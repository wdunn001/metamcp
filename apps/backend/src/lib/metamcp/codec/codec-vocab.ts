/**
 * Vocab map loader + cache for the Codec layer.
 *
 * The MetaMCP gateway is the text/token boundary in the Codec
 * architecture: inference engines emit token IDs, MCP servers
 * speak text JSON-RPC, and this layer converts between them. To do
 * that we need a tokenizer dialect map (a sha256-pinned JSON doc
 * describing one model's BPE merges + vocab + special tokens).
 *
 * Maps are loaded lazily on first reference and cached forever by
 * sha256 hash. The cache is process-local (a single Map) and bounded
 * to MAX_CACHED_MAPS to prevent runaway memory growth if a misbehaving
 * client cycles through hashes. Cache entries are LRU-evicted.
 *
 * The wire-framing layer (codec-transcode.ts) negotiates which map
 * to use per request via the X-Codec-Map header — see
 * negotiateVocabMap() in codec-frame.ts. This module just resolves
 * (url, hash) -> a {Detokenizer, Tokenizer, mapId} bundle.
 *
 * We thin-wrap @codecai/web's existing loadMap / Detokenizer /
 * Tokenizer rather than re-implementing — those are the same
 * decoders the cross-stack matrix uses end-to-end against
 * codec-sglang / codec-vllm / codec-llamacpp, so the bytes line up
 * by construction.
 */
import {
  Detokenizer,
  loadMap,
  pickTokenizer,
  type Tokenizer,
  type TokenizerMap,
} from "@codecai/web";

import logger from "@/utils/logger";

/** Maximum number of distinct vocab maps to keep cached in-process.
 *  Each map is a few MB at most; 32 covers the typical multi-model
 *  agent mesh comfortably and bounds worst-case memory. */
const MAX_CACHED_MAPS = 32;

interface CachedVocab {
  mapId: string;
  map: TokenizerMap;
  detok: Detokenizer;
  tok: Tokenizer;
  /** Last-accessed timestamp for LRU eviction. */
  touchedAt: number;
}

/** Cache keyed by the canonical sha256 hash from the loaded map.
 *  Two different URLs that resolve to identical bytes share a cache
 *  entry — the hash IS the identity. */
const cache = new Map<string, CachedVocab>();

/** Resolve a vocab map handle. Loads + caches on first use, hits
 *  the cache on every subsequent (url, hash) pair with that hash.
 *
 *  Returns the cached bundle so callers never have to re-instantiate
 *  Detokenizer / Tokenizer themselves — those are stateful but
 *  thread-safe (no global mutation in a request handler). The same
 *  Detokenizer can serve concurrent requests because its `render()`
 *  method takes its own buffer arg.
 */
export async function resolveVocabMap(
  url: string,
  hash: string,
): Promise<CachedVocab> {
  const existing = cache.get(hash);
  if (existing) {
    existing.touchedAt = Date.now();
    return existing;
  }

  // First time we see this hash — fetch + verify.
  logger.info(`[Codec] loading vocab map ${hash.slice(0, 14)}... from ${url}`);
  let map: TokenizerMap;
  try {
    map = await loadMap({ url, hash });
  } catch (err) {
    logger.error(`[Codec] failed to load vocab map ${hash}:`, err);
    throw err;
  }

  const entry: CachedVocab = {
    mapId: map.id,
    map,
    detok: new Detokenizer(map),
    tok: pickTokenizer(map),
    touchedAt: Date.now(),
  };
  cache.set(hash, entry);

  // LRU evict if we're over budget. Drop the oldest by touchedAt.
  if (cache.size > MAX_CACHED_MAPS) {
    let oldestKey: string | undefined;
    let oldestAt = Infinity;
    for (const [k, v] of cache.entries()) {
      if (v.touchedAt < oldestAt) {
        oldestAt = v.touchedAt;
        oldestKey = k;
      }
    }
    if (oldestKey) {
      cache.delete(oldestKey);
      logger.info(
        `[Codec] vocab cache evicted ${oldestKey.slice(0, 14)}... (LRU, size=${cache.size})`,
      );
    }
  }

  logger.info(
    `[Codec] vocab map ${entry.mapId} cached as ${hash.slice(0, 14)}... (cache size=${cache.size})`,
  );
  return entry;
}

/** Look up by hash without loading. Useful when a client only sends
 *  the hash on subsequent requests (assuming the map is already
 *  cached). Returns undefined if not cached — caller should require
 *  a full {url, hash} pair on cache miss. */
export function lookupVocabMap(hash: string): CachedVocab | undefined {
  const entry = cache.get(hash);
  if (entry) {
    entry.touchedAt = Date.now();
  }
  return entry;
}

/** Drop a cached map. Mostly for tests + an admin reset endpoint. */
export function evictVocabMap(hash: string): boolean {
  return cache.delete(hash);
}

/** Diagnostic — number of maps cached + their hashes. Used by
 *  the /health/codec endpoint to confirm cache state. */
export function vocabCacheStatus(): { count: number; hashes: string[] } {
  return {
    count: cache.size,
    hashes: [...cache.keys()],
  };
}
