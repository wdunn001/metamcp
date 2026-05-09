/**
 * Streaming compression negotiation for the Codec response path.
 *
 * Mirrors `python/sglang/srt/entrypoints/codec_compression.py` from the
 * sglang Codec PR (#24483) and the equivalent layer in the vllm and
 * llama.cpp PRs: pick the densest encoding the client + this server can
 * BOTH serve, fall through to identity otherwise.
 *
 * Encoding precedence (densest first):
 *   zstd-with-dict  — only when CODEC_MCP_ZSTD_MSGPACK_DICT_PATH is set
 *                     and the dict loaded at boot. Per spec, NO-DICT zstd
 *                     is the worst-of-both-worlds case (TTFB regression
 *                     without offsetting bytes-saved) so we never serve
 *                     it; the dict-gate enforces this.
 *   gzip            — Node stdlib zlib, level 6.
 *   identity        — passthrough.
 *
 * The MCP-shaped dict at `dictionaries/mcp-msgpack-v1.dict` (in the
 * Codec repo) is trained on captured live-bench traffic; on the Phase-1
 * holdout it shrinks responses by 78.8% over no-dict zstd — a 4.7x
 * compression layer on top of the 3.6x msgpack+gzip baseline. See the
 * extract-mcp-corpus + train-zstd-dict scripts in
 * packages/bench/scripts/.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { Transform } from "node:stream";
import { createGzip } from "node:zlib";

// @mongodb-js/zstd is the workhorse for dict-zstd in the Node ecosystem
// (battle-tested in production by the MongoDB driver). One-shot async
// API; we wrap it in a buffering Transform so the route-handler wiring
// stays uniform with gzip's streaming Transform.
import { compress as zstdCompress } from "@mongodb-js/zstd";

import { logger } from "../../logger.js";

export type CodecResponseEncoding = "identity" | "gzip" | "zstd";

// ── Loaded-at-boot zstd dictionary state ─────────────────────────────────────
//
// One dict per (codec) format; today we ship msgpack-only because the
// captured corpus is msgpack-flavored. A protobuf dict slots in the
// same way once a protobuf-flavored corpus is captured. The state is
// module-scoped: one process-lifetime load, no per-request file IO.

interface LoadedDict {
  bytes: Buffer;
  hash: string; // 'sha256:<hex>'
}

const loadedDicts: Map<"msgpack" | "protobuf", LoadedDict> = new Map();

function tryLoadDict(envName: string, label: "msgpack" | "protobuf"): void {
  const path = process.env[envName];
  if (!path) return;
  try {
    if (!existsSync(path)) {
      logger.warn(`[Codec] ${envName}=${path} but file doesn't exist; zstd disabled for ${label}`);
      return;
    }
    const bytes = readFileSync(path);
    const hash =
      "sha256:" + createHash("sha256").update(bytes).digest("hex");
    loadedDicts.set(label, { bytes, hash });
    logger.info(
      `[Codec] loaded ${label} zstd dict from ${path} (${bytes.length} B, ${hash.slice(0, 19)}…)`,
    );
  } catch (e) {
    logger.warn(
      `[Codec] failed to load ${envName}=${path}: ${(e as Error).message} — zstd disabled for ${label}`,
    );
  }
}

// Eager load at module init. Keeps later request paths synchronous —
// negotiateResponseEncoding can answer without IO.
tryLoadDict("CODEC_MCP_ZSTD_MSGPACK_DICT_PATH", "msgpack");
tryLoadDict("CODEC_MCP_ZSTD_PROTOBUF_DICT_PATH", "protobuf");

/** Return the loaded zstd-dict hash for a format, or undefined if none. */
export function getLoadedDictHash(
  format: "msgpack" | "protobuf",
): string | undefined {
  return loadedDicts.get(format)?.hash;
}

/** Whether a dict is loaded for the given format. */
export function hasLoadedDict(format: "msgpack" | "protobuf"): boolean {
  return loadedDicts.has(format);
}

// ── Negotiation ──────────────────────────────────────────────────────────────

/**
 * Pick an encoding from the client's `Accept-Encoding` header.
 *
 * Returns `identity` when the header is missing or doesn't list any
 * encoding we can serve — never throws on unfamiliar encodings. Returns
 * `zstd` only when (a) the client accepts it AND (b) we have a
 * pre-trained dict loaded for this format. Per spec/PROTOCOL.md
 * §"Pre-trained ZSTD dictionaries", no-dict zstd is dominated by gzip on
 * MCP-shaped traffic — we never serve it.
 *
 * The `format` argument lets the caller match the dict to the wire
 * format their response will use (msgpack today; protobuf when
 * captured).
 */
export function negotiateResponseEncoding(
  acceptEncoding: string | undefined,
  format: "msgpack" | "protobuf" = "msgpack",
): CodecResponseEncoding {
  if (!acceptEncoding) return "identity";

  // Parse `gzip;q=0.5, br;q=1.0, identity;q=0` style. We only care
  // about presence — q-values are fine in practice but a client that
  // explicitly q=0's gzip clearly wants identity, so respect that.
  const tokens = acceptEncoding
    .split(",")
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);

  // First pass: zstd if both we and the client support it.
  for (const token of tokens) {
    const [name, ...params] = token.split(";").map((s) => s.trim());
    if (!name) continue;
    const qParam = params.find((p) => p.startsWith("q="));
    const q = qParam ? Number(qParam.slice(2)) : 1;
    if (!Number.isFinite(q) || q <= 0) continue;
    if (name === "zstd" && hasLoadedDict(format)) return "zstd";
  }

  // Second pass: fall through to gzip.
  for (const token of tokens) {
    const [name, ...params] = token.split(";").map((s) => s.trim());
    if (!name) continue;
    const qParam = params.find((p) => p.startsWith("q="));
    const q = qParam ? Number(qParam.slice(2)) : 1;
    if (!Number.isFinite(q) || q <= 0) continue;
    if (name === "gzip") return "gzip";
  }

  return "identity";
}

// ── Compressor factories ─────────────────────────────────────────────────────

/**
 * Buffering Transform wrapper around @mongodb-js/zstd's one-shot async
 * compress(). Collects all input chunks, compresses with the loaded dict
 * on flush, emits one output chunk. MCP responses are small enough
 * (typically <2KB) that buffering imposes no perceptible latency cost
 * vs streaming zstd.
 */
function createZstdDictCompressor(
  format: "msgpack" | "protobuf",
): Transform {
  const dict = loadedDicts.get(format);
  if (!dict) {
    // Should never happen — negotiateResponseEncoding gates this. Fall
    // back to passthrough rather than crashing if someone calls us
    // out-of-order.
    logger.warn(
      `[Codec] zstd compressor requested without loaded ${format} dict — passthrough`,
    );
    return createPassthrough();
  }

  const chunks: Buffer[] = [];
  return new Transform({
    transform(chunk: unknown, _enc: BufferEncoding, cb) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array));
      cb();
    },
    async flush(cb) {
      try {
        const input = Buffer.concat(chunks);
        // level 3 = zstd default. Matches sglang's
        // codec_compression.py and the python ZstdCompressor default.
        const compressed = await zstdCompress(input, 3, dict.bytes);
        cb(null, compressed);
      } catch (e) {
        cb(e as Error);
      }
    },
  });
}

function createPassthrough(): Transform {
  return new Transform({
    transform(chunk, _enc, cb) {
      cb(null, chunk);
    },
  });
}

/**
 * Build the Transform stream that wraps the Codec frame iterator into
 * the negotiated encoding. `identity` returns a passthrough so the
 * caller pipes the same way regardless.
 *
 * The compressor is created per-response; gzip's deflate context and
 * zstd's dict-bound state are scoped to the response so each session
 * gets its own context — matches what sglang's
 * `_codec_compression_iter` does and keeps frames round-trippable
 * through `@codecai/web`'s `decodeStream`.
 */
export function createResponseCompressor(
  encoding: CodecResponseEncoding,
  format: "msgpack" | "protobuf" = "msgpack",
): Transform {
  if (encoding === "zstd") {
    return createZstdDictCompressor(format);
  }
  if (encoding === "gzip") {
    // level 6 = zlib default. Tested to give ~700× wire reduction vs
    // JSON-SSE on Codec frames at 2 K tokens in the cross-stack matrix
    // (sglang row); microbench shows level 9 only buys ~3% extra at
    // the cost of measurable CPU on the hot path. 6 is the sweet spot.
    return createGzip({ level: 6 });
  }

  // PassThrough-equivalent: a Transform that just forwards chunks.
  // Keeps the wiring uniform — the route always pipes(compressor).
  return createPassthrough();
}
