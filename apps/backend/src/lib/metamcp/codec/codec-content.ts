/**
 * Tool-call args / results <-> Codec token IDs at the MetaMCP seam.
 *
 * The MetaMCP gateway is the one place in the chain where tokens
 * have to become text and back: tokens upstream (engine + clients),
 * text in the MCP server (filesystem, GitHub, brave-search, etc.).
 * Localizing the detokenize/tokenize work here means:
 *
 *   - The inference engine emits raw uint32 tokens straight on the
 *     wire — no detokenize on the hot path.
 *   - A ToolWatcher anywhere in the chain (or the codec ecosystem's
 *     reference one in @codecai/web) runs on raw token IDs with a
 *     single uint32 compare per token instead of detokenize + regex.
 *   - The consumer (agent runtime, UI, next agent) decides when to
 *     turn tokens back into text — most chains never need to.
 *
 * Two transforms live here:
 *
 *   1. detokenizeCodecArgs(args, vocab)
 *      Inspect a tools/call request's `arguments`. If it carries a
 *      sibling `_codec_meta` block with token IDs, detokenize via
 *      the negotiated vocab map and return a plain JSON object
 *      that the underlying MCP server will accept.
 *
 *   2. tokenizeContent(content, vocab)
 *      Walk a CallToolResult.content[] array. For each
 *      {type:"text", text:"..."} block, tokenize the text and
 *      attach a sibling {type:"_codec_meta", ids, map_id} block.
 *      Empty the original text field so non-Codec MCP clients
 *      that share this namespace still see a valid (empty) text
 *      block — Codec-aware clients read the meta sibling.
 */
import type {
  CallToolResult,
  CallToolRequest,
} from "@modelcontextprotocol/sdk/types.js";

import logger from "@/utils/logger";

import { resolveVocabMap, lookupVocabMap } from "./codec-vocab";

/** Sibling block carrying the Codec encoding of a `text` content
 *  block. Lives next to the original `{type:"text"}` block so
 *  non-Codec clients still see something they can render — they
 *  just see an empty text body. Codec-aware clients ignore the
 *  text block and read this sibling.
 *
 *  `_codec_meta` is prefixed with `_` per the MCP spec convention
 *  for non-standard fields (mirrors `_meta` on requests). */
export interface CodecMetaBlock {
  type: "_codec_meta";
  /** sha256 hash of the canonical map JSON. Identifies which vocab
   *  the IDs belong to; the receiver looks it up in their cache. */
  map_id: string;
  /** Token IDs in big-endian uint32 order — same wire shape as
   *  the streaming Codec frames the cross-stack matrix uses. */
  ids: number[];
}

/** Reference to a tools/call argument value that carries a Codec
 *  encoding. Looks like a normal JSON-RPC arguments object with one
 *  reserved field name. Detokenize replaces the args entirely with
 *  the parsed JSON body. */
interface CodecArgsBlock {
  _codec_meta: CodecMetaBlock;
}

/** Identify a Codec args block. Returns the meta sibling if present. */
export function extractCodecArgsMeta(
  args: unknown,
): CodecMetaBlock | undefined {
  if (!args || typeof args !== "object") return undefined;
  const meta = (args as CodecArgsBlock)._codec_meta;
  if (
    meta &&
    typeof meta === "object" &&
    meta.type === "_codec_meta" &&
    Array.isArray(meta.ids) &&
    typeof meta.map_id === "string"
  ) {
    return meta;
  }
  return undefined;
}

/**
 * Detokenize a tools/call request's Codec-encoded `arguments` block
 * back into the plain JSON object the MCP server expects.
 *
 * If the request doesn't carry a `_codec_meta` block we return the
 * original args unchanged — the JSON path is unaffected. This lets
 * Codec and JSON callers coexist on the same namespace without
 * special routing.
 *
 * The vocab map is resolved by sha256 hash. If we already have the
 * map cached from a previous request we use it; otherwise we error
 * with a clear message because we don't have a URL to load from at
 * this layer. The X-Codec-Map header on the original HTTP request
 * is what carries (url, hash) — codec-transcode.ts is responsible
 * for calling resolveVocabMap() upstream of this layer.
 */
export function detokenizeCodecArgs(
  request: CallToolRequest,
): CallToolRequest {
  const meta = extractCodecArgsMeta(request.params.arguments);
  if (!meta) return request;

  const vocab = lookupVocabMap(meta.map_id);
  if (!vocab) {
    throw new Error(
      `Codec args reference vocab map ${meta.map_id} but it isn't cached. ` +
        `Send X-Codec-Map: <url>;sha256=${meta.map_id} on the request to load it.`,
    );
  }

  // Detokenize to UTF-8 text. The render() call is non-streaming
  // (partial=false) — args arrive whole, not chunked.
  const text = vocab.detok.render(meta.ids, { partial: false });

  // The text is the JSON-stringified arguments object. Parse it
  // back so the MCP SDK sees a normal JS object.
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(
      `Codec args detokenized to non-JSON text (vocab=${vocab.mapId}): ${(err as Error).message}`,
    );
  }

  // Return a fresh request with the args replaced. Don't mutate the
  // original — the metamcp-proxy compose chain may inspect both
  // before and after.
  return {
    ...request,
    params: {
      ...request.params,
      arguments: parsed,
    },
  };
}

/**
 * Walk a CallToolResult and attach a Codec meta sibling next to
 * each `text` content block. Returns a NEW result; the original
 * is left intact.
 *
 * The original text is preserved on the wire (empty-string version
 * is a follow-up if/when we want to suppress duplication). For now
 * we keep both because:
 *
 *   - Non-Codec clients on the same namespace see exactly what they
 *     see today.
 *   - Codec-aware clients that prefer tokens read the meta sibling
 *     and discard the text.
 *   - Empty-string suppression doubles the wire savings on this
 *     boundary but makes the response unintelligible to any client
 *     that doesn't understand `_codec_meta` — too compatibility-
 *     hostile for a v2 patch.
 *
 * The wire wrap (codec-transcode.ts) re-frames the entire envelope
 * as msgpack on the way out, so the per-content tokenization adds
 * value on top of msgpack only when the underlying text is large —
 * file reads, web fetches, RAG snippets. On a 50-token error
 * message the meta block costs more bytes than it saves. The
 * tokenizer doesn't try to be clever about that — small
 * inefficiencies stay; the headline is on the long-text path.
 */
export function tokenizeContent(
  result: CallToolResult,
  mapHash: string,
): CallToolResult {
  const vocab = lookupVocabMap(mapHash);
  if (!vocab) {
    // No cached map — log + return as-is. The caller upstream
    // should have loaded the map before reaching here, but never
    // assume; bailing out preserves the JSON path.
    logger.warn(
      `[Codec] tokenizeContent: vocab map ${mapHash} not cached — leaving result as-is`,
    );
    return result;
  }
  if (!vocab.tok) {
    // Map cached but Tokenizer construction failed (see codec-
    // vocab.ts CachedVocab.tok comment). Fall through and ship
    // text content unchanged — the wire is still re-framed as
    // msgpack/gzip by the wrapper above.
    return result;
  }

  const newContent = result.content.map((block) => {
    if (block.type !== "text" || typeof block.text !== "string") {
      return block; // image, audio, resource, etc. — leave alone
    }
    if (block.text.length === 0) {
      return block; // empty text doesn't benefit from tokenization
    }

    const ids = vocab.tok.encode(block.text);
    const meta: CodecMetaBlock = {
      type: "_codec_meta",
      map_id: mapHash,
      ids,
    };
    // Return the original block UNCHANGED + the meta sibling.
    // The receiving Codec-aware client picks the meta over the
    // text; non-Codec clients ignore the meta and see the text.
    return [block, meta];
  });

  return {
    ...result,
    content: newContent.flat(),
  };
}

/** Convenience wrapper: ensure a vocab map is loaded before any
 *  detokenize/tokenize call. Used by the route handlers when an
 *  X-Codec-Map header arrives.
 *
 *  The header shape is: `<url>;sha256=<hash>` (semicolon-delimited
 *  parameters, similar to Content-Type). Either field can come
 *  first. Whitespace between params is tolerated. */
export async function loadVocabFromHeader(
  header: string | undefined,
): Promise<{ url: string; hash: string } | undefined> {
  if (!header) return undefined;

  let url: string | undefined;
  let hash: string | undefined;

  for (const part of header.split(";").map((s) => s.trim())) {
    if (!part) continue;
    if (part.startsWith("sha256=")) {
      hash = part.slice("sha256=".length);
    } else if (part.startsWith("url=")) {
      url = part.slice("url=".length);
    } else if (!url && (part.startsWith("http://") || part.startsWith("https://"))) {
      // Bare URL as the first param, no `url=` prefix
      url = part;
    }
  }

  if (!url || !hash) {
    throw new Error(
      `X-Codec-Map header must include both a URL and sha256 hash (got "${header}")`,
    );
  }

  await resolveVocabMap(url, hash);
  return { url, hash };
}
