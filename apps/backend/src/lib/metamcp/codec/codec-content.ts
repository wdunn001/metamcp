/**
 * Tool-call args / results <-> Codec token IDs — the **gateway-tokenize
 * shim** per `spec/PROTOCOL.md § Backward compatibility (legacy
 * text-mode tools)`.
 *
 * # Read this before changing anything in this file
 *
 * The architectural target documented in the spec is
 * **leaf-tokenization**: the MCP server itself tokenizes its result
 * with the session-negotiated vocab and emits a `ToolResultFrame`
 * with raw token IDs. The gateway in that target is a transparent ID
 * pipe — it forwards `ToolCallFrame` / `ToolResultFrame` bytes by
 * `tool_call_id` and never opens the body.
 *
 * What this file implements is **the back-compat shim** for the
 * (currently universal) case where the downstream MCP server does
 * NOT speak Codec. The gateway then has to do the leaf's job on its
 * behalf:
 *
 *   - Inbound: detokenize a Codec-encoded `arguments` block to JSON
 *     so the legacy MCP server sees a normal tools/call request.
 *   - Outbound: walk a CallToolResult.content[] array and attach a
 *     `_codec_meta` sibling carrying tokenized text, so a Codec-aware
 *     client downstream reads IDs instead of UTF-8.
 *
 * That is, today's MetaMCP gateway IS the text/token boundary —
 * because no MCP server in the wild yet emits ToolResultFrames. As
 * MCP servers upgrade, calls to the functions in this file should
 * disappear from any given session: the leaf tokenizes, the gateway
 * never gets here, the spec target is met.
 *
 * Operator visibility: every tokenize/detokenize that happens here
 * bumps a counter (`shimInvocationCount`). Operators can poll it via
 * `getShimMetrics()` to see how much of their MCP traffic is still
 * relying on the shim vs flowing as native Codec. A non-zero count is
 * normal during the legacy → Codec transition; the goal over time is
 * for that counter to flatline against a growing total request count.
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
 *      Both the original text and the meta sibling ship — non-Codec
 *      clients ignore the meta, Codec-aware clients prefer it.
 */
import type {
  CallToolResult,
  CallToolRequest,
} from "@modelcontextprotocol/sdk/types.js";

import logger from "@/utils/logger";

import { resolveVocabMap, lookupVocabMap } from "./codec-vocab";

// ── Shim metrics ────────────────────────────────────────────────────
//
// Per spec/PROTOCOL.md § Backward compatibility, gateway-side
// tokenization MUST be observable to operators so the cost of legacy
// text-mode MCP servers is visible. We keep a tiny in-process counter
// pair: total invocations, and a per-vocab breakdown for sessions
// that have happened to mix vocabs (rare but possible if the gateway
// fronts multiple model families).
//
// Cheap by design — this isn't Prometheus; it's a getter the operator
// can poll over the trpc admin route or grep out of `docker logs`.
// Promote to a real metrics backend if/when MetaMCP grows one.

interface ShimMetrics {
  /** Total `detokenizeCodecArgs` calls that found a `_codec_meta`
   *  block and ran a detokenize. */
  detokenizeCalls: number;
  /** Total `tokenizeContent` calls that produced one or more
   *  `_codec_meta` siblings. */
  tokenizeCalls: number;
  /** Total `tokenizeContent` calls that found the leaf had ALREADY
   *  emitted `_codec_meta` and skipped the shim path. This is the
   *  counter operators want to grow over time — it measures how much
   *  of MCP traffic has graduated to leaf-tokenization, the
   *  architectural target per the spec. */
  leafBypasses: number;
  /** Per-vocab invocation count, keyed by sha256 hash. Lets operators
   *  spot uneven legacy-MCP traffic across model families. */
  byVocab: Record<string, number>;
  /** When the counters were last reset (process start). */
  since: string;
}

const shimMetrics: ShimMetrics = {
  detokenizeCalls: 0,
  tokenizeCalls: 0,
  leafBypasses: 0,
  byVocab: {},
  since: new Date().toISOString(),
};

/** Snapshot of shim invocation counters. Read-only — callers MUST NOT
 *  mutate the returned object. */
export function getShimMetrics(): Readonly<ShimMetrics> {
  return {
    ...shimMetrics,
    byVocab: { ...shimMetrics.byVocab },
  };
}

function bumpShim(kind: "detok" | "tok", vocabHash: string): void {
  if (kind === "detok") shimMetrics.detokenizeCalls += 1;
  else shimMetrics.tokenizeCalls += 1;
  shimMetrics.byVocab[vocabHash] = (shimMetrics.byVocab[vocabHash] ?? 0) + 1;
}

// Once-per-vocab "shim mode engaged" log line. Operators see one
// entry per fresh (vocab, process-lifetime) pair so logs aren't
// flooded but the path is grep-able.
//
// Emitted at WARN level because shim mode IS the degraded path per
// the spec — leaf-tokenizing tools would skip this hop entirely.
// Logger's default LOG_LEVEL is `errors-only`, which mirrors WARN
// and ERROR to console; INFO would be invisible by default and that
// would defeat the spec's "MUST be observable to operators" rule.
// Once-per-(vocab, process) limits volume; the per-call counter in
// `shimMetrics` is the high-rate observability source.
const shimAnnouncedFor = new Set<string>();
function announceShimOnce(vocabHash: string, kind: "detok" | "tok"): void {
  const key = `${vocabHash}:${kind}`;
  if (shimAnnouncedFor.has(key)) return;
  shimAnnouncedFor.add(key);
  logger.warn(
    `[Codec][shim] ${kind === "detok" ? "detokenizing args" : "tokenizing tool result"} ` +
      `for vocab ${vocabHash.slice(0, 12)}… — leaf-mode MCP server would skip this. ` +
      `(spec/PROTOCOL.md § Backward compatibility)`,
  );
}

// Once-per-vocab "leaf-mode tool result observed" log line — the
// inverse of announceShimOnce. Fires the first time a downstream MCP
// server returns a result that already carries a `_codec_meta` block,
// telling us this tool has graduated to leaf-tokenization. Emitted at
// INFO because it's good news, not a degraded path; operators can
// surface it with LOG_LEVEL=info or by polling `getShimMetrics()`.
const leafAnnouncedFor = new Set<string>();
function announceLeafOnce(vocabHash: string): void {
  if (leafAnnouncedFor.has(vocabHash)) return;
  leafAnnouncedFor.add(vocabHash);
  logger.info(
    `[Codec][leaf] downstream tool returned pre-tokenized result for vocab ` +
      `${vocabHash.slice(0, 12)}… — gateway shim bypassed. ` +
      `(spec/PROTOCOL.md § Tool-call calling conventions in the map)`,
  );
}

/**
 * Detect whether a CallToolResult.content array already carries a
 * `_codec_meta` block somewhere — meaning the leaf tool tokenized
 * its own result and the gateway should not re-tokenize.
 *
 * Conservative: returns true only if the meta block has the expected
 * shape (`type === "_codec_meta"`, `map_id` is a string, `ids` is an
 * array). A malformed sibling triggers shim re-tokenization, which is
 * the safer fallback than passing garbage through.
 */
function hasExistingCodecMeta(
  content: ReadonlyArray<{ type?: string; map_id?: unknown; ids?: unknown }>,
): boolean {
  for (const block of content) {
    if (
      block?.type === "_codec_meta" &&
      typeof block.map_id === "string" &&
      Array.isArray(block.ids)
    ) {
      return true;
    }
  }
  return false;
}

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

  // Shim path engaged — record + announce once for operator
  // visibility. Per spec/PROTOCOL.md § Backward compatibility, this
  // bookkeeping is REQUIRED, not optional.
  bumpShim("detok", meta.map_id);
  announceShimOnce(meta.map_id, "detok");

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
  // Idempotence: if the leaf already produced a `_codec_meta` block,
  // the spec target is met — the gateway becomes a transparent ID
  // pipe for this hop. Skip the shim, count the bypass, announce the
  // first time we see it for this vocab. The result still flows as
  // msgpack/gzip on the wire wrap layer; we just don't double-encode.
  if (Array.isArray(result.content) && hasExistingCodecMeta(result.content)) {
    shimMetrics.leafBypasses += 1;
    announceLeafOnce(mapHash);
    return result;
  }

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

  let didEmitMeta = false;
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
    didEmitMeta = true;
    // Return the original block UNCHANGED + the meta sibling.
    // The receiving Codec-aware client picks the meta over the
    // text; non-Codec clients ignore the meta and see the text.
    return [block, meta];
  });

  // Only record a shim invocation if we actually emitted a `_codec_meta`
  // sibling — a result that's all images/empty/etc. didn't trigger
  // any leaf-shim work and shouldn't show up in the counter.
  if (didEmitMeta) {
    bumpShim("tok", mapHash);
    announceShimOnce(mapHash, "tok");
  }

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
