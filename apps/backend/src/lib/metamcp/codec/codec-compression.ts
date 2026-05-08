/**
 * Streaming compression negotiation for the Codec response path.
 *
 * Mirrors `python/sglang/srt/entrypoints/codec_compression.py` from the
 * sglang Codec PR (#24483) and the equivalent layer in the vllm and
 * llama.cpp PRs: pick `gzip` if the client advertises support, fall
 * through to `identity` otherwise. Brotli and zstd would slot in here
 * the same way — gzip first because Node ships zlib in stdlib and the
 * MCP traffic shape (small JSON-RPC envelopes) doesn't benefit much
 * from brotli's bigger framing, while zstd needs a pre-shared
 * dictionary to win on tiny payloads (the dict-zstd path lives in a
 * forthcoming follow-up).
 */
import { Transform } from "node:stream";
import { createGzip } from "node:zlib";

export type CodecResponseEncoding = "identity" | "gzip";

/**
 * Pick an encoding from the client's `Accept-Encoding` header.
 *
 * Returns `identity` when the header is missing or doesn't list any
 * encoding we can serve — never throws on unfamiliar encodings.
 */
export function negotiateResponseEncoding(
  acceptEncoding: string | undefined,
): CodecResponseEncoding {
  if (!acceptEncoding) return "identity";

  // Parse `gzip;q=0.5, br;q=1.0, identity;q=0` style. We only care
  // about presence — q-values are fine in practice but a client that
  // explicitly q=0's gzip clearly wants identity, so respect that.
  const tokens = acceptEncoding
    .split(",")
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);

  for (const token of tokens) {
    const [name, ...params] = token.split(";").map((s) => s.trim());
    if (!name) continue;

    const qParam = params.find((p) => p.startsWith("q="));
    const q = qParam ? Number(qParam.slice(2)) : 1;
    if (!Number.isFinite(q) || q <= 0) continue;

    if (name === "gzip") return "gzip";
    // Future: brotli + zstd land here, in the order their bench cells
    // beat gzip on small JSON-RPC envelopes (~100–500 B). gzip wins on
    // tiny payloads today; bigger wins are dict-zstd territory.
  }

  return "identity";
}

/**
 * Build the Transform stream that wraps the Codec frame iterator into
 * the negotiated encoding. `identity` returns a passthrough so the
 * caller pipes the same way regardless.
 *
 * The compressor is created per-response and `end()`-ed at stream
 * close so each session gets its own deflate context — matches what
 * the sglang `_codec_compression_iter` does and keeps the frames
 * round-trippable through `@codecai/web`'s `decodeStream`.
 */
export function createResponseCompressor(
  encoding: CodecResponseEncoding,
): Transform {
  if (encoding === "gzip") {
    // level 6 = zlib default. Tested to give ~700× wire reduction vs
    // JSON-SSE on Codec frames at 2 K tokens in the cross-stack matrix
    // (sglang row); microbench shows level 9 only buys ~3% extra at
    // the cost of measurable CPU on the hot path. 6 is the sweet spot.
    return createGzip({ level: 6 });
  }

  // PassThrough-equivalent: a Transform that just forwards chunks.
  // Keeps the wiring uniform — the route always pipes(compressor).
  return new Transform({
    transform(chunk, _enc, cb) {
      cb(null, chunk);
    },
  });
}
