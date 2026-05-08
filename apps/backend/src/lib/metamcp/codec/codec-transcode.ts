/**
 * Express request/response transcoding for the Codec path.
 *
 * The MCP SDK's `StreamableHTTPServerTransport` writes JSON-RPC
 * messages straight into `res` as either `application/json` (for
 * single-shot replies) or as `text/event-stream` SSE (for long-running
 * streams with progress notifications). Both paths boil down to one
 * call: `res.write(<JSON-bytes-ending-in-newline>)` per message.
 *
 * To switch the wire format to Codec without forking the SDK we wrap
 * `req` and `res` BEFORE handing them to `transport.handleRequest`:
 *
 *   - `req`: if Content-Type is `application/x-codec-msgpack` (or
 *     `…-protobuf`), decode the body to a JS object and let the SDK
 *     read it from a synthetic stream; the SDK's `JSON.parse` step
 *     becomes a no-op.
 *
 *   - `res`: monkey-patch `setHeader`, `write`, and `end` so the
 *     body's framing changes from JSON+newline to length-prefixed
 *     msgpack/protobuf. Headers swap Content-Type to the negotiated
 *     Codec mime type and add `Content-Encoding` if gzip was
 *     negotiated.
 *
 * This is intentionally surgical — no SDK fork, no new transport
 * subclass, no protocol change for clients that don't opt in. JSON
 * traffic on the same /:uuid/mcp route is byte-for-byte identical to
 * upstream MetaMCP.
 */
import type { Request, Response } from "express";
import { Readable } from "node:stream";

import {
  type CodecStreamFormat,
  contentTypeFor,
  decodeInlineMsgpack,
  decodeInlineProtobuf,
  encodeCodecFrame,
  type JsonRpcMessage,
} from "./codec-frame";
import {
  type CodecResponseEncoding,
  createResponseCompressor,
} from "./codec-compression";

/**
 * Decode a Codec-framed POST body into a JSON-RPC message and replace
 * `req.body` so the SDK's existing JSON-decoded path sees a plain
 * object exactly as if it had come in as JSON.
 *
 * Caller is responsible for ensuring this is only invoked when the
 * request actually carries a Codec content-type — checking that lives
 * one level up in the route handler so the JSON path is untouched.
 */
export function decodeCodecRequestBody(
  req: Request,
  format: CodecStreamFormat,
): void {
  // Express's body parser already left a Buffer in req.body when the
  // type isn't application/json. If the operator hasn't installed a
  // raw-body parser for our content types this will be a Stream — in
  // which case we don't have a synchronous decode path and the SDK
  // will try to JSON.parse a Buffer-as-string which fails loudly.
  // Routes that opt into Codec should mount express.raw() with the
  // matching `type` filter; see codec-router.ts.
  const buf = req.body;
  if (!Buffer.isBuffer(buf)) {
    throw new Error(
      `Codec request: expected raw Buffer body for ${format}, got ${typeof buf}. ` +
        `Mount express.raw({ type: "application/x-codec-${format}" }) before this route.`,
    );
  }

  const decoded: JsonRpcMessage =
    format === "msgpack" ? decodeInlineMsgpack(buf) : decodeInlineProtobuf(buf);

  // Hand the SDK the decoded object directly. StreamableHTTPServerTransport
  // accepts a parsed body via the third argument of handleRequest, so we
  // both replace req.body AND signal to the caller they should pass it
  // explicitly.
  (req as Request & { body: unknown }).body = decoded;
}

/**
 * Wrap `res` so the SDK's write path emits Codec frames instead of
 * newline-delimited JSON. Transparent: the wrapped `res` still has
 * the Express Response type and forwards every method we don't
 * intercept.
 *
 * Returns a cleanup that flushes any buffered compressor output —
 * call from `res.on("close")` if you need belt-and-braces shutdown,
 * but in normal flow `res.end()` triggers it automatically.
 */
export function wrapResponseForCodec(
  res: Response,
  format: CodecStreamFormat,
  encoding: CodecResponseEncoding,
): () => void {
  // Set Codec headers up front so the SDK's later setHeader calls for
  // Content-Type get overridden cleanly. We keep its other headers
  // (Cache-Control, mcp-session-id, etc.) untouched.
  res.setHeader("Content-Type", contentTypeFor(format));
  if (encoding !== "identity") {
    res.setHeader("Content-Encoding", encoding);
    res.setHeader("Vary", "Accept-Encoding");
  }
  // Streamable HTTP responses are unboundedly long; chunked is the
  // only sane framing. Express will set this automatically when we
  // call res.write before res.end, but pinning it here avoids any
  // race with a late Content-Length write from the SDK.
  res.setHeader("Transfer-Encoding", "chunked");

  // Build the compression pipe. Output flows: writeFrame() ->
  // [compressor] -> [res-socket]. For identity the compressor is a
  // passthrough Transform, so the indirection is consistent.
  const compressor = createResponseCompressor(encoding);
  compressor.on("error", (err) => {
    // If the compressor blows up the response is already partially
    // written — there's no clean way to switch back to JSON. End
    // the socket and let the client retry without stream_format.
    if (!res.destroyed) {
      res.destroy(err);
    }
  });
  compressor.pipe(res);

  // Patch the SDK's view of res. The SDK uses three call patterns:
  //
  //   1. `res.writeHead(status, headers).flushHeaders()` — commits
  //      headers immediately for streaming responses (SSE). This
  //      OVERWRITES anything we set via setHeader earlier, so we
  //      have to intercept writeHead and substitute our Codec headers
  //      back in.
  //
  //   2. `res.write(chunk)` — newline-delimited JSON or SSE event
  //      chunks. Parsed back to JS, framed, and piped to the
  //      compressor.
  //
  //   3. `res.writeHead(status).end(JSON.stringify(...))` — short
  //      error path for protocol failures (4xx/406/415/etc.). The
  //      end() chunk goes through the same forwarder so the client
  //      gets a Codec-encoded error envelope rather than mixed JSON.
  const originalWrite = res.write.bind(res);
  const originalEnd = res.end.bind(res);

  // Unused variable suppression — we deliberately don't forward to
  // originalWrite because that would write the original JSON bytes
  // alongside our Codec frames.
  void originalWrite;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const originalWriteHead = (res as any).writeHead.bind(res);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (res as any).writeHead = (
    status: number,
    headersOrReason?: unknown,
    maybeHeaders?: unknown,
  ) => {
    // Pin our Codec headers regardless of what the SDK tries to
    // set. Lets the SDK pick the status code (200 for happy path,
    // 4xx for protocol errors) but the wire bytes underneath stay
    // Codec for the duration of this request.
    const ourHeaders: Record<string, string> = {
      "Content-Type": contentTypeFor(format),
      "Transfer-Encoding": "chunked",
    };
    if (encoding !== "identity") {
      ourHeaders["Content-Encoding"] = encoding;
      ourHeaders["Vary"] = "Accept-Encoding";
    }
    // Carry forward any non-conflicting headers the SDK passed —
    // mcp-session-id, cache-control, access-control-*, etc. The
    // shape of writeHead is overloaded:
    //   writeHead(status, headers)
    //   writeHead(status, statusMessage, headers)
    let sdkHeaders: Record<string, string> | undefined;
    if (headersOrReason && typeof headersOrReason === "object") {
      sdkHeaders = headersOrReason as Record<string, string>;
    } else if (maybeHeaders && typeof maybeHeaders === "object") {
      sdkHeaders = maybeHeaders as Record<string, string>;
    }
    if (sdkHeaders) {
      for (const [key, value] of Object.entries(sdkHeaders)) {
        const lower = key.toLowerCase();
        if (
          lower === "content-type" ||
          lower === "content-encoding" ||
          lower === "content-length" ||
          lower === "transfer-encoding"
        ) {
          continue; // we own these
        }
        ourHeaders[key] = String(value);
      }
    }
    return originalWriteHead(status, ourHeaders);
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (res as any).flushHeaders = () => {
    // No-op: writeHead already commits. The SDK calls flushHeaders
    // after writeHead for SSE; if we forward it the underlying
    // socket sends headers twice. Swallow it.
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (res as any).write = (chunk: any, ...rest: any[]): boolean => {
    try {
      forwardChunkToCodec(chunk, compressor, format);
      // Honor the optional callback on the original signature
      const cb = rest.find((arg) => typeof arg === "function") as
        | ((err?: Error) => void)
        | undefined;
      if (cb) process.nextTick(cb);
      return true;
    } catch (err) {
      compressor.destroy(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (res as any).end = ((chunk?: any, ...rest: any[]): Response => {
    if (chunk) {
      try {
        forwardChunkToCodec(chunk, compressor, format);
      } catch (err) {
        compressor.destroy(err instanceof Error ? err : new Error(String(err)));
        return originalEnd();
      }
    }
    compressor.end();
    // Don't call originalEnd directly — the compressor.pipe(res)
    // wiring handles the end-of-stream propagation when compressor
    // finishes. This keeps gzip's footer bytes inside the response.
    return res;
  }) as Response["end"];

  return () => {
    if (!compressor.destroyed) compressor.end();
  };
}

/**
 * Take a single chunk that the SDK wrote (JSON-RPC line OR SSE event)
 * and emit one Codec frame per JSON-RPC message contained within it.
 *
 * The SDK has two emission modes; we don't try to detect which one
 * we're in — we just look for parsable JSON segments in the chunk.
 */
function forwardChunkToCodec(
  chunk: unknown,
  out: NodeJS.WritableStream,
  format: CodecStreamFormat,
): void {
  const text = chunkToString(chunk);
  if (text.length === 0) return;

  for (const message of extractJsonRpcMessages(text)) {
    out.write(encodeCodecFrame(message, format));
  }
}

function chunkToString(chunk: unknown): string {
  if (typeof chunk === "string") return chunk;
  if (Buffer.isBuffer(chunk)) return chunk.toString("utf8");
  if (chunk instanceof Uint8Array) {
    return Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength).toString("utf8");
  }
  if (chunk == null) return "";
  // Fallback for stream-mode SDK callers that pass an Object — should
  // not happen, but keep the wrapper crash-resistant.
  return String(chunk);
}

/**
 * Pull JSON-RPC objects out of either:
 *   - one or more newline-terminated JSON lines, or
 *   - SSE events of the shape `event: message\ndata: {...}\n\n`.
 *
 * We're permissive here because the SDK's Streamable HTTP transport
 * is allowed to mix initialization metadata into the same stream and
 * we want to forward those objects unchanged too.
 */
function extractJsonRpcMessages(text: string): JsonRpcMessage[] {
  const messages: JsonRpcMessage[] = [];

  // SSE path: split on "\n\n" event boundaries, pick out the data: line.
  if (text.includes("data:")) {
    for (const event of text.split(/\n\n+/)) {
      for (const line of event.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        const parsed = tryParseJson(payload);
        if (parsed) messages.push(parsed);
      }
    }
    if (messages.length > 0) return messages;
  }

  // Single-shot path: one or more newline-terminated JSON objects.
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parsed = tryParseJson(trimmed);
    if (parsed) messages.push(parsed);
  }

  return messages;
}

function tryParseJson(s: string): JsonRpcMessage | undefined {
  try {
    const value = JSON.parse(s);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as JsonRpcMessage;
    }
  } catch {
    // Not JSON — skip; might be an SSE comment line or partial chunk.
  }
  return undefined;
}

/**
 * For tests / future callers that want to pump frames in from a stream
 * (e.g. Codec-encoded request bodies arriving over HTTP/2), expose the
 * raw frame parser. Not used on the response path.
 */
export function readableFromBuffer(buf: Buffer): Readable {
  return Readable.from(buf, { objectMode: false });
}
