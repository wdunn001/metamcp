/**
 * Codec wire format for MetaMCP — msgpack/protobuf framing for the
 * JSON-RPC stream that the MCP SDK normally writes to the response.
 *
 * Same shape as the Codec frames shipped on the OpenAI completions
 * path in sglang (PR #24483), vllm (#41765), and llama.cpp (#22757):
 *
 *     +------------------+------------------------------+
 *     | 4-byte BE length |  msgpack OR protobuf body    |
 *     +------------------+------------------------------+
 *
 * For MetaMCP the body is a JSON-RPC message — request, response,
 * notification — exactly as the MCP server already produces it. We
 * just swap the JSON serializer for msgpack (or protobuf) and
 * length-prefix it. No semantic change to the protocol; bytes only.
 *
 * The JSON path is unchanged when `stream_format` is not negotiated —
 * the existing /:uuid/mcp routes still emit JSON-RPC over Streamable
 * HTTP / SSE. Codec is opt-in per request via either:
 *
 *   - `?stream_format=msgpack`  (or `protobuf`) on the URL, or
 *   - `Accept: application/x-codec-msgpack`  request header.
 *
 * Background and the cross-stack benchmark matrix that motivated
 * this layer: https://codecai.net/docs/protocol/
 */
import { decode as decodeMsgpack, encode as encodeMsgpack } from "@msgpack/msgpack";

/** Codec wire formats we speak today. JSON is the SDK default and not
 *  routed through this module. */
export type CodecStreamFormat = "msgpack" | "protobuf";

/** Anything the MCP SDK serializes to JSON-RPC. We encode it as-is —
 *  msgpack/protobuf preserves the exact same field names and semantics. */
export type JsonRpcMessage = Record<string, unknown>;

/** Build one length-prefixed Codec frame from a single JSON-RPC message.
 *
 *  Mirrors the encoder in @codecai/web's `decodeStream` and the Python
 *  client's `decode_msgpack_stream` — they decode this exact wire shape.
 */
export function encodeCodecFrame(
  message: JsonRpcMessage,
  format: CodecStreamFormat,
): Buffer {
  const body =
    format === "msgpack" ? encodeBody(message) : encodeProtobufBody(message);

  const length = Buffer.alloc(4);
  length.writeUInt32BE(body.length, 0);
  return Buffer.concat([length, body]);
}

function encodeBody(message: JsonRpcMessage): Buffer {
  const encoded = encodeMsgpack(message, { useBigInt64: false });
  // @msgpack/msgpack returns Uint8Array; lift to Buffer for Node stream
  // ergonomics. Buffer is a Uint8Array subclass so this is a view, not
  // a copy.
  return Buffer.from(encoded.buffer, encoded.byteOffset, encoded.byteLength);
}

/** Hand-rolled minimal protobuf encoding for the JSON-RPC envelope.
 *
 *  This stays simple on purpose — MCP messages are nested objects with
 *  arbitrary shapes (varying tool args, varying tool results), so we
 *  don't try to fit them into a strict `.proto` schema. Instead we
 *  msgpack the JS object first and then wrap THAT in a one-field
 *  protobuf message:
 *
 *      message CodecRpcEnvelope {
 *          bytes msgpack_body = 1;
 *      }
 *
 *  Result: clients that already speak `protoc`-generated code can
 *  decode via the same envelope they generate, and the body inside
 *  is plain msgpack — no protobuf reflection over arbitrary tool
 *  arguments. The wire size is essentially identical to msgpack-only;
 *  protobuf clients pay one extra varint of header.
 *
 *  Tool-result text content with its real per-token Codec encoding
 *  lives in a separate (forthcoming) layer that takes a vocab map and
 *  rewrites `content[i].text` → `content[i].ids` — that's where the
 *  big wire reduction lives. This first pass just gets binary framing
 *  on the JSON-RPC envelope.
 */
function encodeProtobufBody(message: JsonRpcMessage): Buffer {
  const inner = encodeBody(message);

  // Protobuf field 1, wire type 2 (length-delimited): tag = (1 << 3) | 2 = 0x0A
  const tag = Buffer.from([0x0a]);
  const len = encodeVarint(inner.length);
  return Buffer.concat([tag, len, inner]);
}

function encodeVarint(value: number): Buffer {
  if (value < 0 || !Number.isSafeInteger(value)) {
    throw new RangeError(`Codec frame: cannot varint-encode ${value}`);
  }
  const out: number[] = [];
  while (value >= 0x80) {
    out.push((value & 0x7f) | 0x80);
    value = Math.floor(value / 128);
  }
  out.push(value);
  return Buffer.from(out);
}

/** Decode an inbound Codec wire body into a JSON-RPC message.
 *
 *  Used on the request path when a client POSTs msgpack/protobuf —
 *  we accept either inline (single frame, no length prefix needed
 *  because the request body already carries Content-Length) or a
 *  length-prefixed stream of multiple messages. This implementation
 *  handles the simple inline case used by HTTP request bodies; the
 *  streaming case is on the response side and is handled by
 *  `encodeCodecFrame` above.
 */
export function decodeInlineMsgpack(body: Buffer): JsonRpcMessage {
  const value = decodeMsgpack(body);
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(
      `Codec frame: expected JSON-RPC object after msgpack decode, got ${typeof value}`,
    );
  }
  return value as JsonRpcMessage;
}

export function decodeInlineProtobuf(body: Buffer): JsonRpcMessage {
  // Expect a single field-1 length-delimited entry: (0x0A) (varint) (msgpack body)
  if (body.length < 2 || body[0] !== 0x0a) {
    throw new Error(
      "Codec frame: protobuf envelope must start with field 1 tag (0x0A)",
    );
  }
  const [innerLen, headerSize] = decodeVarint(body, 1);
  const start = 1 + headerSize;
  if (start + innerLen > body.length) {
    throw new Error(
      `Codec frame: protobuf envelope truncated (header says ${innerLen} bytes, only ${body.length - start} available)`,
    );
  }
  return decodeInlineMsgpack(body.subarray(start, start + innerLen));
}

function decodeVarint(buf: Buffer, offset: number): [number, number] {
  let value = 0;
  let shift = 0;
  let i = offset;
  while (i < buf.length) {
    const b = buf[i];
    if (b === undefined) break;
    value += (b & 0x7f) * Math.pow(2, shift);
    if ((b & 0x80) === 0) {
      return [value, i - offset + 1];
    }
    shift += 7;
    i += 1;
    if (shift > 49) {
      throw new RangeError("Codec frame: varint too large to fit a JS number");
    }
  }
  throw new Error("Codec frame: truncated varint");
}

/** Negotiate the stream format from a request's query + headers.
 *
 *  Resolution order, first match wins:
 *    1. `?stream_format=msgpack|protobuf|json`
 *    2. `Accept: application/x-codec-msgpack` or `application/x-codec-protobuf`
 *    3. fallthrough → undefined (caller should treat as JSON, the SDK default)
 *
 *  `stream_format=json` is a valid explicit opt-out, useful for clients
 *  that toggle per-request without touching headers.
 */
export function negotiateStreamFormat(
  query: Record<string, unknown>,
  acceptHeader: string | undefined,
): CodecStreamFormat | undefined {
  const queryFormat = (query.stream_format ?? query.streamFormat) as
    | string
    | undefined;
  if (queryFormat) {
    const lower = queryFormat.toLowerCase();
    if (lower === "msgpack" || lower === "protobuf") return lower;
    if (lower === "json") return undefined;
  }

  if (acceptHeader) {
    // Accept may be a comma-separated q-list; we just substring-test
    // since the Codec types are unambiguous.
    if (acceptHeader.includes("application/x-codec-msgpack")) return "msgpack";
    if (acceptHeader.includes("application/x-codec-protobuf")) return "protobuf";
  }

  return undefined;
}

/** The Content-Type to advertise on a Codec response. Used by the
 *  Express response wrapper before flushing the first frame. */
export function contentTypeFor(format: CodecStreamFormat): string {
  return format === "msgpack"
    ? "application/x-codec-msgpack"
    : "application/x-codec-protobuf";
}
