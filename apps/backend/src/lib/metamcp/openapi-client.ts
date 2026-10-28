import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { ServerParameters } from "@repo/zod-types";

import logger from "@/utils/logger";

import { transformDockerUrl } from "./client";

/**
 * Native OpenAPI downstream support.
 *
 * An OPENAPI server row points at a plain OpenAPI service (FastAPI sidecars
 * on this fleet: knowledge-api, psych-kb-api, assist-tools, comfy-tools,
 * paperless-tools, memory-tools, calendar-tools, terminal-run). MetaMCP
 * fetches the service's openapi.json, turns each operation into an MCP tool
 * named by its bare operationId, and serves those tools from an in-process
 * MCP server wired to the pool through an in-memory transport pair. The rest
 * of MetaMCP sees an ordinary connected client.
 *
 * Tool names deliberately carry no prefix of their own. MetaMCP prefixes the
 * server name at the namespace layer, so a search_knowledge operation on the
 * `knowledge` server reaches clients as knowledge__search_knowledge. The
 * retired bridge containers self-prefixed, which doubled the name.
 *
 * Per-tool allowlisting and description overrides are NOT done here. The
 * namespace layer already owns those (tool status plus override_description
 * on namespace_tool_mappings).
 */

/** JSON value as OpenAPI specs carry it. */
type Json = Record<string, unknown>;

interface Operation {
  toolName: string;
  method: string;
  path: string;
  /** declared parameters: name plus location (path, query, header) */
  params: Array<{ name: string; in: string; required?: boolean }>;
  hasBody: boolean;
  tool: Tool;
}

const HTTP_METHODS = ["get", "post", "put", "delete", "patch"] as const;
const SPEC_FETCH_TIMEOUT_MS = 30_000;
const DEFAULT_CALL_TIMEOUT_SECONDS = 1800;
const MAX_DESCRIPTION = 4000;
const MAX_REF_DEPTH = 32;

function sanitizeName(raw: string): string {
  const cleaned = raw.replace(/[^A-Za-z0-9_]/g, "_").slice(0, 64);
  return cleaned || "operation";
}

/** Resolve the spec location: a .json URL is used as given, anything else
 * gets /openapi.json appended (the FastAPI convention). */
export function resolveSpecUrl(base: string): string {
  const url = new URL(base);
  if (url.pathname.endsWith(".json")) {
    return url.toString();
  }
  url.pathname = url.pathname.replace(/\/+$/, "") + "/openapi.json";
  return url.toString();
}

/** Inline local $refs so every tool schema is self-contained. Depth-capped so
 * a recursive schema cannot hang the connection. */
function inlineRefs(node: unknown, spec: Json, depth = 0): unknown {
  if (depth > MAX_REF_DEPTH) {
    return {};
  }
  if (Array.isArray(node)) {
    return node.map((v) => inlineRefs(v, spec, depth + 1));
  }
  if (node && typeof node === "object") {
    const obj = node as Json;
    const ref = obj["$ref"];
    if (typeof ref === "string" && ref.startsWith("#/")) {
      let target: unknown = spec;
      for (const part of ref.slice(2).split("/")) {
        target = ((target as Json) || {})[part];
      }
      return inlineRefs(target ?? {}, spec, depth + 1);
    }
    const out: Json = {};
    for (const [k, v] of Object.entries(obj)) {
      if (k === "$ref") continue;
      out[k] = inlineRefs(v, spec, depth + 1);
    }
    return out;
  }
  return node;
}

function oneLineJoin(...parts: Array<string | undefined>): string {
  const cleaned = parts
    .map((p) => (p || "").split(/\s+/).join(" ").trim())
    .filter(Boolean);
  // FastAPI often repeats the summary at the start of the description.
  if (cleaned.length === 2 && cleaned[1].startsWith(cleaned[0])) {
    return cleaned[1];
  }
  return cleaned.join("\n\n");
}

function buildOperations(spec: Json): Operation[] {
  const ops: Operation[] = [];
  const seen = new Set<string>();
  const paths = (spec["paths"] as Json) || {};
  for (const [path, rawOps] of Object.entries(paths)) {
    const methods = (rawOps as Json) || {};
    for (const method of HTTP_METHODS) {
      const op = methods[method] as Json | undefined;
      if (!op) continue;

      const opId =
        (op["operationId"] as string) ||
        `${method}_${path}`.replace(/\//g, "_");
      let toolName = sanitizeName(opId);
      while (seen.has(toolName)) {
        toolName = sanitizeName(`${toolName}_${method}`);
        if (seen.has(toolName)) toolName = sanitizeName(`${toolName}_x`);
      }
      seen.add(toolName);

      const requestBody = (op["requestBody"] as Json) || {};
      const content = (requestBody["content"] as Json) || {};
      const jsonContent = (content["application/json"] as Json) || {};
      const bodySchema = jsonContent["schema"]
        ? (inlineRefs(jsonContent["schema"], spec) as Json)
        : undefined;

      const properties: Json = {
        ...((bodySchema?.["properties"] as Json) || {}),
      };
      const required = new Set<string>(
        (bodySchema?.["required"] as string[]) || [],
      );

      const params: Operation["params"] = [];
      for (const raw of (op["parameters"] as Json[]) || []) {
        const prm = (inlineRefs(raw, spec) as Json) || {};
        const name = prm["name"] as string;
        if (!name) continue;
        const where = (prm["in"] as string) || "query";
        if (where === "header" || where === "cookie") continue;
        params.push({
          name,
          in: where,
          required: Boolean(prm["required"]),
        });
        properties[name] = (prm["schema"] as Json) || { type: "string" };
        if (prm["description"] && !(properties[name] as Json)["description"]) {
          (properties[name] as Json)["description"] = prm["description"];
        }
        if (prm["required"]) required.add(name);
      }

      const description =
        oneLineJoin(
          op["summary"] as string,
          op["description"] as string,
        ).slice(0, MAX_DESCRIPTION) || toolName;

      ops.push({
        toolName,
        method,
        path,
        params,
        hasBody: Boolean(bodySchema),
        tool: {
          name: toolName,
          description,
          inputSchema: {
            type: "object" as const,
            properties: properties as { [key: string]: object },
            ...(required.size ? { required: [...required] } : {}),
          },
        },
      });
    }
  }
  return ops;
}

function upstreamHeaders(serverParams: ServerParameters): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(serverParams.headers || {}),
  };
  const authToken =
    serverParams.oauth_tokens?.access_token || serverParams.bearerToken;
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  return headers;
}

async function callUpstream(
  baseUrl: string,
  op: Operation,
  args: Json,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<{ text: string; isError: boolean }> {
  // Split arguments into path, query, and body the way the spec declares them.
  const payload: Json = {};
  for (const [k, v] of Object.entries(args || {})) {
    if (v !== null && v !== undefined) payload[k] = v;
  }
  let path = op.path;
  const query = new URLSearchParams();
  for (const prm of op.params) {
    if (!(prm.name in payload)) continue;
    const value = payload[prm.name];
    delete payload[prm.name];
    if (prm.in === "path") {
      path = path.replace(
        `{${prm.name}}`,
        encodeURIComponent(String(value)),
      );
    } else {
      query.set(prm.name, String(value));
    }
  }

  const method = op.method.toUpperCase();
  const sendBody = op.hasBody || method !== "GET";
  if (!sendBody) {
    // A GET with stray arguments: pass them as query rather than dropping them.
    for (const [k, v] of Object.entries(payload)) {
      query.set(k, String(v));
    }
  }

  const url = new URL(baseUrl.replace(/\/+$/, "") + path);
  for (const [k, v] of query) url.searchParams.set(k, v);

  const resp = await fetch(url, {
    method,
    headers,
    body: sendBody ? JSON.stringify(payload) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });

  const rawText = await resp.text();
  let text = rawText;
  try {
    text = JSON.stringify(JSON.parse(rawText), null, 2);
  } catch {
    // not JSON, return as-is
  }
  if (!resp.ok) {
    return {
      text: `HTTP ${resp.status} from upstream: ${text.slice(0, 4000)}`,
      isError: true,
    };
  }
  return { text, isError: false };
}

/**
 * Fetch the spec and stand up the in-process MCP server. Returns the client
 * side of the in-memory pair, ready for a Client (the pool) or a proxy bridge
 * (the inspector) to drive.
 */
export async function createOpenApiInProcessTransport(
  serverParams: ServerParameters,
): Promise<Transport> {
  if (!serverParams.url) {
    throw new Error(`OPENAPI server ${serverParams.name} has no URL`);
  }
  const baseUrl = transformDockerUrl(serverParams.url);
  const specUrl = resolveSpecUrl(baseUrl);
  const headers = upstreamHeaders(serverParams);
  const timeoutSeconds = Number(
    serverParams.env?.OPENAPI_TIMEOUT_SECONDS || DEFAULT_CALL_TIMEOUT_SECONDS,
  );
  const timeoutMs =
    (Number.isFinite(timeoutSeconds) && timeoutSeconds > 0
      ? timeoutSeconds
      : DEFAULT_CALL_TIMEOUT_SECONDS) * 1000;

  const resp = await fetch(specUrl, {
    headers,
    signal: AbortSignal.timeout(SPEC_FETCH_TIMEOUT_MS),
  });
  if (!resp.ok) {
    throw new Error(
      `OPENAPI server ${serverParams.name}: ${specUrl} returned HTTP ${resp.status}`,
    );
  }
  const spec = (await resp.json()) as Json;
  const operations = buildOperations(spec);
  if (operations.length === 0) {
    logger.warn(
      `OPENAPI server ${serverParams.name}: spec at ${specUrl} has no operations`,
    );
  }
  const byName = new Map(operations.map((op) => [op.toolName, op]));

  const info = (spec["info"] as Json) || {};
  const server = new Server(
    {
      name: (info["title"] as string) || `openapi-${serverParams.name}`,
      version: (info["version"] as string) || "1.0.0",
    },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: operations.map((op) => op.tool),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const op = byName.get(request.params.name);
    if (!op) {
      return {
        content: [
          { type: "text" as const, text: `Unknown tool: ${request.params.name}` },
        ],
        isError: true,
      };
    }
    try {
      const result = await callUpstream(
        baseUrl,
        op,
        (request.params.arguments as Json) || {},
        headers,
        timeoutMs,
      );
      return {
        content: [{ type: "text" as const, text: result.text }],
        isError: result.isError,
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text" as const,
            text: `Upstream call failed: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  });

  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);

  return clientTransport;
}

/**
 * The pool-facing wrapper: the in-process transport plus an ordinary Client,
 * matching the shape every other server type produces.
 */
export async function createOpenApiClient(
  serverParams: ServerParameters,
): Promise<{ client: Client; transport: Transport }> {
  const transport = await createOpenApiInProcessTransport(serverParams);

  const client = new Client(
    { name: "metamcp-client", version: "2.0.0" },
    {
      capabilities: {
        prompts: {},
        resources: { subscribe: true },
        tools: {},
      },
    },
  );

  return { client, transport };
}
