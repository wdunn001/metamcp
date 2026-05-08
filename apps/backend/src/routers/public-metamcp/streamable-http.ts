import { randomUUID } from "node:crypto";

import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";

import {
  ApiKeyAuthenticatedRequest,
  authenticateApiKey,
} from "@/middleware/api-key-oauth.middleware";
import { lookupEndpoint } from "@/middleware/lookup-endpoint-middleware";
import { rateLimitMiddleware } from "@/middleware/rate-limit.middleware";
import logger from "@/utils/logger";

import { negotiateResponseEncoding } from "../../lib/metamcp/codec/codec-compression";
import {
  extractCodecArgsMeta,
  loadVocabFromHeader,
} from "../../lib/metamcp/codec/codec-content";
import { negotiateStreamFormat } from "../../lib/metamcp/codec/codec-frame";
import {
  decodeCodecRequestBody,
  wrapResponseForCodec,
} from "../../lib/metamcp/codec/codec-transcode";
import { lookupVocabMap } from "../../lib/metamcp/codec/codec-vocab";
import { metaMcpServerPool } from "../../lib/metamcp/metamcp-server-pool";
import { SessionLifetimeManagerImpl } from "../../lib/session-lifetime-manager";

const streamableHttpRouter = express.Router();

// Codec opt-in raw-body parser. Only kicks in when the client posts
// `application/x-codec-msgpack` or `…-protobuf`. JSON-RPC traffic on
// the same routes is parsed by the SDK's existing JSON middleware
// and is byte-for-byte unchanged.
streamableHttpRouter.use(
  express.raw({
    type: [
      "application/x-codec-msgpack",
      "application/x-codec-protobuf",
    ],
    limit: "4mb",
  }),
);

// Session lifetime manager for StreamableHTTP sessions
const sessionManager =
  new SessionLifetimeManagerImpl<StreamableHTTPServerTransport>(
    "StreamableHTTP",
  );

// Cleanup function for a specific session
const cleanupSession = async (
  sessionId: string,
  transport?: StreamableHTTPServerTransport,
) => {
  logger.info(`Cleaning up StreamableHTTP session ${sessionId}`);

  try {
    // Use provided transport or get from session manager
    const sessionTransport = transport || sessionManager.getSession(sessionId);

    if (sessionTransport) {
      logger.info(`Closing transport for session ${sessionId}`);
      await sessionTransport.close();
      logger.info(`Transport cleaned up for session ${sessionId}`);
    } else {
      logger.info(`No transport found for session ${sessionId}`);
    }

    // Remove from session manager
    sessionManager.removeSession(sessionId);

    // Clean up MetaMCP server pool session
    await metaMcpServerPool.cleanupSession(sessionId);

    logger.info(`Session ${sessionId} cleanup completed successfully`);
  } catch (error) {
    logger.error(`Error during cleanup of session ${sessionId}:`, error);
    // Even if cleanup fails, remove the session from manager to prevent memory leaks
    sessionManager.removeSession(sessionId);
    logger.info(`Removed orphaned session ${sessionId} due to cleanup error`);
    throw error;
  }
};

// Health check endpoint to monitor sessions
streamableHttpRouter.get("/health/sessions", (req, res) => {
  const sessionIds = sessionManager.getSessionIds();
  const poolStatus = metaMcpServerPool.getPoolStatus();

  res.json({
    timestamp: new Date().toISOString(),
    streamableHttpSessions: {
      count: sessionIds.length,
      sessionIds: sessionIds,
    },
    metaMcpPoolStatus: poolStatus,
    totalActiveSessions: sessionIds.length + poolStatus.active,
  });
});

streamableHttpRouter.get(
  "/:endpoint_name/mcp",
  lookupEndpoint,
  authenticateApiKey,
  rateLimitMiddleware,
  async (req, res) => {
    // const authReq = req as ApiKeyAuthenticatedRequest;
    // const { namespaceUuid, endpointName } = authReq;
    const sessionId = req.headers["mcp-session-id"] as string;

    // logger.info(
    //   `Received GET message for public endpoint ${endpointName} -> namespace ${namespaceUuid} sessionId ${sessionId}`,
    // );

    try {
      logger.info(`Looking up existing session: ${sessionId}`);
      logger.info(`Available sessions:`, sessionManager.getSessionIds());

      const transport = sessionManager.getSession(sessionId);
      if (!transport) {
        logger.info(`Session ${sessionId} not found in session manager`);
        res.status(404).end("Session not found");
        return;
      } else {
        logger.info(`Found session ${sessionId}, handling request`);
        await transport.handleRequest(req, res);
      }
    } catch (error) {
      logger.error("Error in public endpoint /mcp route:", error);
      res.status(500).json(error);
    }
  },
);

streamableHttpRouter.post(
  "/:endpoint_name/mcp",
  lookupEndpoint,
  authenticateApiKey,
  rateLimitMiddleware,
  async (req, res) => {
    const authReq = req as ApiKeyAuthenticatedRequest;
    const { namespaceUuid, endpointName } = authReq;
    const sessionId = req.headers["mcp-session-id"] as string | undefined;

    // ── Codec negotiation ───────────────────────────────────────────
    // Request and response negotiation are INDEPENDENT:
    //   - Content-Type: application/x-codec-msgpack on the request
    //     means decode the inbound body as msgpack before handing
    //     to the SDK. Always pairs with the matching response format.
    //   - ?stream_format=… or Accept: application/x-codec-… means
    //     wrap the response so the SDK's JSON-RPC writes emit Codec
    //     frames. The inbound body can still be plain JSON — the
    //     client may want JSON-in / Codec-out for migration paths.
    // Pinning these two together (as the first version did) breaks
    // the JSON-in / Codec-out path: a JSON body fails msgpack-decode
    // before the SDK even sees it.
    const reqContentType = req.headers["content-type"] as string | undefined;
    const reqCodecFormat: ReturnType<typeof negotiateStreamFormat> =
      reqContentType?.includes("application/x-codec-msgpack")
        ? "msgpack"
        : reqContentType?.includes("application/x-codec-protobuf")
          ? "protobuf"
          : undefined;
    if (reqCodecFormat) {
      try {
        decodeCodecRequestBody(req, reqCodecFormat);
        // Spoof Content-Type so the SDK's StreamableHTTPServerTransport
        // (which validates against application/json) accepts the request
        // we just decoded. The body is now a parsed JS object — the SDK
        // sees what it would have seen if the client had posted JSON.
        // Same pattern as the Accept-header spoof below for the response
        // direction.
        req.headers["content-type"] = "application/json";
      } catch (error) {
        logger.error(
          `Codec request decode failed (${reqCodecFormat}):`,
          error,
        );
        res.status(400).json({
          jsonrpc: "2.0",
          id: null,
          error: {
            code: -32700,
            message: `Codec request body could not be decoded as ${reqCodecFormat}`,
          },
        });
        return;
      }
    }

    // ── Vocab map negotiation ───────────────────────────────────────
    // Optional X-Codec-Map: <url>;sha256=<hash> header. If present,
    // load + cache the map so detokenize/tokenize transforms can run
    // synchronously downstream. Per-request: the same client can
    // switch vocabs by changing the header.
    let vocabHash: string | undefined;
    const codecMapHeader = req.headers["x-codec-map"] as string | undefined;
    if (codecMapHeader) {
      try {
        const loaded = await loadVocabFromHeader(codecMapHeader);
        vocabHash = loaded?.hash;
      } catch (error) {
        logger.error(`X-Codec-Map header rejected:`, error);
        res.status(400).json({
          jsonrpc: "2.0",
          id: null,
          error: {
            code: -32600,
            message: `Invalid X-Codec-Map header: ${(error as Error).message}`,
          },
        });
        return;
      }
    }

    // ── Detokenize tools/call args carried as Codec ────────────────
    // If the request body is a tools/call whose `arguments` is a
    // Codec-encoded block, replace it with the detokenized JSON
    // object before the SDK ever sees it. The MCP server downstream
    // gets the same JSON it would have gotten without Codec — the
    // tokenizer/detokenizer pair is purely a wire-side optimization.
    const body = req.body as
      | { method?: string; params?: { arguments?: unknown } }
      | undefined;
    if (body?.method === "tools/call") {
      const meta = extractCodecArgsMeta(body.params?.arguments);
      if (meta) {
        const vocab = lookupVocabMap(meta.map_id);
        if (!vocab) {
          res.status(400).json({
            jsonrpc: "2.0",
            id: null,
            error: {
              code: -32600,
              message:
                `tools/call args reference vocab map ${meta.map_id} but it isn't cached. ` +
                `Send X-Codec-Map: <url>;sha256=${meta.map_id} alongside this request.`,
            },
          });
          return;
        }
        const text = vocab.detok.render(meta.ids, { partial: false });
        try {
          (body.params as { arguments: unknown }).arguments = JSON.parse(text);
        } catch (error) {
          res.status(400).json({
            jsonrpc: "2.0",
            id: null,
            error: {
              code: -32700,
              message: `Codec args detokenized to non-JSON text: ${(error as Error).message}`,
            },
          });
          return;
        }
      }
    }

    const respCodecFormat = negotiateStreamFormat(
      req.query as Record<string, unknown>,
      req.headers.accept as string | undefined,
    );
    if (respCodecFormat) {
      const acceptEncoding = req.headers["accept-encoding"] as
        | string
        | undefined;
      const codecEncoding = negotiateResponseEncoding(acceptEncoding);
      // Pass the vocab hash so wrapResponseForCodec runs the
      // CallToolResult content tokenizer on every response in this
      // request's lifecycle. Without a vocab, the wire still gets
      // reframed as msgpack but text content stays as-is.
      wrapResponseForCodec(res, respCodecFormat, codecEncoding, vocabHash);
      // The SDK's StreamableHTTPServerTransport runs its own Accept
      // negotiation against `application/json` + `text/event-stream`
      // and returns 406 for anything else. Spoof the header so the
      // SDK accepts the request — the wrapResponseForCodec layer
      // above will re-frame whatever bytes the SDK writes back into
      // the Codec wire format on the way out.
      req.headers.accept = "application/json, text/event-stream";
    }

    // Log authentication information for debugging
    logger.info(`POST /mcp request for endpoint: ${endpointName}`);
    logger.info(`Authentication method: ${authReq.authMethod || "none"}`);
    logger.info(`Session ID: ${sessionId || "new session"}`);

    if (!sessionId) {
      try {
        logger.info(
          `New public endpoint StreamableHttp connection request for ${endpointName} -> namespace ${namespaceUuid}`,
        );

        // Generate session ID upfront
        const newSessionId = randomUUID();
        logger.info(
          `Generated new session ID: ${newSessionId} for endpoint: ${endpointName}`,
        );

        // Get or create MetaMCP server instance from the pool
        const mcpServerInstance = await metaMcpServerPool.getServer(
          newSessionId,
          namespaceUuid,
        );
        if (!mcpServerInstance) {
          throw new Error("Failed to get MetaMCP server instance from pool");
        }

        logger.info(
          `Using MetaMCP server instance for public endpoint session ${newSessionId} (endpoint: ${endpointName})`,
        );

        // Create transport with the predetermined session ID
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => newSessionId,
          onsessioninitialized: async (sessionId) => {
            try {
              logger.info(`Session initialized for sessionId: ${sessionId}`);
            } catch (error) {
              logger.error(
                `Error initializing public endpoint session ${sessionId}:`,
                error,
              );
            }
          },
        });

        // Note: Cleanup is handled explicitly via DELETE requests
        // StreamableHTTP is designed to persist across multiple requests
        logger.info("Created public endpoint StreamableHttp transport");
        logger.info(
          `Session ${newSessionId} will be cleaned up when DELETE request is received`,
        );

        // Store transport reference
        sessionManager.addSession(newSessionId, transport);

        logger.info(
          `Public Endpoint Client <-> Proxy sessionId: ${newSessionId} for endpoint ${endpointName} -> namespace ${namespaceUuid}`,
        );
        logger.info(`Stored transport for sessionId: ${newSessionId}`);
        logger.info(`Current stored sessions:`, sessionManager.getSessionIds());
        logger.info(
          `Total active sessions: ${sessionManager.getSessionCount()}`,
        );

        // Connect the server to the transport before handling the request
        await mcpServerInstance.server.connect(transport);

        // Now handle the request - server is guaranteed to be ready
        await transport.handleRequest(req, res);
      } catch (error) {
        logger.error("Error in public endpoint /mcp POST route:", error);

        // Provide more detailed error information
        const errorMessage =
          error instanceof Error ? error.message : "Unknown error";
        res.status(500).json({
          error: "Internal server error",
          message: errorMessage,
          endpoint: endpointName,
          timestamp: new Date().toISOString(),
        });
      }
    } else {
      // logger.info(
      //   `Received POST message for public endpoint ${endpointName} -> namespace ${namespaceUuid} sessionId ${sessionId}`,
      // );
      logger.info(`Available session IDs:`, sessionManager.getSessionIds());
      logger.info(`Looking for sessionId: ${sessionId}`);
      try {
        logger.info(`Looking up existing session: ${sessionId}`);
        logger.info(`Available sessions:`, sessionManager.getSessionIds());

        const transport = sessionManager.getSession(sessionId);
        if (!transport) {
          logger.error(
            `Transport not found for sessionId ${sessionId}. Available sessions:`,
            sessionManager.getSessionIds(),
          );
          res.status(404).json({
            error: "Session not found",
            message: `Transport not found for sessionId ${sessionId}`,
            available_sessions: sessionManager.getSessionIds(),
            timestamp: new Date().toISOString(),
          });
        } else {
          logger.info(`Found session ${sessionId}, handling request`);
          await transport.handleRequest(req, res);
        }
      } catch (error) {
        logger.error("Error in public endpoint /mcp route:", error);

        const errorMessage =
          error instanceof Error ? error.message : "Unknown error";
        res.status(500).json({
          error: "Internal server error",
          message: errorMessage,
          session_id: sessionId,
          endpoint: endpointName,
          timestamp: new Date().toISOString(),
        });
      }
    }
  },
);

streamableHttpRouter.delete(
  "/:endpoint_name/mcp",
  lookupEndpoint,
  authenticateApiKey,
  rateLimitMiddleware,
  async (req, res) => {
    const authReq = req as ApiKeyAuthenticatedRequest;
    const { namespaceUuid, endpointName } = authReq;
    const sessionId = req.headers["mcp-session-id"] as string | undefined;

    logger.info(
      `Received DELETE message for public endpoint ${endpointName} -> namespace ${namespaceUuid} sessionId ${sessionId}`,
    );

    if (sessionId) {
      try {
        logger.info(`Starting cleanup for session ${sessionId}`);
        logger.info(
          `Available sessions before cleanup:`,
          sessionManager.getSessionIds(),
        );

        await cleanupSession(sessionId);

        logger.info(
          `Public endpoint session ${sessionId} cleaned up successfully`,
        );
        logger.info(
          `Available sessions after cleanup:`,
          sessionManager.getSessionIds(),
        );

        res.status(200).json({
          message: "Session cleaned up successfully",
          sessionId: sessionId,
          remainingSessions: sessionManager.getSessionIds(),
        });
      } catch (error) {
        logger.error("Error in public endpoint /mcp DELETE route:", error);
        res.status(500).json({
          error: "Cleanup failed",
          message: error instanceof Error ? error.message : "Unknown error",
          sessionId: sessionId,
        });
      }
    } else {
      res.status(400).json({
        error: "Missing sessionId",
        message: "sessionId header is required for cleanup",
      });
    }
  },
);

// Initialize automatic cleanup timer using session manager
sessionManager.startCleanupTimer(async (sessionId, transport) => {
  await cleanupSession(sessionId, transport);
});

export default streamableHttpRouter;
