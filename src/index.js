/**
 * Google MCP Server — Entry Point
 *
 * Transports:
 *   1. StreamableHTTP  POST /mcp          (modern MCP clients)
 *   2. SSE             GET  /sse          (legacy SSE clients)
 *                      POST /messages     (SSE message endpoint)
 *   3. Stdio                              (local CLI / Claude Desktop)
 *
 * OAuth:
 *   GET  /oauth/start?alias=<name>
 *   GET  /oauth/callback
 *   GET  /oauth/revoke?alias=<name>
 *   GET  /health
 */

import "dotenv/config";
import express from "express";
import { randomUUID } from "crypto";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { registerAllTools } from "./tools/index.js";
import { createOAuthApp } from "./oauthServer.js";

// ── Validate env vars ──────────────────────────────────────────────────────
const REQUIRED_ENV = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "OAUTH_REDIRECT_URI"];
const missing = REQUIRED_ENV.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error(`❌  Missing required environment variables: ${missing.join(", ")}`);
  process.exit(1);
}

// ── Factory: create a fresh MCP server instance with all tools registered ──
function createMcpServer() {
  const server = new McpServer({
    name: "google-analytics-search-console",
    version: "1.0.0",
    description: "GA4 + Google Search Console for multiple Google accounts.",
  });
  registerAllTools(server);
  return server;
}

// ── Express app ────────────────────────────────────────────────────────────
const app = createOAuthApp(); // includes /health, /oauth/start, /oauth/callback

// Session store for SSE and StreamableHTTP connections
const sseSessions = new Map();

// ── StreamableHTTP transport  POST /mcp ────────────────────────────────────
app.post("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];

  let transport;

  if (sessionId && sseSessions.has(sessionId)) {
    transport = sseSessions.get(sessionId);
  } else if (!sessionId && isInitializeRequest(req.body)) {
    // New session
    const server = createMcpServer();
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (id) => {
        sseSessions.set(id, transport);
      },
    });
    transport.onclose = () => {
      if (transport.sessionId) sseSessions.delete(transport.sessionId);
    };
    await server.connect(transport);
  } else {
    res.status(400).json({ error: "Bad request: missing or unknown session" });
    return;
  }

  await transport.handleRequest(req, res, req.body);
});

// Handle GET/DELETE for StreamableHTTP session management
app.get("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (!sessionId || !sseSessions.has(sessionId)) {
    res.status(404).json({ error: "Session not found" });
    return;
  }
  await sseSessions.get(sessionId).handleRequest(req, res);
});

app.delete("/mcp", async (req, res) => {
  const sessionId = req.headers["mcp-session-id"];
  if (sessionId && sseSessions.has(sessionId)) {
    sseSessions.get(sessionId).close();
    sseSessions.delete(sessionId);
  }
  res.status(200).json({ ok: true });
});

// ── SSE transport  GET /sse ────────────────────────────────────────────────
app.get("/sse", async (req, res) => {
  const server = createMcpServer();
  const transport = new SSEServerTransport("/messages", res);
  sseSessions.set(transport.sessionId, transport);
  transport.onclose = () => sseSessions.delete(transport.sessionId);
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId;
  const transport = sseSessions.get(sessionId);
  if (!transport) {
    res.status(404).json({ error: "SSE session not found" });
    return;
  }
  await transport.handlePostMessage(req, res, req.body);
});

// ── Start HTTP server ──────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || "3000", 10);

app.listen(PORT, () => {
  console.error(`🚀  Google MCP Server running on port ${PORT}`);
  console.error(`    StreamableHTTP: POST /mcp`);
  console.error(`    SSE:            GET  /sse  |  POST /messages`);
  console.error(`    OAuth:          GET  /oauth/start?alias=<name>`);
  console.error(`    Health:         GET  /health`);
});

// ── Stdio transport (local use) ────────────────────────────────────────────
if (process.env.MCP_STDIO === "true") {
  const stdioServer = createMcpServer();
  const transport = new StdioServerTransport();
  await stdioServer.connect(transport);
  console.error("✅  Stdio transport also active");
}
