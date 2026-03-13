/**
 * Google MCP Server — Entry Point
 *
 * Starts two things concurrently:
 *   1. MCP server over stdio (consumed by AI clients like Claude Desktop)
 *   2. Express HTTP server for the OAuth2 flow (consumed by a browser)
 */

import "dotenv/config";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerAllTools } from "./tools/index.js";
import { createOAuthApp } from "./oauthServer.js";

// ── Validate required environment variables ────────────────────────────────
const REQUIRED_ENV = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "OAUTH_REDIRECT_URI"];
const missing = REQUIRED_ENV.filter((k) => !process.env[k]);
if (missing.length > 0) {
  console.error(`❌  Missing required environment variables: ${missing.join(", ")}`);
  console.error(`    Copy .env.example → .env and fill in the values.`);
  process.exit(1);
}

// ── MCP Server ─────────────────────────────────────────────────────────────
const server = new McpServer({
  name: "google-analytics-search-console",
  version: "1.0.0",
  description:
    "Access Google Analytics 4 and Google Search Console data across multiple Google accounts.",
});

registerAllTools(server);

// ── OAuth HTTP Server ──────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || "3000", 10);
const oauthApp = createOAuthApp();

oauthApp.listen(PORT, () => {
  console.error(`🔐  OAuth server:  http://localhost:${PORT}`);
  console.error(`    Connect an account: /oauth/start?alias=account1`);
  console.error(`    Health check:       /health`);
});

// ── MCP Stdio Transport ────────────────────────────────────────────────────
const transport = new StdioServerTransport();
await server.connect(transport);

console.error("✅  Google MCP Server running (GA4 + Search Console, stdio transport)");
