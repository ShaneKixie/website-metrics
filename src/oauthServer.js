/**
 * OAuth HTTP server (Express)
 *
 * Routes:
 *   GET /health                      — health check + connected accounts list
 *   GET /oauth/start?alias=<name>    — redirect to Google consent screen
 *   GET /oauth/callback              — exchange code, store token, show success page
 *   GET /oauth/revoke?alias=<name>   — remove stored tokens for an account
 */

import express from "express";
import { getAuthUrl, exchangeCode } from "./auth.js";
import { listAliases, removeToken, getTokenFilePath } from "./tokenStore.js";

export function createOAuthApp() {
  const app = express();

  // Parse JSON bodies — must be before any routes that read req.body
  app.use(express.json());

  // ── Health ───────────────────────────────────────────────────────────────
  app.get("/health", (_req, res) => {
    res.json({
      status: "ok",
      server: "google-mcp-server",
      authenticated_accounts: listAliases(),
      token_file: getTokenFilePath(),
      transports: ["StreamableHTTP (POST /mcp)", "SSE (GET /sse)"],
    });
  });

  // ── Start OAuth Flow ─────────────────────────────────────────────────────
  app.get("/oauth/start", (req, res) => {
    const alias = req.query.alias?.trim();
    if (!alias) {
      return res.status(400).send(
        "Missing <code>?alias=</code> parameter. Example: <a href='/oauth/start?alias=account1'>/oauth/start?alias=account1</a>"
      );
    }
    const url = getAuthUrl(alias);
    res.redirect(url);
  });

  // ── OAuth Callback ───────────────────────────────────────────────────────
  app.get("/oauth/callback", async (req, res) => {
    const { code, state: alias, error } = req.query;

    if (error) {
      return res.status(400).send(`<h2>❌ OAuth Error</h2><p>${error}</p>`);
    }
    if (!code || !alias) {
      return res.status(400).send("Missing <code>code</code> or <code>state</code> parameter.");
    }

    try {
      await exchangeCode(alias, code);
      const all = listAliases();
      res.send(`
        <!DOCTYPE html>
        <html>
        <head><title>Account Connected</title>
        <style>
          body { font-family: system-ui, sans-serif; max-width: 600px; margin: 4rem auto; padding: 0 1rem; }
          .success { color: #16a34a; }
          code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }
          ul { line-height: 2; }
        </style>
        </head>
        <body>
          <h2 class="success">✅ Account "${alias}" connected!</h2>
          <p>This Google account has been authenticated. You can close this tab.</p>
          <h3>All connected accounts (${all.length})</h3>
          <ul>${all.map((a) => `<li><code>${a}</code></li>`).join("")}</ul>
          <hr>
          <p>MCP endpoints:</p>
          <ul>
            <li>StreamableHTTP: <code>POST /mcp</code></li>
            <li>SSE: <code>GET /sse</code></li>
          </ul>
        </body>
        </html>
      `);
    } catch (err) {
      res.status(500).send(`<h2>❌ Error</h2><pre>${err.message}</pre>`);
    }
  });

  // ── Revoke Account ───────────────────────────────────────────────────────
  app.get("/oauth/revoke", (req, res) => {
    const alias = req.query.alias?.trim();
    if (!alias) return res.status(400).send("Missing ?alias= parameter.");
    removeToken(alias);
    res.json({ removed: alias, remaining: listAliases() });
  });

  return app;
}
