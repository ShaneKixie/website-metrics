/**
 * Account management tools
 * - list_accounts         (aliases with stored tokens)
 * - list_gsc_sites        (GSC properties per account)
 * - list_ga4_accounts     (GA4 accounts per Google account)
 * - list_ga4_properties   (GA4 properties for a GA4 account)
 */

import { listAliases } from "../auth.js";
import { listSites } from "../gscClient.js";
import { listAccounts, listProperties } from "../ga4Client.js";

export function registerAccountTools(server) {
  // ── List authenticated aliases ─────────────────────────────────────────────
  server.tool(
    "list_accounts",
    "List all Google accounts that have been authenticated with this MCP server.",
    {},
    async () => {
      const aliases = listAliases();
      if (aliases.length === 0) {
        return {
          content: [{
            type: "text",
            text: "No accounts authenticated yet. Visit /oauth/start?alias=<name> to connect an account.",
          }],
        };
      }
      return {
        content: [{
          type: "text",
          text: `Authenticated accounts (${aliases.length}): ${aliases.join(", ")}`,
        }],
      };
    }
  );

  // ── GSC: List sites ────────────────────────────────────────────────────────
  server.tool(
    "list_gsc_sites",
    "List all verified Google Search Console properties for a given account.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias (from list_accounts)." },
      },
      required: ["account"],
    },
    async ({ account }) => {
      const sites = await listSites(account);
      if (sites.length === 0) {
        return { content: [{ type: "text", text: `No verified GSC sites found for "${account}".` }] };
      }
      const list = sites.map((s) => `• ${s.siteUrl}  [${s.permissionLevel}]`).join("\n");
      return { content: [{ type: "text", text: `GSC sites for ${account}:\n${list}` }] };
    }
  );

  // ── GA4: List GA4 accounts ─────────────────────────────────────────────────
  server.tool(
    "list_ga4_accounts",
    "List all Google Analytics 4 accounts accessible to a given Google account.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias (from list_accounts)." },
      },
      required: ["account"],
    },
    async ({ account }) => {
      const accounts = await listAccounts(account);
      if (accounts.length === 0) {
        return { content: [{ type: "text", text: `No GA4 accounts found for "${account}".` }] };
      }
      const list = accounts.map((a) => `• ${a.name}  display: ${a.displayName}`).join("\n");
      return { content: [{ type: "text", text: `GA4 accounts for ${account}:\n${list}` }] };
    }
  );

  // ── GA4: List GA4 properties ───────────────────────────────────────────────
  server.tool(
    "list_ga4_properties",
    "List all GA4 properties under a GA4 account. Use list_ga4_accounts first to get the account name.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias (from list_accounts)." },
        ga4_account: {
          type: "string",
          description: 'GA4 account name, e.g. "accounts/123456" (from list_ga4_accounts).',
        },
      },
      required: ["account", "ga4_account"],
    },
    async ({ account, ga4_account }) => {
      const props = await listProperties(account, ga4_account);
      if (props.length === 0) {
        return { content: [{ type: "text", text: `No GA4 properties found under ${ga4_account}.` }] };
      }
      const list = props
        .map((p) => `• ${p.name}  display: ${p.displayName}  type: ${p.propertyType}`)
        .join("\n");
      return { content: [{ type: "text", text: `GA4 properties for ${ga4_account}:\n${list}` }] };
    }
  );
}
