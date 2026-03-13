/**
 * Google Search Console MCP tools
 *
 * gsc_get_performance  — search analytics (clicks, impressions, CTR, position)
 * gsc_inspect_url      — URL indexing status
 * gsc_submit_url       — submit URL for indexing / deletion
 * gsc_list_sitemaps    — list submitted sitemaps
 */

import {
  getPerformance,
  formatPerformanceRows,
  getUrlStatus,
  submitUrl,
  listSitemaps,
} from "../gscClient.js";

export function registerGscTools(server) {
  // ── Search Performance ─────────────────────────────────────────────────────
  server.tool(
    "gsc_get_performance",
    "Fetch Google Search Console search analytics: clicks, impressions, CTR, and average position.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias (from list_accounts)." },
        site_url: { type: "string", description: 'Verified property URL, e.g. "https://example.com/".' },
        start_date: { type: "string", description: "Start date YYYY-MM-DD." },
        end_date: { type: "string", description: "End date YYYY-MM-DD." },
        dimensions: {
          type: "array",
          items: { type: "string" },
          description: 'Dimensions to group by: "query", "page", "country", "device", "date". Default: ["query"].',
        },
        row_limit: { type: "number", description: "Rows to return (default 1000, max 25000)." },
        search_type: { type: "string", description: '"web" | "image" | "video" | "news". Default: "web".' },
        filter_query: { type: "string", description: "Filter rows to those containing this query string." },
        filter_page: { type: "string", description: "Filter rows to those matching this page URL." },
        filter_country: { type: "string", description: "Filter rows to a specific country code (e.g. USA)." },
        filter_device: { type: "string", description: "Filter rows to a device: DESKTOP | MOBILE | TABLET." },
      },
      required: ["account", "site_url", "start_date", "end_date"],
    },
    async ({
      account,
      site_url,
      start_date,
      end_date,
      dimensions = ["query"],
      row_limit = 1000,
      search_type = "web",
      filter_query,
      filter_page,
      filter_country,
      filter_device,
    }) => {
      const filters = [];
      if (filter_query) filters.push({ dimension: "query", operator: "contains", expression: filter_query });
      if (filter_page) filters.push({ dimension: "page", operator: "contains", expression: filter_page });
      if (filter_country) filters.push({ dimension: "country", operator: "equals", expression: filter_country });
      if (filter_device) filters.push({ dimension: "device", operator: "equals", expression: filter_device });

      const rows = await getPerformance(account, site_url, {
        startDate: start_date,
        endDate: end_date,
        dimensions,
        rowLimit: row_limit,
        searchType: search_type,
        dimensionFilterGroups: filters.length ? [{ filters }] : [],
      });

      const table = formatPerformanceRows(rows, dimensions);
      return {
        content: [{
          type: "text",
          text: `GSC Performance: ${site_url} (${account})\n${start_date} → ${end_date}\n\n${table}`,
        }],
      };
    }
  );

  // ── URL Inspection ─────────────────────────────────────────────────────────
  server.tool(
    "gsc_inspect_url",
    "Inspect a URL's indexing status in Google Search Console.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias." },
        url: { type: "string", description: "Full URL to inspect." },
      },
      required: ["account", "url"],
    },
    async ({ account, url }) => {
      const result = await getUrlStatus(account, url);
      const index = result.inspectionResult?.indexStatusResult || {};
      const mobile = result.inspectionResult?.mobileUsabilityResult || {};
      const rich = result.inspectionResult?.richResultsResult || {};

      const lines = [
        `URL: ${url}`,
        `Account: ${account}`,
        ``,
        `── Indexing ──────────────────────`,
        `Coverage State:    ${index.coverageState || "unknown"}`,
        `Indexing State:    ${index.indexingState || "unknown"}`,
        `Last Crawl Time:   ${index.lastCrawlTime || "N/A"}`,
        `Crawled As:        ${index.crawledAs || "N/A"}`,
        `Robots.txt State:  ${index.robotsTxtState || "N/A"}`,
        `Google Canonical:  ${index.googleCanonical || "N/A"}`,
        `User Canonical:    ${index.userCanonical || "N/A"}`,
        ``,
        `── Mobile Usability ──────────────`,
        `Verdict: ${mobile.verdict || "N/A"}`,
        ...(mobile.issues || []).map((i) => `  • ${i.issueMessage}`),
        ``,
        `── Rich Results ─────────────────`,
        `Verdict: ${rich.verdict || "N/A"}`,
        ...(rich.detectedItems || []).map(
          (i) => `  • ${i.richResultType}: ${i.items?.length || 0} items`
        ),
      ];

      return { content: [{ type: "text", text: lines.join("\n") }] };
    }
  );

  // ── Submit URL ─────────────────────────────────────────────────────────────
  server.tool(
    "gsc_submit_url",
    "Submit a URL to Google for indexing or notify Google of a deletion.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias." },
        url: { type: "string", description: "Full URL to submit." },
        type: { type: "string", description: '"URL_UPDATED" (default) or "URL_DELETED".' },
      },
      required: ["account", "url"],
    },
    async ({ account, url, type = "URL_UPDATED" }) => {
      const result = await submitUrl(account, url, type);
      return {
        content: [{
          type: "text",
          text: `Submitted "${url}" (${type}) for account "${account}".\n${JSON.stringify(result, null, 2)}`,
        }],
      };
    }
  );

  // ── List Sitemaps ──────────────────────────────────────────────────────────
  server.tool(
    "gsc_list_sitemaps",
    "List all submitted sitemaps for a Search Console property.",
    {
      type: "object",
      properties: {
        account: { type: "string", description: "Account alias." },
        site_url: { type: "string", description: "Verified property URL." },
      },
      required: ["account", "site_url"],
    },
    async ({ account, site_url }) => {
      const sitemaps = await listSitemaps(account, site_url);
      if (sitemaps.length === 0) {
        return { content: [{ type: "text", text: `No sitemaps found for ${site_url}.` }] };
      }
      const list = sitemaps
        .map((s) =>
          `• ${s.path}  type:${s.type}  downloaded:${s.lastDownloaded || "N/A"}  warnings:${s.warnings || 0}  errors:${s.errors || 0}`
        )
        .join("\n");
      return { content: [{ type: "text", text: `Sitemaps for ${site_url}:\n${list}` }] };
    }
  );
}
