/**
 * Google Analytics 4 MCP tools
 *
 * ga4_traffic_overview       — overall sessions, users, pageviews, bounce rate
 * ga4_traffic_by_channel     — sessions broken down by channel group
 * ga4_traffic_by_source      — sessions broken down by source/medium
 * ga4_traffic_over_time      — daily sessions/users/pageviews trend
 * ga4_top_pages              — most viewed pages
 * ga4_landing_pages          — top landing pages with sessions + conversions
 * ga4_top_events             — top events by count
 * ga4_conversions            — conversion events with rates
 * ga4_audience_by_country    — users by country
 * ga4_audience_by_device     — users by device category
 * ga4_audience_by_browser    — users by browser
 * ga4_new_vs_returning       — new vs returning user split
 * ga4_realtime               — active users right now
 * ga4_custom_report          — custom dimensions/metrics
 */

import {
  getTrafficOverview,
  getTrafficByChannel,
  getTrafficBySource,
  getTrafficOverTime,
  getTopPages,
  getLandingPages,
  getTopEvents,
  getConversions,
  getAudienceByCountry,
  getAudienceByDevice,
  getAudienceByBrowser,
  getNewVsReturning,
  getRealtimeUsers,
  runCustomReport,
  formatReport,
} from "../ga4Client.js";

// Shared param definitions reused across tools
const BASE_PARAMS = {
  account: { type: "string", description: "Account alias (from list_accounts)." },
  property_id: {
    type: "string",
    description: 'GA4 property ID — numeric only, e.g. "123456789" (from list_ga4_properties, strip "properties/" prefix).',
  },
  start_date: { type: "string", description: "Start date YYYY-MM-DD or relative value like \"7daysAgo\", \"30daysAgo\", \"yesterday\"." },
  end_date: { type: "string", description: "End date YYYY-MM-DD or \"today\", \"yesterday\"." },
};

function makeBase(extra = {}) {
  return {
    type: "object",
    properties: { ...BASE_PARAMS, ...extra },
    required: ["account", "property_id", "start_date", "end_date"],
  };
}

function text(t) {
  return { content: [{ type: "text", text: t }] };
}

export function registerGa4Tools(server) {

  // ── Traffic Overview ───────────────────────────────────────────────────────
  server.tool(
    "ga4_traffic_overview",
    "Get overall GA4 traffic summary: sessions, users, new users, pageviews, bounce rate, average session duration, and engagement rate.",
    makeBase(),
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getTrafficOverview(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Traffic Overview — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Traffic by Channel ─────────────────────────────────────────────────────
  server.tool(
    "ga4_traffic_by_channel",
    "Break down GA4 sessions by default channel group (Organic Search, Direct, Referral, Email, Paid Search, Social, etc.).",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTrafficByChannel(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Traffic by Channel — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Traffic by Source/Medium ───────────────────────────────────────────────
  server.tool(
    "ga4_traffic_by_source",
    "Break down GA4 sessions by source and medium (e.g. google/organic, direct/none, newsletter/email).",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTrafficBySource(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Traffic by Source/Medium — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Traffic Over Time ──────────────────────────────────────────────────────
  server.tool(
    "ga4_traffic_over_time",
    "Get a daily trend of GA4 sessions, users, and pageviews over a date range.",
    makeBase(),
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getTrafficOverTime(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Daily Traffic Trend — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Top Pages ──────────────────────────────────────────────────────────────
  server.tool(
    "ga4_top_pages",
    "Get the most-viewed pages in GA4 with pageviews, users, session duration, bounce rate, and engagement rate.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTopPages(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Top Pages — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Landing Pages ──────────────────────────────────────────────────────────
  server.tool(
    "ga4_landing_pages",
    "Get top landing pages in GA4 with sessions, new users, bounce rate, engagement, and conversions.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getLandingPages(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Landing Pages — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Top Events ─────────────────────────────────────────────────────────────
  server.tool(
    "ga4_top_events",
    "List the most frequently triggered GA4 events with event count and users.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTopEvents(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Top Events — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Conversions ────────────────────────────────────────────────────────────
  server.tool(
    "ga4_conversions",
    "Get GA4 conversion events with total conversions, users, sessions, and conversion rate.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getConversions(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Conversions — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Audience by Country ────────────────────────────────────────────────────
  server.tool(
    "ga4_audience_by_country",
    "Break down GA4 users by country.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 25)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getAudienceByCountry(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Audience by Country — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Audience by Device ─────────────────────────────────────────────────────
  server.tool(
    "ga4_audience_by_device",
    "Break down GA4 users by device category (desktop, mobile, tablet).",
    makeBase(),
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getAudienceByDevice(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Audience by Device — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Audience by Browser ────────────────────────────────────────────────────
  server.tool(
    "ga4_audience_by_browser",
    "Break down GA4 users by browser.",
    makeBase({ row_limit: { type: "number", description: "Max rows (default 15)." } }),
    async ({ account, property_id, start_date, end_date, row_limit = 15 }) => {
      const data = await getAudienceByBrowser(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Audience by Browser — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── New vs Returning ───────────────────────────────────────────────────────
  server.tool(
    "ga4_new_vs_returning",
    "Compare new vs returning users in GA4 — users, sessions, pageviews, session duration, engagement rate.",
    makeBase(),
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getNewVsReturning(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 New vs Returning — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  // ── Realtime ───────────────────────────────────────────────────────────────
  server.tool(
    "ga4_realtime",
    "Get the number of active users on the site right now, broken down by country and device.",
    {
      type: "object",
      properties: {
        account: BASE_PARAMS.account,
        property_id: BASE_PARAMS.property_id,
      },
      required: ["account", "property_id"],
    },
    async ({ account, property_id }) => {
      const data = await getRealtimeUsers(account, property_id);
      return text(formatReport(data, `GA4 Realtime Active Users — property ${property_id} (${account})`));
    }
  );

  // ── Custom Report ──────────────────────────────────────────────────────────
  server.tool(
    "ga4_custom_report",
    "Run a fully custom GA4 report with any combination of dimensions and metrics from the GA4 Data API.",
    {
      type: "object",
      properties: {
        ...BASE_PARAMS,
        dimensions: {
          type: "array",
          items: { type: "string" },
          description: 'GA4 dimension names, e.g. ["pagePath", "country", "deviceCategory", "sessionSource"].',
        },
        metrics: {
          type: "array",
          items: { type: "string" },
          description: 'GA4 metric names, e.g. ["sessions", "totalUsers", "screenPageViews", "bounceRate", "conversions"].',
        },
        row_limit: { type: "number", description: "Max rows (default 1000)." },
        order_by_metric: {
          type: "string",
          description: "Sort descending by this metric name. Optional.",
        },
      },
      required: ["account", "property_id", "start_date", "end_date", "dimensions", "metrics"],
    },
    async ({
      account,
      property_id,
      start_date,
      end_date,
      dimensions,
      metrics,
      row_limit = 1000,
      order_by_metric = null,
    }) => {
      const data = await runCustomReport(
        account,
        property_id,
        start_date,
        end_date,
        dimensions,
        metrics,
        row_limit,
        order_by_metric
      );
      return text(formatReport(data, `GA4 Custom Report — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );
}
