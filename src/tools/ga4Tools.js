import { z } from "zod";
import {
  getTrafficOverview, getTrafficByChannel, getTrafficBySource, getTrafficOverTime,
  getTopPages, getLandingPages, getTopEvents, getConversions,
  getAudienceByCountry, getAudienceByDevice, getAudienceByBrowser,
  getNewVsReturning, getRealtimeUsers, runCustomReport, formatReport,
} from "../ga4Client.js";

const BASE = {
  account: z.string().describe("Account alias (from list_accounts)."),
  property_id: z.string().describe('GA4 numeric property ID, e.g. "123456789" (from list_ga4_properties, strip "properties/" prefix).'),
  start_date: z.string().describe('Start date YYYY-MM-DD or relative e.g. "7daysAgo", "30daysAgo", "yesterday".'),
  end_date: z.string().describe('End date YYYY-MM-DD or "today", "yesterday".'),
};

function text(t) { return { content: [{ type: "text", text: t }] }; }

export function registerGa4Tools(server) {
  server.tool("ga4_traffic_overview",
    "Get overall GA4 traffic summary: sessions, users, new users, pageviews, bounce rate, average session duration, and engagement rate.",
    BASE,
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getTrafficOverview(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Traffic Overview — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_traffic_by_channel",
    "Break down GA4 sessions by default channel group (Organic Search, Direct, Referral, Email, Paid Search, Social, etc.).",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTrafficByChannel(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Traffic by Channel — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_traffic_by_source",
    "Break down GA4 sessions by source and medium (e.g. google/organic, direct/none, newsletter/email).",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTrafficBySource(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Traffic by Source/Medium — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_traffic_over_time",
    "Get a daily trend of GA4 sessions, users, and pageviews over a date range.",
    BASE,
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getTrafficOverTime(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Daily Traffic Trend — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_top_pages",
    "Get the most-viewed pages in GA4 with pageviews, users, session duration, bounce rate, and engagement rate.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTopPages(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Top Pages — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_landing_pages",
    "Get top landing pages in GA4 with sessions, new users, bounce rate, engagement, and conversions.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getLandingPages(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Landing Pages — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_top_events",
    "List the most frequently triggered GA4 events with event count and users.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getTopEvents(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Top Events — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_conversions",
    "Get GA4 conversion events with total conversions, users, sessions, and conversion rate.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getConversions(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Conversions — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_audience_by_country",
    "Break down GA4 users by country.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 25).") },
    async ({ account, property_id, start_date, end_date, row_limit = 25 }) => {
      const data = await getAudienceByCountry(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Audience by Country — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_audience_by_device",
    "Break down GA4 users by device category (desktop, mobile, tablet).",
    BASE,
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getAudienceByDevice(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 Audience by Device — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_audience_by_browser",
    "Break down GA4 users by browser.",
    { ...BASE, row_limit: z.number().optional().describe("Max rows (default 15).") },
    async ({ account, property_id, start_date, end_date, row_limit = 15 }) => {
      const data = await getAudienceByBrowser(account, property_id, start_date, end_date, row_limit);
      return text(formatReport(data, `GA4 Audience by Browser — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_new_vs_returning",
    "Compare new vs returning users in GA4 — users, sessions, pageviews, session duration, engagement rate.",
    BASE,
    async ({ account, property_id, start_date, end_date }) => {
      const data = await getNewVsReturning(account, property_id, start_date, end_date);
      return text(formatReport(data, `GA4 New vs Returning — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );

  server.tool("ga4_realtime",
    "Get the number of active users on the site right now, broken down by country and device.",
    {
      account: z.string().describe("Account alias (from list_accounts)."),
      property_id: z.string().describe('GA4 numeric property ID, e.g. "123456789".'),
    },
    async ({ account, property_id }) => {
      const data = await getRealtimeUsers(account, property_id);
      return text(formatReport(data, `GA4 Realtime Active Users — property ${property_id} (${account})`));
    }
  );

  server.tool("ga4_custom_report",
    "Run a fully custom GA4 report with any combination of dimensions and metrics.",
    {
      ...BASE,
      dimensions: z.array(z.string()).describe('GA4 dimension names, e.g. ["pagePath", "country", "deviceCategory"].'),
      metrics: z.array(z.string()).describe('GA4 metric names, e.g. ["sessions", "totalUsers", "screenPageViews"].'),
      row_limit: z.number().optional().describe("Max rows (default 1000)."),
      order_by_metric: z.string().optional().describe("Sort descending by this metric name."),
    },
    async ({ account, property_id, start_date, end_date, dimensions, metrics, row_limit = 1000, order_by_metric = null }) => {
      const data = await runCustomReport(account, property_id, start_date, end_date, dimensions, metrics, row_limit, order_by_metric);
      return text(formatReport(data, `GA4 Custom Report — property ${property_id} (${account})\n${start_date} → ${end_date}`));
    }
  );
}
