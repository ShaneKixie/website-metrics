/**
 * Google Analytics 4 (GA4) API client
 *
 * Uses the Google Analytics Data API v1beta.
 * All methods accept an `alias` (account identifier) and a `propertyId`
 * (the GA4 numeric property ID, e.g. "123456789").
 */

import { google } from "googleapis";
import { getAuthenticatedClient } from "./auth.js";

async function getAnalyticsData(alias) {
  const auth = await getAuthenticatedClient(alias);
  return google.analyticsdata({ version: "v1beta", auth });
}

async function getAnalyticsAdmin(alias) {
  const auth = await getAuthenticatedClient(alias);
  return google.analyticsadmin({ version: "v1beta", auth });
}

// ── Account / Property Discovery ─────────────────────────────────────────────

/**
 * List all GA4 accounts accessible to this Google account.
 */
export async function listAccounts(alias) {
  const admin = await getAnalyticsAdmin(alias);
  const res = await admin.accounts.list();
  return res.data.accounts || [];
}

/**
 * List all GA4 properties for a given account.
 * @param {string} accountName - e.g. "accounts/123456"
 */
export async function listProperties(alias, accountName) {
  const admin = await getAnalyticsAdmin(alias);
  const res = await admin.properties.list({ filter: `parent:${accountName}` });
  return res.data.properties || [];
}

// ── Core Report Runner ────────────────────────────────────────────────────────

/**
 * Run a GA4 Data API report.
 *
 * @param {string} alias
 * @param {string} propertyId  - Numeric GA4 property ID (without "properties/")
 * @param {object} reportBody  - GA4 RunReportRequest body
 */
export async function runReport(alias, propertyId, reportBody) {
  const analytics = await getAnalyticsData(alias);
  const res = await analytics.properties.runReport({
    property: `properties/${propertyId}`,
    requestBody: reportBody,
  });
  return res.data;
}

// ── Traffic & Sessions ────────────────────────────────────────────────────────

export async function getTrafficOverview(alias, propertyId, startDate, endDate) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    metrics: [
      { name: "sessions" },
      { name: "totalUsers" },
      { name: "newUsers" },
      { name: "screenPageViews" },
      { name: "bounceRate" },
      { name: "averageSessionDuration" },
      { name: "engagementRate" },
    ],
  });
}

export async function getTrafficByChannel(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "sessionDefaultChannelGroup" }],
    metrics: [
      { name: "sessions" },
      { name: "totalUsers" },
      { name: "newUsers" },
      { name: "bounceRate" },
      { name: "averageSessionDuration" },
      { name: "engagementRate" },
    ],
    orderBys: [{ metric: { metricName: "sessions" }, desc: true }],
    limit,
  });
}

export async function getTrafficBySource(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [
      { name: "sessionSource" },
      { name: "sessionMedium" },
    ],
    metrics: [
      { name: "sessions" },
      { name: "totalUsers" },
      { name: "newUsers" },
      { name: "bounceRate" },
      { name: "averageSessionDuration" },
    ],
    orderBys: [{ metric: { metricName: "sessions" }, desc: true }],
    limit,
  });
}

export async function getTrafficOverTime(alias, propertyId, startDate, endDate) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "date" }],
    metrics: [
      { name: "sessions" },
      { name: "totalUsers" },
      { name: "screenPageViews" },
    ],
    orderBys: [{ dimension: { dimensionName: "date" }, desc: false }],
  });
}

// ── Page Performance ──────────────────────────────────────────────────────────

export async function getTopPages(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "pagePath" }, { name: "pageTitle" }],
    metrics: [
      { name: "screenPageViews" },
      { name: "totalUsers" },
      { name: "averageSessionDuration" },
      { name: "bounceRate" },
      { name: "engagementRate" },
      { name: "exits" },
    ],
    orderBys: [{ metric: { metricName: "screenPageViews" }, desc: true }],
    limit,
  });
}

export async function getLandingPages(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "landingPage" }],
    metrics: [
      { name: "sessions" },
      { name: "totalUsers" },
      { name: "newUsers" },
      { name: "bounceRate" },
      { name: "averageSessionDuration" },
      { name: "engagementRate" },
      { name: "conversions" },
    ],
    orderBys: [{ metric: { metricName: "sessions" }, desc: true }],
    limit,
  });
}

// ── Events & Conversions ──────────────────────────────────────────────────────

export async function getTopEvents(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "eventName" }],
    metrics: [
      { name: "eventCount" },
      { name: "totalUsers" },
      { name: "eventCountPerUser" },
    ],
    orderBys: [{ metric: { metricName: "eventCount" }, desc: true }],
    limit,
  });
}

export async function getConversions(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "eventName" }],
    metrics: [
      { name: "conversions" },
      { name: "totalUsers" },
      { name: "sessions" },
      { name: "conversionRate" },
    ],
    dimensionFilter: {
      filter: {
        fieldName: "isConversionEvent",
        stringFilter: { value: "true" },
      },
    },
    orderBys: [{ metric: { metricName: "conversions" }, desc: true }],
    limit,
  });
}

// ── Audience & Demographics ───────────────────────────────────────────────────

export async function getAudienceByCountry(alias, propertyId, startDate, endDate, limit = 25) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "country" }],
    metrics: [
      { name: "totalUsers" },
      { name: "sessions" },
      { name: "screenPageViews" },
      { name: "bounceRate" },
    ],
    orderBys: [{ metric: { metricName: "totalUsers" }, desc: true }],
    limit,
  });
}

export async function getAudienceByDevice(alias, propertyId, startDate, endDate) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "deviceCategory" }],
    metrics: [
      { name: "totalUsers" },
      { name: "sessions" },
      { name: "bounceRate" },
      { name: "averageSessionDuration" },
      { name: "engagementRate" },
    ],
    orderBys: [{ metric: { metricName: "totalUsers" }, desc: true }],
  });
}

export async function getAudienceByBrowser(alias, propertyId, startDate, endDate, limit = 15) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "browser" }],
    metrics: [
      { name: "totalUsers" },
      { name: "sessions" },
      { name: "bounceRate" },
    ],
    orderBys: [{ metric: { metricName: "totalUsers" }, desc: true }],
    limit,
  });
}

export async function getNewVsReturning(alias, propertyId, startDate, endDate) {
  return runReport(alias, propertyId, {
    dateRanges: [{ startDate, endDate }],
    dimensions: [{ name: "newVsReturning" }],
    metrics: [
      { name: "totalUsers" },
      { name: "sessions" },
      { name: "screenPageViews" },
      { name: "averageSessionDuration" },
      { name: "engagementRate" },
    ],
  });
}

// ── Realtime ──────────────────────────────────────────────────────────────────

export async function getRealtimeUsers(alias, propertyId) {
  const analytics = await getAnalyticsData(alias);
  const res = await analytics.properties.runRealtimeReport({
    property: `properties/${propertyId}`,
    requestBody: {
      dimensions: [{ name: "country" }, { name: "deviceCategory" }],
      metrics: [{ name: "activeUsers" }],
    },
  });
  return res.data;
}

// ── Custom Report ─────────────────────────────────────────────────────────────

/**
 * Run a fully custom GA4 report with user-specified dimensions and metrics.
 */
export async function runCustomReport(
  alias,
  propertyId,
  startDate,
  endDate,
  dimensions,
  metrics,
  limit = 1000,
  orderByMetric = null,
  dimensionFilters = null
) {
  const body = {
    dateRanges: [{ startDate, endDate }],
    dimensions: dimensions.map((d) => ({ name: d })),
    metrics: metrics.map((m) => ({ name: m })),
    limit,
  };
  if (orderByMetric) {
    body.orderBys = [{ metric: { metricName: orderByMetric }, desc: true }];
  }
  if (dimensionFilters) {
    body.dimensionFilter = dimensionFilters;
  }
  return runReport(alias, propertyId, body);
}

// ── Format Helpers ────────────────────────────────────────────────────────────

/**
 * Convert a GA4 RunReportResponse into a readable text table.
 */
export function formatReport(data, title = "") {
  if (!data || !data.rows || data.rows.length === 0) {
    return title ? `${title}\nNo data returned.` : "No data returned.";
  }

  const dimHeaders = (data.dimensionHeaders || []).map((h) => h.name);
  const metHeaders = (data.metricHeaders || []).map((h) => h.name);
  const headers = [...dimHeaders, ...metHeaders];

  const rows = data.rows.map((row) => {
    const dims = (row.dimensionValues || []).map((v) => v.value);
    const mets = (row.metricValues || []).map((v) => {
      const val = parseFloat(v.value);
      return isNaN(val) ? v.value : val.toLocaleString();
    });
    return [...dims, ...mets].join("\t");
  });

  const totals = data.totals?.[0];
  let totalLine = "";
  if (totals) {
    const tVals = (totals.metricValues || []).map((v) => {
      const val = parseFloat(v.value);
      return isNaN(val) ? v.value : val.toLocaleString();
    });
    const padding = dimHeaders.map(() => "TOTAL").join("\t");
    totalLine = `\n${padding}\t${tVals.join("\t")}`;
  }

  const header = headers.join("\t");
  const body = rows.join("\n");
  const prefix = title ? `${title}\n\n` : "";
  return `${prefix}${header}\n${body}${totalLine}`;
}
