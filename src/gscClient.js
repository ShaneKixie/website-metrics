/**
 * Google Search Console API client
 *
 * All methods accept an `alias` (account identifier) and use that
 * account's stored credentials.
 */

import { google } from "googleapis";
import { getAuthenticatedClient } from "./auth.js";

async function getWebmasters(alias) {
  const auth = await getAuthenticatedClient(alias);
  return google.webmasters({ version: "v3", auth });
}

async function getIndexing(alias) {
  const auth = await getAuthenticatedClient(alias);
  return google.indexing({ version: "v3", auth });
}

// ── Site Management ───────────────────────────────────────────────────────────

export async function listSites(alias) {
  const wm = await getWebmasters(alias);
  const res = await wm.sites.list();
  return res.data.siteEntry || [];
}

export async function listSitemaps(alias, siteUrl) {
  const wm = await getWebmasters(alias);
  const res = await wm.sitemaps.list({ siteUrl });
  return res.data.sitemap || [];
}

// ── Search Analytics ──────────────────────────────────────────────────────────

/**
 * Fetch search analytics performance data.
 *
 * @param {string} alias
 * @param {string} siteUrl
 * @param {object} options
 * @param {string}   options.startDate
 * @param {string}   options.endDate
 * @param {string[]} options.dimensions         - ["query","page","country","device","date"]
 * @param {number}   [options.rowLimit=1000]    - max 25000
 * @param {string}   [options.searchType="web"] - web|image|video|news
 * @param {string}   [options.aggregationType]  - auto|byPage|byProperty
 * @param {object[]} [options.dimensionFilterGroups]
 */
export async function getPerformance(alias, siteUrl, options = {}) {
  const wm = await getWebmasters(alias);
  const {
    startDate,
    endDate,
    dimensions = ["query"],
    rowLimit = 1000,
    searchType = "web",
    aggregationType = "auto",
    dimensionFilterGroups = [],
  } = options;

  const res = await wm.searchanalytics.query({
    siteUrl,
    requestBody: {
      startDate,
      endDate,
      dimensions,
      rowLimit,
      searchType,
      aggregationType,
      dimensionFilterGroups,
    },
  });

  return res.data.rows || [];
}

// ── URL Indexing ──────────────────────────────────────────────────────────────

export async function submitUrl(alias, url, type = "URL_UPDATED") {
  const indexing = await getIndexing(alias);
  const res = await indexing.urlNotifications.publish({
    requestBody: { url, type },
  });
  return res.data;
}

export async function getUrlStatus(alias, url) {
  const wm = await getWebmasters(alias);
  const siteUrl = new URL(url).origin + "/";
  const res = await wm.urlInspection.index.inspect({
    requestBody: { inspectionUrl: url, siteUrl },
  });
  return res.data;
}

// ── Format Helpers ────────────────────────────────────────────────────────────

export function formatPerformanceRows(rows, dimensions) {
  if (!rows || rows.length === 0) return "No data returned.";
  const header = [...dimensions, "clicks", "impressions", "ctr", "position"].join("\t");
  const body = rows
    .map((r) => {
      const keys = (r.keys || []).join("\t");
      const ctr = ((r.ctr || 0) * 100).toFixed(2) + "%";
      const pos = (r.position || 0).toFixed(1);
      return `${keys}\t${r.clicks}\t${r.impressions}\t${ctr}\t${pos}`;
    })
    .join("\n");
  return `${header}\n${body}`;
}
