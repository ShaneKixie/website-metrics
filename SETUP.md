# Google MCP Server — Setup Guide
### GA4 + Google Search Console · Multiple Accounts

---

## Overview

This MCP server gives your AI assistant access to **Google Analytics 4** and **Google Search Console** across up to N Google accounts. Each account authenticates once via OAuth — tokens are stored persistently and refreshed automatically.

---

## Step 1 — Google Cloud Console

Use your **existing** Google Cloud project. You only need one.

### 1a — Enable APIs

Enable all four APIs (click each link, hit **Enable**):

- [Google Search Console API](https://console.cloud.google.com/apis/library/searchconsole.googleapis.com)
- [Google Indexing API](https://console.cloud.google.com/apis/library/indexing.googleapis.com)
- [Google Analytics Data API](https://console.cloud.google.com/apis/library/analyticsdata.googleapis.com)
- [Google Analytics Admin API](https://console.cloud.google.com/apis/library/analyticsadmin.googleapis.com)

### 1b — Create OAuth Credentials

1. Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**
2. Application type: **Web application**
3. Name: `Google MCP Server`
4. **Authorized redirect URIs** — add both:
   - `http://localhost:3000/oauth/callback`
   - `https://YOUR-APP.railway.app/oauth/callback` *(fill in after Railway deploy)*
5. Click **Create** → copy the **Client ID** and **Client Secret**

### 1c — Add Test Users (if app is in Testing mode)

1. Go to **APIs & Services → OAuth consent screen → Test users**
2. Add all 3 Gmail addresses

> Once you're ready, you can publish the app to remove the test-user restriction.

---

## Step 2 — Deploy to Railway

1. Push this repo to GitHub.

2. In [Railway](https://railway.app): **New Project → Deploy from GitHub repo** → select your repo.

3. Set environment variables (**Settings → Variables**):

   | Variable | Value |
   |---|---|
   | `GOOGLE_CLIENT_ID` | From Step 1b |
   | `GOOGLE_CLIENT_SECRET` | From Step 1b |
   | `OAUTH_REDIRECT_URI` | `https://YOUR-APP.railway.app/oauth/callback` |

4. **(Strongly recommended) Add a persistent volume** so tokens survive redeploys:
   - Railway dashboard → your service → **Volumes → Add Volume**
   - Mount path: `/data`
   - The server auto-detects `/data` and stores `tokens.json` there.

5. After deploy, copy your Railway public URL (e.g. `https://google-mcp.railway.app`).

6. Back in Google Cloud → Credentials → your OAuth client → add `https://YOUR-APP.railway.app/oauth/callback` as an Authorized Redirect URI (if you haven't already).

---

## Step 3 — Authenticate Each Google Account

Open each URL in a browser **while signed into the correct Google account**:

```
https://YOUR-APP.railway.app/oauth/start?alias=account1
https://YOUR-APP.railway.app/oauth/start?alias=account2
https://YOUR-APP.railway.app/oauth/start?alias=account3
```

Choose any alias you like — it's the name you'll reference in MCP tool calls (e.g. `main`, `client_a`, `site2`).

After completing the consent screen you'll see a success page listing all connected accounts.

To verify all three are connected:
```
https://YOUR-APP.railway.app/health
```

**You only do this once per account.** Tokens auto-refresh — no repeated logins.

---

## Step 4 — Connect to Your AI Client

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "google": {
      "command": "node",
      "args": ["/path/to/google-mcp-server/src/index.js"],
      "env": {
        "GOOGLE_CLIENT_ID": "...",
        "GOOGLE_CLIENT_SECRET": "...",
        "OAUTH_REDIRECT_URI": "https://YOUR-APP.railway.app/oauth/callback"
      }
    }
  }
}
```

Restart your AI client. The tools will appear automatically.

---

## Available Tools

### Account Management
| Tool | Description |
|---|---|
| `list_accounts` | Show authenticated account aliases |
| `list_gsc_sites` | List verified GSC properties for an account |
| `list_ga4_accounts` | List GA4 accounts for a Google account |
| `list_ga4_properties` | List GA4 properties under a GA4 account |

### Google Search Console
| Tool | Description |
|---|---|
| `gsc_get_performance` | Clicks, impressions, CTR, position — filterable by query/page/country/device |
| `gsc_inspect_url` | Indexing status, mobile usability, rich results |
| `gsc_submit_url` | Submit URL for indexing or deletion |
| `gsc_list_sitemaps` | List submitted sitemaps |

### Google Analytics 4
| Tool | Description |
|---|---|
| `ga4_traffic_overview` | Sessions, users, pageviews, bounce rate, engagement rate |
| `ga4_traffic_by_channel` | Sessions by channel (Organic, Direct, Referral, etc.) |
| `ga4_traffic_by_source` | Sessions by source/medium |
| `ga4_traffic_over_time` | Daily sessions/users/pageviews trend |
| `ga4_top_pages` | Most viewed pages |
| `ga4_landing_pages` | Top landing pages with sessions + conversions |
| `ga4_top_events` | Most triggered events |
| `ga4_conversions` | Conversion events with rates |
| `ga4_audience_by_country` | Users by country |
| `ga4_audience_by_device` | Users by device category |
| `ga4_audience_by_browser` | Users by browser |
| `ga4_new_vs_returning` | New vs returning user split |
| `ga4_realtime` | Active users right now |
| `ga4_custom_report` | Any dimensions + metrics you specify |

---

## Example Usage

```
# What accounts are connected?
list_accounts

# Find the GA4 property ID for account2
list_ga4_accounts { "account": "account2" }
list_ga4_properties { "account": "account2", "ga4_account": "accounts/123456" }

# Last 30 days traffic overview
ga4_traffic_overview {
  "account": "account1",
  "property_id": "123456789",
  "start_date": "30daysAgo",
  "end_date": "today"
}

# Top 10 organic search queries for a GSC property
gsc_get_performance {
  "account": "account1",
  "site_url": "https://example.com/",
  "start_date": "2026-02-11",
  "end_date": "2026-03-13",
  "dimensions": ["query"],
  "row_limit": 10
}

# Compare traffic across all 3 accounts for the same period
ga4_traffic_overview { "account": "account1", "property_id": "111111111", "start_date": "7daysAgo", "end_date": "today" }
ga4_traffic_overview { "account": "account2", "property_id": "222222222", "start_date": "7daysAgo", "end_date": "today" }
ga4_traffic_overview { "account": "account3", "property_id": "333333333", "start_date": "7daysAgo", "end_date": "today" }
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No tokens found for account "X"` | Visit `/oauth/start?alias=X` in a browser |
| `Access blocked: app not verified` | Add the Gmail to test users in Google Cloud OAuth consent screen |
| Tokens lost after Railway redeploy | Mount a persistent volume at `/data` (Step 2, point 4) |
| `redirect_uri_mismatch` | The `OAUTH_REDIRECT_URI` env var must exactly match what's in Google Cloud Console |
| `No GA4 properties found` | Make sure the Analytics Admin API is enabled and the account has GA4 properties (not Universal Analytics) |
