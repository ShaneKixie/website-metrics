/**
 * OAuth2 helpers — supports multiple Google accounts via per-alias tokens.
 *
 * Scopes cover both Google Search Console and Google Analytics (GA4).
 * Tokens are automatically refreshed on every client request — no manual
 * re-login required after the initial one-time OAuth flow per account.
 */

import { google } from "googleapis";
import { getToken, setToken, listAliases } from "./tokenStore.js";

export const SCOPES = [
  // Search Console
  "https://www.googleapis.com/auth/webmasters.readonly",
  "https://www.googleapis.com/auth/indexing",
  // Google Analytics (GA4)
  "https://www.googleapis.com/auth/analytics.readonly",
];

export function createOAuthClient() {
  return new google.auth.OAuth2(
    process.env.GOOGLE_CLIENT_ID,
    process.env.GOOGLE_CLIENT_SECRET,
    process.env.OAUTH_REDIRECT_URI
  );
}

/**
 * Generate the Google consent-screen URL for a given account alias.
 * The alias is passed as the OAuth `state` param so the callback knows
 * which account to store the token under.
 */
export function getAuthUrl(alias) {
  const client = createOAuthClient();
  return client.generateAuthUrl({
    access_type: "offline",
    scope: SCOPES,
    prompt: "consent", // ensures refresh_token is always returned
    state: alias,
  });
}

/**
 * Exchange an authorization code for tokens and persist them.
 */
export async function exchangeCode(alias, code) {
  const client = createOAuthClient();
  const { tokens } = await client.getToken(code);
  if (!tokens.refresh_token) {
    throw new Error(
      "Google did not return a refresh_token. " +
      "Make sure the account hasn't already authorized this app, or revoke access at " +
      "https://myaccount.google.com/permissions and try again."
    );
  }
  setToken(alias, tokens);
  return tokens;
}

/**
 * Return a fully authenticated OAuth2 client for the given alias.
 * Silently refreshes the access token if it has expired or is close to expiry.
 */
export async function getAuthenticatedClient(alias) {
  const tokens = getToken(alias);
  if (!tokens) {
    throw new Error(
      `No tokens found for account "${alias}". ` +
      `Authenticate first by visiting: /oauth/start?alias=${alias}`
    );
  }

  const client = createOAuthClient();
  client.setCredentials(tokens);

  // Refresh if the access token expires within the next 2 minutes
  const expiryDate = tokens.expiry_date || 0;
  const needsRefresh = Date.now() >= expiryDate - 120_000;

  if (needsRefresh && tokens.refresh_token) {
    try {
      const { credentials } = await client.refreshAccessToken();
      // Preserve the refresh_token — Google only re-issues it on first auth
      const merged = { ...tokens, ...credentials };
      if (!merged.refresh_token) merged.refresh_token = tokens.refresh_token;
      setToken(alias, merged);
      client.setCredentials(merged);
    } catch (err) {
      throw new Error(
        `Failed to refresh token for account "${alias}": ${err.message}. ` +
        `Re-authenticate at /oauth/start?alias=${alias}`
      );
    }
  }

  return client;
}

export { listAliases } from "./tokenStore.js";
