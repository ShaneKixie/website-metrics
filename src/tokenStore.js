/**
 * Token Store
 *
 * Persists OAuth tokens per account alias to a JSON file.
 * Storage priority:
 *   1. TOKEN_FILE_PATH env var (explicit override)
 *   2. /data/tokens.json  (Railway persistent volume, if mounted)
 *   3. ./tokens.json      (local fallback)
 */

import fs from "fs";

const TOKEN_FILE =
  process.env.TOKEN_FILE_PATH ||
  (fs.existsSync("/data") ? "/data/tokens.json" : "./tokens.json");

function load() {
  if (!fs.existsSync(TOKEN_FILE)) return {};
  try {
    return JSON.parse(fs.readFileSync(TOKEN_FILE, "utf8"));
  } catch {
    return {};
  }
}

function save(data) {
  fs.writeFileSync(TOKEN_FILE, JSON.stringify(data, null, 2), "utf8");
}

export function getToken(alias) {
  return load()[alias] || null;
}

export function setToken(alias, tokenData) {
  const all = load();
  all[alias] = { ...tokenData, updatedAt: new Date().toISOString() };
  save(all);
}

export function listAliases() {
  return Object.keys(load());
}

export function removeToken(alias) {
  const all = load();
  delete all[alias];
  save(all);
}

export function getTokenFilePath() {
  return TOKEN_FILE;
}
