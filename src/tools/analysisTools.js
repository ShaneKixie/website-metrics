import { z } from "zod";
import { spawn } from "child_process";
import { getPerformance } from "../gscClient.js";

const CATEGORY_EMOJI = {
  seasonality: "📅",
  motif: "🔄",
  anomaly: "⚠️",
  cluster: "📊",
  trend: "📈",
};

/**
 * Spawn the Python pattern engine with a JSON payload.
 * Pipes payload via stdin, collects stdout, parses JSON.
 */
async function runPatternEngine(payload) {
  return new Promise((resolve, reject) => {
    const python = spawn("python3", ["src/analysis/pattern_engine.py"], {
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (d) => (stdout += d.toString()));
    python.stderr.on("data", (d) => (stderr += d.toString()));

    python.on("close", () => {
      if (!stdout.trim()) {
        reject(new Error(`Pattern engine produced no output. stderr: ${stderr.slice(0, 500)}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error(`Failed to parse pattern engine output: ${stdout.slice(0, 300)}`));
      }
    });

    python.on("error", (err) => {
      reject(
        new Error(
          `Failed to spawn python3: ${err.message}. Ensure Python 3 and requirements.txt dependencies are installed.`
        )
      );
    });

    python.stdin.write(JSON.stringify(payload));
    python.stdin.end();
  });
}

/** Format ranked insights array as readable plain text for the LLM. */
function formatInsights(insights, site) {
  if (!insights || insights.length === 0) {
    return "No significant patterns discovered in the provided data.";
  }

  const lines = [
    `Pattern Discovery Results — ${site}`,
    "=".repeat(60),
    `${insights.length} statistically significant pattern(s) found, ranked by traffic impact:`,
    "",
  ];

  for (const ins of insights) {
    const emoji = CATEGORY_EMOJI[ins.category] ?? "•";
    const impact = Math.round((ins.traffic_impact_score ?? 0) * 100);
    lines.push(`${ins.rank}. ${emoji} [${ins.category.toUpperCase()}] ${ins.title}`);
    lines.push(`   Traffic Impact: ${impact}%`);
    lines.push(`   ${ins.description}`);

    // ── Surface evidence details for discovered_cluster insights ────────────
    if (ins.category === "discovered_cluster" && ins.evidence) {
      const ev = ins.evidence;

      // Archetypal members (highest soft-membership probability = most representative)
      if (ev.archetypal_members && ev.archetypal_members.length > 0) {
        lines.push(`   Most representative members: ${ev.archetypal_members.join(" | ")}`);
      }

      // Top members by clicks (if different from archetypal)
      if (
        ev.top_members_by_clicks &&
        ev.top_members_by_clicks.length > 0 &&
        JSON.stringify(ev.top_members_by_clicks) !== JSON.stringify(ev.archetypal_members)
      ) {
        lines.push(`   Top by clicks: ${ev.top_members_by_clicks.join(" | ")}`);
      }

      // Extreme outliers breakdown (noise insights)
      if (ev.extreme_outlier_count !== undefined) {
        lines.push(
          `   Breakdown: ${ev.extreme_outlier_count} extreme outliers (distance ≥ ${ev.median_centroid_distance}), ` +
          `${ev.near_miss_count} near-miss outliers`
        );
        if (ev.top_extreme_outliers && ev.top_extreme_outliers.length > 0) {
          lines.push(`   Top extreme outliers:`);
          for (const o of ev.top_extreme_outliers) {
            lines.push(`     • ${o.label} (outlier dist: ${o.outlier_distance}, clicks: ${o.clicks})`);
          }
        }
      }
    }

    lines.push("");
  }

  return lines.join("\n");
}

/** Return ISO date strings for a window ending today. */
function dateRange(days) {
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - Math.min(days, 365));
  return {
    startDate: start.toISOString().split("T")[0],
    endDate: end.toISOString().split("T")[0],
  };
}

export function registerAnalysisTools(server) {
  // ── discover_all_patterns ────────────────────────────────────────────────
  server.tool(
    "discover_all_patterns",
    [
      "Automatically discover all statistically significant traffic patterns in a Google Search Console property.",
      "Runs three parallel analyses — daily time series (STUMPY matrix profile + statsmodels MSTL seasonality),",
      "per-page performance (PyOD Isolation Forest anomaly detection + URL clustering),",
      "and per-query intent patterns (long-tail vs head, question queries, high-impression low-CTR).",
      "Returns all findings ranked by traffic impact with no patterns predefined.",
    ].join(" "),
    {
      account: z.string().describe("Account alias (from list_accounts)."),
      site_url: z
        .string()
        .describe('GSC property URL, e.g. "sc-domain:example.com" or "https://example.com/".'),
      days: z
        .number()
        .optional()
        .describe("Analysis window in days (default 180, max 365). More days = better pattern detection."),
    },
    async ({ account, site_url, days = 180 }) => {
      const { startDate, endDate } = dateRange(days);
      const allInsights = [];
      const errors = [];

      // 1. Daily time series — STUMPY + MSTL
      try {
        const rows = await getPerformance(account, site_url, {
          startDate,
          endDate,
          dimensions: ["date"],
          rowLimit: Math.min(days, 365),
        });
        if (rows.length >= 14) {
          const result = await runPatternEngine({ type: "gsc_daily", data: rows, site: site_url, days });
          allInsights.push(...(result.insights ?? []));
        }
      } catch (e) {
        errors.push(`Daily analysis: ${e.message}`);
      }

      // 2. Per-page — PyOD + URL clustering + CTR benchmark
      try {
        const rows = await getPerformance(account, site_url, {
          startDate,
          endDate,
          dimensions: ["page"],
          rowLimit: 5000,
        });
        if (rows.length >= 20) {
          const result = await runPatternEngine({ type: "gsc_pages", data: rows, site: site_url, days });
          allInsights.push(...(result.insights ?? []));
        }
      } catch (e) {
        errors.push(`Page analysis: ${e.message}`);
      }

      // 3. Per-query — intent clustering + long-tail + near-page-1 opportunities
      try {
        const rows = await getPerformance(account, site_url, {
          startDate,
          endDate,
          dimensions: ["query"],
          rowLimit: 5000,
        });
        if (rows.length >= 20) {
          const result = await runPatternEngine({ type: "gsc_queries", data: rows, site: site_url, days });
          allInsights.push(...(result.insights ?? []));
        }
      } catch (e) {
        errors.push(`Query analysis: ${e.message}`);
      }

      // Merge, re-rank by impact score
      allInsights.sort((a, b) => (b.traffic_impact_score ?? 0) - (a.traffic_impact_score ?? 0));
      allInsights.forEach((ins, i) => {
        ins.rank = i + 1;
      });

      const text = formatInsights(allInsights, site_url);
      const footer = errors.length ? `\n\nWarnings during analysis: ${errors.join("; ")}` : "";

      return { content: [{ type: "text", text: text + footer }] };
    }
  );

  // ── find_seasonality ──────────────────────────────────────────────────────
  server.tool(
    "find_seasonality",
    "Decompose traffic into trend, day-of-week, and seasonal components using statsmodels MSTL. Answers: which day of the week gets the most traffic? Is the overall trend up or down? Are there multi-week cycles?",
    {
      account: z.string().describe("Account alias."),
      site_url: z.string().describe("GSC property URL."),
      days: z.number().optional().describe("Analysis window in days (default 180)."),
    },
    async ({ account, site_url, days = 180 }) => {
      const { startDate, endDate } = dateRange(days);
      const rows = await getPerformance(account, site_url, {
        startDate,
        endDate,
        dimensions: ["date"],
        rowLimit: Math.min(days, 365),
      });

      if (rows.length < 14) {
        return {
          content: [{ type: "text", text: "Need at least 14 days of data for seasonality analysis." }],
        };
      }

      const result = await runPatternEngine({ type: "gsc_daily", data: rows, site: site_url, days });
      const seasonal = (result.insights ?? []).filter((i) => ["seasonality", "trend"].includes(i.category));
      seasonal.forEach((ins, i) => { ins.rank = i + 1; });

      return { content: [{ type: "text", text: formatInsights(seasonal, site_url) }] };
    }
  );

  // ── find_anomalous_pages ──────────────────────────────────────────────────
  server.tool(
    "find_anomalous_pages",
    "Use PyOD Isolation Forest to identify pages that are statistical outliers — pages ranking in the top 10 but getting anomalously low CTR compared to peers at the same position.",
    {
      account: z.string().describe("Account alias."),
      site_url: z.string().describe("GSC property URL."),
      days: z.number().optional().describe("Analysis window in days (default 90)."),
    },
    async ({ account, site_url, days = 90 }) => {
      const { startDate, endDate } = dateRange(days);
      const rows = await getPerformance(account, site_url, {
        startDate,
        endDate,
        dimensions: ["page"],
        rowLimit: 5000,
      });

      if (rows.length < 20) {
        return {
          content: [{ type: "text", text: "Need at least 20 pages for anomaly detection." }],
        };
      }

      const result = await runPatternEngine({ type: "gsc_pages", data: rows, site: site_url, days });
      const anomalies = (result.insights ?? []).filter((i) => i.category === "anomaly");
      anomalies.forEach((ins, i) => { ins.rank = i + 1; });

      return { content: [{ type: "text", text: formatInsights(anomalies, site_url) }] };
    }
  );

  // ── find_recurring_patterns ───────────────────────────────────────────────
  server.tool(
    "find_recurring_patterns",
    "Use the STUMPY matrix profile algorithm to discover recurring traffic patterns (motifs) and anomalous periods (discords) in your daily click time series — without specifying what patterns to look for.",
    {
      account: z.string().describe("Account alias."),
      site_url: z.string().describe("GSC property URL."),
      days: z
        .number()
        .optional()
        .describe("Analysis window in days (default 180). More days improves motif detection."),
    },
    async ({ account, site_url, days = 180 }) => {
      const { startDate, endDate } = dateRange(days);
      const rows = await getPerformance(account, site_url, {
        startDate,
        endDate,
        dimensions: ["date"],
        rowLimit: Math.min(days, 365),
      });

      if (rows.length < 28) {
        return {
          content: [{ type: "text", text: "Need at least 28 days of data for recurring pattern detection." }],
        };
      }

      const result = await runPatternEngine({ type: "gsc_daily", data: rows, site: site_url, days });
      const patterns = (result.insights ?? []).filter((i) => ["motif", "anomaly"].includes(i.category));
      patterns.forEach((ins, i) => { ins.rank = i + 1; });

      return { content: [{ type: "text", text: formatInsights(patterns, site_url) }] };
    }
  );
}
