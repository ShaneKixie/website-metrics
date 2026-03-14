#!/usr/bin/env python3
"""
Pattern Discovery Engine

Reads a JSON payload from stdin, runs statistical analysis using STUMPY,
statsmodels MSTL, PyOD, and HDBSCAN, and writes ranked insights as JSON to stdout.

Two layers of discovery:
  1. Predefined checks — hardcoded comparisons for known SEO metrics
  2. Open-ended discovery — HDBSCAN finds natural page/query clusters without
     predefining what to look for; multi-metric STUMPY runs on CTR, position,
     and impressions in addition to clicks

Input:  { "type": "gsc_daily"|"gsc_pages"|"gsc_queries", "data": [...], "site": "...", "days": 180 }
Output: { "insights": [{ "rank", "category", "title", "description", "evidence", "traffic_impact_score" }] }
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

EXPECTED_CTR = {1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07,
                6: 0.06, 7: 0.05, 8: 0.04, 9: 0.04, 10: 0.03}

MAX_INSIGHTS = 50


def analyze(payload):
    analysis_type = payload.get("type")
    data = payload.get("data", [])
    site = payload.get("site", "")

    if not data:
        return {"insights": [], "error": "No data provided"}

    dispatch = {
        "gsc_daily":   analyze_daily,
        "gsc_pages":   analyze_pages,
        "gsc_queries": analyze_queries,
    }
    fn = dispatch.get(analysis_type)
    if not fn:
        return {"insights": [], "error": f"Unknown analysis type: {analysis_type}"}

    insights = fn(data, site)
    insights.sort(key=lambda x: x.get("traffic_impact_score", 0), reverse=True)
    for i, ins in enumerate(insights):
        ins["rank"] = i + 1
    return {"insights": insights[:MAX_INSIGHTS]}


# ─────────────────────────────────────────────────────────────────────────────
# OPEN-ENDED DISCOVERY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hdbscan_clusters(df, feature_cols, label_col, value_col, total_value, category_tag, entity_name="pages"):
    """
    Run HDBSCAN on a feature matrix built from df[feature_cols].
    Returns a list of insights, one per discovered cluster, characterised by
    which features deviate most from the global mean — without any predefined
    pattern categories.
    """
    insights = []
    try:
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler

        sub = df[feature_cols + [label_col, value_col]].dropna()
        if len(sub) < 15:
            return []

        X = StandardScaler().fit_transform(sub[feature_cols].values)
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
        labels = clusterer.fit_predict(X)
        sub = sub.copy()
        sub["_cluster"] = labels

        # Global means for comparison
        global_means = {col: float(sub[col].mean()) for col in feature_cols}

        unique_clusters = [c for c in sorted(sub["_cluster"].unique()) if c != -1]
        if not unique_clusters:
            return []

        for cluster_id in unique_clusters:
            mask = sub["_cluster"] == cluster_id
            cluster = sub[mask]
            cluster_clicks = float(cluster[value_col].sum())
            cluster_size = int(mask.sum())

            # Find top 3 most distinguishing features (largest z-score vs global)
            deviations = []
            for col in feature_cols:
                g_mean = global_means[col]
                g_std = float(sub[col].std()) or 1.0
                c_mean = float(cluster[col].mean())
                z = (c_mean - g_mean) / g_std
                deviations.append((col, c_mean, g_mean, z))

            deviations.sort(key=lambda x: abs(x[3]), reverse=True)
            top_devs = deviations[:3]

            # Build a plain-language description of what distinguishes this cluster
            trait_parts = []
            for col, c_mean, g_mean, z in top_devs:
                if abs(z) < 0.3:
                    continue
                direction = "high" if z > 0 else "low"
                # Format value nicely based on column type
                if "ctr" in col:
                    trait_parts.append(f"{direction} CTR ({c_mean*100:.2f}% vs {g_mean*100:.2f}% avg)")
                elif "position" in col:
                    trait_parts.append(f"{direction} avg position ({c_mean:.1f} vs {g_mean:.1f} avg)")
                elif "clicks" in col:
                    trait_parts.append(f"{direction} clicks ({c_mean:.1f} vs {g_mean:.1f} avg)")
                elif "impressions" in col:
                    trait_parts.append(f"{direction} impressions ({c_mean:.1f} vs {g_mean:.1f} avg)")
                elif "depth" in col:
                    trait_parts.append(f"{'deeper' if z > 0 else 'shallower'} URLs (depth {c_mean:.1f} vs {g_mean:.1f} avg)")
                elif "word_count" in col:
                    trait_parts.append(f"{'longer' if z > 0 else 'shorter'} queries ({c_mean:.1f} words vs {g_mean:.1f} avg)")
                else:
                    trait_parts.append(f"{direction} {col.replace('_', ' ')} ({c_mean:.2f} vs {g_mean:.2f} avg)")

            if not trait_parts:
                continue

            trait_str = "; ".join(trait_parts) if trait_parts else "statistically distinct profile"

            # Example members
            examples = cluster.sort_values(value_col, ascending=False)[label_col].head(3).tolist()
            examples = [str(e).split("?")[0][:80] for e in examples]

            insights.append({
                "category": category_tag,
                "title": f"Discovered cluster of {cluster_size} {entity_name}: {trait_str[:80]}",
                "description": (
                    f"HDBSCAN found {cluster_size} {entity_name} that naturally group together with a distinct statistical profile. "
                    f"Key traits vs site average: {trait_str}. "
                    f"This cluster drives {cluster_clicks:.0f} of {total_value:.0f} total clicks ({cluster_clicks/total_value*100:.1f}%). "
                    f"No pattern was predefined — this grouping emerged purely from the data."
                ),
                "evidence": {
                    "cluster_id": int(cluster_id),
                    "cluster_size": cluster_size,
                    "cluster_clicks": round(cluster_clicks, 0),
                    "click_share_pct": round(cluster_clicks / total_value * 100, 1),
                    "distinguishing_features": [
                        {"feature": col, "cluster_mean": round(c_mean, 4), "global_mean": round(g_mean, 4), "z_score": round(z, 2)}
                        for col, c_mean, g_mean, z in top_devs if abs(z) >= 0.3
                    ],
                    "example_members": examples,
                },
                "traffic_impact_score": min(1.0, cluster_clicks / total_value),
            })

        # Also report noise points (cluster = -1) if significant
        noise = sub[sub["_cluster"] == -1]
        noise_clicks = float(noise[value_col].sum())
        if len(noise) >= 5 and noise_clicks / total_value > 0.05:
            insights.append({
                "category": category_tag,
                "title": f"{len(noise)} {entity_name} don't fit any cluster — statistically unique",
                "description": (
                    f"HDBSCAN could not assign {len(noise)} {entity_name} to any cluster — they are statistical outliers "
                    f"with no peers that share their profile. These represent {noise_clicks/total_value*100:.1f}% of total clicks. "
                    f"Each is uniquely performing and worth individual review."
                ),
                "evidence": {
                    "noise_count": len(noise),
                    "noise_clicks": round(noise_clicks, 0),
                    "click_share_pct": round(noise_clicks / total_value * 100, 1),
                    "examples": noise.sort_values(value_col, ascending=False)[label_col].head(5).tolist(),
                },
                "traffic_impact_score": min(1.0, noise_clicks / total_value * 0.5),
            })

    except Exception:
        pass

    return insights


def stumpy_metric(series, dates, metric_name, m=7):
    """
    Run STUMPY on a single metric time series and return motif/discord insights.
    Returns [] if series is too short or all-zero.
    """
    insights = []
    try:
        import stumpy
        values = np.array(series, dtype=float)
        if len(values) < m * 2 or np.nanstd(values) < 1e-6:
            return []

        overall_mean = float(np.nanmean(values)) or 1.0
        mp = stumpy.stump(values, m=m)
        profile = mp[:, 0].astype(float)

        # Top 2 motifs
        profile_copy = profile.copy()
        for rank in range(1, 3):
            if np.all(np.isinf(profile_copy)):
                break
            idx = int(np.argmin(profile_copy))
            excl = slice(max(0, idx - m), min(len(profile_copy), idx + m))
            profile_copy[excl] = np.inf
            seg_mean = float(np.nanmean(values[idx:idx + m]))
            ratio = seg_mean / overall_mean
            if abs(ratio - 1) > 0.10:
                date_str = str(dates[idx])[:10] if idx < len(dates) else "unknown"
                label = "above-average" if ratio > 1 else "below-average"
                insights.append({
                    "category": "motif",
                    "title": f"Recurring {m}-day {metric_name} pattern #{rank} first seen {date_str} ({ratio:.2f}x avg)",
                    "description": (
                        f"A repeating {m}-day pattern in {metric_name} is {label}: "
                        f"{seg_mean:.3f} vs {overall_mean:.3f} overall average ({ratio:.2f}x). "
                        f"This recurs throughout the dataset with no predefined pattern specified."
                    ),
                    "evidence": {
                        "metric": metric_name,
                        "first_occurrence_date": date_str,
                        "segment_mean": round(seg_mean, 4),
                        "overall_mean": round(overall_mean, 4),
                        "ratio_vs_average": round(ratio, 2),
                        "window_days": m,
                    },
                    "traffic_impact_score": min(1.0, abs(ratio - 1) * 0.4),
                })

        # Top 2 discords
        profile_copy = profile.copy()
        for rank in range(1, 3):
            if np.all(np.isinf(profile_copy)):
                break
            idx = int(np.argmax(profile_copy))
            excl = slice(max(0, idx - m), min(len(profile_copy), idx + m))
            profile_copy[excl] = -np.inf
            if idx + m > len(values):
                continue
            seg_mean = float(np.nanmean(values[idx:idx + m]))
            deviation = abs(seg_mean - overall_mean) / overall_mean
            if deviation > 0.20:
                date_str = str(dates[idx])[:10] if idx < len(dates) else "unknown"
                direction = "spike" if seg_mean > overall_mean else "drop"
                insights.append({
                    "category": "anomaly",
                    "title": f"Anomalous {m}-day {metric_name} {direction} #{rank}: {deviation*100:.0f}% from baseline, week of {date_str}",
                    "description": (
                        f"The {m}-day window starting {date_str} is the most statistically unusual period "
                        f"for {metric_name} — a {direction} of {deviation*100:.0f}% from baseline "
                        f"({seg_mean:.3f} vs {overall_mean:.3f} avg). "
                        f"Investigate what changed: algorithm update, content change, or external event."
                    ),
                    "evidence": {
                        "metric": metric_name,
                        "date": date_str,
                        "segment_mean": round(seg_mean, 4),
                        "baseline_mean": round(overall_mean, 4),
                        "deviation_pct": round(deviation * 100, 1),
                        "direction": direction,
                        "window_days": m,
                    },
                    "traffic_impact_score": min(1.0, deviation * 0.5),
                })

    except Exception:
        pass

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# DAILY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_daily(rows, site):
    insights = []

    df = pd.DataFrame([{
        "date": r["keys"][0],
        "clicks": float(r.get("clicks", 0)),
        "impressions": float(r.get("impressions", 0)),
        "ctr": float(r.get("ctr", 0)),
        "position": float(r.get("position", 50)),
    } for r in rows])

    if df.empty or len(df) < 14:
        return []

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # ── MSTL: weekly + monthly seasonality ───────────────────────────────────
    try:
        from statsmodels.tsa.seasonal import MSTL

        clicks = df["clicks"].fillna(0).values
        if len(clicks) >= 14:
            mstl7 = MSTL(clicks, periods=[7])
            res7 = mstl7.fit()
            trend = res7.trend

            df_dow = df.copy()
            df_dow["dow"] = df_dow.index.dayofweek
            dow_means = df_dow.groupby("dow")["clicks"].mean()
            overall_mean = df_dow["clicks"].mean()

            if overall_mean > 0:
                dow_effects = (dow_means / overall_mean - 1) * 100
                best_idx = dow_effects.idxmax()
                worst_idx = dow_effects.idxmin()
                best_pct = float(dow_effects[best_idx])
                worst_pct = float(dow_effects[worst_idx])

                if abs(best_pct) > 5:
                    insights.append({
                        "category": "seasonality",
                        "title": f"{DAY_NAMES[best_idx]} is your highest-traffic day",
                        "description": (
                            f"{DAY_NAMES[best_idx]} gets {best_pct:.0f}% more clicks than your daily average. "
                            f"{DAY_NAMES[worst_idx]} is the lowest at {abs(worst_pct):.0f}% below average. "
                            f"Scheduling content publication or promotions around {DAY_NAMES[best_idx]} could amplify reach."
                        ),
                        "evidence": {
                            "day_of_week_effects_pct": {DAY_NAMES[i]: round(v, 1) for i, v in dow_effects.items()},
                            "best_day": DAY_NAMES[best_idx],
                            "worst_day": DAY_NAMES[worst_idx],
                        },
                        "traffic_impact_score": min(1.0, abs(best_pct) / 100),
                    })

            if len(trend) > 28:
                recent = float(np.nanmean(trend[-14:]))
                baseline = float(np.nanmean(trend[:14]))
                if baseline > 0:
                    change_pct = (recent - baseline) / baseline * 100
                    if abs(change_pct) > 5:
                        direction = "increasing" if change_pct > 0 else "decreasing"
                        insights.append({
                            "category": "trend",
                            "title": f"Overall traffic trend is {direction} ({abs(change_pct):.0f}%)",
                            "description": (
                                f"After removing seasonal effects, the underlying traffic trend has {direction} "
                                f"by {abs(change_pct):.0f}% over the analysis window. "
                                f"Recent average: {recent:.0f} clicks/day vs baseline: {baseline:.0f} clicks/day."
                            ),
                            "evidence": {
                                "trend_change_pct": round(change_pct, 1),
                                "recent_avg_clicks_per_day": round(recent, 1),
                                "baseline_avg_clicks_per_day": round(baseline, 1),
                            },
                            "traffic_impact_score": min(1.0, abs(change_pct) / 150),
                        })

        if len(clicks) >= 56:
            try:
                mstl28 = MSTL(clicks, periods=[7, 28])
                res28 = mstl28.fit()
                monthly_seasonal = res28.seasonal[:, 1] if res28.seasonal.ndim > 1 else None
                if monthly_seasonal is not None:
                    monthly_amplitude = float(np.nanstd(monthly_seasonal))
                    overall_mean = float(np.nanmean(clicks)) or 1.0
                    monthly_pct = monthly_amplitude / overall_mean * 100
                    if monthly_pct > 5:
                        weekly_sums = [float(np.nansum(monthly_seasonal[i:i+7]))
                                       for i in range(0, len(monthly_seasonal)-6, 7)]
                        if weekly_sums:
                            best_week = int(np.argmax(weekly_sums)) + 1
                            worst_week = int(np.argmin(weekly_sums)) + 1
                            insights.append({
                                "category": "seasonality",
                                "title": f"Monthly traffic cycle detected ({monthly_pct:.0f}% amplitude)",
                                "description": (
                                    f"A recurring monthly pattern explains {monthly_pct:.0f}% of click variance. "
                                    f"Traffic tends to peak around week {best_week} of the month and dip around week {worst_week}. "
                                    f"This could reflect pay-cycle buying patterns, monthly content calendars, or monthly ad budget cycles."
                                ),
                                "evidence": {
                                    "monthly_amplitude_pct_of_mean": round(monthly_pct, 1),
                                    "peak_week_of_month": best_week,
                                    "trough_week_of_month": worst_week,
                                    "analysis_days": len(clicks),
                                },
                                "traffic_impact_score": min(1.0, monthly_pct / 50),
                            })
            except Exception:
                pass
    except Exception:
        pass

    # ── STUMPY: clicks + CTR + position + impressions across multiple windows ─
    try:
        dates = df.index.tolist()
        clicks_arr = df["clicks"].fillna(0).values.astype(float)
        overall_mean = float(clicks_arr.mean()) or 1.0

        window_configs = []
        if len(clicks_arr) >= 28:
            window_configs.append((7, "weekly"))
        if len(clicks_arr) >= 56:
            window_configs.append((14, "biweekly"))
        if len(clicks_arr) >= 84:
            window_configs.append((28, "monthly"))

        import stumpy

        for m, label in window_configs:
            try:
                mp = stumpy.stump(clicks_arr, m=m)
                profile = mp[:, 0].astype(float)

                # Top 3 motifs
                profile_copy = profile.copy()
                motif_count = 0
                for _ in range(3):
                    if np.all(np.isinf(profile_copy)):
                        break
                    motif_idx = int(np.argmin(profile_copy))
                    excl = slice(max(0, motif_idx - m), min(len(profile_copy), motif_idx + m))
                    profile_copy[excl] = np.inf
                    motif_clicks = float(clicks_arr[motif_idx:motif_idx + m].mean())
                    motif_date = str(df.index[motif_idx].date())
                    motif_ratio = motif_clicks / overall_mean
                    if abs(motif_ratio - 1) > 0.10:
                        motif_count += 1
                        label_str = "above-average" if motif_ratio > 1 else "below-average"
                        insights.append({
                            "category": "motif",
                            "title": f"Recurring {label} clicks pattern #{motif_count}: first seen {motif_date} ({motif_ratio:.2f}x avg)",
                            "description": (
                                f"STUMPY found a repeating {m}-day traffic pattern that is {label_str}: "
                                f"{motif_clicks:.0f} vs {overall_mean:.0f} avg clicks/day ({motif_ratio:.2f}x). "
                                f"This pattern recurs consistently — no pattern was predefined."
                            ),
                            "evidence": {
                                "first_occurrence_date": motif_date,
                                "avg_clicks_during_pattern": round(motif_clicks, 1),
                                "overall_avg_clicks": round(overall_mean, 1),
                                "ratio_vs_average": round(motif_ratio, 2),
                                "window_days": m, "window_label": label,
                            },
                            "traffic_impact_score": min(1.0, abs(motif_ratio - 1) * 0.6),
                        })

                # Top 3 discords
                profile_copy = profile.copy()
                discord_count = 0
                for _ in range(3):
                    if np.all(np.isinf(profile_copy)):
                        break
                    discord_idx = int(np.argmax(profile_copy))
                    excl = slice(max(0, discord_idx - m), min(len(profile_copy), discord_idx + m))
                    profile_copy[excl] = -np.inf
                    if discord_idx + m > len(df):
                        continue
                    discord_clicks = float(clicks_arr[discord_idx:discord_idx + m].mean())
                    discord_date = str(df.index[discord_idx].date())
                    deviation = abs(discord_clicks - overall_mean) / overall_mean
                    if deviation > 0.20:
                        discord_count += 1
                        direction = "spike" if discord_clicks > overall_mean else "drop"
                        insights.append({
                            "category": "anomaly",
                            "title": f"Anomalous {label} period #{discord_count}: {direction} of {deviation*100:.0f}% week of {discord_date}",
                            "description": (
                                f"The {m}-day window starting {discord_date} is statistically the most unusual — "
                                f"a traffic {direction} of {deviation*100:.0f}% from baseline "
                                f"({discord_clicks:.0f} vs {overall_mean:.0f} avg clicks/day). "
                                f"Investigate: algorithm update, content publish, link acquisition, or site issue."
                            ),
                            "evidence": {
                                "date": discord_date,
                                "avg_clicks_that_period": round(discord_clicks, 1),
                                "baseline_avg_clicks": round(overall_mean, 1),
                                "deviation_pct": round(deviation * 100, 1),
                                "direction": direction,
                                "window_days": m, "window_label": label,
                            },
                            "traffic_impact_score": min(1.0, deviation * 0.7),
                        })
            except Exception:
                pass

        # ── Open-ended: STUMPY on CTR, position, impressions ─────────────────
        # These run independently — finds patterns we never thought to look for
        for metric_name, col in [("CTR", "ctr"), ("avg_position", "position"), ("impressions", "impressions")]:
            series = df[col].fillna(method="ffill").fillna(0).values.astype(float)
            for m, _ in window_configs:
                new_insights = stumpy_metric(series, dates, metric_name, m=m)
                insights.extend(new_insights)

    except Exception:
        pass

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pages(rows, site):
    insights = []

    df = pd.DataFrame([{
        "page": r["keys"][0],
        "clicks": float(r.get("clicks", 0)),
        "impressions": float(r.get("impressions", 0)),
        "ctr": float(r.get("ctr", 0)),
        "position": float(r.get("position", 50)),
    } for r in rows])

    if df.empty or len(df) < 20:
        return []

    total_clicks = df["clicks"].sum() or 1.0
    total_impressions = df["impressions"].sum() or 1.0

    # ── URL features ──────────────────────────────────────────────────────────
    def path_depth(url):
        return len([p for p in url.split("?")[0].split("/") if p])

    def has_segment(url, keywords):
        parts = url.split("?")[0].lower().split("/")
        return int(any(k in parts for k in keywords))

    df["url_depth"] = df["page"].apply(path_depth)
    df["is_blog"] = df["page"].apply(lambda u: has_segment(u, ["blog", "blogs", "post", "posts", "article", "articles"]))
    df["is_guide"] = df["page"].apply(lambda u: has_segment(u, ["guide", "guides", "how-to", "tutorial", "learn"]))
    df["log_clicks"] = np.log1p(df["clicks"])
    df["log_impressions"] = np.log1p(df["impressions"])

    # ── Predefined: blog/guide CTR comparison ─────────────────────────────────
    try:
        for seg_col, seg_label in [("is_blog", "Blog"), ("is_guide", "Guide")]:
            seg = df[df[seg_col] == 1]
            non_seg = df[df[seg_col] == 0]
            if len(seg) >= 5 and len(non_seg) >= 5:
                seg_ctr = seg["ctr"].mean()
                other_ctr = non_seg["ctr"].mean()
                seg_clicks = seg["clicks"].sum()
                if other_ctr > 0 and abs(seg_ctr - other_ctr) / other_ctr > 0.10:
                    pct = abs(seg_ctr - other_ctr) / other_ctr * 100
                    insights.append({
                        "category": "cluster",
                        "title": f"{seg_label} pages get {pct:.0f}% {'higher' if seg_ctr > other_ctr else 'lower'} CTR than other pages",
                        "description": (
                            f"{seg_label} pages average {seg_ctr*100:.2f}% CTR vs {other_ctr*100:.2f}% for other pages. "
                            f"{seg_label} content drives {seg_clicks:.0f} of {total_clicks:.0f} total clicks ({seg_clicks/total_clicks*100:.0f}%). "
                            f"{'Invest more in this content type.' if seg_ctr > other_ctr else 'Review title/meta on this content type.'}"
                        ),
                        "evidence": {
                            f"{seg_label.lower()}_avg_ctr": round(float(seg_ctr), 4),
                            "other_avg_ctr": round(float(other_ctr), 4),
                            f"{seg_label.lower()}_page_count": len(seg),
                            f"{seg_label.lower()}_total_clicks": float(seg_clicks),
                        },
                        "traffic_impact_score": min(1.0, seg_clicks / total_clicks),
                    })
    except Exception:
        pass

    # ── Predefined: URL depth vs CTR ─────────────────────────────────────────
    try:
        depth_perf = df.groupby("url_depth").agg(
            avg_ctr=("ctr", "mean"), total_clicks=("clicks", "sum"), count=("page", "count")
        ).reset_index()
        if len(depth_perf) > 1:
            best = depth_perf.loc[depth_perf["avg_ctr"].idxmax()]
            worst = depth_perf.loc[depth_perf["avg_ctr"].idxmin()]
            overall_ctr = df["ctr"].mean()
            if best["count"] >= 3 and overall_ctr > 0 and best["avg_ctr"] / overall_ctr > 1.15:
                insights.append({
                    "category": "cluster",
                    "title": f"Pages at URL depth {int(best['url_depth'])} have the highest CTR",
                    "description": (
                        f"Pages with {int(best['url_depth'])} path segments average {best['avg_ctr']*100:.2f}% CTR "
                        f"vs {overall_ctr*100:.2f}% overall — {best['avg_ctr']/overall_ctr:.1f}x the average. "
                        f"This depth ({best['count']} pages) generates {best['total_clicks']:.0f} clicks. "
                        f"Compare against depth {int(worst['url_depth'])} pages at {worst['avg_ctr']*100:.2f}% CTR."
                    ),
                    "evidence": {
                        "best_depth": int(best["url_depth"]),
                        "best_depth_avg_ctr": round(float(best["avg_ctr"]), 4),
                        "worst_depth": int(worst["url_depth"]),
                        "worst_depth_avg_ctr": round(float(worst["avg_ctr"]), 4),
                        "overall_avg_ctr": round(float(overall_ctr), 4),
                        "page_count": int(best["count"]),
                    },
                    "traffic_impact_score": min(1.0, best["total_clicks"] / total_clicks),
                })
    except Exception:
        pass

    # ── Predefined: PyOD anomalous low-CTR + high-CTR pages ──────────────────
    try:
        from pyod.models.iforest import IForest
        from sklearn.preprocessing import StandardScaler

        sig = df[df["impressions"] >= 100].copy()
        if len(sig) >= 20:
            X = StandardScaler().fit_transform(sig[["position", "ctr", "impressions"]].values)
            clf = IForest(contamination=0.15, random_state=42)
            clf.fit(X)
            sig["anomaly"] = clf.labels_

            top10 = sig[sig["position"] <= 10].copy()
            if len(top10) >= 5:
                ctr_thresh = top10["ctr"].quantile(0.33)
                under = top10[top10["ctr"] <= ctr_thresh].sort_values("impressions", ascending=False)
                if len(under) >= 2:
                    avg_top10_ctr = float(top10["ctr"].mean())
                    avg_under_ctr = float(under["ctr"].mean())
                    est_lost = float((under["impressions"] * (avg_top10_ctr - under["ctr"])).sum())
                    examples = [u.split("?")[0] for u in under["page"].head(5).tolist()]
                    insights.append({
                        "category": "anomaly",
                        "title": f"{len(under)} top-10 pages have anomalously low CTR",
                        "description": (
                            f"These pages rank in the top 10 but CTR is in the bottom third for that position band. "
                            f"Average CTR: {avg_under_ctr*100:.1f}% vs {avg_top10_ctr*100:.1f}% for top-10 overall. "
                            f"Improving title tags and meta descriptions could recover ~{est_lost:.0f} clicks."
                        ),
                        "evidence": {
                            "underperforming_page_count": len(under),
                            "avg_ctr_underperformers": round(avg_under_ctr, 4),
                            "avg_ctr_top10": round(avg_top10_ctr, 4),
                            "estimated_recoverable_clicks": round(est_lost, 0),
                            "example_pages": examples,
                        },
                        "traffic_impact_score": min(1.0, est_lost / total_clicks * 3),
                    })

                top10_high = sig[(sig["position"] <= 10) & (sig["anomaly"] == 1)].copy()
                if len(top10_high) >= 4:
                    top10_high_ctr = top10_high[top10_high["ctr"] > top10_high["ctr"].quantile(0.75)]
                    if len(top10_high_ctr) >= 2:
                        avg_high_ctr = float(top10_high_ctr["ctr"].mean())
                        avg_top10_ctr = float(top10["ctr"].mean())
                        examples = [u.split("?")[0] for u in top10_high_ctr.sort_values("impressions", ascending=False)["page"].head(3).tolist()]
                        insights.append({
                            "category": "cluster",
                            "title": f"{len(top10_high_ctr)} pages have anomalously HIGH CTR — study as models",
                            "description": (
                                f"PyOD flagged {len(top10_high_ctr)} top-10 pages as high-side outliers — "
                                f"averaging {avg_high_ctr*100:.1f}% CTR vs {avg_top10_ctr*100:.1f}% for top-10 overall. "
                                f"Analyze these pages' title/meta patterns and replicate across similar pages."
                            ),
                            "evidence": {
                                "high_ctr_page_count": len(top10_high_ctr),
                                "avg_ctr_high_performers": round(avg_high_ctr, 4),
                                "avg_ctr_top10": round(avg_top10_ctr, 4),
                                "example_pages": examples,
                            },
                            "traffic_impact_score": min(1.0, avg_high_ctr / avg_top10_ctr * 0.3),
                        })
    except Exception:
        pass

    # ── Predefined: CTR vs industry benchmark ────────────────────────────────
    try:
        sig = df[df["impressions"] >= 50].copy()
        sig["pos_bucket"] = sig["position"].apply(lambda p: min(10, max(1, round(p))))
        by_pos = sig.groupby("pos_bucket").agg(
            avg_ctr=("ctr", "mean"), total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"), count=("page", "count")
        ).reset_index()
        gaps = []
        for _, row in by_pos.iterrows():
            pos = int(row["pos_bucket"])
            if pos in EXPECTED_CTR and row["count"] >= 3:
                expected = EXPECTED_CTR[pos]
                actual = float(row["avg_ctr"])
                if actual < expected - 0.01:
                    gaps.append({
                        "position": pos,
                        "actual_ctr_pct": round(actual * 100, 2),
                        "expected_ctr_pct": round(expected * 100, 2),
                        "gap_pct": round((expected - actual) * 100, 2),
                        "missed_clicks_estimate": round(float(row["total_impressions"]) * (expected - actual), 0),
                        "page_count": int(row["count"]),
                    })
        if gaps:
            gaps.sort(key=lambda x: x["missed_clicks_estimate"], reverse=True)
            total_missed = sum(g["missed_clicks_estimate"] for g in gaps)
            worst = gaps[0]
            insights.append({
                "category": "cluster",
                "title": f"CTR below industry benchmark across {len(gaps)} position bucket(s)",
                "description": (
                    f"Worst gap at position {worst['position']}: {worst['actual_ctr_pct']}% CTR "
                    f"vs {worst['expected_ctr_pct']}% industry average — ~{worst['missed_clicks_estimate']:.0f} clicks unrealized. "
                    f"Total estimated missed clicks: {total_missed:.0f}."
                ),
                "evidence": {"position_gaps": gaps, "total_missed_clicks_estimate": round(total_missed, 0)},
                "traffic_impact_score": min(1.0, total_missed / total_clicks * 2),
            })
    except Exception:
        pass

    # ── Open-ended: HDBSCAN on page feature vectors ───────────────────────────
    page_features = ["log_clicks", "log_impressions", "ctr", "position", "url_depth", "is_blog", "is_guide"]
    hdbscan_insights = hdbscan_clusters(
        df, page_features, label_col="page", value_col="clicks",
        total_value=total_clicks, category_tag="discovered_cluster", entity_name="pages"
    )
    insights.extend(hdbscan_insights)

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# QUERY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_queries(rows, site):
    insights = []

    df = pd.DataFrame([{
        "query": r["keys"][0],
        "clicks": float(r.get("clicks", 0)),
        "impressions": float(r.get("impressions", 0)),
        "ctr": float(r.get("ctr", 0)),
        "position": float(r.get("position", 50)),
    } for r in rows])

    if df.empty or len(df) < 20:
        return []

    total_clicks = df["clicks"].sum() or 1.0

    question_words = ("how", "what", "why", "when", "where", "who", "which", "can", "does", "is ", "are ", "will ")
    df["word_count"] = df["query"].str.split().str.len()
    df["is_question"] = df["query"].str.lower().apply(lambda q: any(q.startswith(w) for w in question_words))
    df["log_clicks"] = np.log1p(df["clicks"])
    df["log_impressions"] = np.log1p(df["impressions"])

    # ── Predefined: high-impression, low-CTR ─────────────────────────────────
    try:
        imp_75 = df["impressions"].quantile(0.75)
        ctr_25 = df["ctr"].quantile(0.25)
        hi_imp_lo_ctr = df[(df["impressions"] >= imp_75) & (df["ctr"] <= ctr_25)].copy()
        if len(hi_imp_lo_ctr) >= 5:
            hi_imp_lo_ctr["opportunity"] = hi_imp_lo_ctr["impressions"] * (df["ctr"].mean() - hi_imp_lo_ctr["ctr"])
            top = hi_imp_lo_ctr.sort_values("opportunity", ascending=False).head(5)
            total_opp = float(top["opportunity"].sum())
            insights.append({
                "category": "anomaly",
                "title": f"{len(hi_imp_lo_ctr)} high-impression queries have below-average CTR",
                "description": (
                    f"Top 25% impressions but bottom 25% CTR — high visibility, low click-through. "
                    f"Top: '{top.iloc[0]['query']}' — {top.iloc[0]['impressions']:.0f} impressions at {top.iloc[0]['ctr']*100:.1f}% CTR. "
                    f"Fixing titles/meta could yield ~{total_opp:.0f} additional clicks."
                ),
                "evidence": {
                    "query_count": len(hi_imp_lo_ctr),
                    "top_opportunities": top[["query", "impressions", "ctr", "position"]].round(3).to_dict("records"),
                    "estimated_additional_clicks": round(total_opp, 0),
                },
                "traffic_impact_score": min(1.0, total_opp / total_clicks * 2),
            })
    except Exception:
        pass

    # ── Predefined: long-tail vs head vs mid-tail ─────────────────────────────
    try:
        longtail = df[df["word_count"] >= 4]
        head = df[df["word_count"] <= 2]
        mid = df[df["word_count"] == 3]
        for grp, grp_label in [(longtail, "Long-tail (4+ word)"), (mid, "Mid-tail (3-word)")]:
            if len(grp) >= 10 and len(head) >= 5:
                g_ctr = grp["ctr"].mean()
                hd_ctr = head["ctr"].mean()
                g_clicks = grp["clicks"].sum()
                if hd_ctr > 0 and abs(g_ctr - hd_ctr) / hd_ctr > 0.10:
                    pct = abs(g_ctr - hd_ctr) / hd_ctr * 100
                    insights.append({
                        "category": "cluster",
                        "title": f"{grp_label} queries get {pct:.0f}% {'higher' if g_ctr > hd_ctr else 'lower'} CTR",
                        "description": (
                            f"{grp_label} queries average {g_ctr*100:.2f}% CTR vs {hd_ctr*100:.2f}% for head terms. "
                            f"Drives {g_clicks:.0f} clicks ({g_clicks/total_clicks*100:.0f}% of total) across {len(grp)} keywords."
                        ),
                        "evidence": {
                            "group_avg_ctr": round(float(g_ctr), 4),
                            "head_avg_ctr": round(float(hd_ctr), 4),
                            "group_clicks": float(g_clicks),
                            "group_query_count": len(grp),
                        },
                        "traffic_impact_score": min(1.0, g_clicks / total_clicks),
                    })
    except Exception:
        pass

    # ── Predefined: question queries ─────────────────────────────────────────
    try:
        q_df = df[df["is_question"]]
        nq_df = df[~df["is_question"]]
        if len(q_df) >= 10 and len(nq_df) >= 10:
            q_ctr = q_df["ctr"].mean()
            nq_ctr = nq_df["ctr"].mean()
            q_clicks = q_df["clicks"].sum()
            if nq_ctr > 0 and abs(q_ctr - nq_ctr) / nq_ctr > 0.10:
                pct = abs(q_ctr - nq_ctr) / nq_ctr * 100
                direction = "higher" if q_ctr > nq_ctr else "lower"
                insights.append({
                    "category": "cluster",
                    "title": f"Question queries (how/what/why) get {pct:.0f}% {direction} CTR",
                    "description": (
                        f"Question queries average {q_ctr*100:.2f}% CTR vs {nq_ctr*100:.2f}% for others. "
                        f"{len(q_df)} question queries drive {q_clicks:.0f} clicks."
                    ),
                    "evidence": {
                        "question_avg_ctr": round(float(q_ctr), 4),
                        "nonquestion_avg_ctr": round(float(nq_ctr), 4),
                        "question_query_count": len(q_df),
                        "question_total_clicks": float(q_clicks),
                    },
                    "traffic_impact_score": min(1.0, q_clicks / total_clicks),
                })
    except Exception:
        pass

    # ── Predefined: near-page-1 opportunities ────────────────────────────────
    try:
        near_p1 = df[(df["position"] >= 4) & (df["position"] <= 10) & (df["impressions"] >= 100)].copy()
        if len(near_p1) >= 5:
            near_p1["click_gain_if_p3"] = near_p1["impressions"] * (EXPECTED_CTR.get(3, 0.11) - near_p1["ctr"])
            near_p1 = near_p1[near_p1["click_gain_if_p3"] > 0].sort_values("click_gain_if_p3", ascending=False)
            if len(near_p1) >= 3:
                total_gain = float(near_p1["click_gain_if_p3"].sum())
                top3 = near_p1.head(3)[["query", "position", "impressions", "ctr"]].round(2).to_dict("records")
                insights.append({
                    "category": "cluster",
                    "title": f"{len(near_p1)} queries in positions 4-10 are close to a top-3 breakthrough",
                    "description": (
                        f"Small ranking push could yield ~{total_gain:.0f} additional clicks. "
                        f"Top: '{near_p1.iloc[0]['query']}' at position {near_p1.iloc[0]['position']:.1f} with {near_p1.iloc[0]['impressions']:.0f} impressions."
                    ),
                    "evidence": {
                        "near_page1_query_count": len(near_p1),
                        "estimated_additional_clicks_if_top3": round(total_gain, 0),
                        "top_opportunities": top3,
                    },
                    "traffic_impact_score": min(1.0, total_gain / total_clicks * 2),
                })
    except Exception:
        pass

    # ── Predefined: branded vs non-branded ───────────────────────────────────
    try:
        import re
        brand_hints = set()
        brand_match = re.search(r"(?:https?://)?(?:www\.)?([^./]+)", site or "")
        if brand_match:
            brand_hints.add(brand_match.group(1).lower())
        if brand_hints:
            df["is_branded"] = df["query"].str.lower().apply(lambda q: any(b in q for b in brand_hints))
            branded = df[df["is_branded"]]
            nonbranded = df[~df["is_branded"]]
            if len(branded) >= 3 and len(nonbranded) >= 10:
                b_clicks = branded["clicks"].sum()
                nb_clicks = nonbranded["clicks"].sum()
                b_ctr = branded["ctr"].mean()
                nb_ctr = nonbranded["ctr"].mean()
                b_pct = b_clicks / total_clicks * 100
                insights.append({
                    "category": "cluster",
                    "title": f"Branded queries account for {b_pct:.0f}% of total clicks at {b_ctr*100:.1f}% CTR",
                    "description": (
                        f"Branded queries ({len(branded)} keywords): {b_clicks:.0f} clicks at {b_ctr*100:.1f}% CTR. "
                        f"Non-branded ({len(nonbranded)} keywords): {nb_clicks:.0f} clicks at {nb_ctr*100:.1f}% CTR. "
                        f"{'High branded share — diversify non-branded content.' if b_pct > 30 else 'Strong non-branded organic presence.'}"
                    ),
                    "evidence": {
                        "branded_query_count": len(branded),
                        "branded_clicks": float(b_clicks),
                        "branded_avg_ctr": round(float(b_ctr), 4),
                        "nonbranded_clicks": float(nb_clicks),
                        "nonbranded_avg_ctr": round(float(nb_ctr), 4),
                        "branded_click_share_pct": round(float(b_pct), 1),
                    },
                    "traffic_impact_score": min(1.0, b_clicks / total_clicks * 0.5),
                })
    except Exception:
        pass

    # ── Open-ended: HDBSCAN on query feature vectors ──────────────────────────
    query_features = ["log_clicks", "log_impressions", "ctr", "position", "word_count", "is_question"]
    hdbscan_insights = hdbscan_clusters(
        df, query_features, label_col="query", value_col="clicks",
        total_value=total_clicks, category_tag="discovered_cluster", entity_name="queries"
    )
    insights.extend(hdbscan_insights)

    return insights


def main():
    try:
        payload = json.loads(sys.stdin.read())
        result = analyze(payload)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"insights": [], "error": str(e)}))


if __name__ == "__main__":
    main()
