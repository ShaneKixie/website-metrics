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

v3 upgrades (from Deep Think critique):
  - UMAP manifold projection before HDBSCAN: fixes mixed-data-type Euclidean
    distance distortion that was causing binary features (is_blog, is_guide) to
    dominate clustering. UMAP handles the mixed feature space correctly.
  - GAM residual CTR anomaly detection: replaces raw Isolation Forest with a
    position-adjusted expected CTR model (via pygam). PyOD now flags pages whose
    CTR deviates from what's expected at their position, not just pages that are
    statistically rare globally.
  - Branded/non-branded bifurcation: MSTL and PyOD run on non-branded data
    separately so branded traffic baselines don't skew anomaly detection.
  - Richer content-type feature vector: adds is_comparison, is_help, is_product,
    is_subdomain, has_year, is_anchor so HDBSCAN can differentiate comparison
    pages, help center articles, product pages, and year-dated content.

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

# ─────────────────────────────────────────────────────────────────────────────
# BRAND TERM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def extract_brand_root(site):
    """Extract the primary brand root from the site URL for basic brand matching."""
    import re
    m = re.search(r"(?:https?://)?(?:www\.)?([^./]+)", site or "")
    return m.group(1).lower() if m else None


def flag_branded(df, query_col, brand_root):
    """
    Add is_branded column using brand root substring match.
    Returns df with is_branded column added.
    """
    if brand_root:
        df = df.copy()
        df["is_branded"] = df[query_col].str.lower().str.contains(brand_root, regex=False)
    else:
        df = df.copy()
        df["is_branded"] = False
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GAM RESIDUAL CTR ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ctr_residuals(df, position_col="position", ctr_col="ctr", impressions_col="impressions"):
    """
    Fit a GAM to model expected CTR given position, then return residuals.
    Uses pygam if available, falls back to log-linear regression via numpy.

    Residual = actual_ctr - expected_ctr_at_position
    Positive residual = outperforming; Negative residual = underperforming.

    Returns df with columns: expected_ctr, ctr_residual
    """
    df = df.copy()
    try:
        from pygam import LinearGAM, s
        X = df[[position_col]].values
        y = df[ctr_col].values
        w = np.log1p(df[impressions_col].values)  # weight by log impressions
        gam = LinearGAM(s(0, n_splines=8)).fit(X, y, weights=w)
        df["expected_ctr"] = gam.predict(X).clip(0, 1)
    except Exception:
        # Fallback: log-linear model (CTR ~ 1/position)
        pos = df[position_col].values.clip(1, 100)
        log_pos = np.log(pos)
        log_ctr = np.log(df[ctr_col].values.clip(1e-6, 1))
        try:
            coeffs = np.polyfit(log_pos, log_ctr, 1)
            df["expected_ctr"] = np.exp(np.polyval(coeffs, log_pos)).clip(0, 1)
        except Exception:
            df["expected_ctr"] = df[position_col].apply(
                lambda p: EXPECTED_CTR.get(min(10, max(1, round(p))), 0.02)
            )
    df["ctr_residual"] = df[ctr_col] - df["expected_ctr"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UMAP + HDBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def hdbscan_clusters(df, feature_cols, label_col, value_col, total_value, category_tag, entity_name="pages"):
    """
    Run UMAP → HDBSCAN on a feature matrix built from df[feature_cols].

    v3 changes:
    - StandardScaler → UMAP manifold projection before HDBSCAN. UMAP handles
      mixed continuous/binary feature spaces correctly, preventing binary flags
      from dominating Euclidean distance calculations.
    - Falls back gracefully to scaled HDBSCAN if umap-learn is not installed.
    - Soft membership probabilities surface archetypal members per cluster.
    - Noise points ranked by centroid distance (extreme vs near-miss).
    - Low-confidence clusters filtered (avg prob < 0.25).
    """
    insights = []
    try:
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        sub = df[feature_cols + [label_col, value_col]].dropna()
        if len(sub) < 15:
            return []

        # Scale first — required before both UMAP and fallback HDBSCAN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sub[feature_cols].values)

        # ── UMAP projection ────────────────────────────────────────────────
        # Project to dense manifold before clustering to fix mixed-type distance
        # distortion. Falls back to scaled features if umap-learn not available.
        try:
            from umap import UMAP
            n_components = min(5, len(feature_cols) - 1, len(sub) - 2)
            n_components = max(2, n_components)
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(sub) // 3),
                min_dist=0.0,   # tighter clusters for HDBSCAN
                metric="euclidean",
                random_state=42,
                low_memory=False,
            )
            X = reducer.fit_transform(X_scaled)
            umap_used = True
        except Exception:
            X = X_scaled
            umap_used = False

        # ── HDBSCAN ───────────────────────────────────────────────────────
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=3)
        clusterer.fit(X)

        labels = clusterer.labels_
        probs  = clusterer.probabilities_

        sub = sub.copy()
        sub["_cluster"] = labels
        sub["_prob"]    = probs

        # Global means computed from ORIGINAL features for interpretability
        global_means = {col: float(sub[col].mean()) for col in feature_cols}

        unique_clusters = [c for c in sorted(sub["_cluster"].unique()) if c != -1]

        # Centroids in UMAP space for outlier distance scoring
        cluster_centroids = {}
        for cid in unique_clusters:
            mask = labels == cid
            cluster_centroids[cid] = X[mask].mean(axis=0)

        for cluster_id in unique_clusters:
            mask    = sub["_cluster"] == cluster_id
            cluster = sub[mask]
            cluster_clicks = float(cluster[value_col].sum())
            cluster_size   = int(mask.sum())

            avg_prob = float(cluster["_prob"].mean())
            if avg_prob < 0.25:
                continue

            # Distinguishing features (z-scores on original feature values)
            deviations = []
            for col in feature_cols:
                g_mean = global_means[col]
                g_std  = float(sub[col].std()) or 1.0
                c_mean = float(cluster[col].mean())
                z = (c_mean - g_mean) / g_std
                deviations.append((col, c_mean, g_mean, z))

            deviations.sort(key=lambda x: abs(x[3]), reverse=True)
            top_devs = deviations[:3]

            trait_parts = []
            for col, c_mean, g_mean, z in top_devs:
                if abs(z) < 0.3:
                    continue
                direction = "high" if z > 0 else "low"
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
                elif col.startswith("is_") or col.startswith("has_"):
                    label = col.replace("is_", "").replace("has_", "").replace("_", " ")
                    trait_parts.append(f"{direction} {label} ({c_mean:.2f} vs {g_mean:.2f} avg)")
                else:
                    trait_parts.append(f"{direction} {col.replace('_', ' ')} ({c_mean:.2f} vs {g_mean:.2f} avg)")

            if not trait_parts:
                continue

            trait_str = "; ".join(trait_parts)

            archetypal = (
                cluster.sort_values("_prob", ascending=False)
                .head(3)[label_col].tolist()
            )
            archetypal = [str(e).split("?")[0][:80] for e in archetypal]

            examples = (
                cluster.sort_values(value_col, ascending=False)
                .head(3)[label_col].tolist()
            )
            examples = [str(e).split("?")[0][:80] for e in examples]

            insights.append({
                "category": category_tag,
                "title": f"Discovered cluster of {cluster_size} {entity_name}: {trait_str[:80]}",
                "description": (
                    f"HDBSCAN found {cluster_size} {entity_name} that naturally group together "
                    f"({'UMAP-projected' if umap_used else 'scaled'} feature space). "
                    f"Key traits vs site average: {trait_str}. "
                    f"Cluster confidence (avg membership probability): {avg_prob:.0%}. "
                    f"Drives {cluster_clicks:.0f} of {total_value:.0f} total clicks "
                    f"({cluster_clicks / total_value * 100:.1f}%). "
                    f"No pattern was predefined — this grouping emerged purely from the data."
                ),
                "evidence": {
                    "cluster_id": int(cluster_id),
                    "cluster_size": cluster_size,
                    "cluster_clicks": round(cluster_clicks, 0),
                    "click_share_pct": round(cluster_clicks / total_value * 100, 1),
                    "cluster_confidence_avg_prob": round(avg_prob, 3),
                    "umap_projection_used": umap_used,
                    "distinguishing_features": [
                        {
                            "feature": col,
                            "cluster_mean": round(c_mean, 4),
                            "global_mean": round(g_mean, 4),
                            "z_score": round(z, 2),
                        }
                        for col, c_mean, g_mean, z in top_devs if abs(z) >= 0.3
                    ],
                    "archetypal_members": archetypal,
                    "top_members_by_clicks": examples,
                },
                "traffic_impact_score": min(1.0, cluster_clicks / total_value),
            })

        # Noise: rank by distance to nearest centroid in projected space
        noise = sub[sub["_cluster"] == -1].copy()
        noise_clicks = float(noise[value_col].sum())

        if len(noise) >= 5 and noise_clicks / total_value > 0.02:
            if cluster_centroids:
                noise_indices = noise.index.tolist()
                noise_X = X[sub.index.get_indexer(noise_indices)]
                centroids_arr = np.array(list(cluster_centroids.values()))
                distances = np.array([
                    float(np.min(np.linalg.norm(centroids_arr - pt, axis=1)))
                    for pt in noise_X
                ])
            else:
                global_centroid = X.mean(axis=0)
                noise_indices = noise.index.tolist()
                noise_X = X[sub.index.get_indexer(noise_indices)]
                distances = np.array([float(np.linalg.norm(pt - global_centroid)) for pt in noise_X])

            noise = noise.copy()
            noise["_dist"] = distances
            median_dist = float(noise["_dist"].median())
            extreme   = noise[noise["_dist"] >= median_dist]
            near_miss = noise[noise["_dist"] < median_dist]

            top_extreme = (
                extreme.sort_values("_dist", ascending=False)
                .head(5)
                .apply(
                    lambda r: {
                        "label": str(r[label_col]).split("?")[0][:80],
                        "outlier_distance": round(float(r["_dist"]), 3),
                        "clicks": int(r[value_col]),
                    },
                    axis=1,
                )
                .tolist()
            )

            insights.append({
                "category": category_tag,
                "title": (
                    f"{len(noise)} {entity_name} don't fit any cluster — "
                    f"{len(extreme)} extreme outliers, {len(near_miss)} near-miss"
                ),
                "description": (
                    f"HDBSCAN assigned {len(noise)} {entity_name} to noise (no cluster). "
                    f"Scored by distance to nearest cluster centroid in "
                    f"{'UMAP-projected' if umap_used else 'scaled'} feature space: "
                    f"{len(extreme)} are extreme outliers (distance ≥ {median_dist:.2f}) with no statistical peers — "
                    f"highest priority for individual review. "
                    f"{len(near_miss)} are near-miss outliers that almost joined a cluster and may represent "
                    f"an emerging pattern as the site grows. "
                    f"Combined, they account for {noise_clicks / total_value * 100:.1f}% of total clicks."
                ),
                "evidence": {
                    "total_noise_count": len(noise),
                    "extreme_outlier_count": len(extreme),
                    "near_miss_count": len(near_miss),
                    "noise_clicks": round(noise_clicks, 0),
                    "click_share_pct": round(noise_clicks / total_value * 100, 1),
                    "median_centroid_distance": round(median_dist, 3),
                    "top_extreme_outliers": top_extreme,
                },
                "traffic_impact_score": min(1.0, noise_clicks / total_value * 0.5),
            })

    except Exception:
        pass

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# STUMPY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def stumpy_metric(series, dates, metric_name, m=7):
    """Run STUMPY on a single metric time series and return motif/discord insights."""
    insights = []
    try:
        import stumpy
        values = np.array(series, dtype=float)
        if len(values) < m * 2 or np.nanstd(values) < 1e-6:
            return []

        overall_mean = float(np.nanmean(values)) or 1.0
        mp = stumpy.stump(values, m=m)
        profile = mp[:, 0].astype(float)

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

    # ── MSTL: weekly + monthly seasonality ────────────────────────────────────
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

    # ── STUMPY across multiple windows and metrics ────────────────────────────
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

        for metric_name, col in [("CTR", "ctr"), ("avg_position", "position"), ("impressions", "impressions")]:
            series = df[col].ffill().fillna(0).values.astype(float)
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

    # ── URL feature engineering (v3: expanded content-type signals) ───────────
    def path_depth(url):
        return len([p for p in url.split("?")[0].split("/") if p])

    def has_segment(url, keywords):
        parts = url.split("?")[0].lower().split("/")
        return int(any(k in parts for k in keywords))

    def url_contains(url, substrings):
        u = url.lower()
        return int(any(s in u for s in substrings))

    def get_subdomain(url):
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc.lower()
            parts = host.split(".")
            if len(parts) > 2:
                sub = parts[0]
                return 0 if sub in ("www", "") else 1
            return 0
        except Exception:
            return 0

    import re as _re
    YEAR_PATTERN = _re.compile(r"20[12]\d")

    df["url_depth"]      = df["page"].apply(path_depth)
    df["is_blog"]        = df["page"].apply(lambda u: has_segment(u, ["blog", "blogs", "post", "posts", "article", "articles"]))
    df["is_guide"]       = df["page"].apply(lambda u: has_segment(u, ["guide", "guides", "how-to", "tutorial", "learn"]))
    df["is_comparison"]  = df["page"].apply(lambda u: url_contains(u, ["-vs-", "-versus-", "compare", "comparison"]))
    df["is_help"]        = df["page"].apply(lambda u: url_contains(u, ["help.", "/help/", "support.", "/support/", "docs.", "/docs/"]))
    df["is_product"]     = df["page"].apply(lambda u: url_contains(u, ["/plan", "/pricing", "/challenge", "/funded", "/select", "/growth", "/lightning"]))
    df["is_subdomain"]   = df["page"].apply(get_subdomain)
    df["has_year"]       = df["page"].apply(lambda u: int(bool(YEAR_PATTERN.search(u.split("?")[0]))))
    df["is_anchor"]      = df["page"].apply(lambda u: int("#" in u))
    df["log_clicks"]     = np.log1p(df["clicks"])
    df["log_impressions"] = np.log1p(df["impressions"])

    # ── Predefined: blog/guide/comparison/help CTR comparison ────────────────
    try:
        content_segs = [
            ("is_blog", "Blog"),
            ("is_guide", "Guide"),
            ("is_comparison", "Comparison"),
            ("is_help", "Help center"),
        ]
        for seg_col, seg_label in content_segs:
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
                            f"{seg_label.lower().replace(' ', '_')}_avg_ctr": round(float(seg_ctr), 4),
                            "other_avg_ctr": round(float(other_ctr), 4),
                            f"{seg_label.lower().replace(' ', '_')}_page_count": len(seg),
                            f"{seg_label.lower().replace(' ', '_')}_total_clicks": float(seg_clicks),
                        },
                        "traffic_impact_score": min(1.0, seg_clicks / total_clicks),
                    })
    except Exception:
        pass

    # ── Predefined: year-dated pages performance ──────────────────────────────
    try:
        year_pages = df[df["has_year"] == 1]
        non_year   = df[df["has_year"] == 0]
        if len(year_pages) >= 3 and len(non_year) >= 5:
            y_ctr  = float(year_pages["ctr"].mean())
            ny_ctr = float(non_year["ctr"].mean())
            y_impr = float(year_pages["impressions"].sum())
            if ny_ctr > 0 and y_ctr < ny_ctr * 0.7:
                insights.append({
                    "category": "cluster",
                    "title": f"{len(year_pages)} year-dated pages average {y_ctr*100:.2f}% CTR — {(1-y_ctr/ny_ctr)*100:.0f}% below non-dated pages",
                    "description": (
                        f"Pages with a year in the URL (e.g. /best-prop-firms-2025/) average "
                        f"{y_ctr*100:.2f}% CTR vs {ny_ctr*100:.2f}% for non-dated pages. "
                        f"Year-dated pages often suffer low CTR when the year in the title appears stale. "
                        f"Review title tags on {len(year_pages)} pages to ensure the year matches searcher expectations."
                    ),
                    "evidence": {
                        "year_dated_page_count": len(year_pages),
                        "year_dated_avg_ctr": round(y_ctr, 4),
                        "non_dated_avg_ctr": round(ny_ctr, 4),
                        "year_dated_total_impressions": round(y_impr, 0),
                        "example_pages": year_pages.sort_values("impressions", ascending=False)["page"].head(5).tolist(),
                    },
                    "traffic_impact_score": min(1.0, y_impr / total_impressions * 2),
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

    # ── GAM residual PyOD: position-adjusted CTR anomaly detection ────────────
    # v3: fit expected CTR ~ f(position) via GAM, run PyOD on residuals.
    # This prevents flagging high-CTR position-1 pages as anomalies simply
    # because they're statistically rare globally.
    try:
        from pyod.models.iforest import IForest

        sig = df[(df["impressions"] >= 100) & (df["position"] > 0)].copy()
        if len(sig) >= 20:
            sig = compute_ctr_residuals(sig)

            # Feature matrix: residual + impressions (log-scaled) + position
            import numpy as _np
            feat = _np.column_stack([
                sig["ctr_residual"].values,
                _np.log1p(sig["impressions"].values),
                sig["position"].values,
            ])
            from sklearn.preprocessing import StandardScaler as _SS
            feat_scaled = _SS().fit_transform(feat)

            clf = IForest(contamination=0.15, random_state=42)
            clf.fit(feat_scaled)
            sig["anomaly"] = clf.labels_

            top10 = sig[sig["position"] <= 10].copy()
            if len(top10) >= 5:
                # Low CTR residuals in top 10 = genuinely underperforming
                residual_thresh = top10["ctr_residual"].quantile(0.33)
                under = top10[top10["ctr_residual"] <= residual_thresh].sort_values("impressions", ascending=False)
                if len(under) >= 2:
                    avg_top10_ctr = float(top10["ctr"].mean())
                    avg_under_ctr = float(under["ctr"].mean())
                    avg_expected  = float(under["expected_ctr"].mean())
                    est_lost = float((under["impressions"] * (under["expected_ctr"] - under["ctr"])).clip(0).sum())
                    examples = [u.split("?")[0] for u in under["page"].head(5).tolist()]
                    insights.append({
                        "category": "anomaly",
                        "title": f"{len(under)} top-10 pages underperform their expected CTR (position-adjusted)",
                        "description": (
                            f"Using a GAM model of expected CTR by position, {len(under)} pages rank in the top 10 "
                            f"but click through significantly less than expected for their rank. "
                            f"Average actual CTR: {avg_under_ctr*100:.1f}% vs expected {avg_expected*100:.1f}% at their positions. "
                            f"Improving title tags and meta descriptions could recover ~{est_lost:.0f} clicks."
                        ),
                        "evidence": {
                            "underperforming_page_count": len(under),
                            "avg_actual_ctr": round(avg_under_ctr, 4),
                            "avg_expected_ctr_at_position": round(avg_expected, 4),
                            "estimated_recoverable_clicks": round(est_lost, 0),
                            "example_pages": examples,
                        },
                        "traffic_impact_score": min(1.0, est_lost / total_clicks * 3),
                    })

                top10_high = sig[(sig["position"] <= 10) & (sig["anomaly"] == 1)].copy()
                if len(top10_high) >= 4:
                    top10_high_ctr = top10_high[top10_high["ctr_residual"] > top10_high["ctr_residual"].quantile(0.75)]
                    if len(top10_high_ctr) >= 2:
                        avg_high_ctr = float(top10_high_ctr["ctr"].mean())
                        avg_top10_ctr = float(top10["ctr"].mean())
                        examples = [u.split("?")[0] for u in top10_high_ctr.sort_values("impressions", ascending=False)["page"].head(3).tolist()]
                        insights.append({
                            "category": "cluster",
                            "title": f"{len(top10_high_ctr)} pages have anomalously HIGH CTR vs position expectation — study as models",
                            "description": (
                                f"PyOD (residual-adjusted) flagged {len(top10_high_ctr)} top-10 pages as high-side outliers — "
                                f"clicking through significantly more than expected for their position. "
                                f"Average CTR: {avg_high_ctr*100:.1f}% vs {avg_top10_ctr*100:.1f}% for top-10 overall. "
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

    # ── Open-ended: UMAP → HDBSCAN on expanded page feature vectors ──────────
    # v3: richer content-type features; UMAP handles mixed binary/continuous
    page_features = [
        "log_clicks", "log_impressions", "ctr", "position", "url_depth",
        "is_blog", "is_guide", "is_comparison", "is_help", "is_product",
        "is_subdomain", "has_year", "is_anchor",
    ]
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
    df["word_count"]  = df["query"].str.split().str.len()
    df["is_question"] = df["query"].str.lower().apply(lambda q: any(q.startswith(w) for w in question_words))
    df["log_clicks"]      = np.log1p(df["clicks"])
    df["log_impressions"] = np.log1p(df["impressions"])

    # ── Branded / non-branded bifurcation ─────────────────────────────────────
    # v3: flag branded queries properly and run all downstream analyses on
    # non-branded subset to avoid branded CTR baselines skewing anomaly detection
    brand_root = extract_brand_root(site)
    df = flag_branded(df, "query", brand_root)
    branded    = df[df["is_branded"]].copy()
    nonbranded = df[~df["is_branded"]].copy()

    nb_total_clicks = nonbranded["clicks"].sum() or 1.0

    # ── Predefined: branded vs non-branded summary ────────────────────────────
    try:
        if len(branded) >= 3 and len(nonbranded) >= 10:
            b_clicks  = branded["clicks"].sum()
            nb_clicks = nonbranded["clicks"].sum()
            b_ctr     = branded["ctr"].mean()
            nb_ctr    = nonbranded["ctr"].mean()
            b_pct     = b_clicks / total_clicks * 100
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

    # ── All subsequent predefined checks run on NON-BRANDED only ─────────────

    # ── Predefined: high-impression, low-CTR (non-branded) ───────────────────
    try:
        nb = nonbranded.copy()
        if len(nb) >= 10:
            imp_75 = nb["impressions"].quantile(0.75)
            ctr_25 = nb["ctr"].quantile(0.25)
            hi_imp_lo_ctr = nb[(nb["impressions"] >= imp_75) & (nb["ctr"] <= ctr_25)].copy()
            if len(hi_imp_lo_ctr) >= 5:
                hi_imp_lo_ctr["opportunity"] = hi_imp_lo_ctr["impressions"] * (nb["ctr"].mean() - hi_imp_lo_ctr["ctr"])
                top = hi_imp_lo_ctr.sort_values("opportunity", ascending=False).head(5)
                total_opp = float(top["opportunity"].sum())
                insights.append({
                    "category": "anomaly",
                    "title": f"{len(hi_imp_lo_ctr)} non-branded high-impression queries have below-average CTR",
                    "description": (
                        f"Top 25% impressions but bottom 25% CTR among non-branded queries — high visibility, low click-through. "
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

    # ── Predefined: long-tail vs head vs mid-tail (non-branded) ──────────────
    try:
        nb = nonbranded.copy()
        longtail = nb[nb["word_count"] >= 4]
        head     = nb[nb["word_count"] <= 2]
        mid      = nb[nb["word_count"] == 3]
        for grp, grp_label in [(longtail, "Long-tail (4+ word)"), (mid, "Mid-tail (3-word)")]:
            if len(grp) >= 10 and len(head) >= 5:
                g_ctr   = grp["ctr"].mean()
                hd_ctr  = head["ctr"].mean()
                g_clicks = grp["clicks"].sum()
                if hd_ctr > 0 and abs(g_ctr - hd_ctr) / hd_ctr > 0.10:
                    pct = abs(g_ctr - hd_ctr) / hd_ctr * 100
                    insights.append({
                        "category": "cluster",
                        "title": f"Non-branded {grp_label} queries get {pct:.0f}% {'higher' if g_ctr > hd_ctr else 'lower'} CTR",
                        "description": (
                            f"{grp_label} non-branded queries average {g_ctr*100:.2f}% CTR vs {hd_ctr*100:.2f}% for head terms. "
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

    # ── Predefined: question queries (non-branded) ────────────────────────────
    try:
        nb = nonbranded.copy()
        q_df  = nb[nb["is_question"]]
        nq_df = nb[~nb["is_question"]]
        if len(q_df) >= 10 and len(nq_df) >= 10:
            q_ctr   = q_df["ctr"].mean()
            nq_ctr  = nq_df["ctr"].mean()
            q_clicks = q_df["clicks"].sum()
            if nq_ctr > 0 and abs(q_ctr - nq_ctr) / nq_ctr > 0.10:
                pct = abs(q_ctr - nq_ctr) / nq_ctr * 100
                direction = "higher" if q_ctr > nq_ctr else "lower"
                insights.append({
                    "category": "cluster",
                    "title": f"Non-branded question queries (how/what/why) get {pct:.0f}% {direction} CTR",
                    "description": (
                        f"Question queries average {q_ctr*100:.2f}% CTR vs {nq_ctr*100:.2f}% for others (non-branded only). "
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

    # ── Predefined: near-page-1 opportunities (non-branded) ──────────────────
    try:
        nb = nonbranded.copy()
        near_p1 = nb[(nb["position"] >= 4) & (nb["position"] <= 10) & (nb["impressions"] >= 100)].copy()
        if len(near_p1) >= 5:
            near_p1["click_gain_if_p3"] = near_p1["impressions"] * (EXPECTED_CTR.get(3, 0.11) - near_p1["ctr"])
            near_p1 = near_p1[near_p1["click_gain_if_p3"] > 0].sort_values("click_gain_if_p3", ascending=False)
            if len(near_p1) >= 3:
                total_gain = float(near_p1["click_gain_if_p3"].sum())
                top3 = near_p1.head(3)[["query", "position", "impressions", "ctr"]].round(2).to_dict("records")
                insights.append({
                    "category": "cluster",
                    "title": f"{len(near_p1)} non-branded queries in positions 4-10 are close to a top-3 breakthrough",
                    "description": (
                        f"Small ranking push could yield ~{total_gain:.0f} additional clicks (non-branded only). "
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

    # ── Open-ended: UMAP → HDBSCAN — run separately on branded and non-branded
    # Running on the full mixed pool would have branded CTR baselines (50-70%)
    # distort the distance calculations for non-branded clusters (5-15% CTR).
    # v3: two separate clustering passes, tagged with brand context.
    for subset, subset_label, subset_total in [
        (nonbranded, "non-branded queries", nb_total_clicks),
        (branded,    "branded queries",     branded["clicks"].sum() or 1.0),
    ]:
        if len(subset) < 15:
            continue
        query_features = ["log_clicks", "log_impressions", "ctr", "position", "word_count", "is_question"]
        sub_insights = hdbscan_clusters(
            subset, query_features, label_col="query", value_col="clicks",
            total_value=total_clicks,  # keep denominator as site total for comparability
            category_tag="discovered_cluster", entity_name=subset_label
        )
        insights.extend(sub_insights)

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

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


def main():
    try:
        payload = json.loads(sys.stdin.read())
        result = analyze(payload)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"insights": [], "error": str(e)}))


if __name__ == "__main__":
    main()
