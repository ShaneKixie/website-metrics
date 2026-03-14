#!/usr/bin/env python3
"""
Pattern Discovery Engine

Reads a JSON payload from stdin, runs statistical analysis using STUMPY,
statsmodels MSTL, and PyOD, and writes ranked insights as JSON to stdout.

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

# Industry CTR benchmarks by position (Sistrix / Advanced Web Ranking averages)
EXPECTED_CTR = {1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07,
                6: 0.06, 7: 0.05, 8: 0.04, 9: 0.04, 10: 0.03}

# Maximum insights returned (raised from 25)
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


# ── Daily GSC: STUMPY motifs/discords + MSTL seasonality + trend ─────────────

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
    total_clicks = df["clicks"].sum()

    # ── MSTL seasonality — weekly + monthly cycles ───────────────────────────
    try:
        from statsmodels.tsa.seasonal import MSTL

        clicks = df["clicks"].fillna(0).values

        # Weekly seasonality (7-day period)
        if len(clicks) >= 14:
            mstl7 = MSTL(clicks, periods=[7])
            res7 = mstl7.fit()
            trend = res7.trend

            # Day-of-week effect
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

                # Lowered threshold: 5% (was 10%)
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

            # Trend direction — lowered threshold: 5% (was 10%)
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

        # Monthly seasonality (28-day period) — requires 56+ days
        if len(clicks) >= 56:
            try:
                mstl28 = MSTL(clicks, periods=[7, 28])
                res28 = mstl28.fit()
                # Monthly seasonal component is the second column
                monthly_seasonal = res28.seasonal[:, 1] if res28.seasonal.ndim > 1 else None
                if monthly_seasonal is not None:
                    monthly_amplitude = float(np.nanstd(monthly_seasonal))
                    overall_mean = float(np.nanmean(clicks)) or 1.0
                    monthly_pct = monthly_amplitude / overall_mean * 100
                    if monthly_pct > 5:
                        # Find the highest and lowest weeks in the monthly cycle
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

    # ── STUMPY: multiple window sizes for more pattern depth ─────────────────
    try:
        import stumpy

        clicks = df["clicks"].fillna(0).values.astype(float)
        overall_mean = float(clicks.mean()) or 1.0

        # Run matrix profile for each window size: 7 (weekly), 14 (biweekly), 28 (monthly)
        window_configs = []
        if len(clicks) >= 28:
            window_configs.append((7, "weekly"))
        if len(clicks) >= 56:
            window_configs.append((14, "biweekly"))
        if len(clicks) >= 84:
            window_configs.append((28, "monthly"))

        for m, label in window_configs:
            try:
                mp = stumpy.stump(clicks, m=m)
                profile = mp[:, 0].astype(float)

                # Top 3 motifs for this window
                profile_copy = profile.copy()
                motif_count = 0
                motif_indices_used = []

                for _ in range(3):
                    if np.all(np.isinf(profile_copy)):
                        break
                    motif_idx = int(np.argmin(profile_copy))
                    # Exclude zone around this motif (avoid near-duplicates)
                    excl_start = max(0, motif_idx - m)
                    excl_end = min(len(profile_copy), motif_idx + m)
                    profile_copy[excl_start:excl_end] = np.inf

                    motif_clicks = float(clicks[motif_idx:motif_idx + m].mean())
                    motif_date = str(df.index[motif_idx].date())
                    motif_ratio = motif_clicks / overall_mean

                    # Lowered threshold: 10% deviation (was 15%)
                    if abs(motif_ratio - 1) > 0.10:
                        motif_count += 1
                        label_str = "above-average" if motif_ratio > 1 else "below-average"
                        insights.append({
                            "category": "motif",
                            "title": f"Recurring {label} pattern #{motif_count}: first seen {motif_date} ({motif_ratio:.2f}x avg)",
                            "description": (
                                f"STUMPY found a repeating {m}-day traffic pattern that is {label_str} — "
                                f"averaging {motif_ratio:.2f}x your typical daily clicks ({motif_clicks:.0f} vs {overall_mean:.0f}). "
                                f"This pattern recurs consistently, suggesting a structural driver such as "
                                f"content cadence, external promotion, or search algorithm preference."
                            ),
                            "evidence": {
                                "first_occurrence_date": motif_date,
                                "avg_clicks_during_pattern": round(motif_clicks, 1),
                                "overall_avg_clicks": round(overall_mean, 1),
                                "ratio_vs_average": round(motif_ratio, 2),
                                "window_days": m,
                                "window_label": label,
                            },
                            "traffic_impact_score": min(1.0, abs(motif_ratio - 1) * 0.6),
                        })

                # Top 3 discords for this window
                profile_copy = profile.copy()
                discord_count = 0

                for _ in range(3):
                    if np.all(np.isinf(profile_copy)):
                        break
                    discord_idx = int(np.argmax(profile_copy))
                    excl_start = max(0, discord_idx - m)
                    excl_end = min(len(profile_copy), discord_idx + m)
                    profile_copy[excl_start:excl_end] = -np.inf

                    if discord_idx + m > len(df):
                        continue

                    discord_clicks = float(clicks[discord_idx:discord_idx + m].mean())
                    discord_date = str(df.index[discord_idx].date())
                    deviation = abs(discord_clicks - overall_mean) / overall_mean

                    # Lowered threshold: 20% (was 30%)
                    if deviation > 0.20:
                        discord_count += 1
                        direction = "spike" if discord_clicks > overall_mean else "drop"
                        insights.append({
                            "category": "anomaly",
                            "title": f"Anomalous {label} period #{discord_count}: {direction} of {deviation*100:.0f}% week of {discord_date}",
                            "description": (
                                f"The {m}-day window starting {discord_date} is statistically unusual — "
                                f"a traffic {direction} of {deviation*100:.0f}% from baseline "
                                f"({discord_clicks:.0f} vs {overall_mean:.0f} avg clicks/day). "
                                f"Investigate what changed: algorithm update, content publish, link acquisition, or site issue."
                            ),
                            "evidence": {
                                "date": discord_date,
                                "avg_clicks_that_period": round(discord_clicks, 1),
                                "baseline_avg_clicks": round(overall_mean, 1),
                                "deviation_pct": round(deviation * 100, 1),
                                "direction": direction,
                                "window_days": m,
                                "window_label": label,
                            },
                            "traffic_impact_score": min(1.0, deviation * 0.7),
                        })
            except Exception:
                pass
    except Exception:
        pass

    return insights


# ── Per-page GSC: PyOD outliers + URL clustering + CTR vs benchmark ──────────

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

    # ── URL pattern clustering ──────────────────────────────────────────────
    try:
        def path_depth(url):
            return len([p for p in url.split("?")[0].split("/") if p])

        def has_segment(url, keywords):
            parts = url.split("?")[0].lower().split("/")
            return int(any(k in parts for k in keywords))

        df["url_depth"] = df["page"].apply(path_depth)
        df["is_blog"] = df["page"].apply(lambda u: has_segment(u, ["blog", "blogs", "post", "posts", "article", "articles"]))
        df["is_guide"] = df["page"].apply(lambda u: has_segment(u, ["guide", "guides", "how-to", "tutorial", "learn"]))

        # Blog vs non-blog
        for seg_col, seg_label in [("is_blog", "Blog"), ("is_guide", "Guide")]:
            seg = df[df[seg_col] == 1]
            non_seg = df[df[seg_col] == 0]
            if len(seg) >= 5 and len(non_seg) >= 5:
                seg_ctr = seg["ctr"].mean()
                other_ctr = non_seg["ctr"].mean()
                seg_clicks = seg["clicks"].sum()
                # Lowered threshold: 10% (was 15%)
                if other_ctr > 0 and abs(seg_ctr - other_ctr) / other_ctr > 0.10:
                    higher = seg_label if seg_ctr > other_ctr else "Non-" + seg_label
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

        # URL depth vs CTR
        depth_perf = df.groupby("url_depth").agg(
            avg_ctr=("ctr", "mean"), total_clicks=("clicks", "sum"), count=("page", "count")
        ).reset_index()
        if len(depth_perf) > 1:
            best = depth_perf.loc[depth_perf["avg_ctr"].idxmax()]
            worst = depth_perf.loc[depth_perf["avg_ctr"].idxmin()]
            overall_ctr = df["ctr"].mean()
            # Lowered threshold: 1.15x (was 1.2x)
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

    # ── PyOD: anomalously low CTR pages in top-10 positions ─────────────────
    try:
        from pyod.models.iforest import IForest
        from sklearn.preprocessing import StandardScaler

        sig = df[df["impressions"] >= 100].copy()
        if len(sig) >= 20:
            X = StandardScaler().fit_transform(sig[["position", "ctr", "impressions"]].values)
            # Increased contamination: 0.15 (was 0.10) — surfaces more anomalies
            clf = IForest(contamination=0.15, random_state=42)
            clf.fit(X)
            sig["anomaly"] = clf.labels_

            top10 = sig[sig["position"] <= 10].copy()
            if len(top10) >= 5:
                # Report bottom third (was bottom quartile) to surface more
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
                            f"These pages rank in the top 10 but their CTR is in the bottom third for that position band. "
                            f"Average CTR: {avg_under_ctr*100:.1f}% vs {avg_top10_ctr*100:.1f}% for top-10 overall. "
                            f"Improving title tags and meta descriptions on these pages could recover ~{est_lost:.0f} clicks."
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

            # Also flag pages anomalously HIGH CTR for top-10 — models to replicate
            top10_high = sig[(sig["position"] <= 10) & (sig["anomaly"] == 1)].copy()
            top10_high_ctr = top10_high[top10_high["ctr"] > top10_high["ctr"].quantile(0.75)] if len(top10_high) >= 4 else pd.DataFrame()
            if len(top10_high_ctr) >= 2:
                avg_high_ctr = float(top10_high_ctr["ctr"].mean())
                avg_top10_ctr = float(top10["ctr"].mean()) if len(top10) > 0 else avg_high_ctr
                examples = [u.split("?")[0] for u in top10_high_ctr.sort_values("impressions", ascending=False)["page"].head(3).tolist()]
                insights.append({
                    "category": "cluster",
                    "title": f"{len(top10_high_ctr)} pages have anomalously HIGH CTR — study these as models",
                    "description": (
                        f"PyOD flagged {len(top10_high_ctr)} top-10 pages as outliers on the high side — "
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

    # ── CTR vs industry benchmark by position ────────────────────────────────
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
                # Lowered threshold: 0.01 (was 0.02)
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
                "title": f"CTR is below industry benchmark across {len(gaps)} position bucket(s)",
                "description": (
                    f"Worst gap at position {worst['position']}: your pages get {worst['actual_ctr_pct']}% CTR "
                    f"vs the {worst['expected_ctr_pct']}% industry average — leaving ~{worst['missed_clicks_estimate']:.0f} clicks unrealized. "
                    f"Total estimated missed clicks across all gap positions: {total_missed:.0f}. "
                    f"Focus on improving title and meta description quality for these positions."
                ),
                "evidence": {"position_gaps": gaps, "total_missed_clicks_estimate": round(total_missed, 0)},
                "traffic_impact_score": min(1.0, total_missed / total_clicks * 2),
            })
    except Exception:
        pass

    return insights


# ── Per-query GSC: intent patterns, long-tail, high-impression low-CTR ───────

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

    # ── High-impression, low-CTR queries (quick wins) ────────────────────────
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
                    f"These queries are in your top 25% for impressions but bottom 25% for CTR — high visibility, low click-through. "
                    f"Top example: '{top.iloc[0]['query']}' gets {top.iloc[0]['impressions']:.0f} impressions at only {top.iloc[0]['ctr']*100:.1f}% CTR (position {top.iloc[0]['position']:.1f}). "
                    f"Fixing title/meta on these could yield ~{total_opp:.0f} additional clicks."
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

    # ── Long-tail vs head term performance ───────────────────────────────────
    try:
        df["word_count"] = df["query"].str.split().str.len()
        longtail = df[df["word_count"] >= 4]
        head = df[df["word_count"] <= 2]
        mid = df[(df["word_count"] == 3)]

        if len(longtail) >= 10 and len(head) >= 5:
            lt_ctr = longtail["ctr"].mean()
            hd_ctr = head["ctr"].mean()
            lt_clicks = longtail["clicks"].sum()
            # Lowered threshold: 10% (was 15%)
            if hd_ctr > 0 and abs(lt_ctr - hd_ctr) / hd_ctr > 0.10:
                higher = "Long-tail (4+ word)" if lt_ctr > hd_ctr else "Head term (1-2 word)"
                pct = abs(lt_ctr - hd_ctr) / hd_ctr * 100
                insights.append({
                    "category": "cluster",
                    "title": f"{higher} queries get {pct:.0f}% {'higher' if lt_ctr > hd_ctr else 'lower'} CTR",
                    "description": (
                        f"4+ word queries average {lt_ctr*100:.2f}% CTR vs {hd_ctr*100:.2f}% for 1-2 word head terms. "
                        f"Long-tail queries account for {len(longtail)} keywords driving {lt_clicks:.0f} clicks "
                        f"({lt_clicks/total_clicks*100:.0f}% of total). "
                        f"{'Targeting more specific long-tail terms could improve click efficiency.' if lt_ctr > hd_ctr else 'Head terms are outperforming — focus on broad keyword optimization.'}"
                    ),
                    "evidence": {
                        "longtail_avg_ctr": round(float(lt_ctr), 4),
                        "head_avg_ctr": round(float(hd_ctr), 4),
                        "longtail_clicks": float(lt_clicks),
                        "longtail_query_count": len(longtail),
                        "head_query_count": len(head),
                    },
                    "traffic_impact_score": min(1.0, lt_clicks / total_clicks),
                })

        # Also compare 3-word mid-tail if enough data
        if len(mid) >= 10 and len(head) >= 5:
            mid_ctr = mid["ctr"].mean()
            hd_ctr = head["ctr"].mean()
            mid_clicks = mid["clicks"].sum()
            if hd_ctr > 0 and abs(mid_ctr - hd_ctr) / hd_ctr > 0.10:
                pct = abs(mid_ctr - hd_ctr) / hd_ctr * 100
                insights.append({
                    "category": "cluster",
                    "title": f"Mid-tail (3-word) queries get {pct:.0f}% {'higher' if mid_ctr > hd_ctr else 'lower'} CTR than head terms",
                    "description": (
                        f"3-word queries average {mid_ctr*100:.2f}% CTR vs {hd_ctr*100:.2f}% for 1-2 word head terms. "
                        f"Mid-tail queries drive {mid_clicks:.0f} clicks ({mid_clicks/total_clicks*100:.0f}% of total) across {len(mid)} keywords."
                    ),
                    "evidence": {
                        "midtail_avg_ctr": round(float(mid_ctr), 4),
                        "head_avg_ctr": round(float(hd_ctr), 4),
                        "midtail_clicks": float(mid_clicks),
                        "midtail_query_count": len(mid),
                    },
                    "traffic_impact_score": min(1.0, mid_clicks / total_clicks * 0.5),
                })
    except Exception:
        pass

    # ── Question queries vs non-question ─────────────────────────────────────
    try:
        question_words = ("how", "what", "why", "when", "where", "who", "which", "can", "does", "is ", "are ", "will ")
        df["is_question"] = df["query"].str.lower().apply(lambda q: any(q.startswith(w) for w in question_words))
        q_df = df[df["is_question"]]
        nq_df = df[~df["is_question"]]

        if len(q_df) >= 10 and len(nq_df) >= 10:
            q_ctr = q_df["ctr"].mean()
            nq_ctr = nq_df["ctr"].mean()
            q_clicks = q_df["clicks"].sum()
            q_impressions = q_df["impressions"].sum()
            # Lowered threshold: 10% (was 20%)
            if nq_ctr > 0 and abs(q_ctr - nq_ctr) / nq_ctr > 0.10:
                pct = abs(q_ctr - nq_ctr) / nq_ctr * 100
                direction = "higher" if q_ctr > nq_ctr else "lower"
                insights.append({
                    "category": "cluster",
                    "title": f"Question queries (how/what/why) get {pct:.0f}% {direction} CTR",
                    "description": (
                        f"Queries starting with how/what/why/etc average {q_ctr*100:.2f}% CTR vs {nq_ctr*100:.2f}% for other queries. "
                        f"You have {len(q_df)} question queries with {q_impressions:.0f} total impressions driving {q_clicks:.0f} clicks. "
                        f"{'FAQ-style content and featured snippet optimization could amplify this.' if q_ctr > nq_ctr else 'Non-question informational content is outperforming — review question-based page titles.'}"
                    ),
                    "evidence": {
                        "question_avg_ctr": round(float(q_ctr), 4),
                        "nonquestion_avg_ctr": round(float(nq_ctr), 4),
                        "question_query_count": len(q_df),
                        "question_total_clicks": float(q_clicks),
                        "question_total_impressions": float(q_impressions),
                    },
                    "traffic_impact_score": min(1.0, q_clicks / total_clicks),
                })
    except Exception:
        pass

    # ── Position 4-10 queries — near-page-1 opportunities ───────────────────
    try:
        # Lowered impression threshold: 100 (was 200)
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
                        f"These queries have 100+ impressions and are just outside page-1 top-3 — a small ranking push could yield ~{total_gain:.0f} additional clicks. "
                        f"Top opportunity: '{near_p1.iloc[0]['query']}' at position {near_p1.iloc[0]['position']:.1f} with {near_p1.iloc[0]['impressions']:.0f} impressions. "
                        f"Prioritize internal linking, content depth improvements, and E-E-A-T signals for these pages."
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

    # ── Branded vs non-branded query split ───────────────────────────────────
    try:
        # Extract domain name as brand signal (e.g. "kixie" from "https://www.kixie.com/")
        import re
        brand_hints = set()
        brand_match = re.search(r"(?:https?://)?(?:www\.)?([^./]+)", site or "")
        if brand_match:
            brand_hints.add(brand_match.group(1).lower())

        if brand_hints:
            df["is_branded"] = df["query"].str.lower().apply(
                lambda q: any(b in q for b in brand_hints)
            )
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
                        f"Branded queries ({len(branded)} keywords) drive {b_clicks:.0f} clicks at {b_ctr*100:.1f}% CTR. "
                        f"Non-branded queries ({len(nonbranded)} keywords) drive {nb_clicks:.0f} clicks at {nb_ctr*100:.1f}% CTR. "
                        f"{'High branded share — diversifying non-branded content could reduce dependence on brand awareness.' if b_pct > 30 else 'Strong non-branded organic presence — good sign of broad SEO reach.'}"
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
