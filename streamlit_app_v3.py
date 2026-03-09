"""
🤖 SaaS Sales Intelligence Agent v2.0
AI-Powered Self-Serve Analytics for B2B Sales Operations

Built by Payal Gore | GitHub: @PayalGore
v2.0 — Improved accuracy, inline charts, enhanced UI, smarter validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Sales Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Professional, clean, not AI-generic
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #64748b;
        margin-top: -8px;
        margin-bottom: 24px;
        font-weight: 400;
    }
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 8px;
        vertical-align: middle;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .accuracy-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .accuracy-high { background: #d1fae5; color: #065f46; }
    .accuracy-medium { background: #fef3c7; color: #92400e; }
    .accuracy-low { background: #fee2e2; color: #991b1b; }
    .tool-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .stTextInput > div > div > input { font-size: 1.1rem; }
    .insight-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
        border-left: 4px solid #667eea;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INITIALIZE SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "kb_embeddings" not in st.session_state:
    st.session_state.kb_embeddings = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "total_time_saved" not in st.session_state:
    st.session_state.total_time_saved = 0


# ─────────────────────────────────────────────
# DATA GENERATION (same as original, unchanged)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    """Generate synthetic B2B SaaS sales data"""
    np.random.seed(42)

    NUM_DEALS = 800
    NUM_ACCOUNTS = 150
    DATE_START = datetime(2025, 1, 1)

    PRODUCTS = ["IT Management Suite", "Security Platform", "Backup & Recovery",
                "Network Monitoring", "Endpoint Protection"]
    REGIONS = ["Northeast", "Southeast", "West", "Midwest"]
    REPS = {
        "Northeast": ["Sarah Chen", "Mike Rodriguez"],
        "Southeast": ["James Wilson", "Priya Patel"],
        "West": ["Alex Kim", "Jordan Blake"],
        "Midwest": ["Taylor Morgan", "Chris Anderson"]
    }
    STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    STAGE_PROBABILITIES = {"Prospecting": 0.10, "Qualification": 0.25,
                           "Proposal": 0.50, "Negotiation": 0.75,
                           "Closed Won": 1.0, "Closed Lost": 0.0}
    SEGMENTS = {"SMB": (8000, 30000), "Mid-Market": (35000, 120000), "Enterprise": (150000, 500000)}

    COMPANY_PREFIXES = [
        "Apex", "Vertex", "Pinnacle", "Summit", "Nexus", "Catalyst", "Horizon",
        "Quantum", "Velocity", "Vanguard", "Atlas", "Beacon", "Core", "Dynamic",
        "Elevate", "Fusion", "Global", "Harbor", "Insight", "Junction", "Keystone",
        "Ledger", "Matrix", "Noble", "Orbit", "Prime", "Quest", "Ridge", "Spark",
        "Titan", "Unity", "Vivid", "Wave", "Zenith", "Bolt", "Cedar", "Delta",
        "Echo", "Forge", "Grid", "Helix", "Iron", "Jade", "Kite", "Lumen",
        "Metro", "Nova", "Onyx", "Pulse", "Relay", "Scout", "Terra", "Ultra"
    ]
    COMPANY_SUFFIXES = ["Technologies", "Systems", "Solutions", "Digital", "Networks",
                        "Software", "Group", "Services", "Labs", "IT", "Computing",
                        "Dynamics", "Partners", "Global", "Industries"]

    used_names = set()
    def gen_name():
        while True:
            name = f"{np.random.choice(COMPANY_PREFIXES)} {np.random.choice(COMPANY_SUFFIXES)}"
            if name not in used_names:
                used_names.add(name)
                return name

    accounts_data = []
    for i in range(NUM_ACCOUNTS):
        segment = np.random.choice(["SMB", "Mid-Market", "Enterprise"], p=[0.50, 0.35, 0.15])
        region = np.random.choice(REGIONS)
        arr = round(np.random.uniform(*SEGMENTS[segment]), 0)
        base_health = np.random.randint(25, 95)
        if segment == "Enterprise": base_health = min(base_health + 15, 100)
        if region == "West": base_health = max(base_health - 12, 15)
        renewal_month = np.random.choice([1,2,3,7,8,9,10,11,12],
                                          p=[0.15,0.12,0.13,0.12,0.10,0.10,0.08,0.10,0.10])
        rep = np.random.choice(REPS[region])
        support_tickets = np.random.randint(0, 8)
        if base_health < 40: support_tickets = np.random.randint(6, 18)
        nps = np.random.randint(6, 10) if base_health > 70 else (np.random.randint(1, 5) if base_health < 40 else np.random.randint(1, 10))

        accounts_data.append({
            "account_id": f"ACC-{1000+i}", "account_name": gen_name(),
            "segment": segment, "region": region, "arr": arr,
            "health_score": base_health, "nps_score": nps,
            "renewal_date": datetime(2026, renewal_month, np.random.randint(1, 28)).strftime("%Y-%m-%d"),
            "tenure_months": np.random.randint(3, 72),
            "support_tickets_last_90d": support_tickets,
            "assigned_rep": rep, "product": np.random.choice(PRODUCTS),
            "industry": np.random.choice(["Healthcare", "Finance", "Retail", "Manufacturing",
                                           "Education", "Technology", "Legal", "Government"])
        })

    accounts_df = pd.DataFrame(accounts_data)

    deals_data = []
    for i in range(NUM_DEALS):
        account = accounts_data[np.random.randint(0, NUM_ACCOUNTS)]
        deal_type = np.random.choice(["New Logo", "Renewal", "Expansion"], p=[0.35, 0.50, 0.15])
        create_date = DATE_START + timedelta(days=np.random.randint(0, 330))
        seg = account["segment"]
        cycle = np.random.randint(45, 120) if seg == "Enterprise" else (np.random.randint(21, 75) if seg == "Mid-Market" else np.random.randint(7, 45))
        expected_close = create_date + timedelta(days=cycle)

        if expected_close <= datetime.now():
            win_base = 0.42
            if deal_type == "Renewal": win_base += 0.28
            elif deal_type == "Expansion": win_base += 0.15
            if seg == "Enterprise": win_base += 0.08
            elif seg == "SMB": win_base -= 0.08
            if account["region"] == "West": win_base -= 0.12
            if expected_close.month in [7, 8, 9]: win_base -= 0.10
            if account["assigned_rep"] == "Jordan Blake": win_base -= 0.20
            if deal_type == "Renewal" and account["health_score"] < 40: win_base -= 0.25
            win_base = max(0.05, min(0.95, win_base))
            won = np.random.random() < win_base
            stage = "Closed Won" if won else "Closed Lost"
            actual_close = expected_close + timedelta(days=np.random.randint(-3, 10))
            probability = 1.0 if won else 0.0
        else:
            stage = np.random.choice(STAGES, p=[0.15, 0.25, 0.30, 0.30, 0.0, 0.0])
            actual_close = None
            probability = STAGE_PROBABILITIES[stage]
            if stage == "Negotiation" and np.random.random() < 0.3:
                expected_close += timedelta(days=np.random.randint(14, 45))

        loss_reason = None
        if stage == "Closed Lost":
            loss_reason = np.random.choice(["Price too high", "Went with competitor", "Budget cut",
                                             "No decision", "Timeline mismatch", "Product gap"],
                                            p=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10])

        deals_data.append({
            "deal_id": f"DEAL-{5000+i}", "account_id": account["account_id"],
            "account_name": account["account_name"], "deal_type": deal_type,
            "deal_value": deal_value if 'deal_value' in dir() else round(np.random.uniform(*SEGMENTS[seg]), 0),
            "stage": stage, "probability": probability,
            "create_date": create_date.strftime("%Y-%m-%d"),
            "expected_close_date": expected_close.strftime("%Y-%m-%d"),
            "actual_close_date": actual_close.strftime("%Y-%m-%d") if actual_close else None,
            "sales_rep": account["assigned_rep"], "region": account["region"],
            "segment": account["segment"], "product": account["product"],
            "loss_reason": loss_reason
        })

        # Calculate deal_value properly for next iteration
        if deal_type == "Renewal": deal_value = round(account["arr"] * np.random.uniform(0.95, 1.10), 0)
        elif deal_type == "Expansion": deal_value = round(account["arr"] * np.random.uniform(0.15, 0.40), 0)
        else: deal_value = round(np.random.uniform(*SEGMENTS[seg]), 0)
        deals_data[-1]["deal_value"] = deal_value

    deals_df = pd.DataFrame(deals_data)
    deals_df["create_date"] = pd.to_datetime(deals_df["create_date"])
    deals_df["expected_close_date"] = pd.to_datetime(deals_df["expected_close_date"])
    deals_df["actual_close_date"] = pd.to_datetime(deals_df["actual_close_date"])

    # Monthly metrics
    closed = deals_df[deals_df["stage"].isin(["Closed Won", "Closed Lost"])].copy()
    closed["close_month"] = closed["actual_close_date"].dt.to_period("M")
    monthly = closed.groupby("close_month").agg(
        gross_bookings=("deal_value", lambda x: x[closed.loc[x.index, "stage"] == "Closed Won"].sum()),
        deals_won=("stage", lambda x: (x == "Closed Won").sum()),
        deals_lost=("stage", lambda x: (x == "Closed Lost").sum()),
        total_deals_closed=("deal_id", "count"),
        avg_deal_size=("deal_value", "mean"),
        total_pipeline_value=("deal_value", "sum")
    ).reset_index()
    monthly["win_rate"] = round(monthly["deals_won"] / monthly["total_deals_closed"] * 100, 1)
    monthly["close_month"] = monthly["close_month"].astype(str)
    monthly = monthly[monthly["total_deals_closed"] >= 10].reset_index(drop=True)

    # Rep performance
    rep_perf = closed[closed["stage"] == "Closed Won"].groupby("sales_rep").agg(
        total_bookings=("deal_value", "sum"), deals_won=("deal_id", "count"),
        avg_deal_size=("deal_value", "mean")
    ).reset_index()
    rep_losses = closed[closed["stage"] == "Closed Lost"].groupby("sales_rep")["deal_id"].count().reset_index()
    rep_losses.columns = ["sales_rep", "deals_lost"]
    rep_perf = rep_perf.merge(rep_losses, on="sales_rep", how="left")
    rep_perf["deals_lost"] = rep_perf["deals_lost"].fillna(0).astype(int)
    rep_perf["win_rate"] = round(rep_perf["deals_won"] / (rep_perf["deals_won"] + rep_perf["deals_lost"]) * 100, 1)
    quotas = {"Sarah Chen": 1800000, "Mike Rodriguez": 1600000, "James Wilson": 1700000,
              "Priya Patel": 1900000, "Alex Kim": 1700000, "Jordan Blake": 1650000,
              "Taylor Morgan": 1750000, "Chris Anderson": 1600000}
    rep_perf["annual_quota"] = rep_perf["sales_rep"].map(quotas)
    rep_perf["quota_attainment"] = round(rep_perf["total_bookings"] / rep_perf["annual_quota"] * 100, 1)

    # Loss reasons
    loss_reasons = deals_df[deals_df["stage"] == "Closed Lost"]["loss_reason"].value_counts().reset_index()
    loss_reasons.columns = ["reason", "count"]

    # ── NEW: Rep loss reasons (for underperformer + loss reason questions) ──
    rep_loss_reasons = deals_df[deals_df["stage"] == "Closed Lost"].groupby(
        ["sales_rep", "loss_reason"]).size().reset_index(name="count")

    return deals_df, accounts_df, monthly, rep_perf, loss_reasons, rep_loss_reasons


# ─────────────────────────────────────────────
# ANALYSIS TOOLS (upgraded)
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# ANALYSIS TOOLS (v3 with forecasting, scenario simulation, drilldowns)
# ─────────────────────────────────────────────
def get_tools(deals_df, accounts_df, monthly_metrics, rep_performance, loss_reasons, rep_loss_reasons, client):
    from pandasql import sqldf

    def query_data(question):
        latest_month = monthly_metrics["close_month"].iloc[-1]
        previous_month = monthly_metrics["close_month"].iloc[-2]

        def build_prompt(q, err=""):
            err_note = f"\nPrevious attempt failed: {err}. Simplify." if err else ""
            return f"""Convert to SQL query.{err_note}
RULES: SQLite only. No CURRENT_DATE/NOW(). For 'last month' use close_month='{previous_month}'. For 'latest' use close_month='{latest_month}'.

Tables:
deals_df: deal_id, account_id, account_name, deal_type, deal_value, stage, probability, create_date, expected_close_date, actual_close_date, sales_rep, region, segment, product, loss_reason
accounts_df: account_id, account_name, segment, region, arr, health_score, nps_score, renewal_date, tenure_months, support_tickets_last_90d, assigned_rep, product, industry
monthly_metrics: close_month, gross_bookings, deals_won, deals_lost, total_deals_closed, avg_deal_size, total_pipeline_value, win_rate
rep_performance: sales_rep, total_bookings, deals_won, avg_deal_size, deals_lost, win_rate, annual_quota, quota_attainment
loss_reasons: reason, count
rep_loss_reasons: sales_rep, loss_reason, count

IMPORTANT:
- underperformers => rep_performance sorted by win_rate ASC
- loss reasons by rep => rep_loss_reasons
- overall loss reasons => loss_reasons
- closed lost deals have loss_reason populated

Question: {q}
Return ONLY SQL. No markdown."""

        tables = {
            "deals_df": deals_df,
            "accounts_df": accounts_df,
            "monthly_metrics": monthly_metrics,
            "rep_performance": rep_performance,
            "loss_reasons": loss_reasons,
            "rep_loss_reasons": rep_loss_reasons,
        }
        last_error = ""
        for _ in range(3):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "SQL expert. SQLite only. Use exact column names provided."},
                    {"role": "user", "content": build_prompt(question, last_error)},
                ],
                temperature=0,
            )
            sql = resp.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
            try:
                result = sqldf(sql, tables)
                return {
                    "success": True,
                    "data": result.to_string(index=False),
                    "row_count": len(result),
                    "sql_used": sql,
                    "columns": list(result.columns),
                    "raw_values": result.to_dict(orient="records"),
                }
            except Exception as e:
                last_error = str(e)
        return {"success": False, "error": last_error}

    def analyze_trends(metric, period="monthly"):
        if metric not in monthly_metrics.columns:
            return {"error": f"Metric '{metric}' not found."}
        values = monthly_metrics[metric].values.astype(float)
        changes = [round((values[i] - values[i - 1]) / values[i - 1] * 100, 1) if values[i - 1] != 0 else 0 for i in range(1, len(values))]
        mean_val, std_val = float(np.mean(values)), float(np.std(values))
        anomalies = [
            {
                "month": monthly_metrics["close_month"].iloc[i],
                "value": round(float(v), 2),
                "z_score": round(float((v - mean_val) / std_val), 2),
                "severity": "high" if abs((v - mean_val) / std_val) > 2 else "medium",
            }
            for i, v in enumerate(values)
            if std_val > 0 and abs((v - mean_val) / std_val) > 1.5
        ]
        return {
            "metric": metric,
            "current_value": round(float(values[-1]), 2),
            "previous_value": round(float(values[-2]), 2) if len(values) > 1 else None,
            "mom_change_pct": changes[-1] if changes else None,
            "average": round(mean_val, 2),
            "trend": "improving" if changes and changes[-1] > 0 else "declining",
            "anomalies": anomalies,
            "all_values": {monthly_metrics["close_month"].iloc[i]: round(float(v), 2) for i, v in enumerate(values)},
        }

    def find_at_risk(category="deals"):
        if category == "deals":
            open_d = deals_df[~deals_df["stage"].isin(["Closed Won", "Closed Lost"])].copy()
            ref = deals_df["expected_close_date"].max() - timedelta(days=30)
            open_d["days_past"] = (ref - open_d["expected_close_date"]).dt.days
            slip = open_d[open_d["days_past"] > 0].sort_values("deal_value", ascending=False)
            return {
                "type": "slipping_deals",
                "count": len(slip),
                "total_value_at_risk": round(float(slip["deal_value"].sum()), 2),
                "columns": ["deal_id", "account_name", "deal_value", "stage", "expected_close_date", "days_past", "sales_rep"],
                "raw_values": slip[["deal_id", "account_name", "deal_value", "stage", "expected_close_date", "days_past", "sales_rep"]].head(15).to_dict(orient="records"),
                "top_deals": slip[["deal_id", "account_name", "deal_value", "stage", "expected_close_date", "days_past", "sales_rep"]].head(10).to_string(index=False),
            }
        accounts_temp = accounts_df.copy()
        accounts_temp["renewal_date_dt"] = pd.to_datetime(accounts_temp["renewal_date"])
        ref = pd.Timestamp("2026-03-01")
        at_risk = accounts_temp[(accounts_temp["health_score"] < 50) & (accounts_temp["renewal_date_dt"] < ref + timedelta(days=90))].sort_values("arr", ascending=False)
        return {
            "type": "at_risk_accounts",
            "count": len(at_risk),
            "total_arr_at_risk": round(float(at_risk["arr"].sum()), 2),
            "columns": ["account_name", "arr", "health_score", "renewal_date", "segment", "assigned_rep"],
            "raw_values": at_risk[["account_name", "arr", "health_score", "renewal_date", "segment", "assigned_rep"]].head(15).to_dict(orient="records"),
            "top_accounts": at_risk[["account_name", "arr", "health_score", "renewal_date", "segment", "assigned_rep"]].head(10).to_string(index=False),
        }

    def compare_periods(period1, period2):
        p1 = monthly_metrics[monthly_metrics["close_month"] == period1]
        p2 = monthly_metrics[monthly_metrics["close_month"] == period2]
        if p1.empty or p2.empty:
            return {"error": f"Period not found. Available: {monthly_metrics['close_month'].tolist()}"}
        comp = {}
        for col in ["gross_bookings", "deals_won", "deals_lost", "win_rate", "avg_deal_size"]:
            v1, v2 = float(p1[col].iloc[0]), float(p2[col].iloc[0])
            comp[col] = {period1: round(v1, 2), period2: round(v2, 2), "change_pct": round(((v2 - v1) / v1 * 100) if v1 != 0 else 0, 1)}
        return {"comparison": comp}

    def generate_weekly_report():
        start = time.time()
        return {
            "bookings_trend": analyze_trends("gross_bookings"),
            "winrate_trend": analyze_trends("win_rate"),
            "at_risk_deals": find_at_risk("deals"),
            "at_risk_accounts": find_at_risk("accounts"),
            "rep_performance": rep_performance[["sales_rep", "total_bookings", "deals_won", "win_rate", "annual_quota", "quota_attainment"]].sort_values("win_rate").to_dict(orient="records"),
            "latest_month": {k: str(v) for k, v in monthly_metrics.iloc[-1].to_dict().items()},
            "time": round(time.time() - start, 1),
        }

    def forecast_metric(metric="gross_bookings", periods=3):
        if metric not in monthly_metrics.columns:
            return {"error": f"Metric '{metric}' not found."}
        hist = monthly_metrics[["close_month", metric]].copy()
        y = hist[metric].astype(float).values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        future = []
        last_period = pd.Period(hist["close_month"].iloc[-1], freq="M")
        for i in range(1, periods + 1):
            pred = max(0, slope * (len(y) + i - 1) + intercept)
            future.append({"close_month": str(last_period + i), metric: round(float(pred), 2), "type": "forecast"})
        hist_records = [{"close_month": row["close_month"], metric: round(float(row[metric]), 2), "type": "historical"} for _, row in hist.iterrows()]
        return {
            "metric": metric,
            "method": "linear_trend",
            "historical_values": hist_records,
            "forecast_values": future,
            "trend_slope": round(float(slope), 2),
        }

    def analyze_segment_performance(dimension="region", metric="win_rate", top_n=10):
        allowed_dims = ["region", "segment", "product", "sales_rep", "deal_type"]
        if dimension not in allowed_dims:
            return {"error": f"Dimension must be one of {allowed_dims}"}
        closed = deals_df[deals_df["stage"].isin(["Closed Won", "Closed Lost"])].copy()
        grouped = closed.groupby(dimension).agg(
            deals_won=("stage", lambda x: int((x == "Closed Won").sum())),
            deals_lost=("stage", lambda x: int((x == "Closed Lost").sum())),
            total_bookings=("deal_value", lambda x: float(x[closed.loc[x.index, "stage"] == "Closed Won"].sum())),
            avg_deal_size=("deal_value", "mean"),
            total_deals=("deal_id", "count"),
        ).reset_index()
        grouped["win_rate"] = np.where(grouped["total_deals"] > 0, round(grouped["deals_won"] / grouped["total_deals"] * 100, 1), 0)
        grouped["avg_deal_size"] = grouped["avg_deal_size"].round(1)
        sort_col = metric if metric in grouped.columns else "win_rate"
        grouped = grouped.sort_values(sort_col, ascending=False).head(top_n)
        return {
            "dimension": dimension,
            "metric": sort_col,
            "columns": list(grouped.columns),
            "raw_values": grouped.to_dict(orient="records"),
            "summary": grouped.to_string(index=False),
        }

    def simulate_scenario(scenario_type="win_rate_lift", magnitude=5.0, scope="overall"):
        latest = monthly_metrics.iloc[-1]
        result = {"scenario_type": scenario_type, "magnitude": magnitude, "scope": scope}
        if scenario_type == "win_rate_lift":
            current_wr = float(latest["win_rate"])
            new_wr = min(100.0, current_wr + float(magnitude))
            current_bookings = float(latest["gross_bookings"])
            uplift_factor = (new_wr / max(current_wr, 1e-6)) - 1
            incremental_bookings = max(0.0, current_bookings * uplift_factor)
            result.update({
                "current_win_rate": round(current_wr, 1),
                "projected_win_rate": round(new_wr, 1),
                "current_bookings": round(current_bookings, 2),
                "projected_incremental_bookings": round(incremental_bookings, 2),
            })
        elif scenario_type == "churn_reduction":
            at_risk = accounts_df[accounts_df["health_score"] < 50]
            arr_at_risk = float(at_risk["arr"].sum())
            saved_arr = arr_at_risk * (float(magnitude) / 100.0)
            result.update({
                "current_arr_at_risk": round(arr_at_risk, 2),
                "projected_arr_saved": round(saved_arr, 2),
            })
        else:
            open_deals = deals_df[~deals_df["stage"].isin(["Closed Won", "Closed Lost"])]
            pipeline = float(open_deals["deal_value"].sum())
            accelerated = pipeline * (float(magnitude) / 100.0)
            result.update({"open_pipeline": round(pipeline, 2), "projected_accelerated_pipeline": round(accelerated, 2)})
        return result

    return (
        query_data,
        analyze_trends,
        find_at_risk,
        compare_periods,
        generate_weekly_report,
        forecast_metric,
        analyze_segment_performance,
        simulate_scenario,
    )


# ─────────────────────────────────────────────
# RAG LAYER
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "Enterprise segment: accounts with ARR above $150,000. Highest-value clients, dedicated account management.",
    "Mid-Market segment: ARR between $35,000 and $150,000. Primary growth engine.",
    "SMB segment: ARR below $35,000. High volume, higher churn risk.",
    "Healthy win rate benchmark is above 40%. Below 35% requires immediate pipeline review.",
    "Renewal deals should close at 70%+ win rate. Below 60% indicates retention problem.",
    "New logo win rate target is 30-35%. Expansion deals should close at 50%+.",
    "Pipeline coverage ratio should be 3x quarterly target. Below 2.5x is a red flag.",
    "Deals slipping more than 14 days past expected close require manager review.",
    "Q3 historically sees a 15-20% dip in bookings due to budget cycles.",
    "Accounts with health score below 40 require immediate customer success outreach.",
    "Support tickets above 10 in 90 days correlate with higher churn probability.",
    "West region has been underperforming due to recent team transitions.",
    "Jordan Blake has the lowest win rate and needs coaching support.",
    "Rep quota attainment below 80% for two quarters triggers performance review.",
    "Top loss reason Price too high above 25% suggests pricing strategy review.",
    "Annual booking target for the sales team is approximately $14M.",
    "Weekly sales report is due every Monday for leadership standup.",
    "Underperforming reps are identified by win rate below 35%, not quota attainment alone.",
    "NPS below 5 signals high renewal risk requiring proactive outreach.",
    "Late stage negotiation deals slipping are highest priority because committed revenue is at risk.",
]

def init_rag(client):
    resp = client.embeddings.create(model="text-embedding-3-small", input=KNOWLEDGE_BASE)
    return [item.embedding for item in resp.data]


def retrieve_context(query, client, kb_embeddings, top_k=3):
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    q_arr = np.array(q_emb)
    scored = []
    for i, emb in enumerate(kb_embeddings):
        e_arr = np.array(emb)
        sim = float(np.dot(q_arr, e_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(e_arr)))
        scored.append((sim, KNOWLEDGE_BASE[i]))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]


# ─────────────────────────────────────────────
# HALLUCINATION GUARD (v3)
# ─────────────────────────────────────────────
RAG_BENCHMARKS = {14, 15, 20, 21, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 110, 120, 150}

def validate_response(text, tool_results):
    numbers = re.findall(r'\$?([\d,]+\.?\d*)\s*(%|M|K)?', text)
    source_nums = set()
    results_str = json.dumps(tool_results, default=str)
    for n in re.findall(r'([\d,]+\.?\d+)', results_str):
        try:
            source_nums.add(float(n.replace(',', '')))
        except:
            pass
    for val in tool_results.values():
        if isinstance(val, dict) and "raw_values" in val:
            for row in val["raw_values"]:
                for v in row.values():
                    try:
                        source_nums.add(float(v))
                    except:
                        pass
        if isinstance(val, dict) and "historical_values" in val:
            for row in val["historical_values"] + val.get("forecast_values", []):
                for k, v in row.items():
                    if k != "close_month" and k != "type":
                        try:
                            source_nums.add(float(v))
                        except:
                            pass
    list_numbers = set(float(m.group(1)) for m in re.finditer(r'(?:^|\n)\s*(\d+)\.\s', text))
    verified, total, unverified = 0, 0, []
    for num_str, unit in numbers:
        try:
            num = float(num_str.replace(',', ''))
            if num in list_numbers and num < 20:
                continue
            if unit != "%" and num < 5:
                continue
            if unit == "%" and num in RAG_BENCHMARKS and not any(abs(num - s) / max(s, 1) < 0.01 for s in source_nums if s > 0):
                continue
            if unit == "K":
                num *= 1_000
            elif unit == "M":
                num *= 1_000_000
            total += 1
            if any(abs(num - s) / max(s, 1) < 0.05 for s in source_nums if s > 0):
                verified += 1
            else:
                unverified.append(f"{num_str}{unit}")
        except:
            pass
    accuracy = round((verified / max(total, 1)) * 100, 1)
    return accuracy, verified, total, unverified


# ─────────────────────────────────────────────
# DYNAMIC AI CHART GENERATION
# ─────────────────────────────────────────────
def result_to_df(result):
    if isinstance(result, dict):
        if "raw_values" in result and result["raw_values"]:
            return pd.DataFrame(result["raw_values"])
        if "all_values" in result and result["all_values"]:
            return pd.DataFrame({"x": list(result["all_values"].keys()), "y": list(result["all_values"].values())})
        if "historical_values" in result and result.get("forecast_values") is not None:
            hist = pd.DataFrame(result["historical_values"])
            fc = pd.DataFrame(result.get("forecast_values", []))
            return pd.concat([hist, fc], ignore_index=True)
    return None


def build_candidate_datasets(tool_results, monthly_metrics, rep_performance, loss_reasons, rep_loss_reasons):
    datasets = {
        "monthly_metrics": monthly_metrics.copy(),
        "rep_performance": rep_performance.copy(),
        "loss_reasons": loss_reasons.copy(),
        "rep_loss_reasons": rep_loss_reasons.copy(),
    }
    for key, result in tool_results.items():
        df = result_to_df(result)
        if df is not None and not df.empty:
            datasets[key] = df
    return datasets


def ai_plan_chart(question, datasets, client):
    dataset_summary = []
    for name, df in datasets.items():
        dataset_summary.append({
            "name": name,
            "columns": list(df.columns),
            "rows": int(len(df)),
            "sample": df.head(3).to_dict(orient="records"),
        })
    prompt = {
        "question": question,
        "datasets": dataset_summary,
        "instructions": "Return only JSON with keys: dataset, chart_type, x, y, color, title, orientation, top_n, use_forecast_split. chart_type must be one of bar,line,scatter,pie. Choose the most relevant dataset for the user's question. If no chart makes sense, return dataset as null.",
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data visualization planner. Return valid JSON only."},
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0,
        )
        spec = json.loads(resp.choices[0].message.content)
        return spec
    except Exception:
        q = question.lower()
        if "forecast" in q:
            return {"dataset": next((k for k in datasets if 'forecast_metric' in k), 'monthly_metrics'), "chart_type": "line", "x": "close_month", "y": "gross_bookings", "color": "type", "title": "Forecast", "orientation": "v", "top_n": 12, "use_forecast_split": True}
        if any(w in q for w in ["rep", "underperform", "performance", "win rate"]):
            return {"dataset": "rep_performance", "chart_type": "bar", "x": "win_rate", "y": "sales_rep", "color": None, "title": "Rep Win Rate", "orientation": "h", "top_n": 10, "use_forecast_split": False}
        if any(w in q for w in ["loss", "reason", "lost"]):
            return {"dataset": "loss_reasons", "chart_type": "bar", "x": "count", "y": "reason", "color": None, "title": "Loss Reasons", "orientation": "h", "top_n": 10, "use_forecast_split": False}
        return {"dataset": "monthly_metrics", "chart_type": "line", "x": "close_month", "y": "gross_bookings", "color": None, "title": "Monthly Trend", "orientation": "v", "top_n": 12, "use_forecast_split": False}


def render_ai_chart(question, tool_results, monthly_metrics, rep_performance, loss_reasons, rep_loss_reasons, client):
    datasets = build_candidate_datasets(tool_results, monthly_metrics, rep_performance, loss_reasons, rep_loss_reasons)
    spec = ai_plan_chart(question, datasets, client)
    dataset_name = spec.get("dataset")
    if not dataset_name or dataset_name not in datasets:
        return None, None
    df = datasets[dataset_name].copy()
    if df.empty:
        return None, spec
    top_n = spec.get("top_n")
    if top_n and isinstance(top_n, int) and len(df) > top_n and spec.get("chart_type") != "line":
        df = df.head(top_n)
    chart_type = spec.get("chart_type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    title = spec.get("title", f"Chart for: {question}")
    orientation = spec.get("orientation", "v")
    if x not in df.columns and len(df.columns) >= 1:
        x = df.columns[0]
    if y not in df.columns and len(df.columns) >= 2:
        y = df.columns[1]
    if chart_type == "line":
        fig = px.line(df, x=x, y=y, color=color if color in df.columns else None, markers=True, title=title)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, title=title)
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=title)
    else:
        fig = px.bar(df, x=x, y=y, color=color if color in df.columns else None, orientation=orientation, title=title)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig, spec


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────
TOOL_DEFS = [
    {"type": "function", "function": {"name": "query_data", "description": "Query sales data via natural language SQL for lookups, rep stats, loss reasons, underperformers.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "analyze_trends", "description": "Analyze trends and detect anomalies for monthly KPIs.", "parameters": {"type": "object", "properties": {"metric": {"type": "string", "description": "gross_bookings, win_rate, deals_won, deals_lost, avg_deal_size, total_pipeline_value"}, "period": {"type": "string", "default": "monthly"}}, "required": ["metric"]}}},
    {"type": "function", "function": {"name": "find_at_risk", "description": "Find slipping deals or at risk accounts near renewal.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "enum": ["deals", "accounts"]}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "compare_periods", "description": "Compare two months side by side.", "parameters": {"type": "object", "properties": {"period1": {"type": "string"}, "period2": {"type": "string"}}, "required": ["period1", "period2"]}}},
    {"type": "function", "function": {"name": "generate_weekly_report", "description": "Generate complete sales operations report.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "forecast_metric", "description": "Forecast a monthly KPI for the next few periods.", "parameters": {"type": "object", "properties": {"metric": {"type": "string"}, "periods": {"type": "integer", "default": 3}}, "required": ["metric"]}}},
    {"type": "function", "function": {"name": "analyze_segment_performance", "description": "Break down performance by region, segment, product, rep, or deal type.", "parameters": {"type": "object", "properties": {"dimension": {"type": "string", "enum": ["region", "segment", "product", "sales_rep", "deal_type"]}, "metric": {"type": "string"}, "top_n": {"type": "integer", "default": 10}}, "required": ["dimension"]}}},
    {"type": "function", "function": {"name": "simulate_scenario", "description": "Simulate the impact of a win rate lift, churn reduction, or pipeline acceleration scenario.", "parameters": {"type": "object", "properties": {"scenario_type": {"type": "string", "enum": ["win_rate_lift", "churn_reduction", "pipeline_acceleration"]}, "magnitude": {"type": "number", "default": 5}, "scope": {"type": "string", "default": "overall"}}, "required": ["scenario_type"]}}},
]


def run_agent(question, client, tools_funcs, kb_embeddings):
    (
        query_data,
        analyze_trends,
        find_at_risk,
        compare_periods,
        generate_weekly_report,
        forecast_metric,
        analyze_segment_performance,
        simulate_scenario,
    ) = tools_funcs

    tool_map = {
        "query_data": lambda a: query_data(a.get("question", "")),
        "analyze_trends": lambda a: analyze_trends(a.get("metric", ""), a.get("period", "monthly")),
        "find_at_risk": lambda a: find_at_risk(a.get("category", "deals")),
        "compare_periods": lambda a: compare_periods(a.get("period1", ""), a.get("period2", "")),
        "generate_weekly_report": lambda a: generate_weekly_report(),
        "forecast_metric": lambda a: forecast_metric(a.get("metric", "gross_bookings"), int(a.get("periods", 3))),
        "analyze_segment_performance": lambda a: analyze_segment_performance(a.get("dimension", "region"), a.get("metric", "win_rate"), int(a.get("top_n", 10))),
        "simulate_scenario": lambda a: simulate_scenario(a.get("scenario_type", "win_rate_lift"), float(a.get("magnitude", 5)), a.get("scope", "overall")),
    }

    start = time.time()
    messages = [
        {"role": "system", "content": """You are a senior SaaS sales ops analyst with real data access.
RULES:
- Always use tools to get data. Never make up numbers.
- Use exact figures from tool results.
- Be concise, executive-friendly, and under 260 words.
- For forecasts, use forecast_metric.
- For region or segment breakouts, use analyze_segment_performance.
- For what-if or impact questions, use simulate_scenario.
- For underperformers plus loss reasons, use multiple tools when needed.
- End with 1-2 specific actions."""},
        {"role": "user", "content": question},
    ]

    all_results = {}
    tools_used = []
    tool_call_counter = 0

    for _ in range(6):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOL_DEFS, tool_choice="auto", temperature=0)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_used.append(fn)
            result = tool_map[fn](args)
            tool_call_counter += 1
            result_key = f"{fn}_{tool_call_counter}"
            all_results[result_key] = result
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})

    rag_rules = retrieve_context(question, client, kb_embeddings)
    rag_context = "\n".join([f"• {r}" for r in rag_rules])
    messages.append({"role": "user", "content": f"Final answer using ONLY the tool results above. Business context:\n{rag_context}\nUse exact numbers from the data. Under 260 words. End with 1-2 specific actions."})
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    answer = final.choices[0].message.content
    accuracy, verified, total, unverified = validate_response(answer, all_results)
    elapsed = round(time.time() - start, 1)
    return answer, accuracy, verified, total, list(dict.fromkeys(tools_used)), elapsed, all_results, rag_rules, unverified

def plot_bookings_trend(monthly):
    fig = px.bar(monthly, x="close_month", y="gross_bookings",
                 title="Monthly Gross Bookings", color_discrete_sequence=["#667eea"])
    fig.update_layout(xaxis_title="Month", yaxis_title="Bookings ($)", template="plotly_white", height=350)
    return fig

def plot_win_rate_trend(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["close_month"], y=monthly["win_rate"],
                             mode="lines+markers", line=dict(color="#667eea", width=3),
                             marker=dict(size=8), name="Win Rate"))
    fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Benchmark (40%)")
    fig.update_layout(title="Win Rate Trend", xaxis_title="Month", yaxis_title="Win Rate (%)",
                      template="plotly_white", height=350)
    return fig

def plot_rep_performance(rep_perf):
    fig = px.bar(rep_perf.sort_values("win_rate"), x="win_rate", y="sales_rep",
                 orientation="h", title="Rep Win Rate",
                 color="win_rate", color_continuous_scale=["#ef4444","#f59e0b","#22c55e"])
    fig.add_vline(x=40, line_dash="dash", line_color="red")
    fig.update_layout(template="plotly_white", height=350, yaxis_title="", xaxis_title="Win Rate (%)")
    return fig

def plot_loss_reasons(loss_reasons):
    lr = loss_reasons.sort_values("count", ascending=True)
    fig = go.Figure(go.Bar(x=lr["count"], y=lr["reason"], orientation="h",
                           marker_color=["#ef4444","#f59e0b","#667eea","#8b5cf6","#06b6d4","#22c55e"][:len(lr)],
                           text=lr["count"], textposition="outside"))
    fig.update_layout(title="Deal Loss Reasons", template="plotly_white", height=350,
                      xaxis_title="Deals Lost", yaxis_title="")
    return fig




# ─────────────────────────────────────────────
# MAIN APP (v3)
# ─────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Sales Intelligence</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-subtitle">AI-Powered Analytics Agent v3.0</p>', unsafe_allow_html=True)
        st.markdown('---')

        api_key = st.secrets.get("OPENAI_API_KEY", "") or st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...")

        if api_key:
            st.success("✓ API Key loaded")
        else:
            st.warning("Enter your OpenAI API key to start")

        st.markdown('---')
        if st.session_state.query_count > 0:
            st.markdown("### 📊 Session Stats")
            st.metric("Queries Asked", st.session_state.query_count)
            st.metric("Est. Manual Time", f"{st.session_state.query_count * 18} min")
            st.metric("AI Time", f"{st.session_state.total_time_saved:.0f} sec")
            st.markdown('---')

        st.markdown("### 💡 Try asking:")
        example_qs = [
            "How did we do last month?",
            "Which rep has the lowest win rate?",
            "Forecast gross bookings for the next 3 months",
            "Break down win rate by region",
            "What if win rate improves by 5 points next month?",
            "Which accounts are at risk?",
            "Who is underperforming and what are the top loss reasons?",
            "Generate my weekly report",
        ]
        for eq in example_qs:
            if st.button(eq, key=f"sb_{eq}", use_container_width=True):
                st.session_state.selected_question = eq

        st.markdown('---')
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        **Function Calling** → AI picks tools  
        **Text-to-SQL** → NL → SQL → data  
        **RAG** → Business context  
        **AI Chart Planner** → question-driven graph  
        **Simulation + Forecasting** → deeper analysis  
        **Validation** → number-level trust layer
        """)
        st.markdown('---')
        st.markdown("Built by **Payal Gore**")
        st.markdown("[GitHub](https://github.com/PayalGore) · [LinkedIn](https://linkedin.com/in/payalgore)")

    st.markdown("""
    <p class="main-header">
        SaaS Sales Intelligence Agent <span class="version-badge">v3.0</span>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your sales data in plain English. AI analyzes, charts, validates, forecasts, and simulates in seconds.</p>', unsafe_allow_html=True)

    deals_df, accounts_df, monthly, rep_perf, loss_reasons, rep_loss_reasons = generate_data()

    col1, col2, col3, col4 = st.columns(4)
    latest = monthly.iloc[-1]
    prev = monthly.iloc[-2]
    with col1:
        st.metric("Gross Bookings", f"${latest['gross_bookings']:,.0f}", f"{((latest['gross_bookings'] - prev['gross_bookings']) / prev['gross_bookings'] * 100):.1f}%")
    with col2:
        st.metric("Win Rate", f"{latest['win_rate']}%", f"{latest['win_rate'] - prev['win_rate']:.1f}pp")
    with col3:
        st.metric("Deals Won", int(latest['deals_won']), f"{int(latest['deals_won'] - prev['deals_won'])}")
    with col4:
        st.metric("At-Risk Accounts", int((accounts_df['health_score'] < 50).sum()), "health < 50")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bookings", "📈 Win Rate", "👥 Reps", "❌ Losses", "📋 Data"])
    with tab1:
        st.plotly_chart(plot_bookings_trend(monthly), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_win_rate_trend(monthly), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_rep_performance(rep_perf), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_loss_reasons(loss_reasons), use_container_width=True)
    with tab5:
        data_tab = st.selectbox("Select table", ["Pipeline Deals", "Accounts", "Monthly Metrics", "Rep Performance", "Loss Reasons by Rep"])
        if data_tab == "Pipeline Deals":
            st.dataframe(deals_df, use_container_width=True)
        elif data_tab == "Accounts":
            st.dataframe(accounts_df, use_container_width=True)
        elif data_tab == "Monthly Metrics":
            st.dataframe(monthly, use_container_width=True)
        elif data_tab == "Loss Reasons by Rep":
            st.dataframe(rep_loss_reasons, use_container_width=True)
        else:
            st.dataframe(rep_perf, use_container_width=True)

    st.markdown('---')
    st.markdown("### 💬 Ask Your Sales Intelligence Agent")

    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("📊 Last month summary", use_container_width=True):
            st.session_state.selected_question = "How did we do last month?"
        if st.button("👥 Underperformers", use_container_width=True):
            st.session_state.selected_question = "Who is underperforming on the sales team and what are the top reasons we're losing deals?"
    with qcol2:
        if st.button("🔮 Forecast next quarter", use_container_width=True):
            st.session_state.selected_question = "Forecast gross bookings for the next 3 months and show the trend"
        if st.button("🏢 Region breakdown", use_container_width=True):
            st.session_state.selected_question = "Break down win rate by region and show me the weakest region"
    with qcol3:
        if st.button("🧪 Scenario simulation", use_container_width=True):
            st.session_state.selected_question = "What if win rate improves by 5 points next month?"
        if st.button("📋 Weekly report", use_container_width=True):
            st.session_state.selected_question = "Generate my weekly report"

    typed_question = st.text_input("Or type your own question", placeholder="Example: Show me bookings by segment and forecast the next 3 months")
    if st.button("Ask", use_container_width=True, key="ask_button") and typed_question.strip():
        st.session_state.selected_question = typed_question.strip()

    if not api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to start chatting with the agent.")
        return

    client = OpenAI(api_key=api_key)
    if st.session_state.kb_embeddings is None:
        with st.spinner("🔄 Initializing RAG knowledge base..."):
            st.session_state.kb_embeddings = init_rag(client)

    tools_funcs = get_tools(deals_df, accounts_df, monthly, rep_perf, loss_reasons, rep_loss_reasons, client)

    if "selected_question" in st.session_state:
        question = st.session_state.selected_question
        del st.session_state.selected_question
    else:
        question = st.chat_input("Ask anything about your sales data...")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.query_count += 1
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                answer, accuracy, verified, total, tools_used, elapsed, results, rag_rules, unverified = run_agent(question, client, tools_funcs, st.session_state.kb_embeddings)
            st.session_state.total_time_saved += elapsed
            st.markdown(answer)

            chart, chart_spec = render_ai_chart(question, results, monthly, rep_perf, loss_reasons, rep_loss_reasons, client)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)

            acc_class = "accuracy-high" if accuracy >= 70 else ("accuracy-medium" if accuracy >= 50 else "accuracy-low")
            tools_html = " ".join([f'<span class="tool-badge">{t}</span>' for t in tools_used])
            manual_est = len(tools_used) * 18
            st.markdown(f"""
            <div style="margin-top: 12px; padding: 10px 14px; background: #f8f9fa; border-radius: 8px; font-size: 0.85rem; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <span class="accuracy-badge {acc_class}">🔒 {accuracy}% accuracy ({verified}/{total})</span>
                <span>⏱️ {elapsed}s</span>
                <span style="color: #059669; font-weight: 600;">💰 ~{manual_est} min saved</span>
                {tools_html}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 AI Reasoning"):
                st.markdown("**Top RAG rules used:**")
                for rule in rag_rules:
                    st.write(f"• {rule}")
                if chart_spec:
                    st.markdown("**AI chart spec:**")
                    st.json(chart_spec)
                if unverified:
                    st.markdown("**Unverified numbers:**")
                    st.write(unverified)

        full_response = f"{answer}\n\n*🔒 {accuracy}% accuracy ({verified}/{total}) | ⏱️ {elapsed}s | Tools: {', '.join(tools_used)}*"
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
