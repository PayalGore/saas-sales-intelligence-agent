"""
🤖 SaaS Sales Intelligence Agent
AI-Powered Self-Serve Analytics for B2B Sales Operations

Built by Payal Gore | GitHub: @PayalGore
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
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
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
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
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


# ─────────────────────────────────────────────
# DATA GENERATION
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

    # Accounts
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

    # Deals
    deals_data = []
    for i in range(NUM_DEALS):
        account = accounts_data[np.random.randint(0, NUM_ACCOUNTS)]
        deal_type = np.random.choice(["New Logo", "Renewal", "Expansion"], p=[0.35, 0.50, 0.15])
        create_date = DATE_START + timedelta(days=np.random.randint(0, 330))
        seg = account["segment"]
        cycle = np.random.randint(45, 120) if seg == "Enterprise" else (np.random.randint(21, 75) if seg == "Mid-Market" else np.random.randint(7, 45))
        expected_close = create_date + timedelta(days=cycle)

        if deal_type == "Renewal": deal_value = round(account["arr"] * np.random.uniform(0.95, 1.10), 0)
        elif deal_type == "Expansion": deal_value = round(account["arr"] * np.random.uniform(0.15, 0.40), 0)
        else: deal_value = round(np.random.uniform(*SEGMENTS[seg]), 0)

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
            "deal_value": deal_value, "stage": stage, "probability": probability,
            "create_date": create_date.strftime("%Y-%m-%d"),
            "expected_close_date": expected_close.strftime("%Y-%m-%d"),
            "actual_close_date": actual_close.strftime("%Y-%m-%d") if actual_close else None,
            "sales_rep": account["assigned_rep"], "region": account["region"],
            "segment": account["segment"], "product": account["product"],
            "loss_reason": loss_reason
        })

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

    return deals_df, accounts_df, monthly, rep_perf, loss_reasons


# ─────────────────────────────────────────────
# ANALYSIS TOOLS
# ─────────────────────────────────────────────
def get_tools(deals_df, accounts_df, monthly_metrics, rep_performance, loss_reasons, client):
    from pandasql import sqldf

    def query_data(question):
        latest_month = monthly_metrics["close_month"].iloc[-1]
        previous_month = monthly_metrics["close_month"].iloc[-2]

        def build_prompt(q, err=""):
            err_note = f"\nPrevious attempt failed: {err}. Simplify." if err else ""
            return f"""Convert to SQL query.{err_note}
RULES: SQLite only. No CURRENT_DATE/NOW(). For "last month" use close_month='{previous_month}'. For "latest" use close_month='{latest_month}'.

Tables:
deals_df: deal_id, account_id, account_name, deal_type, deal_value, stage, probability, create_date, expected_close_date, actual_close_date, sales_rep, region, segment, product, loss_reason
accounts_df: account_id, account_name, segment, region, arr, health_score, nps_score, renewal_date, tenure_months, support_tickets_last_90d, assigned_rep, product, industry
monthly_metrics: close_month, gross_bookings, deals_won, deals_lost, total_deals_closed, avg_deal_size, total_pipeline_value, win_rate
rep_performance: sales_rep, total_bookings, deals_won, avg_deal_size, deals_lost, win_rate, annual_quota, quota_attainment
loss_reasons: reason, count

Question: {q}
Return ONLY SQL. No markdown."""

        tables = {"deals_df": deals_df, "accounts_df": accounts_df, "monthly_metrics": monthly_metrics,
                  "rep_performance": rep_performance, "loss_reasons": loss_reasons}
        last_error = ""
        for attempt in range(3):
            resp = client.chat.completions.create(model="gpt-4o-mini",
                messages=[{"role": "system", "content": "SQL expert. SQLite only. No date functions."},
                          {"role": "user", "content": build_prompt(question, last_error)}], temperature=0)
            sql = resp.choices[0].message.content.strip().replace("```sql","").replace("```","").strip()
            try:
                result = sqldf(sql, tables)
                return {"success": True, "data": result.to_string(), "row_count": len(result), "sql_used": sql}
            except Exception as e:
                last_error = str(e)
        return {"success": False, "error": last_error}

    def analyze_trends(metric, period="monthly"):
        if metric not in monthly_metrics.columns:
            return {"error": f"Metric '{metric}' not found."}
        values = monthly_metrics[metric].values.astype(float)
        changes = [round((values[i]-values[i-1])/values[i-1]*100, 1) if values[i-1]!=0 else 0 for i in range(1, len(values))]
        mean_val, std_val = float(np.mean(values)), float(np.std(values))
        anomalies = [{"month": monthly_metrics["close_month"].iloc[i], "value": round(float(v),2),
                      "z_score": round(float((v-mean_val)/std_val),2),
                      "severity": "high" if abs((v-mean_val)/std_val) > 2 else "medium"}
                     for i, v in enumerate(values) if std_val > 0 and abs((v-mean_val)/std_val) > 1.5]
        return {"metric": metric, "current_value": round(float(values[-1]),2),
                "previous_value": round(float(values[-2]),2) if len(values)>1 else None,
                "mom_change_pct": changes[-1] if changes else None,
                "average": round(mean_val,2), "trend": "improving" if changes and changes[-1]>0 else "declining",
                "anomalies": anomalies,
                "all_values": {monthly_metrics["close_month"].iloc[i]: round(float(v),2) for i,v in enumerate(values)}}

    def find_at_risk(category="deals"):
        if category == "deals":
            open_d = deals_df[~deals_df["stage"].isin(["Closed Won","Closed Lost"])].copy()
            ref = deals_df["expected_close_date"].max() - timedelta(days=30)
            open_d["days_past"] = (ref - open_d["expected_close_date"]).dt.days
            slip = open_d[open_d["days_past"]>0].sort_values("deal_value", ascending=False)
            return {"type": "slipping_deals", "count": len(slip),
                    "total_value_at_risk": round(float(slip["deal_value"].sum()),2),
                    "top_deals": slip[["deal_id","account_name","deal_value","stage","expected_close_date","days_past","sales_rep"]].head(10).to_string()}
        else:
            accounts_df["renewal_date_dt"] = pd.to_datetime(accounts_df["renewal_date"])
            ref = pd.Timestamp("2026-03-01")
            at_risk = accounts_df[(accounts_df["health_score"]<50)&(accounts_df["renewal_date_dt"]<ref+timedelta(days=90))].sort_values("arr", ascending=False)
            return {"type": "at_risk_accounts", "count": len(at_risk),
                    "total_arr_at_risk": round(float(at_risk["arr"].sum()),2),
                    "top_accounts": at_risk[["account_name","arr","health_score","renewal_date","segment","assigned_rep"]].head(10).to_string()}

    def compare_periods(period1, period2):
        p1 = monthly_metrics[monthly_metrics["close_month"]==period1]
        p2 = monthly_metrics[monthly_metrics["close_month"]==period2]
        if p1.empty or p2.empty:
            return {"error": f"Period not found. Available: {monthly_metrics['close_month'].tolist()}"}
        comp = {}
        for col in ["gross_bookings","deals_won","deals_lost","win_rate","avg_deal_size"]:
            v1, v2 = float(p1[col].iloc[0]), float(p2[col].iloc[0])
            comp[col] = {period1: round(v1,2), period2: round(v2,2), "change_pct": round(((v2-v1)/v1*100) if v1!=0 else 0, 1)}
        return {"comparison": comp}

    def generate_weekly_report():
        start = time.time()
        return {
            "bookings_trend": analyze_trends("gross_bookings"),
            "winrate_trend": analyze_trends("win_rate"),
            "at_risk_deals": find_at_risk("deals"),
            "at_risk_accounts": find_at_risk("accounts"),
            "rep_performance": rep_performance[["sales_rep","total_bookings","deals_won","win_rate","annual_quota","quota_attainment"]].sort_values("win_rate").to_string(),
            "latest_month": {k: str(v) for k,v in monthly_metrics.iloc[-1].to_dict().items()},
            "time": round(time.time()-start, 1)
        }

    return query_data, analyze_trends, find_at_risk, compare_periods, generate_weekly_report


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
    "Q3 (July-September) historically sees 15-20% dip in bookings due to budget cycles.",
    "Accounts with health score below 40 require immediate customer success outreach.",
    "Support tickets above 10 in 90 days correlate with 3x higher churn probability.",
    "West region has been underperforming due to recent team transitions.",
    "Jordan Blake has the lowest win rate — needs coaching and pipeline support.",
    "Rep quota attainment below 80% for two quarters triggers performance improvement plan.",
    "Top loss reason 'Price too high' above 25% suggests pricing strategy review needed.",
    "Annual booking target for the full sales team is approximately $14M.",
    "Weekly sales report is due every Monday for leadership standup.",
    "Underperforming reps identified by win rate below 35%, not quota attainment alone.",
    "NPS score below 5 signals high renewal risk requiring proactive executive outreach.",
    "Late-stage Negotiation deals slipping are highest priority — committed revenue at risk."
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
    return "\n".join([f"• {t}" for _, t in scored[:top_k]])


# ─────────────────────────────────────────────
# HALLUCINATION GUARD
# ─────────────────────────────────────────────
RAG_BENCHMARKS = {14,15,20,21,25,30,35,40,45,50,60,70,75,80,90,100,110,120,150}

def validate_response(text, tool_results):
    numbers = re.findall(r'\$?([\d,]+\.?\d*)\s*(%|M|K)?', text)
    source_nums = set()
    for n in re.findall(r'([\d,]+\.?\d+)', json.dumps(tool_results, default=str)):
        try: source_nums.add(float(n.replace(',','')))
        except: pass
    verified, total, unverified = 0, 0, []
    for num_str, unit in numbers:
        try:
            num = float(num_str.replace(',',''))
            if num < 10: continue
            if num in RAG_BENCHMARKS: continue
            if unit == "%" and num < 30: continue
            total += 1
            if any(abs(num-s)/max(s,0.01)<0.05 for s in source_nums if s>0): verified += 1
            else: unverified.append(f"{num_str}{unit}")
        except: pass
    #return round((verified/max(total,1))*100, 1), verified, total, unverified
    accuracy = round((verified / total) * 100, 1) if total > 0 else 100
    return accuracy, verified, total, unverified


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────
TOOL_DEFS = [
    {"type": "function", "function": {"name": "query_data",
        "description": "Query sales data via natural language SQL. Use for data lookups, rep stats, loss reasons.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "analyze_trends",
        "description": "Analyze trends and detect anomalies. Use for trend/pattern questions.",
        "parameters": {"type": "object", "properties": {"metric": {"type": "string", "description": "gross_bookings, win_rate, deals_won, deals_lost, avg_deal_size, total_pipeline_value"}, "period": {"type": "string", "default": "monthly"}}, "required": ["metric"]}}},
    {"type": "function", "function": {"name": "find_at_risk",
        "description": "Find slipping deals or at-risk accounts near renewal.",
        "parameters": {"type": "object", "properties": {"category": {"type": "string", "enum": ["deals","accounts"]}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "compare_periods",
        "description": "Compare two months side by side.",
        "parameters": {"type": "object", "properties": {"period1": {"type": "string"}, "period2": {"type": "string"}}, "required": ["period1","period2"]}}},
    {"type": "function", "function": {"name": "generate_weekly_report",
        "description": "Generate complete sales operations report. Use only for full report requests.",
        "parameters": {"type": "object", "properties": {}}}}
]

def run_agent(question, client, tools_funcs, kb_embeddings):
    query_data, analyze_trends, find_at_risk, compare_periods, generate_weekly_report = tools_funcs
    tool_map = {"query_data": lambda a: query_data(a.get("question","")),
                "analyze_trends": lambda a: analyze_trends(a.get("metric",""), a.get("period","monthly")),
                "find_at_risk": lambda a: find_at_risk(a.get("category","deals")),
                "compare_periods": lambda a: compare_periods(a.get("period1",""), a.get("period2","")),
                "generate_weekly_report": lambda a: generate_weekly_report()}

    start = time.time()
    messages = [
        {"role": "system", "content": """You are a senior SaaS sales ops analyst with real data access.
RULES: Always use tools. Never make up numbers. Use exact figures. Be concise and executive-friendly. Under 250 words.
When asked about underperformers, order by win_rate ASC."""},
        {"role": "user", "content": question}
    ]

    all_results = {}
    tools_used = []

    for _ in range(5):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages,
                                               tools=TOOL_DEFS, tool_choice="auto", temperature=0)
        msg = resp.choices[0].message
        if not msg.tool_calls: break
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_used.append(fn)
            result = tool_map[fn](args)
            all_results[fn] = result
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})

    rag = retrieve_context(question, client, kb_embeddings)
    messages.append({"role": "user", "content": f"Final answer. Business context:\n{rag}\nUse exact numbers. Under 250 words. End with 1-2 actions."})
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
    answer = final.choices[0].message.content
    accuracy, verified, total, unverified = validate_response(answer, all_results)
    elapsed = round(time.time()-start, 1)

    return answer, accuracy, verified, total, tools_used, elapsed, all_results


# ─────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────
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
    fig = px.pie(loss_reasons, values="count", names="reason", title="Loss Reasons",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=350)
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=60)
        st.markdown("## SaaS Sales Intelligence")
        st.markdown("---")

        #api_key = st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...")
        # Try Streamlit secrets first, then fall back to user input
        api_key = st.secrets.get("OPENAI_API_KEY", "") or st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...")

        if api_key:
            st.success("API Key loaded")
        else:
            st.warning("Enter your OpenAI API key to start")
            st.markdown("""
            **Get a key:**
            1. Go to [platform.openai.com](https://platform.openai.com)
            2. API Keys → Create new
            3. Add $5 credit in Billing
            """)

        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        example_qs = [
            "How did we do last month?",
            "Which rep has the lowest win rate?",
            "Which accounts are at risk?",
            "What is our win rate trend?",
            "Compare 2025-06 vs 2025-09",
            "What are our top loss reasons?",
            "Generate my weekly report"
        ]
        for q in example_qs:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.selected_question = q

        st.markdown("---")
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        **Function Calling** → LLM picks tool
        **Text-to-SQL** → NL → SQL → data
        **RAG** → Business context
        **Validation** → Hallucination guard
        """)

        st.markdown("---")
        st.markdown("Built by **Payal Gore**")
        st.markdown("[GitHub](https://github.com/PayalGore) | [LinkedIn](https://linkedin.com/in/payalgore)")

    # Main content
    st.markdown('<p class="main-header">🤖 SaaS Sales Intelligence Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your sales data in plain English. AI does the rest.</p>', unsafe_allow_html=True)

    # Load data
    deals_df, accounts_df, monthly, rep_perf, loss_reasons = generate_data()

    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    latest = monthly.iloc[-1]
    prev = monthly.iloc[-2]
    with col1:
        st.metric("Gross Bookings", f"${latest['gross_bookings']:,.0f}",
                   f"{((latest['gross_bookings']-prev['gross_bookings'])/prev['gross_bookings']*100):.1f}%")
    with col2:
        st.metric("Win Rate", f"{latest['win_rate']}%",
                   f"{latest['win_rate']-prev['win_rate']:.1f}pp")
    with col3:
        st.metric("Deals Won", int(latest["deals_won"]),
                   f"{int(latest['deals_won']-prev['deals_won'])}")
    with col4:
        st.metric("Avg Deal Size", f"${latest['avg_deal_size']:,.0f}")

    # Charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Bookings", "📈 Win Rate", "👥 Reps", "❌ Losses", "📋 Data"])

    with tab1: st.plotly_chart(plot_bookings_trend(monthly), use_container_width=True)
    with tab2: st.plotly_chart(plot_win_rate_trend(monthly), use_container_width=True)
    with tab3: st.plotly_chart(plot_rep_performance(rep_perf), use_container_width=True)
    with tab4: st.plotly_chart(plot_loss_reasons(loss_reasons), use_container_width=True)
    with tab5:
        data_tab = st.selectbox("Select table", ["Pipeline Deals", "Accounts", "Monthly Metrics", "Rep Performance"])
        if data_tab == "Pipeline Deals": st.dataframe(deals_df, use_container_width=True)
        elif data_tab == "Accounts": st.dataframe(accounts_df, use_container_width=True)
        elif data_tab == "Monthly Metrics": st.dataframe(monthly, use_container_width=True)
        else: st.dataframe(rep_perf, use_container_width=True)

    st.markdown("---")

    # Agent chat
    st.markdown("### 💬 Ask Your Sales Intelligence Agent")

    # Quick-access buttons
    st.markdown("**📌 Single-tool questions:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        if st.button("📊 How did we do last month?", use_container_width=True):
            st.session_state.selected_question = "How did we do last month?"
        if st.button("👥 Lowest win rate rep?", use_container_width=True):
            st.session_state.selected_question = "Which rep has the lowest win rate?"
        if st.button("⚠️ At-risk accounts?", use_container_width=True):
            st.session_state.selected_question = "Which accounts are at risk?"
    with qcol2:
        if st.button("📈 Win rate trend?", use_container_width=True):
            st.session_state.selected_question = "What is our win rate trend?"
        if st.button("❌ Top loss reasons?", use_container_width=True):
            st.session_state.selected_question = "What are our top loss reasons?"
        if st.button("🔄 June vs September", use_container_width=True):
            st.session_state.selected_question = "Compare 2025-06 vs 2025-09"
    with qcol3:
        if st.button("🏢 West region deals?", use_container_width=True):
            st.session_state.selected_question = "Show me all deals in the West region"
        if st.button("💰 Rep performance?", use_container_width=True):
            st.session_state.selected_question = "Show me all reps ranked by win rate"
        if st.button("📉 Slipping deals?", use_container_width=True):
            st.session_state.selected_question = "Which deals are slipping past their expected close date?"

    st.markdown("**🔗 Multi-tool questions (agent chains multiple analyses):**")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if st.button("🔍 Underperformers + loss reasons", use_container_width=True):
            st.session_state.selected_question = "Who is underperforming on the sales team and what are the top reasons we're losing deals?"
        if st.button("📊 Q3 dip analysis", use_container_width=True):
            st.session_state.selected_question = "What happened to our win rate in Q3 and which reps were most affected?"
        if st.button("🏥 Full health check", use_container_width=True):
            st.session_state.selected_question = "Give me a full pipeline health check — at-risk deals, at-risk accounts, and current win rate trend"
    with mcol2:
        if st.button("🔮 Risk + bookings trend", use_container_width=True):
            st.session_state.selected_question = "Which accounts are at risk of churning and what is our bookings trend looking like?"
        if st.button("👥 West region deep dive", use_container_width=True):
            st.session_state.selected_question = "How is the West region performing compared to other regions and which reps there need support?"
        if st.button("📋 Generate weekly report", use_container_width=True):
            st.session_state.selected_question = "Generate my weekly report"

    if not api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to start chatting with the agent.")
        return

    client = OpenAI(api_key=api_key)

    # Initialize RAG
    if st.session_state.kb_embeddings is None:
        with st.spinner("🔄 Initializing RAG knowledge base..."):
            st.session_state.kb_embeddings = init_rag(client)

    tools_funcs = get_tools(deals_df, accounts_df, monthly, rep_perf, loss_reasons, client)

    # Check for sidebar button click
    if "selected_question" in st.session_state:
        question = st.session_state.selected_question
        del st.session_state.selected_question
    else:
        question = st.chat_input("Ask anything about your sales data...")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                answer, accuracy, verified, total, tools_used, elapsed, results = run_agent(
                    question, client, tools_funcs, st.session_state.kb_embeddings
                )

            st.markdown(answer)

            # Metadata bar
            acc_class = "accuracy-high" if accuracy >= 70 else ("accuracy-medium" if accuracy >= 50 else "accuracy-low")
            tools_html = " ".join([f'<span class="tool-badge">{t}</span>' for t in tools_used])

            st.markdown(f"""
            <div style="margin-top: 12px; padding: 10px; background: #f8f9fa; border-radius: 8px; font-size: 0.85rem;">
                <span class="accuracy-badge {acc_class}">🔒 {accuracy}% accuracy ({verified}/{total})</span>
                &nbsp;&nbsp; ⏱️ {elapsed}s &nbsp;&nbsp; {tools_html}
            </div>
            """, unsafe_allow_html=True)

        full_response = f"{answer}\n\n*🔒 {accuracy}% accuracy ({verified}/{total}) | ⏱️ {elapsed}s | Tools: {', '.join(tools_used)}*"
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
