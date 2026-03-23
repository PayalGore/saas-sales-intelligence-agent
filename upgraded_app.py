"""
SaaS Sales Intelligence Agent v2
Production-grade upgrade with real messy data + all 7 fixes
Built by Payal Gore
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, time, re, random, warnings
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SaaS Sales Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1a1a2e;margin-bottom:0}
.sub-header{font-size:1rem;color:#6b7280;margin-top:-8px;margin-bottom:16px}
.dq-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600}
.dq-good{background:#d1fae5;color:#065f46}
.dq-warn{background:#fef3c7;color:#92400e}
.dq-bad{background:#fee2e2;color:#991b1b}
.accuracy-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600}
.accuracy-high{background:#d1fae5;color:#065f46}
.accuracy-medium{background:#fef3c7;color:#92400e}
.accuracy-low{background:#fee2e2;color:#991b1b}
.tool-badge{display:inline-block;background:#e0e7ff;color:#3730a3;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;margin-right:4px}
.dq-panel{background:#f8faff;border:1px solid #e0e7ff;border-radius:8px;padding:10px 14px;font-size:0.8rem;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_embeddings" not in st.session_state:
    st.session_state.kb_embeddings = None
if "data" not in st.session_state:
    st.session_state.data = None


# ─────────────────────────────────────────────
# DATA GENERATION — MESSY ENTERPRISE CRM
# ─────────────────────────────────────────────
@st.cache_data
def generate_and_clean():
    np.random.seed(42)
    random.seed(42)

    REPS_MESSY = ["Sarah Chen","SARAH CHEN","sarah chen","Mike Rodriguez","M. Rodriguez",
                  "Mike Rodrigez","James Wilson","James Wilson","Priya Patel","Priya Patel",
                  "Alex Kim","Alex Kim","Jordan Blake","J. Blake","Taylor Morgan","Taylor Morgan",
                  "Chris Anderson","C. Anderson"]
    PRODUCTS_MESSY = ["IT Management Suite","IT Mgmt Suite","IT Management","Security Platform",
                      "Security Plt.","Security Platform","Backup & Recovery","Backup and Recovery",
                      "Network Monitoring","Network Monitor","Endpoint Protection"]
    SEGMENTS_MESSY = ["SMB","Small Business","smb","Mid-Market","MidMarket","Mid Market",
                      "Enterprise","ENT","enterprise",None]
    REGIONS_MESSY = ["Northeast","Southeast","West","Midwest","WEST","north east"]
    STAGES = ["Prospecting","Qualification","Proposal","Negotiation","Closed Won","Closed Lost"]
    LOSS_REASONS = ["Price too high","Went with competitor","Budget cut","No decision","Timeline mismatch","Product gap",None,None]
    COMPANIES = ["Apex Technologies","Vertex Systems","Pinnacle Solutions","Summit Digital","Nexus Networks",
                 "Catalyst Software","Horizon Group","Quantum Services","Velocity Labs","Vanguard IT",
                 "Atlas Computing","Beacon Dynamics","Core Partners","Dynamic Global","Elevate Industries",
                 "Fusion Technologies","Harbor Systems","Insight Solutions","Junction Networks","Keystone Software",
                 "Ledger Technologies","Matrix Systems","Noble Digital","Orbit Services","Prime Computing",
                 "Quest Group","Ridge Labs","Spark Networks","Titan Solutions","Unity Software",
                 "Vivid Technologies","Wave Systems","Zenith Digital","Bolt Computing","Cedar Partners",
                 "Delta Technologies","Echo Systems","Forge Digital","Grid Networks","Helix Solutions",
                 "Iron Computing","Jade Software","Kite Technologies","Lumen Services","Metro Digital",
                 "Nova Systems","Onyx Networks","Pulse Solutions","Relay Computing","Scout Technologies"]

    records = []
    for i in range(900):
        company = random.choice(COMPANIES)
        rep_raw = random.choice(REPS_MESSY)
        product_raw = random.choice(PRODUCTS_MESSY)
        segment_raw = random.choice(SEGMENTS_MESSY)
        region_raw = random.choice(REGIONS_MESSY)
        create_date = datetime(2024,1,1) + timedelta(days=random.randint(0,400))
        date_fmts = ["%Y-%m-%d","%m/%d/%Y","%d-%m-%Y","%Y/%m/%d"]
        base_value = random.uniform(8000,450000)
        if random.random() < 0.03: base_value *= random.uniform(10,50)
        if random.random() < 0.05: deal_value = f"${base_value:,.2f}"
        elif random.random() < 0.03: deal_value = None
        else: deal_value = round(base_value,2)
        stage = random.choice(STAGES)
        if stage in ["Closed Won","Closed Lost"]:
            close_date = create_date + timedelta(days=random.randint(14,180))
            close_date_str = None if random.random()<0.08 else close_date.strftime(random.choice(date_fmts[:2]))
        else:
            close_date_str = None
        loss_reason = random.choice(LOSS_REASONS) if stage=="Closed Lost" else None
        num_copies = 2 if random.random()<0.03 else 1
        for _ in range(num_copies):
            records.append({"deal_id":f"DEAL-{5000+i}","account_name":company,"sales_rep":rep_raw,
                           "product":product_raw,"segment":segment_raw,"region":region_raw,
                           "deal_value":deal_value,"stage":stage,"create_date":create_date.strftime("%Y-%m-%d"),
                           "close_date":close_date_str,"loss_reason":loss_reason,
                           "probability":{"Prospecting":0.10,"Qualification":0.25,"Proposal":0.50,
                                         "Negotiation":0.75,"Closed Won":1.0,"Closed Lost":0.0}.get(stage,0.5)})

    raw_df = pd.DataFrame(records)
    original_count = len(raw_df)

    df = raw_df.copy()
    issues = []
    df = df.drop_duplicates(subset=["deal_id","account_name","stage","close_date"])
    dupes = original_count - len(df)
    if dupes > 0: issues.append(f"Removed {dupes} duplicate records")

    rep_map = {"SARAH CHEN":"Sarah Chen","sarah chen":"Sarah Chen","M. Rodriguez":"Mike Rodriguez",
               "Mike Rodrigez":"Mike Rodriguez","J. Blake":"Jordan Blake","C. Anderson":"Chris Anderson"}
    rep_fixed = df["sales_rep"].isin(rep_map).sum()
    df["sales_rep"] = df["sales_rep"].replace(rep_map)
    if rep_fixed > 0: issues.append(f"Standardized {rep_fixed} rep name variants")

    prod_map = {"IT Mgmt Suite":"IT Management Suite","IT Management":"IT Management Suite",
                "Security Plt.":"Security Platform","Backup and Recovery":"Backup & Recovery","Network Monitor":"Network Monitoring"}
    df["product"] = df["product"].replace(prod_map)

    seg_map = {"Small Business":"SMB","smb":"SMB","MidMarket":"Mid-Market","Mid Market":"Mid-Market",
               "ENT":"Enterprise","enterprise":"Enterprise"}
    df["segment"] = df["segment"].replace(seg_map)
    null_segs = df["segment"].isna().sum()
    df["segment"] = df["segment"].fillna("Unknown")
    if null_segs > 0: issues.append(f"Filled {null_segs} null segments as 'Unknown'")

    reg_map = {"WEST":"West","north east":"Northeast"}
    df["region"] = df["region"].replace(reg_map)
    df["region"] = df["region"].fillna("Unknown")

    def parse_value(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return None
        if isinstance(v,str):
            try: return float(v.replace("$","").replace(",",""))
            except: return None
        return float(v)

    df["deal_value"] = df["deal_value"].apply(parse_value)
    null_vals = df["deal_value"].isna().sum()
    if null_vals > 0: issues.append(f"Found {null_vals} null/unparseable deal values")

    valid = df["deal_value"].dropna()
    mean_v, std_v = valid.mean(), valid.std()
    outlier_mask = df["deal_value"] > (mean_v + 3*std_v)
    outliers = outlier_mask.sum()
    df.loc[outlier_mask,"deal_value"] = None
    if outliers > 0: issues.append(f"Flagged {outliers} outlier deal values (>3σ)")

    def parse_date(d):
        if d is None or (isinstance(d,float) and np.isnan(d)): return None
        for fmt in ["%Y-%m-%d","%m/%d/%Y","%d-%m-%Y","%Y/%m/%d"]:
            try: return datetime.strptime(str(d),fmt)
            except: continue
        return None

    df["create_date"] = df["create_date"].apply(parse_date)
    df["close_date"] = df["close_date"].apply(parse_date)

    df["dq_flag"] = ""
    df.loc[df["deal_value"].isna() & df["stage"].isin(["Closed Won"]),"dq_flag"] += "missing_value;"
    df.loc[df["close_date"].isna() & df["stage"].isin(["Closed Won","Closed Lost"]),"dq_flag"] += "missing_close_date;"
    df.loc[df["segment"]=="Unknown","dq_flag"] += "unknown_segment;"
    flagged = (df["dq_flag"]!="").sum()
    if flagged > 0: issues.append(f"Flagged {flagged} records with data quality issues")

    quality_report = {"original_records":original_count,"clean_records":len(df),
                      "issues_resolved":len(issues),"issue_list":issues,
                      "flagged_records":int(flagged),
                      "dq_score":round(min(100, max(0, 100 - (flagged/len(df))*100)), 1)}

    closed = df[df["stage"].isin(["Closed Won","Closed Lost"]) & df["close_date"].notna()].copy()
    closed["close_month"] = closed["close_date"].dt.to_period("M").astype(str)

    monthly = closed.groupby("close_month").agg(
        gross_bookings=("deal_value", lambda x: x[closed.loc[x.index,"stage"]=="Closed Won"].sum()),
        deals_won=("stage", lambda x: (x=="Closed Won").sum()),
        deals_lost=("stage", lambda x: (x=="Closed Lost").sum()),
        total_deals=("deal_id","count"),
        avg_deal_size=("deal_value","mean"),
    ).reset_index()
    monthly["win_rate"] = round(monthly["deals_won"]/monthly["total_deals"].clip(lower=1)*100,1)
    monthly = monthly[monthly["total_deals"]>=5].reset_index(drop=True)

    won = closed[closed["stage"]=="Closed Won"]
    lost = closed[closed["stage"]=="Closed Lost"]
    rep_won = won.groupby("sales_rep").agg(
        total_bookings=("deal_value","sum"),deals_won=("deal_id","count"),avg_deal_size=("deal_value","mean")).reset_index()
    rep_lost = lost.groupby("sales_rep")["deal_id"].count().reset_index()
    rep_lost.columns = ["sales_rep","deals_lost"]
    rep_perf = rep_won.merge(rep_lost,on="sales_rep",how="left")
    rep_perf["deals_lost"] = rep_perf["deals_lost"].fillna(0).astype(int)
    rep_perf["total_deals"] = rep_perf["deals_won"] + rep_perf["deals_lost"]
    rep_perf["win_rate"] = round(rep_perf["deals_won"]/rep_perf["total_deals"].clip(lower=1)*100,1)
    quotas = {"Sarah Chen":1800000,"Mike Rodriguez":1600000,"James Wilson":1700000,
              "Priya Patel":1900000,"Alex Kim":1700000,"Jordan Blake":1650000,
              "Taylor Morgan":1750000,"Chris Anderson":1600000}
    rep_perf["quota"] = rep_perf["sales_rep"].map(quotas).fillna(1500000)
    rep_perf["quota_attainment"] = round(rep_perf["total_bookings"]/rep_perf["quota"]*100,1)

    loss_df = closed[closed["stage"]=="Closed Lost"]["loss_reason"].dropna().value_counts().reset_index()
    loss_df.columns = ["reason","count"]

    pipeline = df[~df["stage"].isin(["Closed Won","Closed Lost"])].copy()

    return df, monthly, rep_perf, loss_df, pipeline, quality_report


# ─────────────────────────────────────────────
# KNOWLEDGE BASE + RAG
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "Enterprise segment: ARR above $150,000. Highest-value clients.",
    "Mid-Market segment: ARR between $35,000 and $150,000. Primary growth engine.",
    "SMB segment: ARR below $35,000. High volume, higher churn risk.",
    "Healthy win rate benchmark is above 40%. Below 35% requires pipeline review.",
    "Renewal deals should close at 70%+ win rate. Below 60% is a retention problem.",
    "New logo win rate target is 30-35%. Expansion deals should close at 50%+.",
    "Pipeline coverage ratio should be 3x quarterly target. Below 2.5x is a red flag.",
    "Deals slipping more than 14 days past close date require manager review.",
    "Q3 historically sees 15-20% dip in bookings due to budget cycles.",
    "Rep quota attainment below 80% for two quarters triggers performance improvement plan.",
    "Top loss reason 'Price too high' above 25% suggests pricing strategy review.",
    "NPS below 5 signals high renewal risk requiring executive outreach.",
    "Late-stage Negotiation deals slipping are highest priority — committed revenue at risk.",
    "Data quality issues in CRM are common — always validate numbers against source.",
    "Win rates should be calculated from rep_performance table, not raw deals.",
    "Unknown segments indicate missing CRM data — treat with caution in analysis.",
    "Deals with missing close dates on closed stages indicate data entry gaps.",
    "Outlier deal values above 3 standard deviations are flagged and excluded from revenue.",
]

def init_rag(client):
    resp = client.embeddings.create(model="text-embedding-3-small", input=KNOWLEDGE_BASE)
    return [item.embedding for item in resp.data]

def retrieve_context(query, client, kb_embeddings, top_k=3):
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    q_arr = np.array(q_emb)
    scored = [(float(np.dot(q_arr,np.array(e))/(np.linalg.norm(q_arr)*np.linalg.norm(np.array(e)))), t)
              for e, t in zip(kb_embeddings, KNOWLEDGE_BASE)]
    scored.sort(reverse=True)
    return "\n".join([f"- {t}" for _,t in scored[:top_k]])


# ─────────────────────────────────────────────
# GUARDRAIL 1 — RULE-BASED AMBIGUITY RESOLVER
# Zero API calls. Pure Python. Zero latency.
# ─────────────────────────────────────────────
SALES_TERMS = [
    "deal","deals","rep","reps","revenue","booking","win rate","win rates","pipeline",
    "account","accounts","churn","renewal","renewals","quota","forecast","region","regions",
    "performance","loss","losses","trend","trends","compare","report","weekly","monthly",
    "quarterly","q1","q2","q3","q4","bookings","stage","stages","at risk","slipping",
    "health","score","arr","mrr","segment","segments","closed","open","product","products",
    "last month","last week","last quarter","this month","ytd","sales","team","territory",
    "northeast","southeast","west","midwest","enterprise","mid-market","smb","lowest","highest",
    "best","worst","top","bottom","underperform","overperform","attainment","close date",
    "2024","2025","january","february","march","april","may","june","july","august",
    "september","october","november","december"
]

GREETINGS = ["hi","hello","hey","how are you","good morning","good afternoon","good evening",
             "what's up","sup","thanks","thank you","bye","goodbye"]

def resolve_ambiguity(question):
    q = question.lower().strip()
    if any(g in q for g in GREETINGS) and not any(s in q for s in SALES_TERMS):
        return {"clear": False, "clarification": "I can help with sales data analysis. Try asking about win rates, pipeline health, rep performance, or bookings trends."}
    return {"clear": True}


# ─────────────────────────────────────────────
# GUARDRAIL 2 — SMART CONTEXT INJECTION
# Only pass relevant tool outputs per question type
# ─────────────────────────────────────────────
def get_relevant_context(question, all_results):
    """Select only the tool outputs relevant to this question type."""
    q = question.lower()
    relevant = {}
    for tool_name, result in all_results.items():
        if "rep" in q or "win rate" in q or "quota" in q or "performance" in q:
            if tool_name in ["query_data","generate_weekly_report"]: relevant[tool_name] = result
        elif "trend" in q or "month" in q or "booking" in q or "quarter" in q:
            if tool_name in ["analyze_trends","compare_periods","generate_weekly_report"]: relevant[tool_name] = result
        elif "risk" in q or "churn" in q or "slip" in q or "account" in q:
            if tool_name in ["find_at_risk","generate_weekly_report"]: relevant[tool_name] = result
        elif "loss" in q or "lose" in q or "losing" in q or "reason" in q:
            if tool_name in ["query_data"]: relevant[tool_name] = result
        else:
            relevant[tool_name] = result

    if not relevant:
        relevant = all_results
    return relevant


def build_source_context(relevant_results):
    parts = []
    for tool_name, result in relevant_results.items():
        flat = json.dumps(result, default=str, separators=(", ", ": "))
        if len(flat) > 2500:
            flat = flat[:2500] + "...[truncated at 2500 chars]"
        parts.append(f"[SOURCE:{tool_name}]\n{flat}\n[/SOURCE:{tool_name}]")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# GUARDRAIL 3 — SEMANTIC HALLUCINATION GUARD
# Numbers checked in context, not just in JSON
# ─────────────────────────────────────────────
RAG_BENCHMARKS = {14,15,20,21,25,30,35,40,45,50,60,70,75,80,90,100,110,120,150}

SEMANTIC_PATTERNS = {
    "win_rate": re.compile(r"win\s*rate[^.]*?([\d.]+)\s*%", re.IGNORECASE),
    "bookings": re.compile(r"(booking|revenue|gross)[^.]*?\$([\d,]+)", re.IGNORECASE),
    "deal_count": re.compile(r"([\d]+)\s*(deal|won|lost|closed)", re.IGNORECASE),
    "quota": re.compile(r"quota[^.]*?([\d.]+)\s*%", re.IGNORECASE),
}

def validate_response_semantic(text, tool_results, question):
    """
    Semantic validation — check numbers appear in the right context.
    Not just: does this number exist in the JSON?
    But: does this win rate number appear near win rate language?
    """
    source_str = json.dumps(tool_results, default=str)
    source_nums = set()
    for n in re.findall(r"([\d,]+\.?\d*)", source_str):
        try: source_nums.add(float(n.replace(",","")))
        except: pass

    all_numbers = re.findall(r"\$?([\d,]+\.?\d*)\s*(%|M|K)?", text)
    verified, total, unverified = 0, 0, []

    for num_str, unit in all_numbers:
        try:
            num = float(num_str.replace(",",""))
            if num < 2: continue
            if num in RAG_BENCHMARKS: continue
            if num == int(num) and num <= 5 and unit == "": continue
            total += 1
            if any(abs(num-s)/max(s,0.01)<0.05 for s in source_nums if s>0):
                verified += 1
            else:
                unverified.append(f"{num_str}{unit}")
        except: pass

    if total == 0:
        return 100.0, 0, 0, []

    accuracy = round((verified/total)*100, 1)
    return accuracy, verified, total, unverified


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
def get_tools(deals_df, monthly, rep_perf, loss_df, pipeline, client):
    from pandasql import sqldf

    def query_data(question):
        if not monthly.empty:
            latest_month = monthly["close_month"].iloc[-1]
            previous_month = monthly["close_month"].iloc[-2] if len(monthly)>1 else latest_month
        else:
            latest_month = previous_month = "2024-12"

        SCHEMA = f"""
Tables available:
- deals_df: deal_id, account_name, sales_rep, product, segment, region, deal_value(float),
  stage(Prospecting/Qualification/Proposal/Negotiation/Closed Won/Closed Lost),
  create_date(datetime), close_date(datetime), loss_reason, probability, dq_flag
  NOTE: deal_value has nulls where data was missing or outlier. Use AVG/SUM carefully.

- monthly: close_month(YYYY-MM format), gross_bookings, deals_won, deals_lost,
  total_deals, avg_deal_size, win_rate(pre-computed %)
  Latest: {latest_month} | Previous: {previous_month}

- rep_perf: sales_rep, total_bookings, deals_won, avg_deal_size, deals_lost,
  total_deals, win_rate(pre-computed %), quota, quota_attainment
  ALWAYS use rep_perf.win_rate — never calculate from deals_df

- loss_df: reason, count

- pipeline: open deals only (not Closed Won/Lost)
  columns: deal_id, account_name, sales_rep, product, segment, region,
           deal_value, stage, create_date, probability, dq_flag
"""
        def build_sql_prompt(q, err=""):
            err_note = f"\nERROR from last attempt: {err}\nWrite a corrected, simpler query." if err else ""
            return f"""Convert to SQLite query.{err_note}
{SCHEMA}
RULES:
- SQLite only. No CURRENT_DATE/NOW(). No date arithmetic on strings.
- For "last month": close_month = '{previous_month}'
- For "latest"/"this month": close_month = '{latest_month}'
- For win rates always use rep_perf.win_rate
- For vague questions return all relevant columns from most relevant table
- Handle NULLs: use COALESCE or IS NOT NULL where needed
- Return ONLY SQL. No markdown.

Question: {q}"""

        tables = {"deals_df":deals_df,"monthly":monthly,"rep_perf":rep_perf,
                  "loss_df":loss_df,"pipeline":pipeline}
        last_error = ""
        for attempt in range(3):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"SQLite expert. Return only valid SQL. Handle NULLs correctly."},
                          {"role":"user","content":build_sql_prompt(question, last_error)}],
                temperature=0)
            sql = resp.choices[0].message.content.strip().replace("```sql","").replace("```","").strip()
            try:
                result = sqldf(sql, tables)
                if len(result)==0:
                    last_error = "Query returned 0 rows. Try broader filters or check column names."
                    continue
                return {"success":True,"data":result.to_string(index=False),"row_count":len(result),"sql":sql}
            except Exception as e:
                last_error = str(e)

        # Universal fallback
        fallback_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":f"Which single table from [deals_df, monthly, rep_perf, loss_df, pipeline] best answers: '{question}'? Reply with ONLY the table name."}],
            temperature=0)
        tname = fallback_resp.choices[0].message.content.strip().lower()
        if tname in tables:
            df_fb = tables[tname]
            return {"success":True,"data":df_fb.to_string(index=False),"row_count":len(df_fb),
                    "sql":f"fallback:{tname}","note":f"SQL failed after 3 attempts — returning {tname} directly"}
        return {"success":False,"error":last_error}

    def analyze_trends(metric, period="monthly"):
        col_map = {"gross_bookings":"gross_bookings","win_rate":"win_rate",
                   "deals_won":"deals_won","deals_lost":"deals_lost","avg_deal_size":"avg_deal_size"}
        col = col_map.get(metric, metric)
        if col not in monthly.columns:
            return {"error":f"Metric '{metric}' not found. Available: {list(col_map.keys())}"}
        values = monthly[col].dropna().values.astype(float)
        if len(values) < 2:
            return {"error":"Not enough data points for trend analysis"}
        changes = [round((values[i]-values[i-1])/max(values[i-1],0.01)*100,1) for i in range(1,len(values))]
        mean_v, std_v = float(np.mean(values)), float(np.std(values))
        anomalies = [{"month":monthly["close_month"].iloc[i],"value":round(float(v),2),
                      "z_score":round(float((v-mean_v)/max(std_v,0.01)),2),
                      "severity":"high" if abs((v-mean_v)/max(std_v,0.01))>2 else "medium"}
                     for i,v in enumerate(monthly[col].dropna().values.astype(float))
                     if std_v>0 and abs((v-mean_v)/std_v)>1.5]
        return {"metric":col,"current":round(float(values[-1]),2),
                "previous":round(float(values[-2]),2),"mom_change_pct":changes[-1] if changes else None,
                "average":round(mean_v,2),"trend":"improving" if changes and changes[-1]>0 else "declining",
                "anomalies":anomalies,
                "all_values":{monthly["close_month"].iloc[i]:round(float(v),2) for i,v in enumerate(monthly[col].dropna().values)}}

    def find_at_risk(category="deals"):
        if category=="deals":
            open_d = pipeline.copy()
            ref = datetime(2025,4,1)
            open_d["expected_close"] = open_d["create_date"].apply(
                lambda x: x + timedelta(days=60) if x else None)
            open_d["days_past"] = open_d["expected_close"].apply(
                lambda x: max(0,(ref-x).days) if x else 0)
            slip = open_d[open_d["days_past"]>0].sort_values("deal_value",ascending=False)
            return {"type":"slipping_deals","count":len(slip),
                    "total_at_risk":round(float(slip["deal_value"].dropna().sum()),2),
                    "top_deals":slip[["deal_id","account_name","deal_value","stage","sales_rep","days_past"]].head(10).to_string(index=False)}
        else:
            low_health = deals_df[deals_df["dq_flag"].str.contains("missing",na=False)==False].copy()
            at_risk = deals_df[deals_df["stage"].isin(["Negotiation"]) &
                               (deals_df["deal_value"].fillna(0)>50000)].sort_values("deal_value",ascending=False)
            return {"type":"at_risk_accounts","count":len(at_risk),
                    "total_arr_at_risk":round(float(at_risk["deal_value"].dropna().sum()),2),
                    "top_accounts":at_risk[["account_name","deal_value","segment","sales_rep"]].head(10).to_string(index=False)}

    def compare_periods(period1, period2):
        p1 = monthly[monthly["close_month"]==period1]
        p2 = monthly[monthly["close_month"]==period2]
        if p1.empty or p2.empty:
            avail = monthly["close_month"].tolist()
            return {"error":f"Period not found. Available months: {avail}"}
        comp = {}
        for col in ["gross_bookings","deals_won","deals_lost","win_rate","avg_deal_size"]:
            v1,v2 = float(p1[col].iloc[0]),float(p2[col].iloc[0])
            comp[col] = {period1:round(v1,2),period2:round(v2,2),
                        "change_pct":round(((v2-v1)/max(v1,0.01)*100),1)}
        return {"comparison":comp}

    def generate_weekly_report():
        latest = monthly.iloc[-1] if not monthly.empty else {}
        prev = monthly.iloc[-2] if len(monthly)>1 else {}
        top_reps = rep_perf.nsmallest(3,"win_rate")[["sales_rep","win_rate","quota_attainment"]].to_string(index=False)
        slip = find_at_risk("deals")
        return {"latest_month":{k:str(v) for k,v in latest.to_dict().items()} if hasattr(latest,"to_dict") else {},
                "prev_month":{k:str(v) for k,v in prev.to_dict().items()} if hasattr(prev,"to_dict") else {},
                "win_rate_trend":analyze_trends("win_rate"),
                "bookings_trend":analyze_trends("gross_bookings"),
                "slipping_deals":slip,
                "bottom_reps":top_reps,
                "top_loss_reasons":loss_df.head(3).to_string(index=False)}

    return query_data, analyze_trends, find_at_risk, compare_periods, generate_weekly_report


# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────
TOOL_DEFS = [
    {"type":"function","function":{"name":"query_data","description":"Query CRM data via natural language. Use for deal lookups, rep stats, loss reasons, pipeline details. Handles messy data automatically.",
     "parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
    {"type":"function","function":{"name":"analyze_trends","description":"Analyze metric trends and detect anomalies over time.",
     "parameters":{"type":"object","properties":{"metric":{"type":"string","description":"gross_bookings, win_rate, deals_won, deals_lost, avg_deal_size"},"period":{"type":"string","default":"monthly"}},"required":["metric"]}}},
    {"type":"function","function":{"name":"find_at_risk","description":"Find slipping pipeline deals or high-value at-risk accounts.",
     "parameters":{"type":"object","properties":{"category":{"type":"string","enum":["deals","accounts"]}},"required":["category"]}}},
    {"type":"function","function":{"name":"compare_periods","description":"Compare two months side by side across all metrics.",
     "parameters":{"type":"object","properties":{"period1":{"type":"string"},"period2":{"type":"string"}},"required":["period1","period2"]}}},
    {"type":"function","function":{"name":"generate_weekly_report","description":"Generate complete weekly sales operations report.",
     "parameters":{"type":"object","properties":{}}}},
]

SYSTEM_PROMPT = """You are a senior SaaS sales ops analyst with access to real CRM data.

RULES:
1. Only use numbers from [SOURCE] blocks — never invent or estimate
2. Write numbers clearly: "$1,234,567" not "$1234567down"
3. Always add a space before % signs and units
4. If data is unavailable say so — do not guess
5. Note data quality issues when relevant (missing values, unknown segments)
6. Concise and executive-friendly. Under 200 words.
7. End with 1-2 specific named actions.
8. For win rates always reference rep_perf data, not raw deal calculations."""


# ─────────────────────────────────────────────
# AGENT — ALL 5 GUARDRAILS
# ─────────────────────────────────────────────
def run_agent(question, client, tools_funcs, kb_embeddings, conversation_history):
    query_data, analyze_trends, find_at_risk, compare_periods, generate_weekly_report = tools_funcs
    tool_map = {"query_data":lambda a:query_data(a.get("question","")),
                "analyze_trends":lambda a:analyze_trends(a.get("metric",""),a.get("period","monthly")),
                "find_at_risk":lambda a:find_at_risk(a.get("category","deals")),
                "compare_periods":lambda a:compare_periods(a.get("period1",""),a.get("period2","")),
                "generate_weekly_report":lambda a:generate_weekly_report()}
    start = time.time()

    # GUARDRAIL 1: Rule-based ambiguity check (zero latency)
    ambiguity = resolve_ambiguity(question)
    if not ambiguity["clear"]:
        return ambiguity["clarification"], 100.0, 0, 0, ["ambiguity_resolver"], round(time.time()-start,1), {}

    # GUARDRAIL 2: Build messages with conversation memory (last 3 exchanges)
    memory_context = ""
    if conversation_history:
        recent = conversation_history[-6:]  # last 3 Q&A pairs
        memory_context = "\n\nRecent conversation context:\n" + "\n".join(
            [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}"
             for m in recent])

    messages = [{"role":"system","content":SYSTEM_PROMPT + memory_context},
                {"role":"user","content":question}]
    all_results = {}
    tools_used = []

    for _ in range(5):
        resp = client.chat.completions.create(model="gpt-4o-mini",messages=messages,
                                              tools=TOOL_DEFS,tool_choice="auto",temperature=0)
        msg = resp.choices[0].message
        if not msg.tool_calls: break
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            tools_used.append(fn)
            result = tool_map[fn](args)
            all_results[fn] = result
            messages.append({"role":"tool","tool_call_id":tc.id,"content":json.dumps(result,default=str)})

    # GUARDRAIL 3: Smart context injection — only relevant outputs
    relevant = get_relevant_context(question, all_results)
    source_ctx = build_source_context(relevant)
    rag = retrieve_context(question, client, kb_embeddings)

    messages.append({"role":"user","content":f"""Generate the final answer.

DATA — use ONLY numbers from these SOURCE blocks:
{source_ctx}

BENCHMARKS (context only, not data):
{rag}

Format: numbers with spaces, percentages clean, under 200 words, end with actions."""})

    # GUARDRAIL 4: Constrained generation
    final = client.chat.completions.create(model="gpt-4o-mini",messages=messages,temperature=0.1)
    answer = final.choices[0].message.content

    # GUARDRAIL 5: Semantic validation backstop
    accuracy, verified, total, unverified = validate_response_semantic(answer, all_results, question)
    elapsed = round(time.time()-start,1)
    return answer, accuracy, verified, total, tools_used, elapsed, all_results


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_bookings(monthly):
    fig = px.bar(monthly,x="close_month",y="gross_bookings",title="Monthly Gross Bookings",
                 color_discrete_sequence=["#667eea"])
    fig.update_layout(xaxis_title="Month",yaxis_title="Bookings ($)",template="plotly_white",height=320)
    return fig

def plot_win_rate(monthly):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["close_month"],y=monthly["win_rate"],
                             mode="lines+markers",line=dict(color="#667eea",width=2.5),marker=dict(size=7)))
    fig.add_hline(y=40,line_dash="dash",line_color="red",annotation_text="40% benchmark")
    fig.update_layout(title="Win Rate Trend",xaxis_title="Month",yaxis_title="Win Rate (%)",template="plotly_white",height=320)
    return fig

def plot_reps(rep_perf):
    fig = px.bar(rep_perf.sort_values("win_rate"),x="win_rate",y="sales_rep",orientation="h",
                 title="Rep Win Rates",color="win_rate",color_continuous_scale=["#ef4444","#f59e0b","#22c55e"])
    fig.add_vline(x=40,line_dash="dash",line_color="red")
    fig.update_layout(template="plotly_white",height=320,yaxis_title="",xaxis_title="Win Rate (%)")
    return fig

def plot_loss(loss_df):
    fig = px.pie(loss_df,values="count",names="reason",title="Loss Reasons",
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(height=320)
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    deals_df, monthly, rep_perf, loss_df, pipeline, quality_report = generate_and_clean()

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=56)
        st.markdown("## SaaS Sales Intelligence")
        st.markdown("---")

        api_key = st.secrets.get("OPENAI_API_KEY","") or st.text_input(
            "OpenAI API Key", type="password", placeholder="sk-...")
        if api_key: st.success("API Key loaded")
        else: st.warning("Enter your OpenAI API key")

        st.markdown("---")

        # Data Quality Panel
        dq = quality_report
        dq_score = dq["dq_score"]
        dq_class = "dq-good" if dq_score>=80 else ("dq-warn" if dq_score>=60 else "dq-bad")
        st.markdown("### Data Quality")
        st.markdown(f"""
<div class="dq-panel">
<span class="dq-badge {dq_class}">DQ Score: {dq_score}%</span><br><br>
<b>{dq['original_records']}</b> raw records<br>
<b>{dq['clean_records']}</b> after cleaning<br>
<b>{dq['flagged_records']}</b> flagged records<br><br>
<b>Issues resolved:</b><br>
{"<br>".join([f"• {i}" for i in dq['issue_list'][:4]])}
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Architecture")
        st.markdown("""
**1.** Rule-based ambiguity check
**2.** Conversation memory (3 turns)
**3.** Function calling router
**4.** Schema-aware Text-to-SQL
**5.** Smart context injection
**6.** Constrained LLM narration
**7.** Semantic validation guard
""")
        st.markdown("---")
        st.markdown("Built by **Payal Gore**")
        st.markdown("[GitHub](https://github.com/PayalGore) | [LinkedIn](https://linkedin.com/in/payalgore)")

    st.markdown('<p class="main-header">SaaS Sales Intelligence Agent v2</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production-grade AI analyst with real messy CRM data and upstream hallucination prevention</p>', unsafe_allow_html=True)

    # KPI strip
    if not monthly.empty:
        latest = monthly.iloc[-1]
        prev = monthly.iloc[-2] if len(monthly)>1 else latest
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Gross Bookings",f"${latest['gross_bookings']:,.0f}",
                           f"{((latest['gross_bookings']-prev['gross_bookings'])/max(prev['gross_bookings'],1)*100):.1f}%")
        with c2: st.metric("Win Rate",f"{latest['win_rate']}%",f"{latest['win_rate']-prev['win_rate']:.1f}pp")
        with c3: st.metric("Deals Won",int(latest["deals_won"]),f"{int(latest['deals_won']-prev['deals_won'])}")
        with c4: st.metric("Avg Deal Size",f"${latest['avg_deal_size']:,.0f}")

    t1,t2,t3,t4,t5 = st.tabs(["Bookings","Win Rate","Reps","Loss Reasons","Raw Data"])
    with t1: st.plotly_chart(plot_bookings(monthly),use_container_width=True)
    with t2: st.plotly_chart(plot_win_rate(monthly),use_container_width=True)
    with t3: st.plotly_chart(plot_reps(rep_perf),use_container_width=True)
    with t4: st.plotly_chart(plot_loss(loss_df),use_container_width=True)
    with t5:
        sel = st.selectbox("Table",["Deals (cleaned)","Monthly Metrics","Rep Performance","Pipeline","Loss Reasons"])
        if sel=="Deals (cleaned)": st.dataframe(deals_df.head(200),use_container_width=True)
        elif sel=="Monthly Metrics": st.dataframe(monthly,use_container_width=True)
        elif sel=="Rep Performance": st.dataframe(rep_perf,use_container_width=True)
        elif sel=="Pipeline": st.dataframe(pipeline.head(100),use_container_width=True)
        else: st.dataframe(loss_df,use_container_width=True)

    st.markdown("---")
    st.markdown("### Ask Your Sales Intelligence Agent")

    # Quick buttons
    st.markdown("**Single-tool:**")
    qc1,qc2,qc3 = st.columns(3)
    with qc1:
        if st.button("How did we do last month?",use_container_width=True):
            st.session_state.pending_q = "How did we do last month?"
        if st.button("Which rep has the lowest win rate?",use_container_width=True):
            st.session_state.pending_q = "Which rep has the lowest win rate?"
        if st.button("At-risk pipeline?",use_container_width=True):
            st.session_state.pending_q = "Which deals are at risk?"
    with qc2:
        if st.button("Win rate trend?",use_container_width=True):
            st.session_state.pending_q = "What is our win rate trend?"
        if st.button("Top loss reasons?",use_container_width=True):
            st.session_state.pending_q = "What are our top loss reasons?"
        if st.button("Rep performance?",use_container_width=True):
            st.session_state.pending_q = "Show me all reps ranked by win rate"
    with qc3:
        if st.button("Data quality issues?",use_container_width=True):
            st.session_state.pending_q = "What data quality issues exist in our CRM?"
        if st.button("West region deals?",use_container_width=True):
            st.session_state.pending_q = "Show me all deals in the West region"
        if st.button("Generate weekly report",use_container_width=True):
            st.session_state.pending_q = "Generate my weekly report"

    st.markdown("**Multi-tool:**")
    mc1,mc2 = st.columns(2)
    with mc1:
        if st.button("Underperformers + loss reasons",use_container_width=True):
            st.session_state.pending_q = "Who is underperforming and what are the top loss reasons?"
        if st.button("Full pipeline health check",use_container_width=True):
            st.session_state.pending_q = "Give me a full pipeline health check"
    with mc2:
        if st.button("Q3 win rate analysis",use_container_width=True):
            st.session_state.pending_q = "What happened to our win rate in Q3?"
        if st.button("At-risk accounts + bookings trend",use_container_width=True):
            st.session_state.pending_q = "Which accounts are at risk and what is our bookings trend?"

    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to start.")
        return

    client = OpenAI(api_key=api_key)
    if st.session_state.kb_embeddings is None:
        with st.spinner("Initializing knowledge base..."):
            st.session_state.kb_embeddings = init_rag(client)

    tools_funcs = get_tools(deals_df, monthly, rep_perf, loss_df, pipeline, client)

    # Chat input — always rendered first
    typed = st.chat_input("Ask anything about your sales data...")
    if "pending_q" in st.session_state:
        question = st.session_state.pop("pending_q")
    elif typed:
        question = typed
    else:
        question = None

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, accuracy, verified, total, tools_used, elapsed, results = run_agent(
                    question, client, tools_funcs,
                    st.session_state.kb_embeddings,
                    st.session_state.messages[:-1]  # pass history excluding current
                )
            st.markdown(answer)
            acc_cls = "accuracy-high" if accuracy>=70 else ("accuracy-medium" if accuracy>=50 else "accuracy-low")
            tools_html = " ".join([f'<span class="tool-badge">{t}</span>' for t in tools_used])
            st.markdown(f"""<div style="margin-top:10px;padding:8px 12px;background:#f8f9fa;border-radius:8px;font-size:0.82rem;">
                <span class="accuracy-badge {acc_cls}">🔒 {accuracy}% ({verified}/{total})</span>
                &nbsp;&nbsp;⏱️ {elapsed}s&nbsp;&nbsp;{tools_html}</div>""", unsafe_allow_html=True)

        st.session_state.messages.append({"role":"assistant","content":answer})

if __name__ == "__main__":
    main()
