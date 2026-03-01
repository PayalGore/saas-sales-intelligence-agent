# 🤖 SaaS Sales Intelligence Agent

**An AI-powered self-serve analytics tool for B2B SaaS sales operations.**

Sales ops leaders ask questions in plain English → the AI agent decides what analysis to run, executes it on real data, and returns trusted, validated insights — in seconds, not hours.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green?logo=openai)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 The Problem

In most B2B SaaS companies, sales operations teams face a recurring bottleneck:

- **Data lives in warehouses (Snowflake, Redshift)**, but business users think in Excel
- **Weekly sales reports take 4-6 hours** of manual work — pulling, calculating, writing commentary
- **Pipeline risks go undetected** until it's too late — slipping deals, at-risk renewals, underperforming reps
- **Every ad-hoc question** requires an analyst to write SQL, run queries, and interpret results

The gap between where data lives and who needs it shouldn't require a human bottleneck every week.

---

## 💡 The Solution

An **agentic AI system** that acts as a self-serve sales operations analyst:

```
"How did we do last month?"        →  Full performance breakdown in 8 seconds
"Which rep is underperforming?"    →  Jordan Blake, 16.5% win rate, with coaching recommendation
"Which accounts are at risk?"      →  23 accounts, $1.8M ARR at risk, prioritized by urgency
"Generate my weekly report"        →  Complete executive report in 11 seconds (86% data accuracy)
```

No SQL. No dashboards. No waiting for an analyst. Just ask.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────┐
│   USER ASKS QUESTION (English)       │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   AI ROUTER (OpenAI Function Calling)│
│   LLM decides which tool to use      │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   TOOL EXECUTION (Python / Pandas)   │
│   • Text-to-SQL (NL → SQL → data)   │
│   • Trend analysis + Z-score         │
│   • At-risk detection                │
│   • Period comparison                │
│   • Auto report chain                │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   RAG CONTEXT LAYER                  │
│   20 business rules embedded as      │
│   vectors, retrieved by relevance    │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   LLM INTERPRETATION                 │
│   Executive-ready narrative with     │
│   exact numbers + recommended actions│
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   HALLUCINATION GUARD                │
│   Every number cross-checked against │
│   source data (5% tolerance)         │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│   OUTPUT                             │
│   Insight + Accuracy Score + Time    │
└──────────────────────────────────────┘
```

---

## 🧠 AI Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| **Agentic AI (Function Calling)** | LLM selects the right tool from 5 options based on question intent |
| **Multi-Tool Chaining** | Complex questions trigger multiple tools sequentially — LLM decides when it has enough data |
| **Text-to-SQL** | Natural language → SQLite query → executed on Pandas DataFrames (3-attempt recovery) |
| **RAG (Retrieval Augmented Generation)** | 20 business rules embedded via OpenAI `text-embedding-3-small`, retrieved by cosine similarity |
| **Hallucination Detection** | Every number in LLM output cross-validated against source data with 5% tolerance |
| **Process Automation** | Full weekly sales report generated in ~11 seconds vs ~5 hours manually |

---

## 📊 Dataset

Synthetic B2B SaaS sales data modeled after a Salesforce-style CRM:

| Table | Records | Key Fields |
|-------|---------|------------|
| Pipeline Deals | 800 | Deal value, stage, rep, close date, loss reason |
| Account Metrics | 150 | ARR, health score, NPS, renewal date, industry |
| Monthly Aggregates | 13 | Gross bookings, win rate, deal count |
| Rep Performance | 8 | Quota attainment, win rate, bookings |

**Built-in patterns the AI discovers:**
- West region underperformance (team transitions)
- Q3 seasonal dip (Aug-Sep win rate drops to 27-31%)
- Jordan Blake: 16.5% win rate (lowest, needs coaching)
- Renewals close at higher rates than new logos
- "Price too high" as #1 loss reason (25%+ of losses)
- 23 at-risk accounts representing $1.86M ARR

---

## 🔧 Tools the Agent Can Use

| Tool | Function | Example Trigger |
|------|----------|-----------------|
| `query_data()` | Text-to-SQL with 3-attempt recovery | "Show me all deals in the West region" |
| `analyze_trends()` | Z-score anomaly detection | "What is our win rate trend?" |
| `find_at_risk()` | Slipping deals + renewal risk | "Which accounts are at risk?" |
| `compare_periods()` | Side-by-side month comparison | "Compare June vs September" |
| `generate_weekly_report()` | Chains all tools → full report | "Generate my weekly report" |

---

## 🚀 Quick Start

### Option 1: Streamlit App (Recommended)

```bash
# Clone the repo
git clone https://github.com/PayalGore/saas-sales-intelligence-agent.git
cd saas-sales-intelligence-agent

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then enter your OpenAI API key in the sidebar and start asking questions.

### Option 2: Google Colab Notebook

1. Open `notebook/kaseya_saas_sales_intelligence_agent.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Add your OpenAI API key in Colab Secrets (🔑 icon)
3. Run all cells
4. Start asking questions in Cell 7

---

## 📁 Project Structure

```
saas-sales-intelligence-agent/
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
├── notebook/
│   └── kaseya_saas_sales_intelligence_agent.ipynb  # Original Colab notebook
├── data/
│   └── saas_sales_intelligence_data.xlsx           # Generated dataset (5 sheets)
├── docs/
│   └── architecture.mermaid        # Architecture diagram
└── assets/
    └── demo_screenshot.png         # App screenshot
```

---

## 💰 ROI Estimate

| Metric | Value |
|--------|-------|
| Manual report time | ~5 hours/week |
| AI report time | ~11 seconds |
| API cost per report | ~$0.03 |
| Annual savings per analyst | ~$14,298 |
| For a 5-person team | ~$71,490/year |

---

## 🛡️ Production Considerations

This prototype demonstrates the architecture. For production deployment:

- **Data layer:** Replace Pandas with Snowflake/BigQuery connector
- **Authentication:** Add SSO/OAuth for enterprise access
- **Caching:** Redis for repeated queries and embeddings
- **Monitoring:** Log all queries, tool calls, and accuracy scores
- **Security:** API keys in environment variables, not client-side
- **Scaling:** Deploy on AWS/GCP with load balancing

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit, Plotly |
| AI/LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Data Processing | Pandas, NumPy, pandasql |
| SQL Execution | pandasql (SQLite engine) |
| Visualization | Plotly Express |

---

## 🎯 Why I Built This

I built this to prototype what an **internal AI innovation tool for sales operations** looks like at a B2B SaaS company.

In many organizations, the gap between data warehouses (Snowflake, Redshift) and business users (Excel, email) is still bridged by manual analyst work — pulling data, writing commentary, building reports every week.

This agent demonstrates that with the right architecture — agentic function calling, grounded RAG context, and hallucination validation — AI can handle the heavy lifting while keeping humans in the loop for judgment and action.

The same pattern applies to finance ops, customer success, and any function where recurring reports and ad-hoc data questions consume analyst hours.

---

## 📬 Contact

**Payal Gore**
- [LinkedIn](https://www.linkedin.com/in/payalgore)
- [GitHub](https://github.com/PayalGore)
- Email: payal.gore6433@gmail.com

---

*Built with OpenAI GPT-4o-mini • March 2026*
