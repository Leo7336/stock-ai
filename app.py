import json
import time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="5年周期左侧排雷分析器", layout="wide")
st.title("📊 多市场股票批量排雷与周期分析 (最多10只)")

with st.sidebar:
    st.header("⚙️ 免费 Gemini API 配置")
    api_key = st.text_input("粘贴你的 Gemini API Key (AQ... 或 AIza...)", type="password")
    model_name = st.text_input("模型名称", value="gemini-3.8-flash")
    st.caption("代码格式：美股 GOOGL | 港股 0981.HK | A股 601985.SS / 002430.SZ | 新加坡 ME8U.SI | 马来 1023.KL | 印尼 BBRI.JK")

default_tickers = "GOOGL, MSFT, NVDA, TSM, 0981.HK, 601985.SS, 9988.HK, ME8U.SI"
raw_input = st.text_area("输入股票代码（用逗号或换行分隔，最多10只）：", value=default_tickers, height=70)

target_input = st.text_input(
    "可选：填入对应股票的“5年安全边际目标价”（按顺序用逗号分隔，留空则不计算差距，如：210, 400, 100, 260, , , , 1.50）",
    value=""
)

def normalize_div_yield(div_yield):
    if div_yield is None:
        return None
    if div_yield > 100:
        return round(div_yield / 100.0, 2)
    if div_yield > 15:
        return round(div_yield, 2)
    if div_yield <= 0.25:
        return round(div_yield * 100.0, 2)
    return round(div_yield, 2)

@st.cache_data(ttl=3600)
def analyze_single_ticker(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        pe_ttm = info.get("trailingPE")
        pe_fwd = info.get("forwardPE")
        pb = info.get("priceToBook")
        div_yield = info.get("dividendYield")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        free_cashflow = info.get("freeCashflow")
        operating_cashflow = info.get("operatingCashflow")
        revenue_growth = info.get("revenueGrowth")
        gross_margins = info.get("grossMargins")

        q_bs = stock.quarterly_balance_sheet
        red_flags = []

        if operating_cashflow is not None and operating_cashflow < 0:
            red_flags.append("经营性现金流为负")
        if free_cashflow is not None and free_cashflow < 0:
            red_flags.append("自由现金流为负（注意扩产开支或造血能力）")

        sector = info.get("sector", "")
        if debt_to_equity and debt_to_equity > 150 and "Financial" not in sector and "Real Estate" not in sector:
            red_flags.append(f"有息负债偏高 (D/E: {debt_to_equity:.1f}%)")

        if q_bs is not None and not q_bs.empty and "Inventory" in q_bs.index:
            inv_series = q_bs.loc["Inventory"].dropna()
            if len(inv_series) >= 2 and inv_series.iloc[1] > 0:
                inv_chg = (inv_series.iloc[0] - inv_series.iloc[1]) / inv_series.iloc[1]
                if inv_chg > 0.25:
                    red_flags.append(f"单季存货环比激增 {inv_chg*100:.1f}%")

        return {
            "代码": ticker,
            "名称": info.get("shortName", ticker),
            "币种": info.get("currency", ""),
            "现价": price,
            "硬规则排雷": " | ".join(red_flags) if red_flags else "未触发基础红灯",
            "股息率%": normalize_div_yield(div_yield),
            "TTM_PE": round(pe_ttm, 2) if pe_ttm else None,
            "Fwd_PE": round(pe_fwd, 2) if pe_fwd else None,
            "PB": round(pb, 2) if pb else None,
            "ROE%": round(roe * 100, 2) if roe else None,
            "营收增速%": round(revenue_growth * 100, 2) if revenue_growth else None,
            "毛利率%": round(gross_margins * 100, 2) if gross_margins else None,
            "52周低点": info.get("fiftyTwoWeekLow"),
            "52周高点": info.get("fiftyTwoWeekHigh"),
            "行业": info.get("industry", "")
        }
    except Exception as e:
        return {"代码": ticker, "硬规则排雷": f"抓取异常: {e}"}

def run_llm_single_analysis(stock_item: dict, api_key: str, model: str, max_retries: int = 3) -> dict:
    client = genai.Client(api_key=api_key)

    system_prompt = """
你是一名擅长“5年完整产业周期 + 左侧逆向安全边际”的跨市场股票研究员。
用户会提供单只股票的最新估值与基础财务排雷摘要（JSON格式）。
请你输出结构化 JSON 分析，严格返回如下格式：
{
  "ticker": "股票代码",
  "valuation_status": "低估 / 合理偏低 / 合理 / 偏高",
  "cycle_stage": "周期位置（如：景气高位见顶 / 下行去库 / 底部磨底 / 稳健非周期）",
  "risk_level": "低风险 / 中等观察 / 高警惕",
  "core_risk_summary": "一句话点出财报或行业最大雷点（如竞争预期突变、重资产折旧、库存积压等）",
  "five_year_logic": "5年维度的核心护城河与左侧安全边际观察要点（2-3句话）"
}
注意：重资产晶圆厂（如中芯国际、意法半导体）重点看PB而非PE；银行和REITs重点看股息率与PB。
"""
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=json.dumps(stock_item, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            if isinstance(data, dict) and "results" in data and data["results"]:
                return data["results"][0]
            return data
        except Exception as e:
            last_err = e
            err_str = str(e)
            if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise last_err
        
if st.button("🚀 开始批量分析", type="primary"):
    tickers = [t.strip() for t in raw_input.replace("\n", ",").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:10]

    raw_targets = [x.strip() for x in target_input.split(",")]
    targets = []
    for i in range(len(tickers)):
        if i < len(raw_targets) and raw_targets[i]:
            try:
                targets.append(float(raw_targets[i]))
            except ValueError:
                targets.append(None)
        else:
            targets.append(None)

    if not tickers:
        st.warning("请先输入至少1只股票代码。")
    else:
        with st.spinner(f"正在并发抓取 {len(tickers)} 只股票的行情与财报..."):
            with ThreadPoolExecutor(max_workers=10) as executor:
                raw_results = list(executor.map(analyze_single_ticker, tickers))

        for idx, row in enumerate(raw_results):
            tgt = targets[idx]
            cur_price = row.get("现价")
            if tgt is not None:
                row["安全边际目标价"] = tgt
                if cur_price and tgt > 0:
                    # 负数表示现价还高于目标价多少%，正数表示现价已跌破目标价多少%
                    gap_pct = ((tgt - cur_price) / cur_price) * 100
                    row["距买点差距%"] = round(gap_pct, 1)
                else:
                    row["距买点差距%"] = None

        st.subheader("1️⃣ 客观估值与硬规则排雷总览")
        st.dataframe(pd.DataFrame(raw_results), use_container_width=True)

        if api_key:
            st.subheader("2️⃣ AI 周期位置与 5 年安全边际研判")
            progress_bar = st.progress(0)
            total_n = len(raw_results)

            for idx, item_data in enumerate(raw_results):
                t_code = item_data.get("代码", f"第{idx+1}只")
                with st.spinner(f"正在分析 {t_code} ({idx+1}/{total_n})..."):
                    try:
                        ai_item = run_llm_single_analysis(item_data, api_key, model_name)
                        with st.expander(
                            f"📌 {ai_item.get('ticker', t_code)} | 估值：{ai_item.get('valuation_status', '-')} | 周期：{ai_item.get('cycle_stage', '-')} | 风险：{ai_item.get('risk_level', '-')}",
                            expanded=True
                        ):
                            st.markdown(f"**⚠️ 核心风险/雷点**：{ai_item.get('core_risk_summary', '')}")
                            st.markdown(f"**🛡️ 5年周期与安全边际逻辑**：{ai_item.get('five_year_logic', '')}")
                    except Exception as e:
                        st.warning(f"⚠️ {t_code} 暂时遇到服务端拥挤（503/限流），本次跳过：{e}")

                progress_bar.progress((idx + 1) / total_n)
                if idx < total_n - 1:
                    time.sleep(1.0)
        else:
            st.info("💡 在左侧边栏粘贴你的 Gemini API Key 后，即可生成下方的 AI 深度排雷结论。")