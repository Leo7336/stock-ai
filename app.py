import time
import pandas as pd
import streamlit as st
import yfinance as yf
from google import genai
from openai import OpenAI

st.set_page_config(
    page_title="多市场股票批量排雷与周期分析",
    page_icon="📊",
    layout="wide",
)

DEFAULT_TICKERS = "GOOGL, MSFT, NVDA, TSM, 0981.HK, 601985.SS, 9988.HK, ME8U.SI"

if "tickers_input" not in st.session_state:
    st.session_state["tickers_input"] = DEFAULT_TICKERS
if "search_candidates" not in st.session_state:
    st.session_state["search_candidates"] = []
if "brief_df" not in st.session_state:
    st.session_state["brief_df"] = None
if "metrics_list" not in st.session_state:
    st.session_state["metrics_list"] = []
if "ai_reports" not in st.session_state:
    st.session_state["ai_reports"] = []


# ---------------------------------------------------------
# 1. 侧边栏：多服务商与多模型配置
# ---------------------------------------------------------
with st.sidebar:
    st.header("AI 模型与 API 配置")

    provider = st.selectbox(
        "选择模型服务商",
        options=["Gemini", "DeepSeek"],
        index=0,
    )

    if provider == "Gemini":
        api_key = st.text_input(
            "粘贴你的 Gemini API Key",
            type="password",
            placeholder="AIza... 或 AQ...",
        )
        gemini_options = [
            "gemini-3.8-flash",
            "gemini-4-hydrogen",
            "自定义输入...",
        ]
        selected_gemini = st.selectbox(
            "选择 Gemini 模型",
            options=gemini_options,
            index=0,
        )
        if selected_gemini == "自定义输入...":
            model_name = st.text_input(
                "输入自定义 Gemini 模型名称",
                value="gemini-4-hydrogen",
            )
        else:
            model_name = selected_gemini
        st.caption("Gemini 适合免费配额轮换与快速批量分析。")

    else:
        api_key = st.text_input(
            "粘贴你的 DeepSeek API Key",
            type="password",
            placeholder="sk-...",
        )
        deepseek_options = [
            "deepseek-chat",
            "deepseek-reasoner",
            "自定义输入...",
        ]
        selected_deepseek = st.selectbox(
            "选择 DeepSeek 模型",
            options=deepseek_options,
            index=0,
        )
        if selected_deepseek == "自定义输入...":
            model_name = st.text_input(
                "输入自定义 DeepSeek 模型名称",
                value="deepseek-chat",
            )
        else:
            model_name = selected_deepseek
        st.caption("DeepSeek 走 OpenAI 兼容接口，需确保账户有可用 API 额度。")

    st.divider()
    st.caption(
        "代码格式：美股 GOOGL | 港股 0981.HK | A股 601985.SS / 002430.SZ | "
        "新加坡 ME8U.SI | 马来 1023.KL | 印尼 BBRI.JK"
    )


# ---------------------------------------------------------
# 2. 关键字搜公司代码 & 市场行情拉取
# ---------------------------------------------------------
def search_companies_by_keyword(keyword: str, max_results: int = 8) -> list[dict]:
    kw = (keyword or "").strip()
    if not kw:
        return []
    res = yf.Search(query=kw, max_results=max_results)
    quotes = getattr(res, "quotes", None) or []
    items = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        name = q.get("shortname") or q.get("longname") or ""
        exch = q.get("exchDisp") or q.get("exchange") or ""
        qtype = q.get("quoteType") or ""
        items.append({
            "代码": symbol,
            "名称": name,
            "市场/交易所": exch,
            "类型": qtype,
        })
    return items


def parse_tickers(raw_text: str, max_count: int = 10) -> list[str]:
    parts = [t.strip().upper() for t in raw_text.replace("\n", ",").split(",")]
    cleaned = []
    for p in parts:
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned[:max_count]


def append_tickers_to_input(new_symbols: list[str]) -> None:
    current_list = parse_tickers(st.session_state.get("tickers_input", ""), max_count=20)
    for sym in new_symbols:
        s = sym.strip().upper()
        if s and s not in current_list:
            current_list.append(s)
    st.session_state["tickers_input"] = ", ".join(current_list[:10])


def parse_target_prices(raw_targets: str, count: int) -> list[float | None]:
    if not raw_targets or not raw_targets.strip():
        return [None] * count
    tokens = [x.strip() for x in raw_targets.split(",")]
    result: list[float | None] = []
    for i in range(count):
        if i >= len(tokens) or tokens[i] == "":
            result.append(None)
        else:
            try:
                result.append(float(tokens[i]))
            except ValueError:
                result.append(None)
    return result


def fetch_stock_metrics(ticker: str, target_price: float | None) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    hist = stock.history(period="1y")

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if current_price is None and not hist.empty:
        current_price = float(hist["Close"].iloc[-1])

    currency = info.get("currency") or ""
    pe_ttm = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    debt_to_equity = info.get("debtToEquity")
    free_cashflow = info.get("freeCashflow")
    market_cap = info.get("marketCap")

    high_52w = info.get("fiftyTwoWeekHigh")
    if high_52w is None and not hist.empty:
        high_52w = float(hist["High"].max())

    drawdown_from_high_pct = None
    if current_price is not None and high_52w and high_52w > 0:
        drawdown_from_high_pct = round(
            (float(current_price) - float(high_52w)) / float(high_52w) * 100, 2
        )

    margin_of_safety_pct = None
    if current_price is not None and target_price and target_price > 0:
        margin_of_safety_pct = round(
            (float(target_price) - float(current_price)) / float(target_price) * 100, 2
        )

    red_flags = []
    if pe_ttm is not None and pe_ttm < 0:
        red_flags.append("TTM PE为负(亏损)")
    if debt_to_equity is not None and debt_to_equity > 200:
        red_flags.append(f"负债权益比偏高({debt_to_equity:.0f}%)")
    if free_cashflow is not None and free_cashflow < 0:
        red_flags.append("自由现金流为负")
    if roe is not None and roe < 0:
        red_flags.append("ROE为负")

    return {
        "股票代码": ticker,
        "名称": info.get("shortName") or info.get("longName") or ticker,
        "币种": currency,
        "现价": round(float(current_price), 3) if current_price is not None else None,
        "5年安全边际目标价": target_price,
        "距目标价空间(%)": margin_of_safety_pct,
        "52周高点回撤(%)": drawdown_from_high_pct,
        "PE(TTM)": round(float(pe_ttm), 2) if pe_ttm is not None else None,
        "PB": round(float(pb), 2) if pb is not None else None,
        "ROE(%)": round(float(roe) * 100, 2) if roe is not None else None,
        "负债权益比": round(float(debt_to_equity), 1) if debt_to_equity is not None else None,
        "自由现金流": free_cashflow,
        "市值": market_cap,
        "硬规则排雷": "；".join(red_flags) if red_flags else "无明显硬规则红灯",
    }


def build_prompt(metrics: dict) -> str:
    return f"""
请基于以下股票客观快照，给出简洁、克制、偏重排雷与周期定位的分析（约180~260字，分三点：1. 估值与安全边际；2. 财务与经营排雷；3. 周期与跟踪要点）：

{metrics}
""".strip()


# ---------------------------------------------------------
# 3. 统一 AI 调用（Gemini / DeepSeek）
# ---------------------------------------------------------
def ask_ai_analysis(provider: str, api_key: str, model_name: str, prompt: str) -> str:
    clean_key = (api_key or "").strip()
    clean_model = (model_name or "").strip()

    if not clean_key:
        raise ValueError(f"请先在左侧边栏填入 {provider} 的 API Key。")
    if not clean_model:
        raise ValueError("模型名称不能为空。")

    if provider == "Gemini":
        client = genai.Client(api_key=clean_key)
        response = client.models.generate_content(
            model=clean_model,
            contents=prompt,
        )
        return (response.text or "").strip()

    if provider == "DeepSeek":
        client = OpenAI(
            api_key=clean_key,
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=clean_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位严谨的多市场股票估值与排雷分析师，回答客观、简练、不夸大。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    raise ValueError(f"不支持的服务商：{provider}")


# ---------------------------------------------------------
# 4. 主界面：关键字查代码 + 先出财务简报 + 后跑 AI 分析
# ---------------------------------------------------------
st.title("📊 多市场股票批量排雷与周期分析（最多10只）")

with st.expander("🔍 用关键字搜公司代码（支持中/英文公司名或关键词）", expanded=False):
    col_kw, col_btn = st.columns([4, 1])
    with col_kw:
        keyword = st.text_input(
            "输入公司名或关键字",
            placeholder="例如：腾讯、Micron、Mapletree、Tenaga、中芯",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("搜索代码", use_container_width=True)

    if do_search:
        if not keyword.strip():
            st.warning("请先输入关键字。")
        else:
            with st.spinner("正在搜索匹配的公司代码..."):
                try:
                    st.session_state["search_candidates"] = search_companies_by_keyword(keyword)
                except Exception as e:
                    st.error(f"搜索失败：{e}")

    candidates = st.