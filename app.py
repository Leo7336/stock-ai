import json
import time
import urllib.parse
import urllib.request
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

ALIAS_HINTS = {
    "micron": [("MU", "Micron Technology, Inc.", "NASDAQ", "EQUITY")],
    "美光": [("MU", "Micron Technology, Inc.", "NASDAQ", "EQUITY")],
    "腾讯": [("0700.HK", "Tencent Holdings Ltd.", "Hong Kong", "EQUITY"), ("TCEHY", "Tencent Holdings Ltd.", "OTC", "EQUITY")],
    "阿里": [("9988.HK", "Alibaba Group Holding Ltd.", "Hong Kong", "EQUITY"), ("BABA", "Alibaba Group Holding Ltd.", "NYSE", "EQUITY")],
    "中芯": [("0981.HK", "SMIC", "Hong Kong", "EQUITY"), ("688981.SS", "SMIC", "Shanghai", "EQUITY")],
    "台积电": [("TSM", "Taiwan Semiconductor Manufacturing", "NYSE", "EQUITY"), ("2330.TW", "TSMC", "Taiwan", "EQUITY")],
    "英伟达": [("NVDA", "NVIDIA Corporation", "NASDAQ", "EQUITY")],
    "谷歌": [("GOOGL", "Alphabet Inc.", "NASDAQ", "EQUITY")],
    "微软": [("MSFT", "Microsoft Corporation", "NASDAQ", "EQUITY")],
    "意法": [("STM", "STMicroelectronics N.V.", "NYSE", "EQUITY")],
}

if "tickers_input" not in st.session_state:
    st.session_state["tickers_input"] = DEFAULT_TICKERS
if "search_candidates" not in st.session_state:
    st.session_state["search_candidates"] = []
if "search_status" not in st.session_state:
    st.session_state["search_status"] = ""
if "brief_df" not in st.session_state:
    st.session_state["brief_df"] = None
if "metrics_list" not in st.session_state:
    st.session_state["metrics_list"] = []
if "ai_reports" not in st.session_state:
    st.session_state["ai_reports"] = []


# ---------------------------------------------------------
# 1. 侧边栏：AI 模型配置 + 可选备选行情 API（Finnhub）
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
    st.subheader("可选：备选行情 API")
    data_source_mode = st.selectbox(
        "行情抓取策略",
        options=["yfinance 优先 + Finnhub 补缺", "仅用 yfinance"],
        index=0,
    )
    finnhub_key = st.text_input(
        "Finnhub API Key（可选，主要补美股报价）",
        type="password",
        placeholder="留空则仅用 yfinance",
    )

    st.divider()
    st.caption(
        "代码格式：美股 GOOGL | 港股 0981.HK | A股 601985.SS / 002430.SZ | "
        "新加坡 ME8U.SI | 马来 1023.KL | 印尼 BBRI.JK"
    )


# ---------------------------------------------------------
# 2. 关键字搜公司代码（yf.Search + HTTP 备选 + 别名兜底）
# ---------------------------------------------------------
def _http_yahoo_search(kw: str, max_results: int = 8) -> list[dict]:
    url = (
        "https://query2.finance.yahoo.com/v1/finance/search?q="
        + urllib.parse.quote(kw)
        + f"&quotesCount={max_results}&newsCount=0"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    quotes = data.get("quotes") or []
    items = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        items.append({
            "代码": symbol,
            "名称": q.get("shortname") or q.get("longname") or "",
            "市场/交易所": q.get("exchDisp") or q.get("exchange") or "",
            "类型": q.get("quoteType") or "",
        })
    return items


def search_companies_by_keyword(keyword: str, max_results: int = 8) -> tuple[list[dict], str]:
    kw = (keyword or "").strip()
    if not kw:
        return [], "请输入关键字。"

    items: list[dict] = []
    seen: set[str] = set()
    sources_used: list[str] = []

    def add_rows(rows: list[dict], src_label: str) -> None:
        added = False
        for r in rows:
            sym = (r.get("代码") or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                items.append({
                    "代码": sym,
                    "名称": r.get("名称") or "",
                    "市场/交易所": r.get("市场/交易所") or "",
                    "类型": r.get("类型") or "",
                })
                added = True
        if added and src_label not in sources_used:
            sources_used.append(src_label)

    try:
        res = yf.Search(query=kw, max_results=max_results)
        quotes = getattr(res, "quotes", None) or []
        yf_rows = [
            {
                "代码": q.get("symbol"),
                "名称": q.get("shortname") or q.get("longname") or "",
                "市场/交易所": q.get("exchDisp") or q.get("exchange") or "",
                "类型": q.get("quoteType") or "",
            }
            for q in quotes
            if q.get("symbol")
        ]
        add_rows(yf_rows, "yf.Search")
    except Exception:
        pass

    if len(items) < 3:
        try:
            http_rows = _http_yahoo_search(kw, max_results=max_results)
            add_rows(http_rows, "Yahoo直连")
        except Exception:
            pass

    kw_lower = kw.lower()
    alias_rows = []
    for alias_key, tuple_list in ALIAS_HINTS.items():
        if alias_key in kw_lower or kw_lower in alias_key:
            for sym, nm, exch, qt in tuple_list:
                alias_rows.append({
                    "代码": sym,
                    "名称": nm,
                    "市场/交易所": exch,
                    "类型": qt,
                })
    if alias_rows:
        add_rows(alias_rows, "常用映射")

    final_items = items[:max_results]
    if not final_items:
        return [], f"未搜到“{kw}”的匹配结果，可尝试英文名或直接在下方输入代码。"

    src_text = " / ".join(sources_used) if sources_used else "默认"
    return final_items, f"已找到 {len(final_items)} 条结果（来源：{src_text}）"


# ---------------------------------------------------------
# 3. 股票代码解析、多源行情指标与排雷计算
# ---------------------------------------------------------
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


def _fetch_finnhub_quote(ticker: str, token: str) -> dict:
    """用 Finnhub /quote 拉取补充报价（主要用于无点号的美股 ticker）。"""
    clean_token = (token or "").strip()
    if not clean_token or "." in ticker:
        return {}
    url = (
        f"https://finnhub.io/api/v1/quote?symbol={urllib.parse.quote(ticker)}"
        f"&token={urllib.parse.quote(clean_token)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict) or not data.get("c"):
        return {}
    return {
        "current_price": data.get("c"),
        "high_52w": data.get("h"),
        "low_52w": data.get("l"),
    }


def fetch_stock_metrics(ticker: str, use_finnhub: bool = False, finnhub_token: str = "") -> dict:
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
    low_52w = info.get("fiftyTwoWeekLow")
    if not hist.empty:
        if high_52w is None:
            high_52w = float(hist["High"].max())
        if low_52w is None:
            low_52w = float(hist["Low"].min())

    source_tag = "yfinance"
    if use_finnhub and finnhub_token and (current_price is None or high_52w is None or low_52w is None):
        try:
            fh = _fetch_finnhub_quote(ticker, finnhub_token)
            if fh:
                if current_price is None and fh.get("current_price") is not None:
                    current_price = float(fh["current_price"])
                if high_52w is None and fh.get("high_52w") is not None:
                    high_52w = float(fh["high_52w"])
                if low_52w is None and fh.get("low_52w") is not None:
                    low_52w = float(fh["low_52w"])
                source_tag = "yfinance+Finnhub"
        except Exception:
            pass

    drawdown_from_high_pct = None
    if current_price is not None and high_52w and high_52w > 0:
        drawdown_from_high_pct = round(
            (float(current_price) - float(high_52w)) / float(high_52w) * 100, 2
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
        "52周最高": round(float(high_52w), 3) if high_52w is not None else None,
        "52周最低": round(float(low_52w), 3) if low_52w is not None else None,
        "52周高点回撤(%)": drawdown_from_high_pct,
        "PE(TTM)": round(float(pe_ttm), 2) if pe_ttm is not None else None,
        "PB": round(float(pb), 2) if pb is not None else None,
        "ROE(%)": round(float(roe) * 100, 2) if roe is not None else None,
        "负债权益比": round(float(debt_to_equity), 1) if debt_to_equity is not None else None,
        "自由现金流": free_cashflow,
        "市值": market_cap,
        "数据源": source_tag,
        "硬规则排雷": "；".join(red_flags) if red_flags else "无明显硬规则红灯",
    }


def build_prompt(metrics: dict) -> str:
    return f"""
请基于以下股票客观快照，给出简洁、克制、偏重排雷与周期定位的分析（约180~260字，分三点：1. 估值与区间位置；2. 财务与经营排雷；3. 周期与跟踪要点）：

{metrics}
""".strip()


# ---------------------------------------------------------
# 4. 统一 AI 调用（Gemini / DeepSeek）
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
# 5. 主界面：关键字查代码 + 先出财务简报 + 后跑 AI 分析
# ---------------------------------------------------------
st.title("📊 多市场股票批量排雷与周期分析（最多10只）")

with st.expander("🔍 用关键字搜公司代码（支持中/英文公司名或关键词）", expanded=True):
    col_kw, col_btn = st.columns([4, 1])
    with col_kw:
        keyword = st.text_input(
            "输入公司名或关键字",
            placeholder="例如：micron、腾讯、Mapletree、中芯",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("搜索代码", use_container_width=True)

    if do_search:
        with st.spinner("正在搜索匹配的公司代码..."):
            rows, status_msg = search_companies_by_keyword(keyword, max_results=8)
            st.session_state["search_candidates"] = rows
            st.session_state["search_status"] = status_msg

    if st.session_state.get("search_status"):
        st.caption(st.session_state["search_status"])

    candidates = st.session_state.get("search_candidates", [])
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
        option_labels = [
            f"{c['代码']} | {c['名称']} ({c['市场/交易所']})" for c in candidates
        ]
        label_to_symbol = {
            option_labels[i]: candidates[i]["代码"] for i in range(len(candidates))
        }
        picked_labels = st.multiselect(
            "勾选要加入分析的代码：",
            options=option_labels,
        )
        if st.button("➕ 加入下方股票代码框"):
            picked_symbols = [label_to_symbol[lbl] for lbl in picked_labels]
            if not picked_symbols:
                st.warning("请先勾选至少 1 个搜索结果。")
            else:
                append_tickers_to_input(picked_symbols)
                st.rerun()

raw_tickers = st.text_area(
    "输入股票代码（用逗号或换行分隔，最多10只）：",
    key="tickers_input",
    height=90,
)

start_btn = st.button("🚀 先出财务简报并开始批量分析", type="primary")

brief_section = st.container()
ai_section = st.container()

if start_btn:
    tickers = parse_tickers(raw_tickers, max_count=10)
    if not tickers:
        st.error("请至少输入 1 只有效的股票代码。")
        st.stop()

    use_fh = data_source_mode == "yfinance 优先 + Finnhub 补缺"
    overview_rows = []
    valid_metrics = []

    with brief_section:
        st.subheader("1 客观财务简报与重要指标")
        brief_placeholder = st.empty()
        brief_progress = st.progress(0, text="正在拉取财务简报与核心指标...")

        for idx, ticker in enumerate(tickers):
            brief_progress.progress(
                idx / len(tickers),
                text=f"正在拉取指标 ({idx + 1}/{len(tickers)})：{ticker}",
            )
            try:
                metrics = fetch_stock_metrics(
                    ticker,
                    use_finnhub=use_fh,
                    finnhub_token=finnhub_key,
                )
                overview_rows.append(metrics)
                valid_metrics.append(metrics)
            except Exception as e:
                overview_rows.append({
                    "股票代码": ticker,
                    "硬规则排雷": f"行情/财务拉取异常：{e}",
                })

            brief_placeholder.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

        brief_progress.progress(1.0, text="财务简报已就绪。")

    st.session_state["brief_df"] = pd.DataFrame(overview_rows)
    st.session_state["metrics_list"] = valid_metrics
    st.session_state["ai_reports"] = []

    with ai_section:
        st.subheader(f"2 AI 深度解读（{provider} · {model_name}）")
        if not api_key or not api_key.strip():
            st.info(
                f"上方财务简报已生成。若需生成 AI 解读，请在左侧边栏填入 {provider} 的 API Key 后重新点击按钮。"
            )
        elif not valid_metrics:
            st.warning("没有可用的股票财务快照，跳过 AI 分析。")
        else:
            ai_progress = st.progress(0, text="财务简报已出，开始后台逐只 AI 分析...")
            reports = []

            for idx, metrics in enumerate(valid_metrics):
                ticker = metrics["股票代码"]
                name = metrics.get("名称", ticker)
                ai_progress.progress(
                    idx / len(valid_metrics),
                    text=f"正在 AI 分析 ({idx + 1}/{len(valid_metrics)})：{ticker} [{provider} / {model_name}]",
                )
                try:
                    prompt = build_prompt(metrics)
                    analysis_text = ask_ai_analysis(
                        provider=provider,
                        api_key=api_key,
                        model_name=model_name,
                        prompt=prompt,
                    )
                    reports.append((ticker, name, analysis_text))
                    with st.expander(f"{ticker} — {name}", expanded=True):
                        st.markdown(analysis_text)
                except Exception as e:
                    st.warning(f"{ticker} AI 分析跳过：{e}")

                if idx < len(valid_metrics) - 1:
                    time.sleep(1.0)

            ai_progress.progress(1.0, text="AI 批量分析完成。")
            st.session_state["ai_reports"] = reports

else:
    if st.session_state.get("brief_df") is not None:
        with brief_section:
            st.subheader("1 客观财务简报与重要指标")
            st.dataframe(st.session_state["brief_df"], use_container_width=True)

    if st.session_state.get("ai_reports"):
        with ai_section:
            st.subheader(f"2 AI 深度解读（{provider} · {model_name}）")
            for ticker, name, text in st.session_state["ai_reports"]:
                with st.expander(f"{ticker} — {name}", expanded=True):
                    st.markdown(text)