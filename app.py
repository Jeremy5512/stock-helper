"""FundaScope 基本面透視：輸入代號，自動抓台股（FinMind）或美股（yfinance）財報，算出 11 項指標並給學習重點。
注意：本程式未在實際網路環境測試過，欄位名稱若與資料源不符，請看畫面下方「原始數據」並回報。"""
import datetime as dt
import os
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from urllib.parse import quote

FM = "https://api.finmindtrade.com/api/v4/data"
TODAY = dt.date.today()


# ---------- 共用小工具 ----------
def fm_token():
    try:
        return st.secrets["FINMIND_TOKEN"]
    except Exception:
        return os.environ.get("FINMIND_TOKEN", "")


def fm(dataset, sid, start):
    headers = {"Authorization": f"Bearer {fm_token()}"} if fm_token() else {}
    r = requests.get(FM, params={"dataset": dataset, "data_id": sid, "start_date": start},
                     headers=headers, timeout=30)
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(f"{dataset}: {j.get('msg')}")
    return pd.DataFrame(j["data"])


def qseries(df, types):
    """從 FinMind 長表取出某科目的季度序列（依日期排序）。"""
    for t in types:
        s = df[df["type"] == t]
        if len(s):
            s = s.sort_values("date")
            return pd.Series(s["value"].astype(float).values, index=pd.to_datetime(s["date"]))
    return pd.Series(dtype=float)


def ttm(s, back=0):
    """最近四季加總；back=1 代表再往前推一年（去年同期）。"""
    end = len(s) - 4 * back
    if end < 4:
        return None
    return float(s.iloc[end - 4:end].sum())


def latest(s):
    return float(s.iloc[-1]) if len(s) else None


# ---------- 台股 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_tw(sid):
    s3 = (TODAY - dt.timedelta(days=1100)).isoformat()
    s1 = (TODAY - dt.timedelta(days=500)).isoformat()
    px = fm("TaiwanStockPrice", sid, (TODAY - dt.timedelta(days=1830)).isoformat())
    fs = fm("TaiwanStockFinancialStatements", sid, (TODAY - dt.timedelta(days=1830)).isoformat())
    bs = fm("TaiwanStockBalanceSheet", sid, s1)
    cf = fm("TaiwanStockCashFlowsStatement", sid, s3)
    try:
        dv = fm("TaiwanStockDividend", sid, s3)
    except Exception:
        dv = pd.DataFrame()
    if px.empty or fs.empty or bs.empty:
        raise RuntimeError("查無此代號的資料")
    pxs = px.sort_values("date")
    cl = pxs["close"].astype(float).values
    col = lambda n: pxs[n].astype(float).values if n in pxs else cl
    hist = pd.DataFrame({"date": pd.to_datetime(pxs["date"]).values, "open": col("open"), "high": col("max"),
                         "low": col("min"), "close": cl,
                         "volume": pxs["Trading_Volume"].astype(float).values if "Trading_Volume" in pxs else 0.0})

    rev = qseries(fs, ["Revenue"])
    gp = qseries(fs, ["GrossProfit"])
    ni = qseries(fs, ["EquityAttributableToOwnersOfParent", "IncomeAfterTaxes"])
    eps = qseries(fs, ["EPS"])
    trend = pd.DataFrame({"rev": rev, "gp": gp, "ni": ni, "eps": eps}).dropna(subset=["rev"]).tail(12)
    trend["label"] = [f"{i.year}Q{(i.month - 1) // 3 + 1}" for i in trend.index]
    trend = trend.reset_index(drop=True)

    def bsv(types):
        return latest(qseries(bs, types))

    equity = bsv(["EquityAttributableToOwnersOfParent", "Equity"])
    shares = (bsv(["OrdinaryShare"]) or 0) / 10 or None  # 面額 10 元

    # 現金流量表在台灣是「年初至今累計」，需換算成近四季
    ctypes = ["CashFlowsFromUsedInOperatingActivities", "NetCashInflowFromOperatingActivities"]
    cs = qseries(cf, ctypes)
    if cs.empty:
        alt = [t for t in cf["type"].unique() if "OperatingActivities" in t]
        cs = qseries(cf, alt[:1]) if alt else cs
    cfo = None
    if len(cs):
        d = {(i.year, i.month): v for i, v in cs.items()}
        y, m = cs.index[-1].year, cs.index[-1].month
        if m == 12:
            cfo = d[(y, 12)]
        elif (y - 1, 12) in d and (y - 1, m) in d:
            cfo = d[(y, m)] + d[(y - 1, 12)] - d[(y - 1, m)]

    div = None
    if not dv.empty:
        row = dv.sort_values("date").iloc[-1]
        div = sum(float(row.get(k) or 0) for k in ["CashEarningsDistribution", "CashStatutorySurplus"])

    return dict(
        name=sid, cur="元", ud=1e8, ul="億元", mk="tw", trend=trend, tfreq="季",
        price=float(hist["close"].iloc[-1]), hist=hist,
        eps=ttm(eps), bv=(equity / shares) if equity and shares else None, div=div,
        rev=ttm(rev), prev=ttm(rev, 1), gp=ttm(gp), ni=ttm(ni),
        eq=equity, ta=bsv(["TotalAssets"]), tl=bsv(["Liabilities", "TotalLiabilities"]),
        ca=bsv(["CurrentAssets"]), cl=bsv(["CurrentLiabilities"]), cfo=cfo,
        asof=str(rev.index[-1].date()) if len(rev) else "?",
    )


# ---------- 美股 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def load_us(sym):
    import yfinance as yf
    t = yf.Ticker(sym)
    inc, bs, cf = t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow
    if inc is None or inc.empty:
        raise RuntimeError("查無此代號的資料")

    def row(df, names):
        for n in names:
            if n in df.index:
                s = df.loc[n].dropna()
                return s.sort_index()
        return pd.Series(dtype=float)

    rev = row(inc, ["Total Revenue"])
    gp = row(inc, ["Gross Profit"])
    ni = row(inc, ["Net Income Common Stockholders", "Net Income"])
    eps = row(inc, ["Diluted EPS", "Basic EPS"])
    bsl = lambda names: latest(row(bs, names))
    equity = bsl(["Stockholders Equity", "Common Stock Equity"])
    shares = bsl(["Ordinary Shares Number", "Share Issued"])
    cfo_s = row(cf, ["Operating Cash Flow"])

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    price = None
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = shares or info.get("sharesOutstanding")
    hist = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    try:
        h = t.history(period="5y")
        hist = pd.DataFrame({"date": h.index.tz_localize(None), "open": h["Open"].astype(float).values,
                             "high": h["High"].astype(float).values, "low": h["Low"].astype(float).values,
                             "close": h["Close"].astype(float).values, "volume": h["Volume"].astype(float).values})
    except Exception:
        pass
    if price is None and len(hist):
        price = float(hist["close"].iloc[-1])

    div = None
    try:
        dv = t.dividends
        if len(dv):
            cut = pd.Timestamp.now(tz=dv.index.tz) - pd.Timedelta(days=365)
            div = float(dv[dv.index >= cut].sum())
    except Exception:
        pass

    prev = ttm(rev, 1)
    rev_t = ttm(rev)
    if prev is None:  # 季資料不足，改用年度資料算成長
        try:
            ann = row(t.income_stmt, ["Total Revenue"])
            if len(ann) >= 2:
                rev_t, prev = float(ann.iloc[-1]), float(ann.iloc[-2])
        except Exception:
            pass
    trend = pd.DataFrame()
    try:
        ia = t.income_stmt
        trend = pd.DataFrame({"rev": row(ia, ["Total Revenue"]), "gp": row(ia, ["Gross Profit"]),
                              "ni": row(ia, ["Net Income Common Stockholders", "Net Income"]),
                              "eps": row(ia, ["Diluted EPS", "Basic EPS"])}).dropna(subset=["rev"])
        trend["label"] = [f"FY{i.year}" for i in trend.index]
        trend = trend.reset_index(drop=True)
    except Exception:
        pass
    eps_t = ttm(eps)
    ni_t = ttm(ni)
    if eps_t is None and ni_t and shares:
        eps_t = ni_t / shares

    return dict(
        name=info.get("shortName", sym), cur="美元", ud=1e9, ul="十億美元", mk="us", trend=trend, tfreq="年",
        price=price, hist=hist, eps=eps_t, bv=(equity / shares) if equity and shares else None, div=div,
        rev=rev_t, prev=prev, gp=ttm(gp), ni=ni_t,
        eq=equity, ta=bsl(["Total Assets"]),
        tl=bsl(["Total Liabilities Net Minority Interest", "Total Liabilities"]),
        ca=bsl(["Current Assets"]), cl=bsl(["Current Liabilities"]), cfo=ttm(cfo_s),
        asof=str(rev.index[-1].date()) if len(rev) else "?",
    )


# ---------- 指標與解讀 ----------
def div_(a, b):
    return None if a is None or b in (None, 0) else a / b


def rate(v, good, ok, higher=True):
    if v is None:
        return None
    if higher:
        return "good" if v >= good else "ok" if v >= ok else "bad"
    return "good" if v <= good else "ok" if v <= ok else "bad"


def analyze(d):
    pg = 15 if d["mk"] == "tw" else 20
    pm = 1.5 if d["mk"] == "tw" else 3
    pe, pb = div_(d["price"], d["eps"]), div_(d["price"], d["bv"])
    yl, po = div_(d["div"], d["price"]), div_(d["div"], d["eps"])
    gr = div_(d["rev"], d["prev"])
    gr = None if gr is None else gr - 1
    gm, nm = div_(d["gp"], d["rev"]), div_(d["ni"], d["rev"])
    roe, dr = div_(d["ni"], d["eq"]), div_(d["tl"], d["ta"])
    cr, q = div_(d["ca"], d["cl"]), div_(d["cfo"], d["ni"])
    R = dict(pe=pe, pb=pb, yl=yl, po=po, gr=gr, gm=gm, nm=nm, roe=roe, dr=dr, cr=cr, q=q, pg=pg)
    pct = lambda v: "—" if v is None else f"{v*100:.1f}%"
    x = lambda v: "—" if v is None else f"{v:.2f} 倍"
    pe_lv = None if pe is None else ("bad" if pe <= 0 else rate(pe, pg, pg * 1.6, False))
    po_lv = None if po is None else ("bad" if po > 1 else "good" if 0.3 <= po <= 0.8 else "ok")
    q_lv = None if (q is None or (d["ni"] or 0) <= 0) else rate(q, 1, 0.7)
    M = [
        ("本益比 P/E", "—" if pe is None or pe <= 0 else x(pe), pe_lv, f"股價是每年賺的錢的幾倍，越低越便宜但也可能被看衰。參考約 {pg} 倍。"),
        ("股價淨值比 P/B", x(pb), rate(pb, pm, pm * 2, False), "股價是帳面價值的幾倍，重資產產業較常用。"),
        ("現金殖利率", pct(yl), rate(yl, .04, .02), "每年領到的股利佔股價比例，像股票的利息。"),
        ("配息率", pct(po), po_lv, "賺的錢發出去多少。太高難持續，太低代表留著再投資。"),
        ("營收成長率", pct(gr), rate(gr, .1, 0), "營收比去年同期成長多少。"),
        ("毛利率", pct(gm), rate(gm, .3, .15), "賣東西扣成本後剩幾成，需與同業比較。"),
        ("淨利率", pct(nm), rate(nm, .1, .04), "每 100 元營收最後落袋多少。"),
        ("ROE", pct(roe), rate(roe, .15, .08), "公司用股東的錢賺錢的效率，長期 15% 以上算優秀。"),
        ("負債比", pct(dr), rate(dr, .5, .65, False), "資產中有多少是借來的（金融業除外）。"),
        ("流動比率", x(cr), rate(cr, 1.5, 1), "短期還債能力，低於 1 可能吃緊。"),
        ("獲利現金含量", x(q), q_lv, "營業現金流 ÷ 淨利，長期低於 1 要留意獲利品質。"),
    ]
    return M, R


def learn(R, adj):
    o = []
    pe, pg = R["pe"], R["pg"]
    if pe is not None and pe > pg * 1.6:
        o.append(f"**好公司不等於好價格。** 本益比 {pe:.1f} 倍明顯高於參考的 {pg} 倍，市場已把大量成長預期算進股價。")
    elif pe is not None and 0 < pe <= pg:
        o.append("**本益比偏低要多問一句為什麼。** 可能被低估，也可能獲利即將衰退（價值陷阱）。")
    elif pe is not None and pe <= 0:
        o.append("**公司目前虧損，本益比不適用。** 改看營收成長、現金流與資金是否撐得住。")
    if R["roe"] and R["roe"] > .6 and R["pb"] and R["pb"] > 10:
        o.append("**ROE 與股價淨值比可能被庫藏股扭曲。** 大量買回股票會讓股東權益縮水。")
    elif R["dr"] and R["dr"] > .65:
        o.append("**負債比偏高。** 確認是正常營運還是靠借錢擴張，並搭配現金流判斷。")
    if R["yl"] is not None and R["yl"] < .01 and R["po"] is not None and R["po"] < .3:
        o.append("**殖利率與配息率都低。** 公司傾向把錢拿去再投資，不適合當領息股。")
    if R["po"] and R["po"] > 1:
        o.append("**配息超過獲利。** 長期難以維持，留意降息風險。")
    if R["q"] is not None and R["q"] < .7:
        o.append("**賺的錢沒完全收回來。** 可能是應收帳款、存貨堆積，或淨利含非現金的一次性收益。")
    elif R["q"] is not None and R["q"] >= 1:
        o.append("**獲利品質良好。** 營業現金流不低於淨利。")
    if R["gr"] is not None and R["gr"] < 0:
        o.append("**營收衰退。** 先弄清楚是景氣循環還是競爭力下滑。")
    if R["gm"] and R["nm"] and R["gm"] >= .3 and R["nm"] >= .1:
        o.append("**高毛利加高淨利，常是護城河的跡象。**")
    if adj:
        o.append("**你已扣除一次性損益**，ROE、淨利率、本益比是依扣除後的淨利重算。")
    o.append("**別只看一期。** 把近 3 到 5 年的趨勢拉出來，並和同業比較。")
    return o[:6]


# ---------- 畫面（深色科技風）----------
st.set_page_config(page_title="FundaScope｜基本面透視", page_icon="🔭", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:880px}
h1.t{font-size:1.6rem;margin:0;font-weight:800;letter-spacing:.18em;background:linear-gradient(90deg,#00E5FF,#7C4DFF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:#7f93b2;font-size:.85rem;margin:2px 0 14px}
.hero{background:linear-gradient(135deg,#0f1a2e,#0b1220);border:1px solid #1e3a5f;border-radius:16px;padding:18px 20px;box-shadow:0 0 24px rgba(0,229,255,.08)}
.hero .nm{color:#9fb3d1;font-size:.95rem}
.hero .px{font-size:3.3rem;font-weight:800;line-height:1.1;color:#fff;font-variant-numeric:tabular-nums}
.hero .px small{font-size:1rem;color:#7f93b2;margin-left:8px;font-weight:400}
.hero .chg{font-size:1.2rem;font-weight:700}
.hero .dt{color:#5f7596;font-size:.75rem;margin-top:4px}
.kp{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.kp div{background:#0f1a2e;border:1px solid #1b2c47;border-radius:12px;padding:10px 12px}
.kp span{display:block;color:#7f93b2;font-size:.72rem}
.kp b{font-size:1.2rem;color:#E6F1FF}
.sc{display:flex;align-items:center;gap:16px;background:#0f1a2e;border:1px solid #1b2c47;border-radius:14px;padding:14px 18px;margin:8px 0 14px}
.ring{width:84px;height:84px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex:none}
.ring i{width:66px;height:66px;border-radius:50%;background:#0f1a2e;display:flex;align-items:center;justify-content:center;font-style:normal;font-size:1.5rem;font-weight:800;color:#fff}
.sc .tx{color:#9fb3d1;font-size:.9rem;line-height:1.6}
.card{background:#0f1a2e;border:1px solid #1b2c47;border-left:3px solid var(--c);border-radius:12px;padding:12px 14px;margin-bottom:10px;min-height:124px}
.card .h{display:flex;justify-content:space-between;align-items:center;color:#9fb3d1;font-size:.85rem}
.card .v{font-size:1.6rem;font-weight:800;color:#fff;margin:2px 0}
.card .e{color:#7f93b2;font-size:.78rem;line-height:1.45}
.tag{font-size:.72rem;padding:1px 9px;border-radius:99px;border:1px solid var(--c);color:var(--c)}
.lp{background:#101a2c;border:1px solid #223653;border-radius:10px;padding:10px 14px;margin:8px 0;color:#d6e4f5;font-size:.92rem}
h3.s{color:#00E5FF;font-size:1rem;letter-spacing:.06em;margin:18px 0 6px}
@media(max-width:640px){.kp{grid-template-columns:repeat(2,1fr)}.hero .px{font-size:2.6rem}}
.cmp{width:100%;border-collapse:collapse;font-size:.88rem;color:#E6F1FF}
.cmp th,.cmp td{padding:8px 10px;border-bottom:1px solid #1b2c47;text-align:right;white-space:nowrap}
.cmp th:first-child,.cmp td:first-child{text-align:left;color:#9fb3d1}
.cmp th{color:#00E5FF}
@keyframes fi{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.t,.sub{animation:fi .8s ease both}
h1.t.c{font-size:2.5rem;text-align:center}
.sub.c{text-align:center;font-size:.95rem}
.hint{text-align:center;color:#5f7596;font-size:.85rem;margin-top:14px;animation:fi 1.2s .3s ease both}
div[data-testid="stTextInput"]{max-width:540px;margin:0 auto}
div[data-baseweb="input"]{border-radius:14px;border:1px solid #1e3a5f;box-shadow:0 0 20px rgba(0,229,255,.12)}
.hero{animation:fi .8s ease both}
.kp{animation:fi .8s .15s ease both}
div[data-testid="stRadio"]{animation:fi .8s .25s ease both}
div[data-testid="stPlotlyChart"]{animation:fi .9s .3s ease both}
.sc{animation:fi .8s .45s ease both}
h3.s{animation:fi .8s .55s ease both}
.card{animation:fi .8s .65s ease both}
.lp{animation:fi .8s .8s ease both}
</style>""", unsafe_allow_html=True)
head = st.empty()
sym = st.text_input("股票代號", placeholder="輸入股票代號，例如 2330 或 AAPL", label_visibility="collapsed").strip().upper()
landing = not sym
cls = " c" if landing else ""
hint = '<div class="hint">試試看：2330　2454　AAPL　NVDA</div>' if landing else ""
head.markdown(f'<h1 class="t{cls}">🔭 FUNDASCOPE</h1><div class="sub{cls}">基本面透視　·　輸入代號，看穿財報｜台股、美股｜僅供學習，不構成投資建議</div>', unsafe_allow_html=True)
if landing:
    st.markdown("<style>.block-container{padding-top:26vh !important}</style>" + hint, unsafe_allow_html=True)
adj_in = 0.0
if sym:
    with st.expander("進階：扣除一次性損益"):
        adj_in = st.number_input("金額單位同下方財報（台股：億元，美股：十億美元）", value=0.0, step=1.0)
sym2 = ""
if sym:
    with st.expander("⚖️ 比較第二檔股票（選填）"):
        sym2 = st.text_input("第二檔代號", key="sym2", placeholder="例如 2454 或 NVDA").strip().upper()


def site_links(sym, is_tw):
    if not sym:
        return [("Goodinfo 台灣股市資訊網", "https://goodinfo.tw/"), ("Yahoo 奇摩股市", "https://tw.stock.yahoo.com/"),
                ("公開資訊觀測站（官方財報）", "https://mops.twse.com.tw/"), ("Yahoo Finance", "https://finance.yahoo.com/"),
                ("SEC EDGAR（美股官方財報）", "https://www.sec.gov/edgar/search/"), ("Investing.com", "https://www.investing.com/")]
    q = quote(sym)
    if is_tw:
        return [("Goodinfo 台灣股市資訊網", f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={q}"),
                ("Yahoo 奇摩股市", f"https://tw.stock.yahoo.com/quote/{q}"),
                ("鉅亨網", f"https://www.cnyes.com/twstock/{q}"),
                ("公開資訊觀測站（官方財報）", "https://mops.twse.com.tw/"),
                ("TradingView", f"https://www.tradingview.com/symbols/TWSE-{q}/"),
                ("Investing.com 搜尋", f"https://www.investing.com/search/?q={q}")]
    return [("Yahoo Finance", f"https://finance.yahoo.com/quote/{q}"),
            ("Google Finance", f"https://www.google.com/finance?q={q}"),
            ("SEC EDGAR（官方財報）", f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={q}&type=10-&owner=include&count=40"),
            ("StockAnalysis 財報", f"https://stockanalysis.com/stocks/{q.lower()}/financials/"),
            ("Finviz", f"https://finviz.com/quote.ashx?t={q}"),
            ("TradingView", f"https://www.tradingview.com/symbols/{q}/"),
            ("Investing.com 搜尋", f"https://www.investing.com/search/?q={q}")]


is_tw = bool(re.fullmatch(r"\d{4,6}[A-Z]?", sym)) if sym else False
st.sidebar.markdown("### 🔗 查證資料來源")
st.sidebar.caption("覺得數字怪怪的？點下面的網站，對照本程式抓到的資料。")
for _n, _u in site_links(sym, is_tw):
    st.sidebar.markdown(f"- [{_n}]({_u})")
side_nums = st.sidebar.empty()
st.sidebar.caption("各網站的計算口徑（例如 EPS 是否含一次性項目、近四季或年度）不同，小差異屬正常。")
GLOSS = [
    ("本益比 P/E", "股價 ÷ 每股盈餘，代表以目前獲利要多少年才賺回股價，越高代表市場期待越高。"),
    ("股價淨值比 P/B", "股價 ÷ 每股淨值，看股價是帳面價值的幾倍。"),
    ("EPS 每股盈餘", "公司賺的錢平均分到每一股是多少。"),
    ("現金殖利率", "每年現金股利 ÷ 股價，像買股票的「利息」。"),
    ("配息率", "賺的錢發出去多少比例，太高不易持續。"),
    ("營收成長率", "營收比去年同期成長多少，反映生意有沒有變大。"),
    ("毛利率", "賣東西扣掉直接成本後剩幾成，反映產品競爭力。"),
    ("淨利率", "每 100 元營收最後落袋多少。"),
    ("ROE 股東權益報酬率", "公司拿股東的錢賺錢的效率。"),
    ("負債比", "總負債 ÷ 總資產，越高財務風險越大。"),
    ("流動比率", "流動資產 ÷ 流動負債，看短期還債能力。"),
    ("營業現金流", "本業實際收進來的現金，比帳面獲利更難造假。"),
    ("庫藏股", "公司買回自家股票，會減少流通股數、縮小股東權益。"),
    ("波動度", "股價上下震盪的程度，越大越劇烈。"),
    ("最大回撤", "從高點跌到低點的最大跌幅，衡量最慘會虧多少。"),
    ("ETF 與費用率", "ETF 是一籃子股票的基金；費用率是每年被扣的管理成本。"),
]
QUIZ = [
    ("某股票本益比是 40 倍，最貼切的意思是？",
     ["公司賺的錢是股價的 40 倍", "以目前獲利計算，約要 40 年才賺回股價", "股價一年會漲 40%", "配息率是 40%"], 1,
     "本益比 = 股價 ÷ 每股盈餘，可想成「以現在的獲利，回本要幾年」。"),
    ("毛利率 60% 但淨利率只有 3%，最可能的原因是？",
     ["產品賣得不好", "營業費用或其他支出很高，吃掉大部分毛利", "公司沒有繳稅", "股價太低"], 1,
     "毛利只扣直接成本，淨利還要再扣研發、行銷、管理、利息和稅等費用。"),
    ("獲利現金含量（營業現金流 ÷ 淨利）長期遠低於 1，代表？",
     ["公司很會賺錢", "賺的錢可能還沒真的收回來，要追查原因", "公司負債很低", "股利一定很高"], 1,
     "帳面有獲利但沒有現金流入，可能是應收帳款堆積或一次性收益。"),
    ("最大回撤 -40% 代表？",
     ["股票一年跌 40%", "若買在最高點，最慘時帳面虧損約 40%", "公司虧損 40%", "每年會配息 40%"], 1,
     "最大回撤是從高點跌到低點的最大跌幅，用來衡量可能承受的痛苦。"),
    ("兩檔同類型 ETF 報酬接近，費用率分別是 0.05% 和 0.8%，長期持有通常？",
     ["費用率不重要", "費用率低的比較有利", "費用率高的比較有利", "兩者完全一樣"], 1,
     "費用率每年固定扣除，長期複利下會累積成明顯差距。"),
]
with st.sidebar.expander("📖 新手學院：名詞小字典"):
    for _t, _e in GLOSS:
        st.markdown(f"**{_t}**  \n{_e}")
with st.sidebar.expander("🧠 新手小測驗"):
    with st.form("quiz"):
        _ans = [st.radio(f"Q{i+1}. {q}", o, index=None, key=f"qz{i}") for i, (q, o, _, _) in enumerate(QUIZ)]
        _ok = st.form_submit_button("對答案")
    if _ok:
        _sc = sum(1 for a, (q, o, c, _) in zip(_ans, QUIZ) if a is not None and o.index(a) == c)
        st.markdown(f"**得分：{_sc} / {len(QUIZ)}**")
        for i, (a, (q, o, c, ex)) in enumerate(zip(_ans, QUIZ)):
            right = a is not None and o.index(a) == c
            st.markdown(f"{'✅' if right else '❌'} Q{i+1}：{ex}")

COL = {"good": "#00E5A0", "ok": "#FFC857", "bad": "#FF5C7A", None: "#5f7596"}
LBL = {"good": "良好", "ok": "普通", "bad": "留意", None: "無資料"}


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def render_chart(hist, UP, DN):
    if len(hist) > 2:
        DAYS = {"1M": 30, "3M": 91, "6M": 182, "YTD": None, "1Y": 365, "5Y": 1826, "全部": None}
        tf = st.radio("時間範圍", list(DAYS), index=4, horizontal=True, label_visibility="collapsed")
        k1, k2, k3 = st.columns([2, 1, 3])
        kind = k1.radio("樣式", ["K 線", "折線", "面積"], horizontal=True)
        freq = k2.selectbox("週期", ["日", "週", "月"])
        inds = k3.multiselect("技術指標", ["MA5", "MA20", "MA60", "成交量"], default=["成交量"])

        r = hist.set_index("date")
        if freq != "日":
            r = r.resample("W-FRI" if freq == "週" else "MS").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
        for n_ in (5, 20, 60):
            r[f"MA{n_}"] = r["close"].rolling(n_).mean()
        r = r.reset_index()
        last_d = r["date"].iloc[-1]
        if tf == "YTD":
            h = r[r["date"] >= pd.Timestamp(last_d.year, 1, 1)]
        elif DAYS[tf]:
            h = r[r["date"] >= last_d - pd.Timedelta(days=DAYS[tf])]
        else:
            h = r
        if len(h) < 2:
            h = r.tail(2)
        line_c = UP if float(h["close"].iloc[-1]) >= float(h["close"].iloc[0]) else DN

        lb = h.iloc[-1]
        oc = UP if lb["close"] >= lb["open"] else DN
        st.markdown(f'<div style="font-size:.8rem;color:#7f93b2;margin:2px 0 4px">'
                    f'<span style="color:{oc}">開 {lb["open"]:,.2f}　高 {lb["high"]:,.2f}　低 {lb["low"]:,.2f}　收 {lb["close"]:,.2f}</span>'
                    f'　成交量 {lb["volume"]:,.0f}　（{lb["date"].date()}，{freq}線）</div>', unsafe_allow_html=True)

        vol = "成交量" in inds
        fig = make_subplots(rows=2 if vol else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.78, 0.22] if vol else [1])
        if kind == "K 線":
            fig.add_trace(go.Candlestick(x=h["date"], open=h["open"], high=h["high"], low=h["low"], close=h["close"],
                                         increasing=dict(line=dict(color=UP), fillcolor=UP),
                                         decreasing=dict(line=dict(color=DN), fillcolor=DN), name="價格"), row=1, col=1)
            lo_, hi_ = float(h["low"].min()), float(h["high"].max())
        else:
            fig.add_trace(go.Scatter(x=h["date"], y=h["close"], mode="lines", name="收盤", line=dict(color=line_c, width=2),
                                     fill="tozeroy" if kind == "面積" else None, fillcolor=rgba(line_c, 0.12),
                                     hovertemplate="%{y:,.2f}<extra></extra>"), row=1, col=1)
            lo_, hi_ = float(h["close"].min()), float(h["close"].max())
        for name_, color_ in (("MA5", "#FFC857"), ("MA20", "#00E5FF"), ("MA60", "#B388FF")):
            if name_ in inds:
                fig.add_trace(go.Scatter(x=h["date"], y=h[name_], mode="lines", name=name_, line=dict(color=color_, width=1.3),
                                         hovertemplate="%{y:,.2f}<extra></extra>"), row=1, col=1)
        if vol:
            vc = [rgba(UP if c_ >= o_ else DN, 0.55) for o_, c_ in zip(h["open"], h["close"])]
            fig.add_trace(go.Bar(x=h["date"], y=h["volume"], marker_color=vc, name="成交量", showlegend=False,
                                 hovertemplate="%{y:,.0f}<extra></extra>"), row=2, col=1)
        pad = (hi_ - lo_) * 0.08 or 1
        fig.update_yaxes(range=[lo_ - pad, hi_ + pad], row=1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor="#16263f", color="#5f7596", rangeslider_visible=False,
                         showspikes=True, spikemode="across", spikethickness=1, spikecolor="#5f7596", spikedash="dot")
        fig.update_yaxes(gridcolor="#16263f", color="#5f7596", side="right", showspikes=True, spikecolor="#5f7596", spikethickness=1)
        if freq == "日":
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        # 限制拖曳與縮放範圍：不能超出資料的頭尾，避免出現一片空白
        step = h["date"].diff().median()
        step = step if pd.notna(step) else pd.Timedelta(days=1)
        x0, x1 = h["date"].iloc[0] - step * 0.6, h["date"].iloc[-1] + step * 0.6
        fig.update_xaxes(range=[x0, x1], minallowed=x0, maxallowed=x1)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(height=430 if vol else 350, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", font=dict(color="#9fb3d1"),
                          legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)), dragmode="pan")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})


# ---------- 趨勢、風險、比較 ----------
def _style(fig, title, h=230):
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#9fb3d1"), x=0), height=h,
                      margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, font=dict(color="#9fb3d1"), hovermode="x unified", bargap=0.35)
    fig.update_xaxes(type="category", color="#5f7596", showgrid=False)
    fig.update_yaxes(gridcolor="#16263f", color="#5f7596")
    return fig


def render_trend(d):
    tr = d.get("trend")
    st.markdown(f'<h3 class="s">▌多年趨勢（{d.get("tfreq", "")}）</h3>', unsafe_allow_html=True)
    if tr is None or len(tr) < 3:
        st.caption("歷史財報資料不足，無法繪製趨勢。")
        return
    tr = tr.copy()
    ud, ul, cfg = d["ud"], d["ul"], {"displayModeBar": False}
    x = tr["label"]
    tr["gm"] = tr["gp"] / tr["rev"] * 100
    tr["nm"] = tr["ni"] / tr["rev"] * 100
    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Bar(x=x, y=tr["rev"] / ud, marker_color="#00E5FF", hovertemplate="%{y:,.1f}<extra></extra>"))
    c1.plotly_chart(_style(f1, f"營收（{ul}）"), use_container_width=True, config=cfg)
    f2 = go.Figure(go.Bar(x=x, y=tr["ni"] / ud, marker_color="#7C4DFF", hovertemplate="%{y:,.1f}<extra></extra>"))
    c2.plotly_chart(_style(f2, f"淨利（{ul}）"), use_container_width=True, config=cfg)
    c3, c4 = st.columns(2)
    f3 = go.Figure(go.Bar(x=x, y=tr["eps"], marker_color="#FFC857", hovertemplate="%{y:,.2f}<extra></extra>"))
    c3.plotly_chart(_style(f3, "每股盈餘 EPS"), use_container_width=True, config=cfg)
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=x, y=tr["gm"], name="毛利率", mode="lines+markers", line=dict(color="#00E5A0", width=2),
                            hovertemplate="毛利率 %{y:.1f}%<extra></extra>"))
    f4.add_trace(go.Scatter(x=x, y=tr["nm"], name="淨利率", mode="lines+markers", line=dict(color="#FF5C7A", width=2),
                            hovertemplate="淨利率 %{y:.1f}%<extra></extra>"))
    _style(f4, "毛利率（綠）與淨利率（紅）%")
    c4.plotly_chart(f4, use_container_width=True, config=cfg)

    notes = []
    r = tr["rev"].dropna()
    if len(r) >= 2 and r.iloc[0] > 0:
        a, b = tr.loc[r.index[0], "label"], tr.loc[r.index[-1], "label"]
        notes.append(f"**營收** 從 {a} 到 {b} 變動 {(r.iloc[-1] / r.iloc[0] - 1) * 100:+.0f}%。長期向上是好公司的基本條件。")
    g = tr["gm"].dropna()
    if len(g) >= 2:
        pp = g.iloc[-1] - g.iloc[0]
        notes.append(f"**毛利率** 由 {g.iloc[0]:.1f}% 變為 {g.iloc[-1]:.1f}%（{'改善' if pp >= 0 else '下滑'} {abs(pp):.1f} 個百分點）。"
                     + ("持續改善常代表議價力或產品組合變好。" if pp >= 0 else "持續下滑要留意成本壓力或競爭加劇。"))
    e = tr["eps"].dropna()
    if len(e) >= 2:
        notes.append(f"**EPS** 由 {e.iloc[0]:.2f} 變為 {e.iloc[-1]:.2f}。要看的是長期方向，單期大起大落常是一次性因素。")
    if d.get("tfreq") == "季":
        notes.append("**單季數字有季節性。** 比較時建議拿「同一季度的跨年數字」，不要直接比相鄰兩季。")
    for t in notes:
        body = re.sub(r"[*][*](.+?)[*][*]", lambda m: "<b>" + m.group(1) + "</b>", t)
        st.markdown(f'<div class="lp">{body}</div>', unsafe_allow_html=True)


def risk_stats(hist):
    c = hist.set_index("date")["close"].astype(float).dropna()
    if len(c) < 60:
        return None
    ret = c.pct_change().dropna()
    dd = c / c.cummax() - 1
    low = dd.idxmin()
    return dict(vol=float(ret.tail(252).std() * 252 ** 0.5), mdd=float(dd.min()), t_low=low,
                t_high=c[:low].idxmax(), cur=float(dd.iloc[-1]), worst=float(ret.min()),
                worst_d=ret.idxmin(), years=len(c) / 252, dd=dd)


def render_risk(hist, UP, DN):
    k = risk_stats(hist)
    if not k:
        return
    st.markdown('<h3 class="s">▌風險檢視</h3>', unsafe_allow_html=True)

    def lv(v, g, o):
        return "good" if v <= g else "ok" if v <= o else "bad"

    cards = [
        ("年化波動度（近 1 年）", f"{k['vol']*100:.1f}%", lv(k["vol"], .25, .40),
         "股價上下震盪的幅度。越高代表漲跌越劇烈，越考驗你的心臟。"),
        (f"最大回撤（近 {k['years']:.1f} 年）", f"{k['mdd']*100:.1f}%", lv(-k["mdd"], .25, .45),
         f"從 {k['t_high'].date()} 高點到 {k['t_low'].date()}，最慘帳面虧損這麼多。買在高點就可能遇到。"),
        ("目前距離高點", f"{k['cur']*100:.1f}%", lv(-k["cur"], .10, .25),
         "現在股價比歷史最高點低多少。0% 代表正在創高。"),
        ("單日最大跌幅", f"{k['worst']*100:.1f}%", lv(-k["worst"], .04, .08),
         f"發生在 {k['worst_d'].date()}。一天就可能跌這麼多，要先想好承受得起。"),
    ]
    for i in range(0, 4, 2):
        cols = st.columns(2)
        for col, (n_, v_, l_, e_) in zip(cols, cards[i:i + 2]):
            col.markdown(f'<div class="card" style="--c:{COL[l_]}"><div class="h"><span>{n_}</span><span class="tag">{LBL[l_]}</span></div>'
                         f'<div class="v">{v_}</div><div class="e">{e_}</div></div>', unsafe_allow_html=True)
    dd = k["dd"]
    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, mode="lines", fill="tozeroy", line=dict(color="#FF5C7A", width=1.4),
                               fillcolor="rgba(255,92,122,0.15)", hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(title=dict(text="距離歷史高點的跌幅 %（越深代表越痛）", font=dict(size=13, color="#9fb3d1"), x=0), height=220,
                      margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb3d1"), hovermode="x unified")
    fig.update_xaxes(color="#5f7596", showgrid=False)
    fig.update_yaxes(gridcolor="#16263f", color="#5f7596", side="right")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="lp"><b>報酬和風險是一體兩面。</b>看到高報酬時，先問自己：如果跌到最大回撤那麼深，我撐得住嗎？</div>', unsafe_allow_html=True)


def load_any(s):
    tw = bool(re.fullmatch(r"\d{4,6}[A-Z]?", s))
    etf = is_tw_etf(s) if tw else is_us_etf(s)
    if etf:
        raise RuntimeError("目前只支援個股比較，ETF 請單獨查詢")
    return load_tw(s) if tw else load_us(s)


def render_compare(d1, s1, d2, s2):
    st.markdown('<h3 class="s">▌雙股比較</h3>', unsafe_allow_html=True)
    M1, _ = analyze(d1)
    M2, _ = analyze(d2)

    def sc(M):
        lvs = [m[2] for m in M if m[2]]
        return round(sum({"good": 2, "ok": 1, "bad": 0}[v] for v in lvs) / (len(lvs) * 2) * 100) if lvs else 0

    def cell(val, level):
        return f'<span style="color:{COL[level]}">●</span> {val}'

    pr = lambda d: "—" if d["price"] is None else f"{d['price']:,.2f} {d['cur']}"
    rows = [("基本面評分", f"<b>{sc(M1)}</b>", f"<b>{sc(M2)}</b>"), ("股價", pr(d1), pr(d2))]
    for a, b in zip(M1, M2):
        rows.append((a[0], cell(a[1], a[2]), cell(b[1], b[2])))
    k1, k2 = risk_stats(d1["hist"]), risk_stats(d2["hist"])
    if k1 and k2:
        rows.append(("年化波動度（1 年）", f"{k1['vol']*100:.1f}%", f"{k2['vol']*100:.1f}%"))
        rows.append(("最大回撤", f"{k1['mdd']*100:.1f}%", f"{k2['mdd']*100:.1f}%"))
    body = "".join(f"<tr><td>{n}</td><td>{a}</td><td>{b}</td></tr>" for n, a, b in rows)
    st.markdown(f'<div style="overflow-x:auto"><table class="cmp"><tr><th>指標</th><th>{s1}</th><th>{s2}</th></tr>{body}</table></div>',
                unsafe_allow_html=True)

    rng = st.radio("比較區間", ["3 個月", "6 個月", "1 年", "3 年", "5 年"], index=2, horizontal=True, key="cmp_rng")
    days = {"3 個月": 91, "6 個月": 182, "1 年": 365, "3 年": 1095, "5 年": 1826}[rng]
    h1, h2 = d1["hist"], d2["hist"]
    end = min(h1["date"].iloc[-1], h2["date"].iloc[-1])
    start = max(end - pd.Timedelta(days=days), h1["date"].iloc[0], h2["date"].iloc[0])
    fig = go.Figure()
    for h, s_, col in ((h1, s1, "#00E5FF"), (h2, s2, "#FFC857")):
        x = h[(h["date"] >= start) & (h["date"] <= end)]
        if len(x) > 1:
            fig.add_trace(go.Scatter(x=x["date"], y=x["close"] / x["close"].iloc[0] * 100, name=s_, mode="lines",
                                     line=dict(color=col, width=2), hovertemplate="%{y:.1f}<extra>" + s_ + "</extra>"))
    fig.update_layout(title=dict(text="起點 = 100 的相對走勢", font=dict(size=13, color="#9fb3d1"), x=0), height=300,
                      margin=dict(l=0, r=0, t=34, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9fb3d1"), hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(color="#5f7596", showgrid=False)
    fig.update_yaxes(gridcolor="#16263f", color="#5f7596", side="right")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    tip = "**比較要同類才有意義。** 同產業、規模相近的公司才適合並排看。"
    if d1["mk"] != d2["mk"]:
        tip += " 台股與美股的幣別、會計準則與估值水準不同，數字只能當參考。"
    st.markdown(f'<div class="lp">{re.sub(r"[*][*](.+?)[*][*]", lambda m: "<b>" + m.group(1) + "</b>", tip)}</div>', unsafe_allow_html=True)


# ---------- ETF 模式 ----------
def is_tw_etf(sym):
    return bool(re.fullmatch(r"00\d{2,4}[A-Z]?", sym))


@st.cache_data(ttl=3600, show_spinner=False)
def is_us_etf(sym):
    import yfinance as yf
    try:
        return (yf.Ticker(sym).info or {}).get("quoteType") == "ETF"
    except Exception:
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def load_etf_us(sym):
    import yfinance as yf
    t = yf.Ticker(sym)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    h = t.history(period="5y")
    if h.empty:
        raise RuntimeError("查無此代號的資料")
    hist = pd.DataFrame({"date": h.index.tz_localize(None), "open": h["Open"].astype(float).values,
                         "high": h["High"].astype(float).values, "low": h["Low"].astype(float).values,
                         "close": h["Close"].astype(float).values, "volume": h["Volume"].astype(float).values})
    price = None
    try:
        price = float(t.fast_info["last_price"])
    except Exception:
        price = float(hist["close"].iloc[-1])
    top, sect = [], {}
    try:
        fd = t.funds_data
        hold = fd.top_holdings
        for sym_, r in hold.head(10).iterrows():
            top.append((str(sym_), str(r.get("Name", "")), float(r.get("Holding Percent", 0))))
        sect = {str(k): float(v) for k, v in (fd.sector_weightings or {}).items()}
    except Exception:
        pass
    y = info.get("yield") or info.get("trailingAnnualDividendYield")
    return dict(name=info.get("longName") or info.get("shortName") or sym, cur="美元", price=price, hist=hist,
                yld=y, er=info.get("netExpenseRatio"), aum=info.get("totalAssets"),
                category=info.get("category"), family=info.get("fundFamily"), top=top, sect=sect)


@st.cache_data(ttl=3600, show_spinner=False)
def load_etf_tw(sid):
    px = fm("TaiwanStockPrice", sid, (TODAY - dt.timedelta(days=1830)).isoformat())
    if px.empty:
        raise RuntimeError("查無此代號的資料")
    pxs = px.sort_values("date")
    cl = pxs["close"].astype(float).values
    col = lambda n: pxs[n].astype(float).values if n in pxs else cl
    hist = pd.DataFrame({"date": pd.to_datetime(pxs["date"]).values, "open": col("open"), "high": col("max"),
                         "low": col("min"), "close": cl,
                         "volume": pxs["Trading_Volume"].astype(float).values if "Trading_Volume" in pxs else 0.0})
    price = float(cl[-1])
    yld = None
    try:
        dv = fm("TaiwanStockDividend", sid, (TODAY - dt.timedelta(days=400)).isoformat())
        if not dv.empty:
            tot = 0.0
            for _, r in dv.iterrows():
                tot += float(r.get("CashEarningsDistribution") or 0) + float(r.get("CashStatutorySurplus") or 0)
            yld = tot / price if tot else None
    except Exception:
        pass
    name = sid
    try:
        headers = {"Authorization": f"Bearer {fm_token()}"} if fm_token() else {}
        j = requests.get(FM, params={"dataset": "TaiwanStockInfo"}, headers=headers, timeout=30).json()
        df = pd.DataFrame(j["data"])
        r = df[df["stock_id"] == sid]
        if len(r):
            name = str(r.iloc[0]["stock_name"])
    except Exception:
        pass
    return dict(name=name, cur="元", price=price, hist=hist, yld=yld, er=None, aum=None,
                category=None, family=None, top=[], sect={})


def render_etf(e, sym, is_tw):
    hist = e["hist"]
    UP, DN = ("#FF4D6A", "#00E5A0") if is_tw else ("#00E5A0", "#FF4D6A")
    price = e["price"]
    chg_html, dtxt = "", ""
    if len(hist) >= 2 and price is not None:
        prev = float(hist["close"].iloc[-2])
        diff, pc = price - prev, (price - prev) / prev * 100
        c = UP if diff >= 0 else DN
        arrow = "▲" if diff >= 0 else "▼"
        chg_html = f'<div class="chg" style="color:{c}">{arrow} {abs(diff):.2f}（{abs(pc):.2f}%）</div>'
        dtxt = f"收盤資料日期：{hist['date'].iloc[-1].date()}"
    meta = "｜".join(x for x in [e.get("category"), e.get("family")] if x)
    st.markdown(f'<div class="hero"><div class="nm">{e["name"]}　{sym}　<span class="tag" style="--c:#00E5FF">ETF</span></div>'
                f'<div class="px">{price:,.2f}<small>{e["cur"]}</small></div>{chg_html}'
                f'<div class="dt">{dtxt}｜{meta}</div></div>', unsafe_allow_html=True)

    last_d = hist["date"].iloc[-1]

    def ret(days):
        sub = hist[hist["date"] <= last_d - pd.Timedelta(days=days)]
        if sub.empty:
            return None
        return float(hist["close"].iloc[-1]) / float(sub["close"].iloc[-1]) - 1

    def rt(v):
        if v is None:
            return '<b>—</b>'
        return f'<b style="color:{UP if v >= 0 else DN}">{v*100:+.1f}%</b>'

    yl = e.get("yld")
    er = e.get("er")
    aum = e.get("aum")
    h52 = hist.tail(252)
    st.markdown(
        f'<div class="kp"><div><span>殖利率</span><b>{"—" if yl is None else f"{yl*100:.2f}%"}</b></div>'
        f'<div><span>費用率</span><b>{"—" if er is None else f"{er:.2f}%"}</b></div>'
        f'<div><span>基金規模</span><b>{"—" if not aum else f"{aum/1e9:,.1f}B"}</b></div>'
        f'<div><span>52 週區間</span><b style="font-size:.95rem">{h52["close"].min():,.2f} ~ {h52["close"].max():,.2f}</b></div></div>'
        f'<div class="kp"><div><span>近 1 年（價格）</span>{rt(ret(365))}</div><div><span>近 3 年（價格）</span>{rt(ret(1095))}</div>'
        f'<div><span>近 5 年（價格）</span>{rt(ret(1826))}</div><div><span>資料天數</span><b>{len(hist)}</b></div></div>',
        unsafe_allow_html=True)

    render_chart(hist, UP, DN)
    render_risk(hist, UP, DN)

    top, sect = e.get("top") or [], e.get("sect") or {}
    if top:
        st.markdown('<h3 class="s">▌前十大持股</h3>', unsafe_allow_html=True)
        pcts = [(p * 100 if p <= 1 else p) for _, _, p in top]
        mx = max(pcts) or 1
        rows = "".join(f'<div style="margin:7px 0"><div style="display:flex;justify-content:space-between;font-size:.85rem;color:#d6e4f5">'
                       f'<span>{s_}　<span style="color:#7f93b2">{n_[:26]}</span></span><b>{p_:.1f}%</b></div>'
                       f'<div style="height:5px;background:#1b2c47;border-radius:3px"><div style="height:5px;width:{p_/mx*100:.0f}%;background:#00E5FF;border-radius:3px"></div></div></div>'
                       for (s_, n_, _), p_ in zip(top, pcts))
        st.markdown(f'<div class="card" style="--c:#00E5FF">{rows}<div class="e">前十大合計約 {sum(pcts):.1f}%</div></div>', unsafe_allow_html=True)
    if sect:
        st.markdown('<h3 class="s">▌產業分布</h3>', unsafe_allow_html=True)
        items = sorted(sect.items(), key=lambda kv: -kv[1])[:6]
        rows = "".join(f'<div style="margin:7px 0"><div style="display:flex;justify-content:space-between;font-size:.85rem;color:#d6e4f5">'
                       f'<span>{k_.replace("_", " ").title()}</span><b>{v_*100 if v_ <= 1 else v_:.1f}%</b></div>'
                       f'<div style="height:5px;background:#1b2c47;border-radius:3px"><div style="height:5px;width:{min(v_*100 if v_ <= 1 else v_, 100):.0f}%;background:#7C4DFF;border-radius:3px"></div></div></div>'
                       for k_, v_ in items)
        st.markdown(f'<div class="card" style="--c:#7C4DFF">{rows}</div>', unsafe_allow_html=True)
    if not top:
        st.info("這檔的持股、費用率等資料這裡抓不到，請到左側「查證資料來源」的網站或投信官網查看。")

    st.markdown('<h3 class="s">▌學習重點</h3>', unsafe_allow_html=True)
    tips = ["**ETF 是一籃子股票。** 買一檔就等於同時持有很多公司，風險比單押一家分散，所以新手常從 ETF 開始；但它沒有「公司財報」，不能用本益比、ROE 這類個股指標去評分。"]
    if er is not None:
        tips.append(f"**費用率 {er:.2f}%。** 這是每年固定被扣的成本，長期複利下差距很大；同類型 ETF 通常費用率越低越有利。" if er <= 0.5
                    else f"**費用率 {er:.2f}% 偏高。** 長期持有會明顯侵蝕報酬，建議和同類型 ETF 比較。")
    if top:
        tot = sum((p * 100 if p <= 1 else p) for _, _, p in top)
        tips.append(f"**前十大持股佔約 {tot:.0f}%。** " + ("集中度偏高，少數公司的漲跌會明顯影響整檔 ETF。" if tot > 50 else "集中度不算高，分散程度較好。"))
    if yl is not None:
        tips.append(f"**殖利率約 {yl*100:.2f}%。** 高配息不等於高報酬，配息會讓價格相應下調，要看「含息總報酬」。")
    tips.append("**上面的報酬只算價格漲跌，沒有含配息。** 過去績效不代表未來，也要留意追蹤的指數、規模與成交量（流動性）。")
    for t in tips:
        st.markdown(f'<div class="lp">{re.sub(r"[*][*](.+?)[*][*]", lambda m: "<b>" + m.group(1) + "</b>", t)}</div>', unsafe_allow_html=True)

    lines = ["**本程式抓到的數字**", "", f"- 股價：{price:,.2f} {e['cur']}",
             f"- 殖利率：{'—' if yl is None else f'{yl*100:.2f}%'}", f"- 費用率：{'—' if er is None else f'{er:.2f}%'}",
             f"- 基金規模：{'—' if not aum else f'{aum/1e9:,.1f}B'}"]
    side_nums.markdown("\n".join(lines))


if sym:
    is_tw = bool(re.fullmatch(r"\d{4,6}[A-Z]?", sym))
    etf = is_tw_etf(sym) if is_tw else is_us_etf(sym)
    if etf:
        try:
            with st.spinner("抓取 ETF 資料中…"):
                e = load_etf_tw(sym) if is_tw else load_etf_us(sym)
        except Exception as ex:
            st.error(f"抓取失敗：{ex}")
            st.stop()
        render_etf(e, sym, is_tw)
        st.stop()
    try:
        with st.spinner("抓取資料中…"):
            d = load_tw(sym) if is_tw else load_us(sym)
    except Exception as e:
        st.error(f"抓取失敗：{e}")
        st.stop()

    adj = adj_in * d["ud"]
    if adj and d["ni"]:
        old = d["ni"]
        d["ni"] = old - adj
        d["eps"] = d["eps"] * d["ni"] / old if d["eps"] else d["eps"]
    M, R = analyze(d)
    hist = d["hist"]
    _f = lambda v: "—" if v is None else f"{v / d['ud']:,.2f}"
    _g = lambda v: "—" if v is None else f"{v:,.2f}"
    _lines = ["**本程式抓到的數字**", "",
              f"- 股價：{_g(d['price'])} {d['cur']}", f"- 近四季 EPS：{_g(d['eps'])}",
              f"- 每股淨值：{_g(d['bv'])}", f"- 年度股利：{_g(d['div'])}",
              f"- 近四季營收：{_f(d['rev'])} {d['ul']}", f"- 近四季淨利：{_f(d['ni'])} {d['ul']}",
              f"- 營業現金流：{_f(d['cfo'])} {d['ul']}", f"- 財報最新期別：{d['asof']}"]
    side_nums.markdown("\n".join(_lines))

    # 台股慣例：紅漲綠跌；美股：綠漲紅跌
    UP, DN = ("#FF4D6A", "#00E5A0") if is_tw else ("#00E5A0", "#FF4D6A")
    price = d["price"]
    chg_html, line_c, dtxt = "", "#00E5FF", ""
    if len(hist) >= 2 and price is not None:
        prev = float(hist["close"].iloc[-2])
        diff, pc = price - prev, (price - prev) / prev * 100
        c = UP if diff >= 0 else DN
        line_c = UP if float(hist["close"].iloc[-1]) >= float(hist["close"].iloc[0]) else DN
        arrow = "▲" if diff >= 0 else "▼"
        chg_html = f'<div class="chg" style="color:{c}">{arrow} {abs(diff):.2f}（{abs(pc):.2f}%）</div>'
        dtxt = f"收盤資料日期：{hist['date'].iloc[-1].date()}"
    ptxt = "—" if price is None else f"{price:,.2f}"
    st.markdown(f'<div class="hero"><div class="nm">{d["name"]}　{sym}</div>'
                f'<div class="px">{ptxt}<small>{d["cur"]}</small></div>{chg_html}'
                f'<div class="dt">{dtxt}｜財報最新期別 {d["asof"]}｜金額單位：{d["ul"]}</div></div>', unsafe_allow_html=True)

    h52 = hist.tail(252)
    f = lambda v: "—" if v is None else f"{v:,.2f}"
    pe_t = "—" if not R["pe"] or R["pe"] <= 0 else f"{R['pe']:.1f} 倍"
    yl_t = "—" if R["yl"] is None else f"{R['yl']*100:.2f}%"
    hi = f(float(h52["close"].max())) if len(h52) else "—"
    lo = f(float(h52["close"].min())) if len(h52) else "—"
    st.markdown(f'<div class="kp"><div><span>本益比</span><b>{pe_t}</b></div><div><span>現金殖利率</span><b>{yl_t}</b></div>'
                f'<div><span>52 週最高</span><b>{hi}</b></div><div><span>52 週最低</span><b>{lo}</b></div></div>', unsafe_allow_html=True)

    render_chart(hist, UP, DN)
    render_risk(hist, UP, DN)

    sc_n = {"good": 2, "ok": 1, "bad": 0}
    lv = [m[2] for m in M if m[2]]
    score = round(sum(sc_n[v] for v in lv) / (len(lv) * 2) * 100) if lv else 0
    sc_c = COL["good"] if score >= 70 else COL["ok"] if score >= 45 else COL["bad"]
    st.markdown(f'<div class="sc"><div class="ring" style="background:conic-gradient({sc_c} {score}%,#1b2c47 0)"><i>{score}</i></div>'
                f'<div class="tx"><b style="color:#fff">基本面評分</b><br>良好 {lv.count("good")} 項｜普通 {lv.count("ok")} 項｜留意 {lv.count("bad")} 項</div></div>',
                unsafe_allow_html=True)

    st.markdown('<h3 class="s">▌11 項關鍵指標</h3>', unsafe_allow_html=True)
    for i in range(0, len(M), 2):
        cols = st.columns(2)
        for col, (name, val, level, expl) in zip(cols, M[i:i + 2]):
            col.markdown(f'<div class="card" style="--c:{COL[level]}"><div class="h"><span>{name}</span><span class="tag">{LBL[level]}</span></div>'
                         f'<div class="v">{val}</div><div class="e">{expl}</div></div>', unsafe_allow_html=True)

    render_trend(d)

    st.markdown('<h3 class="s">▌學習重點</h3>', unsafe_allow_html=True)
    for t in learn(R, adj):
        body = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", t)
        st.markdown(f'<div class="lp">{body}</div>', unsafe_allow_html=True)

    if sym2:
        try:
            render_compare(d, sym, load_any(sym2), sym2)
        except Exception as ex2:
            st.warning(f"比較對象載入失敗：{ex2}")

    with st.expander("原始數據（出錯時請截圖回報）"):
        keys = ("rev", "prev", "gp", "ni", "eq", "ta", "tl", "ca", "cl", "cfo")
        st.json({k: (round(v / d["ud"], 3) if k in keys and v else v) for k, v in d.items() if k not in ("ud", "ul", "mk", "cur", "hist")})
