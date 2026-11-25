"""基本面小幫手：輸入代號，自動抓台股（FinMind）或美股（yfinance）財報，算出 11 項指標並給學習重點。
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
    fs = fm("TaiwanStockFinancialStatements", sid, s3)
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
        name=sid, cur="元", ud=1e8, ul="億元", mk="tw",
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
    eps_t = ttm(eps)
    ni_t = ttm(ni)
    if eps_t is None and ni_t and shares:
        eps_t = ni_t / shares

    return dict(
        name=info.get("shortName", sym), cur="美元", ud=1e9, ul="十億美元", mk="us",
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
st.set_page_config(page_title="基本面小幫手", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:880px}
h1.t{font-size:1.6rem;margin:0;font-weight:800;letter-spacing:.04em;background:linear-gradient(90deg,#00E5FF,#7C4DFF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
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
head.markdown(f'<h1 class="t{cls}">📊 基本面小幫手</h1><div class="sub{cls}">輸入代號自動抓財報｜台股、美股｜僅供學習，不構成投資建議</div>', unsafe_allow_html=True)
if landing:
    st.markdown("<style>.block-container{padding-top:26vh !important}</style>" + hint, unsafe_allow_html=True)
adj_in = 0.0
if sym:
    with st.expander("進階：扣除一次性損益"):
        adj_in = st.number_input("金額單位同下方財報（台股：億元，美股：十億美元）", value=0.0, step=1.0)


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

COL = {"good": "#00E5A0", "ok": "#FFC857", "bad": "#FF5C7A", None: "#5f7596"}
LBL = {"good": "良好", "ok": "普通", "bad": "留意", None: "無資料"}


def rgba(h, a):
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


if sym:
    is_tw = bool(re.fullmatch(r"\d{4,6}[A-Z]?", sym))
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
        fig.update_layout(height=430 if vol else 350, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", font=dict(color="#9fb3d1"),
                          legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)), dragmode="pan")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})

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

    st.markdown('<h3 class="s">▌學習重點</h3>', unsafe_allow_html=True)
    for t in learn(R, adj):
        body = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", t)
        st.markdown(f'<div class="lp">{body}</div>', unsafe_allow_html=True)

    with st.expander("原始數據（出錯時請截圖回報）"):
        keys = ("rev", "prev", "gp", "ni", "eq", "ta", "tl", "ca", "cl", "cfo")
        st.json({k: (round(v / d["ud"], 3) if k in keys and v else v) for k, v in d.items() if k not in ("ud", "ul", "mk", "cur", "hist")})
