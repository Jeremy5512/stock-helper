"""基本面小幫手：輸入代號，自動抓台股（FinMind）或美股（yfinance）財報，算出 11 項指標並給學習重點。
注意：本程式未在實際網路環境測試過，欄位名稱若與資料源不符，請看畫面下方「原始數據」並回報。"""
import datetime as dt
import os
import re

import pandas as pd
import requests
import streamlit as st

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
    px = fm("TaiwanStockPrice", sid, (TODAY - dt.timedelta(days=20)).isoformat())
    fs = fm("TaiwanStockFinancialStatements", sid, s3)
    bs = fm("TaiwanStockBalanceSheet", sid, s1)
    cf = fm("TaiwanStockCashFlowsStatement", sid, s3)
    try:
        dv = fm("TaiwanStockDividend", sid, s3)
    except Exception:
        dv = pd.DataFrame()
    if px.empty or fs.empty or bs.empty:
        raise RuntimeError("查無此代號的資料")

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
        price=float(px.sort_values("date")["close"].iloc[-1]),
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
        price=price, eps=eps_t, bv=(equity / shares) if equity and shares else None, div=div,
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


# ---------- 畫面 ----------
st.set_page_config(page_title="基本面小幫手", page_icon="📊")
st.title("📊 基本面小幫手")
st.caption("輸入代號自動抓財報：台股（例 2330）、美股（例 AAPL）。僅供學習，不構成投資建議。")

sym = st.text_input("股票代號", placeholder="2330 或 AAPL").strip().upper()
adj_in = st.number_input("（選填）要扣除的一次性損益，單位同下方金額（億元或十億美元）", value=0.0, step=1.0)

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
    score_n = {"good": 2, "ok": 1, "bad": 0}
    lv = [m[2] for m in M if m[2]]
    score = round(sum(score_n[v] for v in lv) / (len(lv) * 2) * 100) if lv else 0
    st.subheader(f"{d['name']}（{sym}）　評分 {score} / 100")
    st.caption(f"財報最新期別：{d['asof']}；股價為最近一個交易日；金額單位：{d['ul']}。")

    icon = {"good": "🟢 良好", "ok": "🟡 普通", "bad": "🔴 留意", None: "⚪ 無資料"}
    for name, val, level, expl in M:
        st.markdown(f"**{name}**　{val}　{icon[level]}")
        st.caption(expl)

    st.subheader("學習重點")
    for t in learn(R, adj):
        st.info(t)

    with st.expander("原始數據（出錯時請截圖回報）"):
        raw = {k: v for k, v in d.items() if k not in ("ud", "ul", "mk", "cur")}
        st.json({k: (round(v / d["ud"], 3) if k in ("rev", "prev", "gp", "ni", "eq", "ta", "tl", "ca", "cl", "cfo") and v else v)
                 for k, v in raw.items()})
