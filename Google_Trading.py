
import os, json, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# CONFIG 
TICKER = "GOOG"
ALT_TICKER = "GOOGL"           
PERIOD = "8y"                   

REBALANCE = "W-FRI"            
LOOKBACK_D = 252                
FWD_HORIZON_D = 10              
PCTL_WINDOW_WEEKS = 20          
LONG_ONLY = False             
USE_REGIME_FILTER = True
REGIME_WINDOW = 126             # ~6 months


RISK_BAND = "moderate"
RISK_BANDS = {
    "low":      (0.90, 0.10),
    "moderate": (0.80, 0.20),
    "high":     (0.70, 0.30),
}

# ATR-based prompts
ATR_WINDOW = 14
STOP_MULT = 1.5
TAKE_MULT = 3.0

STATE_DIR = "./live_state"
os.makedirs(STATE_DIR, exist_ok=True)
PRED_HISTORY_CSV = os.path.join(STATE_DIR, f"pred_history_{TICKER}.csv")
LEDGER_CSV       = os.path.join(STATE_DIR, f"signals_ledger_{TICKER}.csv")
MODEL_INFO_JSON  = os.path.join(STATE_DIR, f"model_info_{TICKER}.json")
CACHE_CSV        = os.path.join(STATE_DIR, f"cache_{TICKER}.csv")
POS_STATE        = os.path.join(STATE_DIR, "position_state.json")

# local CSV used for updates
LOCAL_CSV = "goog_historical_data.csv"


def stops_for(direction, price, atr_pct, stop_mult=STOP_MULT, take_mult=TAKE_MULT):
    """Return (stop, take) given direction ∈ {-1,0,1}. If direction==0 -> (None,None)."""
    if direction == 1:
        return round(price * (1 - stop_mult * atr_pct), 2), round(price * (1 + take_mult * atr_pct), 2)
    if direction == -1:
        return round(price * (1 + stop_mult * atr_pct), 2), round(price * (1 - take_mult * atr_pct), 2)
    return None, None

def load_position():
    try:
        if os.path.exists(POS_STATE) and os.path.getsize(POS_STATE) > 0:
            with open(POS_STATE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"position": 0, "stop": None, "take": None, "last_ts": None}

def save_position(state):
    try:
        with open(POS_STATE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass

def rsi(series: pd.Series, period=14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0); down = -d.clip(upper=0.0)
    gain = up.rolling(period).mean(); loss = down.rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    m = ema_f - ema_s
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s, m - s

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 1-D feature columns (squeezes any accidental (N,1) frames)."""
    def _col(name, default_zero=False):
        s = df[name] if name in df.columns else (pd.Series(0.0, index=df.index) if default_zero else pd.Series(index=df.index, dtype=float))
        if isinstance(s, pd.DataFrame):  # squeeze (N,1) -> (N,)
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")

    close = _col("Close")
    high  = _col("High")
    low   = _col("Low")
    openp = _col("Open")
    vol   = _col("Volume", default_zero=True).fillna(0.0)

    logp = np.log(close)
    ret1  = logp.diff()
    ret5  = logp.diff(5)
    ret10 = logp.diff(10)
    vol10 = ret1.rolling(10).std()
    vol20 = ret1.rolling(20).std()
    ma10, ma20, ma50, ma200 = (close.rolling(n).mean() for n in (10, 20, 50, 200))
    rsi14 = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    bb_std = close.rolling(20).std()
    bb_z   = (close - ma20) / (2 * bb_std + 1e-12)
    vol_z  = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)
    hl_range  = (high - low) / close.replace(0, np.nan)
    oc_change = (close - openp) / openp.replace(0, np.nan)

    X = pd.DataFrame({
        "ret1": ret1, "ret5": ret5, "ret10": ret10,
        "vol10": vol10, "vol20": vol20,
        "ma10_ratio": close/ma10 - 1, "ma20_ratio": close/ma20 - 1,
        "ma50_ratio": close/ma50 - 1, "ma200_ratio": close/ma200 - 1,
        "rsi14": rsi14,
        "macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "bb_z": bb_z, "vol_z": vol_z,
        "hl_range": hl_range, "oc_change": oc_change,
    }, index=df.index)
    return X


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    rename = {}
    for c in list(df.columns):
        lc = c.lower().replace(" ", "_")
        if lc == "date": rename[c] = "Date"
        elif lc == "open": rename[c] = "Open"
        elif lc == "high": rename[c] = "High"
        elif lc == "low": rename[c] = "Low"
        elif lc == "close": rename[c] = "Close"
        elif lc in ("adj_close", "adjclose"): rename[c] = "Adj Close"
        elif lc in ("volume", "vol"): rename[c] = "Volume"
    df = df.rename(columns=rename)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing after normalization.")
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    elif isinstance(df["Volume"], pd.DataFrame):
        df["Volume"] = df["Volume"].iloc[:, 0]

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df.index.name = "Date"
    return df

def load_from_local_csv(path: str):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _normalize_ohlcv(df)
            if not df.empty:
                return df
    except Exception:
        pass
    return None

def load_from_cache(path: str):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _normalize_ohlcv(df)
            if not df.empty:
                return df
    except Exception:
        pass
    return None

def save_cache(df: pd.DataFrame, path: str):
    try:
        df.reset_index().to_csv(path, index=False)
    except Exception:
        pass

def download_from_yahoo(sym: str, period: str, tries: int = 5):
    try:
        import yfinance as yf
    except ImportError:
    # return None if not available
        return None
    for i in range(tries):
        try:
            df = yf.download(sym, period=period, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.dropna()
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
                df.index.name = "Date"
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                return df
        except Exception:
            pass
        time.sleep(min(1*(2**i), 16))  # 1s,2s,4s,8s,16s
    return None

def download_from_stooq(sym: str):
    try:
        import pandas_datareader.data as web
    except Exception:
        return None
    for ticker in [sym, f"{sym}.US"]:
        try:
            df = web.DataReader(ticker, "stooq")
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.sort_index()
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
                df.index.name = "Date"
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
        except Exception:
            continue
    return None

def try_online_backfill(df, ticker=TICKER):
    """Append newer rows from Yahoo to df; persist back to LOCAL_CSV and CACHE_CSV."""
    try:
        import yfinance as yf
    except Exception:
        return df
    if df.empty:
        return df
    last = df.index.max()
    start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        upd = yf.download(ticker, start=start, interval="1d",
                          auto_adjust=False, progress=False, threads=False)
        if isinstance(upd, pd.DataFrame) and not upd.empty:
            upd = upd.dropna()
            upd.index = pd.to_datetime(upd.index, utc=True).tz_convert(None)
            upd.index.name = "Date"
            upd = upd[["Open","High","Low","Close","Volume"]]
            out = pd.concat([df, upd]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            added = len(out) - len(df)
            if added > 0:
                print(f"[backfill] Appended {added} new day(s) up to {out.index.max().date()}")
            try: out.reset_index().to_csv(LOCAL_CSV, index=False)
            except Exception: pass
            try: out.reset_index().to_csv(CACHE_CSV, index=False)
            except Exception: pass
            return out
    except Exception:
        pass
    return df

def get_price_history() -> pd.DataFrame:
    # 1) Local CSV (then backfill)
    df = load_from_local_csv(LOCAL_CSV)
    if df is not None:
        df = try_online_backfill(df, ticker=TICKER)
        return df
    df = load_from_cache(CACHE_CSV)
    if df is not None:
        return df
    # Yahoo (GOOG → GOOGL)
    for sym in [TICKER, ALT_TICKER]:
        df = download_from_yahoo(sym, PERIOD)
        if df is not None:
            save_cache(df, CACHE_CSV)
            return df
    # Stooq
    df = download_from_stooq(TICKER)
    if df is not None:
        save_cache(df, CACHE_CSV)
        return df
    raise RuntimeError(
        "No data loaded. Provide LOCAL_CSV, or install yfinance / pandas-datareader for online sources."
    )


def load_pred_history(path=PRED_HISTORY_CSV) -> pd.Series:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.Series(dtype=float)
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if "Date" in df.columns:
                df = df.set_index("Date").sort_index()
            else:
                raise ValueError("No 'Date' column")
        except Exception:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.rename_axis("Date").sort_index()
        s = df["pred"] if "pred" in df.columns else df.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)

def save_pred_history(series: pd.Series, path=PRED_HISTORY_CSV):
    s = pd.Series(series.values, index=pd.to_datetime(series.index))
    s.index.name = "Date"
    s.sort_index().to_frame("pred").to_csv(path, index=True)

def weekly_filter(idx: pd.DatetimeIndex, rule="W-FRI"):
    return idx.to_series().resample(rule).last().dropna().index

def rolling_percentile_from_history(pred_today: float, hist: pd.Series, window: int):
    if len(hist) == 0:
        return np.nan
    prev = hist.iloc[-window:] if len(hist) >= window else hist
    return ((prev < pred_today).sum() + 0.5 * (prev == pred_today).sum()) / len(prev)


def main():
    # Data
    df = get_price_history()

    # FEATURES for the whole series; LABEL only where available
    X_all = build_features(df).dropna()
    close_all = df["Close"].reindex(X_all.index)
    y_all = (np.log(close_all.shift(-FWD_HORIZON_D)) - np.log(close_all)).rename("fwd")

    # TRAIN on rows that have labels
    train_df = pd.concat([X_all, y_all], axis=1).dropna()
    X_tr_full = train_df.drop(columns=["fwd"])
    y_tr_full = train_df["fwd"]

    # Train on recent lookback
    X_tr = X_tr_full.iloc[-LOOKBACK_D:]
    y_tr = y_tr_full.iloc[-LOOKBACK_D:]
    if len(X_tr) < 150:
        raise RuntimeError("Not enough training data after lookback. Reduce LOOKBACK_D.")

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
        ("hgb", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05,
                                              max_iter=400, random_state=42)),
    ])
    model.fit(X_tr, y_tr)

    # Predict on the LATEST BAR
    t_ref = X_all.index[-1]
    pred_today = float(model.predict(X_all.iloc[[-1]])[0])

    # Update prediction history & compute percentile (exclude today)
    pred_hist = load_pred_history()
    if t_ref not in pred_hist.index:
        pred_hist.loc[t_ref] = pred_today
        save_pred_history(pred_hist)

    hist_excl = pred_hist.drop(index=t_ref, errors="ignore")
    keep = weekly_filter(hist_excl.index, REBALANCE) if REBALANCE != "D" else hist_excl.index
    hist_excl = hist_excl.reindex(keep).dropna()
    pct_today = rolling_percentile_from_history(pred_today, hist_excl, PCTL_WINDOW_WEEKS)
    if not np.isfinite(pct_today):
        # fallback to training target distribution
        pct_today = ((y_tr < pred_today).sum() + 0.5 * (y_tr == pred_today).sum()) / len(y_tr)

    # Map percentile to SIGNAL direction
    buy_q, sell_q = RISK_BANDS[RISK_BAND]
    signal_dir = 0
    if pct_today >= buy_q:
        signal_dir = 1
    elif (not LONG_ONLY) and pct_today <= sell_q:
        signal_dir = -1

    # Regime filter (on signal)
    if USE_REGIME_FILTER:
        ts_mom = float((close_all / close_all.shift(REGIME_WINDOW) - 1).loc[t_ref])
        if signal_dir == 1 and ts_mom <= 0:
            signal_dir = 0
        if signal_dir == -1 and ts_mom >= 0:
            signal_dir = 0

    # Only APPLY changes on rebalance dates; otherwise keep current position
    on_rebalance = True if REBALANCE == "D" else (t_ref in weekly_filter(X_all.index, REBALANCE))
    state = load_position()
    current_pos = int(state.get("position", 0))

    if on_rebalance:
        target_position = signal_dir
        action_label = {1: "REBALANCE→LONG", 0: "REBALANCE→FLAT", -1: "REBALANCE→SHORT"}[target_position]
    else:
        target_position = current_pos
        action_label = "NO_CHANGE" if signal_dir == current_pos else "PENDING_REBALANCE"

    # ATR & level prompts
    entry = float(df.loc[t_ref, "Close"])           # price aligned to timestamp
    atr14_val = float(atr(df, ATR_WINDOW).loc[t_ref])
    atr_pct = float(atr14_val / entry) if entry else np.nan

    # Suggested = for the *new* signal; Active = for the *applied/held* position
    suggested_stop, suggested_take = stops_for(signal_dir, entry, atr_pct)

    # Active: if we rebalance now, set new active; else keep last saved state
    if on_rebalance:
        active_stop, active_take = stops_for(target_position, entry, atr_pct)
        save_position({"position": int(target_position),
                       "stop": active_stop,
                       "take": active_take,
                       "last_ts": str(t_ref)})
    else:
        # Keep previously saved active levels; if none and in a position, provide fallbacks
        active_stop = state.get("stop")
        active_take = state.get("take")
        if (active_stop is None or active_take is None) and target_position != 0:
            active_stop, active_take = stops_for(target_position, entry, atr_pct)

    # Always include hypotheticals so fields are populated even if flat
    hyp_long_stop,  hyp_long_take  = stops_for( 1, entry, atr_pct)
    hyp_short_stop, hyp_short_take = stops_for(-1, entry, atr_pct)

    # Output + ledger
    signal = {
        "timestamp": str(t_ref),
        "ticker": TICKER,
        "close": round(entry, 2),

        f"predicted_log_return_next_{FWD_HORIZON_D}d": round(pred_today, 6),
        f"pred_percentile_{PCTL_WINDOW_WEEKS}w": round(float(pct_today), 4),

        "risk_band": RISK_BAND,
        "rebalance": REBALANCE,
        "long_only": LONG_ONLY,
        "regime_filter": USE_REGIME_FILTER,

        # signal vs application
        "signal_action": {1:"BUY/LONG", 0:"HOLD/FLAT", -1:"SELL/SHORT"}[signal_dir],
        "action_applied": action_label,
        "current_position": int(target_position),

        "atr_pct": round(atr_pct, 4) if np.isfinite(atr_pct) else None,

        # levels
        "suggested_stop": suggested_stop,
        "suggested_take_profit": suggested_take,
        "active_stop": active_stop,
        "active_take_profit": active_take,

        # hypotheticals (always filled)
        "hypothetical_long_stop": hyp_long_stop,
        "hypothetical_long_take_profit": hyp_long_take,
        "hypothetical_short_stop": hyp_short_stop,
        "hypothetical_short_take_profit": hyp_short_take,
    }

    row = {
        "Date": t_ref,
        "Signal": signal["signal_action"],
        "Applied": signal["action_applied"],
        "Pos": int(target_position),
        "Close": entry,
        "Pred": pred_today,
        "Pct": float(pct_today),
        "Stop": active_stop,
        "Take": active_take
    }
    df_row = pd.DataFrame([row]).set_index("Date")
    if os.path.exists(LEDGER_CSV):
        df_row.to_csv(LEDGER_CSV, mode="a", header=False)
    else:
        df_row.to_csv(LEDGER_CSV, mode="w", header=True)

    with open(MODEL_INFO_JSON, "w") as f:
        json.dump(signal, f, indent=2)

    print(json.dumps(signal, indent=2))


if __name__ == "__main__":
    main()
