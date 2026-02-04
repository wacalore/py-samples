#!/usr/bin/env python3
"""
Options chain analyzer (futures options, EOD).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

try:
    import pandas as pd  # type: ignore[import-not-found]

    HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore[assignment]
    HAVE_PANDAS = False

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    njit = None  # type: ignore[assignment]


SQRT_2PI = math.sqrt(2.0 * math.pi)


def resolve_numba(use_numba: Optional[bool]) -> bool:
    if use_numba is None:
        return HAVE_NUMBA and os.environ.get("USE_NUMBA", "1") != "0"
    return bool(use_numba) and HAVE_NUMBA


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


if HAVE_NUMBA:

    @njit(cache=True)
    def norm_cdf_nb(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @njit(cache=True)
    def norm_pdf_nb(x: float) -> float:
        return math.exp(-0.5 * x * x) / SQRT_2PI

    @njit(cache=True)
    def black76_price_nb(F: float, K: float, T: float, r: float, vol: float, is_call: int) -> float:
        if T <= 0.0:
            intrinsic = max((F - K) if is_call == 1 else (K - F), 0.0)
            return intrinsic
        if vol <= 0.0:
            intrinsic = max((F - K) if is_call == 1 else (K - F), 0.0)
            return math.exp(-r * T) * intrinsic
        df = math.exp(-r * T)
        vsqrt = vol * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * vol * vol * T) / vsqrt
        d2 = d1 - vsqrt
        if is_call == 1:
            return df * (F * norm_cdf_nb(d1) - K * norm_cdf_nb(d2))
        return df * (K * norm_cdf_nb(-d2) - F * norm_cdf_nb(-d1))

    @njit(cache=True)
    def black76_greeks_nb(F: float, K: float, T: float, r: float, vol: float, is_call: int) -> Tuple[float, float, float, float, float]:
        if T <= 0.0 or vol <= 0.0:
            price = black76_price_nb(F, K, T, r, vol, is_call)
            return 0.0, 0.0, 0.0, 0.0, -T * price

        df = math.exp(-r * T)
        vsqrt = vol * math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * vol * vol * T) / vsqrt
        pdf = norm_pdf_nb(d1)

        delta = df * norm_cdf_nb(d1) if is_call == 1 else -df * norm_cdf_nb(-d1)
        gamma = df * pdf / (F * vsqrt)
        vega = df * F * pdf * math.sqrt(T)

        dt = 1.0 / 365.0
        t2 = T - dt
        if t2 <= 0.0:
            t2 = 1.0e-6
        p_now = black76_price_nb(F, K, T, r, vol, is_call)
        p_later = black76_price_nb(F, K, t2, r, vol, is_call)
        theta = (p_later - p_now) / dt

        rho = -T * p_now
        return delta, gamma, vega, theta, rho

    @njit(cache=True)
    def implied_vol_bisect_nb(price: float, F: float, K: float, T: float, r: float, is_call: int) -> float:
        if T <= 0.0:
            return 0.0
        df = math.exp(-r * T)
        intrinsic = max((F - K) if is_call == 1 else (K - F), 0.0)
        lower = df * intrinsic
        if price < lower:
            price = lower + 1.0e-10

        low = 1.0e-6
        high = 3.0
        for _ in range(100):
            mid = 0.5 * (low + high)
            val = black76_price_nb(F, K, T, r, mid, is_call) - price
            if abs(val) < 1.0e-8:
                break
            if val > 0.0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    @njit(cache=True)
    def _iv_greeks_vector_nb(
        F: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        price: np.ndarray,
        is_call: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = F.shape[0]
        iv = np.empty(n, dtype=np.float64)
        delta = np.empty(n, dtype=np.float64)
        gamma = np.empty(n, dtype=np.float64)
        vega = np.empty(n, dtype=np.float64)
        theta = np.empty(n, dtype=np.float64)
        rho = np.empty(n, dtype=np.float64)
        for i in range(n):
            iv[i] = implied_vol_bisect_nb(price[i], F[i], K[i], T[i], r[i], is_call[i])
            d, g, v, t, ro = black76_greeks_nb(F[i], K[i], T[i], r[i], iv[i], is_call[i])
            delta[i] = d
            gamma[i] = g
            vega[i] = v
            theta[i] = t
            rho[i] = ro
        return iv, delta, gamma, vega, theta, rho

    @njit(cache=True)
    def _price_vector_nb(
        F: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        vol: np.ndarray,
        is_call: np.ndarray,
    ) -> np.ndarray:
        n = F.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = black76_price_nb(F[i], K[i], T[i], r[i], vol[i], is_call[i])
        return out


@dataclass
class ZeroCurve:
    terms: np.ndarray  # year fractions
    rates: np.ndarray  # zero rates (cont comp)

    def rate(self, t: float) -> float:
        if t <= self.terms[0]:
            return float(self.rates[0])
        if t >= self.terms[-1]:
            return float(self.rates[-1])
        return float(np.interp(t, self.terms, self.rates))

    def df(self, t: float) -> float:
        r = self.rate(t)
        return math.exp(-r * t)


TableLike = Union[Sequence[Dict[str, object]], "pd.DataFrame"]  # type: ignore[name-defined]

OPTIONS_REQUIRED_COLS = {
    "date",
    "expiry",
    "strike",
    "put_call",
    "settle",
    "underlying",
    "underlying_ric",
}
OPTIONS_NUMERIC_COLS = {"strike", "settle", "underlying"}
OPTIONS_DATE_COLS = {"date", "expiry"}

CURVE_REQUIRED_COLS = {"term", "rate"}
CURVE_NUMERIC_COLS = {"term", "rate"}

ANALYTICS_REQUIRED_COLS = {
    "date",
    "expiry",
    "strike",
    "put_call",
    "settle",
    "underlying",
    "iv",
    "iv_fit",
    "iv_resid",
    "iv_z",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "theo",
}


def _missing_columns(cols: Iterable[str], required: Set[str]) -> List[str]:
    return sorted(required - set(cols))


def _ensure_pandas() -> None:
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for DataFrame inputs.")


def validate_options_df(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    missing = _missing_columns(df.columns, OPTIONS_REQUIRED_COLS)
    if missing:
        raise ValueError(f"options table missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("options table is empty.")

    out = df.copy()
    for col in OPTIONS_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise")  # type: ignore[union-attr]
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise ValueError(f"options table column '{col}' must be finite.")
    for col in OPTIONS_DATE_COLS:
        out[col] = pd.to_datetime(out[col], errors="raise")  # type: ignore[union-attr]

    out["put_call"] = out["put_call"].astype(str).str.upper()
    bad_pc = ~out["put_call"].isin(["C", "P"])
    if bad_pc.any():
        bad_vals = sorted(out.loc[bad_pc, "put_call"].unique())
        raise ValueError(f"options table has invalid put_call values: {bad_vals}")

    null_cols = [c for c in OPTIONS_REQUIRED_COLS if out[c].isna().any()]
    if null_cols:
        raise ValueError(f"options table has nulls in columns: {', '.join(sorted(null_cols))}")

    ric = out["underlying_ric"].astype(str)
    if (ric.str.len() == 0).any():
        raise ValueError("options table has empty underlying_ric values.")

    return out


def validate_curve_df(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    missing = _missing_columns(df.columns, CURVE_REQUIRED_COLS)
    if missing:
        raise ValueError(f"curve table missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("curve table is empty.")

    out = df.copy()
    for col in CURVE_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise")  # type: ignore[union-attr]

    null_cols = [c for c in CURVE_REQUIRED_COLS if out[c].isna().any()]
    if null_cols:
        raise ValueError(f"curve table has nulls in columns: {', '.join(sorted(null_cols))}")

    out = out.sort_values("term").reset_index(drop=True)
    terms = out["term"].to_numpy(dtype=float)
    if not np.isfinite(terms).all():
        raise ValueError("curve term values must be finite.")
    if np.any(terms <= 0.0):
        raise ValueError("curve term values must be positive.")
    if len(terms) > 1 and np.any(np.diff(terms) <= 0.0):
        raise ValueError("curve term values must be strictly increasing.")
    rates = out["rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("curve rate values must be finite.")
    return out


def validate_analytics_df(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    missing = _missing_columns(df.columns, ANALYTICS_REQUIRED_COLS)
    if missing:
        raise ValueError(f"analytics table missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("analytics table is empty.")

    out = df.copy()
    out["put_call"] = out["put_call"].astype(str).str.upper()
    bad_pc = ~out["put_call"].isin(["C", "P"])
    if bad_pc.any():
        bad_vals = sorted(out.loc[bad_pc, "put_call"].unique())
        raise ValueError(f"analytics table has invalid put_call values: {bad_vals}")

    numeric_cols = [c for c in ANALYTICS_REQUIRED_COLS if c not in {"date", "expiry", "put_call"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")  # type: ignore[union-attr]
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise ValueError(f"analytics table column '{col}' must be finite.")

    out["date"] = pd.to_datetime(out["date"], errors="raise")  # type: ignore[union-attr]
    out["expiry"] = pd.to_datetime(out["expiry"], errors="raise")  # type: ignore[union-attr]
    return out


def validate_options_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        raise ValueError("options table is empty.")
    validated: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        missing = OPTIONS_REQUIRED_COLS - set(row.keys())
        if missing:
            raise ValueError(f"options row {idx} missing columns: {', '.join(sorted(missing))}")
        null_cols = [c for c in OPTIONS_REQUIRED_COLS if row.get(c) is None]
        if null_cols:
            raise ValueError(f"options row {idx} has nulls in columns: {', '.join(sorted(null_cols))}")
        try:
            float(row["strike"])
            float(row["settle"])
            float(row["underlying"])
        except Exception as exc:
            raise ValueError(f"options row {idx} has non-numeric strike/settle/underlying.") from exc
        try:
            date.fromisoformat(str(row["date"]))
            date.fromisoformat(str(row["expiry"]))
        except Exception as exc:
            raise ValueError(f"options row {idx} has invalid date/expiry format.") from exc
        pc = str(row["put_call"]).upper()
        if pc not in {"C", "P"}:
            raise ValueError(f"options row {idx} has invalid put_call value: {row['put_call']}")
        if str(row["underlying_ric"]) == "":
            raise ValueError(f"options row {idx} has empty underlying_ric.")
        new_row = dict(row)
        new_row["put_call"] = pc
        validated.append(new_row)
    return validated


def validate_curve_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        raise ValueError("curve table is empty.")
    validated: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        missing = CURVE_REQUIRED_COLS - set(row.keys())
        if missing:
            raise ValueError(f"curve row {idx} missing columns: {', '.join(sorted(missing))}")
        null_cols = [c for c in CURVE_REQUIRED_COLS if row.get(c) is None]
        if null_cols:
            raise ValueError(f"curve row {idx} has nulls in columns: {', '.join(sorted(null_cols))}")
        try:
            term = float(row["term"])
            rate = float(row["rate"])
        except Exception as exc:
            raise ValueError(f"curve row {idx} has non-numeric term/rate.") from exc
        if term <= 0.0:
            raise ValueError(f"curve row {idx} has non-positive term.")
        validated.append({"term": term, "rate": rate})

    validated.sort(key=lambda x: x["term"])
    terms_sorted = [float(r["term"]) for r in validated]
    if len(terms_sorted) > 1 and any(t2 <= t1 for t1, t2 in zip(terms_sorted, terms_sorted[1:])):
        raise ValueError("curve term values must be strictly increasing.")
    return validated


def validate_options_table(table: TableLike) -> Union[List[Dict[str, object]], "pd.DataFrame"]:  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(table, "columns"):
        return validate_options_df(table)  # type: ignore[arg-type]
    rows = _table_to_records(table)
    return validate_options_rows(rows)


def validate_curve_table(table: TableLike) -> Union[List[Dict[str, object]], "pd.DataFrame"]:  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(table, "columns"):
        return validate_curve_df(table)  # type: ignore[arg-type]
    rows = _table_to_records(table)
    return validate_curve_rows(rows)


def _table_to_records(table: TableLike) -> List[Dict[str, object]]:
    if isinstance(table, dict):
        return [table]
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        try:
            return table.to_dict(orient="records")
        except TypeError:
            return table.to_dict("records")
    return list(table)


def _records_to_df(rows: List[Dict[str, object]]):
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required to return a DataFrame.")
    return pd.DataFrame(rows)  # type: ignore[union-attr]


def curve_from_rows(curve_rows: TableLike) -> ZeroCurve:
    validated = validate_curve_table(curve_rows)
    rows = _table_to_records(validated)
    terms: List[float] = []
    rates: List[float] = []
    for row in rows:
        term = float(row["term"])
        rate = float(row["rate"])
        terms.append(term)
        rates.append(rate)
    if not terms:
        raise ValueError("curve_rows is empty.")
    return ZeroCurve(np.array(terms), np.array(rates))


def black76_price(F: float, K: float, T: float, r: float, vol: float, is_call: bool) -> float:
    if T <= 0.0:
        intrinsic = max((F - K) if is_call else (K - F), 0.0)
        return intrinsic
    if vol <= 0.0:
        intrinsic = max((F - K) if is_call else (K - F), 0.0)
        return math.exp(-r * T) * intrinsic
    df = math.exp(-r * T)
    vsqrt = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / vsqrt
    d2 = d1 - vsqrt
    if is_call:
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))


def black76_greeks(F: float, K: float, T: float, r: float, vol: float, is_call: bool) -> Dict[str, float]:
    if T <= 0.0 or vol <= 0.0:
        price = black76_price(F, K, T, r, vol, is_call)
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": -T * price,
        }
    df = math.exp(-r * T)
    vsqrt = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / vsqrt
    pdf = norm_pdf(d1)

    delta = df * norm_cdf(d1) if is_call else -df * norm_cdf(-d1)
    gamma = df * pdf / (F * vsqrt)
    vega = df * F * pdf * math.sqrt(T)

    dt = 1.0 / 365.0
    t2 = max(T - dt, 1.0e-6)
    p_now = black76_price(F, K, T, r, vol, is_call)
    p_later = black76_price(F, K, t2, r, vol, is_call)
    theta = (p_later - p_now) / dt

    rho = -T * p_now
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol_black76(price: float, F: float, K: float, T: float, r: float, is_call: bool) -> float:
    if T <= 0.0:
        return 0.0
    df = math.exp(-r * T)
    intrinsic = max((F - K) if is_call else (K - F), 0.0)
    lower = df * intrinsic
    price = max(price, lower + 1.0e-10)

    def f(vol: float) -> float:
        return black76_price(F, K, T, r, vol, is_call) - price

    return brentq(f, 1.0e-6, 3.0, maxiter=100, xtol=1.0e-10)


def make_dummy_curve(asof: date) -> Tuple[List[Dict[str, str]], ZeroCurve]:
    terms = np.array([1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 5.0])
    rates = np.array([0.045, 0.046, 0.047, 0.048, 0.049, 0.050])
    rows = [{"dt": asof.isoformat(), "term": f"{t:.6f}", "rate": f"{r:.6f}"} for t, r in zip(terms, rates)]
    return rows, ZeroCurve(terms, rates)


def true_iv(F: float, K: float, T: float) -> float:
    k = math.log(K / F)
    base = 0.08 + 0.02 * math.sqrt(T)
    skew = -0.40 * k
    curv = 0.90 * k * k
    vol = base + skew + curv
    return max(vol, 0.03)


def make_dummy_chain(asof: date, F: float, expiries: Iterable[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    strike_step = 0.5
    strikes = np.arange(F - 5.0, F + 5.01, strike_step)
    for d in expiries:
        exp = asof + timedelta(days=d)
        T = d / 365.0
        for K in strikes:
            for pc in ("C", "P"):
                is_call = pc == "C"
                vol = true_iv(F, K, T)
                price = black76_price(F, K, T, 0.047, vol, is_call)
                price = max(price + np.random.normal(0.0, 0.0025), 0.0)
                rows.append(
                    {
                        "date": asof.isoformat(),
                        "expiry": exp.isoformat(),
                        "strike": f"{K:.3f}",
                        "put_call": pc,
                        "settle": f"{price:.6f}",
                        "underlying": f"{F:.6f}",
                        "underlying_ric": "TYZ6",
                    }
                )
    return rows


def fit_surface_per_expiry(rows: List[Dict[str, str]]) -> Dict[Tuple[str, float], Dict[str, float]]:
    by_exp: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_exp.setdefault(r["expiry"], []).append(r)

    fitted: Dict[Tuple[str, float], Dict[str, float]] = {}
    for exp, exp_rows in by_exp.items():
        T = (date.fromisoformat(exp) - date.fromisoformat(exp_rows[0]["date"])).days / 365.0
        if T <= 0.0:
            continue
        ks = []
        ws = []
        wts = []
        for r in exp_rows:
            F = float(r["underlying"])
            K = float(r["strike"])
            iv = float(r["iv"])
            k = math.log(K / F)
            w = iv * iv * T
            vega = float(r["vega"])
            ks.append(k)
            ws.append(w)
            wts.append(max(vega, 1.0e-6))

        ks = np.array(ks)
        ws = np.array(ws)
        wts = np.array(wts)
        if len(ks) < 6:
            for r in exp_rows:
                fitted[(exp, float(r["strike"]))] = {"iv_fit": float(r["iv"])}
            continue

        order = np.argsort(ks)
        ks_sorted = ks[order]
        ws_sorted = ws[order]
        wts_sorted = wts[order]
        spline = UnivariateSpline(ks_sorted, ws_sorted, w=wts_sorted, s=0.5 * len(ks_sorted))
        for r in exp_rows:
            F = float(r["underlying"])
            K = float(r["strike"])
            k = math.log(K / F)
            w_fit = max(float(spline(k)), 1.0e-8)
            iv_fit = math.sqrt(w_fit / T)
            fitted[(exp, float(r["strike"]))] = {"iv_fit": iv_fit}
    return fitted


def _curve_arrays_from(curve: Union[ZeroCurve, "pd.DataFrame"]) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore[name-defined]
    if isinstance(curve, ZeroCurve):
        return curve.terms.astype(float), curve.rates.astype(float)
    _ensure_pandas()
    curve_df = validate_curve_df(curve)  # type: ignore[arg-type]
    return curve_df["term"].to_numpy(dtype=float), curve_df["rate"].to_numpy(dtype=float)


def _iv_greeks_vector_py(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    price: np.ndarray,
    is_call: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = F.shape[0]
    iv = np.empty(n, dtype=np.float64)
    delta = np.empty(n, dtype=np.float64)
    gamma = np.empty(n, dtype=np.float64)
    vega = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    rho = np.empty(n, dtype=np.float64)
    for i in range(n):
        iv[i] = implied_vol_black76(price[i], F[i], K[i], T[i], r[i], bool(is_call[i]))
        greeks = black76_greeks(F[i], K[i], T[i], r[i], iv[i], bool(is_call[i]))
        delta[i] = greeks["delta"]
        gamma[i] = greeks["gamma"]
        vega[i] = greeks["vega"]
        theta[i] = greeks["theta"]
        rho[i] = greeks["rho"]
    return iv, delta, gamma, vega, theta, rho


def _price_vector_py(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    vol: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    n = F.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = black76_price(F[i], K[i], T[i], r[i], vol[i], bool(is_call[i]))
    return out


def compute_analytics_df(
    options_df: "pd.DataFrame",  # type: ignore[name-defined]
    curve: Union[ZeroCurve, "pd.DataFrame"],  # type: ignore[name-defined]
    use_numba: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    opts = validate_options_df(options_df)
    terms, rates = _curve_arrays_from(curve)

    date_ser = pd.to_datetime(opts["date"], errors="raise")  # type: ignore[union-attr]
    expiry_ser = pd.to_datetime(opts["expiry"], errors="raise")  # type: ignore[union-attr]
    T = (expiry_ser - date_ser).dt.days.to_numpy(dtype=float) / 365.0
    if np.any(T < 0.0):
        raise ValueError("options table has expiry earlier than date.")

    F = opts["underlying"].to_numpy(dtype=float)
    K = opts["strike"].to_numpy(dtype=float)
    price = opts["settle"].to_numpy(dtype=float)
    is_call = (opts["put_call"] == "C").to_numpy(dtype=np.int8)
    r = np.interp(T, terms, rates)

    use_nb = resolve_numba(use_numba)
    if use_nb:
        iv, delta, gamma, vega, theta, rho = _iv_greeks_vector_nb(F, K, T, r, price, is_call)
    else:
        iv, delta, gamma, vega, theta, rho = _iv_greeks_vector_py(F, K, T, r, price, is_call)

    out = opts.copy()
    out["T"] = T
    out["rate"] = r
    out["iv"] = iv
    out["delta"] = delta
    out["gamma"] = gamma
    out["vega"] = vega
    out["theta"] = theta
    out["rho"] = rho
    return out


def annotate_surface_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    use_numba: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = analytics_df.copy()
    df["iv_fit"] = np.nan

    for exp, idx in df.groupby("expiry").groups.items():
        sub = df.loc[idx]
        T = float(sub["T"].iloc[0])
        if T <= 0.0:
            df.loc[idx, "iv_fit"] = sub["iv"].to_numpy(dtype=float)
            continue
        F = sub["underlying"].to_numpy(dtype=float)
        K = sub["strike"].to_numpy(dtype=float)
        iv = sub["iv"].to_numpy(dtype=float)
        vega = sub["vega"].to_numpy(dtype=float)
        k = np.log(K / F)
        w = iv * iv * T
        if len(k) < 6:
            iv_fit = iv
        else:
            order = np.argsort(k)
            k_sorted = k[order]
            w_sorted = w[order]
            wts_sorted = np.maximum(vega, 1.0e-6)[order]
            spline = UnivariateSpline(k_sorted, w_sorted, w=wts_sorted, s=0.5 * len(k_sorted))
            w_fit = np.maximum(spline(k), 1.0e-8)
            iv_fit = np.sqrt(w_fit / T)
        df.loc[idx, "iv_fit"] = iv_fit

    df["iv_resid"] = df["iv"] - df["iv_fit"]

    def _zscore(series):
        mu = series.mean()
        sd = series.std()
        if not np.isfinite(sd) or sd <= 1.0e-8:
            return (series - mu) / 1.0
        return (series - mu) / sd

    df["iv_z"] = df.groupby("expiry")["iv_resid"].transform(_zscore)

    F = df["underlying"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    r = df["rate"].to_numpy(dtype=float)
    iv_fit = df["iv_fit"].to_numpy(dtype=float)
    is_call = (df["put_call"] == "C").to_numpy(dtype=np.int8)

    use_nb = resolve_numba(use_numba)
    if use_nb:
        theo = _price_vector_nb(F, K, T, r, iv_fit, is_call)
    else:
        theo = _price_vector_py(F, K, T, r, iv_fit, is_call)
    df["theo"] = theo
    return df


def _default_strategy_templates(widths: Sequence[float]) -> List[Dict[str, object]]:
    templates: List[Dict[str, object]] = [
        {
            "name": "straddle",
            "width": 0.0,
            "legs": [("C", 0.0, 1.0), ("P", 0.0, 1.0)],
        }
    ]
    for w in widths:
        w = float(w)
        templates.append(
            {
                "name": f"call_spread_{w:g}",
                "width": w,
                "legs": [("C", 0.0, 1.0), ("C", w, -1.0)],
            }
        )
        templates.append(
            {
                "name": f"put_spread_{w:g}",
                "width": w,
                "legs": [("P", 0.0, 1.0), ("P", -w, -1.0)],
            }
        )
        templates.append(
            {
                "name": f"strangle_{w:g}",
                "width": w,
                "legs": [("C", w, 1.0), ("P", -w, 1.0)],
            }
        )
        templates.append(
            {
                "name": f"risk_reversal_{w:g}",
                "width": w,
                "legs": [("C", w, 1.0), ("P", -w, -1.0)],
            }
        )
        templates.append(
            {
                "name": f"call_fly_{w:g}",
                "width": w,
                "legs": [("C", -w, 1.0), ("C", 0.0, -2.0), ("C", w, 1.0)],
            }
        )
        templates.append(
            {
                "name": f"put_fly_{w:g}",
                "width": w,
                "legs": [("P", -w, 1.0), ("P", 0.0, -2.0), ("P", w, 1.0)],
            }
        )
    return templates


def build_strategy_book(
    analytics: Union[List[Dict[str, object]], "pd.DataFrame"],  # type: ignore[name-defined]
    widths: Sequence[float] = (0.5, 1.0, 2.0),
    strategy_templates: Optional[List[Dict[str, object]]] = None,
    return_df: bool = True,
) -> Union[List[Dict[str, object]], "pd.DataFrame"]:  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(analytics, "columns"):
        df = build_strategy_book_df(analytics, widths=widths, strategy_templates=strategy_templates)  # type: ignore[arg-type]
        if return_df:
            return df
        return df.to_dict(orient="records")

    rows = _table_to_records(analytics)
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required to build strategy book from rows.")
    df = pd.DataFrame(rows)  # type: ignore[union-attr]
    df = build_strategy_book_df(df, widths=widths, strategy_templates=strategy_templates)
    if return_df:
        return df
    return df.to_dict(orient="records")


def build_strategy_book_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    widths: Sequence[float] = (0.5, 1.0, 2.0),
    strategy_templates: Optional[List[Dict[str, object]]] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df)
    templates = strategy_templates if strategy_templates is not None else _default_strategy_templates(widths)

    rows: List[Dict[str, object]] = []
    grouped = df.groupby(["date", "expiry"])

    for (dt, exp), sub in grouped:
        underlying = float(sub["underlying"].iloc[0])
        strike_vals = sub["strike"].to_numpy(dtype=float)
        atm_idx = int(np.argmin(np.abs(strike_vals - underlying)))
        atm = float(strike_vals[atm_idx])

        lookup: Dict[Tuple[float, str], pd.Series] = {}
        for _, row in sub.iterrows():
            key = (round(float(row["strike"]), 6), str(row["put_call"]))
            lookup[key] = row

        for tmpl in templates:
            legs = tmpl["legs"]
            legs_desc = []
            total_price = 0.0
            total_theo = 0.0
            total_delta = 0.0
            total_gamma = 0.0
            total_vega = 0.0
            total_theta = 0.0
            total_rho = 0.0
            resid_num = 0.0
            z_num = 0.0
            vega_denom = 0.0
            missing_leg = False

            for pc, offset, qty in legs:
                strike = round(atm + float(offset), 6)
                row = lookup.get((strike, str(pc)))
                if row is None:
                    missing_leg = True
                    break
                q = float(qty)
                legs_desc.append(f"{pc}@{offset:+g}x{q:g}")
                total_price += q * float(row["settle"])
                total_theo += q * float(row["theo"])
                total_delta += q * float(row["delta"])
                total_gamma += q * float(row["gamma"])
                total_vega += q * float(row["vega"])
                total_theta += q * float(row["theta"])
                total_rho += q * float(row["rho"])

                w = abs(q * float(row["vega"]))
                resid_num += q * float(row["vega"]) * float(row["iv_resid"])
                z_num += q * float(row["vega"]) * float(row["iv_z"])
                vega_denom += w

            if missing_leg:
                continue

            edge = total_theo - total_price
            edge_per_vega = edge / vega_denom if vega_denom > 1.0e-8 else 0.0
            iv_resid = resid_num / vega_denom if vega_denom > 1.0e-8 else 0.0
            iv_z = z_num / vega_denom if vega_denom > 1.0e-8 else 0.0

            rows.append(
                {
                    "date": dt,
                    "expiry": exp,
                    "strategy": str(tmpl["name"]),
                    "width": float(tmpl["width"]),
                    "atm_strike": atm,
                    "underlying": underlying,
                    "market_price": total_price,
                    "theo_price": total_theo,
                    "edge": edge,
                    "edge_per_vega": edge_per_vega,
                    "delta": total_delta,
                    "gamma": total_gamma,
                    "vega": total_vega,
                    "theta": total_theta,
                    "rho": total_rho,
                    "iv_resid": iv_resid,
                    "iv_z": iv_z,
                    "legs": ",".join(legs_desc),
                }
            )

    return pd.DataFrame(rows)


def analyze_chain_df(
    options_df: "pd.DataFrame",  # type: ignore[name-defined]
    curve: Union[ZeroCurve, "pd.DataFrame"],  # type: ignore[name-defined]
    use_numba: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    analytics_df = compute_analytics_df(options_df, curve, use_numba=use_numba)
    return annotate_surface_df(analytics_df, use_numba=use_numba)


def compute_analytics(
    chain_rows: TableLike,
    curve: ZeroCurve,
    use_numba: Optional[bool] = None,
) -> List[Dict[str, str]]:
    validated = validate_options_table(chain_rows)
    chain_records = _table_to_records(validated)
    use_nb = resolve_numba(use_numba)
    analytics: List[Dict[str, str]] = []

    for r in chain_records:
        F = float(r["underlying"])
        K = float(r["strike"])
        T = (date.fromisoformat(r["expiry"]) - date.fromisoformat(r["date"])).days / 365.0
        rate = curve.rate(T)
        is_call = r["put_call"] == "C"
        price = float(r["settle"])

        if use_nb:
            iv = implied_vol_bisect_nb(price, F, K, T, rate, 1 if is_call else 0)
            delta, gamma, vega, theta, rho = black76_greeks_nb(F, K, T, rate, iv, 1 if is_call else 0)
        else:
            iv = implied_vol_black76(price, F, K, T, rate, is_call)
            greeks = black76_greeks(F, K, T, rate, iv, is_call)
            delta, gamma, vega, theta, rho = (
                greeks["delta"],
                greeks["gamma"],
                greeks["vega"],
                greeks["theta"],
                greeks["rho"],
            )

        r2 = dict(r)
        r2.update(
            {
                "T": f"{T:.8f}",
                "rate": f"{rate:.6f}",
                "iv": f"{iv:.6f}",
                "delta": f"{delta:.8f}",
                "gamma": f"{gamma:.10f}",
                "vega": f"{vega:.8f}",
                "theta": f"{theta:.8f}",
                "rho": f"{rho:.8f}",
            }
        )
        analytics.append(r2)
    return analytics


def annotate_surface(
    analytics: List[Dict[str, str]],
    use_numba: Optional[bool] = None,
) -> List[Dict[str, str]]:
    use_nb = resolve_numba(use_numba)
    fitted = fit_surface_per_expiry(analytics)
    by_exp: Dict[str, List[Dict[str, str]]] = {}
    for r in analytics:
        by_exp.setdefault(r["expiry"], []).append(r)

    for exp, exp_rows in by_exp.items():
        resid = []
        for r in exp_rows:
            key = (exp, float(r["strike"]))
            iv_fit = fitted[key]["iv_fit"]
            r["iv_fit"] = f"{iv_fit:.6f}"
            iv = float(r["iv"])
            resid.append(iv - iv_fit)
        mu = float(np.mean(resid))
        sd = float(np.std(resid)) if np.std(resid) > 1.0e-8 else 1.0
        for r in exp_rows:
            iv = float(r["iv"])
            iv_fit = float(r["iv_fit"])
            z = (iv - iv_fit - mu) / sd
            r["iv_resid"] = f"{iv - iv_fit:.6f}"
            r["iv_z"] = f"{z:.4f}"

            F = float(r["underlying"])
            K = float(r["strike"])
            T = float(r["T"])
            rate = float(r["rate"])
            is_call = r["put_call"] == "C"
            if use_nb:
                theo = black76_price_nb(F, K, T, rate, iv_fit, 1 if is_call else 0)
            else:
                theo = black76_price(F, K, T, rate, iv_fit, is_call)
            r["theo"] = f"{theo:.6f}"
    return analytics


def analyze_chain(
    chain_rows: TableLike,
    curve: Union[ZeroCurve, TableLike],
    use_numba: Optional[bool] = None,
    return_df: bool = False,
) -> Union[List[Dict[str, str]], "pd.DataFrame"]:  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(chain_rows, "columns"):
        if isinstance(curve, ZeroCurve) or (HAVE_PANDAS and hasattr(curve, "columns")):
            curve_input = curve
        else:
            curve_input = curve_from_rows(curve)
        df = analyze_chain_df(chain_rows, curve_input, use_numba=use_numba)  # type: ignore[arg-type]
        if return_df:
            return df
        return df.to_dict(orient="records")

    curve_obj = curve if isinstance(curve, ZeroCurve) else curve_from_rows(curve)
    analytics = compute_analytics(chain_rows, curve_obj, use_numba=use_numba)
    analytics = annotate_surface(analytics, use_numba=use_numba)
    if return_df:
        return _records_to_df(analytics)
    return analytics


def analyze_chain_rows(
    chain_rows: TableLike,
    curve_rows: TableLike,
    use_numba: Optional[bool] = None,
    return_df: bool = False,
) -> Union[List[Dict[str, str]], "pd.DataFrame"]:  # type: ignore[name-defined]
    curve = curve_from_rows(curve_rows)
    return analyze_chain(chain_rows, curve, use_numba=use_numba, return_df=return_df)
