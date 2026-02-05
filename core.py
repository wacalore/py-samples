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


@dataclass
class ExpirySurface:
    expiry: object
    T: float
    spline: UnivariateSpline
    rmse: float
    n: int
    iv_atm: float
    skew: float
    curvature: float


@dataclass
class VolSurface:
    expiries: List[ExpirySurface]

    def _find_expiry(self, expiry: object) -> Optional[ExpirySurface]:
        for exp in self.expiries:
            if exp.expiry == expiry:
                return exp
        return None

    def iv(self, expiry: object, F: float, K: float) -> float:
        exp = self._find_expiry(expiry)
        if exp is None:
            raise KeyError("Expiry not found in surface.")
        k = math.log(K / F)
        w = float(exp.spline(k))
        w = max(w, 1.0e-8)
        return math.sqrt(w / exp.T)

    def metrics(self, expiry: object) -> Dict[str, float]:
        exp = self._find_expiry(expiry)
        if exp is None:
            raise KeyError("Expiry not found in surface.")
        return {
            "iv_atm": float(exp.iv_atm),
            "skew": float(exp.skew),
            "curvature": float(exp.curvature),
            "rmse": float(exp.rmse),
            "n": float(exp.n),
        }

    def iv_by_T(self, T: float, F: float, K: float) -> float:
        if not self.expiries:
            raise ValueError("Surface is empty.")
        if T <= 0.0:
            return 0.0
        exps = sorted(self.expiries, key=lambda x: x.T)
        if T <= exps[0].T:
            k = math.log(K / F)
            w = float(exps[0].spline(k))
            return math.sqrt(max(w, 1.0e-8) / exps[0].T)
        if T >= exps[-1].T:
            k = math.log(K / F)
            w = float(exps[-1].spline(k))
            return math.sqrt(max(w, 1.0e-8) / exps[-1].T)
        for lo, hi in zip(exps, exps[1:]):
            if lo.T <= T <= hi.T:
                k = math.log(K / F)
                w_lo = max(float(lo.spline(k)), 1.0e-8)
                w_hi = max(float(hi.spline(k)), 1.0e-8)
                w = w_lo + (w_hi - w_lo) * (T - lo.T) / (hi.T - lo.T)
                return math.sqrt(max(w, 1.0e-8) / T)
        return 0.0

    def summary_df(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        _ensure_pandas()
        return pd.DataFrame(  # type: ignore[union-attr]
            [
                {"expiry": e.expiry, "T": e.T, "rmse": e.rmse, "n": e.n}
                for e in self.expiries
            ]
        )


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

ANALYTICS_BASE_COLS = {
    "date",
    "expiry",
    "strike",
    "put_call",
    "settle",
    "underlying",
    "iv",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "rate",
    "T",
}
ANALYTICS_SURFACE_COLS = {
    "iv_fit",
    "iv_resid",
    "iv_z",
    "theo",
    "edge",
    "edge_per_vega",
}


def _missing_columns(cols: Iterable[str], required: Set[str]) -> List[str]:
    return sorted(required - set(cols))


def _ensure_pandas() -> None:
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for DataFrame inputs.")


def _surface_group_cols(df: "pd.DataFrame") -> List[str]:  # type: ignore[name-defined]
    if "date" in df.columns and df["date"].nunique() > 1:
        return ["date", "expiry"]
    return ["expiry"]


def _prepare_spline_inputs(
    k: np.ndarray,
    w: np.ndarray,
    wts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(k)
    k_sorted = k[order]
    w_sorted = w[order]
    wts_sorted = wts[order]

    if k_sorted.size == 0:
        return np.array([-1.0e-6, 1.0e-6]), np.array([1.0e-8, 1.0e-8]), np.array([1.0, 1.0])

    # Aggregate duplicate k values (common when both calls/puts share same strike).
    if np.any(np.diff(k_sorted) == 0.0):
        uniq_k: List[float] = []
        uniq_w: List[float] = []
        uniq_wts: List[float] = []
        i = 0
        n = k_sorted.size
        while i < n:
            k0 = float(k_sorted[i])
            j = i + 1
            while j < n and k_sorted[j] == k0:
                j += 1
            w_slice = w_sorted[i:j]
            wt_slice = wts_sorted[i:j]
            wt_sum = float(np.sum(wt_slice))
            if wt_sum <= 0.0:
                wt_sum = float(j - i)
                w_bar = float(np.mean(w_slice))
            else:
                w_bar = float(np.sum(w_slice * wt_slice) / wt_sum)
            uniq_k.append(k0)
            uniq_w.append(w_bar)
            uniq_wts.append(max(wt_sum, 1.0e-6))
            i = j
        k_sorted = np.array(uniq_k)
        w_sorted = np.array(uniq_w)
        wts_sorted = np.array(uniq_wts)

    if k_sorted.size < 2:
        k0 = float(k_sorted[0])
        w0 = float(w_sorted[0])
        k_sorted = np.array([k0 - 1.0e-6, k0 + 1.0e-6])
        w_sorted = np.array([max(w0, 1.0e-8), max(w0, 1.0e-8)])
        wts_sorted = np.array([1.0, 1.0])

    return k_sorted, w_sorted, wts_sorted


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


def validate_analytics_df(df: "pd.DataFrame", require_surface: bool = False) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    required = set(ANALYTICS_BASE_COLS)
    if require_surface:
        required |= set(ANALYTICS_SURFACE_COLS)
    missing = _missing_columns(df.columns, required)
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

    numeric_cols = [c for c in required if c not in {"date", "expiry", "put_call"}]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")  # type: ignore[union-attr]
        if not np.isfinite(out[col].to_numpy(dtype=float)).all():
            raise ValueError(f"analytics table column '{col}' must be finite.")

    out["date"] = pd.to_datetime(out["date"], errors="raise")  # type: ignore[union-attr]
    out["expiry"] = pd.to_datetime(out["expiry"], errors="raise")  # type: ignore[union-attr]
    return out


def compute_realized_vol_df(
    underlying_df: "pd.DataFrame",  # type: ignore[name-defined]
    lookback_days: int = 20,
    annualization: float = 252.0,
    price_col: str = "underlying",
    date_col: str = "date",
    method: str = "log",
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    if lookback_days < 2:
        raise ValueError("lookback_days must be >= 2.")
    df = underlying_df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")  # type: ignore[union-attr]
    df[price_col] = pd.to_numeric(df[price_col], errors="raise")  # type: ignore[union-attr]
    df = df.sort_values(date_col)

    prices = df[price_col].to_numpy(dtype=float)
    if method == "log":
        rets = np.diff(np.log(prices), prepend=np.nan)
    elif method == "simple":
        rets = np.diff(prices, prepend=np.nan) / np.concatenate([[np.nan], prices[:-1]])
    else:
        raise ValueError("method must be 'log' or 'simple'.")

    rets_ser = pd.Series(rets, index=df.index)
    rv = rets_ser.rolling(lookback_days, min_periods=lookback_days).std() * math.sqrt(float(annualization))
    out = df[[date_col]].copy()
    out["rv"] = rv.to_numpy(dtype=float)
    return out


def attach_realized_vol_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    rv_df: "pd.DataFrame",  # type: ignore[name-defined]
    rv_col: str = "rv",
    date_col: str = "date",
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = analytics_df.copy()
    rv = rv_df[[date_col, rv_col]].copy()
    rv[date_col] = pd.to_datetime(rv[date_col], errors="raise")  # type: ignore[union-attr]
    rv[rv_col] = pd.to_numeric(rv[rv_col], errors="raise")  # type: ignore[union-attr]
    out = df.merge(rv, how="left", on=date_col)

    denom = out[rv_col].where(out[rv_col].abs() > 1.0e-12)
    out["iv_minus_rv"] = out["iv"] - out[rv_col]
    out["iv_over_rv"] = out["iv"] / denom
    if "iv_fit" in out.columns:
        out["iv_fit_minus_rv"] = out["iv_fit"] - out[rv_col]
        out["iv_fit_over_rv"] = out["iv_fit"] / denom
    return out


def add_realized_vol_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    lookback_days: int = 20,
    annualization: float = 252.0,
    price_col: str = "underlying",
    date_col: str = "date",
    method: str = "log",
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    if date_col not in analytics_df.columns or price_col not in analytics_df.columns:
        raise ValueError(f"analytics table must include '{date_col}' and '{price_col}' columns.")
    unique_underlying = (
        analytics_df[[date_col, price_col]]
        .drop_duplicates(subset=[date_col])
        .sort_values(date_col)
    )
    rv_df = compute_realized_vol_df(
        unique_underlying,
        lookback_days=lookback_days,
        annualization=annualization,
        price_col=price_col,
        date_col=date_col,
        method=method,
    )
    return attach_realized_vol_df(analytics_df, rv_df, rv_col="rv", date_col=date_col)


def fit_surfaces_by_date_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    return_df: bool = False,
) -> Union[Dict[object, VolSurface], Tuple[Dict[object, VolSurface], "pd.DataFrame"]]:  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df)
    if "date" not in df.columns:
        raise ValueError("analytics table must include a 'date' column.")

    surfaces: Dict[object, VolSurface] = {}
    out_parts: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for dt, sub in df.groupby("date"):
        surface, sub_out = fit_surface_df(sub, return_df=True, group_by_date=False)
        surfaces[dt] = surface
        out_parts.append(sub_out)

    if return_df:
        out = pd.concat(out_parts, axis=0)  # type: ignore[union-attr]
        out = out.reindex(df.index)
        return surfaces, out
    return surfaces


def fit_surfaces_by_date_contract_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    contract_col: str = "underlying_ric",
    return_df: bool = False,
) -> Union[
    Dict[object, Dict[object, VolSurface]],
    Tuple[Dict[object, Dict[object, VolSurface]], "pd.DataFrame"],
]:  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df)
    if contract_col not in df.columns:
        raise ValueError(f"analytics table missing '{contract_col}' column.")

    surfaces: Dict[object, Dict[object, VolSurface]] = {}
    out_parts: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for (dt, contract), sub in df.groupby(["date", contract_col]):
        surface, sub_out = fit_surface_df(sub, return_df=True, group_by_date=False)
        surfaces.setdefault(dt, {})[contract] = surface
        out_parts.append(sub_out)

    if return_df:
        out = pd.concat(out_parts, axis=0)  # type: ignore[union-attr]
        out = out.reindex(df.index)
        return surfaces, out
    return surfaces


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

        ks_sorted, ws_sorted, wts_sorted = _prepare_spline_inputs(ks, ws, wts)
        k_spline = min(3, len(ks_sorted) - 1)
        smooth = 0.0 if len(ks_sorted) < 6 else 0.5 * len(ks_sorted)
        spline = UnivariateSpline(ks_sorted, ws_sorted, w=wts_sorted, k=k_spline, s=smooth)
        for r in exp_rows:
            F = float(r["underlying"])
            K = float(r["strike"])
            k = math.log(K / F)
            w_fit = max(float(spline(k)), 1.0e-8)
            iv_fit = math.sqrt(w_fit / T)
            fitted[(exp, float(r["strike"]))] = {"iv_fit": iv_fit}
    return fitted


def fit_surface_rows(rows: List[Dict[str, object]]) -> VolSurface:
    by_exp: Dict[object, List[Dict[str, object]]] = {}
    for r in rows:
        by_exp.setdefault(r["expiry"], []).append(r)

    expiries: List[ExpirySurface] = []
    for exp, exp_rows in by_exp.items():
        T = float(exp_rows[0]["T"])
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

        ks_arr = np.array(ks)
        ws_arr = np.array(ws)
        wts_arr = np.array(wts)
        ks_sorted, ws_sorted, wts_sorted = _prepare_spline_inputs(ks_arr, ws_arr, wts_arr)
        k_spline = min(3, len(ks_sorted) - 1)
        smooth = 0.0 if len(ks_sorted) < 6 else 0.5 * len(ks_sorted)
        spline = UnivariateSpline(ks_sorted, ws_sorted, w=wts_sorted, k=k_spline, s=smooth)
        w_fit = np.maximum(spline(ks_arr), 1.0e-8)
        iv_fit = np.sqrt(w_fit / T)
        rmse = float(np.sqrt(np.mean((np.array([float(r["iv"]) for r in exp_rows]) - iv_fit) ** 2)))

        dk = 1.0e-3
        w_m = float(spline(-dk))
        w_0 = float(spline(0.0))
        w_p = float(spline(dk))
        w_0 = max(w_0, 1.0e-8)
        iv_atm = math.sqrt(w_0 / T)
        w1 = (w_p - w_m) / (2.0 * dk)
        w2 = (w_p - 2.0 * w_0 + w_m) / (dk * dk)
        denom = max(2.0 * T * iv_atm, 1.0e-12)
        skew = w1 / denom
        curv_denom = max(4.0 * T * T * iv_atm * iv_atm * iv_atm, 1.0e-12)
        curvature = (w2 / denom) - (w1 * w1) / curv_denom

        expiries.append(
            ExpirySurface(
                expiry=exp,
                T=T,
                spline=spline,
                rmse=rmse,
                n=len(ks_sorted),
                iv_atm=iv_atm,
                skew=skew,
                curvature=curvature,
            )
        )

    return VolSurface(expiries=expiries)


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
    breakeven = np.where(is_call == 1, K + price, K - price)
    out["breakeven"] = breakeven
    out["breakeven_move"] = np.abs(breakeven - F)
    return out


def annotate_surface_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    use_numba: Optional[bool] = None,
    group_by_date: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    surface, df = fit_surface_df(analytics_df, return_df=True, group_by_date=group_by_date)
    df["iv_resid"] = df["iv"] - df["iv_fit"]

    if group_by_date is None:
        group_cols = _surface_group_cols(df)
    else:
        group_cols = ["date", "expiry"] if group_by_date else ["expiry"]

    def _zscore(series):
        mu = series.mean()
        sd = series.std()
        if not np.isfinite(sd) or sd <= 1.0e-8:
            return (series - mu) / 1.0
        return (series - mu) / sd

    df["iv_z"] = df.groupby(group_cols)["iv_resid"].transform(_zscore)
    df["iv_rank"] = df.groupby(group_cols)["iv_resid"].rank(method="average")
    df["iv_percentile"] = df.groupby(group_cols)["iv_resid"].rank(pct=True, method="average")

    metrics_rows: List[Dict[str, object]] = []
    for exp in surface.expiries:
        if isinstance(exp.expiry, tuple) and len(exp.expiry) == 2:
            dt_key, exp_key = exp.expiry
        else:
            dt_key, exp_key = None, exp.expiry
        metrics_rows.append(
            {
                "date": dt_key,
                "expiry": exp_key,
                "iv_atm": exp.iv_atm,
                "skew": exp.skew,
                "curvature": exp.curvature,
                "surface_rmse": exp.rmse,
                "surface_n": exp.n,
            }
        )
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        if group_cols == ["expiry"]:
            df = df.merge(metrics_df.drop(columns=["date"]), on="expiry", how="left", sort=False)
        else:
            df = df.merge(metrics_df, on=["date", "expiry"], how="left", sort=False)

    F = df["underlying"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    moneyness = np.divide(K, F, out=np.full_like(K, np.nan), where=F != 0.0)
    df["moneyness"] = moneyness
    log_moneyness = np.full_like(moneyness, np.nan)
    mask = moneyness > 0.0
    log_moneyness[mask] = np.log(moneyness[mask])
    df["log_moneyness"] = log_moneyness

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
    df["edge"] = df["theo"] - df["settle"]
    vega = df["vega"].to_numpy(dtype=float)
    edge = df["edge"].to_numpy(dtype=float)
    edge_per_vega = np.where(np.abs(vega) > 1.0e-8, edge / vega, 0.0)
    df["edge_per_vega"] = edge_per_vega
    df["edge_z"] = df.groupby(group_cols)["edge_per_vega"].transform(_zscore)
    df["edge_percentile"] = df.groupby(group_cols)["edge_per_vega"].rank(pct=True, method="average")
    return df


def fit_surface_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    return_df: bool = False,
    group_by_date: Optional[bool] = None,
) -> Union[VolSurface, Tuple[VolSurface, "pd.DataFrame"]]:  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df)
    out = df.copy()
    out["iv_fit"] = np.nan

    if group_by_date is None:
        group_cols = _surface_group_cols(out)
    else:
        group_cols = ["date", "expiry"] if group_by_date else ["expiry"]

    expiries: List[ExpirySurface] = []
    grouped = out.groupby(group_cols)
    for key, idx in grouped.groups.items():
        sub = out.loc[idx]
        T = float(sub["T"].iloc[0])
        if T <= 0.0:
            out.loc[idx, "iv_fit"] = sub["iv"].to_numpy(dtype=float)
            continue

        F = sub["underlying"].to_numpy(dtype=float)
        K = sub["strike"].to_numpy(dtype=float)
        iv = sub["iv"].to_numpy(dtype=float)
        vega = sub["vega"].to_numpy(dtype=float)
        k = np.log(K / F)
        w = iv * iv * T
        wts = np.maximum(vega, 1.0e-6)

        k_sorted, w_sorted, wts_sorted = _prepare_spline_inputs(k, w, wts)
        k_spline = min(3, len(k_sorted) - 1)
        smooth = 0.0 if len(k_sorted) < 6 else 0.5 * len(k_sorted)
        spline = UnivariateSpline(k_sorted, w_sorted, w=wts_sorted, k=k_spline, s=smooth)
        w_fit = np.maximum(spline(k), 1.0e-8)
        iv_fit = np.sqrt(w_fit / T)
        rmse = float(np.sqrt(np.mean((iv - iv_fit) ** 2))) if len(iv) else 0.0

        out.loc[idx, "iv_fit"] = iv_fit

        dk = 1.0e-3
        w_m = float(spline(-dk))
        w_0 = float(spline(0.0))
        w_p = float(spline(dk))
        w_0 = max(w_0, 1.0e-8)
        iv_atm = math.sqrt(w_0 / T)
        w1 = (w_p - w_m) / (2.0 * dk)
        w2 = (w_p - 2.0 * w_0 + w_m) / (dk * dk)
        denom = max(2.0 * T * iv_atm, 1.0e-12)
        skew = w1 / denom
        curv_denom = max(4.0 * T * T * iv_atm * iv_atm * iv_atm, 1.0e-12)
        curvature = (w2 / denom) - (w1 * w1) / curv_denom

        expiries.append(
            ExpirySurface(
                expiry=key,
                T=T,
                spline=spline,
                rmse=rmse,
                n=len(k_sorted),
                iv_atm=iv_atm,
                skew=skew,
                curvature=curvature,
            )
        )

    surface = VolSurface(expiries=expiries)
    if return_df:
        return surface, out
    return surface


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


def scenario_pnl_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    dF: float = 0.0,
    dVol: float = 0.0,
    dRate: float = 0.0,
    dt_days: float = 0.0,
    use_surface_iv: bool = True,
    vol_bump_type: str = "add",
    use_numba: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df, require_surface=use_surface_iv)
    base_vol = df["iv_fit"] if use_surface_iv else df["iv"]
    base_price = df["theo"] if use_surface_iv else df["settle"]

    F = df["underlying"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    r = df["rate"].to_numpy(dtype=float)
    is_call = (df["put_call"] == "C").to_numpy(dtype=np.int8)

    dt_years = float(dt_days) / 365.0
    T_new = np.maximum(T - dt_years, 1.0e-8)
    F_new = F + float(dF)
    r_new = r + float(dRate)
    if vol_bump_type == "mult":
        vol_new = base_vol.to_numpy(dtype=float) * (1.0 + float(dVol))
    else:
        vol_new = base_vol.to_numpy(dtype=float) + float(dVol)
    vol_new = np.maximum(vol_new, 1.0e-6)

    use_nb = resolve_numba(use_numba)
    if use_nb:
        scen_price = _price_vector_nb(F_new, K, T_new, r_new, vol_new, is_call)
    else:
        scen_price = _price_vector_py(F_new, K, T_new, r_new, vol_new, is_call)

    base_price_arr = np.asarray(base_price, dtype=float)
    pnl = scen_price - base_price_arr

    out = df.copy()
    out["scenario_price"] = scen_price
    out["pnl"] = pnl

    # Greeks-based attribution (1st/2nd order)
    delta = out["delta"].to_numpy(dtype=float)
    gamma = out["gamma"].to_numpy(dtype=float)
    vega = out["vega"].to_numpy(dtype=float)
    theta = out["theta"].to_numpy(dtype=float)
    rho = out["rho"].to_numpy(dtype=float)

    pnl_delta = delta * dF
    pnl_gamma = 0.5 * gamma * (dF ** 2)
    pnl_vega = vega * dVol if vol_bump_type == "add" else vega * (float(dVol) * base_vol.to_numpy(dtype=float))
    pnl_rho = rho * dRate
    pnl_theta = theta * dt_years
    pnl_greeks = pnl_delta + pnl_gamma + pnl_vega + pnl_rho + pnl_theta

    out["pnl_delta"] = pnl_delta
    out["pnl_gamma"] = pnl_gamma
    out["pnl_vega"] = pnl_vega
    out["pnl_rho"] = pnl_rho
    out["pnl_theta"] = pnl_theta
    out["pnl_greeks"] = pnl_greeks
    out["pnl_residual"] = pnl - pnl_greeks
    return out


def scenario_pnl_strategy_df(
    strategy_df: "pd.DataFrame",  # type: ignore[name-defined]
    dF: float = 0.0,
    dVol: float = 0.0,
    dRate: float = 0.0,
    dt_days: float = 0.0,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = strategy_df.copy()
    required = {"delta", "gamma", "vega", "theta", "rho"}
    missing = _missing_columns(df.columns, required)
    if missing:
        raise ValueError(f"strategy table missing required columns: {', '.join(missing)}")

    dt_years = float(dt_days) / 365.0
    pnl_delta = df["delta"].to_numpy(dtype=float) * dF
    pnl_gamma = 0.5 * df["gamma"].to_numpy(dtype=float) * (dF ** 2)
    pnl_vega = df["vega"].to_numpy(dtype=float) * dVol
    pnl_rho = df["rho"].to_numpy(dtype=float) * dRate
    pnl_theta = df["theta"].to_numpy(dtype=float) * dt_years
    pnl_greeks = pnl_delta + pnl_gamma + pnl_vega + pnl_rho + pnl_theta

    df["pnl_delta"] = pnl_delta
    df["pnl_gamma"] = pnl_gamma
    df["pnl_vega"] = pnl_vega
    df["pnl_rho"] = pnl_rho
    df["pnl_theta"] = pnl_theta
    df["pnl_greeks"] = pnl_greeks
    return df


def _normalize_scenarios(
    scenarios: Sequence[object],
) -> List[Dict[str, object]]:
    if not scenarios:
        raise ValueError("scenarios must be a non-empty sequence.")
    normalized: List[Dict[str, object]] = []
    for idx, sc in enumerate(scenarios):
        if isinstance(sc, dict):
            dF = float(sc.get("dF", 0.0))
            dVol = float(sc.get("dVol", 0.0))
            dRate = float(sc.get("dRate", 0.0))
            dt_days = float(sc.get("dt_days", 0.0))
            name = sc.get("name")
        else:
            if not isinstance(sc, (list, tuple)) or len(sc) not in (2, 3, 4):
                raise ValueError("scenario must be dict or tuple (dF, dVol[, dRate[, dt_days]]).")
            dF = float(sc[0])
            dVol = float(sc[1])
            dRate = float(sc[2]) if len(sc) >= 3 else 0.0
            dt_days = float(sc[3]) if len(sc) == 4 else 0.0
            name = None
        normalized.append(
            {
                "scenario_id": idx,
                "scenario_name": str(name) if name is not None else f"scen_{idx}",
                "dF": dF,
                "dVol": dVol,
                "dRate": dRate,
                "dt_days": dt_days,
            }
        )
    return normalized


def scenario_grid_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    scenarios: Sequence[object],
    agg_by: Sequence[str] = ("expiry", "strike"),
    use_surface_iv: bool = True,
    vol_bump_type: str = "add",
    use_numba: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df, require_surface=use_surface_iv)
    normalized = _normalize_scenarios(scenarios)

    group_cols = list(agg_by)
    if "date" in df.columns and "date" not in group_cols:
        group_cols = ["date"] + group_cols

    missing = _missing_columns(df.columns, set(group_cols))
    if missing:
        raise ValueError(f"analytics table missing aggregation columns: {', '.join(missing)}")

    pnl_cols = [
        "pnl",
        "pnl_delta",
        "pnl_gamma",
        "pnl_vega",
        "pnl_theta",
        "pnl_rho",
        "pnl_greeks",
        "pnl_residual",
    ]
    greek_cols = ["delta", "gamma", "vega", "theta", "rho"]

    outputs: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for sc in normalized:
        scen_df = scenario_pnl_df(
            df,
            dF=float(sc["dF"]),
            dVol=float(sc["dVol"]),
            dRate=float(sc["dRate"]),
            dt_days=float(sc["dt_days"]),
            use_surface_iv=use_surface_iv,
            vol_bump_type=vol_bump_type,
            use_numba=use_numba,
        )

        cols = [c for c in pnl_cols + greek_cols if c in scen_df.columns]
        agg = scen_df.groupby(group_cols, as_index=False)[cols].sum()
        agg["scenario_id"] = int(sc["scenario_id"])
        agg["scenario_name"] = str(sc["scenario_name"])
        agg["dF"] = float(sc["dF"])
        agg["dVol"] = float(sc["dVol"])
        agg["dRate"] = float(sc["dRate"])
        agg["dt_days"] = float(sc["dt_days"])
        outputs.append(agg)

    return pd.concat(outputs, axis=0, ignore_index=True)  # type: ignore[union-attr]


def scenario_grid_strategy_df(
    strategy_df: "pd.DataFrame",  # type: ignore[name-defined]
    scenarios: Sequence[object],
    agg_by: Sequence[str] = ("expiry", "strategy"),
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = strategy_df.copy()
    required = {"delta", "gamma", "vega", "theta", "rho"}
    missing = _missing_columns(df.columns, required)
    if missing:
        raise ValueError(f"strategy table missing required columns: {', '.join(missing)}")

    normalized = _normalize_scenarios(scenarios)
    group_cols = list(agg_by)
    if "date" in df.columns and "date" not in group_cols:
        group_cols = ["date"] + group_cols

    missing_group = _missing_columns(df.columns, set(group_cols))
    if missing_group:
        raise ValueError(f"strategy table missing aggregation columns: {', '.join(missing_group)}")

    pnl_cols = [
        "pnl_delta",
        "pnl_gamma",
        "pnl_vega",
        "pnl_theta",
        "pnl_rho",
        "pnl_greeks",
    ]
    greek_cols = ["delta", "gamma", "vega", "theta", "rho"]

    outputs: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for sc in normalized:
        scen_df = scenario_pnl_strategy_df(
            df,
            dF=float(sc["dF"]),
            dVol=float(sc["dVol"]),
            dRate=float(sc["dRate"]),
            dt_days=float(sc["dt_days"]),
        )
        cols = [c for c in pnl_cols + greek_cols if c in scen_df.columns]
        agg = scen_df.groupby(group_cols, as_index=False)[cols].sum()
        agg["scenario_id"] = int(sc["scenario_id"])
        agg["scenario_name"] = str(sc["scenario_name"])
        agg["dF"] = float(sc["dF"])
        agg["dVol"] = float(sc["dVol"])
        agg["dRate"] = float(sc["dRate"])
        agg["dt_days"] = float(sc["dt_days"])
        outputs.append(agg)

    return pd.concat(outputs, axis=0, ignore_index=True)  # type: ignore[union-attr]


def scenario_panel_backtest_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    scenarios: Sequence[object],
    lookback_days: int = 20,
    agg_by: Sequence[str] = ("expiry", "strike"),
    use_surface_iv: bool = True,
    vol_bump_type: str = "add",
    use_numba: Optional[bool] = None,
    reduce: str = "sum",
    return_surfaces: bool = False,
) -> Union["pd.DataFrame", Tuple["pd.DataFrame", Dict[object, VolSurface]]]:  # type: ignore[name-defined]
    _ensure_pandas()
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1.")

    df = analytics_df
    if use_surface_iv and "iv_fit" not in df.columns:
        df = annotate_surface_df(df, use_numba=use_numba, group_by_date=True)

    surfaces_by_date = fit_surfaces_by_date_df(df, return_df=False)
    grid = scenario_grid_df(
        df,
        scenarios=scenarios,
        agg_by=agg_by,
        use_surface_iv=use_surface_iv,
        vol_bump_type=vol_bump_type,
        use_numba=use_numba,
    )

    if "date" not in grid.columns:
        raise ValueError("scenario grid must include a 'date' column for rolling windows.")

    if lookback_days == 1:
        return (grid, surfaces_by_date) if return_surfaces else grid

    if reduce not in {"sum", "mean"}:
        raise ValueError("reduce must be 'sum' or 'mean'.")

    roll_cols = [
        "pnl",
        "pnl_delta",
        "pnl_gamma",
        "pnl_vega",
        "pnl_theta",
        "pnl_rho",
        "pnl_greeks",
        "pnl_residual",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
    roll_cols = [c for c in roll_cols if c in grid.columns]
    suffix = f"_roll{lookback_days}"

    out = grid.copy()
    for col in roll_cols:
        out[f"{col}{suffix}"] = np.nan

    group_cols = ["scenario_id", "scenario_name"] + list(agg_by)
    for _, sub in out.groupby(group_cols):
        sub = sub.sort_values("date")
        rolling = sub[roll_cols].rolling(lookback_days, min_periods=lookback_days)
        roll_vals = rolling.sum() if reduce == "sum" else rolling.mean()
        out.loc[sub.index, [f"{c}{suffix}" for c in roll_cols]] = roll_vals.values

    if return_surfaces:
        return out, surfaces_by_date
    return out


def _parse_strategy_legs(legs: str) -> List[Tuple[str, float, float]]:
    parsed: List[Tuple[str, float, float]] = []
    if not legs:
        return parsed
    for part in str(legs).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            pc, rest = part.split("@", 1)
            offset_str, qty_str = rest.split("x", 1)
            offset = float(offset_str)
            qty = float(qty_str)
            parsed.append((pc.strip().upper(), offset, qty))
        except Exception:
            continue
    return parsed


def _option_breakeven(strike: float, premium: float, is_call: bool) -> float:
    return strike + premium if is_call else strike - premium


def _strategy_breakevens(
    legs: List[Tuple[str, float, float]],
    atm: float,
    premium: float,
) -> List[float]:
    if not legs:
        return []

    strikes: Dict[float, float] = {}
    sum_put_qty = 0.0
    sum_put_qtyK = 0.0
    for pc, offset, qty in legs:
        strike = float(atm + float(offset))
        q = float(qty)
        strikes[strike] = strikes.get(strike, 0.0) + q
        if pc.upper() == "P":
            sum_put_qty += q
            sum_put_qtyK += q * strike

    sorted_strikes = sorted(strikes.keys())
    if not sorted_strikes:
        return []

    a = -sum_put_qty
    b = sum_put_qtyK
    solutions: List[float] = []

    def _solve_interval(a_: float, b_: float, lo: Optional[float], hi: Optional[float]) -> None:
        if abs(a_) < 1.0e-12:
            return
        s = (premium - b_) / a_
        if lo is not None and s < lo - 1.0e-9:
            return
        if hi is not None and s > hi + 1.0e-9:
            return
        solutions.append(float(s))

    # Interval below the first strike
    _solve_interval(a, b, None, sorted_strikes[0])

    # Sweep across strikes
    for i, strike in enumerate(sorted_strikes):
        total_q = strikes[strike]
        a += total_q
        b -= total_q * strike
        hi = sorted_strikes[i + 1] if i + 1 < len(sorted_strikes) else None
        _solve_interval(a, b, strike, hi)

    # Deduplicate near-equal solutions
    solutions_sorted = sorted(solutions)
    deduped: List[float] = []
    for s in solutions_sorted:
        if not deduped or abs(s - deduped[-1]) > 1.0e-6:
            deduped.append(s)
    return deduped


def strategy_screener_df(
    strategy_df: "pd.DataFrame",  # type: ignore[name-defined]
    analytics_df: Optional["pd.DataFrame"] = None,  # type: ignore[name-defined]
    vol_col: str = "iv_atm",
    vol_fallback: Optional[float] = None,
    pop_samples: int = 5000,
    pop_seed: int = 7,
    mispricing_metric: str = "edge_per_vega",
    mispricing_quantile: float = 0.9,
    pop_threshold: float = 0.6,
    ev_threshold: float = 0.0,
    upside_metric: str = "upside_p95",
    upside_quantile: float = 0.8,
    upside_threshold: Optional[float] = None,
    credit_debit_tolerance: float = 1.0e-8,
    filter_only: bool = True,
    top_n: Optional[int] = None,
    weight_mispricing: float = 1.0,
    weight_ev: float = 0.5,
    weight_pop: float = 0.5,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = strategy_df.copy()
    required = {"date", "expiry", "market_price", "theo_price", "edge", "edge_per_vega", "legs", "atm_strike", "underlying"}
    missing = _missing_columns(df.columns, required)
    if missing:
        raise ValueError(f"strategy table missing required columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")  # type: ignore[union-attr]
    df["expiry"] = pd.to_datetime(df["expiry"], errors="raise")  # type: ignore[union-attr]
    T = (df["expiry"] - df["date"]).dt.days.to_numpy(dtype=float) / 365.0
    T = np.maximum(T, 1.0e-8)
    df["T"] = T

    if analytics_df is not None:
        adf = analytics_df.copy()
        if vol_col not in adf.columns:
            adf = annotate_surface_df(adf, group_by_date=True)
        if vol_col not in adf.columns:
            raise ValueError(f"analytics_df does not contain '{vol_col}'.")
        vol_map = adf.groupby(["date", "expiry"], as_index=False)[vol_col].first()
        df = df.merge(vol_map, on=["date", "expiry"], how="left", sort=False)
    elif vol_fallback is not None:
        df[vol_col] = float(vol_fallback)
    else:
        raise ValueError("Provide analytics_df with a vol column or set vol_fallback.")

    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")  # type: ignore[union-attr]
    if df[vol_col].isna().any():
        raise ValueError(f"strategy table has missing '{vol_col}' values.")

    # Mispricing flags
    metric = df[mispricing_metric] if mispricing_metric in df.columns else df["edge_per_vega"]
    metric_abs = metric.abs()
    threshold = metric_abs.quantile(mispricing_quantile)
    df["flag_mispricing"] = metric_abs >= threshold

    # Credit/debit sanity checks
    tol = float(credit_debit_tolerance)
    df["flag_credit_when_debit"] = (df["market_price"] < -tol) & (df["theo_price"] > tol)
    df["flag_debit_when_credit"] = (df["market_price"] > tol) & (df["theo_price"] < -tol)

    # Probability of profit + expected value via lognormal MC
    rng = np.random.default_rng(pop_seed)
    z = rng.standard_normal(int(pop_samples))
    pops = np.zeros(len(df), dtype=float)
    evs = np.zeros(len(df), dtype=float)
    up_p95 = np.zeros(len(df), dtype=float)
    up_mean = np.zeros(len(df), dtype=float)

    legs_parsed = [_parse_strategy_legs(s) for s in df["legs"].tolist()]
    for i, legs in enumerate(legs_parsed):
        if not legs:
            pops[i] = np.nan
            evs[i] = np.nan
            up_p95[i] = np.nan
            up_mean[i] = np.nan
            continue

        F0 = float(df["underlying"].iloc[i])
        atm = float(df["atm_strike"].iloc[i])
        vol = float(df[vol_col].iloc[i])
        t = float(T[i])

        sigma = max(vol, 1.0e-8)
        fwd = F0 * np.exp(-0.5 * sigma * sigma * t + sigma * math.sqrt(t) * z)

        payoff = np.zeros_like(fwd)
        for pc, offset, qty in legs:
            strike = atm + float(offset)
            if pc == "C":
                payoff += float(qty) * np.maximum(fwd - strike, 0.0)
            else:
                payoff += float(qty) * np.maximum(strike - fwd, 0.0)

        premium = float(df["market_price"].iloc[i])
        profit = payoff - premium
        pops[i] = float(np.mean(profit > 0.0))
        evs[i] = float(np.mean(profit))
        up_p95[i] = float(np.quantile(profit, 0.95))
        up_mean[i] = float(np.mean(np.maximum(profit, 0.0)))

    df["pop"] = pops
    df["ev"] = evs
    df["upside_p95"] = up_p95
    df["upside_mean"] = up_mean

    df["flag_pop"] = df["pop"] >= float(pop_threshold)
    df["flag_ev"] = df["ev"] >= float(ev_threshold)

    if upside_metric not in df.columns:
        upside_metric = "upside_p95"
    if upside_threshold is None:
        up_thresh = df[upside_metric].quantile(float(upside_quantile))
    else:
        up_thresh = float(upside_threshold)
    df["flag_upside"] = df[upside_metric] >= up_thresh

    # Combined score (rank-based)
    score = np.zeros(len(df), dtype=float)
    score += float(weight_mispricing) * metric_abs.rank(pct=True).to_numpy(dtype=float)
    score += float(weight_ev) * df["ev"].rank(pct=True).to_numpy(dtype=float)
    score += float(weight_pop) * df["pop"].to_numpy(dtype=float)
    df["score"] = score

    if filter_only:
        mask = (
            df["flag_mispricing"]
            | df["flag_upside"]
            | df["flag_credit_when_debit"]
            | df["flag_debit_when_credit"]
        )
        df = df.loc[mask].copy()

    if top_n is not None and top_n > 0:
        df = df.sort_values("score", ascending=False).head(int(top_n)).copy()
    else:
        df = df.sort_values("score", ascending=False)

    return df


def build_strategy_book_df(
    analytics_df: "pd.DataFrame",  # type: ignore[name-defined]
    widths: Sequence[float] = (0.5, 1.0, 2.0),
    strategy_templates: Optional[List[Dict[str, object]]] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    df = validate_analytics_df(analytics_df, require_surface=True)
    templates = strategy_templates if strategy_templates is not None else _default_strategy_templates(widths)

    rows: List[Dict[str, object]] = []
    group_cols = ["date", "expiry"]
    if "underlying_ric" in df.columns:
        group_cols.append("underlying_ric")
    grouped = df.groupby(group_cols)

    for key, sub in grouped:
        if len(group_cols) == 3:
            dt, exp, ric = key
        else:
            dt, exp = key
            ric = sub["underlying_ric"].iloc[0] if "underlying_ric" in sub.columns else None
        underlying = float(sub["underlying"].iloc[0])
        underlying_ric = ric
        sub = sub.reset_index(drop=True)
        strike_vals = sub["strike"].to_numpy(dtype=float)
        atm_idx = int(np.argmin(np.abs(strike_vals - underlying)))
        atm = float(strike_vals[atm_idx])

        pc_vals = sub["put_call"].astype(str).to_numpy()
        settle_vals = sub["settle"].to_numpy(dtype=float)
        theo_vals = sub["theo"].to_numpy(dtype=float)
        delta_vals = sub["delta"].to_numpy(dtype=float)
        gamma_vals = sub["gamma"].to_numpy(dtype=float)
        vega_vals = sub["vega"].to_numpy(dtype=float)
        theta_vals = sub["theta"].to_numpy(dtype=float)
        rho_vals = sub["rho"].to_numpy(dtype=float)
        resid_vals = sub["iv_resid"].to_numpy(dtype=float)
        z_vals = sub["iv_z"].to_numpy(dtype=float)
        lookup = {(round(strike_vals[i], 6), pc_vals[i]): i for i in range(len(sub))}

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
                idx_row = lookup.get((strike, str(pc)))
                if idx_row is None:
                    missing_leg = True
                    break
                q = float(qty)
                legs_desc.append(f"{pc}@{offset:+g}x{q:g}")
                total_price += q * settle_vals[idx_row]
                total_theo += q * theo_vals[idx_row]
                total_delta += q * delta_vals[idx_row]
                total_gamma += q * gamma_vals[idx_row]
                total_vega += q * vega_vals[idx_row]
                total_theta += q * theta_vals[idx_row]
                total_rho += q * rho_vals[idx_row]

                w = abs(q * vega_vals[idx_row])
                resid_num += q * vega_vals[idx_row] * resid_vals[idx_row]
                z_num += q * vega_vals[idx_row] * z_vals[idx_row]
                vega_denom += w

            if missing_leg:
                continue

            edge = total_theo - total_price
            edge_per_vega = edge / vega_denom if vega_denom > 1.0e-8 else 0.0
            iv_resid = resid_num / vega_denom if vega_denom > 1.0e-8 else 0.0
            iv_z = z_num / vega_denom if vega_denom > 1.0e-8 else 0.0

            breakevens = _strategy_breakevens(legs, atm, total_price)
            if breakevens:
                breakeven_low = float(breakevens[0])
                breakeven_high = float(breakevens[-1])
                breakeven_move = min(
                    abs(breakeven_low - underlying),
                    abs(breakeven_high - underlying),
                )
            else:
                breakeven_low = float("nan")
                breakeven_high = float("nan")
                breakeven_move = float("nan")

            rows.append(
                {
                    "date": dt,
                    "expiry": exp,
                    "strategy": str(tmpl["name"]),
                    "width": float(tmpl["width"]),
                    "atm_strike": atm,
                    "underlying": underlying,
                    "underlying_ric": underlying_ric,
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
                    "breakeven_low": breakeven_low,
                    "breakeven_high": breakeven_high,
                    "breakeven_move": breakeven_move,
                    "legs": ",".join(legs_desc),
                }
            )

    return pd.DataFrame(rows)


def analyze_chain_df(
    options_df: "pd.DataFrame",  # type: ignore[name-defined]
    curve: Union[ZeroCurve, "pd.DataFrame"],  # type: ignore[name-defined]
    use_numba: Optional[bool] = None,
    group_by_date: Optional[bool] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    analytics_df = compute_analytics_df(options_df, curve, use_numba=use_numba)
    return annotate_surface_df(analytics_df, use_numba=use_numba, group_by_date=group_by_date)


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
        breakeven = _option_breakeven(K, price, is_call)
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
                "breakeven": f"{breakeven:.6f}",
                "breakeven_move": f"{abs(breakeven - F):.6f}",
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
            edge = theo - float(r["settle"])
            vega = float(r["vega"])
            r["edge"] = f"{edge:.6f}"
            r["edge_per_vega"] = f"{edge / vega if abs(vega) > 1.0e-8 else 0.0:.6f}"
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
