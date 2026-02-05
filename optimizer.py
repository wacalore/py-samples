#!/usr/bin/env python3
"""
Alpha portfolio optimizer with PCA-aware constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

try:
    import pandas as pd  # type: ignore[import-not-found]

    HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore[assignment]
    HAVE_PANDAS = False


@dataclass
class AlphaOptimizerConfig:
    dt_col: str = "dt"
    sym_col: str = "sym"
    signal_col: str = "sig"
    prev_signal_col: str = "prevSig"
    px_diff_col: str = "pxDiff"
    return_col: Optional[str] = None
    use_prev_signal: bool = True

    window: int = 252
    min_periods: Optional[int] = None
    annualization: float = 252.0

    weight_bounds: Tuple[float, float] = (-1.0, 1.0)
    sum_to_one: bool = True
    risk_free: float = 0.0

    n_components: Optional[int] = None
    variance_threshold: float = 0.9
    pc_exposure_limits: Optional[Union[float, Sequence[float], Dict[int, float]]] = None
    pc_exposure_targets: Optional[Union[float, Sequence[float], Dict[int, float]]] = None
    pc_target_tolerance: Optional[Union[float, Sequence[float], Dict[int, float]]] = None
    pc_target_penalty: float = 1.0

    fillna: float = 0.0
    pca_mode: str = "full"  # "full" or "rolling"

    target_vol: Optional[float] = None
    target_vol_penalty: float = 10.0
    turnover_penalty: float = 0.0


def _ensure_pandas() -> None:
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for optimizer inputs.")


def _as_dataframe(table: object) -> "pd.DataFrame":  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(table, "columns"):
        return table.copy()  # type: ignore[return-value]
    if isinstance(table, dict):
        return pd.DataFrame([table])  # type: ignore[union-attr]
    return pd.DataFrame(list(table))  # type: ignore[union-attr]


def alpha_tables_to_returns_df(
    tables: Sequence[object],
    config: Optional[AlphaOptimizerConfig] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    cfg = config or AlphaOptimizerConfig()
    if not tables:
        raise ValueError("tables must be a non-empty sequence.")

    frames: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for idx, t in enumerate(tables):
        df = _as_dataframe(t)
        df = df.copy()
        df["source_id"] = idx
        frames.append(df)

    df_all = pd.concat(frames, axis=0, ignore_index=True)  # type: ignore[union-attr]

    dt_col = cfg.dt_col if cfg.dt_col in df_all.columns else "time"
    if dt_col not in df_all.columns:
        raise ValueError(f"missing dt column: '{cfg.dt_col}' or 'time'.")

    df_all[dt_col] = pd.to_datetime(df_all[dt_col], errors="raise")  # type: ignore[union-attr]
    if cfg.sym_col not in df_all.columns:
        raise ValueError(f"missing sym column '{cfg.sym_col}'.")

    if cfg.return_col and cfg.return_col in df_all.columns:
        df_all["ret"] = pd.to_numeric(df_all[cfg.return_col], errors="coerce")  # type: ignore[union-attr]
    else:
        sig_col = cfg.prev_signal_col if cfg.use_prev_signal else cfg.signal_col
        if sig_col not in df_all.columns or cfg.px_diff_col not in df_all.columns:
            raise ValueError("missing signal or pxDiff columns for return computation.")
        sig = pd.to_numeric(df_all[sig_col], errors="coerce")  # type: ignore[union-attr]
        px = pd.to_numeric(df_all[cfg.px_diff_col], errors="coerce")  # type: ignore[union-attr]
        df_all["ret"] = sig * px

    grouped = df_all.groupby([dt_col, cfg.sym_col], as_index=False)["ret"].sum()
    wide = grouped.pivot(index=dt_col, columns=cfg.sym_col, values="ret").sort_index()
    if cfg.fillna is not None:
        wide = wide.fillna(cfg.fillna)
    return wide


def pca_from_returns(
    returns_df: "pd.DataFrame",  # type: ignore[name-defined]
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_pandas()
    x = returns_df.to_numpy(dtype=float)
    cov = np.cov(x, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if n_components is None:
        total = np.sum(evals)
        if total <= 0.0:
            n_components = min(1, evecs.shape[1])
        else:
            cum = np.cumsum(evals) / total
            n_components = int(np.searchsorted(cum, variance_threshold) + 1)
            n_components = max(1, min(n_components, evecs.shape[1]))

    loadings = evecs[:, :n_components]
    explained = evals[:n_components]
    factor_returns = x @ loadings
    return loadings, explained, factor_returns


def rolling_sharpe(
    returns_df: "pd.DataFrame",  # type: ignore[name-defined]
    window: int,
    min_periods: Optional[int] = None,
    annualization: float = 252.0,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    mp = min_periods or window
    mu = returns_df.rolling(window, min_periods=mp).mean()
    sd = returns_df.rolling(window, min_periods=mp).std()
    sharpe = mu / sd.replace(0.0, np.nan) * math.sqrt(float(annualization))
    return sharpe


def _pc_limits_array(
    pc_exposure_limits: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_components: int,
) -> Optional[np.ndarray]:
    if pc_exposure_limits is None:
        return None
    if isinstance(pc_exposure_limits, (int, float)):
        return np.full(n_components, float(pc_exposure_limits))
    if isinstance(pc_exposure_limits, dict):
        out = np.full(n_components, np.inf)
        for k, v in pc_exposure_limits.items():
            if 0 <= int(k) < n_components:
                out[int(k)] = float(v)
        return out
    vals = np.array(list(pc_exposure_limits), dtype=float)
    if vals.size < n_components:
        vals = np.pad(vals, (0, n_components - vals.size), constant_values=np.inf)
    return vals[:n_components]


def _pc_targets_array(
    pc_exposure_targets: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_components: int,
) -> Optional[np.ndarray]:
    if pc_exposure_targets is None:
        return None
    if isinstance(pc_exposure_targets, (int, float)):
        return np.full(n_components, float(pc_exposure_targets))
    if isinstance(pc_exposure_targets, dict):
        out = np.full(n_components, np.nan)
        for k, v in pc_exposure_targets.items():
            if 0 <= int(k) < n_components:
                out[int(k)] = float(v)
        return out
    vals = np.array(list(pc_exposure_targets), dtype=float)
    if vals.size < n_components:
        vals = np.pad(vals, (0, n_components - vals.size), constant_values=np.nan)
    return vals[:n_components]


def _pc_tolerance_array(
    pc_target_tolerance: Optional[Union[float, Sequence[float], Dict[int, float]]],
    n_components: int,
) -> Optional[np.ndarray]:
    if pc_target_tolerance is None:
        return None
    if isinstance(pc_target_tolerance, (int, float)):
        return np.full(n_components, float(pc_target_tolerance))
    if isinstance(pc_target_tolerance, dict):
        out = np.full(n_components, np.inf)
        for k, v in pc_target_tolerance.items():
            if 0 <= int(k) < n_components:
                out[int(k)] = float(v)
        return out
    vals = np.array(list(pc_target_tolerance), dtype=float)
    if vals.size < n_components:
        vals = np.pad(vals, (0, n_components - vals.size), constant_values=np.inf)
    return vals[:n_components]


def _max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    bounds: Tuple[float, float],
    sum_to_one: bool,
    pc_loadings: Optional[np.ndarray] = None,
    pc_limits: Optional[np.ndarray] = None,
    risk_free: float = 0.0,
    pc_targets: Optional[np.ndarray] = None,
    pc_target_tolerance: Optional[np.ndarray] = None,
    pc_target_penalty: float = 1.0,
    target_vol: Optional[float] = None,
    target_vol_penalty: float = 10.0,
    turnover_penalty: float = 0.0,
    prev_weights: Optional[np.ndarray] = None,
    annualization: float = 252.0,
) -> np.ndarray:
    n = mu.shape[0]
    if n == 0:
        return mu

    if (
        pc_loadings is None
        and pc_limits is None
        and pc_targets is None
        and target_vol is None
        and turnover_penalty <= 0.0
    ):
        inv = np.linalg.pinv(cov)
        w = inv @ (mu - risk_free)
        if sum_to_one:
            s = np.sum(w)
            if abs(s) > 1.0e-12:
                w = w / s
        return np.clip(w, bounds[0], bounds[1])

    def objective(w: np.ndarray) -> float:
        ret = float(w @ mu) - risk_free
        var = float(w @ cov @ w)
        if var <= 0.0:
            return 1.0e6
        vol = math.sqrt(var * float(annualization))
        obj = -ret / math.sqrt(var)
        if target_vol is not None:
            obj += float(target_vol_penalty) * (vol - float(target_vol)) ** 2
        if turnover_penalty > 0.0 and prev_weights is not None:
            obj += float(turnover_penalty) * float(np.sum(np.abs(w - prev_weights)))
        if pc_targets is not None and pc_loadings is not None:
            exposures = pc_loadings.T @ w
            for j in range(pc_loadings.shape[1]):
                tgt = pc_targets[j] if j < len(pc_targets) else np.nan
                if np.isnan(tgt):
                    continue
                obj += float(pc_target_penalty) * (exposures[j] - float(tgt)) ** 2
        return obj

    cons = []
    if sum_to_one:
        cons.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

    if pc_loadings is not None and pc_limits is not None:
        for j in range(pc_loadings.shape[1]):
            limit = float(pc_limits[j])
            if not np.isfinite(limit):
                continue
            load = pc_loadings[:, j]
            cons.append({"type": "ineq", "fun": lambda w, l=load, lim=limit: lim - np.dot(w, l)})
            cons.append({"type": "ineq", "fun": lambda w, l=load, lim=limit: lim + np.dot(w, l)})

    if pc_loadings is not None and pc_targets is not None and pc_target_tolerance is not None:
        for j in range(pc_loadings.shape[1]):
            tgt = pc_targets[j] if j < len(pc_targets) else np.nan
            tol = pc_target_tolerance[j] if j < len(pc_target_tolerance) else np.inf
            if np.isnan(tgt) or not np.isfinite(tol):
                continue
            load = pc_loadings[:, j]
            cons.append(
                {"type": "ineq", "fun": lambda w, l=load, t=tgt, tol=tol: tol - abs(np.dot(w, l) - t)}
            )

    bnds = [bounds] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        return x0
    return res.x


def optimize_portfolio_with_pca(
    tables: Sequence[object],
    config: Optional[AlphaOptimizerConfig] = None,
) -> Dict[str, object]:
    _ensure_pandas()
    cfg = config or AlphaOptimizerConfig()
    returns_df = alpha_tables_to_returns_df(tables, cfg)

    if cfg.pca_mode not in {"full", "rolling"}:
        raise ValueError("pca_mode must be 'full' or 'rolling'.")

    loadings, explained, factor_returns = pca_from_returns(
        returns_df,
        n_components=cfg.n_components,
        variance_threshold=cfg.variance_threshold,
    )
    pc_limits = _pc_limits_array(cfg.pc_exposure_limits, loadings.shape[1])
    pc_targets = _pc_targets_array(cfg.pc_exposure_targets, loadings.shape[1])
    pc_tolerance = _pc_tolerance_array(cfg.pc_target_tolerance, loadings.shape[1])

    window = cfg.window
    min_periods = cfg.min_periods or window
    dates = returns_df.index

    weights_list: List[np.ndarray] = []
    weight_dates: List[object] = []
    port_rets: List[float] = []
    pc_exposure_list: List[np.ndarray] = []

    prev_w: Optional[np.ndarray] = None
    for i in range(len(returns_df)):
        if i + 1 < min_periods:
            continue
        start = max(0, i + 1 - window)
        window_df = returns_df.iloc[start : i + 1]
        mu = window_df.mean().to_numpy(dtype=float)
        cov = np.cov(window_df.to_numpy(dtype=float), rowvar=False)

        if cfg.pca_mode == "rolling":
            loadings_i, _, _ = pca_from_returns(
                window_df,
                n_components=cfg.n_components,
                variance_threshold=cfg.variance_threshold,
            )
            pc_limits_i = _pc_limits_array(cfg.pc_exposure_limits, loadings_i.shape[1])
            pc_targets_i = _pc_targets_array(cfg.pc_exposure_targets, loadings_i.shape[1])
            pc_tolerance_i = _pc_tolerance_array(cfg.pc_target_tolerance, loadings_i.shape[1])
        else:
            loadings_i = loadings
            pc_limits_i = pc_limits
            pc_targets_i = pc_targets
            pc_tolerance_i = pc_tolerance

        w = _max_sharpe_weights(
            mu,
            cov,
            bounds=cfg.weight_bounds,
            sum_to_one=cfg.sum_to_one,
            pc_loadings=loadings_i,
            pc_limits=pc_limits_i,
            risk_free=cfg.risk_free,
            pc_targets=pc_targets_i,
            pc_target_tolerance=pc_tolerance_i,
            pc_target_penalty=cfg.pc_target_penalty,
            target_vol=cfg.target_vol,
            target_vol_penalty=cfg.target_vol_penalty,
            turnover_penalty=cfg.turnover_penalty,
            prev_weights=prev_w,
            annualization=cfg.annualization,
        )
        weights_list.append(w)
        weight_dates.append(dates[i])
        port_rets.append(float(returns_df.iloc[i].to_numpy(dtype=float) @ w))
        if loadings_i is not None:
            pc_exposure_list.append(loadings_i.T @ w)
        prev_w = w

    weights_df = pd.DataFrame(weights_list, index=weight_dates, columns=returns_df.columns)  # type: ignore[union-attr]
    port_series = pd.Series(port_rets, index=weight_dates, name="portfolio_return")  # type: ignore[union-attr]
    sharpe_df = rolling_sharpe(returns_df, window=window, min_periods=min_periods, annualization=cfg.annualization)
    port_sharpe = rolling_sharpe(port_series.to_frame(), window=window, min_periods=min_periods, annualization=cfg.annualization)

    result: Dict[str, object] = {
        "returns": returns_df,
        "loadings": loadings,
        "explained_variance": explained,
        "factor_returns": pd.DataFrame(  # type: ignore[union-attr]
            factor_returns,
            index=returns_df.index,
            columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
        ),
        "alpha_rolling_sharpe": sharpe_df,
        "portfolio_weights": weights_df,
        "portfolio_returns": port_series,
        "portfolio_rolling_sharpe": port_sharpe,
    }

    if pc_exposure_list:
        result["portfolio_pc_exposure"] = pd.DataFrame(  # type: ignore[union-attr]
            pc_exposure_list,
            index=weight_dates,
            columns=[f"PC{i+1}" for i in range(loadings.shape[1])],
        )
    return result
