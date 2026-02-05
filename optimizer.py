#!/usr/bin/env python3
"""
Alpha portfolio optimizer with PCA-aware constraints.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    alpha_mode: str = "sym"  # "sym" uses sym_col, "alpha" aggregates per alpha table
    alpha_col: Optional[str] = None  # optional column name for alpha id (else uses source_id)
    dt_unit: Optional[str] = None
    dt_origin: Optional[str] = None

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

    cov_shrinkage: Optional[float] = None
    cov_shrinkage_target: str = "diag"
    mean_shrinkage: Optional[float] = None
    mean_shrinkage_target: str = "zero"

    ewma_halflife: Optional[float] = None
    ewma_span: Optional[float] = None
    robust_method: Optional[str] = None
    robust_clip: float = 3.0
    robust_scale: str = "mad"

    reliability_mode: Optional[str] = None
    reliability_clip: float = 2.0
    reliability_floor: float = 0.0
    reliability_power: float = 1.0

    weight_l2_penalty: float = 0.0

    regime_mode: Optional[str] = None
    regime_short_window: int = 20
    regime_long_window: Optional[int] = None
    regime_halflife_bounds: Tuple[float, float] = (0.5, 2.0)
    regime_shrinkage_bounds: Tuple[float, float] = (0.5, 2.0)

    rebalance_step: int = 1
    n_jobs: int = 1


def _ensure_pandas() -> None:
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for optimizer inputs.")


def _dict_looks_columnar(table: dict) -> bool:
    lengths: List[int] = []
    for v in table.values():
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            lengths.append(len(v))
        else:
            return False
    if not lengths:
        return False
    return len(set(lengths)) == 1


def _as_dataframe(table: object) -> "pd.DataFrame":  # type: ignore[name-defined]
    if HAVE_PANDAS and hasattr(table, "columns"):
        return table.copy()  # type: ignore[return-value]
    if isinstance(table, dict):
        if _dict_looks_columnar(table):
            return pd.DataFrame(table)  # type: ignore[union-attr]
        return pd.DataFrame([table])  # type: ignore[union-attr]
    return pd.DataFrame(list(table))  # type: ignore[union-attr]


def _config_from_object(config: Optional[object]) -> AlphaOptimizerConfig:
    if config is None:
        return AlphaOptimizerConfig()
    if isinstance(config, AlphaOptimizerConfig):
        return config
    if isinstance(config, dict):
        return AlphaOptimizerConfig(**config)
    raise TypeError("config must be AlphaOptimizerConfig, dict, or None.")


def alpha_tables_to_returns_df(
    tables: Sequence[object],
    config: Optional[object] = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    _ensure_pandas()
    cfg = _config_from_object(config)
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

    dt_series = df_all[dt_col]
    if cfg.dt_unit and cfg.dt_origin:
        df_all[dt_col] = pd.to_datetime(  # type: ignore[union-attr]
            dt_series,
            unit=cfg.dt_unit,
            origin=cfg.dt_origin,
            errors="raise",
        )
    else:
        df_all[dt_col] = pd.to_datetime(dt_series, errors="raise")  # type: ignore[union-attr]
    if cfg.return_col and cfg.return_col in df_all.columns:
        df_all["ret"] = pd.to_numeric(df_all[cfg.return_col], errors="coerce")  # type: ignore[union-attr]
    else:
        sig_col = cfg.prev_signal_col if cfg.use_prev_signal else cfg.signal_col
        if sig_col not in df_all.columns or cfg.px_diff_col not in df_all.columns:
            raise ValueError("missing signal or pxDiff columns for return computation.")
        sig = pd.to_numeric(df_all[sig_col], errors="coerce")  # type: ignore[union-attr]
        px = pd.to_numeric(df_all[cfg.px_diff_col], errors="coerce")  # type: ignore[union-attr]
        df_all["ret"] = sig * px

    if cfg.alpha_mode == "alpha":
        alpha_col = cfg.alpha_col or "source_id"
        if alpha_col not in df_all.columns:
            raise ValueError(f"missing alpha column '{alpha_col}'.")
        grouped = df_all.groupby([dt_col, alpha_col], as_index=False)["ret"].sum()
        wide = grouped.pivot(index=dt_col, columns=alpha_col, values="ret").sort_index()
    else:
        if cfg.sym_col not in df_all.columns:
            raise ValueError(f"missing sym column '{cfg.sym_col}'.")
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


def _shrink_cov(cov: np.ndarray, alpha: float, target: str) -> np.ndarray:
    if not np.isfinite(alpha) or alpha <= 0.0:
        return cov
    alpha = float(min(max(alpha, 0.0), 1.0))
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    n = cov.shape[0]
    if n == 0:
        return cov
    target_key = target.lower()
    if target_key in {"identity", "eye", "i"}:
        avg_var = float(np.trace(cov)) / float(n)
        shrink_target = np.eye(n) * avg_var
    elif target_key in {"diag", "diagonal"}:
        shrink_target = np.diag(np.diag(cov))
    elif target_key in {"constant_correlation", "constcorr", "cc"}:
        std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        denom = np.outer(std, std)
        corr = np.zeros_like(cov)
        mask = denom > 0.0
        corr[mask] = cov[mask] / denom[mask]
        if n > 1:
            avg_corr = float((np.sum(corr) - np.trace(corr)) / (n * (n - 1)))
        else:
            avg_corr = 0.0
        shrink_target = np.outer(std, std) * avg_corr
        np.fill_diagonal(shrink_target, np.diag(cov))
    else:
        raise ValueError(f"unknown cov_shrinkage_target '{target}'.")
    return (1.0 - alpha) * cov + alpha * shrink_target


def _shrink_mean(mu: np.ndarray, alpha: float, target: str) -> np.ndarray:
    if not np.isfinite(alpha) or alpha <= 0.0:
        return mu
    alpha = float(min(max(alpha, 0.0), 1.0))
    target_key = target.lower()
    if target_key in {"zero", "zeros"}:
        shrink_target = np.zeros_like(mu)
    elif target_key in {"grand_mean", "mean"}:
        grand = float(np.mean(mu))
        shrink_target = np.full_like(mu, grand)
    else:
        raise ValueError(f"unknown mean_shrinkage_target '{target}'.")
    return (1.0 - alpha) * mu + alpha * shrink_target


def _ewma_weights(n: int, halflife: Optional[float], span: Optional[float]) -> Optional[np.ndarray]:
    if n <= 0:
        return None
    if halflife is not None and span is not None:
        raise ValueError("set only one of ewma_halflife or ewma_span.")
    if halflife is None and span is None:
        return None
    if halflife is not None:
        if halflife <= 0:
            raise ValueError("ewma_halflife must be positive.")
        alpha = 1.0 - math.exp(math.log(0.5) / float(halflife))
    else:
        if span is None or span <= 0:
            raise ValueError("ewma_span must be positive.")
        alpha = 2.0 / (float(span) + 1.0)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("invalid EWMA alpha.")
    idx = np.arange(n, dtype=float)
    w = (1.0 - alpha) ** (n - 1 - idx)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return None
    return w / w_sum


def _weighted_mean_cov(x: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be 2D.")
    if weights.ndim != 1 or weights.shape[0] != x.shape[0]:
        raise ValueError("weights must be 1D and match rows of x.")
    w = weights.astype(float)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("sum of weights must be positive.")
    w = w / w_sum
    mu = w @ x
    x_c = x - mu
    cov = (x_c.T * w) @ x_c
    return mu, cov


def _robust_winsorize(x: np.ndarray, clip: float, scale: str) -> np.ndarray:
    if clip <= 0:
        return x
    med = np.median(x, axis=0)
    scale_key = scale.lower()
    if scale_key in {"mad", "median_abs_dev", "median_absolute_deviation"}:
        mad = np.median(np.abs(x - med), axis=0)
        s = 1.4826 * mad
    elif scale_key in {"std", "stdev"}:
        s = np.std(x, axis=0, ddof=1)
    else:
        raise ValueError(f"unknown robust_scale '{scale}'.")
    s = np.where(np.isfinite(s) & (s > 0.0), s, 0.0)
    lower = med - clip * s
    upper = med + clip * s
    return np.minimum(np.maximum(x, lower), upper)


def _effective_sample_size(weights: Optional[np.ndarray], n: int) -> float:
    if weights is None:
        return float(n)
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return float(n)
    return 1.0 / denom


def _reliability_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    n_eff: float,
    mode: str,
    clip: float,
    floor: float,
    power: float,
) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    sd = np.where(sd > 0.0, sd, np.nan)
    sr_daily = mu / sd
    mode_key = mode.lower()
    if mode_key in {"sharpe", "sr"}:
        score = np.abs(sr_daily)
    elif mode_key in {"tstat", "t"}:
        score = np.abs(sr_daily) * math.sqrt(max(n_eff, 1.0))
    else:
        raise ValueError("reliability_mode must be 'sharpe' or 'tstat'.")
    clip_val = float(max(clip, 1.0e-12))
    w = np.clip(score / clip_val, 0.0, 1.0)
    if power != 1.0:
        w = np.power(w, float(power))
    if floor > 0.0:
        w = np.maximum(w, float(floor))
    w = np.nan_to_num(w, nan=0.0, posinf=1.0, neginf=0.0)
    return w


def _regime_adjustments(
    x: np.ndarray,
    short_window: int,
    long_window: Optional[int],
    base_halflife: Optional[float],
    base_span: Optional[float],
    base_shrink: Optional[float],
    hl_bounds: Tuple[float, float],
    shrink_bounds: Tuple[float, float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if x.size == 0:
        return base_halflife, base_span, base_shrink
    short_w = max(2, int(short_window))
    long_w = int(long_window) if long_window else x.shape[0]
    long_w = max(short_w, min(long_w, x.shape[0]))
    short_x = x[-short_w:]
    long_x = x[-long_w:]

    short_vol = float(np.nanmean(np.std(short_x, axis=0, ddof=1)))
    long_vol = float(np.nanmean(np.std(long_x, axis=0, ddof=1)))
    if not np.isfinite(short_vol) or not np.isfinite(long_vol) or long_vol <= 0.0:
        return base_halflife, base_span, base_shrink

    ratio = short_vol / long_vol
    hl_lo, hl_hi = hl_bounds
    sh_lo, sh_hi = shrink_bounds
    hl_mult = np.clip(1.0 / ratio, hl_lo, hl_hi)
    sh_mult = np.clip(ratio, sh_lo, sh_hi)

    halflife = base_halflife * hl_mult if base_halflife is not None else None
    span = base_span * hl_mult if base_span is not None else None
    shrink = base_shrink * sh_mult if base_shrink is not None else None
    if shrink is not None:
        shrink = float(min(max(shrink, 0.0), 1.0))
    return halflife, span, shrink


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
    weight_l2_penalty: float = 0.0,
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
        and weight_l2_penalty <= 0.0
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
        if weight_l2_penalty > 0.0:
            obj += float(weight_l2_penalty) * float(np.sum(w * w))
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
    config: Optional[object] = None,
) -> Dict[str, object]:
    _ensure_pandas()
    cfg = _config_from_object(config)
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

    rebalance_step = max(1, int(cfg.rebalance_step))
    n_jobs = int(cfg.n_jobs) if cfg.n_jobs is not None else 1
    if n_jobs == 0:
        n_jobs = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + n_jobs + 1)
    n_jobs = max(1, n_jobs)
    if n_jobs != 1 and cfg.turnover_penalty > 0.0:
        raise ValueError("n_jobs>1 is not supported when turnover_penalty > 0.")

    def _solve_window(
        i: int,
        prev_weights: Optional[np.ndarray],
        turnover_penalty: float,
    ) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
        start = max(0, i + 1 - window)
        window_df = returns_df.iloc[start : i + 1]
        window_x = window_df.to_numpy(dtype=float)
        if cfg.robust_method:
            if cfg.robust_method.lower() not in {"winsor", "winsorize"}:
                raise ValueError("robust_method must be 'winsor' if set.")
            window_x = _robust_winsorize(window_x, cfg.robust_clip, cfg.robust_scale)

        ewma_halflife = cfg.ewma_halflife
        ewma_span = cfg.ewma_span
        cov_shrink = cfg.cov_shrinkage
        if cfg.regime_mode:
            if cfg.regime_mode.lower() not in {"vol_ratio", "vol"}:
                raise ValueError("regime_mode must be 'vol_ratio' if set.")
            ewma_halflife, ewma_span, cov_shrink = _regime_adjustments(
                window_x,
                cfg.regime_short_window,
                cfg.regime_long_window,
                ewma_halflife,
                ewma_span,
                cov_shrink,
                cfg.regime_halflife_bounds,
                cfg.regime_shrinkage_bounds,
            )

        ewma_w = _ewma_weights(window_x.shape[0], ewma_halflife, ewma_span)
        if ewma_w is None:
            mu = window_x.mean(axis=0)
            cov = np.cov(window_x, rowvar=False)
        else:
            mu, cov = _weighted_mean_cov(window_x, ewma_w)

        if cfg.reliability_mode:
            n_eff = _effective_sample_size(ewma_w, window_x.shape[0])
            rel_w = _reliability_weights(
                mu,
                cov,
                n_eff,
                cfg.reliability_mode,
                cfg.reliability_clip,
                cfg.reliability_floor,
                cfg.reliability_power,
            )
            mu = mu * rel_w

        if cfg.mean_shrinkage:
            mu = _shrink_mean(mu, float(cfg.mean_shrinkage), cfg.mean_shrinkage_target)
        if cov_shrink:
            cov = _shrink_cov(cov, float(cov_shrink), cfg.cov_shrinkage_target)

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

        use_pc_constraints = False
        if pc_limits_i is not None and np.any(np.isfinite(pc_limits_i)):
            use_pc_constraints = True
        if pc_targets_i is not None and np.any(np.isfinite(pc_targets_i)):
            use_pc_constraints = True

        pc_loadings_opt = loadings_i if use_pc_constraints else None
        pc_limits_opt = pc_limits_i if use_pc_constraints else None
        pc_targets_opt = pc_targets_i if use_pc_constraints else None
        pc_tolerance_opt = pc_tolerance_i if use_pc_constraints else None

        w = _max_sharpe_weights(
            mu,
            cov,
            bounds=cfg.weight_bounds,
            sum_to_one=cfg.sum_to_one,
            pc_loadings=pc_loadings_opt,
            pc_limits=pc_limits_opt,
            risk_free=cfg.risk_free,
            pc_targets=pc_targets_opt,
            pc_target_tolerance=pc_tolerance_opt,
            pc_target_penalty=cfg.pc_target_penalty,
            target_vol=cfg.target_vol,
            target_vol_penalty=cfg.target_vol_penalty,
            turnover_penalty=turnover_penalty,
            weight_l2_penalty=cfg.weight_l2_penalty,
            prev_weights=prev_weights,
            annualization=cfg.annualization,
        )
        return i, w, loadings_i

    prev_w: Optional[np.ndarray] = None
    prev_loadings: Optional[np.ndarray] = loadings if cfg.pca_mode == "full" else None

    start_idx = min_periods - 1
    if start_idx < 0:
        start_idx = 0
    eligible_indices = list(range(start_idx, len(returns_df)))
    rebalance_indices = [i for i in eligible_indices if (i - start_idx) % rebalance_step == 0]

    if n_jobs == 1 or len(rebalance_indices) <= 1:
        for i in range(len(returns_df)):
            if i + 1 < min_periods:
                continue
            do_rebalance = prev_w is None
            if not do_rebalance:
                offset = (i + 1 - min_periods)
                do_rebalance = (offset % rebalance_step) == 0

            if do_rebalance:
                _, w, loadings_i = _solve_window(i, prev_w, cfg.turnover_penalty)
                prev_w = w
                if cfg.pca_mode == "rolling":
                    prev_loadings = loadings_i
                else:
                    prev_loadings = loadings
            else:
                w = prev_w

            weights_list.append(w)
            weight_dates.append(dates[i])
            port_rets.append(float(returns_df.iloc[i].to_numpy(dtype=float) @ w))
            if prev_loadings is not None:
                pc_exposure_list.append(prev_loadings.T @ w)
    else:
        results: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        max_workers = min(n_jobs, len(rebalance_indices))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_solve_window, i, None, 0.0) for i in rebalance_indices]
            for fut in as_completed(futures):
                idx, w_i, loadings_i = fut.result()
                results[idx] = (w_i, loadings_i)

        last_w: Optional[np.ndarray] = None
        last_loadings: Optional[np.ndarray] = loadings if cfg.pca_mode == "full" else None
        for i in eligible_indices:
            if i in results:
                last_w, last_loadings = results[i]
            if last_w is None:
                continue
            weights_list.append(last_w)
            weight_dates.append(dates[i])
            port_rets.append(float(returns_df.iloc[i].to_numpy(dtype=float) @ last_w))
            if last_loadings is not None:
                pc_exposure_list.append(last_loadings.T @ last_w)

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


def optimizer_result_to_dict(
    result: Dict[str, object],
    date_col: str = "dt",
    date_mode: str = "datetime64[D]",
    epoch: str = "2000-01-01",
) -> Dict[str, object]:
    """
    Convert optimizer output into a q-friendly dict.

    - DataFrames/Series are converted to dict-of-lists with index moved to `date_col`.
    - numpy arrays are returned as-is.
    """
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for optimizer_result_to_dict.")

    out: Dict[str, object] = {}
    epoch_ts = pd.Timestamp(epoch)  # type: ignore[union-attr]

    def _index_to_col(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
        df = df.copy()
        idx = df.index
        if np.issubdtype(idx.dtype, np.datetime64):
            if date_mode == "days":
                df[date_col] = (idx - epoch_ts).days.astype("int32")
            else:
                df[date_col] = idx.values.astype("datetime64[D]")
        else:
            df[date_col] = idx
        return df.reset_index(drop=True)

    for key, val in result.items():
        if HAVE_PANDAS and isinstance(val, pd.Series):  # type: ignore[union-attr]
            df = val.to_frame()
            df = _index_to_col(df)
            out[key] = {c: df[c].tolist() for c in df.columns}
        elif HAVE_PANDAS and isinstance(val, pd.DataFrame):  # type: ignore[union-attr]
            df = _index_to_col(val)
            out[key] = {c: df[c].tolist() for c in df.columns}
        elif isinstance(val, np.ndarray):
            out[key] = val
        else:
            out[key] = val
    return out


def optimizer_result_table(
    result: Dict[str, object],
    key: str,
    date_col: str = "dt",
    date_mode: str = "datetime64[D]",
    epoch: str = "2000-01-01",
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Return a single result entry as a DataFrame with index moved to `date_col`.
    """
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required for optimizer_result_table.")
    if key not in result:
        raise KeyError(f"result has no key '{key}'.")
    val = result[key]
    epoch_ts = pd.Timestamp(epoch)  # type: ignore[union-attr]

    def _index_to_col(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
        df = df.copy()
        idx = df.index
        if np.issubdtype(idx.dtype, np.datetime64):
            if date_mode == "days":
                df[date_col] = (idx - epoch_ts).days.astype("int32")
            else:
                df[date_col] = idx.values.astype("datetime64[D]")
        else:
            df[date_col] = idx
        return df.reset_index(drop=True)

    if HAVE_PANDAS and isinstance(val, pd.Series):  # type: ignore[union-attr]
        return _index_to_col(val.to_frame())
    if HAVE_PANDAS and isinstance(val, pd.DataFrame):  # type: ignore[union-attr]
        return _index_to_col(val)
    if isinstance(val, np.ndarray):
        if val.ndim == 1:
            df = pd.DataFrame({key: val})
        else:
            cols = [f"c{i}" for i in range(val.shape[1])]
            df = pd.DataFrame(val, columns=cols)
        return _index_to_col(df)
    return pd.DataFrame({key: [val]})


def optimizer_result_tables(
    result: Dict[str, object],
    date_col: str = "dt",
    date_mode: str = "datetime64[D]",
    epoch: str = "2000-01-01",
) -> Tuple[List[str], List["pd.DataFrame"]]:  # type: ignore[name-defined]
    """
    Return (names, tables) where each table is a DataFrame (index moved to `date_col`).
    This is intended for embedPy to convert to a list of q tables without nested dicts.
    """
    names: List[str] = []
    tables: List["pd.DataFrame"] = []  # type: ignore[name-defined]
    for key in sorted(result.keys()):
        names.append(key)
        tables.append(optimizer_result_table(result, key, date_col=date_col, date_mode=date_mode, epoch=epoch))
    return names, tables
