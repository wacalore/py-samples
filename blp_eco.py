"""Bloomberg economic history helper via blpapi.

This module is designed to be called from q/embedPy wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class BbgConfig:
    host: str = "localhost"
    port: int = 8194
    service: str = "//blp/refdata"
    timeout_ms: int = 10000
    periodicity_selection: str = "DAILY"
    max_data_points: Optional[int] = None
    keep_raw_fields: bool = False


DEFAULT_FIELD_MAP: Dict[str, List[str]] = {
    "realized": [
        "ACTUAL_RELEASE",
        "ECO_RELEASE_ACTUAL",
        "ECO_ACTUAL_VALUE",
        "PX_LAST",
    ],
    "survey": [
        "SURVEY_MEDIAN",
        "BN_SURVEY_MEDIAN",
        "ECO_RELEASE_SURVEY_MEDIAN",
    ],
    "surprise": [
        "SURPRISE",
        "ECO_RELEASE_SURPRISE",
        "BN_SURPRISE",
    ],
    "revision": [
        "REVISION",
        "ECO_RELEASE_REVISION",
        "BN_REVISION",
    ],
}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _date_to_yyyymmdd(x: Any) -> str:
    if isinstance(x, pd.Timestamp):
        x = x.date()
    if isinstance(x, datetime):
        x = x.date()
    if isinstance(x, date):
        return x.strftime("%Y%m%d")
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return s
    # Accept q-style YYYY.MM.DD and ISO YYYY-MM-DD
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    raise ValueError(f"unable to parse date: {x!r}")


def _elem_to_python(elem: Any) -> Any:
    # blpapi Datatype values are converted to native Python types here.
    try:
        if elem.isNull():
            return None
    except Exception:
        pass
    try:
        return elem.getValueAsFloat()
    except Exception:
        pass
    try:
        return elem.getValueAsInteger()
    except Exception:
        pass
    try:
        return elem.getValueAsString()
    except Exception:
        pass
    try:
        return elem.getValueAsDatetime()
    except Exception:
        pass
    try:
        return elem.getValue()
    except Exception:
        return None


def _coalesce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    out = pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        out = out.where(~out.isna(), v)
    return out


def _normalize_field_map(cfg: Mapping[str, Any]) -> Dict[str, List[str]]:
    fmap = DEFAULT_FIELD_MAP.copy()
    raw = cfg.get("field_map")
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            kk = str(k).strip().lower()
            vv = _as_list(v)
            fmap[kk] = [str(x).strip() for x in vv if str(x).strip()]
    return fmap


def _collect_field_list(field_map: Mapping[str, Sequence[str]], cfg: Mapping[str, Any]) -> List[str]:
    fields: List[str] = []
    for vv in field_map.values():
        for f in vv:
            if f and f not in fields:
                fields.append(f)
    extra = _as_list(cfg.get("extra_fields"))
    for f in extra:
        fs = str(f).strip()
        if fs and fs not in fields:
            fields.append(fs)
    return fields


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "security", "realized", "survey", "surprise", "revision"])


def _parse_cfg(cfg: Optional[Mapping[str, Any]]) -> BbgConfig:
    c = dict(cfg or {})
    return BbgConfig(
        host=str(c.get("host", "localhost")),
        port=int(c.get("port", 8194)),
        service=str(c.get("service", "//blp/refdata")),
        timeout_ms=int(c.get("timeout_ms", 10000)),
        periodicity_selection=str(c.get("periodicity_selection", "DAILY")),
        max_data_points=None
        if c.get("max_data_points") in (None, "")
        else int(c.get("max_data_points")),
        keep_raw_fields=bool(c.get("keep_raw_fields", False)),
    )


def _response_rows(
    securities: Sequence[str],
    start_date: str,
    end_date: str,
    fields: Sequence[str],
    cfg: BbgConfig,
) -> List[Dict[str, Any]]:
    try:
        import blpapi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "blpapi is unavailable; install Bloomberg's Python blpapi and ensure a running Bloomberg session."
        ) from e

    opts = blpapi.SessionOptions()
    opts.setServerHost(cfg.host)
    opts.setServerPort(cfg.port)
    session = blpapi.Session(opts)
    if not session.start():
        raise RuntimeError("unable to start Bloomberg session")
    if not session.openService(cfg.service):
        session.stop()
        raise RuntimeError(f"unable to open Bloomberg service: {cfg.service}")

    svc = session.getService(cfg.service)
    req = svc.createRequest("HistoricalDataRequest")

    sec_el = req.getElement("securities")
    for sec in securities:
        sec_el.appendValue(sec)

    fld_el = req.getElement("fields")
    for f in fields:
        fld_el.appendValue(f)

    req.set("startDate", start_date)
    req.set("endDate", end_date)
    req.set("periodicitySelection", cfg.periodicity_selection)
    if cfg.max_data_points is not None:
        req.set("maxDataPoints", cfg.max_data_points)

    session.sendRequest(req)

    rows: List[Dict[str, Any]] = []
    while True:
        event = session.nextEvent(cfg.timeout_ms)
        et = event.eventType()
        for msg in event:
            if msg.hasElement("responseError"):
                session.stop()
                raise RuntimeError(str(msg.getElement("responseError")))

            if msg.messageType() != blpapi.Name("HistoricalDataResponse"):
                continue

            sdata = msg.getElement("securityData")
            sec = sdata.getElementAsString("security")
            if sdata.hasElement("securityError"):
                continue

            fdata = sdata.getElement("fieldData")
            for i in range(fdata.numValues()):
                fd = fdata.getValueAsElement(i)
                row: Dict[str, Any] = {"security": sec}
                if fd.hasElement("date"):
                    try:
                        row["date"] = fd.getElementAsDatetime("date").date()
                    except Exception:
                        pass
                for f in fields:
                    if not fd.hasElement(f):
                        continue
                    row[f] = _elem_to_python(fd.getElement(f))
                rows.append(row)
        if et == blpapi.Event.RESPONSE:
            break

    session.stop()
    return rows


def get_eco_history(
    securities: Iterable[str],
    start_date: Any,
    end_date: Any,
    cfg: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Return Bloomberg eco history in canonical shape.

    Output columns:
    - date
    - security
    - realized
    - survey
    - surprise
    - revision
    """

    c = dict(cfg or {})
    sec_list = [str(s).strip() for s in _as_list(securities) if str(s).strip()]
    if not sec_list:
        return _empty_result()

    # Test path without Bloomberg connectivity.
    mock_rows = c.get("mock_rows")
    if mock_rows is not None:
        if isinstance(mock_rows, Mapping):
            raw_df = pd.DataFrame(mock_rows)
        else:
            raw_df = pd.DataFrame(_as_list(mock_rows))
    else:
        s0 = _date_to_yyyymmdd(start_date)
        e0 = _date_to_yyyymmdd(end_date)
        field_map = _normalize_field_map(c)
        fields = _collect_field_list(field_map, c)
        rows = _response_rows(sec_list, s0, e0, fields, _parse_cfg(c))
        raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        return _empty_result()

    if "date" in raw_df.columns:
        if pd.api.types.is_numeric_dtype(raw_df["date"]):
            raw_df["date"] = pd.to_datetime(
                raw_df["date"], unit="D", origin="2000-01-01", errors="coerce"
            ).dt.date
        else:
            raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce").dt.date
    else:
        raw_df["date"] = pd.NaT

    if "security" not in raw_df.columns:
        if "ticker" in raw_df.columns:
            raw_df["security"] = raw_df["ticker"]
        else:
            raw_df["security"] = sec_list[0]

    field_map = _normalize_field_map(c)
    out = pd.DataFrame(
        {
            "date": raw_df["date"],
            "security": raw_df["security"].astype(str),
            "realized": _coalesce_numeric(raw_df, field_map.get("realized", [])),
            "survey": _coalesce_numeric(raw_df, field_map.get("survey", [])),
            "surprise": _coalesce_numeric(raw_df, field_map.get("surprise", [])),
            "revision": _coalesce_numeric(raw_df, field_map.get("revision", [])),
        }
    )

    if bool(c.get("keep_raw_fields", False)):
        canon = {"date", "security", "realized", "survey", "surprise", "revision"}
        keep_cols = [x for x in raw_df.columns if x not in canon]
        if keep_cols:
            out = out.join(raw_df[keep_cols], how="left")

    out = out.sort_values(["date", "security"]).reset_index(drop=True)
    return out
