"""Bloomberg economic history helper via blpapi.

This module is designed to be called from q/embedPy wrappers.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class BbgConfig:
    host: str = "localhost"
    port: int = 8194
    service: str = "//blp/refdata"
    auth_service: str = "//blp/apiauth"
    auth_options: Optional[str] = None
    authorize: bool = False
    uuid: Optional[Any] = None
    emrsid: Optional[Any] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    token: Optional[str] = None
    app_name: Optional[str] = None
    auth_timeout_ms: int = 10000
    auth_fields: Optional[Dict[str, Any]] = None
    timeout_ms: int = 10000
    periodicity_selection: str = "DAILY"
    periodicity_adjustment: Optional[str] = None
    max_data_points: Optional[int] = None
    overrides: Optional[Dict[str, Any]] = None
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


def _normalize_securities(securities: Any) -> List[str]:
    def _is_seq(x: Any) -> bool:
        return isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _char_seq_to_str(x: Sequence[Any]) -> Optional[str]:
        if len(x) == 0:
            return None
        if all(isinstance(c, str) and len(c) == 1 for c in x):
            s = "".join(x).strip()
            return s or None
        return None

    def _as_tokens(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            s = x.strip()
            if s and ((s[0] == "[" and s[-1] == "]") or (s[0] == "(" and s[-1] == ")")):
                try:
                    parsed = ast.literal_eval(s)
                    # If this was a serialized list/tuple from q conversion, recurse into it.
                    if _is_seq(parsed):
                        return _as_tokens(parsed)
                except Exception:
                    pass
            return [s] if s else []
        if isinstance(x, (bytes, bytearray)):
            s = x.decode("utf-8", errors="ignore").strip()
            return [s] if s else []
        if _is_seq(x):
            seq = list(x)
            s0 = _char_seq_to_str(seq)
            if s0 is not None:
                return [s0]
            if len(seq) == 1:
                return _as_tokens(seq[0])
            out: List[str] = []
            for z in seq:
                out.extend(_as_tokens(z))
            return out
        s = str(x).strip()
        return [s] if s else []

    return _as_tokens(securities)


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


def _normalize_bbg_date(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        # blpapi Datetime-like object with year/month/day attributes.
        if all(hasattr(v, x) for x in ("year", "month", "day")):
            return date(int(v.year), int(v.month), int(v.day))
    except Exception:
        pass
    try:
        s = str(v).strip()
        if s:
            return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        pass
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


def _normalize_name_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (bytes, bytearray)):
        s = x.decode("utf-8", errors="ignore").strip()
        return [s] if s else []
    if isinstance(x, Sequence):
        out: List[str] = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            else:
                s = str(v).strip()
                if s:
                    out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "security", "realized", "survey", "surprise", "revision"])


def _parse_cfg(cfg: Optional[Mapping[str, Any]]) -> BbgConfig:
    c = dict(cfg or {})
    auth_raw = c.get("auth")
    auth = dict(auth_raw) if isinstance(auth_raw, Mapping) else {}
    def _pick(name: str, default: Any = None) -> Any:
        return c.get(name, auth.get(name, default))

    uuid_raw = _pick("uuid")
    if uuid_raw in (None, ""):
        uuid_val = None
    else:
        uuid_val = str(uuid_raw).strip() if isinstance(uuid_raw, str) else uuid_raw

    emrs_raw = c.get("emrsid", auth.get("emrsid", c.get("emrsId", auth.get("emrsId"))))
    if emrs_raw in (None, ""):
        emrs_val = None
    else:
        emrs_val = str(emrs_raw).strip() if isinstance(emrs_raw, str) else emrs_raw

    user_raw = c.get("username", auth.get("username", c.get("user", auth.get("user"))))
    username_val = None if user_raw in (None, "") else str(user_raw).strip()

    ov_raw = c.get("overrides")
    overrides_val: Optional[Dict[str, Any]] = None
    if isinstance(ov_raw, Mapping):
        overrides_val = dict(ov_raw)
    elif isinstance(ov_raw, Sequence) and not isinstance(ov_raw, (str, bytes, bytearray)):
        tmp: Dict[str, Any] = {}
        for it in ov_raw:
            if isinstance(it, Mapping):
                k = it.get("fieldId", it.get("field", it.get("name")))
                v = it.get("value")
                if k is not None:
                    tmp[str(k)] = v
            elif isinstance(it, Sequence) and not isinstance(it, (str, bytes, bytearray)) and len(it) >= 2:
                tmp[str(it[0])] = it[1]
        overrides_val = tmp if tmp else None

    if overrides_val is None:
        of = c.get("override_fields")
        ov = c.get("override_values")
        if isinstance(of, Sequence) and not isinstance(of, (str, bytes, bytearray)):
            fields = list(of)
            values = list(ov) if isinstance(ov, Sequence) and not isinstance(ov, (str, bytes, bytearray)) else []
            n = min(len(fields), len(values))
            if n > 0:
                overrides_val = {str(fields[i]): values[i] for i in range(n)}

    return BbgConfig(
        host=str(_pick("host", "localhost")),
        port=int(_pick("port", 8194)),
        service=str(_pick("service", "//blp/refdata")),
        auth_service=str(_pick("auth_service", "//blp/apiauth")),
        auth_options=(
            None
            if _pick("auth_options") in (None, "")
            else str(_pick("auth_options"))
        ),
        authorize=bool(_pick("authorize", False)),
        uuid=uuid_val,
        emrsid=emrs_val,
        username=username_val,
        ip_address=(
            None if _pick("ip_address") in (None, "") else str(_pick("ip_address"))
        ),
        token=None if _pick("token") in (None, "") else str(_pick("token")),
        app_name=None if _pick("app_name") in (None, "") else str(_pick("app_name")),
        auth_timeout_ms=int(_pick("auth_timeout_ms", 10000)),
        auth_fields=dict(_pick("auth_fields", {}))
        if isinstance(_pick("auth_fields", {}), Mapping)
        else None,
        timeout_ms=int(_pick("timeout_ms", 10000)),
        periodicity_selection=str(_pick("periodicity_selection", "DAILY")),
        periodicity_adjustment=(
            None
            if _pick("periodicity_adjustment") in (None, "")
            else str(_pick("periodicity_adjustment"))
        ),
        max_data_points=None
        if _pick("max_data_points") in (None, "")
        else int(_pick("max_data_points")),
        overrides=overrides_val,
        keep_raw_fields=bool(_pick("keep_raw_fields", False)),
    )


def _auth_needed(cfg: BbgConfig) -> bool:
    return bool(
        cfg.authorize
        or cfg.uuid is not None
        or cfg.emrsid is not None
        or cfg.username
        or cfg.ip_address
        or cfg.token
        or cfg.app_name
        or (cfg.auth_fields and len(cfg.auth_fields) > 0)
    )


def _maybe_set_req_field(req: Any, key: str, value: Any) -> None:
    if value is None:
        return
    try:
        req.set(key, value)
    except Exception as e:
        raise RuntimeError(f"failed to set authorization field {key!r}: {e}") from e


def _try_set_req_fields(req: Any, keys: Sequence[str], value: Any) -> bool:
    if value is None:
        return False
    for k in keys:
        try:
            req.set(k, value)
            return True
        except Exception:
            pass
    return False


def _authorize_identity(session: Any, cfg: BbgConfig) -> Any:
    import blpapi  # type: ignore

    if not session.openService(cfg.auth_service):
        raise RuntimeError(f"unable to open Bloomberg auth service: {cfg.auth_service}")

    auth_svc = session.getService(cfg.auth_service)
    ao = (cfg.auth_options or "").upper()
    os_logon = "AUTHENTICATIONTYPE=OS_LOGON" in ao
    has_user_fields = bool(
        cfg.username or (cfg.uuid is not None) or (cfg.emrsid is not None) or cfg.ip_address
    )

    def _set_auth_fields(auth_req: Any, include_user_fields: bool) -> None:
        if include_user_fields and cfg.username:
            # Different Bloomberg deployments name this field differently.
            _try_set_req_fields(
                auth_req, ("userId", "username", "userName", "user"), cfg.username
            )

        if include_user_fields:
            emrs_v = cfg.emrsid
            if isinstance(emrs_v, str):
                ev = emrs_v.strip()
                if ev:
                    if ev.isdigit():
                        if not _try_set_req_fields(auth_req, ("emrsId", "emrsid", "EMRSID"), int(ev)):
                            _try_set_req_fields(auth_req, ("emrsId", "emrsid", "EMRSID"), ev)
                    else:
                        _try_set_req_fields(auth_req, ("emrsId", "emrsid", "EMRSID"), ev)
            else:
                _try_set_req_fields(auth_req, ("emrsId", "emrsid", "EMRSID"), emrs_v)

            uuid_v = cfg.uuid
            if isinstance(uuid_v, str):
                u = uuid_v.strip()
                if u:
                    # Bloomberg often documents UUID as integer, but some environments pass it as string.
                    if u.isdigit():
                        try:
                            _maybe_set_req_field(auth_req, "uuid", int(u))
                        except Exception:
                            _maybe_set_req_field(auth_req, "uuid", u)
                    else:
                        _maybe_set_req_field(auth_req, "uuid", u)
            else:
                _maybe_set_req_field(auth_req, "uuid", uuid_v)

            _try_set_req_fields(auth_req, ("ipAddress", "ip_address"), cfg.ip_address)

        _try_set_req_fields(auth_req, ("token",), cfg.token)
        _try_set_req_fields(auth_req, ("appName", "applicationName"), cfg.app_name)

        if cfg.auth_fields:
            for k, v in cfg.auth_fields.items():
                _maybe_set_req_field(auth_req, str(k), v)

    def _attempt(include_user_fields: bool, label: str) -> tuple[Optional[Any], str]:
        auth_req = auth_svc.createAuthorizationRequest()
        _set_auth_fields(auth_req, include_user_fields=include_user_fields)
        identity = session.createIdentity()
        q = blpapi.EventQueue()
        session.sendAuthorizationRequest(auth_req, identity, blpapi.CorrelationId(label), q)

        last_fail: Optional[str] = None
        while True:
            ev = q.nextEvent(cfg.auth_timeout_ms)
            et = ev.eventType()
            if et == blpapi.Event.TIMEOUT:
                return None, f"{label}: authorization timed out"
            for msg in ev:
                mt = str(msg.messageType())
                if mt == "AuthorizationSuccess":
                    return identity, ""
                if mt in ("AuthorizationFailure", "RequestFailure"):
                    last_fail = f"{label}: {msg}"
            if et == blpapi.Event.RESPONSE:
                break
        return None, (last_fail or f"{label}: authorization failed: no success message")

    # OS_LOGON usually works best with minimal authorization request fields.
    if os_logon:
        attempts = [("minimal", False)]
        if has_user_fields:
            attempts.append(("explicit", True))
    else:
        attempts = [("explicit", True)] if has_user_fields else []
        attempts.append(("minimal", False))

    errs: List[str] = []
    for label, include_user_fields in attempts:
        ident, err = _attempt(include_user_fields=include_user_fields, label=label)
        if ident is not None:
            return ident
        errs.append(err)

    raise RuntimeError("authorization failed; " + " | ".join([e for e in errs if e]))


def _apply_overrides(req: Any, overrides: Optional[Mapping[str, Any]]) -> None:
    if not overrides:
        return
    try:
        ovs = req.getElement("overrides")
    except Exception:
        return
    for k, v in overrides.items():
        if v is None:
            continue
        fld = str(k).strip()
        if not fld:
            continue
        ov = ovs.appendElement()
        ov.setElement("fieldId", fld)
        ov.setElement("value", str(v))


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
    if cfg.auth_options:
        opts.setAuthenticationOptions(cfg.auth_options)
    session = blpapi.Session(opts)
    if not session.start():
        raise RuntimeError("unable to start Bloomberg session")
    try:
        identity = _authorize_identity(session, cfg) if _auth_needed(cfg) else None

        if not session.openService(cfg.service):
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
        if cfg.periodicity_adjustment:
            req.set("periodicityAdjustment", cfg.periodicity_adjustment)
        if (cfg.max_data_points is not None) and (cfg.max_data_points > 0):
            req.set("maxDataPoints", cfg.max_data_points)
        _apply_overrides(req, cfg.overrides)

        if identity is None:
            session.sendRequest(req)
        else:
            session.sendRequest(req, identity)

        rows: List[Dict[str, Any]] = []
        while True:
            event = session.nextEvent(cfg.timeout_ms)
            et = event.eventType()
            for msg in event:
                if msg.hasElement("responseError"):
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
                        d0 = fd.getElementAsDatetime("date")
                        row["date"] = _normalize_bbg_date(d0)
                    except Exception:
                        try:
                            row["date"] = _normalize_bbg_date(fd.getElementAsString("date"))
                        except Exception:
                            pass
                    for f in fields:
                        if not fd.hasElement(f):
                            continue
                        row[f] = _elem_to_python(fd.getElement(f))
                    rows.append(row)
            if et == blpapi.Event.RESPONSE:
                break

        return rows
    finally:
        session.stop()


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
    sec_list = _normalize_securities(securities)
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
        req_fields = _normalize_name_list(c.get("request_fields"))
        fields = req_fields if req_fields else _collect_field_list(field_map, c)
        req_cfg = _parse_cfg(c)

        anchor_field = str(c.get("anchor_field", "PX_LAST")).strip()
        force_anchor_history = bool(c.get("force_anchor_history", True))
        if force_anchor_history and anchor_field:
            anchor_rows = _response_rows(sec_list, s0, e0, [anchor_field], req_cfg)
            anchor_df = pd.DataFrame(anchor_rows)

            extra_fields = [f for f in fields if f != anchor_field]
            if extra_fields:
                extra_rows = _response_rows(sec_list, s0, e0, extra_fields, req_cfg)
                extra_df = pd.DataFrame(extra_rows)
            else:
                extra_df = pd.DataFrame()

            if anchor_df.empty:
                raw_df = extra_df
            elif extra_df.empty:
                raw_df = anchor_df
            else:
                # Keep full history from anchor field and enrich with extras where available.
                join_keys = [k for k in ("date", "security") if (k in anchor_df.columns and k in extra_df.columns)]
                if join_keys:
                    raw_df = anchor_df.merge(extra_df, on=join_keys, how="left")
                else:
                    raw_df = anchor_df
        else:
            rows = _response_rows(sec_list, s0, e0, fields, req_cfg)
            raw_df = pd.DataFrame(rows)

    if raw_df.empty:
        return _empty_result()

    if "date" in raw_df.columns:
        if pd.api.types.is_numeric_dtype(raw_df["date"]):
            raw_df["date"] = pd.to_datetime(
                raw_df["date"], unit="D", origin="2000-01-01", errors="coerce"
            ).dt.date
        else:
            ds = raw_df["date"]
            if ds.dtype == object:
                ds = ds.map(lambda z: None if z is None else str(z))
                ds = ds.replace("0Nd", None).replace("0Np", None)
                ds = ds.str.replace("D", "T", regex=False)
                raw_df["date"] = pd.to_datetime(ds, errors="coerce").dt.date
            else:
                raw_df["date"] = pd.to_datetime(ds, errors="coerce").dt.date
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
