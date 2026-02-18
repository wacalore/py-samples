\l p.q

\d .oca

/ Minimal embedPy helper for options_chain_analyzer optimizer
inited:0b
date_mode:`days
epoch:"2000-01-01"
epoch_date:2000.01.01
epoch_ts:2000.01.01D00:00:00.000000000
last_libpath:""

unwrap:{ $[105h=type x; x`.; x] }

init:{[libpath; dm; ep]
  if[libpath~(::); libpath:system "pwd"];
  if[0h=type libpath; libpath:raze libpath];
  if[not libpath~(::);
    if[0=count string libpath; libpath:system "pwd"];
  ];
  if[0h=type libpath; libpath:raze libpath];
  if[10h<>type libpath; libpath:string libpath];
  last_libpath::libpath;
  if[not dm~(::); date_mode::dm];
  if[not ep~(::);
    ep_str:$[10h=type ep; ep; string ep];
    if[0<count ep_str;
      epoch::ep;
      epoch_date::"D"$ep_str;
      epoch_ts::epoch_date + 0D00:00:00.000000000;
    ];
  ];
  .p.e "import sys, importlib";
  cmd:raze ("p = r'''"; libpath; "'''");
  .p.e cmd;
  cmd2:raze ("import sys; p = r'''"; libpath; "'''; sys.path.insert(0,p) if p not in sys.path else None");
  .p.e cmd2;
  .p.e "importlib.invalidate_caches()";
  .p.e "mods=[m for m in list(sys.modules.keys()) if (m=='options_chain_analyzer') or m.startswith('options_chain_analyzer.') or (m=='blp_eco') or m.startswith('blp_eco.')]; [sys.modules.pop(m, None) for m in mods]";
  .p.e "import options_chain_analyzer as oca";
  .p.e "def oca_fix_temporal_df(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if not isinstance(x, pd.DataFrame):\n    return x\n  date_only_cols = {'date', 'expiry', 'reb_date', 'end_date', 'curve_date'}\n  temporal_cols = date_only_cols | {'dt'}\n  out = x.copy()\n  for col in out.columns:\n    c = str(col).lower()\n    if (c not in temporal_cols) and (not c.endswith('_date')):\n      continue\n    s = out[col]\n    if pd.api.types.is_datetime64_any_dtype(s):\n      continue\n    try:\n      if pd.api.types.is_numeric_dtype(s):\n        if c == 'dt':\n          nz = s.dropna()\n          unit = 'D'\n          if len(nz) > 0:\n            vmax = float(nz.abs().max())\n            if vmax > 1.0e10:\n              unit = 'ns'\n          out[col] = pd.to_datetime(s, unit=unit, origin='2000-01-01', errors='coerce')\n        else:\n          out[col] = pd.to_datetime(s, unit='D', origin='2000-01-01', errors='coerce')\n        continue\n    except Exception:\n      pass\n    try:\n      if s.dtype == object:\n        ss = s.map(lambda z: None if z is None else str(z))\n      else:\n        ss = s.astype(str)\n      ss = ss.replace('0Nd', None).replace('0Np', None)\n      ss = ss.str.replace('D', 'T', regex=False)\n      out[col] = pd.to_datetime(ss, errors='coerce')\n    except Exception:\n      try:\n        out[col] = pd.to_datetime(s, errors='coerce')\n      except Exception:\n        pass\n  return out";
  .p.e "def oca_as_df(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if isinstance(x, pd.DataFrame):\n    return oca_fix_temporal_df(x)\n  if isinstance(x, dict) or isinstance(x, (list, tuple)):\n    try:\n      return oca_fix_temporal_df(pd.DataFrame(x))\n    except Exception:\n      return x\n  return x";
  .p.e "def oca_df_to_dict(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if not isinstance(x, pd.DataFrame):\n    return x\n  date_only_cols = {'date', 'expiry', 'reb_date', 'end_date', 'curve_date'}\n  temporal_cols = date_only_cols | {'dt'}\n  out = {}\n  for col in x.columns:\n    s = x[col]\n    c = str(col).lower()\n    date_only = (c in date_only_cols) or c.endswith('_date')\n    is_named_temporal = (c in temporal_cols) or c.endswith('_date')\n    is_dt = pd.api.types.is_datetime64_any_dtype(s)\n    if (not is_dt) and is_named_temporal and (s.dtype == object):\n      for vv in s:\n        if vv is None:\n          continue\n        try:\n          if pd.isna(vv):\n            continue\n        except Exception:\n          pass\n        is_dt = isinstance(vv, pd.Timestamp)\n        if not is_dt:\n          try:\n            pd.Timestamp(vv)\n            is_dt = True\n          except Exception:\n            is_dt = False\n        break\n    if is_dt:\n      fmt = '%Y.%m.%d' if date_only else '%Y.%m.%dD%H:%M:%S.%f'\n      vals = []\n      for vv in s:\n        if vv is None:\n          vals.append('0Nd' if date_only else '0Np')\n          continue\n        try:\n          if pd.isna(vv):\n            vals.append('0Nd' if date_only else '0Np')\n            continue\n        except Exception:\n          pass\n        try:\n          vals.append(pd.Timestamp(vv).strftime(fmt))\n        except Exception:\n          vals.append(vv)\n      out[col] = vals\n    else:\n      out[col] = s.tolist()\n  return out";
  .p.e "def oca_opt_wrapper(tables, cfg=None): return oca.optimize_portfolio_with_pca(tables, cfg)";
  .p.e "def oca_opt_simple_wrapper(tables, cfg=None): return oca.optimize_portfolio(tables, cfg)";
  .p.e "def oca_opt_cvar_wrapper(tables, cfg=None): return oca.optimize_portfolio_cvar(tables, cfg)";
  .p.e "def oca_opt_to_dict(res, date_mode='days', epoch='2000-01-01'): return oca.optimizer_result_to_dict(res, date_mode=date_mode, epoch=epoch)";
  .p.e "def oca_analyze_chain_df(options_df, curve, cfg=None):\n  cfg = {} if cfg is None else cfg\n  options_df = oca_as_df(options_df)\n  curve = oca_as_df(curve)\n  out = oca.analyze_chain_df(options_df, curve, use_numba=cfg.get('use_numba'), group_by_date=cfg.get('group_by_date', cfg.get('group_by_dt')), curve_date_col=cfg.get('curve_date_col'), surface_mode=cfg.get('surface_mode', 'separate'), stitch_contract_col=cfg.get('stitch_contract_col', 'underlying_ric'))\n  return oca_df_to_dict(out)";
  .p.e "def oca_build_strategy_book_df(analytics_df, cfg=None):\n  cfg = {} if cfg is None else cfg\n  analytics_df = oca_as_df(analytics_df)\n  widths = cfg.get('widths', (0.5, 1.0, 2.0))\n  templates = cfg.get('strategy_templates')\n  out = oca.build_strategy_book_df(analytics_df, widths=widths, strategy_templates=templates)\n  return oca_df_to_dict(out)";
  .p.e "def oca_strategy_screener_df(strategy_df, analytics_df=None, cfg=None):\n  cfg = {} if cfg is None else cfg\n  strategy_df = oca_as_df(strategy_df)\n  if analytics_df is not None:\n    analytics_df = oca_as_df(analytics_df)\n  out = oca.strategy_screener_df(\n    strategy_df,\n    analytics_df=analytics_df,\n    vol_col=cfg.get('vol_col', 'iv_atm'),\n    vol_fallback=cfg.get('vol_fallback'),\n    pop_samples=cfg.get('pop_samples', 5000),\n    pop_seed=cfg.get('pop_seed', 7),\n    mispricing_metric=cfg.get('mispricing_metric', 'edge_per_vega'),\n    mispricing_quantile=cfg.get('mispricing_quantile', 0.9),\n    pop_threshold=cfg.get('pop_threshold', 0.6),\n    ev_threshold=cfg.get('ev_threshold', 0.0),\n    upside_metric=cfg.get('upside_metric', 'upside_p95'),\n    upside_quantile=cfg.get('upside_quantile', 0.8),\n    upside_threshold=cfg.get('upside_threshold'),\n    credit_debit_tolerance=cfg.get('credit_debit_tolerance', 1.0e-8),\n    filter_only=cfg.get('filter_only', True),\n    top_n=cfg.get('top_n'),\n    weight_mispricing=cfg.get('weight_mispricing', 1.0),\n    weight_ev=cfg.get('weight_ev', 0.5),\n    weight_pop=cfg.get('weight_pop', 0.5)\n  )\n  return oca_df_to_dict(out)";
  .p.e "def oca_scenario_pnl_strategy_df(strategy_df, cfg=None):\n  cfg = {} if cfg is None else cfg\n  strategy_df = oca_as_df(strategy_df)\n  out = oca.scenario_pnl_strategy_df(\n    strategy_df,\n    dF=cfg.get('dF', 0.0),\n    dVol=cfg.get('dVol', 0.0),\n    dRate=cfg.get('dRate', 0.0),\n    dt_days=cfg.get('dt_days', 0.0)\n  )\n  return oca_df_to_dict(out)";
  .p.e "def oca_bbg_eco_history(securities, start_date, end_date, cfg=None):\n  cfg = {} if cfg is None else cfg\n  import importlib\n  import blp_eco\n  importlib.reload(blp_eco)\n  out = blp_eco.get_eco_history(securities, start_date, end_date, cfg=cfg)\n  return oca_df_to_dict(out)";
  opt_wrapper::.p.get[`oca_opt_wrapper];
  opt_wrapper_simple::.p.get[`oca_opt_simple_wrapper];
  opt_wrapper_cvar::.p.get[`oca_opt_cvar_wrapper];
  opt_to_dict::.p.get[`oca_opt_to_dict];
  analyze_chain_wrapper::.p.get[`oca_analyze_chain_df];
  strategy_book_wrapper::.p.get[`oca_build_strategy_book_df];
  strategy_screener_wrapper::.p.get[`oca_strategy_screener_df];
  scenario_pnl_strategy_wrapper::.p.get[`oca_scenario_pnl_strategy_df];
  bbg_eco_wrapper::.p.get[`oca_bbg_eco_history];
  inited::1b;
  :1b;
 }

ensure_init:{[libpath]
  lp:libpath;
  if[lp~(::);
    if[10h=type last_libpath;
      if[0<count last_libpath; lp:last_libpath];
    ];
  ];
  if[not inited; init[lp;date_mode;epoch]];
 }

reload:{[libpath]
  lp:libpath;
  if[lp~(::);
    if[10h=type last_libpath;
      if[0<count last_libpath; lp:last_libpath];
    ];
  ];
  inited::0b;
  init[lp;date_mode;epoch]
 }

is_date_key:{[k]
  s:lower string k;
  (s~"date") or (s~"expiry") or (s~"reb_date") or (s~"end_date") or (s~"curve_date") or s like "*_date"
 }

cast_date_like:{[v; k; dm_str]
  s:lower string k;
  t:abs type v;
  if[t in 14 12 15h; :v];
  if[s~"dt";
    if[t in 6 7h;
      if[dm_str in ("days";"day"); : .oca.epoch_date + `int$v];
      if[dm_str in ("ns";"nanoseconds";"timestamp";"datetime64[ns]"); : .oca.epoch_ts + `long$v];
      :v;
    ];
    if[dm_str in ("ns";"nanoseconds";"timestamp";"datetime64[ns]");
      if[t=11h; : "P"$string each v];
      if[t=0h; : "P"$v];
      :v;
    ];
    if[t=11h; : "D"$string each v];
    if[t=0h; : "D"$v];
    :v;
  ];
  if[not .oca.is_date_key k; :v];
  if[t in 6 7h; : .oca.epoch_date + `int$v];
  if[t=11h; : `date$"P"$string each v];
  if[t=0h; : `date$"P"$v];
  :v;
 }

fix_dt:{[t]
  dm_str:$[10h=type .oca.date_mode; .oca.date_mode; string .oca.date_mode];
  if[99h=type t;
    if[98h=type key t; :fix_dt 0!t];
    k:key t;
    klist:$[10h=type k; enlist k; k];
    ksym:$[11h=type klist; klist; `$string each klist];
    j:0;
    while[j<count ksym;
      kn:ksym j;
      if[(kn=`dt) or .oca.is_date_key kn;
        kk:$[10h=type k; k; k j];
        t[kk]:.oca.cast_date_like[t kk; kn; dm_str];
      ];
      j+:1;
    ];
    :t;
  ];
  if[98h<>type t; :t];
  d:flip t;
  d:.oca.fix_dt d;
  :flip d;
 }

to_table:{[v]
  if[98h=type v; :fix_dt v];
  if[99h=type v;
    if[98h=type key v; :fix_dt 0!v];
    k:key v;
    if[0=count k; :([])];
    if[11h=type k; :fix_dt flip v];
    if[10h=type k; :fix_dt flip ((`$k)!value v)];
    sym_key:{[x] $[11h=type x; x; 10h=type x; `$x; `$string x]};
    ksym: sym_key each k;
    if[(count distinct ksym) <> count ksym; '"non-unique keys after symbolization"];
    :fix_dt flip (ksym!value v);
  ];
  v
 }

args_dict:{[args]
  $[99h=type args; args;
    a:$[0h=type args; args; enlist args];
    if[(count a)<5; a:a,(5-count a)#(::)];
    (`tables`cfg`date_mode`epoch`libpath)!a
   ]
 }

normalize_cfg:{[cfg]
  $[cfg~(::); ()!(); cfg]
 }

cfg_to_dict:{[cfg]
  c0:$[cfg~(::); ()!(); cfg];
  t:type c0;
  if[t=99h;
    if[98h=type key c0; :()!()];
    :c0
  ];
  if[t=20h;
    k:.oca.to_sym key c0;
    v:value c0;
    vv:$[(type v)>0h and (count v)=1; first v; v];
    :(enlist k)!enlist vv
  ];
  ()!()
 }

to_sym:{[x]
  t:type x;
  $[t=-11h; x;
    t=11h; first x;
    t=10h; first `$enlist x;
    `$string x]
 }

norm_side:{[side]
  if[side~(::); :1f];
  ts:type side;
  ats:abs ts;
  if[(ts<0h) and (ats in 1 4 5 6 7 8 9h); :1f*side];
  if[(ts>0h) and (ats in 1 4 5 6 7 8 9h);
    if[0=count side; '"side must be non-empty"];
    :1f*first side;
  ];
  if[ts=0h;
    if[0=count side; '"side must be non-empty"];
    :.oca.norm_side first side;
  ];
  s:$[ats=11h; lower string $[ts<0h; side; first side];
      ats=10h; lower $[ts<0h; side; first side];
      ""];
  sy:.oca.to_sym s;
  if[(sy~`1) or (sy~`long) or (sy~`buy) or (sy~`b) or (sy~`l) or (sy~`pos) or (sy~`positive) or (sy~`f) or (s~"+1") or (s~"1"); :1f];
  if[(sy~`short) or (sy~`sell) or (sy~`s) or (sy~`neg) or (sy~`negative) or (s~"-1"); :-1f];
  '"side must be numeric (+/-1) or one of `long/`short/`buy/`sell"
 }

norm_wing_steps:{[wing_steps]
  if[wing_steps~(::); :1];
  ts:type wing_steps;
  ats:abs ts;
  if[(ts<0h) and (ats in 1 4 5 6 7 8 9h);
    wi:`int$wing_steps;
    if[wi<1; '"wing_steps must be >= 1"];
    :wi;
  ];
  if[(ts>0h) and (ats in 1 4 5 6 7 8 9h);
    if[0=count wing_steps; '"wing_steps must be non-empty"];
    :.oca.norm_wing_steps first wing_steps;
  ];
  if[ts=0h;
    if[0=count wing_steps; '"wing_steps must be non-empty"];
    :.oca.norm_wing_steps first wing_steps;
  ];
  '"wing_steps must be an integer >= 1"
 }

norm_wing_mode:{[wing_mode]
  if[wing_mode~(::); :`rank];
  ts:type wing_mode;
  ats:abs ts;
  if[(ts>0h) or (ts=0h);
    if[0=count wing_mode; :`rank];
    :.oca.norm_wing_mode first wing_mode;
  ];
  s:$[ats=11h; lower string wing_mode;
      ats=10h; lower wing_mode;
      lower string wing_mode];
  if[(s~"rank") or (s~"step") or (s~"index") or (s~"ladder"); :`rank];
  if[(s~"strict") or (s~"strict_symmetric") or (s~"symmetric") or (s~"sym"); :`strict_symmetric];
  '"wing_mode must be `rank or `strict_symmetric"
 }

norm_wing_step_size:{[wing_step_size]
  if[wing_step_size~(::); :(::)];
  ts:type wing_step_size;
  ats:abs ts;
  if[(ts>0h) and (ats in 1 4 5 6 7 8 9h);
    if[0=count wing_step_size; :(::)];
    :.oca.norm_wing_step_size first wing_step_size;
  ];
  if[ts=0h;
    if[0=count wing_step_size; :(::)];
    :.oca.norm_wing_step_size first wing_step_size;
  ];
  if[(ts<0h) and (ats in 1 4 5 6 7 8 9h);
    w:1f*wing_step_size;
    if[w<=0f; '"wing_step_size must be > 0"];
    :w;
  ];
  '"wing_step_size must be numeric > 0 or ::"
 }

side_wing_cfg:{[side]
  if[99h=type side;
    ks:key side;
    s0:$[`side in ks; side`side; 1f];
    w0:$[`wing_steps in ks; side`wing_steps; 1];
    m0:$[`wing_mode in ks; side`wing_mode; `rank];
    z0:$[`wing_step_size in ks; side`wing_step_size; (::)];
    :(`side`wing_steps`wing_mode`wing_step_size)!(
      .oca.norm_side s0;
      .oca.norm_wing_steps w0;
      .oca.norm_wing_mode m0;
      .oca.norm_wing_step_size z0);
  ];
  (`side`wing_steps`wing_mode`wing_step_size)!(.oca.norm_side side; 1; `rank; (::))
 }

norm_bool:{[v; d]
  if[v~(::); :d];
  tv:type v;
  atv:abs tv;
  if[(tv<0h) and (atv in 1 4 5 6 7 8 9h); :0<>v];
  if[(tv>0h) and (atv in 1 4 5 6 7 8 9h);
    if[0=count v; :d];
    :0<>first v;
  ];
  if[tv=0h;
    if[0=count v; :d];
    :.oca.norm_bool[first v; d];
  ];
  s:lower string .oca.to_sym v;
  if[(s~"1") or (s~"true") or (s~"t") or (s~"yes") or (s~"y") or (s~"on"); :1b];
  if[(s~"0") or (s~"false") or (s~"f") or (s~"no") or (s~"n") or (s~"off"); :0b];
  d
 }

price_cfg:{[price_mode]
  if[99h<>type price_mode;
    pm:$[price_mode~(::); `market; .oca.to_sym price_mode];
    :(`mode`price_col`anchor_col`fallback_col`fallback_weird`weird_rel_lo`weird_rel_hi)!(pm; `settle; `settle; `theo; 1b; 0.2f; 5f);
  ];
  ks:key price_mode;
  m0:$[`mode in ks; price_mode`mode; $[`price_mode in ks; price_mode`price_mode; `market]];
  p0:$[`price_col in ks; price_mode`price_col; `settle];
  a0:$[`anchor_col in ks; price_mode`anchor_col; $[`settle_col in ks; price_mode`settle_col; `settle]];
  f0:$[`fallback_col in ks; price_mode`fallback_col; `theo];
  fw0:$[`fallback_weird in ks; price_mode`fallback_weird; 1b];
  lo0:$[`weird_rel_lo in ks; price_mode`weird_rel_lo; 0.2f];
  hi0:$[`weird_rel_hi in ks; price_mode`weird_rel_hi; 5f];
  (`mode`price_col`anchor_col`fallback_col`fallback_weird`weird_rel_lo`weird_rel_hi)!(
    .oca.to_sym m0;
    .oca.to_sym p0;
    .oca.to_sym a0;
    .oca.to_sym f0;
    .oca.norm_bool[fw0; 1b];
    1f*lo0;
    1f*hi0)
 }

norm_put_call:{[v]
  / Normalize put/call tags to `C/`P so downstream masks are type-stable.
  t:abs type v;
  to_s1:{[x]
    tx:abs type x;
    $[tx=10h; x; tx=11h; string x; string x]
   };
  s:$[t=11h; string each v;
      t=10h; enlist v;
      t=0h; to_s1 each v;
      ::];
  if[s~(::); :v];
  u:upper each s;
  map1:{[x] $[(x~"C") or (x~"CALL"); `C; (x~"P") or (x~"PUT"); `P; .oca.to_sym x]};
  map1 each u
 }

norm_quote_perm_id:{[v]
  to1:{[x]
    to_s1:{[y]
      ty:abs type y;
      if[ty=10h; :y];
      if[ty=11h; :string y];
      if[ty=0h;
        if[0=count y; :""];
        ys:.z.s each y;
        if[10h=type ys; :ys];
        if[0h=type ys; :raze ys];
        :string ys;
      ];
      string y
    };
    tx:abs type x;
    if[x~(::); :`];
    if[tx in 8 9h;
      if[null x; :`];
    ];
    if[tx=11h; :x];
    if[tx=10h; :$[0=count x; `; `$x]];
    sx:lower to_s1 x;
    if[(sx~"0n") or (sx~"0w") or (sx~"nan") or (sx~"none") or (sx~"null") or (sx~""); :`];
    `$sx
  };
  t:abs type v;
  $[t=11h; v;
    t=10h; enlist to1 v;
    t=0h; to1 each v;
    to1 each enlist v]
 }

as_tables:{[tbls]
  $[98h=type tbls; enlist tbls;
    99h=type tbls; enlist 0!tbls;
    0h=type tbls; tbls;
    enlist tbls]
 }

full_cfg:{[tbls; cfg]
  c: normalize_cfg cfg;
  ts: as_tables tbls;
  if[0=count ts; '"tables must be non-empty"];
  dtc:$[`dt_col in key c; c`dt_col; `dt];
  dtc: .oca.to_sym dtc;
  if[not dtc in cols first ts;
    if[`dt_col in key c; '"dt_col not found in tables"];
    dtc:$[`dt in cols first ts; `dt; `time];
  ];
  if[not all dtc in/: cols each ts; '"dt column not found in all tables"];
  n: count distinct raze ({[t;c] t c})'[ts; (count ts)#enlist dtc];
  c[`window]: n;
  c[`min_periods]: n;
  c
 }

norm_strategy:{[strategy]
  s: lower string .oca.to_sym $[strategy~(::); `straddle; strategy];
  if[(s~"straddle") or (s~"atm_straddle"); :`straddle];
  if[s~"call"; :`call];
  if[s~"put"; :`put];
  if[(s~"call_spread") or (s~"bull_call_spread"); :`call_spread];
  if[(s~"put_spread") or (s~"bull_put_spread"); :`put_spread];
  if[s~"strangle"; :`strangle];
  if[(s~"risk_reversal") or (s~"rr"); :`risk_reversal];
  if[s~"collar"; :`collar];
  if[s~"call_fly"; :`call_fly];
  if[s~"put_fly"; :`put_fly];
  if[(s~"iron_fly") or (s~"iron_butterfly"); :`iron_fly];
  if[s~"iron_condor"; :`iron_condor];
  if[s~"call_ratio"; :`call_ratio];
  if[s~"put_ratio"; :`put_ratio];
  if[s~"synthetic_long"; :`synthetic_long];
  if[s~"synthetic_short"; :`synthetic_short];
  '"unknown strategy"
 }

default_atm_strategies:{
  `straddle`call`put`call_spread`put_spread`strangle`risk_reversal`collar`call_fly`put_fly`iron_fly`iron_condor`call_ratio`put_ratio`synthetic_long`synthetic_short
 }

as_strategy_list:{[strategies]
  if[strategies~(::); : .oca.default_atm_strategies[]];
  t:type strategies;
  if[t=11h; : .oca.norm_strategy each strategies];
  if[t=10h; : enlist .oca.norm_strategy `$strategies];
  if[t=0h; : .oca.norm_strategy each strategies];
  enlist .oca.norm_strategy strategies
 }

strategy_strike_at:{[strikes; pos; off]
  idx: pos + off;
  if[(idx<0) or (idx>=count strikes); : 0n];
  strikes idx
 }

strategy_find_strike:{[strikes; target]
  if[0=count strikes; :0n];
  d: abs strikes - target;
  md: exec min d from ([] d:d);
  tol: 0.00000001 + 0.0000000001 * abs target;
  if[md>tol; :0n];
  first strikes where d=md
 }

strategy_sym_dists:{[strikes; k0]
  if[0=count strikes; :0#0f];
  ds0: asc distinct k0 - strikes where strikes < k0;
  ds0: ds0 where ds0 > 0;
  if[0=count ds0; :0#0f];
  ds:0#0f;
  i:0;
  while[i<count ds0;
    d:(ds0 i);
    kl:.oca.strategy_find_strike[strikes; k0-d];
    kr:.oca.strategy_find_strike[strikes; k0+d];
    if[(not null kl) and (not null kr); ds,:enlist d];
    i+:1;
  ];
  asc distinct ds
 }

strategy_legs_wing:{[sub; atm; strat; wing_steps; wing_mode; wing_step_size; require_quotes]
  if[0=count sub; :()];
  ws:.oca.norm_wing_steps wing_steps;
  wm:.oca.norm_wing_mode wing_mode;
  wss:.oca.norm_wing_step_size wing_step_size;
  rq:.oca.norm_bool[require_quotes; 1b];
  strikes: asc distinct sub`strike;
  if[0=count strikes; :()];
  d: abs strikes - atm;
  md: exec min d from ([] d:d);
  atm_k: first strikes where d=md;
  pos: first where strikes=atm_k;
  k0: atm_k;
  use_sym:(wm=`strict_symmetric) and strat in `strangle`risk_reversal`collar`call_fly`put_fly`iron_fly`iron_condor;
  km2:0n;
  km1:0n;
  kp1:0n;
  kp2:0n;
  d1:$[wss~(::); 0n; ws*wss];
  d2:$[wss~(::); 0n; 2*ws*wss];
  if[use_sym;
    if[wss~(::);
      sds:.oca.strategy_sym_dists[strikes; k0];
      d1:$[ws<=count sds; sds (ws-1); 0n];
      d2:$[(2*ws)<=count sds; sds ((2*ws)-1); 0n];
    ];
    km1:$[null d1; 0n; .oca.strategy_find_strike[strikes; k0-d1]];
    kp1:$[null d1; 0n; .oca.strategy_find_strike[strikes; k0+d1]];
    km2:$[null d2; 0n; .oca.strategy_find_strike[strikes; k0-d2]];
    kp2:$[null d2; 0n; .oca.strategy_find_strike[strikes; k0+d2]];
  ];
  if[not use_sym;
    if[wss~(::);
      km2: .oca.strategy_strike_at[strikes; pos; 0 - 2*ws];
      km1: .oca.strategy_strike_at[strikes; pos; 0 - ws];
      kp1: .oca.strategy_strike_at[strikes; pos; ws];
      kp2: .oca.strategy_strike_at[strikes; pos; 2*ws];
    ];
    if[not wss~(::);
      km1:.oca.strategy_find_strike[strikes; k0-d1];
      kp1:.oca.strategy_find_strike[strikes; k0+d1];
      km2:.oca.strategy_find_strike[strikes; k0-d2];
      kp2:.oca.strategy_find_strike[strikes; k0+d2];
    ];
  ];

  pcs:();
  ks:();
  qs:();

  if[strat=`call; pcs:enlist `C; ks:enlist k0; qs:enlist 1f];
  if[strat=`put; pcs:enlist `P; ks:enlist k0; qs:enlist 1f];
  if[strat=`straddle; pcs:`C`P; ks:k0,k0; qs:1f,1f];
  if[strat=`call_spread; pcs:`C`C; ks:k0,kp1; qs:1f,-1f];
  if[strat=`put_spread; pcs:`P`P; ks:k0,km1; qs:1f,-1f];
  if[strat=`strangle; pcs:`C`P; ks:kp1,km1; qs:1f,1f];
  if[strat=`risk_reversal; pcs:`C`P; ks:kp1,km1; qs:1f,-1f];
  if[strat=`collar; pcs:`C`P; ks:kp1,km1; qs:-1f,1f];
  if[strat=`call_fly; pcs:`C`C`C; ks:km1,k0,kp1; qs:1f,-2f,1f];
  if[strat=`put_fly; pcs:`P`P`P; ks:km1,k0,kp1; qs:1f,-2f,1f];
  if[strat=`iron_fly; pcs:`P`P`C`C; ks:km1,k0,k0,kp1; qs:1f,-1f,-1f,1f];
  if[strat=`iron_condor; pcs:`P`P`C`C; ks:km2,km1,kp1,kp2; qs:1f,-1f,-1f,1f];
  if[strat=`call_ratio; pcs:`C`C; ks:k0,kp1; qs:1f,-2f];
  if[strat=`put_ratio; pcs:`P`P; ks:k0,km1; qs:1f,-2f];
  if[strat=`synthetic_long; pcs:`C`P; ks:k0,k0; qs:1f,-1f];
  if[strat=`synthetic_short; pcs:`C`P; ks:k0,k0; qs:-1f,1f];

  if[0=count pcs; :()];
  if[any null ks; :()];

  legs: flip `put_call`strike`qty!(pcs; ks; qs);
  if[not rq; :legs];
  avail: 0!select hasC:any put_call=`C, hasP:any put_call=`P by strike from sub;
  ok:1b;
  i:0;
  while[i<count legs;
    pc: (legs`put_call) i;
    k: (legs`strike) i;
    a: avail where (avail`strike)=k;
    if[0=count a;
      ok:0b;
      i:count legs;
    ];
    if[ok;
      has:$[pc=`C; first a`hasC; first a`hasP];
      if[not has;
        ok:0b;
        i:count legs;
      ];
    ];
    i+:1;
  ];
  if[not ok; :()];
  legs
 }

strategy_legs:{[sub; atm; strat]
  .oca.strategy_legs_wing[sub; atm; strat; 1; `rank; (::); 1b]
 }

atm_strike_order:{[sub; u; strat]
  if[0=count sub; :`float$()];
  cand:([] strike:asc distinct sub`strike);
  cand:update m:abs(strike-u) from cand;
  / ATM strike is nearest to underlying; tie-break by lower strike.
  cand:cand @ iasc cand`strike;
  cand:cand @ iasc cand`m;
  cand`strike
 }

rep_underlying:{[sub]
  if[0=count sub; :0n];
  u:sub`underlying;
  ok:u where not null u;
  if[0<count ok; :med ok];
  first u
 }

strike_eq_mask:{[x; k; tol]
  t:abs type x;
  if[t in 8 9h; :(abs(x-k))<=tol];
  if[t=0h;
    xf: .[`float$; enlist x; {::}];
    if[not xf~(::); :(abs(xf-k))<=tol];
  ];
  x=k
 }

lin_interp:{[x; y; x0]
  n:count x;
  if[0=n; :0n];
  if[1=n; : first y];
  ord:iasc x;
  xs:(`float$x) ord;
  ys:(`float$y) ord;
  if[x0<=first xs;
    x1:first xs; x2:xs 1; y1:first ys; y2:ys 1;
    if[x2=x1; :y1];
    :y1 + (y2-y1)*(x0-x1)%(x2-x1);
  ];
  if[x0>=last xs;
    x1:xs (n-2); x2:last xs; y1:ys (n-2); y2:last ys;
    if[x2=x1; :y2];
    :y1 + (y2-y1)*(x0-x1)%(x2-x1);
  ];
  i:first where xs>=x0;
  if[(xs i)=x0; :ys i];
  x1:xs (i-1); x2:xs i; y1:ys (i-1); y2:ys i;
  if[x2=x1; :y1];
  y1 + (y2-y1)*(x0-x1)%(x2-x1)
 }

fallback_leg_price:{[tt; d; exp_date; ric; pc; k; fcol]
  m:(tt`date)=d;
  m:m & (tt`expiry)=exp_date;
  if[ric<>`; m:m & ((tt`underlying_ric)=ric)];
  m:m & ((tt`put_call)=pc);
  m:m & not null tt[;fcol];
  m:0<>m;
  sub:tt where m;
  if[0=count sub; :0n];
  tol: 0.00000001 + 0.0000000001 * abs k;
  ex:sub where .oca.strike_eq_mask[sub`strike; k; tol];
  if[0<count ex;
    pxs:ex[;fcol];
    pxs:pxs where not null pxs;
    if[0<count pxs; :med `float$pxs];
  ];
  base:([] strike:`float$sub`strike; px:`float$sub[;fcol]);
  base:base where not null base`px;
  if[0=count base; :0n];
  base:0!select px:med px by strike from base;
  .oca.lin_interp[`float$base`strike; `float$base`px; 1f*k]
 }

fallback_leg_cap:{[tt; d; exp_date; ric; pc; k]
  m:(tt`date)=d;
  m:m & (tt`expiry)=exp_date;
  if[ric<>`; m:m & ((tt`underlying_ric)=ric)];
  m:0<>m;
  sub:tt where m;
  if[0=count sub; :0n];
  u:.oca.rep_underlying sub;
  uf: .[`float$; enlist u; {0n}];
  kf:1f*k;
  if[pc=`C;
    if[null uf; :0n];
    : max 0f, uf;
  ];
  max 0f, kf
 }

atm_pick_score:{[sub; k; u; strat]
  / Keep score strictly on ATM distance so strike choice is underlying-nearest only.
  abs(k-u)
 }

legs_desc:{[legs]
  if[(98h<>type legs) or (0=count legs); :""];
  d:();
  i:0;
  while[i<count legs;
    pc:(legs`put_call) i;
    k:(legs`strike) i;
    q:(legs`qty) i;
    d,: enlist raze (string pc; "@"; string k; "x"; string q);
    i+:1;
  ];
  "," sv d
 }

atm_strategy_returns:{[t; rebalance_days; target_dte; min_dte; max_dte; price_mode; strategy; side]
  r: $[rebalance_days~(::); 5; rebalance_days];
  td: $[target_dte~(::); 30; target_dte];
  mind: $[min_dte~(::); 7; min_dte];
  maxd: $[max_dte~(::); ::; max_dte];
  pcfg:.oca.price_cfg price_mode;
  pm_in:pcfg`mode;
  market_price_col:pcfg`price_col;
  anchor_col:pcfg`anchor_col;
  fallback_col:pcfg`fallback_col;
  fallback_weird:pcfg`fallback_weird;
  weird_rel_lo:pcfg`weird_rel_lo;
  weird_rel_hi:pcfg`weird_rel_hi;
  known_pm:`market`mkt`settle`theo`market_cont`mkt_cont`settle_cont`theo_cont`market_reset`mkt_reset`settle_reset`theo_reset`market_fallback`mkt_fallback`settle_fallback`market_fallback_cont`mkt_fallback_cont`settle_fallback_cont`market_fallback_reset`mkt_fallback_reset`settle_fallback_reset;
  if[not pm_in in known_pm; '"unknown price_mode"];
  cont_mode: not pm_in in `market_reset`mkt_reset`settle_reset`theo_reset`market_fallback_reset`mkt_fallback_reset`settle_fallback_reset;
  pm: $[pm_in in `market`mkt`settle`market_cont`mkt_cont`settle_cont`market_reset`mkt_reset`settle_reset`market_fallback`mkt_fallback`settle_fallback`market_fallback_cont`mkt_fallback_cont`settle_fallback_cont`market_fallback_reset`mkt_fallback_reset`settle_fallback_reset; `market; `theo];
  strat: .oca.norm_strategy strategy;
  fallback_mode: pm_in in `market_fallback`mkt_fallback`settle_fallback`market_fallback_cont`mkt_fallback_cont`settle_fallback_cont`market_fallback_reset`mkt_fallback_reset`settle_fallback_reset;
  / For straddles, keep strict nearest-ATM strike and synthesize missing leg marks if needed.
  require_quotes:not (fallback_mode or (strat=`straddle));
  swcfg:.oca.side_wing_cfg side;
  s:swcfg`side;
  ws:swcfg`wing_steps;
  wm:swcfg`wing_mode;
  wss:swcfg`wing_step_size;
  price_col: $[pm=`market; market_price_col; `theo];
  req: `date`expiry`strike`put_call`underlying;
  if[not all req in cols t; '"analytics table missing required columns"];
  if[not price_col in cols t; '"analytics table missing price column"];
  if[fallback_mode;
    if[not fallback_col in cols t; '"analytics table missing fallback_col required for market_fallback mode"];
    if[weird_rel_lo<=0f; weird_rel_lo:0.2f];
    if[weird_rel_hi<=weird_rel_lo; weird_rel_hi:5f];
  ];
  r: max 1, `int$r;
  if[r < 1; '"rebalance_days must be >= 1"];
  tt: t;
  dty: abs type (tt`date);
  if[dty in 12 15h; tt: update date:date date from tt];
  if[not dty in 14 12 15h; '"date column must be date or timestamp/datetime"];
  ety: abs type (tt`expiry);
  if[ety in 12 15h; tt: update expiry:date expiry from tt];
  if[not ety in 14 12 15h; '"expiry column must be date or timestamp/datetime"];
  tt: update put_call:.oca.norm_put_call put_call from tt;
  if[not all ((tt`put_call) in `C`P); '"put_call column must contain C/P (or call/put)"];
  if[`quote_perm_id in cols tt;
    tt:update quote_perm_id:.oca.norm_quote_perm_id quote_perm_id from tt;
    / Collapse duplicate quote rows so each date+quote_perm_id contributes once.
    t_nn: tt where (tt`quote_perm_id)<>`;
    t_n0: tt where (tt`quote_perm_id)=`;
    if[0<count t_nn;
      ix:0!select i:first i by date,quote_perm_id from update i:til count t_nn from t_nn;
      t_nn: t_nn ix`i;
    ];
    if[0<count t_n0; t_n0: distinct t_n0];
    tt:$[0<count t_nn; $[0<count t_n0; t_nn,t_n0; t_nn]; t_n0];
  ];
  have_ric:`underlying_ric in cols tt;
  if[have_ric; tt:update underlying_ric:.oca.to_sym each underlying_ric from tt];
  use_mkt_anchor: (price_col=`theo) and (anchor_col in cols tt);
  if[use_mkt_anchor; tt:update anchor_sel:tt[;anchor_col] from tt];
  tt: update price_sel: tt[;price_col] from tt;
  if[fallback_mode; tt: update fallback_sel: tt[;fallback_col] from tt];
  tt: update dte: expiry - date from tt;
  dates: asc distinct tt`date;
  idx: til `int$count dates;
  reb_dates: dates where (idx mod r) = 0;

  pick:{[d; td; mind; maxd; tt; strat; use_mkt_anchor; sw]
    ws:sw`wing_steps;
    wm:sw`wing_mode;
    wss:sw`wing_step_size;
    rq:sw`require_quotes;
    have_ric:`underlying_ric in cols tt;
    sub: tt where (tt`date)=d;
    sub: sub where (sub`dte) >= mind;
    if[not maxd~(::); sub: sub where (sub`dte) <= maxd];
    if[0=count sub; :()];
    sub_pick: sub;
    if[use_mkt_anchor;
      sub_mkt: sub where not null sub`anchor_sel;
      if[0<count sub_mkt; sub_pick: sub_mkt];
    ];
    if[0=count sub_pick; :()];
    exp_tbl: 0!select dte:first dte by expiry from sub_pick;
    diffs: abs ((exp_tbl`dte) - td);
    md: exec min d from ([] d: diffs);
    exp_exp: exp_tbl`expiry;
    exp_sel: exp_exp where diffs = md;
    if[0=count exp_sel; :()];
    exp_sel: exp_sel 0;
    sub2: sub_pick where (sub_pick`expiry)=exp_sel;
    if[0=count sub2; :()];
    if[have_ric;
      ric_vals: asc distinct sub2`underlying_ric;
      k:(::);
      u:(::);
      ric_sel:`;
      best_sc:0w;
      j:0;
      while[j<count ric_vals;
        r1:(ric_vals j);
        subr: sub2 where (sub2`underlying_ric)=r1;
        if[0<count subr;
          u1:.oca.rep_underlying subr;
          ks: .oca.atm_strike_order[subr; u1; strat];
          best_k_r:(::);
          best_sc_r:0w;
          i:0;
          while[i<count ks;
            k0:(ks i);
            legs:.oca.strategy_legs_wing[subr; k0; strat; ws; wm; wss; rq];
            if[(98h=type legs) and (0<count legs);
              sc:.oca.atm_pick_score[subr; k0; u1; strat];
              if[(best_k_r~(::)) or (sc<best_sc_r);
                best_k_r:k0;
                best_sc_r:sc;
              ];
            ];
            i+:1;
          ];
          if[not best_k_r~(::);
            if[(ric_sel~`) or (best_sc_r<best_sc);
              best_sc:best_sc_r;
              k:best_k_r;
              u:u1;
              ric_sel:r1;
            ];
          ];
        ];
        j+:1;
      ];
      if[(k~(::)) or (ric_sel~`); :()];
      :(`reb_date`expiry`strike`underlying`underlying_ric)! (d; exp_sel; k; u; ric_sel);
    ];
    u: .oca.rep_underlying sub2;
    ks: .oca.atm_strike_order[sub2; u; strat];
    if[0=count ks; :()];
    k:(::);
    best_sc_k:0w;
    i:0;
    while[i<count ks;
      k0:(ks i);
      legs:.oca.strategy_legs_wing[sub2; k0; strat; ws; wm; wss; rq];
      if[(98h=type legs) and (0<count legs);
        sc:.oca.atm_pick_score[sub2; k0; u; strat];
        if[(k~(::)) or (sc<best_sc_k);
          k:k0;
          best_sc_k:sc;
        ];
      ];
      i+:1;
    ];
    if[k~(::); :()];
    (`reb_date`expiry`strike`underlying`underlying_ric)! (d; exp_sel; k; u; `)
  };

  swpick:(`wing_steps`wing_mode`wing_step_size`require_quotes)!(ws; wm; wss; require_quotes);
  picks: pick'[reb_dates; (count reb_dates)#enlist td; (count reb_dates)#enlist mind; (count reb_dates)#enlist maxd; (count reb_dates)#enlist tt; (count reb_dates)#enlist strat; (count reb_dates)#enlist use_mkt_anchor; (count reb_dates)#enlist swpick];
  picks: picks except enlist ();
  if[0=count picks; '"no valid rebalance dates"];
  picks_tbl: flip (`reb_date`expiry`strike`underlying`underlying_ric)! (picks`reb_date; picks`expiry; picks`strike; picks`underlying; picks`underlying_ric);

  reb: picks_tbl`reb_date;
  end_dates: 1 _ reb, enlist (1 + last dates);

  seg_tbl: update end_date:end_dates from picks_tbl;
  env: (`tt`dates`strat`side`price_mode`cont_mode`wing_steps`wing_mode`wing_step_size`require_quotes`fallback_mode`fallback_col`fallback_weird`weird_rel_lo`weird_rel_hi)!(
    tt; dates; strat; s; pm; cont_mode; ws; wm; wss; require_quotes; fallback_mode; fallback_col; fallback_weird; weird_rel_lo; weird_rel_hi);

  seg_fn:{[seg; env]
    rb: seg`reb_date;
    re: seg`end_date;
    exp_date: seg`expiry;
    strike: seg`strike;
    ric: seg`underlying_ric;
    tt: env`tt;
    dates: env`dates;
    strat: env`strat;
    s: env`side;
    pm: env`price_mode;
    cont: env`cont_mode;
    ws: env`wing_steps;
    wm: env`wing_mode;
    wss: env`wing_step_size;
    rq: env`require_quotes;
    fb_mode: env`fallback_mode;
    fb_col: env`fallback_col;
    fb_weird: env`fallback_weird;
    fb_lo: env`weird_rel_lo;
    fb_hi: env`weird_rel_hi;
    seg_dates: $[cont; dates where (dates>=rb) & (dates<=re); dates where (dates>=rb) & (dates<re)];
    if[0=count seg_dates; :()];
    sub_rb: tt where (tt`date)=rb;
    sub_rb: sub_rb where (sub_rb`expiry)=exp_date;
    if[ric<>`; sub_rb: sub_rb where (sub_rb`underlying_ric)=ric];
    legs: .oca.strategy_legs_wing[sub_rb; strike; strat; ws; wm; wss; rq];
    if[(98h<>type legs) or (0=count legs); :()];
    leg_desc: `$ .oca.legs_desc legs;
    have_qid:`quote_perm_id in cols tt;
    ord_cand:{[x; pref]
      if[0=count x; :x];
      x: x @ iasc x`quote_perm_id;
      x: update qrank:til count x from x;
      nc:$[`n_cov in cols x; x`n_cov; (count x)#1];
      nmx:$[0<count nc; max nc; 1];
      pf: .[`float$; enlist pref; {0n}];
      pgap: x`gap;
      if[not null pf; pgap:abs((`float$x`px) - pf)];
      sc:(1f*pgap) + 0.000000001f*(1f*(nmx - nc)) + 0.000000000001f*(1f*x`qrank);
      x: x @ iasc sc;
      x
    };
    leg_tbls:();
    i:0;
    while[i<count legs;
      pc: (legs`put_call) i;
      k: (legs`strike) i;
      q: (legs`qty) i;
      tol: 0.00000001 + 0.0000000001 * abs k;
      mask0: (tt`date) in seg_dates;
      mask0: mask0 & (tt`expiry)=exp_date;
      if[ric<>`; mask0: mask0 & ((tt`underlying_ric)=ric)];
      mask0: mask0 & ((tt`put_call)=pc);
      mask0: mask0 & .oca.strike_eq_mask[tt`strike; k; tol];
      mask0: 0 <> mask0;
      leg0_all: tt where mask0;
      if[not `quote_perm_id in cols leg0_all;
        leg0_all:update quote_perm_id:(count leg0_all)#` from leg0_all;
      ];
      leg_pick:([] date:`date$(); leg_price:`float$(); leg_qid:`symbol$(); leg_model:`boolean$());
      if[0<count leg0_all;
        if[have_qid;
          cand: leg0_all where (leg0_all`quote_perm_id)<>`;
          if[0<count cand;
            cand: 0!select px:first price_sel by date,quote_perm_id from cand;
            qcov: 0!select n_cov:count distinct date by quote_perm_id from cand;
            cand: cand lj `quote_perm_id xkey qcov;
            dmed: 0!select day_med:med px by date from cand;
            cand: cand lj `date xkey dmed;
            cand: update gap:abs(px-day_med) from cand;

            qid_anchor:`;
            rb_c: cand where (cand`date)=rb;
            if[0<count rb_c;
              px_ref:0n;
              if[strat=`straddle;
                other_pc:$[pc=`C; `P; `C];
                mref:(tt`date)=rb;
                mref:mref & (tt`expiry)=exp_date;
                if[ric<>`; mref:mref & ((tt`underlying_ric)=ric)];
                mref:mref & ((tt`put_call)=other_pc);
                mref:mref & .oca.strike_eq_mask[tt`strike; k; tol];
                mref:0<>mref;
                oth:tt where mref;
                if[0<count oth; px_ref:med oth`price_sel];
              ];
              rb_c: ord_cand[rb_c; px_ref];
              qid_anchor: first rb_c`quote_perm_id;
            ];

            if[qid_anchor<>`;
              leg_a: cand where (cand`quote_perm_id)=qid_anchor;
              if[0<count leg_a;
                / Hard-anchor only when the quote id spans multiple days.
                if[(count distinct leg_a`date)>1;
                  leg_pick: 0!select leg_price:first px, leg_qid:first quote_perm_id, leg_model:first 0b by date from leg_a;
                ];
              ];
            ];

            miss_dates: seg_dates except $[0<count leg_pick; asc distinct leg_pick`date; `date$()];
            if[0<count miss_dates;
              rest: cand where (cand`date) in miss_dates;
              if[0<count rest;
                pref_rest:0n;
                if[0<count leg_pick;
                  rbp: leg_pick where (leg_pick`date)=rb;
                  if[0<count rbp; pref_rest:first rbp`leg_price];
                ];
                rest: ord_cand[rest; pref_rest];
                rest: 0!select leg_price:first px, leg_qid:first quote_perm_id, leg_model:first 0b by date from rest;
                leg_pick: $[0<count leg_pick; leg_pick,rest; rest];
              ];
            ];
          ];
        ];
        if[0=count leg_pick;
          / No usable quote-id anchoring: choose an actual row nearest each day's center.
          base: update row_i:til count leg0_all from leg0_all;
          dmed0: 0!select day_med:med price_sel by date from base;
          base: base lj `date xkey dmed0;
          base: update gap:abs(price_sel-day_med) from base;
          base: base @ iasc base`row_i;
          base: base @ iasc base`gap;
          leg_pick: 0!select leg_price:first price_sel, leg_qid:first quote_perm_id, leg_model:first 0b by date from base;
        ];
      ];
      leg1:([] date:seg_dates; leg_price:(count seg_dates)#0n; leg_qid:(count seg_dates)#`; leg_model:(count seg_dates)#0b);
      scaffold:([] date:seg_dates);
      if[0<count leg_pick;
        leg1: scaffold lj `date xkey leg_pick;
        if[not `leg_model in cols leg1; leg1:update leg_model:(count leg1)#0b from leg1];
      ];
      mkt_px:(count seg_dates)#0n;
      if[not rq;
        / Keep nearest-ATM leg set usable by interpolating missing market marks across strikes.
        mkt_px:.oca.fallback_leg_price'[ (count seg_dates)#enlist tt; seg_dates; (count seg_dates)#enlist exp_date; (count seg_dates)#enlist ric; (count seg_dates)#enlist pc; (count seg_dates)#enlist k; (count seg_dates)#enlist `price_sel];
        mkt_miss: where (null leg1`leg_price) & (not null mkt_px);
        if[0<count mkt_miss;
          leg1:update
            leg_price:@[leg_price; mkt_miss; :; mkt_px mkt_miss],
            leg_qid:@[leg_qid; mkt_miss; :; (count mkt_miss)#`interp],
            leg_model:@[leg_model; mkt_miss; :; (count mkt_miss)#1b]
            from leg1;
        ];
      ];
      if[fb_mode;
        mdl_px:.oca.fallback_leg_price'[ (count seg_dates)#enlist tt; seg_dates; (count seg_dates)#enlist exp_date; (count seg_dates)#enlist ric; (count seg_dates)#enlist pc; (count seg_dates)#enlist k; (count seg_dates)#enlist fb_col];
        ub_px:.oca.fallback_leg_cap'[ (count seg_dates)#enlist tt; seg_dates; (count seg_dates)#enlist exp_date; (count seg_dates)#enlist ric; (count seg_dates)#enlist pc; (count seg_dates)#enlist k];
        md0:`float$mdl_px;
        neg_idx:where (not null md0) & (md0<0f);
        if[0<count neg_idx;
          repl:mkt_px neg_idx;
          replf:`float$repl;
          missr:where null replf;
          if[0<count missr; repl:@[repl; missr; :; (count missr)#0f]];
          mdl_px:@[mdl_px; neg_idx; :; repl];
        ];
        md0:`float$mdl_px;
        ub_idx:where (not null md0) & (not null ub_px) & (md0>ub_px);
        if[0<count ub_idx;
          repl:mkt_px ub_idx;
          replf:`float$repl;
          ubv:ub_px ub_idx;
          badr:where (null replf) | (replf>ubv);
          if[0<count badr; repl:@[repl; badr; :; ubv badr]];
          mdl_px:@[mdl_px; ub_idx; :; repl];
        ];
        if[fb_weird;
          md0:`float$mdl_px;
          mr0:`float$mkt_px;
          rel_ok:(not null md0) & (not null mr0) & (abs mr0)>0.000000001f;
          bad_rel: where rel_ok & ((md0 > fb_hi*mr0) | (md0 < fb_lo*mr0));
          if[0<count bad_rel; mdl_px:@[mdl_px; bad_rel; :; mr0 bad_rel]];
        ];
        miss_idx: where (null leg1`leg_price) & (not null mdl_px);
        if[0<count miss_idx;
          leg1:update
            leg_price:@[leg_price; miss_idx; :; mdl_px miss_idx],
            leg_qid:@[leg_qid; miss_idx; :; (count miss_idx)#`model],
            leg_model:@[leg_model; miss_idx; :; (count miss_idx)#1b]
            from leg1;
        ];
        if[fb_weird;
          px0:`float$leg1`leg_price;
          md0:`float$mdl_px;
          rel_ok:(not null px0) & (not null md0) & (abs md0)>0.000000001f;
          bad_idx: where rel_ok & ((px0 > fb_hi*md0) | (px0 < fb_lo*md0) | (px0 < 0f));
          if[0<count bad_idx;
            leg1:update
              leg_price:@[leg_price; bad_idx; :; mdl_px bad_idx],
              leg_qid:@[leg_qid; bad_idx; :; (count bad_idx)#`model],
              leg_model:@[leg_model; bad_idx; :; (count bad_idx)#1b]
              from leg1;
          ];
        ];
      ];
      leg1: update leg_price:fills leg_price, leg_qid:fills leg_qid from leg1;
      leg1: update put_call:pc, leg_qty:q, leg_strike:k from leg1;
      leg1: update contrib:q*leg_price from leg1;
      leg_tbls,: enlist leg1;
      i+:1;
    ];
    if[0=count leg_tbls; :()];
    leg_all: raze leg_tbls;
    px: 0!select price:sum contrib by date from leg_all;
    mx: 0!select n_model_legs:sum leg_model by date from leg_all;
    px: px lj `date xkey mx;
    if[0=count px; px:([] date:seg_dates; price:(count seg_dates)#0n)];
    if[strat=`straddle;
      cp:0!select call_leg_price:first leg_price by date from leg_all where (leg_all`put_call)=`C;
      pp:0!select put_leg_price:first leg_price by date from leg_all where (leg_all`put_call)=`P;
      cq:0!select call_leg_qid:first leg_qid by date from leg_all where (leg_all`put_call)=`C;
      pq:0!select put_leg_qid:first leg_qid by date from leg_all where (leg_all`put_call)=`P;
      px: px lj `date xkey cp;
      px: px lj `date xkey pp;
      px: px lj `date xkey cq;
      px: px lj `date xkey pq;
      px: update straddle_leg_sum:call_leg_price + put_leg_price from px;
    ];
    if[not `call_leg_price in cols px; px:update call_leg_price:(count px)#0n from px];
    if[not `put_leg_price in cols px; px:update put_leg_price:(count px)#0n from px];
    if[not `call_leg_qid in cols px; px:update call_leg_qid:(count px)#` from px];
    if[not `put_leg_qid in cols px; px:update put_leg_qid:(count px)#` from px];
    if[not `n_model_legs in cols px; px:update n_model_legs:(count px)#0 from px];
    if[not `straddle_leg_sum in cols px; px:update straddle_leg_sum:(count px)#0n from px];
    px: px @ iasc px`date;
    scaffold:([] date:seg_dates);
    px: scaffold lj `date xkey px;
    px: update price:fills price from px;
    px: update reb_date:rb, expiry:exp_date, strike:strike, underlying_ric:ric, strategy:strat, legs:leg_desc, price_mode:pm, side:s from px;
    px: update pnl: s * (price - prev price) from px;
    px: update ret: pnl % abs prev price from px;
    px: update pnl:0f^pnl, ret:0f^ret from px;
    px
  };

  segs: seg_fn'[seg_tbl; (count seg_tbl)#enlist env];
  segs: segs where 0 < count each segs;
  if[0=count segs; '"no pricing rows for selected ATM strategy"];
  out: raze segs;
  if[cont_mode;
    overlap: 1 _ reb;
    keep: (not ((out`date) in overlap)) or ((out`reb_date) < out`date);
    out: out where keep;
  ];
  out: out @ iasc out`reb_date;
  out: out @ iasc out`date;
  if[(count out) > count distinct out`date;
    ix:exec first i by date from update i:til count out from out;
    out: out ix;
    out: out @ iasc out`date;
  ];
  if[cont_mode;
    / Recompute from final published series so prev price always matches emitted prior row.
    out: out @ iasc out`date;
    out: update pnl: s * (price - prev price) from out;
    out: update ret: pnl % abs prev price from out;
    out: update pnl:0f^pnl, ret:0f^ret from out;
  ];
  out
 }

atm_strategy_returns_wing:{[t; rebalance_days; target_dte; min_dte; max_dte; price_mode; strategy; cfg]
  c:$[99h=type cfg; cfg; ()!()];
  s:$[`side in key c; c`side; 1f];
  w:$[`wing_steps in key c; c`wing_steps; 1];
  m:$[`wing_mode in key c; c`wing_mode; `rank];
  z:$[`wing_step_size in key c; c`wing_step_size; (::)];
  sw:(`side`wing_steps`wing_mode`wing_step_size)!(s; w; m; z);
  args:(t; rebalance_days; target_dte; min_dte; max_dte; price_mode; strategy; sw);
  .oca.atm_strategy_returns . args
 }

atm_strategy_suite_returns:{[t; rebalance_days; target_dte; min_dte; max_dte; price_mode; side; strategies]
  strats: .oca.as_strategy_list strategies;
  swcfg:.oca.side_wing_cfg side;
  s:swcfg`side;
  ws:swcfg`wing_steps;
  wm:swcfg`wing_mode;
  wss:swcfg`wing_step_size;
  sw:(`side`wing_steps`wing_mode`wing_step_size)!(s; ws; wm; wss);
  if[0=count strats; '"strategy list is empty"];
  outs:();
  i:0;
  while[i<count strats;
    st:(strats i);
    out1:.[.oca.atm_strategy_returns; (t; rebalance_days; target_dte; min_dte; max_dte; price_mode; st; sw); {::}];
    if[98h=type out1; outs,: enlist out1];
    i+:1;
  ];
  if[0=count outs; '"no pricing rows for selected ATM strategy suite"];
  raze outs
 }

atm_strategy_suite_returns_wing:{[t; rebalance_days; target_dte; min_dte; max_dte; price_mode; strategies; cfg]
  c:$[99h=type cfg; cfg; ()!()];
  s:$[`side in key c; c`side; 1f];
  w:$[`wing_steps in key c; c`wing_steps; 1];
  m:$[`wing_mode in key c; c`wing_mode; `rank];
  z:$[`wing_step_size in key c; c`wing_step_size; (::)];
  sw:(`side`wing_steps`wing_mode`wing_step_size)!(s; w; m; z);
  args:(t; rebalance_days; target_dte; min_dte; max_dte; price_mode; sw; strategies);
  .oca.atm_strategy_suite_returns . args
 }

atm_strategy_pnl_diagnostics:{[rets; top_n]
  t:rets;
  if[99h=type t; t:0!t];
  if[98h<>type t; '"returns input must be a table"];
  req:`date`reb_date`price`pnl;
  if[not all req in cols t; '"returns table missing required columns (`date`reb_date`price`pnl)"];
  if[0=count t;
    e:([]);
    :(`summary`by_trade`daily`top_days`worst_days`top_trades`worst_trades)!(e;e;e;e;e;e;e);
  ];

  dty: abs type t`date;
  if[dty in 12 15h; t:update date:date date from t];
  if[not dty in 14 12 15h; '"date column must be date or timestamp/datetime"];
  rty: abs type t`reb_date;
  if[rty in 12 15h; t:update reb_date:date reb_date from t];
  if[not rty in 14 12 15h; '"reb_date column must be date or timestamp/datetime"];

  tn: $[top_n~(::); 10; top_n];
  tn: max 1, `int$tn;

  if[not `strategy in cols t; t:update strategy:`all from t];
  if[not `side in cols t; t:update side:1f from t];
  if[not `price_mode in cols t; t:update price_mode:`unknown from t];
  if[not `expiry in cols t; t:update expiry:0Nd from t];
  if[not `strike in cols t; t:update strike:0n from t];

  trade:0!select
    entry_date:first date,
    exit_date:last date,
    hold_days:count i,
    entry_price:first price,
    exit_price:last price,
    segment_pnl:sum pnl,
    segment_abs_pnl:sum abs pnl,
    max_day_pnl:max pnl,
    min_day_pnl:min pnl
    by reb_date,strategy,side,price_mode,expiry,strike from t;
  trade:update segment_ret: segment_pnl % abs entry_price from trade;
  trade:update segment_pnl:0f^segment_pnl, segment_ret:0f^segment_ret from trade;

  svals: distinct trade`strategy;
  out_trade:();
  i:0;
  while[i<count svals;
    s1:(svals i);
    sub: trade where (trade`strategy)=s1;
    sub: sub @ iasc sub`reb_date;
    sub: update cum_pnl:sums segment_pnl from sub;
    sub: update cum_peak:maxs cum_pnl from sub;
    sub: update drawdown:cum_pnl - cum_peak from sub;
    out_trade,:enlist sub;
    i+:1;
  ];
  trade: raze out_trade;

  daily:$[
    `ret in cols t;
    0!select day_pnl:sum pnl, day_abs_ret:sum abs ret by date,strategy from t;
    0!select day_pnl:sum pnl by date,strategy from t
  ];
  if[not `day_pnl in cols daily; '"internal error: daily summary missing day_pnl"];
  daily: daily @ reverse iasc daily`day_pnl;
  top_days: tn#daily;
  worst_days: tn#(daily @ iasc daily`day_pnl);

  top_trades: tn#(trade @ reverse iasc trade`segment_pnl);
  worst_trades: tn#(trade @ iasc trade`segment_pnl);

  summary:0!select
    start_date:min date,
    end_date:max date,
    n_days:count distinct date,
    rebalance_count:count distinct reb_date,
    total_pnl:sum pnl,
    avg_day_pnl:avg pnl,
    pnl_stdev:dev pnl,
    win_rate:avg pnl>0f,
    best_day:max pnl,
    worst_day:min pnl
    by strategy from t;
  nz:t where (abs t`pnl)>1e-12f;
  nzStats:0!select win_rate_nz:avg pnl>0f by strategy from nz;
  summary:summary lj `strategy xkey nzStats;
  summary:update win_rate:win_rate_nz from summary;
  summary:delete win_rate_nz from summary;

  (`summary`by_trade`daily`top_days`worst_days`top_trades`worst_trades)!(summary; trade; daily; top_days; worst_days; top_trades; worst_trades)
 }

atm_strategy_trade_table:{[rets]
  d: .oca.atm_strategy_pnl_diagnostics[rets; 10];
  d`by_trade
 }

is_numeric_col:{[t;c]
  if[not c in cols t; :0b];
  (abs type t c) in 1 4 5 6 7 8 9h
 }

safe_corr:{[x;y]
  n:count x;
  if[(n<2) or ((count y)<2); :0n];
  dx:dev x;
  dy:dev y;
  if[(null dx) or (null dy) or (dx<=1e-12) or (dy<=1e-12); :0n];
  cor[x;y]
 }

safe_beta:{[x;y]
  n:count x;
  if[(n<2) or ((count y)<2); :0n];
  vx:var x;
  if[(null vx) or (vx<=1e-12); :0n];
  cov[x;y] % vx
 }

alpha_subtype_guess:{[s]
  nm:lower string s;
  if[(nm like "*mom*") or (nm like "*trend*") or (nm like "*break*") or (nm like "*donch*") or (nm like "*time*"); :`momentum];
  if[(nm like "*carry*") or (nm like "*roll*") or (nm like "*basis*") or (nm like "*curve*"); :`carry];
  if[(nm like "*mean*") or (nm like "*revert*") or (nm like "*mr*") or (nm like "*zscore*"); :`mean_reversion];
  if[(nm like "*vol*") or (nm like "*skew*") or (nm like "*gamma*") or (nm like "*straddle*") or (nm like "*fly*") or (nm like "*condor*"); :`volatility];
  if[(nm like "*event*") or (nm like "*news*") or (nm like "*flow*") or (nm like "*impulse*"); :`event];
  if[(nm like "*macro*") or (nm like "*fund*") or (nm like "*econ*") or (nm like "*cpi*") or (nm like "*nfp*"); :`fundamental];
  `other
 }

alpha_assign_group_subtype:{[attrib; corrHi; concHi; minSubtype]
  out:attrib;
  grp:`symbol$();
  stype:`symbol$();
  stat:`symbol$();
  i:0;
  while[i<count out;
    p:(out`sum_pnl) i;
    cr:(out`corr_total) i;
    tc:(out`topn_pnl_conc) i;
    wr:(out`win_rate) i;
    crPos:(not null cr) and (cr>=corrHi);
    crNeg:(not null cr) and (cr<=neg corrHi);
    tcHi:(not null tc) and (tc>=concHi);
    g1:$[
      (p>0f) and crPos and (not tcHi); `core_contributor;
      (p>0f) and crNeg; `diversifying_contributor;
      (p>0f) and tcHi; `event_contributor;
      (p<=0f) and crNeg; `hedge_cost;
      (p<=0f) and crPos; `drag;
      `mixed
    ];
    s0:.oca.alpha_subtype_guess (out`strategy) i;
    if[s0=`other;
      s0:$[
        tcHi; `event;
        crPos and (not null wr) and (wr>=0.55); `trend_like;
        crNeg; `diversifier;
        (p>0f) and (not null wr) and (wr<0.5); `convex;
        (p>0f); `carry_like;
        `other
      ];
    ];
    st1:$[
      p>0f; `working;
      g1=`hedge_cost; `insurance_cost;
      `not_working
    ];
    grp,:enlist g1;
    stype,:enlist s0;
    stat,:enlist st1;
    i+:1;
  ];
  out:update alpha_group:grp, alpha_subtype:stype, status:stat from out;
  g:group out`alpha_subtype;
  ks:key g;
  ns:count each value g;
  ms:max 1, `int$minSubtype;
  small:ks where ns<ms;
  if[(count small)>0;
    m:(out`alpha_subtype) in small;
    if[(sum m)>0;
      out:update alpha_subtype:@[alpha_subtype; where m; :; (sum m)#`other] from out;
    ];
  ];
  out
 }

alpha_ensure_min_subtypes:{[attrib; minDistinct]
  out:attrib;
  tgt:max 0, `int$minDistinct;
  if[tgt<=0; :out];
  n:count out;
  if[n<=1; :out];
  tgt:min tgt,n;
  dcnt:count distinct out`alpha_subtype;
  while[dcnt<tgt;
    g:group out`alpha_subtype;
    ks:key g;
    vix:value g;
    ns:count each vix;
    spl:where ns>1;
    if[0=count spl; :out];
    candN:ns spl;
    ord:idesc candN;
    best:spl first ord;
    base:ks best;
    idxs:vix best;
    cs:`float$(out`corr_total) idxs;
    hasC:sum not null cs;
    ordLocal:$[hasC>1; idxs @ iasc cs; idxs @ iasc (`float$(out`sum_pnl) idxs)];
    mcnt:count ordLocal;
    if[mcnt<2; :out];
    splitN:mcnt div 2;
    iA:splitN#ordLocal;
    iB:(mcnt-splitN)#splitN _ ordLocal;
    if[(count iA)=0 or (count iB)=0; :out];
    sA:`$raze (string base;"_a");
    sB:`$raze (string base;"_b");
    out:update alpha_subtype:@[alpha_subtype; iA; :; (count iA)#sA] from out;
    out:update alpha_subtype:@[alpha_subtype; iB; :; (count iB)#sB] from out;
    dcnt:count distinct out`alpha_subtype;
  ];
  out
 }

alpha_monthly_status:{[alphaLong; attrib]
  am:update month:`month$date from alphaLong;
  m:0!select n_days:count i, month_pnl:sum pnl, avg_day_pnl:avg pnl, pnl_stdev:dev pnl, win_rate:avg pnl>0f by strategy,month from am;
  amAct:am where (abs am`pnl)>1e-12f;
  mAct:0!select active_n_days:count i, active_win_rate:avg pnl>0f by strategy,month from amAct;
  m:m lj `strategy`month xkey mAct;
  m:update win_rate:active_win_rate from m;
  mNZ:m where (abs m`month_pnl)>1e-12f;
  mstats:0!select monthly_avg_pnl:avg month_pnl, monthly_pnl_stdev:dev month_pnl by strategy from mNZ;
  mstats:update monthly_sharpe:(sqrt 12f) * monthly_avg_pnl % (1e-12f + 0f^monthly_pnl_stdev) from mstats;
  mstats:update fragility_ratio:(abs monthly_avg_pnl) % (1e-12f + 0f^monthly_pnl_stdev) from mstats;
  mstats:update fragile_edge:fragility_ratio<0.25f from mstats;
  m:m lj `strategy xkey mstats;
  m:m lj `strategy xkey (select strategy,alpha_group,alpha_subtype,status from attrib);
  ms:(count m)#`flat;
  ip:where (m`month_pnl)>0f;
  ineg:where (m`month_pnl)<0f;
  if[(count ip)>0; ms:@[ms; ip; :; (count ip)#`working]];
  if[(count ineg)>0; ms:@[ms; ineg; :; (count ineg)#`not_working]];
  m:update month_status:ms from m;
  m:(`month`strategy) xasc m;
  (`alpha_monthly`working_monthly`not_working_monthly)!(
    m;
    m where (m`month_status)=`working;
    m where (m`month_status)=`not_working
  )
 }

subtype_behavior_enrich:{[st]
  if[0=count st; :st];
  lbl:`symbol$();
  txt:`symbol$();
  i:0;
  while[i<count st;
    p:(st`subtype_pnl) i;
    wr:(st`avg_win_rate) i;
    tc:(st`avg_topn_conc) i;
    cr:(st`avg_corr_total) i;
    fs:(st`fragile_share) i;
    b:$[
      (p>=0f) and (tc>=0.6) and (wr<0.55); `event_driven_convex;
      (p<0f) and (tc>=0.6); `event_driven_drag;
      (p>0f) and (wr>=0.55) and (tc<0.5) and (fs<0.6); `steady_carry_trend;
      (p<0f) and (cr<0f); `diversifying_cost;
      (p<0f) and (cr>=0f); `structural_drag;
      fs>=0.7; `fragile_mixed;
      `mixed
    ];
    t:$[
      b=`event_driven_convex; "Event-driven: gains come from a small number of outsized upside days; hit-rate can be low while convexity is positive.";
      b=`event_driven_drag; "Event-driven drag: losses are concentrated in a small number of outsized downside days.";
      b=`steady_carry_trend; "Steady edge: more frequent smaller wins, lower tail concentration, and more stable day-to-day drift.";
      b=`diversifying_cost; "Diversifying cost: standalone pnl is negative but correlation to portfolio is negative, so it can hedge stress.";
      b=`structural_drag; "Structural drag: negative pnl with positive/neutral correlation, so it tends to lose with the book.";
      b=`fragile_mixed; "Fragile mixed profile: average edge is small versus realized volatility, so outcomes are noisy and unstable.";
      "Mixed profile: no single regime dominates behavior."
    ];
    lbl,:enlist b;
    txt,:enlist `$t;
    i+:1;
  ];
  update behavior_label:lbl, behavior_text:txt from st
 }

edge_tail_recheck:{[daily_path; svals; trimN]
  rows:();
  i:0;
  while[i<count svals;
    s1:svals i;
    sub:daily_path where (daily_path`strategy)=s1;
    sub:sub @ iasc sub`date;
    n:count sub;
    k:min trimN, max 0, (n-1) div 2;
    idx:til n;
    ordUp:reverse iasc sub`day_pnl;
    ordDn:iasc sub`day_pnl;
    dropIdx:distinct (k#ordUp),(k#ordDn);
    keepIdx:idx where not idx in dropIdx;
    kept:sub keepIdx;
    nk:count kept;
    ttot:sum kept`day_pnl;
    tavg:$[nk=0; 0n; avg kept`day_pnl];
    tstd:$[nk<2; 0n; dev kept`day_pnl];
    tsh:(sqrt 252f) * tavg % (1e-12f + 0f^tstd);
    tfrag:(abs tavg) % (1e-12f + 0f^tstd);
    tfragF:tfrag<0.25f;
    twr:$[nk=0; 0n; avg (kept`day_pnl)>0f];
    oavg:$[n=0; 0n; avg sub`day_pnl];
    ret:$[(null oavg) or ((abs oavg)<=1e-12f); 0n; (abs tavg) % (abs oavg)];
    dec:$[null ret; 0n; 1f-ret];
    dec:1f & (0f | dec);
    rows,:enlist ([] strategy:enlist s1; trim_n:enlist k; n_days:enlist n; trimmed_n_days:enlist nk; dropped_days:enlist n-nk; dropped_frac:enlist $[n=0; 0n; 1f*(n-nk)%n]; trimmed_total_pnl:enlist ttot; trimmed_avg_day_pnl:enlist tavg; trimmed_pnl_stdev:enlist tstd; trimmed_sharpe:enlist tsh; trimmed_fragility_ratio:enlist tfrag; trimmed_fragile_edge:enlist tfragF; trimmed_win_rate:enlist twr; edge_retention:enlist ret; edge_decay_pct:enlist dec);
    i+:1;
  ];
  $[
    0=count rows;
    ([] strategy:`symbol$(); trim_n:`int$(); n_days:`int$(); trimmed_n_days:`int$(); dropped_days:`int$(); dropped_frac:`float$(); trimmed_total_pnl:`float$(); trimmed_avg_day_pnl:`float$(); trimmed_pnl_stdev:`float$(); trimmed_sharpe:`float$(); trimmed_fragility_ratio:`float$(); trimmed_fragile_edge:`boolean$(); trimmed_win_rate:`float$(); edge_retention:`float$(); edge_decay_pct:`float$());
    raze rows
  ]
 }

strategy_driver_effects:{[daily_path; svals; dcols]
  drvRows:();
  i:0;
  while[i<count svals;
    s1:svals i;
    sub:daily_path where (daily_path`strategy)=s1;
    j:0;
    while[j<count dcols;
      c1:dcols j;
      x:`float$(sub c1);
      y:`float$sub`day_pnl;
      ok:(not null x) & not null y;
      x:x where ok;
      y:y where ok;
      n:count x;
      md:$[n=0; 0n; med x];
      hi:x>=md;
      lo:x<md;
      hm:$[(sum hi)=0; 0n; avg y where hi];
      lm:$[(sum lo)=0; 0n; avg y where lo];
      hwr:$[(sum hi)=0; 0n; avg (y where hi)>0f];
      lwr:$[(sum lo)=0; 0n; avg (y where lo)>0f];
      cr:.oca.safe_corr[x;y];
      bt:.oca.safe_beta[x;y];
      sp:$[(null hm) or (null lm); 0n; hm-lm];
      drvRows,:enlist ([] strategy:enlist s1; driver:enlist c1; n_obs:enlist n; corr:enlist cr; beta:enlist bt; median_driver:enlist md; high_mean_pnl:enlist hm; low_mean_pnl:enlist lm; pnl_spread:enlist sp; high_win_rate:enlist hwr; low_win_rate:enlist lwr);
      j+:1;
    ];
    i+:1;
  ];
  $[
    0=count drvRows;
    ([] strategy:`symbol$(); driver:`symbol$(); n_obs:`int$(); corr:`float$(); beta:`float$(); median_driver:`float$(); high_mean_pnl:`float$(); low_mean_pnl:`float$(); pnl_spread:`float$(); high_win_rate:`float$(); low_win_rate:`float$());
    raze drvRows
  ]
 }

strategy_performance_narrative:{[svals; flags; drv; tn]
  narrReason:();
  narrDiag:`symbol$();
  narrPosDrv:`symbol$();
  narrPosSp:`float$();
  narrNegDrv:`symbol$();
  narrNegSp:`float$();
  i:0;
  while[i<count svals;
    s1:svals i;
    f:flags where (flags`strategy)=s1;
    d1:$[0=count f; `flat; first f`diagnosis];
    tp:$[0=count f; 0n; first f`total_pnl];
    wr:$[0=count f; 0n; first f`win_rate];
    tc:$[0=count f; 0b; first f`tail_concentrated];
    ct:$[0=count f; 0n; first f`topn_pnl_conc];

    dsub:drv where (drv`strategy)=s1;
    posDrv:`;
    negDrv:`;
    posSp:0n;
    negSp:0n;
    if[(count dsub)>0;
      dpos:dsub @ reverse iasc dsub`pnl_spread;
      dneg:dsub @ iasc dsub`pnl_spread;
      posDrv:first dpos`driver;
      negDrv:first dneg`driver;
      posSp:first dpos`pnl_spread;
      negSp:first dneg`pnl_spread;
    ];

    base:$[
      d1=`positive_right_tail; "PnL is positive with sub-50% win rate; large right-tail days dominate.";
      d1=`negative_left_tail; "Win rate is above 50% but total PnL is negative; left-tail losses dominate.";
      d1=`event_driven; "A large share of total PnL comes from a small number of days.";
      d1=`broad_positive; "Performance is broadly positive across days.";
      d1=`broad_negative; "Performance is broadly negative across days.";
      "Performance is close to flat."
    ];

    concTxt:$[tc; raze (" Concentration is high (top-",string tn," days / total abs PnL = ",string ct,")."); ""];
    posTxt:$[posDrv~`; ""; raze (" Positive regime driver: ",string posDrv," (high-low avg day pnl spread ",string posSp,").")];
    negTxt:$[negDrv~`; ""; raze (" Negative regime driver: ",string negDrv," (high-low avg day pnl spread ",string negSp,").")];
    lvlTxt:raze (" Total pnl ",string tp,", win rate ",string wr,".");
    msg:raze (base; lvlTxt; concTxt; posTxt; negTxt);

    narrDiag,:enlist d1;
    narrReason,:enlist msg;
    narrPosDrv,:enlist posDrv;
    narrPosSp,:enlist posSp;
    narrNegDrv,:enlist negDrv;
    narrNegSp,:enlist negSp;
    i+:1;
  ];
  reasonSym:{[x] `$raze x} each narrReason;
  flip `strategy`diagnosis`reason`top_driver_pos`top_driver_pos_spread`top_driver_neg`top_driver_neg_spread!(
    svals;
    narrDiag;
    reasonSym;
    narrPosDrv;
    narrPosSp;
    narrNegDrv;
    narrNegSp
  )
 }

/ Explain good/bad performance drivers for a returns stream.
/ Inputs:
/   rets: table containing at least `date`pnl (optionally `strategy)
/   cfg keys (all optional):
/     `top_n (default 10) number of top/worst days used in concentration metrics
/     `event_trim_n (default top_n) symmetric tail days removed on each side for edge re-check
/     `driver_cols (symbol or symbol list) numeric columns to attribute performance to
/     `drivers_tbl (table keyed by `date with extra daily drivers to join onto `rets)
/ Returns dict:
/   `summary`daily`monthly`top_days`worst_days`edge_recheck`driver_effects`flags`narrative
atm_strategy_performance_explain:{[rets; cfg]
  t:.oca.to_table rets;
  if[98h<>type t; '"returns input must be a table"];
  c:.oca.cfg_to_dict cfg;
  req:`date`pnl;
  if[not all req in cols t; '"returns table missing required columns (`date`pnl)"];

  if[0=count t;
    e:([]);
    :(`summary`daily`monthly`top_days`worst_days`edge_recheck`driver_effects`flags`narrative)!(e;e;e;e;e;e;e;e;e);
  ];

  dty:abs type t`date;
  if[dty in 12 15h; t:update date:date date from t];
  if[not dty in 14 12 15h; '"date column must be date or timestamp/datetime"];
  if[not `strategy in cols t; t:update strategy:`all from t];

  if[`drivers_tbl in key c;
    dtb:.oca.to_table c`drivers_tbl;
    if[98h<>type dtb; '"drivers_tbl must be a table"];
    if[0<count dtb;
      if[not `date in cols dtb; '"drivers_tbl must include `date"];
      dty2:abs type dtb`date;
      if[dty2 in 12 15h; dtb:update date:date date from dtb];
      if[not dty2 in 14 12 15h; '"drivers_tbl `date must be date or timestamp/datetime"];
      if[(count distinct dtb`date)<>count dtb; '"drivers_tbl `date must be unique"];
      t:t lj `date xkey dtb;
    ];
  ];

  tn:$[`top_n in key c; c`top_n; 10];
  tn:max 1, `int$tn;
  trimN:$[`event_trim_n in key c; c`event_trim_n; tn];
  trimN:max 0, `int$trimN;

  drv_raw:$[`driver_cols in key c; c`driver_cols; `symbol$()];
  dty_drv:type drv_raw;
  dcols:$[
    drv_raw~(::); `symbol$();
    dty_drv=-11h; enlist drv_raw;
    dty_drv=10h; enlist `$drv_raw;
    dty_drv=11h; drv_raw;
    dty_drv=0h; .oca.to_sym each drv_raw;
    enlist .oca.to_sym drv_raw];
  dcols:dcols where dcols in cols t;

  if[(0=count dcols) and (`drivers_tbl in key c);
    cs:cols t;
    i:0;
    auto:`symbol$();
    while[i<count cs;
      c1:cs i;
      if[(not c1 in `date`strategy`pnl`price`ret`reb_date) and .oca.is_numeric_col[t;c1];
        auto,:enlist c1];
      i+:1;
    ];
    dcols:auto;
  ];

  / Daily aggregated pnl by strategy/date
  daily:0!select day_pnl:sum pnl by date,strategy from t;

  / Attach daily driver averages if requested
  i:0;
  gk:group flip `date`strategy!(t`date; t`strategy);
  gkTab:key gk;
  gkIdx:value gk;
  while[i<count dcols;
    c1:dcols i;
    vals:`float$avg each (t c1) gkIdx;
    dv:flip (`date`strategy,c1)!(gkTab`date; gkTab`strategy; vals);
    daily:daily lj `date`strategy xkey dv;
    i+:1;
  ];

  svals:asc distinct daily`strategy;
  paths:();
  i:0;
  while[i<count svals;
    s1:svals i;
    sub:daily where (daily`strategy)=s1;
    sub:sub @ iasc sub`date;
    sub:update cum_pnl:sums day_pnl from sub;
    sub:update cum_peak:maxs cum_pnl from sub;
    sub:update drawdown:cum_pnl - cum_peak from sub;
    paths,:enlist sub;
    i+:1;
  ];
  daily_path:raze paths;

  summary:0!select
    start_date:min date,
    end_date:max date,
    n_days:count i,
    total_pnl:sum day_pnl,
    avg_day_pnl:avg day_pnl,
    pnl_stdev:dev day_pnl,
    win_rate:avg day_pnl>0f,
    avg_win:avg day_pnl where day_pnl>0f,
    avg_loss:avg day_pnl where day_pnl<0f,
    best_day:max day_pnl,
    worst_day:min day_pnl
    by strategy from daily_path;
  / Active-day stats: sharpe/fragility should be computed on non-zero pnl days only.
  ad:daily_path where (abs daily_path`day_pnl)>1e-12f;
  aStats:0!select active_avg_day_pnl:avg day_pnl, active_pnl_stdev:dev day_pnl, active_win_rate:avg day_pnl>0f by strategy from ad;
  summary:summary lj `strategy xkey aStats;
  summary:update win_rate:active_win_rate, sharpe:(sqrt 252f) * active_avg_day_pnl % (1e-12f + 0f^active_pnl_stdev), fragility_ratio:(abs active_avg_day_pnl) % (1e-12f + 0f^active_pnl_stdev), fragile_edge:((abs active_avg_day_pnl) % (1e-12f + 0f^active_pnl_stdev))<0.25f from summary;

  dd:0!select max_drawdown:min drawdown, max_dd_date:first date where drawdown=min drawdown by strategy from daily_path;
  summary:summary lj `strategy xkey dd;

  concRows:();
  i:0;
  while[i<count svals;
    s1:svals i;
    sub:daily_path where (daily_path`strategy)=s1;
    subD:sub @ reverse iasc sub`day_pnl;
    subW:sub @ iasc sub`day_pnl;
    tp:sum tn#subD`day_pnl;
    wl:sum abs tn#subW`day_pnl;
    totAbs:abs sum sub`day_pnl;
    cTop:$[totAbs<=1e-12; 0n; tp % totAbs];
    cW:$[totAbs<=1e-12; 0n; wl % totAbs];
    concRows,:enlist ([] strategy:enlist s1; topn_pnl_conc:enlist cTop; worstn_abs_pnl_conc:enlist cW);
    i+:1;
  ];
  conc:$[0=count concRows; ([] strategy:`symbol$(); topn_pnl_conc:`float$(); worstn_abs_pnl_conc:`float$()); raze concRows];
  summary:summary lj `strategy xkey conc;
  summary:update tail_balance_score:(0f^topn_pnl_conc) - (0f^worstn_abs_pnl_conc) from summary;

  edgeRe:.oca.edge_tail_recheck[daily_path; svals; trimN];
  summary:summary lj `strategy xkey edgeRe;
  concScore:1f & abs 0f^summary`topn_pnl_conc;
  fragBase:0f^summary`fragility_ratio;
  fragScore:1f & (0f | (0.25f-fragBase) % 0.25f);
  trimScore:1f & (0f | 0f^summary`edge_decay_pct);
  ofr:100f * (0.35f*concScore + 0.30f*fragScore + 0.35f*trimScore);
  ofb:{[x] $[x>=70f; `high; x>=40f; `medium; `low]} each ofr;
  summary:update overfit_risk_score:ofr, overfit_risk_bucket:ofb from summary;
  edgeRe:edgeRe lj `strategy xkey (select strategy,overfit_risk_score,overfit_risk_bucket from summary);

  dm:update month:`month$date from daily_path;
  monthly:0!select n_days:count i, month_pnl:sum day_pnl, avg_day_pnl:avg day_pnl, pnl_stdev:dev day_pnl, win_rate:avg day_pnl>0f by strategy,month from dm;
  dmAct:dm where (abs dm`day_pnl)>1e-12f;
  mAct:0!select active_win_rate:avg day_pnl>0f by strategy,month from dmAct;
  monthly:monthly lj `strategy`month xkey mAct;
  monthly:update win_rate:active_win_rate from monthly;
  / Monthly fragility is computed from the strategy monthly return series (not within-month daily dispersion).
  monthlyNZ:monthly where (abs monthly`month_pnl)>1e-12f;
  mStats:0!select monthly_n_months:count i, monthly_avg_pnl:avg month_pnl, monthly_pnl_stdev:dev month_pnl by strategy from monthlyNZ;
  mStats:update monthly_sharpe:(sqrt 12f) * monthly_avg_pnl % (1e-12f + 0f^monthly_pnl_stdev) from mStats;
  mStats:update monthly_fragility_ratio:(abs monthly_avg_pnl) % (1e-12f + 0f^monthly_pnl_stdev) from mStats;
  mStats:update monthly_fragile_edge:monthly_fragility_ratio<0.25f from mStats;
  monthly:monthly lj `strategy xkey mStats;
  monthly:update fragility_ratio:monthly_fragility_ratio, fragile_edge:monthly_fragile_edge from monthly;
  monthly:(`strategy`month) xasc monthly;

  top_days:tn#(daily_path @ reverse iasc daily_path`day_pnl);
  worst_days:tn#(daily_path @ iasc daily_path`day_pnl);

  drv:.oca.strategy_driver_effects[daily_path; svals; dcols];

  flags:update
    tail_concentrated:topn_pnl_conc>0.6,
    right_tail_profile:(win_rate<0.5) & (total_pnl>0f),
    left_tail_drag:(win_rate>0.5) & (total_pnl<0f),
    fragile_edge:fragility_ratio<0.25f,
    overfit_risk_high:overfit_risk_score>=70f
    from summary;
  flags:0!select
    strategy,total_pnl,win_rate,topn_pnl_conc,worstn_abs_pnl_conc,tail_balance_score,overfit_risk_score,overfit_risk_bucket,
    tail_concentrated,right_tail_profile,left_tail_drag,fragile_edge,overfit_risk_high
    from flags;

  diag:`symbol$();
  i:0;
  while[i<count flags;
    tp:(flags`total_pnl) i;
    wr:(flags`win_rate) i;
    tc:(flags`tail_concentrated) i;
    d1:$[(tp>0f) and (wr<0.5); `positive_right_tail;
        (tp<0f) and (wr>0.5); `negative_left_tail;
        tc; `event_driven;
        tp>0f; `broad_positive;
        tp<0f; `broad_negative;
        `flat];
    diag,:enlist d1;
    i+:1;
  ];
  flags:update diagnosis:diag from flags;
  narrative:.oca.strategy_performance_narrative[svals; flags; drv; tn];

  (`summary`daily`monthly`top_days`worst_days`edge_recheck`driver_effects`flags`narrative)!(summary; daily_path; monthly; top_days; worst_days; edgeRe; drv; flags; narrative)
 }

/ Portfolio-level explanation from:
/   - alpha_wide: wide table (date + one pnl column per alpha)
/   - total_tbl: total strategy pnl table (date + pnl-like numeric column)
/ It auto-detects alpha columns and total pnl column, assigns alpha groups,
/ and returns attribution + working/not-working summaries.
/ cfg optional keys:
/   `date_col (default auto: `date then `dt)
/   `total_col (default auto: `pnl else first numeric non-date col in total_tbl)
/   `alpha_cols (default auto: all numeric non-date cols in alpha_wide)
/   `top_n (default 10)
/   `group_corr_hi (default 0.2)
/   `group_conc_hi (default 0.6)
/   `min_subtype_members (default 2; subtype buckets smaller than this fold into `other)
/   `min_subtypes (default 0; enforce at least this many distinct subtype buckets when possible)
/   `min_subtype_size (deprecated alias for `min_subtypes)
/   `drivers_tbl, `driver_cols (passed through to atm_strategy_performance_explain)
alpha_portfolio_explain:{[alpha_wide; total_tbl; cfg]
  aw:.oca.to_table alpha_wide;
  tt:.oca.to_table total_tbl;
  if[98h<>type aw; '"alpha_wide must be a table"];
  if[98h<>type tt; '"total_tbl must be a table"];
  c:.oca.cfg_to_dict cfg;

  / Resolve date columns
  dcol_aw:$[
    `date_col in key c; .oca.to_sym c`date_col;
    `date in cols aw; `date;
    `dt in cols aw; `dt;
    '"alpha_wide must include `date (or `dt) or pass cfg`date_col"
  ];
  dcol_tt:$[
    `total_date_col in key c; .oca.to_sym c`total_date_col;
    `date in cols tt; `date;
    `dt in cols tt; `dt;
    dcol_aw
  ];

  if[not dcol_aw in cols aw; '"cfg`date_col not found in alpha_wide"];
  if[not dcol_tt in cols tt; '"total date column not found in total_tbl"];

  if[dcol_aw<>`date; aw:update date:aw dcol_aw from aw];
  if[dcol_tt<>`date; tt:update date:tt dcol_tt from tt];

  dty1:abs type aw`date;
  if[dty1 in 12 15h; aw:update date:date date from aw];
  if[not dty1 in 14 12 15h; '"alpha_wide date column must be date or timestamp/datetime"];
  dty2:abs type tt`date;
  if[dty2 in 12 15h; tt:update date:date date from tt];
  if[not dty2 in 14 12 15h; '"total_tbl date column must be date or timestamp/datetime"];

  / Resolve total pnl column
  tcol:$[`total_col in key c; .oca.to_sym c`total_col; `pnl];
  if[not tcol in cols tt;
    cs:cols tt;
    num:`symbol$();
    i:0;
    while[i<count cs;
      c1:cs i;
      if[(c1<>`date) and .oca.is_numeric_col[tt;c1]; num,:enlist c1];
      i+:1;
    ];
    if[0=count num; '"total_tbl has no numeric pnl column"];
    tcol:first num;
  ];
  tcol:.oca.to_sym tcol;
  if[not .oca.is_numeric_col[tt;tcol]; '"resolved total pnl column is not numeric"];

  / Resolve alpha columns
  ac_raw:$[`alpha_cols in key c; c`alpha_cols; `symbol$()];
  ac_t:type ac_raw;
  acols:$[
    ac_raw~(::); `symbol$();
    ac_t=-11h; enlist ac_raw;
    ac_t=10h; enlist `$ac_raw;
    ac_t=11h; ac_raw;
    ac_t=0h; .oca.to_sym each ac_raw;
    enlist .oca.to_sym ac_raw
  ];
  if[0=count acols;
    cs:cols aw;
    acols:`symbol$();
    i:0;
    while[i<count cs;
      c1:cs i;
      if[(c1<>`date) and .oca.is_numeric_col[aw;c1]; acols,:enlist c1];
      i+:1;
    ];
  ];
  acols:acols where acols in cols aw;
  if[0=count acols; '"no alpha columns resolved in alpha_wide"];

  / Keep only required columns
  awVals:enlist aw`date;
  i:0;
  while[i<count acols;
    c1:acols i;
    cvals:`float$(aw c1);
    awVals,:enlist cvals;
    i+:1;
  ];
  awk:flip (`date,acols)!awVals;
  if[not (tcol in cols tt); '"internal error: resolved total pnl column missing in total_tbl"];
  ttk:([] date:tt`date; portfolio_pnl:`float$(tt tcol));

  / Aggregate duplicates by date (sum)
  gaw:group awk`date;
  dta:key gaw;
  ixa:value gaw;
  aggVals:enlist dta;
  i:0;
  while[i<count acols;
    c1:acols i;
    vals:`float$sum each (0f^`float$(awk c1)) ixa;
    aggVals,:enlist vals;
    i+:1;
  ];
  awAgg:flip (`date,acols)!aggVals;
  gtt:group ttk`date;
  dtt:key gtt;
  ixt:value gtt;
  totVals:`float$sum each (0f^ttk`portfolio_pnl) ixt;
  ttAgg:([] date:dtt; portfolio_pnl:totVals);

  / Align on shared dates
  j:awAgg lj `date xkey ttAgg;
  j:j where not null j`portfolio_pnl;
  if[0=count j; '"no overlapping dates between alpha_wide and total_tbl"];

  tn:$[`top_n in key c; c`top_n; 10];
  tn:max 1, `int$tn;

  / Build long alpha table for reusable explain helper
  parts:();
  i:0;
  while[i<count acols;
    c1:acols i;
    p1:([] date:j`date; strategy:(count j)#c1; pnl:`float$(j c1));
    parts,:enlist p1;
    i+:1;
  ];
  alphaLong:raze parts;
  alphaLong:alphaLong where not null alphaLong`pnl;

  / Portfolio table
  portTbl:([] date:j`date; strategy:(count j)#`portfolio; pnl:`float$(j`portfolio_pnl));

  / Pass-through config (atm_strategy_performance_explain handles defaults itself)
  alphaExp:.oca.atm_strategy_performance_explain[alphaLong; c];
  portExp:.oca.atm_strategy_performance_explain[portTbl; c];

  / Attribution of each alpha to total
  topRows:tn#(j @ reverse iasc j`portfolio_pnl);
  worstRows:tn#(j @ iasc j`portfolio_pnl);
  topDates:topRows`date;
  worstDates:worstRows`date;
  attrRows:();
  i:0;
  while[i<count acols;
    s1:acols i;
    x:`float$(j s1);
    y:`float$(j`portfolio_pnl);
    ok:(not null x) & not null y;
    x:x where ok;
    y:y where ok;
    n:count x;
    sx:sum x;
    sy:sum y;
    sax:sum abs x;
    say:sum abs y;
    contrib:$[(abs sy)<=1e-12; 0n; sx % sy];
    absContrib:$[say<=1e-12; 0n; sax % say];
    cr:.oca.safe_corr[x;y];
    bt:.oca.safe_beta[x;y];
    sxsgn:(`float$(x>0f)) - (`float$(x<0f));
    sysgn:(`float$(y>0f)) - (`float$(y<0f));
    aMask:(sxsgn<>0f) & (sysgn<>0f);
    agree:$[(sum aMask)=0; 0n; avg (sxsgn where aMask) = (sysgn where aMask)];
    upM:y>0f;
    dnM:y<0f;
    upCap:$[(sum upM)=0; 0n; avg x where upM];
    dnCap:$[(sum dnM)=0; 0n; avg x where dnM];
    jt:j where (j`date) in topDates;
    jw:j where (j`date) in worstDates;
    xt:`float$(jt s1);
    yt:`float$(jt`portfolio_pnl);
    xw:`float$(jw s1);
    yw:`float$(jw`portfolio_pnl);
    tShare:$[(abs sum yt)<=1e-12; 0n; sum xt % sum yt];
    wShare:$[(abs sum yw)<=1e-12; 0n; sum xw % sum yw];
    attrRows,:enlist ([] strategy:enlist s1; n_obs:enlist n; sum_pnl:enlist sx; contrib_pct:enlist contrib; abs_contrib_pct:enlist absContrib; corr_total:enlist cr; beta_total:enlist bt; sign_agree_rate:enlist agree; up_capture:enlist upCap; down_capture:enlist dnCap; top_days_share:enlist tShare; worst_days_share:enlist wShare);
    i+:1;
  ];
  attrib:$[
    0=count attrRows;
    ([] strategy:`symbol$(); n_obs:`int$(); sum_pnl:`float$(); contrib_pct:`float$(); abs_contrib_pct:`float$(); corr_total:`float$(); beta_total:`float$(); sign_agree_rate:`float$(); up_capture:`float$(); down_capture:`float$(); top_days_share:`float$(); worst_days_share:`float$());
    raze attrRows
  ];

  as:alphaExp`summary;
  as:0!select strategy,alpha_total_pnl:total_pnl,win_rate,sharpe,max_drawdown,topn_pnl_conc,worstn_abs_pnl_conc,tail_balance_score,avg_day_pnl,pnl_stdev,fragility_ratio,fragile_edge,edge_retention,edge_decay_pct,overfit_risk_score,overfit_risk_bucket from as;
  attrib:attrib lj `strategy xkey as;

  / Auto grouping
  corrHi:$[`group_corr_hi in key c; 1f*c`group_corr_hi; 0.2f];
  concHi:$[`group_conc_hi in key c; 1f*c`group_conc_hi; 0.6f];
  minMembers:$[`min_subtype_members in key c; c`min_subtype_members; 2];
  minMembers:max 1, `int$minMembers;
  minDistinct:$[
    `min_subtypes in key c; c`min_subtypes;
    `min_subtype_size in key c; c`min_subtype_size;
    0
  ];
  minDistinct:max 0, `int$minDistinct;
  attrib:.oca.alpha_assign_group_subtype[attrib; corrHi; concHi; minMembers];
  attrib:.oca.alpha_ensure_min_subtypes[attrib; minDistinct];
  attrib:attrib @ reverse iasc attrib`sum_pnl;

  groupSummary:0!select
    n_alpha:count i,
    group_pnl:sum sum_pnl,
    group_abs_pnl:sum abs sum_pnl,
    avg_corr_total:avg corr_total
    by alpha_group from attrib;
  groupSummary:groupSummary @ reverse iasc groupSummary`group_pnl;
  subtypeSummary:0!select
    n_alpha:count i,
    subtype_pnl:sum sum_pnl,
    subtype_abs_pnl:sum abs sum_pnl,
    avg_corr_total:avg corr_total,
    avg_win_rate:avg win_rate,
    avg_topn_conc:avg topn_pnl_conc,
    avg_tail_balance_score:avg tail_balance_score,
    fragile_share:avg fragile_edge,
    avg_fragility_ratio:avg fragility_ratio,
    avg_edge_decay_pct:avg edge_decay_pct,
    avg_overfit_risk_score:avg overfit_risk_score
    by alpha_subtype from attrib;
  subtypeSummary:subtypeSummary @ reverse iasc subtypeSummary`subtype_pnl;
  subtypeSummary:.oca.subtype_behavior_enrich subtypeSummary;

  work:attrib where (attrib`sum_pnl)>0f;
  work:tn#work;
  drag:attrib where (attrib`sum_pnl)<0f;
  drag:tn#(drag @ iasc drag`sum_pnl);
  mres:.oca.alpha_monthly_status[alphaLong; attrib];
  alphaMonthly:mres`alpha_monthly;
  monthlyFrag:0!select n_alpha:count i, month_pnl:sum month_pnl, avg_fragility_ratio:avg fragility_ratio, fragile_share:avg fragile_edge by month from alphaMonthly;
  monthlyFrag:(`month) xasc monthlyFrag;
  monthlyFragSubtype:0!select n_alpha:count i, month_pnl:sum month_pnl, avg_win_rate:avg win_rate, avg_fragility_ratio:avg fragility_ratio, fragile_share:avg fragile_edge by month,alpha_subtype from alphaMonthly;
  subtypeBeh:0!select alpha_subtype,behavior_label,behavior_text from subtypeSummary;
  monthlyFragSubtype:monthlyFragSubtype lj `alpha_subtype xkey subtypeBeh;
  monthlyFragSubtype:(`month`alpha_subtype) xasc monthlyFragSubtype;

  / High-level narrative
  psTbl:portExp`summary;
  hasPs:(count psTbl)>0;
  pTot:$[hasPs; first psTbl`total_pnl; 0n];
  pN:$[hasPs; first psTbl`n_days; 0N];
  pWr:$[hasPs; first psTbl`win_rate; 0n];
  posN:sum ((attrib`sum_pnl)>0f);
  negN:sum ((attrib`sum_pnl)<0f);
  topA:$[0=count work; `; first work`strategy];
  topP:$[0=count work; 0n; first work`sum_pnl];
  badA:$[0=count drag; `; first drag`strategy];
  badP:$[0=count drag; 0n; first drag`sum_pnl];
  h1:$[
    hasPs;
    raze ("Portfolio pnl "; string pTot; " over "; string pN; " days (win rate "; string pWr; ").");
    "No portfolio summary rows."
  ];
  h2:raze (string posN; " alphas are working (positive pnl) and "; string negN; " are not across "; string count distinct attrib`alpha_subtype; " subtypes.");
  h3:raze ("Top contributor: "; string topA; " ("; string topP; "). Largest drag: "; string badA; " ("; string badP; ").");
  ntext:`$ (h1;h2;h3);
  narrative:([] section:`headline`breadth`leaders; text:ntext);

  (`portfolio_summary`portfolio_flags`portfolio_narrative`portfolio_edge_recheck`alpha_summary`alpha_driver_effects`alpha_flags`alpha_narrative`alpha_edge_recheck`alpha_attribution`group_summary`subtype_summary`working`not_working`alpha_monthly`working_monthly`not_working_monthly`monthly_edge_fragility`monthly_edge_fragility_by_subtype`narrative)!(
    portExp`summary;
    portExp`flags;
    portExp`narrative;
    portExp`edge_recheck;
    alphaExp`summary;
    alphaExp`driver_effects;
    alphaExp`flags;
    alphaExp`narrative;
    alphaExp`edge_recheck;
    attrib;
    groupSummary;
    subtypeSummary;
    work;
    drag;
    mres`alpha_monthly;
    mres`working_monthly;
    mres`not_working_monthly;
    monthlyFrag;
    monthlyFragSubtype;
    narrative
  )
 }

min_abs_diff:{[vals; p]
  if[p~(::); :0w];
  if[null p; :0w];
  t:type vals;
  xs:$[t=0h; vals; t>0h; vals; enlist vals];
  if[0=count xs; :0w];
  to_f:{[x] .[`float$; enlist x; {0n}]};
  pf:to_f p;
  if[null pf; :0w];
  xf:to_f each xs;
  xf:xf where not null xf;
  if[0=count xf; :0w];
  min abs(xf - pf)
 }

atm_strategy_price_check:{[adf; rets; price_col; tol]
  a:adf;
  r:rets;
  if[99h=type a; a:0!a];
  if[99h=type r; r:0!r];
  if[98h<>type a; '"adf must be a table"];
  if[98h<>type r; '"returns must be a table"];
  pc:$[price_col~(::); `settle; .oca.to_sym price_col];
  eps:$[tol~(::); 0.000000001f; 1f*tol];

  req_a:`date`expiry`strike`put_call;
  if[not all req_a in cols a; '"adf missing required columns (`date`expiry`strike`put_call)"];
  if[not pc in cols a; '"adf missing selected price column"];
  req_r:`date`expiry`strike`call_leg_price`put_leg_price;
  if[not all req_r in cols r; '"returns missing required leg columns"];

  a:update put_call:.oca.norm_put_call put_call from a;
  a:update price_chk:a[;pc] from a;
  if[not `underlying_ric in cols a; a:update underlying_ric:(count a)#` from a];
  if[not `underlying_ric in cols r; r:update underlying_ric:(count r)#` from r];
  a:update underlying_ric:.oca.to_sym each underlying_ric from a;
  r:update underlying_ric:.oca.to_sym each underlying_ric from r;
  if[not `strategy in cols r; r:update strategy:(count r)#`unknown from r];
  if[not `price_mode in cols r; r:update price_mode:(count r)#$[pc=`settle;`market;`theo] from r];

  calls:0!select call_vals:price_chk by date,expiry,strike,underlying_ric from a where (a`put_call)=`C;
  puts:0!select put_vals:price_chk by date,expiry,strike,underlying_ric from a where (a`put_call)=`P;
  out:r lj `date`expiry`strike`underlying_ric xkey calls;
  out:out lj `date`expiry`strike`underlying_ric xkey puts;

  out:update call_min_diff:.oca.min_abs_diff'[call_vals; call_leg_price], put_min_diff:.oca.min_abs_diff'[put_vals; put_leg_price] from out;
  out:update call_match:call_min_diff<=eps, put_match:put_min_diff<=eps from out;
  out:update legs_match:call_match & put_match from out;

  summary:0!select
    n_rows:count i,
    n_match:sum legs_match,
    n_mismatch:sum not legs_match,
    max_call_diff:max call_min_diff,
    max_put_diff:max put_min_diff
    by strategy,price_mode from out;
  mism: out where not out`legs_match;
  (`summary`mismatches`joined)! (summary; mism; out)
 }

scale_strategy_returns:{[rets; scale]
  t:rets;
  if[99h=type t; t:0!t];
  if[98h<>type t; '"returns input must be a table"];
  s:$[scale~(::); 1f; 1f*scale];
  if[s=1f; :t];
  if[`price in cols t; t:update price:s*price from t];
  if[`pnl in cols t; t:update pnl:s*pnl from t];
  if[`call_leg_price in cols t; t:update call_leg_price:s*call_leg_price from t];
  if[`put_leg_price in cols t; t:update put_leg_price:s*put_leg_price from t];
  if[`straddle_leg_sum in cols t; t:update straddle_leg_sum:s*straddle_leg_sum from t];
  t
 }

optimize_raw1:{[args]
  d:args_dict args;
  tbls:d`tables;
  cfg:d`cfg;
  dm_arg:d`date_mode;
  ep_arg:d`epoch;
  libpath:d`libpath;
  .oca.ensure_init libpath;
  if[not dm_arg~(::); .oca.date_mode::dm_arg];
  if[not ep_arg~(::);
    ep_arg_str:$[10h=type ep_arg; ep_arg; string ep_arg];
    if[0<count ep_arg_str;
      .oca.epoch::ep_arg;
      .oca.epoch_date::"D"$ep_arg_str;
      .oca.epoch_ts::.oca.epoch_date + 0D00:00:00.000000000;
    ];
  ];
  res: .oca.opt_wrapper[tbls; cfg];
  dm:$[dm_arg~(::); .oca.date_mode; dm_arg];
  ep:$[ep_arg~(::); .oca.epoch; ep_arg];
  dm_str:$[10h=type dm; dm; string dm];
  ep_str:$[10h=type ep; ep; string ep];
  resd_py: .oca.opt_to_dict[res; dm_str; ep_str];
  .p.py2q .oca.unwrap resd_py
 }

optimize_raw_simple1:{[args]
  d:args_dict args;
  tbls:d`tables;
  cfg:d`cfg;
  dm_arg:d`date_mode;
  ep_arg:d`epoch;
  libpath:d`libpath;
  .oca.ensure_init libpath;
  if[not dm_arg~(::); .oca.date_mode::dm_arg];
  if[not ep_arg~(::);
    ep_arg_str:$[10h=type ep_arg; ep_arg; string ep_arg];
    if[0<count ep_arg_str;
      .oca.epoch::ep_arg;
      .oca.epoch_date::"D"$ep_arg_str;
      .oca.epoch_ts::.oca.epoch_date + 0D00:00:00.000000000;
    ];
  ];
  res: .oca.opt_wrapper_simple[tbls; cfg];
  dm:$[dm_arg~(::); .oca.date_mode; dm_arg];
  ep:$[ep_arg~(::); .oca.epoch; ep_arg];
  dm_str:$[10h=type dm; dm; string dm];
  ep_str:$[10h=type ep; ep; string ep];
  resd_py: .oca.opt_to_dict[res; dm_str; ep_str];
  .p.py2q .oca.unwrap resd_py
 }

optimize_raw_cvar1:{[args]
  d:args_dict args;
  tbls:d`tables;
  cfg:d`cfg;
  dm_arg:d`date_mode;
  ep_arg:d`epoch;
  libpath:d`libpath;
  .oca.ensure_init libpath;
  if[not dm_arg~(::); .oca.date_mode::dm_arg];
  if[not ep_arg~(::);
    ep_arg_str:$[10h=type ep_arg; ep_arg; string ep_arg];
    if[0<count ep_arg_str;
      .oca.epoch::ep_arg;
      .oca.epoch_date::"D"$ep_arg_str;
      .oca.epoch_ts::.oca.epoch_date + 0D00:00:00.000000000;
    ];
  ];
  res: .oca.opt_wrapper_cvar[tbls; cfg];
  dm:$[dm_arg~(::); .oca.date_mode; dm_arg];
  ep:$[ep_arg~(::); .oca.epoch; ep_arg];
  dm_str:$[10h=type dm; dm; string dm];
  ep_str:$[10h=type ep; ep; string ep];
  resd_py: .oca.opt_to_dict[res; dm_str; ep_str];
  .p.py2q .oca.unwrap resd_py
 }

optimize_tables1:{[args]
  resd: .oca.optimize_raw1 args;
  k:key resd;
  v:value resd;
  k!.oca.to_table each v
 }

optimize_tables_simple1:{[args]
  resd: .oca.optimize_raw_simple1 args;
  k:key resd;
  v:value resd;
  k!.oca.to_table each v
 }

optimize_tables_cvar1:{[args]
  resd: .oca.optimize_raw_cvar1 args;
  k:key resd;
  v:value resd;
  k!.oca.to_table each v
 }

optimize_weights1:{[args]
  resd: .oca.optimize_tables1 args;
  .oca.fix_dt resd[`portfolio_weights]
 }

optimize_weights_simple1:{[args]
  resd: .oca.optimize_tables_simple1 args;
  .oca.fix_dt resd[`portfolio_weights]
 }

optimize_weights_cvar1:{[args]
  resd: .oca.optimize_tables_cvar1 args;
  .oca.fix_dt resd[`portfolio_weights]
 }

optimize_raw:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_raw1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_raw_simple:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_raw_simple1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_raw_cvar:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_raw_cvar1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_tables:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_tables_simple:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables_simple1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_full_simple:{[tbls; cfg; dm; ep; libpath]
  c: .oca.full_cfg[tbls; cfg];
  .oca.optimize_tables_simple1[`tables`cfg`date_mode`epoch`libpath!(tbls;c;dm;ep;libpath)]
 }

optimize_tables_cvar:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables_cvar1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_simple:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables_simple1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_cvar:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables_cvar1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_full_cvar:{[tbls; cfg; dm; ep; libpath]
  c: .oca.full_cfg[tbls; cfg];
  .oca.optimize_tables_cvar1[`tables`cfg`date_mode`epoch`libpath!(tbls;c;dm;ep;libpath)]
 }

optimize_weights:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_weights1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_weights_simple:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_weights_simple1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_weights_cvar:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_weights_cvar1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

analyze_chain_df:{[options; curve; cfg; libpath]
  .oca.ensure_init libpath;
  c: .oca.normalize_cfg cfg;
  opt: options;
  if[99h=type opt; opt:.oca.to_table opt];
  cur: curve;
  if[99h=type cur; cur:.oca.to_table cur];
  res: .oca.analyze_chain_wrapper[opt; cur; c];
  out: .p.py2q .oca.unwrap res;
  out:$[99h=type out; .oca.to_table out; out];
  if[98h=type out;
    if[`put_call in cols out; out:update put_call:.oca.norm_put_call put_call from out];
    if[`quote_perm_id in cols out; out:update quote_perm_id:.oca.norm_quote_perm_id quote_perm_id from out];
    if[`underlying_ric in cols out; out:update underlying_ric:.oca.to_sym each underlying_ric from out];
  ];
  out
 }

build_strategy_book_df:{[analytics; cfg; libpath]
  .oca.ensure_init libpath;
  c: .oca.normalize_cfg cfg;
  a: analytics;
  if[99h=type a; a:.oca.to_table a];
  res: .oca.strategy_book_wrapper[a; c];
  out: .p.py2q .oca.unwrap res;
  $[99h=type out; .oca.to_table out; out]
 }

strategy_screener_df:{[strategy_tbl; analytics_tbl; cfg; libpath]
  .oca.ensure_init libpath;
  c: .oca.normalize_cfg cfg;
  st: strategy_tbl;
  if[99h=type st; st:.oca.to_table st];
  at: analytics_tbl;
  if[99h=type at; at:.oca.to_table at];
  res: .oca.strategy_screener_wrapper[st; at; c];
  out: .p.py2q .oca.unwrap res;
  $[99h=type out; .oca.to_table out; out]
 }

scenario_pnl_strategy_df:{[strategy_tbl; cfg; libpath]
  .oca.ensure_init libpath;
  c: .oca.normalize_cfg cfg;
  st: strategy_tbl;
  if[99h=type st; st:.oca.to_table st];
  res: .oca.scenario_pnl_strategy_wrapper[st; c];
  out: .p.py2q .oca.unwrap res;
  $[99h=type out; .oca.to_table out; out]
 }

bbg_eco_history:{[securities; start_date; end_date; cfg; libpath]
  .oca.ensure_init libpath;
  c:.oca.cfg_to_dict cfg;
  if[0=count key c; c:.oca.normalize_cfg cfg];
  sd:string start_date;
  ed:string end_date;
  res:.oca.bbg_eco_wrapper[securities; sd; ed; c];
  out:.p.py2q .oca.unwrap res;
  out:$[99h=type out; .oca.to_table out; out];
  $[98h=type out; .oca.fix_dt out; out]
 }

\d .
