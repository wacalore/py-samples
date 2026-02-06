\l p.q

\d .oca

/ Minimal embedPy helper for options_chain_analyzer optimizer
inited:0b
date_mode:`days
epoch:"2000-01-01"
epoch_date:2000.01.01
epoch_ts:2000.01.01D00:00:00.000000000

unwrap:{ $[105h=type x; x`.; x] }

init:{[libpath; dm; ep]
  if[libpath~(::); libpath:system "pwd"];
  if[0h=type libpath; libpath:raze libpath];
  if[not libpath~(::);
    if[0=count string libpath; libpath:system "pwd"];
  ];
  if[0h=type libpath; libpath:raze libpath];
  if[10h<>type libpath; libpath:string libpath];
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
  .p.e "sys.modules.pop('options_chain_analyzer.optimizer', None)";
  .p.e "sys.modules.pop('options_chain_analyzer', None)";
  .p.e "import options_chain_analyzer as oca";
  .p.e "def oca_fix_temporal_df(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if not isinstance(x, pd.DataFrame):\n    return x\n  date_only_cols = {'date', 'expiry', 'reb_date', 'end_date', 'curve_date'}\n  temporal_cols = date_only_cols | {'dt'}\n  out = x.copy()\n  for col in out.columns:\n    c = str(col).lower()\n    if (c not in temporal_cols) and (not c.endswith('_date')):\n      continue\n    s = out[col]\n    if pd.api.types.is_datetime64_any_dtype(s):\n      continue\n    try:\n      if pd.api.types.is_numeric_dtype(s):\n        if c == 'dt':\n          nz = s.dropna()\n          unit = 'D'\n          if len(nz) > 0:\n            vmax = float(nz.abs().max())\n            if vmax > 1.0e10:\n              unit = 'ns'\n          out[col] = pd.to_datetime(s, unit=unit, origin='2000-01-01', errors='coerce')\n        else:\n          out[col] = pd.to_datetime(s, unit='D', origin='2000-01-01', errors='coerce')\n        continue\n    except Exception:\n      pass\n    try:\n      if s.dtype == object:\n        ss = s.map(lambda z: None if z is None else str(z))\n      else:\n        ss = s.astype(str)\n      ss = ss.replace('0Nd', None).replace('0Np', None)\n      ss = ss.str.replace('D', 'T', regex=False)\n      out[col] = pd.to_datetime(ss, errors='coerce')\n    except Exception:\n      try:\n        out[col] = pd.to_datetime(s, errors='coerce')\n      except Exception:\n        pass\n  return out";
  .p.e "def oca_as_df(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if isinstance(x, pd.DataFrame):\n    return oca_fix_temporal_df(x)\n  if isinstance(x, dict) or isinstance(x, (list, tuple)):\n    try:\n      return oca_fix_temporal_df(pd.DataFrame(x))\n    except Exception:\n      return x\n  return x";
  .p.e "def oca_df_to_dict(x):\n  try:\n    import pandas as pd\n  except Exception:\n    return x\n  if not isinstance(x, pd.DataFrame):\n    return x\n  date_only_cols = {'date', 'expiry', 'reb_date', 'end_date', 'curve_date'}\n  temporal_cols = date_only_cols | {'dt'}\n  out = {}\n  for col in x.columns:\n    s = x[col]\n    c = str(col).lower()\n    date_only = (c in date_only_cols) or c.endswith('_date')\n    is_named_temporal = (c in temporal_cols) or c.endswith('_date')\n    is_dt = pd.api.types.is_datetime64_any_dtype(s)\n    if (not is_dt) and is_named_temporal and (s.dtype == object):\n      for vv in s:\n        if vv is None:\n          continue\n        try:\n          if pd.isna(vv):\n            continue\n        except Exception:\n          pass\n        is_dt = isinstance(vv, pd.Timestamp)\n        if not is_dt:\n          try:\n            pd.Timestamp(vv)\n            is_dt = True\n          except Exception:\n            is_dt = False\n        break\n    if is_dt:\n      fmt = '%Y.%m.%d' if date_only else '%Y.%m.%dD%H:%M:%S.%f'\n      vals = []\n      for vv in s:\n        if vv is None:\n          vals.append('0Nd' if date_only else '0Np')\n          continue\n        try:\n          if pd.isna(vv):\n            vals.append('0Nd' if date_only else '0Np')\n            continue\n        except Exception:\n          pass\n        try:\n          vals.append(pd.Timestamp(vv).strftime(fmt))\n        except Exception:\n          vals.append(vv)\n      out[col] = vals\n    else:\n      out[col] = s.tolist()\n  return out";
  .p.e "def oca_opt_wrapper(tables, cfg=None): return oca.optimize_portfolio_with_pca(tables, cfg)";
  .p.e "def oca_opt_simple_wrapper(tables, cfg=None): return oca.optimize_portfolio(tables, cfg)";
  .p.e "def oca_opt_cvar_wrapper(tables, cfg=None): return oca.optimize_portfolio_cvar(tables, cfg)";
  .p.e "def oca_opt_to_dict(res, date_mode='days', epoch='2000-01-01'): return oca.optimizer_result_to_dict(res, date_mode=date_mode, epoch=epoch)";
  .p.e "def oca_analyze_chain_df(options_df, curve, cfg=None):\n  cfg = {} if cfg is None else cfg\n  options_df = oca_as_df(options_df)\n  curve = oca_as_df(curve)\n  out = oca.analyze_chain_df(options_df, curve, use_numba=cfg.get('use_numba'), group_by_date=cfg.get('group_by_date'), curve_date_col=cfg.get('curve_date_col'))\n  return oca_df_to_dict(out)";
  .p.e "def oca_build_strategy_book_df(analytics_df, cfg=None):\n  cfg = {} if cfg is None else cfg\n  analytics_df = oca_as_df(analytics_df)\n  widths = cfg.get('widths', (0.5, 1.0, 2.0))\n  templates = cfg.get('strategy_templates')\n  out = oca.build_strategy_book_df(analytics_df, widths=widths, strategy_templates=templates)\n  return oca_df_to_dict(out)";
  .p.e "def oca_strategy_screener_df(strategy_df, analytics_df=None, cfg=None):\n  cfg = {} if cfg is None else cfg\n  strategy_df = oca_as_df(strategy_df)\n  if analytics_df is not None:\n    analytics_df = oca_as_df(analytics_df)\n  out = oca.strategy_screener_df(\n    strategy_df,\n    analytics_df=analytics_df,\n    vol_col=cfg.get('vol_col', 'iv_atm'),\n    vol_fallback=cfg.get('vol_fallback'),\n    pop_samples=cfg.get('pop_samples', 5000),\n    pop_seed=cfg.get('pop_seed', 7),\n    mispricing_metric=cfg.get('mispricing_metric', 'edge_per_vega'),\n    mispricing_quantile=cfg.get('mispricing_quantile', 0.9),\n    pop_threshold=cfg.get('pop_threshold', 0.6),\n    ev_threshold=cfg.get('ev_threshold', 0.0),\n    upside_metric=cfg.get('upside_metric', 'upside_p95'),\n    upside_quantile=cfg.get('upside_quantile', 0.8),\n    upside_threshold=cfg.get('upside_threshold'),\n    credit_debit_tolerance=cfg.get('credit_debit_tolerance', 1.0e-8),\n    filter_only=cfg.get('filter_only', True),\n    top_n=cfg.get('top_n'),\n    weight_mispricing=cfg.get('weight_mispricing', 1.0),\n    weight_ev=cfg.get('weight_ev', 0.5),\n    weight_pop=cfg.get('weight_pop', 0.5)\n  )\n  return oca_df_to_dict(out)";
  .p.e "def oca_scenario_pnl_strategy_df(strategy_df, cfg=None):\n  cfg = {} if cfg is None else cfg\n  strategy_df = oca_as_df(strategy_df)\n  out = oca.scenario_pnl_strategy_df(\n    strategy_df,\n    dF=cfg.get('dF', 0.0),\n    dVol=cfg.get('dVol', 0.0),\n    dRate=cfg.get('dRate', 0.0),\n    dt_days=cfg.get('dt_days', 0.0)\n  )\n  return oca_df_to_dict(out)";
  opt_wrapper::.p.get[`oca_opt_wrapper];
  opt_wrapper_simple::.p.get[`oca_opt_simple_wrapper];
  opt_wrapper_cvar::.p.get[`oca_opt_cvar_wrapper];
  opt_to_dict::.p.get[`oca_opt_to_dict];
  analyze_chain_wrapper::.p.get[`oca_analyze_chain_df];
  strategy_book_wrapper::.p.get[`oca_build_strategy_book_df];
  strategy_screener_wrapper::.p.get[`oca_strategy_screener_df];
  scenario_pnl_strategy_wrapper::.p.get[`oca_scenario_pnl_strategy_df];
  inited::1b;
  :1b;
 }

ensure_init:{[libpath]
  if[not inited; init[libpath;date_mode;epoch]];
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
  if[t=11h; : "D"$string each v];
  if[t=0h; : "D"$v];
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
    if[count a<5; a:a,(5-count a)#(::)];
    (`tables`cfg`date_mode`epoch`libpath)!a
   ]
 }

normalize_cfg:{[cfg]
  $[cfg~(::); ()!(); cfg]
 }

to_sym:{[x]
  t:type x;
  $[t=-11h; x;
    t=11h; first x;
    `$string x]
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

atm_strategy_returns:{[t; rebalance_days; target_dte; min_dte; max_dte; price_mode; strategy; side]
  r: $[rebalance_days~(::); 5; rebalance_days];
  td: $[target_dte~(::); 30; target_dte];
  mind: $[min_dte~(::); 7; min_dte];
  maxd: $[max_dte~(::); ::; max_dte];
  pm: $[price_mode~(::); `market; price_mode];
  strat: $[strategy~(::); `straddle; strategy];
  s: $[side~(::); 1f; side];
  price_col: $[pm in (`market;`mkt;`settle); `settle; `theo];
  req: `date`expiry`strike`put_call`underlying;
  if[not all req in cols t; '"analytics table missing required columns"];
  if[not price_col in cols t; '"analytics table missing price column"];
  r: max 1, `int$r;
  if[r < 1; '"rebalance_days must be >= 1"];
  tt: t;
  dty: abs type (tt`date);
  if[dty in 12 15h; tt: update date:date date from tt];
  if[not dty in 14 12 15h; '"date column must be date or timestamp/datetime"];
  ety: abs type (tt`expiry);
  if[ety in 12 15h; tt: update expiry:date expiry from tt];
  if[not ety in 14 12 15h; '"expiry column must be date or timestamp/datetime"];
  tt: update price_sel: tt[;price_col] from tt;
  tt: update dte: expiry - date from tt;
  dates: asc distinct tt`date;
  idx: til `int$count dates;
  reb_dates: dates where (idx mod r) = 0;

  pick:{[d; td; mind; maxd; tt]
    sub: tt where (tt`date)=d;
    sub: sub where (sub`dte) >= mind;
    if[not maxd~(::); sub: sub where (sub`dte) <= maxd];
    if[0=count sub; :()];
    exp_tbl: 0!select dte:first dte by expiry from sub;
    diffs: abs ((exp_tbl`dte) - td);
    md: exec min d from ([] d: diffs);
    exp_exp: exp_tbl`expiry;
    exp_sel: exp_exp where diffs = md;
    if[0=count exp_sel; :()];
    exp_sel: exp_sel 0;
    sub2: sub where (sub`expiry)=exp_sel;
    u: first sub2`underlying;
    sub2: update m: abs(strike - u) from sub2;
    m0: exec min m from sub2;
    k: first ((sub2`strike) where (sub2`m)=m0);
    (`reb_date`expiry`strike`underlying)! (d; exp_sel; k; u)
  };

  picks: pick'[reb_dates; (count reb_dates)#enlist td; (count reb_dates)#enlist mind; (count reb_dates)#enlist maxd; (count reb_dates)#enlist tt];
  picks: picks except enlist ();
  if[0=count picks; '"no valid rebalance dates"];
  picks_tbl: flip (`reb_date`expiry`strike`underlying)! (picks`reb_date; picks`expiry; picks`strike; picks`underlying);

  reb: picks_tbl`reb_date;
  end_dates: 1 _ reb, enlist (1 + last dates);
  pc_set: $[strat=`call; enlist `C; strat=`put; enlist `P; `C`P];

  seg_tbl: update end_date:end_dates from picks_tbl;
  env: (`tt`dates`pc_set`strat`side`price_mode)!(tt; dates; pc_set; strat; s; pm);

  seg_fn:{[seg; env]
    rb: seg`reb_date;
    re: seg`end_date;
    exp_date: seg`expiry;
    strike: seg`strike;
    tt: env`tt;
    dates: env`dates;
    pc_set: env`pc_set;
    strat: env`strat;
    s: env`side;
    pm: env`price_mode;
    seg_dates: dates where (dates>=rb) & (dates<re);
    if[0=count seg_dates; :()];
    mask: (tt`date) in seg_dates;
    mask: mask & (tt`expiry)=exp_date;
    mask: mask & (tt`strike)=strike;
    mask: mask & (tt`put_call) in pc_set;
    leg: select date, put_call, price: price_sel from tt where mask;
    if[strat=`straddle;
      cnt_map: count each group leg`date;
      ncol: cnt_map leg`date;
      mask2: ncol = 2;
      leg: leg where mask2;
    ];
    if[0=count leg; :()];
    px: 0!select price: sum price by date from leg;
    px: px @ iasc px`date;
    px: update reb_date:rb, expiry:exp_date, strike:strike, strategy:strat, price_mode:pm, side:s from px;
    px: update pnl: s * (price - prev price) from px;
    px: update ret: pnl % abs prev price from px;
    px
  };

  segs: seg_fn'[seg_tbl; (count seg_tbl)#enlist env];
  segs: segs where 0 < count each segs;
  if[0=count segs; '"no pricing rows for selected ATM strategy"];
  raze segs
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
  $[99h=type out; .oca.to_table out; out]
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

\d .
