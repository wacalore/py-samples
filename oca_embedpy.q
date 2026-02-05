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
  .p.e "def oca_opt_wrapper(tables, cfg=None): return oca.optimize_portfolio_with_pca(tables, cfg)";
  .p.e "def oca_opt_to_dict(res, date_mode='days', epoch='2000-01-01'): return oca.optimizer_result_to_dict(res, date_mode=date_mode, epoch=epoch)";
  opt_wrapper::.p.get[`oca_opt_wrapper];
  opt_to_dict::.p.get[`oca_opt_to_dict];
  inited::1b;
  :1b;
 }

ensure_init:{[libpath]
  if[not inited; init[libpath;date_mode;epoch]];
 }

fix_dt:{[t]
  if[not `dt in cols t; :t];
  dm_str:$[10h=type .oca.date_mode; .oca.date_mode; string .oca.date_mode];
  if[dm_str in ("days";"day");
    if[6h=type t`dt; :update dt:.oca.epoch_date + dt from t];
    if[7h=type t`dt; :update dt:.oca.epoch_date + `int$dt from t];
    :t;
  ];
  if[dm_str in ("ns";"nanoseconds";"timestamp";"datetime64[ns]");
    if[6h=type t`dt; :update dt:.oca.epoch_ts + `long$dt from t];
    if[7h=type t`dt; :update dt:.oca.epoch_ts + dt from t];
    :t;
  ];
  :t;
 }

to_table:{[v]
  $[98h=type v; fix_dt v;
    99h=type v;
      $[98h=type key v; fix_dt 0!v; fix_dt[flip v]];
    v]
 }

args_dict:{[args]
  $[99h=type args; args;
    a:$[0h=type args; args; enlist args];
    if[count a<5; a:a,(5-count a)#(::)];
    (`tables`cfg`date_mode`epoch`libpath)!a
   ]
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

optimize_tables1:{[args]
  resd: .oca.optimize_raw1 args;
  k:key resd;
  v:value resd;
  k!.oca.to_table each v
 }

optimize_weights1:{[args]
  resd: .oca.optimize_tables1 args;
  .oca.fix_dt resd[`portfolio_weights]
 }

optimize_raw:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_raw1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_tables:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_tables1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

optimize_weights:{[tbls; cfg; dm; ep; libpath]
  .oca.optimize_weights1[`tables`cfg`date_mode`epoch`libpath!(tbls;cfg;dm;ep;libpath)]
 }

\d .
