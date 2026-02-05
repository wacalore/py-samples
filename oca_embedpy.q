\l p.q

\d oca

/ Minimal embedPy helper for options_chain_analyzer optimizer
inited:0b
date_mode:`days
epoch:"2000-01-01"
epoch_date:2000.01.01

unwrap:{ $[105h=type x; x`.; x] }

init:{[libpath; dm; ep]
  if[null libpath; libpath:system "pwd"];
  libpath:string libpath;
  if[not null dm; date_mode::dm];
  if[not null ep;
    epoch::ep;
    epoch_date::$["D"$string ep];
  ];
  .p.eval["import sys"];
  .p.eval["p = r'''",libpath,"'''"];
  .p.eval["import sys; p = r'''",libpath,"'''; sys.path.insert(0,p) if p not in sys.path else None"];
  .p.eval["import options_chain_analyzer as oca"];
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
  if[6h=type t`dt; :update dt:epoch_date + dt from t];
  :t;
 }

to_table:{[v]
  $[98h=type v; fix_dt v;
    99h=type v; fix_dt flip v;
    v]
 }

optimize_raw:{[tables; cfg; dm; ep; libpath]
  ensure_init libpath;
  res: opt_wrapper[tables; cfg];
  dm:$[null dm; date_mode; dm];
  ep:$[null ep; epoch; ep];
  resd_py: opt_to_dict[res; string dm; string ep];
  .p.py2q unwrap resd_py
 }

optimize_tables:{[tables; cfg; dm; ep; libpath]
  resd: optimize_raw[tables; cfg; dm; ep; libpath];
  k:key resd;
  k!{to_table resd x} each k
 }

optimize_weights:{[tables; cfg; dm; ep; libpath]
  resd: optimize_tables[tables; cfg; dm; ep; libpath];
  fix_dt resd[`portfolio_weights]
 }

\d .
