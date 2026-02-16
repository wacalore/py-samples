// =============================================================================
// ALPHA RESEARCH PIPELINE WITH BAYESIAN OPTIMIZATION
// =============================================================================
// Systematic alpha evaluation and parameter optimization for both
// cross-sectional and time-series signals.
// Version: 0.1.0
// Dependency: kdbtools.q (kRBF, distSq, mm, minv, eye, fwdReturn)
// Optional:   cond.q (rollingIC), pcrisk.q (alphaListOptimize)

\d .alphalab

// -----------------------------------------------------------------------------
// CONFIGURATION
// -----------------------------------------------------------------------------

defaultCfg:`nFolds`nInit`nIter`nCandidates`horizons`lamTO`foldAgg`noise`retCol`icWindow!(
    5;             // temporal cross-validation folds
    15;            // Latin Hypercube initial design points
    50;            // Bayesian optimization iterations
    500;           // random EI candidates per iteration
    1 2 5 10 20;   // forward return horizons for IC decay (CS only)
    0.1;           // turnover penalty weight
    `conservative; // fold aggregation: `conservative (min), `mean, `penalized
    1e-4;          // GP observation noise
    `ret;          // return column name in data
    60             // rolling IC window for TS alphas
    )

// Merge user config with defaults
mergeCfg:{[cfg] defaultCfg,$[99h=type cfg;cfg;()!()]}

// -----------------------------------------------------------------------------
// ALPHA REGISTRY
// -----------------------------------------------------------------------------

registry:()!()

// Register an alpha signal
// name:       symbol identifier (e.g., `csMom)
// type:       `ts (time-series) or `cs (cross-sectional)
// fn:         signal function {[data;params] -> ([] dt;sym;sig)} or ([] dt;sig)
// paramBounds: dict of 2-element float lists, e.g., `window`decay!(5 200f;0.5 0.99)
defineAlpha:{[name;typ;fn;paramBounds]
    if[not typ in `ts`cs; '"alphaType must be `ts or `cs"];
    pnames:key paramBounds;
    registry[name]:`fn`type`bounds`paramNames`nParams!(fn;typ;paramBounds;pnames;count pnames);
    name}

// List registered alphas
listAlphas:{[] if[0=count registry; :([] name:`$(); atype:`$(); nParams:`long$())];
    ([] name:key registry; atype:{x`type} each value registry; nParams:{x`nParams} each value registry)}

// Remove an alpha
removeAlpha:{[name] registry::name _ registry; name}

// Get alpha definition
getAlpha:{[name] registry name}

// -----------------------------------------------------------------------------
// STATISTICAL HELPERS
// -----------------------------------------------------------------------------

// Standard normal PDF (vectorized)
normPDF:{[x] (reciprocal sqrt 2 * acos -1f) * exp neg (x xexp 2) % 2}

// Standard normal CDF — Abramowitz-Stegun rational approximation (max error ~7.5e-8)
// Horner form with explicit parentheses for right-to-left safety
normCDF:{[x]
    isAtom:0h > type x;
    x:$[isAtom;enlist x;x];
    ax:abs x;
    t:reciprocal 1 + (0.2316419 * ax);
    // Horner polynomial: d = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    d:t * (0.319381530 + (t * (neg[0.356563782] + (t * (1.781477937 + (t * (neg[1.821255978] + (t * 1.330274429))))))));
    phi:1 - (d * normPDF ax);
    r:?[x >= 0; phi; 1 - phi];
    $[isAtom;first r;r]}

// Cross-sectional Spearman rank correlation (two vectors)
spearmanCorCS:{[x;y]
    valid:where not null[x] & not null y;
    $[2 > count valid; 0n; cor[iasc iasc x valid; iasc iasc y valid]]}

// -----------------------------------------------------------------------------
// CROSS-SECTIONAL EVALUATION (CS PIPELINE)
// -----------------------------------------------------------------------------
// Used exclusively for `cs alphas. The key new capability is cross-sectional IC:
// at each date, rank-correlate signal across ALL symbols with forward returns.

// Cross-sectional IC: at each date, Spearman rank-correlate signal vs N-period forward return
// sigTable: ([] dt; sym; sig)
// data:     table with dt, sym, and return column
// N:        forward return horizon
// cfg:      config dict (needs `retCol)
csIC:{[sigTable;data;N;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    // Compute forward returns per symbol
    syms:asc distinct data`sym;
    fwdData:raze {[d;rc;N;s]
        sub:select from d where sym=s;
        sub:`dt xasc sub;
        r:sub[rc];
        fr:.kdbtools.fwdReturn[N;r];
        update fwdRet:fr from sub
        }[data;rc;N] each syms;
    // Join signal with forward returns
    jn:sigTable lj `dt`sym xkey select dt,sym,fwdRet from fwdData;
    // Group by date, compute cross-sectional Spearman correlation
    dates:asc distinct jn`dt;
    ics:{[jn;d]
        sub:select from jn where dt=d, not null sig, not null fwdRet;
        $[2 > count sub; 0n; spearmanCorCS[sub`sig;sub`fwdRet]]
        }[jn] each dates;
    ([] dt:dates; ic:ics)}

// IC Information Ratio
icIR:{[icTable]
    v:icTable[`ic] where not null icTable`ic;
    $[0 = count v; 0n; (avg v) % dev v]}

// IC decay profile: IC and IC_IR at each forward horizon
// Returns ([] horizon; meanIC; icIR)
icDecayProfile:{[sigTable;data;horizons;cfg]
    res:{[sigTable;data;cfg;h]
        ict:csIC[sigTable;data;h;cfg];
        v:ict[`ic] where not null ict`ic;
        mic:$[0 = count v; 0n; avg v];
        ir:$[0 = count v; 0n; mic % dev v];
        (h;mic;ir)
        }[sigTable;data;cfg] each horizons;
    ([] horizon:res[;0]; meanIC:res[;1]; icIR:res[;2])}

// Long/short Sharpe: rank signals cross-sectionally, long top quintile, short bottom
longShortSharpe:{[sigTable;data;N;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    // Compute forward returns
    syms:asc distinct data`sym;
    fwdData:raze {[d;rc;N;s]
        sub:`dt xasc select from d where sym=s;
        fr:.kdbtools.fwdReturn[N;sub[rc]];
        update fwdRet:fr from sub
        }[data;rc;N] each syms;
    jn:sigTable lj `dt`sym xkey select dt,sym,fwdRet from fwdData;
    dates:asc distinct jn`dt;
    // At each date: long top quintile, short bottom quintile
    lsRets:{[jn;d]
        sub:select from jn where dt=d, not null sig, not null fwdRet;
        n:count sub;
        if[n < 5; :0n];
        ranked:iasc iasc sub`sig;
        q:n % 5;
        longIdx:where ranked >= (n - ceiling q);
        shortIdx:where ranked < ceiling q;
        if[(0 = count longIdx) or 0 = count shortIdx; :0n];
        ((avg sub[`fwdRet] longIdx) - avg sub[`fwdRet] shortIdx)
        }[jn] each dates;
    v:lsRets where not null lsRets;
    $[0 = count v; 0n; (sqrt 252) * (avg v) % dev v]}

// -----------------------------------------------------------------------------
// TIME-SERIES EVALUATION (TS PIPELINE)
// -----------------------------------------------------------------------------
// Used exclusively for `ts alphas. Primary metric: signal-weighted return Sharpe.

// Signal-weighted return Sharpe (annualized)
// Uses previous signal to avoid lookahead: pnl_t = prev_signal_t * return_t
tsSharpe:{[sigTable;data;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    hasSym:`sym in cols sigTable;
    $[hasSym;
        [
        // Multi-symbol: compute per-symbol PnL, average
        syms:asc distinct sigTable`sym;
        pnls:{[sigTable;data;rc;s]
            sig:`dt xasc select from sigTable where sym=s;
            dat:`dt xasc select from data where sym=s;
            jn:sig lj `dt xkey ([] dt:dat`dt; ret:dat[rc]);
            prevSig:prev sig`sig;
            (fills prevSig) * jn`ret
            }[sigTable;data;rc] each syms;
        // Average PnL across symbols
        pnl:avg pnls
        ];
        [
        // Single series
        jn:(`dt xasc sigTable) lj `dt xkey ([] dt:data`dt; ret:data[rc]);
        prevSig:prev jn`sig;
        pnl:(fills prevSig) * jn`ret
        ]
    ];
    v:pnl where not null pnl;
    $[0 = count v; 0n; (sqrt 252) * (avg v) % dev v]}

// Time-series rolling IC (wrapper around cond.rollingIC if available)
tsRollingIC:{[sigTable;data;window;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    hasSym:`sym in cols sigTable;
    if[hasSym;
        // For multi-symbol, compute per-symbol, average IC across symbols
        syms:asc distinct sigTable`sym;
        icTables:{[sigTable;data;rc;window;s]
            sig:`dt xasc select from sigTable where sym=s;
            dat:`dt xasc select from data where sym=s;
            jn:sig lj `dt xkey ([] dt:dat`dt; ret:dat[rc]);
            f:fills jn`sig;
            r:jn`ret;
            ic:$[@[{.cond.rollingIC};0b;{0b}]~0b;
                // Fallback: manual rolling IC
                {[f;r;w] wins1:{1_x,y}\[w#0n;fills f]; wins2:{1_x,y}\[w#0n;r];
                 {$[(count x)<2;0n;any null x,y;0n;cor[x where not null x;y where not null x]]}.' flip (wins1;wins2)}[f;r;window];
                .cond.rollingIC[f;r;window]
            ];
            ([] dt:jn`dt; ic:ic)
            }[sigTable;data;rc;window] each syms;
        // Average ICs across symbols by date
        combined:raze icTables;
        :select ic:avg ic by dt from combined
    ];
    // Single series
    jn:(`dt xasc sigTable) lj `dt xkey ([] dt:data`dt; ret:data[rc]);
    f:fills jn`sig;
    r:jn`ret;
    ic:$[@[{.cond.rollingIC};0b;{0b}]~0b;
        {[f;r;w] wins1:{1_x,y}\[w#0n;fills f]; wins2:{1_x,y}\[w#0n;r];
         {$[(count x)<2;0n;any null x,y;0n;cor[x where not null x;y where not null x]]}.' flip (wins1;wins2)}[f;r;window];
        .cond.rollingIC[f;r;window]
    ];
    ([] dt:jn`dt; ic:ic)}

// Hit rate: fraction of dates where sign(prev_signal) = sign(return)
tsHitRate:{[sigTable;data;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    hasSym:`sym in cols sigTable;
    $[hasSym;
        [
        syms:asc distinct sigTable`sym;
        hrs:{[sigTable;data;rc;s]
            sig:`dt xasc select from sigTable where sym=s;
            dat:`dt xasc select from data where sym=s;
            jn:sig lj `dt xkey ([] dt:dat`dt; ret:dat[rc]);
            prevSig:prev jn`sig;
            valid:where not null[prevSig] & not null jn`ret;
            $[0 = count valid; 0n; avg (signum prevSig valid) = signum jn[`ret] valid]
            }[sigTable;data;rc] each syms;
        avg hrs where not null hrs
        ];
        [
        jn:(`dt xasc sigTable) lj `dt xkey ([] dt:data`dt; ret:data[rc]);
        prevSig:prev jn`sig;
        valid:where not null[prevSig] & not null jn`ret;
        $[0 = count valid; 0n; avg (signum prevSig valid) = signum jn[`ret] valid]
        ]
    ]}

// Full PnL series
tsPnLSeries:{[sigTable;data;cfg]
    cfg:mergeCfg cfg;
    rc:cfg`retCol;
    hasSym:`sym in cols sigTable;
    $[hasSym;
        [
        syms:asc distinct sigTable`sym;
        pnlTables:{[sigTable;data;rc;s]
            sig:`dt xasc select from sigTable where sym=s;
            dat:`dt xasc select from data where sym=s;
            jn:sig lj `dt xkey ([] dt:dat`dt; ret:dat[rc]);
            prevSig:prev jn`sig;
            ([] dt:jn`dt; pnl:(fills prevSig) * jn`ret)
            }[sigTable;data;rc] each syms;
        // Average PnL across symbols
        combined:raze pnlTables;
        select pnl:avg pnl by dt from combined
        ];
        [
        jn:(`dt xasc sigTable) lj `dt xkey ([] dt:data`dt; ret:data[rc]);
        prevSig:prev jn`sig;
        ([] dt:jn`dt; pnl:(fills prevSig) * jn`ret)
        ]
    ]}

// -----------------------------------------------------------------------------
// SIGNAL TURNOVER
// -----------------------------------------------------------------------------

// Mean absolute change / mean absolute level, averaged across symbols
signalTurnover:{[sigTable]
    hasSym:`sym in cols sigTable;
    $[hasSym;
        [
        syms:asc distinct sigTable`sym;
        tos:{[sigTable;s]
            sub:`dt xasc select from sigTable where sym=s;
            sig:sub`sig;
            dsig:1 _ deltas sig;
            valid:dsig where not null dsig;
            lvl:sig where not null sig;
            $[(0 = count valid) or 0 = count lvl; 0n;
              $[0 = avg abs lvl; 0n; (avg abs valid) % avg abs lvl]]
            }[sigTable] each syms;
        avg tos where not null tos
        ];
        [
        sig:(`dt xasc sigTable)`sig;
        dsig:1 _ deltas sig;
        valid:dsig where not null dsig;
        lvl:sig where not null sig;
        $[(0 = count valid) or 0 = count lvl; 0n;
          $[0 = avg abs lvl; 0n; (avg abs valid) % avg abs lvl]]
        ]
    ]}

// -----------------------------------------------------------------------------
// EVALUATION HARNESS
// -----------------------------------------------------------------------------

// Single evaluation of an alpha with given params
evalAlpha:{[alphaName;params;data;cfg]
    cfg:mergeCfg cfg;
    if[not alphaName in key registry; '"alpha not found: ",string alphaName];
    alpha:getAlpha alphaName;
    // Generate signal
    sigTable:alpha[`fn][data;params];
    // Forward-fill signals within each sym (or single series)
    hasSym:`sym in cols sigTable;
    sigTable:$[hasSym;
        raze {[t;s] `dt xasc update sig:fills sig from select from t where sym=s}[sigTable] each asc distinct sigTable`sym;
        `dt xasc update sig:fills sig from sigTable
    ];
    // Dispatch based on type
    $[alpha[`type]~`cs;
        [
        // Cross-sectional evaluation
        ict:csIC[sigTable;data;1;cfg];
        ir:icIR ict;
        decay:icDecayProfile[sigTable;data;cfg`horizons;cfg];
        lsSharpe:longShortSharpe[sigTable;data;1;cfg];
        to:signalTurnover sigTable;
        `atype`sigTable`params`csIC`icIR`icDecay`longShortSharpe`turnover!(
            `cs;sigTable;params;ict;ir;decay;lsSharpe;to)
        ];
        [
        // Time-series evaluation
        sharpe:tsSharpe[sigTable;data;cfg];
        hr:tsHitRate[sigTable;data;cfg];
        to:signalTurnover sigTable;
        `atype`sigTable`params`tsSharpe`tsHitRate`turnover!(
            `ts;sigTable;params;sharpe;hr;to)
        ]
    ]}

// K-fold temporal cross-validation
// Split sorted unique dates into K non-overlapping folds.
// For each fold: generate signal on all data (alpha is causal),
// but measure metrics only on the fold's date range.
evalFolds:{[alphaName;params;data;nFolds;cfg]
    cfg:mergeCfg cfg;
    alpha:getAlpha alphaName;
    dates:asc distinct data`dt;
    nDates:count dates;
    foldSize:nDates div nFolds;
    // Assign fold boundaries
    foldBounds:{[dates;nFolds;foldSize;i]
        startIdx:i * foldSize;
        endIdx:$[i = nFolds - 1; count dates; (i + 1) * foldSize];
        (dates startIdx; dates endIdx - 1)
        }[dates;nFolds;foldSize] each til nFolds;
    // Generate signal on full data (alpha is causal/expanding)
    sigTable:alpha[`fn][data;params];
    hasSym:`sym in cols sigTable;
    sigTable:$[hasSym;
        raze {[t;s] `dt xasc update sig:fills sig from select from t where sym=s}[sigTable] each asc distinct sigTable`sym;
        `dt xasc update sig:fills sig from sigTable
    ];
    // Evaluate each fold (restrict metrics to fold date range)
    foldMetrics:{[sigTable;data;alpha;cfg;foldBound]
        startDt:foldBound 0; endDt:foldBound 1;
        foldSig:select from sigTable where dt >= startDt, dt <= endDt;
        foldData:select from data where dt >= startDt, dt <= endDt;
        if[0 = count foldSig; :`atype`turnover!(alpha`type;0n)];
        to:signalTurnover foldSig;
        $[alpha[`type]~`cs;
            [
            ict:csIC[foldSig;foldData;1;cfg];
            ir:icIR ict;
            `atype`icIR`turnover!(`cs;ir;to)
            ];
            [
            sharpe:tsSharpe[foldSig;foldData;cfg];
            `atype`tsSharpe`turnover!(`ts;sharpe;to)
            ]
        ]
        }[sigTable;data;alpha;cfg] each foldBounds;
    `alphaName`atype`params`foldMetrics`nFolds!(alphaName;alpha`type;params;foldMetrics;nFolds)}

// -----------------------------------------------------------------------------
// OBJECTIVE FUNCTION
// -----------------------------------------------------------------------------

// Scalar score from fold evaluation (dispatches per alpha type)
// CS: IC_IR - lamTO * turnover
// TS: Sharpe - lamTO * turnover
objective:{[foldResult;cfg]
    cfg:mergeCfg cfg;
    fm:foldResult`foldMetrics;
    typ:foldResult`atype;
    // Extract per-fold scores
    scores:{[typ;lamTO;m]
        primary:$[typ~`cs; m`icIR; m`tsSharpe];
        to:m`turnover;
        primary:$[null primary; neg 1e10; primary];
        to:$[null to; 0f; to];
        primary - (lamTO * to)
        }[typ;cfg`lamTO] each fm;
    scores:scores where not null scores;
    if[0 = count scores; :neg 1e10];
    // Aggregate across folds
    $[cfg[`foldAgg]~`conservative; min scores;
      cfg[`foldAgg]~`mean; avg scores;
      cfg[`foldAgg]~`penalized; (avg scores) - dev scores;
      avg scores]}

// -----------------------------------------------------------------------------
// PARAMETER NORMALIZATION
// -----------------------------------------------------------------------------

// Normalize params dict to [0,1] vector based on bounds
normalizeParams:{[params;bounds]
    pnames:key bounds;
    {[params;bounds;pn]
        v:params pn;
        lo:first bounds pn;
        hi:last bounds pn;
        $[hi = lo; 0.5; (v - lo) % hi - lo]
        }[params;bounds] each pnames}

// Denormalize [0,1] vector back to params dict
denormalizeParams:{[normVec;bounds]
    pnames:key bounds;
    bvals:value bounds;
    vals:{[normVec;i;bv]
        lo:first bv;
        hi:last bv;
        v:lo + (normVec[i] * (hi - lo));
        lo | hi & v  // clamp to bounds
        }[normVec] .' flip (til count pnames; bvals);
    pnames!vals}

// -----------------------------------------------------------------------------
// LATIN HYPERCUBE SAMPLING
// -----------------------------------------------------------------------------

// Latin Hypercube Sample: n points in d dimensions, each in [0,1]
lhs:{[n;d]
    if[n < 1; :()];
    // For each dimension, create stratified random samples
    {[n;di]
        perm:neg[n]?n;  // random permutation
        ((perm + n?1f) % n)  // stratified: (perm + U[0,1)) / n
        }[n] each til d}

// LHS scaled to parameter bounds, returns list of param dicts
lhsScaled:{[n;bounds]
    d:count bounds;
    pnames:key bounds;
    raw:lhs[n;d];  // d x n matrix
    // Transpose to n x d, then scale each point
    points:flip raw;
    {[pnames;bounds;pt]
        vals:{[bounds;pn;v]
            lo:first bounds pn;
            hi:last bounds pn;
            lo + (v * (hi - lo))
            }[bounds] .' flip (pnames;pt);
        pnames!vals
        }[pnames;bounds] each points}

// -----------------------------------------------------------------------------
// GAUSSIAN PROCESS
// -----------------------------------------------------------------------------

// Auto-select RBF gamma from median pairwise distance
medianHeuristic:{[X]
    if[1 >= count X; :1f];
    dists:.kdbtools.distSq[X;X];
    // Get upper triangle distances (exclude diagonal zeros)
    n:count X;
    upTri:raze {[dists;n;i] dists[i] (i + 1) + til (n - i) - 1}[dists;n] each til n - 1;
    if[0 = count upTri; :1f];
    md:med upTri;
    $[md < 1e-12; 1f; reciprocal 2 * md]}

// Fit GP model
// X: list of d-vectors (n observations), y: n-vector of objectives
// kfn: kernel function (e.g., kRBF[gamma]), noise: observation noise
gpFit:{[X;y;kfn;noise]
    n:count y;
    K:kfn[X;X];
    Kn:K + noise * .kdbtools.eye n;
    Kinv:.kdbtools.minv Kn;
    alpha:.kdbtools.mm[Kinv;y];
    `X`y`kfn`noise`Kinv`alpha!(X;y;kfn;noise;Kinv;alpha)}

// GP predictive mean and variance at new points
// Returns dict `mu`var
gpPredict:{[model;Xnew]
    if[0 = count Xnew; :`mu`var!(();())];
    Ks:model[`kfn][model`X;Xnew];  // n_train x n_new
    // Mean: Ks' @ alpha
    mu:{[Ks;alpha;j] sum Ks[;j] * alpha}[Ks;model`alpha] each til count Xnew;
    // Variance: k(x*,x*) - Ks' Kinv Ks (diagonal only)
    Kss:model[`kfn][Xnew;Xnew];  // n_new x n_new (only need diagonal)
    KsKinv:.kdbtools.mm[flip Ks;model`Kinv];  // n_new x n_train
    // diag(Ks' Kinv Ks) = sum_j KsKinv[i;j] * Ks[j;i]
    varRed:{[KsKinv;Ks;i] sum KsKinv[i] * Ks[;i]}[KsKinv;Ks] each til count Xnew;
    diagKss:{[Kss;i] Kss[i;i]}[Kss] each til count Xnew;
    variance:0f | diagKss - varRed;  // clamp to non-negative
    `mu`var!(mu;variance)}

// Expected Improvement acquisition function
expectedImprovement:{[model;Xnew;bestY]
    pred:gpPredict[model;Xnew];
    mu:pred`mu;
    sigma:sqrt pred`var;
    // EI = (mu - bestY) * Phi(z) + sigma * phi(z) where z = (mu - bestY) / sigma
    {[mu1;sig1;bestY]
        if[sig1 < 1e-10; :0f];
        z:(mu1 - bestY) % sig1;
        ((mu1 - bestY) * normCDF z) + sig1 * normPDF z
        }[;; bestY] .' flip (mu; sigma)}

// -----------------------------------------------------------------------------
// BAYESIAN OPTIMIZATION LOOP
// -----------------------------------------------------------------------------

bayesOpt:{[alphaName;data;cfg]
    cfg:mergeCfg cfg;
    alpha:getAlpha alphaName;
    if[not alphaName in key registry; '"alpha not found: ",string alphaName];
    bounds:alpha`bounds;
    nInit:cfg`nInit;
    nIter:cfg`nIter;
    nCand:cfg`nCandidates;
    nFolds:cfg`nFolds;
    t0:.z.P;

    // 1. Generate initial design via Latin Hypercube
    initParams:lhsScaled[nInit;bounds];

    // 2. Evaluate initial points
    -1 "  [bayesOpt] Evaluating ",string[nInit]," initial points for ",string alphaName;
    initObs:{[alphaName;data;nFolds;cfg;bounds;p]
        foldRes:evalFolds[alphaName;p;data;nFolds;cfg];
        obj:objective[foldRes;cfg];
        normP:normalizeParams[p;bounds];
        (normP;obj)
        }[alphaName;data;nFolds;cfg;bounds] each initParams;
    Xobs:initObs[;0];
    yobs:initObs[;1];

    // 3. Bayesian optimization loop
    -1 "  [bayesOpt] Starting ",string[nIter]," optimization iterations";
    d:alpha`nParams;
    // Pack loop context into a dict to stay within Q's 8-param limit
    ctx:`alphaName`data`nFolds`cfg`bounds`d`nCand`nIter!(alphaName;data;nFolds;cfg;bounds;d;nCand;nIter);
    state:`Xobs`yobs!(Xobs;yobs);
    state:{[ctx;state;iter]
        Xobs:state`Xobs;
        yobs:state`yobs;
        d:ctx`d; nCand:ctx`nCand; bounds:ctx`bounds;
        // Fit GP with adaptive kernel
        gamma:medianHeuristic Xobs;
        kfn:.kdbtools.kRBF gamma;
        model:gpFit[Xobs;yobs;kfn;ctx[`cfg]`noise];
        bestY:max yobs;
        // Generate random candidates in [0,1]^d
        candidates:{[d;n] {[d;i] d?1f}[d] each til n}[d;nCand];
        // Compute EI
        ei:expectedImprovement[model;candidates;bestY];
        bestCandIdx:ei?max ei;
        bestCandNorm:candidates bestCandIdx;
        // Denormalize and evaluate
        bestCandParams:denormalizeParams[bestCandNorm;bounds];
        foldRes:evalFolds[ctx`alphaName;bestCandParams;ctx`data;ctx`nFolds;ctx`cfg];
        obj:objective[foldRes;ctx`cfg];
        // Append observation
        Xobs:Xobs,enlist bestCandNorm;
        yobs:yobs,obj;
        // Log progress
        if[0 = (iter + 1) mod 10;
            -1 "  [bayesOpt] iter ",string[iter + 1],"/",string[ctx`nIter],
               " | best: ",string[max yobs],
               " | current: ",string obj];
        `Xobs`yobs!(Xobs;yobs)
        }[ctx] over enlist[state],til nIter;

    Xobs:state`Xobs;
    yobs:state`yobs;

    // 4. Extract best
    bestIdx:yobs?max yobs;
    bestNorm:Xobs bestIdx;
    bestParams:denormalizeParams[bestNorm;bounds];
    bestObj:yobs bestIdx;

    // Final GP model for diagnostics
    gamma:medianHeuristic Xobs;
    kfn:.kdbtools.kRBF gamma;
    gpModel:gpFit[Xobs;yobs;kfn;cfg`noise];

    elapsed:(`second$.z.P - t0);
    -1 "  [bayesOpt] Done. Best objective: ",string[bestObj]," | Time: ",string elapsed;

    `alphaName`alphaType`bestParams`bestObj`allXobs`allYobs`gpModel`nInit`nIter`elapsed!(
        alphaName;alpha`type;bestParams;bestObj;Xobs;yobs;gpModel;nInit;nIter;elapsed)}

// -----------------------------------------------------------------------------
// FULL PIPELINE & REPORTING
// -----------------------------------------------------------------------------

// Run Bayesian optimization for a list of alphas
run:{[alphaNames;data;cfg]
    cfg:mergeCfg cfg;
    -1 "=== .alphalab.run: optimizing ",string[count alphaNames]," alphas ===";
    results:{[data;cfg;nm]
        -1 "";
        -1 "--- Optimizing: ",string nm," ---";
        bayesOpt[nm;data;cfg]
        }[data;cfg] each alphaNames;
    -1 "";
    -1 "=== All alphas optimized ===";
    // Generate best signals for each alpha
    bestSignals:{[data;r]
        alpha:getAlpha r`alphaName;
        alpha[`fn][data;r`bestParams]
        }[data] each results;
    `results`bestSignals!(results;bestSignals)}

// Display report for a single optimization result
report:{[result]
    -1 "";
    -1 "=============================================================================";
    -1 "  ALPHA OPTIMIZATION REPORT: ",string result`alphaName;
    -1 "=============================================================================";
    -1 "";
    -1 "  Type:       ",string result`alphaType;
    -1 "  Best Obj:   ",string result`bestObj;
    -1 "  Iterations: ",string[result`nInit]," init + ",string[result`nIter]," opt";
    -1 "  Elapsed:    ",string result`elapsed;
    -1 "";
    -1 "  Best Parameters:";
    bp:result`bestParams;
    {[bp;pn] -1 "    ",string[pn]," = ",string bp pn}[bp] each key bp;
    -1 "";
    // Objective trajectory
    yobs:result`allYobs;
    runBest:{max x#y}[;yobs] each 1 + til count yobs;
    -1 "  Objective trajectory (running best):";
    milestones:(ceiling (count runBest) * 0.25 0.5 0.75 1.0) - 1;
    {[rb;m] -1 "    iter ",string[1 + m],": ",string rb m}[runBest] each milestones;
    -1 "";
    // Full evaluation of best params
    -1 "  Full evaluation of best params:";
    alpha:getAlpha result`alphaName;
    // Just show the best objective breakdown
    -1 "    Final objective score: ",string result`bestObj;
    -1 ""}

// -----------------------------------------------------------------------------
// DIAGNOSTICS
// -----------------------------------------------------------------------------

// GP posterior mean/std on a grid (1D or 2D)
// resolution: number of grid points per dimension
paramSurface:{[result;resolution]
    bounds:(getAlpha result`alphaName)`bounds;
    pnames:key bounds;
    d:count pnames;
    model:result`gpModel;
    if[d = 1;
        // 1D grid
        grid:til[resolution] % resolution - 1;
        grid:grid,\:();  // make each element a 1-element list
        pred:gpPredict[model;grid];
        params:{[bounds;pnames;g] lo:first bounds pnames 0; hi:last bounds pnames 0; lo + (g[0] * (hi - lo))}[bounds;pnames] each grid;
        :([] param:params; mu:pred`mu; std:sqrt pred`var)
    ];
    if[d = 2;
        // 2D grid (fix other dims at best values)
        g1:til[resolution] % resolution - 1;
        g2:g1;
        bestNorm:normalizeParams[result`bestParams;bounds];
        grid:raze {[g2;bestNorm;i] {[bestNorm;i;j] @[bestNorm;0;:;i]; @[bestNorm;1;:;j]}[bestNorm;i] each g2}[g2;bestNorm] each g1;
        pred:gpPredict[model;grid];
        p1Vals:{[bounds;pnames;g] lo:first bounds pnames 0; hi:last bounds pnames 0; lo + (g[0] * (hi - lo))}[bounds;pnames] each grid;
        p2Vals:{[bounds;pnames;g] lo:first bounds pnames 1; hi:last bounds pnames 1; lo + (g[1] * (hi - lo))}[bounds;pnames] each grid;
        :([] p1:p1Vals; p2:p2Vals; mu:pred`mu; std:sqrt pred`var)
    ];
    // d > 2: project onto first two dims, fix rest at best
    -1 "  [paramSurface] Only 1D and 2D supported (d=",string[d],")";
    ([]mu:();std:())}

// Parameter sensitivity: perturb each param ±X%, measure objective change
paramSensitivity:{[alphaName;bestParams;data;perturbPct;cfg]
    cfg:mergeCfg cfg;
    alpha:getAlpha alphaName;
    bounds:alpha`bounds;
    pnames:key bounds;
    // Baseline
    baseFold:evalFolds[alphaName;bestParams;data;cfg`nFolds;cfg];
    baseObj:objective[baseFold;cfg];
    // Perturb each param
    sensRows:{[alphaName;bestParams;data;cfg;bounds;baseObj;perturbPct;pn]
        lo:first bounds pn;
        hi:last bounds pn;
        base:bestParams pn;
        delta:(hi - lo) * perturbPct;
        // Up
        upParams:@[bestParams;pn;:;lo | hi & base + delta];
        upFold:evalFolds[alphaName;upParams;data;cfg`nFolds;cfg];
        upObj:objective[upFold;cfg];
        // Down
        downParams:@[bestParams;pn;:;lo | hi & base - delta];
        downFold:evalFolds[alphaName;downParams;data;cfg`nFolds;cfg];
        downObj:objective[downFold;cfg];
        sens:(upObj - downObj) % 2 * delta;
        `param`baseVal`baseObj`upObj`downObj`sensitivity!(pn;base;baseObj;upObj;downObj;sens)
        }[alphaName;bestParams;data;cfg;bounds;baseObj;perturbPct] each pnames;
    sensRows}

// -----------------------------------------------------------------------------
// HELP & USAGE
// -----------------------------------------------------------------------------

help:{[]
    -1 "";
    -1 "=== .alphalab ALPHA RESEARCH PIPELINE v0.1.0 ===";
    -1 "";
    -1 "ALPHA REGISTRY:";
    -1 "  defineAlpha[name;type;fn;paramBounds] - register alpha (`ts or `cs)";
    -1 "  listAlphas[]                          - table of registered alphas";
    -1 "  removeAlpha[name]                     - unregister alpha";
    -1 "  getAlpha[name]                        - lookup alpha definition";
    -1 "";
    -1 "CROSS-SECTIONAL EVALUATION (type=`cs):";
    -1 "  csIC[sigTable;data;N;cfg]             - cross-sectional IC per date";
    -1 "  icIR[icTable]                         - IC information ratio";
    -1 "  icDecayProfile[sig;data;horizons;cfg] - IC at multiple horizons";
    -1 "  longShortSharpe[sig;data;N;cfg]       - long/short quintile Sharpe";
    -1 "";
    -1 "TIME-SERIES EVALUATION (type=`ts):";
    -1 "  tsSharpe[sigTable;data;cfg]           - signal-weighted return Sharpe";
    -1 "  tsRollingIC[sigTable;data;window;cfg] - rolling IC (time-series)";
    -1 "  tsHitRate[sigTable;data;cfg]          - directional hit rate";
    -1 "  tsPnLSeries[sigTable;data;cfg]        - full PnL series";
    -1 "";
    -1 "SIGNAL TURNOVER:";
    -1 "  signalTurnover[sigTable]              - mean |delta sig| / mean |sig|";
    -1 "";
    -1 "ALPHA EVALUATION SUITE:";
    -1 "  alphaEval[t;cfg]                      - unified perf report";
    -1 "    Returns: sharpe, winsorizedSharpe, sortino, calmar, annReturn, annVol,";
    -1 "      maxDD, skew, kurtosis, winRate, profitFactor, avgWin, avgLoss,";
    -1 "      winLossRatio, cvar95, medianMonthlySharpe, medianMonthlyHitRate,";
    -1 "      monthlyTable, turnover, ic, icIR, icHitRate, retTstat, retPval,";
    -1 "      icTstat, icPval, minTRL, retAutoCorr, icDecay";
    -1 "  monthlyBreakdown[dts;rets]            - per-month Sharpe, hit rate, return, vol";
    -1 "  cfg keys: dtCol symCol sigCol pnlCol  - column name overrides";
    -1 "            rf periods winsorizePct      - (0; 252; 0.01)";
    -1 "";
    -1 "EVALUATION:";
    -1 "  evalAlpha[name;params;data;cfg]       - single evaluation (type-dispatched)";
    -1 "  evalFolds[name;params;data;nFolds;cfg]- K-fold temporal CV";
    -1 "  objective[foldResult;cfg]             - scalar objective from fold eval";
    -1 "";
    -1 "BAYESIAN OPTIMIZATION:";
    -1 "  bayesOpt[alphaName;data;cfg]          - optimize alpha parameters";
    -1 "  run[alphaNames;data;cfg]              - optimize multiple alphas";
    -1 "  report[result]                        - display optimization report";
    -1 "";
    -1 "GAUSSIAN PROCESS:";
    -1 "  gpFit[X;y;kfn;noise]                 - fit GP model";
    -1 "  gpPredict[model;Xnew]                - predictive mean + variance";
    -1 "  expectedImprovement[model;Xnew;bestY]- EI acquisition function";
    -1 "  medianHeuristic[X]                   - auto-select RBF gamma";
    -1 "";
    -1 "DIAGNOSTICS:";
    -1 "  paramSurface[result;resolution]       - GP posterior on 1D/2D grid";
    -1 "  paramSensitivity[name;params;data;pct;cfg] - perturbation analysis";
    -1 "";
    -1 "HELPERS:";
    -1 "  normPDF[x]         normCDF[x]         - standard normal PDF/CDF";
    -1 "  spearmanCorCS[x;y]                    - Spearman rank correlation";
    -1 "  lhs[n;d]           lhsScaled[n;bounds]- Latin Hypercube sampling";
    -1 "  normalizeParams[params;bounds]        - scale to [0,1]";
    -1 "  denormalizeParams[normVec;bounds]     - scale back to original";
    -1 "";
    -1 "CONFIGURATION (pass as cfg dict, or use defaults):";
    -1 "  nFolds:5  nInit:15  nIter:50  nCandidates:500  lamTO:0.1";
    -1 "  foldAgg:`conservative  noise:1e-4  retCol:`ret  icWindow:60";
    -1 "  horizons:1 2 5 10 20";
    -1 ""}

usage:help

// -----------------------------------------------------------------------------
// EXAMPLE
// -----------------------------------------------------------------------------

exampleData:{[]
    system "S 42";
    nSym:20; nDays:500;
    syms:`$"SYM",/: string til nSym;
    dates:("D"$"2020.01.01") + til nDays;
    // Generate per-symbol return data with a cross-sectional factor
    rows:raze {[syms;dates;nDays;i]
        s:syms i;
        // Each symbol has slightly different mean/vol
        mu:(neg[0.001] + 0.002 * i % count syms);
        vol:0.01 + (0.005 * i % count syms);
        rets:(vol * nDays?1f) + mu % 252;
        // Add common factor (cross-sectional structure)
        ([] dt:dates; sym:nDays#s; ret:rets; price:100 * prds 1 + rets; volume:1000 + nDays?9000f)
        }[syms;dates;nDays] each til nSym;
    `dt`sym xasc rows}

// -----------------------------------------------------------------------------
// ALPHA EVALUATION SUITE
// -----------------------------------------------------------------------------

// Sample skewness
alphaSkew:{[x]
    n:count x; mu:avg x; s:dev x;
    z:(x - mu) % s;
    ((n % (n - 1)) % (n - 2)) * sum z xexp 3}

// Excess kurtosis
alphaKurt:{[x]
    n:count x; mu:avg x; s:dev x;
    z:(x - mu) % s;
    s4:sum z xexp 4;
    adj:((n * (n + 1)) % ((n - 1) * (n - 2) * (n - 3))) * s4;
    adj - (3f * ((n - 1) * (n - 1)) % ((n - 2) * (n - 3)))}

// Cross-sectional IC from position-level table: rank-correlate sig vs pnl per day
// Returns ([] dt; ic) — one IC per day with >= 2 non-null positions
alphaIC:{[t;cDt;cSym;cSig;cPnl]
    dates:asc distinct t cDt;
    {[t;cDt;cSig;cPnl;d]
        sub:?[t;enlist(=;cDt;d);0b;(cSig,cPnl)!(cSig,cPnl)];
        s:0f^sub cSig; p:0f^sub cPnl;
        valid:where (not null s) & not null p;
        $[2 > count valid; 0n; cor[iasc iasc s valid; iasc iasc p valid]]
        }[t;cDt;cSig;cPnl] each dates}

// CS IC at lagged horizons: for each lag, shift pnl forward by lag per sym, then daily CS IC
alphaICDecay:{[t;cDt;cSym;cSig;cPnl;lags]
    syms:asc distinct t cSym;
    cs:(cDt,cSig,cPnl)!(cDt,cSig,cPnl);
    {[t;cDt;cSym;cSig;cPnl;cs;syms;lag]
        // Build shifted table: sig at t, fwdPnl = pnl at t+lag
        shifted:raze {[t;cDt;cSym;cSig;cPnl;cs;lag;s]
            sub:cDt xasc ?[t;enlist(=;cSym;enlist s);0b;cs];
            fp:((lag # 0n), neg[lag] _ sub cPnl);
            ([] dt:sub cDt; sym:(count sub cDt)#s; sig:sub cSig; fwdPnl:fp)
            }[t;cDt;cSym;cSig;cPnl;cs;lag] each syms;
        dates:asc distinct shifted`dt;
        ics:{[shifted;d]
            sub:select from shifted where dt=d, not null sig, not null fwdPnl;
            $[2 > count sub; 0n; cor[iasc iasc sub`sig; iasc iasc sub`fwdPnl]]
            }[shifted] each dates;
        v:ics where not null ics;
        mic:$[0 < count v; avg v; 0n];
        ir:$[(1 < count v) and (dev v) > 0; (avg v) % dev v; 0n];
        `lag`meanIC`icIR!(lag;mic;ir)
        }[t;cDt;cSym;cSig;cPnl;cs;syms] each lags}

// Time-series IC: per symbol, correlate sig(t) vs pnl(t) over time, then avg across syms
// Measures whether signal predicts each position's own return directionally
alphaTSIC:{[t;cDt;cSym;cSig;cPnl]
    syms:asc distinct t cSym;
    cs:(cDt,cSig,cPnl)!(cDt,cSig,cPnl);
    ics:{[t;cDt;cSym;cSig;cPnl;cs;s]
        sub:cDt xasc ?[t;enlist(=;cSym;enlist s);0b;cs];
        sg:0f^sub cSig; p:0f^sub cPnl;
        valid:where (not null sg) & not null p;
        $[5 > count valid; 0n; cor[sg valid; p valid]]
        }[t;cDt;cSym;cSig;cPnl;cs] each syms;
    v:ics where not null ics;
    `tsIC`tsICBySymAvg`tsICBySymStd`tsICBySym!(
        $[0 < count v; avg v; 0n];
        $[0 < count v; avg v; 0n];
        $[1 < count v; dev v; 0n];
        syms!ics)}

// Time-series IC decay: per symbol, correlate sig(t) vs pnl(t+lag), avg across syms
alphaTSICDecay:{[t;cDt;cSym;cSig;cPnl;lags]
    syms:asc distinct t cSym;
    cs:(cDt,cSig,cPnl)!(cDt,cSig,cPnl);
    {[t;cDt;cSym;cSig;cPnl;cs;syms;lag]
        ics:{[t;cDt;cSym;cSig;cPnl;cs;lag;s]
            sub:cDt xasc ?[t;enlist(=;cSym;enlist s);0b;cs];
            sg:sub cSig; p:sub cPnl;
            sg2:neg[lag] _ sg;
            p2:lag _ p;
            valid:where (not null 0f^sg2) & not null 0f^p2;
            $[5 > count valid; 0n; cor[sg2 valid; p2 valid]]
            }[t;cDt;cSym;cSig;cPnl;cs;lag] each syms;
        v:ics where not null ics;
        mic:$[0 < count v; avg v; 0n];
        ir:$[(1 < count v) and (dev v) > 0; (avg v) % dev v; 0n];
        `lag`meanTSIC`tsICIR!(lag;mic;ir)
        }[t;cDt;cSym;cSig;cPnl;cs;syms] each lags}

// Lag-1 autocorrelation
autoCorr1:{[x] v:x where not null x; $[2 > count v; 0n; cor[neg[1] _ v; 1 _ v]]}

// Monthly breakdown: Sharpe, hit rate, return, vol, nDays per month
monthlyBreakdown:{[dts;rets]
    months:`month$dts;
    uMonths:asc distinct months;
    {[dts;rets;months;m]
        idx:where months = m;
        r:rets idx;
        nz:r where r <> 0;
        nd:count r;
        nnz:count nz;
        s:dev nz;
        sh:$[(nnz > 1) and s > 0; (avg[nz] % s) * sqrt 252; 0n];
        hr:$[nnz > 0; (sum nz > 0) % nnz; 0n];
        `month`sharpe`hitRate`ret`vol`nDays!(m;sh;hr;sum r;s;nd)
        }[dts;rets;months] each uMonths}

// Main alpha evaluation function
// t: table with per-position daily rows
// cfg: optional dict:
//   `dtCol   - date column (default `dt)
//   `symCol  - symbol column (default `sym)
//   `sigCol  - signal column (default `sig)
//   `pnlCol  - P&L column (default `pnl)
//   `rf      - risk-free rate per period (default 0)
//   `periods - annualization factor (default 252)
//   `winsorizePct - winsorize percentile (default 0.01)
alphaEval:{[t;cfg]
    rf:$[`rf in key cfg; cfg`rf; 0f];
    periods:$[`periods in key cfg; cfg`periods; 252];
    wpct:$[`winsorizePct in key cfg; cfg`winsorizePct; 0.01];
    cDt:$[`dtCol in key cfg; cfg`dtCol; `dt];
    cSym:$[`symCol in key cfg; cfg`symCol; `sym];
    cSig:$[`sigCol in key cfg; cfg`sigCol; `sig];
    cPnl:$[`pnlCol in key cfg; cfg`pnlCol; `pnl];

    // Aggregate daily portfolio return, fill nulls with 0
    daily:0!?[t;();(enlist cDt)!enlist cDt;(enlist`ret)!enlist(sum;cPnl)];
    dts:daily cDt;
    r:0f^daily`ret;
    n:count r;

    // Non-zero returns — zero days are excluded entirely
    nz:r where r <> 0;
    nnz:count nz;

    // Annualized return and vol (non-zero days only)
    annRet:periods * avg nz;
    annVol:(sqrt periods) * dev nz;

    // Sharpe
    sharpe:$[annVol > 0; (annRet - (periods * rf)) % annVol; 0n];

    // Winsorized Sharpe: clip at wpct/1-wpct percentiles, then Sharpe
    lo:(asc nz) @ `long$wpct * nnz;
    hi:(asc nz) @ `long$(1 - wpct) * nnz;
    wRets:lo | nz & hi;
    wAnnRet:periods * avg wRets;
    wAnnVol:(sqrt periods) * dev wRets;
    winsorizedSharpe:$[wAnnVol > 0; (wAnnRet - (periods * rf)) % wAnnVol; 0n];

    // Sortino (non-zero days)
    sortino:.kdbtools.sortino[periods;rf;nz];

    // Max drawdown on cumulative returns (all days, preserves timeline)
    cumRets:sums r;
    maxDD:min cumRets - maxs cumRets;

    // Calmar
    calmar:$[(maxDD < 0) and (not null maxDD); neg annRet % maxDD; 0n];

    // CVaR 95% (non-zero days)
    cutoff:(asc nz) @ `long$0.05 * nnz;
    cvar95:neg avg nz where nz <= cutoff;

    // Distribution (non-zero days)
    skw:alphaSkew nz;
    krt:alphaKurt nz;

    // Hit/loss (non-zero days)
    wins:nz where nz > 0;
    losses:nz where nz < 0;
    winRate:$[nnz > 0; (count wins) % nnz; 0n];
    avgWin:$[0 < count wins; avg wins; 0n];
    avgLoss:$[0 < count losses; avg losses; 0n];
    profitFactor:$[(0 < count losses) and (0 < count wins); (sum wins) % neg sum losses; 0n];
    winLossRatio:$[(not null avgWin) and (not null avgLoss) and avgLoss < 0; avgWin % neg avgLoss; 0n];

    // Monthly breakdown
    mt:monthlyBreakdown[dts;r];
    mSh:mt[;`sharpe]; mHr:mt[;`hitRate]; mRet:mt[;`ret];
    // Filter to months with non-zero total return, drop nulls
    shValid:mSh where (mRet <> 0) and not null mSh;
    hrValid:mHr where (mRet <> 0) and not null mHr;
    medSh:$[0 < count shValid; med shValid; 0n];
    medHr:$[0 < count hrValid; med hrValid; 0n];

    // Turnover from sig column
    syms:asc distinct t cSym;
    tos:{[t;cDt;cSym;cSig;s]
        sub:cDt xasc ?[t;enlist(=;cSym;enlist s);0b;(cDt,cSig)!(cDt,cSig)];
        sig:sub cSig;
        dsig:1 _ deltas sig;
        valid:dsig where not null dsig;
        lvl:sig where not null sig;
        $[(0 < count valid) and (0 < avg abs lvl);
            (avg abs valid) % avg abs lvl;
            0n]
        }[t;cDt;cSym;cSig] each syms;
    turnover:avg tos where not null tos;

    // IC metrics (cross-sectional: rank-correlate sig vs pnl per day)
    nSym:count syms;
    dailyICs:$[nSym >= 2; alphaIC[t;cDt;cSym;cSig;cPnl]; n # 0n];
    icValid:dailyICs where not null dailyICs;
    nICDays:count icValid;
    icMean:$[nICDays > 0; avg icValid; 0n];
    icStd:$[nICDays > 1; dev icValid; 0n];
    icIR:$[(nICDays > 1) and icStd > 0; icMean % icStd; 0n];
    icHitRate:$[nICDays > 0; (sum icValid > 0) % nICDays; 0n];

    // Signal vs noise: t-stats and p-values
    // t-stat on returns
    retTstat:$[nnz > 1; (avg nz) % (dev nz) % sqrt nnz; 0n];
    // t-stat on IC
    icTstat:$[(nICDays > 1) and icStd > 0; icMean % icStd % sqrt nICDays; 0n];
    // p-values (two-sided, normal approx for large n)
    retPval:$[not null retTstat; 2 * 1 - .kdbtools.normCDF abs retTstat; 0n];
    icPval:$[not null icTstat; 2 * 1 - .kdbtools.normCDF abs icTstat; 0n];

    // Minimum Track Record Length (MinTRL)
    // Days needed for observed Sharpe to be significant at 95% (z=1.96)
    sr:$[annVol > 0; (avg[nz] % dev nz); 0n];
    minTRL:$[(not null sr) and sr <> 0;
        `long$(1 + ((krt % 4) * sr * sr) - ((skw % 2) * sr)) * (1.96 * 1.96) % (sr * sr);
        0N];

    // Return autocorrelation (lag-1)
    retAutoCorr:autoCorr1 nz;

    // IC decay profile (cross-sectional)
    icDecay:$[nSym >= 2; alphaICDecay[t;cDt;cSym;cSig;cPnl;1 2 3 5 10]; ()];

    // Time-series IC: per sym, correlate sig(t) vs pnl(t), avg across syms
    tsicRes:alphaTSIC[t;cDt;cSym;cSig;cPnl];
    tsIC:tsicRes`tsIC;
    tsICBySym:tsicRes`tsICBySym;
    // t-stat on tsIC across syms
    tsICVals:(value tsICBySym) where not null value tsICBySym;
    nTSIC:count tsICVals;
    tsICTstat:$[(nTSIC > 1) and (dev tsICVals) > 0; (avg tsICVals) % (dev tsICVals) % sqrt nTSIC; 0n];
    tsICPval:$[not null tsICTstat; 2 * 1 - .kdbtools.normCDF abs tsICTstat; 0n];

    // Time-series IC decay
    tsICDecay:alphaTSICDecay[t;cDt;cSym;cSig;cPnl;1 2 3 5 10];

    // Build result dict
    (`sharpe`winsorizedSharpe`sortino`calmar`annReturn`annVol`maxDD,
     `skew`kurtosis,
     `winRate`profitFactor`avgWin`avgLoss`winLossRatio,
     `cvar95,
     `medianMonthlySharpe`medianMonthlyHitRate`monthlyTable,
     `turnover,
     `ic`icIR`icHitRate,
     `tsIC`tsICBySym`tsICTstat`tsICPval,
     `retTstat`retPval`icTstat`icPval`minTRL`retAutoCorr,
     `icDecay`tsICDecay,
     `nDays`nNonZeroDays`nPositions)!
    (sharpe;winsorizedSharpe;sortino;calmar;annRet;annVol;maxDD;
     skw;krt;
     winRate;profitFactor;avgWin;avgLoss;winLossRatio;
     cvar95;
     medSh;medHr;mt;
     turnover;
     icMean;icIR;icHitRate;
     tsIC;tsICBySym;tsICTstat;tsICPval;
     retTstat;retPval;icTstat;icPval;minTRL;retAutoCorr;
     icDecay;tsICDecay;
     n;nnz;count syms)}

// -----------------------------------------------------------------------------
// ALPHA REPORT — lightweight signal evaluation
// -----------------------------------------------------------------------------
// Evaluate a strategy's alpha. Performance metrics based on alpha column
// (the strategy's daily PnL), summed across symbols by date.
// All metrics computed on non-zero alpha days.
//
// t:   table with columns:
//   dt       - date/time column
//   ricRoot  - symbol identifier
//   sig      - position signal (used for IC and turnover)
//   alpha    - strategy alpha / daily PnL per symbol
//   pxDiff   - (optional) raw price difference for cross-sectional IC
//
// cfg: optional config dict:
//   `dtCol         - date column (default: `dt)
//   `symCol        - symbol column (default: `ricRoot)
//   `sigCol        - signal column for IC/turnover (default: `sig)
//   `retCol        - strategy alpha column for performance (default: `alpha)
//   `pxDiffCol     - raw price diff for cross-sectional IC (default: `pxDiff)
//   `topN          - top N days for PnL concentration (default: 5 10)
//   `winsorizePct  - winsorize percentile (default: 0.025)
//   `icLags        - forward horizons for IC decay (default: 1 2 5 10 20)
//   `episodeGap    - trading day gap to separate episodes (default: 5)
//   `vrQ           - VR lag for regime detection (default: 10)
//   `vrW           - VR rolling window for regime detection (default: 63)
//   `regimeCol     - optional pre-computed regime column (>0.5=trending)
//
// Returns: dict with:
//   n, ann_return, ann_vol, sharpe, sortino, win_sharpe,
//   hit_rate, monthly_hit_rate, profit_factor, payoff_ratio,
//   tail_ratio, return_skew, max_dd, max_dd_length, calmar,
//   max_consec_loss, pnl_topN, turnover, ic, ic_ir, tsIc, ic_decay,
//   episodes, regime

// --- alphaReport helpers (private) ---

arParseCfg_:{[t;cfg]
    cDt:$[99h=type cfg;$[`dtCol in key cfg;cfg`dtCol;`dt];`dt];
    cSym:$[99h=type cfg;$[`symCol in key cfg;cfg`symCol;`ricRoot];`ricRoot];
    cSig:$[99h=type cfg;$[`sigCol in key cfg;cfg`sigCol;`sig];`sig];
    cRet:$[99h=type cfg;$[`retCol in key cfg;cfg`retCol;`alpha];`alpha];
    cPx:$[99h=type cfg;$[`pxDiffCol in key cfg;cfg`pxDiffCol;`pxDiff];`pxDiff];
    topN:$[99h=type cfg;$[`topN in key cfg;cfg`topN;5 10];5 10];
    winPct:$[99h=type cfg;$[`winsorizePct in key cfg;cfg`winsorizePct;0.025];0.025];
    icLags:$[99h=type cfg;$[`icLags in key cfg;cfg`icLags;1 2 5 10 20];1 2 5 10 20];
    epGap:$[99h=type cfg;$[`episodeGap in key cfg;cfg`episodeGap;5];5];
    vrQ:$[99h=type cfg;$[`vrQ in key cfg;cfg`vrQ;10];10];
    vrW:$[99h=type cfg;$[`vrW in key cfg;cfg`vrW;63];63];
    cReg:$[99h=type cfg;$[`regimeCol in key cfg;cfg`regimeCol;`];`];
    hasReg:(cReg <> `) and cReg in cols t;
    `cDt`cSym`cSig`cRet`cPx`hasPx`topN`winPct`icLags`epGap`vrQ`vrW`cRegime`hasRegime!
        (cDt;cSym;cSig;cRet;cPx;cPx in cols t;topN;winPct;icLags;epGap;vrQ;vrW;cReg;hasReg)}

arExtract_:{[t;pcfg]
    grp:group t pcfg`cSym;
    sd:{[t;pcfg;idx]
        sub:pcfg[`cDt] xasc t idx;
        d:`dt`sig`alpha!(sub pcfg`cDt; "f"$sub pcfg`cSig; "f"$sub pcfg`cRet);
        if[pcfg`hasPx; d[`pxDiff]:"f"$sub pcfg`cPx];
        if[pcfg`hasRegime; d[`regime]:"f"$sub pcfg`cRegime];
        d
    }[t;pcfg;] each value grp;
    `grp`symData!(grp;sd)}

arPerf_:{[nzr;nzDts;n;winPct]
    annRet:252 * avg nzr;
    annVol:(sqrt 252f) * dev nzr;
    sharpe:$[annVol > 1e-10; annRet % annVol; 0n];
    downside:nzr where nzr < 0f;
    dsVol:$[0 < count downside; (sqrt 252f) * sqrt avg downside * downside; 0n];
    sortino:$[(not null dsVol) and dsVol > 1e-10; annRet % dsVol; 0n];
    sorted:asc nzr;
    loIdx:1 | `long$winPct * n;
    hiIdx:(n - 2) & `long$(1f - winPct) * n;
    wRet:(sorted loIdx) | nzr & sorted hiIdx;
    wVol:(sqrt 252f) * dev wRet;
    winSharpe:$[wVol > 1e-10; (252 * avg wRet) % wVol; 0n];
    wins:nzr where nzr > 0f;
    losses:nzr where nzr < 0f;
    hitRate:(count wins) % n;
    mPnl:0!select mp:sum x by m from ([] m:`month$nzDts; x:nzr);
    monthlyHitRate:avg (mPnl`mp) > 0f;
    `ann_return`ann_vol`sharpe`sortino`win_sharpe`hit_rate`monthly_hit_rate!
        (annRet;annVol;sharpe;sortino;winSharpe;hitRate;monthlyHitRate)}

arRisk_:{[r;annRet]
    cumPnl:sums r;
    peak:maxs cumPnl;
    dd:cumPnl - peak;
    maxDD:min dd;
    inDD:dd < neg 1e-10;
    ddLens:$[any inDD;
        [d:deltas "i"$inDD;
         sts:where d = 1i;
         eds:where d = -1i;
         if[inDD[0] and ((0 = count sts) or sts[0] > 0); sts:0,sts];
         if[(count sts) > count eds; eds:eds,count dd];
         $[0 < count sts; eds - sts; enlist 0]];
        enlist 0];
    maxDDLen:max ddLens;
    calmar:$[(maxDD < neg 1e-10) and not null maxDD; neg annRet % maxDD; 0n];
    `max_dd`max_dd_length`calmar!(maxDD;maxDDLen;calmar)}

arPayoff_:{[nzr;n;topN]
    sorted:asc nzr;
    wins:nzr where nzr > 0f;
    losses:nzr where nzr < 0f;
    profitFactor:$[(0 < count losses) and 0 < count wins;
        (sum wins) % neg sum losses; 0n];
    payoffRatio:$[(0 < count losses) and 0 < count wins;
        (avg wins) % neg avg losses; 0n];
    tailRatio:$[n > 20;
        [p95:sorted `long$0.95 * n;
         p05:sorted `long$0.05 * n;
         $[p05 < neg 1e-10; p95 % neg p05; 0n]];
        0n];
    mu:avg nzr; s:dev nzr;
    returnSkew:$[(n > 3) and s > 1e-10; avg ((nzr - mu) % s) xexp 3; 0n];
    maxConsecLoss:max {$[y;x+1;0]}\[0;"i"$nzr < 0f];
    totalPnl:sum nzr;
    pnlConc:{[nzr;tp;k]
        if[(k > count nzr) or (abs tp) < 1e-10; :0n];
        (sum k # desc nzr) % tp
    }[nzr;totalPnl;] each topN;
    pnlConcKeys:`$"pnl_top" ,/: string topN;
    base:`profit_factor`payoff_ratio`tail_ratio`return_skew`max_consec_loss!
        (profitFactor;payoffRatio;tailRatio;returnSkew;maxConsecLoss);
    base,pnlConcKeys!pnlConc}

arEpisodes_:{[nzr;nzDts;n;topN;epGap]
    maxTopN:n & max topN;
    topRank:iasc neg nzr;
    topIdx:asc topRank til maxTopN;
    topPnls:nzr topIdx;
    epBreaks:where epGap < 1 _ deltas topIdx;
    epStarts:0,1 + epBreaks;
    epEnds:(1 + epBreaks),maxTopN;
    epSlices:epStarts,'epEnds;
    nEp:count epSlices;
    epPnls:{[nzr;topIdx;se] sum nzr topIdx se[0] + til se[1] - se 0}[nzr;topIdx;] each epSlices;
    totalTopPnl:1e-10 | sum topPnls;
    `n_episodes`largest_ep_pct`avg_ep_days!(nEp; (max epPnls) % totalTopPnl; maxTopN % 1 | nEp)}

arTurnover_:{[symData]
    avg {[x]
        sig:x`sig;
        valid:where not null sig;
        if[(count valid) < 2; :0n];
        vSig:sig valid;
        avgAbs:1e-10 | avg abs vSig;
        (avg abs 1 _ deltas vSig) % avgAbs
    } each symData}

arIC_:{[symData;grp;nSyms;pcfg]
    hasPx:pcfg`hasPx;
    icLags:pcfg`icLags;
    nullIc:`ic`panelIc`tsIc`ic_decay!(0n;0n;(key grp)!(count symData)#0n;icLags!(count icLags)#0n);
    if[not hasPx; :nullIc];
    // Build flat table: prev sig per sym, then combine
    tab:raze {[nm;sd]
        ([] dt:sd`dt; sym:(count sd`dt)#nm; ps:prev sd`sig; r:sd`pxDiff)
    }.'flip (key grp; symData);
    if[0 = count tab; :nullIc];
    // panelIc: pooled cor(prev sig, pxDiff) across all syms and times
    pv:where (not null tab`ps) and not null tab`r;
    rawPanel:$[5 < count pv; cor[tab[`ps] pv; tab[`r] pv]; 0n];
    panelIc:$[rawPanel within -1 1f; rawPanel; 0n];
    // ic: average cross-sectional cor(prev sig, pxDiff) grouped by time
    // Group by dt, compute cor per group, fill null->0, avg non-zero
    icByDt:{[tab;idx]
        ps:tab[`ps] idx; r:tab[`r] idx;
        v:where (not null ps) and not null r;
        $[2 > count v; 0n; cor[ps v; r v]]
    }[tab;] each value group tab`dt;
    // Replace null/0w/-0w with 0 — valid cor is always in [-1,1]
    icByDt:@[icByDt; where not icByDt within -1 1f; :; 0f];
    icNZ:icByDt where icByDt <> 0f;
    ic:$[0 < count icNZ; avg icNZ; panelIc];
    // Per-sym IC: cor(prev sig, pxDiff) within each sym
    tsIcBySym:(key grp)!{[x]
        ps:prev x`sig; r:x`pxDiff;
        valid:where (not null ps) and not null r;
        v:$[5 < count valid; cor[ps valid; r valid]; 0n];
        $[v within -1 1f; v; 0n]
    } each symData;
    // IC decay: cor(prev sig, fwd h-day pxDiff) at each horizon
    icDecay:{[symData;h]
        ics:{[sd;h]
            ps:prev sd`sig; r:sd`pxDiff; nn:count ps;
            fwdRet:msum[h; 1 rotate r];
            fwdRet:@[fwdRet; ((nn - h) + til h); :; 0n];
            valid:where (not null ps) and not null fwdRet;
            c:$[5 < count valid; cor[ps valid; fwdRet valid]; 0n];
            $[c within -1 1f; c; 0n]
        }[;h] each symData;
        v:ics where not null ics;
        $[0 < count v; avg v; 0n]
    }[symData;] each icLags;
    icStd:$[1 < count icNZ; dev icNZ; 0n];
    icIR:$[icStd > 1e-10; ic % icStd; 0n];
    `ic`ic_ir`panelIc`tsIc`ic_decay!(ic;icIR;panelIc;tsIcBySym;icLags!icDecay)}

arRegime_:{[symData;dts;nzr;nzDts;n;pcfg]
    hasRegime:pcfg`hasRegime; hasPx:pcfg`hasPx;
    vrQ:pcfg`vrQ; vrW:pcfg`vrW;
    regimeByDt:$[hasRegime;
        [regTab:raze {([] dt:x`dt; regime:x`regime)} each symData;
         regDaily:0!select regime:avg regime by dt from regTab;
         regDaily:`dt xasc regDaily; regDaily`regime];
        [vrPerSym:{[sd;hasPx;vrQ;vrW]
            x:$[hasPx; sd`pxDiff; sd`alpha];
            vr:.cond.varianceRatio[x; vrQ; vrW];
            ([] dt:sd`dt; vr:vr)
         }[;hasPx;vrQ;vrW] each symData;
         vrTab:raze vrPerSym;
         vrDaily:0!select vr:avg vr by dt from vrTab;
         vrDaily:`dt xasc vrDaily; vrDaily`vr]];
    regimeLookup:dts!regimeByDt;
    nzRegime:regimeLookup nzDts;
    isTrending:$[hasRegime; nzRegime > 0.5; nzRegime > 1f];
    trendR:nzr where isTrending;
    mrR:nzr where not isTrending;
    rm:{[r;nAll]
        nn:count r;
        if[nn < 3; :`n`frac`sharpe`hit_rate`avg_daily!(nn;nn % 1 | nAll;0n;0n;0n)];
        ar:252 * avg r; av:(sqrt 252f) * dev r;
        sh:$[av > 1e-10; ar % av; 0n];
        `n`frac`sharpe`hit_rate`avg_daily!(nn;nn % 1 | nAll;sh;avg r > 0f;avg r)
    }[;n];
    `trending`mr!(rm trendR;rm mrR)}

// --- Main alphaReport function ---
alphaReport:{[t;cfg]
    pcfg:arParseCfg_[t;cfg];
    ext:arExtract_[t;pcfg];
    grp:ext`grp; symData:ext`symData;
    nSyms:count key grp;
    // Aggregate daily alpha
    allRows:raze {([] dt:x`dt; alpha:x`alpha)} each symData;
    daily:0!select alpha:sum alpha by dt from allRows;
    daily:`dt xasc daily;
    dts:daily`dt; r:daily`alpha;
    nzr:r where r <> 0f;
    nzDts:dts where r <> 0f;
    n:count nzr;
    if[n < 3; :(`n`sharpe`sortino`ann_return)!(n;0n;0n;0n)];
    // Compute all sections
    perf:arPerf_[nzr;nzDts;n;pcfg`winPct];
    risk:arRisk_[r;perf`ann_return];
    payoff:arPayoff_[nzr;n;pcfg`topN];
    ep:arEpisodes_[nzr;nzDts;n;pcfg`topN;pcfg`epGap];
    turn:(enlist `turnover)!enlist arTurnover_ symData;
    icm:arIC_[symData;grp;nSyms;pcfg];
    reg:arRegime_[symData;dts;nzr;nzDts;n;pcfg];
    (enlist[`n]!enlist n),perf,risk,payoff,turn,icm,`episodes`regime!(ep;reg)}

example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                .alphalab EXAMPLE: Alpha Research Pipeline";
    -1 "=============================================================================";
    -1 "";

    // 1. Generate synthetic data
    -1 "1. Generating synthetic multi-symbol data (20 syms, 500 days)...";
    data:exampleData[];
    -1 "   Data: ",string[count data]," rows, ",string[count distinct data`sym]," symbols, ",
        string[count distinct data`dt]," dates";
    -1 "";

    // 2. Define a cross-sectional alpha (momentum)
    -1 "2. Defining alphas...";
    -1 "";
    -1 "   CS Alpha: Cross-sectional momentum (ranks symbols by rolling return)";
    csMomFn:{[data;params]
        w:`int$params`window;
        syms:asc distinct data`sym;
        raze {[data;w;s]
            sub:`dt xasc select from data where sym=s;
            sig:mavg[w;sub`ret];
            ([] dt:sub`dt; sym:(count sub`dt)#s; sig:sig)
            }[data;w] each syms
        };
    defineAlpha[`csMom;`cs;csMomFn;enlist[`window]!enlist 10 100f];
    -1 "   Registered `csMom with param: window in [10,100]";

    // 3. Define a time-series alpha (mean reversion z-score)
    -1 "   TS Alpha: Mean reversion z-score (single parameter: window)";
    tsMRFn:{[data;params]
        w:`int$params`window;
        syms:asc distinct data`sym;
        raze {[data;w;s]
            sub:`dt xasc select from data where sym=s;
            mu:mavg[w;sub`ret];
            sd:mdev[w;sub`ret];
            sig:neg (sub[`ret] - mu) % sd;  // negative z-score = mean reversion
            ([] dt:sub`dt; sym:(count sub`dt)#s; sig:sig)
            }[data;w] each syms
        };
    defineAlpha[`tsMR;`ts;tsMRFn;enlist[`window]!enlist 5 60f];
    -1 "   Registered `tsMR with param: window in [5,60]";
    -1 "";

    // 4. Single evaluation
    -1 "3. Single evaluation (CS alpha: csMom, window=20)...";
    csEval:evalAlpha[`csMom;enlist[`window]!enlist 20f;data;()!()];
    -1 "   IC IR:            ",string csEval`icIR;
    -1 "   Long/Short Sharpe:",string csEval`longShortSharpe;
    -1 "   Turnover:         ",string csEval`turnover;
    -1 "   IC Decay:";
    show csEval`icDecay;
    -1 "";

    -1 "4. Single evaluation (TS alpha: tsMR, window=20)...";
    tsEval:evalAlpha[`tsMR;enlist[`window]!enlist 20f;data;()!()];
    -1 "   Sharpe:   ",string tsEval`tsSharpe;
    -1 "   Hit Rate: ",string tsEval`tsHitRate;
    -1 "   Turnover: ",string tsEval`turnover;
    -1 "";

    // 5. Bayesian optimization (small budget for demo)
    -1 "5. Bayesian optimization (nInit=5, nIter=10, nFolds=3)...";
    -1 "";
    smallCfg:`nInit`nIter`nFolds`nCandidates!(5;10;3;100);

    -1 "--- Optimizing CS alpha: csMom ---";
    csResult:bayesOpt[`csMom;data;smallCfg];
    -1 "";
    report csResult;

    -1 "--- Optimizing TS alpha: tsMR ---";
    tsResult:bayesOpt[`tsMR;data;smallCfg];
    -1 "";
    report tsResult;

    // 6. Parameter sensitivity
    -1 "6. Parameter sensitivity (csMom)...";
    sens:paramSensitivity[`csMom;csResult`bestParams;data;0.1;smallCfg];
    -1 "   Sensitivity analysis:";
    show sens;
    -1 "";

    // Cleanup
    removeAlpha `csMom;
    removeAlpha `tsMR;

    -1 "=== Example complete ===";
    -1 "";
    `csResult`tsResult!(csResult;tsResult)}

// Return to root namespace
\d .
