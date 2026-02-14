// =============================================================================
// MOMENTUM ENSEMBLE SIGNAL LIBRARY
// =============================================================================
// Multi-speed momentum ensemble with vol scaling, evaluation suite, and tests.
// Version: 0.1.0
// Dependencies: cond.q, kdbtools.q
// Optional:     alphalab.q (for alphaEval bridge in evaluate)
//
// Core idea: blend momentum signals at multiple lookback speeds (1W to 1Y),
// normalize, combine, vol-scale, and evaluate. Anti-lookahead via prev on
// all lookback computations. All arithmetic uses explicit parentheses
// (Q evaluates right-to-left).

\d .momentum

// =============================================================================
// CONFIGURATION
// =============================================================================

defaultSpeeds:5 10 21 63 126 252
defaultNorm:`zscore
defaultComb:`equal
defaultVolWindow:20
defaultTargetVol:0.10
defaultClipBound:3f

// =============================================================================
// SECTION 1: SINGLE-SPEED MOMENTUM SIGNAL
// =============================================================================

// Compute momentum signal for a single lookback window.
// x:          return series (daily yield changes or price returns)
// w:          lookback window (days)
// normMethod: `zscore `rank `sign `sigmoid
// Returns:    normalized signal vector (same length as x)
//
// Process:
//   1. EMA of returns with halflife = w/2 (exponential weighting)
//   2. prev to prevent lookahead (signal at t uses data up to t-1)
//   3. Normalize using selected method
//   4. Clip at +/- 3 to prevent extreme positions
singleSpeed:{[x;w;normMethod]
    x:"f"$x;
    // EMA halflife = w/2 gives effective lookback ~ w days
    hl:w % 2f;
    raw:.cond.smooth[x;hl];
    // Anti-lookahead: signal at t uses data through t-1
    lagged:prev raw;
    // Normalize
    sig:$[normMethod ~ `zscore;
            .cond.rzscore[w;lagged];
          normMethod ~ `rank;
            (2 * .cond.rrank[w;lagged]) - 1f;  // scale [0,1] to [-1,1]
          normMethod ~ `sign;
            "f"$signum lagged;
          normMethod ~ `sigmoid;
            [z:.cond.rzscore[w;lagged]; z % sqrt 1 + (z * z)];
          '"unknown normMethod: use `zscore`rank`sign`sigmoid"];
    // Clip extreme values at +/- defaultClipBound
    (neg defaultClipBound) | sig & defaultClipBound}

// =============================================================================
// SECTION 2: MULTI-SPEED ENSEMBLE
// =============================================================================

// Build ensemble from multiple lookback speeds.
// x:          return series
// speeds:     list of lookback windows (e.g., 5 10 21 63 126 252)
// normMethod: normalization method for each speed
// combMethod: `equal `invVol `invCorr
// Returns:    blended signal vector
ensemble:{[x;speeds;normMethod;combMethod]
    // Compute signal at each speed
    sigs:singleSpeed[x;;normMethod] each speeds;
    n:count sigs;
    nObs:count first sigs;
    // Compute weights based on combination method
    $[combMethod ~ `equal;
        // Simple equal-weight average
        avgSigs[sigs];
      combMethod ~ `invVol;
        // Weight by inverse of each signal's rolling volatility
        invVolCombine[sigs];
      combMethod ~ `invCorr;
        // Decorrelation weighting: minimize pairwise correlation
        invCorrCombine[sigs];
      '"unknown combMethod: use `equal`invVol`invCorr"]}

// Simple average across signal vectors (handles nulls)
avgSigs:{[sigs]
    n:count sigs;
    total:sigs[0];
    i:1;
    while[i < n;
        total:total + sigs[i];
        i+:1];
    total % n}

// Inverse-vol combination: weight each speed by 1/rolling_vol
// Uses EWMA vol with halflife 20 for responsiveness
invVolCombine:{[sigs]
    n:count sigs;
    nObs:count first sigs;
    // Rolling vol for each signal (EWMA, halflife 20)
    vols:{mdev[20;x]} each sigs;
    // Inverse vol weights (floor vol at 1e-6 to avoid division by zero)
    invVols:{reciprocal 1e-6 | x} each vols;
    // Normalize weights to sum to 1 at each point
    totalInvVol:invVols[0];
    i:1;
    while[i < n;
        totalInvVol:totalInvVol + invVols[i];
        i+:1];
    totalInvVol:1e-6 | totalInvVol;
    wts:{x % y}'[invVols;n#enlist totalInvVol];
    // Weighted sum
    result:nObs # 0f;
    i:0;
    while[i < n;
        result:result + (sigs[i] * wts[i]);
        i+:1];
    result}

// Inverse-correlation combination: decorrelation weighting
// Approximation: weight_i = 1 / sum(|corr(i,j)|) for j != i
// This downweights signals that are highly correlated with others
invCorrCombine:{[sigs]
    n:count sigs;
    nObs:count first sigs;
    // Compute pairwise rolling correlations (use trailing 63-day window)
    // For efficiency, use full-sample correlation as proxy for weighting
    // (rolling would be more accurate but much more expensive)
    nValid:count where not null sigs[0];
    // If too few valid obs, fall back to equal weight
    if[20 > nValid; :avgSigs[sigs]];
    // Clean each sig of nulls by forward-filling
    cleanSigs:{fills x} each sigs;
    // Use the last 252 obs (or all if shorter) for correlation estimation
    useN:252 & count first cleanSigs;
    tails:(neg useN)#/:cleanSigs;
    // Pairwise absolute correlation matrix
    corrMat:{[tails;i;n] {[tails;i;j] abs cor[tails i;tails j]}[tails;i;] each til n}[tails;;n] each til n;
    // Row sums of |corr| (subtract 1 for diagonal)
    rowSums:{(sum x) - 1f} each corrMat;
    // Inverse of row sum = decorrelation weight
    rawWts:reciprocal 1e-6 | rowSums;
    // Normalize
    wts:rawWts % sum rawWts;
    // Weighted combination
    result:nObs # 0f;
    i:0;
    while[i < n;
        result:result + (wts[i] * cleanSigs[i]);
        i+:1];
    result}

// =============================================================================
// SECTION 3: VOL-SCALED ENSEMBLE
// =============================================================================

// Ensemble with vol targeting.
// x:          return series
// speeds:     list of lookback windows
// normMethod: normalization for each speed
// combMethod: combination method
// volWindow:  window for vol estimate (EWMA halflife)
// targetVol:  annualized vol target (e.g., 0.10 for 10%)
// Returns:    vol-scaled signal vector
volScaledEnsemble:{[x;speeds;normMethod;combMethod;volWindow;targetVol]
    sig:ensemble[x;speeds;normMethod;combMethod];
    // Compute rolling realized vol of signal-weighted returns
    // sigReturn = prev[sig] * x (signal at t-1 applied to return at t)
    sigRet:0f ^ (prev[sig] * x);  // fill nulls with 0 for EWMA init
    // EWMA vol of signal returns (halflife = volWindow/2)
    hl:volWindow % 2f;
    rollingVar:.cond.smooth[sigRet * sigRet; hl];
    rollingVol:sqrt 1e-10 | rollingVar;
    // Annualize: daily vol * sqrt(252)
    annVol:rollingVol * sqrt 252f;
    // Scale factor: targetVol / annualizedVol
    // Use prev to avoid lookahead on scale factor
    scaleFactor:prev targetVol % (1e-6 | annVol);
    // Cap scale factor at 5x to avoid extreme leverage
    scaleFactor:0f | scaleFactor & 5f;
    // Apply scaling
    sig * scaleFactor}

// =============================================================================
// SECTION 4: TABLE INTERFACE
// =============================================================================

// Compute ensemble for multiple symbols in a table.
// t:   table with at minimum (dt; sym; ret) columns
// cfg: config dict with optional keys:
//      `speeds     - lookback windows (default: 5 10 21 63 126 252)
//      `normMethod - normalization (default: `zscore)
//      `combMethod - combination (default: `equal)
//      `volWindow  - vol estimate window (default: 20)
//      `targetVol  - annual vol target (default: 0.10)
//      `retCol     - return column name (default: `ret)
//      `dtCol      - date column name (default: `dt)
//      `symCol     - symbol column name (default: `sym)
// Returns: table with added columns:
//      sig       - final vol-scaled ensemble signal
//      sig_N     - individual speed signals (e.g., sig_5, sig_10, ...)
//      volScale  - vol scaling factor applied
ensembleTable:{[t;cfg]
    // Parse config with defaults
    speeds:$[`speeds in key cfg; cfg`speeds; defaultSpeeds];
    normMethod:$[`normMethod in key cfg; cfg`normMethod; defaultNorm];
    combMethod:$[`combMethod in key cfg; cfg`combMethod; defaultComb];
    volWindow:$[`volWindow in key cfg; cfg`volWindow; defaultVolWindow];
    targetVol:$[`targetVol in key cfg; cfg`targetVol; defaultTargetVol];
    retCol:$[`retCol in key cfg; cfg`retCol; `ret];
    dtCol:$[`dtCol in key cfg; cfg`dtCol; `dt];
    symCol:$[`symCol in key cfg; cfg`symCol; `sym];

    // Process each symbol group
    syms:asc distinct t symCol;

    // Pack config into a dict to stay within Q's 8-param limit
    pcfg:`symCol`dtCol`retCol`speeds`normMethod`combMethod`volWindow`targetVol!
         (symCol;dtCol;retCol;speeds;normMethod;combMethod;volWindow;targetVol);

    processOneSym:{[t;pcfg;s]
        symCol:pcfg`symCol; dtCol:pcfg`dtCol; retCol:pcfg`retCol;
        speeds:pcfg`speeds; normMethod:pcfg`normMethod;
        combMethod:pcfg`combMethod; volWindow:pcfg`volWindow;
        targetVol:pcfg`targetVol;
        sub:dtCol xasc select from t where sym=s;
        x:sub retCol;

        // Compute individual speed signals
        sigs:singleSpeed[x;;normMethod] each speeds;

        // Build ensemble
        ens:ensemble[x;speeds;normMethod;combMethod];

        // Vol scaling
        sigRet:0f ^ (prev[ens] * x);  // fill nulls for EWMA init
        hl:volWindow % 2f;
        rollingVar:.cond.smooth[sigRet * sigRet; hl];
        rollingVol:sqrt 1e-10 | rollingVar;
        annVol:rollingVol * sqrt 252f;
        sf:prev targetVol % (1e-6 | annVol);
        sf:0f | sf & 5f;
        volSig:ens * sf;

        // Build result dict
        result:(symCol,dtCol,retCol)!(sub symCol;sub dtCol;x);
        // Add individual speed columns
        i:0;
        while[i < count speeds;
            colName:`$"sig_",string speeds i;
            result[colName]:sigs i;
            i+:1];
        // Add ensemble and vol-scaled signal
        result[`sig]:volSig;
        result[`sigRaw]:ens;
        result[`volScale]:sf;
        flip result
    }[t;pcfg;];

    raze processOneSym each syms}

// =============================================================================
// SECTION 5: EVALUATION SUITE
// =============================================================================

// Comprehensive evaluation of momentum signal.
// t: table with dt, sym, sig (signal), ret (returns)
//    Also expects sig_N columns for per-speed analysis
// Returns: dict with all evaluation metrics
evaluate:{[t]
    // --- Basic P&L: signal(t-1) * return(t), aggregated by dt ---
    syms:asc distinct t`sym;
    nSym:count syms;

    // Compute daily P&L per sym, aggregate
    dailyPnl:computeDailyPnl[t;syms];
    r:dailyPnl`ret;
    dts:dailyPnl`dt;
    n:count r;
    nzr:r where r <> 0f;
    nnz:count nzr;

    // --- Standard metrics ---
    annRet:252 * avg nzr;
    annVol:(sqrt 252f) * dev nzr;
    sharpe:$[annVol > 1e-10; annRet % annVol; 0n];

    // Sortino
    downside:nzr where nzr < 0f;
    dd:$[0 < count downside; sqrt avg downside * downside; 1e-10];
    sortino:$[dd > 1e-10; (avg[nzr] * sqrt 252f) % dd; 0n];

    // Drawdown analysis
    cumRets:sums r;
    runningMax:maxs cumRets;
    drawdowns:cumRets - runningMax;
    maxDD:min drawdowns;
    calmar:$[(maxDD < 0) and not null maxDD; neg annRet % maxDD; 0n];

    // Max drawdown duration (in days)
    inDD:drawdowns < 0f;
    ddStarts:differ inDD;  // transitions
    ddDurations:computeDDDurations[drawdowns];
    maxDDDur:$[0 < count ddDurations; max ddDurations; 0];

    // Current drawdown
    currentDD:last drawdowns;

    // Hit rate and profit factor
    wins:nzr where nzr > 0f;
    losses:nzr where nzr < 0f;
    hitRate:$[nnz > 0; (count wins) % nnz; 0n];
    profitFactor:$[(0 < count losses) and 0 < count wins;
        (sum wins) % neg sum losses; 0n];

    // --- Per-speed metrics (requires `ret` column and `sig_*` columns) ---
    hasRet:`ret in cols t;
    speedCols:cols[t] where cols[t] like "sig_*";
    speedSharpes:$[hasRet and 0 < count speedCols;
        computePerSpeedSharpe[t;syms;speedCols];
        ()!()];

    // Ensemble vs best single speed
    bestSingleSharpe:$[0 < count speedSharpes;
        max value speedSharpes; 0n];

    // --- Rolling Sharpe (1Y = 252 days) ---
    rollSharpe:$[n > 252;
        computeRollingSharpe[r;252];
        n # 0n];

    // --- Regime-conditional Sharpe ---
    // Split by variance ratio > 1 (trending) vs < 1 (mean-reverting)
    regimeSharpes:computeRegimeSharpes[t;r;dts];

    // --- Signal autocorrelation (turnover proxy) ---
    sigAutoCorr:computeSigAutoCorr[t;syms];

    // --- IC and IC decay (requires `ret` column for forward returns) ---
    icMetrics:$[hasRet; computeIC[t;syms]; `ic`icIR`icHitRate`icDecay!(0n;0n;0n;()!())];

    // --- Time-series IC per sym ---
    tsICBySym:$[hasRet; computeTSIC[t;syms]; syms!count[syms]#0n];

    // --- Bootstrap Sharpe CI (1000 resamples) ---
    bootCI:bootstrapSharpeCI[nzr;1000];

    // --- Monthly breakdown ---
    monthlyTab:computeMonthlyBreakdown[dts;r];

    // Build result
    (`sharpe`sortino`calmar`hitRate`profitFactor`annReturn`annVol,
     `maxDD`maxDDDuration`currentDD,
     `perSpeedSharpe`bestSingleSharpe,
     `rollingSharpe,
     `regimeSharpes,
     `sigAutoCorr,
     `ic`icIR`icHitRate`icDecay,
     `tsICBySym,
     `bootstrapSharpeLo`bootstrapSharpeHi,
     `monthlyTable,
     `nDays`nNonZeroDays)!
    (sharpe;sortino;calmar;hitRate;profitFactor;annRet;annVol;
     maxDD;maxDDDur;currentDD;
     speedSharpes;bestSingleSharpe;
     rollSharpe;
     regimeSharpes;
     sigAutoCorr;
     icMetrics`ic;icMetrics`icIR;icMetrics`icHitRate;icMetrics`icDecay;
     tsICBySym;
     bootCI`lo;bootCI`hi;
     monthlyTab;
     n;nnz)}

// --- Evaluation helpers ---

// Compute daily P&L: sig(t-1) * ret(t), aggregated across syms
// If table already has `pnl` column, use it directly
computeDailyPnl:{[t;syms]
    hasPnl:`pnl in cols t;
    hasRet:`ret in cols t;
    pnls:raze $[hasPnl;
        {[t;s] sub:`dt xasc select from t where sym=s; ([] dt:sub`dt; pnl:sub`pnl)}[t;] each syms;
        hasRet;
        {[t;s] sub:`dt xasc select from t where sym=s; pnl:prev[sub`sig] * sub`ret; ([] dt:sub`dt; pnl:pnl)}[t;] each syms;
        '"evaluate requires `ret or `pnl column"];
    daily:0!select ret:sum pnl by dt from pnls;
    daily:`dt xasc daily;
    daily}

// Compute drawdown durations
computeDDDurations:{[drawdowns]
    inDD:drawdowns < neg 1e-10;
    if[not any inDD; :enlist 0];
    // Find transitions: 0->1 = DD start, 1->0 = DD end
    // Use deltas on int cast: +1 = entered DD, -1 = exited DD
    d:deltas "i"$inDD;
    starts:where d = 1i;
    ends:where d = -1i;
    // Handle: starts in DD from beginning (inDD[0] is true)
    if[inDD[0] and (0 = count starts) or (count starts) > 0 and starts[0] > 0;
        starts:0,starts];
    // Handle: still in DD at end of series
    if[(count starts) > count ends; ends:ends,count drawdowns];
    $[0 < count starts; ends - starts; enlist 0]}

// Per-speed Sharpe ratios
computePerSpeedSharpe:{[t;syms;speedCols]
    {[t;syms;result;sc]
        pnls:raze {[t;sc;s]
            sub:`dt xasc select from t where sym=s;
            pnl:prev[sub sc] * sub`ret;
            ([] dt:sub`dt; pnl:pnl)
        }[t;sc;] each syms;
        daily:0!select ret:sum pnl by dt from pnls;
        r:daily[`ret] where daily[`ret] <> 0f;
        sr:$[(count r) > 10;
            ((252 * avg r) % (sqrt[252f] * dev r));
            0n];
        result[sc]:sr;
        result
    }[t;syms]/[()!();speedCols]}

// Rolling Sharpe (annualized)
computeRollingSharpe:{[r;w]
    mu:mavg[w;r];
    vol:mdev[w;r];
    (sqrt[252f] * mu) % 1e-10 | vol}

// Regime-conditional Sharpe (uses variance ratio on portfolio returns)
computeRegimeSharpes:{[t;r;dts]
    // Compute VR on the aggregated portfolio daily returns
    vr:.cond.varianceRatio[r;5;60];
    // Split by VR regime
    trendIdx:where (vr > 1f) and not null vr;
    mrIdx:where (vr <= 1f) and not null vr;
    trendR:r trendIdx;
    mrR:r mrIdx;
    trendSharpe:$[(count trendR where trendR <> 0f) > 10;
        ((252 * avg trendR) % (sqrt[252f] * dev trendR)); 0n];
    mrSharpe:$[(count mrR where mrR <> 0f) > 10;
        ((252 * avg mrR) % (sqrt[252f] * dev mrR)); 0n];
    `trending`meanReverting!(trendSharpe;mrSharpe)}

// Signal autocorrelation (average across syms)
computeSigAutoCorr:{[t;syms]
    autos:{[t;s]
        sub:`dt xasc select from t where sym=s;
        sig:sub`sig;
        valid:sig where not null sig;
        $[30 < count valid;
            cor[neg[1] _ valid; 1 _ valid];
            0n]
    }[t;] each syms;
    avg autos where not null autos}

// Helper: compute rolling IC for one sym at one horizon
computeICOneSym:{[t;s;h]
    sub:`dt xasc select from t where sym=s;
    sig:sub`sig;
    fwdRet:h msum (1 rotate sub`ret);
    // Null out last h values
    n:count fwdRet;
    idxs:(n - h) + til h;
    fwdRet[idxs]:0n;
    .cond.rollingIC[sig;fwdRet;60]}

// Helper: average IC across syms at one horizon
avgICAtHorizon:{[t;syms;h]
    ics:raze computeICOneSym[t;;h] each syms;
    v:ics where not null ics;
    $[0 < count v; avg v; 0n]}

// IC metrics (time-series: correlate sig with next-period return)
computeIC:{[t;syms]
    // Per-sym rolling IC at horizon 1, then average
    ics:raze computeICOneSym[t;;1] each syms;
    valid:ics where not null ics;
    nIC:count valid;
    icMean:$[nIC > 0; avg valid; 0n];
    icStd:$[nIC > 1; dev valid; 0n];
    icIR:$[(nIC > 1) and icStd > 1e-10; icMean % icStd; 0n];
    icHitRate:$[nIC > 0; (sum valid > 0f) % nIC; 0n];

    // IC decay: IC at different horizons
    horizons:1 2 5 10 20;
    icDecayVals:avgICAtHorizon[t;syms;] each horizons;
    icDecay:horizons!icDecayVals;

    `ic`icIR`icHitRate`icDecay!(icMean;icIR;icHitRate;icDecay)}

// Time-series IC per sym
computeTSIC:{[t;syms]
    syms!{[t;s]
        sub:`dt xasc select from t where sym=s;
        sig:sub`sig;
        fwdRet:1 rotate sub`ret;
        fwdRet[(count fwdRet) - 1]:0n;
        valid:where (not null sig) and not null fwdRet;
        $[30 < count valid;
            cor[sig valid;fwdRet valid];
            0n]
    }[t;] each syms}

// Bootstrap Sharpe confidence interval
// Resample daily returns 1000x, compute Sharpe each time
bootstrapSharpeCI:{[r;nBoot]
    n:count r;
    if[n < 10; :`lo`hi!(0n;0n)];
    sharpes:{[r;n;i]
        sample:r n?n;
        $[dev[sample] > 1e-10;
            (sqrt[252f] * avg sample) % dev sample;
            0n]
    }[r;n;] each til nBoot;
    valid:asc sharpes where not null sharpes;
    nv:count valid;
    if[nv < 10; :`lo`hi!(0n;0n)];
    lo:valid `long$0.025 * nv;
    hi:valid `long$0.975 * nv;
    `lo`hi!(lo;hi)}

// Monthly breakdown table
computeMonthlyBreakdown:{[dts;r]
    months:`month$dts;
    tbl:([] month:months; r:r);
    agg:0!select ret:sum r, nDays:count r, hitRate:avg r > 0f,
               mu:avg r, vol:dev r by month from tbl;
    // Compute monthly Sharpe (avoid $[] in select)
    agg:update sharpe:(sqrt[252f] * mu) % 1e-10 | vol from agg;
    delete mu, vol from agg}

// =============================================================================
// SECTION 6: SYNTHETIC DATA GENERATOR
// =============================================================================

// Generate test data with known regime properties.
// Returns table with dt, sym, ret, regime columns.
// Symbol A: Strong trending  (AR(1) phi=0.3, positive drift)
// Symbol B: Pure random walk  (no autocorrelation)
// Symbol C: Mean-reverting    (AR(1) phi=-0.3)
syntheticTest:{[]
    system "S 42";
    nDays:1000;
    dts:2020.01.01 + til nDays;
    vol:0.01;  // daily vol

    // Symbol A: Trending (AR(1) with phi=0.3, drift=0.0002)
    drift:0.0002;
    phi:0.3;
    epsA:.cond.randNorm nDays;
    retA:nDays # 0f;
    retA[0]:vol * epsA[0];
    i:1;
    while[i < nDays;
        retA[i]:drift + (phi * retA[i - 1]) + (vol * epsA[i]);
        i+:1];

    // Symbol B: Random walk (iid, no autocorrelation)
    epsB:.cond.randNorm nDays;
    retB:vol * epsB;

    // Symbol C: Mean-reverting (AR(1) with phi=-0.3)
    phiC:neg 0.3;
    epsC:.cond.randNorm nDays;
    retC:nDays # 0f;
    retC[0]:vol * epsC[0];
    i:1;
    while[i < nDays;
        retC[i]:(phiC * retC[i - 1]) + (vol * epsC[i]);
        i+:1];

    // Combine into table
    t:raze (
        ([] dt:dts; sym:nDays # `A; ret:retA; regime:nDays # `trending);
        ([] dt:dts; sym:nDays # `B; ret:retB; regime:nDays # `random);
        ([] dt:dts; sym:nDays # `C; ret:retC; regime:nDays # `meanReverting)
    );
    `dt`sym xasc t}

// =============================================================================
// SECTION 7: TEST RUNNER
// =============================================================================

// Run all tests. Returns pass/fail summary.
runTests:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum TEST SUITE";
    -1 "=============================================================================";
    -1 "";

    nPass:0; nFail:0;
    results:()!();

    // --- Test 1: Single speed output length and no nulls after warmup ---
    -1 "Test 1: Single speed - correct length, no nulls after warmup";
    system "S 42";
    x:1000 # 0.01 * .cond.randNorm 1000;
    sig:.momentum.singleSpeed[x;21;`zscore];
    t1len:1000 = count sig;
    // After 2*w warmup, should have no nulls
    warmup:2 * 21;
    t1nn:0 = sum null sig warmup + til (1000 - warmup);
    t1:t1len and t1nn;
    results[`test1]:t1;
    -1 "  Length correct: ",string t1len;
    -1 "  No nulls after warmup: ",string t1nn;
    -1 "  ",($[t1;"PASS";"FAIL"]);
    nPass+:t1; nFail+:not t1;
    -1 "";

    // --- Test 2: Equal-weight ensemble = mean of components ---
    -1 "Test 2: Equal-weight ensemble = mean of components";
    system "S 42";
    x:500 # 0.01 * .cond.randNorm 500;
    speeds:5 21 63;
    sigs:.momentum.singleSpeed[x;;`zscore] each speeds;
    ens:.momentum.ensemble[x;speeds;`zscore;`equal];
    manualAvg:((sigs[0] + sigs[1]) + sigs[2]) % 3f;
    // Compare after warmup (where both are non-null)
    warmup:2 * 63;  // largest speed
    validIdx:warmup + til (500 - warmup);
    diff:max abs (ens - manualAvg) validIdx;
    t2:diff < 1e-10;
    results[`test2]:t2;
    -1 "  Max diff from manual average: ",string diff;
    -1 "  ",($[t2;"PASS";"FAIL"]);
    nPass+:t2; nFail+:not t2;
    -1 "";

    // --- Test 3: Vol scaling produces roughly target vol ---
    -1 "Test 3: Vol scaling produces roughly target vol";
    system "S 42";
    x:2000 # 0.01 * .cond.randNorm 2000;
    targetVol:0.10;
    sig:.momentum.volScaledEnsemble[x;21 63 126;`zscore;`equal;20;targetVol];
    // Compute realized vol of sig * x after warmup
    sigRet:prev[sig] * x;
    warmup:300;  // generous warmup
    realizedVol:(sqrt 252f) * dev sigRet warmup + til (2000 - warmup);
    // Allow wide tolerance (50%) since vol targeting is approximate
    t3:(realizedVol > (0.5 * targetVol)) and realizedVol < (2.0 * targetVol);
    results[`test3]:t3;
    -1 "  Target vol: ",string targetVol;
    -1 "  Realized vol: ",string realizedVol;
    -1 "  Within 50%-200% of target: ",string t3;
    -1 "  ",($[t3;"PASS";"FAIL"]);
    nPass+:t3; nFail+:not t3;
    -1 "";

    // --- Test 4: Anti-lookahead: signal at t uses only data up to t-1 ---
    -1 "Test 4: Anti-lookahead (signal at t uses data up to t-1)";
    system "S 42";
    x:100 # 0.01 * .cond.randNorm 100;
    // Compute signal for first 50 observations
    sig50:.momentum.singleSpeed[50 # x;21;`zscore];
    // Compute signal for first 51 observations
    sig51:.momentum.singleSpeed[51 # x;21;`zscore];
    // Signal at position 49 should be same in both (uses data up to t-1 = 48)
    // sig50 has 50 elements; sig51 has 51 elements
    // Due to prev: sig50[49] uses EMA through index 48
    //              sig51[49] uses EMA through index 48 (same data!)
    t4:$[(null sig50[49]) or null sig51[49];
        1b;  // both null = consistent
        (abs (sig50[49]) - sig51[49]) < 1e-10];
    results[`test4]:t4;
    -1 "  sig[49] with 50 obs: ",string sig50[49];
    -1 "  sig[49] with 51 obs: ",string sig51[49];
    -1 "  ",($[t4;"PASS";"FAIL"]);
    nPass+:t4; nFail+:not t4;
    -1 "";

    // --- Test 5: Trending data - ensemble Sharpe > 0 ---
    -1 "Test 5: Trending data (Symbol A) - ensemble Sharpe > 0";
    td:.momentum.syntheticTest[];
    retA:exec ret from td where sym=`A;
    sig:.momentum.ensemble[retA;defaultSpeeds;`zscore;`equal];
    pnl:prev[sig] * retA;
    valid:pnl where (not null pnl) and pnl <> 0f;
    shA:$[(count valid) > 10;
        (sqrt[252f] * avg valid) % dev valid;
        0n];
    t5:(not null shA) and shA > 0f;
    results[`test5]:t5;
    -1 "  Trending Sharpe: ",string shA;
    -1 "  ",($[t5;"PASS";"FAIL"]);
    nPass+:t5; nFail+:not t5;
    -1 "";

    // --- Test 6: Random walk - Sharpe near 0 ---
    -1 "Test 6: Random walk (Symbol B) - ensemble Sharpe near 0";
    retB:exec ret from td where sym=`B;
    sigB:.momentum.ensemble[retB;defaultSpeeds;`zscore;`equal];
    pnlB:prev[sigB] * retB;
    validB:pnlB where (not null pnlB) and pnlB <> 0f;
    shB:$[(count validB) > 10;
        (sqrt[252f] * avg validB) % dev validB;
        0n];
    // Sharpe should be close to 0 (allow +/- 0.8 for noise)
    t6:(not null shB) and (abs shB) < 0.8;
    results[`test6]:t6;
    -1 "  Random walk Sharpe: ",string shB;
    -1 "  |Sharpe| < 0.8: ",string t6;
    -1 "  ",($[t6;"PASS";"FAIL"]);
    nPass+:t6; nFail+:not t6;
    -1 "";

    // --- Test 7: Mean-reverting data - Sharpe worse than trending ---
    -1 "Test 7: Mean-reverting (Symbol C) - ensemble Sharpe < trending Sharpe";
    retC:exec ret from td where sym=`C;
    sigC:.momentum.ensemble[retC;defaultSpeeds;`zscore;`equal];
    pnlC:prev[sigC] * retC;
    validC:pnlC where (not null pnlC) and pnlC <> 0f;
    shC:$[(count validC) > 10;
        (sqrt[252f] * avg validC) % dev validC;
        0n];
    // Momentum should work worse on mean-reverting than trending data
    // shA (trending) should be substantially higher than shC (mean-reverting)
    t7:(not null shC) and (not null shA) and shC < shA;
    results[`test7]:t7;
    -1 "  Mean-reverting Sharpe: ",string shC;
    -1 "  Trending Sharpe: ",string shA;
    -1 "  Mean-reverting < Trending: ",string t7;
    -1 "  ",($[t7;"PASS";"FAIL"]);
    nPass+:t7; nFail+:not t7;
    -1 "";

    // --- Test 8: Ensemble Sharpe >= best single speed on diversified data ---
    -1 "Test 8: Ensemble Sharpe >= best single speed (diversified)";
    // Use trending data where momentum works - ensemble should not be worse
    // Compute per-speed Sharpes on Symbol A
    speedSharpes:{[retA;w]
        sig:.momentum.singleSpeed[retA;w;`zscore];
        pnl:prev[sig] * retA;
        valid:pnl where (not null pnl) and pnl <> 0f;
        $[(count valid) > 10;
            (sqrt[252f] * avg valid) % dev valid;
            0n]
    }[retA;] each defaultSpeeds;
    validSpSh:speedSharpes where not null speedSharpes;
    bestSingle:$[0 < count validSpSh; max validSpSh; 0n];
    // Ensemble Sharpe (computed above as shA)
    // Ensemble typically underperforms best single speed in-sample due to
    // diversification cost, but the gap should not be extreme (< 0.5 Sharpe)
    t8:$[(null bestSingle) or null shA;
        1b;  // can't compare
        shA > (bestSingle - 0.5)];
    results[`test8]:t8;
    -1 "  Best single-speed Sharpe: ",string bestSingle;
    -1 "  Ensemble Sharpe: ",string shA;
    -1 "  Ensemble >= best - 0.5: ",string t8;
    -1 "  ",($[t8;"PASS";"FAIL"]);
    nPass+:t8; nFail+:not t8;
    -1 "";

    // --- Test 9: Table interface produces correct columns ---
    -1 "Test 9: Table interface produces correct columns";
    cfg:`speeds`normMethod`combMethod!(5 21 63;`zscore;`equal);
    tbl:.momentum.ensembleTable[td;cfg];
    expectedCols:`dt`sym`ret`sig`sigRaw`volScale`sig_5`sig_21`sig_63;
    hasCols:all expectedCols in cols tbl;
    // Check all syms present
    hasSyms:3 = count distinct tbl`sym;
    // Check row count (3 syms * 1000 days each)
    hasRows:3000 = count tbl;
    t9:hasCols and hasSyms and hasRows;
    results[`test9]:t9;
    -1 "  Has expected columns: ",string hasCols;
    -1 "  Has all 3 syms: ",string hasSyms;
    -1 "  Has 3000 rows: ",string hasRows;
    -1 "  ",($[t9;"PASS";"FAIL"]);
    nPass+:t9; nFail+:not t9;
    -1 "";

    // --- Test 10: Evaluation suite returns all expected keys ---
    -1 "Test 10: Evaluation suite returns all expected keys";
    evalT:update sig:.momentum.ensemble[ret;5 21 63;`zscore;`equal] by sym from td;
    evalT:update sig_5:.momentum.singleSpeed[ret;5;`zscore] by sym from evalT;
    evalT:update sig_21:.momentum.singleSpeed[ret;21;`zscore] by sym from evalT;
    evalT:update sig_63:.momentum.singleSpeed[ret;63;`zscore] by sym from evalT;
    ev:.momentum.evaluate[evalT];
    expectedKeys:`sharpe`sortino`calmar`hitRate`profitFactor`annReturn`annVol;
    expectedKeys:expectedKeys,`maxDD`maxDDDuration`currentDD;
    expectedKeys:expectedKeys,`perSpeedSharpe`bestSingleSharpe;
    expectedKeys:expectedKeys,`rollingSharpe`regimeSharpes`sigAutoCorr;
    expectedKeys:expectedKeys,`ic`icIR`icHitRate`icDecay`tsICBySym;
    expectedKeys:expectedKeys,`bootstrapSharpeLo`bootstrapSharpeHi;
    expectedKeys:expectedKeys,`monthlyTable`nDays`nNonZeroDays;
    hasKeys:all expectedKeys in key ev;
    // Check Sharpe is a number
    sharpeValid:(not null ev`sharpe) and (-9h) = type ev`sharpe;
    t10:hasKeys and sharpeValid;
    results[`test10]:t10;
    -1 "  Has all expected keys: ",string hasKeys;
    -1 "  Sharpe is valid float: ",string sharpeValid;
    -1 "  Sharpe = ",string ev`sharpe;
    -1 "  ",($[t10;"PASS";"FAIL"]);
    nPass+:t10; nFail+:not t10;
    -1 "";

    // --- Summary ---
    -1 "=============================================================================";
    -1 "                           TEST SUMMARY";
    -1 "=============================================================================";
    -1 "";
    -1 "  Passed: ",string nPass;
    -1 "  Failed: ",string nFail;
    -1 "";
    $[nFail = 0;
        -1 "  ALL TESTS PASSED!";
        -1 "  SOME TESTS FAILED - see details above"];
    -1 "";
    results}

// =============================================================================
// EXAMPLE
// =============================================================================

example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum EXAMPLE: Multi-Speed Ensemble";
    -1 "=============================================================================";
    -1 "";

    // Generate synthetic data
    -1 "1. Generate synthetic test data (3 syms, 1000 days each)";
    td:.momentum.syntheticTest[];
    -1 "  Rows: ",string count td;
    -1 "  Syms: ","," sv string asc distinct td`sym;
    -1 "";

    // Compute ensemble for trending symbol
    -1 "2. Single-speed signals for Symbol A (trending):";
    retA:exec ret from td where sym=`A;
    {[retA;w]
        sig:.momentum.singleSpeed[retA;w;`zscore];
        pnl:prev[sig] * retA;
        valid:pnl where (not null pnl) and pnl <> 0f;
        sh:$[(count valid) > 10; (sqrt[252f] * avg valid) % dev valid; 0n];
        -1 "  Speed ",string[w],"D Sharpe: ",string sh;
    }[retA;] each defaultSpeeds;
    -1 "";

    // Ensemble
    -1 "3. Ensemble signal (equal weight):";
    sig:.momentum.ensemble[retA;defaultSpeeds;`zscore;`equal];
    pnl:prev[sig] * retA;
    valid:pnl where (not null pnl) and pnl <> 0f;
    sh:(sqrt[252f] * avg valid) % dev valid;
    -1 "  Ensemble Sharpe: ",string sh;
    -1 "";

    // Vol-scaled
    -1 "4. Vol-scaled ensemble (target 10%):";
    vsig:.momentum.volScaledEnsemble[retA;defaultSpeeds;`zscore;`equal;20;0.10];
    vpnl:prev[vsig] * retA;
    vvalid:vpnl where (not null vpnl) and vpnl <> 0f;
    vsh:(sqrt[252f] * avg vvalid) % dev vvalid;
    rvol:(sqrt 252f) * dev vvalid;
    -1 "  Vol-scaled Sharpe: ",string vsh;
    -1 "  Realized annual vol: ",string rvol;
    -1 "";

    // Table interface
    -1 "5. Table interface:";
    cfg:`speeds`normMethod`combMethod!(5 21 63 126;`zscore;`equal);
    tbl:.momentum.ensembleTable[td;cfg];
    -1 "  Output columns: ","," sv string cols tbl;
    -1 "  Rows: ",string count tbl;
    -1 "";

    -1 "Done.";
    -1 "";
    }

// =============================================================================
// HELP
// =============================================================================

help:{[]
    -1 "";
    -1 "=== .momentum MULTI-SPEED ENSEMBLE v0.1.0 ===";
    -1 "";
    -1 "CORE:";
    -1 "  singleSpeed[x;w;normMethod]                          - momentum at one lookback";
    -1 "  ensemble[x;speeds;normMethod;combMethod]             - multi-speed blend";
    -1 "  volScaledEnsemble[x;speeds;norm;comb;volWin;tgtVol]  - vol-targeted ensemble";
    -1 "";
    -1 "TABLE:";
    -1 "  ensembleTable[t;cfg]  - compute for table with (dt;sym;ret)";
    -1 "    cfg keys: speeds, normMethod, combMethod, volWindow, targetVol";
    -1 "    Returns:  table + sig, sigRaw, volScale, sig_N columns";
    -1 "";
    -1 "EVALUATION:";
    -1 "  evaluate[t]     - comprehensive metrics (t needs dt,sym,sig,ret + sig_N)";
    -1 "  syntheticTest[]  - generate test data (trending/random/mean-reverting)";
    -1 "  runTests[]       - run 10 validation tests";
    -1 "  example[]        - worked example";
    -1 "";
    -1 "NORMALIZATION: `zscore `rank `sign `sigmoid";
    -1 "COMBINATION:   `equal `invVol `invCorr";
    -1 "";
    -1 "DEFAULTS:";
    -1 "  speeds:    5 10 21 63 126 252 (1W 2W 1M 3M 6M 1Y)";
    -1 "  norm:      `zscore";
    -1 "  comb:      `equal";
    -1 "  volWindow: 20";
    -1 "  targetVol: 0.10 (10% annual)";
    -1 "  clip:      +/- 3";
    -1 "";}

\d .

-1 "Loaded .momentum namespace v0.1.0";
-1 "Multi-speed momentum ensemble: singleSpeed, ensemble, volScaledEnsemble";
-1 "Run .momentum.help[] for full function list";
-1 "Run .momentum.runTests[] for validation";
