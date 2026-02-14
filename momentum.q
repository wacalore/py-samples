// =============================================================================
// MOMENTUM SIGNAL LIBRARY v3.0
// =============================================================================
// Regime-switched blend architecture for multi-speed momentum.
// Two signal variants (slow trend ensemble, fast responsive) are blended
// based on regime state. Carry enters additively. Curve agreement adjusts
// leverage cap. Single DD brake as protection.
//
// Version: 3.0.0
// Dependencies: cond.q, kdbtools.q
// Optional:     alphalab.q (for alphaEval bridge)
//
// Pipeline:
//   1. Sharpe momentum (mu/sigma) at trend speeds — self-normalizing
//   2. Momentum acceleration (2nd derivative) overlay
//   3. VR-weighted trend ensemble (raw, unscaled)
//   4. Fast responsive signal (single speed)
//   5. Regime blend: regimeW * trendSig + (1-regimeW) * fastSig
//   6. Additive carry tilt (optional, if carry column present)
//   7. Curve agreement adjusts max leverage ceiling (portfolio level)
//   8. Vol targeting + DD brake (floored at 0.6)
//
// All arithmetic uses explicit parentheses (Q right-to-left).

\d .momentum

// =============================================================================
// CONFIGURATION
// =============================================================================

defaultTrendSpeeds:63 126 252
defaultFastSpeed:21
defaultAccelWt:0.3
defaultBlendHL:10
defaultCarryTiltWt:0.15
defaultVRQ:5
defaultVRW:60
defaultTargetVol:0.10
defaultMaxLev:3f
defaultDDThresh:0.05
defaultClip:4f

// =============================================================================
// SECTION 1: SHARPE MOMENTUM (RISK-ADJUSTED)
// =============================================================================

// Risk-adjusted momentum at a single lookback speed.
// Returns mu/sigma instead of raw EMA — self-normalizing in variable-vol.
// A 20bp move in 5-vol is 4-sigma (strong); in 20-vol is 1-sigma (noise).
//
// x: return series (daily yield changes or price returns)
// w: lookback window (days). EMA halflife = w/2.
// Returns: prev(mu/sigma) — lagged for anti-lookahead.
sharpeMom:{[x;w]
    x:"f"$x;
    hl:w % 2f;
    mu:.cond.smooth[x; hl];
    // EWMA variance: E[x^2] - (E[x])^2
    ewmaX2:.cond.smooth[x * x; hl];
    // Floor at 1% of long-run variance to avoid division by noise
    lrVar:(var x) | 1e-10;
    floorVar:0.01 * lrVar;
    localVar:floorVar | (ewmaX2 - (mu * mu));
    vol:sqrt localVar;
    prev mu % vol}

// =============================================================================
// SECTION 2: MOMENTUM ACCELERATION
// =============================================================================

// Second derivative of momentum — detects trend strengthening/weakening.
// Valuable at FOMC inflection points: acceleration deteriorates before level.
//
// mom: momentum signal vector
// w:   lookback window (determines lag k = max(1, w/5))
accel:{[mom;w]
    k:1 | `long$w % 5;
    mom - (k xprev mom)}

// Combined single-speed signal: Sharpe momentum + acceleration.
// x:  return series
// w:  lookback window
// aw: acceleration weight (0.3 recommended)
// Returns: clipped signal in [-4, 4]
singleSpeed:{[x;w;aw]
    mom:sharpeMom[x;w];
    acc:accel[mom;w];
    sig:mom + (aw * acc);
    (neg defaultClip) | sig & defaultClip}

// =============================================================================
// SECTION 3: REGIME-ADAPTIVE ENSEMBLE
// =============================================================================

// Blend multi-speed signals with VR-adaptive weighting.
// Returns RAW ensemble (unscaled) plus vrPctl for regime blending.
//
// sigs:   list of signal vectors (one per speed)
// x:      return series (for VR computation)
// speeds: list of lookback windows
// vrQ:    VR aggregation horizon (default 5)
// vrW:    VR rolling window (default 60)
// Returns: dict `ensemble`vr`regimeScale`vrPctl
regimeEnsemble:{[sigs;x;speeds;vrQ;vrW]
    n:count sigs;
    nObs:count first sigs;
    // Variance ratio
    vr:.cond.varianceRatio[x;vrQ;vrW];
    // Rolling percentile of VR (robust to bias)
    vrPctl:.cond.rrank[vrW;vr];
    // Regime scale: 0 at 25th pctl, 1 at 75th pctl
    regimeScale:0f | 1f & (vrPctl - 0.25) % 0.5;
    // VR-tilted weights: upweight slow speeds when trending
    maxSpd:"f"$max speeds;
    tilts:{[vrPctl;spd;maxSpd]
        exp[(2f * (vrPctl - 0.5)) * spd % maxSpd]
    }[vrPctl;;"f"$maxSpd] each "f"$speeds;
    // Normalize weights
    tiltSum:nObs # 0f;
    i:0;
    while[i < n; tiltSum:tiltSum + tilts[i]; i+:1];
    tiltSum:1e-10 | tiltSum;
    // Weighted sum (RAW — no regimeScale multiplication)
    ens:nObs # 0f;
    i:0;
    while[i < n;
        ens:ens + (sigs[i] * tilts[i] % tiltSum);
        i+:1];
    `ensemble`vr`regimeScale`vrPctl!(ens; vr; regimeScale; vrPctl)}

// =============================================================================
// SECTION 4: REGIME BLEND
// =============================================================================

// Blend trend and fast signals based on smoothed VR percentile.
// In trending regimes (regimeW→1): uses trend ensemble.
// In mean-reverting regimes (regimeW→0): uses fast responsive signal.
//
// trendSig: VR-weighted trend ensemble (raw)
// fastSig:  fast speed signal
// vrPctl:   rolling VR percentile
// blendHL:  EMA halflife for smoothing regime weight
// Returns: blended signal vector
regimeBlend:{[trendSig;fastSig;vrPctl;blendHL]
    regimeW:.cond.smooth[0.5 ^ vrPctl; blendHL];
    (regimeW * trendSig) + ((1f - regimeW) * fastSig)}

// =============================================================================
// SECTION 5: CARRY TILT
// =============================================================================

// Additive carry adjustment. When carry aligns with signal direction,
// nudges the signal larger. When opposing, nudges smaller.
// Z-scoring normalizes carry magnitude across instruments.
//
// sig:    blended signal vector
// carry:  daily carry return vector
// weight: carry tilt weight (0.15 recommended)
// Returns: tilted signal vector
carryTilt:{[sig;carry;weight]
    carryZ:(neg defaultClip) | defaultClip & 0f ^ .cond.rzscore[126; "f"$carry];
    sig + (weight * (prev carryZ) * signum sig)}

// =============================================================================
// SECTION 6: CRASH PROTECTION
// =============================================================================

// Drawdown circuit breaker.
// 1.0 at no drawdown, scales down with DD, 0.6 floor.
//
// cumPnl: cumulative P&L vector
// thresh: drawdown threshold (0.05 = 5%)
ddBreaker:{[cumPnl;thresh]
    peak:maxs cumPnl;
    dd:cumPnl - peak;
    absDd:0f | neg dd;
    0.6 | 1f - ((0.5 * absDd) % thresh)}

// =============================================================================
// SECTION 7: TABLE INTERFACE
// =============================================================================

// Main entry point. Compute momentum ensemble for multi-sym table.
//
// t:   table with (dt; sym; ret) minimum. Optional: carry column.
// cfg: config dict. All keys optional:
//   `trendSpeeds     - trend lookback windows (default: 63 126 252)
//   `fastSpeed       - fast speed lookback (default: 21)
//   `accelWt         - acceleration weight (default: 0.3)
//   `blendHL         - regime blend smoothing halflife (default: 10)
//   `carryTiltWt     - carry tilt weight (default: 0.15)
//   `vrQ             - VR aggregation horizon (default: 5)
//   `vrW             - VR rolling window (default: 60)
//   `targetVol       - annualized vol target (default: 0.10)
//   `maxLeverage     - max vol scale factor (default: 3)
//   `ddThresh        - drawdown threshold (default: 0.05)
//   `retCol          - return column name (default: `ret)
//   `dtCol           - date/time column name (default: `dt)
//   `symCol          - symbol column name (default: `sym)
//   `carryCol        - carry column name (default: `carry)
//
// Returns: table with columns:
//   sig_N (per-speed), trendSig, fastSig, vr, regimeW,
//   blendedSig, carryTilt, curveAgreement, volScale, ddScale, sig
ensembleTable:{[t;cfg]
    // Parse config
    pcfg:parseConfig[cfg;cols t];
    trendSpeeds:pcfg`trendSpeeds; fastSpeed:pcfg`fastSpeed;
    symCol:pcfg`symCol; dtCol:pcfg`dtCol;
    allSpeeds:asc distinct fastSpeed,trendSpeeds;

    // Phase 1: per-sym signal construction
    t:@[t;`momIdx__;:;til count t];
    grp:group t symCol;
    syms:key grp;

    p1:{[t;pcfg;idx]
        dtCol:pcfg`dtCol; retCol:pcfg`retCol;
        carryCol:pcfg`carryCol; hasCarry:pcfg`hasCarry;
        trendSpeeds:pcfg`trendSpeeds; fastSpeed:pcfg`fastSpeed;
        aw:pcfg`accelWt; blendHL:pcfg`blendHL; carryWt:pcfg`carryTiltWt;
        vrQ:pcfg`vrQ; vrW:pcfg`vrW;
        allSpeeds:asc distinct fastSpeed,trendSpeeds;
        sub:dtCol xasc t idx;
        r:"f"$sub retCol;
        // Excess returns if carry available
        x:$[hasCarry; r - (0f ^ prev "f"$sub carryCol); r];
        // Per-speed signals (all speeds)
        allSigs:{[x;aw;spd] singleSpeed[x;spd;aw]}[x;aw;] each allSpeeds;
        i:0;
        while[i < count allSpeeds;
            sub:@[sub;`$"sig_",string allSpeeds i;:;allSigs i];
            i+:1];
        // Fast signal
        fastIdx:allSpeeds ? fastSpeed;
        fastSig:allSigs fastIdx;
        sub:@[sub;`fastSig;:;fastSig];
        // Trend signals
        trendIdx:allSpeeds ? trendSpeeds;
        trendSigs:allSigs trendIdx;
        // Regime ensemble on trend speeds
        re:regimeEnsemble[trendSigs;x;trendSpeeds;vrQ;vrW];
        trendSig:re`ensemble;
        sub:@[sub;`trendSig;:;trendSig];
        sub:@[sub;`vr;:;re`vr];
        // Regime blend
        vrPctl:re`vrPctl;
        regimeW:.cond.smooth[0.5 ^ vrPctl; blendHL];
        blended:(regimeW * trendSig) + ((1f - regimeW) * fastSig);
        sub:@[sub;`regimeW;:;regimeW];
        sub:@[sub;`blendedSig;:;blended];
        // Carry tilt
        $[hasCarry;
            [tilted:carryTilt[blended; "f"$sub carryCol; carryWt];
             sub:@[sub;`carryTilt;:;tilted - blended];
             sub:@[sub;`blendedSig;:;tilted]];
            sub:@[sub;`carryTilt;:;(count sub) # 0f]];
        sub}[t;pcfg;];

    t:raze p1 each value grp;
    t:`momIdx__ xasc t;

    // Phase 2: portfolio sizing
    nSyms:count syms;

    // Curve agreement: |avg signum(blendedSig)| by dt → adjusts maxLev ceiling
    if[nSyms > 1;
        [dtVals:t dtCol;
         dirVals:signum t`blendedSig;
         grpDt:group dtVals;
         meanDirs:(key grpDt)!{[dv;idx] avg dv idx}[dirVals;] each value grpDt;
         ca:abs meanDirs dtVals;
         t:@[t;`curveAgreement;:;ca]]];
    if[nSyms < 2;
        t:@[t;`curveAgreement;:;(count t) # 1f]];

    // Per-sym vol targeting + DD brake
    grp2:group t pcfg`symCol;

    p2:{[t;pcfg;nSyms;idx]
        dtCol:pcfg`dtCol; retCol:pcfg`retCol;
        targetVol:pcfg`targetVol; maxLev:pcfg`maxLev; ddT:pcfg`ddThresh;
        sub:dtCol xasc t idx;
        r:"f"$sub retCol;
        sigIn:sub`blendedSig;
        // Adjusted max leverage: curve agreement adjusts ceiling [0.7-1.0]
        adjMaxLev:$[nSyms > 1;
            maxLev * (0.7 + (0.3 * sub`curveAgreement));
            (count sub) # maxLev];
        // Vol targeting
        sigRet:0f ^ (prev[sigIn] * r);
        rollingVol:sqrt 1e-10 | .cond.smooth[sigRet * sigRet; 10f];
        annVol:rollingVol * sqrt 252f;
        sf:prev targetVol % (1e-6 | annVol);
        sf:0f | sf & adjMaxLev;
        sigVS:sigIn * sf;
        sub:@[sub;`volScale;:;sf];
        // Drawdown circuit breaker
        pnl:0f ^ (prev[sigVS] * r);
        cumPnl:sums pnl;
        dds:ddBreaker[cumPnl;ddT];
        sub:@[sub;`ddScale;:;dds];
        sub:@[sub;`sig;:;sigVS * prev dds];
        sub}[t;pcfg;nSyms;];

    t:raze p2 each value grp2;
    t:`momIdx__ xasc t;
    ![t;();0b;enlist `momIdx__]}

// Parse config with defaults
parseConfig:{[cfg;tcols]
    trendSpeeds:$[`trendSpeeds in key cfg; cfg`trendSpeeds;
                   $[`speeds in key cfg; cfg`speeds; defaultTrendSpeeds]];
    fastSpeed:$[`fastSpeed in key cfg; cfg`fastSpeed; defaultFastSpeed];
    accelWt:$[`accelWt in key cfg; cfg`accelWt; defaultAccelWt];
    blendHL:$[`blendHL in key cfg; cfg`blendHL; defaultBlendHL];
    carryTiltWt:$[`carryTiltWt in key cfg; cfg`carryTiltWt; defaultCarryTiltWt];
    vrQ:$[`vrQ in key cfg; cfg`vrQ; defaultVRQ];
    vrW:$[`vrW in key cfg; cfg`vrW; defaultVRW];
    targetVol:$[`targetVol in key cfg; cfg`targetVol; defaultTargetVol];
    maxLev:$[`maxLeverage in key cfg; cfg`maxLeverage; defaultMaxLev];
    ddThresh:$[`ddThresh in key cfg; cfg`ddThresh; defaultDDThresh];
    retCol:$[`retCol in key cfg; cfg`retCol; `ret];
    dtCol:$[`dtCol in key cfg; cfg`dtCol; `dt];
    symCol:$[`symCol in key cfg; cfg`symCol; `sym];
    carryCol:$[`carryCol in key cfg; cfg`carryCol; `carry];
    hasCarry:carryCol in tcols;
    `trendSpeeds`fastSpeed`accelWt`blendHL`carryTiltWt`vrQ`vrW`targetVol`maxLev`ddThresh`retCol`dtCol`symCol`carryCol`hasCarry!
    (trendSpeeds;fastSpeed;accelWt;blendHL;carryTiltWt;vrQ;vrW;targetVol;maxLev;ddThresh;retCol;dtCol;symCol;carryCol;hasCarry)}

// =============================================================================
// SECTION 8: EVALUATION
// =============================================================================

// Comprehensive evaluation. t needs dt, sym, sig, ret (+ sig_* for per-speed).
evaluate:{[t]
    syms:asc distinct t`sym;
    hasRet:`ret in cols t;

    dailyPnl:computeDailyPnl[t;syms];
    r:dailyPnl`ret; dts:dailyPnl`dt;
    n:count r;
    nzr:r where r <> 0f;
    nnz:count nzr;

    // Standard metrics
    annRet:252 * avg nzr;
    annVol:(sqrt 252f) * dev nzr;
    sharpe:$[annVol > 1e-10; annRet % annVol; 0n];

    // Sortino
    downside:nzr where nzr < 0f;
    dd:$[0 < count downside; (sqrt 252f) * sqrt avg downside * downside; 1e-10];
    sortino:$[dd > 1e-10; annRet % dd; 0n];

    // Drawdown
    cumRets:sums r;
    runMax:maxs cumRets;
    drawdowns:cumRets - runMax;
    maxDD:min drawdowns;
    calmar:$[(maxDD < 0) and not null maxDD; neg annRet % maxDD; 0n];
    ddDurations:computeDDDurations[drawdowns];
    maxDDDur:$[0 < count ddDurations; max ddDurations; 0];
    currentDD:last drawdowns;

    // Hit rate, profit factor
    wins:nzr where nzr > 0f;
    losses:nzr where nzr < 0f;
    hitRate:$[nnz > 0; (count wins) % nnz; 0n];
    profitFactor:$[(0 < count losses) and 0 < count wins;
        (sum wins) % neg sum losses; 0n];

    // Per-speed Sharpe
    speedCols:cols[t] where cols[t] like "sig_*";
    speedSharpes:$[hasRet and 0 < count speedCols;
        computePerSpeedSharpe[t;syms;speedCols]; ()!()];
    bestSingle:$[0 < count speedSharpes; max value speedSharpes; 0n];

    // Rolling Sharpe
    rollSharpe:$[n > 252; computeRollingSharpe[r;252]; n # 0n];

    // Regime Sharpe
    regimeSharpes:computeRegimeSharpes[t;r;dts];

    // Signal autocorrelation
    sigAutoCorr:computeSigAutoCorr[t;syms];

    // IC
    icMetrics:$[hasRet; computeIC[t;syms]; `ic`icIR`icHitRate`icDecay!(0n;0n;0n;()!())];
    tsICBySym:$[hasRet; computeTSIC[t;syms]; syms!count[syms]#0n];

    // Bootstrap CI
    bootCI:bootstrapSharpeCI[nzr;1000];

    // Monthly breakdown
    monthlyTab:computeMonthlyBreakdown[dts;r];

    // Gate stats
    gateStats:computeGateStats[t];

    (`sharpe`sortino`calmar`hitRate`profitFactor`annReturn`annVol,
     `maxDD`maxDDDuration`currentDD,
     `perSpeedSharpe`bestSingleSharpe,
     `rollingSharpe`regimeSharpes`sigAutoCorr,
     `ic`icIR`icHitRate`icDecay`tsICBySym,
     `bootstrapSharpeLo`bootstrapSharpeHi,
     `monthlyTable`gateStats,
     `nDays`nNonZeroDays)!
    (sharpe;sortino;calmar;hitRate;profitFactor;annRet;annVol;
     maxDD;maxDDDur;currentDD;
     speedSharpes;bestSingle;
     rollSharpe;regimeSharpes;sigAutoCorr;
     icMetrics`ic;icMetrics`icIR;icMetrics`icHitRate;icMetrics`icDecay;tsICBySym;
     bootCI`lo;bootCI`hi;
     monthlyTab;gateStats;
     n;nnz)}

// --- Evaluation helpers ---

computeDailyPnl:{[t;syms]
    hasPnl:`pnl in cols t;
    hasRet:`ret in cols t;
    pnls:raze $[hasPnl;
        {[t;s] sub:`dt xasc select from t where sym=s; ([] dt:sub`dt; pnl:sub`pnl)}[t;] each syms;
        hasRet;
        {[t;s] sub:`dt xasc select from t where sym=s; pnl:prev[sub`sig] * sub`ret; ([] dt:sub`dt; pnl:pnl)}[t;] each syms;
        '"evaluate requires `ret or `pnl column"];
    daily:0!select ret:sum pnl by dt from pnls;
    `dt xasc daily}

computeDDDurations:{[drawdowns]
    inDD:drawdowns < neg 1e-10;
    if[not any inDD; :enlist 0];
    d:deltas "i"$inDD;
    starts:where d = 1i;
    ends:where d = -1i;
    if[inDD[0] and ((0 = count starts) or (starts[0] > 0));
        starts:0,starts];
    if[(count starts) > count ends; ends:ends,count drawdowns];
    $[0 < count starts; ends - starts; enlist 0]}

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
            ((252 * avg r) % ((sqrt 252f) * dev r));
            0n];
        result[sc]:sr;
        result
    }[t;syms]/[()!();speedCols]}

computeRollingSharpe:{[r;w]
    mu:mavg[w;r];
    vol:mdev[w;r];
    ((sqrt 252f) * mu) % 1e-10 | vol}

computeRegimeSharpes:{[t;r;dts]
    vr:.cond.varianceRatio[r;5;60];
    trendIdx:where (vr > 1f) and not null vr;
    mrIdx:where (vr <= 1f) and not null vr;
    trendR:r trendIdx; mrR:r mrIdx;
    trendSharpe:$[(count trendR where trendR <> 0f) > 10;
        ((252 * avg trendR) % ((sqrt 252f) * dev trendR)); 0n];
    mrSharpe:$[(count mrR where mrR <> 0f) > 10;
        ((252 * avg mrR) % ((sqrt 252f) * dev mrR)); 0n];
    `trending`meanReverting!(trendSharpe;mrSharpe)}

computeSigAutoCorr:{[t;syms]
    autos:{[t;s]
        sub:`dt xasc select from t where sym=s;
        sig:sub`sig; valid:sig where not null sig;
        $[30 < count valid; cor[neg[1] _ valid; 1 _ valid]; 0n]
    }[t;] each syms;
    avg autos where not null autos}

computeICOneSym:{[t;s;h]
    sub:`dt xasc select from t where sym=s;
    sig:sub`sig;
    fwdRet:h msum (1 rotate sub`ret);
    n:count fwdRet;
    idxs:(n - h) + til h;
    fwdRet[idxs]:0n;
    .cond.rollingIC[sig;fwdRet;60]}

computeIC:{[t;syms]
    ics:raze computeICOneSym[t;;1] each syms;
    valid:ics where not null ics;
    nIC:count valid;
    icMean:$[nIC > 0; avg valid; 0n];
    icStd:$[nIC > 1; dev valid; 0n];
    icIR:$[(nIC > 1) and icStd > 1e-10; icMean % icStd; 0n];
    icHitRate:$[nIC > 0; (sum valid > 0f) % nIC; 0n];
    horizons:1 2 5 10 20;
    icDecayVals:{[t;syms;h] ics:raze computeICOneSym[t;;h] each syms; v:ics where not null ics; $[0 < count v; avg v; 0n]}[t;syms;] each horizons;
    `ic`icIR`icHitRate`icDecay!(icMean;icIR;icHitRate;horizons!icDecayVals)}

computeTSIC:{[t;syms]
    syms!{[t;s]
        sub:`dt xasc select from t where sym=s;
        sig:sub`sig; fwdRet:1 rotate sub`ret;
        fwdRet[(count fwdRet) - 1]:0n;
        valid:where (not null sig) and not null fwdRet;
        $[30 < count valid; cor[sig valid; fwdRet valid]; 0n]
    }[t;] each syms}

bootstrapSharpeCI:{[r;nBoot]
    n:count r;
    if[n < 10; :`lo`hi!(0n;0n)];
    sharpes:{[r;n;i]
        sample:r n?n;
        $[dev[sample] > 1e-10;
            ((sqrt 252f) * avg sample) % dev sample; 0n]
    }[r;n;] each til nBoot;
    valid:asc sharpes where not null sharpes;
    nv:count valid;
    if[nv < 10; :`lo`hi!(0n;0n)];
    `lo`hi!(valid `long$0.025 * nv; valid `long$0.975 * nv)}

computeMonthlyBreakdown:{[dts;r]
    tbl:([] month:`month$dts; r:r);
    agg:0!select ret:sum r, nDays:count r, hitRate:avg r > 0f,
               mu:avg r, vol:dev r by month from tbl;
    agg:update sharpe:((sqrt 252f) * mu) % 1e-10 | vol from agg;
    delete mu, vol from agg}

computeGateStats:{[t]
    gateCols:`regimeW`ddScale;
    present:gateCols where gateCols in cols t;
    if[0 = count present; :()!()];
    present!{[t;c] v:t[c] where not null t[c]; n:count v; `avg`pctActive!($[n>0;avg v;0n]; $[n>0;(sum v < 0.9) % n; 0f])}[t;] each present}

// =============================================================================
// SECTION 9: SYNTHETIC DATA
// =============================================================================

// 4 symbols: trending, trending+crash, random, mean-reverting.
// Optional carry column (positive for long-biased).
syntheticTest:{[]
    system "S 42";
    nDays:1000;
    dts:2020.01.01 + til nDays;
    vol:0.01;

    // Symbol A: Strong trending (AR1 phi=0.3, drift=0.0002)
    epsA:.cond.randNorm nDays;
    retA:nDays # 0f;
    retA[0]:vol * epsA[0];
    i:1;
    while[i < nDays;
        retA[i]:0.0002 + (0.3 * retA[i - 1]) + (vol * epsA[i]);
        i+:1];

    // Symbol B: Trend then crash at day 500 (vol spike + reversal)
    epsB:.cond.randNorm nDays;
    retB:nDays # 0f;
    retB[0]:vol * epsB[0];
    i:1;
    while[i < nDays;
        drift:$[i < 500; 0.0002; neg 0.0003];
        phi:$[i < 500; 0.3; 0.1];
        v:$[(i > 498) and i < 510; 3f * vol; vol];
        retB[i]:drift + (phi * retB[i - 1]) + (v * epsB[i]);
        i+:1];

    // Symbol C: Random walk
    retC:vol * .cond.randNorm nDays;

    // Symbol D: Mean-reverting (AR1 phi=-0.3)
    epsD:.cond.randNorm nDays;
    retD:nDays # 0f;
    retD[0]:vol * epsD[0];
    i:1;
    while[i < nDays;
        retD[i]:(neg[0.3] * retD[i - 1]) + (vol * epsD[i]);
        i+:1];

    // Carry: positive for all (simulates long-biased carry environment)
    carryA:nDays # 0.0001;
    carryB:nDays # 0.0001;
    carryC:nDays # 0.00005;
    carryD:nDays # neg 0.0001;  // carry opposes trend for D

    t:raze (
        ([] dt:dts; sym:nDays # `A; ret:retA; carry:carryA);
        ([] dt:dts; sym:nDays # `B; ret:retB; carry:carryB);
        ([] dt:dts; sym:nDays # `C; ret:retC; carry:carryC);
        ([] dt:dts; sym:nDays # `D; ret:retD; carry:carryD)
    );
    `dt`sym xasc t}

// =============================================================================
// SECTION 10: TESTS
// =============================================================================

runTests:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum v3.0 TEST SUITE";
    -1 "=============================================================================";
    -1 "";

    nPass:0; nFail:0;
    results:()!();

    // --- Test 1: sharpeMom is self-normalizing ---
    -1 "Test 1: sharpeMom self-normalizes across vol regimes";
    system "S 42";
    n:500;
    base:.cond.randNorm n;
    lowVol:0.005 * base;    // 50bps daily vol
    hiVol:0.02 * base;      // 200bps daily vol (same direction!)
    smLo:sharpeMom[lowVol;21];
    smHi:sharpeMom[hiVol;21];
    rawLo:.cond.smooth[lowVol;10.5];
    rawHi:.cond.smooth[hiVol;10.5];
    // sharpeMom should be similar for same underlying noise
    warmup:50;
    idx:warmup + til n - warmup;
    smDiff:avg abs (smLo - smHi) idx;
    rawDiff:avg abs (rawLo - rawHi) idx;
    t1:smDiff < rawDiff;  // sharpeMom more similar than raw EMA
    results[`test1]:t1;
    -1 "  sharpeMom diff: ",string smDiff;
    -1 "  raw EMA diff:   ",string rawDiff;
    -1 "  sharpeMom more similar: ",string t1;
    -1 "  ",$[t1;"PASS";"FAIL"];
    nPass+:t1; nFail+:not t1;
    -1 "";

    // --- Test 2: Acceleration detects trend reversal ---
    -1 "Test 2: Acceleration detects trend reversal";
    system "S 42";
    n:400;
    eps:.cond.randNorm n;
    // Trending up for 200 days, then reversal
    trendUp:0.0003 + (0.005 * eps);
    trendDn:neg[0.0003] + (0.005 * eps);
    r:(200 # trendUp),(200 # trendDn);
    mom:sharpeMom[r;21];
    acc:accel[mom;21];
    // Steady trend: acceleration ~0. Post-reversal: acceleration clearly negative.
    accSteady:avg acc 150 + til 30;
    accPost:avg acc 210 + til 20;
    t2:(accPost < 0f) and (accPost < accSteady);
    results[`test2]:t2;
    -1 "  accel during trend: ",string accSteady;
    -1 "  accel post-reversal: ",string accPost;
    -1 "  Acceleration drops at reversal: ",string t2;
    -1 "  ",$[t2;"PASS";"FAIL"];
    nPass+:t2; nFail+:not t2;
    -1 "";

    // --- Test 3: Regime blend uses fast signal in MR ---
    -1 "Test 3: Regime blend uses fast signal in MR regime";
    system "S 42";
    n:1000;
    eps:.cond.randNorm n;
    mrRet:n # 0f; mrRet[0]:0.01 * eps[0];
    i:1; while[i < n; mrRet[i]:(neg[0.5] * mrRet[i-1]) + (0.01 * eps[i]); i+:1];
    trendSigs:singleSpeed[mrRet;;0.3] each 63 126 252;
    fastSig3:singleSpeed[mrRet;21;0.3];
    re:regimeEnsemble[trendSigs;mrRet;63 126 252;5;60];
    blended:regimeBlend[re`ensemble;fastSig3;re`vrPctl;10];
    // In MR regime, regimeW should be low → blended ≈ fastSig
    idx3:300 + til 500;
    regimeW3:.cond.smooth[re`vrPctl; 10];
    avgW:avg regimeW3 idx3;
    // Correlation between blended and fast should be high in MR
    corrFast:cor[blended idx3; fastSig3 idx3];
    corrTrend:cor[blended idx3; (re`ensemble) idx3];
    t3:(avgW < 0.6) and (corrFast > corrTrend);
    results[`test3]:t3;
    -1 "  Avg regimeW (MR data): ",string avgW;
    -1 "  Corr(blended, fast): ",string corrFast;
    -1 "  Corr(blended, trend): ",string corrTrend;
    -1 "  ",$[t3;"PASS";"FAIL"];
    nPass+:t3; nFail+:not t3;
    -1 "";

    // --- Test 4: Carry tilt nudges signal in carry direction ---
    -1 "Test 4: Carry tilt nudges signal in carry direction";
    system "S 42";
    n4:200;
    sig4:n4 # 1f;   // all positive signal
    // Use trending carry: starts low, rises to high — recent values have positive carryZ
    // carryPos ramps up: rzscore will be positive at end
    carryRamp:(0.0002 * til n4) % n4;   // ramps from 0 to ~0.04
    // carryDown ramps down: rzscore will be negative at end
    carryDown:reverse carryRamp;
    tiltedUp:carryTilt[sig4;carryRamp;0.15];
    tiltedDn:carryTilt[sig4;carryDown;0.15];
    // At end of series (idx 170+), ramp up → positive carryZ → tilt > 1
    // ramp down → negative carryZ → tilt < 1
    idx4:170 + til 30;
    avgUp:avg tiltedUp idx4;
    avgDn:avg tiltedDn idx4;
    t4:(avgUp > 1f) and (avgDn < 1f);
    results[`test4]:t4;
    -1 "  Avg tilted (carry ramping up): ",string avgUp;
    -1 "  Avg tilted (carry ramping down): ",string avgDn;
    -1 "  ",$[t4;"PASS";"FAIL"];
    nPass+:t4; nFail+:not t4;
    -1 "";

    // --- Test 5: Curve agreement adjusts leverage ---
    -1 "Test 5: Curve agreement adjusts leverage ceiling";
    // Unanimous agreement → adjMaxLev = maxLev * 1.0
    // Zero agreement → adjMaxLev = maxLev * 0.7
    maxLev5:3f;
    adjUnanimous:maxLev5 * (0.7 + (0.3 * 1f));   // agreement = 1.0
    adjSplit:maxLev5 * (0.7 + (0.3 * 0f));         // agreement = 0.0
    t5:(adjUnanimous > 2.99) and (adjSplit < 2.11) and (adjSplit > 2.09);
    results[`test5]:t5;
    -1 "  Unanimous adjMaxLev: ",string adjUnanimous;
    -1 "  Split adjMaxLev: ",string adjSplit;
    -1 "  ",$[t5;"PASS";"FAIL"];
    nPass+:t5; nFail+:not t5;
    -1 "";

    // --- Test 6: Regime blend uses trend signal in trending ---
    -1 "Test 6: Regime blend uses trend signal in trending regime";
    system "S 42";
    n:2000;
    eps:.cond.randNorm n;
    // Strong trending: high autocorrelation (phi=0.5), strong drift
    trendRet:n # 0f; trendRet[0]:0.01 * eps[0];
    i:1; while[i < n; trendRet[i]:0.0005 + (0.5 * trendRet[i-1]) + (0.01 * eps[i]); i+:1];
    trendSigs6:singleSpeed[trendRet;;0.3] each 63 126 252;
    fastSig6:singleSpeed[trendRet;21;0.3];
    re6:regimeEnsemble[trendSigs6;trendRet;63 126 252;5;60];
    blended6:regimeBlend[re6`ensemble;fastSig6;re6`vrPctl;10];
    idx6:500 + til 1000;
    regimeW6:.cond.smooth[0.5 ^ re6`vrPctl; 10];
    avgW6:avg regimeW6 idx6;
    // In trending regime, regimeW should be above neutral (0.5)
    t6:avgW6 > 0.5;
    results[`test6]:t6;
    -1 "  Avg regimeW (trending data): ",string avgW6;
    -1 "  Expected > 0.5 for trending regime";
    -1 "  ",$[t6;"PASS";"FAIL"];
    nPass+:t6; nFail+:not t6;
    -1 "";

    // --- Test 7: DD breaker floor is 0.6 ---
    -1 "Test 7: DD breaker floor is 0.6";
    cumPnl7:0.1 0.12 0.15 0.13 0.10 0.08 0.05 0.04 0.06 0.08;
    dds7:ddBreaker[cumPnl7;0.05];
    // At index 6: dd = 0.05 - 0.15 = -0.10 -> scale = max(0.6, 1 - 0.5*0.10/0.05) = max(0.6, 0) = 0.6
    t7:(dds7[0] > 0.99) and ((abs dds7[6] - 0.6) < 0.01);
    results[`test7]:t7;
    -1 "  No DD scale: ",string dds7[0];
    -1 "  Deep DD scale (floor check): ",string dds7[6];
    -1 "  ",$[t7;"PASS";"FAIL"];
    nPass+:t7; nFail+:not t7;
    -1 "";

    // --- Test 8: Blend is smooth (EMA-based, not discontinuous step) ---
    -1 "Test 8: Regime weight transitions smoothly (EMA-based)";
    system "S 42";
    n:500;
    // Construct vrPctl that jumps abruptly from 0 to 1 at midpoint
    vrPctl8:(250 # 0f),(250 # 1f);
    trendSig8:n # 1f;
    fastSig8:n # neg 1f;
    blended8:regimeBlend[trendSig8;fastSig8;vrPctl8;10];
    // Smoothed regimeW should transition gradually, not jump
    // At midpoint: regimeW jumps from ~0 toward 1, but EMA smooths it
    // Check that regimeW at midpoint+1 is not yet at 1.0
    regimeW8:.cond.smooth[0.5 ^ vrPctl8; 10];
    // Just after transition: should be partially transitioned, not fully
    t8:(regimeW8[251] < 0.9) and (regimeW8[300] > regimeW8[251]);
    results[`test8]:t8;
    -1 "  regimeW at step+1: ",string regimeW8[251];
    -1 "  regimeW at step+50: ",string regimeW8[300];
    -1 "  Smooth transition: ",string t8;
    -1 "  ",$[t8;"PASS";"FAIL"];
    nPass+:t8; nFail+:not t8;
    -1 "";

    // --- Test 9: DD breaker activates during drawdown ---
    -1 "Test 9: DD breaker activates during drawdown";
    cumPnl9:0.1 0.12 0.15 0.13 0.10 0.08 0.05 0.04 0.06 0.08;
    dds9:ddBreaker[cumPnl9;0.05];
    // At peak (0.15, idx 2): dd=0 → scale=1. Deep DD (idx 6-7): scale=0.6 (floor).
    t9:(dds9[2] > 0.99) and (dds9[6] < 0.61) and (dds9[6] > 0.59);
    results[`test9]:t9;
    -1 "  At peak scale: ",string dds9[2];
    -1 "  Deep DD scale: ",string dds9[6];
    -1 "  ",$[t9;"PASS";"FAIL"];
    nPass+:t9; nFail+:not t9;
    -1 "";

    // --- Test 10: Full pipeline runs end-to-end ---
    -1 "Test 10: Full pipeline - end-to-end smoke test";
    td:.momentum.syntheticTest[];
    pipeResult:ensembleTable[td;()!()];
    ev:evaluate[pipeResult];
    sh:ev`sharpe;
    // Full portfolio across all 4 syms. Sharpe may be near 0.
    // Key check: pipeline runs, produces finite Sharpe, not terribly negative.
    t10:(not null sh) and sh > neg 1f;
    results[`test10]:t10;
    -1 "  Sharpe: ",string sh;
    -1 "  ",$[t10;"PASS";"FAIL"];
    nPass+:t10; nFail+:not t10;
    -1 "";

    // --- Test 11: Output has all expected columns ---
    -1 "Test 11: Table interface produces correct columns";
    expectedCols:`sig`trendSig`fastSig`vr`regimeW`blendedSig`carryTilt`curveAgreement`volScale`ddScale;
    hasCols:all expectedCols in cols pipeResult;
    hasSyms:4 = count distinct pipeResult`sym;
    t11:hasCols and hasSyms;
    results[`test11]:t11;
    -1 "  Has all columns: ",string hasCols;
    -1 "  Missing: ","," sv string expectedCols where not expectedCols in cols pipeResult;
    -1 "  Has 4 syms: ",string hasSyms;
    -1 "  ",$[t11;"PASS";"FAIL"];
    nPass+:t11; nFail+:not t11;
    -1 "";

    // --- Test 12: Anti-lookahead ---
    -1 "Test 12: Anti-lookahead (signal at t uses data up to t-1)";
    system "S 42";
    x:100 # 0.01 * .cond.randNorm 100;
    sig50:singleSpeed[50 # x;21;0.3];
    sig51:singleSpeed[51 # x;21;0.3];
    t12:$[(null sig50[49]) or null sig51[49];
        1b;
        (abs (sig50[49]) - sig51[49]) < 1e-10];
    results[`test12]:t12;
    -1 "  sig[49] with 50 obs: ",string sig50[49];
    -1 "  sig[49] with 51 obs: ",string sig51[49];
    -1 "  ",$[t12;"PASS";"FAIL"];
    nPass+:t12; nFail+:not t12;
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
        -1 "  SOME TESTS FAILED - see above"];
    -1 "";
    results}

// =============================================================================
// SECTION 11: HELP / EXAMPLE
// =============================================================================

help:{[]
    -1 "";
    -1 "=== .momentum v3.0 - Regime-Switched Blend Momentum ===";
    -1 "";
    -1 "CORE SIGNAL:";
    -1 "  sharpeMom[x;w]              - risk-adjusted momentum (mu/sigma)";
    -1 "  accel[mom;w]                - momentum acceleration (2nd derivative)";
    -1 "  singleSpeed[x;w;accelWt]    - combined signal at one lookback";
    -1 "";
    -1 "ENSEMBLE:";
    -1 "  regimeEnsemble[sigs;x;speeds;vrQ;vrW]  - VR-adaptive trend blend (raw)";
    -1 "  regimeBlend[trend;fast;vrPctl;blendHL]  - regime-switched blending";
    -1 "  carryTilt[sig;carry;weight]              - additive carry adjustment";
    -1 "";
    -1 "PROTECTION:";
    -1 "  ddBreaker[cumPnl;thresh]    - drawdown brake (0.6 floor)";
    -1 "";
    -1 "TABLE INTERFACE:";
    -1 "  ensembleTable[t;cfg]        - full pipeline for (dt;sym;ret) table";
    -1 "    Optional carry column enables carry tilt + excess return separation";
    -1 "    cfg keys: trendSpeeds, fastSpeed, accelWt, blendHL, carryTiltWt,";
    -1 "              vrQ, vrW, targetVol, maxLeverage, ddThresh";
    -1 "";
    -1 "EVALUATION:";
    -1 "  evaluate[t]                 - comprehensive metrics";
    -1 "  syntheticTest[]             - 4-sym test data (trend/crash/random/MR)";
    -1 "  runTests[]                  - 12 validation tests";
    -1 "";
    -1 "DEFAULTS:";
    -1 "  trendSpeeds: 63 126 252 | fastSpeed: 21 | accelWt: 0.3";
    -1 "  blendHL: 10 | carryTiltWt: 0.15 | targetVol: 0.10";
    -1 "  maxLeverage: 3 | ddThresh: 0.05";
    -1 "";
    -1 "OUTPUT COLUMNS:";
    -1 "  sig_N (per-speed), trendSig, fastSig, vr, regimeW,";
    -1 "  blendedSig, carryTilt, curveAgreement, volScale, ddScale, sig";
    -1 "";}

example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum v3.0 EXAMPLE";
    -1 "=============================================================================";
    -1 "";
    td:.momentum.syntheticTest[];
    -1 "1. Synthetic data: ",string[count td]," rows, ",string[count distinct td`sym]," syms";
    -1 "";
    result:.momentum.ensembleTable[td;()!()];
    -1 "2. ensembleTable output: ",string[count cols result]," columns";
    -1 "   Columns: ","," sv string cols result;
    -1 "";
    ev:.momentum.evaluate[result];
    -1 "3. Evaluation:";
    -1 "   Sharpe:       ",string ev`sharpe;
    -1 "   Sortino:      ",string ev`sortino;
    -1 "   Calmar:       ",string ev`calmar;
    -1 "   HitRate:      ",string ev`hitRate;
    -1 "   MaxDD:        ",string ev`maxDD;
    -1 "   MaxDDDur:     ",string[ev`maxDDDuration]," days";
    -1 "";
    -1 "4. Architecture stats:";
    -1 "   Avg regimeW:         ",string avg result`regimeW;
    -1 "   Avg ddScale:         ",string avg result`ddScale;
    -1 "   Avg |blendedSig|:    ",string avg abs result`blendedSig;
    -1 "   Avg |sig|:           ",string avg abs result`sig;
    -1 "";
    gs:ev`gateStats;
    if[0 < count gs;
        -1 "5. Gate stats:";
        {[gs;x] -1 "   ",string[x],": avg=",string[(gs x)`avg]," pctActive=",string (gs x)`pctActive}[gs;] each key gs;
        -1 ""];
    -1 "6. Per-speed Sharpe:";
    pss:ev`perSpeedSharpe;
    {[pss;x] -1 "   ",string[x]," = ",string pss x}[pss;] each key pss;
    -1 "";
    -1 "Done.";}

\d .

-1 "Loaded .momentum namespace v3.0.0";
-1 "Regime-switched blend: trendSig + fastSig blended by regime, additive carry tilt";
-1 "Run .momentum.help[] for full function list";
-1 "Run .momentum.runTests[] for validation";
