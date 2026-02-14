// =============================================================================
// MOMENTUM SIGNAL LIBRARY v2.0
// =============================================================================
// Risk-adjusted multi-speed momentum with regime gating, carry alignment,
// curve confirmation, crash protection, and portfolio vol targeting.
//
// Version: 2.0.0
// Dependencies: cond.q, kdbtools.q
// Optional:     alphalab.q (for alphaEval bridge)
//
// Pipeline:
//   1. Sharpe momentum (mu/sigma) at each speed — self-normalizing
//   2. Momentum acceleration (2nd derivative) overlay
//   3. Regime-adaptive ensemble (VR-based speed weighting + scaling)
//   4. Carry alignment gate (optional, if carry column present)
//   5. Curve confirmation (cross-contract directional agreement)
//   6. Extreme taper at cumulative z-score extremes
//   7. Crash protection (vol break + drawdown circuit breaker)
//   8. Vol targeting with leverage cap
//
// All arithmetic uses explicit parentheses (Q right-to-left).

\d .momentum

// =============================================================================
// CONFIGURATION
// =============================================================================

defaultSpeeds:21 63 126
defaultAccelWt:0.3
defaultVRQ:5
defaultVRW:60
defaultTargetVol:0.10
defaultMaxLev:3f
defaultDDThresh:0.05
defaultVolBreakThresh:2.0
defaultExtremeZ:2.0
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
// Uses rolling percentile of VR (robust to VR level bias).
// In trending regimes: upweight slow speeds, full size.
// In mean-reverting regimes: scale toward zero.
//
// sigs:   list of signal vectors (one per speed)
// x:      return series (for VR computation)
// speeds: list of lookback windows
// vrQ:    VR aggregation horizon (default 5)
// vrW:    VR rolling window (default 60)
// Returns: dict `ensemble`vr`regimeScale
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
    // Weighted sum
    ens:nObs # 0f;
    i:0;
    while[i < n;
        ens:ens + (sigs[i] * tilts[i] % tiltSum);
        i+:1];
    `ensemble`vr`regimeScale!(ens * regimeScale; vr; regimeScale)}

// =============================================================================
// SECTION 4: CARRY ALIGNMENT GATE
// =============================================================================

// Gate multiplier based on carry direction alignment.
// 1.0 when signal and carry agree; 0.3 when opposing.
//
// sig:   momentum signal vector
// carry: daily carry return vector
// Returns: multiplier vector (0.3 or 1.0)
carryGateVal:{[sig;carry]
    carryDir:signum 0f ^ prev carry;
    sigDir:signum sig;
    opposing:(sigDir <> 0f) and (carryDir <> 0f) and (sigDir <> carryDir);
    1f - (0.7 * opposing)}

// =============================================================================
// SECTION 5: CURVE CONFIRMATION
// =============================================================================

// Cross-contract directional agreement.
// |mean(sign(sig))| across contracts at each time step.
// Mapped to [0.3, 1.0]: unanimous = 1.0, max disagreement = 0.3.
//
// sigs: list of signal vectors (one per sym, aligned by index)
// Returns: multiplier vector
curveConfirmVec:{[sigs]
    n:count sigs;
    if[n < 2; :count[first sigs] # 1f];
    dirs:{signum x} each sigs;
    sumDirs:dirs[0];
    i:1;
    while[i < n; sumDirs:sumDirs + dirs[i]; i+:1];
    agreement:abs sumDirs % "f"$n;
    0.3 + (0.7 * agreement)}

// =============================================================================
// SECTION 6: EXTREME TAPER
// =============================================================================

// Fade signal when cumulative returns reach extreme z-scores.
// After 200bps of yield decline in a year, continuation probability drops.
//
// x:      return series
// w:      window for cumulative z-score (252 = 1Y)
// thresh: z-score threshold before taper begins (2.0)
// Returns: multiplier (0 to 1). 1.0 when |cumZ| < thresh.
extremeTaper:{[x;w;thresh]
    cumX:sums x;
    cumZ:.cond.rzscore[w;cumX];
    excess:0f | (abs cumZ) - thresh;
    exp neg (excess * excess) % 2f}

// =============================================================================
// SECTION 7: CRASH PROTECTION
// =============================================================================

// Vol break: fast vol / slow vol ratio.
// Scales down linearly from 1.0 at thresh to 0.0 at thresh+1.
//
// x:       return series
// fastHL:  fast vol EWMA halflife (5)
// slowHL:  slow vol EWMA halflife (63)
// thresh:  ratio threshold to begin scale-down (2.0)
volBreak:{[x;fastHL;slowHL;thresh]
    fastVol:sqrt 1e-10 | .cond.smooth[x * x; fastHL];
    slowVol:sqrt 1e-10 | .cond.smooth[x * x; slowHL];
    ratio:prev fastVol % slowVol;
    0f | 1f & (1f - (0f | ratio - thresh))}

// Drawdown circuit breaker.
// 1.0 at no drawdown, 0.5 at -thresh, 0.25 floor.
//
// cumPnl: cumulative P&L vector
// thresh: drawdown threshold (0.05 = 5%)
ddBreaker:{[cumPnl;thresh]
    peak:maxs cumPnl;
    dd:cumPnl - peak;
    absDd:0f | neg dd;
    0.25 | 1f - ((0.5 * absDd) % thresh)}

// =============================================================================
// SECTION 8: TABLE INTERFACE
// =============================================================================

// Main entry point. Compute momentum ensemble for multi-sym table.
//
// t:   table with (dt; sym; ret) minimum. Optional: carry column.
// cfg: config dict. All keys optional:
//   `speeds          - lookback windows (default: 21 63 126)
//   `accelWt         - acceleration weight (default: 0.3)
//   `vrQ             - VR aggregation horizon (default: 5)
//   `vrW             - VR rolling window (default: 60)
//   `targetVol       - annualized vol target (default: 0.10)
//   `maxLeverage     - max vol scale factor (default: 3)
//   `ddThresh        - drawdown threshold (default: 0.05)
//   `volBreakThresh  - vol break threshold (default: 2.0)
//   `extremeZ        - cumulative z threshold (default: 2.0)
//   `retCol          - return column name (default: `ret)
//   `dtCol           - date/time column name (default: `dt)
//   `symCol          - symbol column name (default: `sym)
//   `carryCol        - carry column name (default: `carry)
//
// Returns: table with columns:
//   sig_N (per-speed), sigRaw, vr, regimeScale, carryGate,
//   curveConfirm, extremeTaper, volBreak, volScale, ddScale, sig
ensembleTable:{[t;cfg]
    // Parse config
    pcfg:parseConfig[cfg;cols t];
    speeds:pcfg`speeds; symCol:pcfg`symCol; dtCol:pcfg`dtCol;

    // Phase 1: per-sym signal computation
    t:@[t;`momIdx__;:;til count t];
    grp:group t symCol;
    syms:key grp;

    p1:{[t;pcfg;idx]
        dtCol:pcfg`dtCol; retCol:pcfg`retCol;
        carryCol:pcfg`carryCol; hasCarry:pcfg`hasCarry;
        speeds:pcfg`speeds; aw:pcfg`accelWt;
        vrQ:pcfg`vrQ; vrW:pcfg`vrW;
        ezT:pcfg`extremeZ; vbT:pcfg`volBreakThresh;
        sub:dtCol xasc t idx;
        r:"f"$sub retCol;
        // Excess returns if carry available
        x:$[hasCarry; r - (0f ^ prev "f"$sub carryCol); r];
        // Per-speed signals
        sigs:singleSpeed[x;;aw] each speeds;
        i:0;
        while[i < count speeds;
            sub:@[sub;`$"sig_",string speeds i;:;sigs i];
            i+:1];
        // Regime ensemble
        re:regimeEnsemble[sigs;x;speeds;vrQ;vrW];
        sub:@[sub;`sigRaw;:;re`ensemble];
        sub:@[sub;`vr;:;re`vr];
        sub:@[sub;`regimeScale;:;re`regimeScale];
        // Extreme taper
        sub:@[sub;`extremeTaper;:;extremeTaper[x;252;ezT]];
        // Vol break
        sub:@[sub;`volBreak;:;volBreak[x;5;63;vbT]];
        // Carry gate
        sub:@[sub;`carryGate;:;$[hasCarry;
            carryGateVal[re`ensemble; "f"$sub carryCol];
            (count sub) # 1f]];
        sub}[t;pcfg;];

    t:raze p1 each value grp;
    t:`momIdx__ xasc t;

    // Phase 2: curve confirmation
    nSyms:count syms;
    if[nSyms > 1;
        [dtVals:t dtCol;
         dirVals:signum t`sigRaw;
         grpDt:group dtVals;
         meanDirs:(key grpDt)!{[dv;idx] avg dv idx}[dirVals;] each value grpDt;
         cc:0.3 + (0.7 * abs meanDirs dtVals);
         t:@[t;`curveConfirm;:;cc]]];
    if[nSyms < 2;
        t:@[t;`curveConfirm;:;(count t) # 1f]];

    // Phase 3: per-sym gates + vol scale + dd breaker
    grp2:group t pcfg`symCol;

    p3:{[t;pcfg;idx]
        dtCol:pcfg`dtCol; retCol:pcfg`retCol;
        targetVol:pcfg`targetVol; maxLev:pcfg`maxLev; ddT:pcfg`ddThresh;
        sub:dtCol xasc t idx;
        r:"f"$sub retCol;
        // Apply all multiplicative gates
        sigGated:sub[`sigRaw] * sub[`carryGate] * sub[`curveConfirm];
        sigGated:sigGated * sub[`extremeTaper] * sub[`volBreak];
        // Vol targeting
        sigRet:0f ^ (prev[sigGated] * r);
        rollingVol:sqrt 1e-10 | .cond.smooth[sigRet * sigRet; 10f];
        annVol:rollingVol * sqrt 252f;
        sf:prev targetVol % (1e-6 | annVol);
        sf:0f | sf & maxLev;
        sigVS:sigGated * sf;
        sub:@[sub;`volScale;:;sf];
        // Drawdown circuit breaker
        pnl:0f ^ (prev[sigVS] * r);
        cumPnl:sums pnl;
        dds:ddBreaker[cumPnl;ddT];
        sub:@[sub;`ddScale;:;dds];
        sub:@[sub;`sig;:;sigVS * prev dds];
        sub}[t;pcfg;];

    t:raze p3 each value grp2;
    t:`momIdx__ xasc t;
    ![t;();0b;enlist `momIdx__]}

// Parse config with defaults
parseConfig:{[cfg;tcols]
    speeds:$[`speeds in key cfg; cfg`speeds; defaultSpeeds];
    accelWt:$[`accelWt in key cfg; cfg`accelWt; defaultAccelWt];
    vrQ:$[`vrQ in key cfg; cfg`vrQ; defaultVRQ];
    vrW:$[`vrW in key cfg; cfg`vrW; defaultVRW];
    targetVol:$[`targetVol in key cfg; cfg`targetVol; defaultTargetVol];
    maxLev:$[`maxLeverage in key cfg; cfg`maxLeverage; defaultMaxLev];
    ddThresh:$[`ddThresh in key cfg; cfg`ddThresh; defaultDDThresh];
    vbThresh:$[`volBreakThresh in key cfg; cfg`volBreakThresh; defaultVolBreakThresh];
    ezThresh:$[`extremeZ in key cfg; cfg`extremeZ; defaultExtremeZ];
    retCol:$[`retCol in key cfg; cfg`retCol; `ret];
    dtCol:$[`dtCol in key cfg; cfg`dtCol; `dt];
    symCol:$[`symCol in key cfg; cfg`symCol; `sym];
    carryCol:$[`carryCol in key cfg; cfg`carryCol; `carry];
    hasCarry:carryCol in tcols;
    `speeds`accelWt`vrQ`vrW`targetVol`maxLev`ddThresh`volBreakThresh`extremeZ`retCol`dtCol`symCol`carryCol`hasCarry!
    (speeds;accelWt;vrQ;vrW;targetVol;maxLev;ddThresh;vbThresh;ezThresh;retCol;dtCol;symCol;carryCol;hasCarry)}

// =============================================================================
// SECTION 9: EVALUATION
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
    gateCols:`carryGate`curveConfirm`extremeTaper`volBreak`ddScale;
    present:gateCols where gateCols in cols t;
    if[0 = count present; :()!()];
    present!{[t;c] v:t[c] where not null t[c]; n:count v; `avg`pctActive!($[n>0;avg v;0n]; $[n>0;(sum v < 0.9) % n; 0f])}[t;] each present}

// =============================================================================
// SECTION 10: SYNTHETIC DATA
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
// SECTION 11: TESTS
// =============================================================================

runTests:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum v2.0 TEST SUITE";
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

    // --- Test 3: Regime scaling is low for mean-reverting data ---
    -1 "Test 3: Regime scaling is low for mean-reverting data";
    system "S 42";
    n:1000;
    eps:.cond.randNorm n;
    mrRet:n # 0f; mrRet[0]:0.01 * eps[0];
    i:1; while[i < n; mrRet[i]:(neg[0.5] * mrRet[i-1]) + (0.01 * eps[i]); i+:1];
    sigs:singleSpeed[mrRet;;0.3] each 21 63 126;
    re:regimeEnsemble[sigs;mrRet;21 63 126;5;60];
    avgRS:avg (re`regimeScale) 200 + til 800;
    t3:avgRS < 0.6;
    results[`test3]:t3;
    -1 "  Average regime scale (MR data): ",string avgRS;
    -1 "  Below 0.6: ",string t3;
    -1 "  ",$[t3;"PASS";"FAIL"];
    nPass+:t3; nFail+:not t3;
    -1 "";

    // --- Test 4: Carry gate reduces signal when opposing ---
    -1 "Test 4: Carry gate reduces when opposing carry";
    sig:100 # 1f;   // all positive signal
    carryPos:100 # 0.001;
    carryNeg:100 # neg 0.001;
    gAligned:carryGateVal[sig;carryPos];
    gOpposing:carryGateVal[sig;carryNeg];
    // After warmup (prev), check from index 2
    t4:((avg gAligned 2 + til 98) > 0.9) and ((avg gOpposing 2 + til 98) < 0.4);
    results[`test4]:t4;
    -1 "  Aligned gate avg: ",string avg gAligned 2 + til 98;
    -1 "  Opposing gate avg: ",string avg gOpposing 2 + til 98;
    -1 "  ",$[t4;"PASS";"FAIL"];
    nPass+:t4; nFail+:not t4;
    -1 "";

    // --- Test 5: Curve confirmation unanimous = 1.0 ---
    -1 "Test 5: Curve confirmation - unanimous agreement";
    sigs:(100 # 1f; 100 # 2f; 100 # 0.5);  // all positive
    cc:curveConfirmVec sigs;
    t5:(avg cc) > 0.99;
    results[`test5]:t5;
    -1 "  Unanimous CC avg: ",string avg cc;
    -1 "  ",$[t5;"PASS";"FAIL"];
    nPass+:t5; nFail+:not t5;
    -1 "";

    // --- Test 6: Curve confirmation mixed < unanimous ---
    -1 "Test 6: Curve confirmation - mixed directions";
    sigs2:(100 # 1f; 100 # neg 1f; 100 # 0.5);  // 2 pos, 1 neg
    cc2:curveConfirmVec sigs2;
    t6:(avg cc2) < (avg cc);
    results[`test6]:t6;
    -1 "  Mixed CC avg: ",string avg cc2;
    -1 "  ",$[t6;"PASS";"FAIL"];
    nPass+:t6; nFail+:not t6;
    -1 "";

    // --- Test 7: Extreme taper activates at large cumZ ---
    -1 "Test 7: Extreme taper activates at extremes";
    // Constant returns have cumZ bounded at sqrt(3) ≈ 1.73 (never hits 2.0).
    // Use flat-then-burst: cumZ spikes at the acceleration point.
    flatBurst:(800 # 0.0001),(200 # 0.01);
    et:extremeTaper[flatBurst;252;2.0];
    // Flat phase: cumZ ~ 1.73 (under 2.0) -> taper = 1.0
    // Burst transition (~day 830): cumZ > 3 -> taper << 1.0
    early:avg et 200 + til 100;
    late:avg et 830 + til 40;
    t7:(early > 0.95) and (late < 0.5);
    results[`test7]:t7;
    -1 "  Early taper avg: ",string early;
    -1 "  Late taper avg: ",string late;
    -1 "  ",$[t7;"PASS";"FAIL"];
    nPass+:t7; nFail+:not t7;
    -1 "";

    // --- Test 8: Vol break activates during vol spike ---
    -1 "Test 8: Vol break activates during vol spike";
    system "S 42";
    n:500;
    eps:.cond.randNorm n;
    calm:0.005 * eps;
    spike:0.025 * eps;
    r:(300 # calm),(50 # spike),(150 # calm);
    vb:volBreak[r;5;63;2.0];
    calmVB:avg vb 250 + til 40;
    spikeVB:avg vb 310 + til 30;
    t8:(calmVB > 0.8) and (spikeVB < calmVB);
    results[`test8]:t8;
    -1 "  Calm vol break: ",string calmVB;
    -1 "  Spike vol break: ",string spikeVB;
    -1 "  ",$[t8;"PASS";"FAIL"];
    nPass+:t8; nFail+:not t8;
    -1 "";

    // --- Test 9: DD breaker activates during drawdown ---
    -1 "Test 9: DD breaker activates during drawdown";
    cumPnl:0.1 0.12 0.15 0.13 0.10 0.08 0.05 0.04 0.06 0.08;
    dds:ddBreaker[cumPnl;0.05];
    // At index 6: dd = 0.05 - 0.15 = -0.10 -> scale = max(0.25, 1 - 0.5*0.10/0.05) = max(0.25, 0)
    t9:(dds[0] > 0.99) and (dds[6] < 0.5);
    results[`test9]:t9;
    -1 "  No DD scale: ",string dds[0];
    -1 "  Deep DD scale: ",string dds[6];
    -1 "  ",$[t9;"PASS";"FAIL"];
    nPass+:t9; nFail+:not t9;
    -1 "";

    // --- Test 10: Full pipeline runs end-to-end ---
    -1 "Test 10: Full pipeline - end-to-end smoke test";
    td:.momentum.syntheticTest[];
    pipeResult:ensembleTable[td;()!()];
    ev:evaluate[pipeResult];
    sh:ev`sharpe;
    // Full portfolio across all 4 syms (trend+crash+random+MR). Sharpe may be near 0.
    // Key check: pipeline runs, produces finite Sharpe, not terribly negative.
    t10:(not null sh) and sh > neg 1f;
    results[`test10]:t10;
    -1 "  Sharpe: ",string sh;
    -1 "  ",$[t10;"PASS";"FAIL"];
    nPass+:t10; nFail+:not t10;
    -1 "";

    // --- Test 11: Output has all expected columns ---
    -1 "Test 11: Table interface produces correct columns";
    expectedCols:`sig`sigRaw`vr`regimeScale`carryGate`curveConfirm`extremeTaper`volBreak`volScale`ddScale;
    hasCols:all expectedCols in cols pipeResult;
    hasSyms:4 = count distinct pipeResult`sym;
    t11:hasCols and hasSyms;
    results[`test11]:t11;
    -1 "  Has all columns: ",string hasCols;
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
// SECTION 12: HELP / EXAMPLE
// =============================================================================

help:{[]
    -1 "";
    -1 "=== .momentum v2.0 - Risk-Adjusted Momentum Ensemble ===";
    -1 "";
    -1 "CORE SIGNAL:";
    -1 "  sharpeMom[x;w]              - risk-adjusted momentum (mu/sigma)";
    -1 "  accel[mom;w]                - momentum acceleration (2nd derivative)";
    -1 "  singleSpeed[x;w;accelWt]    - combined signal at one lookback";
    -1 "";
    -1 "ENSEMBLE:";
    -1 "  regimeEnsemble[sigs;x;speeds;vrQ;vrW]  - VR-adaptive blend";
    -1 "";
    -1 "GATES:";
    -1 "  carryGateVal[sig;carry]     - carry alignment (0.3 or 1.0)";
    -1 "  curveConfirmVec[sigs]       - cross-contract agreement (0.3-1.0)";
    -1 "  extremeTaper[x;w;thresh]    - fade at cumulative extremes";
    -1 "  volBreak[x;fastHL;slowHL;thresh]  - vol spike protection";
    -1 "  ddBreaker[cumPnl;thresh]    - drawdown circuit breaker";
    -1 "";
    -1 "TABLE INTERFACE:";
    -1 "  ensembleTable[t;cfg]        - full pipeline for (dt;sym;ret) table";
    -1 "    Optional carry column enables carry gate + excess return separation";
    -1 "    cfg keys: speeds, accelWt, vrQ, vrW, targetVol, maxLeverage,";
    -1 "              ddThresh, volBreakThresh, extremeZ, retCol, dtCol, symCol";
    -1 "";
    -1 "EVALUATION:";
    -1 "  evaluate[t]                 - comprehensive metrics";
    -1 "  syntheticTest[]             - 4-sym test data (trend/crash/random/MR)";
    -1 "  runTests[]                  - 12 validation tests";
    -1 "";
    -1 "DEFAULTS:";
    -1 "  speeds: 21 63 126 | accelWt: 0.3 | targetVol: 0.10";
    -1 "  maxLeverage: 3 | ddThresh: 0.05 | volBreakThresh: 2.0";
    -1 "";}

example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              .momentum v2.0 EXAMPLE";
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
    -1 "4. Gate stats:";
    gs:ev`gateStats;
    {[gs;x] -1 "   ",string[x],": avg=",string[(gs x)`avg]," pctActive=",string (gs x)`pctActive}[gs;] each key gs;
    -1 "";
    -1 "5. Per-speed Sharpe:";
    pss:ev`perSpeedSharpe;
    {[pss;x] -1 "   ",string[x]," = ",string pss x}[pss;] each key pss;
    -1 "";
    -1 "Done.";}

\d .

-1 "Loaded .momentum namespace v2.0.0";
-1 "Risk-adjusted momentum: sharpeMom, regimeEnsemble, carry/curve/crash gates";
-1 "Run .momentum.help[] for full function list";
-1 "Run .momentum.runTests[] for validation";
