// =============================================================================
// SIGNAL CONDITIONING
// =============================================================================
// Condition one alpha/signal on another using rolling methods
// Version: 0.3.0

\d .cond

// -----------------------------------------------------------------------------
// CONFIGURATION
// -----------------------------------------------------------------------------

normTypes:`zscore`rank`percentile`minmax`raw

// -----------------------------------------------------------------------------
// ROLLING PRIMITIVES
// -----------------------------------------------------------------------------

ffill:{fills x}
rmean:{[w;x] mavg[w;x]}
rstd:{[w;x] mdev[w;x]}
rzscore:{[w;x] (x - rmean[w;x]) % rstd[w;x]}
rrank:{[w;x] wins:prev {1_x,y}\[w#0n;x]; {y:y where not null y; $[0=n:count y;0n;(sum y<x)%n]}.' flip (x;wins)}
rpercentile:rrank
rmin:{[w;x] (w-1) mmin x}
rmax:{[w;x] (w-1) mmax x}
rminmax:{[w;x] mn:rmin[w;x]; mx:rmax[w;x]; (x - mn) % (mx - mn)}

// Normalize by type
normalize:{[w;ntype;x] $[ntype~`zscore;rzscore[w;x];ntype~`rank;rrank[w;x];ntype~`percentile;rpercentile[w;x];ntype~`minmax;rminmax[w;x];ntype~`raw;x;'`unknownNormType]}

// -----------------------------------------------------------------------------
// ROLLING REGRESSION
// -----------------------------------------------------------------------------

rbeta:{[w;x;y] mx:rmean[w;x]; my:rmean[w;y]; cv:rmean[w;x*y] - mx * my; varx:rmean[w;x*x] - mx * mx; cv % varx}
ralpha:{[w;x;y] b:rbeta[w;x;y]; rmean[w;y] - b * rmean[w;x]}
rresid:{[w;x;y] b:rbeta[w;x;y]; a:ralpha[w;x;y]; y - a - b * x}
rrsq:{[w;x;y] res:rresid[w;x;y]; sstot:rmean[w;(y - rmean[w;y]) xexp 2]; ssres:rmean[w;res * res]; 1 - ssres % sstot}

// -----------------------------------------------------------------------------
// CALENDAR UTILITIES
// -----------------------------------------------------------------------------

// Check if date is weekend (0=Sat, 1=Sun in q date mod 7)
isWeekend:{(x mod 7) in 0 1}

// First day of month (cast to month, then back to date)
firstOfMonth:{[d] "d"$"m"$d}

// Last day of month
lastOfMonth:{[d] -1 + firstOfMonth[d] + 32}

// Last business day of month
lastBizDayOfMonth:{[d]
    lom:lastOfMonth d;
    dow:lom mod 7;  // 0=Sat, 1=Sun
    $[dow = 0; lom - 1; dow = 1; lom - 2; lom]}

// First day of quarter (Q1=Jan, Q2=Apr, Q3=Jul, Q4=Oct)
firstOfQuarter:{[d]
    m:"m"$d;  // e.g., 2026.03m
    qStart:m - ((`mm$d) - 1) mod 3;  // subtract 0, 1, or 2 months to get quarter start
    "d"$qStart}

// Last day of quarter (last day of quarter's final month)
lastOfQuarter:{[d] -1 + "d"$3 + "m"$firstOfQuarter d}

// Last business day of quarter
lastBizDayOfQuarter:{[d]
    loq:lastOfQuarter d;
    dow:loq mod 7;
    $[dow = 0; loq - 1; dow = 1; loq - 2; loq]}

// Count business days between two dates (signed: negative if d1 > d2)
bizDaysBetween:{[d1;d2]
    $[d1 = d2; 0;
      d1 < d2; sum not isWeekend d1 + til 1 + d2 - d1;
      neg sum not isWeekend d2 + til 1 + d1 - d2]}

// Business days to month-end for each date (negative = before, positive = after)
// Returns 0 on last biz day of month
daysToMonthEnd:{[dates]
    monthEnds:lastBizDayOfMonth each dates;
    neg bizDaysBetween'[dates;monthEnds]}

// Business days to quarter-end for each date (negative = before, positive = after)
daysToQuarterEnd:{[dates]
    quarterEnds:lastBizDayOfQuarter each dates;
    neg bizDaysBetween'[dates;quarterEnds]}

// -----------------------------------------------------------------------------
// CALENDAR PROXIMITY FILTERS
// -----------------------------------------------------------------------------

// Gaussian proximity filter for month-end
// dates  - date vector
// offset - peak day relative to month-end (e.g., -3 = 3 biz days before)
// sigma  - decay width in business days
// Returns: 0-1 filter, peaks at offset, 0 after month-end
monthEndGaussian:{[dates;offset;sigma]
    dte:daysToMonthEnd dates;
    d:dte - offset;  // distance from peak
    raw:exp neg (d * d) % 2 * sigma * sigma;
    raw * dte <= 0}  // zero after month-end

// Exponential proximity filter for month-end
// halflife - decay halflife in business days
monthEndExp:{[dates;offset;halflife]
    dte:daysToMonthEnd dates;
    d:abs dte - offset;
    lam:log[2] % halflife;
    raw:exp neg lam * d;
    raw * dte <= 0}

// Gaussian proximity filter for quarter-end
quarterEndGaussian:{[dates;offset;sigma]
    dte:daysToQuarterEnd dates;
    d:dte - offset;
    raw:exp neg (d * d) % 2 * sigma * sigma;
    raw * dte <= 0}

// Exponential proximity filter for quarter-end
quarterEndExp:{[dates;offset;halflife]
    dte:daysToQuarterEnd dates;
    d:abs dte - offset;
    lam:log[2] % halflife;
    raw:exp neg lam * d;
    raw * dte <= 0}

// Generic proximity filter
// event - `monthEnd or `quarterEnd
// decay - `gaussian or `exp
// param - sigma (gaussian) or halflife (exp)
proximityFilter:{[dates;offset;decay;param;event]
    dte:$[event ~ `monthEnd; daysToMonthEnd dates;
          event ~ `quarterEnd; daysToQuarterEnd dates;
          '`unknownEvent];
    d:dte - offset;
    raw:$[decay ~ `gaussian; exp neg (d * d) % 2 * param * param;
          decay ~ `exp; exp neg (log[2] % param) * abs d;
          '`unknownDecay];
    raw * dte <= 0}

// -----------------------------------------------------------------------------
// CORE CONDITIONING FUNCTIONS
// -----------------------------------------------------------------------------

// Gate: f1 when normalized f2 > threshold, else 0
gate:{[f1;f2;window;ntype;threshold] f2n:normalize[window;ntype;ffill f2]; ffill[f1] * f2n > threshold}

// Gate between: f1 when f2 in range [lo,hi]
gateBetween:{[f1;f2;window;ntype;lo;hi] f2n:normalize[window;ntype;ffill f2]; ffill[f1] * (f2n >= lo) and f2n <= hi}

// Scale: multiply f1 by normalized f2
scale:{[f1;f2;window;ntype] f2n:normalize[window;ntype;ffill f2]; ffill[f1] * f2n}

// Scale positive: only scale by positive f2 values
scalePos:{[f1;f2;window;ntype] f2n:normalize[window;ntype;ffill f2]; ffill[f1] * 0f | f2n}

// Percentile filter: keep f1 when f2 rank in [loP,hiP]
percentile:{[f1;f2;window;loP;hiP] f2r:rrank[window;ffill f2]; ffill[f1] * (f2r >= loP) and f2r <= hiP}

// Top percentile
top:{[f1;f2;window;pct] percentile[f1;f2;window;1-pct;1.0]}

// Bottom percentile
bottom:{[f1;f2;window;pct] percentile[f1;f2;window;0.0;pct]}

// Regime: assign regime labels based on f2 quantiles
regime:{[f1;f2;window;nBuckets] f2r:rrank[window;ffill f2]; buckets:(nBuckets-1)&`long$nBuckets*f2r; ([]f1:ffill f1;f2:ffill f2;regime:buckets)}

// Residualize: orthogonalize f1 with respect to f2
residualize:{[f1;f2;window] rresid[window;ffill f2;ffill f1]}

// Interact: multiply normalized signals
interact:{[f1;f2;window;ntype] f1n:normalize[window;ntype;ffill f1]; f2n:normalize[window;ntype;ffill f2]; f1n * f2n}

// Tilt: blend f1 toward f2
tilt:{[f1;f2;window;ntype;weight] f1n:normalize[window;ntype;ffill f1]; f2n:normalize[window;ntype;ffill f2]; ((1-weight)*f1n)+weight*f2n}

// -----------------------------------------------------------------------------
// SIGNAL PROCESSING
// -----------------------------------------------------------------------------

// Smooth: exponential moving average (causal - only uses past)
smooth:{[f;halflife] lam:exp neg log[2]%halflife; {(x*y)+(1-x)*z}[lam]\[first f;ffill f]}

// Rolling clip: cap at rolling percentile bounds (no look-ahead)
clip:{[f;window;loPct;hiPct]
    wins:prev {1_x,y}\[window#0n;f];
    bounds:{[lo;hi;w] v:asc w where not null w; n:count v; $[n<2;(0n;0n);(v `long$lo*n;v `long$hi*n)]}[loPct;hiPct] each wins;
    lo:bounds[;0]; hi:bounds[;1];
    lo|f&hi}

// Winsorize: alias for clip
winsorize:clip

// Decay: EMA-style decay (causal - same as smooth, clearer name)
decay:{[f;halflife] smooth[f;halflife]}

// Fast-attack slow-decay filter
// Snaps to input when magnitude is growing, decays gradually when fading
// @param f - signal vector
// @param decayHL - halflife for decay (e.g. 5-10)
attackDecay:{[f;decayHL]
    ld:exp neg log[2] % decayHL;
    {[ld;y;z] $[(abs z) >= abs y; z; (ld * y) + (1 - ld) * z]}[ld]\[first f;ffill f]}

// Lag: shift signal by n periods (positive = look back)
lag:{[f;n] if[0=count f; :f]; $[n>0; (n#0n),neg[n]_f; (neg[n]_f),abs[n]#0n]}

// Diff: signal change (momentum of signal)
diff:{[f;n] f - lag[f;n]}

// Pctchange: percentage change of signal
pctChange:{[f;n] prev_f:lag[f;n]; (f - prev_f) % abs prev_f}

// -----------------------------------------------------------------------------
// MEAN-REVERSION / SNAP-BACK
// -----------------------------------------------------------------------------

// Accumulated tension: z-score of running cumulative sum
// Measures how far a cumulative signal has drifted from its rolling norm
accumTension:{[w;x] rzscore[w; sums ffill x]}

// Conditional reversal: reverse long-horizon signal when short-horizon is extreme
//
// Logic: When short-term z-score exceeds threshold, the series has moved too far
// too fast, so reverse the longer-term signal to bet on mean reversion.
// Otherwise, follow the longer-term signal direction.
//
// x         - input series (e.g., returns or prices)
// shortHL   - short halflife for EMA (used for z-score calculation)
// longHL    - long halflife for the signal to potentially reverse
// zWindow   - window for computing z-score of the EMAs
// threshold - z-score threshold for reversal (e.g., 2.0)
// mode      - `zscore (long signal is z-score of EMA) or `ema (long signal is EMA deviation from slower EMA)
//
// Returns: signal that reverses when short-term is extreme
conditionalReverse:{[x;shortHL;longHL;zWindow;threshold;mode]
    x:ffill "f"$x;
    // Short-term EMA and its z-score (using smooth which takes halflife)
    shortEMA:smooth[x;shortHL];
    shortZ:rzscore[zWindow;shortEMA];
    // Long-term EMA and signal
    longEMA:smooth[x;longHL];
    longSig:$[mode~`zscore;
        rzscore[zWindow;longEMA];
        longEMA - smooth[x;longHL*2]];  // EMA deviation from slower EMA
    // Reverse when |shortZ| > threshold (treat null as not extreme)
    extreme:(abs[shortZ] > threshold) and not null shortZ;
    ?[extreme; neg longSig; longSig]}

// Simplified version with defaults: z-score mode, threshold=2
condReverse:{[x;shortHL;longHL;zWindow] conditionalReverse[x;shortHL;longHL;zWindow;2.0;`zscore]}

// Dual-horizon z-score reversal (simpler - just uses rolling z-scores directly)
// Computes z-scores at both horizons, reverses long z-score when short is extreme
// More direct interpretation: fade the long-term trend when short-term is overextended
//
// x            - input series
// shortWindow  - window for short-term z-score
// longWindow   - window for long-term z-score
// threshold    - z-score threshold for reversal
dualHorizonReverse:{[x;shortWindow;longWindow;threshold]
    x:ffill "f"$x;
    shortZ:rzscore[shortWindow;x];
    longZ:rzscore[longWindow;x];
    extreme:(abs[shortZ] > threshold) and not null shortZ;
    ?[extreme; neg longZ; longZ]}

// Snap-revert: accumulated tension + trigger -> mean-reverting held position
//
// When trigger fires, takes position opposite to tension (-tension).
// Position then either tracks tension continuously (track mode) or
// decays exponentially (decay mode). Exits when tension sign reverses
// (but not before minHold periods have elapsed).
//
// tension - float vector: directional measure (e.g. accumTension output)
// trigger - float vector: binary trigger (>0.5 = fire)
// mode    - `track (position = -tension while sign holds) or
//           `decay (position decays with halflife)
// holdHL  - halflife for decay mode (ignored in track mode)
// minHold - minimum periods to hold after trigger before sign-flip exit (default 0)
snapRevert:{[tension;trigger;mode;holdHL;minHold]
    ten:@[ffill "f"$tension;where null ffill "f"$tension;:;0f];
    trig:@[ffill "f"$trigger;where null ffill "f"$trigger;:;0f];
    mh:$[null minHold;0f;"f"$minHold];
    lam:$[mode~`decay; exp neg log[2] % holdHL; 0f];
    init:(0f; 0f; 0f);
    step:$[mode~`track;
        {[mh;s;t;tr]
            pos:s 0; esign:s 1; hc:s 2; ts:signum t;
            $[(tr>0.5) and t<>0f;        (neg t; ts; 1f);
              (esign<>0f) and (hc>=mh) and (ts<>esign) and ts<>0f; (0f;0f;0f);
              esign<>0f;                  (neg t; esign; hc+1);
                                          (0f;0f;0f)]}[mh];
        {[lam;mh;s;t;tr]
            pos:s 0; esign:s 1; hc:s 2; ts:signum t;
            $[(tr>0.5) and t<>0f;        (neg t; ts; 1f);
              (esign<>0f) and (hc>=mh) and (ts<>esign) and ts<>0f; (0f;0f;0f);
              esign<>0f;                  (pos*lam; esign; hc+1);
                                          (0f;0f;0f)]}[lam;mh]];
    (step\[init;ten;trig])[;0]}

// -----------------------------------------------------------------------------
// CONVICTION / CONFIDENCE
// -----------------------------------------------------------------------------

// Conviction: scale by absolute magnitude of signal (stronger signal = bigger position)
conviction:{[f;window;ntype] fn:normalize[window;ntype;ffill f]; fn * abs fn}

// Confidence: weight signal by its rolling IC with forward returns
confidence:{[f;fwdRet;window] ic:rollingIC[f;fwdRet;window]; ffill[f] * 0f | ffill ic}

// Agree: f1 only when f1 and f2 have same sign
agree:{[f1;f2] f1f:ffill f1; f2f:ffill f2; f1f * (signum[f1f]=signum[f2f])}

// AgreeN: f1 only when N signals agree on direction (signals is list of vectors)
agreeN:{[f1;signals;minAgree] f1f:ffill f1; signs:signum each ffill each signals; agreement:sum each flip signs; f1f * abs[agreement] >= minAgree}

// Disagree: f1 only when f1 and f2 have opposite signs (contrarian)
disagree:{[f1;f2] f1f:ffill f1; f2f:ffill f2; f1f * (signum[f1f]<>signum[f2f])}

// Confirm: f1 only when f2 confirms (same sign and f2 above threshold)
confirm:{[f1;f2;window;ntype;thresh] f1f:ffill f1; f2n:normalize[window;ntype;ffill f2]; f1f * (signum[f1f]=signum[f2n]) and abs[f2n]>thresh}

// -----------------------------------------------------------------------------
// RISK-BASED CONDITIONING
// -----------------------------------------------------------------------------

// VolAdjust: scale signal by inverse rolling volatility (vol-target at signal level)
volAdjust:{[f;window;targetVol] vol:rstd[window;ffill f]; scale:targetVol % vol; ffill[f] * ffill scale}

// DrawdownGate: reduce/zero signal when cumulative signal in drawdown
drawdownGate:{[f;window;ddThresh] cf:sums ffill f; roll:{(x-1) mmax y}[window]; maxCf:roll cf; dd:(cf - maxCf) % abs maxCf; f * dd > neg ddThresh}

// SharpeGate: gate signal by its rolling Sharpe ratio
sharpeGate:{[f;fwdRet;window;minSharpe] mu:rmean[window;fwdRet*signum ffill f]; vol:rstd[window;fwdRet*signum ffill f]; sharpe:mu % vol; ffill[f] * (sharpe > minSharpe % sqrt 252)}

// IcDecay: decay signal strength based on rolling IC
icDecay:{[f;fwdRet;window] ic:rollingIC[f;fwdRet;window]; icNorm:0f | ic % rmean[window*2;abs ic]; ffill[f] * ffill icNorm}

// HitRateGate: gate by rolling hit rate
hitRateGate:{[f;fwdRet;window;minHitRate] hits:rmean[window;(signum ffill f)=signum fwdRet]; ffill[f] * hits >= minHitRate}

// MaxLossGate: zero signal after large loss
maxLossGate:{[f;fwdRet;window;maxLoss] losses:rmean[window;0f & fwdRet * signum ffill f]; ffill[f] * losses > neg maxLoss}

// -----------------------------------------------------------------------------
// VARIANCE RATIO & REGIME DETECTION
// -----------------------------------------------------------------------------

// Variance ratio: detects trending vs mean-reverting regimes
// Compares variance of q-period returns to q * variance of 1-period returns
// VR > 1 = trending (positive autocorrelation), VR < 1 = mean-reverting
// @param x - return series (daily changes)
// @param q - multi-period horizon (e.g. 5 for weekly)
// @param w - lookback window for rolling estimates
varianceRatio:{[x;q;w]
    x:ffill x;
    qRet:q msum x;
    vQ:mdev[w; qRet] xexp 2;
    v1:mdev[w; x] xexp 2;
    vQ % 1e-10 | q * v1}

// Variance ratio table interface (grouped by sym)
// @param t - table sorted by (bycol, time)
// @param bycol - group column (e.g. `sym)
// @param col - return column
// @param q - multi-period horizon
// @param w - lookback window
varianceRatioTable:{[t;bycol;col;q;w]
    f:{[c;q;w;g] vr:varianceRatio[g c;q;w]; g,'flip `vr`vrTrending!(vr;vr > 1f)}[col;q;w];
    t:update vrIdx__:i from t;
    grp:group t bycol;
    r:raze f each {[t;idx] t idx}[t] each value grp;
    r:`vrIdx__ xasc r;
    ![r;();0b;enlist `vrIdx__]}

// Gated momentum signal: directional signal active only in trending regimes
// @param x - return series
// @param momWindow - momentum lookback (e.g. 20)
// @param q - variance ratio horizon (e.g. 5)
// @param vrWindow - variance ratio lookback (e.g. 60)
// @param vrThresh - VR threshold for trending (e.g. 1.0)
gatedMomentum:{[x;momWindow;q;vrWindow;vrThresh]
    x:ffill x;
    direction:signum mavg[momWindow; x];
    vr:varianceRatio[x;q;vrWindow];
    direction * vr > vrThresh}

// Gated momentum table interface (grouped by sym)
// @param t - table sorted by (bycol, time)
// @param bycol - group column
// @param col - return column
// @param momWindow - momentum lookback
// @param q - variance ratio horizon
// @param vrWindow - variance ratio lookback
// @param vrThresh - VR threshold
gatedMomentumTable:{[t;bycol;col;momWindow;q;vrWindow;vrThresh]
    f:{[c;mw;q;vrw;vrt;g] r:gatedMomentum[g c;mw;q;vrw;vrt]; vr:varianceRatio[g c;q;vrw]; dir:signum mavg[mw; g c]; g,'flip `vr`direction`gatedSig!(vr;dir;r)}[col;momWindow;q;vrWindow;vrThresh];
    t:update gmIdx__:i from t;
    grp:group t bycol;
    r:raze f each {[t;idx] t idx}[t] each value grp;
    r:`gmIdx__ xasc r;
    ![r;();0b;enlist `gmIdx__]}

// -----------------------------------------------------------------------------
// CROSS-SECTIONAL (operate across assets at each time point)
// -----------------------------------------------------------------------------

// csRank: rank across assets (columns) at each time point
// Input: matrix where rows=time, cols=assets; Output: same shape with ranks 0-1
csRank:{[M] {(iasc iasc r) % count r:x} each M}

// csZscore: z-score across assets at each time point
csZscore:{[M] {(r - avg r) % dev r:x} each M}

// csNeutralize: demean across assets (market neutral)
csNeutralize:{[M] {r - avg r:x} each M}

// csSpread: spread vs cross-sectional mean (alias)
csSpread:csNeutralize

// csWinsorize: winsorize across assets
csWinsorize:{[M;loPct;hiPct] {[loPct;hiPct;row] s:asc row; lo:s `long$loPct*count s; hi:s `long$hiPct*count s; lo|row&hi}[loPct;hiPct] each M}

// csResidNeutralize: residualize each asset vs cross-sectional mean
csResidNeutralize:{[M;window] means:avg each M; {[w;m;col] rresid[w;m;col]}[window;means] each flip M}

// csSmooth: joint EMA smoothing of level and cross-sectional components
// Decomposes signal into CS mean (level) and CS deviation, smooths each independently,
// rescales deviations to match raw dispersion, then recombines.
// t: table with time, sym, and signal columns
// halflife: default EMA halflife (used for both level and CS unless overridden in cfg)
// cfg: dict with optional keys:
//   `time`sym`sig  - column names (defaults: `time`ricRoot`rawSig)
//   `levelHL       - halflife for level/mean smoothing (default: halflife)
//   `csHL          - halflife for cross-sectional deviation smoothing (default: halflife)
//   `invert        - negate CS deviations (default: 0b)
// Returns: t with added `smoothSig column
csSmooth:{[t;halflife;cfg]
    dc:`time`sym`sig`invert`levelHL`csHL!(`time;`ricRoot;`rawSig;0b;halflife;halflife);
    c:$[99h = type cfg; dc,cfg; dc];
    tc:c`time; sc:c`sym; sigc:c`sig; doInv:c`invert;
    lHL:c`levelHL; cHL:c`csHL;
    t:(tc,sc) xasc t;
    raw:"f"$t sigc;
    rawMu:(avg; raw) fby t tc;
    rawDev:raw - rawMu;
    smoMu:(smooth[;lHL]; rawMu) fby t sc;
    smoDev:(smooth[;cHL]; rawDev) fby t sc;
    rawStd:(dev; raw) fby t tc;
    smoDevSd:(dev; smoDev) fby t tc;
    csdev:0f ^ smoDev * rawStd % 1e-10 | smoDevSd;
    t[`smoothSig]:smoMu + $[doInv; neg csdev; csdev];
    t}

// csInvert: invert cross-sectional deviations while preserving the level
// At each time step: output = csMean - (sig - csMean) = 2*csMean - sig
// t: table with time, sym, and signal columns
// cfg: dict with optional keys `time`sym`sig (defaults: `time`ricRoot`rawSig)
// Returns: t with sig column replaced by its cs-inverted values
csInvert:{[t;cfg]
    dc:`time`sym`sig!(`time;`ricRoot;`rawSig);
    c:$[99h = type cfg; dc,cfg; dc];
    tc:c`time; sigc:c`sig;
    mu:(avg; "f"$t sigc) fby t tc;
    t[sigc]:(2 * mu) - "f"$t sigc;
    t}

// sigOpt internals (namespace-level to avoid q closure scoping issues)
// Clean infinities/nulls: replace 0w/-0w with 0n
soClean_:{@[x;where not x within (-1e308;1e308);:;0n]};
soPsh_:{[sym;ret;rsk;tm;ann;sig]
    sig:soClean_ sig;
    scaled:soClean_ sig%rsk;
    ps:(prev;0f^scaled) fby sym;
    pnl:ps*ret;
    tab:([]t:tm;p:pnl);
    byTm:0!select sp:sum p by t from tab;
    v:byTm[`sp] where not null byTm`sp;
    $[(1e-10<dev v)and 2<count v;((avg v)%dev v)*sqrt ann;0n]};
soFmet_:{[sym;ret;rsk;tm;ann;sig]
    sig:soClean_ sig;
    if[all null sig;:`sharpe`ic`hitRate`turnover`maxDD`profitFactor!(0n;0n;0n;0n;0n;0n)];
    scaled:soClean_ sig%rsk;
    ps:(prev;0f^scaled) fby sym;
    pnl:ps*ret;
    tab:([]t:tm;p:pnl);
    byTm:0!select sp:sum p by t from tab;
    v:byTm[`sp] where not null byTm`sp;
    sharpe:$[(1e-10<dev v)and 2<count v;((avg v)%dev v)*sqrt ann;0n];
    tab2:([]t:tm;ps0:ps;r:ret);
    byTm2:0!select ps0,r by t from tab2;
    ics:{cor[0f^x;0f^y]}'[byTm2`ps0;byTm2`r];
    ics:@[ics;where not ics within -1 1f;:;0n];
    icV:ics where not null ics;
    ic:$[0<count icV;avg icV;0n];
    mask:(not null ps)&not null ret;
    hr:$[0<sum mask;(sum ((signum ps)=signum ret)&mask)%sum mask;0n];
    dlt:soClean_(deltas;0f^scaled) fby sym;
    to:avg abs dlt where not null dlt;
    cumPnl:sums v;mdd:min cumPnl-maxs cumPnl;
    posP:sum v where v>0;negP:abs sum v where v<0;
    pf:$[negP>1e-10;posP%negP;0n];
    `sharpe`ic`hitRate`turnover`maxDD`profitFactor!(sharpe;ic;hr;to;mdd;pf)};
soGsrch_:{[nn;psh;grid;gen]
    sigs:{[nn;gen;p] soClean_ @[gen;p;{[n;e] n#0n}[nn;]]}[nn;gen;] each grid;
    sharpes:psh each sigs;
    valid:where not null sharpes;
    if[0=count valid;:(grid 0;0n;nn#0n)];
    best:valid first idesc sharpes valid;
    (grid best;sharpes best;sigs best)};

// --- Stateful signal helpers for sigOpt ---
// Threshold entry/exit state machine: returns +1 (long), -1 (short), 0 (flat)
// Enter long when indicator <= entryLo, exit long when indicator >= exitLo
// Enter short when indicator >= entryHi, exit short when indicator <= exitHi
soThreshHold_:{[indicator;entryLo;exitLo;entryHi;exitHi]
    n:count indicator; if[n=0; :`float$()]; pos:n#0f; st:0f; i:0;
    while[i<n;
        v:indicator i;
        if[not null v;
            $[st = 0f;
                $[v <= entryLo; st:1f; v >= entryHi; st:-1f; (::)];
              st = 1f;
                if[v >= exitLo; st:0f];
              // st = -1f
                if[v <= exitHi; st:0f]]];
        pos[i]:st;
        i+:1];
    pos};
// Breakout entry with time-based hold: +1 on new high, -1 on new low, hold for N bars
soBreakHold_:{[feat;w;holdN]
    n:count feat; if[n=0; :`float$()]; pos:n#0f;
    hh:mmax[w;feat]; ll:mmin[w;feat];
    cnt:0; i:w;
    while[i<n;
        if[not null feat i;
            $[feat[i] >= hh i; cnt:holdN;
              feat[i] <= ll i; cnt:neg holdN;
              cnt > 0; cnt-:1;
              cnt < 0; cnt+:1;
              (::)]];
        pos[i]:"f"$signum cnt;
        i+:1];
    pos};

// --- Bayesian Optimization for sigOpt ---
// Transform unit [0,1] → parameter value (handles log-scale and integer rounding)
soToParam_:{[lo;hi;isLog;isInt;u]
    u:0f|1f&u;
    v:$[isLog;
        exp (log lo) + (u * (log hi) - log lo);
        lo + (u * (hi - lo))];
    $[isInt; `long$ 0.5 + v; v]};
// Unit vector → parameter dict
soU2P_:{[space;u] space[`names]!soToParam_'[space`lo;space`hi;space`log;space`int;u]};
// RBF kernel: single row against all rows of X2
soKernelRow_:{[x1;X2;ls]
    sqdists:{sum d*d:x-y}[x1;] each X2;
    exp neg 0.5 * sqdists % ls * ls};
// Full N1 x N2 kernel matrix
soKernel_:{[X1;X2;ls] soKernelRow_[;X2;ls] each X1};
// Cholesky decomposition → lower triangular L where A = L L'
soChol_:{[A]
    n:count A; L:(n;n)#0f; j:0;
    while[j<n;
        s:A[j;j] - $[j>0; sum L[j;til j] * L[j;til j]; 0f];
        L[j;j]:sqrt 1e-10 | s;
        i:j+1;
        while[i<n;
            s:A[i;j] - $[j>0; sum L[i;til j] * L[j;til j]; 0f];
            L[i;j]:s % L[j;j];
            i+:1];
        j+:1];
    L};
// Forward substitution: solve L x = b
soFwdSolve_:{[L;b]
    n:count b; x:n#0f; i:0;
    while[i<n;
        x[i]:(b[i] - $[i>0; sum L[i;til i] * x til i; 0f]) % L[i;i];
        i+:1];
    x};
// Backward substitution: solve L' x = b
soBwdSolve_:{[L;b]
    n:count b; x:n#0f; i:n-1;
    while[i>=0;
        idx:(i+1) + til (n-1) - i;
        s:$[0 < count idx; sum L[idx;i] * x idx; 0f];
        x[i]:(b[i] - s) % L[i;i];
        i-:1];
    x};
// GP predict mean + variance at test points given training data
soGPpred_:{[Xtrain;ytrain;Xtest;ls;noise]
    n:count Xtrain;
    Kxx:soKernel_[Xtrain;Xtrain;ls];
    Kxx:Kxx + noise * (til n) =/:\: til n;
    L:soChol_ Kxx;
    alpha:soBwdSolve_[L; soFwdSolve_[L;ytrain]];
    Kxs:soKernel_[Xtest;Xtrain;ls];
    mu:{sum x*y}[;alpha] each Kxs;
    vars:{[L;krow] v:soFwdSolve_[L;krow]; 0f | 1f - sum v*v}[L;] each Kxs;
    (mu;vars)};
// Expected Improvement acquisition function
soEI_:{[mu;vars;bestY]
    sigma:sqrt vars;
    imp:mu - bestY;
    z:imp % 1e-10 | sigma;
    pdf:(exp neg 0.5 * z * z) % sqrt 2 * acos neg 1f;
    cdf:1f % 1f + exp neg 1.7023 * z;
    (imp * cdf) + sigma * pdf};
// Safe signal evaluation → Sharpe
soEvalOne_:{[nn;psh;gen;p] psh soClean_ @[gen;p;{[n;e] n#0n}[nn;]]};
// Create parameter space: names, bounds, log/int flags, optional validator
soMkSpace_:{[nms;lo;hi;lg;it;vf]
    `names`lo`hi`log`int`valid!((),nms;(),"f"$lo;(),"f"$hi;(),lg;(),it;vf)};
// Main BO search loop (replaces grid search)
soBOsrch_:{[nn;psh;gen;space;nInit;nIter;nCand]
    d:count space`names;
    // 0-parameter signals: evaluate once
    if[d=0;
        p:space[`names]!`float$();
        sig:soClean_ @[gen;p;{[n;e] n#0n}[nn;]];
        :(p; psh sig; sig)];
    ev:soEvalOne_[nn;psh;gen;];
    vld:space`valid; hasVld:not (::) ~ vld;
    // Phase 1: random initial points (d floats per point)
    us:d cut (nInit * d) ? 1f;
    ps:soU2P_[space;] each us;
    // Rejection-sample invalid points
    if[hasVld;
        idx:0;
        while[idx < nInit;
            att:0;
            while[(not vld ps idx) and att < 100;
                us[idx]:d?1f; ps[idx]:soU2P_[space; us idx]; att+:1];
            idx+:1]];
    sharpes:ev each ps;
    // Phase 2: BO iterations
    boI:0;
    while[boI < nIter;
        ys:"f"$sharpes;
        ys:@[ys; where null ys; :; -10f];
        ymu:avg ys; ysd:1e-10 | dev ys;
        yn:(ys - ymu) % ysd;
        bestYn:max yn;
        // Candidates
        candU:d cut (nCand * d) ? 1f;
        candP:soU2P_[space;] each candU;
        // Filter by constraint
        if[hasVld;
            mask:vld each candP;
            candU:candU where mask;
            candP:candP where mask;
            // Resample if too few valid
            if[10 > count candU;
                extra:d cut (200 * d) ? 1f;
                extraP:soU2P_[space;] each extra;
                eMask:vld each extraP;
                candU:candU, extra where eMask;
                candP:candP, extraP where eMask;
                candU:(nCand & count candU) # candU;
                candP:(nCand & count candP) # candP]];
        if[0 = count candU; boI+:1; :boI];
        // GP predict + EI
        nc:count candU;
        gpRes:@[soGPpred_[us;yn;;0.3;0.1]; candU; {[nc;e] (nc#0f;nc#1f)}[nc;]];
        mu:gpRes 0; vars:gpRes 1;
        ei:soEI_[mu;vars;bestYn];
        ei:@[ei; where null ei; :; 0f];
        bestIdx:first idesc ei;
        bestU:candU bestIdx;
        bestP:soU2P_[space;bestU];
        newSharpe:ev bestP;
        us:us,enlist bestU;
        ps:ps,enlist bestP;
        sharpes:sharpes,newSharpe;
        boI+:1];
    // Return best
    valid:where not null sharpes;
    if[0=count valid; :(ps 0; 0n; nn#0n)];
    best:valid first idesc "f"$sharpes valid;
    bestSig:soClean_ @[gen;ps best;{[n;e] n#0n}[nn;]];
    (ps best; sharpes best; bestSig)};

// sigOpt: signal optimization boilerplate
// Generates 15 momentum/regression signals from a feature, optimizes params by Sharpe
// t: table with dt, time, ricRoot, risk, pxDiff + feature column
// featureCol: symbol name of feature column
// cfg: optional config dict (`sym`time`ret`risk to override column names)
// Returns: table with signal name, best params, sharpe, ic, hitRate, turnover, maxDD, profitFactor
sigOpt:{[t;featureCol;cfg]
    dc:`sym`time`ret`risk!(`ricRoot;`time;`pxDiff;`risk);
    c:$[99h=type cfg;dc,cfg;dc];
    symC:c`sym;tmC:c`time;retC:c`ret;rskC:c`risk;
    t:(tmC,symC) xasc t;
    feat:soClean_ "f"$t featureCol;
    ret:soClean_ "f"$t retC;
    sym:t symC;
    tm:t tmC;
    rsk:1e-10|soClean_ $[rskC in cols t;"f"$t rskC;1f+(0*feat)];
    nn:count feat;
    nT:count distinct tm;
    ann:$[nT>500;252f;$[nT>100;52f;12f]];
    ctx:`feat`ret`sym`tm`rsk`nn`ann!(feat;ret;sym;tm;rsk;nn;ann);
    // Bind helpers via projection
    psh:soPsh_[sym;ret;rsk;tm;ann;];
    fmet:soFmet_[sym;ret;rsk;tm;ann;];
    gsrch:soGsrch_[nn;psh;;];
    // --- 15 Signal Generators (project ctx to capture data) ---
    g1:{[c;p] (.cond.rzscore[p`w;];c`feat) fby c`sym}[ctx;];
    g2:{[c;p] ((.cond.rrank[p`w;];c`feat) fby c`sym)-0.5}[ctx;];
    g3:{[c;p] (.cond.diff[;p`n];c`feat) fby c`sym}[ctx;];
    g4:{[c;p] d:(.cond.diff[;p`n];c`feat) fby c`sym; (.cond.smooth[;p`hl];0f^d) fby c`sym}[ctx;];
    g5:{[c;p] (.cond.diff[;p`n];(.cond.diff[;p`n];c`feat) fby c`sym) fby c`sym}[ctx;];
    g6:{[c;p] ((.cond.smooth[;p`fast];c`feat) fby c`sym)-((.cond.smooth[;p`slow];c`feat) fby c`sym)}[ctx;];
    g7:{[c;p] neg(.cond.rzscore[p`w;];c`feat) fby c`sym}[ctx;];
    g8:{[c;p] (c`feat)%1e-10|(mdev[`long$p`w;];c`feat) fby c`sym}[ctx;];
    g9:{[c;p] (c`feat)-(.kdbtools.rpctl[`long$p`w;0.5];c`feat) fby c`sym}[ctx;];
    g10:{[c;p] (.cond.decay[;p`hl];c`feat) fby c`sym}[ctx;];
    g11:{[c;p]
        tb:([]t:c`tm;s:c`sym;r:c`ret;f:c`feat);
        res:`t`s xasc .kdbtools.rollingRidgeTable[tb;`s;enlist`f;`r;`long$p`w;p`lam];
        res`yhat}[ctx;];
    g12:{[c;p]
        tb:([]t:c`tm;s:c`sym;r:c`ret;f:c`feat);
        res:`t`s xasc .kdbtools.rollingRidgeTable[tb;`s;enlist`f;`r;`long$p`w;0f];
        res`yhat}[ctx;];
    g13:{[c;p] mu:(avg;c`feat) fby c`tm;sd:(dev;c`feat) fby c`tm;((c`feat)-mu)%1e-10|sd}[ctx;];
    g14:{[c;p]
        byT:0!select f:f by t:t from([]t:c`tm;f:c`feat);
        raze{r:rank x;(r%((-1+count r)|1))-0.5}each byT`f}[ctx;];
    g15:{[c;p] zs:(.cond.rzscore[p`w;];c`feat) fby c`sym; (.cond.smooth[;p`hl];0f^zs) fby c`sym}[ctx;];
    // --- 10 Additional Signal Generators ---
    // RSI centered at 0: (RSI - 50) / 50 → range [-1, 1]
    g16:{[c;p] (((.kdbtools.rsi[`long$p`w;];c`feat) fby c`sym) % 50f) - 1f}[ctx;];
    // Bollinger band position: (x - sma) / (k * mdev)
    g17:{[c;p] (.kdbtools.bbpos[`long$p`w;p`k;];c`feat) fby c`sym}[ctx;];
    // MACD histogram: extract hist from macd dict
    g18:{[c;p] fn:{[fa;sl;sg;d] (.kdbtools.macd[fa;sl;sg;d])`hist}[`long$p`fast;`long$p`slow;`long$p`sig;]; (fn;c`feat) fby c`sym}[ctx;];
    // Slope t-statistic (trend significance)
    g19:{[c;p] (.kdbtools.slopeT[`long$p`w;];c`feat) fby c`sym}[ctx;];
    // CCI (Commodity Channel Index)
    g20:{[c;p] (.kdbtools.cci[`long$p`w;];c`feat) fby c`sym}[ctx;];
    // Stochastic %K centered at 0: (stochK - 50) / 50 → range [-1, 1]
    g21:{[c;p] (((.kdbtools.stochk[`long$p`w;];c`feat) fby c`sym) % 50f) - 1f}[ctx;];
    // Fisher transform (normalized momentum)
    g22:{[c;p] (.kdbtools.fisher[`long$p`w;];c`feat) fby c`sym}[ctx;];
    // Chande Momentum Oscillator, scale to [-1, 1]
    g23:{[c;p] ((.kdbtools.cmo[`long$p`w;];c`feat) fby c`sym) % 100f}[ctx;];
    // TRIX (triple-smoothed EMA rate of change)
    g24:{[c;p] (.kdbtools.trix[`long$p`w;];c`feat) fby c`sym}[ctx;];
    // Kalman residual (deviation from adaptive filter → mean-reversion)
    g25:{[c;p] kf:({$[0=count z;z;.kdbtools.kalman[x;y;z]]}[p`q;p`r;];c`feat) fby c`sym; (c`feat) - kf}[ctx;];
    // --- 5 Stateful (entry/exit/hold) Signal Generators ---
    // RSI retrace: enter long on oversold, short on overbought, hold until neutral
    g26:{[c;p]
        rsi:(.kdbtools.rsi[`long$p`w;];c`feat) fby c`sym;
        (soThreshHold_[;p`entryLo;p`exitMid;100f - p`entryLo;100f - p`exitMid]; rsi) fby c`sym}[ctx;];
    // Bollinger band retrace: enter on band touch, hold until mid-band
    g27:{[c;p]
        bb:(.kdbtools.bbpos[`long$p`w;p`k;];c`feat) fby c`sym;
        (soThreshHold_[;neg p`entry;0f;p`entry;0f]; bb) fby c`sym}[ctx;];
    // Z-score snap: enter on extreme z-score, hold until mean-reversion
    g28:{[c;p]
        zs:(.cond.rzscore[`long$p`w;];c`feat) fby c`sym;
        (soThreshHold_[;neg p`zEntry;0f;p`zEntry;0f]; zs) fby c`sym}[ctx;];
    // Breakout-and-hold: enter on new high/low, hold for N periods
    g29:{[c;p] (soBreakHold_[;`long$p`w;`long$p`hold];c`feat) fby c`sym}[ctx;];
    // Tension snapback: enter when cumulative drift is extreme, hold until reversion
    g30:{[c;p]
        tension:(.cond.accumTension[`long$p`w;];c`feat) fby c`sym;
        (soThreshHold_[;neg p`thresh;0f;p`thresh;0f]; tension) fby c`sym}[ctx;];
    gens:(g1;g2;g3;g4;g5;g6;g7;g8;g9;g10;g11;g12;g13;g14;g15;g16;g17;g18;g19;g20;g21;g22;g23;g24;g25;g26;g27;g28;g29;g30);
    // --- Parameter Spaces (broad continuous ranges for Bayesian optimization) ---
    mks:soMkSpace_;
    noVld:(::);
    spaces:(
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // zscore
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // rank
        mks[enlist`n; 1; 126; 0b; 1b; noVld];                           // momentum
        mks[`n`hl; 3 2; 63 63; 00b; 11b; noVld];                        // smoothMom
        mks[enlist`n; 1; 63; 0b; 1b; noVld];                            // accel
        mks[`fast`slow; 2 5; 42 252; 00b; 11b; {x[`fast]<x`slow}];     // emaCross
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // meanRev
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // volAdj
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // breakout
        mks[enlist`hl; 1; 126; 0b; 1b; noVld];                          // decay
        mks[`w`lam; 21 0.001; 252 10; 01b; 10b; noVld];                 // ridge
        mks[enlist`w; 21; 252; 0b; 1b; noVld];                          // ols
        mks[`$(); `float$(); `float$(); `boolean$(); `boolean$(); noVld]; // csZscore
        mks[`$(); `float$(); `float$(); `boolean$(); `boolean$(); noVld]; // csRank
        mks[`w`hl; 5 2; 252 63; 00b; 11b; noVld];                       // smoothZscore
        mks[enlist`w; 2; 126; 0b; 1b; noVld];                           // rsi
        mks[`w`k; 5 0.5; 252 5.0; 00b; 10b; noVld];                    // bbandPos
        mks[`fast`slow`sig; 2 8 2; 26 63 26; 000b; 111b; {x[`fast]<x`slow}]; // macdHist
        mks[enlist`w; 5; 252; 0b; 1b; noVld];                           // slopeT
        mks[enlist`w; 5; 126; 0b; 1b; noVld];                           // cci
        mks[enlist`w; 2; 126; 0b; 1b; noVld];                           // stochastic
        mks[enlist`w; 2; 126; 0b; 1b; noVld];                           // fisher
        mks[enlist`w; 2; 126; 0b; 1b; noVld];                           // cmo
        mks[enlist`w; 3; 63; 0b; 1b; noVld];                            // trix
        mks[`q`r; 0.001 0.01; 10 100; 11b; 00b; noVld];                // kalmanResid
        // --- Stateful (entry/exit/hold) signals ---
        mks[`w`entryLo`exitMid; 5 15 40; 63 40 65; 000b; 100b; {x[`entryLo]<x`exitMid}]; // rsiRetrace
        mks[`w`k`entry; 10 1.0 0.5; 126 3.0 1.5; 000b; 100b; noVld];                     // bbRetrace
        mks[`w`zEntry; 10 1.0; 252 4.0; 00b; 10b; noVld];                                 // zscoreSnap
        mks[`w`hold; 5 3; 126 63; 00b; 11b; noVld];                                       // breakHold
        mks[`w`thresh; 10 1.0; 252 4.0; 00b; 10b; noVld]);                                // tensionSnap
    names:`zscore`rank`momentum`smoothMom`accel`emaCross`meanRev`volAdj`breakout`decay`ridge`ols`csZscore`csRank`smoothZscore`rsi`bbandPos`macdHist`slopeT`cci`stochastic`fisher`cmo`trix`kalmanResid`rsiRetrace`bbRetrace`zscoreSnap`breakHold`tensionSnap;
    // BO config (overridable)
    boInit:$[`boInit in key c; c`boInit; 20];
    boIter:$[`boIter in key c; c`boIter; 25];
    boCand:$[`boCand in key c; c`boCand; 200];
    // Run Bayesian optimization
    nSig:count names;
    -1"sigOpt: optimizing ",string[nSig]," signals via Bayesian optimization (",string[boInit]," init + ",string[boIter]," BO iter each)...";
    bosrch:soBOsrch_[nn;psh;;;boInit;boIter;boCand];
    results:bosrch'[gens;spaces];
    bestParams:results[;0];
    bestSigs:results[;2];
    mets:fmet each bestSigs;
    // Format params
    pfmt:{k:key[x] except enlist`x;$[0=count k;"none";" " sv{(string x),"=",string y}'[k;x k]]};
    ([]signal:names;
       params:pfmt each bestParams;
       sharpe:mets`sharpe;
       ic:mets`ic;
       hitRate:mets`hitRate;
       turnover:mets`turnover;
       maxDD:mets`maxDD;
       profitFactor:mets`profitFactor)}

// -----------------------------------------------------------------------------
// FORWARD SIMULATION
// -----------------------------------------------------------------------------

// Rolling volatility (alias for mdev)
rollingVol:{[w;x] mdev[w;x]}

// Estimate AR(1) coefficient (phi) from data
// Returns trailing phi estimate
estimatePhi:{[w;x]
    xlag:prev x;
    rbeta[w;xlag;x]}

// Box-Muller transform for normal random numbers
// n - count of random numbers needed
randNorm:{[n]
    u1:n?1f; u2:n?1f;
    sqrt[neg 2 * log u1] * cos 2 * 3.14159265359 * u2}

// Student-t random numbers via ratio of uniforms
// n  - count of random numbers
// df - degrees of freedom (higher = closer to normal, lower = fatter tails)
// Note: df > 2 for finite variance, df > 4 for finite kurtosis
randT:{[n;df]
    // Use normal/chi-squared ratio: T = Z / sqrt(V/df) where V ~ chi-sq(df)
    z:randNorm n;
    // Chi-squared via sum of squared normals (approximation for speed)
    v:sum each (ceiling df) cut randNorm[n * ceiling df] xexp 2;
    z % sqrt v % df}

// Skew-normal random numbers (Azzalini method)
// n     - count of random numbers
// alpha - skewness parameter (0 = normal, positive = right skew, negative = left skew)
randSkewNorm:{[n;alpha]
    delta:alpha % sqrt 1 + alpha * alpha;
    u0:randNorm n;
    u1:randNorm n;
    // Correlated normal construction
    delta * abs[u0] + sqrt[1 - delta * delta] * u1}

// Empirical distribution: sample from historical data with replacement
// rets - historical returns/changes
// n    - count of samples
randEmpirical:{[rets;n]
    rets n?count rets}

// Random number dispatcher
// n      - count of random numbers
// dist   - `normal, `t, `skew, or `empirical
// params - dist-specific: `t needs `df; `skew needs `alpha; `empirical needs `rets
randDist:{[n;dist;params]
    $[dist ~ `normal;   randNorm n;
      dist ~ `t;        randT[n;params`df];
      dist ~ `skew;     randSkewNorm[n;params`alpha];
      dist ~ `empirical; randEmpirical[params`rets;n];
      '`unknownDist]}

// -----------------------------------------------------------------------------
// MULTI-STEP PATH SIMULATION
// -----------------------------------------------------------------------------

// Generate multi-step forward paths (matrix output)
// x      - historical series
// nPaths - number of simulation paths
// nSteps - number of forward steps
// method - `rw (random walk), `ar1 (mean-reverting), `boot (bootstrap)
// params - dict with:
//          `vol (float) - volatility
//          `phi (float) - AR(1) coefficient (for ar1 method)
//          `window (int) - lookback for bootstrap
//          `dist (sym) - distribution: `normal (default), `t, `skew, `empirical
//          `df (int) - degrees of freedom for t-dist
//          `alpha (float) - skewness for skew-normal
// Returns: nPaths x nSteps matrix of path values (cumulative from x_last)
simPathsND:{[x;nPaths;nSteps;method;params]
    xlast:last x;
    vol:params`vol;
    dist:$[`dist in key params; params`dist; `normal];

    // Generate all random shocks: nPaths x nSteps
    shocks:$[dist ~ `normal;
                (nPaths;nSteps)#randNorm nPaths * nSteps;
             dist ~ `t;
                (nPaths;nSteps)#randT[nPaths * nSteps;params`df];
             dist ~ `skew;
                (nPaths;nSteps)#randSkewNorm[nPaths * nSteps;params`alpha];
             dist ~ `empirical;
                // For empirical, sample from recent returns
                rets:neg[params`window]#1 _ deltas x;
                (nPaths;nSteps)#randEmpirical[rets;nPaths * nSteps];
             '`unknownDist];

    // AR(1) helper: iterate returns with autocorrelation, then cumsum
    ar1Path:{[phi;vol;x0;eps] n:count eps; rets:n#0f; rets[0]:vol*eps 0; i:1; while[i<n; rets[i]:(phi*rets i-1)+vol*eps i; i+:1]; x0+sums rets};

    $[method ~ `rw;
        xlast + vol * sums each shocks;
      method ~ `ar1;
        ar1Path[params`phi;vol;xlast;] each shocks;
      method ~ `boot;
        xlast + sums each shocks;
      '`unknownSimMethod]}

// Simulate signal change over multiple steps
// x       - historical series
// sigFn   - signal function (unary)
// nPaths  - number of simulation paths
// nSteps  - number of forward steps
// method  - `rw, `ar1, or `boot
// params  - method-specific parameters dict
// Returns: dict with `currentSig`pathSigs`meanByStep`stdByStep`finalMean`finalStd
simSignalChangeND:{[x;sigFn;nPaths;nSteps;method;params]
    currentSig:last sigFn x;
    paths:simPathsND[x;nPaths;nSteps;method;params];
    // Apply signal to each path - use peach for parallel if available
    pathSigs:{[sigFn;x;path] {[sigFn;x;i;path] last sigFn x,i#path}[sigFn;x;;path] each 1+til count path}[sigFn;x] peach paths;
    byStep:flip pathSigs;
    meanByStep:avg each byStep;
    stdByStep:dev each byStep;
    finalSigs:last each pathSigs;
    finalChanges:finalSigs - currentSig;
    `currentSig`pathSigs`meanByStep`stdByStep`finalMean`finalStd!(currentSig;pathSigs;meanByStep;stdByStep;avg finalChanges;dev finalChanges)}

// Fast version - combined signal change + percentiles in one pass (avoids duplicate computation)
// Optimization: call sigFn once per path instead of nSteps times (nPaths calls vs nPaths*nSteps)
simSignalFast:{[x;sigFn;nPaths;nSteps;method;params]
    currentSig:last sigFn x;
    paths:simPathsND[x;nPaths;nSteps;method;params];
    pathSigs:{[sigFn;x;nSteps;path] (neg nSteps)#sigFn x,path}[sigFn;x;nSteps;] peach paths;
    byStep:flip pathSigs;
    meanByStep:avg each byStep;
    stdByStep:dev each byStep;
    pctls:{s:asc x; s `long$(0.05 0.25 0.5 0.75 0.95)*count x} each byStep;
    finalSigs:last each pathSigs;
    pUp:(sum finalSigs > currentSig) % nPaths;
    pCross:$[currentSig>0;sum finalSigs<0;sum finalSigs>0] % nPaths;
    `currentSig`meanByStep`stdByStep`p5`p25`p50`p75`p95`probUp`probCross0!(currentSig;meanByStep;stdByStep;pctls[;0];pctls[;1];pctls[;2];pctls[;3];pctls[;4];pUp;pCross)}

// Convenience: simulate EMA z-score over multiple steps
simEmaZscoreND:{[x;emaHL;zWindow;nPaths;nSteps;method;params]
    sigFn:{[hl;w;s] rzscore[w;smooth[s;hl]]}[emaHL;zWindow];
    simSignalChangeND[x;sigFn;nPaths;nSteps;method;params]}

// Convenience: simulate rolling z-score over multiple steps
simRzscoreND:{[x;window;nPaths;nSteps;method;params]
    sigFn:rzscore[window;];
    simSignalChangeND[x;sigFn;nPaths;nSteps;method;params]}

// Percentile bands from simulation
// Returns dict with `p5`p25`p50`p75`p95 by step
simPercentiles:{[x;sigFn;nPaths;nSteps;method;params]
    result:simSignalChangeND[x;sigFn;nPaths;nSteps;method;params];
    pathSigs:result`pathSigs;
    // Transpose to get values by step
    byStep:flip pathSigs;
    pctls:{s:asc x; s `long$(0.05 0.25 0.5 0.75 0.95) * count x} each byStep;
    `p5`p25`p50`p75`p95`currentSig!(pctls[;0];pctls[;1];pctls[;2];pctls[;3];pctls[;4];result`currentSig)}

// -----------------------------------------------------------------------------
// DISTRIBUTION FITTING
// -----------------------------------------------------------------------------

// Estimate t-distribution df from data (method of moments)
// Lower df = fatter tails
estimateDF:{[x]
    // Compute excess kurtosis directly
    n:count x; m:avg x; s:dev x;
    k:((avg (x - m) xexp 4) % s xexp 4) - 3;
    // For t-dist: kurtosis = 6/(df-4) for df > 4
    // Solving: df = 4 + 6/kurtosis
    // Clamp to reasonable range [3, 30]
    3 | 30 & `int$4 + 6 % 0.01 | k}

// Estimate skewness parameter alpha for skew-normal
estimateAlpha:{[x]
    // Compute skewness directly
    n:count x; m:avg x; s:dev x;
    sk:(avg (x - m) xexp 3) % s xexp 3;
    // Approximate: alpha ≈ skewness for small skewness
    // Clamp to reasonable range [-5, 5]
    -5f | 5f & sk}

// Auto-estimate simulation params including distribution
// x      - historical series
// window - lookback for estimation
// method - `rw, `ar1, or `boot
// dist   - `normal, `t, `skew, or `empirical
// Returns: params dict suitable for simPathsND
autoSimParamsND:{[x;window;method;dist]
    vol:last rollingVol[window;x];
    phi:last estimatePhi[window;x];
    base:$[method ~ `rw;    `vol`dist!(vol;dist);
           method ~ `ar1;   `vol`phi`dist!(vol;phi;dist);
           method ~ `boot;  `window`dist!(window;dist);
           '`unknownSimMethod];
    // Add distribution-specific params
    $[dist ~ `t;        base,enlist[`df]!enlist estimateDF neg[window]#x;
      dist ~ `skew;     base,enlist[`alpha]!enlist estimateAlpha neg[window]#x;
      dist ~ `empirical; base;  // no extra params needed
      base]}  // normal needs no extra params

// Generate 1-day forward paths
// x      - historical series
// nPaths - number of simulation paths
// method - `rw (random walk), `ar1 (mean-reverting), `boot (bootstrap)
// params - dict with method-specific params:
//          `rw: `vol (float) - volatility for random walk
//          `ar1: `vol`phi (floats) - volatility and AR coefficient
//          `boot: `window (int) - lookback for resampling returns
simPaths1D:{[x;nPaths;method;params]
    xlast:last x;
    $[method ~ `rw;
        // Random walk: x_last + N(0, vol^2)
        xlast + params[`vol] * randNorm nPaths;
      method ~ `ar1;
        // AR(1): phi * x_last + N(0, vol^2)
        // vol is the innovation (shock) std dev
        (params[`phi] * xlast) + params[`vol] * randNorm nPaths;
      method ~ `boot;
        // Bootstrap: x_last + sample from recent returns
        xlast + (neg[params`window]#1 _ deltas x) nPaths?params`window;
      '`unknownSimMethod]}

// Apply signal function to series extended by each simulated path
// sigFn - signal function (unary: takes series, returns series of same length)
// x     - historical series
// paths - nPaths simulated forward values
// Returns: nPaths signal values (last value from each extended series)
applySignalToPath:{[sigFn;x;paths]
    {[sigFn;x;p] last sigFn x,p}[sigFn;x] each paths}

// Main API: simulate expected signal change
// x      - historical series
// sigFn  - signal function (unary)
// nPaths - number of simulation paths
// method - `rw, `ar1, or `boot
// params - method-specific parameters dict
// Returns: dict with `currentSig`mean`std`simSigs
simSignalChange:{[x;sigFn;nPaths;method;params]
    // Current signal value
    currentSig:last sigFn x;
    // Generate forward paths
    paths:simPaths1D[x;nPaths;method;params];
    // Apply signal to each path
    simSigs:applySignalToPath[sigFn;x;paths];
    // Compute statistics
    changes:simSigs - currentSig;
    `currentSig`mean`std`simSigs!(currentSig;avg changes;dev changes;simSigs)}

// Convenience: simulate EMA z-score signal change
// x       - historical series
// emaHL   - EMA halflife
// zWindow - z-score window
// nPaths  - number of paths
// method  - simulation method
// params  - method params
simEmaZscore:{[x;emaHL;zWindow;nPaths;method;params]
    sigFn:{[hl;w;s] rzscore[w;smooth[s;hl]]}[emaHL;zWindow];
    simSignalChange[x;sigFn;nPaths;method;params]}

// Convenience: simulate rolling z-score signal change
simRzscore:{[x;window;nPaths;method;params]
    sigFn:rzscore[window;];
    simSignalChange[x;sigFn;nPaths;method;params]}

// Auto-estimate simulation params from data
// x      - historical series
// window - lookback for estimation
// method - `rw, `ar1, or `boot
// Returns: params dict suitable for simPaths1D
autoSimParams:{[x;window;method]
    vol:last rollingVol[window;x];
    phi:last estimatePhi[window;x];
    $[method ~ `rw;
        enlist[`vol]!enlist vol;
      method ~ `ar1;
        `vol`phi!(vol;phi);
      method ~ `boot;
        enlist[`window]!enlist window;
      '`unknownSimMethod]}

// -----------------------------------------------------------------------------
// COMBINATION
// -----------------------------------------------------------------------------

// Blend: weighted average of multiple signals
blend:{[signals;weights] wts:weights % sum weights; sum wts * signals}

// BlendAdaptive: blend signals weighted by their rolling IC
blendAdaptive:{[signals;fwdRet;window] ics:{[f;r;w] rollingIC[f;r;w]}[;fwdRet;window] each signals; icsPos:0f|/:ics; wts:icsPos %\: sum each flip icsPos; sum signals * wts}

// Switch: use f1 when condition true, else f2
switch:{[f1;f2;cond] (ffill[f1] * cond) + ffill[f2] * not cond}

// SwitchRegime: use f1 in high regime, f2 in low regime
switchRegime:{[f1;f2;regime;window;threshold] rn:rzscore[window;ffill regime]; switch[f1;f2;rn > threshold]}

// Stack: apply multiple conditioning functions in sequence
// ops is list of (func;args) where func is conditioning function
stack:{[f;ops] {[f;op] op[0][f],op[1]}[;]/[f;ops]}

// Best: select signal with best recent IC at each point
best:{[signals;fwdRet;window] ics:{[f;r;w] rollingIC[f;r;w]}[;fwdRet;window] each signals; bestIdx:ics?/:max each flip ics; signals ./: flip (til count first signals;bestIdx)}

// Ensemble: average of top N signals by recent IC
ensemble:{[signals;fwdRet;window;topN] ics:{[f;r;w] rollingIC[f;r;w]}[;fwdRet;window] each signals; ranked:{[n;x] x rank neg x}[topN] each flip ics; wts:(ranked < topN) % topN; sum signals * wts}

// -----------------------------------------------------------------------------
// TABLE INTERFACE
// -----------------------------------------------------------------------------

// Apply conditioning to table columns, add new column
apply:{[t;f1col;f2col;method;params]
    f1:t f1col; f2:t f2col;
    newcol:`$string[f1col],"_",string[method];
    result:$[method~`gate;gate[f1;f2;params 0;params 1;params 2];method~`gateBetween;gateBetween[f1;f2;params 0;params 1;params 2;params 3];method~`scale;scale[f1;f2;params 0;params 1];method~`scalePos;scalePos[f1;f2;params 0;params 1];method~`percentile;percentile[f1;f2;params 0;params 1;params 2];method~`top;top[f1;f2;params 0;params 1];method~`bottom;bottom[f1;f2;params 0;params 1];method~`residualize;residualize[f1;f2;params 0];method~`interact;interact[f1;f2;params 0;params 1];method~`tilt;tilt[f1;f2;params 0;params 1;params 2];'`unknownMethod];
    ![t;();0b;enlist[newcol]!enlist result]}

// Analyze f1 performance by f2 regime
analyze:{[t;f1col;f2col;fwdRetCol;window;nBuckets]
    f1:t f1col; f2:t f2col; fwdRet:t fwdRetCol;
    f2r:rrank[window;ffill f2];
    buckets:(nBuckets-1)&`long$nBuckets*f2r;
    stats:{[f1;fwdRet;buckets;b] idx:where buckets=b; f1b:f1 idx; retb:fwdRet idx; `bucket`n`ic`meanRet`hitRate!(b;count idx;cor[f1b;retb];avg retb;avg retb>0)}[f1;fwdRet;buckets] each til nBuckets;
    flip stats}

// IC by regime
icByRegime:{[f1;f2;fwdRet;window;nBuckets] f2r:rrank[window;ffill f2]; buckets:(nBuckets-1)&`long$nBuckets*f2r; {[f1;fwdRet;buckets;b] idx:where buckets=b; cor[f1 idx;fwdRet idx]}[ffill f1;fwdRet;buckets] each til nBuckets}

// Rolling IC
rollingIC:{[f1;fwdRet;window] wins1:{1_x,y}\[window#0n;ffill f1]; wins2:{1_x,y}\[window#0n;fwdRet]; {$[(count x)<2;0n;any null x,y;0n;cor[x where not null x;y where not null x]]}.' flip (wins1;wins2)}

// -----------------------------------------------------------------------------
// TABLE-BASED SIMULATION
// -----------------------------------------------------------------------------

// Simulate signal forward by sym from a table
// t        - table with sym, date/time, and price columns
// symCol   - symbol column name (e.g., `sym)
// priceCol - price column name (e.g., `price or `close)
// sigFn    - signal function (takes price vector, returns signal vector)
// nPaths   - number of Monte Carlo paths
// nSteps   - number of forward steps
// method   - `rw, `ar1, or `boot
// params   - simulation params dict (if `vol not provided, estimated per sym; add `parallel for peach)
// Returns: table with sym, currentSig, finalMean, finalStd, p5, p50, p95, probUp, probCross0
simBySym:{[t;symCol;priceCol;sigFn;nPaths;nSteps;method;params]
    syms:distinct t symCol;
    getPrices:{[t;symCol;priceCol;s] ?[t;enlist (=;symCol;enlist s);();priceCol]};
    priceVecs:getPrices[t;symCol;priceCol;] each syms;
    // Use fast combined function, parallel over syms if requested
    runOne:{[sigFn;nPaths;nSteps;method;params;prices] p:$[`vol in key params;params;params,enlist[`vol]!enlist dev 1_deltas prices]; r:simSignalFast[prices;sigFn;nPaths;nSteps;method;p]; `currentSig`finalMean`finalStd`p5`p50`p95`probUp`probCross0!(r`currentSig;last r`meanByStep - r`currentSig;last r`stdByStep;last r`p5;last r`p50;last r`p95;r`probUp;r`probCross0)};
    iter:$[`parallel in key params;$[params`parallel;peach;each];each];
    results:runOne[sigFn;nPaths;nSteps;method;params;] iter priceVecs;
    ([] sym:syms; currentSig:results@\:`currentSig; finalMean:results@\:`finalMean; finalStd:results@\:`finalStd; p5:results@\:`p5; p50:results@\:`p50; p95:results@\:`p95; probUp:results@\:`probUp; probCross0:results@\:`probCross0)}

// Snapshot simulation - run once per sym using full history (fast, single point)
// Returns: table with one row per sym
simSnapshot:{[t;cfg]
    symCol:cfg`sym; priceCol:cfg`price; sigFn:cfg`sigFn;
    nPaths:cfg`nPaths; nSteps:cfg`nSteps; method:cfg`method;
    simBySym[t;symCol;priceCol;sigFn;nPaths;nSteps;method;cfg]}

// Rolling simulation - run at EACH date for each sym
// t   - table with dt/sym/price columns (sorted by sym,dt)
// cfg - dict with keys:
//       Required: `dt`sym`price`sigFn`nPaths`nSteps`method`minHist
//       Optional: `vol`dist`df`phi`window`parallel (1b to use peach)
// minHist - minimum history required before running sim (e.g., 60 for 60-day lookback)
// Returns: table with dt, sym, currentSig, p5, p50, p95, probUp, probCross0

// Helper: run sim at one index using fast combined function
// Auto-estimates missing params: vol, df (for t-dist), alpha (for skew)
simAtIdx:{[cfg;prices;i]
    px:prices til i+1;
    rets:1_deltas px;
    p:cfg;
    if[not `vol in key p; p:p,enlist[`vol]!enlist dev rets];
    if[(p[`dist]~`t) and not `df in key p; p:p,enlist[`df]!enlist estimateDF rets];
    if[(p[`dist]~`skew) and not `alpha in key p; p:p,enlist[`alpha]!enlist estimateAlpha rets];
    r:simSignalFast[px;cfg`sigFn;cfg`nPaths;cfg`nSteps;cfg`method;p];
    `currentSig`mean`std`p5`p50`p95`probUp`probCross0!(r`currentSig;last r`meanByStep;last r`stdByStep;last r`p5;last r`p50;last r`p95;r`probUp;r`probCross0)}

// Helper: run sim for one sym's data (parallel over dates if cfg`parallel)
// Sort group by date first to ensure correct time ordering
simOneSym:{[cfg;grp] grp:cfg[`dt] xasc grp; dts:grp cfg`dt; prices:grp cfg`price; n:count prices; mh:cfg`minHist; idxs:mh+til n-mh; res:$[cfg`parallel; simAtIdx[cfg;prices;] peach idxs; simAtIdx[cfg;prices;] each idxs]; ([] dt:dts idxs; currentSig:res@\:`currentSig; mean:res@\:`mean; std:res@\:`std; p5:res@\:`p5; p50:res@\:`p50; p95:res@\:`p95; probUp:res@\:`probUp; probCross0:res@\:`probCross0)}

// Main rolling sim - parallel over syms if cfg`parallel
// Defaults: minHist=60, nPaths=100, nSteps=5, method=`rw, dist=`normal, parallel=0b
simRolling:{[t;cfg] defaults:`minHist`nPaths`nSteps`method`dist`parallel!(60;100;5;`rw;`normal;0b); cfg:defaults,cfg; bySym:t group t cfg`sym; pairs:flip (key bySym;value bySym); results:$[cfg`parallel; {[cfg;p] r:simOneSym[cfg;p 1]; update sym:p 0 from r}[cfg;] peach pairs; {[cfg;p] r:simOneSym[cfg;p 1]; update sym:p 0 from r}[cfg;] each pairs]; raze results}

// Convenience wrapper - all config in dict
// cfg must have: `dt`sym`price`sigFn`nPaths`nSteps`method`minHist
simTable:{[t;cfg] simRolling[t;cfg]}

// Convenience: EMA z-score rolling simulation
// cfg must have: `dt`sym`price`emaHL`zWindow`nPaths`nSteps`method`minHist
simEmaZscoreRolling:{[t;cfg] cfg[`sigFn]:{[hl;w;x] rzscore[w;smooth[x;hl]]}[cfg`emaHL;cfg`zWindow]; simRolling[t;cfg]}

// Convenience: rolling z-score simulation
// cfg must have: `dt`sym`price`window`nPaths`nSteps`method`minHist
simRzscoreRolling:{[t;cfg] cfg[`sigFn]:rzscore[cfg`window;]; simRolling[t;cfg]}

// Legacy alias
// cfg dict must include: `sym`price`window`nPaths`nSteps`method
simRzscoreBySym:{[t;cfg] cfg[`sigFn]:rzscore[cfg`window;]; simTable[t;cfg]}

// -----------------------------------------------------------------------------
// HELP
// -----------------------------------------------------------------------------

help:{[]
    -1 "";
    -1 "=== .cond SIGNAL CONDITIONING v0.3.0 ===";
    -1 "";
    -1 "NORMALIZATION: `zscore `rank `percentile `minmax `raw";
    -1 "";
    -1 "PRIMITIVES:";
    -1 "  rzscore[w;x]     rrank[w;x]      rminmax[w;x]     rmean[w;x]   rstd[w;x]";
    -1 "  rbeta[w;x;y]     ralpha[w;x;y]   rresid[w;x;y]    rrsq[w;x;y]";
    -1 "";
    -1 "CORE CONDITIONING:";
    -1 "  gate[f1;f2;w;ntype;thresh]       - f1 where norm(f2)>thresh";
    -1 "  gateBetween[f1;f2;w;ntype;lo;hi] - f1 where norm(f2) in [lo,hi]";
    -1 "  scale[f1;f2;w;ntype]             - f1 * norm(f2)";
    -1 "  scalePos[f1;f2;w;ntype]          - f1 * max(0,norm(f2))";
    -1 "  percentile[f1;f2;w;loP;hiP]      - f1 where rank(f2) in [loP,hiP]";
    -1 "  top[f1;f2;w;pct]                 - f1 where f2 in top pct";
    -1 "  bottom[f1;f2;w;pct]              - f1 where f2 in bottom pct";
    -1 "  regime[f1;f2;w;nBuckets]         - table with regime labels";
    -1 "  residualize[f1;f2;w]             - f1 orthogonalized vs f2";
    -1 "  interact[f1;f2;w;ntype]          - norm(f1) * norm(f2)";
    -1 "  tilt[f1;f2;w;ntype;weight]       - blend f1 toward f2";
    -1 "";
    -1 "SIGNAL PROCESSING:";
    -1 "  smooth[f;halflife]               - EMA smoothing";
    -1 "  clip[f;w;loPct;hiPct]            - rolling percentile clip";
    -1 "  winsorize[f;w;loPct;hiPct]       - alias for clip";
    -1 "  decay[f;halflife]                - EMA decay (alias for smooth)";
    -1 "  lag[f;n]                         - shift signal by n periods";
    -1 "  diff[f;n]                        - signal change (momentum)";
    -1 "  pctChange[f;n]                   - percentage change";
    -1 "";
    -1 "MEAN-REVERSION:";
    -1 "  accumTension[w;x]                - z-score of cumsum (rolling tension)";
    -1 "  snapRevert[ten;trig;mode;hl;mh]  - mean-revert on trigger, track or decay";
    -1 "";
    -1 "CONVICTION / CONFIDENCE:";
    -1 "  conviction[f;w;ntype]            - scale by |signal| magnitude";
    -1 "  confidence[f;fwdRet;w]           - weight by rolling IC";
    -1 "  agree[f1;f2]                     - f1 when same sign as f2";
    -1 "  agreeN[f1;signals;minN]          - f1 when N signals agree";
    -1 "  disagree[f1;f2]                  - f1 when opposite sign (contrarian)";
    -1 "  confirm[f1;f2;w;ntype;thresh]    - f1 when f2 confirms direction";
    -1 "";
    -1 "RISK-BASED:";
    -1 "  volAdjust[f;w;targetVol]         - scale by inverse vol";
    -1 "  drawdownGate[f;w;ddThresh]       - gate by drawdown level";
    -1 "  sharpeGate[f;fwdRet;w;minSharpe] - gate by rolling Sharpe";
    -1 "  icDecay[f;fwdRet;w]              - decay by rolling IC";
    -1 "  hitRateGate[f;fwdRet;w;minHR]    - gate by hit rate";
    -1 "  maxLossGate[f;fwdRet;w;maxLoss]  - gate after large losses";
    -1 "";
    -1 "CROSS-SECTIONAL (matrix input: rows=time, cols=assets):";
    -1 "  csRank[M]                        - rank across assets";
    -1 "  csZscore[M]                      - z-score across assets";
    -1 "  csNeutralize[M]                  - demean (market neutral)";
    -1 "  csWinsorize[M;loPct;hiPct]       - winsorize across assets";
    -1 "";
    -1 "CALENDAR PROXIMITY (date-based filters):";
    -1 "  daysToMonthEnd[dates]            - biz days to month-end (neg=before)";
    -1 "  daysToQuarterEnd[dates]          - biz days to quarter-end";
    -1 "  monthEndGaussian[dates;off;sig]  - Gaussian decay, peak at offset";
    -1 "  monthEndExp[dates;off;hl]        - exponential decay, halflife hl";
    -1 "  quarterEndGaussian[dates;off;sig]- quarter-end Gaussian";
    -1 "  quarterEndExp[dates;off;hl]      - quarter-end exponential";
    -1 "  proximityFilter[d;off;dec;p;evt] - generic (dec=`gaussian`exp)";
    -1 "";
    -1 "FORWARD SIMULATION:";
    -1 "  simPaths1D[x;n;meth;params]      - generate n 1-day forward paths";
    -1 "  simPathsND[x;n;steps;m;p]        - generate n multi-step paths (matrix)";
    -1 "  simSignalChange[x;fn;n;m;p]      - expected 1-step signal change";
    -1 "  simSignalChangeND[x;fn;n;s;m;p]  - multi-step signal change";
    -1 "  simPercentiles[x;fn;n;s;m;p]     - percentile bands by step";
    -1 "  simEmaZscore[x;hl;w;n;m;p]       - simulate EMA z-score (1-step)";
    -1 "  simEmaZscoreND[x;hl;w;n;s;m;p]   - simulate EMA z-score (N-step)";
    -1 "  simRzscore[x;w;n;m;p]            - simulate z-score (1-step)";
    -1 "  simRzscoreND[x;w;n;s;m;p]        - simulate z-score (N-step)";
    -1 "  autoSimParams[x;w;method]        - auto-estimate 1-step params";
    -1 "  autoSimParamsND[x;w;method;dist] - auto-estimate with distribution";
    -1 "  Methods: `rw (random walk), `ar1 (mean-revert), `boot (bootstrap)";
    -1 "  Distributions: `normal (default), `t (fat tails), `skew, `empirical";
    -1 "";
    -1 "COMBINATION:";
    -1 "  blend[signals;weights]           - weighted average";
    -1 "  blendAdaptive[signals;fwdRet;w]  - blend by rolling IC";
    -1 "  switch[f1;f2;cond]               - f1 when cond, else f2";
    -1 "  switchRegime[f1;f2;reg;w;thresh] - regime-based switching";
    -1 "  best[signals;fwdRet;w]           - best signal by IC";
    -1 "  ensemble[signals;fwdRet;w;topN]  - avg of top N by IC";
    -1 "";
    -1 "TABLE INTERFACE:";
    -1 "  apply[t;`f1;`f2;method;params]   - add conditioned column";
    -1 "  analyze[t;`f1;`f2;`ret;w;nB]     - stats by regime";
    -1 "  rollingIC[f;fwdRet;w]            - rolling information coefficient";
    -1 "  simTable[t;cfg]                  - simulate any signal by sym";
    -1 "    cfg keys: `sym`price`sigFn`nPaths`nSteps`method + optional `vol`dist`df";
    -1 "  simEmaZscoreBySym[t;cfg]         - EMA z-score (cfg adds `emaHL`zWindow)";
    -1 "  simRzscoreBySym[t;cfg]           - z-score (cfg adds `window)";
    -1 "";
    -1 "EXAMPLES:";
    -1 "  .cond.gate[alpha;vol;60;`zscore;0]         // alpha when vol z>0";
    -1 "  .cond.top[alpha;mom;60;0.2]                // top 20% momentum";
    -1 "  .cond.smooth[alpha;10]                     // 10-period EMA";
    -1 "  .cond.confidence[alpha;ret;60]             // weight by IC";
    -1 "  .cond.agree[alpha1;alpha2]                 // when both agree";
    -1 "  .cond.volAdjust[alpha;60;0.01]             // vol-target to 1%";
    -1 "  .cond.blend[(a1;a2;a3);(0.5;0.3;0.2)]      // weighted blend";
    -1 "  .cond.switchRegime[momAlpha;meanRevAlpha;vix;60;1]";
    -1 "";}

// Generate sample data for examples
exampleData:{[]
    system "S 42";
    n:252;
    alpha1:sums (n?1f) - 0.5;
    alpha2:(n?1f) - 0.5;
    alpha3:0.5*alpha1 + 0.5*alpha2;
    vol:0.01 + 0.02 * abs (n?1f)-0.5;
    momentum:mavg[20;alpha1];
    fwdRet:0.2*alpha1 + 0.1*alpha2 + 0.7*((n?1f)-0.5);
    t:([] date:2024.01.01 + til n; alpha1:alpha1; alpha2:alpha2; vol:vol; momentum:momentum; fwdRet:fwdRet);
    `n`alpha1`alpha2`alpha3`vol`momentum`fwdRet`t!(n;alpha1;alpha2;alpha3;vol;momentum;fwdRet;t)}

// Example usage
example:{[]
    -1 "=== .cond SIGNAL CONDITIONING EXAMPLES ===";
    -1 "";
    -1 "Generate sample data: d:.cond.exampleData[]";
    -1 "";
    d:exampleData[];
    -1 "Sample data:";
    -1 "  n = ",string d`n;
    -1 "  alpha1, alpha2, alpha3: signal vectors";
    -1 "  vol, momentum: conditioning signals";
    -1 "  fwdRet: forward returns";
    -1 "";
    -1 "--- ROLLING PRIMITIVES ---";
    -1 "  rzscore[60;alpha1]  - Rolling z-score";
    -1 "  rrank[60;alpha1]    - Rolling percentile rank";
    -1 "  rbeta[60;x;y]       - Rolling beta";
    -1 "  rresid[60;x;y]      - Rolling residuals";
    -1 "";
    -1 "--- CORE CONDITIONING ---";
    -1 "  gate[f1;f2;w;ntype;thresh]   - f1 when norm(f2)>thresh";
    -1 "  scale[f1;f2;w;ntype]         - f1 * norm(f2)";
    -1 "  top[f1;f2;w;pct]             - f1 in top pct of f2";
    -1 "  residualize[f1;f2;w]         - orthogonalize f1 vs f2";
    -1 "  interact[f1;f2;w;ntype]      - norm(f1) * norm(f2)";
    -1 "";
    -1 "--- SIGNAL PROCESSING ---";
    -1 "  smooth[f;halflife]           - EMA smoothing";
    -1 "  clip[f;loPct;hiPct]          - Winsorize";
    -1 "  lag[f;n]                     - Shift signal";
    -1 "  diff[f;n]                    - Signal change";
    -1 "";
    -1 "--- CONVICTION/CONFIDENCE ---";
    -1 "  conviction[f;w;ntype]        - Scale by |signal|";
    -1 "  confidence[f;fwdRet;w]       - Weight by IC";
    -1 "  agree[f1;f2]                 - f1 when same sign";
    -1 "  disagree[f1;f2]              - f1 when opposite sign";
    -1 "";
    -1 "--- RISK-BASED ---";
    -1 "  volAdjust[f;w;targetVol]     - Inverse vol scaling";
    -1 "  hitRateGate[f;ret;w;minHR]   - Gate by hit rate";
    -1 "  icDecay[f;ret;w]             - Decay by IC";
    -1 "";
    -1 "--- CROSS-SECTIONAL (matrix input) ---";
    -1 "  csRank[M]                    - Rank across assets";
    -1 "  csZscore[M]                  - Z-score across assets";
    -1 "  csNeutralize[M]              - Demean (market neutral)";
    -1 "";
    -1 "--- COMBINATION ---";
    -1 "  blend[signals;weights]       - Weighted average";
    -1 "  blendAdaptive[sigs;ret;w]    - Blend by IC";
    -1 "  switch[f1;f2;cond]           - f1 if cond else f2";
    -1 "  best[sigs;ret;w]             - Best by IC";
    -1 "  ensemble[sigs;ret;w;topN]    - Avg top N";
    -1 "";
    -1 "--- TABLE INTERFACE ---";
    -1 "  apply[t;`f1;`f2;method;params] - Add column";
    -1 "  analyze[t;`f1;`f2;`ret;w;nB]   - Stats by regime";
    -1 "  rollingIC[f;ret;w]             - Rolling IC";
    -1 "";
    -1 "NORMALIZATION TYPES: `zscore `rank `percentile `minmax `raw";
    -1 "";
    -1 "EXAMPLE USAGE:";
    -1 "  d:.cond.exampleData[]";
    -1 "  .cond.gate[d`alpha1;d`vol;60;`zscore;0]";
    -1 "  .cond.scale[d`alpha1;d`momentum;60;`rank]";
    -1 "  .cond.top[d`alpha1;d`momentum;60;0.3]";
    -1 "  .cond.blendAdaptive[(d`alpha1;d`alpha2);d`fwdRet;60]";
    -1 "";
    d}

\d .

-1 "Loaded .cond namespace v0.3.0";
-1 "Signal conditioning: gate, scale, smooth, confidence, volAdjust, blend, switch + more";
-1 "Run .cond.help[] for full function list";
