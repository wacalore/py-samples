// =============================================================================
// SIGNAL CONDITIONING
// =============================================================================
// Condition one alpha/signal on another using rolling methods
// Version: 0.1.0

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

// Smooth: exponential moving average
smooth:{[f;halflife] lambda:exp neg log[2]%halflife; {(y*1-x)+x*z}[lambda]\[first f;ffill f]}

// Clip: cap values at percentile bounds
clip:{[f;loPct;hiPct] s:asc f; lo:s `long$loPct*count s; hi:s `long$hiPct*count s; lo|f&hi}

// Winsorize: same as clip (alias)
winsorize:clip

// Decay: exponential decay weighting (recent values matter more)
decay:{[f;halflife] lambda:exp neg log[2]%halflife; n:count f; wts:lambda xexp reverse til n; f * wts % max wts}

// Lag: shift signal by n periods (positive = look back)
lag:{[f;n] $[n>0; (n#0n),neg[n]_f; (neg[n]_f),abs[n]#0n]}

// Diff: signal change (momentum of signal)
diff:{[f;n] f - lag[f;n]}

// Pctchange: percentage change of signal
pctChange:{[f;n] prev_f:lag[f;n]; (f - prev_f) % abs prev_f}

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
// HELP
// -----------------------------------------------------------------------------

help:{[]
    -1 "";
    -1 "=== .cond SIGNAL CONDITIONING v0.2.0 ===";
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
    -1 "  clip[f;loPct;hiPct]              - cap at percentile bounds";
    -1 "  winsorize[f;loPct;hiPct]         - alias for clip";
    -1 "  decay[f;halflife]                - exponential decay weighting";
    -1 "  lag[f;n]                         - shift signal by n periods";
    -1 "  diff[f;n]                        - signal change (momentum)";
    -1 "  pctChange[f;n]                   - percentage change";
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

-1 "Loaded .cond namespace v0.2.0";
-1 "Signal conditioning: gate, scale, smooth, confidence, volAdjust, blend, switch + more";
-1 "Run .cond.help[] for full function list";
