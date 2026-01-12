// kdbtools.q - Quantitative Signal Library for KDB+/Q
// Generic time series utilities for signal generation
// All functions are instrument-agnostic and work on any numeric series
// Namespace: .kdbtools

// =============================================================================
// DEVELOPER NOTES
// =============================================================================
//
// Q ENVIRONMENT:
//   Executable: /Users/tonycalore/Downloads/m64/m64/q
//   License:    /Users/tonycalore/Downloads/m64/m64/kc.lic
//   QML:        /Users/tonycalore/Downloads/m64/m64/qml.q (run q from this dir)
//
// KDB/Q EVALUATION ORDER:
//   Q evaluates RIGHT TO LEFT. This affects operator precedence:
//     5 + 2*x - 1.5*y  -->  5 + (2*(x - (1.5*y)))   // NOT what you expect!
//     5 + (2*x) - (1.5*y)  -->  correct form        // Use explicit parens
//
// PERFORMANCE PRINCIPLES:
//   1. VECTORIZE EVERYTHING - operate on entire columns/lists, not row-by-row
//   2. AVOID LOOPS - use built-in vector ops (each, over, scan) not while/do
//   3. USE BUILT-INS - mavg, msum, mdev, deltas, ratios are optimized C code
//   4. TABLE-NATIVE - write functions that operate on table columns directly:
//        GOOD: update sig:mavg[20;price] from t
//        BAD:  {[row] mavg[20;row`price]} each t
//   5. MINIMIZE COPIES - update in place, avoid intermediate allocations
//   6. PREFER ADVERBS - each, peach, over (\), scan (/) are faster than loops
//
// PERFORMANCE OPTIMIZATION PATTERNS:
//   1. CUMSUM TRICKS - Many rolling calculations can use msum/sums differences:
//        - Rolling product: prds[x] % n xprev prds[x]  (not prd each window)
//        - WMA: Use msum[n;t*x] and msum[n;x] with algebra
//        - OLS slope: (n*Σxy - Σx*Σy)/(n*Σx² - (Σx)²) via msum
//   2. USE BUILT-IN EMA - q's ema[alpha;x] is C-optimized, ~100x faster than scan
//        - BAD:  {[a;p;c](a*c)+(1-a)*p}\[first x;alpha;x]  // scan-based, slow
//        - GOOD: ema[alpha;x]                              // built-in, fast
//   3. VECTORIZE CONDITIONALS - avoid scan with $[cond;...] per element:
//        - BAD:  {[p;r;v]$[v<0;p*1+r;p]}\[1f;ret;deltas vol]  // scan, slow
//        - GOOD: prds 1 + ret * (deltas vol)<0                // vectorized, fast
//   4. SLIDING WINDOWS - swin:{[n;x] flip (1-til n) xprev\: x}
//        - Faster than scan-based {1_x,y}\[n#0n;x] for building windows
//        - But still O(n*w), unavoidable for median/percentile/argmax
//   5. mwindow DOESN'T EXIST in q4 - don't assume it's available
//
// INHERENTLY SLOW OPERATIONS (O(n*w), cannot easily vectorize):
//   - Rolling median (rmed): needs sorted window, no cumsum trick
//   - Rolling percentile (rpctl): needs sorted window
//   - Rolling skewness/kurtosis: needs full window for moments
//   - Aroon (argmax/argmin position): needs to track position in window
//   - These require specialized streaming data structures to optimize further
//
// BUILT-IN FUNCTIONS (use these directly, no wrapper needed):
//   mavg[n;x]    - simple moving average
//   msum[n;x]    - rolling sum
//   mmax[n;x]    - rolling max
//   mmin[n;x]    - rolling min
//   mdev[n;x]    - rolling standard deviation
//   mcount[n;x]  - rolling count of non-nulls
//   ema[alpha;x] - exponential moving average (alpha-based, C-optimized)
//   deltas x     - first difference
//   ratios x     - successive ratios
//   sums x       - cumulative sum
//   prds x       - cumulative product
//   cor[x;y]     - correlation
//   cov[x;y]     - covariance
//   var x        - variance
//   dev x        - standard deviation
//   NOTE: mcov/mcor don't exist in q4 - use .kdbtools.rcov/.kdbtools.rcor instead
//
// GOTCHAS:
//   1. do[n;body] does NOT provide loop index - use while with explicit counter
//   2. Multi-line functions in .q files need proper semicolons, no trailing comments
//   3. msum/mavg handle nulls at boundaries - your vectorized version should match
//   4. Test vectorized rewrites against reference implementation before replacing
//
// TABLE USAGE PATTERNS:
//   All functions operate on vectors (table columns). Use update/select syntax:
//
//   1. SINGLE SYMBOL - direct column update:
//        t:update sma5:.kdbtools.sma[5;close] from t
//        t:update ret:.kdbtools.ret[close], rsi:.kdbtools.rsi[14;close] from t
//
//   2. MULTI-SYMBOL - use "by sym" for per-symbol rolling windows:
//        t:update sma5:.kdbtools.sma[5;close] by sym from t
//        t:update slope:.kdbtools.slope[20;close] by sym from t
//
//   3. DICT-RETURNING FUNCTIONS - extract to table, then update:
//        bb:.kdbtools.bband[20;2;t`close];
//        t:update bb_mid:bb`mid, bb_upper:bb`upper, bb_pctb:bb`pctb from t
//
//        macd:.kdbtools.macd[12;26;9;t`close];
//        t:update macd:macd`macd, signal:macd`signal from t
//
//   4. MULTIPLE INDICATORS - chain or combine:
//        t:update sma5:.kdbtools.sma[5;close], sma20:.kdbtools.sma[20;close],
//                 rsi14:.kdbtools.rsi[14;close], slope10:.kdbtools.slope[10;close] from t
//
//   FUNCTIONS RETURNING DICTS (need unpacking):
//     bband    -> `mid`upper`lower`width`pctb
//     macd     -> `macd`signal`hist
//     aroon    -> `up`down`osc
//     adx      -> `pdi`ndi`dx`adx
//     keltner  -> `mid`upper`lower
//     vortex   -> `vip`vim`diff
//     elderRay -> `bull`bear
//
// =============================================================================
// CONFIGURATION & INITIALIZATION
// =============================================================================

\d .kdbtools

// Version info
version:"0.7.0"  // Added digital filters (supersmoother, zlema, butterworth, kalman, savgol, laguerre, impulseResponse)

// Try to load QML if available (for advanced linear algebra)
qmlLoaded:@[{system"l qml.q";1b};::;0b]

// =============================================================================
// ROLLING WINDOW PRIMITIVES
// =============================================================================

// Sliding window helper - returns list of n-element windows
// Uses flip of lagged series (faster than scan for large n)
swin:{[n;x] flip (1-til n) xprev\: x}

// Rolling apply - apply any function over rolling window
roll:{[f;n;x] f each swin[n;x]}

// Simple moving average (alias for mavg - clearer name)
sma:{[n;x] mavg[n;x]}

// Exponential moving average (span-based, like pandas ewm(span=n))
expma:{[n;x] ema[2%n+1;x]}

// Exponential moving standard deviation (span-based)
// Uses EMA of squared deviations from EMA mean
emadev:{[n;x] mu:expma[n;x]; sqrt expma[n;(x-mu) xexp 2]}

// Exponential moving standard deviation with explicit alpha
emadeva:{[alpha;x] mu:ema[alpha;x]; sqrt ema[alpha;(x-mu) xexp 2]}

// Weighted moving average (linear weights) - vectorized using cumsum
// Returns null for first n-1 values (need full window)
wma:{[n;x] t:til count x; Sx:msum[n;x]; Stx:msum[n;t*x]; r:(Stx - (t-n)*Sx) % n*(n+1)%2; @[r;til n-1;:;0n]}

// Rolling product - vectorized using cumulative products
rprod:{[n;x] cp:prds x; cp % n xprev cp}

// Rolling variance (sample)
rvar:{[n;x] mdev[n;x] xexp 2}

// Rolling median - using swin
rmed:{[n;x] med each swin[n;x]}

// Rolling percentile (p between 0-1) - using swin
rpctl:{[n;p;x] {[p;w] (asc w)@`long$p*-1+count w}[p] each swin[n;x]}

// Rolling skewness - using swin
rskew:{[n;x] {[w] m:avg w; s:dev w; $[s=0;0n;avg ((w-m)%s) xexp 3]} each swin[n;x]}

// Rolling kurtosis (excess) - using swin
rkurt:{[n;x] {[w] m:avg w; s:dev w; $[s=0;0n;-3+avg ((w-m)%s) xexp 4]} each swin[n;x]}

// =============================================================================
// DIFFERENCING & RETURNS
// =============================================================================

// N-period difference
diffn:{[n;x] x - n xprev x}

// Simple returns
ret:{(deltas x) % prev x}

// N-period returns
retn:{[n;x] (x - n xprev x) % n xprev x}

// Log returns
logret:{log x % prev x}

// N-period log returns
logretn:{[n;x] log x % n xprev x}

// Cumulative returns from simple returns
cumret:{prds 1+x}

// =============================================================================
// STATISTICAL FUNCTIONS
// =============================================================================

// Z-score (standardization over entire series)
zscore:{(x - avg x) % dev x}

// Rolling z-score
zscorer:{[n;x] (x - sma[n;x]) % mdev[n;x]}

// Cross-sectional z-score
zscorex:{[x] (x - avg x) % dev x}

// Winsorize at percentiles (lo, hi between 0-1)
winsorize:{[lo;hi;x] p:(asc x)@`long$(lo;hi)*-1+count x; p[0]|x&p[1]}

// Rank (cross-sectional, 1 to n)
rnk:{1+iasc iasc x}

// Percentile rank (0-1)
pctrank:{(rnk x) % count x}

// Beta (regression slope): cov(x,y)/var(x)
beta:{[x;y] cov[x;y] % var x}

// Rolling covariance: E[xy] - E[x]*E[y]
rcov:{[n;x;y] mavg[n;x*y] - mavg[n;x]*mavg[n;y]}

// Rolling correlation: cov(x,y) / (std(x)*std(y))
rcor:{[n;x;y] rcov[n;x;y] % mdev[n;x]*mdev[n;y]}

// Rolling beta: cov(x,y)/var(x)
betar:{[n;x;y] rcov[n;x;y] % rvar[n;x]}

// OLS regression coefficients [intercept; slope]
ols:{[x;y] b:cov[x;y] % var x; a:(avg y) - b * avg x; (a;b)}

// OLS residuals
olsresid:{[x;y] c:ols[x;y]; y - c[0] + c[1] * x}

// R-squared
rsq:{[x;y] r:cor[x;y]; r*r}

// Autocorrelation at lag n
autocorr:{[n;x] cor[x;n xprev x]}

// Partial autocorrelation (approximation)
pacf:{[n;x] r:x; do[n-1;r:olsresid[prev r;r]]; cor[n xprev x;r]}

// =============================================================================
// RISK & VOLATILITY METRICS
// =============================================================================

// Maximum drawdown (returns negative value representing max peak-to-trough decline)
maxdd:{[x] maxs[x] - x; min (x - maxs x) % maxs x}

// Drawdown series (current drawdown at each point)
drawdown:{[x] (x - maxs x) % maxs x}

// Calmar ratio: annualized return / abs(max drawdown)
// @param x - price series
// @param periods - periods per year (252 for daily, 12 for monthly)
calmar:{[periods;x] r:retn[count[x]-1;x]; annRet:r[count[r]-1] * periods % count x; annRet % abs maxdd x}

// Sortino ratio: excess return / downside deviation
// @param x - returns series
// @param rf - risk-free rate (per period)
// @param periods - periods per year for annualization
sortino:{[periods;rf;x] excess:x - rf; downside:x where x < rf; dd:sqrt avg downside * downside; (avg[excess] * sqrt periods) % dd}

// Information ratio: excess return / tracking error
// @param x - strategy returns
// @param bench - benchmark returns
ir:{[x;bench] excess:x - bench; (avg excess) % dev excess}

// Value at Risk (historical, returns loss as positive number)
// @param p - confidence level (e.g., 0.95 for 95% VaR)
// @param x - returns series
varHist:{[p;x] neg (asc x) @ `long$(1-p) * count x}

// Conditional VaR / Expected Shortfall (average of losses beyond VaR)
// @param p - confidence level
// @param x - returns series
cvar:{[p;x] cutoff:(asc x) @ `long$(1-p) * count x; neg avg x where x <= cutoff}

// Rolling VaR
varHistr:{[n;p;x] {[p;w] neg (asc w) @ `long$(1-p) * count w}[p] each swin[n;x]}

// Parkinson volatility (high-low based, more efficient than close-close)
// @param n - lookback window
// @param h - high prices
// @param l - low prices
parkinson:{[n;h;l] k:1 % 4 * log 2; sqrt k * mavg[n;(log[h]-log l) xexp 2]}

// Garman-Klass volatility (OHLC based, even more efficient)
// @param n - lookback window
// @param o,h,l,c - OHLC prices
garmanKlass:{[n;o;h;l;c]
    hl2:(log[h]-log l) xexp 2;
    co2:(log[c]-log o) xexp 2;
    sqrt mavg[n;(0.5*hl2) - (2*log[2]-1)*co2]}

// Yang-Zhang volatility (handles overnight gaps, most accurate)
// @param n - lookback window
// @param o,h,l,c - OHLC prices
yangZhang:{[n;o;h;l;c]
    // Overnight volatility (close-to-open)
    co:log o % prev c;
    vo:mdev[n;co] xexp 2;
    // Open-to-close volatility
    oc:log c % o;
    vc:mdev[n;oc] xexp 2;
    // Rogers-Satchell volatility
    rs:(log[h]-log o)*(log[h]-log c) + (log[l]-log o)*(log[l]-log c);
    vrs:mavg[n;rs];
    // Yang-Zhang combination (k = 0.34 is optimal for efficiency)
    k:0.34 % 1 + (n+1) % n - 1;
    sqrt vo + (k*vc) + (1-k)*vrs}

// Exponentially weighted volatility (same as emadev but clearer name for vol context)
ewmvol:{[n;x] emadev[n;x]}

// GARCH(1,1) volatility forecast
// Estimates: sigma^2_t = omega + alpha*r^2_{t-1} + beta*sigma^2_{t-1}
// Uses variance targeting: omega = (1-alpha-beta)*long_run_var
// @param x - returns series
// @param alpha - reaction to recent return (typically 0.05-0.15)
// @param beta - persistence (typically 0.8-0.95)
garch11:{[alpha;beta;x]
    omega:(1-alpha-beta) * var x;
    r2:x * x;
    // Recursive: sig2[t] = omega + alpha*r2[t-1] + beta*sig2[t-1]
    // Use scan with initial variance
    init:var x;
    {[o;a;b;s;r] o + (a*r) + b*s}\[init;omega;alpha;beta;prev r2]}

// Rolling realized volatility (annualized)
// @param n - lookback window
// @param periods - periods per year
realizedVol:{[n;periods;x] r:ret x; sqrt periods * mavg[n;r*r]}

// Volatility ratio (current vol / historical vol) for regime detection
volRatio:{[fast;slow;x] r:ret x; mdev[fast;r] % mdev[slow;r]}

// =============================================================================
// MOMENTUM / TREND INDICATORS
// =============================================================================

// Rate of change (percentage)
roc:{[n;x] 100 * (x - n xprev x) % n xprev x}

// Momentum (absolute)
mom:{[n;x] x - n xprev x}

// MACD - returns dict with `macd`signal`hist
macd:{[fast;slow;sig;x] m:expma[fast;x] - expma[slow;x]; s:expma[sig;m]; `macd`signal`hist!(m;s;m-s)}

// RSI (Relative Strength Index) 0-100
rsi:{[n;x] d:deltas x; u:d*d>0; dd:neg d*d<0; rs:expma[n;u] % expma[n;dd]; 100 - 100 % 1+rs}

// Stochastic %K (0-100)
stochk:{[n;x] ll:mmin[n;x]; hh:mmax[n;x]; 100 * (x - ll) % hh - ll}

// Stochastic %D
stochd:{[n;d;x] sma[d;stochk[n;x]]}

// Williams %R (-100 to 0)
willr:{[n;x] hh:mmax[n;x]; ll:mmin[n;x]; -100 * (hh - x) % hh - ll}

// CCI (Commodity Channel Index)
cci:{[n;x] m:sma[n;x]; md:sma[n;abs x-m]; (x - m) % 0.015 * md}

// ADX components - returns dict with `pdi`ndi`dx`adx
adx:{[n;h;l;c] ph:prev h; pl:prev l; pdm:0|h-ph; ndm:0|pl-l; pdm:pdm*pdm>ndm; ndm:ndm*ndm>pdm; tr:mmax[1;h]-mmin[1;l]; atr:ema[1%n;tr]; pdi:100*ema[1%n;pdm]%atr; ndi:100*ema[1%n;ndm]%atr; dx:100*abs[pdi-ndi]%pdi+ndi; adxv:ema[1%n;dx]; `pdi`ndi`dx`adx!(pdi;ndi;dx;adxv)}

// ATR (Average True Range)
atr:{[n;h;l;c] pc:prev c; tr:(h-l)|(abs h-pc)|abs l-pc; ema[1%n;tr]}

// Bollinger Band position (-1 to 1 at bands)
bbpos:{[n;k;x] m:sma[n;x]; s:mdev[n;x]; (x - m) % k * s}

// Donchian breakout signal (1 upper, -1 lower, 0 none)
donchian:{[n;x] hh:mmax[n;x]; ll:mmin[n;x]; (x = hh) - x = ll}

// Linear regression slope - vectorized using cumsum formulas
// slope = (n*Σiy - Σi*Σy) / (n*Σi² - (Σi)²) where i=0..n-1
// Returns null for first n-1 positions (partial windows)
slope:{[n;x]
    si:n*(n-1)%2;                    // Σi for i=0..n-1 (constant)
    si2:n*(n-1)*((2*n)-1)%6;         // Σi² (constant) - NOTE: (2*n)-1 not 2*n-1
    denom:(n*si2)-(si*si);           // denominator (constant)
    sy:msum[n;x];                    // rolling Σy
    // For Σiy: at position t, window is [t-n+1..t], relative idx i = j-(t-n+1) for j in window
    // Σiy = Σ(j-(t-n+1))*x[j] = Σj*x[j] - (t-n+1)*Σx[j] = msum[t*x] - (t-n+1)*msum[x]
    t:til count x;
    offset:(t-n)+1;                  // NOTE: parens critical! (t-n)+1 not t-n+1
    siy:(msum[n;t*x]) - offset*sy;   // rolling Σiy
    r:((n*siy) - si*sy) % denom;
    @[r;til n-1;:;0n]}               // null for partial windows

// Trend strength (abs slope of regression)
trendstr:{[n;x] abs slope[n;x]}

// =============================================================================
// MEAN REVERSION SIGNALS
// =============================================================================

// Bollinger Bands - returns dict with `mid`upper`lower`width`pctb
bband:{[n;k;x] m:sma[n;x]; s:mdev[n;x]; u:m+k*s; l:m-k*s; w:(u-l)%m; pb:(x-l)%(u-l); `mid`upper`lower`width`pctb!(m;u;l;w;pb)}

// Keltner Channels - returns dict
keltner:{[n;k;h;l;c] m:expma[n;c]; a:atr[n;h;l;c]; `mid`upper`lower!(m;m+k*a;m-k*a)}

// Mean reversion z-score signal
mrzscore:{[fast;slow;x] f:sma[fast;x]; s:sma[slow;x]; sd:mdev[slow;x]; (f - s) % sd}

// Half-life of mean reversion (from AR1)
halflife:{[x] y:1_ x; x0:-1_ x; b:beta[x0;y-x0]; neg log[2] % log[1+b]}

// Hurst exponent (R/S method, simplified)
hurst:{[x] n:count x; r:sums x - avg x; rs:(max r) - min r; s:dev x; log[rs%s] % log n}

// Pairs spread (residual)
spread:{[x;y] olsresid[x;y]}

// Z-scored spread
spreadz:{[n;x;y] zscorer[n;spread[x;y]]}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Fill forward nulls
ffill:{fills x}

// Fill backward
bfill:{reverse fills reverse x}

// Shift series (positive=lag, negative=lead)
shift:{[n;x] $[n>=0;n xprev x;reverse neg[n] xprev reverse x]}

// Lag alias
lag:{[n;x] n xprev x}

// Lead (look ahead) - xnext doesn't exist in q4, use reverse trick
lead:{[n;x] reverse n xprev reverse x}

// Clip to range
clip:{[lo;hi;x] lo|x&hi}

// Sign (-1, 0, 1)
sgn:{(x>0)-x<0}

// Normalize to 0-1
normalize:{(x - min x) % (max x) - min x}

// Rolling normalize
normalizer:{[n;x] (x - mmin[n;x]) % mmax[n;x] - mmin[n;x]}

// Cross above
crossabove:{[x;y] (x>y) and prev[x]<=prev y}

// Cross below
crossbelow:{[x;y] (x<y) and prev[x]>=prev y}

// Exponential decay of signal
decay:{[hl;x] a:1 - exp neg log[2] % hl; ema[a;x]}

// Signal to position with threshold
sig2pos:{[thresh;x] s:sgn x; {[th;p;s] $[abs[s]>th;s;p]}\[0;thresh;s]}

// =============================================================================
// COMPOSITE SIGNAL HELPERS
// =============================================================================

// Equal weight combination
combineEq:{avg x}

// Weighted combination
combineWt:{[w;sigs] sum w*sigs}

// Rank-based combination
combineRank:{[sigs] ranks:pctrank each flip sigs; avg each flip ranks}

// Decay-weighted combination
combineDecay:{[hl;sigs] n:count sigs; w:exp neg (til n) * log[2] % hl; w:w % sum w; sum w * sigs}

// =============================================================================
// VOLUME-BASED INDICATORS
// =============================================================================

// VWAP (Volume Weighted Average Price) - cumulative
vwap:{[p;v] (sums p*v) % sums v}

// Rolling VWAP
vwapr:{[n;p;v] (msum[n;p*v]) % msum[n;v]}

// OBV (On Balance Volume)
obv:{[c;v] sums v * sgn deltas c}

// Accumulation/Distribution Line
adl:{[h;l;c;v] mfm:((c-l)-(h-c)) % (h-l); sums mfm * v}

// Chaikin Money Flow
cmf:{[n;h;l;c;v] mfm:((c-l)-(h-c)) % (h-l); mfv:mfm*v; msum[n;mfv] % msum[n;v]}

// Money Flow Index (volume-weighted RSI)
mfi:{[n;tp;v] mf:tp*v; pmf:mf*tp>prev tp; nmf:mf*tp<prev tp; mr:msum[n;pmf]%msum[n;nmf]; 100-100%1+mr}

// Force Index
forceIdx:{[n;c;v] expma[n;v * deltas c]}

// Ease of Movement
eom:{[n;h;l;v] dm:(h+l)%2 - (prev[h]+prev l)%2; br:v%1e6%(h-l); expma[n;dm%br]}

// Price Volume Trend
pvt:{[c;v] sums v * ret c}

// Negative Volume Index - vectorized: apply return only when volume decreases
nvi:{[c;v] r:ret c; vdown:(deltas v)<0; prds 1 + r * vdown}

// Positive Volume Index - vectorized: apply return only when volume increases
pvi:{[c;v] r:ret c; vup:(deltas v)>0; prds 1 + r * vup}

// =============================================================================
// ADVANCED TECHNICAL INDICATORS
// =============================================================================

// Aroon Oscillator - using swin (faster than scan)
aroon:{[n;h;l] up:100*(n - {x?max x} each swin[n;h])%n; dn:100*(n - {x?min x} each swin[n;l])%n; `up`down`osc!(up;dn;up-dn)}

// TRIX (triple smoothed EMA rate of change)
trix:{[n;x] e1:expma[n;x]; e2:expma[n;e1]; e3:expma[n;e2]; 10000 * ret e3}

// Ultimate Oscillator
ultOsc:{[s1;s2;s3;h;l;c] pc:prev c; bp:c-l&pc; tr:(h|pc)-(l&pc); a1:msum[s1;bp]%msum[s1;tr]; a2:msum[s2;bp]%msum[s2;tr]; a3:msum[s3;bp]%msum[s3;tr]; 100*(4*a1+2*a2+a3)%7}

// Elder Ray (Bull and Bear Power)
elderRay:{[n;h;l;c] e:expma[n;c]; `bull`bear!(h-e;l-e)}

// Mass Index
massIdx:{[n;h;l] r:h-l; e1:expma[9;r]; e2:expma[9;e1]; msum[n;e1%e2]}

// Vortex Indicator
vortex:{[n;h;l;c] vmp:abs h - prev l; vmm:abs l - prev h; tr:(h|prev c)-(l&prev c); vip:msum[n;vmp]%msum[n;tr]; vim:msum[n;vmm]%msum[n;tr]; `vip`vim`diff!(vip;vim;vip-vim)}

// Coppock Curve
coppock:{[s;l;m;x] wma[m;roc[s;x]+roc[l;x]]}

// Chande Momentum Oscillator
cmo:{[n;x] d:deltas x; su:msum[n;d*d>0]; sd:msum[n;abs d*d<0]; 100*(su-sd)%su+sd}

// Detrended Price Oscillator
dpo:{[n;x] x - (`int$1+n%2) xprev sma[n;x]}

// Know Sure Thing
kst:{[x] r1:sma[10;roc[10;x]]; r2:sma[10;roc[15;x]]; r3:sma[10;roc[20;x]]; r4:sma[15;roc[30;x]]; r1+2*r2+3*r3+4*r4}

// Percentage Price Oscillator
ppo:{[fast;slow;x] 100*(expma[fast;x]-expma[slow;x])%expma[slow;x]}

// Schaff Trend Cycle
stc:{[fast;slow;k;x] m:expma[fast;x]-expma[slow;x]; pf:stochk[k;m]; pff:expma[3;pf]; pfs:stochk[k;pff]; expma[3;pfs]}

// Fisher Transform
fisher:{[n;x] v:2*normalizer[n;x]-1; v:clip[-0.999;0.999;v]; 0.5*log (1+v)%1-v}

// Ichimoku Cloud - returns dict with all 5 lines
// @param tenkan - conversion line period (default 9)
// @param kijun - base line period (default 26)
// @param senkou - leading span B period (default 52)
// @param h,l,c - high, low, close prices
ichimoku:{[tenkan;kijun;senkou;h;l;c]
    // Tenkan-sen (Conversion): (highest high + lowest low) / 2 over tenkan periods
    tenkanSen:(mmax[tenkan;h] + mmin[tenkan;l]) % 2;
    // Kijun-sen (Base): (highest high + lowest low) / 2 over kijun periods
    kijunSen:(mmax[kijun;h] + mmin[kijun;l]) % 2;
    // Senkou Span A (Leading A): (tenkan + kijun) / 2, shifted forward kijun periods
    senkouA:lead[kijun;(tenkanSen + kijunSen) % 2];
    // Senkou Span B (Leading B): (highest + lowest) / 2 over senkou periods, shifted forward
    senkouB:lead[kijun;(mmax[senkou;h] + mmin[senkou;l]) % 2];
    // Chikou Span (Lagging): close shifted back kijun periods
    chikouSpan:kijun xprev c;
    `tenkan`kijun`senkouA`senkouB`chikou!(tenkanSen;kijunSen;senkouA;senkouB;chikouSpan)}

// SuperTrend indicator
// @param n - ATR period
// @param mult - ATR multiplier
// @param h,l,c - OHLC prices
supertrend:{[n;mult;h;l;c]
    atrv:atr[n;h;l;c];
    hl2:(h+l) % 2;
    basicUpper:hl2 + mult * atrv;
    basicLower:hl2 - mult * atrv;
    pc:prev c;
    // Combine into table of inputs for scan
    inputs:flip `bu`bl`pc!(basicUpper;basicLower;pc);
    init:`upper`lower`st`dir!(first basicUpper;first basicLower;first basicLower;1);
    step:{[s;inp] bu:inp`bu; bl:inp`bl; cv:inp`pc; pu:s`upper; pl:s`lower; pd:s`dir; fu:$[cv<=pu;bu&pu;bu]; fl:$[cv>=pl;bl|pl;bl]; dir:$[cv>fu;1;cv<fl;-1;pd]; st:$[dir=1;fl;fu]; `upper`lower`st`dir!(fu;fl;st;dir)};
    states:step\[init;inputs];
    `st`dir`upper`lower!(states`st;states`dir;states`upper;states`lower)}

// Parabolic SAR
// @param af_step - acceleration factor step (default 0.02)
// @param af_max - max acceleration factor (default 0.2)
// @param h,l - high, low prices
psar:{[af_step;af_max;h;l]
    init:`sar`ep`af`up!(first l;first h;af_step;1b);
    inputs:flip `h`l!(h;l);
    step:{[afs;afm;s;inp] hv:inp`h; lv:inp`l; psar0:s`sar; pep:s`ep; paf:s`af; pup:s`up; upSar:psar0 + paf * pep - psar0; upRev:lv < upSar; upNewEp:hv|pep; upNewAf:$[hv>pep;afm&paf+afs;paf]; upResult:$[upRev;`sar`ep`af`up!(pep;lv;afs;0b);`sar`ep`af`up!(upSar;upNewEp;upNewAf;1b)]; dnSar:psar0 - paf * psar0 - pep; dnRev:hv > dnSar; dnNewEp:lv&pep; dnNewAf:$[lv<pep;afm&paf+afs;paf]; dnResult:$[dnRev;`sar`ep`af`up!(pep;hv;afs;1b);`sar`ep`af`up!(dnSar;dnNewEp;dnNewAf;0b)]; $[pup;upResult;dnResult]};
    states:step[af_step;af_max]\[init;inputs];
    `sar`direction!(states`sar;states`up)}

// Pivot Points (Floor/Standard method)
// @param h,l,c - previous period high, low, close (scalars or vectors)
// Returns dict with pivot, support (s1,s2,s3) and resistance (r1,r2,r3) levels
pivotPoints:{[h;l;c]
    pp:(h+l+c) % 3;
    r1:(2*pp) - l;
    s1:(2*pp) - h;
    r2:pp + h - l;
    s2:pp - (h - l);
    r3:h + 2 * pp - l;
    s3:l - 2 * (h - pp);
    `pp`r1`r2`r3`s1`s2`s3!(pp;r1;r2;r3;s1;s2;s3)}

// Woodie Pivot Points (alternative method)
pivotWoodie:{[h;l;c]
    pp:(h+l+2*c) % 4;
    r1:(2*pp) - l;
    s1:(2*pp) - h;
    r2:pp + h - l;
    s2:pp - (h - l);
    `pp`r1`r2`s1`s2!(pp;r1;r2;s1;s2)}

// Camarilla Pivot Points
pivotCamarilla:{[h;l;c]
    r:h-l;
    r4:c + r * 1.1 % 2;
    r3:c + r * 1.1 % 4;
    r2:c + r * 1.1 % 6;
    r1:c + r * 1.1 % 12;
    s1:c - r * 1.1 % 12;
    s2:c - r * 1.1 % 6;
    s3:c - r * 1.1 % 4;
    s4:c - r * 1.1 % 2;
    `r4`r3`r2`r1`s1`s2`s3`s4!(r4;r3;r2;r1;s1;s2;s3;s4)}

// Heikin-Ashi OHLC transformation
// Smoothed candlesticks that filter noise
// @param o,h,l,c - original OHLC prices
heikinAshi:{[o;h;l;c]
    // HA Close: average of OHLC
    hac:(o+h+l+c) % 4;
    // HA Open: midpoint of previous HA candle body (recursive)
    // Start with first open, then scan
    hao:{[po;o;c] (po+c) % 2}\[first o;prev o;prev hac];
    // Overwrite first value
    hao[0]:first o;
    // HA High: max of high, HA open, HA close
    hah:h|hao|hac;
    // HA Low: min of low, HA open, HA close
    hal:l&hao&hac;
    `open`high`low`close!(hao;hah;hal;hac)}

// Chandelier Exit (trailing stop based on ATR)
// @param n - ATR period
// @param mult - ATR multiplier (typically 3)
// @param h,l,c - OHLC prices
chandelier:{[n;mult;h;l;c]
    atrv:atr[n;h;l;c];
    highestHigh:mmax[n;h];
    lowestLow:mmin[n;l];
    exitLong:highestHigh - mult * atrv;
    exitShort:lowestLow + mult * atrv;
    `long`short!(exitLong;exitShort)}

// Keltner Channel Width (squeeze detection)
keltnerWidth:{[n;k;h;l;c] kc:keltner[n;k;h;l;c]; (kc[`upper] - kc[`lower]) % kc[`mid]}

// Bollinger Bandwidth (squeeze detection)
bbWidth:{[n;k;x] bb:bband[n;k;x]; bb`width}

// Squeeze indicator: Bollinger inside Keltner = low volatility squeeze
squeeze:{[bb_n;bb_k;kc_n;kc_k;h;l;c]
    bb:bband[bb_n;bb_k;c];
    kc:keltner[kc_n;kc_k;h;l;c];
    // Squeeze on when BB is inside KC (use bracket notation to avoid parsing issues)
    sqzOn:(bb[`lower] > kc[`lower]) & (bb[`upper] < kc[`upper]);
    // Momentum: linear regression slope of midline deviation
    mom:slope[20;c - sma[20;c]];
    `squeeze`momentum!(sqzOn;mom)}

// =============================================================================
// DIGITAL FILTERS & SIGNAL PROCESSING
// =============================================================================

// SuperSmoother (John Ehlers) - 2-pole filter with less lag than EMA
// Attempt to get the best of both worlds: smoothing without excessive lag
// @param n - cutoff period
// @param x - price series
supersmoother:{[n;x]
    pi:acos -1;
    a1:exp neg sqrt[2] * pi % n;
    b1:2 * a1 * cos sqrt[2] * pi % n;
    c2:b1;
    c3:neg a1 * a1;
    c1:(1 - c2) - c3;  // explicit parens for RTL eval
    // IIR filter: y = c1*(x + prev_x)/2 + c2*y1 + c3*y2
    n2:count x;
    y:n2#0f;
    y[0]:x 0;
    y[1]:(c1 * (x[1] + x[0]) % 2) + (c2 + c3) * y[0];
    i:2;
    while[i < n2;
        y[i]:(c1 * (x[i] + x[i-1]) % 2) + (c2 * y[i-1]) + c3 * y[i-2];
        i+:1];
    y}

// Ehlers 2-pole High-Pass Filter
// Removes low-frequency trend to isolate cycles
// @param n - cutoff period
// @param x - price series
ehlers2poleHP:{[n;x]
    pi:acos -1;
    angle:0.707 * 2 * pi % n;
    alpha1:((cos[angle] + sin[angle]) - 1) % cos angle;
    // HP = (1 - alpha/2)^2 * (x - 2*x1 + x2) + 2*(1-alpha)*hp1 - (1-alpha)^2*hp2
    c0:(1 - alpha1 % 2) xexp 2;
    c1:2 * 1 - alpha1;
    c2:neg (1 - alpha1) xexp 2;
    n2:count x;
    hp:n2#0f;
    i:2;
    while[i < n2;
        // Second difference: x[i] - 2*x[i-1] + x[i-2]
        diff:(x[i] - (2 * x[i-1])) + x[i-2];
        hp[i]:(c0 * diff) + (c1 * hp[i-1]) + c2 * hp[i-2];
        i+:1];
    hp}

// Decycler (Ehlers) - Removes cycles, leaves pure trend
// Complement of high-pass filter
// @param n - cutoff period (removes cycles shorter than n)
// @param x - price series
decycler:{[n;x] x - ehlers2poleHP[n;x]}

// Zero-Lag EMA - EMA with lag correction
// Uses double-EMA technique to reduce lag
// @param n - period
// @param x - price series
zlema:{[n;x]
    e1:expma[n;x];
    e2:expma[n;e1];
    (2 * e1) - e2}

// Zero-Lag EMA (alternative) - Instantaneous trendline
// Uses momentum correction
// @param n - period
// @param x - price series
zlemaAlt:{[n;x]
    gain:(n - 1) % 2;
    momentum:deltas x;  // x - prev x, with first value = first x
    momentum[0]:0f;     // set first momentum to 0
    expma[n;x + momentum * gain]}

// Butterworth Low-Pass Filter (2-pole)
// Maximally flat frequency response in passband
// @param n - cutoff period
// @param x - price series
butterworth2:{[n;x]
    pi:acos -1;
    a:exp neg sqrt[2] * pi % n;
    b:2 * a * cos sqrt[2] * pi % n;
    c2:b;
    c3:neg a * a;
    c1:((1 - c2) - c3) % 4;  // explicit parens for RTL
    // y = c1*(x + 2*x1 + x2) + c2*y1 + c3*y2
    n2:count x;
    y:n2#0f;
    y[0]:x 0;
    y[1]:x 1;
    i:2;
    while[i < n2;
        y[i]:(c1 * (x[i] + (2 * x[i-1]) + x[i-2])) + (c2 * y[i-1]) + c3 * y[i-2];
        i+:1];
    y}

// Butterworth Low-Pass Filter (3-pole) - cascade 2-pole + 1-pole
// Implemented as cascade of butterworth2 and single-pole EMA for stability
// @param n - cutoff period
// @param x - price series
butterworth3:{[n;x]
    // Use cascade: apply 2-pole then 1-pole for 3-pole total
    bw2:butterworth2[n;x];
    // Simple 1-pole as final stage (EMA equivalent)
    alpha:2.0 % n + 1;
    ema[alpha;bw2]}

// Gaussian Filter - weights follow Gaussian (normal) distribution
// Smooth with bell-curve weights, good for noise reduction
// @param n - window size (odd recommended)
// @param sigma - standard deviation (controls smoothness)
// @param x - price series
gaussian:{[n;sigma;x]
    center:(n - 1) % 2;
    idx:til n;
    w:exp neg 0.5 * ((idx - center) % sigma) xexp 2;
    w:w % sum w;  // normalize
    // Pad input so we have enough for centered windows
    pad:n - 1;
    xpad:((pad#first x),x,(pad#last x));
    // Apply convolution, drop initial partial windows
    wins:swin[n;xpad];
    res:{[w;win] sum w * win}[w] each wins;
    // swin has n-1 partial windows at start; take count[x] from index n-1
    (count[x])#((n - 1) _ res)}

// Hodrick-Prescott Filter - Separate trend from cycle
// Minimizes: sum((y-trend)^2) + lambda*sum(d2_trend^2)
// @param lambda - smoothing parameter (1600 quarterly, 129600 monthly, 6.25 daily)
// @param x - price series
// Note: Requires matrix operations, uses approximation for speed
hpfilter:{[lambda;x]
    if[not qmlLoaded;'"HP filter requires QML for full version"];
    n:count x;
    // Build second-difference matrix K (n-2 x n)
    // K[i] = [0..0, 1, -2, 1, 0..0]
    K:{[n;i] @[n#0f;i + til 3;:;1 -2 1f]}[n] each til n - 2;
    // Trend = (I + lambda*K'K)^-1 * y
    I:eye n;
    KtK:mm[mT K;K];
    A:I + lambda * KtK;
    Ainv:minv A;
    first each mm[Ainv;enlist each x]}

// HP Filter (fast approximation using double EMA)
// Good for real-time use, approximates HP trend
// @param n - period (higher = smoother)
// @param x - price series
hpfast:{[n;x]
    alpha:2.0 % n + 1;
    // Double EMA approximation of HP trend
    e1:ema[alpha;x];
    e2:ema[alpha;e1];
    (2 * e1) - e2}

// Kalman Filter - Adaptive smoothing for noisy data
// Simple 1D Kalman for price smoothing
// @param q - process noise variance (higher = more responsive)
// @param r - measurement noise variance (higher = more smoothing)
// @param x - price series
kalman:{[q;r;x]
    n:count x;
    est:n#0f;
    p:1.0;
    est[0]:x 0;
    i:1;
    while[i < n;
        // Predict: state unchanged, covariance grows
        p_pred:p + q;
        // Update with measurement
        k:p_pred % p_pred + r;
        est[i]:est[i-1] + k * x[i] - est[i-1];
        p:(1 - k) * p_pred;
        i+:1];
    est}

// Kalman Filter with auto-tuned parameters
// Estimates noise from data
// @param n - lookback for noise estimation
// @param x - price series
kalmanAuto:{[n;x]
    // Estimate measurement noise from rolling std of residuals
    trend:sma[n;x];
    resid:x - trend;
    r:var resid;
    // Process noise: fraction of measurement noise
    q:r % 10;
    kalman[q;r;x]}

// Savitzky-Golay Filter - Polynomial smoothing
// Fits polynomial to each window, evaluates at center
// Preserves peaks better than simple MA
// @param n - window size (odd number)
// @param order - polynomial order (2 or 3 typical)
// @param x - price series
savgol:{[n;order;x]
    if[0 = n mod 2;'"Window size must be odd"];
    if[order >= n;'"Order must be less than window size"];
    // Build Vandermonde matrix for window
    center:(n - 1) % 2;
    t:(til n) - center;  // centered indices: -2,-1,0,1,2 for n=5
    // X = [1, t, t^2, t^3, ...] up to order
    X:flip {[t;p] t xexp p}[t] each til order + 1;
    // Compute smoothing weights: w = (X'X)^-1 * X' evaluated at center (row 0)
    XtX:mm[mT X;X];
    XtXinv:minv XtX;
    // Weight for center point: first row of XtXinv * X'
    w:first mm[XtXinv;mT X];
    // Pad input so we have enough for centered windows
    pad:n - 1;
    xpad:((pad#first x),x,(pad#last x));
    // Apply convolution, drop initial partial windows
    wins:swin[n;xpad];
    res:{[w;win] sum w * win}[w] each wins;
    // swin has n-1 partial windows at start; take count[x] from index n-1
    (count[x])#((n - 1) _ res)}

// Laguerre Filter (Ehlers) - Variable-length smoothing
// Uses Laguerre polynomials for adaptive filtering
// @param gamma - damping factor 0 < gamma < 1 (higher = more smoothing)
// @param x - price series
laguerre:{[gamma;x]
    n:count x;
    x0:first x;
    filt:n#x0;
    // Use scalar state variables for L0-L3 previous values (memory efficient)
    L0p:x0; L1p:x0; L2p:x0; L3p:x0;
    g1:1 - gamma;
    ng:neg gamma;
    i:1;
    while[i < n;
        L0:(g1 * x i) + gamma * L0p;
        L1:(ng * L0) + L0p + gamma * L1p;
        L2:(ng * L1) + L1p + gamma * L2p;
        L3:(ng * L2) + L2p + gamma * L3p;
        filt[i]:(L0 + (2 * L1) + (2 * L2) + L3) % 6;
        L0p:L0; L1p:L1; L2p:L2; L3p:L3;
        i+:1];
    filt}

// =============================================================================
// FILTER ANALYSIS
// =============================================================================

// Impulse Response - Shows filter weights/behavior
// Feed a unit impulse through filter to see its response
// @param filter - filter function (takes x, returns filtered x)
// @param n - number of samples to compute
impulseResponse:{[filter;n]
    // Create impulse: [1, 0, 0, 0, ...]
    impulse:@[n#0f;0;:;1f];
    filter impulse}

// Step Response - Shows how filter responds to step change
// @param filter - filter function
// @param n - number of samples
stepResponse:{[filter;n]
    // Create step: [0, 0, ..., 1, 1, 1, ...]
    step:@[n#1f;til n div 2;:;0f];
    filter step}

// Filter Lag - Estimate effective lag of filter
// Uses step response: measures when output reaches 50% of step height
// @param filter - filter function
// @param n - samples for step response
filterLag:{[filter;n]
    // Step from 0 to 100 at index 0
    step:n#100f;
    step[0]:0f;
    resp:filter step;
    // Measure when response reaches 50 (half of 100)
    first where resp >= 50}

// Compare Filters - Visual comparison of multiple filters
// @param filters - list of filter functions
// @param names - list of filter names
// @param x - price series
compareFilters:{[filters;names;x]
    results:filters @\: x;
    ([] name:names) ,' flip (`$string each til count x)!flip results}

// =============================================================================
// QML MATRIX OPERATIONS (requires QML)
// =============================================================================

// Identity matrix
eye:{[n] @[n#0f;;:;1f] each til n}

// Design matrix with intercept (prepend column of 1s)
addIntercept:{[X] (count[X]#1f),'X}

// Matrix multiply (wrapper for QML or fallback)
mm:{[A;B] $[qmlLoaded;.qml.mm[A;B];A mmu B]}

// Matrix inverse (wrapper for QML or fallback)
minv:{[A] $[qmlLoaded;.qml.minv[A];inv A]}

// Matrix transpose
mT:{flip x}

// Covariance matrix from data matrix (rows=obs, cols=features)
covmat:{[X] n:count X; mu:avg each flip X; Xc:X -\: mu; mm[mT Xc;Xc] % n-1}

// Correlation matrix
cormat:{[X] C:covmat X; s:sqrt each C[;til count C]; C % s *\: s}

// =============================================================================
// REGULARIZED REGRESSION (for high-noise systems)
// =============================================================================

// Ridge Regression: (X'X + lambda*I)^-1 * X'y
// @param X - design matrix (n x p), should include intercept column if desired
// @param y - target vector (n x 1)
// @param lambda - regularization parameter (L2 penalty)
// @return - coefficient vector
ridge:{[X;y;lambda]
    p:count first X;
    XtX:mm[mT X;X];
    reg:lambda * eye p;
    Xty:enlist each sum each (mT X) *\: y;  // X'y as column vector
    first each mm[minv XtX + reg;Xty]}

// Ridge with intercept (auto-adds intercept, centers data)
ridgeInt:{[X;y;lambda]
    Xa:addIntercept X;
    ridge[Xa;y;lambda]}

// Ridge with cross-validation to select lambda
ridgeCV:{[X;y;lambdas;nfolds]
    n:count y;
    foldIdx:n#til nfolds;  // assign each obs to a fold
    cvOneLambda:{[X;y;fi;nfolds;lam]
        mses:{[X;y;fi;lam;fold]
            testIdx:where fi=fold;
            trainIdx:where not fi=fold;
            Xtr:X trainIdx; ytr:y trainIdx;
            Xte:X testIdx; yte:y testIdx;
            b:ridge[Xtr;ytr;lam];
            pred:sum each Xte *\: b;
            avg (yte - pred) xexp 2}[X;y;fi;lam] each til nfolds;
        avg mses};
    scores:cvOneLambda[X;y;foldIdx;nfolds] each lambdas;
    bestIdx:scores?min scores;
    bestLam:lambdas bestIdx;
    `lambda`coeffs`cv_scores!(bestLam;ridge[X;y;bestLam];scores)}

// Squared Euclidean distance matrix
distSq:{[X1;X2] {[r;X2] sum each (X2 -\: r) xexp 2}[;X2] each X1}

// Kernel function: RBF (Gaussian)
kRBF:{[gamma;X1;X2] exp neg gamma * distSq[X1;X2]}

// Kernel function: Polynomial
kPoly:{[d;c;X1;X2] (c + {[r;X2] sum each X2 *\: r}[;X2] each X1) xexp d}

// Kernel function: Linear
kLinear:{[X1;X2] {[r;X2] sum each X2 *\: r}[;X2] each X1}

// Kernel Ridge Regression
// @param X - training features (n x p)
// @param y - target (n x 1)
// @param lambda - regularization
// @param kfn - kernel function, e.g., kRBF[0.1]
// @return - dict with alpha coefficients and training data for prediction
krr:{[X;y;lambda;kfn]
    K:kfn[X;X];
    n:count y;
    alpha:first each mm[minv K + lambda * eye n;enlist each y];
    `alpha`X`kfn!(alpha;X;kfn)}

// Kernel Ridge prediction
krrPredict:{[model;Xnew]
    Ktest:model[`kfn][model`X;Xnew];
    sum each Ktest *\: model`alpha}

// Elastic Net via coordinate descent
// @param X - design matrix (n x p)
// @param y - target (n x 1)
// @param lambda - regularization strength
// @param alpha - mixing (0=ridge, 1=lasso)
// @param maxIter - max iterations
// @param tol - convergence tolerance
elasticNet:{[X;y;lambda;alpha;maxIter;tol]
    p:count first X;
    l1:lambda*alpha; l2:lambda*(1-alpha);
    b:p#0f; bOld:p#1f;
    niter:0;
    while[(niter<maxIter) and tol < max abs b - bOld;
        bOld:b;
        // One full cycle through all variables
        j:0; while[j<p;
            Xj:X[;j];
            r:y - sum each X *\: b;
            rj:r + b[j]*Xj;
            rho:sum Xj * rj;
            zj:sum Xj * Xj;
            b[j]:softThresh[rho;l1] % zj + l2;
            j+:1];
        niter+:1];
    b}

// Soft thresholding operator (for Lasso)
softThresh:{[x;t] (x - t) * x > t | (x + t) * x < neg t}

// Lasso regression (L1 penalty via coordinate descent)
lasso:{[X;y;lambda;maxIter;tol] elasticNet[X;y;lambda;1f;maxIter;tol]}

// =============================================================================
// ROBUST REGRESSION METHODS
// =============================================================================

// Huber loss weight function
huberWeight:{[delta;r] $[abs[r]<=delta;1f;delta%abs r]}

// IRLS (Iteratively Reweighted Least Squares) for robust regression
// @param X - design matrix
// @param y - target
// @param wfn - weight function (e.g., huberWeight[1.345])
// @param maxIter - max iterations
// @param tol - tolerance
irls:{[X;y;wfn;maxIter;tol]
    b:ridge[X;y;0.001];
    do[maxIter;
        bOld:b;
        r:y - sum each X *\: b;  // residuals
        s:1.4826 * med abs r;
        w:wfn each r % s | 0.0001;
        Xw:X * w;
        b:ridge[Xw;y*w;0.0001];
        if[tol > max abs b - bOld; :b]];
    b}

// Huber regression (robust to outliers)
huber:{[X;y;delta] irls[X;y;huberWeight[delta];50;1e-6]}

// Tukey biweight function
tukeyWeight:{[c;r] ar:abs r; $[ar<=c;(1-(r%c) xexp 2) xexp 2;0f]}

// Tukey's bisquare regression (very robust)
tukey:{[X;y;c] irls[X;y;tukeyWeight[c];50;1e-6]}

// Quantile regression via IRLS approximation
// @param X - design matrix
// @param y - target
// @param tau - quantile (0.5 = median regression)
// @param maxIter - max iterations
quantileReg:{[X;y;tau;maxIter]
    b:ridge[X;y;0.001];
    eps:1e-6;
    do[maxIter;
        bOld:b;
        r:y - sum each X *\: b;  // residuals
        w:{[tau;eps;r] $[r>0;tau;1-tau] % abs[r]|eps}[tau;eps] each r;
        Xw:X * w;
        b:ridge[Xw;y*w;0.0001];
        if[1e-6 > max abs b - bOld; :b]];
    b}

// Theil-Sen estimator (median of pairwise slopes) - for simple regression
theilSen:{[x;y]
    n:count x;
    pairs:(til n) cross til n;
    pairs:pairs where pairs[;0] < pairs[;1];
    slopes:{[x;y;p] (y[p 1]-y[p 0])%(x[p 1]-x[p 0])}[x;y] each pairs;
    b:med slopes;
    a:med y - b * x;
    (a;b)}

// =============================================================================
// PCA AND DIMENSIONALITY REDUCTION (requires QML)
// =============================================================================

// PCA via eigendecomposition of covariance matrix
// @param X - data matrix (n obs x p features)
// @param k - number of components to keep (0 = all)
// @return - dict with components, explained variance, loadings
pca:{[X;k]
    if[not qmlLoaded;'"PCA requires QML"];
    n:count X; p:count first X;
    mu:avg each flip X;
    Xc:X -\: mu;
    C:covmat Xc;
    ev:.qml.mev C;
    vals:ev 0; vecs:ev 1;
    ord:idesc vals;
    vals:vals ord; vecs:vecs[;ord];
    k:$[k=0;p;k];
    totVar:sum vals;
    expVar:vals % totVar;
    cumVar:sums expVar;
    loadings:vecs[;til k];
    scores:mm[Xc;loadings];
    `mean`loadings`eigenvalues`explained`cumulative`scores!(mu;loadings;k#vals;k#expVar;k#cumVar;scores)}

// Transform new data using PCA model
pcaTransform:{[model;Xnew]
    Xc:Xnew -\: model`mean;
    mm[Xc;model`loadings]}

// Inverse transform (reconstruct from reduced dimensions)
pcaInverse:{[model;scores]
    mm[scores;mT model`loadings] +\: model`mean}

// =============================================================================
// ROLLING REGRESSION METHODS
// =============================================================================

// Rolling OLS - uses swin to create index windows
// Returns null coefficients for first n-1 positions (partial windows)
olsr:{[n;X;y] idx:swin[n;til count y]; p:count first X; {[X;y;p;i] $[any null i;p#0n;ridge[X i;y i;1e-8]]}[X;y;p] each idx}

// Rolling Ridge
ridger:{[n;X;y;lambda] idx:swin[n;til count y]; p:count first X; {[X;y;lam;p;i] $[any null i;p#0n;ridge[X i;y i;lam]]}[X;y;lambda;p] each idx}

// Rolling Huber (robust)
huberr:{[n;X;y;delta] idx:swin[n;til count y]; p:count first X; {[X;y;d;p;i] $[any null i;p#0n;huber[X i;y i;d]]}[X;y;delta;p] each idx}

// Rolling Lasso (L1 regularized)
lassor:{[n;X;y;lambda;maxIter;tol] idx:swin[n;til count y]; p:count first X; {[X;y;lam;mi;t;p;i] $[any null i;p#0n;lasso[X i;y i;lam;mi;t]]}[X;y;lambda;maxIter;tol;p] each idx}

// Rolling Elastic Net (L1+L2 regularized)
elasticNetr:{[n;X;y;lambda;alpha;maxIter;tol] idx:swin[n;til count y]; p:count first X; {[X;y;lam;a;mi;t;p;i] $[any null i;p#0n;elasticNet[X i;y i;lam;a;mi;t]]}[X;y;lambda;alpha;maxIter;tol;p] each idx}

// Rolling Tukey (bisquare robust)
tukeyr:{[n;X;y;c] idx:swin[n;til count y]; p:count first X; {[X;y;c;p;i] $[any null i;p#0n;tukey[X i;y i;c]]}[X;y;c;p] each idx}

// Rolling Quantile Regression
quantileRegr:{[n;X;y;tau;maxIter] idx:swin[n;til count y]; p:count first X; {[X;y;tau;mi;p;i] $[any null i;p#0n;quantileReg[X i;y i;tau;mi]]}[X;y;tau;maxIter;p] each idx}

// Rolling Theil-Sen (robust median slope estimator)
theilSenr:{[n;x;y] idx:swin[n;til count y]; {[x;y;i] $[any null i;2#0n;theilSen[x i;y i]]}[x;y] each idx}

// Rolling prediction using rolling coefficients
rollingPredict:{[Xall;coeffs] {sum x*y}'[Xall;coeffs]}

// Recursive least squares (online updating)
// @param X - new observation features (vector)
// @param y - new observation target (scalar)
// @param state - dict with `b (coeffs), `P (inverse covariance), `lambda (forgetting)
// @return - updated state
rlsUpdate:{[X;y;state]
    b:state`b; P:state`P; lam:state`lambda;
    Px:sum each P *\: X;  // P @ x
    k:Px % lam + sum X * Px;  // gain vector
    e:y - sum X * b;  // prediction error
    bNew:b + e * k;
    PNew:(P - k */: Px) % lam;
    `b`P`lambda!(bNew;PNew;lam)}

// Initialize RLS state
rlsInit:{[p;lambda]
    `b`P`lambda!(p#0f;1000*eye p;lambda)}

// =============================================================================
// FACTOR MODEL UTILITIES
// =============================================================================

// Orthogonalize (Gram-Schmidt) - fixed loop variables
orthogonalize:{[X]
    n:count X; p:count first X;
    Q:();
    i:0; while[i<p;
        v:X[;i];
        j:0; while[j<count Q; v:v - (sum v * Q[j]) * Q[j]; j+:1];
        Q:Q,enlist v % sqrt sum v * v;
        i+:1];
    flip Q}

// Residualize Y against X (remove X's influence)
residualize:{[X;Y]
    b:ridge[X;Y;0.001];
    Y - sum each X *\: b}

// Factor exposure (regression betas to factors)
factorExposure:{[returns;factors]
    ridge[addIntercept factors;returns;0]}

// Factor mimicking portfolio weights
factorMimick:{[returns;chars]
    C:covmat returns;
    Cinv:minv C;
    fc:avg each flip chars;
    charDev:chars -\: fc;
    mm[Cinv;flip charDev] % sum each mm[flip charDev;Cinv] *' flip charDev}

\d .

// =============================================================================
// TESTS
// =============================================================================

.test.kdbtools:{[]
    x:100+sums 100?1f;
    y:100+sums 100?1f;
    v:1000+100?500f;  // volume
    results:()!();

    // Basic rolling primitives
    results[`sma]:100=count .kdbtools.sma[10;x];
    results[`expma]:100=count .kdbtools.expma[10;x];
    results[`emadev]:100=count .kdbtools.emadev[10;x];
    results[`rmed]:100=count .kdbtools.rmed[10;x];
    results[`ret]:100=count .kdbtools.ret x;
    results[`logret]:100=count .kdbtools.logret x;
    results[`zscore]:abs[avg .kdbtools.zscore x]<1e-10;
    results[`cor]:(cor[x;x]) within 0.9999 1.0001;
    results[`beta]:abs[.kdbtools.beta[x;x]-1]<1e-10;
    results[`ols]:2=count .kdbtools.ols[x;y];
    results[`roc]:100=count .kdbtools.roc[10;x];
    results[`rsi]:100=count .kdbtools.rsi[14;x];
    results[`macd]:3=count .kdbtools.macd[12;26;9;x];
    results[`bband]:5=count .kdbtools.bband[20;2;x];
    results[`zscorer]:100=count .kdbtools.zscorer[20;x];
    results[`ffill]:0=sum null .kdbtools.ffill @[x;10 20 30;:;0n];
    results[`clip]:all .kdbtools.clip[50;150;x] within (50;150);
    results[`crossabove]:100=count .kdbtools.crossabove[x;y];

    // Volume indicators
    results[`vwap]:100=count .kdbtools.vwap[x;v];
    results[`obv]:100=count .kdbtools.obv[x;v];

    // Advanced indicators
    results[`aroon]:3=count .kdbtools.aroon[14;x;y];
    results[`trix]:100=count .kdbtools.trix[15;x];
    results[`fisher]:100=count .kdbtools.fisher[10;x];

    // Matrix operations
    results[`eye]:3=count .kdbtools.eye 3;
    results[`covmat]:3=count .kdbtools.covmat flip (x;y;x+y);

    // Ridge regression
    X:flip (100#1f;x;y);
    z:0.5*x + 0.3*y + 10 + 100?1f;
    results[`ridge]:3=count .kdbtools.ridge[X;z;0.1];

    // Kernel Ridge (RBF)
    Xk:flip (x;y);
    model:.kdbtools.krr[Xk;z;1.0;.kdbtools.kRBF[0.01]];
    results[`krr]:3=count model;
    results[`krrPredict]:100=count .kdbtools.krrPredict[model;Xk];

    // Robust regression
    results[`huber]:3=count .kdbtools.huber[X;z;1.345];
    results[`theilSen]:2=count .kdbtools.theilSen[x;z];

    // Quantile regression
    results[`quantileReg]:3=count .kdbtools.quantileReg[X;z;0.5;20];

    // PCA (if QML available)
    results[`pca]:$[.kdbtools.qmlLoaded;6=count .kdbtools.pca[flip(x;y;x+y);2];1b];

    // Risk & Volatility metrics
    results[`maxdd]:(type .kdbtools.maxdd x) in -9 -8h;  // returns float
    results[`drawdown]:100=count .kdbtools.drawdown x;
    results[`sortino]:(type .kdbtools.sortino[252;0;.kdbtools.ret x]) in -9 -8h;
    results[`ir]:(type .kdbtools.ir[.kdbtools.ret x;.kdbtools.ret y]) in -9 -8h;
    results[`varHist]:(type .kdbtools.varHist[0.95;.kdbtools.ret x]) in -9 -8h;
    results[`cvar]:(type .kdbtools.cvar[0.95;.kdbtools.ret x]) in -9 -8h;
    results[`parkinson]:100=count .kdbtools.parkinson[20;x*1.01;x*0.99];
    results[`garmanKlass]:100=count .kdbtools.garmanKlass[20;x;x*1.01;x*0.99;x];
    results[`yangZhang]:100=count .kdbtools.yangZhang[20;x;x*1.01;x*0.99;x];
    results[`garch11]:100=count .kdbtools.garch11[0.1;0.85;.kdbtools.ret x];
    results[`realizedVol]:100=count .kdbtools.realizedVol[20;252;x];

    // New technical indicators
    h:x*1.01; l:x*0.99; c:x; o:prev x;
    results[`ichimoku]:5=count .kdbtools.ichimoku[9;26;52;h;l;c];
    results[`supertrend]:4=count .kdbtools.supertrend[10;3;h;l;c];
    results[`psar]:2=count .kdbtools.psar[0.02;0.2;h;l];
    results[`pivotPoints]:7=count .kdbtools.pivotPoints[last h;last l;last c];
    results[`pivotWoodie]:5=count .kdbtools.pivotWoodie[last h;last l;last c];
    results[`pivotCamarilla]:8=count .kdbtools.pivotCamarilla[last h;last l;last c];
    results[`heikinAshi]:4=count .kdbtools.heikinAshi[o;h;l;c];
    results[`chandelier]:2=count .kdbtools.chandelier[22;3;h;l;c];
    results[`squeeze]:2=count .kdbtools.squeeze[20;2;20;1.5;h;l;c];

    passed:sum results;
    total:count results;
    -1 "KDB Tools tests: ",string[passed],"/",string[total]," passed";
    $[passed=total;-1 "All tests PASSED";[-1 "FAILED: ",", " sv string where not results;'"Tests failed"]];
    results}

// =============================================================================
// EXAMPLES
// =============================================================================

.example.kdbtools:{[]
    -1 "=== KDB Tools Examples ===\n";
    n:252;
    prices:100*prds 1+0.0002+(n?0.02)-0.01;
    -1 "Sample price series (first 10):";
    show 10#prices;

    -1 "\n1. Rolling Statistics";
    -1 "20-day SMA (last 5): ",-3!5#reverse .kdbtools.sma[20;prices];
    -1 "20-day EMA Std (last 5): ",-3!5#reverse .kdbtools.emadev[20;prices];

    -1 "\n2. Returns";
    rets:.kdbtools.ret prices;
    -1 "Mean daily return: ",string avg rets;
    -1 "Std daily return: ",string dev rets;
    -1 "Annualized Sharpe: ",string (252*avg rets) % sqrt[252]*dev rets;

    -1 "\n3. Momentum";
    rsiv:.kdbtools.rsi[14;prices];
    -1 "RSI(14): ",string last rsiv;
    macdv:.kdbtools.macd[12;26;9;prices];
    -1 "MACD: ",-3!last each macdv;

    -1 "\n4. Mean Reversion";
    bb:.kdbtools.bband[20;2;prices];
    -1 "Bollinger %B: ",string last bb`pctb;
    -1 "Z-score(20): ",string last .kdbtools.zscorer[20;prices];

    -1 "\n5. Regression";
    -1 "20-day slope: ",string last .kdbtools.slope[20;prices];

    -1 "\n=== End Examples ==="}

// Load message
-1 "Loaded .kdbtools library v",.kdbtools.version;
-1 "QML available: ",string .kdbtools.qmlLoaded;
-1 "Run .test.kdbtools[] to verify";
-1 "Run .example.kdbtools[] for examples";
