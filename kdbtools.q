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
// Detect if already loaded (e.g., user loaded qml/src/qml.q before kdbtools)
qmlLoaded:$[not () ~ key `.qml; 1b; @[{system"l qml.q";1b};::;0b]]

// =============================================================================
// ROLLING WINDOW PRIMITIVES
// =============================================================================

// Sliding window helper - returns list of n-element windows
// Each window contains only PAST data (causal, no look-ahead)
// Window at position i contains: x[i], x[i-1], ..., x[i-n+1]
// First n-1 positions have nulls (incomplete history)
swin:{[n;x] flip (til n) xprev\: x}

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
rsi:{[n;x] d:x - prev x; u:0f^d*d>0; dd:0f^neg d*d<0; rs:expma[n;u] % expma[n;dd]; 100 - 100 % 1+rs}

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

// Rolling Hurst exponent
rHurst:{[n;x] hurst each mwin[n;x]}

// Rolling half-life
rHalflife:{[n;x] halflife each mwin[n;x]}

// Pairs spread (residual)
spread:{[x;y] olsresid[x;y]}

// Z-scored spread
spreadz:{[n;x;y] zscorer[n;spread[x;y]]}

// -----------------------------------------------------------------------------
// ZERO-CROSSING ANALYSIS
// -----------------------------------------------------------------------------

// Count of zero crossings (demeaned series)
zeroCrossings:{[x]
    centered:x - avg x;
    sum 0 < abs deltas signum centered}

// Zero-crossing rate (normalized by length)
zeroCrossRate:{[x]
    (zeroCrossings x) % count x}

// Zero-crossing ratio vs random walk expectation
// >1 = mean reverting, <1 = trending
zeroCrossRatio:{[x]
    n:count x;
    actual:zeroCrossings x;
    expected:0.6366 * sqrt n;
    actual % expected}

// Rolling zero-crossing ratio
rZeroCrossRatio:{[n;x]
    zeroCrossRatio each mwin[n;x]}

// -----------------------------------------------------------------------------
// VARIANCE RATIO TESTS
// -----------------------------------------------------------------------------

// Variance ratio: Var(k-period returns) / (k * Var(1-period returns))
// VR < 1 indicates mean reversion, VR > 1 indicates momentum
varianceRatio:{[k;x]
    r1:1 _ deltas x;
    rk:(k _ x) - (neg k) _ x;
    (var rk) % k * var r1}

// Rolling variance ratio
rVarianceRatio:{[n;k;x]
    {[k;w] varianceRatio[k;w]}[k] each mwin[n;x]}

// Variance ratio profile across multiple horizons
vrProfile:{[x;maxK]
    {[x;k] varianceRatio[k;x]}[x] each 2 + til maxK - 1}

// -----------------------------------------------------------------------------
// ORNSTEIN-UHLENBECK PARAMETERS
// -----------------------------------------------------------------------------

// Estimate OU process parameters: dx = θ(μ - x)dt + σdW
// Returns dict with theta, mu, sigma, halfLife
ouParams:{[x]
    xLag:1 _ prev x;
    dx:1 _ deltas x;
    n:count dx;
    sumX:sum xLag;
    sumY:sum dx;
    sumXY:sum xLag * dx;
    sumX2:sum xLag * xLag;
    denom:n * sumX2 - sumX * sumX;
    b:$[0 = denom; 0n; (n * sumXY - sumX * sumY) % denom];
    a:$[0 = denom; 0n; (sumY - b * sumX) % n];
    theta:neg b;
    mu:$[theta = 0; 0n; a % theta];
    sigma:dev dx - a + b * xLag;
    hl:$[theta <= 0; 0w; log[2] % theta];
    `theta`mu`sigma`halfLife!(theta;mu;sigma;hl)}

// Rolling OU half-life
rOUHalfLife:{[n;x]
    {ouParams[x]`halfLife} each mwin[n;x]}

// Rolling OU theta (mean reversion speed)
rOUTheta:{[n;x]
    {ouParams[x]`theta} each mwin[n;x]}

// -----------------------------------------------------------------------------
// AUTOCORRELATION STRUCTURE
// -----------------------------------------------------------------------------

// Autocorrelation at lag k
autocorr:{[k;x]
    xm:x - avg x;
    (sum (k _ xm) * (neg k) _ xm) % sum xm * xm}

// Autocorrelation profile up to maxLag
acfProfile:{[maxLag;x]
    autocorr[;x] each 1 + til maxLag}

// Sum of autocorrelations (negative = mean reverting)
acfSum:{[maxLag;x]
    sum autocorr[;x] each 1 + til maxLag}

// First-order autocorrelation (quick MR indicator)
// Negative = mean reverting, Positive = trending
rho1:{[x] autocorr[1;x]}

// Rolling first-order autocorrelation
rRho1:{[n;x]
    rho1 each mwin[n;x]}

// -----------------------------------------------------------------------------
// REVERSION STRENGTH INDICATORS
// -----------------------------------------------------------------------------

// Reversion beta: regress future returns on distance from MA
// Beta < 0 means gap predicts reversal (mean reversion)
reversionBeta:{[n;horizon;x]
    ma:mavg[n;x];
    gap:x - ma;
    futRet:(horizon _ x) % (neg[horizon] _ x) - 1;
    gap:neg[horizon] _ gap;
    $[0 = var gap; 0n; cov[gap;futRet] % var gap]}

// Rolling reversion beta
rReversionBeta:{[win;n;horizon;x]
    {[n;h;w] reversionBeta[n;h;w]}[n;horizon] each mwin[win;x]}

// -----------------------------------------------------------------------------
// CROSSING RATE INDICATORS
// -----------------------------------------------------------------------------

// MA crossing rate (crosses per window)
maCrossRate:{[n;window;x]
    ma:mavg[n;x];
    aboveMA:x > ma;
    crossings:0 < abs deltas aboveMA;
    msum[window;crossings]}

// Band touch rate (touches of Bollinger bands per window)
bandTouchRate:{[n;k;window;x]
    bb:bband[n;k;x];
    touchUpper:x >= bb`upper;
    touchLower:x <= bb`lower;
    touches:(0 < abs deltas touchUpper) or 0 < abs deltas touchLower;
    msum[window;touches]}

// Mean crossing frequency
meanCrossFreq:{[window;x]
    mu:mavg[window;x];
    crosses:0 < abs deltas x > mu;
    msum[window;crosses]}

// -----------------------------------------------------------------------------
// RANGE AND EFFICIENCY MEASURES
// -----------------------------------------------------------------------------

// Range efficiency: net move vs total path traveled
// Low efficiency = mean reverting (lots of back and forth)
rangeEfficiency:{[n;x]
    netMove:abs x - (n-1) xprev x;
    totalPath:msum[n;abs deltas x];
    netMove % totalPath}

// Fractal efficiency / dimension proxy
fractalEff:{[n;x]
    netMove:abs x - (n-1) xprev x;
    totalPath:msum[n;abs deltas x];
    log[netMove % totalPath] % log n}

// Price range ratio: current range vs historical
rangeRatio:{[n;window;h;l]
    currentRange:mmax[n;h] - mmin[n;l];
    avgRange:mavg[window;currentRange];
    currentRange % avgRange}

// -----------------------------------------------------------------------------
// COINTEGRATION / SPREAD INDICATORS
// -----------------------------------------------------------------------------

// Spread z-score with explicit beta
spreadZscore:{[n;beta;x;y]
    sprd:y - beta * x;
    (sprd - mavg[n;sprd]) % mdev[n;sprd]}

// Rolling hedge ratio (beta) via rolling regression
rollingHedgeRatio:{[n;x;y]
    {[xw;yw] $[0 = var xw; 0n; cov[xw;yw] % var xw]}'[mwin[n;x]; mwin[n;y]]}

// Dynamic spread z-score (uses rolling hedge ratio)
dynamicSpreadZ:{[n;x;y]
    beta:rollingHedgeRatio[n;x;y];
    sprd:y - beta * x;
    (sprd - mavg[n;sprd]) % mdev[n;sprd]}

// Error correction model alpha (speed of adjustment)
// Alpha < 0 indicates mean reversion, magnitude = speed
ecmAlpha:{[n;x;y]
    beta:$[0 = var x; 0n; cov[x;y] % var x];
    sprd:y - beta * x;
    sprdLag:prev sprd;
    deltaY:deltas y;
    {[sw;dw] $[0 = var sw; 0n; cov[sw;dw] % var sw]}'[mwin[n;1 _ sprdLag]; mwin[n;1 _ deltaY]]}

// Cointegration residual stationarity (rolling ADF-like measure)
// More negative = more stationary/mean-reverting
residualMR:{[n;x;y]
    beta:rollingHedgeRatio[n;x;y];
    sprd:y - beta * x;
    // Simplified ADF: regression coefficient of lagged level
    {[w] sprdLag:prev w; dSprd:deltas w; sprdLag:1 _ sprdLag; dSprd:1 _ dSprd;
     $[0 = var sprdLag; 0n; cov[sprdLag;dSprd] % var sprdLag]} each mwin[n;sprd]}

// -----------------------------------------------------------------------------
// ENTROPY-BASED MEASURES
// -----------------------------------------------------------------------------

// Direction change rate
// High rate = mean reverting (frequent reversals)
dirChangeRate:{[n;x]
    dirs:signum deltas x;
    changes:0 < abs deltas dirs;
    mavg[n;changes]}

// Direction entropy (predictability of direction changes)
// High entropy = random, Low = predictable pattern
dirEntropy:{[n;x]
    dirs:signum deltas x;
    changes:0 < abs deltas dirs;
    p:mavg[n;changes];
    // Avoid log(0)
    p:0.0001 | p & 0.9999;
    neg (p * log p) + (1-p) * log 1-p}

// Reversal intensity: magnitude of direction changes
reversalIntensity:{[n;x]
    rets:deltas[x] % prev x;
    dirs:signum rets;
    reversals:0 > dirs * prev dirs;
    // Average return magnitude on reversal days
    revMag:abs rets * reversals;
    mavg[n;revMag] % mavg[n;abs rets]}

// -----------------------------------------------------------------------------
// COMPOSITE MEAN-REVERSION SCORE
// -----------------------------------------------------------------------------

// Composite MR indicator (combines multiple signals)
// Positive = mean reverting, Negative = trending
mrScore:{[n;x]
    // Hurst score (0.5 - hurst, so < 0.5 gives positive)
    h:hurst x;
    hurstScore:2 * (0.5 - h);
    // Variance ratio score (1 - VR, so < 1 gives positive)
    vr:varianceRatio[5;x];
    vrScore:1 - vr;
    // Zero crossing score (ratio - 1, so > 1 gives positive)
    zcr:zeroCrossRatio x;
    zcrScore:zcr - 1;
    // Autocorrelation score (negative AC is positive for MR)
    ac:neg rho1 x;
    // Combine with equal weights, clip to [-1, 1]
    score:(hurstScore + vrScore + zcrScore + ac) % 4;
    -1 | score & 1}

// Rolling composite MR score
rMrScore:{[win;x]
    mrScore[win] each mwin[win;x]}

// Detailed MR diagnostics (returns dict with all components)
mrDiagnostics:{[x]
    h:hurst x;
    vr:varianceRatio[5;x];
    zcr:zeroCrossRatio x;
    ac1:rho1 x;
    ou:ouParams x;
    `hurst`varianceRatio`zeroCrossRatio`autocorr1`ouTheta`ouHalfLife`mrScore!(h;vr;zcr;ac1;ou`theta;ou`halfLife;mrScore[count x;x])}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Sliding window: returns list of n-length windows from x
mwin:{[n;x] x (til n) +\: til 1 + count[x] - n}

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
// .qml.mm requires both args to be 2D matrices; vectors must be wrapped
mm:{[A;B]
    if[not qmlLoaded; :A mmu B];
    // Detect vector args (type > 0h means typed list, i.e. vector not matrix)
    aVec:0h < type A;
    bVec:0h < type B;
    // If both vectors, fall back to mmu (dot product / outer product)
    if[aVec and bVec; :A mmu B];
    // Wrap vector→matrix for QML, then unwrap result
    A2:$[aVec; enlist A; A];
    B2:$[bVec; flip enlist B; B];
    r:@[.qml.mm[A2;]; B2; {[A;B;e] A mmu B}[A;B]];  // fallback to mmu on error
    // Unwrap: if B was vector, extract column; if A was vector, extract row
    r:$[bVec; first each r; r];
    r:$[aVec; first r; r];
    r}

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
olsr:{[n;X;y] idx:swin[n;til count y]; p:count first X; {[X;y;p;i] $[any null i;p#0n;ridge[X i;y i;1e-8]]}[X;y;p] peach idx}

// Rolling Ridge (original implementation - uses window slicing)
ridger:{[n;X;y;lambda] idx:swin[n;til count y]; p:count first X; {[X;y;lam;p;i] $[any null i;p#0n;ridge[X i;y i;lam]]}[X;y;lambda;p] each idx}

// Rolling Ridge (optimized using cumulative sums + mmu)
// ~2.5x faster than original ridger:
//   1. Cumulative sums avoid redundant X'X computation per window
//   2. Uses mmu (matrix-vector multiply) instead of mm with column matrices
//   3. Uses minv wrapper for QML acceleration when available
ridgerFast:{[n;X;y;lam]
    nObs:count y;
    p:count first X;
    // Outer product for each row: x_i' * x_i (p x p matrix)
    outerProds:{x */:\: x} each X;
    cumXtX:sums outerProds;
    // X'y cumsum: each row x_i multiplied by scalar y_i
    cumXty:sums X *' y;
    // Regularization
    reg:lam * eye p;
    // Beta computation for each window position
    computeBeta:{[cumXtX;cumXty;reg;p;n;start]
        if[null start; :p#0n];
        endIdx:start + n - 1;
        XtX:$[start = 0; cumXtX endIdx; cumXtX[endIdx] - cumXtX[start - 1]];
        Xty:$[start = 0; cumXty endIdx; cumXty[endIdx] - cumXty[start - 1]];
        (minv XtX + reg) mmu Xty  // minv uses QML when available
    }[cumXtX;cumXty;reg;p;n];
    // Generate start indices (null for warmup period)
    starts:((n-1)#0N),(til 1 + nObs - n);
    computeBeta each starts}

// Rolling Ridge with within-group parallelization
// Use when you have few groups but large data per group
// Parallelizes the beta computation across windows within each group
ridgerFastParallel:{[n;X;y;lam]
    nObs:count y;
    p:count first X;
    outerProds:{x */:\: x} each X;
    cumXtX:sums outerProds;
    cumXty:sums X *' y;
    reg:lam * eye p;
    computeBeta:{[cumXtX;cumXty;reg;p;n;start]
        if[null start; :p#0n];
        endIdx:start + n - 1;
        XtX:$[start = 0; cumXtX endIdx; cumXtX[endIdx] - cumXtX[start - 1]];
        Xty:$[start = 0; cumXty endIdx; cumXty[endIdx] - cumXty[start - 1]];
        (minv XtX + reg) mmu Xty
    }[cumXtX;cumXty;reg;p;n];
    starts:((n-1)#0N),(til 1 + nObs - n);
    computeBeta peach starts}  // peach for within-group parallelization

// Rolling Huber (robust)
huberr:{[n;X;y;delta] idx:swin[n;til count y]; p:count first X; {[X;y;d;p;i] $[any null i;p#0n;huber[X i;y i;d]]}[X;y;delta;p] peach idx}

// Rolling Lasso (L1 regularized)
lassor:{[n;X;y;lambda;maxIter;tol] idx:swin[n;til count y]; p:count first X; {[X;y;lam;mi;t;p;i] $[any null i;p#0n;lasso[X i;y i;lam;mi;t]]}[X;y;lambda;maxIter;tol;p] peach idx}

// Rolling Elastic Net (L1+L2 regularized)
elasticNetr:{[n;X;y;lambda;alpha;maxIter;tol] idx:swin[n;til count y]; p:count first X; {[X;y;lam;a;mi;t;p;i] $[any null i;p#0n;elasticNet[X i;y i;lam;a;mi;t]]}[X;y;lambda;alpha;maxIter;tol;p] peach idx}

// Rolling Tukey (bisquare robust)
tukeyr:{[n;X;y;c] idx:swin[n;til count y]; p:count first X; {[X;y;c;p;i] $[any null i;p#0n;tukey[X i;y i;c]]}[X;y;c;p] peach idx}

// Rolling Quantile Regression
quantileRegr:{[n;X;y;tau;maxIter] idx:swin[n;til count y]; p:count first X; {[X;y;tau;mi;p;i] $[any null i;p#0n;quantileReg[X i;y i;tau;mi]]}[X;y;tau;maxIter;p] peach idx}

// Rolling Theil-Sen (robust median slope estimator)
theilSenr:{[n;x;y] idx:swin[n;til count y]; {[x;y;i] $[any null i;2#0n;theilSen[x i;y i]]}[x;y] peach idx}

// Rolling prediction using rolling coefficients
rollingPredict:{[Xall;coeffs] {sum x*y}'[Xall;coeffs]}

// Rolling Ridge on table with grouping
// @param t - table with data
// @param bycol - column name to group by (e.g. `sym)
// @param xc - list of feature column names (e.g. `mom`val`size)
// @param ycol - target column name (e.g. `ret)
// @param window - rolling window size
// @param lam - ridge regularization parameter (lambda is reserved in q)
// @return - original table with b_<feature> columns and yhat (next-step prediction)
rollingRidgeHelper:{[xc;ycol;w;lam;g]
    X:flip g xc;
    y:g ycol;
    nf:count xc;
    betas:ridgerFast[w;X;y;lam];  // Sequential within group (parallel across groups)
    // Handle malformed betas: wrong length, contains nulls/infinities
    cleanBeta:{[nf;b]
        $[(count b)<>nf; nf#0n;
          any null b; nf#0n;
          any (b=0w) or b=-0w; nf#0n;  // infinities from singular matrix
          b]}[nf];
    betasCleaned:cleanBeta each betas;
    prevBetas:prev betasCleaned;
    // Note: prev returns empty list for first element, check count first
    pred:{[nf;x;b] $[(count b)<>nf; 0n; any null b; 0n; sum x*b]}[nf]'[X;prevBetas];
    bcols:(`$"b_",/:string xc)!flip betasCleaned;
    g,'flip bcols,'(enlist`yhat)!enlist pred}

// Helper for within-group parallelization (use when few groups, large data per group)
rollingRidgeHelperParallel:{[xc;ycol;w;lam;g]
    X:flip g xc;
    y:g ycol;
    nf:count xc;
    betas:ridgerFastParallel[w;X;y;lam];  // Parallel within group
    cleanBeta:{[nf;b]
        $[(count b)<>nf; nf#0n;
          any null b; nf#0n;
          any (b=0w) or b=-0w; nf#0n;
          b]}[nf];
    betasCleaned:cleanBeta each betas;
    prevBetas:prev betasCleaned;
    pred:{[nf;x;b] $[(count b)<>nf; 0n; any null b; 0n; sum x*b]}[nf]'[X;prevBetas];
    bcols:(`$"b_",/:string xc)!flip betasCleaned;
    g,'flip bcols,'(enlist`yhat)!enlist pred}

// Rolling ridge regression by group
// Uses peach for automatic parallelization when slave threads available
// Start q with -s N for parallel processing (e.g., q -s 4)
// Without slaves, peach behaves like each (no overhead)
// Parallelization strategy:
//   - Many groups (>=4): parallel across groups (default)
//   - Few groups (<4): parallel within each group
rollingRidgeTable:{[t;bycol;xc;ycol;window;lam]
    vals:distinct t bycol;
    groups:{[t;col;s] ?[t;enlist (=;col;enlist s);0b;()]}[t;bycol] each vals;
    nGroups:count groups;
    // Choose parallelization strategy based on number of groups
    $[nGroups >= 4;
        // Many groups: parallelize across groups
        raze rollingRidgeHelper[xc;ycol;window;lam] peach groups;
        // Few groups: parallelize within each group
        raze rollingRidgeHelperParallel[xc;ycol;window;lam] each groups]}

// Forward N-period simple return: sum(ret[t+1]..ret[t+N])
// @param N - number of periods forward
// @param x - simple return vector
fwdReturn:{[N;x] (neg[N] xprev sums x) - sums x}

// Rolling ridge regression predicting forward N-period simple return
// Features are NOT lagged — the rolling ridge fits on windows where the target
// (fwdReturn) is realized, and predicts using current features. For the last N
// rows where fwdReturn is null, betas are forward-filled from the last valid
// model and yhat is recomputed with current features.
// @param t      - table (must be sorted by bycol, time)
// @param bycol  - group column (e.g. `sym)
// @param retcol - return column (simple returns)
// @param xc     - list of feature columns
// @param N      - forward return horizon (periods)
// @param window - rolling regression window
// @param lam    - ridge regularization
// @return original table + fwdRetN + b_<feature> columns + yhat
rollingRidgeN:{[t;bycol;retcol;xc;N;window;lam]
    bc:$[11h=type bycol; bycol; enlist bycol];
    by:bc!bc;
    ycol:`fwdRetN;
    // Add forward N-period return as target (null for last N rows per group)
    t:![t; (); by; (enlist ycol)!enlist (`.kdbtools.fwdReturn;N;retcol)];
    // Run rolling ridge — features are current, target has trailing nulls
    // rollingRidgeTable fits on window, predicts at t using features[t] * betas[t-1]
    // Betas go null when window includes null-target rows
    t:rollingRidgeTable[t;bycol;xc;ycol;window;lam];
    // Forward-fill betas per group so the last valid model extends to present
    bcols:`$"b_",/:string xc;
    t:![t; (); by; bcols!{(fills;x)} each bcols];
    // Recompute yhat = sum(prev(filled_betas) * features) for all rows
    // prev ensures no lookahead: model at t uses betas fitted up to t-1
    terms:{(*;(prev;x);y)}'[bcols;xc];
    yexpr:terms 0;
    i:1; while[i < count terms; yexpr:(+;yexpr;terms i); i+:1];
    t:![t; (); by; (enlist `yhat)!enlist yexpr];
    t}

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
// MARKOV-SWITCHING REGRESSION (2-Regime Hamilton Filter)
// =============================================================================

// Precompute constant
sqrt2pi:sqrt 2 * acos -1f
isqrt2pi:1%sqrt2pi

// Gaussian probability density function
// @param x - observation (scalar)
// @param mu - mean (scalar)
// @param sigma - standard deviation (scalar, >0)
// @return - density value (scalar)
gaussPdf:{[x;mu;sigma] z:(x-mu)%sigma; (exp neg 0.5*z*z) % sigma*sqrt2pi}

// Hamilton filter step (for scan operator, K=2 regimes)
// @param state - `fp0`fp1`ll dict (filtered probs + cumulative log-likelihood)
// @param inp - `eta0`eta1`p00`p11 dict (regime likelihoods + transition probs)
// @return - updated state dict
hamiltonStep:{[state;inp]
    fp0:state`fp0; fp1:state`fp1;
    p00:inp`p00; p11:inp`p11;
    // Prediction: P(s_t=k | Y_{1:t-1})
    pp0:(p00*fp0)+(1-p11)*fp1;
    pp1:((1-p00)*fp0)+p11*fp1;
    // Update: P(s_t=k | Y_{1:t})
    j0:inp[`eta0]*pp0;
    j1:inp[`eta1]*pp1;
    fsum:(j0+j1)|1e-300;
    `fp0`fp1`ll!(j0%fsum; j1%fsum; state[`ll]+log fsum)}

// Hamilton filter (forward pass, 2 regimes, no backward smoother)
// @param y - target vector (T)
// @param X - feature matrix (T x p, list of p-vectors)
// @param b0 - regime 0 betas (p-vector)
// @param b1 - regime 1 betas (p-vector)
// @param s0 - regime 0 sigma (scalar)
// @param s1 - regime 1 sigma (scalar)
// @param p00 - P(stay in regime 0) (scalar)
// @param p11 - P(stay in regime 1) (scalar)
// @return - `fp0`fp1`ll (T-vectors of filtered probs + scalar log-likelihood)
hamiltonFilter:{[y;X;b0;b1;s0;s1;p00;p11]
    T:count y; Xt:flip X;
    // Vectorized regime likelihoods (mmu for speed)
    mu0:first (enlist b0) mmu Xt;
    mu1:first (enlist b1) mmu Xt;
    inv0:1%s0; inv1:1%s1;
    r0:(y-mu0)*inv0; r1:(y-mu1)*inv1;
    eta0:(exp neg 0.5*r0*r0)*inv0*isqrt2pi;
    eta1:(exp neg 0.5*r1*r1)*inv1*isqrt2pi;
    // While loop (avoids dict overhead of scan)
    ofp0:T#0f; ofp1:T#0f;
    p01:1-p00; p10:1-p11;
    fp0:0.5; fp1:0.5; ll:0f;
    i:0; while[i<T;
        pp0:(p00*fp0)+p10*fp1;
        pp1:(p01*fp0)+p11*fp1;
        e0:eta0 i; e1:eta1 i;
        $[((e0=0n) or e1=0n) or (e0=0f) and e1=0f;
            [fp0:pp0; fp1:pp1];  // null/missing obs: predict only, no update
            [j0:e0*pp0; j1:e1*pp1;
             fsum:(j0+j1)|1e-300;
             fp0:j0%fsum; fp1:j1%fsum;
             ll+:log fsum]];
        ofp0[i]:fp0; ofp1[i]:fp1;
        i+:1];
    `fp0`fp1`ll!(ofp0;ofp1;ll)}

// Initialize Markov-switching regression parameters
// Splits data by residual magnitude into low-vol (regime 0) and high-vol (regime 1)
// @param X - feature matrix (T x p)
// @param y - target vector (T)
// @param lam - ridge regularization
// @return - `b0`b1`s0`s1`p00`p11 parameter dict
msRegInit:{[X;y;lam]
    p:count first X;
    // Filter to valid (non-null) rows
    vld:where not (null y) | any each null each X;
    Xv:X vld; yv:y vld;
    bAll:ridge[Xv;yv;lam];
    resid:yv - first (enlist bAll) mmu flip Xv;
    absR:abs resid;
    medR:med absR;
    idx0:where absR<=medR;
    idx1:where absR>medR;
    // Guard: need enough obs per regime for regression
    if[(count[idx0]<2*p) or count[idx1]<2*p;
        s0:dev yv[idx0]; s1:dev yv[idx1];
        :`b0`b1`s0`s1`p00`p11!(bAll;bAll;s0|1e-6;s1|1e-6;0.95;0.95)];
    b0:ridge[Xv idx0;yv idx0;lam];
    b1:ridge[Xv idx1;yv idx1;lam];
    r0:yv[idx0] - first (enlist b0) mmu flip Xv idx0;
    r1:yv[idx1] - first (enlist b1) mmu flip Xv idx1;
    s0:sqrt avg r0*r0;
    s1:sqrt avg r1*r1;
    // Enforce convention: regime 0 = low vol
    if[s0>s1; tmp:b0; b0:b1; b1:tmp; tmp:s0; s0:s1; s1:tmp];
    `b0`b1`s0`s1`p00`p11!(b0;b1;s0|1e-6;s1|1e-6;0.95;0.95)}

// Markov-switching regression via EM (forward filter only, no backward pass)
// @param X - feature matrix (T x p)
// @param y - target vector (T)
// @param lam - ridge regularization
// @param maxIter - maximum EM iterations
// @param tol - log-likelihood convergence tolerance
// @param params0 - initial parameters (dict or 0N for auto-init)
// @return - `b0`b1`s0`s1`p00`p11`fp0`fp1`ll`niter`converged
msRegEM:{[X;y;lam;maxIter;tol;params0]
    T:count y; p:count first X; Xt:flip X;
    // Valid (non-null) row indices for M-step
    vld:where not (null y) | any each null each X;
    Xv:X vld; yv:y vld; Xvt:flip Xv;
    params:$[99h=type params0; params0; msRegInit[X;y;lam]];
    b0:params`b0; b1:params`b1;
    s0:params`s0; s1:params`s1;
    p00:params`p00; p11:params`p11;
    prevLL:neg 1e30; iter:0; converged:0b;
    // E-step first, then iterate M->E until converged
    filt:hamiltonFilter[y;X;b0;b1;s0;s1;p00;p11];
    fp0:filt`fp0; fp1:filt`fp1; ll:filt`ll;
    while[(iter<maxIter) and not converged;
        // Check convergence
        if[(abs[ll-prevLL]<tol) and iter>0; converged:1b];
        prevLL:ll;
        if[not converged;
            // M-step on valid rows only
            fp0v:fp0 vld; fp1v:fp1 vld;
            // WLS betas (sqrt-weighting)
            sqw0:sqrt fp0v|1e-10; sqw1:sqrt fp1v|1e-10;
            b0:ridge[sqw0*'Xv;sqw0*yv;lam];
            b1:ridge[sqw1*'Xv;sqw1*yv;lam];
            // Sigmas (mmu for speed)
            r0:yv - first (enlist b0) mmu Xvt;
            r1:yv - first (enlist b1) mmu Xvt;
            s0:sqrt(sum[fp0v*r0*r0]%sum fp0v)|1e-6;
            s1:sqrt(sum[fp1v*r1*r1]%sum fp1v)|1e-6;
            // Transition probs (valid rows only)
            pfp0:neg[1]_ fp0v; nfp0:1_ fp0v;
            pfp1:neg[1]_ fp1v; nfp1:1_ fp1v;
            p00:0.01|((sum pfp0*nfp0)%sum pfp0)&0.99;
            p11:0.01|((sum pfp1*nfp1)%sum pfp1)&0.99;
            // Enforce label: regime 0 = low vol
            if[s0>s1;
                tmp:b0; b0:b1; b1:tmp;
                tmp:s0; s0:s1; s1:tmp;
                tmp:p00; p00:p11; p11:tmp;
                tmp:fp0; fp0:fp1; fp1:tmp];
            // E-step (full series — filter skips nulls internally)
            filt:hamiltonFilter[y;X;b0;b1;s0;s1;p00;p11];
            fp0:filt`fp0; fp1:filt`fp1; ll:filt`ll];
        iter+:1];
    `b0`b1`s0`s1`p00`p11`fp0`fp1`ll`niter`converged!(
        b0;b1;s0;s1;p00;p11;fp0;fp1;ll;iter;converged)}

// Rolling Markov-switching regression
// At each window position, runs EM with warm-start from previous window
// @param n - window size
// @param X - feature matrix (T x p)
// @param y - target vector (T)
// @param lam - ridge regularization
// @param maxIter - max EM iterations per window
// @param tol - EM convergence tolerance
// @return - dict with fp0,fp1,b0,b1,s0,s1,p00,p11,yhat,regime (T-vectors)
msRegRolling:{[n;X;y;lam;maxIter;tol]
    T:count y; p:count first X;
    idx:swin[n;til T];
    // Step function for scan (warm-start carry)
    step:{[X;y;lam;maxIter;tol;p;state;widx]
        if[any null widx;
            :`params`fp0`fp1`b0`b1`s0`s1`p00`p11!(
                0N;0n;0n;p#0n;p#0n;0n;0n;0n;0n)];
        Xw:X widx; yw:y widx;
        pars:state`params;
        res:msRegEM[Xw;yw;lam;maxIter;tol;pars];
        `params`fp0`fp1`b0`b1`s0`s1`p00`p11!(
            `b0`b1`s0`s1`p00`p11#res;
            last res`fp0;
            last res`fp1;
            res`b0; res`b1; res`s0; res`s1; res`p00; res`p11)
        }[X;y;lam;maxIter;tol;p];
    init:`params`fp0`fp1`b0`b1`s0`s1`p00`p11!(
        0N;0n;0n;p#0n;p#0n;0n;0n;0n;0n);
    states:step\[init;idx];
    // Extract T-vectors
    fp0:states`fp0; fp1:states`fp1;
    b0s:states`b0; b1s:states`b1;
    s0:states`s0; s1:states`s1;
    p00:states`p00; p11:states`p11;
    // Anti-lookahead: use prev betas for prediction
    prevB0:prev b0s; prevB1:prev b1s;
    nf:p;
    yhat:{[nf;x;b0;b1;f0;f1]
        $[(count b0)<>nf; 0n;
          any null b0; 0n;
          (f0*sum x*b0)+f1*sum x*b1]
        }[nf]'[X;prevB0;prevB1;fp0;fp1];
    regime:?[fp1>0.5;1;0];
    `fp0`fp1`b0`b1`s0`s1`p00`p11`yhat`regime!(
        fp0;fp1;b0s;b1s;s0;s1;p00;p11;yhat;regime)}

// Regime-weighted prediction from fitted model
// @param X - feature matrix (T x p)
// @param b0 - regime 0 betas (p-vector)
// @param b1 - regime 1 betas (p-vector)
// @param fp0 - P(regime=0) (T-vector)
// @param fp1 - P(regime=1) (T-vector)
// @return - T-vector of blended predictions
msRegPredict:{[X;b0;b1;fp0;fp1]
    pred0:sum each X*\:b0;
    pred1:sum each X*\:b1;
    (fp0*pred0)+fp1*pred1}

// Rolling Markov-switching regression on table with grouping
// @param t - table (sorted by bycol, time)
// @param bycol - group column (e.g. `sym)
// @param xc - list of feature columns
// @param ycol - target column
// @param window - rolling window size
// @param lam - ridge regularization
// @param maxIter - max EM iterations per window
// @param tol - EM convergence tolerance
// @return - original table + fp0,fp1,s0,s1,p00,p11,yhat,regime,b0_*,b1_* columns
msRegTable:{[t;bycol;xc;ycol;window;lam;maxIter;tol]
    vals:distinct t bycol;
    helper:{[xc;ycol;w;lam;mi;tol;g]
        X:flip g xc; y:g ycol;
        r:msRegRolling[w;X;y;lam;mi;tol];
        bcols:(`$"b0_",/:string xc)!flip r`b0;
        bcols:bcols,(`$"b1_",/:string xc)!flip r`b1;
        main:`fp0`fp1`s0`s1`p00`p11`yhat`regime!(
            r`fp0;r`fp1;r`s0;r`s1;r`p00;r`p11;r`yhat;r`regime);
        g,'(flip main),'flip bcols
    }[xc;ycol;window;lam;maxIter;tol];
    groups:{[t;col;s] ?[t;enlist (=;col;enlist s);0b;()]}[t;bycol] each vals;
    raze helper each groups}

// =============================================================================
// CHANGEPOINT DETECTION
// =============================================================================

// Log-gamma function (Lanczos approximation, g=7)
// Handles both scalar and vector inputs
lgamma:{[z]
    f:{[z0] c:0.99999999999980993 676.5203681218851 -1259.1392167224028 771.32342877765313 -176.61502916214059 12.507343278686905 -0.13857109526572012 9.9843695780195716e-6 1.5056327351493116e-7; z1:z0 - 1f; x:c[0] + sum (c 1 + til 8) % z1 + 1f + til 8; t:z1 + 7.5; (0.5 * log[2f * acos neg 1f]) + ((z1 + 0.5) * log t) + (log x) - t};
    $[0h < type z; f each z; f z]}

// Student-t probability density function
// @param x - observation (scalar or vector)
// @param mu - location (scalar or vector, same length as nu/sigma)
// @param sigma - scale (scalar or vector, >0)
// @param nu - degrees of freedom (scalar or vector, >0)
// @return - density value(s)
studentTpdf:{[x;mu;sigma;nu]
    z:(x - mu) % sigma;
    logc:(lgamma 0.5 * nu + 1) - ((lgamma 0.5 * nu) + 0.5 * log nu * acos neg 1f);
    logp:(logc - log sigma) + (neg 0.5 * nu + 1) * log 1f + z * z % nu;
    exp logp}

// Gaussian segment cost using cumulative sums (O(1) per query)
// @param cs - cumulative sum of x (length n+1, cs[0]=0)
// @param cs2 - cumulative sum of x^2 (length n+1, cs2[0]=0)
// @param s - start index (inclusive)
// @param e - end index (inclusive)
// @return - scalar cost: n * log(var)
segCost:{[cs;cs2;s;e]
    n:1 + e - s;
    if[n < 2; :0w];
    sx:cs[e + 1] - cs s;
    sx2:cs2[e + 1] - cs2 s;
    v:(sx2 % n) - (sx % n) xexp 2;
    n * log v | 1e-10}

// Page's bilateral CUSUM changepoint detection
// @param x - observation series (T-vector)
// @param mu - target mean (scalar)
// @param k - allowance/slack parameter (e.g., 0.5 * dev x)
// @param h - alarm threshold (e.g., 4 * dev x)
// @return - `sPlus`sMinus`stat`alarm`cp
cusum:{[x;mu;k;h]
    T:count x;
    sP:T#0n; sM:T#0n; st:T#0n; al:T#0b;
    sp:0f; sm:0f; cpList:();
    i:0; while[i < T;
        xi:x i;
        $[null xi;
            [sP[i]:0n; sM[i]:0n; st[i]:0n; al[i]:0b];
            [sp:0f | sp + xi - mu + k;
             sm:0f | sm + mu - k - xi;
             sP[i]:sp; sM[i]:sm;
             st[i]:sp | sm;
             al[i]:st[i] > h;
             if[al i; cpList,:i; sp:0f; sm:0f]]];
        i+:1];
    `sPlus`sMinus`stat`alarm`cp!(sP;sM;st;al;cpList)}

// Rolling CUSUM with rolling mean/std (anti-lookahead)
// @param w - lookback window for rolling mean/std
// @param x - observation series (T-vector)
// @param k - allowance in sigma units
// @param h - threshold in sigma units
// @return - `sPlus`sMinus`stat`alarm`cp
cusumr:{[w;x;k;h]
    T:count x;
    mu:prev mavg[w;x];
    sigma:prev mdev[w;x];
    sP:T#0n; sM:T#0n; st:T#0n; al:T#0b;
    sp:0f; sm:0f; cpList:();
    i:0; while[i < T;
        xi:x i; mi:mu i; si:sigma i;
        $[(null xi) or (null mi) or (null si) or si = 0f;
            [sP[i]:0n; sM[i]:0n; st[i]:0n; al[i]:0b];
            [kAbs:k * si; hAbs:h * si;
             sp:0f | sp + xi - mi + kAbs;
             sm:0f | sm + mi - kAbs - xi;
             sP[i]:sp; sM[i]:sm;
             st[i]:sp | sm;
             al[i]:st[i] > hAbs;
             if[al i; cpList,:i; sp:0f; sm:0f]]];
        i+:1];
    `sPlus`sMinus`stat`alarm`cp!(sP;sM;st;al;cpList)}

// PELT: Pruned Exact Linear Time changepoint detection
// @param x - observation series (T-vector)
// @param pen - penalty per changepoint (e.g., log count x for BIC)
// @param minSize - minimum segment size (default 2)
// @return - sorted list of changepoint indices
pelt:{[x;pen;minSize]
    ms:minSize | 2;
    // Filter nulls, work on valid data
    vld:where not null x;
    xv:x vld; N:count xv;
    if[N < 2 * ms; :()];
    // Precompute cumulative sums
    cs:0f, sums xv;
    cs2:0f, sums xv * xv;
    // DP arrays
    F:(N + 1)#0w;
    F[0]:neg pen;
    bp:(N + 1)#0Ni;  // backpointer
    t:ms;
    while[t <= N;
        // Evaluate all candidates from 0 to t-ms
        valid:til (t - ms) + 1;
        valid:valid where (t - valid) >= ms;
        if[count valid;
            costs:F[valid] + {[cs;cs2;s;t] segCost[cs;cs2;s;t - 1]}[cs;cs2;;t] each valid;
            cpen:costs + pen;
            bestIdx:cpen ? min cpen;
            if[(min cpen) < F t;
                F[t]:min cpen;
                bp[t]:valid bestIdx]];
        t+:1];
    // Backtrack
    cps:();
    pos:N;
    while[(pos > 0) and not null bp pos;
        if[bp[pos] > 0; cps,:bp pos];
        pos:bp pos];
    // Map back to original indices
    cps:asc cps;
    if[count cps; cps:vld cps];
    cps}

// Binary segmentation changepoint detection
// @param x - observation series (T-vector)
// @param pen - penalty per changepoint
// @param minSize - minimum segment size
// @param maxCP - maximum number of changepoints
// @return - sorted list of changepoint indices
binseg:{[x;pen;minSize;maxCP]
    ms:minSize | 2;
    vld:where not null x;
    xv:x vld; N:count xv;
    if[N < 2 * ms; :()];
    cs:0f, sums xv;
    cs2:0f, sums xv * xv;
    // Find best split in segment [s,e]
    bestSplit:{[cs;cs2;ms;s;e]
        if[(e - s + 1) < 2 * ms; :(0Ni; neg 0w)];
        mpts:(s + ms) + til (e - ms) - (s + ms) + 1;
        if[0 = count mpts; :(0Ni; neg 0w)];
        baseCost:segCost[cs;cs2;s;e];
        splitCosts:{[cs;cs2;s;e;m]
            segCost[cs;cs2;s;m - 1] + segCost[cs;cs2;m;e]
            }[cs;cs2;s;e] each mpts;
        improvements:baseCost - splitCosts;
        bestIdx:improvements ? max improvements;
        (mpts bestIdx; improvements bestIdx)}[cs;cs2;ms];
    // Iterative: maintain list of segments, greedily split best
    segs:enlist (0; N - 1);
    cps:();
    nCP:0;
    while[nCP < maxCP;
        // Find best split across all segments
        splits:bestSplit ./: segs;
        imps:splits[;1];
        bestSeg:imps ? max imps;
        if[(max imps) < pen; :asc $[count cps; vld cps; cps]];
        m:splits[bestSeg;0];
        cps,:m;
        // Replace segment with two sub-segments
        seg:segs bestSeg;
        newSegs:((seg 0; m - 1); (m; seg 1));
        segs:((bestSeg # segs), newSegs, (bestSeg + 1) _ segs);
        nCP+:1];
    asc $[count cps; vld cps; cps]}

// Compute statistics per segment defined by changepoints
// @param x - observation series (T-vector)
// @param cps - sorted list of changepoint indices
// @return - table ([] start; end; mean; var; n)
segmentStats:{[x;cps]
    T:count x;
    starts:0, cps;
    ends:(cps - 1), T - 1;
    stats:{[x;s;e]
        seg:x s + til 1 + e - s;
        v:where not null seg;
        segv:seg v;
        `start`end`mean`var`n!(s; e; avg segv; var segv; count segv)
        }[x] .' flip (starts; ends);
    flip stats}

// Bayesian Online Changepoint Detection (Adams & MacKay 2007)
// @param x - observation series (T-vector)
// @param hazard - expected run length (e.g., 250)
// @param mu0 - prior mean
// @param kappa0 - prior precision (pseudo-observations for mean)
// @param alpha0 - prior shape for variance (e.g., 2.0)
// @param beta0 - prior scale for variance
// @return - `cpProb`runLength`postMean`postVar (T-vectors)
bocpd:{[x;hazard;mu0;kappa0;alpha0;beta0]
    T:count x; maxRL:500;
    H:1f % hazard;
    cpP:T#0n; rl:T#0Ni; pMu:T#0n; pVar:T#0n;
    // Run-length distribution and sufficient stats (muS stores kappa*mu)
    R:enlist 1f;
    muS:enlist mu0 * kappa0;
    kapS:enlist kappa0;
    alpS:enlist alpha0;
    betS:enlist beta0;
    i:0; while[i < T;
        xi:x i;
        $[null xi;
            [cpP[i]:0n; rl[i]:0Ni; pMu[i]:0n; pVar[i]:0n];
            [muN:muS % kapS;
             nuN:2f * alpS;
             sig2:betS * (kapS + 1f) % (alpS * kapS);
             predLL:studentTpdf[xi; muN; sqrt sig2; nuN];
             growth:R * predLL * 1f - H;
             cpMass:sum R * predLL * H;
             R1:(enlist cpMass), growth;
             rSum:sum R1;
             R:$[rSum > 0; R1 % rSum; R1];
             if[count[R] > maxRL; R:maxRL # R; rS:sum R; if[rS > 0; R:R % rS]];
             // Beta update BEFORE mu/kap/alp update (uses old posterior mean)
             betNew:betS + (kapS % kapS + 1f) * 0.5 * (xi - muN) * xi - muN;
             // Now update other stats
             muS:(enlist mu0 * kappa0), muS + xi;
             kapS:(enlist kappa0), kapS + 1f;
             alpS:(enlist alpha0), alpS + 0.5;
             betS:(enlist beta0), betNew;
             if[count[muS] > maxRL; muS:maxRL # muS; kapS:maxRL # kapS; alpS:maxRL # alpS; betS:maxRL # betS];
             cpP[i]:R 0;
             mapR:R ? max R;
             rl[i]:mapR;
             pMu[i]:muS[mapR] % kapS mapR;
             pVar[i]:betS[mapR] % (alpS[mapR] - 1f) | 1e-300]];
        i+:1];
    `cpProb`runLength`postMean`postVar!(cpP;rl;pMu;pVar)}

// BOCPD with auto-estimated priors
// @param x - observation series (T-vector)
// @param hazard - expected run length (e.g., 250)
// @return - same as bocpd
bocpdAuto:{[x;hazard]
    x0:(100 & sum not null x) # x where not null x;
    mu0:avg x0;
    kappa0:1f;
    alpha0:2f;
    beta0:var[x0] * alpha0 - 1f;
    bocpd[x;hazard;mu0;kappa0;alpha0;beta0 | 1e-10]}

// CUSUM table interface
cpTableCusum:{[t;bycol;col;params]
    w:params`window; k:params`k; h:params`h;
    vals:distinct t bycol;
    helper:{[col;w;k;h;g]
        r:cusumr[w;g col;k;h];
        g,'flip `cusumStat`cusumAlarm!(r`stat;r`alarm)
    }[col;w;k;h];
    groups:{[t;c;s] ?[t;enlist (=;c;enlist s);0b;()]}[t;bycol] each vals;
    raze helper each groups}

// PELT table interface
cpTablePelt:{[t;bycol;col;params]
    pen:params`pen; ms:params`minSize;
    vals:distinct t bycol;
    helper:{[col;pen;ms;g]
        x:g col; n:count x;
        cps:pelt[x;pen;ms];
        isCP:n#0b;
        if[count cps; isCP[cps]:1b];
        // Broadcast segment stats to each row
        starts:0, cps; ends:(cps - 1), n - 1;
        segMu:n#0n; segVar:n#0n;
        j:0; while[j < count starts;
            idx:starts[j] + til 1 + ends[j] - starts j;
            seg:x idx; segv:seg where not null seg;
            segMu[idx]:avg segv; segVar[idx]:var segv;
            j+:1];
        g,'flip `isChangepoint`segMean`segVar!(isCP;segMu;segVar)
    }[col;pen;ms];
    groups:{[t;c;s] ?[t;enlist (=;c;enlist s);0b;()]}[t;bycol] each vals;
    raze helper each groups}

// BinSeg table interface
cpTableBinseg:{[t;bycol;col;params]
    pen:params`pen; ms:params`minSize; mc:params`maxCP;
    vals:distinct t bycol;
    helper:{[col;pen;ms;mc;g]
        x:g col; n:count x;
        cps:binseg[x;pen;ms;mc];
        isCP:n#0b;
        if[count cps; isCP[cps]:1b];
        starts:0, cps; ends:(cps - 1), n - 1;
        segMu:n#0n; segVar:n#0n;
        j:0; while[j < count starts;
            idx:starts[j] + til 1 + ends[j] - starts j;
            seg:x idx; segv:seg where not null seg;
            segMu[idx]:avg segv; segVar[idx]:var segv;
            j+:1];
        g,'flip `isChangepoint`segMean`segVar!(isCP;segMu;segVar)
    }[col;pen;ms;mc];
    groups:{[t;c;s] ?[t;enlist (=;c;enlist s);0b;()]}[t;bycol] each vals;
    raze helper each groups}

// BOCPD table interface
bocpdTable:{[t;bycol;col;hazard]
    vals:distinct t bycol;
    helper:{[col;hazard;g]
        r:bocpdAuto[g col;hazard];
        g,'flip `cpProb`runLength`postMean`postVar!(r`cpProb;r`runLength;r`postMean;r`postVar)
    }[col;hazard];
    groups:{[t;c;s] ?[t;enlist (=;c;enlist s);0b;()]}[t;bycol] each vals;
    raze helper each groups}

// Unified changepoint detection table interface
// @param t - table (sorted by bycol, time)
// @param bycol - group column
// @param col - data column
// @param method - `cusum`pelt`binseg`bocpd
// @param params - dict of method-specific parameters
// @return - original table + method-specific output columns
cpTable:{[t;bycol;col;method;params]
    $[method = `cusum;  cpTableCusum[t;bycol;col;params];
      method = `pelt;   cpTablePelt[t;bycol;col;params];
      method = `binseg;  cpTableBinseg[t;bycol;col;params];
      method = `bocpd;  bocpdTable[t;bycol;col;params`hazard];
      '"Unknown changepoint method: ",string method]}

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

// =============================================================================
// DECISION TREES (Optimized)
// =============================================================================

// Gini impurity for classification
gini:{[y] p:count each group y; p%:sum p; 1 - sum p*p}

// Weighted Gini for split evaluation
giniSplit:{[yL;yR] nL:count yL; nR:count yR; n:nL+nR; ((nL%n)*gini yL) + (nR%n)*gini yR}

// MSE for regression splits
mseSplit:{[yL;yR] nL:count yL; nR:count yR; n:nL+nR; ((nL%n)*var yL) + (nR%n)*var yR}

// Mode (most frequent value) for classification
mode:{first key desc count each group x}

// =============================================================================
// OPTIMIZED SPLIT FINDING - O(n log n) instead of O(n²)
// =============================================================================

// Fast MSE split finder - fully vectorized, no loops
// Returns (threshold; weightedMSE) or (0n; 0w) if no valid split
bestSplitFeatFast:{[x;y;minLeaf]
    n:count y;
    if[n < 2*minLeaf; :(0n;0w)];
    ord:iasc x;
    xs:x ord;
    ys:y ord;
    splitIdx:where differ xs;
    if[0=count splitIdx; :(0n;0w)];
    validIdx:splitIdx where (splitIdx >= minLeaf) & (splitIdx <= n - minLeaf);
    if[0=count validIdx; :(0n;0w)];
    cumSum:sums ys;
    cumSumSq:sums ys * ys;
    totalSum:last cumSum;
    totalSumSq:last cumSumSq;
    // Vectorized MSE calculation - no each loop
    sumL:cumSum validIdx - 1;
    sumSqL:cumSumSq validIdx - 1;
    nL:`float$validIdx;
    nR:`float$n - validIdx;
    sumR:totalSum - sumL;
    sumSqR:totalSumSq - sumSqL;
    meanL:sumL % nL;
    meanR:sumR % nR;
    varL:(sumSqL % nL) - meanL * meanL;
    varR:(sumSqR % nR) - meanR * meanR;
    mseScores:(nL * varL) + nR * varR;
    bestLocalIdx:mseScores?min mseScores;
    bestSplitIdx:validIdx bestLocalIdx;
    (0.5 * xs[bestSplitIdx-1] + xs bestSplitIdx; mseScores bestLocalIdx)}

// Fast Gini split finder - fully vectorized matrix operations
bestSplitFeatFastCls:{[x;y;minLeaf]
    n:count y;
    if[n < 2*minLeaf; :(0n;0w)];
    ord:iasc x;
    xs:x ord;
    ys:y ord;
    splitIdx:where differ xs;
    if[0=count splitIdx; :(0n;0w)];
    validIdx:splitIdx where (splitIdx >= minLeaf) & (splitIdx <= n - minLeaf);
    if[0=count validIdx; :(0n;0w)];
    classes:asc distinct ys;
    k:count classes;
    classIdx:classes?ys;
    cumCounts:sums each (til k) =\: classIdx;
    totalCounts:cumCounts[;n-1];
    // Vectorized Gini: k x m matrices where m = count validIdx
    countsL:cumCounts[;validIdx - 1];
    countsR:totalCounts -' countsL;
    nL:`float$validIdx;
    nR:`float$n - validIdx;
    // Each-left broadcasts vector division across matrix rows
    pL:countsL %\: nL;
    pR:countsR %\: nR;
    gL:1 - sum pL * pL;
    gR:1 - sum pR * pR;
    giniScores:((nL % n) * gL) + (nR % n) * gR;
    bestLocalIdx:giniScores?min giniScores;
    bestSplitIdx:validIdx bestLocalIdx;
    (0.5 * xs[bestSplitIdx-1] + xs bestSplitIdx; giniScores bestLocalIdx)}

// Find best split across all features (optimized)
// Uses peach for parallel execution when threads available (start q with -s N)
bestSplitFast:{[X;y;minLeaf;treeType]
    splitFn:$[treeType=`cls;bestSplitFeatFastCls[;;minLeaf];bestSplitFeatFast[;;minLeaf]];
    feats:flip X;
    splits:$[0 < system"s";
        splitFn[;y] peach feats;
        splitFn[;y] each feats];
    impurities:splits[;1];
    bestFeat:impurities?min impurities;
    (bestFeat; splits[bestFeat;0]; splits[bestFeat;1])}

// =============================================================================
// INDEX-BASED TREE BUILDING - Avoid data copying
// =============================================================================

// Build tree using index arrays (avoids copying X and y)
dtBuildIdx:{[X;y;idx;maxDepth;minLeaf;treeType]
    n:count idx;
    leafVal:$[treeType=`cls;mode;avg];
    if[(n<2*minLeaf) or maxDepth<1; :`leaf`val!(1b;leafVal y idx)];
    // Get subset views
    Xsub:X idx;
    ysub:y idx;
    split:bestSplitFast[Xsub;ysub;minLeaf;treeType];
    if[0w=split 2; :`leaf`val!(1b;leafVal ysub)];
    feat:split 0; thresh:split 1;
    mask:Xsub[;feat] <= thresh;
    idxL:idx where mask;
    idxR:idx where not mask;
    if[(minLeaf>count idxL) or minLeaf>count idxR; :`leaf`val!(1b;leafVal ysub)];
    // Recurse with index subsets
    left:dtBuildIdx[X;y;idxL;maxDepth-1;minLeaf;treeType];
    right:dtBuildIdx[X;y;idxR;maxDepth-1;minLeaf;treeType];
    `leaf`feat`thresh`left`right!(0b;feat;thresh;left;right)}

// Fit decision tree (optimized version)
// @param X - matrix (n obs x p features)
// @param y - target (n values)
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param treeType - `reg (regression) or `cls (classification)
// @return - tree model dict
dtFit:{[X;y;maxDepth;minLeaf;treeType]
    tree:dtBuildIdx[X;y;til count y;maxDepth;minLeaf;treeType];
    `tree`treeType`nFeatures!(tree;treeType;count first X)}

// Predict single observation
dtPredictOne:{[tree;x]
    node:tree;
    while[not node`leaf;
        node:$[x[node`feat] <= node`thresh; node`left; node`right]];
    node`val}

// Vectorized tree prediction - process all rows at once
// Uses index tracking through tree traversal
dtPredictVec:{[tree;X]
    n:count X;
    preds:n#0f;
    queue:enlist (tree; til n);
    while[count queue;
        item:first queue;
        queue:1 _ queue;
        node:item 0;
        idx:item 1;
        if[0 = count idx; :preds];
        if[node`leaf; preds[idx]:node`val];
        if[not node`leaf;
            feat:node`feat;
            thresh:node`thresh;
            vals:X[idx;feat];
            leftMask:vals <= thresh;
            leftIdx:idx where leftMask;
            rightIdx:idx where not leftMask;
            if[count leftIdx; queue,:enlist (node`left; leftIdx)];
            if[count rightIdx; queue,:enlist (node`right; rightIdx)]
        ]
    ];
    preds}

// Predict using fitted tree model (vectorized)
dtPredict:{[model;X]
    dtPredictVec[model`tree;X]}

// Predict using fitted tree model (row-by-row, for single predictions)
dtPredictSlow:{[model;X]
    dtPredictOne[model`tree] each X}

// Recursive helper to count splits per feature
dtCountSplits:{[node;counts]
    if[node`leaf; :counts];
    counts[node`feat]+:1;
    counts:dtCountSplits[node`left;counts];
    counts:dtCountSplits[node`right;counts];
    counts}

// Feature importance (simple: split count per feature)
dtImportance:{[model]
    p:model`nFeatures;
    counts:dtCountSplits[model`tree;p#0];
    counts % sum counts | 1}

// =============================================================================
// RULE-BASED SIGNAL TREE
// =============================================================================

// Apply a single rule: (featureIdx; op; threshold) -> boolean mask
// ops: `lt `le `gt `ge `eq
applyRule:{[X;rule]
    feat:rule 0; op:rule 1; thresh:rule 2;
    col:X[;feat];
    $[op=`lt; col < thresh;
      op=`le; col <= thresh;
      op=`gt; col > thresh;
      op=`ge; col >= thresh;
      op=`eq; col = thresh;
      count[col]#1b]}

// Rule tree node format:
// Leaf: (val)  - just a value
// Branch: (rule; trueNode; falseNode)
// rule: (featureIdx; op; threshold)

// Evaluate rule tree for single observation
ruleTreeOne:{[node;x]
    // Leaf is just a value (atom or single-element list)
    if[1>=count node; :first node,()];
    rule:node 0; trueNode:node 1; falseNode:node 2;
    feat:rule 0; op:rule 1; thresh:rule 2;
    v:x feat;
    cond:$[op=`lt; v < thresh;
           op=`le; v <= thresh;
           op=`gt; v > thresh;
           op=`ge; v >= thresh;
           op=`eq; v = thresh;
           1b];
    $[cond; ruleTreeOne[trueNode;x]; ruleTreeOne[falseNode;x]]}

// Evaluate rule tree for matrix X
// @param tree - rule tree structure
// @param X - matrix (n obs x p features)
// @return - predictions for each row
ruleTree:{[tree;X] ruleTreeOne[tree] each X}

// Helper to build rule tree from readable format
// Example:
// tree: (`rsi;`lt;30; 1; (`macd;`gt;0; 0.5; -0.5))
// Means: if rsi<30 then 1, else if macd>0 then 0.5 else -0.5
// Features can be indices or symbols (with featureNames dict)
ruleTreeBuild:{[tree;featureNames]
    if[1>=count tree; :tree];  // leaf
    rule:3#tree;
    // Convert symbol feature name to index if needed
    if[-11h=type rule 0; rule[0]:featureNames?rule 0];
    trueNode:ruleTreeBuild[tree 3;featureNames];
    falseNode:ruleTreeBuild[tree 4;featureNames];
    (rule;trueNode;falseNode)}

// Convenience: build and evaluate rule tree with named features
// @param rules - rule tree with symbol feature names
// @param featureNames - list of symbols mapping to column indices
// @param X - data matrix
ruleTreeEval:{[rules;featureNames;X]
    tree:ruleTreeBuild[rules;featureNames];
    ruleTree[tree;X]}

// =============================================================================
// RANDOM FOREST
// =============================================================================

// Bootstrap sample (sample n indices with replacement)
bootstrap:{[n] n?n}

// Build tree with random feature subset at each split
// mtry = number of features to consider at each split
dtBuildIdxRF:{[X;y;idx;maxDepth;minLeaf;treeType;mtry]
    n:count idx;
    leafVal:$[treeType=`cls;mode;avg];
    if[(n<2*minLeaf) or maxDepth<1; :`leaf`val!(1b;leafVal y idx)];
    Xsub:X idx;
    ysub:y idx;
    // Random feature subset
    p:count first X;
    featIdx:neg[mtry]?p;
    // Find best split only among selected features
    splitFn:$[treeType=`cls;bestSplitFeatFastCls[;;minLeaf];bestSplitFeatFast[;;minLeaf]];
    splits:splitFn[;ysub] each (flip Xsub)[featIdx];
    impurities:splits[;1];
    bestLocal:impurities?min impurities;
    if[0w=splits[bestLocal;1]; :`leaf`val!(1b;leafVal ysub)];
    feat:featIdx bestLocal;
    thresh:splits[bestLocal;0];
    mask:Xsub[;feat] <= thresh;
    idxL:idx where mask;
    idxR:idx where not mask;
    if[(minLeaf>count idxL) or minLeaf>count idxR; :`leaf`val!(1b;leafVal ysub)];
    left:dtBuildIdxRF[X;y;idxL;maxDepth-1;minLeaf;treeType;mtry];
    right:dtBuildIdxRF[X;y;idxR;maxDepth-1;minLeaf;treeType;mtry];
    `leaf`feat`thresh`left`right!(0b;feat;thresh;left;right)}

// Fit single random forest tree (bootstrap + feature subsampling)
rfTreeFit:{[X;y;maxDepth;minLeaf;treeType;mtry]
    n:count y;
    bootIdx:bootstrap n;
    tree:dtBuildIdxRF[X;y;bootIdx;maxDepth;minLeaf;treeType;mtry];
    `tree`treeType`nFeatures!(tree;treeType;count first X)}

// Fit random forest
// @param X - matrix (n obs x p features)
// @param y - target vector
// @param nTrees - number of trees
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param treeType - `reg or `cls
// @param mtry - features per split (default: sqrt(p) for cls, p/3 for reg)
// @return - forest model
rfFit:{[X;y;nTrees;maxDepth;minLeaf;treeType;mtry]
    p:count first X;
    mtry:$[null mtry; $[treeType=`cls; ceiling sqrt p; ceiling p%3]; mtry];
    trees:{[X;y;md;ml;tt;mt;i] rfTreeFit[X;y;md;ml;tt;mt]}[X;y;maxDepth;minLeaf;treeType;mtry] peach til nTrees;
    `trees`treeType`nFeatures`nTrees`mtry!(trees;treeType;p;nTrees;mtry)}

// Predict with random forest
rfPredict:{[model;X]
    preds:dtPredict[;X] peach model`trees;
    $[model[`treeType]=`cls;
        mode each flip preds;   // majority vote
        avg each flip preds]}   // average

// Out-of-bag error estimate (uses samples not in bootstrap)
rfOOB:{[X;y;nTrees;maxDepth;minLeaf;treeType;mtry]
    n:count y;
    p:count first X;
    mtry:$[null mtry; $[treeType=`cls; ceiling sqrt p; ceiling p%3]; mtry];
    // Track predictions and counts for each observation
    oobPreds:n#enlist();
    i:0;
    do[nTrees;
        bootIdx:bootstrap n;
        oobIdx:(til n) except distinct bootIdx;
        tree:dtBuildIdxRF[X;y;bootIdx;maxDepth;minLeaf;treeType;mtry];
        model:`tree`treeType`nFeatures!(tree;treeType;p);
        preds:dtPredict[model;X oobIdx];
        // Store OOB predictions
        j:0;
        {[oobPreds;idx;pred] oobPreds[idx],:pred; oobPreds}[;oobIdx j;preds j]/[oobPreds;til count oobIdx]
    ];
    // Aggregate OOB predictions
    finalPreds:$[treeType=`cls;
        {$[0=count x;0N;mode x]} each oobPreds;
        {$[0=count x;0n;avg x]} each oobPreds];
    valid:where not null finalPreds;
    $[treeType=`cls;
        1 - avg y[valid]=finalPreds valid;   // misclassification rate
        avg (y[valid] - finalPreds valid) xexp 2]}  // MSE

// =============================================================================
// GRADIENT BOOSTING - REGRESSION (OPTIMIZED)
// =============================================================================

// Fit gradient boosted trees for regression
// Optimized: pre-allocated tree list, vectorized operations
// @param X - matrix (n obs x p features)
// @param y - target vector
// @param nTrees - number of boosting iterations
// @param maxDepth - max tree depth (typically 3-6)
// @param minLeaf - min samples in leaf
// @param lr - learning rate / shrinkage (0.01-0.3 typical)
// @return - gradient boosting model
gbFit:{[X;y;nTrees;maxDepth;minLeaf;lr]
    n:count y;
    p:count first X;
    init:avg y;
    preds:n#init;
    trees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        resid:y - preds;
        tree:dtBuildIdx[X;resid;idx;maxDepth;minLeaf;`reg];
        treeModel:`tree`treeType`nFeatures!(tree;`reg;p);
        treePreds:dtPredictVec[tree;X];
        preds+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees!(trees;init;lr;p;nTrees)}

// Stochastic GB fit - each tree trained on random row subsample
// subsample: fraction of rows to use per tree (0.3-0.8 typical)
gbFitStoch:{[X;y;nTrees;maxDepth;minLeaf;lr;subsample]
    n:count y;
    p:count first X;
    sampleSize:`int$n*subsample;
    init:avg y;
    preds:n#init;
    trees:nTrees#enlist (::);
    i:0;
    do[nTrees;
        sampleIdx:neg[sampleSize]?n;
        residSub:(y - preds) sampleIdx;
        tree:dtBuildIdx[X sampleIdx;residSub;til sampleSize;maxDepth;minLeaf;`reg];
        treeModel:`tree`treeType`nFeatures!(tree;`reg;p);
        treePreds:dtPredictVec[tree;X];
        preds+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees!(trees;init;lr;p;nTrees)}

// Update leaf values in existing tree structure (keep splits, recompute values)
// Used for warm-starting
dtUpdateLeaves:{[tree;X;y;idx]
    if[tree`leaf;
        :`leaf`val!(1b;avg y idx)
    ];
    feat:tree`feat;
    thresh:tree`thresh;
    mask:X[idx;feat] <= thresh;
    idxL:idx where mask;
    idxR:idx where not mask;
    left:$[count idxL; dtUpdateLeaves[tree`left;X;y;idxL]; tree`left];
    right:$[count idxR; dtUpdateLeaves[tree`right;X;y;idxR]; tree`right];
    `leaf`feat`thresh`left`right!(0b;feat;thresh;left;right)}

// Warm-start GB fit - reuse tree structure from previous model
// Much faster: O(n) leaf updates vs O(n log n) split finding
gbFitWarm:{[prevModel;X;y]
    n:count y;
    p:count first X;
    lr:prevModel`lr;
    nTrees:prevModel`nTrees;
    init:avg y;
    preds:n#init;
    trees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        resid:y - preds;
        oldTree:prevModel[`trees][i][`tree];
        tree:dtUpdateLeaves[oldTree;X;resid;idx];
        treeModel:`tree`treeType`nFeatures!(tree;`reg;p);
        treePreds:dtPredictVec[tree;X];
        preds+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees!(trees;init;lr;p;nTrees)}

// Batch predict for multiple trees (vectorized)
// Returns matrix: nTrees x n observations
dtPredictBatch:{[trees;X]
    {[tree;X] dtPredictVec[tree`tree;X]}[;X] each trees}

// Predict with gradient boosting regression model (vectorized)
gbPredict:{[model;X]
    init:model`init;
    lr:model`lr;
    n:count X;
    preds:n#init;
    {[X;lr;preds;tree]
        preds + lr * dtPredictVec[tree`tree;X]
    }[X;lr]/[preds;model`trees]}

// Staged predictions (predictions after each boosting round)
gbPredictStaged:{[model;X]
    init:model`init;
    lr:model`lr;
    treePreds:dtPredictBatch[model`trees;X];
    init + lr * sums treePreds}

// =============================================================================
// GRADIENT BOOSTING - BINARY CLASSIFICATION (OPTIMIZED)
// =============================================================================

// Sigmoid function (vectorized, numerically stable)
sigmoid:{1 % 1 + exp neg x}

// Log-loss gradient: y - p (where y in {0,1} and p = sigmoid(F))
// Hessian: p * (1 - p)

// Fit gradient boosted trees for binary classification
// Optimized: pre-allocated trees, optional Newton-Raphson leaf values
// @param X - matrix (n obs x p features)
// @param y - binary target (0/1)
// @param nTrees - number of boosting iterations
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param lr - learning rate
// @return - gradient boosting classification model
gbcFit:{[X;y;nTrees;maxDepth;minLeaf;lr]
    n:count y;
    p:count first X;
    // Initialize with log-odds
    pPos:avg y;
    init:log (pPos + 1e-10) % (1 - pPos + 1e-10);
    F:n#init;
    // Pre-allocate tree list
    trees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        prob:sigmoid F;
        negGrad:y - prob;
        tree:dtBuildIdx[X;negGrad;idx;maxDepth;minLeaf;`reg];
        treeModel:`tree`treeType`nFeatures!(tree;`reg;p);
        treePreds:dtPredictVec[tree;X];
        F+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees!(trees;init;lr;p;nTrees)}

// Newton-Raphson boosted classification (second-order, faster convergence)
// Uses Hessian weighting for optimal leaf values: leaf = sum(grad) / sum(hess)
// @param X - matrix (n obs x p features)
// @param y - binary target (0/1)
// @param nTrees - number of boosting iterations
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param lr - learning rate
// @return - gradient boosting classification model
gbcFitNewton:{[X;y;nTrees;maxDepth;minLeaf;lr]
    n:count y;
    p:count first X;
    pPos:avg y;
    init:log (pPos + 1e-10) % (1 - pPos + 1e-10);
    F:n#init;
    trees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        prob:sigmoid F;
        grad:y - prob;
        hess:prob * 1 - prob;
        tree:dtBuildIdx[X;grad;idx;maxDepth;minLeaf;`reg];
        tree:dtNewtonLeaves[tree;X;grad;hess];
        treeModel:`tree`treeType`nFeatures!(tree;`reg;p);
        treePreds:dtPredictVec[tree;X];
        F+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees!(trees;init;lr;p;nTrees)}

// Update leaf values using Newton-Raphson: sum(grad) / (sum(hess) + lambda)
// Traverses tree and recomputes leaf values based on gradient/hessian
dtNewtonLeaves:{[node;X;grad;hess]
    if[node`leaf;
        // Newton leaf value: sum(grad) / sum(hess + regularization)
        leafVal:(sum grad) % (sum hess) | 0.001;  // min hess for stability
        :node,`val`nval!(leafVal;count grad)];
    // Recurse on children
    feat:node`feat;
    thresh:node`thresh;
    mask:X[;feat] <= thresh;
    leftIdx:where mask;
    rightIdx:where not mask;
    left:dtNewtonLeaves[node`left;X leftIdx;grad leftIdx;hess leftIdx];
    right:dtNewtonLeaves[node`right;X rightIdx;grad rightIdx;hess rightIdx];
    node,`left`right!(left;right)}

// Predict probabilities with gradient boosting classification
gbcPredictProb:{[model;X]
    init:model`init;
    lr:model`lr;
    n:count X;
    F:n#init;
    F:{[X;lr;F;tree]
        F + lr * dtPredictVec[tree`tree;X]
    }[X;lr]/[F;model`trees];
    sigmoid F}

// Predict class labels with gradient boosting classification
gbcPredict:{[model;X]
    probs:gbcPredictProb[model;X];
    probs >= 0.5}

// =============================================================================
// GRADIENT BOOSTING - MULTI-CLASS CLASSIFICATION (OPTIMIZED)
// =============================================================================

// Softmax function (for vector of logits, numerically stable)
softmax:{[x] e:exp x - max x; e % sum e}

// Softmax for matrix (each row is a sample)
softmaxRows:{[X] {e:exp x - max x; e % sum e} each X}

// Fit gradient boosted trees for multi-class using softmax (true multinomial)
// More efficient than one-vs-rest: builds K trees per round jointly
// @param X - matrix (n obs x p features)
// @param y - class labels (integers 0 to K-1)
// @param nTrees - number of boosting rounds
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param lr - learning rate
// @return - multi-class gradient boosting model
gbmcFit:{[X;y;nTrees;maxDepth;minLeaf;lr]
    n:count y;
    p:count first X;
    classes:asc distinct y;
    K:count classes;
    // Map y to class indices 0..K-1
    yIdx:classes?y;
    // Initialize F matrix: n x K
    F:(n;K)#0f;
    // One-hot encode y as floats
    yOneHot:`float$(til K) =\: yIdx;  // K x n matrix
    // Pre-allocate: list of K trees per round
    allTrees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        // Compute probabilities via softmax
        probs:softmaxRows F;  // n x K
        // For each class, compute gradient and fit tree
        roundTrees:K#enlist (::);
        k:0;
        do[K;
            grad:yOneHot[k] - probs[;k];
            tree:dtBuildIdx[X;grad;idx;maxDepth;minLeaf;`reg];
            roundTrees[k]:`tree`treeType`nFeatures!(tree;`reg;p);
            treePreds:dtPredictVec[tree;X];
            F[;k]+:lr * treePreds;
            k+:1
        ];
        allTrees[i]:roundTrees;
        i+:1
    ];
    `allTrees`classes`nFeatures`nTrees`lr`K!(allTrees;classes;p;nTrees;lr;K)}

// Predict class probabilities for multi-class (optimized)
gbmcPredictProb:{[model;X]
    n:count X;
    K:model`K;
    lr:model`lr;
    F:(n;K)#0f;
    // Accumulate predictions from all trees
    F:{[X;lr;K;F;roundTrees]
        newF:F;
        k:0;
        do[K;
            treePreds:dtPredictVec[roundTrees[k]`tree;X];
            newF:.[newF;(::;k);+;lr * treePreds];
            k+:1
        ];
        newF
    }[X;lr;K]/[F;model`allTrees];
    softmaxRows F}

// Predict class labels for multi-class
gbmcPredict:{[model;X]
    probs:gbmcPredictProb[model;X];
    model[`classes] probs?'max each probs}

// =============================================================================
// ROLLING GRADIENT BOOSTING
// =============================================================================

// Rolling GB Regression - train on window, predict next point
// Returns out-of-sample predictions (first n values are null)
// @param n - window size
// @param X - feature matrix
// @param y - target vector
// @param nTrees - trees per model
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
gbr:{[n;X;y;nTrees;maxDepth;minLeaf;lr]
    len:count y;
    preds:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:gbFit[Xwin;ywin;nTrees;maxDepth;minLeaf;lr];
        preds[i]:first gbPredict[model;enlist X i];
        i+:1
    ];
    preds}

// Fast Rolling GB Regression - retrains every k steps (stride)
// @param n - window size
// @param X - feature matrix
// @param y - target
// @param nTrees - trees per model
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
// @param stride - retrain every stride steps (e.g., 5 = 5x faster)
gbrF:{[n;X;y;nTrees;maxDepth;minLeaf;lr;stride]
    len:count y;
    preds:len#0n;
    model:();
    i:n;
    while[i < len;
        // Retrain model every 'stride' steps or on first iteration
        if[(0 = (i-n) mod stride) or (0 = count model);
            idx:(i-n) + til n;
            model:gbFit[X idx;y idx;nTrees;maxDepth;minLeaf;lr]
        ];
        preds[i]:first gbPredict[model;enlist X i];
        i+:1
    ];
    preds}

// Batch Rolling GB Regression - predicts k steps ahead with each model
// Faster but predictions use slightly older models
// @param n - window size
// @param X - feature matrix
// @param y - target
// @param nTrees - trees per model
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
// @param batch - predict this many steps per model (e.g., 5 = 5x faster)
gbrB:{[n;X;y;nTrees;maxDepth;minLeaf;lr;batch]
    len:count y;
    preds:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        model:gbFit[X idx;y idx;nTrees;maxDepth;minLeaf;lr];
        // Predict for next 'batch' steps (or remaining)
        batchEnd:len & i+batch;
        batchIdx:i + til batchEnd - i;
        batchPreds:gbPredict[model;X batchIdx];
        preds[batchIdx]:batchPreds;
        i:batchEnd
    ];
    preds}

// Warm-start Rolling GB - reuses tree structure, only updates leaf values
// Fastest option: ~3-5x faster than gbr
// @param n - window size
// @param X - feature matrix
// @param y - target
// @param nTrees - trees per model
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
// @param rebuildEvery - rebuild tree structure every N steps (0=never)
gbrW:{[n;X;y;nTrees;maxDepth;minLeaf;lr;rebuildEvery]
    len:count y;
    preds:len#0n;
    model:();
    i:n;
    stepsSinceRebuild:0;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:$[(0 = count model) or ((rebuildEvery > 0) and stepsSinceRebuild >= rebuildEvery);
            [stepsSinceRebuild::0; gbFit[Xwin;ywin;nTrees;maxDepth;minLeaf;lr]];
            [stepsSinceRebuild+::1; gbFitWarm[model;Xwin;ywin]]
        ];
        preds[i]:first gbPredict[model;enlist X i];
        i+:1
    ];
    preds}

// Stochastic Rolling GB - uses row subsampling for faster training
// subsample: fraction of window to use per tree (0.3-0.5 typical)
gbrS:{[n;X;y;nTrees;maxDepth;minLeaf;lr;subsample]
    len:count y;
    preds:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        model:gbFitStoch[X idx;y idx;nTrees;maxDepth;minLeaf;lr;subsample];
        preds[i]:first gbPredict[model;enlist X i];
        i+:1
    ];
    preds}

// Rolling GB Classification - returns probabilities
// @param n - window size
// @param X - feature matrix
// @param y - binary target (0/1 floats)
// @param nTrees - trees per model
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
gbcr:{[n;X;y;nTrees;maxDepth;minLeaf;lr]
    len:count y;
    probs:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:gbcFit[Xwin;ywin;nTrees;maxDepth;minLeaf;lr];
        probs[i]:first gbcPredictProb[model;enlist X i];
        i+:1
    ];
    probs}

// Fast Rolling GB Classification - retrains every k steps
// @param stride - retrain every stride steps (e.g., 5 = 5x faster)
gbcrF:{[n;X;y;nTrees;maxDepth;minLeaf;lr;stride]
    len:count y;
    probs:len#0n;
    model:();
    i:n;
    while[i < len;
        if[(0 = (i-n) mod stride) or (0 = count model);
            idx:(i-n) + til n;
            model:gbcFit[X idx;y idx;nTrees;maxDepth;minLeaf;lr]
        ];
        probs[i]:first gbcPredictProb[model;enlist X i];
        i+:1
    ];
    probs}

// Batch Rolling GB Classification - predicts k steps ahead
// @param batch - predict this many steps per model (e.g., 5 = 5x faster)
gbcrB:{[n;X;y;nTrees;maxDepth;minLeaf;lr;batch]
    len:count y;
    probs:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        model:gbcFit[X idx;y idx;nTrees;maxDepth;minLeaf;lr];
        batchEnd:len & i+batch;
        batchIdx:i + til batchEnd - i;
        batchProbs:gbcPredictProb[model;X batchIdx];
        probs[batchIdx]:batchProbs;
        i:batchEnd
    ];
    probs}

// Rolling GB Classification with Newton-Raphson
gbcnr:{[n;X;y;nTrees;maxDepth;minLeaf;lr]
    len:count y;
    probs:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:gbcFitNewton[Xwin;ywin;nTrees;maxDepth;minLeaf;lr];
        probs[i]:first gbcPredictProb[model;enlist X i];
        i+:1
    ];
    probs}

// Rolling Random Forest Regression
rfr:{[n;X;y;nTrees;maxDepth;minLeaf]
    len:count y;
    preds:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:rfFit[Xwin;ywin;nTrees;maxDepth;minLeaf;`reg;0N];
        preds[i]:first rfPredict[model;enlist X i];
        i+:1
    ];
    preds}

// Rolling Random Forest Classification - returns probabilities
// For classification, returns proportion of trees voting for class 1
rfcr:{[n;X;y;nTrees;maxDepth;minLeaf]
    len:count y;
    probs:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:rfFit[Xwin;ywin;nTrees;maxDepth;minLeaf;`cls;0N];
        // Get votes from each tree
        votes:{dtPredictOne[x`tree;y]}[;X i] each model`trees;
        probs[i]:avg votes;
        i+:1
    ];
    probs}

// Rolling Histogram-based GB (fastest)
// Uses default nBins=64, lambda=0.1
gbHistr:{[n;X;y;nTrees;maxDepth;minLeaf;lr]
    len:count y;
    preds:len#0n;
    i:n;
    while[i < len;
        idx:(i-n) + til n;
        Xwin:X idx;
        ywin:y idx;
        model:gbHistFit[Xwin;ywin;nTrees;maxDepth;minLeaf;lr;64;0.1];
        preds[i]:first gbPredict[model;enlist X i];
        i+:1
    ];
    preds}

// =============================================================================
// TABLE-FRIENDLY ROLLING ENSEMBLE FUNCTIONS
// =============================================================================
// These functions work directly with tables and column names
// Usage: .kdbtools.gbcrT[t;`rsi`macd`sma20;`target;60;20;3;5;0.1]

// Rolling GB Classification on table
// @param tbl - table with features and target
// @param xc - feature column names (symbol list)
// @param yc - target column name (symbol)
// @param n - window size
// @param nTrees - number of trees
// @param maxDepth - max tree depth
// @param minLeaf - min samples per leaf
// @param lr - learning rate
// @return column of probabilities (same length as table)
gbcrT:{[tbl;xc;yc;n;nTrees;maxDepth;minLeaf;lr]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbcr[n;X;y;nTrees;maxDepth;minLeaf;lr]}

// Rolling GB Classification Newton on table
gbcnrT:{[tbl;xc;yc;n;nTrees;maxDepth;minLeaf;lr]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbcnr[n;X;y;nTrees;maxDepth;minLeaf;lr]}

// Rolling GB Regression on table
gbrT:{[tbl;xc;yc;n;nTrees;maxDepth;minLeaf;lr]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbr[n;X;y;nTrees;maxDepth;minLeaf;lr]}

// Rolling Random Forest Classification on table
rfcrT:{[tbl;xc;yc;n;nTrees;maxDepth;minLeaf]
    X:flip tbl xc;
    y:`float$tbl yc;
    rfcr[n;X;y;nTrees;maxDepth;minLeaf]}

// Rolling Random Forest Regression on table
rfrT:{[tbl;xc;yc;n;nTrees;maxDepth;minLeaf]
    X:flip tbl xc;
    y:`float$tbl yc;
    rfr[n;X;y;nTrees;maxDepth;minLeaf]}

// Fast Rolling GB Classification on table (stride)
// Uses stride=5 by default for 5x speedup
// @param n - window size
// @param nTrees - trees (default minLeaf=5, lr=0.1)
gbcrFT:{[tbl;xc;yc;n;nTrees;maxDepth]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbcrF[n;X;y;nTrees;maxDepth;5;0.1;5]}

// Fast Rolling GB Regression on table (stride)
gbrFT:{[tbl;xc;yc;n;nTrees;maxDepth]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbrF[n;X;y;nTrees;maxDepth;5;0.1;5]}

// Batch Rolling GB Classification on table
gbcrBT:{[tbl;xc;yc;n;nTrees;maxDepth]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbcrB[n;X;y;nTrees;maxDepth;5;0.1;5]}

// Batch Rolling GB Regression on table
gbrBT:{[tbl;xc;yc;n;nTrees;maxDepth]
    X:flip tbl xc;
    y:`float$tbl yc;
    gbrB[n;X;y;nTrees;maxDepth;5;0.1;5]}

// =============================================================================
// HISTOGRAM-BASED GRADIENT BOOSTING (XGBoost-style, QML optimized)
// =============================================================================

// Histogram binning for faster split finding
// Bins continuous features into discrete buckets
histBin:{[x;nBins]
    mn:min x; mx:max x;
    edges:mn + (mx - mn) * (til nBins + 1) % nBins;
    bins:edges bin x;
    bins:bins & nBins - 1;  // clamp to valid range
    (bins;edges)}

// Build histogram of gradients/hessians per bin (vectorized)
// Returns dict with gradient sum, hessian sum, count per bin
histGradSum:{[bins;grad;hess;nBins]
    // Group and sum for each bin present
    grpd:group bins;
    binKeys:key grpd;
    idxLists:value grpd;
    // Sum gradients and hessians for each bin
    gVals:sum each grad idxLists;
    hVals:sum each hess idxLists;
    cVals:count each idxLists;
    // Build result arrays using @
    gSum:@[nBins#0f;binKeys;:;gVals];
    hSum:@[nBins#0f;binKeys;:;hVals];
    cnt:@[nBins#0;binKeys;:;cVals];
    `gSum`hSum`cnt!(gSum;hSum;cnt)}

// Find best split using histogram (O(nBins) instead of O(n))
histBestSplit:{[hist;minLeaf;lambda]
    nBins:count hist`gSum;
    totalG:sum hist`gSum;
    totalH:sum hist`hSum;
    totalN:sum hist`cnt;
    // Compute cumulative sums
    cumG:sums hist`gSum;
    cumH:sums hist`hSum;
    cumN:sums hist`cnt;
    // Compute gain at each split point using vectorized ops
    // gain = 0.5 * (GL^2/(HL+λ) + GR^2/(HR+λ) - (GL+GR)^2/(HL+HR+λ))
    validIdx:til nBins - 1;
    nL:cumN validIdx;
    nR:totalN - nL;
    validMask:(nL >= minLeaf) & nR >= minLeaf;
    gL:cumG validIdx;
    gR:totalG - gL;
    hL:cumH validIdx;
    hR:totalH - hL;
    gainL:(gL * gL) % hL + lambda;
    gainR:(gR * gR) % hR + lambda;
    gainP:(totalG * totalG) % totalH + lambda;
    gains:0.5 * (gainL + gainR - gainP);
    gains:@[gains;where not validMask;:;neg 0w];
    bestBin:gains?max gains;
    (bestBin;gains bestBin)}

// Histogram-based gradient boosting (XGBoost-style)
// Uses binned features for O(nBins) split finding
// @param X - matrix (n obs x p features)
// @param y - target vector
// @param nTrees - number of boosting iterations
// @param maxDepth - max tree depth
// @param minLeaf - min samples in leaf
// @param lr - learning rate
// @param nBins - number of histogram bins (default 256)
// @param lambda - L2 regularization on leaf weights
// @return - gradient boosting model
gbHistFit:{[X;y;nTrees;maxDepth;minLeaf;lr;nBins;lambda]
    n:count y;
    p:count first X;
    // Pre-bin all features
    binData:{[col;nBins] histBin[col;nBins]}[;nBins] each flip X;
    Xbins:flip binData[;0];  // binned feature matrix
    edges:binData[;1];       // bin edges per feature
    init:avg y;
    preds:n#init;
    trees:nTrees#enlist (::);
    idx:til n;
    i:0;
    do[nTrees;
        resid:y - preds;
        tree:dtBuildHist[Xbins;resid;idx;maxDepth;minLeaf;lambda;nBins;edges];
        treeModel:`tree`treeType`nFeatures`edges!(tree;`reg;p;edges);
        treePreds:dtPredictVec[tree;X];
        preds+:lr * treePreds;
        trees[i]:treeModel;
        i+:1
    ];
    `trees`init`lr`nFeatures`nTrees`nBins`lambda!(trees;init;lr;p;nTrees;nBins;lambda)}

// Build tree using histogram-based split finding
dtBuildHist:{[Xbins;y;idx;maxDepth;minLeaf;lambda;nBins;edges]
    n:count idx;
    if[(n<2*minLeaf) or maxDepth<1; :`leaf`val!(1b;avg y idx)];
    Xsub:Xbins idx;
    ysub:y idx;
    // Gradient and Hessian (for regression, grad=resid, hess=1)
    grad:ysub;
    hess:n#1f;
    // Find best split using histograms
    p:count first Xbins;
    bestGain:neg 0w;
    bestFeat:0;
    bestBin:0;
    f:0;
    do[p;
        hist:histGradSum[Xsub[;f];grad;hess;nBins];
        split:histBestSplit[hist;minLeaf;lambda];
        if[split[1] > bestGain;
            bestGain:split 1;
            bestFeat:f;
            bestBin:split 0
        ];
        f+:1
    ];
    if[bestGain <= 0; :`leaf`val!(1b;avg ysub)];
    // Threshold is upper edge of best bin
    thresh:edges[bestFeat] bestBin + 1;
    mask:Xsub[;bestFeat] <= bestBin;
    idxL:idx where mask;
    idxR:idx where not mask;
    if[(minLeaf>count idxL) or minLeaf>count idxR; :`leaf`val!(1b;avg ysub)];
    left:dtBuildHist[Xbins;y;idxL;maxDepth-1;minLeaf;lambda;nBins;edges];
    right:dtBuildHist[Xbins;y;idxR;maxDepth-1;minLeaf;lambda;nBins;edges];
    `leaf`feat`thresh`left`right!(0b;bestFeat;thresh;left;right)}

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

    // Decision Trees
    Xdt:flip (100?1f;100?1f;100?1f);
    ydt:sum each Xdt;
    ydtCls:ydt > 1.5;
    dtReg:.kdbtools.dtFit[Xdt;ydt;4;5;`reg];
    dtCls:.kdbtools.dtFit[Xdt;ydtCls;4;5;`cls];
    results[`dtFitReg]:3=count dtReg;
    results[`dtFitCls]:3=count dtCls;
    results[`dtPredict]:100=count .kdbtools.dtPredict[dtReg;Xdt];
    results[`dtImportance]:3=count .kdbtools.dtImportance dtReg;

    // Rule Tree
    ruleTree:(`rsi;`lt;30;1;(`macd;`gt;0;0.5;-0.5));
    results[`ruleTreeBuild]:3=count .kdbtools.ruleTreeBuild[ruleTree;`rsi`macd];

    // Random Forest
    rfReg:.kdbtools.rfFit[Xdt;ydt;5;3;5;`reg;0N];
    rfCls:.kdbtools.rfFit[Xdt;ydtCls;5;3;5;`cls;0N];
    results[`rfFitReg]:5=count rfReg;
    results[`rfFitCls]:5=count rfCls;
    results[`rfPredict]:100=count .kdbtools.rfPredict[rfReg;Xdt];

    // Gradient Boosting
    gbReg:.kdbtools.gbFit[Xdt;ydt;10;3;5;0.1];
    results[`gbFit]:5=count gbReg;
    results[`gbPredict]:100=count .kdbtools.gbPredict[gbReg;Xdt];
    ydtClsF:`float$ydtCls;
    gbCls:.kdbtools.gbcFit[Xdt;ydtClsF;10;3;5;0.1];
    results[`gbcFit]:5=count gbCls;
    results[`gbcPredict]:100=count .kdbtools.gbcPredict[gbCls;Xdt];
    results[`gbcPredictProb]:100=count .kdbtools.gbcPredictProb[gbCls;Xdt];

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
