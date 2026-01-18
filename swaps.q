// =============================================================================
// SWAP PRICING LIBRARY
// =============================================================================
// A comprehensive interest rate swap pricing library for KDB+/Q
// Supports curve building, interpolation, swap pricing, and risk analytics
//
// Author: kdbtools
// Version: 0.1.0
// =============================================================================

\d .swaps

// Version info
version:"0.1.0"

// =============================================================================
// CONFIGURATION AND DEFAULTS
// =============================================================================

// Default configuration (dict-based, following optimizer.q pattern)
// asOfDate: valuation date (defaults to today), dayCount: ACT360, spotDays: 2,
// compounding: continuous, frequency: 6M, interpMethod: loglinear
defaults:`asOfDate`dayCount`spotDays`compounding`frequency`interpMethod!(.z.d;`ACT360;2i;`continuous;`6M;`loglinear)

// Supported conventions
dayCounts:`ACT360`ACT365`30360`ACTACT
frequencies:`1M`3M`6M`12M
compoundings:`continuous`annual`semiannual`quarterly

// =============================================================================
// TENOR PARSING AND DATE UTILITIES
// =============================================================================

// Parse single tenor symbol to year fraction
// tenorToYF[`3M] -> 0.25
// tenorToYF[`1Y] -> 1.0
// tenorToYF[`10Y] -> 10.0
tenorToYF:{[tenor]
    s:string tenor;
    n:"F"$-1_s;
    unit:upper last s;
    $[unit="M"; n%12;
      unit="Y"; n;
      unit="W"; n%52;
      unit="D"; n%365;
      '"Unknown tenor unit: ",s]}

// Parse list of tenors to year fractions (vectorized)
tenorsToYF:{[tenors] tenorToYF each tenors}

// Generate payment year fractions for a given maturity and frequency
// genPaySchedule[5;`6M] -> 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5
genPaySchedule:{[maturity;freq]
    dt:tenorToYF freq;
    n:1 | `long$maturity % dt;
    sched:dt * 1 + til n;
    $[last[sched] > maturity; @[sched;n-1;:;maturity]; sched]}

// =============================================================================
// DAY COUNT CONVENTIONS
// =============================================================================

// ACT/360: actual days / 360
dcfACT360:{[d1;d2] (d2 - d1) % 360f}

// ACT/365: actual days / 365
dcfACT365:{[d1;d2] (d2 - d1) % 365f}

// 30/360 (Bond basis): each month treated as 30 days
dcf30360:{[d1;d2]
    y1:`year$d1; m1:`mm$d1; dd1:`dd$d1;
    y2:`year$d2; m2:`mm$d2; dd2:`dd$d2;
    dd1:dd1 & 30;
    dd2:$[dd1>=30; dd2 & 30; dd2];
    ((360*y2-y1) + (30*m2-m1) + dd2-dd1) % 360f}

// ACT/ACT: actual days / actual days in year
dcfACTACT:{[d1;d2]
    y1:`year$d1;
    daysInYear:$[(0=y1 mod 4) and (0<>y1 mod 100) or 0=y1 mod 400; 366f; 365f];
    (d2 - d1) % daysInYear}

// Day count fraction dispatcher
// For year fractions (no actual dates), uses simple division
dcf:{[t1;t2;convention]
    $[convention~`ACT360; (t2-t1);
      convention~`ACT365; (t2-t1);
      convention~`30360;  (t2-t1);
      convention~`ACTACT; (t2-t1);
      (t2-t1)]}

// Day count fraction between two actual dates
dcfDates:{[d1;d2;convention]
    $[convention~`ACT360; dcfACT360[d1;d2];
      convention~`ACT365; dcfACT365[d1;d2];
      convention~`30360;  dcf30360[d1;d2];
      convention~`ACTACT; dcfACTACT[d1;d2];
      dcfACT360[d1;d2]]}  // default to ACT/360

// =============================================================================
// DATE UTILITIES (RELATIVE TO CURVE VALUATION DATE)
// =============================================================================

// Convert date to year fraction from curve's asOfDate
// dateToYF[curve; 2025.06.15] -> year fraction using curve's day count
dateToYF:{[curve;dt]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    dcfDates[asOf;dt;dc]}

// Convert multiple dates to year fractions (vectorized)
datesToYF:{[curve;dates]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    dcfDates[asOf;;dc] each dates}

// Convert year fraction to date from curve's asOfDate
// yfToDate[curve; 0.5] -> date approximately 6 months from asOfDate
yfToDate:{[curve;yf]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    // Approximate conversion (inverse of day count)
    days:$[dc~`ACT360; `int$yf * 360;
           dc~`ACT365; `int$yf * 365;
           dc~`30360;  `int$yf * 360;
           dc~`ACTACT; `int$yf * 365;
           `int$yf * 365];
    asOf + days}

// Convert multiple year fractions to dates
yfsToDate:{[curve;yfs] yfToDate[curve;] each yfs}

// Add tenor to a date
// addTenor[2025.01.15; `3M] -> 2025.04.15
// addTenor[2025.01.15; `1Y] -> 2026.01.15
addTenor:{[dt;tenor]
    s:string tenor;
    n:"J"$-1_s;
    unit:upper last s;
    $[unit="D"; dt + `int$n;
      unit="W"; dt + `int$7*n;
      unit="M"; addMonthsToDate[dt;`int$n];
      unit="Y"; addMonthsToDate[dt;`int$12*n];
      dt]}  // unknown unit, return original

// Helper: add n months to a date (optimized - no string operations)
addMonthsToDate:{[dt;n]
    // Direct integer extraction (faster than string conversion)
    y:`int$`year$dt;
    m:`int$`mm$dt;
    d:`int$`dd$dt;
    // Calculate new month
    newM:m + n;
    yAdj:(newM - 1) div 12;
    mAdj:1 + (newM - 1) mod 12;
    // Handle negative months
    if[mAdj <= 0; yAdj-:1; mAdj+:12];
    newY:y + yAdj;
    // Max days in target month (inline for speed)
    isLeap:(0=newY mod 4) and (0<>newY mod 100) or 0=newY mod 400;
    maxD:$[mAdj in 1 3 5 7 8 10 12; 31; mAdj in 4 6 9 11; 30; mAdj=2; $[isLeap;29;28]; 31];
    newD:d & maxD;
    // Build date: Jan 1 of target year + days offset
    // Use date arithmetic: base date + days
    jan1:("D"$string[newY],".01.01");
    cumDays:0 31 59 90 120 151 181 212 243 273 304 334;
    daysFromJan1:(cumDays mAdj-1) + newD - 1;
    // Adjust for leap year if past Feb
    if[isLeap and mAdj > 2; daysFromJan1+:1];
    jan1 + daysFromJan1}

// Days in month for a given year
daysInMonthForYear:{[m;y]
    isLeap:(0=y mod 4) and (0<>y mod 100) or 0=y mod 400;
    $[m in 1 3 5 7 8 10 12; 31;
      m in 4 6 9 11; 30;
      m=2; $[isLeap;29;28];
      31]}

// Generate payment dates for a swap given start date, end date, and frequency
// genPayDates[2025.01.15; 2030.01.15; `3M] -> list of payment dates
genPayDates:{[startDate;endDate;freq]
    // Vectorized approach: calculate number of periods and generate all at once
    freqMonths:$[freq~`1M;1;freq~`3M;3;freq~`6M;6;freq~`12M;12;freq~`1Y;12;3];
    // Approximate number of periods
    approxMonths:12 * (endDate - startDate) % 365;
    nPeriods:1 + `int$approxMonths % freqMonths;
    // Generate dates by adding months from start
    monthOffsets:freqMonths * 1 + til nPeriods;
    dates:addMonthsToDate[startDate;] each monthOffsets;
    // Filter to valid range and ensure endDate is included
    dates:dates where dates <= endDate;
    if[not endDate in dates; dates:dates,endDate];
    dates}

// Generate payment year fractions for dated swap (relative to curve asOfDate)
genPayScheduleDated:{[curve;startDate;endDate;freq]
    dates:genPayDates[startDate;endDate;freq];
    datesToYF[curve;dates]}

// =============================================================================
// INTERPOLATION - LINEAR
// =============================================================================

// Linear interpolation
// x: knot points, y: values at knots, xi: points to interpolate
interpLinear:{[x;y;xi]
    n:count x;
    idx:{[x;v] 0 | (count[x]-2) & x bin v}[x] each xi;
    t:(xi - x idx) % (x[idx+1] - x idx) + 1e-15;
    y[idx] + t * y[idx+1] - y idx}

// Log-linear interpolation (better for discount factors)
interpLogLinear:{[x;y;xi]
    logY:log y;
    exp interpLinear[x;logY;xi]}

// =============================================================================
// INTERPOLATION - CUBIC SPLINE
// =============================================================================

// Thomas algorithm for solving tridiagonal systems
// a: lower diagonal, b: main diagonal, c: upper diagonal, d: RHS
solveTridiag:{[a;b;c;d]
    n:count b;
    cp:enlist c[0] % b[0];
    dp:enlist d[0] % b[0];
    i:1;
    while[i < n-1;
        denom:b[i] - a[i] * cp[i-1];
        cp:cp,c[i] % denom;
        dp:dp,(d[i] - a[i] * dp[i-1]) % denom;
        i+:1];
    dp:dp,(d[n-1] - a[n-1] * dp[n-2]) % (b[n-1] - a[n-1] * cp[n-2]);
    x:enlist dp[n-1];
    i:n-2;
    while[i >= 0;
        x:(dp[i] - cp[i] * x[0]),x;
        i-:1];
    x}

// Build natural cubic spline coefficients
// Returns dict with x, a, b, c, d coefficients for each segment
cubicSplineCoeffs:{[x;y]
    n:count x;
    if[n < 3; '"Need at least 3 points for cubic spline"];
    h:(1_ x) - -1_ x;
    delta:((1_ y) - -1_ y) % h;
    // For natural spline: M[0] = M[n-1] = 0, solve for M[1]...M[n-2]
    // System size is (n-2) x (n-2)
    m:n - 2;
    if[m < 1; '"Need at least 3 points"];
    diagMain:2 * (h[til m] + h[1 + til m]);
    diagLo:h[1 + til m-1];
    diagHi:h[1 + til m-1];
    rhs:6 * ((1_ delta) - -1_ delta);
    Mint:solveTridiag[(0f),diagLo;diagMain;diagHi,(0f);rhs];
    M:(0f),Mint,0f;
    a:-1_ y;
    cc:0.5 * -1_ M;
    dd:((1_ M) - -1_ M) % 6 * h;
    bb:((1_ y) - -1_ y) % h - (h * (2 * -1_ M) + 1_ M) % 6;
    `x`a`b`c`d!(x;a;bb;cc;dd)}

// Evaluate cubic spline at points xi
cubicSplineEval:{[coeffs;xi]
    x:coeffs`x; a:coeffs`a; b:coeffs`b; c:coeffs`c; d:coeffs`d;
    xi:`float$xi;
    idx:{[x;v] 0 | (count[x]-2) & x bin v}[x] each xi;
    dx:xi - x idx;
    a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))}

// =============================================================================
// INTERPOLATION - SMOOTHING SPLINE
// =============================================================================

// Smoothing spline (Reinsch algorithm, scipy-style)
// p: smoothing parameter in (0,1), 0=interpolating, 1=linear regression
smoothingSpline:{[x;y;p]
    n:count x;
    if[n < 3; '"Need at least 3 points for smoothing spline"];
    h:(1_ x) - -1_ x;
    lambda:$[p<=0; 1e-10; p>=1; 1e10; p % (1-p)];
    m:n - 2;
    diagR:(h[til m] + h[1 + til m]) % 3;
    offR:$[m > 1; h[1 + til m-1] % 6; ()];
    diagMod:diagR + lambda;
    delta:((1_ y) - -1_ y) % h;
    Qty:(1_ delta) - -1_ delta;
    gamma:$[m = 1; enlist Qty[0] % diagMod[0]; solveTridiag[(0f),offR;diagMod;offR,(0f);Qty]];
    gamma:(0f),gamma,0f;
    a:-1_ y;
    cc:0.5 * -1_ gamma;
    dd:((1_ gamma) - -1_ gamma) % 6 * h;
    bb:((1_ y) - -1_ y) % h - (h * (2 * -1_ gamma) + 1_ gamma) % 6;
    `x`a`b`c`d`lambda`p!(x;a;bb;cc;dd;lambda;p)}

// Evaluate smoothing spline (same as cubic)
smoothingSplineEval:{[coeffs;xi] cubicSplineEval[coeffs;xi]}

// =============================================================================
// INTERPOLATION - MASTER DISPATCHER
// =============================================================================

// Master interpolation function
// method: `linear`loglinear`cubic`smooth
interp:{[x;y;xi;method]
    $[method~`linear;    interpLinear[x;y;xi];
      method~`loglinear; interpLogLinear[x;y;xi];
      method~`cubic;     cubicSplineEval[cubicSplineCoeffs[x;y];xi];
      method~`smooth;    smoothingSplineEval[smoothingSpline[x;y;0.001];xi];
      '"Unknown interpolation method: ",string method]}

// =============================================================================
// CURVE BUILDING - ZERO/DF CONVERSIONS
// =============================================================================

// Convert zero rate to discount factor
zero2df:{[rate;t;compounding]
    $[t=0; 1f;
      compounding~`continuous;  exp neg rate * t;
      compounding~`annual;      xexp[1+rate;neg t];
      compounding~`semiannual;  xexp[1+rate%2;neg 2*t];
      compounding~`quarterly;   xexp[1+rate%4;neg 4*t];
      exp neg rate * t]}

// Convert discount factor to zero rate
df2zero:{[df;t;compounding]
    $[t=0; 0f;
      df<=0; 0f;
      compounding~`continuous;  neg log[df] % t;
      compounding~`annual;      xexp[df;neg 1%t] - 1;
      compounding~`semiannual;  2 * xexp[df;neg 1%2*t] - 1;
      compounding~`quarterly;   4 * xexp[df;neg 1%4*t] - 1;
      neg log[df] % t]}

// =============================================================================
// CURVE BUILDING - BOOTSTRAP
// =============================================================================

// Bootstrap discount factors from par swap rates
// yearFracs: maturities, parRates: corresponding par rates
// freq: payment frequency, config: configuration dict
bootstrap:{[yearFracs;parRates;freq;config]
    n:count yearFracs;
    dt:tenorToYF freq;
    dfs:n#0f;
    i:0;
    while[i < n;
        T:yearFracs i;
        r:parRates i;
        nPay:`long$T % dt;
        payTimes:dt * 1 + til nPay;
        $[i=0;
            [dfs[i]:1 % 1 + r * T];
            [priorIdx:where payTimes < T;
             if[0 < count priorIdx;
                 priorTimes:payTimes priorIdx;
                 priorDFs:interpLogLinear[yearFracs til i;dfs til i;priorTimes];
                 priorSum:sum priorDFs * dt;
                 dfs[i]:(1 - r * priorSum) % 1 + r * dt];
             if[0 = count priorIdx;
                 dfs[i]:1 % 1 + r * T]]];
        i+:1];
    dfs}

// Convert year fraction to tenor label (for display)
yfToTenor:{[yf]
    days:`int$yf*365;
    weeks:`int$yf*52;
    months:`int$yf*12;
    $[days < 7;        `$string[days],"D";
      yf < 1%12;       `$string[weeks],"W";
      yf < 1;          `$string[months],"M";
      yf = `int$yf;    `$string[`int$yf],"Y";
      `$string[yf],"Y"]}

// Smooth par rates using smoothing spline
// Takes sparse par rate input and returns smoothed rates at same points
// p: smoothing parameter (0=interpolating, higher=smoother)
smoothParRates:{[yf;rates;p]
    coeffs:smoothingSpline[yf;rates;p];
    smoothingSplineEval[coeffs;yf]}

// Interpolate par rates at finer grid using smoothing spline
// yf: original year fractions, rates: original par rates
// yfNew: new year fractions to interpolate at
// p: smoothing parameter
interpParRatesSmooth:{[yf;rates;yfNew;p]
    coeffs:smoothingSpline[yf;rates;p];
    smoothingSplineEval[coeffs;yfNew]}

// Build complete yield curve from market data
// tenors: list of tenor symbols (e.g., `3M`6M`1Y`2Y`5Y`10Y) OR year fractions (e.g., 0.25 0.5 1 2 5 10)
// rates: corresponding par swap rates
// config: optional configuration overrides
//   `smoothPar - if set, smooth par rates first using smoothing spline (value is smoothing param, e.g., 0.001)
//   `interpMethod - method for interpolating zeros after bootstrap
buildCurve:{[tenors;rates;config]
    cfg:defaults,config;
    isSymbol:11h=abs type tenors;
    yf:$[isSymbol; tenorsToYF tenors; `float$tenors];
    labels:$[isSymbol; tenors; yfToTenor each yf];
    sortIdx:iasc yf;
    yf:yf sortIdx;
    rates:rates sortIdx;
    labels:labels sortIdx;
    // Apply smoothing spline to par rates if requested
    rates:$[`smoothPar in key cfg;
        smoothParRates[yf;rates;cfg`smoothPar];
        rates];
    df:bootstrap[yf;rates;cfg`frequency;cfg];
    zeros:{[df;t;comp] df2zero[df;t;comp]}[;;cfg`compounding]'[df;yf];
    `tenors`yearFracs`discountFactors`zeroRates`parRates`config!(labels;yf;df;zeros;rates;cfg)}

// Build curve with finer grid by first smoothing par rates
// Builds curve at more points for smoother forward rates
// yf: input year fractions, rates: input par rates
// gridStep: spacing for finer grid (e.g., 0.25 for quarterly points)
// p: smoothing parameter for spline on par rates
// config: other curve config
buildCurveFineGrid:{[yf;rates;gridStep;p;config]
    // Sort inputs
    sortIdx:iasc yf;
    yf:yf sortIdx;
    rates:rates sortIdx;
    // Create finer grid from 0 to max maturity
    maxYF:last yf;
    minYF:first yf;
    nPts:1 + `long$(maxYF - minYF) % gridStep;
    yfFine:minYF + gridStep * til nPts;
    // Interpolate par rates at fine grid using smoothing spline
    ratesFine:interpParRatesSmooth[yf;rates;yfFine;p];
    // Build curve at fine grid
    buildCurve[yfFine;ratesFine;config,enlist[`smoothPar]!enlist 0n]}

// Rebuild discount factors from zero rates (for curve bumping)
rebuildFromZeros:{[curve]
    yf:curve`yearFracs;
    zeros:curve`zeroRates;
    comp:curve[`config]`compounding;
    dfs:{[z;t;c] zero2df[z;t;c]}[;;comp]'[zeros;yf];
    @[curve;`discountFactors;:;dfs]}

// Interpolate zero rate at arbitrary time
interpZero:{[curve;t]
    cfg:curve`config;
    im:cfg`interpMethod;
    method:$[im~`loglinear;`linear;im];
    r:interp[curve`yearFracs;curve`zeroRates;enlist `float$t;method];
    r 0}

// Interpolate discount factor at arbitrary time (via zero rate interpolation)
// This approach produces smoother forward rates than interpolating DFs directly
interpDF:{[curve;t]
    if[t=0; :1f];
    zeroInt:interpZero[curve;t];
    zero2df[zeroInt;t;curve[`config]`compounding]}

// Interpolate discount factors at multiple times (via zero rate interpolation)
interpDFs:{[curve;ts]
    ts:`float$ts;
    cfg:curve`config;
    im:cfg`interpMethod;
    method:$[im~`loglinear;`linear;im];
    comp:cfg`compounding;
    zeroInts:interp[curve`yearFracs;curve`zeroRates;ts;method];
    {[z;t;c] zero2df[z;t;c]}[;;comp]'[zeroInts;ts]}

// =============================================================================
// FORWARD RATES
// =============================================================================

// Forward rate from t1 to t2
fwdRate:{[curve;t1;t2]
    df1:$[t1=0;1f;interpDF[curve;t1]];
    df2:interpDF[curve;t2];
    ((df1 % df2) - 1) % (t2 - t1)}

// Forward discount factor from t1 to t2
fwdDF:{[curve;t1;t2]
    df1:$[t1=0;1f;interpDF[curve;t1]];
    df2:interpDF[curve;t2];
    df2 % df1}

// Generate forward rates for a payment schedule
genForwardRates:{[curve;payDates]
    n:count payDates;
    starts:(0f),payDates til n-1;
    {[c;t1;t2] fwdRate[c;t1;t2]}[curve]'[starts;payDates]}

// =============================================================================
// SWAP PRICING
// =============================================================================

// Fixed leg present value
pvFixedLeg:{[curve;notional;fixedRate;payDates]
    n:count payDates;
    starts:(0f),payDates til n-1;
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    notional * fixedRate * sum dcfs * dfs}

// Floating leg present value (single curve: PV = N * (1 - df(T)))
pvFloatingLeg:{[curve;notional;maturity]
    dfT:interpDF[curve;maturity];
    notional * 1 - dfT}

// Floating leg PV with explicit forward rates
pvFloatingLegExplicit:{[curve;notional;payDates]
    n:count payDates;
    starts:(0f),payDates til n-1;
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    fwds:{[c;t1;t2] fwdRate[c;t1;t2]}[curve]'[starts;payDates];
    notional * sum fwds * dcfs * dfs}

// Price vanilla interest rate swap
// curve: yield curve from buildCurve
// notional: notional amount
// fixedRate: fixed rate (e.g., 0.05 for 5%)
// maturity: swap maturity in years
// freq: payment frequency
// isReceiver: 1b = receive fixed, 0b = pay fixed
priceSwap:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    payDates:genPaySchedule[maturity;freq];
    pvFixed:pvFixedLeg[curve;notional;fixedRate;payDates];
    pvFloat:pvFloatingLeg[curve;notional;maturity];
    pv:$[isReceiver; pvFixed - pvFloat; pvFloat - pvFixed];
    `pv`pvFixed`pvFloat`notional`fixedRate`maturity`frequency`isReceiver!(pv;pvFixed;pvFloat;notional;fixedRate;maturity;freq;isReceiver)}

// Par swap rate (rate that makes PV = 0)
parRate:{[curve;maturity;freq]
    payDates:genPaySchedule[maturity;freq];
    if[0=count payDates; payDates:enlist maturity];
    n:count payDates;
    starts:(0f),payDates til n-1;
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    dfT:last dfs;
    (1 - dfT) % sum dcfs * dfs}

// =============================================================================
// FORWARD STARTING SWAPS
// =============================================================================

// Generate payment schedule for forward starting swap
// Returns payment dates from fwdStart to fwdStart+tenor
genFwdPaySchedule:{[fwdStart;tenor;freq]
    dt:tenorToYF freq;
    n:1 | `long$tenor % dt;
    fwdStart + dt * 1 + til n}

// Fixed leg PV for forward starting swap
pvFixedLegFwd:{[curve;notional;fixedRate;fwdStart;payDates]
    n:count payDates;
    starts:fwdStart,(payDates til n-1);
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    notional * fixedRate * sum dcfs * dfs}

// Floating leg PV for forward starting swap (single curve)
// PV = N * (df(fwdStart) - df(maturity))
pvFloatingLegFwd:{[curve;notional;fwdStart;maturity]
    dfStart:interpDF[curve;fwdStart];
    dfEnd:interpDF[curve;maturity];
    notional * dfStart - dfEnd}

// Price forward starting swap
// fwdStart: when the swap starts (in years from today)
// tenor: length of the swap (in years)
// e.g., fwdStart=1, tenor=5 is a 1Y forward 5Y swap (1x6)
priceForwardSwap:{[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver]
    maturity:fwdStart + tenor;
    payDates:genFwdPaySchedule[fwdStart;tenor;freq];
    pvFixed:pvFixedLegFwd[curve;notional;fixedRate;fwdStart;payDates];
    pvFloat:pvFloatingLegFwd[curve;notional;fwdStart;maturity];
    pv:$[isReceiver; pvFixed - pvFloat; pvFloat - pvFixed];
    `pv`pvFixed`pvFloat`notional`fixedRate`fwdStart`tenor`maturity`frequency`isReceiver!(pv;pvFixed;pvFloat;notional;fixedRate;fwdStart;tenor;maturity;freq;isReceiver)}

// Par rate for forward starting swap
forwardParRate:{[curve;fwdStart;tenor;freq]
    maturity:fwdStart + tenor;
    payDates:genFwdPaySchedule[fwdStart;tenor;freq];
    n:count payDates;
    starts:fwdStart,(payDates til n-1);
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    dfStart:interpDF[curve;fwdStart];
    dfEnd:last dfs;
    (dfStart - dfEnd) % sum dcfs * dfs}

// =============================================================================
// DATED SWAP PRICING (USING ACTUAL DATES)
// =============================================================================

// Price swap using actual dates (relative to curve's asOfDate)
// startDate: effective date (use curve[`config]`asOfDate for spot starting)
// endDate: maturity date
priceSwapDated:{[curve;notional;fixedRate;startDate;endDate;freq;isReceiver]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;

    // Convert dates to year fractions from asOfDate
    startYF:$[startDate <= asOf; 0f; dcfDates[asOf;startDate;dc]];
    endYF:dcfDates[asOf;endDate;dc];

    // Generate payment schedule using actual dates
    paymentDates:genPayDates[startDate;endDate;freq];
    payYFs:dcfDates[asOf;;dc] each paymentDates;

    // For spot starting swaps
    if[startYF <= 0;
        pvFixed:pvFixedLeg[curve;notional;fixedRate;payYFs];
        pvFloat:pvFloatingLeg[curve;notional;endYF];
        pv:$[isReceiver; pvFixed - pvFloat; pvFloat - pvFixed];
        :`pv`pvFixed`pvFloat`notional`fixedRate`startDate`endDate`frequency`isReceiver`asOfDate!(
            pv;pvFixed;pvFloat;notional;fixedRate;startDate;endDate;freq;isReceiver;asOf)];

    // For forward starting swaps
    pvFixed:pvFixedLegFwd[curve;notional;fixedRate;startYF;payYFs];
    pvFloat:pvFloatingLegFwd[curve;notional;startYF;endYF];
    pv:$[isReceiver; pvFixed - pvFloat; pvFloat - pvFixed];
    `pv`pvFixed`pvFloat`notional`fixedRate`startDate`endDate`frequency`isReceiver`asOfDate!(
        pv;pvFixed;pvFloat;notional;fixedRate;startDate;endDate;freq;isReceiver;asOf)}

// Par rate for swap with actual dates
parRateDated:{[curve;startDate;endDate;freq]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;

    startYF:$[startDate <= asOf; 0f; dcfDates[asOf;startDate;dc]];
    endYF:dcfDates[asOf;endDate;dc];

    paymentDates:genPayDates[startDate;endDate;freq];
    payYFs:dcfDates[asOf;;dc] each paymentDates;

    // Spot starting
    if[startYF <= 0;
        n:count payYFs;
        starts:(0f),payYFs til n-1;
        dcfs:payYFs - starts;
        dfs:interpDFs[curve;payYFs];
        dfT:last dfs;
        :(1 - dfT) % sum dcfs * dfs];

    // Forward starting
    n:count payYFs;
    starts:startYF,(payYFs til n-1);
    dcfs:payYFs - starts;
    dfs:interpDFs[curve;payYFs];
    dfStart:interpDF[curve;startYF];
    dfEnd:last dfs;
    (dfStart - dfEnd) % sum dcfs * dfs}

// Full analytics for dated swap
analyticsDated:{[curve;notional;fixedRate;startDate;endDate;freq;isReceiver]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;

    startYF:$[startDate <= asOf; 0f; dcfDates[asOf;startDate;dc]];
    endYF:dcfDates[asOf;endDate;dc];

    // Get basic pricing
    swap:priceSwapDated[curve;notional;fixedRate;startDate;endDate;freq;isReceiver];

    // Risk analytics (using year fractions)
    maturity:endYF;
    dv:$[startYF <= 0;
        dv01[curve;notional;fixedRate;maturity;freq;isReceiver];
        dv01[curve;notional;fixedRate;maturity;freq;isReceiver]];

    pv0:swap`pv;
    modDur:$[pv0=0; 0f; neg dv % pv0 * 0.0001];

    par:parRateDated[curve;startDate;endDate;freq];

    swap,`dv01`modDuration`parRate!(dv;modDur;par)}

// =============================================================================
// RISK ANALYTICS
// =============================================================================

// DV01: PV change for 1bp parallel shift in curve
dv01:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    curveUp:@[curve;`zeroRates;+;0.0001];
    curveUp:rebuildFromZeros curveUp;
    curveDn:@[curve;`zeroRates;-;0.0001];
    curveDn:rebuildFromZeros curveDn;
    pvUp:(priceSwap[curveUp;notional;fixedRate;maturity;freq;isReceiver])`pv;
    pvDn:(priceSwap[curveDn;notional;fixedRate;maturity;freq;isReceiver])`pv;
    (pvDn - pvUp) % 2}

// Key rate DV01s (bucketed sensitivities) - optimized version
// Pre-computes payment schedule and uses direct DF interpolation
keyRateDV01:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    tenors:curve`tenors;
    yfs:curve`yearFracs;
    zeros:curve`zeroRates;
    cfg:curve`config;
    im:cfg`interpMethod;
    method:$[im~`loglinear;`linear;im];
    comp:cfg`compounding;

    // Only compute sensitivities for tenors <= maturity (others are 0)
    relevantIdx:where yfs <= maturity + 0.01;
    if[0 = count relevantIdx; :tenors!count[tenors]#0f];

    // Pre-compute payment schedule once
    payDates:genPaySchedule[maturity;freq];
    n:count payDates;
    starts:(0f),payDates til n-1;
    dcfs:payDates - starts;
    payDatesF:`float$payDates;

    // Helper to compute PV from DFs at payment dates
    // pvFixed = N * r * sum(dcfs * dfs), pvFloat = N * (1 - dfT)
    // For receiver: pv = pvFixed - pvFloat
    calcPV:{[notional;fixedRate;dcfs;isReceiver;dfs]
        pvFixed:notional * fixedRate * sum dcfs * dfs;
        pvFloat:notional * 1 - last dfs;
        $[isReceiver; pvFixed - pvFloat; pvFloat - pvFixed]};

    // Compute DFs at payment dates for bumped zeros
    bumpedDFs:{[yfs;zeros;payDatesF;method;comp;idx;dr]
        zerosB:@[zeros;idx;+;dr];
        zeroInts:interp[yfs;zerosB;payDatesF;method];
        {[z;t;c] zero2df[z;t;c]}[;;comp]'[zeroInts;payDatesF]
    }[yfs;zeros;payDatesF;method;comp];

    // Calculate sensitivity for each relevant tenor
    sens:relevantIdx!{[calcPV;bumpedDFs;notional;fixedRate;dcfs;isReceiver;idx]
        dfsUp:bumpedDFs[idx;0.0001];
        dfsDn:bumpedDFs[idx;-0.0001];
        pvUp:calcPV[notional;fixedRate;dcfs;isReceiver;dfsUp];
        pvDn:calcPV[notional;fixedRate;dcfs;isReceiver;dfsDn];
        (pvDn - pvUp) % 2
    }[calcPV;bumpedDFs;notional;fixedRate;dcfs;isReceiver] each relevantIdx;

    // Build full result with 0s for tenors beyond maturity
    result:count[tenors]#0f;
    result[relevantIdx]:value sens;
    tenors!result}

// Modified duration
modDuration:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    dv:dv01[curve;notional;fixedRate;maturity;freq;isReceiver];
    $[pv0=0; 0f; neg dv % pv0 * 0.0001]}

// Macaulay duration (weighted average time of cash flows)
macDuration:{[curve;notional;fixedRate;maturity;freq]
    payDates:genPaySchedule[maturity;freq];
    n:count payDates;
    starts:(0f),payDates til n-1;
    dcfs:payDates - starts;
    dfs:interpDFs[curve;payDates];
    cfs:notional * fixedRate * dcfs;
    cfs[n-1]+:notional;
    pvCfs:cfs * dfs;
    totalPV:sum pvCfs;
    $[totalPV=0; 0f; sum[payDates * pvCfs] % totalPV]}

// Convexity
convexity:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    dr:0.0001;
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    curveUp:@[curve;`zeroRates;+;dr];
    curveUp:rebuildFromZeros curveUp;
    curveDn:@[curve;`zeroRates;-;dr];
    curveDn:rebuildFromZeros curveDn;
    pvUp:(priceSwap[curveUp;notional;fixedRate;maturity;freq;isReceiver])`pv;
    pvDn:(priceSwap[curveDn;notional;fixedRate;maturity;freq;isReceiver])`pv;
    d2PV:(pvUp - 2*pv0 + pvDn) % dr*dr;
    $[pv0=0; 0f; d2PV % pv0]}

// Gamma: change in DV01 for 1bp move (second-order sensitivity)
gamma:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    dr:0.0001;  // 1bp
    curveUp:rebuildFromZeros @[curve;`zeroRates;+;dr];
    curveDn:rebuildFromZeros @[curve;`zeroRates;-;dr];
    dv01Up:dv01[curveUp;notional;fixedRate;maturity;freq;isReceiver];
    dv01Dn:dv01[curveDn;notional;fixedRate;maturity;freq;isReceiver];
    (dv01Up - dv01Dn) % 2}

// Break-even rate: fixed rate that makes PV = 0 (same as par rate)
breakEvenRate:{[curve;maturity;freq] parRate[curve;maturity;freq]}

// Carry (annualized): net income from fixed vs floating
carryAnnualized:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    floatRate:fwdRate[curve;0;tenorToYF freq];
    netRate:$[isReceiver; fixedRate - floatRate; floatRate - fixedRate];
    notional * netRate}

// Curve risk: sensitivity to 2s10s steepening (10Y up, 2Y down)
curveRisk2s10s:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    // Steepener: 10Y +1bp, 2Y -1bp
    tenors:curve`tenors;
    zeros:curve`zeroRates;
    idx2Y:first where tenors in `2Y`24M;
    idx10Y:first where tenors = `10Y;
    if[null idx2Y; idx2Y:first where (curve`yearFracs) >= 2];
    if[null idx10Y; idx10Y:first where (curve`yearFracs) >= 10];
    // Bump 2Y down, 10Y up
    zerosSteep:@[zeros;idx2Y;-;0.0001];
    zerosSteep:@[zerosSteep;idx10Y;+;0.0001];
    curveSteep:rebuildFromZeros @[curve;`zeroRates;:;zerosSteep];
    pvSteep:(priceSwap[curveSteep;notional;fixedRate;maturity;freq;isReceiver])`pv;
    pvBase:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    pvSteep - pvBase}

// PV01: PV change for 1% (100bp) parallel shift
pv01:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    100 * dv01[curve;notional;fixedRate;maturity;freq;isReceiver]}

// Scenario analysis: P&L for various parallel shifts
scenarioPnL:{[curve;notional;fixedRate;maturity;freq;isReceiver;shocksBps]
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    calcPnL:{[curve;notional;fixedRate;maturity;freq;isReceiver;pv0;shock]
        bumped:rebuildFromZeros @[curve;`zeroRates;+;shock*0.0001];
        pv1:(priceSwap[bumped;notional;fixedRate;maturity;freq;isReceiver])`pv;
        pv1 - pv0
    }[curve;notional;fixedRate;maturity;freq;isReceiver;pv0];
    shocksBps!calcPnL each shocksBps}

// VaR (parametric): Value at Risk assuming normal distribution
// Uses DV01 and rate volatility (in bps, annualized)
// varConfig: `confidenceLevel`holdingDays`rateVolBps!(0.95;10;50)
//   confidenceLevel: 0.95 or 0.99
//   holdingDays: typically 1 or 10
//   rateVolBps: annualized rate volatility in bps (e.g., 50 = 0.5%)
varParametric:{[curve;notional;fixedRate;maturity;freq;isReceiver;varConfig]
    dv:dv01[curve;notional;fixedRate;maturity;freq;isReceiver];
    confidenceLevel:varConfig`confidenceLevel;
    holdingDays:varConfig`holdingDays;
    rateVolBps:varConfig`rateVolBps;
    // Z-scores: 1.645 for 95%, 2.326 for 99%
    zScore:$[confidenceLevel >= 0.99; 2.326; confidenceLevel >= 0.95; 1.645; 1.282];
    // Scale vol to holding period
    volScaled:rateVolBps * sqrt holdingDays % 252;
    // VaR = DV01 * vol * z-score
    abs dv * volScaled * zScore}

// Accrued interest: for swaps that have already started
// Returns accrued on fixed leg since last payment
accruedInterest:{[curve;notional;fixedRate;maturity;freq]
    dt:tenorToYF freq;
    // Time since last payment (assuming we're partway through a period)
    asOf:curve[`config]`asOfDate;
    // Simplified: assume we're at start of period (accrued = 0)
    // In practice, would need actual last payment date
    0f}

// Next payment date and days to payment
nextPayment:{[curve;maturity;freq]
    asOf:curve[`config]`asOfDate;
    dt:tenorToYF freq;
    // Generate schedule and find next date after asOf
    payDates:genPaySchedule[maturity;freq];
    payDatesDays:asOf + `int$(payDates * 365);
    nextIdx:first where payDatesDays > asOf;
    if[null nextIdx; :()!()];
    nextDate:payDatesDays nextIdx;
    `nextPayDate`daysToPayment`yearFrac!(nextDate; nextDate - asOf; payDates nextIdx)}

// Full analytics suite (enhanced)
analytics:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    swap:priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver];
    dv:dv01[curve;notional;fixedRate;maturity;freq;isReceiver];
    modDur:modDuration[curve;notional;fixedRate;maturity;freq;isReceiver];
    macDur:macDuration[curve;notional;fixedRate;maturity;freq];
    conv:convexity[curve;notional;fixedRate;maturity;freq;isReceiver];
    gam:gamma[curve;notional;fixedRate;maturity;freq;isReceiver];
    par:parRate[curve;maturity;freq];
    krDV01:keyRateDV01[curve;notional;fixedRate;maturity;freq;isReceiver];
    pv1:pv01[curve;notional;fixedRate;maturity;freq;isReceiver];
    cr2s10s:curveRisk2s10s[curve;notional;fixedRate;maturity;freq;isReceiver];
    carryAnn:carryAnnualized[curve;notional;fixedRate;maturity;freq;isReceiver];
    bpFromPar:10000 * fixedRate - par;
    scenarios:scenarioPnL[curve;notional;fixedRate;maturity;freq;isReceiver;-100 -50 -25 25 50 100];
    swap,`dv01`pv01`gamma`modDuration`macDuration`convexity`parRate`bpsFromPar`carryAnnualized`curveRisk2s10s`keyRateDV01`scenarios!(
        dv;pv1;gam;modDur;macDur;conv;par;bpFromPar;carryAnn;cr2s10s;krDV01;scenarios)}

// =============================================================================
// THETA AND CARRY/ROLL-DOWN
// =============================================================================

// Roll curve forward in time by dt years
// This shifts all year fractions down, removes expired tenors, and advances asOfDate
rollCurve:{[curve;dt]
    yf:curve`yearFracs;
    newYF:yf - dt;
    // Keep only tenors that haven't expired
    validIdx:where newYF > 0;
    if[0 = count validIdx; '"All tenors expired after roll"];
    newCurve:@[curve;`yearFracs;:;newYF validIdx];
    newCurve:@[newCurve;`tenors;:;(curve`tenors) validIdx];
    newCurve:@[newCurve;`zeroRates;:;(curve`zeroRates) validIdx];
    newCurve:@[newCurve;`parRates;:;(curve`parRates) validIdx];
    // Advance the asOfDate
    cfg:newCurve`config;
    oldAsOf:cfg`asOfDate;
    dc:cfg`dayCount;
    // Convert dt to days based on day count
    daysToAdd:$[dc~`ACT360; `int$dt * 360;
                dc~`ACT365; `int$dt * 365;
                dc~`30360;  `int$dt * 360;
                `int$dt * 365];
    newAsOf:oldAsOf + daysToAdd;
    newCfg:@[cfg;`asOfDate;:;newAsOf];
    newCurve:@[newCurve;`config;:;newCfg];
    // Rebuild DFs for new year fractions
    rebuildFromZeros newCurve}

// Theta: PV change from 1-day time decay (curve unchanged)
// Returns daily theta (negative for receiver swaps that are in-the-money)
theta:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    dt:1 % 365;  // 1 day
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    // Roll curve forward 1 day
    rolledCurve:rollCurve[curve;dt];
    // New maturity is shorter by 1 day
    newMat:maturity - dt;
    if[newMat <= 0; :neg pv0];  // Swap expires, theta is negative of remaining PV
    pv1:(priceSwap[rolledCurve;notional;fixedRate;newMat;freq;isReceiver])`pv;
    pv1 - pv0}

// Theta for forward starting swap
thetaForward:{[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver]
    dt:1 % 365;
    pv0:(priceForwardSwap[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver])`pv;
    rolledCurve:rollCurve[curve;dt];
    newFwdStart:fwdStart - dt;
    // If fwdStart becomes 0 or negative, it becomes a spot starting swap
    if[newFwdStart <= 0;
        newMat:tenor + newFwdStart;  // Adjust maturity
        pv1:(priceSwap[rolledCurve;notional;fixedRate;newMat;freq;isReceiver])`pv;
        :pv1 - pv0];
    pv1:(priceForwardSwap[rolledCurve;notional;fixedRate;newFwdStart;tenor;freq;isReceiver])`pv;
    pv1 - pv0}

// Carry: Income from holding the swap for a period
// For receiver: you receive fixed and pay floating
// Carry = (fixedRate - currentFloatRate) * notional * accrualFraction
carry:{[curve;notional;fixedRate;maturity;freq;isReceiver;holdingPeriod]
    // Current floating rate for the next period
    dt:tenorToYF freq;
    floatRate:fwdRate[curve;0;dt & maturity];
    // Accrual for holding period
    accrual:holdingPeriod;
    // Carry calculation
    netRate:$[isReceiver; fixedRate - floatRate; floatRate - fixedRate];
    notional * netRate * accrual}

// Roll-down: PV change from moving along the curve (assuming curve shape unchanged)
// This captures the benefit of "rolling down" a steep curve
rollDown:{[curve;notional;fixedRate;maturity;freq;isReceiver;holdingPeriod]
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    // Price the same swap but with shorter maturity (as if time passed)
    newMat:maturity - holdingPeriod;
    if[newMat <= 0; :neg pv0];
    // Use current curve (not rolled) - this isolates the roll-down effect
    pv1:(priceSwap[curve;notional;fixedRate;newMat;freq;isReceiver])`pv;
    pv1 - pv0}

// Total carry and roll-down analysis
// holdingPeriod: time horizon in years (e.g., 1/12 for 1 month, 1/365 for 1 day)
carryRollDown:{[curve;notional;fixedRate;maturity;freq;isReceiver;holdingPeriod]
    pv0:(priceSwap[curve;notional;fixedRate;maturity;freq;isReceiver])`pv;
    // Carry component
    carryPnL:carry[curve;notional;fixedRate;maturity;freq;isReceiver;holdingPeriod];
    // Roll-down component
    rollPnL:rollDown[curve;notional;fixedRate;maturity;freq;isReceiver;holdingPeriod];
    // Theta (pure time decay with curve roll)
    thetaPnL:holdingPeriod * 365 * theta[curve;notional;fixedRate;maturity;freq;isReceiver];
    // Current floating rate
    dt:tenorToYF freq;
    floatRate:fwdRate[curve;0;dt & maturity];
    `pv`carry`rollDown`theta`totalExpected`fixedRate`floatRate`netRate`holdingPeriod!(
        pv0;
        carryPnL;
        rollPnL;
        thetaPnL;
        carryPnL + rollPnL;
        fixedRate;
        floatRate;
        $[isReceiver;fixedRate - floatRate;floatRate - fixedRate];
        holdingPeriod)}

// Annualized carry and roll-down (scales to 1-year equivalent)
carryRollDownAnn:{[curve;notional;fixedRate;maturity;freq;isReceiver]
    carryRollDown[curve;notional;fixedRate;maturity;freq;isReceiver;1f]}

// Carry and roll-down for a specific horizon using tenor notation
// horizon: tenor symbol like `1D`1W`1M`3M`6M`1Y or year fraction
carryRoll:{[curve;notional;fixedRate;maturity;freq;isReceiver;horizon]
    // Convert horizon to year fraction if symbol
    hp:$[-11h = type horizon; tenorToYF horizon; horizon];
    crd:carryRollDown[curve;notional;fixedRate;maturity;freq;isReceiver;hp];
    // Add horizon info
    crd,`horizon`horizonYF!(horizon;hp)}

// Carry and roll-down for forward starting swaps
carryRollForward:{[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver;horizon]
    hp:$[-11h = type horizon; tenorToYF horizon; horizon];
    maturity:fwdStart + tenor;
    pv0:(priceForwardSwap[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver])`pv;
    // Carry: difference between fixed and implied forward rate
    dt:tenorToYF freq;
    floatRate:fwdRate[curve;fwdStart;fwdStart + dt & tenor];
    netRate:$[isReceiver; fixedRate - floatRate; floatRate - fixedRate];
    carryPnL:notional * netRate * hp;
    // Roll-down: price with reduced fwdStart (closer to spot)
    newFwdStart:0f | fwdStart - hp;
    newTenor:$[newFwdStart = 0; tenor + fwdStart - hp; tenor];
    pv1:$[newFwdStart = 0;
        (priceSwap[curve;notional;fixedRate;newTenor;freq;isReceiver])`pv;
        (priceForwardSwap[curve;notional;fixedRate;newFwdStart;tenor;freq;isReceiver])`pv];
    rollPnL:pv1 - pv0;
    // Theta
    thetaPnL:hp * 365 * thetaForward[curve;notional;fixedRate;fwdStart;tenor;freq;isReceiver];
    `pv`carry`rollDown`theta`totalExpected`fixedRate`floatRate`netRate`fwdStart`tenor`horizon`horizonYF!(
        pv0; carryPnL; rollPnL; thetaPnL; carryPnL + rollPnL;
        fixedRate; floatRate; netRate; fwdStart; tenor; horizon; hp)}

// Multi-horizon carry/roll-down table
carryRollTable:{[curve;notional;fixedRate;maturity;freq;isReceiver;horizons]
    horizons:(),horizons;  // Ensure list
    results:carryRoll[curve;notional;fixedRate;maturity;freq;isReceiver;] each horizons;
    // Build table
    ([] horizon:horizons;
        horizonYF:results@\:`horizonYF;
        carry:results@\:`carry;
        rollDown:results@\:`rollDown;
        theta:results@\:`theta;
        total:results@\:`totalExpected)}

// =============================================================================
// SWAP ANALYSIS - SIMPLIFIED INTERFACE
// =============================================================================

// Analyze carry/roll for a swap defined by start, end, rate
// start: 0 for spot, or tenor/year-frac for forward starting (e.g., `1Y or 1.0)
// end: swap end date as tenor or year-frac (e.g., `5Y or 5.0)
// rate: fixed rate
// horizon: holding period as tenor or year-frac (e.g., `3M or 0.25)
// isReceiver: 1b = receive fixed, 0b = pay fixed
swapCarryRoll:{[curve;notional;start;end;rate;freq;isReceiver;horizon]
    // Convert inputs to year fractions
    startYF:$[-11h = type start; tenorToYF start; `float$start];
    endYF:$[-11h = type end; tenorToYF end; `float$end];
    hp:$[-11h = type horizon; tenorToYF horizon; `float$horizon];

    // Determine if spot or forward starting
    isSpot:startYF = 0;
    tenor:endYF - startYF;

    // Get current PV and par rate
    pv0:$[isSpot;
        (priceSwap[curve;notional;rate;endYF;freq;isReceiver])`pv;
        (priceForwardSwap[curve;notional;rate;startYF;tenor;freq;isReceiver])`pv];

    parRt:$[isSpot;
        parRate[curve;endYF;freq];
        forwardParRate[curve;startYF;tenor;freq]];

    // Current floating rate (for the first period)
    dt:tenorToYF freq;
    t1:$[isSpot; 0f; startYF];
    t2:$[isSpot; dt & endYF; startYF + (dt & tenor)];
    floatRate:fwdRate[curve;t1;t2];

    // Net rate (what you earn/pay)
    netRate:$[isReceiver; rate - floatRate; floatRate - rate];

    // Carry: accrued income over horizon
    carryPnL:notional * netRate * hp;

    // Roll-down: PV change from moving along curve
    newStartYF:0f | startYF - hp;
    newEndYF:endYF - hp;

    // Handle case where swap hasn't started yet vs already started
    rollPV:$[newEndYF <= 0;
        neg pv0;  // Swap expires
        newStartYF = 0;
        (priceSwap[curve;notional;rate;newEndYF;freq;isReceiver])`pv;
        (priceForwardSwap[curve;notional;rate;newStartYF;tenor;freq;isReceiver])`pv];

    rollPnL:rollPV - pv0;

    // Theta
    dailyTheta:$[isSpot;
        theta[curve;notional;rate;endYF;freq;isReceiver];
        thetaForward[curve;notional;rate;startYF;tenor;freq;isReceiver]];
    thetaPnL:hp * 365 * dailyTheta;

    // DV01
    dv:$[isSpot;
        dv01[curve;notional;rate;endYF;freq;isReceiver];
        dv01[curve;notional;rate;endYF;freq;isReceiver]];  // Approximate for fwd

    // Build result
    `start`end`tenor`rate`parRate`pv`horizon`carry`rollDown`theta`totalExpected`floatRate`netRate`dv01`isReceiver!(
        start; end; tenor; rate; parRt; pv0; horizon; carryPnL; rollPnL; thetaPnL;
        carryPnL + rollPnL; floatRate; netRate; dv; isReceiver)}

// Shorthand for receiver swap
swapCR:{[curve;notional;start;end;rate;freq;horizon]
    swapCarryRoll[curve;notional;start;end;rate;freq;1b;horizon]}

// Show formatted carry/roll analysis
showCarryRoll:{[result]
    -1 "";
    -1 "=== SWAP CARRY/ROLL ANALYSIS ===";
    -1 "";
    -1 "Swap:     ",string[result`start]," -> ",string[result`end]," (",string[result`tenor],"Y tenor)";
    -1 "Type:     ",$[result`isReceiver;"Receiver";"Payer"];
    -1 "Rate:     ",string[100*result`rate],"%";
    -1 "Par Rate: ",string[100*result`parRate],"%";
    -1 "PV:       $",string[`int$result`pv];
    -1 "DV01:     $",string[`int$result`dv01];
    -1 "";
    -1 "Horizon:  ",string result`horizon;
    -1 "-----------------------------------";
    -1 "Carry:      $",string[`int$result`carry];
    -1 "Roll-Down:  $",string[`int$result`rollDown];
    -1 "Theta:      $",string[`int$result`theta];
    -1 "-----------------------------------";
    -1 "Total:      $",string[`int$result`totalExpected];
    -1 "";
    -1 "Float Rate: ",(string 100*result`floatRate),"%";
    nr:result`netRate;
    ir:result`isReceiver;
    isPositive:$[ir; nr > 0; nr < 0];
    carrySign:$[isPositive;"positive";"negative"];
    -1 "Net Rate:   ",(string 100*nr),"% (",carrySign," carry)";
    -1 "";}

// =============================================================================
// DATED CARRY/ROLL ANALYSIS
// =============================================================================

// Carry/roll analysis for swap with actual dates
// startDate: swap start date
// endDate: swap end date
// rate: fixed rate
// horizon: holding period as tenor (e.g., `3M) or year fraction
swapCarryRollDated:{[curve;notional;startDate;endDate;rate;freq;isReceiver;horizon]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    hp:$[-11h = type horizon; tenorToYF horizon; `float$horizon];

    // Convert dates to year fractions from asOfDate
    startYF:$[startDate <= asOf; 0f; dcfDates[asOf;startDate;dc]];
    endYF:dcfDates[asOf;endDate;dc];

    // Determine if spot or forward starting
    isSpot:startYF <= 0;
    tenor:endYF - startYF;

    // Get current PV and par rate
    pv0:(priceSwapDated[curve;notional;rate;startDate;endDate;freq;isReceiver])`pv;
    parRt:parRateDated[curve;startDate;endDate;freq];

    // Current floating rate (for the first period)
    dt:tenorToYF freq;
    t1:$[isSpot; 0f; startYF];
    t2:$[isSpot; dt & endYF; startYF + (dt & tenor)];
    floatRate:fwdRate[curve;t1;t2];

    // Net rate (what you earn/pay)
    netRate:$[isReceiver; rate - floatRate; floatRate - rate];

    // Carry: accrued income over horizon
    carryPnL:notional * netRate * hp;

    // Calculate horizon end date
    horizonDate:$[-11h = type horizon;
        addTenor[asOf;horizon];
        asOf + `int$hp * 365];

    // New swap dates after horizon
    newStartDate:$[startDate <= horizonDate; horizonDate; startDate];
    newEndYF:endYF - hp;

    // Roll-down: PV change from moving along curve
    rollPV:$[newEndYF <= 0;
        neg pv0;  // Swap expires
        [rolledCurve:rollCurve[curve;hp];
         newStart:$[isSpot; rolledCurve[`config]`asOfDate; newStartDate];
         newEnd:yfToDate[rolledCurve;newEndYF - (hp * (not isSpot))];
         (priceSwapDated[rolledCurve;notional;rate;newStart;newEnd;freq;isReceiver])`pv]];

    rollPnL:rollPV - pv0;

    // Theta
    dailyTheta:$[isSpot;
        theta[curve;notional;rate;endYF;freq;isReceiver];
        thetaForward[curve;notional;rate;startYF;tenor;freq;isReceiver]];
    thetaPnL:hp * 365 * dailyTheta;

    // DV01
    dv:dv01[curve;notional;rate;endYF;freq;isReceiver];

    // Build result
    `startDate`endDate`tenor`rate`parRate`pv`asOfDate`horizon`horizonDate`carry`rollDown`theta`totalExpected`floatRate`netRate`dv01`isReceiver!(
        startDate; endDate; tenor; rate; parRt; pv0; asOf; horizon; horizonDate; carryPnL; rollPnL; thetaPnL;
        carryPnL + rollPnL; floatRate; netRate; dv; isReceiver)}

// Shorthand for dated receiver swap
swapCRDated:{[curve;notional;startDate;endDate;rate;freq;horizon]
    swapCarryRollDated[curve;notional;startDate;endDate;rate;freq;1b;horizon]}

// Show formatted carry/roll analysis for dated swap
showCarryRollDated:{[result]
    -1 "";
    -1 "=== DATED SWAP CARRY/ROLL ANALYSIS ===";
    -1 "";
    -1 "As Of Date: ",string result`asOfDate;
    -1 "Start Date: ",string result`startDate;
    -1 "End Date:   ",string result`endDate;
    -1 "Tenor:      ",(string result`tenor)," years";
    -1 "Type:       ",$[result`isReceiver;"Receiver";"Payer"];
    -1 "Rate:       ",(string 100*result`rate),"%";
    -1 "Par Rate:   ",(string 100*result`parRate),"%";
    -1 "PV:         $",string[`int$result`pv];
    -1 "DV01:       $",string[`int$result`dv01];
    -1 "";
    -1 "Horizon:      ",string result`horizon;
    -1 "Horizon Date: ",string result`horizonDate;
    -1 "-----------------------------------";
    -1 "Carry:      $",string[`int$result`carry];
    -1 "Roll-Down:  $",string[`int$result`rollDown];
    -1 "Theta:      $",string[`int$result`theta];
    -1 "-----------------------------------";
    -1 "Total:      $",string[`int$result`totalExpected];
    -1 "";
    -1 "Float Rate: ",(string 100*result`floatRate),"%";
    nr:result`netRate;
    ir:result`isReceiver;
    isPositive:$[ir; nr > 0; nr < 0];
    carrySign:$[isPositive;"positive";"negative"];
    -1 "Net Rate:   ",(string 100*nr),"% (",carrySign," carry)";
    -1 "";}

// =============================================================================
// DISPLAY UTILITIES
// =============================================================================

// Display curve summary
showCurve:{[curve]
    -1 "";
    -1 "=== YIELD CURVE ===";
    -1 "As Of Date:       ",string curve[`config]`asOfDate;
    -1 "Tenors:           ",-3!curve`tenors;
    -1 "Year Fracs:       ",-3!curve`yearFracs;
    -1 "Discount Factors: ",-3!curve`discountFactors;
    -1 "Zero Rates (%):   ",-3!100*curve`zeroRates;
    -1 "Par Rates (%):    ",-3!100*curve`parRates;
    -1 "Day Count:        ",string curve[`config]`dayCount;
    -1 "Frequency:        ",string curve[`config]`frequency;
    -1 "Compounding:      ",string curve[`config]`compounding;
    -1 "";}

// Display swap summary
showSwap:{[result]
    -1 "";
    -1 "=== SWAP VALUATION ===";
    -1 "PV:           $",string result`pv;
    -1 "Fixed Leg PV: $",string result`pvFixed;
    -1 "Float Leg PV: $",string result`pvFloat;
    -1 "Notional:     $",string result`notional;
    -1 "Fixed Rate:   ",string 100*result`fixedRate;"%";
    -1 "Maturity:     ",string result`maturity;" years";
    -1 "Type:         ",$[result`isReceiver;"Receiver (receive fixed)";"Payer (pay fixed)"];
    -1 "";}

// =============================================================================
// EXAMPLE AND HELP
// =============================================================================

// Example demonstrating all swap functions
example:{[]
    -1 "";
    -1 "=== SWAP PRICING LIBRARY EXAMPLE ===";
    -1 "";
    -1 "1. BUILDING YIELD CURVE FROM MARKET DATA";
    -1 "   Market par swap rates:";
    tenors:`3M`6M`1Y`2Y`3Y`5Y`7Y`10Y;
    rates:0.045 0.0475 0.05 0.052 0.053 0.054 0.0545 0.055;
    -1 "   Tenors: ",-3!tenors;
    -1 "   Rates:  ",-3!rates;
    -1 "";
    curve:buildCurve[tenors;rates;()!()];
    showCurve curve;
    -1 "2. PRICING A 5Y RECEIVER SWAP";
    -1 "   Notional: $10,000,000";
    -1 "   Fixed Rate: 5.4%";
    -1 "   Maturity: 5 years";
    -1 "   Frequency: Semi-annual";
    -1 "";
    notional:10000000f;
    fixedRate:0.054;
    maturity:5f;
    freq:`6M;
    result:priceSwap[curve;notional;fixedRate;maturity;freq;1b];
    showSwap result;
    -1 "3. RISK ANALYTICS";
    full:analytics[curve;notional;fixedRate;maturity;freq;1b];
    -1 "   DV01:              $",string full`dv01;
    -1 "   Modified Duration: ",string full`modDuration;
    -1 "   Macaulay Duration: ",string full`macDuration;
    -1 "   Convexity:         ",string full`convexity;
    -1 "   Par Rate:          ",string 100*full`parRate;"%";
    -1 "";
    -1 "   Key Rate DV01s:";
    krDV01:full`keyRateDV01;
    {-1 "     ",string[x],": $",string y}'[key krDV01;value krDV01];
    -1 "";
    -1 "4. FORWARD RATES";
    -1 "   1Y1Y Forward: ",string 100*fwdRate[curve;1f;2f];"%";
    -1 "   2Y2Y Forward: ",string 100*fwdRate[curve;2f;4f];"%";
    -1 "   5Y5Y Forward: ",string 100*fwdRate[curve;5f;10f];"%";
    -1 "";
    -1 "5. VERIFY BOOTSTRAP (par rates should match inputs)";
    -1 "   Computed par rates:";
    computedPars:parRate[curve;;`6M] each curve`yearFracs;
    {-1 "     ",string[x],": ",string 100*y;"%"}'[curve`tenors;computedPars];
    -1 "";
    -1 "=== END EXAMPLE ===";
    -1 "";
    full}

// Usage reference with actual function calls
usage:{[]
    -1 "";
    -1 "=== .swaps USAGE REFERENCE ===";
    -1 "";
    -1 "// 1. BUILD A CURVE";
    -1 "// -----------------";
    -1 "tenors:`3M`6M`1Y`2Y`3Y`5Y`7Y`10Y";
    -1 "rates:0.05 0.051 0.052 0.053 0.054 0.055 0.056 0.057";
    -1 "";
    -1 "curve:.swaps.buildCurve[tenors;rates;()!()]                              / defaults (asOfDate=today)";
    -1 "curve:.swaps.buildCurve[tenors;rates;enlist[`asOfDate]!enlist 2025.01.15] / specific valuation date";
    -1 "curve:.swaps.buildCurve[tenors;rates;enlist[`frequency]!enlist `3M]      / quarterly";
    -1 "curve:.swaps.buildCurve[0.25 0.5 1 2 3 5 7 10f;rates;()!()]              / year fractions";
    -1 "";
    -1 "// SMOOTHING SPLINE ON PAR RATES";
    -1 "// Smooth the input par rates BEFORE bootstrapping (removes noise, produces smooth forwards)";
    -1 "curve:.swaps.buildCurve[tenors;rates;enlist[`smoothPar]!enlist 0.001]    / smooth par rates (p=0.001)";
    -1 "curve:.swaps.buildCurve[tenors;rates;`smoothPar`frequency!(0.001;`3M)]   / smooth + quarterly";
    -1 "";
    -1 "// FINE GRID CURVE (interpolate par rates to finer grid, then bootstrap)";
    -1 "/ Produces more curve points for smoother forward rate profile";
    -1 "yf:0.25 0.5 1 2 3 5 7 10f";
    -1 "curve:.swaps.buildCurveFineGrid[yf;rates;0.25;0.001;()!()]               / 0.25Y grid, smooth p=0.001";
    -1 "";
    -1 "// 2. QUERY THE CURVE";
    -1 "// -------------------";
    -1 ".swaps.interpZero[curve;4.5]           / zero rate at 4.5Y";
    -1 ".swaps.interpDF[curve;4.5]             / discount factor at 4.5Y";
    -1 ".swaps.fwdRate[curve;2;5]              / forward rate from 2Y to 5Y (2Y3Y fwd)";
    -1 ".swaps.fwdRate[curve;5;10]             / 5Y5Y forward";
    -1 ".swaps.parRate[curve;5;`3M]            / 5Y par swap rate (quarterly)";
    -1 ".swaps.parRate[curve;10;`6M]           / 10Y par swap rate (semi-annual)";
    -1 "";
    -1 "// 3. PRICE A SWAP";
    -1 "// ----------------";
    -1 "/ priceSwap[curve; notional; fixedRate; maturity; freq; isReceiver]";
    -1 "/ isReceiver: 1b = receive fixed, 0b = pay fixed";
    -1 "";
    -1 ".swaps.priceSwap[curve; 10000000; 0.055; 5; `3M; 1b]    / 5Y receiver @ 5.5%";
    -1 ".swaps.priceSwap[curve; 10000000; 0.055; 5; `3M; 0b]    / 5Y payer @ 5.5%";
    -1 "";
    -1 "/ ATM swap (PV = 0)";
    -1 "par:.swaps.parRate[curve;5;`3M]";
    -1 ".swaps.priceSwap[curve; 10000000; par; 5; `3M; 1b]";
    -1 "";
    -1 "// 4. FORWARD STARTING SWAPS";
    -1 "// --------------------------";
    -1 "/ priceForwardSwap[curve; notional; fixedRate; fwdStart; tenor; freq; isReceiver]";
    -1 ".swaps.priceForwardSwap[curve; 10000000; 0.055; 1; 5; `3M; 1b]  / 1Y fwd 5Y receiver";
    -1 ".swaps.priceForwardSwap[curve; 10000000; 0.056; 2; 5; `3M; 1b]  / 2Y fwd 5Y receiver";
    -1 ".swaps.forwardParRate[curve; 1; 5; `3M]                         / 1x6 par rate";
    -1 ".swaps.forwardParRate[curve; 2; 5; `3M]                         / 2x7 par rate";
    -1 "";
    -1 "// 5. RISK ANALYTICS";
    -1 "// ------------------";
    -1 ".swaps.dv01[curve;10000000;0.055;5;`3M;1b]              / DV01 (PV change for 1bp)";
    -1 ".swaps.keyRateDV01[curve;10000000;0.055;5;`3M;1b]       / bucketed sensitivities";
    -1 ".swaps.macDuration[curve;10000000;0.055;5;`3M]          / Macaulay duration";
    -1 ".swaps.modDuration[curve;10000000;0.055;5;`3M;1b]       / Modified duration";
    -1 ".swaps.convexity[curve;10000000;0.055;5;`3M;1b]         / Convexity";
    -1 ".swaps.analytics[curve;10000000;0.055;5;`3M;1b]         / full analytics suite";
    -1 "";
    -1 "// 6. THETA AND CARRY/ROLL-DOWN";
    -1 "// -----------------------------";
    -1 ".swaps.theta[curve;10000000;0.055;5;`3M;1b]             / daily theta (time decay)";
    -1 ".swaps.thetaForward[curve;10000000;0.055;1;5;`3M;1b]    / theta for fwd swap";
    -1 "";
    -1 "/ Carry/roll with tenor horizon (e.g., `1M`3M`6M`1Y)";
    -1 ".swaps.carryRoll[curve;10000000;0.055;5;`3M;1b;`3M]     / 3-month horizon";
    -1 ".swaps.carryRoll[curve;10000000;0.055;5;`3M;1b;`1Y]     / 1-year horizon";
    -1 ".swaps.carryRollForward[curve;10000000;0.057;1;5;`3M;1b;`3M]  / fwd swap, 3M horizon";
    -1 "";
    -1 "/ Multi-horizon table";
    -1 ".swaps.carryRollTable[curve;10000000;0.055;5;`3M;1b;`1M`3M`6M`1Y]";
    -1 "";
    -1 "/ Lower-level functions";
    -1 ".swaps.carry[curve;10000000;0.055;5;`3M;1b;0.25]        / carry (year frac)";
    -1 ".swaps.rollDown[curve;10000000;0.055;5;`3M;1b;0.25]     / roll-down (year frac)";
    -1 "";
    -1 "// 7. CURVE MANIPULATION";
    -1 "// ----------------------";
    -1 "bumpedCurve:@[curve;`zeroRates;+;0.001]                 / +10bp parallel bump";
    -1 "bumpedCurve:.swaps.rebuildFromZeros bumpedCurve         / rebuild DFs from zeros";
    -1 ".swaps.rollCurve[curve;1%12]                            / roll curve forward 1 month";
    -1 "";
    -1 "// 8. UTILITIES";
    -1 "// -------------";
    -1 ".swaps.tenorToYF[`3M]                    / 0.25";
    -1 ".swaps.tenorToYF[`5Y]                    / 5";
    -1 ".swaps.tenorToYF[`18M]                   / 1.5";
    -1 ".swaps.tenorsToYF[`1Y`2Y`5Y]             / 1 2 5f";
    -1 ".swaps.genPaySchedule[2;`3M]             / 0.25 0.5 0.75 1 1.25 1.5 1.75 2";
    -1 ".swaps.genPaySchedule[5;`6M]             / 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5";
    -1 ".swaps.zero2df[0.05;5;`continuous]       / zero rate -> DF";
    -1 ".swaps.df2zero[0.78;5;`continuous]       / DF -> zero rate";
    -1 "";
    -1 "// 9. DISPLAY";
    -1 "// ----------";
    -1 ".swaps.showCurve[curve]                  / print curve summary (includes asOfDate)";
    -1 ".swaps.showSwap[swap]                    / print swap valuation";
    -1 ".swaps.help[]                            / function reference";
    -1 ".swaps.example[]                         / run full example";
    -1 "";
    -1 "// 10. DATED FUNCTIONS (USING ACTUAL DATES)";
    -1 "// -----------------------------------------";
    -1 "/ All dated functions use curve's asOfDate as reference point";
    -1 "";
    -1 "/ Date utilities";
    -1 ".swaps.dateToYF[curve; 2025.06.15]       / date to year fraction from asOfDate";
    -1 ".swaps.datesToYF[curve; 2025.06.15 2026.01.15]  / multiple dates";
    -1 ".swaps.yfToDate[curve; 0.5]              / year fraction to date";
    -1 ".swaps.addTenor[2025.01.15; `3M]         / add tenor to date -> 2025.04.15";
    -1 ".swaps.addTenor[2025.01.15; `1Y]         / add tenor to date -> 2026.01.15";
    -1 ".swaps.genPayDates[2025.01.15; 2030.01.15; `3M]  / payment dates for swap";
    -1 "";
    -1 "/ Dated swap pricing";
    -1 "asOf:curve[`config]`asOfDate";
    -1 "endDate:asOf + 365*5                     / 5Y from asOf";
    -1 ".swaps.priceSwapDated[curve; 10000000; 0.055; asOf; endDate; `3M; 1b]  / spot receiver";
    -1 ".swaps.parRateDated[curve; asOf; endDate; `3M]                         / par rate";
    -1 ".swaps.analyticsDated[curve; 10000000; 0.055; asOf; endDate; `3M; 1b]  / full analytics";
    -1 "";
    -1 "/ Forward starting (startDate > asOfDate)";
    -1 "startDate:asOf + 365                     / 1Y forward";
    -1 "endDate:asOf + 365*6                     / ends at 6Y";
    -1 ".swaps.priceSwapDated[curve; 10000000; 0.057; startDate; endDate; `3M; 1b]";
    -1 "";
    -1 "/ Dated carry/roll analysis";
    -1 ".swaps.swapCarryRollDated[curve;10000000;asOf;asOf+365*5;0.055;`3M;1b;`3M]";
    -1 ".swaps.swapCRDated[curve;10000000;asOf;asOf+365*5;0.055;`3M;`3M]  / shorthand";
    -1 ".swaps.showCarryRollDated[result]        / formatted output";
    -1 "";
    -1 "// CONFIG OPTIONS";
    -1 "// ---------------";
    -1 "/ `asOfDate     - Curve valuation date (default: .z.d = today)";
    -1 "/ `interpMethod - `linear`loglinear`cubic`smooth (default: loglinear)";
    -1 "/ `frequency    - `3M`6M (default: 6M)";
    -1 "/ `dayCount     - `ACT360`ACT365`30360`ACTACT (default: ACT360)";
    -1 "/ `compounding  - `continuous`annual`semiannual (default: continuous)";
    -1 "";
    }

// Help function
help:{[]
    -1 "";
    -1 "=== .swaps FUNCTIONS ===";
    -1 "";
    -1 "CURVE BUILDING:";
    -1 "  buildCurve[tenors;rates;config]    - Build curve from par swap rates";
    -1 "    config`smoothPar: smooth par rates before bootstrap (e.g., 0.001)";
    -1 "  buildCurveFineGrid[yf;rates;step;p;cfg] - Smooth par rates to fine grid";
    -1 "  rebuildFromZeros[curve]            - Rebuild DFs from zero rates";
    -1 "  rollCurve[curve;dt]                - Roll curve forward by dt years";
    -1 "  interpDF[curve;t]                  - Interpolate discount factor";
    -1 "  interpZero[curve;t]                - Interpolate zero rate";
    -1 "";
    -1 "INTERPOLATION:";
    -1 "  interp[x;y;xi;method]              - Interpolate (linear/loglinear/cubic/smooth)";
    -1 "  cubicSplineCoeffs[x;y]             - Build cubic spline";
    -1 "  smoothingSpline[x;y;p]             - Build smoothing spline (p=smoothness)";
    -1 "  smoothParRates[yf;rates;p]         - Smooth par rates at same points";
    -1 "  interpParRatesSmooth[yf;rates;yfNew;p] - Interpolate par rates to new grid";
    -1 "";
    -1 "FORWARD RATES:";
    -1 "  fwdRate[curve;t1;t2]               - Forward rate from t1 to t2";
    -1 "  fwdDF[curve;t1;t2]                 - Forward discount factor";
    -1 "  genForwardRates[curve;payDates]    - All forwards for schedule";
    -1 "";
    -1 "SWAP PRICING:";
    -1 "  priceSwap[curve;N;r;mat;freq;rcv]        - Price vanilla IRS";
    -1 "  priceForwardSwap[curve;N;r;fwd;tnr;freq;rcv] - Price forward starting swap";
    -1 "  pvFixedLeg[curve;N;rate;dates]           - Fixed leg PV";
    -1 "  pvFloatingLeg[curve;N;mat]               - Floating leg PV";
    -1 "  parRate[curve;mat;freq]                  - Par swap rate";
    -1 "  forwardParRate[curve;fwd;tnr;freq]       - Forward starting par rate";
    -1 "";
    -1 "RISK ANALYTICS:";
    -1 "  dv01[curve;N;r;mat;freq;rcv]         - Dollar value of 01";
    -1 "  keyRateDV01[curve;N;r;mat;freq;rcv]  - Bucketed DV01s";
    -1 "  modDuration[curve;N;r;mat;freq;rcv]  - Modified duration";
    -1 "  macDuration[curve;N;r;mat;freq]      - Macaulay duration";
    -1 "  convexity[curve;N;r;mat;freq;rcv]    - Convexity";
    -1 "  analytics[curve;N;r;mat;freq;rcv]    - Full analytics suite";
    -1 "";
    -1 "THETA & CARRY:";
    -1 "  theta[curve;N;r;mat;freq;rcv]        - Daily theta (time decay)";
    -1 "  thetaForward[curve;N;r;fwd;tnr;freq;rcv] - Theta for fwd swap";
    -1 "  carryRoll[curve;N;r;mat;freq;rcv;horizon] - Carry/roll for horizon (`1M`3M etc)";
    -1 "  carryRollForward[curve;N;r;fwd;tnr;freq;rcv;horizon] - For fwd swaps";
    -1 "  carryRollTable[curve;N;r;mat;freq;rcv;horizons] - Multi-horizon table";
    -1 "  carry[curve;N;r;mat;freq;rcv;period] - Carry component (year frac)";
    -1 "  rollDown[curve;N;r;mat;freq;rcv;period] - Roll-down component (year frac)";
    -1 "";
    -1 "UTILITIES:";
    -1 "  tenorToYF[tenor]                   - Parse tenor to year fraction";
    -1 "  tenorsToYF[tenors]                 - Parse list of tenors";
    -1 "  genPaySchedule[mat;freq]           - Generate payment schedule";
    -1 "  zero2df[rate;t;comp]               - Zero to discount factor";
    -1 "  df2zero[df;t;comp]                 - Discount factor to zero";
    -1 "  showCurve[curve]                   - Display curve summary";
    -1 "  showSwap[result]                   - Display swap result";
    -1 "";
    -1 "DATE UTILITIES:";
    -1 "  dateToYF[curve;date]               - Date to year frac from asOfDate";
    -1 "  datesToYF[curve;dates]             - Multiple dates to year fracs";
    -1 "  yfToDate[curve;yf]                 - Year frac to date";
    -1 "  addTenor[date;tenor]               - Add tenor to date";
    -1 "  genPayDates[start;end;freq]        - Generate payment dates";
    -1 "  dcfDates[d1;d2;conv]               - Day count fraction between dates";
    -1 "";
    -1 "DATED SWAP PRICING:";
    -1 "  priceSwapDated[curve;N;r;start;end;freq;rcv]    - Price swap with dates";
    -1 "  parRateDated[curve;start;end;freq]              - Par rate with dates";
    -1 "  analyticsDated[curve;N;r;start;end;freq;rcv]    - Analytics with dates";
    -1 "  swapCarryRollDated[curve;N;start;end;r;freq;rcv;horizon] - Carry/roll";
    -1 "  swapCRDated[curve;N;start;end;r;freq;horizon]   - Shorthand (receiver)";
    -1 "  showCarryRollDated[result]                      - Display dated result";
    -1 "";
    -1 "CONFIG OPTIONS (pass in config dict):";
    -1 "  `asOfDate     - Curve valuation date (default .z.d = today)";
    -1 "  `dayCount     - `ACT360`ACT365`30360`ACTACT (default ACT360)";
    -1 "  `frequency    - `3M`6M (default 6M)";
    -1 "  `compounding  - `continuous`annual`semiannual (default continuous)";
    -1 "  `interpMethod - `linear`loglinear`cubic`smooth (default loglinear)";
    -1 "  `smoothPar    - Smooth par rates BEFORE bootstrap (e.g., 0.001 = light smoothing)";
    -1 "                  Use this for noisy market data to get smooth forward rates";
    -1 "";
    -1 "Run .swaps.usage[] for quick reference or .swaps.example[] for demo.";
    -1 "";}

\d .

// Load message
-1 "Loaded .swaps namespace v0.1.0";
-1 "Functions: buildCurve, priceSwap, analytics, dv01, duration, convexity";
-1 "Run .swaps.usage[] for quick reference, .swaps.help[] for functions, .swaps.example[] for demo";
