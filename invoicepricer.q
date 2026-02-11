// Invoice Spread Library
// Namespace: .invoicepricer
// Version: 0.1.0
// Dependencies: swaps.q, ctd.q
//
// Purpose: Calculate and analyze invoice spreads - the difference between
// swap rates and Treasury futures implied yields. A key relative value metric
// for fixed income trading.
//
// Key Concepts:
//   Invoice Spread = Swap Rate - Invoice Yield
//   Invoice Price = Futures Price * CF + Accrued Interest (at delivery)
//   Invoice Yield = YTM implied by the invoice price
//   Positive spread = swaps trading wide of Treasuries
//
// Date Convention:
//   Swap starts at futures delivery date
//   Swap matures at CTD bond maturity date

\d .invoicepricer

version:"0.1.0"

// =============================================================================
// CONFIGURATION & CONSTANTS
// =============================================================================

// Default payment frequency for swaps
defaultFreq:`1Y

// =============================================================================
// CONVENTION CONFIGURATION
// =============================================================================
// Swap conventions (OIS style):
//   - Fixed leg: Annual (1Y), ACT/360
//   - Floating leg: Daily SOFR compounded, ACT/360
//
// Bond yield conventions (US Treasury Street):
//   - Compounding: Semi-annual
//   - Discounting: Period count (time measured in semi-annual periods)
//   - Accrued interest: ACT/ACT (actual days / actual days in period)

swapConv:`freq`dayCount`compounding!(`1Y;`ACT360;`annual);
bondConv:`compounding`discounting`aiDayCount!(`semiannual;`periodCount;`ACTACT);

// =============================================================================
// PARALLELIZATION
// =============================================================================
// Parallelization: use `peach` if secondary threads available, else `each`
// Set .invoicepricer.useParallel:1b to force parallel, 0b to force sequential
// By default, auto-detect based on \s setting (returns count of secondary threads)
useParallel:0 < system "s"

// Parallel-aware each: uses peach when enabled and beneficial
// Only parallelize when count >= threshold (overhead otherwise)
pmap:{[f;threshold;xs]
    $[useParallel and (count xs) >= threshold;
        f peach xs;
        f each xs]}

// =============================================================================
// CORE SPREAD CALCULATIONS
// =============================================================================

// Invoice price: Futures × CF + AI at delivery
// The price the short delivers the bond for
invoicePrice:{[deliveryDate;futuresPrice;ctdBond]
    // Invoice price = futures * CF + AI
    ai:.ctd.accruedInterest[deliveryDate;ctdBond`maturity;ctdBond`coupon;`SA;100f];
    (futuresPrice * ctdBond`cf) + ai}

// Invoice yield: YTM implied by invoice price
// What yield does the long earn if they receive the bond at the invoice price?
// Uses US Treasury Street convention:
//   - Semi-annual compounding
//   - Period counting (time in semi-annual periods, not calendar days)
//   - ACT/ACT for accrued interest
invoiceYield:{[deliveryDate;futuresPrice;ctdBond]
    // Clean price = futures * CF
    cleanPrice:futuresPrice * ctdBond`cf;
    // Use Street convention solver
    invoiceYieldStreet[deliveryDate;ctdBond`maturity;ctdBond`coupon;cleanPrice]}

// Street convention YTM solver (US Treasury standard)
// Uses period counting: DF = (1 + y/2)^(-n) where n is number of semi-annual periods
invoiceYieldStreet:{[settleDate;maturityDate;coupon;cleanPrice]
    // Generate semi-annual coupon dates from maturity date backwards
    // Coupon months are maturity month and maturity month +/- 6
    matMonth:`mm$maturityDate;
    matDay:`dd$maturityDate;

    // Coupon months: maturity month and 6 months offset
    cpnMonth1:matMonth;
    cpnMonth2:$[matMonth > 6; matMonth - 6; matMonth + 6];

    // Generate coupon date with day adjustment for month length
    genCpnDate:{[md;y;m]
        // Days in each month
        dim:31 28 31 30 31 30 31 31 30 31 30 31;
        // Adjust for leap year February
        dim:@[dim;1;:;$[(0 = y mod 4) and (0 <> y mod 100) or (0 = y mod 400); 29; 28]];
        d:min (md; dim m-1);
        "D"$string[y],".",("0"^-2$string m),".",("0"^-2$string d)
      }[matDay];

    // Generate all coupon dates from settle year-1 to maturity
    matYear:`year$maturityDate;
    settleYear:`year$settleDate;
    years:(settleYear - 1) + til 2 + matYear - settleYear;

    allCpnDates:raze {[f;m1;m2;y] (f[y;m1]; f[y;m2])}[genCpnDate;cpnMonth1;cpnMonth2;] each years;
    allCpnDates:asc distinct allCpnDates where not null allCpnDates;

    // Filter: after settlement and <= maturity
    cpnDates:allCpnDates where (allCpnDates > settleDate) & allCpnDates <= maturityDate;

    // Handle edge case: no coupons remaining (bond very close to maturity)
    if[0 = count cpnDates;
        // Just maturity payment
        nDays:maturityDate - settleDate;
        y:coupon;  // Simple approximation for very short bonds
        :(y)];

    // Previous coupon (for accrued interest)
    prevCpnDates:allCpnDates where allCpnDates <= settleDate;
    prevCoupon:$[count prevCpnDates; last prevCpnDates; settleDate - 182];
    nextCoupon:first cpnDates;

    // Accrued interest (ACT/ACT)
    daysAccrued:settleDate - prevCoupon;
    daysPeriod:nextCoupon - prevCoupon;
    semiCoupon:coupon * 50f;
    ai:semiCoupon * daysAccrued % daysPeriod;

    // Cash flows
    nCoupons:count cpnDates;
    cfs:((nCoupons - 1)#semiCoupon),semiCoupon + 100f;

    // Time to first coupon as fraction of period
    t1:(nextCoupon - settleDate) % daysPeriod;
    periods:t1 + til nCoupons;

    // Newton-Raphson to solve for yield
    y:0.04;  // Initial guess
    do[25;
        dfs:xexp[1 + y % 2; neg periods];
        dirty:sum cfs * dfs;
        price:dirty - ai;
        // Derivative: d(price)/dy
        dpdy:sum cfs * dfs * neg periods % (2 * (1 + y % 2));
        y:y - (price - cleanPrice) % dpdy
    ];
    y}

// Legacy invoice yield using ctd.q (for comparison)
invoiceYieldLegacy:{[deliveryDate;futuresPrice;ctdBond]
    cleanPrice:futuresPrice * ctdBond`cf;
    .ctd.yieldToMaturity[deliveryDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;cleanPrice]}

// Swap rate: Forward par rate from delivery date to CTD maturity
// Uses OIS convention: 1Y fixed, ACT/360, daily SOFR compounded
swapRate:{[curve;deliveryDate;ctdMaturity;freq]
    .swaps.parRateDated[curve;deliveryDate;ctdMaturity;freq]}

// =============================================================================
// OIS CURVE BOOTSTRAPPING
// =============================================================================
// Bootstrap an OIS curve from par OIS rates
// Conventions:
//   - Tenors < 1Y: simple interest, single payment at maturity
//   - Tenors >= 1Y: annual fixed payments, ACT/360

// Log-linear DF interpolation (internal)
oisInterpDF:{[yfs;dfs;t]
    if[t <= first yfs; :first dfs];
    if[t >= last yfs; :last dfs * exp (t - last yfs) * log[last dfs] % last yfs];
    idx:(yfs bin t) & count[yfs]-2;
    df0:dfs idx; df1:dfs idx+1; t0:yfs idx; t1:yfs idx+1;
    exp log[df0] + (t - t0) * (log[df1] - log[df0]) % (t1 - t0)}

// Bootstrap OIS curve from par rates
// yf: year fractions (float list)
// parRates: par OIS rates
// curveDate: valuation date
// Returns dict with `yf (year fractions) and `df (discount factors)
oisBootstrap:{[yf;parRates;curveDate]
    // Short end (< 1Y): DF = 1 / (1 + r * t) simple interest
    shortIdx:where yf < 1;
    shortYF:yf shortIdx;
    shortDF:1 % 1 + (parRates shortIdx) * shortYF;

    // Initialize curve with short end
    allYF:shortYF;
    allDF:shortDF;

    // Bootstrap annual tenors (>= 1Y)
    annualIdx:where yf >= 1;
    dcf:365 % 360f;  // ACT/360 approx

    n:0;
    do[count annualIdx;
        t:yf annualIdx n;
        r:parRates annualIdx n;

        // Annual payment schedule: 1, 2, ..., floor(t), then t if stub
        nY:`long$floor t;
        payT:1f + til nY;
        stub:t - nY;
        if[stub > 0.01; payT:payT,t];

        // DCFs (365/360 for full years, stub*365/360 for stub)
        dcfs:(count payT)#dcf;
        if[stub > 0.01; dcfs:@[dcfs;-1+count dcfs;:;stub*dcf]];

        // Bootstrap: DF_t = (1 - r * annuity_prev) / (1 + r * dcf_last)
        prevDF:oisInterpDF[allYF;allDF;] each -1 _ payT;
        ann:$[count[prevDF]>0; sum prevDF * -1 _ dcfs; 0f];
        newDF:(1 - r * ann) % 1 + r * last dcfs;

        allYF,:t;
        allDF,:newDF;
        n+:1
    ];

    `yf`df`curveDate!(allYF;allDF;curveDate)}

// Spot OIS swap rate - interpolates par rate at swap tenor
// This is the market convention for invoice spreads:
// Compare invoice yield to spot swap rate at the same tenor as CTD maturity from delivery
// oisCurve: OIS curve with `yf`df`parRates (if available) or `yf`df
// deliveryDate: swap effective date (for calculating tenor)
// maturityDate: swap maturity
// curveDate: valuation date
oisSpotSwapRate:{[oisCurve;curveDate;deliveryDate;maturityDate]
    // Swap tenor from delivery to maturity
    swapTenor:(maturityDate - deliveryDate) % 365f;

    // If parRates available in curve, interpolate directly
    if[`parRates in key oisCurve;
        yfs:oisCurve`yf;
        prs:oisCurve`parRates;
        // Linear interpolation on par rates
        idx:yfs bin swapTenor;
        if[idx >= -1 + count yfs; :prs idx];
        if[idx < 0; :prs 0];
        t1:yfs idx; t2:yfs idx+1;
        r1:prs idx; r2:prs idx+1;
        w:(swapTenor - t1) % (t2 - t1);
        :r1 + w * (r2 - r1)
    ];

    // Otherwise use DFs to compute implied par rate at spot tenor
    // For spot starting swap at tenor T: parRate = (1 - DF_T) / annuity
    yfs:oisCurve`yf;
    dfs:oisCurve`df;

    // DF at swap tenor (from today, not from delivery)
    dfMat:oisInterpDF[yfs;dfs;swapTenor];

    // Annual payment dates for spot swap with given tenor
    nYears:floor swapTenor;
    stub:swapTenor - nYears;

    // Payment year fractions: 1, 2, ..., nYears, then swap tenor if stub exists
    payYFs:$[nYears >= 1; 1f + til "j"$nYears; `float$()];
    if[stub > 0.01; payYFs:payYFs,swapTenor];  // add stub period if > ~4 days
    if[0 = count payYFs; payYFs:enlist swapTenor];  // short tenor < 1Y

    // DFs at payments
    dfPays:oisInterpDF[yfs;dfs;] each payYFs;

    // DCFs: year fraction differences * 365/360
    dcfs:deltas[0f,payYFs] * 365 % 360;

    annuity:sum dfPays * dcfs;
    (1 - last dfPays) % annuity}

// Forward OIS swap rate from bootstrapped OIS curve
// deliveryDate: swap effective date
// maturityDate: swap maturity
// curveDate: valuation date
oisFwdSwapRate:{[oisCurve;curveDate;deliveryDate;maturityDate]
    yfs:oisCurve`yf;
    dfs:oisCurve`df;

    // Year fractions from curve date
    yfDel:(deliveryDate - curveDate) % 365f;
    yfMat:(maturityDate - curveDate) % 365f;

    // DFs at delivery and maturity
    dfDel:oisInterpDF[yfs;dfs;yfDel];
    dfMat:oisInterpDF[yfs;dfs;yfMat];

    // Generate annual payment dates from delivery to maturity
    // Compute year offsets (1, 2, 3, ... until we pass maturity)
    delYr:`year$deliveryDate;
    delMo:`mm$deliveryDate;
    delDy:`dd$deliveryDate;

    // Generate annual anniversaries from delivery
    genAnnDate:{[yr;mo;dy;n]
        newYr:yr + n;
        // Handle Feb 29 -> Feb 28 for non-leap years
        adjDy:$[(mo=2) and (dy>28) and (0<>newYr mod 4); 28; dy];
        // Build date string with explicit parts
        yrStr:string newYr;
        moStr:"0"^-2$string mo;
        dyStr:"0"^-2$string adjDy;
        "D"$yrStr,".",moStr,".",dyStr
      };

    // Generate enough annual dates
    maxYrs:1 + "j"$yfMat - yfDel;
    annDates:genAnnDate[delYr;delMo;delDy;] each 1 + til maxYrs;

    // Keep only dates <= maturity
    payDates:annDates where annDates <= maturityDate;

    // Always include maturity as final payment
    if[not maturityDate in payDates; payDates:payDates,maturityDate];
    payDates:asc distinct payDates;

    // YFs and DFs at each payment
    yfPays:{[cd;x] (x - cd) % 365f}[curveDate;] each payDates;
    dfPays:oisInterpDF[yfs;dfs;] each yfPays;

    // Forward DFs (relative to delivery)
    fwdDFs:dfPays % dfDel;

    // DCFs (ACT/360)
    allDates:deliveryDate,payDates;
    daysBetween:1_ deltas allDates;
    dcfs:daysBetween % 360f;

    // Forward par rate = (1 - fwdDF_mat) / annuity
    annuity:sum fwdDFs * dcfs;
    (1 - last fwdDFs) % annuity}

// Convenience: build OIS curve and compute forward swap rate
oisSwapRate:{[tenors;parRates;curveDate;deliveryDate;maturityDate]
    oisCurve:oisBootstrap[tenors;parRates;curveDate];
    oisFwdSwapRate[oisCurve;curveDate;deliveryDate;maturityDate]}

// Invoice spread: Swap Rate - Invoice Yield
// Positive spread = swaps trading wide of Treasuries
invoiceSpread:{[curve;deliveryDate;futuresPrice;ctdBond;freq]
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];
    ctdMat:ctdBond`maturity;
    swpRt:swapRate[curve;deliveryDate;ctdMat;freq];
    spread:swpRt - invYld;
    `invoiceYield`swapRate`invoiceSpread`spreadBps!(invYld;swpRt;spread;spread*10000)}

// Invoice spread using OIS curve and Street conventions
// This is the preferred function for accurate invoice spread calculation
// Conventions:
//   - Swap: OIS (1Y fixed, ACT/360, daily SOFR)
//   - Bond: Street (semi-annual, period count, ACT/ACT AI)
//
// Parameters:
//   tenors: OIS curve tenors (e.g., `1D`1W`1M`3M`6M`1Y`2Y`5Y`7Y`10Y...)
//   oisRates: USD SOFR par OIS rates
//   curveDate: valuation date
//   deliveryDate: futures delivery date
//   futuresPrice: futures price (decimal, e.g., 111.65625 for 111-21)
//   ctdBond: CTD bond dict with `sym`coupon`maturity`cf
invoiceSpreadOIS:{[tenors;oisRates;curveDate;deliveryDate;futuresPrice;ctdBond]
    // Invoice yield using Street convention
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];

    // Swap rate using OIS bootstrap
    swpRt:oisSwapRate[tenors;oisRates;curveDate;deliveryDate;ctdBond`maturity];

    spread:swpRt - invYld;
    `invoiceYield`swapRate`invoiceSpread`spreadBps!(invYld;swpRt;spread;spread*10000)}

// Full invoice spread with OIS curve object (for reuse)
invoiceSpreadWithOIS:{[oisCurve;curveDate;deliveryDate;futuresPrice;ctdBond]
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];
    swpRt:oisFwdSwapRate[oisCurve;curveDate;deliveryDate;ctdBond`maturity];
    spread:swpRt - invYld;
    `invoiceYield`swapRate`invoiceSpread`spreadBps!(invYld;swpRt;spread;spread*10000)}

// =============================================================================
// DURATION CALCULATIONS (for DV01)
// =============================================================================

// Modified duration from yield (semiannual bond)
// settleDate: valuation date (use deliveryDate for forward duration)
// maturityDate: bond maturity
// coupon: annual coupon rate (e.g., 0.045 for 4.5%)
// ytm: yield to maturity (annual, bond-equivalent)
// Returns modified duration in years
modifiedDuration:{[settleDate;maturityDate;coupon;ytm]
    // Calculate number of periods (semiannual)
    // Days to maturity, converted to approximate semiannual periods
    days:maturityDate - settleDate;
    nPeriods:`long$1 + (days % 182);  // ~182 days per semiannual period

    // Time to each cash flow in years (semiannual)
    times:(1 + til nPeriods) % 2f;

    // Cash flows: coupon/2 for each period, plus 100 at maturity
    // Note: must parenthesize due to Q's right-associative operators
    semiCoupon:coupon * 50f;
    cfs:((nPeriods - 1)#semiCoupon),semiCoupon + 100f;

    // Discount factors: (1 + y/2)^(-2t)
    dfs:xexp[1 + ytm % 2; neg 2 * times];

    // Dirty price
    price:sum cfs * dfs;

    // Macaulay duration = sum(t * cf * df) / price
    macDur:(sum times * cfs * dfs) % price;

    // Modified duration = Macaulay / (1 + y/2)
    macDur % 1 + ytm % 2}

// Forward modified duration for CTD bond at delivery
// Uses invoice yield (forward YTM) and delivery date
forwardModDur:{[deliveryDate;futuresPrice;ctdBond]
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];
    modifiedDuration[deliveryDate;ctdBond`maturity;ctdBond`coupon;invYld]}

// =============================================================================
// BATCH FUNCTIONS (parallelized for multiple bonds)
// =============================================================================

// Batch invoice yields for multiple bonds (parallelized)
invoiceYieldBatch:{[deliveryDate;futuresPrice;ctdBonds]
    pmap[{[d;p;b] invoiceYield[d;p;b]}[deliveryDate;futuresPrice;];3;ctdBonds]}

// Batch invoice spreads for multiple bonds (parallelized)
// Returns table with sym, invoiceYield, swapRate, spread, spreadBps
invoiceSpreadBatch:{[curve;deliveryDate;futuresPrice;ctdBonds;freq]
    // Compute invoice yields in parallel
    invYlds:invoiceYieldBatch[deliveryDate;futuresPrice;ctdBonds];
    // Compute swap rates for each maturity
    mats:ctdBonds@\:`maturity;
    swpRts:swapRate[curve;deliveryDate;;freq] each mats;
    spreads:swpRts - invYlds;
    syms:ctdBonds@\:`sym;
    ([] sym:syms; invoiceYield:invYlds; swapRate:swpRts;
        invoiceSpread:spreads; spreadBps:spreads*10000)}

// Batch hedge ratios for multiple bonds (parallelized)
// Uses forward DV01 (invoice yield duration) for correct hedging
hedgeRatioBatch:{[curve;deliveryDate;futuresPrices;ctdBonds;swapNotional;freq]
    // Compute futures DV01s in parallel (per $1MM) using forward mod duration
    futDV01s:futuresDV01Batch[deliveryDate;futuresPrices;ctdBonds];
    // Compute swap DV01s (one per maturity)
    mats:ctdBonds@\:`maturity;
    swpDV01s:swapDV01Dated[curve;swapNotional;deliveryDate;;freq] each mats;
    futNotionals:swapNotional * swpDV01s % futDV01s;
    syms:ctdBonds@\:`sym;
    ([] sym:syms; swapDV01:swpDV01s; futuresDV01:futDV01s; futuresNotional:futNotionals)}

// =============================================================================
// OPTION-ADJUSTED INVOICE SPREAD (OAS)
// =============================================================================
// The delivery option gives the futures short the right to deliver any bond
// from the basket. This option has value when:
//   1. Quality option: CTD may switch as rates move
//   2. Timing option: When to deliver during the month (first vs last)
//   3. Wild card option: Delivery after futures stop trading
//
// OAS (short-futures invoice spread convention) = Invoice Spread + (Delivery Option Value / DV01)
// A higher option value makes the received spread MORE positive (or less negative)
//
// Key methodology:
//   - CTD is determined by HIGHEST IMPLIED REPO (not lowest net basis)
//   - Implied repo uses MARKET PRICES (bond`cleanPrice), not curve prices
//   - Timing option considers both first and last delivery dates

// Standard normal CDF (used for option calculations)
normCDF:{[x]
    t:1 % 1 + 0.2316419 * abs x;
    d:0.3989423 * exp[neg 0.5 * x * x];
    p:d * t * 0.3193815 + t * neg[0.3565638] + t * 1.781478 + t * neg[1.821256] + t * 1.330274;
    $[x >= 0; 1 - p; p]}

// Implied repo for a single bond using MARKET prices
// This is the correct way to determine CTD
// bond must have: sym, coupon, maturity, cleanPrice, cf
bondImpliedRepo:{[settleDate;deliveryDate;futuresPrice;bond]
    cleanPrice:bond`cleanPrice;
    coupon:bond`coupon;
    maturity:bond`maturity;
    cf:bond`cf;
    // AI at settle and delivery
    aiSettle:.ctd.accruedInterest[settleDate;maturity;coupon;`SA;100f];
    aiDelivery:.ctd.accruedInterest[deliveryDate;maturity;coupon;`SA;100f];
    // Coupon payments between settle and delivery
    numCoupons:.ctd.couponPaymentsBetween[settleDate;deliveryDate;maturity];
    couponIncome:numCoupons * coupon * 50f;  // 50 = 100 face * 0.5 for semi-annual
    // Use ctd.q implied repo calculation
    .ctd.impliedRepoRate[settleDate;deliveryDate;cleanPrice;aiSettle;aiDelivery;futuresPrice;cf;couponIncome]}

// Rank all bonds by implied repo (highest = CTD)
// Uses MARKET prices from bond`cleanPrice
// Returns table sorted by implied repo descending
rankByImpliedRepo:{[settleDate;deliveryDate;futuresPrice;bonds]
    irs:bondImpliedRepo[settleDate;deliveryDate;futuresPrice;] each bonds;
    syms:bonds@\:`sym;
    t:([] sym:syms; impliedRepo:irs; bond:bonds);
    `impliedRepo xdesc t}

// Net basis using MARKET prices (consistent with implied repo)
// Net basis = Forward Price - Invoice Price (in price terms)
// Lower (more negative) net basis = higher implied repo = more likely CTD
netBasisMarket:{[settleDate;deliveryDate;futuresPrice;bond;repoRate]
    cleanPrice:bond`cleanPrice;
    coupon:bond`coupon;
    maturity:bond`maturity;
    cf:bond`cf;
    days:deliveryDate - settleDate;
    // AI at settle and delivery
    aiSettle:.ctd.accruedInterest[settleDate;maturity;coupon;`SA;100f];
    aiDelivery:.ctd.accruedInterest[deliveryDate;maturity;coupon;`SA;100f];
    // Dirty price and forward dirty
    dirtySpot:cleanPrice + aiSettle;
    fwdDirty:dirtySpot * 1 + repoRate * days % 360;
    fwdClean:fwdDirty - aiDelivery;
    // Invoice clean
    invoiceClean:futuresPrice * cf;
    // Net basis = forward - invoice (negative = bond is cheap to deliver)
    fwdClean - invoiceClean}

// Legacy net basis using curve prices (kept for backward compatibility)
netBasis:{[curve;settleDate;deliveryDate;futuresPrice;bond;repoRate]
    // Price bond on curve
    bondPrice:.ctd.priceBond[curve;settleDate;bond`maturity;bond`coupon;`SA;100f];
    // Gross basis
    grossBasis:bondPrice - (futuresPrice * bond`cf);
    // Carry to delivery
    ai0:.ctd.accruedInterest[settleDate;bond`maturity;bond`coupon;`SA;100f];
    dirtyPrice:bondPrice + ai0;
    daysToDelivery:deliveryDate - settleDate;
    carry:(dirtyPrice * repoRate * daysToDelivery) % 360;
    // Coupon accrual (simplified - full calculation would check coupon dates)
    couponAccrual:(bond[`coupon] * 100 * daysToDelivery) % 365;
    netCarry:couponAccrual - carry;
    grossBasis - netCarry}

// Legacy rank by net basis (kept for backward compatibility)
rankByNetBasis:{[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate]
    nbs:netBasis[curve;settleDate;deliveryDate;futuresPrice;;repoRate] each bonds;
    syms:bonds@\:`sym;
    t:([] sym:syms; netBasis:nbs; bond:bonds);
    `netBasis xasc t}

// Calculate switch points using MARKET prices
// Returns rate shift (in bp) where CTD changes
// Uses net basis differential and futures DV01 sensitivity
// As rates rise, higher duration bonds cheapen relative to lower duration
switchPointsMarket:{[settleDate;deliveryDate;futuresPrice;bonds;repoRate]
    if[2 > count bonds; :([] sym:(); switchBps:(); bond:())];
    // Current ranking by implied repo
    ranked:rankByImpliedRepo[settleDate;deliveryDate;futuresPrice;bonds];
    ctd:first ranked;
    ctdBond:ctd`bond;
    others:1 _ ranked;
    if[0 = count others; :([] sym:(); switchBps:(); bond:())];
    // Get DV01 for each bond (price sensitivity to 1bp yield change)
    getBondDV01:{[s;b]
        ytm:.ctd.yieldToMaturity[s;b`maturity;b`coupon;`SA;100f;b`cleanPrice];
        pUp:.ctd.priceBondFromYield[s;b`maturity;b`coupon;`SA;100f;ytm+0.0001];
        pDn:.ctd.priceBondFromYield[s;b`maturity;b`coupon;`SA;100f;ytm-0.0001];
        (pDn - pUp) % 2};
    ctdDV01:getBondDV01[settleDate;ctdBond];
    // CTD net basis (should be lowest/most negative)
    ctdNB:netBasisMarket[settleDate;deliveryDate;futuresPrice;ctdBond;repoRate];
    // For each other bond, calculate switch point
    // Net basis changes with rates: ΔNB ≈ (DV01/CF) × Δrate
    // Switch when: bondNB - ctdNB + (bondDV01/CF - ctdDV01/CF) × Δrate = 0
    result:{[s;deliv;fp;repo;ctdDV01;ctdCF;ctdNB;row]
        bond:row`bond;
        bondIR:row`impliedRepo;
        // Bond DV01
        ytm:.ctd.yieldToMaturity[s;bond`maturity;bond`coupon;`SA;100f;bond`cleanPrice];
        pUp:.ctd.priceBondFromYield[s;bond`maturity;bond`coupon;`SA;100f;ytm+0.0001];
        pDn:.ctd.priceBondFromYield[s;bond`maturity;bond`coupon;`SA;100f;ytm-0.0001];
        bondDV01:(pDn - pUp) % 2;
        // Net basis for this bond
        bondNB:netBasisMarket[s;deliv;fp;bond;repo];
        // Net basis sensitivity to yield changes
        // NB = Price - Futures × CF
        // When yields rise: dPrice/dy = -bondDV01, dFutures/dy = -ctdDV01/ctdCF
        // So d(NB)/dy = -bondDV01 - bondCF × (-ctdDV01/ctdCF)
        //             = -bondDV01 + bondCF × ctdDV01/ctdCF
        //             = bondCF × ctdDV01/ctdCF - bondDV01
        // For CTD: ctdCF × ctdDV01/ctdCF - ctdDV01 = 0 ✓
        nbSensDiff:(bond[`cf] * ctdDV01 % ctdCF) - bondDV01;
        // Net basis gap (positive = bond is more expensive than CTD)
        nbGap:bondNB - ctdNB;
        // Rate move for switch: nbGap + nbSensDiff × Δrate = 0
        // Δrate = -nbGap / nbSensDiff
        // If nbSensDiff > 0 (bond has higher duration), rates need to FALL for bond to become CTD
        // If nbSensDiff < 0 (bond has lower duration), rates need to RISE for bond to become CTD
        // switchRate is already in bp (since DV01 is $/bp and nbGap is $)
        // Use small threshold (1e-6) to avoid division by zero but allow small sensitivity differences
        switchRate:$[abs[nbSensDiff] > 1e-6; neg nbGap % nbSensDiff; 0n];
        `sym`switchBps`netBasis`impliedRepo`bond!(row`sym;switchRate;bondNB;bondIR;bond)
        }[settleDate;deliveryDate;futuresPrice;repoRate;ctdDV01;ctdBond`cf;ctdNB] each others;
    result}

// Legacy: Calculate switch points using curve prices (kept for backward compatibility)
// Returns the rate move needed for each bond to become CTD
switchPoints:{[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate]
    // Current ranking
    ranked:rankByNetBasis[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate];
    ctd:first ranked;
    ctdBond:ctd`bond;
    ctdNB:ctd`netBasis;
    // For each non-CTD bond, estimate rate move to become CTD
    // Approximation: ΔNB ≈ (DV01_ctd/CF_ctd - DV01_bond/CF_bond) × Δrate
    // When ΔNB = (bond NB - CTD NB), that bond becomes CTD
    others:1 _ ranked;
    if[0 = count others; :([] sym:(); switchBps:(); bond:())];
    // Get DV01s using bumped curves
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];
    // Helper to get bond DV01
    getBondDV01:{[cu;cd;s;b]
        p1:.ctd.priceBond[cu;s;b`maturity;b`coupon;`SA;100f];
        p2:.ctd.priceBond[cd;s;b`maturity;b`coupon;`SA;100f];
        (p2 - p1) % 2};
    ctdDV01:getBondDV01[curveUp;curveDn;settleDate;ctdBond];
    ctdFutDV01:ctdDV01 % ctdBond`cf;
    // For each other bond, calculate switch point
    result:{[cu;cd;s;ctdFDV01;ctdNB;row]
        bond:row`bond;
        bondNB:row`netBasis;
        // Calculate bond DV01 inline
        p1:.ctd.priceBond[cu;s;bond`maturity;bond`coupon;`SA;100f];
        p2:.ctd.priceBond[cd;s;bond`maturity;bond`coupon;`SA;100f];
        bondDV01:(p2 - p1) % 2;
        bondFutDV01:bondDV01 % bond`cf;
        // Net basis sensitivity to rate change
        // When rates rise, lower duration bonds gain relative advantage
        nbSens:ctdFDV01 - bondFutDV01;  // positive if CTD has higher duration
        nbGap:bondNB - ctdNB;
        // Rate move for this bond to become CTD (already in bp since DV01 is $/bp)
        switchRate:$[abs[nbSens] > 1e-6; nbGap % nbSens; 0n];
        `sym`switchBps`bond!(row`sym;switchRate;bond)
        }[curveUp;curveDn;settleDate;ctdFutDV01;ctdNB] each others;
    result}

// Estimate delivery option value using MARKET prices
// Includes both quality option (CTD switch) and timing option (first vs last delivery)
// vol: annualized rate volatility (e.g., 0.01 = 100bp/year)
// firstDelivery/lastDelivery: first and last delivery dates
// Returns dict with qualityOption, timingOption, and total
deliveryOptionValueMarket:{[settleDate;firstDelivery;lastDelivery;futuresPrice;bonds;repoRate;vol]
    if[1 >= count bonds; :`qualityOption`timingOption`total!(0f;0f;0f)];
    deliveryDate:lastDelivery;
    horizon:(deliveryDate - settleDate) % 365f;
    if[horizon <= 0; :`qualityOption`timingOption`total!(0f;0f;0f)];
    sigmaT:vol * sqrt horizon;
    // Rank once — reused for CTD identification and switch points
    ranked:rankByImpliedRepo[settleDate;deliveryDate;futuresPrice;bonds];
    ctdBond:(first ranked)`bond;
    ctdIR:(first ranked)`impliedRepo;
    // CTD net basis (computed once, reused in quality option loop)
    ctdNB:netBasisMarket[settleDate;deliveryDate;futuresPrice;ctdBond;repoRate];
    ctdCF:ctdBond`cf;
    // CTD DV01 (computed once, reused in switch point estimation)
    ctdYtm:.ctd.yieldToMaturity[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdBond`cleanPrice];
    ctdPUp:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdYtm+0.0001];
    ctdPDn:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdYtm-0.0001];
    ctdDV01:(ctdPDn - ctdPUp) % 2;
    // Compute switch points + net basis for all non-CTD bonds in one pass
    others:1 _ ranked;
    sigmaTbps:sigmaT * 10000;
    maxSwitchBps:300f;
    qualityOpt:0f;
    if[0 < count others;
        // Compute YTM, DV01, net basis for all non-CTD bonds in one pass
        otherBonds:others@\:`bond;
        ytmFn:{[s;b] .ctd.yieldToMaturity[s;b`maturity;b`coupon;`SA;100f;b`cleanPrice]}[settleDate;];
        otherYtms:ytmFn each otherBonds;
        pFn:{[s;b;y] .ctd.priceBondFromYield[s;b`maturity;b`coupon;`SA;100f;y]}[settleDate;];
        otherPUp:pFn'[otherBonds;otherYtms+0.0001];
        otherPDn:pFn'[otherBonds;otherYtms-0.0001];
        otherDV01s:(otherPDn - otherPUp) % 2;
        otherNBs:netBasisMarket[settleDate;deliveryDate;futuresPrice;;repoRate] each otherBonds;
        // Switch points: Δrate = -nbGap / nbSensDiff
        nbGaps:otherNBs - ctdNB;
        nbSensDiffs:((otherBonds@\:`cf) * ctdDV01 % ctdCF) - otherDV01s;
        switchBpsList:?[abs[nbSensDiffs] > 1e-6; neg nbGaps % nbSensDiffs; count[otherBonds]#0n];
        // Quality option: max over probability-weighted net basis gaps
        i:0;
        do[count otherBonds;
            switchBps:switchBpsList i;
            if[not[null switchBps] and (abs switchBps) < maxSwitchBps;
                zScore:$[sigmaTbps > 1e-6; switchBps % sigmaTbps; 0n];
                zScore:$[null zScore; 0n; -4f | 4f & zScore];
                pSwitch:$[null zScore; 0f; 1 - normCDF abs zScore];
                qualityOpt:qualityOpt | pSwitch * abs nbGaps i];
            i+:1]];
    // TIMING OPTION
    daysDiff:lastDelivery - firstDelivery;
    carrySpread:ctdIR - repoRate;
    aiSettle:.ctd.accruedInterest[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f];
    dirtyPrice:ctdBond[`cleanPrice] + aiSettle;
    dailyCarryAbs:abs (dirtyPrice * carrySpread) % 360;
    timingOpt:0.5 * dailyCarryAbs * daysDiff;
    `qualityOption`timingOption`total`carrySpread!(qualityOpt;timingOpt;qualityOpt + timingOpt;carrySpread)}

// =============================================================================
// WILDCARD OPTION
// =============================================================================
// The wildcard option arises from the 6-hour window between:
//   - 2:00 PM CT: Futures settlement price is locked
//   - 8:00 PM CT: Deadline for short to declare intent to deliver
//
// During this window, the cash bond market continues trading.
// If bond prices FALL after 2 PM, short can buy cheap and deliver at locked price.
// If bond prices RISE, short waits for another day.
//
// This is essentially a daily at-the-money call option on the bond price move.
// =============================================================================

// Wildcard option value for a single day
// intradayVol: annualized intraday volatility (e.g., 0.005 = 50bp/year)
//              This should reflect 2PM-8PM price volatility, typically 30-50% of daily vol
// Returns: option value in price points (per 100 face)
wildcardDaily:{[ctdDV01;ctdCF;intradayVol]
    // Convert annual vol to 6-hour vol
    // Daily vol = annual vol / sqrt(252)
    // 6-hour vol = daily vol * sqrt(6/24) = daily vol * 0.5
    dailyVol:intradayVol % sqrt 252f;
    sixHourVol:dailyVol * sqrt 6 % 24;
    // Price volatility for 6-hour window (in price points)
    // priceVol = DV01 * yieldVol * 10000 (convert decimal vol to bp effect)
    priceVol:ctdDV01 * sixHourVol * 10000;
    // ATM call option value ≈ 0.4 * sigma (for normal distribution)
    // This is E[max(0, X)] where X ~ N(0, sigma)
    dailyValue:0.4 * priceVol;
    // Convert to futures terms (divide by CF)
    dailyValue % ctdCF}

// Wildcard option value over the delivery month
// settleDate: current date
// firstDelivery: first day of delivery month
// lastDelivery: last delivery day
// ctdBond: CTD bond dict with `coupon`maturity`cleanPrice`cf
// intradayVol: annualized intraday volatility
// Returns: dict with `daily`nDays`total
wildcardOption:{[settleDate;firstDelivery;lastDelivery;ctdBond;intradayVol]
    // Compute CTD DV01
    ytm:.ctd.yieldToMaturity[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdBond`cleanPrice];
    pUp:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ytm+0.0001];
    pDn:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ytm-0.0001];
    ctdDV01:(pDn - pUp) % 2;
    ctdCF:ctdBond`cf;
    // Number of delivery days remaining
    // If before delivery month, count all days
    // If during delivery month, count remaining days
    startDate:settleDate | firstDelivery;
    nDays:1 + lastDelivery - startDate;
    if[nDays <= 0; :`daily`nDays`total!(0f;0;0f)];
    // Daily wildcard value
    daily:wildcardDaily[ctdDV01;ctdCF;intradayVol];
    // Total wildcard value
    // The short can exercise on any day, but exercising forfeits remaining wildcards
    // Use sqrt(N) scaling to account for optimal exercise (not just sum)
    // This is a simplification; true value requires dynamic programming
    total:daily * sqrt nDays;
    `daily`nDays`total!(daily;nDays;total)}

// Wildcard option in basis points (yield terms)
wildcardOptionBps:{[settleDate;firstDelivery;lastDelivery;ctdBond;intradayVol]
    wc:wildcardOption[settleDate;firstDelivery;lastDelivery;ctdBond;intradayVol];
    // Convert to yield terms: option_bps = option_points / (DV01/CF) / 100
    // Or equivalently: option_bps = option_points * CF / DV01 / 100
    ytm:.ctd.yieldToMaturity[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdBond`cleanPrice];
    pUp:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ytm+0.0001];
    pDn:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ytm-0.0001];
    ctdDV01:(pDn - pUp) % 2;
    wcBps:(wc`total) % ctdDV01;
    `daily`nDays`total`totalBps!(wc`daily;wc`nDays;wc`total;wcBps)}

// Full delivery option value including wildcard
// Returns: dict with quality, timing, wildcard, and total
fullDeliveryOptionValue:{[settleDate;firstDelivery;lastDelivery;futuresPrice;bonds;repoRate;rateVol;intradayVol]
    // Quality and timing options
    qtOpt:deliveryOptionValueMarket[settleDate;firstDelivery;lastDelivery;futuresPrice;bonds;repoRate;rateVol];
    // Find CTD for wildcard calculation
    ranked:rankByImpliedRepo[settleDate;lastDelivery;futuresPrice;bonds];
    ctdBond:first[ranked]`bond;
    // Wildcard option
    wc:wildcardOption[settleDate;firstDelivery;lastDelivery;ctdBond;intradayVol];
    // Total
    total:(qtOpt`total) + wc`total;
    `qualityOption`timingOption`wildcardOption`total`wildcardDaily`wildcardDays!(
        qtOpt`qualityOption;
        qtOpt`timingOption;
        wc`total;
        total;
        wc`daily;
        wc`nDays)}

// Legacy: Estimate delivery option value using curve prices
// vol: annualized rate volatility (e.g., 0.01 = 100bp/year)
// horizon: time to delivery in years
deliveryOptionValue:{[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate;vol]
    if[1 >= count bonds; :0f];  // No option with single bond
    // Time to delivery
    horizon:(deliveryDate - settleDate) % 365f;
    if[horizon <= 0; :0f];
    // Volatility over horizon
    sigmaT:vol * sqrt horizon;
    // Get switch points
    switches:switchPoints[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate];
    if[0 = count switches; :0f];
    // Current ranking for CTD info
    ranked:rankByNetBasis[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate];
    ctdBond:(first ranked)`bond;
    // For each potential switch, calculate probability and value
    // P(switch) ≈ N(-switchBps / sigmaT) for upward switches (rates rise)
    // P(switch) ≈ N(switchBps / sigmaT) for downward switches (rates fall)
    // Value = P(switch) × |net basis difference|
    // Standard normal CDF approximation
    normCDF:{[x]
        t:1 % 1 + 0.2316419 * abs x;
        d:0.3989423 * exp[neg 0.5 * x * x];
        p:d * t * 0.3193815 + t * neg[0.3565638] + t * 1.781478 + t * neg[1.821256] + t * 1.330274;
        $[x >= 0; 1 - p; p]};
    optVal:0f;
    i:0;
    do[count switches;
        sw:switches i;
        switchBps:sw`switchBps;
        bond:sw`bond;
        // Net basis gap (value of switching)
        bondNB:netBasis[curve;settleDate;deliveryDate;futuresPrice;bond;repoRate];
        ctdNB:netBasis[curve;settleDate;deliveryDate;futuresPrice;ctdBond;repoRate];
        nbGap:bondNB - ctdNB;  // Positive = bond is more expensive than CTD
        // Probability of switch (rate moves enough)
        zScore:$[(sigmaT * 100) > 1e-6; switchBps % (sigmaT * 100); 0n];
        // If switch requires rates to rise (positive switchBps), use right tail
        // If switch requires rates to fall (negative switchBps), use left tail
        pSwitch:$[null zScore; 0f; switchBps > 0; 1 - normCDF zScore; normCDF neg zScore];
        // Option value contribution (in price terms per 100 face)
        // The short gains |nbGap| by switching, weighted by probability
        contribution:pSwitch * abs nbGap;
        optVal+:contribution;
        i+:1];
    optVal}

// Option-adjusted invoice spread using MARKET prices (with wildcard option)
// Uses implied repo ranking and includes timing option + wildcard option
// delivery: dict with `first`last keys for delivery dates
// Works with OIS curve (from oisBootstrap) or swaps curve (from .swaps.buildCurve)
// cfg: dict with `vol`intradayVol`freq keys (intradayVol defaults to 0.003 = 30bp/year)
// Returns standard spread, option breakdown (quality + timing + wildcard), and OAS
oasInvoiceSpreadMarketFull:{[curve;delivery;futuresPrice;ctdBond;basket;repoRate;cfg]
    // Extract config with defaults
    vol:cfg`vol;
    intradayVol:$[`intradayVol in key cfg; cfg`intradayVol; 0.003];
    freq:$[`freq in key cfg; cfg`freq; `6M];
    // Determine curve type and get settle date
    isOIS:`curveDate in key curve;
    settleDate:$[isOIS; curve`curveDate; curve[`config]`asOfDate];
    firstDelivery:delivery`first;
    lastDelivery:delivery`last;
    // Invoice yield from market price (same for both curve types)
    invYld:invoiceYield[lastDelivery;futuresPrice;ctdBond];
    // Swap rate: use appropriate function based on curve type
    ctdMat:ctdBond`maturity;
    swpRt:$[isOIS;
        oisFwdSwapRate[curve;settleDate;lastDelivery;ctdMat];
        .swaps.parRateDated[curve;lastDelivery;ctdMat;freq]];
    spread:swpRt - invYld;
    spreadBps:spread * 10000;
    // Quality + Timing option value
    optResult:deliveryOptionValueMarket[settleDate;firstDelivery;lastDelivery;futuresPrice;basket;repoRate;vol];
    // Wildcard option value
    wc:wildcardOption[settleDate;firstDelivery;lastDelivery;ctdBond;intradayVol];
    // Total option value
    totalOptVal:(optResult`total) + wc`total;
    // CTD DV01 for yield conversion — reuse from deliveryOptionValue if available
    // (it already computed CTD YTM + price bumps internally)
    ctdYtm:.ctd.yieldToMaturity[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdBond`cleanPrice];
    ctdPUp:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdYtm+0.0001];
    ctdPDn:.ctd.priceBondFromYield[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdYtm-0.0001];
    ctdDV01:(ctdPDn - ctdPUp) % 2;
    optYieldAdj:$[ctdDV01 > 1e-6; totalOptVal % ctdDV01; 0f];
    // OAS (short-futures convention) = standard spread + option adjustment
    oas:spread + (optYieldAdj * 0.0001);
    oasBps:oas * 10000;
    // Net basis for CTD (using market prices)
    ctdNetBasis:netBasisMarket[settleDate;lastDelivery;futuresPrice;ctdBond;repoRate];
    `invoiceYield`swapRate`invoiceSpread`spreadBps`qualityOption`timingOption`wildcardOption`totalOption`optionAdjBps`oas`oasBps`netBasis!(
        invYld;
        swpRt;
        spread;
        spreadBps;
        optResult`qualityOption;
        optResult`timingOption;
        wc`total;
        totalOptVal;
        optYieldAdj;
        oas;
        oasBps;
        ctdNetBasis)}

// Wrapper with simple signature (includes wildcard with default 30bp intraday vol)
oasInvoiceSpreadMarket:{[curve;delivery;futuresPrice;ctdBond;basket;repoRate;vol;freq]
    oasInvoiceSpreadMarketFull[curve;delivery;futuresPrice;ctdBond;basket;repoRate;`vol`freq!(vol;freq)]}

// Full OAS analysis using MARKET prices (with wildcard option)
// Identifies CTD by implied repo and includes full option breakdown
// delivery: dict with `first`last keys for delivery dates
// cfg: dict with `vol`intradayVol`freq keys
oasAnalysisMarketFull:{[curve;delivery;futuresPrice;basket;repoRate;cfg]
    vol:cfg`vol;
    intradayVol:$[`intradayVol in key cfg; cfg`intradayVol; 0.003];
    freq:$[`freq in key cfg; cfg`freq; `6M];
    // Handle both OIS curve (curveDate) and swaps curve (config`asOfDate)
    settleDate:$[`curveDate in key curve; curve`curveDate; curve[`config]`asOfDate];
    firstDelivery:delivery`first;
    lastDelivery:delivery`last;
    // Rank bonds by implied repo (at last delivery)
    ranked:rankByImpliedRepo[settleDate;lastDelivery;futuresPrice;basket];
    ctdBond:(first ranked)`bond;
    ctdIR:(first ranked)`impliedRepo;
    // Calculate OAS (with custom intradayVol)
    oas:oasInvoiceSpreadMarketFull[curve;delivery;futuresPrice;ctdBond;basket;repoRate;cfg];
    // Switch points using market prices
    switches:switchPointsMarket[settleDate;lastDelivery;futuresPrice;basket;repoRate];
    // Package results
    `oas`ctdBond`ctdImpliedRepo`ranking`switchPoints`vol`intradayVol`settleDate`firstDelivery`lastDelivery!(
        oas;ctdBond;ctdIR;ranked;switches;vol;intradayVol;settleDate;firstDelivery;lastDelivery)}

// Wrapper with simple signature (includes wildcard with default 30bp intraday vol)
oasAnalysisMarket:{[curve;delivery;futuresPrice;basket;repoRate;vol;freq]
    oasAnalysisMarketFull[curve;delivery;futuresPrice;basket;repoRate;`vol`freq!(vol;freq)]}

// Legacy: Option-adjusted invoice spread using curve prices
// Returns standard spread, option value, and OAS
oasInvoiceSpread:{[curve;deliveryDate;futuresPrice;ctdBond;basket;repoRate;vol;freq]
    settleDate:curve[`config]`asOfDate;
    // Standard invoice spread (using current CTD)
    std:invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq];
    // Delivery option value
    optVal:deliveryOptionValue[curve;settleDate;deliveryDate;futuresPrice;basket;repoRate;vol];
    // Convert option value to yield terms
    // Option value is in price terms (per 100 face)
    // DV01 converts price to yield: Δyield ≈ Δprice / DV01
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];
    ctdDV01:(.ctd.priceBond[curveDn;settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f] -
             .ctd.priceBond[curveUp;settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f]) % 2;
    // Option adjustment in yield terms (per 1bp)
    optYieldAdj:$[ctdDV01 > 1e-6; optVal % ctdDV01; 0f];
    // OAS (short-futures convention) = standard spread + option adjustment
    // The option benefits the short futures leg, so add to swap-minus-CTD spread
    oas:std[`invoiceSpread] + (optYieldAdj * 0.0001);
    oasBps:oas * 10000;
    `invoiceYield`swapRate`invoiceSpread`spreadBps`optionValue`optionAdjBps`oas`oasBps!(
        std`invoiceYield;
        std`swapRate;
        std`invoiceSpread;
        std`spreadBps;
        optVal;
        optYieldAdj;
        oas;
        oasBps)}

// Full OAS analysis with breakdown
oasAnalysis:{[curve;deliveryDate;futuresPrice;basket;repoRate;vol;freq]
    settleDate:curve[`config]`asOfDate;
    // Rank bonds to find CTD
    ranked:rankByNetBasis[curve;settleDate;deliveryDate;futuresPrice;basket;repoRate];
    ctdBond:(first ranked)`bond;
    // Calculate OAS
    oas:oasInvoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;basket;repoRate;vol;freq];
    // Switch points
    switches:switchPoints[curve;settleDate;deliveryDate;futuresPrice;basket;repoRate];
    // Package results
    `oas`ctdBond`ranking`switchPoints`vol`settleDate`deliveryDate!(
        oas;ctdBond;ranked;switches;vol;settleDate;deliveryDate)}

// =============================================================================
// DV01 & HEDGE RATIO
// =============================================================================

// Futures DV01 per $1MM notional (in dollars)
// Uses forward modified duration from invoice yield
// Futures DV01 = ModDur × Futures Price × 0.0001 × 10000 = ModDur × Futures Price
// This is the correct forward DV01 for invoice spread hedging
futuresDV01:{[deliveryDate;futuresPrice;ctdBond]
    modDur:forwardModDur[deliveryDate;futuresPrice;ctdBond];
    // DV01 per $1MM = ModDur * futuresPrice (in dollars)
    modDur * futuresPrice}

// Futures DV01 using curve (old method - kept for comparison)
// Uses spot DV01, not forward DV01
futuresDV01Curve:{[curve;settleDate;ctdBond]
    mat:ctdBond`maturity; cpn:ctdBond`coupon; cf:ctdBond`cf;
    // Bump curve ±1bp (central difference for accuracy)
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];
    // Bond DV01 = (P_down - P_up) / 2 per 100 face
    bondDV01:(.ctd.priceBond[curveDn;settleDate;mat;cpn;`SA;100f] -
              .ctd.priceBond[curveUp;settleDate;mat;cpn;`SA;100f]) % 2;
    // Futures DV01 per $1MM = Bond DV01 / CF * 1000000 / 100
    bondDV01 * 10000 % cf}

// Batch futures DV01 for multiple bonds
// Uses forward modified duration - much faster than curve bumping
futuresDV01Batch:{[deliveryDate;futuresPrices;ctdBonds]
    modDurs:forwardModDur[deliveryDate;;]'[futuresPrices;ctdBonds];
    modDurs * futuresPrices}

// Swap DV01 for the delivery-to-maturity swap (in dollars)
swapDV01Dated:{[curve;notional;deliveryDate;ctdMaturity;freq]
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    // Convert dates to year fractions for DV01 calculation
    startYF:.swaps.dcfDates[asOf;deliveryDate;dc];
    endYF:.swaps.dcfDates[asOf;ctdMaturity;dc];
    maturity:endYF - startYF;
    // Par rate for this swap
    parRt:.swaps.parRateDated[curve;deliveryDate;ctdMaturity;freq];
    // DV01 approximation: use standard DV01 with maturity from delivery
    // For forward starting swaps, DV01 is similar to spot swap of same length
    .swaps.dv01[curve;notional;parRt;endYF;freq;1b]}

// Hedge ratio: futures notional needed to hedge swap notional
// Returns DV01s (both per $1MM) and the futures notional for a DV01-neutral hedge
// Uses forward DV01 (invoice yield duration) for correct hedging
hedgeRatio:{[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq]
    ctdMat:ctdBond`maturity;
    // Swap DV01 for the given notional
    swpDV01:swapDV01Dated[curve;swapNotional;deliveryDate;ctdMat;freq];
    // Futures DV01 per $1MM (uses forward modified duration)
    futDV01perMM:futuresDV01[deliveryDate;futuresPrice;ctdBond];
    // Futures notional needed: (futNotional/1MM) * futDV01perMM = swpDV01
    // So: futNotional = swpDV01 * 1MM / futDV01perMM
    futNotional:swpDV01 * 1000000 % futDV01perMM;
    `swapDV01`futuresDV01`swapNotional`futuresNotional!(swpDV01;futDV01perMM;swapNotional;futNotional)}

// =============================================================================
// POSITION SIZING
// =============================================================================

// Calculate futures notional from swap notional (convenience wrapper)
futuresNotionalFromSwap:{[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq]
    hr:hedgeRatio[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq];
    hr`futuresNotional}

// Calculate swap notional from futures notional
swapNotionalFromFutures:{[curve;deliveryDate;futuresPrice;ctdBond;futuresNotional;freq]
    ctdMat:ctdBond`maturity;
    // Futures DV01 per $1MM (uses forward modified duration)
    futDV01perMM:futuresDV01[deliveryDate;futuresPrice;ctdBond];
    // Total futures DV01
    totalFutDV01:futuresNotional * futDV01perMM % 1000000;
    // Swap DV01 per $1MM
    swpDV01perMM:swapDV01Dated[curve;1000000f;deliveryDate;ctdMat;freq];
    // Swap notional that produces same DV01
    1000000f * totalFutDV01 % swpDV01perMM}

// Create a balanced position (notional-based)
// direction: `long (receive fixed, short futures) or `short (pay fixed, long futures)
createPosition:{[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq;direction]
    hr:hedgeRatio[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq];
    futNotional:hr`futuresNotional;
    spread:invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq];
    // Long spread = receive fixed on swap, short futures (benefit if spread widens)
    isReceiver:direction~`long;
    `deliveryDate`futuresPrice`ctdBond`swapNotional`futuresNotional`freq`isReceiver`direction`spread`hedgeRatio!(
        deliveryDate;
        futuresPrice;
        ctdBond;
        swapNotional;
        futNotional;
        freq;
        isReceiver;
        direction;
        spread;
        hr)}

// =============================================================================
// P&L DECOMPOSITION
// =============================================================================

// Swap P&L from curve change
swapPnL:{[curve;position;newCurve]
    deliveryDate:position`deliveryDate;
    ctdMat:position[`ctdBond]`maturity;
    notional:position`swapNotional;
    freq:position`freq;
    isReceiver:position`isReceiver;
    // Original swap rate
    swpRt0:.swaps.parRateDated[curve;deliveryDate;ctdMat;freq];
    // Price swap at original rate on original curve (should be ~0 at inception)
    pv0:(.swaps.priceSwapDated[curve;notional;swpRt0;deliveryDate;ctdMat;freq;isReceiver])`pv;
    // Price swap at original rate on new curve
    pv1:(.swaps.priceSwapDated[newCurve;notional;swpRt0;deliveryDate;ctdMat;freq;isReceiver])`pv;
    // New swap rate (for information)
    swpRt1:.swaps.parRateDated[newCurve;deliveryDate;ctdMat;freq];
    `swapRate0`swapRate1`swapRateChange`pv0`pv1`pnl!(swpRt0;swpRt1;swpRt1-swpRt0;pv0;pv1;pv1-pv0)}

// Futures P&L from price change (notional-based)
// P&L = notional * (price change / 100)
futuresPnL:{[position;newFuturesPrice]
    futNotional:position`futuresNotional;
    price0:position`futuresPrice;
    direction:position`direction;
    // P&L = notional * (newPrice - oldPrice) / 100
    totalPnL:futNotional * (newFuturesPrice - price0) % 100;
    // Adjust for direction: long spread = short futures (negate P&L)
    signedPnL:$[direction~`long; neg totalPnL; totalPnL];
    `futuresPrice0`futuresPrice1`priceChange`totalPnL`signedPnL!(
        price0;newFuturesPrice;newFuturesPrice-price0;totalPnL;signedPnL)}

// Full P&L decomposition (optimized: reuses swap rates, minimal YTM solves)
pnlDecomposition:{[curve;position;newCurve;newFuturesPrice]
    swpPnL:swapPnL[curve;position;newCurve];
    futPnL:futuresPnL[position;newFuturesPrice];
    totalPnL:swpPnL[`pnl] + futPnL[`signedPnL];
    // Spread change: reuse swap rates from swpPnL, compute invoice yields once
    deliveryDate:position`deliveryDate;
    ctdBond:position`ctdBond;
    // Invoice yields (clean price = futures * CF)
    invYld0:invoiceYield[deliveryDate;position`futuresPrice;ctdBond];
    invYld1:invoiceYield[deliveryDate;newFuturesPrice;ctdBond];
    // Build spread dicts using already-computed swap rates
    spread0:`invoiceYield`swapRate`invoiceSpread`spreadBps!(
        invYld0;swpPnL`swapRate0;swpPnL[`swapRate0]-invYld0;(swpPnL[`swapRate0]-invYld0)*10000);
    spread1:`invoiceYield`swapRate`invoiceSpread`spreadBps!(
        invYld1;swpPnL`swapRate1;swpPnL[`swapRate1]-invYld1;(swpPnL[`swapRate1]-invYld1)*10000);
    spreadChange:spread1[`invoiceSpread] - spread0[`invoiceSpread];
    `swapPnL`futuresPnL`totalPnL`spread0`spread1`spreadChangeBps!(
        swpPnL;futPnL;totalPnL;spread0;spread1;spreadChange*10000)}

// =============================================================================
// CARRY ANALYSIS
// =============================================================================

// Swap carry for the position
swapCarry:{[curve;position;horizon]
    deliveryDate:position`deliveryDate;
    ctdMat:position[`ctdBond]`maturity;
    notional:position`swapNotional;
    freq:position`freq;
    isReceiver:position`isReceiver;
    swpRt:.swaps.parRateDated[curve;deliveryDate;ctdMat;freq];
    .swaps.swapCarryRollDated[curve;notional;deliveryDate;ctdMat;swpRt;freq;isReceiver;horizon]}

// Futures carry (uses ctd.futuresCarryRoll1D scaled to horizon)
futuresCarry:{[curve;position;repoRate;horizon]
    ctdBond:position`ctdBond;
    settleDate:curve[`config]`asOfDate;
    // Daily carry from ctd.q
    daily:.ctd.futuresCarryRoll1D[curve;settleDate;ctdBond;repoRate];
    // Scale to horizon
    hp:$[-11h = type horizon; .swaps.tenorToYF horizon; `float$horizon];
    days:`int$hp * 365;
    scaledCarry:daily[`futuresCarry] * days;
    scaledRoll:daily[`futuresRollDown] * days;
    scaledTheta:daily[`futuresTheta] * days;
    daily,`horizonCarry`horizonRollDown`horizonTheta`horizonDays`horizon!(
        scaledCarry;scaledRoll;scaledTheta;days;horizon)}

// Net spread carry (swap - futures for long spread)
// Uses notional-based calculations
spreadCarry:{[curve;position;repoRate;horizon]
    swpCry:swapCarry[curve;position;horizon];
    futCry:futuresCarry[curve;position;repoRate;horizon];
    direction:position`direction;
    futNotional:position`futuresNotional;
    // Scale futures carry to position size (carry is per 100 face, notional is face value)
    futCarryVal:futCry[`horizonCarry] * futNotional % 100;
    futRollVal:futCry[`horizonRollDown] * futNotional % 100;
    // Swap carry from swaps.swapCarryRollDated
    swapCarryVal:swpCry`carry;
    swapRollVal:swpCry`rollDown;
    // Net: for long spread (receive fixed, short futures), we add swap P&L and subtract futures P&L
    // Swap carry is positive if we earn fixed > floating, futures carry is positive if bond carry is positive
    // Long spread benefits from swap carry, loses from futures carry (since we're short)
    netCarry:$[direction~`long; swapCarryVal - futCarryVal; futCarryVal - swapCarryVal];
    netRoll:$[direction~`long; swapRollVal - futRollVal; futRollVal - swapRollVal];
    `swapCarry`swapRollDown`futuresCarry`futuresRollDown`netCarry`netRollDown`totalExpected`horizon!(
        swapCarryVal;swapRollVal;futCarryVal;futRollVal;netCarry;netRoll;netCarry+netRoll;horizon)}

// =============================================================================
// SCENARIO ANALYSIS
// =============================================================================

// Parallel rate shift scenario (internal, uses cached base price)
parallelShiftScenario_:{[curve;position;price0;shiftBps]
    newCurve:.swaps.bumpCurve[curve;shiftBps*0.0001];
    ctdBond:position`ctdBond;
    settleDate:curve[`config]`asOfDate;
    price1:.ctd.priceBond[newCurve;settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f];
    newFuturesPrice:position[`futuresPrice] + (price1 - price0) % ctdBond`cf;
    pnlDecomposition[curve;position;newCurve;newFuturesPrice]}

// Parallel rate shift scenario (public API)
parallelShiftScenario:{[curve;position;shiftBps;repoRate]
    ctdBond:position`ctdBond;
    settleDate:curve[`config]`asOfDate;
    price0:.ctd.priceBond[curve;settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f];
    parallelShiftScenario_[curve;position;price0;shiftBps]}

// Multiple parallel shift scenarios (parallelized)
parallelShiftScenarios:{[curve;position;shiftsBps;repoRate]
    // Cache base price once (avoid recalculating in each scenario)
    ctdBond:position`ctdBond;
    settleDate:curve[`config]`asOfDate;
    price0:.ctd.priceBond[curve;settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f];
    // Run scenarios in parallel (threshold=3 for parallelization benefit)
    results:pmap[parallelShiftScenario_[curve;position;price0;];3;shiftsBps];
    // Extract results (vectorized where possible)
    ([] shiftBps:shiftsBps;
        swapPnL:{x[`swapPnL]`pnl} each results;
        futuresPnL:{x[`futuresPnL]`signedPnL} each results;
        totalPnL:results@\:`totalPnL;
        spreadChangeBps:results@\:`spreadChangeBps)}

// Spread-specific scenario (swap-Treasury basis move)
// Moves swap rates while keeping futures price constant
spreadScenario:{[curve;position;spreadChangeBps;repoRate]
    newCurve:.swaps.bumpCurve[curve;spreadChangeBps*0.0001];
    pnlDecomposition[curve;position;newCurve;position`futuresPrice]}

// Multiple spread scenarios (parallelized)
spreadScenarios:{[curve;position;spreadChangesBps;repoRate]
    // Run scenarios in parallel
    results:pmap[spreadScenario[curve;position;;repoRate];3;spreadChangesBps];
    ([] spreadChangeBps:spreadChangesBps;
        swapPnL:{x[`swapPnL]`pnl} each results;
        futuresPnL:{x[`futuresPnL]`signedPnL} each results;
        totalPnL:results@\:`totalPnL)}

// =============================================================================
// RICH/CHEAP ANALYSIS
// =============================================================================

// Compare current spread to historical range
richCheap:{[currentSpread;history;lookback]
    // Filter to recent history
    recent:select from history where date >= .z.d - lookback;
    if[0 = count recent;
        :`zScore`percentile`min`max`mean`std`current`status!(0n;0n;0n;0n;0n;0n;currentSpread;`unknown)];
    spreads:recent`invoiceSpread;
    mn:min spreads;
    mx:max spreads;
    mean:avg spreads;
    std:dev spreads;
    zScore:$[std > 0; (currentSpread - mean) % std; 0f];
    // Percentile rank
    pctile:(sum spreads < currentSpread) % count spreads;
    // Status
    status:$[zScore > 2; `rich;
             zScore < -2; `cheap;
             zScore > 1; `slightlyRich;
             zScore < -1; `slightlyCheap;
             `fair];
    `zScore`percentile`min`max`mean`std`current`status!(zScore;pctile;mn;mx;mean;std;currentSpread;status)}

// Add observation to history
addToHistory:{[history;curve;deliveryDate;futuresPrice;ctdBond;freq]
    settleDate:curve[`config]`asOfDate;
    spread:invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq];
    // Calculate z-score if history exists
    recent:select from history where date >= settleDate - 252;
    zsc:$[count[recent] >= 20;
        (spread[`invoiceSpread] - avg recent`invoiceSpread) % dev recent`invoiceSpread;
        0n];
    pctile:$[count[recent] >= 20;
        (sum recent[`invoiceSpread] < spread`invoiceSpread) % count recent;
        0n];
    newRow:([] date:enlist settleDate;
        futuresPrice:enlist futuresPrice;
        invoiceYield:enlist spread`invoiceYield; swapRate:enlist spread`swapRate;
        invoiceSpread:enlist spread`invoiceSpread; spreadBps:enlist spread`spreadBps;
        zScore:enlist zsc; percentile:enlist pctile);
    history,newRow}

// Create empty history table
newHistory:{[]
    ([] date:`date$();
        futuresPrice:`float$();
        invoiceYield:`float$(); swapRate:`float$();
        invoiceSpread:`float$(); spreadBps:`float$();
        zScore:`float$(); percentile:`float$())}

// =============================================================================
// FULL ANALYSIS SUITE
// =============================================================================

// Complete invoice spread analysis (notional-based)
analyze:{[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;repoRate;freq]
    settleDate:curve[`config]`asOfDate;
    // Core spread
    spread:invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq];
    // Hedge ratio (using forward DV01)
    hr:hedgeRatio[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq];
    // Create position
    pos:createPosition[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq;`long];
    // Carry analysis (3M horizon)
    carry:spreadCarry[curve;pos;repoRate;`3M];
    // Scenarios
    scenarios:parallelShiftScenarios[curve;pos;-50 -25 0 25 50;repoRate];
    // Spread scenarios
    spreadScens:spreadScenarios[curve;pos;-25 -10 0 10 25;repoRate];
    // Package result
    `spread`hedgeRatio`position`carry`parallelScenarios`spreadScenarios`settleDate`deliveryDate`ctdBond!(
        spread;hr;pos;carry;scenarios;spreadScens;settleDate;deliveryDate;ctdBond)}

// =============================================================================
// DISPLAY FUNCTIONS
// =============================================================================

// Format number with commas
fmtNum:{[n]
    s:string `int$n;
    neg_:s[0]="-";
    s:$[neg_; 1_ s; s];
    len:count s;
    if[len <= 3; :$[neg_; "-",s; s]];
    // Pad to multiple of 3
    padLen:(3 - len mod 3) mod 3;
    padded:(padLen # "0"),s;
    // Split into groups of 3 and join with commas
    grouped:"," sv 3 cut padded;
    // Drop leading zeros
    res:padLen _ grouped;
    $[neg_; "-",res; res]}

// Show spread analysis
showAnalysis:{[result]
    spread:result`spread;
    hr:result`hedgeRatio;
    carry:result`carry;
    pos:result`position;
    -1 "";
    -1 "=============================================================================";
    -1 "                       INVOICE SPREAD ANALYSIS";
    -1 "=============================================================================";
    -1 "";
    -1 "Settlement Date:   ",string result`settleDate;
    -1 "Delivery Date:     ",string result`deliveryDate;
    -1 "CTD Bond:          ",string result[`ctdBond]`sym;
    -1 "CTD Maturity:      ",string result[`ctdBond]`maturity;
    -1 "CTD Coupon:        ",(string 100*result[`ctdBond]`coupon),"%";
    -1 "CTD CF:            ",string result[`ctdBond]`cf;
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "INVOICE SPREAD";
    -1 "-----------------------------------------------------------------------------";
    -1 "Invoice Yield:     ",(string 0.01 * `int$10000*spread`invoiceYield),"%";
    -1 "Swap Rate:         ",(string 0.01 * `int$10000*spread`swapRate),"%";
    -1 "Invoice Spread:    ",(string `int$spread`spreadBps)," bps (swaps ",$[(spread`invoiceSpread)>0;"wide";"tight"],")";
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "POSITION";
    -1 "-----------------------------------------------------------------------------";
    -1 "Direction:         ",string pos`direction;
    -1 "Swap Notional:     $",fmtNum pos`swapNotional;
    -1 "Futures Notional:  $",fmtNum pos`futuresNotional;
    -1 "Futures Price:     ",string pos`futuresPrice;
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "HEDGE RATIO";
    -1 "-----------------------------------------------------------------------------";
    -1 "Swap DV01:         $",(fmtNum hr`swapDV01);
    -1 "Futures DV01:      $",(fmtNum hr`futuresDV01)," per $1MM";
    -1 "Swap Notional:     $",fmtNum hr`swapNotional;
    -1 "Futures Notional:  $",fmtNum hr`futuresNotional;
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "CARRY ANALYSIS (3M)";
    -1 "-----------------------------------------------------------------------------";
    -1 "Swap Carry:        $",(fmtNum carry`swapCarry);
    -1 "Swap Roll-Down:    $",(fmtNum carry`swapRollDown);
    -1 "Futures Carry:     $",(fmtNum carry`futuresCarry);
    -1 "Futures Roll-Down: $",(fmtNum carry`futuresRollDown);
    -1 "Net Carry:         $",(fmtNum carry`netCarry);
    -1 "Net Roll-Down:     $",(fmtNum carry`netRollDown);
    -1 "Total Expected:    $",(fmtNum carry`totalExpected);
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "PARALLEL SHIFT SCENARIOS";
    -1 "-----------------------------------------------------------------------------";
    show result`parallelScenarios;
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "SPREAD SCENARIOS (Swap-Treasury Basis)";
    -1 "-----------------------------------------------------------------------------";
    show result`spreadScenarios;
    -1 "";
    -1 "=============================================================================";
    -1 "";
    result}

// Show rich/cheap analysis
showRichCheap:{[result]
    -1 "";
    -1 "=============================================================================";
    -1 "                        RICH/CHEAP ANALYSIS";
    -1 "=============================================================================";
    -1 "";
    -1 "Current Spread:   ",(string `int$10000*result`current)," bps";
    -1 "Historical Mean:  ",(string `int$10000*result`mean)," bps";
    -1 "Historical Std:   ",(string `int$10000*result`std)," bps";
    -1 "Z-Score:          ",(string 0.01*`int$100*result`zScore);
    -1 "Percentile:       ",(string `int$100*result`percentile),"%";
    -1 "Status:           ",string result`status;
    -1 "Historical Range: ",(string `int$10000*result`min)," to ",(string `int$10000*result`max)," bps";
    -1 "";
    -1 "=============================================================================";
    -1 "";
    result}

// Show position
showPosition:{[position]
    -1 "";
    -1 "=============================================================================";
    -1 "                        INVOICE SPREAD POSITION";
    -1 "=============================================================================";
    -1 "";
    -1 "Direction:         ",string position`direction;
    -1 "Delivery Date:     ",string position`deliveryDate;
    -1 "Futures Price:     ",string position`futuresPrice;
    -1 "Swap Notional:     $",fmtNum position`swapNotional;
    -1 "Futures Notional:  $",fmtNum position`futuresNotional;
    -1 "";
    -1 "CTD Bond:          ",string position[`ctdBond]`sym;
    -1 "CTD Coupon:        ",(string 100*position[`ctdBond]`coupon),"%";
    -1 "CTD Maturity:      ",string position[`ctdBond]`maturity;
    -1 "CTD CF:            ",string position[`ctdBond]`cf;
    -1 "";
    -1 "Spread:            ",(string `int$position[`spread]`spreadBps)," bps";
    -1 "";
    -1 "=============================================================================";
    -1 "";
    position}

// =============================================================================
// DAILY INVOICE SPREAD FUNCTIONS
// =============================================================================
// These functions integrate with tsy.q baskets and daily price data to compute
// invoice spreads and OAS across multiple contracts and dates.
//
// Inputs (same pattern as .tsy.dailyCTD):
//   baskets: from .tsy.allBasketsRange[] - table with deliveryCode, contract, deliveryMonth, cusip, cf, etc.
//   prices: ([] date; cusip; price) - daily bond prices
//   futures: ([] date; contract; deliveryMonth; price) - daily futures prices
//            OR ([] date; deliveryCode; price) where deliveryCode = e.g. `TYH25
//   curves: one of:
//           - static curve dict (used for all dates)
//           - builder function {[dt] ...} returning curve for date
//           - table ([] date; curve) with pre-built curves per date
//   repoRate: scalar OR table ([] date; rate)
//   vol: annualized rate volatility for OAS (e.g., 0.01 = 100bp/year)
//   useLastDelivery: optional, 0b (default) = first delivery day, 1b = last delivery day
//                   `both or 2 = run both and combine results with deliveryType column

// Helper: convert basket + prices to ctdBonds format (same as tsy.toCtdBonds)
toCtdBondsInternal:{[basket;priceDict]
    {[pd;row]
        p:pd row`cusip;
        if[null p; :()];
        `sym`coupon`maturity`cleanPrice`cf!(
            row`cusip;
            row`coupon;
            row`maturityDate;
            p;
            row`cf)
    }[priceDict;] each basket}

// Helper: analyze one delivery code for invoice spread
dailyISAnalyzeOne:{[ctx;curve;dayPrices;futPrices;repo;freq;dt;dc]
    basket:ctx[`basketsByCode][dc];
    if[0=count basket; :()];

    if[not dc in key futPrices; :()];
    futPrice:futPrices dc;
    if[null futPrice; :()];

    ctdBonds:toCtdBondsInternal[basket;dayPrices];
    ctdBonds:ctdBonds where not (::)~/: ctdBonds;
    if[0=count ctdBonds; :()];

    deliveryDate:ctx[`deliveryDateByCode][dc];

    // Run CTD analysis to find the CTD
    ctdResults:.ctd.ctdAnalysis[curve;dt;deliveryDate;futPrice;repo;ctdBonds];
    if[0=count ctdResults; :()];

    ctdRow:first select from ctdResults where isCTD;
    if[0=count ctdRow; :()];

    // Get CTD bond dict (find by sym match)
    ctdBond:first ctdBonds where (ctdRow`sym)~/: ctdBonds@\:`sym;
    if[(::)~ctdBond; :()];

    // Calculate invoice spread for CTD
    spread:invoiceSpread[curve;deliveryDate;futPrice;ctdBond;freq];

    // Calculate hedge ratio (using forward DV01)
    hr:hedgeRatio[curve;deliveryDate;futPrice;ctdBond;1000000f;freq];

    // Build result row
    ([] date:enlist dt;
        deliveryCode:enlist dc;
        contract:enlist ctx[`contractByCode][dc];
        deliveryMonth:enlist ctx[`monthByCode][dc];
        deliveryDate:enlist deliveryDate;
        ctdSym:enlist ctdBond`sym;
        ctdCoupon:enlist ctdBond`coupon;
        ctdMaturity:enlist ctdBond`maturity;
        ctdCF:enlist ctdBond`cf;
        futuresPrice:enlist futPrice;
        invoiceYield:enlist spread`invoiceYield;
        swapRate:enlist spread`swapRate;
        invoiceSpread:enlist spread`invoiceSpread;
        spreadBps:enlist spread`spreadBps;
        swapDV01:enlist hr`swapDV01;
        futuresDV01:enlist hr`futuresDV01;
        impliedRepo:enlist ctdRow`impliedRepo)}

// =============================================================================
// DAILY INVOICE SPREAD FROM PRE-COMPUTED CTD
// =============================================================================
// Simplified daily invoice spread using pre-computed CTD table.
// This avoids recomputing CTD analysis - you supply the CTD bonds directly.
//
// Required columns in ctdTable:
//   date         - valuation date
//   deliveryDate - futures delivery date
//   futuresPrice - futures price
//   sym          - CTD bond identifier
//   coupon       - CTD coupon (decimal, e.g., 0.045)
//   maturity     - CTD maturity date
//   cf           - conversion factor
//   cleanPrice   - CTD clean price (optional if you want hedge ratio)
//
// Optional columns (passed through to output):
//   deliveryCode, contract, deliveryMonth, impliedRepo, etc.

// Helper: process one row of CTD table
dailyISFromCTDRow:{[curve;freq;hasCleanPrice;row]
    // Build ctdBond dict
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        row`sym; row`coupon; row`maturity;
        $[hasCleanPrice; row`cleanPrice; 100f];
        row`cf);

    // Calculate invoice spread
    spread:invoiceSpread[curve;row`deliveryDate;row`futuresPrice;ctdBond;freq];

    // Calculate hedge ratio (using forward DV01)
    hr:hedgeRatio[curve;row`deliveryDate;row`futuresPrice;ctdBond;1000000f;freq];

    // Return result dict
    `invoiceYield`swapRate`invoiceSpread`spreadBps`swapDV01`futuresDV01!(
        spread`invoiceYield; spread`swapRate; spread`invoiceSpread;
        spread`spreadBps; hr`swapDV01; hr`futuresDV01)}

// Daily invoice spread from pre-computed CTD table (row-by-row, slower)
// Use dailyInvoiceSpreadFromCTD for better performance
dailyInvoiceSpreadFromCTDSlow:{[ctdTable;curves;freq]
    if[0=count ctdTable; :ctdTable];

    // Handle frequency default
    frq:$[freq~(::); `6M; freq];

    // Get unique dates
    dates:asc distinct ctdTable`date;
    nDates:count dates;

    // Pre-build curves by date
    curveType:$[99h=type curves; `dict;
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `dict];

    curvesByDate:$[curveType=`dict; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // Check if cleanPrice column exists
    hasCleanPrice:`cleanPrice in cols ctdTable;

    // Process each row
    results:{[cbd;frq;hcp;row]
        curve:cbd row`date;
        dailyISFromCTDRow[curve;frq;hcp;row]
    }[curvesByDate;frq;hasCleanPrice;] each ctdTable;

    // Combine with original table
    // When count=1, results is already a table (type 98)
    // When count>1, results is a list of dicts - flip to make table
    resultTable:$[98h=type results; results; flip results];
    ctdTable,'resultTable}

// Optimized helper: process all rows for one date (batched DV01 computation)
// Uses forward modified duration for futures DV01 (much faster than curve bumping)
dailyISFromCTDBatch:{[curve;freq;hasCleanPrice;rows]
    if[0=count rows; :rows];
    deliveryDate:first rows`deliveryDate;

    // Build bond dicts for all rows
    ctdBonds:{[hcp;r] `sym`coupon`maturity`cleanPrice`cf!(r`sym;r`coupon;r`maturity;$[hcp;r`cleanPrice;100f];r`cf)}[hasCleanPrice;] each rows;

    // Batch invoice yields (vectorizable via Newton-Raphson)
    futPrices:rows`futuresPrice;
    invYlds:invoiceYield[deliveryDate;;]'[futPrices;ctdBonds];

    // Batch swap rates - one per unique maturity
    mats:rows`maturity;
    uniqueMats:distinct mats;
    swpRtByMat:uniqueMats!swapRate[curve;deliveryDate;;freq] each uniqueMats;
    swpRts:swpRtByMat mats;

    // Invoice spreads
    spreads:swpRts - invYlds;
    spreadsBps:spreads * 10000;

    // Batch futures DV01 using forward modified duration (no curve bumping needed!)
    // futuresDV01 = forwardModDur * futuresPrice
    futDV01s:futuresDV01Batch[deliveryDate;futPrices;ctdBonds];

    // Batch swap DV01s - one per unique maturity
    asOf:curve[`config]`asOfDate;
    dc:curve[`config]`dayCount;
    swpDV01ByMat:uniqueMats!{[c;ao;dc;dd;frq;m]
        startYF:.swaps.dcfDates[ao;dd;dc];
        endYF:.swaps.dcfDates[ao;m;dc];
        parRt:.swaps.parRateDated[c;dd;m;frq];
        .swaps.dv01[c;1000000f;parRt;endYF;frq;1b]
    }[curve;asOf;dc;deliveryDate;freq;] each uniqueMats;
    swpDV01s:swpDV01ByMat mats;

    // Return results
    ([] invoiceYield:invYlds; swapRate:swpRts; invoiceSpread:spreads;
        spreadBps:spreadsBps; swapDV01:swpDV01s; futuresDV01:futDV01s)}

// Daily invoice spread from pre-computed CTD table
// Optimized: batches by date, ~3x faster than row-by-row
// ctdTable: table with date, deliveryDate, futuresPrice, sym, coupon, maturity, cf
// curves: static dict, builder {[dt]...}, or ([] date; curve)
// freq: payment frequency (`6M default)
// Supports parallel processing when secondary threads available (\s N)
dailyInvoiceSpreadFromCTD:{[ctdTable;curves;freq]
    if[0=count ctdTable; :ctdTable];

    // Handle frequency default
    frq:$[freq~(::); `6M; freq];

    // Get unique dates
    dates:asc distinct ctdTable`date;
    nDates:count dates;

    // Pre-build curves by date
    curveType:$[99h=type curves; `dict;
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `dict];

    curvesByDate:$[curveType=`dict; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // Check if cleanPrice column exists
    hasCleanPrice:`cleanPrice in cols ctdTable;

    // Group rows by date
    rowsByDate:ctdTable group ctdTable`date;

    // Process each date in batch (parallel if threads available)
    processDate:{[cbd;frq;hcp;dtrows]
        dt:dtrows 0; rows:dtrows 1;
        curve:cbd dt;
        dailyISFromCTDBatch[curve;frq;hcp;rows]
    }[curvesByDate;frq;hasCleanPrice;];

    // Process dates - parallel if available, sequential otherwise
    dateRowPairs:flip (key rowsByDate; value rowsByDate);
    resultsByDate:$[useParallel and nDates >= 3;
        processDate peach dateRowPairs;
        processDate each dateRowPairs];

    // Reconstruct table in original order
    results:raze resultsByDate;

    // Reorder to match original ctdTable order
    idx:ctdTable`date;
    order:rank idx;
    sortedResults:results iasc order;

    ctdTable,'sortedResults}

// =============================================================================
// DAILY OAS FROM PRE-COMPUTED CTD + BASKET
// =============================================================================
// Option-adjusted spread using pre-computed CTD table and basket.
// Skips CTD identification but needs basket for option value calculation.
//
// ctdTable columns (one row per date/deliveryCode):
//   date, deliveryCode, deliveryDate, futuresPrice, sym, coupon, maturity, cf
//   Optional: cleanPrice (defaults to 100)
//
// basketTable columns (all deliverable bonds per date/deliveryCode):
//   date, deliveryCode, sym, coupon, maturity, cf, cleanPrice

// Helper: build bond dict from row (hasCP flag passed separately)
rowToBondWithFlag:{[hasCP;row]
    `sym`coupon`maturity`cleanPrice`cf!(
        row`sym; row`coupon; row`maturity;
        $[hasCP; row`cleanPrice; 100f];
        row`cf)}

// Internal: get basket from cache, apply daily prices if needed
basketFromCache_:{[ctx;dc;dt]
    cachedBasket:ctx[`basketCache] dc;
    if[0 = count ctx`pricesByDateSym; :cachedBasket];
    {[px;dt;b]
        p:px (dt;b`sym);
        $[null p; b; @[b;`cleanPrice;:;p]]
    }[ctx`pricesByDateSym;dt] each cachedBasket}

// Internal: get basket from table (no cache), build bond dicts
basketFromTable_:{[ctx;dc;dt]
    basketRows:$[ctx`hasBasketDateCol;
        select from ctx[`basketTable] where date=dt, deliveryCode=dc;
        select from ctx[`basketTable] where deliveryCode=dc];
    if[(0 < count basketRows) and (0 < count ctx`pricesByDateSym);
        pxBySym:{[px;d;s] px (d;s)}[ctx`pricesByDateSym;dt;] each basketRows`sym;
        basketRows:$[`cleanPrice in cols basketRows;
            update cleanPrice:{$[null x; y; x]}'[pxBySym;cleanPrice] from basketRows;
            update cleanPrice:pxBySym from basketRows]];
    if[0 = count basketRows; :()];
    hasCleanPrice:`cleanPrice in cols basketRows;
    rowToBondWithFlag[hasCleanPrice;] each basketRows}

// Helper: process one CTD row for OAS
// Supports both OIS curves (from buildOISCurves) and swap curves (from .swaps.buildCurve)
dailyOASFromCTDRow:{[ctx;row]
    dt:row`date;
    dc:row`deliveryCode;
    curve:ctx[`curvesByDate] dt;
    freq:ctx`freq;
    isOIS:ctx`isOIS;

    // Get contract from deliveryCode if not in row
    contract:$[`contract in key row; row`contract; .tsy.contractFromCode dc];

    // Get repo - may be by date, by (date,contract), or scalar
    repoRate:$[`repoByDateContract in key ctx;
        ctx[`repoByDateContract] (dt;contract);
        `repoByDate in key ctx;
        ctx[`repoByDate] dt;
        ctx`repo];

    // Get vol - may be by date, by (date,contract), or scalar
    vol:$[`volByDateContract in key ctx;
        ctx[`volByDateContract] (dt;contract);
        `volByDate in key ctx;
        ctx[`volByDate] dt;
        ctx`vol];

    // Build CTD bond dict
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        row`sym; row`coupon; row`maturity;
        $[ctx`hasCtdCleanPrice; row`cleanPrice; 100f];
        row`cf);

    // Get delivery dates - for OIS we need both first and last
    deliveryDate:row`deliveryDate;
    firstDelivery:$[`firstDelivery in key row; row`firstDelivery; deliveryDate];
    lastDelivery:$[`lastDelivery in key row; row`lastDelivery; deliveryDate];
    delivery:`first`last!(firstDelivery;lastDelivery);

    // Get basket bond dicts - use cache when available
    useCache:(0 < count ctx`basketCache) and dc in key ctx`basketCache;
    basket:$[useCache;
        basketFromCache_[ctx;dc;dt];
        basketFromTable_[ctx;dc;dt]];

    if[0=count basket;
        // No basket - return invoice spread only, no option value
        spread:$[isOIS;
            invoiceSpreadWithOIS[curve;dt;deliveryDate;row`futuresPrice;ctdBond];
            invoiceSpread[curve;deliveryDate;row`futuresPrice;ctdBond;freq]];
        hr:$[isOIS;
            `swapDV01`futuresDV01!(0f;0f);
            hedgeRatio[curve;deliveryDate;row`futuresPrice;ctdBond;1000000f;freq]];
        :`invoiceYield`swapRate`invoiceSpread`spreadBps`qualityOption`timingOption`wildcardOption`optionValue`optionAdjBps`oas`oasBps`swapDV01`futuresDV01`basketSize!(
            spread`invoiceYield; spread`swapRate; spread`invoiceSpread; spread`spreadBps;
            0f; 0f; 0f; 0f; 0f; spread`invoiceSpread; spread`spreadBps;
            hr`swapDV01; hr`futuresDV01; 0)];

    // Calculate OAS - use market-based function for OIS curves
    oasResult:$[isOIS;
        oasInvoiceSpreadMarket[curve;delivery;row`futuresPrice;ctdBond;basket;repoRate;vol;`SA];
        oasInvoiceSpread[curve;deliveryDate;row`futuresPrice;ctdBond;basket;repoRate;vol;freq]];

    // Calculate hedge ratio (using forward DV01)
    hr:$[isOIS;
        `swapDV01`futuresDV01!(0f;0f);
        hedgeRatio[curve;deliveryDate;row`futuresPrice;ctdBond;1000000f;freq]];

    // Return - handle different output formats between OIS and swap
    optVal:$[isOIS; oasResult`totalOption; oasResult`optionValue];
    optAdj:oasResult`optionAdjBps;
    // Option breakdown (only available for OIS/market-based)
    qualityOpt:$[isOIS; oasResult`qualityOption; optVal];
    timingOpt:$[isOIS; oasResult`timingOption; 0f];
    wildcardOpt:$[isOIS; oasResult`wildcardOption; 0f];

    `invoiceYield`swapRate`invoiceSpread`spreadBps`qualityOption`timingOption`wildcardOption`optionValue`optionAdjBps`oas`oasBps`swapDV01`futuresDV01`basketSize!(
        oasResult`invoiceYield; oasResult`swapRate; oasResult`invoiceSpread; oasResult`spreadBps;
        qualityOpt; timingOpt; wildcardOpt;
        optVal; optAdj; oasResult`oas; oasResult`oasBps;
        hr`swapDV01; hr`futuresDV01; count basket)}

// Daily OAS from pre-computed CTD table and basket
// Supports both OIS curves (from buildOISCurves) and swap curves (from .swaps.buildCurve)
//
// ctdTable: table with date, deliveryCode, deliveryDate, futuresPrice, sym, coupon, maturity, cf
//           Optional: firstDelivery, lastDelivery (for timing option with OIS curves)
// basketTable: table with deliveryCode, sym, coupon, maturity, cf (static - no date column)
// bondPx: ([] date; sym; cleanPrice) - daily bond prices, joined to basket
//         OR (::) to use cleanPrice from basketTable directly (basketTable must have cleanPrice)
// curves: OIS curves dict (date -> oisCurve from buildOISCurves)
//         OR swap curves: static dict, builder {[dt]...}, or ([] date; curve)
// repoRate: scalar or ([] date; rate)
// vol: scalar, ([] date; vol), or ([] date; contract; vol) for contract-level
// freq: payment frequency (`6M for swap curves, ignored for OIS which uses `SA)
dailyOASFromCTD:{[ctdTable;basketTable;bondPx;curves;repoRate;vol;freq]
    if[0=count ctdTable; :ctdTable];

    // Add delivery dates if missing (parses deliveryCode to get firstDelivery/lastDelivery)
    ctdTable:addDeliveryDates ctdTable;

    // Handle defaults
    frq:$[freq~(::); `6M; freq];

    // basketTable should NOT have date column - it's static per deliveryCode
    // bondPx provides daily prices: ([] date; sym; cleanPrice)
    // Prices are joined per-row in dailyOASFromCTDRow
    hasBasketDateCol:`date in cols basketTable;
    hasBondPx:not (bondPx~(::)) and (0 < count bondPx);

    // Build price lookup: (date;sym) -> cleanPrice
    pricesByDateSym:$[hasBondPx;
        (exec (date,'sym)!cleanPrice from bondPx);
        ()!()];

    // Get unique dates from CTD table
    dates:asc distinct ctdTable`date;
    nDates:count dates;

    // Pre-build curves by date
    // Handle different curve input formats:
    // - dict with date keys (from buildOISCurves): date -> oisCurve
    // - dict with curve keys (single swap curve): replicate for all dates
    // - builder function: call for each date
    // - table: ([] date; curve)
    curveType:$[99h=type curves;
                    // Dict - check if keys are dates (type 14h) or curve keys (symbols)
                    // buildOISCurves returns date -> curve dict
                    // Single swap curve has symbol keys like `config`tenors`zeroRates
                    $[14h=type key curves; `dateDict; `singleCurve];
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `singleCurve];

    curvesByDate:$[curveType=`dateDict; dates!curves dates;
                   curveType=`singleCurve; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // Detect if OIS curves (have `curveDate) or swap curves (have `config)
    sampleCurve:first value curvesByDate;
    isOIS:`curveDate in key sampleCurve;

    // Build repo lookup - supports scalar, dict, table by date, table by (date,contract)
    repoCtx:$[repoRate~(::);
        // Default
        enlist[`repo]!enlist 0.0375;
        -9h=type repoRate;
        // Scalar
        enlist[`repo]!enlist repoRate;
        99h=type repoRate;
        // Dict by date
        enlist[`repoByDate]!enlist repoRate;
        98h=type repoRate;
        // Table - check if has contract column
        $[`contract in cols repoRate;
            // By (date, contract)
            enlist[`repoByDateContract]!enlist (exec (date,'contract)!rate from repoRate);
            // By date only
            enlist[`repoByDate]!enlist (exec date!rate from repoRate)];
        // Default scalar
        enlist[`repo]!enlist repoRate];

    // Build vol lookup - same structure as repo
    volCtx:$[vol~(::);
        // Default
        enlist[`vol]!enlist 0.01;
        -9h=type vol;
        // Scalar
        enlist[`vol]!enlist vol;
        99h=type vol;
        // Dict by date
        enlist[`volByDate]!enlist vol;
        98h=type vol;
        // Table - check if has contract column
        $[`contract in cols vol;
            // By (date, contract)
            enlist[`volByDateContract]!enlist (exec (date,'contract)!vol from vol);
            // By date only
            enlist[`volByDate]!enlist (exec date!vol from vol)];
        // Default scalar
        enlist[`vol]!enlist vol];

    // Check if cleanPrice column exists
    hasCtdCleanPrice:`cleanPrice in cols ctdTable;
    hasBasketCleanPrice:`cleanPrice in cols basketTable;

    // Pre-build basket cache by deliveryCode (avoids repeated select+dict construction)
    hasBondPxPrices:0 < count pricesByDateSym;
    basketCache:$[hasBasketDateCol;
        ()!();  // can't cache when basket varies by date
        [dcs:distinct basketTable`deliveryCode;
         hasCP:`cleanPrice in cols basketTable;
         dcs!{[bt;hasCP;dc]
             rows:select from bt where deliveryCode=dc;
             rowToBondWithFlag[hasCP;] each rows
         }[basketTable;hasCP] each dcs]];

    // Pack context
    ctx:`curvesByDate`basketTable`pricesByDateSym`freq`hasCtdCleanPrice`hasBasketCleanPrice`hasBasketDateCol`isOIS`basketCache!(
        curvesByDate;basketTable;pricesByDateSym;frq;hasCtdCleanPrice;hasBasketCleanPrice;hasBasketDateCol;isOIS;basketCache);
    ctx:ctx,repoCtx,volCtx;

    // Process each CTD row
    fn:dailyOASFromCTDRow[ctx;];
    results:$[useParallel and 10 < count ctdTable; fn peach ctdTable; fn each ctdTable];

    // Combine with original table
    resultTable:$[98h=type results; results; flip results];
    ctdTable,'resultTable}

// =============================================================================
// DAILY INVOICE SPREAD FROM BASKETS (full CTD recomputation)
// =============================================================================
// Use this version when you don't have pre-computed CTD data.
// It runs full CTD analysis internally.

// Daily invoice spread across all contracts and dates
// Returns: table with date, deliveryCode, contract, invoiceSpread columns
dailyInvoiceSpread:{[baskets;prices;futures;curves;repoRate;freq;useLastDelivery]
    // Handle `both option - run both first and last, combine results
    if[(useLastDelivery~`both) or (useLastDelivery~2);
        r1:dailyInvoiceSpread[baskets;prices;futures;curves;repoRate;freq;0b];
        r2:dailyInvoiceSpread[baskets;prices;futures;curves;repoRate;freq;1b];
        r1:update deliveryType:`first from r1;
        r2:update deliveryType:`last from r2;
        :`date`deliveryCode`deliveryType xasc r1,r2
    ];

    // Handle optional useLastDelivery parameter (default to first delivery day)
    useLast:$[useLastDelivery~(::); 0b; useLastDelivery];

    // Get unique dates from prices
    dates:asc distinct prices`date;
    nDates:count dates;

    // Get unique delivery codes
    deliveryCodes:distinct baskets`deliveryCode;

    // Pre-build curves by date
    curveType:$[99h=type curves; `dict;
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `dict];

    curvesByDate:$[curveType=`dict; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // Pre-index prices and futures by date
    pricesByDate:dates!{[p;dt] exec cusip!price from p where date=dt}[prices;] each dates;

    hasDC:`deliveryCode in cols futures;
    futuresByDate:$[hasDC;
        dates!{[f;dt] exec deliveryCode!price from f where date=dt}[futures;] each dates;
        dates!{[f;dt] exec deliveryCode!price from update deliveryCode:`$string[contract],'string[deliveryMonth] from f where date=dt}[futures;] each dates];

    // Pre-compute repo by date
    repoByDate:$[-9h=type repoRate; dates!nDates#repoRate;
                 99h=type repoRate; dates!nDates#repoRate`rate;
                 98h=type repoRate; dates!(exec date!rate from repoRate) dates;
                 dates!nDates#repoRate];

    // Pre-filter baskets by code
    basketsByCode:deliveryCodes!{[b;dc] select from b where deliveryCode=dc}[baskets;] each deliveryCodes;

    // Pre-compute delivery dates by code (TU/FV use 3rd biz day for last delivery)
    contractByCode:deliveryCodes!{[bc;dc] first bc[dc]`contract}[basketsByCode;] each deliveryCodes;
    deliveryDateByCode:$[useLast;
        deliveryCodes!{[bc;cc;dc] .tsy.lastDeliveryDayForContract[cc dc; first bc[dc]`deliveryMonth]}[basketsByCode;contractByCode;] each deliveryCodes;
        deliveryCodes!{[bc;dc] .tsy.firstDeliveryDay first bc[dc]`deliveryMonth}[basketsByCode;] each deliveryCodes];
    monthByCode:deliveryCodes!{[bc;dc] first bc[dc]`deliveryMonth}[basketsByCode;] each deliveryCodes;

    // Pack context
    ctx:`basketsByCode`pricesByDate`futuresByDate`curvesByDate`repoByDate`deliveryDateByCode`contractByCode`monthByCode!
        (basketsByCode;pricesByDate;futuresByDate;curvesByDate;repoByDate;deliveryDateByCode;contractByCode;monthByCode);

    // Handle frequency default
    frq:$[freq~(::); `6M; freq];

    // Process by date
    analyzeDate:{[ctx;deliveryCodes;frq;dt]
        curve:ctx[`curvesByDate][dt];
        dayPrices:ctx[`pricesByDate][dt];
        futPrices:ctx[`futuresByDate][dt];
        repo:ctx[`repoByDate][dt];
        results:dailyISAnalyzeOne[ctx;curve;dayPrices;futPrices;repo;frq;dt;] each deliveryCodes;
        results:results where not (::)~/: results;
        if[0=count results; :()];
        raze results}[ctx;deliveryCodes;frq;] each dates;

    result:raze analyzeDate where not (::)~/: analyzeDate;
    if[0=count result; :([] date:`date$(); deliveryCode:`$(); contract:`$(); deliveryMonth:`month$();
        deliveryDate:`date$(); ctdSym:`$(); ctdCoupon:`float$(); ctdMaturity:`date$();
        ctdCF:`float$(); futuresPrice:`float$(); invoiceYield:`float$(); swapRate:`float$();
        invoiceSpread:`float$(); spreadBps:`float$(); swapDV01:`float$(); futuresDV01:`float$();
        impliedRepo:`float$())];
    `date`deliveryCode xasc result}

// Helper: analyze one delivery code for OAS
// Uses context dict to stay within 8 parameter limit
// ctx must include: basketsByCode, deliveryDateByCode, contractByCode, monthByCode, vol, freq
dailyOASAnalyzeOne:{[ctx;curve;dayPrices;futPrices;repo;dt;dc]
    basket:ctx[`basketsByCode][dc];
    if[0=count basket; :()];

    if[not dc in key futPrices; :()];
    futPrice:futPrices dc;
    if[null futPrice; :()];

    ctdBonds:toCtdBondsInternal[basket;dayPrices];
    ctdBonds:ctdBonds where not (::)~/: ctdBonds;
    if[0=count ctdBonds; :()];

    deliveryDate:ctx[`deliveryDateByCode][dc];
    vol:ctx`vol;
    freq:ctx`freq;

    // Run CTD analysis to find the CTD
    ctdResults:.ctd.ctdAnalysis[curve;dt;deliveryDate;futPrice;repo;ctdBonds];
    if[0=count ctdResults; :()];

    ctdRow:first select from ctdResults where isCTD;
    if[0=count ctdRow; :()];

    // Get CTD bond dict (find by sym match)
    ctdBond:first ctdBonds where (ctdRow`sym)~/: ctdBonds@\:`sym;
    if[(::)~ctdBond; :()];

    // Calculate OAS for CTD with basket
    oasResult:oasInvoiceSpread[curve;deliveryDate;futPrice;ctdBond;ctdBonds;repo;vol;freq];

    // Build result row
    ([] date:enlist dt;
        deliveryCode:enlist dc;
        contract:enlist ctx[`contractByCode][dc];
        deliveryMonth:enlist ctx[`monthByCode][dc];
        deliveryDate:enlist deliveryDate;
        ctdSym:enlist ctdBond`sym;
        ctdCoupon:enlist ctdBond`coupon;
        ctdMaturity:enlist ctdBond`maturity;
        basketSize:enlist count ctdBonds;
        futuresPrice:enlist futPrice;
        invoiceYield:enlist oasResult`invoiceYield;
        swapRate:enlist oasResult`swapRate;
        invoiceSpread:enlist oasResult`invoiceSpread;
        spreadBps:enlist oasResult`spreadBps;
        optionValue:enlist oasResult`optionValue;
        optionAdjBps:enlist oasResult`optionAdjBps;
        oas:enlist oasResult`oas;
        oasBps:enlist oasResult`oasBps;
        impliedRepo:enlist ctdRow`impliedRepo)}

// Daily option-adjusted spread across all contracts and dates
// vol: annualized rate volatility (e.g., 0.01 = 100bp/year)
// Returns: table with date, deliveryCode, invoiceSpread, optionValue, oas columns
dailyOAS:{[baskets;prices;futures;curves;repoRate;vol;freq;useLastDelivery]
    // Handle `both option
    if[(useLastDelivery~`both) or (useLastDelivery~2);
        r1:dailyOAS[baskets;prices;futures;curves;repoRate;vol;freq;0b];
        r2:dailyOAS[baskets;prices;futures;curves;repoRate;vol;freq;1b];
        r1:update deliveryType:`first from r1;
        r2:update deliveryType:`last from r2;
        :`date`deliveryCode`deliveryType xasc r1,r2
    ];

    // Handle optional useLastDelivery parameter
    useLast:$[useLastDelivery~(::); 0b; useLastDelivery];

    // Get unique dates from prices
    dates:asc distinct prices`date;
    nDates:count dates;

    // Get unique delivery codes
    deliveryCodes:distinct baskets`deliveryCode;

    // Pre-build curves by date
    curveType:$[99h=type curves; `dict;
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `dict];

    curvesByDate:$[curveType=`dict; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // Pre-index prices and futures by date
    pricesByDate:dates!{[p;dt] exec cusip!price from p where date=dt}[prices;] each dates;

    hasDC:`deliveryCode in cols futures;
    futuresByDate:$[hasDC;
        dates!{[f;dt] exec deliveryCode!price from f where date=dt}[futures;] each dates;
        dates!{[f;dt] exec deliveryCode!price from update deliveryCode:`$string[contract],'string[deliveryMonth] from f where date=dt}[futures;] each dates];

    // Pre-compute repo by date
    repoByDate:$[-9h=type repoRate; dates!nDates#repoRate;
                 99h=type repoRate; dates!nDates#repoRate`rate;
                 98h=type repoRate; dates!(exec date!rate from repoRate) dates;
                 dates!nDates#repoRate];

    // Pre-filter baskets by code
    basketsByCode:deliveryCodes!{[b;dc] select from b where deliveryCode=dc}[baskets;] each deliveryCodes;

    // Pre-compute delivery dates by code (TU/FV use 3rd biz day for last delivery)
    contractByCode:deliveryCodes!{[bc;dc] first bc[dc]`contract}[basketsByCode;] each deliveryCodes;
    deliveryDateByCode:$[useLast;
        deliveryCodes!{[bc;cc;dc] .tsy.lastDeliveryDayForContract[cc dc; first bc[dc]`deliveryMonth]}[basketsByCode;contractByCode;] each deliveryCodes;
        deliveryCodes!{[bc;dc] .tsy.firstDeliveryDay first bc[dc]`deliveryMonth}[basketsByCode;] each deliveryCodes];
    monthByCode:deliveryCodes!{[bc;dc] first bc[dc]`deliveryMonth}[basketsByCode;] each deliveryCodes;

    // Handle defaults
    frq:$[freq~(::); `6M; freq];
    v:$[vol~(::); 0.01; vol];  // Default 100bp vol

    // Pack context (includes vol and freq for dailyOASAnalyzeOne)
    ctx:`basketsByCode`pricesByDate`futuresByDate`curvesByDate`repoByDate`deliveryDateByCode`contractByCode`monthByCode`vol`freq!
        (basketsByCode;pricesByDate;futuresByDate;curvesByDate;repoByDate;deliveryDateByCode;contractByCode;monthByCode;v;frq);

    // Process by date
    analyzeDate:{[ctx;deliveryCodes;dt]
        curve:ctx[`curvesByDate][dt];
        dayPrices:ctx[`pricesByDate][dt];
        futPrices:ctx[`futuresByDate][dt];
        repo:ctx[`repoByDate][dt];
        results:dailyOASAnalyzeOne[ctx;curve;dayPrices;futPrices;repo;dt;] each deliveryCodes;
        results:results where not (::)~/: results;
        if[0=count results; :()];
        raze results}[ctx;deliveryCodes;] each dates;

    result:raze analyzeDate where not (::)~/: analyzeDate;
    if[0=count result; :([] date:`date$(); deliveryCode:`$(); contract:`$(); deliveryMonth:`month$();
        deliveryDate:`date$(); ctdSym:`$(); ctdCoupon:`float$(); ctdMaturity:`date$();
        basketSize:`int$(); futuresPrice:`float$(); invoiceYield:`float$(); swapRate:`float$();
        invoiceSpread:`float$(); spreadBps:`float$(); optionValue:`float$(); optionAdjBps:`float$();
        oas:`float$(); oasBps:`float$(); impliedRepo:`float$())];
    `date`deliveryCode xasc result}

// =============================================================================
// DAILY INVOICE SPREAD WITH OIS CONVENTIONS
// =============================================================================
// These functions compute invoice spreads using proper market conventions:
//   - Bond Yield: US Treasury Street convention (semi-annual, period count, ACT/ACT AI)
//   - Swap Rate: OIS bootstrap (1Y fixed, ACT/360, daily SOFR compounded)
//
// Input tables:
//   ctdTable: ([] date; deliveryDate; futuresPrice; sym; coupon; maturity; cf)
//   oisRates: ([] date; rate1; rate2; ...; rateN) where columns are tenor rates
//             OR dict of date -> rate vector
//             OR static rate vector (same rates for all dates)
//   tenors: symbol list of OIS tenors matching rate columns
//
// Example:
//   tenors:`1D`1W`1M`3M`6M`1Y`2Y`5Y`7Y`10Y;
//   oisRates:([] date:dates; r1D; r1W; r1M; r3M; r6M; r1Y; r2Y; r5Y; r7Y; r10Y);
//   result:.invoicepricer.dailyInvoiceSpreadOIS[ctdTable; tenors; oisRates]

// Helper: process one row using OIS conventions
dailyISFromCTDRowOIS:{[oisCurve;curveDate;row]
    // Build ctdBond dict
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        row`sym; row`coupon; row`maturity; 100f; row`cf);

    // Invoice yield using Street convention
    invYld:invoiceYield[row`deliveryDate; row`futuresPrice; ctdBond];

    // Swap rate using OIS curve
    swpRt:oisFwdSwapRate[oisCurve; curveDate; row`deliveryDate; row`maturity];

    // Spread
    spread:swpRt - invYld;

    // Futures DV01 (forward modified duration)
    futDV01:forwardModDur[row`deliveryDate; row`futuresPrice; ctdBond] * row`futuresPrice;

    `invoiceYield`swapRate`invoiceSpread`spreadBps`futuresDV01!(
        invYld; swpRt; spread; spread * 10000; futDV01)}

// Helper: batch process all rows for one date
dailyISFromCTDBatchOIS:{[oisCurve;curveDate;rows]
    if[0=count rows; :rows];

    // Build bond dicts
    ctdBonds:{[r] `sym`coupon`maturity`cleanPrice`cf!(r`sym;r`coupon;r`maturity;100f;r`cf)} each rows;

    // Batch invoice yields
    deliveryDates:rows`deliveryDate;
    futPrices:rows`futuresPrice;
    invYlds:invoiceYield'[deliveryDates; futPrices; ctdBonds];

    // Batch swap rates - one per unique maturity (delivery date is same for batch)
    mats:rows`maturity;
    deliveryDate:first deliveryDates;
    uniqueMats:distinct mats;
    swpRtByMat:uniqueMats!oisFwdSwapRate[oisCurve;curveDate;deliveryDate;] each uniqueMats;
    swpRts:swpRtByMat mats;

    // Spreads
    spreads:swpRts - invYlds;
    spreadsBps:spreads * 10000;

    // Futures DV01s
    futDV01s:{[dd;fp;b] forwardModDur[dd;fp;b] * fp}'[deliveryDates; futPrices; ctdBonds];

    ([] invoiceYield:invYlds; swapRate:swpRts; invoiceSpread:spreads;
        spreadBps:spreadsBps; futuresDV01:futDV01s)}

// Daily invoice spread from CTD table using OIS conventions
// ctdTable: ([] date; deliveryDate; futuresPrice; sym; coupon; maturity; cf)
// oisCurves: dict of date -> oisCurve (built via buildOISCurves)
// Returns: ctdTable with invoiceYield, swapRate, spreadBps, futuresDV01 columns added
dailyInvoiceSpreadOIS:{[ctdTable;oisCurves]
    if[0=count ctdTable; :ctdTable];

    // Get unique dates
    dates:asc distinct ctdTable`date;
    nDates:count dates;

    // Group rows by date
    rowsByDate:ctdTable group ctdTable`date;

    // Process each date
    processDate:{[ocbd;dtrows]
        dt:dtrows 0; rows:dtrows 1;
        oisCurve:ocbd dt;
        dailyISFromCTDBatchOIS[oisCurve;dt;rows]
    }[oisCurves;];

    dateRowPairs:flip (key rowsByDate; value rowsByDate);
    resultsByDate:$[useParallel and nDates >= 3;
        processDate peach dateRowPairs;
        processDate each dateRowPairs];

    // Reconstruct in original order
    results:raze resultsByDate;
    idx:ctdTable`date;
    order:rank idx;
    sortedResults:results iasc order;

    ctdTable,'sortedResults}

// Convert a bootstrapped OIS curve (`yf`df`curveDate) to swaps.q curve format.
// This lets us reuse existing position, PnL, and carry machinery for OIS analytics.
oisCurveToSwapCurve:{[oisCurve]
    if[not all (`yf`df`curveDate in key oisCurve); '"oisCurve missing required keys: `yf`df`curveDate"];
    yfs:`float$oisCurve`yf;
    dfs:`float$oisCurve`df;
    if[0=count yfs; '"oisCurve has empty tenor grid"];
    ord:iasc yfs;
    yfs:yfs ord;
    dfs:dfs ord;
    zeros:{[df;t] .swaps.df2zero[df;t;`continuous]}'[dfs;yfs];
    tenors:.swaps.yfToTenor each yfs;
    cfg:.swaps.defaults,`asOfDate`dayCount`frequency`compounding`interpMethod`calibrate!(
        oisCurve`curveDate;`ACT360;`1Y;`continuous;`loglinear;0b);
    `tenors`yearFracs`discountFactors`zeroRates`parRates`zeroSpline`fwdSpline`config!(
        tenors;yfs;dfs;zeros;(count yfs)#0n;()!();()!();cfg)}

// Repo lookup helper for scalar/dict/table repo inputs.
// Supported tabular columns:
//   ([] date; repo)
//   ([] date; rate)
//   ([] date; repoRate)
//   ([] date; value)
repoForDate:{[repoInput;dt]
    if[repoInput~(::); :0n];
    dt0:$[(abs type dt) in 12 15h; date dt; dt];
    tr:type repoInput;
    atr:abs tr;
    if[(tr<0h) and (atr in 1 4 5 6 7 8 9h); :1f*repoInput];
    if[(tr>0h) and (atr in 1 4 5 6 7 8 9h);
        if[0=count repoInput; :0n];
        :1f*first repoInput];
    if[tr=99h;
        / keyed table -> plain table path
        if[98h=type key repoInput;
            repoInput:0!repoInput;
            tr:98h;
        ];
    ];
    if[tr=99h;
        ks:key repoInput;
        if[(abs type ks) in 12 15h; ks:date each ks];
        m:ks=dt0;
        if[any m; :1f*first (value repoInput) where m];
        :0n];
    if[tr=98h;
        cols_:cols repoInput;
        if[not `date in cols_; :0n];
        rt:repoInput;
        if[(abs type rt`date) in 12 15h; rt:update date:date date from rt];
        valCol:$[`repo in cols_; `repo;
            $[`rate in cols_; `rate;
                $[`repoRate in cols_; `repoRate;
                    $[`value in cols_; `value; `]]]];
        if[valCol~`; :0n];
        rr:rt where (rt`date)=dt0;
        if[0<count rr; :1f*first rr valCol];
        :0n;
    ];
    0n}

// Daily invoice spread with OIS conventions + realized day-over-day PnL and carry.
// Uses existing createPosition/pnlDecomposition/spreadCarry logic via OIS->swaps curve adapter.
// cfg options:
//   `swapNotional (default 1000000f; carry/roll notional, and used when notionalMode=`fixed)
//   `notionalMode (`perBpDV01 default, or `fixed)
//   `targetSwapDV01 (default 1f; used when notionalMode is DV01-based)
//   `direction    (`long default, or `short)
//   `freq         (default `1Y)
//   `horizon      (default `1D for carry projection only; does not affect realized 1D pnl)
//   `repo         (optional: scalar, date->rate dict, or ([] date; repo/value))
//   `groupCols    (optional symbol list; default deliveryCode+contract if present, else sym)
//   `initPnl      (default 0f on first row per group)
dailyInvoiceSpreadOISPnL:{[ctdTable;oisCurves;cfg]
    if[0=count ctdTable; :ctdTable];
    c:$[99h=type cfg; cfg; ()!()];
    baseSwapNotional:$[`swapNotional in key c; 1f*c`swapNotional; 1000000f];
    notionalMode:`$lower string $[`notionalMode in key c; c`notionalMode; `perBpDV01];
    dv01Modes:`perbpdv01`dv01`unitdv01;
    isDv01Mode:notionalMode in dv01Modes;
    if[not (isDv01Mode or notionalMode~`fixed); '"cfg.notionalMode must be `perBpDV01/`dv01/`unitdv01 or `fixed"];
    targetSwapDV01:$[`targetSwapDV01 in key c; 1f*c`targetSwapDV01; 1f];
    if[targetSwapDV01<=0; '"cfg.targetSwapDV01 must be > 0"];
    freq:$[`freq in key c; c`freq; `1Y];
    horizon:$[`horizon in key c; c`horizon; `1D];
    initPnl:$[`initPnl in key c; 1f*c`initPnl; 0f];
    dirRaw:$[`direction in key c; c`direction; `long];
    direction:`$lower string dirRaw;
    if[not direction in `long`short; '"cfg.direction must be `long or `short"];
    repoInput:$[`repo in key c; c`repo; (::)];

    gcols:$[`groupCols in key c; c`groupCols; ()];
    if[gcols~(::); gcols:()];
    if[-11h=type gcols; gcols:enlist gcols];
    if[10h=type gcols; gcols:enlist `$gcols];
    if[0h=type gcols;
        if[(count gcols)=1;
            g0:first gcols;
            if[11h=abs type g0; gcols:g0];
        ]];
    if[0h=type gcols;
        if[0<count gcols;
            if[10h=abs type first gcols; gcols:`$gcols];
        ]];
    if[0=count gcols;
        if[`deliveryCode in cols ctdTable; gcols,:enlist `deliveryCode];
        if[`contract in cols ctdTable; gcols,:enlist `contract];
        if[0=count gcols; gcols:enlist `sym]];
    if[11h<>type gcols; '"cfg.groupCols must be a symbol or symbol list"];
    if[not all (gcols in cols ctdTable); '"cfg.groupCols contains missing columns in ctdTable"];

    out:dailyInvoiceSpreadOIS[ctdTable;oisCurves];
    n:count out;
    hasClean:`cleanPrice in cols out;

    dates:asc distinct out`date;
    if[not all (dates in key oisCurves); '"oisCurves missing one or more dates from ctdTable"];
    convCurves:dates!.invoicepricer.oisCurveToSwapCurve each oisCurves dates;

    swapPnL:n#0n;
    futuresPnL:n#0n;
    totalPnL:n#0n;
    spreadChangeBps:n#0n;
    swapCarryV:n#0n;
    swapRollV:n#0n;
    swapTotalV:n#0n;
    futuresCarryV:n#0n;
    futuresRollV:n#0n;
    futuresTotalV:n#0n;
    netCarryV:n#0n;
    netRollV:n#0n;
    totalExpectedV:n#0n;
    repoRateV:n#0n;
    futuresNotionalV:n#0n;
    swapNotionalV:n#0n;
    carryFuturesNotionalV:n#0n;
    carrySwapNotionalV:n#baseSwapNotional;
    directionV:n#direction;
    pnlSwapNotionalV:n#0n;
    pnlFuturesNotionalV:n#0n;

    grpKeysTbl:flip gcols!(out gcols);
    grp:group grpKeysTbl;
    grpVals:value grp;
    gi:0;
    while[gi<count grpVals;
        idxs:grpVals gi;
        if[0<count idxs;
            sub:out idxs;
            ord:iasc sub`date;
            sub:sub ord;
            idxs:idxs ord;

            prevPos:(::);
            prevCurve:(::);
            sj:0;
            while[sj<count sub;
                r:sub sj;
                d:r`date;
                curve:convCurves d;
                cp:$[hasClean; r`cleanPrice; 100f];
                ctdBond:`sym`coupon`maturity`cleanPrice`cf!(r`sym;r`coupon;r`maturity;cp;r`cf);
                rowSwapNotional:baseSwapNotional;
                if[isDv01Mode;
                    / Size to requested swap DV01 units per 1bp (default: 1bp DV01).
                    dv01PerUnit:swapDV01Dated[curve;1f;r`deliveryDate;ctdBond`maturity;freq];
                    rowSwapNotional:$[(null dv01PerUnit) or 0f=abs dv01PerUnit; baseSwapNotional; targetSwapDV01 % abs dv01PerUnit];
                ];
                pos:createPosition[curve;r`deliveryDate;r`futuresPrice;ctdBond;rowSwapNotional;freq;direction];
                / Carry/roll is always reported in flat-notional units.
                carryPos:createPosition[curve;r`deliveryDate;r`futuresPrice;ctdBond;baseSwapNotional;freq;direction];
                oi:idxs sj;
                swapNotionalV[oi]:pos`swapNotional;
                futuresNotionalV[oi]:pos`futuresNotional;
                carryFuturesNotionalV[oi]:carryPos`futuresNotional;

                rp:.invoicepricer.repoForDate[repoInput;d];
                repoRateV[oi]:rp;
                if[not null rp;
                    cr:spreadCarry[curve;carryPos;rp;horizon];
                    swapCarryV[oi]:cr`swapCarry;
                    swapRollV[oi]:cr`swapRollDown;
                    swapTotalV[oi]:(cr`swapCarry) + (cr`swapRollDown);
                    futuresCarryV[oi]:cr`futuresCarry;
                    futuresRollV[oi]:cr`futuresRollDown;
                    futuresTotalV[oi]:(cr`futuresCarry) + (cr`futuresRollDown);
                    netCarryV[oi]:cr`netCarry;
                    netRollV[oi]:cr`netRollDown;
                    totalExpectedV[oi]:cr`totalExpected;
                ];

                if[sj=0;
                    pnlSwapNotionalV[oi]:pos`swapNotional;
                    pnlFuturesNotionalV[oi]:pos`futuresNotional;
                    swapPnL[oi]:initPnl;
                    futuresPnL[oi]:initPnl;
                    totalPnL[oi]:initPnl;
                    spreadChangeBps[oi]:0f;
                ];
                if[sj>0;
                    pd:pnlDecomposition[prevCurve;prevPos;curve;r`futuresPrice];
                    pnlSwapNotionalV[oi]:prevPos`swapNotional;
                    pnlFuturesNotionalV[oi]:prevPos`futuresNotional;
                    swapPnL[oi]:pd[`swapPnL]`pnl;
                    futuresPnL[oi]:pd[`futuresPnL]`signedPnL;
                    totalPnL[oi]:pd`totalPnL;
                    spreadChangeBps[oi]:pd`spreadChangeBps;
                ];

                prevPos:pos;
                prevCurve:curve;
                sj+:1;
            ];
        ];
        gi+:1;
    ];

    update
        swapNotional:swapNotionalV,
        futuresNotional:futuresNotionalV,
        carrySwapNotional:carrySwapNotionalV,
        carryFuturesNotional:carryFuturesNotionalV,
        direction:directionV,
        repoRate:repoRateV,
        pnlSwapNotional:pnlSwapNotionalV,
        pnlFuturesNotional:pnlFuturesNotionalV,
        swapPnL:swapPnL,
        futuresPnL:futuresPnL,
        totalPnL:totalPnL,
        realizedSwapPnL1D:swapPnL,
        realizedFuturesPnL1D:futuresPnL,
        realizedTotalPnL1D:totalPnL,
        realizedSwapPnLPrev:swapPnL,
        realizedFuturesPnLPrev:futuresPnL,
        realizedTotalPnLPrev:totalPnL,
        spreadChangeBpsPrev:spreadChangeBps,
        spreadChangeBps1D:spreadChangeBps,
        swapCarry:swapCarryV,
        swapRollDown:swapRollV,
        swapTotalExpected:swapTotalV,
        futuresCarry:futuresCarryV,
        futuresRollDown:futuresRollV,
        futuresTotalExpected:futuresTotalV,
        netCarry:netCarryV,
        netRollDown:netRollV,
        totalExpectedCarry:totalExpectedV
      from out}

// Extract CTD data from dailyCTD output for invoice spread calculation
// When dailyCTD is run with `both mode, this selects the winning CTD per (date, deliveryCode)
// based on isCTDBoth (highest implied repo across first/last delivery scenarios)
//
// ctdResults: output from .tsy.dailyCTD with useLastDelivery=`both
// Returns: table with columns needed for dailyInvoiceSpreadOIS:
//          date, deliveryDate, futuresPrice, sym, coupon, maturity, cf
//          Plus: deliveryCode, contract, deliveryMonth, deliveryType for reference
ctdForInvoiceSpread:{[ctdResults]
    // Check if this is `both mode output (has isCTDBoth column)
    if[`isCTDBoth in cols ctdResults;
        // Filter for winning CTD (highest implied repo across both delivery scenarios)
        ctd:select from ctdResults where isCTDBoth;
        :select date, deliveryDate, futuresPrice, sym, coupon, maturity, cf,
                deliveryCode, contract, deliveryMonth, deliveryType
         from ctd];

    // Non-both mode: use isCTD to get CTD per (date, deliveryCode)
    if[`isCTD in cols ctdResults;
        ctd:select from ctdResults where isCTD;
        :select date, deliveryDate, futuresPrice, sym, coupon, maturity, cf,
                deliveryCode, contract, deliveryMonth
         from ctd];

    // Fallback: return as-is with required columns selected
    select date, deliveryDate, futuresPrice, sym, coupon, maturity, cf from ctdResults}

// Full invoice spread pipeline from dailyCTD output
// Combines ctdForInvoiceSpread + buildOISCurves + dailyInvoiceSpreadOIS
//
// ctdResults: output from .tsy.dailyCTD (ideally with `both mode)
// oisRateTable: ([] date; tenors; rates) - OIS curve data
// Returns: CTD table with invoice spread columns added
invoiceSpreadFromCTD:{[ctdResults;oisRateTable]
    // Extract winning CTDs
    ctdTable:ctdForInvoiceSpread ctdResults;
    if[0=count ctdTable; :ctdTable];

    // Build OIS curves for dates in CTD table
    dates:distinct ctdTable`date;
    oisRates:select from oisRateTable where date in dates;
    oisCurves:buildOISCurves oisRates;

    // Calculate invoice spreads
    dailyInvoiceSpreadOIS[ctdTable;oisCurves]}

// =============================================================================
// MARKET-BASED OAS TABLE FUNCTIONS
// =============================================================================
// These functions use market prices and implied repo for CTD identification

// Process single row for market-based OAS
// Internal helper for oasFromCTDMarket
oasFromCTDMarketRow:{[ctx;row]
    dt:row`date;
    deliveryCode:row`deliveryCode;
    contract:row`contract;

    // Get curve for this date
    curve:ctx[`curvesByDate] dt;

    // Get repo - may be by date, by (date,contract), or scalar
    repo:$[`repoByDateContract in key ctx;
        ctx[`repoByDateContract] (dt;contract);
        `repoByDate in key ctx;
        ctx[`repoByDate] dt;
        ctx`repo];

    // Get vol - may be by date, by (date,contract), or scalar
    vol:$[`volByDateContract in key ctx;
        ctx[`volByDateContract] (dt;contract);
        `volByDate in key ctx;
        ctx[`volByDate] dt;
        ctx`vol];

    freq:ctx`freq;

    // Extract CTD bond with market price
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        row`sym;
        row`coupon;
        row`maturity;
        row`cleanPrice;
        row`cf);

    // Delivery dates
    firstDelivery:row`firstDelivery;
    lastDelivery:row`lastDelivery;
    delivery:`first`last!(firstDelivery;lastDelivery);
    futuresPrice:row`futuresPrice;

    // Get basket for this (date, deliveryCode) if available
    basket:$[`basketsByKey in key ctx;
        ctx[`basketsByKey] (dt;deliveryCode);
        enlist ctdBond];  // Default to just CTD

    // Calculate OAS using market-based function
    result:oasInvoiceSpreadMarket[curve;delivery;futuresPrice;ctdBond;basket;repo;vol;freq];

    // Return key metrics
    `invoiceYield`swapRate`invoiceSpread`spreadBps`qualityOption`timingOption`optionAdjBps`oasBps`netBasis!(
        result`invoiceYield;
        result`swapRate;
        result`invoiceSpread;
        result`spreadBps;
        result`qualityOption;
        result`timingOption;
        result`optionAdjBps;
        result`oasBps;
        result`netBasis)}

// Market-based OAS from CTD table
// Uses market prices and implied repo for CTD identification
//
// ctdTable: table with columns:
//   date, deliveryCode, futuresPrice, sym, coupon, maturity, cleanPrice, cf
//   firstDelivery, lastDelivery (or use helper to compute)
// basketTable: (optional) full basket for each (date, deliveryCode) for quality option
//   Must have: date, deliveryCode, sym, coupon, maturity, cleanPrice, cf
// oisCurves: dict of date -> oisCurve (from buildOISCurves or oisBootstrap)
// repoRate: scalar, dict (date -> rate), or table ([] date; rate)
// vol: annualized rate volatility (e.g., 0.008 = 80bp/year)
// freq: swap frequency (`SA for semi-annual)
//
// Returns: ctdTable with OAS columns added
oasFromCTDMarket:{[ctdTable;basketTable;oisCurves;repoRate;vol;freq]
    if[0=count ctdTable; :ctdTable];

    // Add delivery dates if missing (parses deliveryCode if needed)
    ctdTable:addDeliveryDates ctdTable;

    // Handle defaults
    frq:$[freq~(::); `SA; freq];

    // Get unique dates and contracts
    dates:asc distinct ctdTable`date;
    contracts:distinct ctdTable`contract;
    nDates:count dates;

    // Build curves by date (handle dict or builder)
    curvesByDate:$[99h=type oisCurves;
        $[all dates in key oisCurves; oisCurves; dates!oisCurves each dates];
        dates!nDates#enlist oisCurves];

    // Build repo lookup - supports:
    //   scalar: same for all
    //   dict: keyed by date
    //   table with (date, rate): by date
    //   table with (date, contract, rate): by (date, contract)
    repoCtx:$[-9h=type repoRate;
        // Scalar
        enlist[`repo]!enlist repoRate;
        99h=type repoRate;
        // Dict by date
        enlist[`repoByDate]!enlist repoRate;
        98h=type repoRate;
        // Table - check if has contract column
        $[`contract in cols repoRate;
            // By (date, contract)
            enlist[`repoByDateContract]!enlist (exec (date,'contract)!rate from repoRate);
            // By date only
            enlist[`repoByDate]!enlist (exec date!rate from repoRate)];
        // Default scalar
        enlist[`repo]!enlist repoRate];

    // Build vol lookup - same structure as repo
    volCtx:$[vol~(::);
        // Default
        enlist[`vol]!enlist 0.008;
        -9h=type vol;
        // Scalar
        enlist[`vol]!enlist vol;
        99h=type vol;
        // Dict by date
        enlist[`volByDate]!enlist vol;
        98h=type vol;
        // Table - check if has contract column
        $[`contract in cols vol;
            // By (date, contract)
            enlist[`volByDateContract]!enlist (exec (date,'contract)!vol from vol);
            // By date only
            enlist[`volByDate]!enlist (exec date!vol from vol)];
        // Default scalar
        enlist[`vol]!enlist 0.008];

    // Build basket index by (date, deliveryCode) if basket provided
    basketsByKey:$[(98h=type basketTable) and 0 < count basketTable;
        {[t]
            // Group by (date, deliveryCode)
            grp:t group ([]date:t`date;deliveryCode:t`deliveryCode);
            // Convert each group to list of bond dicts
            keys_:key grp;
            vals_:{[rows]
                {`sym`coupon`maturity`cleanPrice`cf!(x`sym;x`coupon;x`maturity;x`cleanPrice;x`cf)} each rows
            } each value grp;
            (keys_[;`date],'keys_[;`deliveryCode])!vals_
        } basketTable;
        ()!()];

    // Pack context
    ctx:`curvesByDate`basketsByKey`freq!(curvesByDate;basketsByKey;frq);
    ctx:ctx,repoCtx,volCtx;

    // Process each CTD row
    results:oasFromCTDMarketRow[ctx;] each ctdTable;

    // Combine with original table
    resultTable:$[98h=type results; results; flip results];
    ctdTable,'resultTable}

// Convenience: Add delivery dates and contract info to CTD table if missing
// Parses deliveryCode (e.g., `TUH26) to extract contract, deliveryMonth, and delivery dates
// Uses contract type to determine first/last delivery conventions
addDeliveryDates:{[ctdTable]
    // If already has delivery dates, return as-is
    if[(`firstDelivery in cols ctdTable) and `lastDelivery in cols ctdTable; :ctdTable];

    // Check what columns we have
    hasContract:`contract in cols ctdTable;
    hasMonth:`deliveryMonth in cols ctdTable;
    hasDC:`deliveryCode in cols ctdTable;

    // If missing contract/month but have deliveryCode, parse it
    t:$[hasDC and not hasContract;
        update contract:.tsy.contractFromCode each deliveryCode from ctdTable;
        ctdTable];
    t:$[hasDC and not hasMonth;
        $[`deliveryDate in cols t;
            / Prefer deliveryDate-derived month when available; this is robust to short codes like TYH6.
            update deliveryMonth:`month$deliveryDate from t;
            update deliveryMonth:.tsy.monthFromCode each deliveryCode from t];
        t];

    // Now add delivery dates based on contract and delivery month
    update
        firstDelivery:.tsy.firstDeliveryDayForContract'[contract;deliveryMonth],
        lastDelivery:.tsy.lastDeliveryDayForContract'[contract;deliveryMonth]
    from t}

// Full market-based OAS pipeline from dailyCTD output
// Combines CTD extraction + delivery dates + OIS curves + market OAS
//
// ctdResults: output from .tsy.dailyCTD with `both mode
//   Must include cleanPrice column (market prices)
//   Expected columns: date, deliveryDate, futuresPrice, sym, coupon, maturity, cf,
//                     deliveryCode, contract, deliveryMonth
// basketResults: (optional) full basket output from dailyCTD for quality option
// oisRateTable: ([] date; tenors; rates) - OIS curve data
// repoRate: scalar or ([] date; rate)
// vol: annualized rate volatility
//
// Returns: CTD table with full market-based OAS analysis
oasFromCTDMarketPipeline:{[ctdResults;basketResults;oisRateTable;repoRate;vol]
    // Check if already has firstDelivery/lastDelivery columns
    hasBothDates:(`firstDelivery in cols ctdResults) and `lastDelivery in cols ctdResults;

    // Extract winning CTDs (only if using dailyCTD format with isCTDBoth)
    ctdTable:$[hasBothDates;
        ctdResults;  // Already formatted, use directly
        ctdForInvoiceSpread ctdResults];
    if[0=count ctdTable; :ctdTable];

    // Ensure cleanPrice exists
    if[not `cleanPrice in cols ctdTable;
        '"cleanPrice column required - need market prices"];

    // Add delivery dates if not present
    ctdTable:addDeliveryDates ctdTable;

    // Build OIS curves
    dates:distinct ctdTable`date;
    oisRates:select from oisRateTable where date in dates;
    oisCurves:buildOISCurves oisRates;

    // Prepare basket table (if provided)
    basketTable:$[(98h=type basketResults) and 0 < count basketResults;
        select date, deliveryCode, sym, coupon, maturity, cleanPrice, cf from basketResults;
        ()];

    // Calculate market-based OAS
    oasFromCTDMarket[ctdTable;basketTable;oisCurves;repoRate;vol;`SA]}

// Build OIS curves from rate table
// oisRateTable: ([] date; tenor; rate) - wide format table
//   date: valuation date
//   tenor: LIST of tenors (symbols like `1D`1W or strings like "1D","1W")
//   rate: LIST of OIS par rates (same length as tenor)
// Returns dict: date -> oisCurve
buildOISCurves:{[oisRateTable]
    // Build curve for each row
    // Supports column names: tenor/tenors and rate/rates
    cols_:cols oisRateTable;
    tenorCol:$[`tenors in cols_; `tenors; `tenor];
    rateCol:$[`rates in cols_; `rates; `rate];

    buildOne:{[tCol;rCol;row]
        dt:row`date;
        tenors:row tCol;
        rates:row rCol;
        // Convert strings to symbols if needed
        if[10h=type first tenors; tenors:`$tenors];
        // Convert tenors to year fractions and sort
        yf:.swaps.tenorsToYF tenors;
        ord:iasc yf;
        yf:yf ord;
        rates:rates ord;
        // Bootstrap
        oisBootstrap[yf;rates;dt]
    }[tenorCol;rateCol];

    // Return dict: date -> curve
    oisRateTable[`date]!buildOne each oisRateTable}

// Debug version - use this to diagnose issues
buildOISCurvesDebug:{[oisRateTable]
    -1 "DEBUG: Input table meta:";
    show meta oisRateTable;
    -1 "";
    cols_:cols oisRateTable;
    -1 "DEBUG: Columns: ",", " sv string cols_;
    tenorCol:$[`tenors in cols_; `tenors; `tenor];
    rateCol:$[`rates in cols_; `rates; `rate];
    -1 "DEBUG: Using tenorCol=",string[tenorCol]," rateCol=",string rateCol;
    -1 "";
    -1 "DEBUG: First row:";
    row:oisRateTable 0;
    -1 "  date: ",string row`date;
    -1 "  ",string[tenorCol]," type: ",string type row tenorCol;
    -1 "  ",string[tenorCol]," count: ",string count row tenorCol;
    -1 "  ",string[tenorCol]," value: ",-3!row tenorCol;
    -1 "  first tenor type: ",string type first row tenorCol;
    -1 "  first tenor value: ",-3!first row tenorCol;
    -1 "";
    -1 "DEBUG: Calling tenorsToYF on first row...";
    tenors:row tenorCol;
    yf:.swaps.tenorsToYF tenors;
    -1 "  Result: ",-3!yf;
    -1 "";
    -1 "DEBUG: Building all curves...";
    buildOISCurves[oisRateTable]}

// =============================================================================
// OIS DAILY EXAMPLE
// =============================================================================

// Example: Daily invoice spread with OIS conventions
exampleOIS:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "           DAILY INVOICE SPREAD WITH OIS CONVENTIONS";
    -1 "=============================================================================";
    -1 "";

    // Sample tenors
    tenors:`1D`1W`2W`3W`1M`2M`3M`4M`5M`6M`7M`8M`9M`10M`11M`1Y`2Y`3Y`4Y`5Y`6Y`7Y`8Y`9Y`10Y;

    // Sample OIS rates (USD SOFR par rates)
    baseRates:0.0364 0.0367 0.0369 0.0368 0.0367 0.0368 0.0367 0.0366 0.0365 0.0364 0.0362 0.036 0.0358 0.0356 0.0354 0.0353 0.0343 0.0346 0.0351 0.0357 0.0364 0.0371 0.0377 0.0383 0.0389;

    // Create sample dates
    dates:("D"$"2026.01.21") + til 5;

    -1 "Step 1: Create OIS Rate Table (wide format)";
    -1 "";

    // Build rate table: ([] date; tenor; rate)
    // Each row has one date with lists of tenors and rates
    oisRateTable:([] date:dates; tenor:count[dates]#enlist tenors; rate:count[dates]#enlist baseRates);

    -1 "  OIS Rate Table:";
    show oisRateTable;
    -1 "";
    -1 "  Each row has date + list of tenors + list of rates";
    -1 "  Rows: ",string count oisRateTable;
    -1 "";

    -1 "Step 2: Create CTD Table (sample data)";
    -1 "";

    // Sample CTD data (5 days, same bond, varying futures price)
    ctdTable:([]
        date:dates;
        deliveryDate:5#"D"$"2026.03.31";
        futuresPrice:111.5 + 0.1 * til 5;
        sym:5#`T_4_125_2032;
        coupon:5#0.04125;
        maturity:5#"D"$"2032.11.15";
        cf:5#0.9003);

    -1 "  CTD Table:";
    show ctdTable;
    -1 "";

    -1 "Step 3: Build OIS Curves";
    -1 "  oisCurves:.invoicepricer.buildOISCurves[oisRateTable]";
    -1 "";

    // Build curves from rate table
    oisCurves:buildOISCurves[oisRateTable];
    -1 "  Built ",string[count oisCurves]," OIS curves";
    -1 "  Sample DF at 7Y: ",string oisCurves[first dates][`df][oisCurves[first dates][`yf]?7f];
    -1 "";

    -1 "Step 4: Calculate Daily Invoice Spreads";
    -1 "  result:.invoicepricer.dailyInvoiceSpreadOIS[ctdTable;oisCurves]";
    -1 "";

    // Calculate spreads using pre-built OIS curves
    result:dailyInvoiceSpreadOIS[ctdTable;oisCurves];

    -1 "  Results:";
    show select date, futuresPrice, invoiceYield, swapRate, spreadBps, futuresDV01 from result;
    -1 "";

    -1 "=============================================================================";
    -1 "Conventions used:";
    -1 "  Bond Yield:  Street convention (semi-annual, period count, ACT/ACT AI)";
    -1 "  Swap Rate:   OIS bootstrap (annual fixed, ACT/360, SOFR compounded)";
    -1 "=============================================================================";
    -1 "";

    -1 "Workflow:";
    -1 "  1. Build OIS curves: oisCurves:.invoicepricer.buildOISCurves[oisRateTable]";
    -1 "  2. Calc spreads:     result:.invoicepricer.dailyInvoiceSpreadOIS[ctdTable;oisCurves]";
    -1 "";

    result}

// =============================================================================
// EXAMPLE & HELP
// =============================================================================

// Example demonstrating all functions
example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                    INVOICE SPREAD LIBRARY EXAMPLE";
    -1 "=============================================================================";
    -1 "";
    // Check dependencies
    if[not `buildCurve in key `.swaps;
        -1 "ERROR: Load swaps.q first (\\l swaps.q)";
        :()];
    if[not `yieldToMaturity in key `.ctd;
        -1 "ERROR: Load ctd.q first (\\l ctd.q)";
        :()];

    -1 "Building yield curve...";
    // Build curve
    curveDate:"D"$"2025.01.15";
    tenors:`3M`6M`1Y`2Y`3Y`5Y`7Y`10Y`20Y`30Y;
    rates:0.0425 0.0438 0.0452 0.0468 0.0479 0.0495 0.0508 0.0522 0.0545 0.0560;
    curve:.swaps.buildCurve[tenors;rates;`asOfDate`frequency!(curveDate;`6M)];

    -1 "Defining CTD bond...";
    // Define CTD bond
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        `T_4_500_2035;0.045;"D"$"2035.02.15";101.125;0.8756);

    // Parameters
    settleDate:curveDate;
    deliveryDate:"D"$"2025.03.20";
    futuresPrice:116.50;
    swapNotional:10000000f;
    repoRate:0.045;
    freq:`6M;

    -1 "";
    -1 "=== BASIC CALCULATIONS ===";
    -1 "";

    // Invoice price
    invPrice:invoicePrice[deliveryDate;futuresPrice;ctdBond];
    -1 "Invoice Price:     ",string invPrice;

    // Invoice yield
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];
    -1 "Invoice Yield:     ",(string 0.01*`int$10000*invYld),"%";

    // Swap rate
    swpRt:swapRate[curve;deliveryDate;ctdBond`maturity;freq];
    -1 "Swap Rate:         ",(string 0.01*`int$10000*swpRt),"%";

    // Invoice spread
    spread:invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq];
    -1 "Invoice Spread:    ",(string `int$spread`spreadBps)," bps";

    -1 "";
    -1 "=== HEDGE RATIO ===";
    -1 "";

    hr:hedgeRatio[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq];
    -1 "Swap DV01:         $",(fmtNum hr`swapDV01);
    -1 "Futures DV01:      $",(fmtNum hr`futuresDV01)," per $1MM";
    -1 "Swap Notional:     $",fmtNum hr`swapNotional;
    -1 "Futures Notional:  $",fmtNum hr`futuresNotional;

    -1 "";
    -1 "=== FULL ANALYSIS ===";
    -1 "";

    // Run full analysis
    result:analyze[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;repoRate;freq];
    showAnalysis result;

    result}

// Usage reference
usage:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                     .invoicepricer USAGE REFERENCE";
    -1 "=============================================================================";
    -1 "";
    -1 "DEPENDENCIES: Load swaps.q and ctd.q first";
    -1 "";
    -1 "All functions work in NOTIONAL terms - no contract size needed.";
    -1 "You can convert notional to contracts yourself: contracts = notional / contractSize";
    -1 "";
    -1 "// CORE SPREAD CALCULATIONS";
    -1 ".invoicepricer.invoicePrice[deliveryDate;futuresPrice;ctdBond]";
    -1 ".invoicepricer.invoiceYield[deliveryDate;futuresPrice;ctdBond]";
    -1 ".invoicepricer.swapRate[curve;deliveryDate;ctdMaturity;freq]";
    -1 ".invoicepricer.invoiceSpread[curve;deliveryDate;futuresPrice;ctdBond;freq]";
    -1 "";
    -1 "// OIS CONVENTIONS (PREFERRED for accurate spread calculation)";
    -1 "// Uses: Semi-annual period count YTM for bonds, OIS bootstrap for swaps";
    -1 ".invoicepricer.invoiceSpreadOIS[tenors;oisRates;curveDate;deliveryDate;futuresPrice;ctdBond]";
    -1 "    // tenors: `1D`1W`1M`3M`6M`1Y`2Y... oisRates: USD SOFR par OIS rates";
    -1 "    // Returns: invoiceYield, swapRate, invoiceSpread, spreadBps";
    -1 ".invoicepricer.invoiceSpreadWithOIS[oisCurve;curveDate;deliveryDate;futuresPrice;ctdBond]";
    -1 "    // Use pre-built oisCurve for multiple calculations";
    -1 ".invoicepricer.oisBootstrap[tenors;oisRates;curveDate]";
    -1 "    // Build OIS curve from par rates. Returns dict with `yf`df`curveDate";
    -1 ".invoicepricer.oisSwapRate[tenors;oisRates;curveDate;deliveryDate;maturityDate]";
    -1 "    // Forward par OIS rate from delivery to maturity";
    -1 "";
    -1 "// HEDGE RATIO & POSITION SIZING (all notional-based)";
    -1 ".invoicepricer.futuresDV01[deliveryDate;futuresPrice;ctdBond]      // DV01 per $1MM (forward mod dur)";
    -1 ".invoicepricer.swapDV01Dated[curve;notional;deliveryDate;ctdMaturity;freq]";
    -1 ".invoicepricer.hedgeRatio[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq]";
    -1 "    // Returns: swapDV01, futuresDV01, swapNotional, futuresNotional";
    -1 ".invoicepricer.futuresNotionalFromSwap[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq]";
    -1 ".invoicepricer.swapNotionalFromFutures[curve;deliveryDate;futuresPrice;ctdBond;futuresNotional;freq]";
    -1 ".invoicepricer.createPosition[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq;direction]";
    -1 "";
    -1 "// P&L DECOMPOSITION";
    -1 ".invoicepricer.swapPnL[curve;position;newCurve]";
    -1 ".invoicepricer.futuresPnL[position;newFuturesPrice]";
    -1 ".invoicepricer.pnlDecomposition[curve;position;newCurve;newFuturesPrice]";
    -1 "";
    -1 "// CARRY ANALYSIS";
    -1 ".invoicepricer.swapCarry[curve;position;horizon]";
    -1 ".invoicepricer.futuresCarry[curve;position;repoRate;horizon]";
    -1 ".invoicepricer.spreadCarry[curve;position;repoRate;horizon]";
    -1 "";
    -1 "// SCENARIOS";
    -1 ".invoicepricer.parallelShiftScenario[curve;position;shiftBps;repoRate]";
    -1 ".invoicepricer.parallelShiftScenarios[curve;position;shiftsBps;repoRate]";
    -1 ".invoicepricer.spreadScenario[curve;position;spreadChangeBps;repoRate]";
    -1 ".invoicepricer.spreadScenarios[curve;position;spreadChangesBps;repoRate]";
    -1 "";
    -1 "// RICH/CHEAP ANALYSIS";
    -1 ".invoicepricer.richCheap[currentSpread;history;lookback]";
    -1 ".invoicepricer.addToHistory[history;curve;deliveryDate;futuresPrice;ctdBond;freq]";
    -1 ".invoicepricer.newHistory[]";
    -1 "";
    -1 "// FULL ANALYSIS";
    -1 ".invoicepricer.analyze[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;repoRate;freq]";
    -1 ".invoicepricer.showAnalysis[result]";
    -1 ".invoicepricer.showRichCheap[result]";
    -1 ".invoicepricer.showPosition[position]";
    -1 ".invoicepricer.example[]";
    -1 "";
    -1 "// CTD BOND FORMAT";
    -1 "ctdBond:`sym`coupon`maturity`cleanPrice`cf!(";
    -1 "    `T_4_500_2035;    // symbol";
    -1 "    0.045;            // coupon (4.5%)";
    -1 "    \"D\"$\"2035.02.15\"; // maturity date";
    -1 "    101.125;          // clean price";
    -1 "    0.8756            // conversion factor";
    -1 "  )";
    -1 "";
    -1 "// BATCH FUNCTIONS (parallelized for multiple bonds)";
    -1 ".invoicepricer.invoiceYieldBatch[deliveryDate;futuresPrice;ctdBonds]";
    -1 ".invoicepricer.invoiceSpreadBatch[curve;deliveryDate;futuresPrice;ctdBonds;freq]";
    -1 ".invoicepricer.futuresDV01Batch[deliveryDate;futuresPrices;ctdBonds]";
    -1 ".invoicepricer.hedgeRatioBatch[curve;deliveryDate;futuresPrices;ctdBonds;swapNotional;freq]";
    -1 "";
    -1 "// DAILY FUNCTIONS";
    -1 "";
    -1 "// OIS conventions (PREFERRED - uses proper market conventions)";
    -1 "// Step 1: Build curves";
    -1 ".invoicepricer.buildOISCurves[yf;oisRates;dates]";
    -1 "    // yf: year fractions (float list)";
    -1 "    // oisRates: static vector or dict (date->rates)";
    -1 "    // Returns dict: date -> oisCurve";
    -1 "// Step 2: Calculate spreads";
    -1 ".invoicepricer.dailyInvoiceSpreadOIS[ctdTable;oisCurves]";
    -1 "    // ctdTable: date, deliveryDate, futuresPrice, sym, coupon, maturity, cf";
    -1 "    // oisCurves: dict from buildOISCurves";
    -1 "    // Returns: ctdTable + invoiceYield, swapRate, spreadBps, futuresDV01";
    -1 ".invoicepricer.exampleOIS[]";
    -1 "    // Run full OIS daily example";
    -1 "";
    -1 "// Generic daily (uses swaps.q curves, not OIS)";
    -1 ".invoicepricer.dailyInvoiceSpreadFromCTD[ctdTable;curves;freq]";
    -1 "    // ctdTable: date, deliveryDate, futuresPrice, sym, coupon, maturity, cf";
    -1 "    // Returns: ctdTable + invoiceYield, swapRate, spreadBps, swapDV01, futuresDV01";
    -1 "";
    -1 ".invoicepricer.dailyOASFromCTD[ctdTable;basketTable;curves;repoRate;vol;freq]";
    -1 "    // ctdTable: date, deliveryCode, deliveryDate, futuresPrice, sym, coupon, maturity, cf";
    -1 "    // basketTable: date, deliveryCode, sym, coupon, maturity, cf, cleanPrice (all bonds)";
    -1 "    // Returns: ctdTable + spreadBps, optionValue, oasBps, swapDV01, futuresDV01, basketSize";
    -1 "";
    -1 "// From baskets (recomputes CTD internally - slower)";
    -1 ".invoicepricer.dailyInvoiceSpread[baskets;prices;futures;curves;repoRate;freq;useLastDelivery]";
    -1 ".invoicepricer.dailyOAS[baskets;prices;futures;curves;repoRate;vol;freq;useLastDelivery]";
    -1 "";
    -1 "// PARALLELIZATION";
    -1 ".invoicepricer.useParallel      // 1b if secondary threads available, 0b otherwise";
    -1 ".invoicepricer.useParallel:1b   // force parallel execution";
    -1 ".invoicepricer.useParallel:0b   // force sequential execution";
    -1 "";
    -1 "// DIRECTION";
    -1 "`long  - receive fixed on swap, short futures (benefit if spread widens)";
    -1 "`short - pay fixed on swap, long futures (benefit if spread narrows)";
    -1 "";
    -1 "=============================================================================";
    -1 "";}

// Help function
help:{[] usage[]}

\d .

// Display load message
-1 "";
-1 "Invoice Spread Library v",.invoicepricer.version," loaded";
-1 "Namespace: .invoicepricer";
-1 "Dependencies: swaps.q, ctd.q";
-1 "Run .invoicepricer.example[] for demonstration";
-1 "Run .invoicepricer.usage[] for function reference";
-1 "";
