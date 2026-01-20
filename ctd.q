// =============================================================================
// CTD (CHEAPEST-TO-DELIVER) BOND FUTURES PRICER
// =============================================================================
// Prices bond futures deliverables and identifies the CTD bond
// Integrates with .swaps curve infrastructure
//
// Usage: \l ctd.q (after loading swaps.q)
// =============================================================================

\d .ctd

version:"0.1.0"

// =============================================================================
// CONFIGURATION
// =============================================================================

// Standard coupon for conversion factor calculation (6% for most contracts)
defaults:`standardCoupon`dayCount`settleDelay`futuresTickSize!(0.06;`ACT365;1i;1%32)

// =============================================================================
// CURVE BUILDING FROM BONDS
// =============================================================================

// Build a yield curve from a basket of bonds (using their YTMs)
// This gives a curve consistent with the deliverable bonds
// bonds: list of dicts with `sym`coupon`maturity`cleanPrice`cf
// settleDate: valuation date
curveFromBonds:{[settleDate;bonds]
    // Calculate YTM for each bond
    ytms:{[settle;b]
        yieldToMaturity[settle;b`maturity;b`coupon;`SA;100f;b`cleanPrice]
    }[settleDate] each bonds;

    // Get maturities as year fractions
    yfs:{(x - y) % 365f}[;settleDate] each bonds`maturity;

    // Sort by maturity
    ord:iasc yfs;
    yfs:yfs ord;
    ytms:ytms ord;

    // Build curve using YTMs as zero rates (approximation)
    // For more accuracy, would need to bootstrap from bond prices
    .swaps.buildCurve[yfs;ytms;`asOfDate`frequency`interpMethod!(settleDate;`6M;`linear)]}

// Quick curve from a single bond's YTM (flat curve assumption)
// Useful when you just need approximate roll-down
curveFromBond:{[settleDate;ctdBond]
    ytm:yieldToMaturity[settleDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;ctdBond`cleanPrice];
    yf:(ctdBond[`maturity] - settleDate) % 365f;
    // Create flat curve at the YTM level
    tenors:0.25 0.5 1 2 3 5 7 10 20 30f;
    rates:count[tenors]#ytm;
    .swaps.buildCurve[tenors;rates;`asOfDate`frequency!(settleDate;`6M)]}

// =============================================================================
// BOND PRICING
// =============================================================================

// Generate bond cash flow dates from settlement to maturity
genBondCFDates:{[settleDate;maturityDate;couponFreq]
    // couponFreq: `SA (semi-annual) or `Q (quarterly) or `A (annual)
    monthsPerPeriod:$[couponFreq~`SA;6;couponFreq~`Q;3;couponFreq~`A;12;6];
    // Work backwards from maturity to find coupon dates
    dates:();
    dt:maturityDate;
    while[dt > settleDate;
        dates,:dt;
        dt:.swaps.addMonthsToDate[dt;neg monthsPerPeriod]];
    asc dates}

// Accrued interest calculation
// Uses ACT/ACT for Treasury bonds
accruedInterest:{[settleDate;maturityDate;coupon;couponFreq;faceValue]
    monthsPerPeriod:$[couponFreq~`SA;6;couponFreq~`Q;3;couponFreq~`A;12;6];
    periodsPerYear:12 % monthsPerPeriod;
    couponPayment:faceValue * coupon % periodsPerYear;

    // Find previous and next coupon dates
    cfDates:genBondCFDates[settleDate - 365;maturityDate;couponFreq];
    prevCpn:max cfDates where cfDates <= settleDate;
    nextCpn:min cfDates where cfDates > settleDate;

    // Days accrued / days in period (ACT/ACT)
    daysAccrued:settleDate - prevCpn;
    daysInPeriod:nextCpn - prevCpn;

    couponPayment * daysAccrued % daysInPeriod}

// Price a bond given a yield curve
// Returns clean price (excludes accrued interest)
priceBond:{[curve;settleDate;maturityDate;coupon;couponFreq;faceValue]
    if[settleDate >= maturityDate; :faceValue];  // Matured

    // Generate cash flows
    cfDates:genBondCFDates[settleDate;maturityDate;couponFreq];
    if[0 = count cfDates; :faceValue];

    monthsPerPeriod:$[couponFreq~`SA;6;couponFreq~`Q;3;couponFreq~`A;12;6];
    periodsPerYear:12 % monthsPerPeriod;
    couponPayment:faceValue * coupon % periodsPerYear;

    // Cash flows: coupons + principal at maturity
    cfs:count[cfDates]#couponPayment;
    cfs[count[cfs]-1]+:faceValue;

    // Discount factors from curve
    cfYFs:.swaps.datesToYF[curve;cfDates];
    dfs:.swaps.interpDFs[curve;cfYFs];

    // PV of cash flows
    sum cfs * dfs}

// Full bond valuation
valueBond:{[curve;settleDate;maturityDate;coupon;couponFreq;faceValue]
    cleanPrice:priceBond[curve;settleDate;maturityDate;coupon;couponFreq;faceValue];
    ai:accruedInterest[settleDate;maturityDate;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;
    `cleanPrice`dirtyPrice`accruedInterest`faceValue`coupon`maturity`settleDate!(
        cleanPrice;dirtyPrice;ai;faceValue;coupon;maturityDate;settleDate)}

// Price bond from yield (yield-to-maturity)
priceBondFromYield:{[settleDate;maturityDate;coupon;couponFreq;faceValue;ytm]
    if[settleDate >= maturityDate; :faceValue];

    cfDates:genBondCFDates[settleDate;maturityDate;couponFreq];
    if[0 = count cfDates; :faceValue];

    monthsPerPeriod:$[couponFreq~`SA;6;couponFreq~`Q;3;couponFreq~`A;12;6];
    periodsPerYear:12 % monthsPerPeriod;
    couponPayment:faceValue * coupon % periodsPerYear;

    cfs:count[cfDates]#couponPayment;
    cfs[count[cfs]-1]+:faceValue;

    // Time to each cash flow in years
    cfYFs:(cfDates - settleDate) % 365f;

    // Discount at YTM (continuous compounding)
    dfs:exp neg ytm * cfYFs;

    sum cfs * dfs}

// Yield to maturity (solve for yield given price)
yieldToMaturity:{[settleDate;maturityDate;coupon;couponFreq;faceValue;cleanPrice]
    ai:accruedInterest[settleDate;maturityDate;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;

    // Newton-Raphson to solve for yield
    ytm:coupon;  // Initial guess
    do[20;
        pv:priceBondFromYield[settleDate;maturityDate;coupon;couponFreq;faceValue;ytm];
        pvAI:ai;
        pvDirty:pv + pvAI;

        // Numerical derivative
        dy:0.0001;
        pvUp:priceBondFromYield[settleDate;maturityDate;coupon;couponFreq;faceValue;ytm+dy] + pvAI;
        dPdY:(pvUp - pvDirty) % dy;

        if[abs[dPdY] < 1e-10; :ytm];

        ytm:ytm - (pvDirty - dirtyPrice) % dPdY;
        if[abs[pvDirty - dirtyPrice] < 1e-8; :ytm]];
    ytm}

// =============================================================================
// CONVERSION FACTORS
// =============================================================================

// Calculate conversion factor for a deliverable bond
// CF makes bonds with different coupons comparable
// Assumes standard 6% yield, rounded to first day of delivery month
conversionFactor:{[deliveryDate;maturityDate;coupon;couponFreq]
    stdCoupon:defaults`standardCoupon;

    // Time to maturity in months, rounded down to quarters
    monthsToMat:(`month$maturityDate) - `month$deliveryDate;
    if[monthsToMat <= 0; :1f];

    // Round down to nearest quarter
    quarters:`int$monthsToMat % 3;
    n:quarters % 2;  // Number of semi-annual periods
    z:(quarters mod 2) * 3 % 12f;  // Fractional first period (0 or 0.25)

    if[n = 0; :1f];

    monthsPerPeriod:$[couponFreq~`SA;6;couponFreq~`Q;3;couponFreq~`A;12;6];
    periodsPerYear:12 % monthsPerPeriod;
    c:coupon % periodsPerYear;  // Periodic coupon rate
    y:stdCoupon % periodsPerYear;  // Periodic yield (6% semi-annual = 3%)

    // CF formula (CBOT standard)
    // CF = (c/y) * (1 - 1/(1+y)^n) + 1/(1+y)^n - (c * z)
    // Adjusted for fractional period
    factor:(c % y) * (1 - 1 % xexp[1+y;n]) + (1 % xexp[1+y;n]);
    factor:factor * 1 % xexp[1+y;z*2];  // Adjust for fractional period
    factor:factor - c * z;

    // Round to 4 decimal places (exchange convention)
    0.0001 * `long$factor * 10000}

// Alternative: accept conversion factor as input (from exchange)
// Most practitioners use published CFs rather than calculating

// =============================================================================
// BASIS CALCULATIONS
// =============================================================================

// Gross basis = Clean Price - (Futures Price * Conversion Factor)
grossBasis:{[bondCleanPrice;futuresPrice;cf]
    bondCleanPrice - futuresPrice * cf}

// Forward price of bond to delivery date
forwardBondPrice:{[curve;settleDate;deliveryDate;bondCleanPrice;coupon;couponFreq;faceValue;repoRate]
    // Days to delivery
    daysToDelivery:deliveryDate - settleDate;
    if[daysToDelivery <= 0; :bondCleanPrice];

    // Accrued interest at settlement
    // Need maturity date - estimate from coupon dates or pass in
    aiSettle:0f;  // Simplified - should pass in maturity
    dirtyPrice:bondCleanPrice + aiSettle;

    // Financing cost
    financingCost:dirtyPrice * repoRate * daysToDelivery % 360;

    // Coupon income during holding period (simplified - assumes no coupon)
    couponIncome:0f;

    // Forward dirty price
    fwdDirtyPrice:dirtyPrice + financingCost - couponIncome;

    fwdDirtyPrice}

// Net basis = Gross Basis - Carry
// Carry = Coupon Income - Financing Cost
// Net basis represents the value of delivery option
netBasis:{[bondCleanPrice;futuresPrice;cf;carry]
    grossBasis[bondCleanPrice;futuresPrice;cf] - carry}

// Convenience function: compute net basis from bond parameters
// Does not require curve - uses inputs directly
bondNetBasis:{[settleDate;deliveryDate;maturityDate;cleanPrice;coupon;cf;futuresPrice;repoRate]
    faceValue:100f;
    couponFreq:`SA;

    // Accrued interest
    ai:accruedInterest[settleDate;maturityDate;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;

    // Gross basis
    gb:grossBasis[cleanPrice;futuresPrice;cf];

    // Carry (coupon accrual - financing cost)
    carry:bondCarry[settleDate;deliveryDate;dirtyPrice;coupon;faceValue;repoRate];

    // Net basis
    nb:gb - carry;

    `grossBasis`carry`netBasis`cleanPrice`dirtyPrice`ai!(gb;carry;nb;cleanPrice;dirtyPrice;ai)}

// Count coupon payments between two dates
// Returns number of semi-annual coupons paid (exclusive of start, inclusive of end)
couponPaymentsBetween:{[settleDate;deliveryDate;maturityDate]
    if[deliveryDate <= settleDate; :0];
    // Generate coupon dates working back from maturity
    // Semi-annual: 6 months apart
    matMonth:`month$maturityDate;
    matDay:`dd$maturityDate;
    // Generate months going back 50 years (100 semi-annual periods)
    cpnMonths:matMonth - 6 * til 100;
    // Convert back to dates (same day of month as maturity)
    cpnDates:`date$cpnMonths;
    cpnDates:((`dd$cpnDates) - 1 + matDay) + cpnDates;  // Adjust to correct day
    cpnDates:cpnDates where cpnDates > 1990.01.01;  // Reasonable limit
    // Count coupons strictly after settle and on or before delivery
    count cpnDates where (cpnDates > settleDate) & cpnDates <= deliveryDate}

// Calculate carry for a bond to delivery
// Carry = (Coupon Income Received - Repo Cost) during holding period
// NOTE: Only include coupons actually PAID during the period, not accrual
bondCarry:{[settleDate;deliveryDate;dirtyPrice;coupon;faceValue;repoRate]
    daysHeld:deliveryDate - settleDate;
    if[daysHeld <= 0; :0f];

    // Financing cost (repo) - ACT/360
    repoCost:dirtyPrice * repoRate * daysHeld % 360;

    // For simplicity in this version, assume no coupon payments
    // Most delivery periods (settle to delivery) are < 3 months
    // A full implementation would need maturity date to check coupon dates
    couponIncome:0f;

    // Carry = income - cost (usually negative for short holding periods)
    couponIncome - repoCost}

// Calculate carry with full maturity info (preferred)
// Carry = Change in AI + Coupon Income - Financing Cost
// This is the cash flow from holding the bond to delivery
bondCarryFull:{[settleDate;deliveryDate;maturity;dirtyPrice;coupon;faceValue;repoRate]
    daysHeld:deliveryDate - settleDate;
    if[daysHeld <= 0; :0f];

    // Financing cost (repo) - ACT/360
    repoCost:dirtyPrice * repoRate * daysHeld % 360;

    // Change in accrued interest (you earn this even without coupon payment)
    ai_settle:accruedInterest[settleDate;maturity;coupon;`SA;faceValue];
    ai_delivery:accruedInterest[deliveryDate;maturity;coupon;`SA;faceValue];
    aiChange:ai_delivery - ai_settle;

    // Check for actual coupon payments
    numCoupons:couponPaymentsBetween[settleDate;deliveryDate;maturity];
    couponPayment:faceValue * coupon * 0.5;  // Semi-annual payment
    couponIncome:numCoupons * couponPayment;

    // Carry = AI change + coupon income - financing cost
    aiChange + couponIncome - repoCost}

// Implied repo rate
// The repo rate that makes net basis = 0
// Higher implied repo = cheaper to deliver
impliedRepoRate:{[settleDate;deliveryDate;bondCleanPrice;bondAI;futuresPrice;cf;couponAccrual]
    daysToDelivery:deliveryDate - settleDate;
    if[daysToDelivery <= 0; :0f];

    dirtyPrice:bondCleanPrice + bondAI;
    invoicePrice:futuresPrice * cf;  // What you receive at delivery

    // Invoice price at delivery includes AI at delivery
    // Simplified: assume same AI
    invoicePriceWithAI:invoicePrice + bondAI;

    // Solve for repo rate:
    // DirtyPrice * (1 + r * days/360) = InvoicePrice + CouponIncome
    // r = (InvoicePrice + CouponIncome - DirtyPrice) / (DirtyPrice * days/360)

    numerator:invoicePriceWithAI + couponAccrual - dirtyPrice;
    denominator:dirtyPrice * daysToDelivery % 360;

    if[denominator = 0; :0f];
    numerator % denominator}

// =============================================================================
// CTD ANALYSIS
// =============================================================================

// Deliverable bond structure
// bonds: table with columns `sym`coupon`maturity`cleanPrice`cf
// or provide conversion factors

// Analyze a single deliverable bond
analyzeBond:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bond]
    sym:bond`sym;
    coupon:bond`coupon;
    maturity:bond`maturity;
    cleanPrice:bond`cleanPrice;
    cf:bond`cf;
    faceValue:100f;
    couponFreq:`SA;

    // Accrued interest
    ai:accruedInterest[settleDate;maturity;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;

    // Basis calculations
    gb:grossBasis[cleanPrice;futuresPrice;cf];

    // Carry with actual coupon payment check
    numCoupons:couponPaymentsBetween[settleDate;deliveryDate;maturity];
    couponPayment:faceValue * coupon * 0.5;
    couponIncome:numCoupons * couponPayment;
    carry:bondCarryFull[settleDate;deliveryDate;maturity;dirtyPrice;coupon;faceValue;repoRate];

    nb:netBasis[cleanPrice;futuresPrice;cf;carry];

    // Implied repo - use actual coupon income
    ir:impliedRepoRate[settleDate;deliveryDate;cleanPrice;ai;futuresPrice;cf;couponIncome];

    // Invoice price (what short receives upon delivery)
    invoicePrice:futuresPrice * cf + ai;

    // Yield to maturity
    ytm:yieldToMaturity[settleDate;maturity;coupon;couponFreq;faceValue;cleanPrice];

    `sym`coupon`maturity`cleanPrice`dirtyPrice`ai`cf`grossBasis`carry`netBasis`impliedRepo`invoicePrice`ytm!(
        sym;coupon;maturity;cleanPrice;dirtyPrice;ai;cf;gb;carry;nb;ir;invoicePrice;ytm)}

// Full CTD analysis for a basket of deliverable bonds
// Returns ranked table with CTD at top
ctdAnalysis:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]
    results:analyzeBond[curve;settleDate;deliveryDate;futuresPrice;repoRate] peach bonds;

    // Convert to table
    t:flip `sym`coupon`maturity`cleanPrice`dirtyPrice`ai`cf`grossBasis`carry`netBasis`impliedRepo`invoicePrice`ytm!
        (results`sym;results`coupon;results`maturity;results`cleanPrice;results`dirtyPrice;
         results`ai;results`cf;results`grossBasis;results`carry;results`netBasis;
         results`impliedRepo;results`invoicePrice;results`ytm);

    // Sort by implied repo (highest = CTD) or net basis (lowest = CTD)
    t:`impliedRepo xdesc t;

    // Add CTD flag
    t:update isCTD:(i=0) from t;

    t}

// Quick CTD identification
findCTD:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]
    results:ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];
    first results}

// =============================================================================
// FUTURES PRICING
// =============================================================================

// Theoretical futures price from CTD bond
// Forward price formula: FwdDirty = Dirty × (1 + r×t) - CouponPayments
theoreticalFuturesPrice:{[curve;settleDate;deliveryDate;ctdBond;repoRate]
    coupon:ctdBond`coupon;
    maturity:ctdBond`maturity;
    cleanPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;
    faceValue:100f;
    couponFreq:`SA;

    ai:accruedInterest[settleDate;maturity;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;

    daysToDelivery:deliveryDate - settleDate;

    // Check for coupon payments during holding period
    numCoupons:couponPaymentsBetween[settleDate;deliveryDate;maturity];
    couponPayment:faceValue * coupon * 0.5;  // Semi-annual payment

    // Forward dirty price = Dirty × (1 + r×t) - coupon payments received
    // (simplified: ignore reinvestment of coupons)
    fwdDirtyPrice:dirtyPrice * (1 + repoRate * daysToDelivery % 360);
    fwdDirtyPrice:fwdDirtyPrice - (numCoupons * couponPayment);

    // AI at delivery
    aiDelivery:accruedInterest[deliveryDate;maturity;coupon;couponFreq;faceValue];
    fwdCleanPrice:fwdDirtyPrice - aiDelivery;

    // Futures price = Forward Clean Price / CF
    fwdCleanPrice % cf}

// Basis point value (BPV/DV01) of futures position
futuresDV01:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;ctdBond;numContracts;contractSize]
    // DV01 of CTD bond
    coupon:ctdBond`coupon;
    maturity:ctdBond`maturity;
    cleanPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;

    // Bump curve and reprice using proper curve bumping
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];

    priceUp:priceBond[curveUp;settleDate;maturity;coupon;`SA;100f];
    priceDn:priceBond[curveDn;settleDate;maturity;coupon;`SA;100f];

    bondDV01:(priceDn - priceUp) % 2;

    // Futures DV01 = Bond DV01 / CF * contracts * contract size
    futDV01:bondDV01 % cf * numContracts * contractSize % 100;

    `bondDV01`futuresDV01`cf`contracts!(bondDV01;futDV01;cf;numContracts)}

// =============================================================================
// ONE-DAY FUTURES CARRY
// =============================================================================

// One-day carry for holding the CTD bond (underlying)
// Carry = Coupon accrual - Financing cost
// Positive carry means you earn income by holding
bondCarry1D:{[settleDate;maturityDate;cleanPrice;coupon;repoRate]
    faceValue:100f;
    couponFreq:`SA;

    // Accrued interest at settlement
    ai:accruedInterest[settleDate;maturityDate;coupon;couponFreq;faceValue];
    dirtyPrice:cleanPrice + ai;

    // Daily coupon accrual (ACT/365 for Treasuries)
    dailyCouponAccrual:faceValue * coupon % 365;

    // Daily financing cost (ACT/360 for repo)
    dailyFinancingCost:dirtyPrice * repoRate % 360;

    // Net carry (in price terms)
    dailyCouponAccrual - dailyFinancingCost}

// One-day carry for futures position
// Long futures = synthetically long CTD bond
// Carry is the bond carry divided by conversion factor
futuresCarry1D:{[settleDate;ctdBond;repoRate]
    coupon:ctdBond`coupon;
    maturity:ctdBond`maturity;
    cleanPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;

    // Bond carry
    bondCarry:bondCarry1D[settleDate;maturity;cleanPrice;coupon;repoRate];

    // Futures carry = Bond carry / CF
    futCarry:bondCarry % cf;

    `bondCarry1D`futuresCarry1D`cf`coupon`repoRate!(bondCarry;futCarry;cf;coupon;repoRate)}

// One-day carry AND roll-down for futures (requires curve)
// This gives the complete expected 1-day P&L assuming rates unchanged
// Components:
//   carry: coupon accrual - financing cost
//   rollDown: price change from bond aging 1 day on unchanged curve
//   total: carry + rollDown (theta)
futuresCarryRoll1D:{[curve;settleDate;ctdBond;repoRate]
    coupon:ctdBond`coupon;
    maturity:ctdBond`maturity;
    cleanPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;
    faceValue:100f;

    // 1. Carry component (same as before)
    bondCarry:bondCarry1D[settleDate;maturity;cleanPrice;coupon;repoRate];
    futCarry:bondCarry % cf;

    // 2. Roll-down component
    // Price bond today
    priceT0:priceBond[curve;settleDate;maturity;coupon;`SA;faceValue];

    // Roll curve forward 1 day and reprice
    curveT1:.swaps.rollCurve[curve;1%365];
    settleT1:settleDate + 1;
    priceT1:priceBond[curveT1;settleT1;maturity;coupon;`SA;faceValue];

    // Roll-down = change in clean price (positive if price rises as bond ages)
    bondRollDown:priceT1 - priceT0;
    futRollDown:bondRollDown % cf;

    // 3. Total theta = carry + roll-down
    bondTheta:bondCarry + bondRollDown;
    futTheta:futCarry + futRollDown;

    // 4. Mark-to-market vs theoretical
    // If we have a market price, show the difference
    priceDiff:cleanPrice - priceT0;

    `bondCarry`bondRollDown`bondTheta`futuresCarry`futuresRollDown`futuresTheta`futuresThetaTicks`cf`modelPrice`marketPrice`richCheap!(
        bondCarry;bondRollDown;bondTheta;
        futCarry;futRollDown;futTheta;32f*futTheta;
        cf;priceT0;cleanPrice;priceDiff)}

// Annualized carry (for comparison with yields)
futuresCarryAnn:{[settleDate;ctdBond;repoRate]
    carry:futuresCarry1D[settleDate;ctdBond;repoRate];
    bondCarryAnn:carry[`bondCarry1D] * 365;
    futCarryAnn:carry[`futuresCarry1D] * 365;
    carry,`bondCarryAnn`futuresCarryAnn!(bondCarryAnn;futCarryAnn)}

// Carry in ticks (1/32nds) for futures
// Treasury futures are quoted in 32nds
futuresCarry1DTicks:{[settleDate;ctdBond;repoRate]
    carry:futuresCarry1D[settleDate;ctdBond;repoRate];
    futCarryTicks:32f * carry`futuresCarry1D;
    carry,enlist[`futuresCarry1DTicks]!enlist futCarryTicks}

// Carry vs implied repo analysis
// Shows whether futures are rich or cheap based on carry
carryAnalysis:{[settleDate;deliveryDate;ctdBond;futuresPrice;repoRate]
    coupon:ctdBond`coupon;
    maturity:ctdBond`maturity;
    cleanPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;
    faceValue:100f;

    // Accrued interest
    ai:accruedInterest[settleDate;maturity;coupon;`SA;faceValue];
    dirtyPrice:cleanPrice + ai;

    // Days to delivery
    daysToDelivery:`int$deliveryDate - settleDate;

    // One-day carry
    carry1D:futuresCarry1D[settleDate;ctdBond;repoRate];

    // Total carry to delivery
    bondCarryTotal:carry1D[`bondCarry1D] * daysToDelivery;
    futCarryTotal:carry1D[`futuresCarry1D] * daysToDelivery;

    // Implied repo
    couponAccrual:faceValue * coupon * daysToDelivery % 365;
    implRepo:impliedRepoRate[settleDate;deliveryDate;cleanPrice;ai;futuresPrice;cf;couponAccrual];

    // Carry spread: implied repo - actual repo
    // Positive = futures cheap (earn more than financing), Negative = futures rich
    carrySpread:implRepo - repoRate;
    carrySpreadBps:carrySpread * 10000;

    `sym`cleanPrice`cf`daysToDelivery`bondCarry1D`futuresCarry1D`futuresCarry1DTicks`bondCarryTotal`futCarryTotal`repoRate`impliedRepo`carrySpreadBps!(
        ctdBond`sym;cleanPrice;cf;`float$daysToDelivery;
        carry1D`bondCarry1D;carry1D`futuresCarry1D;32f * carry1D`futuresCarry1D;
        bondCarryTotal;futCarryTotal;
        repoRate;implRepo;carrySpreadBps)}

// =============================================================================
// DELIVERY OPTIONS
// =============================================================================
// Treasury futures have three embedded options that benefit the short:
//   1. Wild Card Option - deliver after futures close using settlement price
//   2. Quality Option - choose which bond to deliver as yields change
//   3. End-of-Month Option - time delivery after last trading day

// -----------------------------------------------------------------------------
// WILD CARD OPTION
// -----------------------------------------------------------------------------
// After futures close (2pm CT), short has until 8pm CT to decide whether to
// deliver using that day's settlement price. If bond prices fall after 2pm,
// short can buy cheap and deliver at the higher settlement price.
//
// Model: Series of daily put options during delivery month
// Value = σ_intraday × √(hours/year) × price × N(0) × days_remaining

// Wild card option value for a single day
// intradayVol: annualized volatility during 2pm-8pm window (typically 0.10-0.20 of daily vol)
// bondPrice: CTD bond price
wildCardDaily:{[bondPrice;intradayVol]
    // 6 hours = 6/8760 of a year (252 trading days × ~8 hours)
    // Using simplified at-the-money option approximation: value ≈ 0.4 × σ × √T × S
    hoursPerYear:252 * 8;
    timeToExpiry:6 % hoursPerYear;
    0.4 * intradayVol * sqrt[timeToExpiry] * bondPrice}

// Total wild card option value for remaining delivery days
// daysRemaining: trading days left in delivery month
// intradayVol: intraday volatility (annualized)
wildCardOption:{[ctdBond;daysRemaining;intradayVol]
    bondPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;

    // Daily option value
    dailyValue:wildCardDaily[bondPrice;intradayVol];

    // Total value = sum of daily options (not quite additive, but approximation)
    // Discount slightly for overlapping nature
    totalBondValue:dailyValue * sqrt daysRemaining;

    // Convert to futures terms (divide by CF) and to 32nds
    futuresValue:totalBondValue % cf;
    ticks:32 * futuresValue;

    `bondValue`futuresValue`ticks32`daysRemaining`intradayVol`dailyValue!(
        totalBondValue;futuresValue;ticks;daysRemaining;intradayVol;dailyValue)}

// -----------------------------------------------------------------------------
// QUALITY OPTION
// -----------------------------------------------------------------------------
// The option to switch which bond is delivered if CTD changes as yields move.
// Value depends on:
//   - Net basis spread between CTD and next-cheapest bonds
//   - Yield volatility and duration differences
//   - Time to delivery

// Find breakeven yield shift for CTD to switch
// Returns the yield change (in bp) at which a different bond becomes CTD
qualityBreakeven:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]
    // Current CTD analysis
    baseResults:ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];
    currentCTD:first exec sym from baseResults where isCTD;

    // Test yield shifts from -100bp to +100bp
    shifts:-100 -75 -50 -25 0 25 50 75 100;

    findCTDAtShift:{[curve;settle;delivery;fut;repo;bonds;shift]
        // Bump curve
        bumpedCurve:.swaps.bumpCurve[curve;shift % 10000];
        // Reprice bonds at new yields
        repricedBonds:{[c;s;b]
            newPrice:priceBond[c;s;b`maturity;b`coupon;`SA;100f];
            b,enlist[`cleanPrice]!enlist newPrice
        }[bumpedCurve;settle] each bonds;
        // Run CTD analysis
        results:ctdAnalysis[bumpedCurve;settle;delivery;fut;repo;repricedBonds];
        ctd:first exec sym from results where isCTD;
        `shift`ctd!(shift;ctd)
    };

    shiftResults:findCTDAtShift[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds] peach shifts;

    // Find where CTD switches
    switches:select from ([]shift:shiftResults`shift;ctd:shiftResults`ctd) where ctd <> currentCTD;

    // Breakeven is first switch point
    if[0 = count switches;
        :`currentCTD`switchPoints`breakevenUp`breakevenDown!(currentCTD;();0Ni;0Ni)];

    upSwitches:select from switches where shift > 0;
    downSwitches:select from switches where shift < 0;

    breakevenUp:$[count upSwitches; first upSwitches`shift; 0Ni];
    breakevenDown:$[count downSwitches; last downSwitches`shift; 0Ni];

    `currentCTD`switchPoints`breakevenUp`breakevenDown`shiftResults!(
        currentCTD;switches;breakevenUp;breakevenDown;shiftResults)}

// Quality option value using scenario analysis
// yieldVol: annualized yield volatility (e.g., 0.01 = 100bp/year)
// horizon: time to delivery in years
qualityOption:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds;yieldVol;horizon]
    // Get breakeven analysis
    be:qualityBreakeven[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];

    // If no switch point, option has minimal value
    if[(null be`breakevenUp) and null be`breakevenDown;
        :`optionValue`futuresValue`ticks32`breakevenUp`breakevenDown`currentCTD`probSwitch!(
            0f;0f;0f;0Ni;0Ni;be`currentCTD;0f)];

    // Expected yield move over horizon
    expectedMove:yieldVol * sqrt horizon;
    expectedMoveBps:expectedMove * 10000;

    // Probability of hitting breakeven (simplified normal approximation)
    // P(|move| > breakeven) using normal distribution
    probUp:$[null be`breakevenUp; 0f;
        1 - .5 * (1 + .qmath.erf[(be`breakevenUp) % (expectedMoveBps * 1.414)])];
    probDown:$[null be`breakevenDown; 0f;
        .5 * (1 + .qmath.erf[(neg be`breakevenDown) % (expectedMoveBps * 1.414)])];
    probSwitch:probUp + probDown;

    // Get current results to find basis spread to 2nd cheapest
    baseResults:ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];
    ctdBasis:first exec netBasis from baseResults where isCTD;
    secondBasis:exec netBasis from baseResults where not isCTD;
    basisSpread:$[count secondBasis; (min secondBasis) - ctdBasis; 0f];

    // Option value = probability of switch × expected savings
    // Savings ≈ basis spread to 2nd cheapest
    optionValue:probSwitch * abs basisSpread;
    ticks:32 * optionValue;

    // Get CF for futures conversion
    cf:first exec cf from baseResults where isCTD;
    futuresValue:optionValue % cf;

    `optionValue`futuresValue`ticks32`breakevenUp`breakevenDown`currentCTD`probSwitch`basisSpread`expectedMoveBps!(
        optionValue;futuresValue;32*futuresValue;be`breakevenUp;be`breakevenDown;be`currentCTD;probSwitch;basisSpread;expectedMoveBps)}

// Simple error function approximation (if .qmath not available)
// erf(x) ≈ sign(x) * sqrt(1 - exp(-x^2 * (4/π + ax^2)/(1 + ax^2)))
// where a = 8(π-3)/(3π(4-π))
.qmath.erf:{[x]
    if[x = 0; :0f];
    a:0.147;  // Approximation constant
    x2:x * x;
    inner:(4 % 3.14159) + (a * x2);
    inner:inner % (1 + a * x2);
    sign:$[x > 0; 1; -1];
    sign * sqrt 1 - exp neg x2 * inner}

// -----------------------------------------------------------------------------
// END-OF-MONTH OPTION
// -----------------------------------------------------------------------------
// Futures stop trading ~7 business days before month end, but delivery
// continues until last business day. Short can wait to see how bonds trade.
//
// Model: Option to time delivery optimally during EOM period
// Similar to wild card but over multiple full trading days

// End-of-month option value
// eomDays: business days between last trading day and last delivery day (~7)
// dailyVol: annualized daily volatility of CTD bond
endOfMonthOption:{[ctdBond;eomDays;dailyVol]
    bondPrice:ctdBond`cleanPrice;
    cf:ctdBond`cf;

    // Model as option to deliver on best of N days
    // Value ≈ σ_daily × E[max of N normals] × price
    // E[max of N standard normals] ≈ sqrt(2 × ln(N)) for large N
    // σ_daily = σ_annual / sqrt(252)

    // Convert annualized vol to daily standard deviation
    dailyStdDev:dailyVol % sqrt 252;

    // Expected max of N iid normals approximation
    expectedMax:$[eomDays <= 1; 0f; sqrt 2 * log eomDays];

    // Option value = daily std dev × expected max factor × price
    bondValue:dailyStdDev * expectedMax * bondPrice;

    futuresValue:bondValue % cf;
    ticks:32 * futuresValue;

    `bondValue`futuresValue`ticks32`eomDays`dailyVol`expectedMax`dailyStdDev!(
        bondValue;futuresValue;ticks;eomDays;dailyVol;expectedMax;dailyStdDev)}

// -----------------------------------------------------------------------------
// COMBINED DELIVERY OPTIONS
// -----------------------------------------------------------------------------

// Default volatility parameters
// Calibrated to typical Treasury market conditions
// Note: These are PRICE volatilities (annualized), not yield vols
defaultVolParams:`intradayVol`dailyVol`yieldVol!(
    0.005;   // ~0.5% annualized intraday (2pm-8pm window), roughly 1/3 of daily
    0.015;   // ~1.5% annualized daily price vol for 10Y Treasury (~10bp/day yield)
    0.01     // ~100bp annualized yield vol
  )

// Total delivery option value
// Combines wild card, quality, and end-of-month options
deliveryOptionValue:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds;params]
    // Merge with defaults
    p:defaultVolParams,params;

    // Get CTD bond
    ctdBond:findCTD[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];

    // Days calculations
    daysToDelivery:deliveryDate - settleDate;
    deliveryMonth:`month$deliveryDate;
    lastDeliveryDay:.tsy.lastDeliveryDay[deliveryMonth];
    lastTradingDay:lastDeliveryDay - 7;  // Approximate
    eomDays:lastDeliveryDay - lastTradingDay;

    // Time to delivery in years
    horizon:daysToDelivery % 365;

    // 1. Wild Card Option
    wc:wildCardOption[ctdBond;daysToDelivery;p`intradayVol];

    // 2. Quality Option
    qo:qualityOption[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds;p`yieldVol;horizon];

    // 3. End-of-Month Option
    eom:endOfMonthOption[ctdBond;eomDays;p`dailyVol];

    // Total (not simply additive due to correlation, apply haircut)
    correlationHaircut:0.85;  // Options are somewhat correlated
    totalFutValue:(wc[`futuresValue] + qo[`futuresValue] + eom[`futuresValue]) * correlationHaircut;
    totalTicks:32 * totalFutValue;

    `wildCard`quality`endOfMonth`total`totalTicks32`ctdBond`params!(
        wc;qo;eom;totalFutValue;totalTicks;ctdBond`sym;p)}

// Summary display for delivery options
deliveryOptionsSummary:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds;params]
    result:deliveryOptionValue[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds;params];

    wc:result`wildCard;
    qo:result`quality;
    eom:result`endOfMonth;

    -1 "";
    -1 "=============================================================================";
    -1 "                    DELIVERY OPTIONS ANALYSIS";
    -1 "=============================================================================";
    -1 "";
    -1 "Settlement:     ",string settleDate;
    -1 "Delivery:       ",string deliveryDate;
    -1 "Futures Price:  ",string futuresPrice;
    -1 "CTD Bond:       ",string result`ctdBond;
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "WILD CARD OPTION (2pm-8pm daily timing)";
    -1 "-----------------------------------------------------------------------------";
    -1 "  Days remaining:     ",string wc`daysRemaining;
    -1 "  Intraday vol:       ",string[100*wc`intradayVol],"%";
    -1 "  Daily value:        ",string[0.001*`long$1000*wc`dailyValue]," pts";
    -1 "  Total value:        ",string[0.01*`long$100*wc`ticks32]," /32nds";
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "QUALITY OPTION (CTD switch)";
    -1 "-----------------------------------------------------------------------------";
    -1 "  Current CTD:        ",string qo`currentCTD;
    -1 "  Breakeven up:       ",$[null qo`breakevenUp; "No switch"; string[qo`breakevenUp],"bp"];
    -1 "  Breakeven down:     ",$[null qo`breakevenDown; "No switch"; string[qo`breakevenDown],"bp"];
    -1 "  Expected move:      ",string[`int$qo`expectedMoveBps],"bp";
    -1 "  Switch probability: ",string[`int$100*qo`probSwitch],"%";
    -1 "  Basis spread:       ",string[0.01*`long$100*32*qo`basisSpread]," /32nds";
    -1 "  Option value:       ",string[0.01*`long$100*qo`ticks32]," /32nds";
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "END-OF-MONTH OPTION (post last-trade delivery timing)";
    -1 "-----------------------------------------------------------------------------";
    -1 "  EOM days:           ",string eom`eomDays;
    -1 "  Daily vol:          ",string[100*eom`dailyVol],"%";
    -1 "  Option value:       ",string[0.01*`long$100*eom`ticks32]," /32nds";
    -1 "";
    -1 "-----------------------------------------------------------------------------";
    -1 "TOTAL DELIVERY OPTION VALUE";
    -1 "-----------------------------------------------------------------------------";
    -1 "  Wild Card:          ",string[0.01*`long$100*wc`ticks32]," /32nds";
    -1 "  Quality:            ",string[0.01*`long$100*qo`ticks32]," /32nds";
    -1 "  End-of-Month:       ",string[0.01*`long$100*eom`ticks32]," /32nds";
    -1 "  ----------------------------------------";
    -1 "  TOTAL (w/haircut):  ",string[0.01*`long$100*result`totalTicks32]," /32nds";
    -1 "";
    -1 "=============================================================================";
    -1 "";

    result}

// =============================================================================
// DISPLAY FUNCTIONS
// =============================================================================

showCTDAnalysis:{[results]
    -1 "";
    -1 "=== CTD ANALYSIS ===";
    -1 "";
    -1 "Bond          Coupon  Maturity    Price    CF      Gross   Net     Impl Repo  CTD";
    -1 "----          ------  --------    -----    --      -----   ---     ---------  ---";
    {[r]
        ctdFlag:$[r`isCTD;"  <--";""];
        -1 (10$string r`sym),"  ",
           (6$string 100*r`coupon),"%  ",
           (10$string r`maturity),"  ",
           (7$string 0.01*`long$100*r`cleanPrice),"  ",
           (6$string r`cf),"  ",
           (6$string 0.01*`long$100*r`grossBasis),"  ",
           (6$string 0.01*`long$100*r`netBasis),"  ",
           (8$string 0.01*`long$10000*r`impliedRepo),"%",
           ctdFlag
    } each results;
    -1 "";
    -1 "CTD Bond: ",string (first results)`sym;
    -1 "Implied Repo: ",string[100*(first results)`impliedRepo],"%";
    -1 "";}

// =============================================================================
// EXAMPLE
// =============================================================================

example:{[]
    -1 "";
    -1 "=== CTD PRICER EXAMPLE ===";
    -1 "";

    // Need swaps.q loaded for curve
    if[not `buildCurve in key `.swaps;
        -1 "ERROR: Load swaps.q first (\\l swaps.q)";
        :()];

    // Build a yield curve
    curveDate:"D"$"2025.01.15";
    tenors:`3M`6M`1Y`2Y`3Y`5Y`7Y`10Y`20Y`30Y;
    rates:0.0425 0.0438 0.0452 0.0468 0.0479 0.0495 0.0508 0.0522 0.0545 0.0560;
    curve:.swaps.buildCurve[tenors;rates;`asOfDate`frequency!(curveDate;`3M)];

    -1 "Curve Date: ",string curveDate;
    -1 "";

    // Define deliverable bonds for 10Y Treasury futures
    // Typically 6.5-10Y remaining maturity
    bonds:(
        `sym`coupon`maturity`cleanPrice`cf!(
            `T_4_125_2034;0.04125;"D"$"2034.08.15";98.50;0.8432);
        `sym`coupon`maturity`cleanPrice`cf!(
            `T_4_375_2034;0.04375;"D"$"2034.11.15";100.25;0.8621);
        `sym`coupon`maturity`cleanPrice`cf!(
            `T_4_500_2035;0.045;"D"$"2035.02.15";101.125;0.8756);
        `sym`coupon`maturity`cleanPrice`cf!(
            `T_4_750_2035;0.0475;"D"$"2035.05.15";103.00;0.8892);
        `sym`coupon`maturity`cleanPrice`cf!(
            `T_5_000_2035;0.05;"D"$"2035.08.15";105.50;0.9127)
    );

    -1 "Deliverable Bonds:";
    show ([]
        sym:bonds`sym;
        coupon:100*bonds`coupon;
        maturity:bonds`maturity;
        price:bonds`cleanPrice;
        cf:bonds`cf);
    -1 "";

    // Futures parameters
    settleDate:curveDate;
    deliveryDate:"D"$"2025.03.20";  // March futures delivery
    futuresPrice:116.50;  // Quoted futures price
    repoRate:0.045;  // Repo rate for financing

    -1 "Futures Parameters:";
    -1 "  Settlement:    ",string settleDate;
    -1 "  Delivery:      ",string deliveryDate;
    -1 "  Futures Price: ",string futuresPrice;
    -1 "  Repo Rate:     ",string[100*repoRate],"%";
    -1 "";

    // Run CTD analysis
    results:ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds];

    showCTDAnalysis results;

    // Show detailed CTD bond analysis
    ctd:first results;
    -1 "=== CTD BOND DETAILS ===";
    -1 "";
    -1 "Symbol:         ",string ctd`sym;
    -1 "Coupon:         ",string[100*ctd`coupon],"%";
    -1 "Maturity:       ",string ctd`maturity;
    -1 "Clean Price:    ",string ctd`cleanPrice;
    -1 "Dirty Price:    ",string ctd`dirtyPrice;
    -1 "Accrued Int:    ",string ctd`ai;
    -1 "Conv Factor:    ",string ctd`cf;
    -1 "YTM:            ",string[100*ctd`ytm],"%";
    -1 "";
    -1 "Gross Basis:    ",string[ctd`grossBasis]," (ticks: ",string[32*ctd`grossBasis],")";
    -1 "Carry:          ",string ctd`carry;
    -1 "Net Basis:      ",string[ctd`netBasis]," (ticks: ",string[32*ctd`netBasis],")";
    -1 "Implied Repo:   ",string[100*ctd`impliedRepo],"%";
    -1 "Invoice Price:  ",string ctd`invoicePrice;
    -1 "";

    // Theoretical futures price
    theoFut:theoreticalFuturesPrice[curve;settleDate;deliveryDate;ctd;repoRate];
    -1 "Theoretical Futures: ",string theoFut;
    -1 "Market Futures:      ",string futuresPrice;
    -1 "Rich/Cheap:          ",string[futuresPrice - theoFut]," (",string[32*(futuresPrice-theoFut)]," ticks)";
    -1 "";

    results}

// Usage help
usage:{[]
    -1 "";
    -1 "=== .ctd USAGE REFERENCE ===";
    -1 "";
    -1 "// 1. SETUP (load swaps.q first)";
    -1 "\\l swaps.q";
    -1 "\\l ctd.q";
    -1 "";
    -1 "// 2. BUILD CURVE";
    -1 "curve:.swaps.buildCurve[tenors;rates;`asOfDate`frequency!(curveDate;`3M)]";
    -1 "";
    -1 "// 3. DEFINE DELIVERABLE BONDS";
    -1 "bonds:(";
    -1 "    `sym`coupon`maturity`cleanPrice`cf!(`T_4_125_2034;0.04125;2034.08.15;98.50;0.8432);";
    -1 "    `sym`coupon`maturity`cleanPrice`cf!(`T_4_375_2034;0.04375;2034.11.15;100.25;0.8621);";
    -1 "    ...)";
    -1 "";
    -1 "// 4. RUN CTD ANALYSIS";
    -1 "results:.ctd.ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]";
    -1 ".ctd.showCTDAnalysis results";
    -1 "";
    -1 "// 5. GET CTD BOND";
    -1 "ctd:.ctd.findCTD[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]";
    -1 "";
    -1 "// 6. BOND PRICING";
    -1 ".ctd.priceBond[curve;settleDate;maturity;coupon;`SA;100]";
    -1 ".ctd.valueBond[curve;settleDate;maturity;coupon;`SA;100]";
    -1 ".ctd.accruedInterest[settleDate;maturity;coupon;`SA;100]";
    -1 ".ctd.yieldToMaturity[settleDate;maturity;coupon;`SA;100;cleanPrice]";
    -1 "";
    -1 "// 7. BASIS CALCULATIONS";
    -1 ".ctd.grossBasis[bondPrice;futuresPrice;cf]";
    -1 ".ctd.netBasis[bondPrice;futuresPrice;cf;carry]";
    -1 ".ctd.impliedRepoRate[settleDate;deliveryDate;bondPrice;ai;futuresPrice;cf;couponAccrual]";
    -1 ".ctd.bondCarry[settleDate;deliveryDate;dirtyPrice;coupon;faceValue;repoRate]";
    -1 "";
    -1 "// 8. FUTURES";
    -1 ".ctd.theoreticalFuturesPrice[curve;settleDate;deliveryDate;ctdBond;repoRate]";
    -1 ".ctd.futuresDV01[curve;settleDate;deliveryDate;futuresPrice;repoRate;ctdBond;numContracts;contractSize]";
    -1 "";
    -1 "// 9. ONE-DAY CARRY";
    -1 ".ctd.futuresCarry1D[settleDate;ctdBond;repoRate]           / one-day carry (no curve)";
    -1 ".ctd.futuresCarryRoll1D[curve;settleDate;ctdBond;repoRate] / carry + roll-down (with curve)";
    -1 ".ctd.futuresCarry1DTicks[settleDate;ctdBond;repoRate]      / carry in 32nds";
    -1 ".ctd.futuresCarryAnn[settleDate;ctdBond;repoRate]          / annualized carry";
    -1 ".ctd.carryAnalysis[settleDate;deliveryDate;ctdBond;futuresPrice;repoRate]  / full analysis";
    -1 "";
    -1 "// 10. DELIVERY OPTIONS (wild card, quality, end-of-month)";
    -1 "params:`intradayVol`dailyVol`yieldVol!(0.03;0.15;0.01)     / vol parameters";
    -1 ".ctd.wildCardOption[ctdBond;daysRemaining;intradayVol]     / wild card value";
    -1 ".ctd.qualityBreakeven[curve;settle;delivery;fut;repo;bonds] / CTD switch points";
    -1 ".ctd.qualityOption[curve;settle;delivery;fut;repo;bonds;yieldVol;horizon]";
    -1 ".ctd.endOfMonthOption[ctdBond;eomDays;dailyVol]            / EOM option value";
    -1 ".ctd.deliveryOptionValue[curve;settle;delivery;fut;repo;bonds;params] / all options";
    -1 ".ctd.deliveryOptionsSummary[curve;settle;delivery;fut;repo;bonds;params] / display";
    -1 "";
    -1 "// 11. EXAMPLE";
    -1 ".ctd.example[]";
    -1 "";}

help:{[]
    -1 "";
    -1 "=== .ctd FUNCTION REFERENCE ===";
    -1 "";
    -1 "BOND PRICING:";
    -1 "  priceBond[curve;settle;mat;cpn;freq;face]  - Clean price from curve";
    -1 "  valueBond[curve;settle;mat;cpn;freq;face]  - Full valuation dict";
    -1 "  priceBondFromYield[settle;mat;cpn;freq;face;ytm] - Price from yield";
    -1 "  yieldToMaturity[settle;mat;cpn;freq;face;price]  - YTM from price";
    -1 "  accruedInterest[settle;mat;cpn;freq;face]  - Accrued interest";
    -1 "  genBondCFDates[settle;mat;freq]            - Cash flow dates";
    -1 "";
    -1 "CONVERSION FACTORS:";
    -1 "  conversionFactor[delivery;mat;cpn;freq]    - Calculate CF";
    -1 "";
    -1 "BASIS ANALYSIS:";
    -1 "  grossBasis[bondPrice;futPrice;cf]          - Gross basis";
    -1 "  netBasis[bondPrice;futPrice;cf;carry]      - Net basis";
    -1 "  bondCarry[settle;delivery;dirty;cpn;face;repo] - Carry to delivery";
    -1 "  impliedRepoRate[settle;delivery;price;ai;fut;cf;cpnAccr] - Implied repo";
    -1 "";
    -1 "CTD ANALYSIS:";
    -1 "  analyzeBond[curve;settle;delivery;fut;repo;bond] - Analyze one bond";
    -1 "  ctdAnalysis[curve;settle;delivery;fut;repo;bonds] - Full CTD table";
    -1 "  findCTD[curve;settle;delivery;fut;repo;bonds]     - Get CTD bond";
    -1 "  showCTDAnalysis[results]                          - Display results";
    -1 "";
    -1 "FUTURES:";
    -1 "  theoreticalFuturesPrice[curve;settle;delivery;ctd;repo] - Theo price";
    -1 "  futuresDV01[curve;settle;delivery;fut;repo;ctd;n;size]  - Futures DV01";
    -1 "";
    -1 "ONE-DAY CARRY:";
    -1 "  bondCarry1D[settle;mat;price;cpn;repo]          - One-day bond carry";
    -1 "  futuresCarry1D[settle;ctdBond;repo]             - One-day futures carry (no curve)";
    -1 "  futuresCarryRoll1D[curve;settle;ctdBond;repo]   - Carry + roll-down (with curve)";
    -1 "  futuresCarry1DTicks[settle;ctdBond;repo]        - Carry in 32nds";
    -1 "  futuresCarryAnn[settle;ctdBond;repo]            - Annualized carry";
    -1 "  carryAnalysis[settle;delivery;ctd;fut;repo]     - Full carry vs implied repo";
    -1 "";
    -1 "DELIVERY OPTIONS:";
    -1 "  wildCardOption[ctdBond;daysRemaining;intradayVol] - Wild card (2pm-8pm timing)";
    -1 "  qualityBreakeven[curve;settle;delivery;fut;repo;bonds] - CTD switch breakevens";
    -1 "  qualityOption[curve;settle;delivery;fut;repo;bonds;yieldVol;horizon] - Quality option";
    -1 "  endOfMonthOption[ctdBond;eomDays;dailyVol]        - End-of-month timing option";
    -1 "  deliveryOptionValue[curve;settle;delivery;fut;repo;bonds;params] - All three options";
    -1 "  deliveryOptionsSummary[curve;settle;delivery;fut;repo;bonds;params] - Display summary";
    -1 "";
    -1 "UTILITIES:";
    -1 "  example[]  - Run full example";
    -1 "  usage[]    - Quick reference";
    -1 "  help[]     - This help";
    -1 "";}

\d .

-1 "Loaded .ctd namespace v",.ctd.version;
-1 "Functions: ctdAnalysis, findCTD, priceBond, impliedRepoRate";
-1 "Run .ctd.usage[] for quick reference, .ctd.example[] for demo";
