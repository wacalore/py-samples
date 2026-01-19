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

// Calculate carry for a bond to delivery
// Carry = (Coupon Accrual - Repo Cost) during holding period
bondCarry:{[settleDate;deliveryDate;dirtyPrice;coupon;faceValue;repoRate]
    daysHeld:deliveryDate - settleDate;
    if[daysHeld <= 0; :0f];

    // Financing cost (repo)
    repoCost:dirtyPrice * repoRate * daysHeld % 360;

    // Coupon accrual (simplified - linear accrual)
    couponAccrual:faceValue * coupon * daysHeld % 365;

    // Carry = income - cost
    couponAccrual - repoCost}

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

    // Carry (coupon accrual - financing)
    daysToDelivery:deliveryDate - settleDate;
    couponAccrual:faceValue * coupon * daysToDelivery % 365;
    carry:bondCarry[settleDate;deliveryDate;dirtyPrice;coupon;faceValue;repoRate];

    nb:netBasis[cleanPrice;futuresPrice;cf;carry];

    // Implied repo
    ir:impliedRepoRate[settleDate;deliveryDate;cleanPrice;ai;futuresPrice;cf;couponAccrual];

    // Invoice price (what short receives upon delivery)
    invoicePrice:futuresPrice * cf + ai;

    // Yield to maturity
    ytm:yieldToMaturity[settleDate;maturity;coupon;couponFreq;faceValue;cleanPrice];

    `sym`coupon`maturity`cleanPrice`dirtyPrice`ai`cf`grossBasis`carry`netBasis`impliedRepo`invoicePrice`ytm!(
        sym;coupon;maturity;cleanPrice;dirtyPrice;ai;cf;gb;carry;nb;ir;invoicePrice;ytm)}

// Full CTD analysis for a basket of deliverable bonds
// Returns ranked table with CTD at top
ctdAnalysis:{[curve;settleDate;deliveryDate;futuresPrice;repoRate;bonds]
    results:analyzeBond[curve;settleDate;deliveryDate;futuresPrice;repoRate] each bonds;

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

    // Forward dirty price
    financingCost:dirtyPrice * repoRate * daysToDelivery % 360;
    couponIncome:faceValue * coupon * daysToDelivery % 365;  // Simplified
    fwdDirtyPrice:dirtyPrice + financingCost - couponIncome;

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
    -1 "// 10. EXAMPLE";
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
    -1 "UTILITIES:";
    -1 "  example[]  - Run full example";
    -1 "  usage[]    - Quick reference";
    -1 "  help[]     - This help";
    -1 "";}

\d .

-1 "Loaded .ctd namespace v",.ctd.version;
-1 "Functions: ctdAnalysis, findCTD, priceBond, impliedRepoRate";
-1 "Run .ctd.usage[] for quick reference, .ctd.example[] for demo";
