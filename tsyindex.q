// =============================================================================
// TREASURY INDEX REBALANCING PREDICTION
// =============================================================================
// Predicts monthly index rebalancing for Treasury indices
// Supports Bloomberg US Treasury Index and ICE BofA US Treasury Index
//
// Usage: \l tsyindex.q
// Dependencies: tsy.q, ctd.q, invoicepricer.q
// =============================================================================

\d .tsyindex

version:"0.1.0"

// =============================================================================
// CONFIGURATION & DATA STRUCTURES
// =============================================================================

// Index rules configuration
// Bloomberg: $300M min outstanding, 1Y+ remaining maturity
// ICE BofA: $250M min outstanding, 1Y+ remaining maturity
indexRules:([index:`BBERG`ICE] minOutstanding:300e6 250e6; minRemMaturity:1.0 1.0; maxRemMaturity:100 100f; secTypes:((`Note`Bond);(`Note`Bond)); excludeTIPS:11b; excludeFRN:11b; rebalanceDay:`monthEnd`monthEnd)

// Treasury auction schedule (typical patterns as of 2025)
// Note: actual dates may vary; this is for forecasting
auctionSchedule:([]
    origTerm:`2Y`3Y`5Y`7Y`10Y`20Y`30Y;
    auctionWeek:2 2 2 4 2 3 2i;          // Week of month (1-4)
    auctionDayOfWeek:1 1 3 4 3 3 4i;     // 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri
    frequency:`Monthly`Monthly`Monthly`Monthly`Monthly`Quarterly`Quarterly;
    settleDays:1 1 1 1 1 1 1i;           // T+1 for all coupons
    typicalSize:57e9 56e9 64e9 44e9 42e9 13e9 22e9  // Typical auction size
  )

// Quarterly months for 20Y/30Y auctions (Feb, May, Aug, Nov)
quarterlyMonths:2 5 8 11i

// Maturity buckets
maturityBuckets:`1_3Y`3_5Y`5_7Y`7_10Y`10_20Y`20_30Y
bucketRanges:((1f;3f);(3f;5f);(5f;7f);(7f;10f);(10f;20f);(20f;30f))

// =============================================================================
// DATE UTILITIES
// =============================================================================

// First day of month
firstOfMonth:{[ym]
    y:`year$ym; m:`mm$ym;
    "D"$string[y],".",(-2#"0",string m),".01"}

// Last day of month
lastOfMonth:{[ym]
    -1 + firstOfMonth ym + 1}

// Check if weekend
isWeekend:{[d] (d mod 7) in 0 1}

// Next business day
nextBizDay:{[d] while[isWeekend d; d+:1]; d}

// Previous business day
prevBizDay:{[d] while[isWeekend d; d-:1]; d}

// Last business day of month
lastBizDayOfMonth:{[ym] prevBizDay lastOfMonth ym}

// Remaining maturity in years
remainingYears:{[maturityDate;asOfDate]
    (maturityDate - asOfDate) % 365.25}

// Get nth occurrence of day-of-week in month
// dow: 1=Mon, 2=Tue, ..., 7=Sun
// n: 1=first, 2=second, etc.
nthDayOfWeek:{[ym;dow;n]
    fom:firstOfMonth ym;
    // Day of week for first of month (q: 0=Sat, 1=Sun, 2=Mon, ...)
    fomDow:2 + fom mod 7;
    if[fomDow > 7; fomDow-:7];
    // Days to add to get to target dow
    daysToAdd:(dow - fomDow) mod 7;
    // First occurrence
    firstOcc:fom + daysToAdd;
    // Nth occurrence
    firstOcc + 7 * n - 1}

// =============================================================================
// MATURITY BUCKET FUNCTIONS
// =============================================================================

// Assign maturity bucket based on remaining years
assignBucket:{[remYears]
    if[remYears < 1; :`UNDER_1Y];
    if[remYears < 3; :`1_3Y];
    if[remYears < 5; :`3_5Y];
    if[remYears < 7; :`5_7Y];
    if[remYears < 10; :`7_10Y];
    if[remYears < 20; :`10_20Y];
    `20_30Y}

// Vectorized bucket assignment
assignBuckets:{[remYearsList]
    assignBucket each remYearsList}

// =============================================================================
// DURATION CALCULATIONS
// =============================================================================

// Modified duration for a bond (uses invoicepricer.q if available, else computes)
// settleDate: settlement date
// maturityDate: bond maturity
// coupon: annual coupon rate (e.g., 0.045 for 4.5%)
// ytm: yield to maturity (annual, e.g., 0.04 for 4%)
bondDuration:{[settleDate;maturityDate;coupon;ytm]
    days:maturityDate - settleDate;
    if[days <= 0; :0f];

    // Number of semiannual periods
    nPeriods:`long$1 + days % 182;

    // Time to each cash flow in years
    times:(1 + til nPeriods) % 2f;

    // Cash flows: coupon/2 each period, plus 100 at maturity
    semiCoupon:coupon * 50f;
    cfs:((nPeriods - 1)#semiCoupon),semiCoupon + 100f;

    // Discount factors
    dfs:xexp[1 + ytm % 2; neg 2 * times];

    // Dirty price
    price:sum cfs * dfs;
    if[price <= 0; :0f];

    // Macaulay duration
    macDur:(sum times * cfs * dfs) % price;

    // Modified duration = Macaulay / (1 + y/2)
    macDur % 1 + ytm % 2}

// Batch duration calculation for a table of bonds
// Assumes bonds table has: maturityDate, coupon, ytm (or use coupon as proxy)
bondDurationBatch:{[settleDate;bonds]
    // If ytm column exists, use it; otherwise use coupon as proxy
    ytms:$[`ytm in cols bonds; bonds`ytm; bonds`coupon];
    .tsyindex.bondDuration[settleDate;;]'[bonds`maturityDate;bonds`coupon;ytms]}

// =============================================================================
// YIELD SOLVER
// =============================================================================

// Calculate dirty price from yield (helper for solver)
// Returns dirty price given settlement, maturity, coupon, and yield
dirtyPriceFromYield:{[settleDate;maturityDate;coupon;ytm]
    days:maturityDate - settleDate;
    if[days <= 0; :0f];

    nPeriods:`long$1 + days % 182;
    times:(1 + til nPeriods) % 2f;
    semiCoupon:coupon * 50f;
    cfs:((nPeriods - 1)#semiCoupon),semiCoupon + 100f;
    dfs:xexp[1 + ytm % 2; neg 2 * times];
    sum cfs * dfs}

// Solve for yield given clean price (Newton-Raphson)
// settleDate: settlement date
// maturityDate: bond maturity
// coupon: annual coupon rate
// cleanPrice: market clean price
// Returns: yield to maturity (annual)
ytmFromPrice:{[settleDate;maturityDate;coupon;cleanPrice]
    days:maturityDate - settleDate;
    if[days <= 0; :0f];

    // Estimate accrued interest (simple approximation)
    // Days since last coupon (assume semiannual, 182 day periods)
    daysSinceCoupon:(days mod 182);
    ai:coupon * 50f * daysSinceCoupon % 182;
    dirtyPrice:cleanPrice + ai;

    // Initial guess: simple yield approximation
    remYears:days % 365.25;
    ytm0:(coupon + (100 - cleanPrice) % remYears) % ((100 + cleanPrice) % 2);
    ytm0:0.001|ytm0;  // floor at 0.1%

    // Newton-Raphson iteration
    ytm:ytm0;
    do[20;
        price:.tsyindex.dirtyPriceFromYield[settleDate;maturityDate;coupon;ytm];
        if[0 = price; :ytm];

        // Numerical derivative (dP/dy)
        bump:0.0001;
        priceUp:.tsyindex.dirtyPriceFromYield[settleDate;maturityDate;coupon;ytm+bump];
        dPdY:(priceUp - price) % bump;
        if[0 = dPdY; :ytm];

        // Newton step
        delta:(price - dirtyPrice) % dPdY;
        ytm:0.0001|ytm - delta;  // floor at 0.01%

        // Check convergence
        if[0.00001 > abs delta; :ytm]];
    ytm}

// Vectorized yield solver
ytmFromPriceBatch:{[settleDate;maturityDates;coupons;cleanPrices]
    .tsyindex.ytmFromPrice[settleDate;;;]'[maturityDates;coupons;cleanPrices]}

// =============================================================================
// INDEX ELIGIBILITY
// =============================================================================

// Filter bonds eligible for index
// bonds: table from tsy.q with cusip, securityType, coupon, maturityDate, issueDate, origTerm
// asOfDate: date to evaluate eligibility
// indexName: `BBERG or `ICE
filterEligibleForIndex:{[bonds;asOfDate;indexName]
    rules:indexRules indexName;

    // Start with bonds outstanding as of date
    eligible:select from bonds where issueDate <= asOfDate, maturityDate > asOfDate;

    // Filter by security type
    eligible:select from eligible where securityType in rules`secTypes;

    // Calculate remaining maturity
    eligible:update remYears:(maturityDate - asOfDate) % 365.25 from eligible;

    // Filter by minimum remaining maturity
    eligible:select from eligible where remYears >= rules`minRemMaturity;

    // Exclude TIPS (identified by origTerm containing "TIP" or very specific patterns)
    if[rules`excludeTIPS;
        eligible:select from eligible where not origTerm like "*TIP*"];

    // Exclude FRNs (identified by origTerm containing "FRN")
    if[rules`excludeFRN;
        eligible:select from eligible where not origTerm like "*FRN*"];

    // Add maturity bucket
    eligible:update bucket:.tsyindex.assignBuckets remYears from eligible;

    eligible}

// =============================================================================
// INDEX COMPOSITION
// =============================================================================

// Build full index composition
// bonds: bond table from tsy.q
// asOfDate: composition date
// indexName: `BBERG or `ICE
// prices: dict cusip->cleanPrice (optional, use (::) for par assumption)
// outstandings: dict cusip->outstanding amount (optional, use (::) for auction size proxy)
buildIndexComposition:{[bonds;asOfDate;indexName;prices;outstandings]
    // Get eligible bonds
    eligible:.tsyindex.filterEligibleForIndex[bonds;asOfDate;indexName];
    if[0 = count eligible; :eligible];

    // Add prices (use par=100 if not provided)
    eligible:update cleanPrice:100f from eligible;
    hasPrices:not prices ~ (::);
    if[hasPrices;
        eligible:update cleanPrice:prices cusip from eligible where cusip in key prices];

    // Use outstanding amounts from bonds table (total outstanding, not SOMA-adjusted)
    // Bloomberg index uses total outstanding, not publicFloat
    eligible:update parAmount:outstanding from eligible;
    eligible:update parAmount:?[null parAmount; 1e9; parAmount] from eligible;  // fallback default
    // Override with user-provided outstandings if given
    if[not outstandings ~ (::);
        eligible:update parAmount:outstandings cusip from eligible where cusip in key outstandings];

    // Calculate yields: solve from price if prices provided, else use coupon (par assumption)
    eligible:$[hasPrices;
        update ytm:.tsyindex.ytmFromPrice[asOfDate;;;]'[maturityDate;coupon;cleanPrice] from eligible;
        update ytm:coupon from eligible];

    // Calculate durations using solved yields
    eligible:update modDuration:.tsyindex.bondDuration[asOfDate;;]'[maturityDate;coupon;ytm] from eligible;

    // Calculate market values (par amount * price / 100)
    eligible:update marketValue:parAmount * cleanPrice % 100 from eligible;

    // Calculate weights
    totalMV:sum eligible`marketValue;
    eligible:update weight:marketValue % totalMV from eligible;

    // Calculate duration contribution
    eligible:update durationContrib:weight * modDuration from eligible;

    // Select final columns
    select cusip, securityType, coupon, maturityDate, issueDate, origTerm,
           remYears, bucket, cleanPrice, ytm, parAmount, marketValue,
           weight, modDuration, durationContrib
    from eligible}

// Calculate index-level statistics
indexStats:{[composition]
    if[0 = count composition;
        :`totalMV`indexDuration`avgCoupon`bondCount!(0f;0f;0f;0i)];

    totalMV:sum composition`marketValue;
    indexDur:sum composition`durationContrib;
    avgCoupon:wavg[composition`weight;composition`coupon];

    `totalMV`indexDuration`avgCoupon`bondCount!(totalMV;indexDur;avgCoupon;count composition)}

// Duration breakdown by maturity bucket
durationByBucket:{[composition]
    if[0 = count composition;
        :([] bucket:maturityBuckets; weight:6#0f; durationContrib:6#0f; avgDuration:6#0f; bondCount:6#0i)];

    byBucket:select
        weight:sum weight,
        durationContrib:sum durationContrib,
        avgDuration:wavg[weight;modDuration],
        bondCount:count i
    by bucket from composition;

    0!byBucket}

// =============================================================================
// AUCTION DATE PREDICTION
// =============================================================================

// Estimate auction date for a given month and original term
estimateAuctionDate:{[targetMonth;oTerm]
    // Get schedule for this term
    sched:first select from auctionSchedule where origTerm = oTerm;
    if[0 = count sched; :0Nd];

    // Check if quarterly auction
    if[sched[`frequency] ~ `Quarterly;
        // Only in Feb, May, Aug, Nov
        if[not (`mm$targetMonth) in quarterlyMonths; :0Nd]];

    // Calculate nth day of week in month
    nthDayOfWeek[targetMonth; sched`auctionDayOfWeek; sched`auctionWeek]}

// Estimate settlement date (T+1 for coupons)
estimateSettleDate:{[auctionDate;oTerm]
    if[null auctionDate; :0Nd];
    sched:first select from auctionSchedule where origTerm = oTerm;
    settleDays:$[0 = count sched; 1; sched`settleDays];
    nextBizDay auctionDate + settleDays}

// Estimate coupon for new issuance based on recent auctions
estimateCoupon:{[bonds;asOfDate;oTerm]
    // Find most recent auction of same term
    recent:select from bonds where origTerm = oTerm, auctionDate <= asOfDate;
    if[0 = count recent; :0.04];  // Default 4%
    recent:`auctionDate xdesc recent;
    first recent`coupon}

// Predict new issuance for a target month
// Returns table of expected auctions that will settle before month-end
// netNewOnly: if true, only count quarterly 10Y/20Y/30Y as "new" (Bloomberg methodology)
//             if false, count all auctions settling in the month
predictNewIssuance:{[bonds;targetMonth;asOfDate]
    .tsyindex.predictNewIssuanceEx[bonds;targetMonth;asOfDate;1b]}  // Default: net new only

predictNewIssuanceEx:{[bonds;targetMonth;asOfDate;netNewOnly]
    monthEnd:.tsyindex.lastBizDayOfMonth targetMonth;

    // If netNewOnly, only count quarterly 10Y/20Y/30Y (original issues)
    // Monthly 2Y/3Y/5Y/7Y are considered replacements for maturing bonds
    terms:$[netNewOnly; `10Y`20Y`30Y; exec origTerm from auctionSchedule];

    // For each term in schedule, check if auction settles before month-end
    newIssues:{[bonds;targetMonth;monthEnd;asOfDate;term]
        auctionDate:.tsyindex.estimateAuctionDate[targetMonth;term];
        if[null auctionDate; :()];

        settleDate:.tsyindex.estimateSettleDate[auctionDate;term];
        if[settleDate > monthEnd; :()];
        if[settleDate <= asOfDate; :()];  // Already happened

        sched:first select from auctionSchedule where origTerm = term;
        estCoupon:.tsyindex.estimateCoupon[bonds;asOfDate;term];

        // Estimate duration based on term
        termYears:$[term ~ `2Y; 2f; term ~ `3Y; 3f; term ~ `5Y; 5f;
                    term ~ `7Y; 7f; term ~ `10Y; 10f; term ~ `20Y; 20f; 30f];
        estDuration:.tsyindex.bondDuration[settleDate;settleDate + `long$termYears * 365;estCoupon;estCoupon];

        ([] origTerm:enlist term;
            auctionDate:enlist auctionDate;
            settleDate:enlist settleDate;
            estimatedCoupon:enlist estCoupon;
            estimatedDuration:enlist estDuration;
            typicalSize:enlist sched`typicalSize)
    }[bonds;targetMonth;monthEnd;asOfDate] each terms;

    raze newIssues}

// =============================================================================
// ROLL-OFF PREDICTION
// =============================================================================

// Predict bonds that will exit the index
// (remaining maturity < 1 year at target month-end)
predictRolloff:{[composition;targetMonth;indexName]
    rules:indexRules indexName;
    monthEnd:.tsyindex.lastBizDayOfMonth targetMonth;
    minRem:rules`minRemMaturity;

    // Find bonds that will be below threshold at month-end
    rolloffs:select from composition where
        ((maturityDate - monthEnd) % 365.25) < minRem;

    rolloffs}

// Preview bonds near rolloff threshold
bondsNearRolloff:{[composition;asOfDate;horizonMonths]
    // Find bonds that will exit within horizon
    horizonDate:asOfDate + `long$horizonMonths * 30;

    select cusip, coupon, maturityDate, remYears, weight, modDuration, durationContrib,
           exitMonth:`month$(maturityDate - 365)
    from composition
    where maturityDate < (horizonDate + 365)}

// =============================================================================
// DURATION EXTENSION FORECASTING
// =============================================================================

// Forecast index composition for a future month
forecastMonth:{[bonds;currentComp;targetMonth;indexName;asOfDate]
    monthEnd:.tsyindex.lastBizDayOfMonth targetMonth;

    // 1. Remove rolloffs
    rolloffs:.tsyindex.predictRolloff[currentComp;targetMonth;indexName];
    rolloffCusips:rolloffs`cusip;
    remaining:select from currentComp where not cusip in rolloffCusips;

    // 2. Age remaining bonds
    remaining:update remYears:(maturityDate - monthEnd) % 365.25 from remaining;
    remaining:update bucket:.tsyindex.assignBuckets remYears from remaining;
    remaining:update modDuration:.tsyindex.bondDuration[monthEnd;;]'[maturityDate;coupon;ytm] from remaining;

    // 3. Add new issuance (use ALL auctions for duration calculation, not just net new)
    newIssues:.tsyindex.predictNewIssuanceEx[bonds;targetMonth;asOfDate;0b];

    // Convert new issues to composition format
    if[0 < count newIssues;
        termYrs:{$[x~`2Y;2;x~`3Y;3;x~`5Y;5;x~`7Y;7;x~`10Y;10;x~`20Y;20;30]} each newIssues`origTerm;
        termYrsF:{$[x~`2Y;2f;x~`3Y;3f;x~`5Y;5f;x~`7Y;7f;x~`10Y;10f;x~`20Y;20f;30f]} each newIssues`origTerm;
        newRows:([]
            cusip:`$"NEW_",/:string newIssues`origTerm;
            securityType:count[newIssues]#`Note;
            coupon:newIssues`estimatedCoupon;
            maturityDate:newIssues[`settleDate] + `long$365 * termYrs;
            issueDate:newIssues`settleDate;
            origTerm:newIssues`origTerm;
            remYears:termYrsF;
            bucket:.tsyindex.assignBuckets termYrsF;
            cleanPrice:count[newIssues]#100f;
            ytm:newIssues`estimatedCoupon;
            parAmount:newIssues`typicalSize;
            marketValue:newIssues`typicalSize;
            weight:count[newIssues]#0f;
            modDuration:newIssues`estimatedDuration;
            durationContrib:count[newIssues]#0f);
        remaining:remaining,newRows];

    // 4. Recalculate weights
    totalMV:sum remaining`marketValue;
    remaining:update weight:marketValue % totalMV from remaining;
    remaining:update durationContrib:weight * modDuration from remaining;

    // Return forecast composition plus metadata
    `composition`rolloffs`newIssues`asOfDate`targetMonth!(remaining;rolloffs;newIssues;asOfDate;targetMonth)}

// Decompose duration change into components
decomposeDurationChange:{[prevComp;forecast]
    currentDur:sum prevComp`durationContrib;
    projectedDur:sum forecast[`composition]`durationContrib;

    // Duration lost from rolloffs
    rolloffDur:sum forecast[`rolloffs]`durationContrib;

    // Duration gained from new issuance
    newIssues:forecast`newIssues;
    newDur:$[0 < count newIssues;
        sum forecast[`composition;`durationContrib] where forecast[`composition;`cusip] like "NEW_*";
        0f];

    // Duration change from aging (existing bonds)
    agingDur:projectedDur - currentDur + rolloffDur - newDur;

    `currentDuration`projectedDuration`extensionTotal`extensionFromNew`extensionFromRolloff`extensionFromAging!(
        currentDur;projectedDur;projectedDur - currentDur;newDur;neg rolloffDur;agingDur)}

// Full duration extension forecast for multiple months
forecastDurationExtension:{[bonds;asOfDate;horizonMonths;indexName;prices;outstandings]
    // Build current composition
    currentComp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;prices;outstandings];
    currentStats:.tsyindex.indexStats currentComp;

    // Forecast each month
    months:(`month$asOfDate) + 1 + til horizonMonths;

    // Iterate over months, accumulating results
    step:{[bonds;indexName;asOfDate;acc;targetMonth]
        prevComp:acc`composition;
        forecast:.tsyindex.forecastMonth[bonds;prevComp;targetMonth;indexName;asOfDate];
        decomp:.tsyindex.decomposeDurationChange[prevComp;forecast];

        // Build row for summary table
        row:([] forecastMonth:enlist targetMonth;
             index:enlist indexName;
             currentDuration:enlist decomp`currentDuration;
             projectedDuration:enlist decomp`projectedDuration;
             extensionTotal:enlist decomp`extensionTotal;
             extensionFromNew:enlist decomp`extensionFromNew;
             extensionFromRolloff:enlist decomp`extensionFromRolloff;
             extensionFromAging:enlist decomp`extensionFromAging);

        // Return updated accumulator
        `rows`composition!(acc[`rows],row;forecast`composition)
    }[bonds;indexName;asOfDate];

    init:`rows`composition!(();currentComp);
    results:step/[init;months];

    // Get summary (rows is already a table from accumulation)
    summaryRows:results`rows;

    `summary`currentComposition`currentStats!(summaryRows;currentComp;currentStats)}

// =============================================================================
// HIGH-LEVEL API
// =============================================================================

// Main entry point: Full rebalancing forecast
indexRebalanceForecast:{[bonds;asOfDate;horizonMonths;indexName]
    forecastDurationExtension[bonds;asOfDate;horizonMonths;indexName;(::);(::)]}

// Quick view of next month's changes
nextMonthChanges:{[bonds;asOfDate;indexName]
    currentComp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;(::);(::)];
    nextMonth:1 + `month$asOfDate;

    rolloffs:.tsyindex.predictRolloff[currentComp;nextMonth;indexName];
    newIssues:.tsyindex.predictNewIssuance[bonds;nextMonth;asOfDate];

    `rolloffs`newIssues`nextMonth!(rolloffs;newIssues;nextMonth)}

// Pretty-print forecast
showRebalanceForecast:{[forecast]
    -1 "";
    -1 "=== TREASURY INDEX DURATION FORECAST ===";
    -1 "";

    stats:forecast`currentStats;
    -1 "Current Index Statistics:";
    -1 "  Total Market Value: $",string[`long$(stats`totalMV) % 1e9],"B";
    -1 "  Index Duration:     ",string[stats`indexDuration]," years";
    -1 "  Bond Count:         ",string stats`bondCount;
    -1 "  Average Coupon:     ",string[100 * stats`avgCoupon],"%";
    -1 "";

    -1 "Duration Extension Forecast:";
    -1 "-------------------------------------------";
    summary:forecast`summary;
    {-1 "  ",string[x`forecastMonth],": ",string[x`currentDuration]," -> ",string[x`projectedDuration],
       " (extension: ",string[x`extensionTotal],")"} each summary;
    -1 "";

    -1 "Decomposition (last month):";
    last_:last summary;
    -1 "  From new issuance: +",string last_`extensionFromNew;
    -1 "  From rolloffs:     ",string last_`extensionFromRolloff;
    -1 "  From aging:        ",string last_`extensionFromAging;
    -1 "";
    }

// =============================================================================
// API: INDEX EXTENSION FROM MARKET PRICES
// =============================================================================

// Calculate index extension forecast from market price data
// Input:
//   bonds: bond reference data from .tsy.loadBondCache[]
//   priceData: table with columns (date; cusip; price) - daily prices
//   indexName: `BBERG or `ICE
// Output:
//   Table with one row per date containing:
//     - date, forecastMonth
//     - bondCount, totalMV, avgCoupon
//     - currentDuration, projectedDuration, extensionTotal
//     - extensionFromNew, extensionFromRolloff, extensionFromAging
//     - bondsEntering, bondsLeaving
//     - durationByBucket (nested dict)
//
// Example:
//   prices:([] date:2026.01.31 2026.01.31; cusip:`ABC`DEF; price:99.5 101.2)
//   .tsyindex.extensionFromPrices[bonds; prices; `BBERG]

extensionFromPrices:{[bonds;priceData;indexName]
    // Get unique dates
    dates:asc distinct priceData`date;

    // Process each date (buildIndexComposition handles outstanding amounts internally)
    results:{[bonds;priceData;indexName;asOfDate]
        // Get prices for this date as dict cusip->price
        dayPrices:exec cusip!price from priceData where date = asOfDate;

        // Build composition with market prices (outstandings from bonds table)
        comp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;dayPrices;(::)];
        if[0 = count comp; :()];

        stats:.tsyindex.indexStats comp;

        // Forecast next month
        nextMonth:1 + `month$asOfDate;
        forecast:.tsyindex.forecastMonth[bonds;comp;nextMonth;indexName;asOfDate];
        decomp:.tsyindex.decomposeDurationChange[comp;forecast];

        // Get entering/leaving counts
        newIssues:.tsyindex.predictNewIssuance[bonds;nextMonth;asOfDate];  // Net new only
        rolloffs:forecast`rolloffs;

        // Duration by bucket
        buckets:.tsyindex.durationByBucket comp;
        bucketDict:buckets[`bucket]!buckets[`durationContrib];

        // Build result row
        ([]
            date:enlist asOfDate;
            forecastMonth:enlist nextMonth;
            bondCount:enlist stats`bondCount;
            totalMV:enlist stats`totalMV;
            avgCoupon:enlist stats`avgCoupon;
            avgPrice:enlist wavg[comp`weight; comp`cleanPrice];
            currentDuration:enlist decomp`currentDuration;
            projectedDuration:enlist decomp`projectedDuration;
            extensionTotal:enlist decomp`extensionTotal;
            extensionFromNew:enlist decomp`extensionFromNew;
            extensionFromRolloff:enlist decomp`extensionFromRolloff;
            extensionFromAging:enlist decomp`extensionFromAging;
            bondsEntering:enlist count newIssues;
            bondsLeaving:enlist count rolloffs;
            weight1_3Y:enlist sum exec weight from comp where bucket = `1_3Y;
            weight3_5Y:enlist sum exec weight from comp where bucket = `3_5Y;
            weight5_7Y:enlist sum exec weight from comp where bucket = `5_7Y;
            weight7_10Y:enlist sum exec weight from comp where bucket = `7_10Y;
            weight10_20Y:enlist sum exec weight from comp where bucket = `10_20Y;
            weight20_30Y:enlist sum exec weight from comp where bucket = `20_30Y;
            dur1_3Y:enlist sum exec durationContrib from comp where bucket = `1_3Y;
            dur3_5Y:enlist sum exec durationContrib from comp where bucket = `3_5Y;
            dur5_7Y:enlist sum exec durationContrib from comp where bucket = `5_7Y;
            dur7_10Y:enlist sum exec durationContrib from comp where bucket = `7_10Y;
            dur10_20Y:enlist sum exec durationContrib from comp where bucket = `10_20Y;
            dur20_30Y:enlist sum exec durationContrib from comp where bucket = `20_30Y)
    }[bonds;priceData;indexName] each dates;

    raze results}

// Simplified version: single date, returns full detail
// Input:
//   bonds: bond reference data
//   prices: dict cusip->price OR table (cusip; price)
//   asOfDate: valuation date
//   indexName: `BBERG or `ICE
// Output:
//   Dict with keys: summary, composition, rolloffs, newIssues, buckets

extensionDetail:{[bonds;prices;asOfDate;indexName]
    // Convert prices to dict if table
    priceDict:$[99h = type prices; prices; prices[`cusip]!prices`price];

    // Build composition (outstandings from bonds table)
    comp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;priceDict;(::)];
    stats:.tsyindex.indexStats comp;

    // Forecast
    nextMonth:1 + `month$asOfDate;
    forecast:.tsyindex.forecastMonth[bonds;comp;nextMonth;indexName;asOfDate];
    decomp:.tsyindex.decomposeDurationChange[comp;forecast];

    // New issues and rolloffs
    newIssues:.tsyindex.predictNewIssuance[bonds;nextMonth;asOfDate];

    // Buckets
    buckets:.tsyindex.durationByBucket comp;

    // Summary
    summary:`date`forecastMonth`bondCount`totalMV`avgCoupon`avgPrice`currentDuration`projectedDuration`extensionTotal`extensionFromNew`extensionFromRolloff`extensionFromAging`bondsEntering`bondsLeaving!(
        asOfDate;
        nextMonth;
        stats`bondCount;
        stats`totalMV;
        stats`avgCoupon;
        wavg[comp`weight; comp`cleanPrice];
        decomp`currentDuration;
        decomp`projectedDuration;
        decomp`extensionTotal;
        decomp`extensionFromNew;
        decomp`extensionFromRolloff;
        decomp`extensionFromAging;
        count newIssues;
        count forecast`rolloffs);

    `summary`composition`rolloffs`newIssues`buckets!(summary;comp;forecast`rolloffs;newIssues;buckets)}

// =============================================================================
// EXAMPLE
// =============================================================================

example:{[]
    -1 "";
    -1 "=== TREASURY INDEX REBALANCING EXAMPLE ===";
    -1 "";
    -1 "First, load bond data:";
    -1 "  \\l tsy.q";
    -1 "  bonds:.tsy.loadBondCache[]";
    -1 "";
    -1 "Then run duration forecast:";
    -1 "  forecast:.tsyindex.indexRebalanceForecast[bonds;.z.d;3;`BBERG]";
    -1 "  .tsyindex.showRebalanceForecast forecast";
    -1 "";
    -1 "Quick view of next month:";
    -1 "  changes:.tsyindex.nextMonthChanges[bonds;.z.d;`BBERG]";
    -1 "  changes`rolloffs";
    -1 "  changes`newIssues";
    -1 "";
    -1 "Duration by bucket:";
    -1 "  comp:forecast`currentComposition";
    -1 "  .tsyindex.durationByBucket comp";
    -1 "";
    }

help:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                         .tsyindex FUNCTIONS";
    -1 "=============================================================================";
    -1 "";
    -1 "HIGH-LEVEL API:";
    -1 "  indexRebalanceForecast[bonds;asOfDate;horizonMonths;indexName]";
    -1 "    Main entry point - forecasts duration extension";
    -1 "    indexName: `BBERG or `ICE";
    -1 "";
    -1 "  nextMonthChanges[bonds;asOfDate;indexName]";
    -1 "    Quick view of next month's additions and removals";
    -1 "";
    -1 "  showRebalanceForecast[forecast]";
    -1 "    Pretty-print forecast results";
    -1 "";
    -1 "COMPOSITION:";
    -1 "  buildIndexComposition[bonds;asOfDate;indexName;prices;outstandings]";
    -1 "    Build full index composition (prices/outstandings optional)";
    -1 "";
    -1 "  indexStats[composition]";
    -1 "    Aggregate statistics (duration, MV, count)";
    -1 "";
    -1 "  durationByBucket[composition]";
    -1 "    Duration breakdown by maturity bucket";
    -1 "";
    -1 "ELIGIBILITY:";
    -1 "  filterEligibleForIndex[bonds;asOfDate;indexName]";
    -1 "    Filter bonds meeting index criteria";
    -1 "";
    -1 "PREDICTION:";
    -1 "  predictNewIssuance[bonds;targetMonth;asOfDate]";
    -1 "    Predict auctions settling before month-end";
    -1 "";
    -1 "  predictRolloff[composition;targetMonth;indexName]";
    -1 "    Predict bonds exiting index";
    -1 "";
    -1 "DURATION:";
    -1 "  .tsyindex.bondDuration[settleDate;maturityDate;coupon;ytm]";
    -1 "    Modified duration for a single bond";
    -1 "";
    -1 "  bondDurationBatch[settleDate;bonds]";
    -1 "    Vectorized duration calculation";
    -1 "";
    -1 "CONFIGURATION:";
    -1 "  indexRules           - Index eligibility rules";
    -1 "  auctionSchedule      - Typical auction schedule";
    -1 "  maturityBuckets      - Bucket definitions";
    -1 "";
    }

// Startup message
-1 "Loaded .tsyindex namespace v",version;
-1 "Functions: indexRebalanceForecast, nextMonthChanges, buildIndexComposition";
-1 "Run .tsyindex.help[] for usage, .tsyindex.example[] for demo";

\d .
