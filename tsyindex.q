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
//
// IMPORTANT: Sizing methodology for extension calculation:
//   - Use offeringAmount (public market allocation), NOT outstanding or publicFloat
//   - SOMA add-on at auction = totalAccepted - offeringAmount (typically 7-15%)
//   - The offeringAmount is what goes to public market; SOMA add-on goes to Fed
//   - publicFloat in data reflects accumulated SOMA holdings over time (not just auction add-on)
//
// AUCTION SCHEDULE BY MONTH TYPE:
//   Non-refunding months (Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec):
//     - 2Y, 3Y, 5Y, 7Y, 10Y (reopen) auctions
//   Refunding months (Feb, May, Aug, Nov):
//     - 3Y, 10Y (original), 30Y auctions only
//     - NO 2Y, 5Y, 7Y, 20Y auctions in refunding months
//
// SETTLEMENT TIMING:
//   - Late-month auctions (e.g., 7Y in week 4) often settle in the NEXT month
//   - Example: Jan 27 auction settles Feb 02
//   - The actual issueDate from API data is used to determine which month bonds enter
//
// NOTE: This schedule is for reference only. The extension calculation uses actual
// issueDate from the API data via bondsEnteringMonth(), not estimated dates.

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

// First business day of month
firstBizDayOfMonth:{[ym] nextBizDay firstOfMonth ym}

// Add months to a date, preserving day-of-month (capped at month end)
// e.g., addMonths[2027.01.31;-6] -> 2026.07.31
addMonths:{[dt;n]
    d:`dd$dt;
    // Get new month
    newMon:(`month$dt) + n;
    // First day of new month
    firstDay:`date$newMon;
    // Last day of new month
    lastDay:-1 + `date$newMon + 1;
    maxD:`dd$lastDay;
    // Return date with same day-of-month (or last day if original day > month length)
    firstDay + (d - 1) & maxD - 1}

// N-th business day before a date
nBizDaysBefore:{[d;n]
    i:0;
    while[i < n;
        d-:1;
        if[not isWeekend d; i+:1]];
    d}

// Index determination date: 1 business day before month end
// Bonds AUCTIONED on or before this date enter the NEXT month's index
// (even if they settle after the determination date)
determinationDate:{[ym]
    lbd:lastBizDayOfMonth ym;
    nBizDaysBefore[lbd;1]}

// Check if an auction date qualifies for the next month's index
// auctionDate must be ON OR BEFORE the determination date
auctionedForNextMonth:{[ym;auctionDate]
    auctionDate <= determinationDate ym}

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
// OUTSTANDING CALCULATION FROM AUCTION DATA
// =============================================================================
// Compute outstanding amounts as of a given date from auction data
// This gives the correct historical outstanding, not just the latest snapshot

// Compute outstanding for all CUSIPs as of asOfDate
// auctions: auction table with columns (cusip; issueDate; offeringAmt)
// asOfDate: date to compute outstanding as of
// Returns: dict cusip -> outstanding
outstandingAsOf:{[auctions;asOfDate]
    // Sum totalAccepted for auctions that have settled by asOfDate
    // totalAccepted = total issued including SOMA add-on at auction
    // This is the correct amount to use when computing publicFloat = outstanding - SOMA
    // Fall back to offeringAmt if totalAccepted column doesn't exist (old cache format)
    amtCol:$[`totalAccepted in cols auctions; `totalAccepted; `offeringAmt];
    settled:?[auctions; enlist (<=;`issueDate;asOfDate); (enlist `cusip)!(enlist `cusip); (enlist `totalOut)!(enlist (sum;amtCol))];
    exec cusip!totalOut from settled}

// Compute outstanding for all CUSIPs as of each date in a list
// Returns: dict date -> (dict cusip -> outstanding)
outstandingByDate:{[auctions;dates]
    dates!outstandingAsOf[auctions;] each dates}

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

    // Generate actual coupon dates by going backwards from maturity in 6-month steps
    // Treasury coupons are semiannual on same day-of-month as maturity
    couponDates:enlist maturityDate;
    dt:maturityDate;
    while[(dt:addMonths[dt;-6]) > settleDate; couponDates,:dt];

    // Reverse to chronological order (earliest first)
    couponDates:reverse couponDates;
    nCoupons:count couponDates;
    if[nCoupons = 0; :0f];

    // Time to each coupon in years (ACT/365.25)
    times:(couponDates - settleDate) % 365.25;

    // Cash flows: coupon/2 each period, plus 100 at maturity
    semiCoupon:coupon * 50f;
    cfs:((nCoupons - 1)#semiCoupon),semiCoupon + 100f;

    // Discount factors using semiannual compounding
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

    // Generate actual coupon dates from maturity
    couponDates:enlist maturityDate;
    dt:maturityDate;
    while[(dt:addMonths[dt;-6]) > settleDate; couponDates,:dt];
    couponDates:reverse couponDates;
    nCoupons:count couponDates;
    if[nCoupons = 0; :0f];

    times:(couponDates - settleDate) % 365.25;
    semiCoupon:coupon * 50f;
    cfs:((nCoupons - 1)#semiCoupon),semiCoupon + 100f;
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

    // Calculate actual accrued interest
    // Find last coupon date by going back 6 months from maturity until <= settleDate
    lastCoupon:maturityDate;
    while[lastCoupon > settleDate; lastCoupon:addMonths[lastCoupon;-6]];
    nextCoupon:addMonths[lastCoupon;6];
    daysSinceCoupon:settleDate - lastCoupon;
    couponPeriod:nextCoupon - lastCoupon;
    ai:coupon * 50f * daysSinceCoupon % couponPeriod;
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

    // Index inclusion rule: bonds in month M's index must have issued before
    // the determination date of month M-1 (2 biz days before M-1 month end)
    // Example: November index includes bonds issued before October det date (~Oct 29)
    priorMonth:(`month$asOfDate) - 1;
    priorDetDate:determinationDate priorMonth;

    // Start with bonds that issued before prior month's determination date
    eligible:select from bonds where issueDate < priorDetDate, maturityDate > asOfDate;

    // Filter by security type
    eligible:select from eligible where securityType in rules`secTypes;

    // Filter by minimum outstanding amount
    // Use outstanding (not publicFloat) because publicFloat has missing data for some bonds
    // publicFloat = outstanding - SOMA holdings, but API sometimes returns 0
    // Note: This gives ~293 bonds vs Bloomberg's ~305. The gap is likely due to:
    //   - Reopened issues counted separately by Bloomberg
    //   - Different data sources / timing
    // If outstanding column is missing, skip this filter
    if[`outstanding in cols eligible;
        eligible:select from eligible where outstanding >= rules`minOutstanding];

    // Calculate remaining maturity
    eligible:update remYears:(maturityDate - asOfDate) % 365f from eligible;

    // Filter by minimum remaining maturity (based on MONTH, not exact days)
    // Include bonds maturing in the current month + 12 months or later
    // This ensures bonds like 12/15/26 are included on 12/31/25 (same month + 12)
    minMaturityMonth:(`month$asOfDate) + `long$12 * rules`minRemMaturity;
    eligible:select from eligible where (`month$maturityDate) >= minMaturityMonth;

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

    // Use outstanding (total market value) for index weighting
    // Bloomberg uses total outstanding, not SOMA-adjusted publicFloat
    // publicFloat makes new issues have disproportionately large weight
    eligible:$[`outstanding in cols eligible;
        update parAmount:outstanding from eligible;
        update parAmount:1e9 from eligible];
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

// Get offer amount for a term (for extension calculation)
// Uses offeringAmount (public market allocation), NOT outstanding
// isReopen: true for 10Y in non-refunding months, always true for 20Y/30Y
getOfferAmount:{[bonds;asOfDate;oTerm;isReopen]
    // Get schedule for fallback sizes
    sched:first select from auctionSchedule where origTerm = oTerm;
    fallbackSize:$[isReopen; sched`typicalReopenSize; sched`typicalOfferSize];

    // Try to get from recent auction data
    recent:select from bonds where origTerm = oTerm, auctionDate <= asOfDate;
    if[0 = count recent; :fallbackSize];
    recent:`auctionDate xdesc recent;

    // Use offeringAmount if available, else fall back to typical size
    // Note: For 10Y/20Y/30Y, offeringAmount in the API is CUMULATIVE across reopens
    // so we use typical sizes for those terms
    $[oTerm in `10Y`20Y`30Y;
        fallbackSize;
        // For 2Y/3Y/5Y/7Y, use offeringAmount from most recent auction
        $[`offeringAmount in cols recent;
            first recent`offeringAmount;
            fallbackSize]]}

// Legacy wrapper (for backward compatibility)
getRecentAuctionSize:{[bonds;asOfDate;oTerm]
    // Default: assume original issue (not reopen)
    .tsyindex.getOfferAmount[bonds;asOfDate;oTerm;0b]}

// Check if month is a quarterly refunding month (Feb, May, Aug, Nov)
isRefundingMonth:{[targetMonth]
    (`mm$targetMonth) in quarterlyMonths}

// =============================================================================
// BONDS ENTERING INDEX - DETERMINATION DATE METHODOLOGY
// =============================================================================
// Index determination date = 2 business days before month end
// Bonds settling BEFORE determination date enter the NEXT month's index
// Bonds settling ON or AFTER determination date enter the month AFTER next
//
// Example: For Feb index (Jan→Feb extension):
//   - Look at auctions in January
//   - Filter to those settling BEFORE Jan determination date (~Jan 29)
//   - These bonds enter February's index
//   - Auctions settling Jan 29 or later enter March's index
// =============================================================================

// Get bonds entering the index for a future month
// indexMonth: the month whose index we're building (e.g., Jan for Dec→Jan extension)
// auctions: (optional) pre-loaded auction cache - pass (::) to load from file
// Returns bonds AUCTIONED in the window between:
//   - Prior month's prior determination date (exclusive, already in prior index)
//   - Prior month's determination date (inclusive, cutoff for this index)
// Example for January index: bonds auctioned between Nov 26 (exclusive) and Dec 27 (inclusive)
// NOTE: Uses auctionDate - bonds can enter even if they settle in the next month
bondsEnteringForIndexWithAuctions:{[bonds;indexMonth;indexName;auctions]
    // The determination date cutoff for this index (auctions ON this date included)
    detDate:determinationDate indexMonth - 1;
    // The prior determination date (auctions ON OR BEFORE this are in prior index)
    priorDetDate:determinationDate indexMonth - 2;
    bondsEnteringByDateRangeWithAuctions[bonds;indexName;priorDetDate;detDate;(::);auctions]}

// Backward-compatible wrapper
bondsEnteringForIndex:{[bonds;indexMonth;indexName]
    bondsEnteringForIndexWithAuctions[bonds;indexMonth;indexName;(::)]}

// Get bonds entering for a future month - PREDICTED (with announcement filter)
// predictionDate: only include auctions announced by this date
bondsEnteringForIndexPredicted:{[bonds;indexMonth;indexName;predictionDate]
    detDate:determinationDate indexMonth - 1;
    priorDetDate:determinationDate indexMonth - 2;
    bondsEnteringByDateRange[bonds;indexName;priorDetDate;detDate;predictionDate]}

// Core function: get bonds entering by date range
// startDate: exclusive lower bound (auctions AFTER this date)
// endDate: inclusive upper bound (auctions ON OR BEFORE this date)
// announceCutoff: only include auctions announced by this date (for predictions)
// auctions: (optional) pre-loaded auction cache - pass (::) to load from file
// Returns ALL auctions (both original and reopenings) auctioned in the date range
// NOTE: Uses auctionDate, not issueDate - bonds auctioned on/before determination date
//       can enter the next month's index even if they settle in the next month
bondsEnteringByDateRangeWithAuctions:{[bonds;indexName;startDate;endDate;announceCutoff;auctions]
    rules:indexRules indexName;

    // Use provided auction data or load from cache
    auctionData:$[auctions ~ (::);
        @[.tsy.loadAuctionCache; ::; {[e] ([] cusip:`symbol$())}];
        auctions];

    // Filter to auctions AUCTIONED in the date range (not settled)
    // Bonds auctioned on or before determination date enter next month's index
    if[0 < count auctionData;
        auctionData:select from auctionData where auctionDate > startDate, auctionDate <= endDate];

    // Filter by security type and exclude TIPS/FRN
    if[0 < count auctionData;
        auctionData:select from auctionData where securityType in rules`secTypes];
    if[(0 < count auctionData) and rules`excludeTIPS;
        auctionData:select from auctionData where not origTerm like "*TIP*"];
    if[(0 < count auctionData) and rules`excludeFRN;
        auctionData:select from auctionData where not origTerm like "*FRN*"];
    // Exclude FRNs by null coupon (FRNs have no fixed coupon)
    if[0 < count auctionData;
        auctionData:select from auctionData where not null coupon];
    // Cross-reference with bonds cache to exclude TIPS (bonds cache excludes TIPS/FRN)
    // Only keep CUSIPs that are in bonds cache
    if[0 < count auctionData;
        validCusips:exec distinct cusip from bonds;
        auctionData:select from auctionData where cusip in validCusips];

    // Apply announcement cutoff filter (for predictions)
    if[(not announceCutoff ~ (::)) and 0 < count auctionData;
        if[`announcementDate in cols auctionData;
            auctionData:select from auctionData where announcementDate <= announceCutoff]];

    // Also get from bonds cache for bonds not in auction data (use auctionDate)
    enteringCache:select from bonds where auctionDate > startDate, auctionDate <= endDate;
    enteringCache:select from enteringCache where securityType in rules`secTypes;
    if[rules`excludeTIPS; enteringCache:select from enteringCache where not origTerm like "*TIP*"];
    if[rules`excludeFRN; enteringCache:select from enteringCache where not origTerm like "*FRN*"];

    // Start with auction data as primary source
    // Use totalAccepted if available (total issued including SOMA add-on)
    // Fall back to offeringAmt for backward compatibility with old cache format
    entering:$[0 < count auctionData;
        $[`totalAccepted in cols auctionData;
            select cusip, securityType, coupon, issueDate, maturityDate, origTerm, auctionDate,
                   outstanding:totalAccepted, isReopening from auctionData;
            select cusip, securityType, coupon, issueDate, maturityDate, origTerm, auctionDate,
                   outstanding:offeringAmt, isReopening from auctionData];
        ([] cusip:`symbol$(); securityType:`symbol$(); coupon:`float$(); issueDate:`date$();
            maturityDate:`date$(); origTerm:`symbol$(); auctionDate:`date$();
            outstanding:`float$(); isReopening:`boolean$())];

    // Add bonds from cache that aren't in auction data
    if[0 < count enteringCache;
        missingCusips:enteringCache[`cusip] except entering`cusip;
        if[0 < count missingCusips;
            missing:select cusip, securityType, coupon, issueDate, maturityDate, origTerm, auctionDate
                           from enteringCache where cusip in missingCusips;
            missing:update outstanding:0n, isReopening:0b from missing;
            entering:entering,missing]];

    // For amounts: use auction offering amount, else estimate
    entering:update estimatedAmt:.tsyindex.getTypicalOfferSize each origTerm from entering;
    entering:update outstanding:?[null outstanding; estimatedAmt; outstanding] from entering;

    // Filter by minimum outstanding
    entering:select from entering where outstanding >= rules`minOutstanding;

    // Fill null coupons with estimated rate
    entering:update coupon:?[null coupon; 0.045; coupon] from entering;

    // Calculate duration at index month end
    indexMonth:1 + `month$endDate;  // endDate is determination date of prior month
    monthEnd:lastBizDayOfMonth indexMonth;
    entering:update modDuration:.tsyindex.bondDuration[monthEnd;;]'[maturityDate;coupon;coupon] from entering;

    // Add bucket
    entering:update remYears:(maturityDate - monthEnd) % 365f from entering;
    entering:update bucket:.tsyindex.assignBuckets remYears from entering;

    entering}

// Backward-compatible wrapper (loads auction cache if not provided)
bondsEnteringByDateRange:{[bonds;indexName;startDate;endDate;announceCutoff]
    bondsEnteringByDateRangeWithAuctions[bonds;indexName;startDate;endDate;announceCutoff;(::)]}

// Legacy function - kept for backward compatibility
// auctionMonth: the month to look for auctions (bonds issued during this month)
// settleCutoff: only include bonds settling by this date (determination date)
// announceCutoff: only include auctions announced by this date (for predictions)
bondsEnteringFiltered:{[bonds;auctionMonth;indexName;settleCutoff;announceCutoff]
    rules:indexRules indexName;

    // Get per-auction offering amounts from Fiscal Data API
    // This is the primary source for bonds issuing in the auction month
    auctionData:@[.tsy.getAuctionsForExtension; auctionMonth; {[e] ([] cusip:`symbol$(); offeringAmt:`float$(); issueDate:`date$(); announcementDate:`date$())}];

    // Apply announcement cutoff filter FIRST (for predictions - only use announced auctions)
    if[(not announceCutoff ~ (::)) and 0 < count auctionData;
        if[`announcementDate in cols auctionData;
            auctionData:select from auctionData where announcementDate <= announceCutoff]];

    // Apply settlement cutoff filter (determination date) - strictly less than
    // Bonds settling ON the determination date go to the NEXT month
    if[(not settleCutoff ~ (::)) and 0 < count auctionData;
        auctionData:select from auctionData where issueDate < settleCutoff];

    // Get bonds from cache issued in auctionMonth
    enteringCache:select from bonds where (`month$issueDate) = auctionMonth;
    // Apply same settlement filter to cache
    if[not settleCutoff ~ (::);
        enteringCache:select from enteringCache where issueDate < settleCutoff];
    enteringCache:select from enteringCache where securityType in rules`secTypes;
    if[rules`excludeTIPS; enteringCache:select from enteringCache where not origTerm like "*TIP*"];
    if[rules`excludeFRN; enteringCache:select from enteringCache where not origTerm like "*FRN*"];

    // Supplement with auction data for bonds not in cache
    // This ensures we capture all bonds from the Fiscal Data API
    if[0 < count auctionData;
        auctionData:select from auctionData where securityType in rules`secTypes;
        missingCusips:auctionData[`cusip] except enteringCache`cusip;
        if[0 < count missingCusips;
            // Add missing bonds from auction data
            missing:select cusip, securityType, coupon, issueDate, maturityDate, origTerm, auctionDate
                           from auctionData where cusip in missingCusips;
            // Select only essential columns from cache to match
            enteringCache:select cusip, securityType, coupon, issueDate, maturityDate, origTerm, auctionDate
                                 from enteringCache;
            enteringCache:enteringCache,missing]];

    entering:enteringCache;

    // Initialize columns
    entering:update auctionOfferAmt:0n, isReopening:0b from entering;

    // Join auction data with entering bonds for per-auction amounts and reopening status
    // For CUSIPs with multiple auctions (within the filter window):
    //   - Sum all offering amounts
    //   - Use min isReopening (if any auction is original, bond is new to market)
    if[0 < count auctionData;
        auctionAmts:exec sum offeringAmt by cusip from auctionData;
        entering:update auctionOfferAmt:auctionAmts cusip from entering;
        // Use isReopening flag directly from Fiscal Data API if available
        // min() because if ANY auction is original (0), the CUSIP is new to market
        if[`isReopening in cols auctionData;
            reopenFlags:exec min isReopening by cusip from auctionData;
            entering:update isReopening:reopenFlags cusip from entering]];

    // For amounts: use auction offering amount, else estimate
    entering:update estimatedAmt:.tsyindex.getTypicalOfferSize each origTerm from entering;

    // Set final amount: prefer auction data, else estimate
    entering:update finalAmt:?[not null auctionOfferAmt; auctionOfferAmt; estimatedAmt] from entering;

    // Filter by minimum outstanding
    entering:select from entering where finalAmt >= rules`minOutstanding;

    // Fill null coupons with estimated rate (~4.5% as of 2025-2026)
    entering:update coupon:?[null coupon; 0.045; coupon] from entering;

    // Calculate duration at next month end (when these bonds will be in the index)
    indexMonth:auctionMonth + 1;
    monthEnd:lastBizDayOfMonth indexMonth;
    entering:update modDuration:.tsyindex.bondDuration[monthEnd;;]'[maturityDate;coupon;coupon] from entering;

    // Add bucket
    entering:update remYears:(maturityDate - monthEnd) % 365f from entering;
    entering:update bucket:.tsyindex.assignBuckets remYears from entering;

    // Use finalAmt as outstanding for calculations
    entering:update outstanding:finalAmt from entering;

    entering}

// Legacy wrapper - kept for backward compatibility
bondsEnteringMonth:{[bonds;targetMonth;indexName]
    bondsEnteringForIndex[bonds;targetMonth;indexName]}

// Get bonds that settle AFTER determination date (these enter the month AFTER next)
// These are late-settling bonds that miss the current month's index cutoff
bondsEnteringLateMonth:{[bonds;auctionMonth;indexName]
    // Get all entering, then filter to late month
    // Note: amounts here include full monthly aggregation since we don't have a "after cutoff" filter
    entering:.tsyindex.bondsEnteringMonth[bonds;targetMonth;indexName];
    fbd:firstBizDayOfMonth targetMonth;
    cutoff:fbd + settlementWindow;
    select from entering where issueDate > cutoff}

// Typical auction sizes by term (as of 2025, approximate)
typicalOfferSizes:`2Y`3Y`5Y`7Y`10Y`20Y`30Y!(69e9; 58e9; 70e9; 44e9; 42e9; 16e9; 22e9)

// Helper to get typical offer size for a term
getTypicalOfferSize:{[term]
    sz:typicalOfferSizes term;
    $[null sz; 50e9; sz]}  // default 50B if not found

// Predict new issuance for a target month
// Uses actual issueDate from API data
// Note: asOfDate parameter kept for backward compatibility but not used
predictNewIssuance:{[bonds;targetMonth;asOfDate]
    .tsyindex.bondsEnteringMonth[bonds;targetMonth;`BBERG]}

// For extension calculation: uses actual bond data from API
// Returns bonds entering in targetMonth with their actual outstanding amounts
predictNewIssuanceForExtension:{[bonds;targetMonth;indexName]
    .tsyindex.bondsEnteringMonth[bonds;targetMonth;indexName]}

// Legacy function for backward compatibility
predictNewIssuanceEx:{[bonds;targetMonth;asOfDate;netNewOnly]
    terms:$[netNewOnly; `10Y`20Y`30Y; exec origTerm from auctionSchedule];
    .tsyindex.predictNewIssuanceTerms[bonds;targetMonth;asOfDate;terms]}

// Core function that takes explicit list of terms
// Determines original vs reopen based on term and month:
//   - 10Y in non-refunding months = REOPEN (use typicalReopenSize)
//   - 10Y in refunding months (Feb/May/Aug/Nov) = ORIGINAL (use typicalOfferSize)
//   - 20Y/30Y = always REOPEN (use typicalReopenSize)
//   - 2Y/3Y/5Y/7Y = always ORIGINAL (use offeringAmount or typicalOfferSize)
predictNewIssuanceTerms:{[bonds;targetMonth;asOfDate;terms]
    monthEnd:.tsyindex.lastBizDayOfMonth targetMonth;
    isRefunding:.tsyindex.isRefundingMonth targetMonth;

    // For each term in schedule, check if auction settles before month-end
    newIssues:{[bonds;targetMonth;monthEnd;asOfDate;isRefunding;term]
        auctionDate:.tsyindex.estimateAuctionDate[targetMonth;term];
        if[null auctionDate; :()];

        settleDate:.tsyindex.estimateSettleDate[auctionDate;term];
        if[settleDate > monthEnd; :()];
        if[settleDate <= asOfDate; :()];  // Already happened

        estCoupon:.tsyindex.estimateCoupon[bonds;asOfDate;term];

        // Determine if this is a reopen
        // 10Y: reopen in non-refunding months, original in refunding months
        // 20Y/30Y: always reopens
        // 2Y/3Y/5Y/7Y: always originals
        isReopen:$[term ~ `10Y; not isRefunding;
                   term in `20Y`30Y; 1b;
                   0b];

        // Get offer size with reopen flag
        estSize:.tsyindex.getOfferAmount[bonds;asOfDate;term;isReopen];

        // Estimate duration based on term
        termYears:$[term ~ `2Y; 2f; term ~ `3Y; 3f; term ~ `5Y; 5f;
                    term ~ `7Y; 7f; term ~ `10Y; 10f; term ~ `20Y; 20f; 30f];
        estDuration:.tsyindex.bondDuration[settleDate;settleDate + `long$termYears * 365;estCoupon;estCoupon];

        ([] origTerm:enlist term;
            auctionDate:enlist auctionDate;
            settleDate:enlist settleDate;
            estimatedCoupon:enlist estCoupon;
            estimatedDuration:enlist estDuration;
            estimatedSize:enlist estSize;
            isReopen:enlist isReopen)
    }[bonds;targetMonth;monthEnd;asOfDate;isRefunding] each terms;

    raze newIssues}

// =============================================================================
// ROLL-OFF PREDICTION
// =============================================================================

// Predict bonds that will exit the index
// Bonds exit when their remaining maturity falls below 1 year
// At month-end rebalance, bonds maturing in the same month 12 months ahead exit
// (e.g., Dec 31 rebalance removes all Dec 2026 maturities)
predictRolloff:{[composition;targetMonth;indexName]
    // targetMonth is the month we're projecting into (e.g., Jan 2026 for Dec->Jan)
    // Bonds maturing in the current month + 12 exit
    // Current month = targetMonth - 1, so maturity month = targetMonth + 11
    targetMaturityMonth:targetMonth + 11;

    // All bonds maturing in that month exit
    rolloffs:select from composition where (`month$maturityDate) = targetMaturityMonth;

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
// Uses determination date methodology:
//   - Bonds settling BEFORE prior month's determination date enter this month's index
//   - targetMonth is the index month (e.g., Feb for Jan->Feb extension)
//   - auctions: (optional) pre-loaded auction cache
forecastMonthWithAuctions:{[bonds;currentComp;targetMonth;indexName;asOfDate;auctions]
    monthEnd:.tsyindex.lastBizDayOfMonth targetMonth;

    // 1. Remove rolloffs (bonds maturing in targetMonth with < 1Y remaining)
    rolloffs:.tsyindex.predictRolloff[currentComp;targetMonth;indexName];
    rolloffCusips:rolloffs`cusip;
    remaining:select from currentComp where not cusip in rolloffCusips;

    // 2. Age remaining bonds to targetMonth end
    remaining:update remYears:(maturityDate - monthEnd) % 365.25 from remaining;
    remaining:update bucket:.tsyindex.assignBuckets remYears from remaining;
    remaining:update modDuration:.tsyindex.bondDuration[monthEnd;;]'[maturityDate;coupon;ytm] from remaining;

    // 3. Add new issuance - bonds from prior month settling before determination date
    // These are bonds auctioned in (targetMonth - 1) that settle before the determination date
    newIssues:.tsyindex.bondsEnteringForIndexWithAuctions[bonds;targetMonth;indexName;auctions];

    // Handle new issues: separate truly new CUSIPs from reopenings
    if[0 < count newIssues;
        existingCusips:remaining`cusip;

        // Reopenings: CUSIPs already in the index - add incremental amount
        reopenCusips:newIssues[`cusip] inter existingCusips;
        if[0 < count reopenCusips;
            reopenAmts:exec first outstanding by cusip from newIssues where cusip in reopenCusips;
            remaining:update marketValue:marketValue + reopenAmts cusip,
                            parAmount:parAmount + reopenAmts cusip
                     from remaining where cusip in reopenCusips];

        // Truly new CUSIPs: add as new rows
        newCusips:newIssues[`cusip] except existingCusips;
        if[0 < count newCusips;
            trulyNew:select from newIssues where cusip in newCusips;
            newRows:([]
                cusip:trulyNew`cusip;
                securityType:trulyNew`securityType;
                coupon:trulyNew`coupon;
                maturityDate:trulyNew`maturityDate;
                issueDate:trulyNew`issueDate;
                origTerm:trulyNew`origTerm;
                remYears:trulyNew`remYears;
                bucket:trulyNew`bucket;
                cleanPrice:count[trulyNew]#100f;  // assume par at issuance
                ytm:trulyNew`coupon;  // ytm ~ coupon at issuance
                parAmount:trulyNew`outstanding;
                marketValue:trulyNew`outstanding;  // at par
                weight:count[trulyNew]#0f;
                modDuration:trulyNew`modDuration;
                durationContrib:count[trulyNew]#0f);
            remaining:remaining,newRows]];

    // 4. Recalculate weights
    totalMV:sum remaining`marketValue;
    remaining:update weight:marketValue % totalMV from remaining;
    remaining:update durationContrib:weight * modDuration from remaining;

    // Return forecast composition plus metadata
    `composition`rolloffs`newIssues`asOfDate`targetMonth!(remaining;rolloffs;newIssues;asOfDate;targetMonth)}

// Backward-compatible wrapper
forecastMonth:{[bonds;currentComp;targetMonth;indexName;asOfDate]
    forecastMonthWithAuctions[bonds;currentComp;targetMonth;indexName;asOfDate;(::)]}

// Decompose duration change into components
decomposeDurationChange:{[prevComp;forecast]
    currentDur:sum prevComp`durationContrib;
    projectedDur:sum forecast[`composition]`durationContrib;

    // Duration lost from rolloffs
    rolloffDur:sum forecast[`rolloffs]`durationContrib;

    // Duration gained from new issuance (including reopenings)
    // For reopenings: use INCREMENTAL contribution (projected - current)
    // For truly new: use full projected contribution
    newIssues:forecast`newIssues;
    enteringCusips:$[0 < count newIssues; newIssues`cusip; `symbol$()];

    // Get dur contrib of entering CUSIPs in both compositions
    projDurEntering:$[0 < count enteringCusips;
        sum forecast[`composition;`durationContrib] where forecast[`composition;`cusip] in enteringCusips;
        0f];
    currDurEntering:$[0 < count enteringCusips;
        sum prevComp[`durationContrib] where prevComp[`cusip] in enteringCusips;
        0f];

    // Incremental duration from entering bonds
    newDur:projDurEntering - currDurEntering;

    // Duration change from aging (existing bonds, excluding entering)
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

        // Get entering/leaving counts (refunding months: 3, non-refunding: 4)
        newIssues:.tsyindex.predictNewIssuance[bonds;nextMonth;asOfDate];
        rolloffs:forecast`rolloffs;

        // Duration by bucket
        buckets:.tsyindex.durationByBucket comp;
        bucketDict:buckets[`bucket]!buckets[`durationContrib];

        // Calculate rebalancing effect (new - rolloffs) and aging effect
        rebalEffect:decomp[`extensionFromNew] + decomp`extensionFromRolloff;  // rolloff is already negative
        agingEffect:decomp`extensionFromAging;
        totalEffect:decomp`extensionTotal;

        // DV01 from rebalancing (net change due to adds/removes)
        // DV01 = duration × marketValue × 0.0001
        projComp:forecast`composition;
        newCusips:$[0 < count newIssues; newIssues`cusip; `symbol$()];
        newInProj:select from projComp where cusip in newCusips;
        dv01New:$[0 < count newInProj; sum newInProj[`modDuration] * newInProj[`marketValue] * 0.0001; 0f];
        dv01Rolloff:$[0 < count rolloffs; sum rolloffs[`modDuration] * rolloffs[`marketValue] * 0.0001; 0f];
        dv01Net:dv01New - dv01Rolloff;

        // Rebalancing effect by bucket
        // New issues: 10Y->7_10Y, 20Y->10_20Y, 30Y->20_30Y
        // Rolloffs: typically from 1_3Y bucket (bonds near maturity)
        newBuckets:select bucket, durationContrib from newInProj;
        newByBucket:exec sum durationContrib by bucket from newBuckets;
        rolloffByBucket:exec sum durationContrib by bucket from rolloffs;

        // Net rebalancing by bucket (new - rolloff)
        // Handle nulls before subtraction (null -> 0)
        getVal:{$[null x; 0f; x]};
        rebal1_3Y:(getVal newByBucket`1_3Y) - getVal rolloffByBucket`1_3Y;
        rebal3_5Y:(getVal newByBucket`3_5Y) - getVal rolloffByBucket`3_5Y;
        rebal5_7Y:(getVal newByBucket`5_7Y) - getVal rolloffByBucket`5_7Y;
        rebal7_10Y:(getVal newByBucket`7_10Y) - getVal rolloffByBucket`7_10Y;
        rebal10_20Y:(getVal newByBucket`10_20Y) - getVal rolloffByBucket`10_20Y;
        rebal20_30Y:(getVal newByBucket`20_30Y) - getVal rolloffByBucket`20_30Y;

        // Build result row
        ([]
            date:enlist asOfDate;
            forecastMonth:enlist nextMonth;
            bondCount:enlist stats`bondCount;
            totalMV:enlist stats`totalMV;
            dv01Net:enlist dv01Net;
            avgCoupon:enlist stats`avgCoupon;
            avgPrice:enlist wavg[comp`weight; comp`cleanPrice];
            currentDuration:enlist decomp`currentDuration;
            projectedDuration:enlist decomp`projectedDuration;
            rebalancingEffect:enlist rebalEffect;
            agingEffect:enlist agingEffect;
            totalDurationChange:enlist totalEffect;
            extensionFromNew:enlist decomp`extensionFromNew;
            extensionFromRolloff:enlist decomp`extensionFromRolloff;
            bondsEntering:enlist count newIssues;
            bondsLeaving:enlist count rolloffs;
            rebal1_3Y:enlist rebal1_3Y;
            rebal3_5Y:enlist rebal3_5Y;
            rebal5_7Y:enlist rebal5_7Y;
            rebal7_10Y:enlist rebal7_10Y;
            rebal10_20Y:enlist rebal10_20Y;
            rebal20_30Y:enlist rebal20_30Y;
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
//   weighting: (optional) `outstanding (default) or `publicFloat (SOMA-adjusted)
// Output:
//   Dict with keys: summary, composition, rolloffs, newIssues, buckets

extensionDetailWithAuctions:{[bonds;prices;asOfDate;indexName;weighting;auctions]
    // Forward to full version with no SOMA data
    extensionDetailFull[bonds;prices;asOfDate;indexName;weighting;auctions;(::)]}

// Full version with all data sources
extensionDetailFull:{[bonds;prices;asOfDate;indexName;weighting;auctions;somaCache]
    // Handle optional weighting parameter
    weighting:$[weighting ~ (::); `outstanding; weighting];

    // Convert prices to dict if table, or keep (::) for par pricing
    priceDict:$[prices ~ (::); (::); 99h = type prices; prices; prices[`cusip]!prices`price];

    // Compute outstanding amounts as of asOfDate from auction data
    // This gives correct historical outstanding, not just latest snapshot
    outstandings:$[0 < count auctions;
        .tsyindex.outstandingAsOf[auctions;asOfDate];
        (::)];

    // Build composition with date-dependent outstandings
    comp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;priceDict;outstandings];

    // Apply SOMA-adjusted weighting if requested and we have SOMA data
    if[(weighting ~ `publicFloat) and 0 < count somaCache;
        comp:.tsyindex.applyWeightingFromSoma[comp;auctions;somaCache;asOfDate]];

    stats:.tsyindex.indexStats comp;

    // Forecast (pass auctions through)
    nextMonth:1 + `month$asOfDate;
    forecast:.tsyindex.forecastMonthWithAuctions[bonds;comp;nextMonth;indexName;asOfDate;auctions];
    projComp:forecast`composition;

    // Apply same SOMA weighting to projected composition
    if[(weighting ~ `publicFloat) and 0 < count somaCache;
        projComp:.tsyindex.applyWeightingFromSoma[projComp;auctions;somaCache;asOfDate]];

    // All bonds with issuance in target month
    allEntering:forecast`newIssues;
    rolloffs:forecast`rolloffs;
    // Note: rolloffs come from comp which already has the weighting applied

    // Separate truly new CUSIPs from reopenings using API isReopening field
    // Truly new = first auction of this CUSIP (isReopening = false per API)
    // Reopening = adding to existing CUSIP (isReopening = true per API)
    existingCusips:comp`cusip;

    // Use isReopening flag from API if available, else fall back to CUSIP comparison
    hasReopenFlag:`isReopening in cols allEntering;

    trulyNew:$[0 < count allEntering;
        $[hasReopenFlag;
            select from allEntering where not isReopening;
            select from allEntering where not cusip in existingCusips];
        allEntering];
    reopenings:$[0 < count allEntering;
        $[hasReopenFlag;
            select from allEntering where isReopening;
            select from allEntering where cusip in existingCusips];
        0#allEntering];

    trulyNewCusips:$[0 < count trulyNew; trulyNew`cusip; `symbol$()];
    reopeningCusips:$[0 < count reopenings; reopenings`cusip; `symbol$()];

    // For reopenings, further split into:
    //   - alreadyInIndex: CUSIPs that were in the index (size increase only)
    //   - newToIndex: CUSIPs not in index before (new to index but market reopening)
    reopenAlreadyIn:reopeningCusips inter existingCusips;
    reopenNewToIndex:reopeningCusips except existingCusips;

    // Extension from truly new CUSIPs (first auction, not in market before)
    extFromTrulyNew:$[0 < count trulyNewCusips;
        sum exec durationContrib from projComp where cusip in trulyNewCusips;
        0f];

    // Extension from reopenings (size changes)
    // For bonds new to index: full projected duration contrib
    extFromReopenNew:$[0 < count reopenNewToIndex;
        sum exec durationContrib from projComp where cusip in reopenNewToIndex;
        0f];
    // For bonds already in index: incremental (projected - current)
    reopenCurrDur:$[0 < count reopenAlreadyIn;
        sum exec durationContrib from comp where cusip in reopenAlreadyIn;
        0f];
    reopenProjDur:$[0 < count reopenAlreadyIn;
        sum exec durationContrib from projComp where cusip in reopenAlreadyIn;
        0f];
    extFromReopenIncr:reopenProjDur - reopenCurrDur;

    extFromSizeChanges:extFromReopenNew + extFromReopenIncr;

    // Extension from rolloffs
    extFromRolloff:neg sum rolloffs`durationContrib;

    // Total rebalancing effect
    rebalEffect:extFromTrulyNew + extFromSizeChanges + extFromRolloff;

    // Aging effect (everything else)
    currentDur:sum comp`durationContrib;
    projectedDur:sum projComp`durationContrib;
    agingEffect:(projectedDur - currentDur) - rebalEffect;

    // Buckets
    buckets:.tsyindex.durationByBucket comp;

    // DV01 from rebalancing
    trulyNewInProj:$[0 < count trulyNewCusips; select from projComp where cusip in trulyNewCusips; 0#projComp];
    dv01New:$[0 < count trulyNewInProj; sum trulyNewInProj[`modDuration] * trulyNewInProj[`marketValue] * 0.0001; 0f];
    dv01Rolloff:$[0 < count rolloffs; sum rolloffs[`modDuration] * rolloffs[`marketValue] * 0.0001; 0f];
    dv01Net:dv01New - dv01Rolloff;

    // Rebalancing effect by bucket (for truly new only)
    newBuckets:$[0 < count trulyNewInProj; select bucket, durationContrib from trulyNewInProj; ([] bucket:`symbol$(); durationContrib:`float$())];
    newByBucket:exec sum durationContrib by bucket from newBuckets;
    rolloffByBucket:exec sum durationContrib by bucket from rolloffs;

    // Net rebalancing by bucket
    rebalByBucket:`1_3Y`3_5Y`5_7Y`7_10Y`10_20Y`20_30Y!{
        n:$[null x; 0f; x];
        r:$[null y; 0f; y];
        n - r
    }'[newByBucket`1_3Y`3_5Y`5_7Y`7_10Y`10_20Y`20_30Y; rolloffByBucket`1_3Y`3_5Y`5_7Y`7_10Y`10_20Y`20_30Y];

    // Summary with separated new vs size changes
    summary:`date`forecastMonth`weighting`bondCount`totalMV`dv01Net`avgCoupon`avgPrice`currentDuration`projectedDuration`rebalancingEffect`agingEffect`totalDurationChange`extFromTrulyNew`extFromSizeChanges`extFromRolloff`bondsEntering`bondsLeaving`sizeChanges`rebalByBucket!(
        asOfDate;
        nextMonth;
        weighting;
        stats`bondCount;
        stats`totalMV;
        dv01Net;
        stats`avgCoupon;
        wavg[comp`weight; comp`cleanPrice];
        currentDur;
        projectedDur;
        rebalEffect;
        agingEffect;
        projectedDur - currentDur;
        extFromTrulyNew;
        extFromSizeChanges;
        extFromRolloff;
        count trulyNew;
        count rolloffs;
        count reopenings;
        rebalByBucket);

    `summary`composition`projectedComp`rolloffs`newIssues`reopenings`buckets!(summary;comp;projComp;rolloffs;trulyNew;reopenings;buckets)}

// Backward-compatible wrapper (loads auction cache internally)
extensionDetail:{[bonds;prices;asOfDate;indexName;weighting]
    extensionDetailWithAuctions[bonds;prices;asOfDate;indexName;weighting;(::)]}

// =============================================================================
// SOMA-ADJUSTED WEIGHTING
// =============================================================================
// Apply publicFloat (SOMA-adjusted) weighting to a composition
// weighting: `outstanding (default) or `publicFloat (SOMA-adjusted)

applyWeighting:{[comp;bonds;weighting]
    if[weighting ~ `outstanding; :comp];
    if[not weighting ~ `publicFloat; :comp];

    // Get publicFloat from bonds
    pfDict:exec cusip!publicFloat from bonds where not null publicFloat, publicFloat > 0;

    // Replace parAmount with publicFloat where available
    // For CUSIPs not in pfDict (e.g., new issues), keep original parAmount (100% public at issuance)
    comp:update parAmount:?[cusip in key pfDict; pfDict cusip; parAmount] from comp;

    // Recalculate market value and weights
    comp:update marketValue:parAmount * cleanPrice % 100 from comp;
    totalMV:sum comp`marketValue;
    comp:update weight:marketValue % totalMV from comp;
    comp:update durationContrib:weight * modDuration from comp;

    comp}

// Apply SOMA-adjusted weighting using historical SOMA cache
// comp: index composition table
// auctions: auction cache (for computing outstanding as of date)
// somaCache: historical SOMA holdings cache
// asOfDate: date for SOMA snapshot lookup
applyWeightingFromSoma:{[comp;auctions;somaCache;asOfDate]
    // Get outstanding (totalAccepted) from auction data as of asOfDate
    outstanding:.tsyindex.outstandingAsOf[auctions;asOfDate];

    // Get SOMA holdings as of asOfDate (finds nearest prior weekly snapshot)
    soma:.tsy.somaHoldingsAsOf[somaCache;asOfDate];

    // For bonds where SOMA holdings aren't published yet, estimate SOMA as
    // totalAccepted - offeringAmt (the SOMA add-on at auction)
    // Build dict of estimated SOMA from auction data
    settledAuctions:select from auctions where issueDate <= asOfDate;
    estimatedSoma:$[(`totalAccepted in cols settledAuctions) and `offeringAmt in cols settledAuctions;
        exec cusip!somaAddon from select somaAddon:sum totalAccepted-offeringAmt by cusip from settledAuctions;
        ()!()];

    // Calculate public float = outstanding - SOMA for each CUSIP
    // For new issues not yet settled (outstanding = 0), use existing parAmount
    // (which is the totalAccepted from the entering bonds table)
    cusips:comp`cusip;
    existingParAmounts:exec cusip!parAmount from comp;

    pfDict:cusips!{[out;soma;estSoma;existing;c]
        o:$[c in key out; out c; 0f];
        // Use actual SOMA if available, else use estimated SOMA (add-on at auction)
        s:$[c in key soma; soma c; $[c in key estSoma; estSoma c; 0f]];
        // If outstanding is 0 (not yet settled), use existing parAmount
        // (which is totalAccepted from entering bonds - 100% public until SOMA buys)
        $[o = 0f; existing c; 0f | o - s]
    }[outstanding;soma;estimatedSoma;existingParAmounts;] each cusips;

    // Replace parAmount with publicFloat
    comp:update parAmount:pfDict cusip from comp;

    // Recalculate market value and weights
    comp:update marketValue:parAmount * cleanPrice % 100 from comp;
    totalMV:sum comp`marketValue;
    comp:update weight:marketValue % totalMV from comp;
    comp:update durationContrib:weight * modDuration from comp;

    comp}

// =============================================================================
// PREDICTED VS ACTUAL EXTENSION
// =============================================================================
// For predictions (before month end): only use auctions announced by prediction date
// For actuals (after month end): use all actual auction data

// Predicted extension - only uses auctions announced by prediction date
// predictionDate: typically the determination date of the current month
// Example: Predicting Feb extension on Jan 29 (Jan determination date)
// weighting: (optional) `outstanding (default) or `publicFloat (SOMA-adjusted)
extensionDetailPredicted:{[bonds;prices;predictionDate;indexName;weighting]
    // Handle optional weighting parameter
    weighting:$[weighting ~ (::); `outstanding; weighting];

    priceDict:$[prices ~ (::); (::); 99h = type prices; prices; prices[`cusip]!prices`price];

    // Build current composition as of prediction date
    comp:.tsyindex.buildIndexComposition[bonds;predictionDate;indexName;priceDict;(::)];

    // Apply SOMA-adjusted weighting if requested
    comp:.tsyindex.applyWeighting[comp;bonds;weighting];
    stats:.tsyindex.indexStats comp;

    // Index month = next month after prediction date
    indexMonth:1 + `month$predictionDate;
    auctionMonth:`month$predictionDate;

    // Get bonds entering with announcement filter
    detDate:determinationDate auctionMonth;
    newIssues:.tsyindex.bondsEnteringFiltered[bonds;auctionMonth;indexName;detDate;predictionDate];

    // Build projected composition
    rolloffs:.tsyindex.predictRolloff[comp;indexMonth;indexName];
    monthEnd:lastBizDayOfMonth indexMonth;

    // Start with current comp, remove rolloffs, age bonds
    remaining:select from comp where not cusip in rolloffs`cusip;
    remaining:update remYears:(maturityDate - monthEnd) % 365.25 from remaining;
    remaining:update modDuration:.tsyindex.bondDuration[monthEnd;;]'[maturityDate;coupon;ytm] from remaining;

    // Add new issues
    existingCusips:remaining`cusip;
    if[0 < count newIssues;
        reopenCusips:newIssues[`cusip] inter existingCusips;
        if[0 < count reopenCusips;
            reopenAmts:exec first outstanding by cusip from newIssues where cusip in reopenCusips;
            remaining:update marketValue:marketValue + reopenAmts cusip,
                            parAmount:parAmount + reopenAmts cusip
                     from remaining where cusip in reopenCusips];
        newCusips:newIssues[`cusip] except existingCusips;
        if[0 < count newCusips;
            trulyNew:select from newIssues where cusip in newCusips;
            newRows:([]
                cusip:trulyNew`cusip; securityType:trulyNew`securityType;
                coupon:trulyNew`coupon; maturityDate:trulyNew`maturityDate;
                issueDate:trulyNew`issueDate; origTerm:trulyNew`origTerm;
                remYears:trulyNew`remYears; bucket:trulyNew`bucket;
                cleanPrice:count[trulyNew]#100f; ytm:trulyNew`coupon;
                parAmount:trulyNew`outstanding; marketValue:trulyNew`outstanding;
                weight:count[trulyNew]#0f; modDuration:trulyNew`modDuration;
                durationContrib:count[trulyNew]#0f);
            remaining:remaining,newRows]];

    // Recalculate weights
    totalMV:sum remaining`marketValue;
    projComp:update weight:marketValue % totalMV from remaining;
    projComp:update durationContrib:weight * modDuration from projComp;

    // Calculate extension components (same logic as extensionDetail)
    hasReopenFlag:`isReopening in cols newIssues;
    trulyNew:$[0 < count newIssues;
        $[hasReopenFlag; select from newIssues where not isReopening;
          select from newIssues where not cusip in existingCusips]; newIssues];
    reopenings:$[0 < count newIssues;
        $[hasReopenFlag; select from newIssues where isReopening;
          select from newIssues where cusip in existingCusips]; 0#newIssues];

    trulyNewCusips:$[0 < count trulyNew; trulyNew`cusip; `symbol$()];
    extFromTrulyNew:$[0 < count trulyNewCusips;
        sum exec durationContrib from projComp where cusip in trulyNewCusips; 0f];
    extFromRolloff:neg sum rolloffs`durationContrib;

    currentDur:sum comp`durationContrib;
    projectedDur:sum projComp`durationContrib;
    rebalEffect:extFromTrulyNew + extFromRolloff;
    agingEffect:(projectedDur - currentDur) - rebalEffect;

    summary:`predictionDate`indexMonth`weighting`bondCount`currentDuration`projectedDuration`rebalancingEffect`agingEffect`totalDurationChange`extFromTrulyNew`extFromRolloff`bondsEntering`bondsLeaving`determinationDate!(
        predictionDate; indexMonth; weighting; stats`bondCount; currentDur; projectedDur;
        rebalEffect; agingEffect; projectedDur - currentDur;
        extFromTrulyNew; extFromRolloff; count trulyNew; count rolloffs; detDate);

    `summary`composition`projectedComp`rolloffs`newIssues!(summary;comp;projComp;rolloffs;trulyNew)}

// Actual extension - uses all actual auction data (for post-month analysis)
// asOfDate: typically the last business day of the month that just ended
// Example: Calculating actual Feb extension on Feb 28 (after Feb ends)
// weighting: (optional) `outstanding (default) or `publicFloat (SOMA-adjusted)
extensionDetailActual:{[bonds;prices;asOfDate;indexName;weighting]
    // Handle optional weighting parameter
    weighting:$[weighting ~ (::); `outstanding; weighting];

    priceDict:$[prices ~ (::); (::); 99h = type prices; prices; prices[`cusip]!prices`price];

    // The month that just ended
    endedMonth:`month$asOfDate;
    // Prior month end for starting composition
    priorMonthEnd:lastBizDayOfMonth[endedMonth - 1];

    // Build composition at prior month end
    comp:.tsyindex.buildIndexComposition[bonds;priorMonthEnd;indexName;priceDict;(::)];
    comp:.tsyindex.applyWeighting[comp;bonds;weighting];
    stats:.tsyindex.indexStats comp;

    // Get actual bonds that entered this month (no announcement filter)
    auctionMonth:endedMonth - 1;  // Look at prior month's auctions
    detDate:determinationDate auctionMonth;
    newIssues:.tsyindex.bondsEnteringFiltered[bonds;auctionMonth;indexName;detDate;(::)];

    // Build actual end-of-month composition
    actualComp:.tsyindex.buildIndexComposition[bonds;asOfDate;indexName;priceDict;(::)];
    actualComp:.tsyindex.applyWeighting[actualComp;bonds;weighting];
    actualStats:.tsyindex.indexStats actualComp;

    // Identify rolloffs (in prior comp but not in actual)
    rolloffCusips:comp[`cusip] except actualComp`cusip;
    rolloffs:select from comp where cusip in rolloffCusips;

    // Calculate actual extension
    hasReopenFlag:`isReopening in cols newIssues;
    existingCusips:comp`cusip;
    trulyNew:$[0 < count newIssues;
        $[hasReopenFlag; select from newIssues where not isReopening;
          select from newIssues where not cusip in existingCusips]; newIssues];
    reopenings:$[0 < count newIssues;
        $[hasReopenFlag; select from newIssues where isReopening;
          select from newIssues where cusip in existingCusips]; 0#newIssues];

    trulyNewCusips:$[0 < count trulyNew; trulyNew`cusip; `symbol$()];
    extFromTrulyNew:$[0 < count trulyNewCusips;
        sum exec durationContrib from actualComp where cusip in trulyNewCusips; 0f];
    extFromRolloff:neg sum rolloffs`durationContrib;

    priorDur:sum comp`durationContrib;
    actualDur:sum actualComp`durationContrib;
    rebalEffect:extFromTrulyNew + extFromRolloff;
    agingEffect:(actualDur - priorDur) - rebalEffect;

    summary:`asOfDate`endedMonth`weighting`bondCountPrior`bondCountActual`priorDuration`actualDuration`rebalancingEffect`agingEffect`totalDurationChange`extFromTrulyNew`extFromRolloff`bondsEntered`bondsLeft`determinationDate!(
        asOfDate; endedMonth; weighting; stats`bondCount; actualStats`bondCount;
        priorDur; actualDur; rebalEffect; agingEffect; actualDur - priorDur;
        extFromTrulyNew; extFromRolloff; count trulyNew; count rolloffs; detDate);

    `summary`priorComposition`actualComposition`rolloffs`newIssues!(summary;comp;actualComp;rolloffs;trulyNew)}

// =============================================================================
// TIME SERIES EXTENSION CALCULATION
// =============================================================================
// Calculate extension for each date in a prices table

// extensionTimeSeriesWithAuctions: Calculate index extension with user-provided data
// bonds: bond table from .tsy.loadBondCache[]
// auctions: auction table from .tsy.loadAuctionCache[]
// pricesTable: table with columns (date; cusip; price) - one row per cusip per date
// indexName: `BBERG or `ICE
// weighting: `outstanding or `publicFloat (SOMA-adjusted)
// Returns: table with one row per date showing extension components
// NOTE: For publicFloat weighting, use extensionTimeSeriesFull which accepts SOMA cache
extensionTimeSeriesWithAuctions:{[bonds;auctions;pricesTable;indexName;weighting]
    extensionTimeSeriesFull[bonds;auctions;(::);pricesTable;indexName;weighting]}

// Full version with SOMA cache for publicFloat weighting
// bonds: bond table from .tsy.loadBondCache[]
// auctions: auction table from .tsy.loadAuctionCache[]
// somaCache: SOMA holdings from .tsy.loadSomaCache[] (required for publicFloat weighting)
// pricesTable: table with columns (date; cusip; price)
// indexName: `BBERG or `ICE
// weighting: `outstanding or `publicFloat (SOMA-adjusted)
extensionTimeSeriesFull:{[bonds;auctions;somaCache;pricesTable;indexName;weighting]
    weighting:$[weighting ~ (::); `outstanding; weighting];

    // Warn if publicFloat requested but no SOMA data
    if[(weighting ~ `publicFloat) and (somaCache ~ (::)) or 0 = count somaCache;
        -1 "WARNING: publicFloat weighting requested but no SOMA cache provided. Using outstanding.";
        weighting:`outstanding];

    // Get unique dates
    dates:asc distinct pricesTable`date;

    // Process each date
    calcOneDate:{[bonds;auctions;somaCache;pricesTable;indexName;weighting;dt]
        // Extract prices for this date as dict
        dayPrices:exec cusip!price from pricesTable where date = dt;

        // Calculate extension using Full version (with error handling)
        result:.[.tsyindex.extensionDetailFull; (bonds;dayPrices;dt;indexName;weighting;auctions;somaCache);
            {[e] (enlist `error)!(enlist e)}];

        // Handle errors
        if[`error in key result;
            :([] date:enlist dt; duration:0n; projectedDuration:0n;
               rebalancing:0n; aging:0n; totalChange:0n;
               newIssues:0N; sizeChanges:0N; rolloffs:0N; error:enlist result`error)];

        sum1:result`summary;

        ([] date:enlist dt;
            duration:enlist sum1`currentDuration;
            projectedDuration:enlist sum1`projectedDuration;
            rebalancing:enlist sum1`rebalancingEffect;
            aging:enlist sum1`agingEffect;
            totalChange:enlist sum1`totalDurationChange;
            newIssues:enlist sum1`bondsEntering;
            sizeChanges:enlist sum1`sizeChanges;
            rolloffs:enlist sum1`bondsLeaving;
            error:enlist "")
    };

    results:calcOneDate[bonds;auctions;somaCache;pricesTable;indexName;weighting;] each dates;
    raze results}

// Backward-compatible wrapper (loads auction cache internally)
extensionTimeSeries:{[bonds;pricesTable;indexName;weighting]
    auctions:@[.tsy.loadAuctionCache; ::; {[e] ([] cusip:`symbol$())}];
    extensionTimeSeriesFull[bonds;auctions;(::);pricesTable;indexName;weighting]}

// extensionTimeSeriesFromPriceMatrix: Calculate extension from wide-format price matrix
// bonds: bond table from .tsy.loadBondCache[]
// priceMatrix: table with date column and one column per cusip (wide format)
// indexName: `BBERG or `ICE
// weighting: `outstanding or `publicFloat (SOMA-adjusted)
// Returns: table with one row per date showing extension components
extensionTimeSeriesFromPriceMatrix:{[bonds;priceMatrix;indexName;weighting]
    // Convert wide to long format
    dates:priceMatrix`date;
    cusipCols:cols[priceMatrix] except `date;

    // Build long-format table
    pricesLong:raze {[pm;dt;cusips]
        prices:pm[pm[`date]?dt;cusips];
        ([] date:count[cusips]#dt; cusip:cusips; price:prices)
    }[priceMatrix;;cusipCols] each dates;

    // Call main function
    extensionTimeSeries[bonds;pricesLong;indexName;weighting]}

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
    -1 "    Predict auctions settling before month-end (for bond count)";
    -1 "";
    -1 "  predictNewIssuanceForExtension[bonds;targetMonth;asOfDate]";
    -1 "    Predict issuance for extension calc (includes 10Y reopen)";
    -1 "";
    -1 "  predictRolloff[composition;targetMonth;indexName]";
    -1 "    Predict bonds exiting index";
    -1 "";
    -1 "  getOfferAmount[bonds;asOfDate;oTerm;isReopen]";
    -1 "    Get offer amount for term (uses offeringAmount, not outstanding)";
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
    -1 "SIZING METHODOLOGY:";
    -1 "  New issuance uses offeringAmount (public market allocation).";
    -1 "  Index MV uses outstanding (total issued) as denominator.";
    -1 "";
    -1 "  Non-refunding months (Jan, Mar, Apr, Jun, Jul, Sep, Oct, Dec):";
    -1 "    - 2Y, 3Y, 5Y, 7Y, 10Y (reopen) auctions";
    -1 "    - Expected rebalancing effect: ~0.06 years";
    -1 "";
    -1 "  Refunding months (Feb, May, Aug, Nov):";
    -1 "    - 3Y, 10Y (original), 30Y auctions";
    -1 "    - Plus 2Y, 5Y, 7Y from prior month (settle in current month)";
    -1 "    - Expected rebalancing effect: ~0.05 years";
    -1 "";
    -1 "  NOTE: Uses actual issueDate from API data, not estimated dates.";
    -1 "";
    }

// Startup message
-1 "Loaded .tsyindex namespace v",version;
-1 "Functions: indexRebalanceForecast, nextMonthChanges, buildIndexComposition";
-1 "Run .tsyindex.help[] for usage, .tsyindex.example[] for demo";

\d .
