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
defaultFreq:`6M

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
// Optimized: clean price = futures * CF (AI cancels out)
invoiceYield:{[deliveryDate;futuresPrice;ctdBond]
    // Clean price = invoice price - AI = (futures * CF + AI) - AI = futures * CF
    cleanPrice:futuresPrice * ctdBond`cf;
    // Solve for yield (Newton-Raphson via ctd.q)
    .ctd.yieldToMaturity[deliveryDate;ctdBond`maturity;ctdBond`coupon;`SA;100f;cleanPrice]}

// Swap rate: Forward par rate from delivery date to CTD maturity
swapRate:{[curve;deliveryDate;ctdMaturity;freq]
    .swaps.parRateDated[curve;deliveryDate;ctdMaturity;freq]}

// Invoice spread: Swap Rate - Invoice Yield
// Positive spread = swaps trading wide of Treasuries
invoiceSpread:{[curve;deliveryDate;futuresPrice;ctdBond;freq]
    invYld:invoiceYield[deliveryDate;futuresPrice;ctdBond];
    ctdMat:ctdBond`maturity;
    swpRt:swapRate[curve;deliveryDate;ctdMat;freq];
    spread:swpRt - invYld;
    `invoiceYield`swapRate`invoiceSpread`spreadBps!(invYld;swpRt;spread;spread*10000)}

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
hedgeRatioBatch:{[curve;settleDate;deliveryDate;ctdBonds;swapNotional;freq]
    // Compute futures DV01s in parallel (per $1MM)
    futDV01s:futuresDV01Batch[curve;settleDate;ctdBonds];
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
//   2. Timing option: When to deliver during the month
//   3. Wild card option: Delivery after futures stop trading
//
// OAS = Invoice Spread - (Delivery Option Value / DV01)
// A higher option value means the standard spread OVERSTATES the true spread

// Net basis for a single bond (used to rank CTD candidates)
// Net basis = Gross basis - Carry = Bond price - (Futures * CF) - Carry
// Lower net basis = more likely to be delivered (higher implied repo)
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

// Rank all bonds by net basis to find CTD and near-CTD
// Returns table sorted by net basis (lowest = CTD)
rankByNetBasis:{[curve;settleDate;deliveryDate;futuresPrice;bonds;repoRate]
    nbs:netBasis[curve;settleDate;deliveryDate;futuresPrice;;repoRate] each bonds;
    syms:bonds@\:`sym;
    t:([] sym:syms; netBasis:nbs; bond:bonds);
    `netBasis xasc t}

// Calculate switch points: rate shift (in bp) where CTD changes
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
        // Rate move for this bond to become CTD
        switchRate:$[abs[nbSens] > 1e-6; nbGap % nbSens; 0n];
        `sym`switchBps`bond!(row`sym;switchRate * 10000;bond)
        }[curveUp;curveDn;settleDate;ctdFutDV01;ctdNB] each others;
    result}

// Estimate delivery option value using switch point analysis
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

// Option-adjusted invoice spread
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
    // OAS = standard spread - option adjustment
    // The option makes futures cheaper for the short, so standard spread overstates
    oas:std[`invoiceSpread] - (optYieldAdj * 0.0001);
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
// Bond DV01 / CF, normalized to $1MM for direct comparison with swap DV01
// Uses central difference for accuracy
futuresDV01:{[curve;settleDate;ctdBond]
    mat:ctdBond`maturity; cpn:ctdBond`coupon; cf:ctdBond`cf;
    // Bump curve ±1bp (central difference for accuracy)
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];
    // Bond DV01 = (P_down - P_up) / 2 per 100 face
    bondDV01:(.ctd.priceBond[curveDn;settleDate;mat;cpn;`SA;100f] -
              .ctd.priceBond[curveUp;settleDate;mat;cpn;`SA;100f]) % 2;
    // Futures DV01 per $1MM = Bond DV01 / CF * 1000000 / 100
    bondDV01 * 10000 % cf}

// Batch futures DV01 for multiple bonds (parallelized)
// Returns DV01 per $1MM notional for each bond
futuresDV01Batch:{[curve;settleDate;ctdBonds]
    curveUp:.swaps.bumpCurve[curve;0.0001];
    curveDn:.swaps.bumpCurve[curve;-0.0001];
    // Price all bonds in parallel
    priceUp:pmap[{[c;s;b].ctd.priceBond[c;s;b`maturity;b`coupon;`SA;100f]}[curveUp;settleDate;];3;ctdBonds];
    priceDn:pmap[{[c;s;b].ctd.priceBond[c;s;b`maturity;b`coupon;`SA;100f]}[curveDn;settleDate;];3;ctdBonds];
    bondDV01s:(priceDn - priceUp) % 2;
    cfs:ctdBonds@\:`cf;
    // DV01 per $1MM = Bond DV01 / CF * 10000
    bondDV01s * 10000 % cfs}

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
hedgeRatio:{[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq]
    ctdMat:ctdBond`maturity;
    // Swap DV01 for the given notional
    swpDV01:swapDV01Dated[curve;swapNotional;deliveryDate;ctdMat;freq];
    // Futures DV01 per $1MM
    futDV01perMM:futuresDV01[curve;settleDate;ctdBond];
    // Futures notional needed: (futNotional/1MM) * futDV01perMM = swpDV01
    // So: futNotional = swpDV01 * 1MM / futDV01perMM
    futNotional:swpDV01 * 1000000 % futDV01perMM;
    `swapDV01`futuresDV01`swapNotional`futuresNotional!(swpDV01;futDV01perMM;swapNotional;futNotional)}

// =============================================================================
// POSITION SIZING
// =============================================================================

// Calculate futures notional from swap notional (convenience wrapper)
futuresNotionalFromSwap:{[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq]
    hr:hedgeRatio[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq];
    hr`futuresNotional}

// Calculate swap notional from futures notional
swapNotionalFromFutures:{[curve;settleDate;deliveryDate;ctdBond;futuresNotional;freq]
    ctdMat:ctdBond`maturity;
    // Futures DV01 per $1MM
    futDV01perMM:futuresDV01[curve;settleDate;ctdBond];
    // Total futures DV01
    totalFutDV01:futuresNotional * futDV01perMM % 1000000;
    // Swap DV01 per $1MM
    swpDV01perMM:swapDV01Dated[curve;1000000f;deliveryDate;ctdMat;freq];
    // Swap notional that produces same DV01
    1000000f * totalFutDV01 % swpDV01perMM}

// Create a balanced position (notional-based)
// direction: `long (receive fixed, short futures) or `short (pay fixed, long futures)
createPosition:{[curve;deliveryDate;futuresPrice;ctdBond;swapNotional;freq;direction]
    settleDate:curve[`config]`asOfDate;
    hr:hedgeRatio[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq];
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
    // Hedge ratio
    hr:hedgeRatio[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq];
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

    // Calculate hedge ratio
    hr:hedgeRatio[curve;dt;deliveryDate;ctdBond;1000000f;freq];

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

    // Calculate hedge ratio (using curve's asOfDate as settle)
    settleDate:curve[`config]`asOfDate;
    hr:hedgeRatio[curve;settleDate;row`deliveryDate;ctdBond;1000000f;freq];

    // Return result dict
    `invoiceYield`swapRate`invoiceSpread`spreadBps`swapDV01`futuresDV01!(
        spread`invoiceYield; spread`swapRate; spread`invoiceSpread;
        spread`spreadBps; hr`swapDV01; hr`futuresDV01)}

// Daily invoice spread from pre-computed CTD table
// ctdTable: table with date, deliveryDate, futuresPrice, sym, coupon, maturity, cf
// curves: static dict, builder {[dt]...}, or ([] date; curve)
// freq: payment frequency (`6M default)
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

// Helper: process one CTD row for OAS
dailyOASFromCTDRow:{[ctx;row]
    dt:row`date;
    dc:row`deliveryCode;
    curve:ctx[`curvesByDate] dt;
    vol:ctx`vol;
    freq:ctx`freq;
    repoRate:ctx[`repoByDate] dt;

    // Build CTD bond dict
    ctdBond:`sym`coupon`maturity`cleanPrice`cf!(
        row`sym; row`coupon; row`maturity;
        $[ctx`hasCtdCleanPrice; row`cleanPrice; 100f];
        row`cf);

    // Get basket for this date/deliveryCode
    basketRows:select from ctx[`basketTable] where date=dt, deliveryCode=dc;
    if[0=count basketRows;
        // No basket - return invoice spread only, no option value
        spread:invoiceSpread[curve;row`deliveryDate;row`futuresPrice;ctdBond;freq];
        settleDate:curve[`config]`asOfDate;
        hr:hedgeRatio[curve;settleDate;row`deliveryDate;ctdBond;1000000f;freq];
        :`invoiceYield`swapRate`invoiceSpread`spreadBps`optionValue`optionAdjBps`oas`oasBps`swapDV01`futuresDV01`basketSize!(
            spread`invoiceYield; spread`swapRate; spread`invoiceSpread; spread`spreadBps;
            0f; 0f; spread`invoiceSpread; spread`spreadBps;
            hr`swapDV01; hr`futuresDV01; 0)];

    // Convert basket rows to bond dicts
    basket:rowToBondWithFlag[ctx`hasBasketCleanPrice;] each basketRows;

    // Calculate OAS
    oasResult:oasInvoiceSpread[curve;row`deliveryDate;row`futuresPrice;ctdBond;basket;repoRate;vol;freq];

    // Calculate hedge ratio
    settleDate:curve[`config]`asOfDate;
    hr:hedgeRatio[curve;settleDate;row`deliveryDate;ctdBond;1000000f;freq];

    `invoiceYield`swapRate`invoiceSpread`spreadBps`optionValue`optionAdjBps`oas`oasBps`swapDV01`futuresDV01`basketSize!(
        oasResult`invoiceYield; oasResult`swapRate; oasResult`invoiceSpread; oasResult`spreadBps;
        oasResult`optionValue; oasResult`optionAdjBps; oasResult`oas; oasResult`oasBps;
        hr`swapDV01; hr`futuresDV01; count basket)}

// Daily OAS from pre-computed CTD table and basket
// ctdTable: table with date, deliveryCode, deliveryDate, futuresPrice, sym, coupon, maturity, cf
// basketTable: table with date, deliveryCode, sym, coupon, maturity, cf, cleanPrice (all deliverable bonds)
// curves: static dict, builder {[dt]...}, or ([] date; curve)
// repoRate: scalar or ([] date; rate)
// vol: annualized rate volatility (e.g., 0.01 = 100bp/year)
// freq: payment frequency (`6M default)
dailyOASFromCTD:{[ctdTable;basketTable;curves;repoRate;vol;freq]
    if[0=count ctdTable; :ctdTable];

    // Handle defaults
    frq:$[freq~(::); `6M; freq];
    v:$[vol~(::); 0.01; vol];

    // Get unique dates from CTD table
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

    // Pre-compute repo by date
    repoByDate:$[-9h=type repoRate; dates!nDates#repoRate;
                 99h=type repoRate; dates!nDates#repoRate`rate;
                 98h=type repoRate; dates!(exec date!rate from repoRate) dates;
                 dates!nDates#repoRate];

    // Check if cleanPrice column exists
    hasCtdCleanPrice:`cleanPrice in cols ctdTable;
    hasBasketCleanPrice:`cleanPrice in cols basketTable;

    // Pack context
    ctx:`curvesByDate`repoByDate`basketTable`vol`freq`hasCtdCleanPrice`hasBasketCleanPrice!(
        curvesByDate;repoByDate;basketTable;v;frq;hasCtdCleanPrice;hasBasketCleanPrice);

    // Process each CTD row
    results:dailyOASFromCTDRow[ctx;] each ctdTable;

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
    deliveryFn:$[useLast; .tsy.lastDeliveryDay; .tsy.firstDeliveryDay];

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

    // Pre-compute delivery dates by code
    deliveryDateByCode:deliveryCodes!{[bc;dc;fn] fn first bc[dc]`deliveryMonth}[basketsByCode;;deliveryFn] each deliveryCodes;
    contractByCode:deliveryCodes!{[bc;dc] first bc[dc]`contract}[basketsByCode;] each deliveryCodes;
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
    deliveryFn:$[useLast; .tsy.lastDeliveryDay; .tsy.firstDeliveryDay];

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

    // Pre-compute delivery dates by code
    deliveryDateByCode:deliveryCodes!{[bc;dc;fn] fn first bc[dc]`deliveryMonth}[basketsByCode;;deliveryFn] each deliveryCodes;
    contractByCode:deliveryCodes!{[bc;dc] first bc[dc]`contract}[basketsByCode;] each deliveryCodes;
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

    hr:hedgeRatio[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq];
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
    -1 "// HEDGE RATIO & POSITION SIZING (all notional-based)";
    -1 ".invoicepricer.futuresDV01[curve;settleDate;ctdBond]               // DV01 per $1MM notional";
    -1 ".invoicepricer.swapDV01Dated[curve;notional;deliveryDate;ctdMaturity;freq]";
    -1 ".invoicepricer.hedgeRatio[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq]";
    -1 "    // Returns: swapDV01, futuresDV01, swapNotional, futuresNotional";
    -1 ".invoicepricer.futuresNotionalFromSwap[curve;settleDate;deliveryDate;ctdBond;swapNotional;freq]";
    -1 ".invoicepricer.swapNotionalFromFutures[curve;settleDate;deliveryDate;ctdBond;futuresNotional;freq]";
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
    -1 ".invoicepricer.futuresDV01Batch[curve;settleDate;ctdBonds]";
    -1 ".invoicepricer.hedgeRatioBatch[curve;settleDate;deliveryDate;ctdBonds;swapNotional;freq]";
    -1 "";
    -1 "// DAILY FUNCTIONS";
    -1 "";
    -1 "// From pre-computed CTD table (PREFERRED - faster, no CTD recomputation)";
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
