// =============================================================================
// TREASURY FUTURES DELIVERABLE BASKET LIBRARY
// =============================================================================
// Manages Treasury bond data and generates deliverable baskets for futures
//
// Usage: \l tsy.q
// =============================================================================

\d .tsy

version:"0.1.0"

// =============================================================================
// CONFIGURATION & CONSTANTS
// =============================================================================

// Supported futures contracts
contracts:`TU`FV`TY`US`WN

// CME delivery specifications (as of March 2025)
// Reference: https://www.cmegroup.com/markets/interest-rates/us-treasury/
//
// Contract | Remaining Maturity      | Original Maturity  | Type
// ---------|-------------------------|--------------------|-----
// TU (2Y)  | >= 1y9m, <= 2y*         | <= 5y 3m           | Note
// FV (5Y)  | >= 4y2m                 | <= 5y 3m           | Note
// TY (10Y) | >= 6y6m, < 8y           | <= 10y             | Note
// US (Bond)| >= 15y, < 25y           | (any)              | Bond
// WN (Ultra)| >= 25y                 | (any)              | Bond
// * TU max is from last day of delivery month, others from first day
//
deliverySpecs:([contract:`TU`FV`TY`US`WN]
    name:("2-Year Note";"5-Year Note";"10-Year Note (6.5-8Y)";"Treasury Bond";"Ultra T-Bond");
    minYears:1.75 4.167 6.5 15.0 25.0f;
    maxYears:2.0 100.0 8.0 25.0 100f;    // FV/WN have no max (set to 100)
    minInclusive:11111b;                  // all minimums are inclusive (>=)
    maxInclusive:10001b;                  // TU/WN max inclusive, others exclusive (<)
    origMaxYears:5.25 5.25 10.0 100.0 100f;  // original maturity constraint (100 = no limit)
    origTerms:((`2Y`3Y`5Y);(`5Y`7Y);(`7Y`10Y);(`20Y`30Y);(`20Y`30Y));
    secType:(`Note`Note`Note`Bond`Bond);
    useRounding:01111b                    // all use 3-month rounding except TU
  )

// Standard coupon for conversion factor (6% semi-annual)
standardYield:0.06

// API endpoints
apiBaseUrl:"https://www.treasurydirect.gov/TA_WS/securities/announced"

// Cache file location
cacheDir:"/tmp/"
bondCacheFile:"tsy_bonds.csv"

// =============================================================================
// DATE UTILITIES
// =============================================================================

// Parse security term to symbol (e.g., "10-Year" -> `10Y, "30-Year" -> `30Y)
// Handles cases like "29-Year 10-Month" by taking just the year part
parseSecurityTerm:{[term]
    t:$[10h = type term; term; string term];
    // Split on space and take first part (year component)
    parts:" " vs t;
    yearPart:first parts;
    // Extract just the number before "-Year"
    n:"I"$yearPart where yearPart in "0123456789";
    `$string[n],"Y"}

// Generate delivery months (Mar, Jun, Sep, Dec) from startYear to endYear
genDeliveryMonths:{[startYear;endYear]
    years:startYear + til 1 + endYear - startYear;
    raze {[y] `month$("D"$string[y],".03.01";"D"$string[y],".06.01";"D"$string[y],".09.01";"D"$string[y],".12.01")} each years}

// First day of month
firstOfMonth:{[ym]
    y:`year$ym;
    m:`mm$ym;
    "D"$string[y],".",(-2#"0",string m),".01"}

// Last day of month
lastOfMonth:{[ym]
    nextMonth:ym + 1;
    -1 + firstOfMonth nextMonth}

// First business day of month (simplified - doesn't account for holidays)
firstBizDay:{[ym]
    d:firstOfMonth ym;
    dow:d mod 7;
    d + $[dow = 5; 2; dow = 6; 1; 0]}

// Last business day of month
lastBizDay:{[ym]
    d:lastOfMonth ym;
    dow:d mod 7;
    d - $[dow = 5; 1; dow = 6; 2; 0]}

// First delivery day of contract month (first business day)
firstDeliveryDay:{[ym] firstBizDay ym}

// Last delivery day (last business day of month)
lastDeliveryDay:{[ym] lastBizDay ym}

// Remaining maturity in years
remainingYears:{[maturityDate;asOfDate]
    (maturityDate - asOfDate) % 365.25}

// Round remaining maturity down to complete 3-month (quarter) increments
// CME uses calendar months for this calculation:
//   1. Calculate months from 1st of delivery month to 1st of maturity month
//   2. Round down to complete quarters (3-month periods)
// maturityDate: bond maturity date
// deliveryDate: first day of delivery month
roundToQuarters:{[maturityDate;deliveryDate]
    // Calculate complete calendar months between dates
    months:(`month$maturityDate) - `month$deliveryDate;
    // Round down to complete quarters
    quarters:months div 3;
    // Convert back to years
    quarters % 4f}

// =============================================================================
// TREASURYDIRECT API FUNCTIONS
// =============================================================================

// Search API endpoint (supports date ranges)
apiSearchUrl:"https://www.treasurydirect.gov/TA_WS/securities/search"

// Fetch securities from TreasuryDirect API (recent only, max 250)
// secType: `Note or `Bond
fetchSecurities:{[secType]
    url:apiBaseUrl,"?format=json&type=",string[secType],"&pagesize=250";
    resp:system "curl -s \"",url,"\"";
    .j.k "" sv resp}

// Fetch securities for a date range using search endpoint
// secType: `Note or `Bond
// startDate, endDate: dates as strings "YYYY-MM-DD" or Q dates
fetchSecuritiesDateRange:{[secType;startDate;endDate]
    // Convert Q dates to YYYY-MM-DD format (API requires dashes, not dots)
    sd:$[10h = type startDate; startDate; ssr[string startDate;".";"-"]];
    ed:$[10h = type endDate; endDate; ssr[string endDate;".";"-"]];
    url:apiSearchUrl,"?format=json&type=",string[secType],"&pagesize=250&startDate=",sd,"&endDate=",ed;
    resp:system "curl -s \"",url,"\"";
    result:.j.k "" sv resp;
    // API returns empty list if no results
    $[0 = count result; (); result]}

// Fetch all notes (recent)
fetchNotes:{[] .tsy.fetchSecurities`Note}

// Fetch all bonds (recent)
fetchBonds:{[] .tsy.fetchSecurities`Bond}

// Fetch one year of data for a security type
fetchOneYear:{[secType;y]
    sd:string[y],"-01-01";
    ed:string[y],"-12-31";
    -1 "  ",string[y],"...";
    result:.tsy.fetchSecuritiesDateRange[secType;sd;ed];
    result}

// Fetch historical data for a security type across multiple years
// Returns raw API data (list of dicts)
fetchHistorical:{[secType;startYear;endYear]
    years:startYear + til 1 + endYear - startYear;
    -1 "Fetching ",string[secType]," from ",string[startYear]," to ",string[endYear],"...";

    // Fetch year by year to avoid hitting limits
    data:raze .tsy.fetchOneYear[secType;] each years;

    -1 "  Total: ",string[count data]," records";
    data}

// Build complete historical cache (one-time operation)
// Fetches all Notes and Bonds from startYear to present
buildHistoricalCache:{[startYear]
    endYear:`year$.z.d;

    -1 "=== Building Historical Treasury Cache ===";
    -1 "Date range: ",string[startYear]," to ",string endYear;
    -1 "";

    // Fetch Notes
    notesRaw:.tsy.fetchHistorical[`Note;startYear;endYear];
    notes:.tsy.parseApiResponse notesRaw;
    -1 "Notes parsed: ",string count notes;
    -1 "";

    // Fetch Bonds
    bondsRaw:.tsy.fetchHistorical[`Bond;startYear;endYear];
    bonds:.tsy.parseApiResponse bondsRaw;
    -1 "Bonds parsed: ",string count bonds;
    -1 "";

    // Combine and deduplicate
    combined:notes,bonds;
    combined:`auctionDate xasc combined;
    combined:select securityType:first securityType, coupon:first coupon, issueDate:first issueDate,
                    maturityDate:first maturityDate, origTerm:first origTerm, auctionDate:first auctionDate
             by cusip from combined;
    combined:0!combined;

    -1 "Total unique securities: ",string count combined;
    -1 "";

    // Save to cache
    .tsy.saveBondCache combined;
    combined}

// Update existing cache with new data (incremental update)
// Fetches data from last cache date to present and merges
updateCache:{[]
    // Load existing cache
    existing:.tsy.loadBondCache[];
    lastDate:max existing`auctionDate;

    -1 "=== Updating Treasury Cache ===";
    -1 "Last auction in cache: ",string lastDate;
    -1 "";

    // Fetch from last date to now
    startDate:lastDate + 1;
    endDate:.z.d;

    if[startDate > endDate;
        -1 "Cache is already up to date";
        :existing];

    sd:ssr[string startDate;".";"-"];
    ed:ssr[string endDate;".";"-"];

    -1 "Fetching new data from ",sd," to ",ed,"...";

    // Fetch new Notes
    newNotesRaw:.tsy.fetchSecuritiesDateRange[`Note;sd;ed];
    newNotes:$[0 < count newNotesRaw; .tsy.parseApiResponse newNotesRaw; 0#existing];
    -1 "  New Notes: ",string count newNotes;

    // Fetch new Bonds
    newBondsRaw:.tsy.fetchSecuritiesDateRange[`Bond;sd;ed];
    newBonds:$[0 < count newBondsRaw; .tsy.parseApiResponse newBondsRaw; 0#existing];
    -1 "  New Bonds: ",string count newBonds;

    // Combine with existing
    combined:existing,newNotes,newBonds;
    combined:`auctionDate xasc combined;
    combined:select securityType:first securityType, coupon:first coupon, issueDate:first issueDate,
                    maturityDate:first maturityDate, origTerm:first origTerm, auctionDate:first auctionDate
             by cusip from combined;
    combined:0!combined;

    newCount:(count combined) - count existing;
    -1 "";
    -1 "Added ",string[newCount]," new securities";
    -1 "Total securities: ",string count combined;

    // Save updated cache
    .tsy.saveBondCache combined;
    combined}

// Parse API response to standard bond table format
parseApiResponse:{[data]
    // data is a list of dictionaries from JSON - convert to table
    rawTable:flip data;

    // Extract and convert fields
    cusips:`$rawTable`cusip;
    secTypes:`$rawTable`securityType;
    coupons:0.01 * "F"$rawTable`interestRate;
    issueDates:"D"$10#/:rawTable`issueDate;
    matDates:"D"$10#/:rawTable`maturityDate;
    origTerms:.tsy.parseSecurityTerm each rawTable`securityTerm;
    auctionDates:"D"$10#/:rawTable`auctionDate;

    // Build table
    ([] cusip:cusips; securityType:secTypes; coupon:coupons;
        issueDate:issueDates; maturityDate:matDates;
        origTerm:origTerms; auctionDate:auctionDates)}

// Fetch and parse all Treasury securities
fetchAllSecurities:{[]
    -1 "Fetching Notes from TreasuryDirect...";
    notes:parseApiResponse fetchNotes[];
    -1 "  Retrieved ",string[count notes]," notes";

    -1 "Fetching Bonds from TreasuryDirect...";
    bonds:parseApiResponse fetchBonds[];
    -1 "  Retrieved ",string[count bonds]," bonds";

    // Combine and deduplicate by CUSIP (keep earliest auction for each CUSIP)
    combined:notes,bonds;
    combined:`auctionDate xasc combined;
    combined:select securityType:first securityType, coupon:first coupon, issueDate:first issueDate,
                    maturityDate:first maturityDate, origTerm:first origTerm, auctionDate:first auctionDate
             by cusip from combined;
    combined:0!combined;
    -1 "Total unique securities: ",string count combined;
    combined}

// =============================================================================
// BOND DATA MANAGEMENT
// =============================================================================

// Empty bond table schema
emptyBonds:([]
    cusip:`symbol$();
    securityType:`symbol$();
    coupon:`float$();
    issueDate:`date$();
    maturityDate:`date$();
    origTerm:`symbol$();
    auctionDate:`date$()
  )

// Save bonds to CSV cache
saveBondCache:{[bonds]
    filepath:cacheDir,bondCacheFile;
    path:`$":",filepath;
    (path) 0: csv 0: bonds;
    -1 "Saved ",string[count bonds]," bonds to ",filepath;
    bonds}

// Load bonds from CSV cache
loadBondCache:{[]
    filepath:cacheDir,bondCacheFile;
    path:`$":",filepath;
    if[()~key path; '"Bond cache not found at ",filepath,". Run .tsy.fetchAllSecurities[] first"];
    bonds:("SSFDDSD";enlist csv) 0: path;
    -1 "Loaded ",string[count bonds]," bonds from cache";
    bonds}

// Load bonds from custom CSV file
loadBondsFromCSV:{[filepath]
    bonds:("SSFDDSD";enlist csv) 0: `$filepath;
    -1 "Loaded ",string[count bonds]," bonds from ",filepath;
    bonds}

// Filter to eligible securities (exclude TIPS, FRNs, CMBs)
filterEligible:{[bonds]
    // Keep only Note and Bond types
    bonds:select from bonds where securityType in `Note`Bond;
    // Exclude TIPS (typically have "TIP" in term or very low coupons indicating TIPS)
    // This is a simplified filter - real implementation would need better TIPS identification
    bonds}

// Get bonds outstanding as of a specific date
bondsAsOf:{[bonds;asOfDate]
    select from bonds where issueDate <= asOfDate, maturityDate > asOfDate}

// =============================================================================
// CONVERSION FACTOR CALCULATION
// =============================================================================

// Calculate conversion factor for a deliverable bond
// Uses CME standard: price at 6% yield, rounded to first day of delivery month
// deliveryMonth: month of delivery (e.g., 2025.03m)
// maturityDate: bond maturity date
// coupon: annual coupon rate (e.g., 0.045 for 4.5%)
conversionFactor:{[deliveryMonth;maturityDate;coupon]
    // First day of delivery month
    deliveryDate:firstOfMonth deliveryMonth;

    // Time to maturity in months, rounded down to quarters
    monthsToMat:(`month$maturityDate) - `month$deliveryDate;
    if[monthsToMat <= 0; :1f];

    // Round down to nearest quarter (3 months)
    quarters:`int$monthsToMat % 3;
    n:quarters div 2;  // Number of semi-annual periods
    z:(quarters mod 2) * 3 % 12f;  // Fractional first period (0 or 0.25)

    if[n = 0; :1f];

    // Semi-annual coupon and yield
    c:coupon % 2;           // Semi-annual coupon
    y:standardYield % 2;    // Semi-annual yield (3%)

    // CF formula (CBOT standard)
    // CF = (c/y) * (1 - 1/(1+y)^n) + 1/(1+y)^n - (c * z)
    // Adjusted for fractional period
    factor:(c % y) * (1 - 1 % xexp[1+y;n]) + (1 % xexp[1+y;n]);
    factor:factor * 1 % xexp[1+y;z*2];  // Adjust for fractional period
    factor:factor - c * z;

    // Round to 4 decimal places (exchange convention)
    0.0001 * `long$factor * 10000}

// =============================================================================
// ELIGIBILITY FILTERING
// =============================================================================

// Check if a bond is eligible for a specific contract at delivery
isEligible:{[bond;contract;deliveryMonth]
    spec:deliverySpecs contract;
    deliveryDate:firstDeliveryDay deliveryMonth;

    // Check security type
    if[not bond[`securityType] = spec`secType; :0b];

    // Check original term (if specified)
    if[not bond[`origTerm] in spec`origTerms; :0b];

    // Check remaining maturity with inclusive/exclusive boundaries
    // Apply 3-month rounding for all contracts except TU
    remYrsExact:.tsy.remainingYears[bond`maturityDate;deliveryDate];
    remYrs:$[spec`useRounding; .tsy.roundToQuarters[bond`maturityDate;deliveryDate]; remYrsExact];

    // Min check: >= if inclusive, > if exclusive
    minOk:$[spec`minInclusive; remYrs >= spec`minYears; remYrs > spec`minYears];
    if[not minOk; :0b];

    // Max check: <= if inclusive, < if exclusive
    maxOk:$[spec`maxInclusive; remYrs <= spec`maxYears; remYrs < spec`maxYears];
    if[not maxOk; :0b];

    // Must be issued before delivery
    if[bond[`issueDate] > deliveryDate; :0b];

    1b}

// Get all eligible bonds for a contract/delivery month
eligibleBonds:{[bonds;contract;deliveryMonth]
    select from bonds where isEligible[;contract;deliveryMonth] each bonds}

// Alternative using table filtering (faster for large tables)
filterEligibleBonds:{[bonds;contract;deliveryMonth]
    spec:deliverySpecs contract;
    deliveryDate:firstDeliveryDay deliveryMonth;

    // Filter by security type and original term
    b:select from bonds where
        securityType = spec`secType,
        origTerm in spec`origTerms,
        issueDate <= deliveryDate;

    // Calculate remaining maturity
    b:update remYearsExact:.tsy.remainingYears'[maturityDate;deliveryDate] from b;

    // Apply 3-month rounding for eligibility (all contracts except TU)
    // CME rounds remaining maturity down to complete 3-month increments
    b:$[spec`useRounding;
        update remYears:.tsy.roundToQuarters'[maturityDate;deliveryDate] from b;
        update remYears:remYearsExact from b];

    // Apply min bound (>= if inclusive, > if exclusive)
    b:$[spec`minInclusive;
        select from b where remYears >= spec`minYears;
        select from b where remYears > spec`minYears];

    // Apply max bound (<= if inclusive, < if exclusive)
    $[spec`maxInclusive;
        select from b where remYears <= spec`maxYears;
        select from b where remYears < spec`maxYears]}

// =============================================================================
// BASKET GENERATION
// =============================================================================

// Generate deliverable basket for one contract/delivery month
genBasket:{[bonds;contract;deliveryMonth]
    // Get eligible bonds
    eligible:filterEligibleBonds[bonds;contract;deliveryMonth];

    if[0 = count eligible; :([] contract:(); deliveryMonth:(); cusip:(); coupon:(); maturityDate:(); remainingYears:(); cf:(); origTerm:())];

    // Calculate conversion factors
    eligible:update cf:.tsy.conversionFactor'[deliveryMonth;maturityDate;coupon] from eligible;

    // Format output
    select
        contract:contract,
        deliveryMonth:deliveryMonth,
        cusip,
        coupon,
        maturityDate,
        remainingYears:remYears,
        cf,
        origTerm
    from eligible}

// Generate baskets for all contracts for one delivery month
genAllBaskets:{[bonds;deliveryMonth]
    baskets:.tsy.genBasket[bonds;;deliveryMonth] peach .tsy.contracts;
    raze baskets}

// Generate historical baskets from startYear to current
genHistoricalBaskets:{[bonds;startYear]
    endYear:`year$.z.d;
    months:genDeliveryMonths[startYear;endYear];
    // Filter to past/current months only
    months:months where months <= `month$.z.d;

    -1 "Generating baskets for ",string[count months]," delivery months...";
    baskets:raze genAllBaskets[bonds;] peach months;
    -1 "Generated ",string[count baskets]," total basket entries";
    baskets}

// Generate all baskets for all contracts and delivery months in a date range
// Returns comprehensive table with contract, deliveryMonth, and all bond details
// startYear/endYear: year range (e.g., 2020, 2026)
// includeAll: if 1b, include future months; if 0b (default), only up to current month
allBaskets:{[bonds;startYear;endYear;includeAll]
    months:genDeliveryMonths[startYear;endYear];

    // Optionally filter to past/current months only
    if[not includeAll; months:months where months <= `month$.z.d];

    -1 "Generating baskets for ",string[count months]," delivery months x 5 contracts...";
    baskets:raze genAllBaskets[bonds;] peach months;

    // Add formatted delivery code (e.g., "TYH26" for March 2026 TY)
    // Month codes: H=Mar(3), M=Jun(6), U=Sep(9), Z=Dec(12)
    mc:"HMUZ";
    mkCode:{[mc;c;m] `$string[c],mc[((`mm$m)-3) div 3],-2#string 100+(`year$m) mod 100};
    baskets:update deliveryCode:mkCode[mc;;]'[contract;deliveryMonth] from baskets;

    // Reorder columns to put delivery info first
    outcols:`deliveryCode`contract`deliveryMonth`cusip`coupon`maturityDate`remainingYears`cf`origTerm;
    -1 "Generated ",string[count baskets]," total basket entries";
    outcols xcols baskets}

// Convenience wrapper: all baskets from startYear to endYear (historical only)
allBasketsRange:{[bonds;startYear;endYear]
    allBaskets[bonds;startYear;endYear;0b]}

// Convenience wrapper: all baskets from startYear to current
allBasketsFrom:{[bonds;startYear]
    allBaskets[bonds;startYear;`year$.z.d;0b]}

// Main entry point: build all deliverable baskets
buildDeliverableBaskets:{[startYear]
    // Load or fetch bond data
    bonds:$[()~key `$":",cacheDir,bondCacheFile;
        [b:.tsy.fetchAllSecurities[]; .tsy.saveBondCache b; b];
        .tsy.loadBondCache[]];

    // Generate historical baskets
    .tsy.genHistoricalBaskets[bonds;startYear]}

// =============================================================================
// ANALYSIS & DISPLAY
// =============================================================================

// Basket summary statistics
basketStats:{[basket]
    if[0 = count basket; :`count`minCoupon`maxCoupon`minCF`maxCF`minRemYrs`maxRemYrs!(0;0n;0n;0n;0n;0n;0n)];
    `count`minCoupon`maxCoupon`minCF`maxCF`minRemYrs`maxRemYrs!(
        count basket;
        min basket`coupon;
        max basket`coupon;
        min basket`cf;
        max basket`cf;
        min basket`remainingYears;
        max basket`remainingYears)}

// Show basket summary
showBasket:{[basket]
    if[0 = count basket; -1 "Empty basket"; :()];

    contract:first basket`contract;
    deliveryMonth:first basket`deliveryMonth;

    -1 "";
    -1 "=== DELIVERABLE BASKET: ",string[contract]," ",string[deliveryMonth]," ===";
    -1 "";
    -1 "Eligible bonds: ",string count basket;
    -1 "";

    // Sort by CF descending
    basket:`cf xdesc basket;

    -1 "CUSIP      Coupon   Maturity    RemYrs   CF";
    -1 "------------------------------------------------";
    {-1 (string x`cusip),"  ",(5$string 100*x`coupon),"%   ",(string x`maturityDate),"  ",(5$string x`remainingYears),"   ",(string x`cf)} each basket;
    -1 "";
    basket}

// Compare basket counts across delivery months
basketSummary:{[baskets]
    select cnt:count i,
           minCF:min cf,
           maxCF:max cf,
           avgCF:avg cf
    by contract, deliveryMonth from baskets}

// Show all baskets for a delivery month (all 5 contracts)
showAllBaskets:{[bonds;deliveryMonth]
    -1 "";
    -1 "=============================================================================";
    -1 "           DELIVERABLE BASKETS FOR ",string deliveryMonth;
    -1 "=============================================================================";

    // Generate all baskets
    allBaskets:genAllBaskets[bonds;deliveryMonth];

    // Show each contract
    showOneContract:{[specs;baskets;c]
        b:select from baskets where contract = c;
        n:(specs c)`name;
        -1 "";
        $[0 = count b;
            -1 "--- ",string[c]," (",n,") - No eligible bonds ---";
            [
                -1 "--- ",string[c]," (",n,") - ",string[count b]," bonds ---";
                -1 "";
                b:`cf xdesc b;
                -1 "    CUSIP      Coupon   Maturity    RemYrs   CF       OrigTerm";
                -1 "    ----------------------------------------------------------------";
                {-1 "    ",(string x`cusip),"  ",(5$string 100*x`coupon),"%   ",(string x`maturityDate),"  ",(5$string x`remainingYears),"   ",(6$string x`cf),"  ",string x`origTerm} each b
            ]
        ];
        1b  // Always return same type
    };
    showOneContract[deliverySpecs;allBaskets;] each contracts;

    -1 "";
    -1 "=============================================================================";
    -1 "";

    // Return the combined table
    allBaskets}

// Get summary table for all contracts for a delivery month
allBasketsSummary:{[bonds;deliveryMonth]
    baskets:genAllBaskets[bonds;deliveryMonth];
    summary:([]
        contract:contracts;
        name:(deliverySpecs each contracts)`name;
        cnt:{[b;c] count select from b where contract=c}[baskets;] each contracts;
        minCF:{[b;c] exec min cf from b where contract=c}[baskets;] each contracts;
        maxCF:{[b;c] exec max cf from b where contract=c}[baskets;] each contracts;
        minYrs:{[b;c] exec min remainingYears from b where contract=c}[baskets;] each contracts;
        maxYrs:{[b;c] exec max remainingYears from b where contract=c}[baskets;] each contracts
      );
    summary}

// =============================================================================
// INTEGRATION WITH CTD.Q
// =============================================================================

// Convert basket to ctd.q bond format (requires market prices)
// prices: dictionary of cusip -> cleanPrice
toCtdBonds:{[basket;prices]
    {[prices;b]
        price:prices[b`cusip];
        if[null price; price:100f];  // Default to par if no price
        `sym`coupon`maturity`cleanPrice`cf!(b`cusip;b`coupon;b`maturityDate;price;b`cf)
    }[prices;] each basket}

// =============================================================================
// HELP & DOCUMENTATION
// =============================================================================

example:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "              TREASURY DELIVERABLE BASKET LIBRARY EXAMPLE";
    -1 "=============================================================================";
    -1 "";

    -1 "1. FETCH TREASURY DATA";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // Fetch from TreasuryDirect API (requires internet)";
    -1 "   bonds:.tsy.fetchAllSecurities[]";
    -1 "   .tsy.saveBondCache bonds   // Save to cache";
    -1 "";
    -1 "   // Or load from cache";
    -1 "   bonds:.tsy.loadBondCache[]";
    -1 "";

    -1 "2. GENERATE BASKET FOR SPECIFIC CONTRACT/MONTH";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // March 2025 10-Year Note futures";
    -1 "   basket:.tsy.genBasket[bonds;`TY;2025.03m]";
    -1 "   .tsy.showBasket basket";
    -1 "";

    -1 "3. GENERATE ALL HISTORICAL BASKETS";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // All contracts from 2018 to present";
    -1 "   baskets:.tsy.buildDeliverableBaskets[2018]";
    -1 "";
    -1 "   // Summary by contract/month";
    -1 "   .tsy.basketSummary baskets";
    -1 "";

    -1 "4. INTEGRATION WITH CTD ANALYSIS";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // Get basket and add market prices";
    -1 "   basket:.tsy.genBasket[bonds;`TY;2025.03m]";
    -1 "   prices:basket[`cusip]!100+0.1*til count basket  // Mock prices";
    -1 "   ctdBonds:.tsy.toCtdBonds[basket;prices]";
    -1 "";
    -1 "   // Run CTD analysis (requires ctd.q)";
    -1 "   // .ctd.ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;ctdBonds]";
    -1 "";

    -1 "=============================================================================";
    -1 "                         END EXAMPLE";
    -1 "=============================================================================";
    -1 "";}

usage:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                       .tsy USAGE REFERENCE";
    -1 "=============================================================================";
    -1 "";
    -1 "// DATA LOADING";
    -1 ".tsy.fetchAllSecurities[]              // Fetch Notes & Bonds from TreasuryDirect";
    -1 ".tsy.saveBondCache[bonds]              // Save to CSV cache";
    -1 ".tsy.loadBondCache[]                   // Load from CSV cache";
    -1 ".tsy.loadBondsFromCSV[filepath]        // Load from custom CSV";
    -1 "";
    -1 "// BASKET GENERATION";
    -1 ".tsy.genBasket[bonds;contract;deliveryMonth]   // Single contract/month";
    -1 ".tsy.genAllBaskets[bonds;deliveryMonth]        // All contracts for month";
    -1 ".tsy.genHistoricalBaskets[bonds;startYear]     // All from startYear";
    -1 ".tsy.buildDeliverableBaskets[startYear]        // Full pipeline";
    -1 "";
    -1 "// ANALYSIS";
    -1 ".tsy.showBasket[basket]                // Display basket details";
    -1 ".tsy.basketStats[basket]               // Summary statistics";
    -1 ".tsy.basketSummary[baskets]            // Cross-month comparison";
    -1 "";
    -1 "// CONVERSION FACTOR";
    -1 ".tsy.conversionFactor[deliveryMonth;maturityDate;coupon]";
    -1 "";
    -1 "// INTEGRATION";
    -1 ".tsy.toCtdBonds[basket;prices]         // Convert to ctd.q format";
    -1 "";
    -1 "// CONTRACTS: `TU (2Y) `FV (5Y) `TY (10Y) `US (Bond) `WN (Ultra)";
    -1 "";}

help:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                         .tsy FUNCTIONS";
    -1 "=============================================================================";
    -1 "";
    -1 "DATA LOADING:";
    -1 "  fetchAllSecurities[]          - Fetch Notes & Bonds from TreasuryDirect API";
    -1 "  fetchNotes[]                  - Fetch Notes only";
    -1 "  fetchBonds[]                  - Fetch Bonds only";
    -1 "  saveBondCache[bonds]          - Save bonds to CSV cache";
    -1 "  loadBondCache[]               - Load bonds from CSV cache";
    -1 "  loadBondsFromCSV[filepath]    - Load bonds from custom CSV";
    -1 "  bondsAsOf[bonds;asOfDate]     - Filter bonds outstanding as of date";
    -1 "";
    -1 "ELIGIBILITY:";
    -1 "  isEligible[bond;contract;deliveryMonth]       - Check single bond";
    -1 "  filterEligibleBonds[bonds;contract;deliveryMonth] - Filter table";
    -1 "";
    -1 "CONVERSION FACTOR:";
    -1 "  conversionFactor[deliveryMonth;maturityDate;coupon] - CME CF formula";
    -1 "";
    -1 "BASKET GENERATION:";
    -1 "  genBasket[bonds;contract;deliveryMonth]  - Single contract/month basket";
    -1 "  genAllBaskets[bonds;deliveryMonth]       - All 5 contracts for month";
    -1 "  allBaskets[bonds;startYr;endYr;inclFut]  - All contracts, all months in range";
    -1 "  allBasketsRange[bonds;startYr;endYr]     - All baskets (historical only)";
    -1 "  allBasketsFrom[bonds;startYr]            - All baskets from startYr to now";
    -1 "  genHistoricalBaskets[bonds;startYear]    - Historical baskets (legacy)";
    -1 "  buildDeliverableBaskets[startYear]       - Full pipeline (fetch + generate)";
    -1 "";
    -1 "ANALYSIS:";
    -1 "  showBasket[basket]            - Display single basket with details";
    -1 "  showAllBaskets[bonds;deliveryMonth] - Display ALL 5 contracts for month";
    -1 "  allBasketsSummary[bonds;deliveryMonth] - Summary table for all contracts";
    -1 "  basketStats[basket]           - Summary statistics for one basket";
    -1 "  basketSummary[baskets]        - Cross-month comparison table";
    -1 "";
    -1 "UTILITIES:";
    -1 "  genDeliveryMonths[startYear;endYear]  - Generate Mar/Jun/Sep/Dec months";
    -1 "  firstDeliveryDay[deliveryMonth]       - First delivery day";
    -1 "  lastDeliveryDay[deliveryMonth]        - Last delivery day";
    -1 "  remainingYears[maturityDate;asOfDate] - Years to maturity";
    -1 "";
    -1 "INTEGRATION:";
    -1 "  toCtdBonds[basket;prices]     - Convert to ctd.q bond format";
    -1 "";
    -1 "DELIVERY SPECIFICATIONS (as of March 2025):";
    -1 "  Contract   Remaining Maturity     Original Maturity   Type";
    -1 "  ---------  ---------------------  ------------------  ----";
    -1 "  TU (2Y)    >= 1y9m, <= 2y*        <= 5y 3m            Note";
    -1 "  FV (5Y)    >= 4y2m                <= 5y 3m            Note";
    -1 "  TY (10Y)   >= 6y6m, < 8y          <= 10y              Note";
    -1 "  US (Bond)  >= 15y, < 25y          (any)               Bond";
    -1 "  WN (Ultra) >= 25y                 (any)               Bond";
    -1 "  * TU max is from last day of delivery month";
    -1 "";
    -1 "  3-MONTH ROUNDING: All contracts except TU use 3-month rounding";
    -1 "  increments. Remaining maturity is calculated as calendar months";
    -1 "  from delivery month, then rounded DOWN to complete quarters.";
    -1 "";
    -1 "Run .tsy.usage[] for quick reference or .tsy.example[] for demo.";
    -1 "";}

// =============================================================================
// INITIALIZATION
// =============================================================================

-1 "Loaded .tsy namespace v",version;
-1 "Functions: fetchAllSecurities, genBasket, buildDeliverableBaskets";
-1 "Run .tsy.usage[] for quick reference, .tsy.help[] for functions, .tsy.example[] for demo";

\d .
