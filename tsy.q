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
    // Handle formats like "4-Week", "13-Week", "2-Year", "10-Year", "42-Day"
    tl:lower t;
    // Determine suffix: D for days, W for weeks, Y for years
    suffix:$[tl like "*day*"; "D"; tl like "*week*"; "W"; "Y"];
    // Extract the number
    n:"I"$t where t in "0123456789";
    `$string[n],suffix}

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

// HTTP fetch helper using system curl
// Returns JSON string or signals error
httpGet:{[url]
    cmd:"curl -sLk --connect-timeout 30 \"",url,"\"";
    resp:@[system;cmd;{[e] '"curl error: ",e}];
    if[0 = count resp; '"Empty response from ",url];
    "" sv resp}

// Fetch securities from TreasuryDirect API (recent only, max 250)
// secType: `Note or `Bond
fetchSecurities:{[secType]
    url:apiBaseUrl,"?format=json&type=",string[secType],"&pagesize=250";
    .j.k httpGet url}

// Fetch securities for a date range using search endpoint
// secType: `Note or `Bond
// startDate, endDate: dates as strings "YYYY-MM-DD" or Q dates
fetchSecuritiesDateRange:{[secType;startDate;endDate]
    // Convert Q dates to YYYY-MM-DD format (API requires dashes, not dots)
    sd:$[10h = type startDate; startDate; ssr[string startDate;".";"-"]];
    ed:$[10h = type endDate; endDate; ssr[string endDate;".";"-"]];
    url:apiSearchUrl,"?format=json&type=",string[secType],"&pagesize=250&startDate=",sd,"&endDate=",ed;
    result:.j.k httpGet url;
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
// Fetches all Notes and Bonds from startYear to present and saves to cache
buildHistoricalCache:{[startYear]
    combined:.tsy.fetchAllSecurities[startYear];
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

// Fetch and parse all Treasury securities (full historical dataset)
// startYear: optional, defaults to 2010 (needed for 2018+ baskets due to 30Y bonds)
fetchAllSecurities:{[startYear]
    startYear:$[startYear ~ (::); 2010; startYear];
    endYear:`year$.z.d;

    -1 "=== Fetching Treasury Securities ===";
    -1 "Date range: ",string[startYear]," to ",string endYear;
    -1 "";

    // Fetch Notes year by year
    notesRaw:.tsy.fetchHistorical[`Note;startYear;endYear];
    notes:.tsy.parseApiResponse notesRaw;
    -1 "Notes parsed: ",string count notes;
    -1 "";

    // Fetch Bonds year by year
    bondsRaw:.tsy.fetchHistorical[`Bond;startYear;endYear];
    bonds:.tsy.parseApiResponse bondsRaw;
    -1 "Bonds parsed: ",string count bonds;
    -1 "";

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
// T-BILL SUPPORT
// =============================================================================
// T-Bills are fetched and cached separately from Notes/Bonds
// They are NOT included in futures basket/CTD logic
// Used for building on-the-run Treasury yield curves

billCacheFile:"tsy_bills.csv"

// Fetch T-Bills (recent only, max 250)
fetchBills:{[] .tsy.fetchSecurities`Bill}

// Fetch one year of bills
fetchBillsOneYear:{[y]
    sd:string[y],"-01-01";
    ed:string[y],"-12-31";
    -1 "  ",string[y],"...";
    result:.tsy.fetchSecuritiesDateRange[`Bill;sd;ed];
    result}

// Fetch historical T-Bills across multiple years
fetchBillsHistorical:{[startYear;endYear]
    years:startYear + til 1 + endYear - startYear;
    -1 "Fetching Bills from ",string[startYear]," to ",string[endYear],"...";
    data:raze .tsy.fetchBillsOneYear each years;
    -1 "  Total: ",string[count data]," records";
    data}

// Parse T-Bill API response
// T-Bills have no coupon; they trade at a discount
parseBillResponse:{[data]
    if[0 = count data; :([] cusip:`symbol$(); securityType:`symbol$();
        discountRate:`float$(); issueDate:`date$(); maturityDate:`date$();
        origTerm:`symbol$(); auctionDate:`date$(); price:`float$())];

    rawTable:flip data;

    cusips:`$rawTable`cusip;
    secTypes:`$rawTable`securityType;
    // Bills have discount rate instead of coupon (use averageMedianDiscountRate)
    discRates:0.01 * "F"$rawTable`averageMedianDiscountRate;
    issueDates:"D"$10#/:rawTable`issueDate;
    matDates:"D"$10#/:rawTable`maturityDate;
    origTerms:.tsy.parseSecurityTerm each rawTable`securityTerm;
    auctionDates:"D"$10#/:rawTable`auctionDate;
    // Some responses include price
    prices:$[`pricePer100 in key rawTable;
        "F"$rawTable`pricePer100;
        count[cusips]#0n];

    ([] cusip:cusips; securityType:secTypes; discountRate:discRates;
        issueDate:issueDates; maturityDate:matDates;
        origTerm:origTerms; auctionDate:auctionDates; price:prices)}

// Fetch all T-Bills from startYear to present
fetchAllBills:{[startYear]
    startYear:$[startYear ~ (::); 2020; startYear];
    endYear:`year$.z.d;

    -1 "=== Fetching Treasury Bills ===";
    -1 "Date range: ",string[startYear]," to ",string endYear;
    -1 "";

    billsRaw:.tsy.fetchBillsHistorical[startYear;endYear];
    bills:.tsy.parseBillResponse billsRaw;
    -1 "Bills parsed: ",string count bills;

    // Deduplicate by CUSIP (keep first by auction date)
    bills:`auctionDate xasc distinct bills;
    if[count bills;
        bills:select from bills where i = (first;i) fby cusip];
    -1 "After dedup: ",string count bills;

    bills}

// Save bills to CSV cache
saveBillCache:{[bills]
    filepath:cacheDir,billCacheFile;
    path:`$":",filepath;
    (path) 0: csv 0: bills;
    -1 "Saved ",string[count bills]," bills to ",filepath;
    bills}

// Load bills from CSV cache
loadBillCache:{[]
    filepath:cacheDir,billCacheFile;
    path:`$":",filepath;
    if[()~key path; '"Bill cache not found at ",filepath,". Run .tsy.fetchAllBills[] first"];
    bills:("SSFDDSDJ";enlist csv) 0: path;
    -1 "Loaded ",string[count bills]," bills from cache";
    bills}

// Build and save bill cache
// startDate/endDate: dates, or year integers, or (::) for defaults
// Examples:
//   buildBillCache[2020;(::)]                  - year 2020 to present
//   buildBillCache[2020;2024]                  - years 2020-2024
//   buildBillCache["D"$"2020-01-15";.z.d]      - specific date range
buildBillCache:{[startDate;endDate]
    // Convert inputs to dates
    // Accept: date, year (int/long), or (::) for default
    toDate:{[x;isEnd]
        $[x ~ (::);
            $[isEnd; .z.d; "D"$"2018.01.01"];  // defaults
          -14h = type x;
            x;  // already a date
          (type x) in -6 -7h;
            // Year integer: convert to Jan 1 or Dec 31
            $[isEnd; "D"$(string x),".12.31"; "D"$(string x),".01.01"];
          '"Invalid date type. Use date or year integer"]};

    startDate:toDate[startDate;0b];
    endDate:toDate[endDate;1b];

    // Extract years for API calls
    startYear:`year$startDate;
    endYear:`year$endDate;

    -1 "=== Building Bill Cache ===";
    -1 "Date range: ",string[startDate]," to ",string endDate;

    // Fetch by year range
    billsRaw:.tsy.fetchBillsHistorical[startYear;endYear];
    bills:.tsy.parseBillResponse billsRaw;

    // Filter to exact date range (by issue date)
    bills:select from bills where issueDate >= startDate, issueDate <= endDate;
    -1 "Bills in date range: ",string count bills;

    // Deduplicate by CUSIP (keep first by auction date)
    bills:`auctionDate xasc distinct bills;
    if[count bills;
        bills:select from bills where i = (first;i) fby cusip];
    -1 "After dedup: ",string count bills;

    .tsy.saveBillCache bills;
    bills}

// =============================================================================
// ON-THE-RUN TREASURY CURVE
// =============================================================================
// On-the-run = most recently auctioned security at each maturity point
// Combines T-Bills (short end) with Notes/Bonds (long end)

// Standard on-the-run tenors
otrTenors:`4W`8W`13W`17W`26W`52W`2Y`3Y`5Y`7Y`10Y`20Y`30Y

// Map tenor symbols to approximate year fractions
tenorToYF:{[t]
    mapping:`4W`8W`13W`17W`26W`52W`2Y`3Y`5Y`7Y`10Y`20Y`30Y!
        (4%52;8%52;13%52;17%52;26%52;1f;2f;3f;5f;7f;10f;20f;30f);
    mapping t}

// Find on-the-run security for each tenor as of a given date
// Returns table with tenor, cusip, maturity, yield, etc.
findOnTheRun:{[bonds;bills;asOfDate]
    // Filter to securities issued on or before asOfDate and not yet matured
    activeBonds:select from bonds where issueDate <= asOfDate, maturityDate > asOfDate;

    // For each original term, find most recently auctioned
    otrBonds:select cusip, securityType, coupon, maturityDate, origTerm, auctionDate from activeBonds where auctionDate = (max;auctionDate) fby origTerm;

    // Build result table for bonds (2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)
    bondTenors:`2Y`3Y`5Y`7Y`10Y`20Y`30Y;
    bondRows:{[otrBonds;asOfDate;t]
        b:select from otrBonds where origTerm = t;
        if[0 = count b; :([] tenor:`symbol$(); cusip:`symbol$(); securityType:`symbol$(); coupon:`float$(); maturityDate:`date$(); auctionDate:`date$(); remainingYears:`float$(); price:`float$())];
        b:first b;
        ([] tenor:enlist t; cusip:enlist b`cusip; securityType:enlist b`securityType; coupon:enlist b`coupon; maturityDate:enlist b`maturityDate; auctionDate:enlist b`auctionDate; remainingYears:enlist (b[`maturityDate] - asOfDate) % 365f; price:enlist 0n)
    }[otrBonds;asOfDate;] each bondTenors;

    raze bondRows}

// Bill yield from discount rate and days to maturity
// Bank discount yield to bond-equivalent yield conversion
billBEY:{[discountRate;daysToMat]
    // BEY = (365 * discount rate) / (360 - discount rate * days)
    (365 * discountRate) % (360 - (discountRate * daysToMat))}

// Bill yield from price (if price available)
billYieldFromPrice:{[price;daysToMat]
    // Simple yield = ((100 - price) / price) * (365 / days)
    ((100 - price) % price) * (365 % daysToMat)}

// Build on-the-run yield curve for a SINGLE date (internal)
// priceDict: cusip -> price
buildOTRCurveOne:{[bonds;bills;asOfDate;priceDict]
    // Get on-the-run securities
    otr:findOnTheRun[bonds;bills;asOfDate];

    // Add bills to OTR
    activeBills:select from bills where issueDate <= asOfDate, maturityDate > asOfDate;
    otrBills:select cusip, securityType, discountRate, maturityDate, origTerm, auctionDate from activeBills where auctionDate = (max;auctionDate) fby origTerm;

    billTenors:`4W`8W`13W`17W`26W`52W;
    emptyBillTable:([] tenor:`symbol$(); cusip:`symbol$(); securityType:`symbol$(); coupon:`float$(); maturityDate:`date$(); auctionDate:`date$(); remainingYears:`float$(); price:`float$(); yield:`float$());
    billRows:{[otrBills;asOfDate;priceDict;emptyBillTable;t]
        b:select from otrBills where origTerm = t;
        if[0 = count b; :emptyBillTable];
        b:first b;
        daysToMat:b[`maturityDate] - asOfDate;
        // Try price from dict, else use discount rate
        px:$[(b`cusip) in key priceDict; priceDict b`cusip; 0n];
        yld:$[not null px; billYieldFromPrice[px;daysToMat]; billBEY[b`discountRate;daysToMat]];
        ([] tenor:enlist t; cusip:enlist b`cusip; securityType:enlist b`securityType; coupon:enlist 0f; maturityDate:enlist b`maturityDate; auctionDate:enlist b`auctionDate; remainingYears:enlist daysToMat%365f; price:enlist px; yield:enlist yld)
    }[otrBills;asOfDate;priceDict;emptyBillTable;] each billTenors;
    billTable:raze billRows;

    // Add yields to bond OTR (need prices)
    otr:update yield:{[settle;mat;cpn;px]
        $[null px; 0n; .ctd.yieldToMaturity[settle;mat;cpn;`SA;100f;px]]
        }[asOfDate]'[maturityDate;coupon;priceDict cusip] from otr;

    // Update bond prices from input
    otr:update price:priceDict cusip from otr;

    // Combine bills and bonds
    combined:billTable,otr;

    // Sort by remaining years and add date column
    update date:asOfDate from `remainingYears xasc combined}

// Build on-the-run yield curve
// Supports single date or multi-date (time series) mode
//
// Single date mode:
//   buildOTRCurve[bonds;bills;"D"$"2025.01.20";cusips!prices]
//   buildOTRCurve[bonds;bills;"D"$"2025.01.20";priceTable]
//
// Multi-date mode (asOfDate = (::), prices table must have date column):
//   buildOTRCurve[bonds;bills;(::);priceTable]
//   OTR securities are determined for EACH date from bonds/bills
//
// prices can be:
//   - dict: cusip -> price (single date only)
//   - table with `date`cusip`price columns (multi-date)
//   - table with `cusip`price columns (single date, uses asOfDate)
//
buildOTRCurve:{[bonds;bills;asOfDate;prices]
    // Check if prices is a table (98h) vs dict (99h)
    isTable:98h = type prices;

    // Determine price column name (only for tables)
    priceCol:$[not isTable; `price;
               `cleanPrice in cols prices; `cleanPrice;
               `price];

    // Multi-date mode: asOfDate is (::) and prices is table with date column
    if[(asOfDate ~ (::)) and isTable and (`date in cols prices);
        if[not `cusip in cols prices; '"prices table must have cusip column"];
        if[not priceCol in cols prices; '"prices table must have price or cleanPrice column"];
        dates:asc distinct prices`date;
        -1 "Building OTR curves for ",string[count dates]," dates...";
        result:raze {[bonds;bills;prices;priceCol;dt]
            px:select from prices where date = dt;
            priceDict:px[`cusip]!px priceCol;
            buildOTRCurveOne[bonds;bills;dt;priceDict]
        }[bonds;bills;prices;priceCol;] each dates;
        // Reorder columns: date first
        :(`date,cols[result] except `date) xcols result];

    // Single date mode
    if[asOfDate ~ (::); '"asOfDate required for single-date mode (or provide prices table with date column)"];

    // Normalize prices to dict
    priceDict:$[
        isTable;
        [
            // If table has date column, filter to asOfDate
            px:$[`date in cols prices;
                select from prices where date = asOfDate;
                prices];
            if[not priceCol in cols px; '"prices table must have price or cleanPrice column"];
            if[not `cusip in cols px; '"prices table must have cusip column"];
            px[`cusip]!px priceCol
        ];
        // Dict input - use directly
        prices
    ];

    buildOTRCurveOne[bonds;bills;asOfDate;priceDict]}

// Build swap-style curve from OTR yields
// Returns curve dict compatible with .swaps functions
otrToSwapCurve:{[otrTable;asOfDate]
    // Filter to securities with valid yields
    valid:select from otrTable where not null yield;
    if[0 = count valid; '"No valid yields in OTR table"];

    // Build curve
    tenors:valid`tenor;
    yearFracs:valid`remainingYears;
    yields:valid`yield;

    // Use swaps buildCurve with year fractions
    .swaps.buildCurve[yearFracs;yields;`asOfDate`frequency!(asOfDate;`6M)]}

// Convenience: load caches and build OTR curve
quickOTRCurve:{[asOfDate;prices]
    bonds:.tsy.loadBondCache[];
    bills:.tsy.loadBillCache[];
    otr:.tsy.buildOTRCurve[bonds;bills;asOfDate;prices];
    otr}

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
    // IMPORTANT: Use explicit parentheses due to Q's right-to-left evaluation
    dfn:1 % xexp[1+y;n];                          // discount factor for n periods
    factor:((c % y) * (1 - dfn)) + dfn;           // PV of coupons + principal
    factor:factor * (1 % xexp[1+y;z*2]);          // Adjust for fractional period
    factor:factor - (c * z);                       // Subtract accrued coupon

    // Round to 4 decimal places (exchange convention)
    0.0001 * `long$(factor * 10000)}

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
// DAILY PRICING SUPPORT
// =============================================================================

// Expected prices table schema:
// prices:([] date:`date$(); cusip:`symbol$(); price:`float$())
//
// Expected futures table schema:
// futures:([] date:`date$(); contract:`symbol$(); price:`float$())

// Join basket with prices for a specific date
// Returns ctd-format bonds with prices filled in
// basket: from genBasket
// prices: table with date, cusip, price columns
// asOfDate: date to get prices for
joinPricesForDate:{[basket;prices;asOfDate]
    // Get prices for this date
    dayPrices:exec cusip!price from prices where date=asOfDate;
    // Convert basket to ctd format with these prices
    toCtdBonds[basket;dayPrices]}

// Run CTD analysis for a single date
// Returns CTD analysis results table with date column added
// basket: from genBasket
// prices: table (date;cusip;price)
// futures: table (date;contract;price) or dictionary `contract`price!(...)
// curve: yield curve (or curveBuilder function)
// asOfDate: analysis date
// deliveryDate: futures delivery date
// repoRate: repo financing rate
ctdForDate:{[basket;prices;futures;curve;asOfDate;deliveryDate;repoRate]
    // Get bond prices for this date
    ctdBonds:joinPricesForDate[basket;prices;asOfDate];

    // Get futures price - handle both table and dict formats
    futuresPrice:$[99h=type futures;
        futures`price;                                    // dict format
        first exec price from futures where date=asOfDate // table format
    ];

    if[null futuresPrice; :([] date:(); sym:(); isCTD:())];  // No futures price

    // Run CTD analysis
    results:.ctd.ctdAnalysis[curve;asOfDate;deliveryDate;futuresPrice;repoRate;ctdBonds];

    // Add date column
    update date:asOfDate from results}

// Run CTD analysis across multiple dates
// Returns table with daily CTD results
// basket: from genBasket (static for the contract/delivery month)
// prices: table (date;cusip;price) - daily bond prices
// futures: table (date;contract;price) - daily futures prices
// curve: yield curve or curveBuilder:{[asOfDate] ...} function
// dates: list of dates to analyze
// deliveryDate: futures delivery date
// repoRate: repo rate (scalar or table with date column)
ctdTimeSeries:{[basket;prices;futures;curveOrBuilder;dates;deliveryDate;repoRate]
    // Determine if curve is static or needs rebuilding per date
    isBuilder:100h=type curveOrBuilder;  // function type

    // Handle repo rate - scalar or table
    getRepo:{[repoRate;dt]
        $[99h=type repoRate; repoRate`rate;              // dict with `rate
          98h=type repoRate; first exec rate from repoRate where date=dt;  // table
          repoRate]                                       // scalar
    }[repoRate;];

    // Run analysis for each date
    analyzeOne:{[basket;prices;futures;curveOrBuilder;isBuilder;deliveryDate;getRepo;dt]
        curve:$[isBuilder; curveOrBuilder[dt]; curveOrBuilder];
        repo:getRepo[dt];
        ctdForDate[basket;prices;futures;curve;dt;deliveryDate;repo]
    }[basket;prices;futures;curveOrBuilder;isBuilder;deliveryDate;getRepo;];

    raze analyzeOne each dates}

// Get CTD bond for each date (summary view)
// Returns table: date, ctdCusip, ctdCoupon, impliedRepo, grossBasis, netBasis
ctdSummary:{[ctdResults]
    select date,
           ctdCusip:sym,
           ctdCoupon:coupon,
           cleanPrice,
           impliedRepo,
           grossBasis,
           netBasis
    from ctdResults where isCTD}

// Detect CTD switches (when CTD changes from one bond to another)
ctdSwitches:{[ctdResults]
    summary:ctdSummary[ctdResults];
    summary:update prevCTD:prev ctdCusip from summary;
    select date, fromCTD:prevCTD, toCTD:ctdCusip from summary where ctdCusip<>prevCTD, not null prevCTD}

// =============================================================================
// MULTI-CONTRACT DAILY ANALYSIS
// =============================================================================

// Main entry point for daily CTD analysis across all contracts and delivery months
//
// Helper for dailyCTD - analyze one delivery code (defined at namespace level for peach visibility)
dailyCTDAnalyzeOne:{[ctx;curve;dayPrices;futPrices;repo;dt;dc]
    basket:ctx[`basketsByCode][dc];
    if[0=count basket; :()];

    if[not dc in key futPrices; :()];
    futPrice:futPrices dc;
    if[null futPrice; :()];

    ctdBonds:toCtdBonds[basket;dayPrices];
    if[0=count ctdBonds; :()];

    deliveryDate:ctx[`deliveryDateByCode][dc];
    // Use vectorized analysis if available
    results:$[`ctdAnalysisVec in key `.ctd;
        .ctd.ctdAnalysisVec[curve;dt;deliveryDate;futPrice;repo;ctdBonds];
        .ctd.ctdAnalysis[curve;dt;deliveryDate;futPrice;repo;ctdBonds]];

    update date:dt, deliveryCode:dc, contract:ctx[`contractByCode][dc], deliveryMonth:ctx[`monthByCode][dc] from results}

// Inputs:
//   baskets: from allBasketsRange[] - table with deliveryCode, contract, deliveryMonth, cusip, cf, etc.
//   prices: ([] date; cusip; price) - daily bond prices
//   futures: ([] date; contract; deliveryMonth; price) - daily futures prices
//            OR ([] date; deliveryCode; price) where deliveryCode = e.g. `TYH25
//   curves: one of:
//           - static curve dict (used for all dates)
//           - builder function {[dt] ...} returning curve for date
//           - table ([] date; curve) with pre-built curves per date
//   repoRate: scalar OR table ([] date; rate)
//   useLastDelivery: optional, 0b (default) = first delivery day, 1b = last delivery day
//                   `both or 2 = run both and combine results with deliveryType column
//
// Returns: table with date, deliveryCode, contract, deliveryMonth, + all CTD analysis columns
//          If `both: adds deliveryType (`first/`last), deliveryDate columns, isCTD is per-scenario
dailyCTD:{[baskets;prices;futures;curves;repoRate;useLastDelivery]
    // Handle `both option - run both first and last, combine results
    if[(useLastDelivery~`both) or (useLastDelivery~2);
        r1:dailyCTD[baskets;prices;futures;curves;repoRate;0b];
        r2:dailyCTD[baskets;prices;futures;curves;repoRate;1b];
        // Add delivery type and date columns
        r1:update deliveryType:`first from r1;
        r2:update deliveryType:`last from r2;
        combined:r1,r2;
        // Re-determine CTD across both scenarios per (date, deliveryCode)
        :update isCTDBoth:(impliedRepo = max impliedRepo) by date, deliveryCode from
            `date`deliveryCode`deliveryType xasc combined
    ];

    // Handle optional useLastDelivery parameter (default to first delivery day)
    useLast:$[useLastDelivery~(::); 0b; useLastDelivery];
    deliveryFn:$[useLast; lastDeliveryDay; firstDeliveryDay];

    // Get unique dates from prices
    dates:asc distinct prices`date;
    nDates:count dates;

    // Get unique delivery codes (contract + month combinations)
    deliveryCodes:distinct baskets`deliveryCode;

    // === OPTIMIZATION 1: Pre-build curves by date (not per combo) ===
    curveType:$[99h=type curves; `dict;
                (type curves) in 100 104h; `builder;
                98h=type curves; `table;
                `dict];

    curvesByDate:$[curveType=`dict; dates!nDates#enlist curves;
                   curveType=`builder; dates!curves each dates;
                   curveType=`table; dates!(exec date!curve from curves) dates;
                   dates!nDates#enlist curves];

    // === OPTIMIZATION 2: Pre-index prices and futures by date ===
    pricesByDate:dates!{[p;dt] exec cusip!price from p where date=dt}[prices;] each dates;

    hasDC:`deliveryCode in cols futures;
    futuresByDate:$[hasDC;
        dates!{[f;dt] exec deliveryCode!price from f where date=dt}[futures;] each dates;
        dates!{[f;dt] exec deliveryCode!price from update deliveryCode:`$string[contract],'string[deliveryMonth] from f where date=dt}[futures;] each dates];

    // === OPTIMIZATION 3: Pre-compute repo by date ===
    repoByDate:$[-9h=type repoRate; dates!nDates#repoRate;
                 99h=type repoRate; dates!nDates#repoRate`rate;
                 98h=type repoRate; dates!(exec date!rate from repoRate) dates;
                 dates!nDates#repoRate];

    // === OPTIMIZATION 4: Pre-filter baskets by code (avoid repeated filtering) ===
    basketsByCode:deliveryCodes!{[b;dc] select from b where deliveryCode=dc}[baskets;] each deliveryCodes;

    // === OPTIMIZATION 5: Pre-compute delivery dates by code ===
    deliveryDateByCode:deliveryCodes!{[bc;dc;fn] fn first bc[dc]`deliveryMonth}[basketsByCode;;deliveryFn] each deliveryCodes;
    contractByCode:deliveryCodes!{[bc;dc] first bc[dc]`contract}[basketsByCode;] each deliveryCodes;
    monthByCode:deliveryCodes!{[bc;dc] first bc[dc]`deliveryMonth}[basketsByCode;] each deliveryCodes;

    // Pack context
    ctx:`basketsByCode`pricesByDate`futuresByDate`curvesByDate`repoByDate`deliveryDateByCode`contractByCode`monthByCode!
        (basketsByCode;pricesByDate;futuresByDate;curvesByDate;repoByDate;deliveryDateByCode;contractByCode;monthByCode);

    // Process by date (sequential), parallelize across delivery codes within each date
    analyzeDate:{[ctx;deliveryCodes;dt]
        curve:ctx[`curvesByDate][dt];
        if[not 99h=type curve; :()];
        repo:ctx[`repoByDate][dt];
        if[null repo; :()];
        dayPrices:ctx[`pricesByDate][dt];
        futPrices:ctx[`futuresByDate][dt];
        // Parallel across delivery codes for this date
        raze dailyCTDAnalyzeOne[ctx;curve;dayPrices;futPrices;repo;dt;] peach deliveryCodes
    }[ctx;deliveryCodes;];

    // Run analysis: sequential over dates, parallel over delivery codes
    results:raze analyzeDate each dates;

    if[0=count results; :([] date:`date$(); deliveryCode:`symbol$(); contract:`symbol$(); deliveryMonth:`month$())];

    `date`contract`deliveryMonth xasc results}

// Summary view: CTD for each deliveryCode/date
ctdDailySummary:{[ctdResults]
    select date,
           deliveryCode,
           contract,
           deliveryMonth,
           ctdCusip:sym,
           ctdCoupon:coupon,
           cleanPrice,
           impliedRepo,
           grossBasis,
           netBasis
    from ctdResults where isCTD}

// Detect CTD switches by delivery code
ctdDailySwitches:{[ctdResults]
    summary:ctdDailySummary[ctdResults];
    summary:update prevCTD:prev ctdCusip by deliveryCode from summary;
    select date, deliveryCode, contract, deliveryMonth, fromCTD:prevCTD, toCTD:ctdCusip
    from summary
    where ctdCusip<>prevCTD, not null prevCTD}

// =============================================================================
// DAILY QUALITY BREAKEVEN ANALYSIS
// =============================================================================

// Helper: analyze one date/deliveryCode combination for quality breakeven
dailyQBAnalyzeOne:{[ctx;curve;dayPrices;futPrices;repo;dt;dc]
    basket:ctx[`basketsByCode][dc];
    if[0=count basket; :()];

    if[not dc in key futPrices; :()];
    futPrice:futPrices dc;
    if[null futPrice; :()];

    ctdBonds:toCtdBonds[basket;dayPrices];
    if[0=count ctdBonds; :()];

    deliveryDate:ctx[`deliveryDateByCode][dc];

    // Run quality breakeven analysis
    be:.ctd.qualityBreakeven[curve;dt;deliveryDate;futPrice;repo;ctdBonds];

    // Flatten switch points into result - return as single-row table for proper raze behavior
    nSwitches:count be`switchPoints;
    switchUp:$[null be`breakevenUp; 0Ni; be`breakevenUp];
    switchDown:$[null be`breakevenDown; 0Ni; be`breakevenDown];

    // Return as single-row table (enlist dict) so raze works correctly
    enlist `date`deliveryCode`contract`deliveryMonth`currentCTD`breakevenUp`breakevenDown`nSwitchPoints!(
        dt;dc;ctx[`contractByCode][dc];ctx[`monthByCode][dc];
        be`currentCTD;switchUp;switchDown;nSwitches)}

// Daily quality breakeven analysis across all contracts and dates
// Same inputs as dailyCTD
// Returns: table with date, deliveryCode, currentCTD, breakevenUp, breakevenDown, nSwitchPoints
dailyQualityBreakeven:{[baskets;prices;futures;curves;repoRate;useLastDelivery]
    // Handle `both option - run both first and last, combine results
    if[(useLastDelivery~`both) or (useLastDelivery~2);
        r1:dailyQualityBreakeven[baskets;prices;futures;curves;repoRate;0b];
        r2:dailyQualityBreakeven[baskets;prices;futures;curves;repoRate;1b];
        r1:update deliveryType:`first from r1;
        r2:update deliveryType:`last from r2;
        :`date`deliveryCode`deliveryType xasc r1,r2
    ];

    // Handle optional useLastDelivery parameter
    useLast:$[useLastDelivery~(::); 0b; useLastDelivery];
    deliveryFn:$[useLast; lastDeliveryDay; firstDeliveryDay];

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

    // Process by date
    analyzeDate:{[ctx;deliveryCodes;dt]
        curve:ctx[`curvesByDate][dt];
        if[not 99h=type curve; :()];
        repo:ctx[`repoByDate][dt];
        if[null repo; :()];
        dayPrices:ctx[`pricesByDate][dt];
        futPrices:ctx[`futuresByDate][dt];
        raze dailyQBAnalyzeOne[ctx;curve;dayPrices;futPrices;repo;dt;] each deliveryCodes
    }[ctx;deliveryCodes;];

    // Run analysis - raze tables from each date
    results:raze analyzeDate each dates;

    // Handle empty results
    if[0=count results;
        :([] date:`date$(); deliveryCode:`symbol$(); contract:`symbol$();
           deliveryMonth:`month$(); currentCTD:`symbol$();
           breakevenUp:`int$(); breakevenDown:`int$(); nSwitchPoints:`int$())];

    `date`contract`deliveryMonth xasc results}

// Convenience: build baskets + run analysis
// For when you don't have pre-built baskets
runDailyCTD:{[bonds;startYear;endYear;prices;futures;curves;repoRate;useLastDelivery]
    baskets:allBasketsRange[bonds;startYear;endYear];
    dailyCTD[baskets;prices;futures;curves;repoRate;useLastDelivery]}

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

    -1 "4. SINGLE-DAY CTD ANALYSIS";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // Get basket and add market prices";
    -1 "   basket:.tsy.genBasket[bonds;`TY;2025.03m]";
    -1 "   prices:basket[`cusip]!100+0.1*til count basket  // Mock prices";
    -1 "   ctdBonds:.tsy.toCtdBonds[basket;prices]";
    -1 "";
    -1 "   // Run CTD analysis (requires ctd.q)";
    -1 "   // .ctd.ctdAnalysis[curve;settleDate;deliveryDate;futuresPrice;repoRate;ctdBonds]";
    -1 "";

    -1 "5. DAILY CTD ANALYSIS (MULTI-CONTRACT, MULTI-DAY)";
    -1 "-----------------------------------------------------------------------------";
    -1 "   // Build baskets for all contracts/months";
    -1 "   baskets:.tsy.allBasketsRange[bonds;2018;2025]";
    -1 "";
    -1 "   // Prepare daily price tables";
    -1 "   // prices:([] date; cusip; price) - daily bond prices";
    -1 "   // futures:([] date; deliveryCode; price) - daily futures prices";
    -1 "";
    -1 "   // Curves: static dict, builder fn, or pre-built table";
    -1 "   curveBuilder:{[dt] .swaps.buildCurve[tenors;rates;`asOfDate`frequency!(dt;`6M)]}";
    -1 "";
    -1 "   // Run daily analysis (~2,200 combos/sec)";
    -1 "   results:.tsy.dailyCTD[baskets;prices;futures;curveBuilder;0.045]";
    -1 "";
    -1 "   // View results";
    -1 "   .tsy.ctdDailySummary[results]     // CTD for each deliveryCode/date";
    -1 "   .tsy.ctdDailySwitches[results]    // Dates where CTD changed";
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
    -1 ".tsy.allBasketsRange[bonds;startYear;endYear]  // All baskets for year range";
    -1 ".tsy.buildDeliverableBaskets[startYear]        // Full pipeline";
    -1 "";
    -1 "// DAILY CTD ANALYSIS (multi-contract, multi-day)";
    -1 ".tsy.dailyCTD[baskets;prices;futures;curves;repo]  // Main entry point";
    -1 ".tsy.ctdDailySummary[results]          // CTD for each deliveryCode/date";
    -1 ".tsy.ctdDailySwitches[results]         // Dates where CTD changed";
    -1 "";
    -1 "// Input formats:";
    -1 "//   baskets: from allBasketsRange[]";
    -1 "//   prices:  ([] date; cusip; price)";
    -1 "//   futures: ([] date; deliveryCode; price)";
    -1 "//   curves:  static dict | builder fn | ([] date; curve)";
    -1 "//   repo:    scalar | ([] date; rate)";
    -1 "";
    -1 "// BASKET ANALYSIS";
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
    -1 "DATA LOADING (Notes & Bonds):";
    -1 "  fetchAllSecurities[]          - Fetch Notes & Bonds from TreasuryDirect API";
    -1 "  fetchNotes[]                  - Fetch Notes only";
    -1 "  fetchBonds[]                  - Fetch Bonds only";
    -1 "  saveBondCache[bonds]          - Save bonds to CSV cache";
    -1 "  loadBondCache[]               - Load bonds from CSV cache";
    -1 "  loadBondsFromCSV[filepath]    - Load bonds from custom CSV";
    -1 "  bondsAsOf[bonds;asOfDate]     - Filter bonds outstanding as of date";
    -1 "";
    -1 "T-BILL LOADING (separate from bonds - NOT used in CTD/basket logic):";
    -1 "  fetchBills[]                  - Fetch recent T-Bills";
    -1 "  fetchAllBills[startYear]      - Fetch T-Bills from startYear to present";
    -1 "  saveBillCache[bills]          - Save bills to CSV cache";
    -1 "  loadBillCache[]               - Load bills from CSV cache";
    -1 "  buildBillCache[startDate;endDate] - Fetch bills in date range and cache";
    -1 "     Examples: buildBillCache[2020;(::)]  - year 2020 to present";
    -1 "              buildBillCache[\"D\"$\"2020-01-01\";\"D\"$\"2024-12-31\"]  - date range";
    -1 "";
    -1 "ON-THE-RUN TREASURY CURVE:";
    -1 "  findOnTheRun[bonds;bills;asOfDate]  - Find OTR securities";
    -1 "  buildOTRCurve[bonds;bills;asOfDate;prices] - Build OTR yield table";
    -1 "     Single date: asOfDate + dict OR table with cusip/price";
    -1 "     Multi-date:  asOfDate=(::) + table with date/cusip/price";
    -1 "                  OTR determined from bonds/bills for EACH date";
    -1 "  otrToSwapCurve[otrTable;asOfDate]   - Convert OTR to swap curve";
    -1 "  quickOTRCurve[asOfDate;prices]      - Convenience (loads caches)";
    -1 "  billBEY[discRate;days]              - Discount rate to bond-equiv yield";
    -1 "  billYieldFromPrice[price;days]      - Price to yield";
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
    -1 "BASKET ANALYSIS:";
    -1 "  showBasket[basket]            - Display single basket with details";
    -1 "  showAllBaskets[bonds;deliveryMonth] - Display ALL 5 contracts for month";
    -1 "  allBasketsSummary[bonds;deliveryMonth] - Summary table for all contracts";
    -1 "  basketStats[basket]           - Summary statistics for one basket";
    -1 "  basketSummary[baskets]        - Cross-month comparison table";
    -1 "";
    -1 "DAILY CTD ANALYSIS:";
    -1 "  dailyCTD[baskets;prices;futures;curves;repo] - Multi-contract, multi-day CTD";
    -1 "  ctdDailySummary[results]      - CTD for each deliveryCode/date";
    -1 "  ctdDailySwitches[results]     - Dates where CTD changed";
    -1 "";
    -1 "  Input formats:";
    -1 "    baskets: from allBasketsRange[]";
    -1 "    prices:  ([] date; cusip; price)";
    -1 "    futures: ([] date; deliveryCode; price)";
    -1 "    curves:  static dict | builder fn | ([] date; curve)";
    -1 "    repo:    scalar | ([] date; rate)";
    -1 "";
    -1 "  Performance: ~2,200 combos/sec (~15-25 sec for 7 years)";
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
