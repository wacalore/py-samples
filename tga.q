// =============================================================================
// TGA (TREASURY GENERAL ACCOUNT) LIBRARY
// =============================================================================
// Fetch, analyze, and forecast Treasury General Account data
// Version: 0.1.0

\d .tga

version:"0.1.0"

// =============================================================================
// CONFIGURATION
// =============================================================================

// Fiscal Data API (no authentication required)
fiscalDataUrl:"https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

// FRED API (requires API key)
fredUrl:"https://api.stlouisfed.org/fred/series/observations"
fredApiKey:""

// Cache configuration
cacheDir:"/tmp/"
tgaCacheFile:"tga_balance.csv"
dtsCacheFile:"tga_dts.csv"

// Set FRED API key (optional - only needed for FRED data source)
setFredKey:{[key] fredApiKey::key; -1 "FRED API key set"}

// =============================================================================
// DATE UTILITIES
// =============================================================================

// Convert q date to API format (YYYY-MM-DD)
formatDateApi:{[d] ssr[string d;".";"-"]}

// Parse API date string to q date
parseDate:{[s] "D"$ssr[s;"-";"."] }

// =============================================================================
// HTTP UTILITIES
// =============================================================================

// HTTP GET with curl
httpGet:{[url]
    cmd:"curl -sLk --connect-timeout 30 \"",url,"\"";
    resp:@[system;cmd;{[e] '"HTTP error: ",e}];
    if[0 = count resp; '"Empty response from ",url];
    "" sv resp}

// =============================================================================
// FISCAL DATA API
// =============================================================================

// Generic Fiscal Data API fetch with pagination
// endpoint - API endpoint (e.g., "/v1/accounting/dts/operating_cash_balance")
// params   - dict of query parameters
// Returns: list of data records
fiscalDataFetch:{[endpoint;params]
    // Build URL with parameters (URL encode brackets)
    baseUrl:fiscalDataUrl,endpoint;
    paramStrs:{[k;v] (string k),"=",$[10h = type v; v; string v]}'[key params;value params];
    url:baseUrl,"?","&" sv paramStrs;
    url:url,"&format=json&page%5Bsize%5D=10000";  // URL encoded [size]

    // Fetch and parse
    resp:httpGet url;
    parsed:.j.k resp;

    // Extract data and metadata
    records:parsed`data;
    totalPages:parsed[`meta;`$"total-pages"];

    // Fetch remaining pages if any
    if[(not null totalPages) and totalPages > 1;
        pages:2 + til totalPages - 1;
        fetchPage:{[url;p] resp:httpGet url,"&page%5Bnumber%5D=",string p; (.j.k resp)`data};
        moreRecords:raze fetchPage[url;] each pages;
        records:records,moreRecords];

    records}

// =============================================================================
// TGA BALANCE FETCHING
// =============================================================================

// Fetch TGA balance from Fiscal Data API
// startDate, endDate - date range (q dates or YYYY-MM-DD strings)
// Returns: table with date, balance
// Note: API changed April 2022 - closing balance moved to open_today_bal column for "TGA Closing Balance" rows
fetchTGA:{[startDate;endDate]
    // Convert dates to API format
    sd:$[-14h = type startDate; formatDateApi startDate; startDate];
    ed:$[-14h = type endDate; formatDateApi endDate; endDate];

    // Fetch operating cash balance
    params:`filter`sort!("record_date:gte:",sd,",record_date:lte:",ed;"-record_date");
    -1 "Fetching TGA balance from ",sd," to ",ed,"...";

    data:fiscalDataFetch["/v1/accounting/dts/operating_cash_balance";params];

    if[0 = count data; -1 "No data returned"; :([] date:`date$(); balance:`float$())];

    // For recent data (post April 2022): use "TGA Closing Balance" rows, value in open_today_bal
    acctTypes:data[;`account_type];
    // Use ss (substring search) instead of like for compatibility
    tgaClosingMask:{(0 < count x ss "TGA") and 0 < count x ss "Closing"} each acctTypes;
    tgaClosing:data where tgaClosingMask;

    if[0 < count tgaClosing;
        dates:parseDate each tgaClosing[;`record_date];
        // The balance is in open_today_bal for recent data (confusingly named)
        balances:"F"$tgaClosing[;`open_today_bal];
        result:([] date:dates; balance:balances);
        result:`date xasc distinct result;
        -1 "Fetched ",string[count result]," TGA records";
        :result];

    // For older data: use "Federal Reserve Account" rows, value in close_today_bal
    fraMask:{0 < count x ss "Federal Reserve Account"} each acctTypes;
    fraData:data where fraMask;

    if[0 < count fraData;
        dates:parseDate each fraData[;`record_date];
        balances:"F"$fraData[;`close_today_bal];
        result:([] date:dates; balance:balances);
        result:`date xasc distinct result;
        -1 "Fetched ",string[count result]," TGA records (legacy FRA format)";
        :result];

    -1 "No TGA data found in response";
    ([] date:`date$(); balance:`float$())}

// Fetch TGA from FRED (requires API key)
fetchTGAFred:{[startDate;endDate]
    if[0 = count fredApiKey; '"FRED API key not set. Use .tga.setFredKey[\"your-key\"]"];

    sd:$[-14h = type startDate; formatDateApi startDate; startDate];
    ed:$[-14h = type endDate; formatDateApi endDate; endDate];

    // WDTGAL = Wednesday Level TGA
    url:fredUrl,"?series_id=WDTGAL&api_key=",fredApiKey,"&file_type=json";
    url:url,"&observation_start=",sd,"&observation_end=",ed;

    -1 "Fetching TGA from FRED...";
    resp:httpGet url;
    data:.j.k resp;

    obs:data`observations;
    if[0 = count obs; -1 "No data returned"; :()];

    dates:parseDate each obs[;`date];
    values:"F"$obs[;`value];

    result:([] date:dates; balance:values; source:`fred);
    result:`date xasc result;

    -1 "Fetched ",string[count result]," FRED records";
    result}

// =============================================================================
// DTS (DAILY TREASURY STATEMENT) FETCHING
// =============================================================================

// Fetch DTS deposits (tax receipts and other income)
fetchDTSDeposits:{[startDate;endDate]
    sd:$[-14h = type startDate; formatDateApi startDate; startDate];
    ed:$[-14h = type endDate; formatDateApi endDate; endDate];

    params:`filter`sort!("record_date:gte:",sd,",record_date:lte:",ed;"-record_date");
    -1 "Fetching DTS deposits from ",sd," to ",ed,"...";

    data:fiscalDataFetch["/v1/accounting/dts/deposits_withdrawals_operating_cash";params];

    if[0 = count data; -1 "No data returned"; :()];

    // Filter for deposits (positive transaction types)
    txnTypes:data[;`transaction_type];
    deposits:data where {0 < count x ss "Deposit"} each txnTypes;

    // Parse fields
    dates:parseDate each deposits[;`record_date];
    txnTypes:`$deposits[;`transaction_type];
    todayAmt:"F"$deposits[;`transaction_today_amt];
    mtdAmt:"F"$deposits[;`transaction_mtd_amt];
    ytdAmt:"F"$deposits[;`transaction_ytd_amt];

    result:([] date:dates; category:txnTypes; todayAmt:todayAmt; mtdAmt:mtdAmt; ytdAmt:ytdAmt);
    result:`date xasc result;

    -1 "Fetched ",string[count result]," deposit records";
    result}

// Fetch DTS withdrawals (spending)
fetchDTSWithdrawals:{[startDate;endDate]
    sd:$[-14h = type startDate; formatDateApi startDate; startDate];
    ed:$[-14h = type endDate; formatDateApi endDate; endDate];

    params:`filter`sort!("record_date:gte:",sd,",record_date:lte:",ed;"-record_date");
    -1 "Fetching DTS withdrawals from ",sd," to ",ed,"...";

    data:fiscalDataFetch["/v1/accounting/dts/deposits_withdrawals_operating_cash";params];

    if[0 = count data; -1 "No data returned"; :()];

    // Filter for withdrawals (use ss instead of like for compatibility)
    txnTypes:data[;`transaction_type];
    withdrawals:data where {0 < count x ss "Withdraw"} each txnTypes;

    dates:parseDate each withdrawals[;`record_date];
    txnTypes:`$withdrawals[;`transaction_type];
    todayAmt:"F"$withdrawals[;`transaction_today_amt];
    mtdAmt:"F"$withdrawals[;`transaction_mtd_amt];
    ytdAmt:"F"$withdrawals[;`transaction_ytd_amt];

    result:([] date:dates; category:txnTypes; todayAmt:todayAmt; mtdAmt:mtdAmt; ytdAmt:ytdAmt);
    result:`date xasc result;

    -1 "Fetched ",string[count result]," withdrawal records";
    result}

// Fetch public debt transactions
fetchDTSDebt:{[startDate;endDate]
    sd:$[-14h = type startDate; formatDateApi startDate; startDate];
    ed:$[-14h = type endDate; formatDateApi endDate; endDate];

    params:`filter`sort!("record_date:gte:",sd,",record_date:lte:",ed;"-record_date");
    -1 "Fetching DTS debt transactions from ",sd," to ",ed,"...";

    data:fiscalDataFetch["/v1/accounting/dts/public_debt_transactions";params];

    if[0 = count data; -1 "No data returned"; :()];

    dates:parseDate each data[;`record_date];
    txnTypes:`$data[;`transaction_type];
    todayAmt:"F"$data[;`transaction_today_amt];
    mtdAmt:"F"$data[;`transaction_mtd_amt];

    result:([] date:dates; category:txnTypes; todayAmt:todayAmt; mtdAmt:mtdAmt);
    result:`date xasc result;

    -1 "Fetched ",string[count result]," debt transaction records";
    result}

// =============================================================================
// CACHING
// =============================================================================

// Save TGA data to cache
saveTGACache:{[data]
    filepath:cacheDir,tgaCacheFile;
    path:`$":",filepath;
    (path) 0: csv 0: data;
    -1 "Saved ",string[count data]," TGA records to ",filepath;
    data}

// Load TGA data from cache
loadTGACache:{[]
    filepath:cacheDir,tgaCacheFile;
    path:`$":",filepath;
    if[()~key path; '"TGA cache not found at ",filepath,". Run .tga.fetchTGA[] first"];
    data:("DFFF";enlist csv) 0: path;
    -1 "Loaded ",string[count data]," TGA records from cache";
    data}

// Save DTS data to cache
saveDTSCache:{[data]
    filepath:cacheDir,dtsCacheFile;
    path:`$":",filepath;
    (path) 0: csv 0: data;
    -1 "Saved ",string[count data]," DTS records to ",filepath;
    data}

// Load DTS data from cache
loadDTSCache:{[]
    filepath:cacheDir,dtsCacheFile;
    path:`$":",filepath;
    if[()~key path; '"DTS cache not found at ",filepath];
    data:("DSFFF";enlist csv) 0: path;
    -1 "Loaded ",string[count data]," DTS records from cache";
    data}

// =============================================================================
// ANALYSIS FUNCTIONS
// =============================================================================

// Calculate daily TGA changes
tgaChange:{[data]
    update change:balance - prev balance, pctChange:100*(balance - prev balance) % prev balance from data}

// Rolling z-score of TGA balance
tgaZscore:{[data;window]
    bal:data`balance;
    mu:mavg[window;bal];
    sigma:mdev[window;bal];
    zscore:(bal - mu) % sigma;
    update zscore:zscore from data}

// TGA drawdown from rolling peak
tgaDrawdown:{[data;window]
    bal:data`balance;
    peak:(window-1) mmax bal;
    dd:(bal - peak) % peak;
    update drawdown:dd, peak:peak from data}

// Tax receipt seasonality by month
taxSeasonality:{[receipts;nYears]
    // Aggregate by month
    byMonth:select totalReceipts:sum todayAmt by month:`mm$date from receipts;

    // Calculate average and std
    avgByMonth:select avgReceipts:avg totalReceipts, stdReceipts:dev totalReceipts by month from byMonth;

    // Percentage of annual
    annualAvg:avg avgByMonth`avgReceipts;
    update pctOfAnnual:avgReceipts % 12 * annualAvg from avgByMonth}

// Receipts breakdown by category
receiptsByCategory:{[deposits]
    select totalAmt:sum todayAmt, avgDaily:avg todayAmt, cnt:count i by category from deposits}

// Net borrowing (issuance - redemptions)
netBorrowing:{[debtTxns]
    // Separate issues and redemptions (use ss instead of like for compatibility)
    cats:string each debtTxns`category;
    issueMask:{0 < count x ss "Issue"} each cats;
    redemptionMask:{0 < count x ss "Redemption"} each cats;
    issues:select issued:sum todayAmt by date from debtTxns where issueMask;
    redemptions:select redeemed:sum todayAmt by date from debtTxns where redemptionMask;

    // Join and calculate net
    combined:0!issues lj redemptions;
    update netBorrowing:issued - redeemed from combined}

// =============================================================================
// FORECASTING
// =============================================================================

// Simple moving average forecast
tgaForecastMA:{[data;window;horizon]
    bal:data`balance;
    lastVal:last bal;
    mu:avg (neg window)#bal;
    trend:(lastVal - mu) % window;

    // Project forward
    futureDates:(last data`date) + 1 + til horizon;
    futureBal:lastVal + trend * 1 + til horizon;

    ([] date:futureDates; forecast:futureBal; method:`ma)}

// Seasonal forecast based on historical patterns
tgaForecastSeasonal:{[data;horizon]
    // Calculate average change by day of week
    dataWithDow:update dow:date mod 7 from tgaChange data;
    avgByDow:exec avg change by dow from dataWithDow;

    // Project forward
    lastDate:last data`date;
    lastBal:last data`balance;
    futureDates:lastDate + 1 + til horizon;
    futureDow:futureDates mod 7;
    dailyChanges:avgByDow futureDow;
    futureBal:lastBal + sums dailyChanges;

    ([] date:futureDates; forecast:futureBal; method:`seasonal)}

// =============================================================================
// DISPLAY FUNCTIONS
// =============================================================================

// Show TGA summary
showTGA:{[data]
    if[0 = count data; -1 "No data"; :()];

    -1 "";
    -1 "=== TGA BALANCE SUMMARY ===";
    -1 "";
    -1 "Date range: ",string[min data`date]," to ",string max data`date;
    -1 "Records: ",string count data;
    -1 "";
    -1 "Current balance: $",string[0.001*last data`balance],"B";
    -1 "Min balance:     $",string[0.001*min data`balance],"B";
    -1 "Max balance:     $",string[0.001*max data`balance],"B";
    -1 "Avg balance:     $",string[0.001*avg data`balance],"B";
    -1 "";

    // Recent values
    -1 "Recent values:";
    show (neg 5)#`date`balance xcols data;
    -1 "";
    data}

// Show TGA with z-score
showTGAZscore:{[data;window]
    dataZ:tgaZscore[data;window];
    -1 "";
    -1 "=== TGA Z-SCORE (window=",string[window],") ===";
    -1 "";
    -1 "Current z-score: ",string last dataZ`zscore;
    -1 "";
    show (neg 10)#`date`balance`zscore xcols dataZ;
    -1 "";
    dataZ}

// Show seasonality
showSeasonality:{[seasonal]
    -1 "";
    -1 "=== TAX RECEIPT SEASONALITY ===";
    -1 "";
    -1 "Month  AvgReceipts  StdReceipts  PctOfAnnual";
    -1 "-----  -----------  -----------  -----------";
    {-1 (2$string x`month),"     ",(11$string `int$x`avgReceipts),"  ",(11$string `int$x`stdReceipts),"  ",(string `int$100*x`pctOfAnnual),"%"} each seasonal;
    -1 "";
    seasonal}

// TGA statistics summary
tgaSummary:{[data]
    dataC:tgaChange data;
    `currentBal`minBal`maxBal`avgBal`stdBal`avgChange`stdChange`minChange`maxChange!(
        last data`balance;
        min data`balance;
        max data`balance;
        avg data`balance;
        dev data`balance;
        avg dataC`change;
        dev dataC`change;
        min dataC`change;
        max dataC`change)}

// =============================================================================
// HELP & EXAMPLES
// =============================================================================

usage:{[]
    -1 "";
    -1 "=============================================================================";
    -1 "                       .tga USAGE REFERENCE";
    -1 "=============================================================================";
    -1 "";
    -1 "// FETCH DATA (Fiscal Data API - no key needed)";
    -1 ".tga.fetchTGA[startDate;endDate]        // TGA balance history";
    -1 ".tga.fetchDTSDeposits[startDate;endDate] // Tax receipts";
    -1 ".tga.fetchDTSWithdrawals[startDate;endDate] // Spending";
    -1 ".tga.fetchDTSDebt[startDate;endDate]    // Debt transactions";
    -1 "";
    -1 "// FETCH DATA (FRED - requires API key)";
    -1 ".tga.setFredKey[\"your-api-key\"]";
    -1 ".tga.fetchTGAFred[startDate;endDate]";
    -1 "";
    -1 "// CACHING";
    -1 ".tga.saveTGACache[data]                 // Save to CSV";
    -1 ".tga.loadTGACache[]                     // Load from CSV";
    -1 "";
    -1 "// ANALYSIS";
    -1 ".tga.tgaChange[data]                    // Daily changes";
    -1 ".tga.tgaZscore[data;window]             // Rolling z-score";
    -1 ".tga.tgaDrawdown[data;window]           // Drawdown from peak";
    -1 ".tga.taxSeasonality[receipts;nYears]    // Seasonal patterns";
    -1 ".tga.receiptsByCategory[deposits]       // Category breakdown";
    -1 ".tga.netBorrowing[debtTxns]             // Net issuance";
    -1 "";
    -1 "// FORECASTING";
    -1 ".tga.tgaForecastMA[data;window;horizon] // Moving average";
    -1 ".tga.tgaForecastSeasonal[data;horizon]  // Seasonal projection";
    -1 "";
    -1 "// DISPLAY";
    -1 ".tga.showTGA[data]                      // TGA summary";
    -1 ".tga.showTGAZscore[data;window]         // With z-score";
    -1 ".tga.showSeasonality[seasonal]          // Seasonal factors";
    -1 ".tga.tgaSummary[data]                   // Statistics dict";
    -1 "";
    -1 "=============================================================================";
    -1 ""}

example:{[]
    -1 "";
    -1 "=== TGA LIBRARY EXAMPLE ===";
    -1 "";
    -1 "// 1. Fetch TGA balance (last 2 years)";
    -1 "tga:.tga.fetchTGA[.z.d - 730; .z.d]";
    -1 "";
    -1 "// 2. Show summary";
    -1 ".tga.showTGA tga";
    -1 "";
    -1 "// 3. Calculate z-score";
    -1 ".tga.showTGAZscore[tga;60]";
    -1 "";
    -1 "// 4. Forecast 30 days ahead";
    -1 "forecast:.tga.tgaForecastMA[tga;20;30]";
    -1 "show forecast";
    -1 "";
    -1 "// 5. Save to cache";
    -1 ".tga.saveTGACache tga";
    -1 "";
    -1 "// 6. Load from cache later";
    -1 "tga:.tga.loadTGACache[]";
    -1 "";
    -1 "=============================================================================";
    -1 ""}

help:{[]
    -1 "";
    -1 "=== .tga TGA LIBRARY v",version," ===";
    -1 "";
    -1 "DATA FETCHING:";
    -1 "  fetchTGA[sd;ed]           - TGA balance from Fiscal Data API";
    -1 "  fetchTGAFred[sd;ed]       - TGA balance from FRED (needs API key)";
    -1 "  fetchDTSDeposits[sd;ed]   - Tax receipts and deposits";
    -1 "  fetchDTSWithdrawals[sd;ed]- Government spending";
    -1 "  fetchDTSDebt[sd;ed]       - Public debt transactions";
    -1 "";
    -1 "CACHING:";
    -1 "  saveTGACache[data]        - Save TGA to CSV";
    -1 "  loadTGACache[]            - Load TGA from CSV";
    -1 "  saveDTSCache[data]        - Save DTS to CSV";
    -1 "  loadDTSCache[]            - Load DTS from CSV";
    -1 "";
    -1 "ANALYSIS:";
    -1 "  tgaChange[data]           - Calculate daily changes";
    -1 "  tgaZscore[data;w]         - Rolling z-score";
    -1 "  tgaDrawdown[data;w]       - Drawdown from rolling peak";
    -1 "  taxSeasonality[rcpts;n]   - Monthly seasonal factors";
    -1 "  receiptsByCategory[deps]  - Breakdown by tax type";
    -1 "  netBorrowing[debt]        - Net issuance (issues - redemptions)";
    -1 "";
    -1 "FORECASTING:";
    -1 "  tgaForecastMA[data;w;h]   - Moving average forecast";
    -1 "  tgaForecastSeasonal[data;h]- Seasonal projection";
    -1 "";
    -1 "DISPLAY:";
    -1 "  showTGA[data]             - Summary display";
    -1 "  showTGAZscore[data;w]     - Display with z-score";
    -1 "  showSeasonality[s]        - Seasonal factor display";
    -1 "  tgaSummary[data]          - Statistics dictionary";
    -1 "";
    -1 "CONFIGURATION:";
    -1 "  setFredKey[key]           - Set FRED API key";
    -1 "";
    -1 "Run .tga.usage[] for quick reference";
    -1 "Run .tga.example[] for example workflow";
    -1 ""}

\d .

-1 "Loaded .tga namespace v",(.tga.version);
-1 "Treasury General Account: fetchTGA, fetchDTSDeposits, tgaZscore, taxSeasonality + more";
-1 "Run .tga.help[] for full function list";
