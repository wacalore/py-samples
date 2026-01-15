// =============================================================================
// PORTFOLIO OPTIMIZER
// =============================================================================
// Optimization methods for portfolio construction
// Version: 0.1.0

\d .optimizer

// -----------------------------------------------------------------------------
// UTILITY FUNCTIONS
// -----------------------------------------------------------------------------

// Annualization factor (default 252 trading days)
annFactor:252

// Portfolio return given weights and asset returns matrix
// w: weight vector, R: matrix where each row is asset returns
portRet:{[w;R] sum w * avg each R}

// Portfolio volatility given weights and covariance matrix
portVol:{[w;C] sqrt sum w * .kdbtools.mm[C;w]}

// Portfolio Sharpe ratio (annualized)
// Returns annualize by multiplying by periods (252)
// Volatility annualizes by multiplying by sqrt(periods)
portSharpe:{[w;R;C;rf]
    ret:annFactor * sum w * avg each R;
    vol:sqrt[annFactor] * portVol[w;C];
    (ret - rf) % vol}

// Median return (more robust to outliers)
medianRet:{[R] med each R}

// Portfolio median Sharpe
portMedianSharpe:{[w;R;C;rf]
    ret:annFactor * sum w * medianRet R;
    vol:sqrt[annFactor] * portVol[w;C];
    (ret - rf) % vol}

// Normalize weights to sum to 1
normalize:{x % sum x}

// Clip weights to bounds
clipWeights:{[lo;hi;w] lo | w & hi}

// Project weights to simplex (sum to 1, non-negative)
projectSimplex:{[w]
    n:count w;
    u:desc w;
    cssv:sums u;
    rho:last where (u + (1 - cssv) % 1 + til n) > 0;
    theta:(cssv[rho] - 1) % rho + 1;
    0f | w - theta}

// -----------------------------------------------------------------------------
// TABLE CONVERSION HELPERS
// -----------------------------------------------------------------------------

// Convert a table of returns to matrix format for optimizer
// Each column becomes one asset's return series (one row in matrix)
// Usage: R:.optimizer.fromTable[t]
//        R:.optimizer.fromTable[delete date from t]
fromTable:{[t] value flip t}

// Convert table, excluding specified columns (e.g., date, sym)
// Usage: R:.optimizer.fromTableEx[`date`sym;t]
fromTableEx:{[excludeCols;t] fromTable (cols[t] except excludeCols)#t}

// Convert long-format table (date;sym;ret) to matrix
// Each unique sym becomes one row
// Usage: R:.optimizer.fromLongTable[`sym;`ret;t]
fromLongTable:{[symCol;retCol;t] value exec retCol by symCol from t}

// Get asset names from table (for labeling results)
// Usage: assets:.optimizer.assetNames[`date;t]
assetNames:{[excludeCols;t] cols[t] except excludeCols}

// Attach asset names to optimization result weights
// Usage: .optimizer.labelWeights[`date;t;result]
labelWeights:{[excludeCols;t;result]
    assets:assetNames[excludeCols;t];
    result,enlist[`weightsByAsset]!enlist assets!result`weights}

// Convenience wrapper: optimize directly from table
// Usage: .optimizer.optimizeTable[t;`date;0.02;`maxSharpe;10000]
//        .optimizer.optimizeTable[t;`date;0.02;`constrained;0.05;0.4;10000]
optimizeTable:{[t;excludeCols;rf;method;args]
    R:fromTableEx[excludeCols;t];
    assets:assetNames[excludeCols;t];
    result:$[method~`maxSharpe;        maxSharpe[R;rf;args 0];
             method~`constrained;      maxSharpeConstrained[R;rf;args 0;args 1;args 2];
             method~`hillClimb;        maxSharpeHillClimb[R;rf;args 0;args 1];
             method~`hybrid;           maxSharpeHybrid[R;rf;args 0;args 1;args 2];
             method~`median;           maxMedianSharpe[R;rf;args 0];
             method~`riskParity;       riskParity[R;args 0;args 1];
             method~`minVariance;      minVariance[R;args 0];
             '`unknownMethod];
    result,enlist[`assets]!enlist assets!result`weights}

// -----------------------------------------------------------------------------
// VECTORIZED HELPERS (OPTIMIZED)
// -----------------------------------------------------------------------------

// Generate random weight matrix: nIter rows, n columns, each row sums to 1
randWeightMatrix:{[nIter;n]
    raw:(nIter;n)#(nIter*n)?1f;
    {x%sum x} each raw}

// Vectorized portfolio variances: w'Cw for each row of W
// W: nIter x n, C: n x n -> returns nIter vector
portVarsVec:{[W;C]
    WC:.kdbtools.mm[W;C];          // nIter x n
    sum each WC * W}                // diagonal of W*C*W'

// Vectorized Sharpe ratios for weight matrix
// W: nIter x n, mu: n vector (mean returns), C: n x n covariance
sharpesVec:{[W;mu;C;rf]
    rets:annFactor * sum each W *\: mu;   // nIter annualized returns
    vars:portVarsVec[W;C];                 // nIter variances
    vols:sqrt[annFactor] * sqrt vars;      // nIter annualized vols
    (rets - rf) % vols}

// -----------------------------------------------------------------------------
// MAX SHARPE - VECTORIZED RANDOM SEARCH
// -----------------------------------------------------------------------------

// Find max Sharpe portfolio via vectorized random search
maxSharpe:{[R;rf;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    // Generate all random portfolios as matrix
    W:randWeightMatrix[nIter;n];

    // Vectorized Sharpe computation
    sharpes:sharpesVec[W;mu;C;rf];

    // Find best
    bestIdx:sharpes?max sharpes;
    bestW:W bestIdx;

    `weights`sharpe`return`volatility!(
        bestW;
        sharpes bestIdx;
        annFactor * sum bestW * mu;
        sqrt[annFactor] * sqrt portVarsVec[enlist bestW;C] 0)}

// -----------------------------------------------------------------------------
// MAX MEDIAN SHARPE - VECTORIZED
// -----------------------------------------------------------------------------

// Vectorized median Sharpe
medianSharpesVec:{[W;medR;C;rf]
    rets:annFactor * sum each W *\: medR;
    vars:portVarsVec[W;C];
    vols:sqrt[annFactor] * sqrt vars;
    (rets - rf) % vols}

// Find max median Sharpe portfolio (more robust to outliers)
maxMedianSharpe:{[R;rf;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    medR:med each R;

    W:randWeightMatrix[nIter;n];
    sharpes:medianSharpesVec[W;medR;C;rf];

    bestIdx:sharpes?max sharpes;
    bestW:W bestIdx;

    `weights`sharpe`medianReturn`volatility!(
        bestW;
        sharpes bestIdx;
        annFactor * sum bestW * medR;
        sqrt[annFactor] * sqrt portVarsVec[enlist bestW;C] 0)}

// -----------------------------------------------------------------------------
// MAX SHARPE WITH WEIGHT CONSTRAINTS - VECTORIZED
// -----------------------------------------------------------------------------

// Generate constrained random weight matrix
randWeightMatrixConstrained:{[nIter;n;lo;hi]
    // Generate nIter rows, each with n elements in [lo,hi]
    raw:{[lo;hi;n;i] lo + (hi-lo)*n?1f}[lo;hi;n] each til nIter;
    // Normalize and clip iteratively
    step:{[lo;hi;W] {[lo;hi;w] lo | (w%sum w) & hi}[lo;hi] each W};
    5 step[lo;hi]/raw}

// Find max Sharpe with weight limits
maxSharpeConstrained:{[R;rf;lo;hi;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    loVec:$[0h = type lo; n#lo; lo];
    hiVec:$[0h = type hi; n#hi; hi];

    W:randWeightMatrixConstrained[nIter;n;loVec;hiVec];
    sharpes:sharpesVec[W;mu;C;rf];

    bestIdx:sharpes?max sharpes;
    bestW:W bestIdx;

    `weights`sharpe`return`volatility`bounds!(
        bestW;
        sharpes bestIdx;
        annFactor * sum bestW * mu;
        sqrt[annFactor] * sqrt portVarsVec[enlist bestW;C] 0;
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// GRADIENT-BASED OPTIMIZATION (OPTIMIZED)
// -----------------------------------------------------------------------------

// ANALYTICAL gradient of Sharpe ratio (much faster than numerical)
// Sharpe = (w'μ - rf) / sqrt(w'Cw)
// ∂Sharpe/∂w = (μ*σ - (r-rf)*Cw/σ) / σ² = (μ - sharpe*Cw/σ) / σ
sharpeGradAnalytical:{[mu;C;rf;w]
    Cw:.kdbtools.mm[C;w];
    variance:sum w * Cw;
    sigma:sqrt variance;
    ret:sum w * mu;
    sharpe:(ret - rf) % sigma;
    (mu - sharpe * Cw % sigma) % sigma}

// Numerical gradient (kept for comparison/fallback)
sharpeGradNumerical:{[mu;C;rf;w;eps]
    n:count w;
    portSh:{[mu;C;rf;w] (sum[w*mu] - rf) % sqrt sum w * .kdbtools.mm[C;w]};
    base:portSh[mu;C;rf;w];
    {[mu;C;rf;w;eps;base;ps;i]
        wUp:@[w;i;+;eps];
        wUp:wUp % sum wUp;
        (ps[mu;C;rf;wUp] - base) % eps
    }[mu;C;rf;w;eps;base;portSh] each til n}

// Hill climbing with analytical gradient and momentum
// Uses normalized gradient for stable convergence
maxSharpeHillClimb:{[R;rf;nIter;lr]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    // Start with inverse volatility (better than equal weight)
    vars:{x[y;y]}[C] each til n;
    w:normalize 1f % sqrt vars;

    // Gradient ascent with momentum and normalized gradient
    beta:0.8;  // Momentum coefficient
    state:`w`v!(w;n#0f);

    step:{[mu;C;rf;lr;beta;s]
        g:sharpeGradAnalytical[mu;C;rf;s`w];
        gNorm:g % sqrt sum g*g;  // Normalize gradient
        v:(beta * s`v) + (1-beta) * gNorm;
        wNew:s[`w] + lr * v;
        wNew:projectSimplex wNew;
        `w`v!(wNew;v)};

    state:nIter step[mu;C;rf;lr;beta]/state;
    w:state`w;

    `weights`sharpe`return`volatility!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w])}

// Hill climbing with weight constraints (analytical gradient + momentum)
// Uses normalized gradient for stable convergence
maxSharpeHillClimbConstrained:{[R;rf;lo;hi;nIter;lr]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    loVec:$[0h = type lo; n#lo; lo];
    hiVec:$[0h = type hi; n#hi; hi];

    // Start with clipped inverse volatility
    vars:{x[y;y]}[C] each til n;
    w:loVec | (normalize 1f % sqrt vars) & hiVec;
    w:normalize w;

    // Gradient ascent with momentum, normalized gradient, and projection
    beta:0.8;
    state:`w`v!(w;n#0f);

    step:{[mu;C;rf;lr;beta;lo;hi;s]
        g:sharpeGradAnalytical[mu;C;rf;s`w];
        gNorm:g % sqrt sum g*g;  // Normalize gradient
        v:(beta * s`v) + (1-beta) * gNorm;
        wNew:s[`w] + lr * v;
        wNew:lo | wNew & hi;
        wNew:wNew % sum wNew;
        `w`v!(wNew;v)};

    state:nIter step[mu;C;rf;lr;beta;loVec;hiVec]/state;
    w:state`w;

    `weights`sharpe`return`volatility`bounds!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w];
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// COMBINED APPROACH: RANDOM START + HILL CLIMBING
// -----------------------------------------------------------------------------

// Best of both: random search to find good starting point, then refine
// Uses normalized gradient for stable convergence
maxSharpeHybrid:{[R;rf;nRandom;nClimb;lr]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    // Random search phase
    randResult:maxSharpe[R;rf;nRandom];
    w:randResult`weights;

    // Refine with analytical gradient + momentum (normalized)
    beta:0.8;
    state:`w`v!(w;n#0f);

    step:{[mu;C;rf;lr;beta;s]
        g:sharpeGradAnalytical[mu;C;rf;s`w];
        gNorm:g % sqrt sum g*g;  // Normalize gradient
        v:(beta * s`v) + (1-beta) * gNorm;
        wNew:s[`w] + lr * v;
        wNew:projectSimplex wNew;
        `w`v!(wNew;v)};

    state:nClimb step[mu;C;rf;lr;beta]/state;
    w:state`w;

    `weights`sharpe`return`volatility!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w])}

// Hybrid with constraints
// Uses normalized gradient for stable convergence
maxSharpeHybridConstrained:{[R;rf;lo;hi;nRandom;nClimb;lr]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    loVec:$[0h = type lo; n#lo; lo];
    hiVec:$[0h = type hi; n#hi; hi];

    // Random search phase
    randResult:maxSharpeConstrained[R;rf;lo;hi;nRandom];
    w:randResult`weights;

    // Refine with analytical gradient + momentum (normalized)
    beta:0.8;
    state:`w`v!(w;n#0f);

    step:{[mu;C;rf;lr;beta;lo;hi;s]
        g:sharpeGradAnalytical[mu;C;rf;s`w];
        gNorm:g % sqrt sum g*g;  // Normalize gradient
        v:(beta * s`v) + (1-beta) * gNorm;
        wNew:s[`w] + lr * v;
        wNew:lo | wNew & hi;
        wNew:wNew % sum wNew;
        `w`v!(wNew;v)};

    state:nClimb step[mu;C;rf;lr;beta;loVec;hiVec]/state;
    w:state`w;

    `weights`sharpe`return`volatility`bounds!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w];
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// EQUAL RISK CONTRIBUTION (RISK PARITY)
// -----------------------------------------------------------------------------

// Simple risk parity: weights inverse to volatility
riskParity:{[R;nIter;lr]
    n:count R;
    C:.kdbtools.covmat flip R;

    // Extract diagonal (variances)
    vars:{x[y;y]}[C] each til n;
    vols:sqrt vars;
    w:normalize 1f % vols;

    `weights`sharpe`return`volatility!(
        w;
        portSharpe[w;R;C;0f];
        annFactor * portRet[w;R];
        sqrt[annFactor] * portVol[w;C])}

// -----------------------------------------------------------------------------
// MINIMUM VARIANCE PORTFOLIO
// -----------------------------------------------------------------------------

// Minimum variance portfolio (uses inverse covariance)
// For long-only, use iterative projection
minVariance:{[R;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;

    // Start with risk parity as initial guess
    vars:{x[y;y]}[C] each til n;
    w:normalize 1f % sqrt vars;

    // Gradient descent on portfolio variance
    step:{[C;n;w]
        grad:2 * .kdbtools.mm[C;w];
        wNew:w - 0.1 * grad;
        wNew:0f | wNew;  // Long only
        wNew % sum wNew};

    w:nIter step[C;n]/w;

    `weights`sharpe`return`volatility!(
        w;
        portSharpe[w;R;C;0f];
        annFactor * portRet[w;R];
        sqrt[annFactor] * portVol[w;C])}

// -----------------------------------------------------------------------------
// EXAMPLE AND DOCUMENTATION
// -----------------------------------------------------------------------------

// Example function demonstrating all optimizer functions
// Run: .optimizer.example[]
example:{[]
    -1 "";
    -1 "=== PORTFOLIO OPTIMIZER EXAMPLE ===";
    -1 "";

    // Generate sample data: 5 assets, 252 trading days
    -1 "1. DATA SETUP";
    -1 "   R: matrix where each row is one asset's return series";
    -1 "   Shape: n_assets x n_periods (e.g., 5 x 252)";
    -1 "";

    system "S 42";  // Set seed for reproducibility
    n:5; T:252;

    // Simulate returns with different characteristics
    dailyMu:0.0001 * 3 2 1 2.5 1.5;           // Daily expected returns
    dailyVol:0.01 * 1.5 1.2 0.5 1.8 0.8;      // Daily volatilities
    R:{[mu;vol;T] mu + vol * (T?1f) - 0.5}'[dailyMu;dailyVol;T];

    rf:0.02;  // 2% annual risk-free rate

    -1 "   Generated: ",string[n]," assets, ",string[T]," periods";
    -1 "   Risk-free rate: ",string[rf]," (annual)";
    -1 "";

    // Show asset characteristics
    -1 "   Asset characteristics (annualized):";
    {-1 "     Asset ",string[x],": return=",string[252*avg y],", vol=",string sqrt[252]*dev y}'[til n;R];
    -1 "";

    // ---------------------------------------------------------------------
    -1 "2. OPTIMIZATION FUNCTIONS";
    -1 "";

    // maxSharpe - Unconstrained max Sharpe via random search
    -1 "   maxSharpe[R;rf;nIter]";
    -1 "   - R: return matrix (n_assets x n_periods)";
    -1 "   - rf: annual risk-free rate";
    -1 "   - nIter: number of random portfolios to try";
    r1:maxSharpe[R;rf;10000];
    -1 "   Result: Sharpe=",string[r1`sharpe],", weights=",-3!r1`weights;
    -1 "";

    // maxSharpeConstrained - With weight limits
    -1 "   maxSharpeConstrained[R;rf;lo;hi;nIter]";
    -1 "   - lo: minimum weight per asset (scalar or vector)";
    -1 "   - hi: maximum weight per asset (scalar or vector)";
    r2:maxSharpeConstrained[R;rf;0.05;0.4;10000];
    -1 "   Result: Sharpe=",string[r2`sharpe],", weights=",-3!r2`weights;
    -1 "";

    // maxSharpeHillClimb - Gradient-based optimization
    -1 "   maxSharpeHillClimb[R;rf;nIter;lr]";
    -1 "   - nIter: number of gradient steps";
    -1 "   - lr: learning rate (try 0.01 to 0.05)";
    r3:maxSharpeHillClimb[R;rf;500;0.02];
    -1 "   Result: Sharpe=",string[r3`sharpe],", weights=",-3!r3`weights;
    -1 "";

    // maxSharpeHybrid - Random search + gradient refinement
    -1 "   maxSharpeHybrid[R;rf;nRandom;nClimb;lr]";
    -1 "   - nRandom: random search iterations";
    -1 "   - nClimb: gradient refinement steps";
    r4:maxSharpeHybrid[R;rf;5000;200;0.02];
    -1 "   Result: Sharpe=",string[r4`sharpe],", weights=",-3!r4`weights;
    -1 "";

    // maxMedianSharpe - Robust to outliers
    -1 "   maxMedianSharpe[R;rf;nIter]";
    -1 "   - Uses median returns instead of mean (more robust)";
    r5:maxMedianSharpe[R;rf;10000];
    -1 "   Result: Sharpe=",string[r5`sharpe],", weights=",-3!r5`weights;
    -1 "";

    // riskParity - Equal risk contribution
    -1 "   riskParity[R;nIter;lr]";
    -1 "   - Weights inversely proportional to volatility";
    r6:riskParity[R;100;0.01];
    -1 "   Result: Sharpe=",string[r6`sharpe],", weights=",-3!r6`weights;
    -1 "";

    // minVariance - Minimum volatility portfolio
    -1 "   minVariance[R;nIter]";
    -1 "   - Minimizes portfolio variance (long-only)";
    r7:minVariance[R;500];
    -1 "   Result: Sharpe=",string[r7`sharpe],", vol=",string[r7`volatility];
    -1 "            weights=",-3!r7`weights;
    -1 "";

    // ---------------------------------------------------------------------
    -1 "3. RESULT DICTIONARY KEYS";
    -1 "";
    -1 "   All functions return a dictionary with:";
    -1 "   - weights: portfolio weights (sum to 1)";
    -1 "   - sharpe: annualized Sharpe ratio";
    -1 "   - return: annualized expected return";
    -1 "   - volatility: annualized volatility";
    -1 "   - bounds: (constrained only) lo/hi weight limits";
    -1 "";

    // ---------------------------------------------------------------------
    -1 "4. RECOMMENDED USAGE";
    -1 "";
    -1 "   For diversified portfolios:";
    -1 "     .optimizer.maxSharpeConstrained[R;rf;0.05;0.3;20000]";
    -1 "";
    -1 "   For robust optimization:";
    -1 "     .optimizer.maxSharpeHybridConstrained[R;rf;0.05;0.3;10000;500;0.02]";
    -1 "";
    -1 "   For equal risk contribution:";
    -1 "     .optimizer.riskParity[R;100;0.01]";
    -1 "";
    -1 "   For minimum volatility:";
    -1 "     .optimizer.minVariance[R;500]";
    -1 "";

    // ---------------------------------------------------------------------
    -1 "5. WORKING WITH TABLES";
    -1 "";
    -1 "   Most data comes as tables. Use these helpers to convert:";
    -1 "";

    // Create sample table
    t:([] date:2024.01.01+til 100; AAPL:100?0.02-0.01; GOOG:100?0.02-0.01; MSFT:100?0.02-0.01);
    -1 "   Sample table:";
    -1 "   t:([] date:dates; AAPL:rets; GOOG:rets; MSFT:rets)";
    -1 "";

    -1 "   METHOD 1: Manual conversion (most control)";
    -1 "   ------------------------------------------";
    -1 "   // Convert table to matrix (exclude date column)";
    -1 "   R:.optimizer.fromTableEx[enlist`date;t]";
    Rt:fromTableEx[enlist`date;t];
    -1 "   // Shape: ",string[count Rt]," assets x ",string[count first Rt]," periods";
    -1 "";
    -1 "   // Optimize";
    -1 "   result:.optimizer.maxSharpeConstrained[R;0.02;0.1;0.5;10000]";
    rt1:maxSharpeConstrained[Rt;0.02;0.1;0.5;10000];
    -1 "";
    -1 "   // Add asset labels to weights";
    -1 "   result:.optimizer.labelWeights[enlist`date;t;result]";
    rt1:labelWeights[enlist`date;t;rt1];
    -1 "   result`weightsByAsset:";
    -1 "     ",raze{string[x],"| ",string[y],"  "}'[key rt1`weightsByAsset;value rt1`weightsByAsset];
    -1 "";

    -1 "   METHOD 2: All-in-one wrapper";
    -1 "   -----------------------------";
    -1 "   result:.optimizer.optimizeTable[t;enlist`date;rf;method;args]";
    -1 "";
    -1 "   Methods and args:";
    -1 "     `maxSharpe      enlist nIter           e.g. enlist 10000";
    -1 "     `constrained    (lo;hi;nIter)          e.g. (0.05;0.4;10000)";
    -1 "     `hillClimb      (nIter;lr)             e.g. (500;0.02)";
    -1 "     `hybrid         (nRandom;nClimb;lr)    e.g. (5000;200;0.02)";
    -1 "     `median         enlist nIter           e.g. enlist 10000";
    -1 "     `riskParity     (nIter;lr)             e.g. (100;0.01)";
    -1 "     `minVariance    enlist nIter           e.g. enlist 500";
    -1 "";

    -1 "   Example:";
    -1 "   result:.optimizer.optimizeTable[t;enlist`date;0.02;`constrained;(0.1;0.5;10000)]";
    rt2:optimizeTable[t;enlist`date;0.02;`constrained;(0.1;0.5;10000)];
    -1 "   result`assets:";
    -1 "     ",raze{string[x],"| ",string[y],"  "}'[key rt2`assets;value rt2`assets];
    -1 "";

    -1 "=== END EXAMPLE ===";
    -1 "";

    // Return summary table
    ([] func:`maxSharpe`maxSharpeConstrained`maxSharpeHillClimb`maxSharpeHybrid`maxMedianSharpe`riskParity`minVariance;
       sharpe:r1[`sharpe],r2[`sharpe],r3[`sharpe],r4[`sharpe],r5[`sharpe],r6[`sharpe],r7[`sharpe])}

// Show available functions
help:{[]
    -1 "";
    -1 "=== .optimizer FUNCTIONS ===";
    -1 "";
    -1 "TABLE HELPERS (convert tables to matrix format):";
    -1 "  fromTable[t]                       - Table to matrix (all columns)";
    -1 "  fromTableEx[`date;t]               - Exclude columns, then convert";
    -1 "  fromLongTable[`sym;`ret;t]         - Long format (sym,ret) to matrix";
    -1 "  optimizeTable[t;`date;rf;`maxSharpe;enlist 10000]  - Direct from table";
    -1 "";
    -1 "SHARPE MAXIMIZATION:";
    -1 "  maxSharpe[R;rf;nIter]              - Random search";
    -1 "  maxSharpeConstrained[R;rf;lo;hi;n] - With weight bounds";
    -1 "  maxSharpeHillClimb[R;rf;nIter;lr]  - Gradient descent";
    -1 "  maxSharpeHybrid[R;rf;nR;nC;lr]     - Random + gradient";
    -1 "  maxMedianSharpe[R;rf;nIter]        - Median-based (robust)";
    -1 "";
    -1 "CONSTRAINED VARIANTS:";
    -1 "  maxSharpeHillClimbConstrained[R;rf;lo;hi;nIter;lr]";
    -1 "  maxSharpeHybridConstrained[R;rf;lo;hi;nR;nC;lr]";
    -1 "";
    -1 "RISK-BASED:";
    -1 "  riskParity[R;nIter;lr]             - Equal risk contribution";
    -1 "  minVariance[R;nIter]               - Minimum volatility";
    -1 "";
    -1 "UTILITIES:";
    -1 "  example[]                          - Run example with sample data";
    -1 "  help[]                             - Show this help";
    -1 "";
    -1 "PARAMETERS:";
    -1 "  R     - Return matrix: n_assets rows x n_periods columns";
    -1 "  rf    - Annual risk-free rate (e.g., 0.02 for 2%)";
    -1 "  nIter - Number of iterations";
    -1 "  lr    - Learning rate for gradient descent (0.01-0.05)";
    -1 "  lo/hi - Min/max weight bounds (scalar or vector)";
    -1 "";
    -1 "Run .optimizer.example[] for a complete demonstration.";
    -1 "";}

\d .

// Load message
-1 "Loaded .optimizer namespace v0.1.0";
-1 "Functions: maxSharpe, maxMedianSharpe, maxSharpeConstrained, maxSharpeHybrid";
-1 "          riskParity, minVariance";
-1 "Run .optimizer.help[] for usage or .optimizer.example[] for demo";
