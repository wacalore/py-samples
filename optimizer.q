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
             method~`slsqp;            maxSharpeSLSQP[R;rf;args 0;args 1;args 2];
             method~`hillClimb;        maxSharpeHillClimb[R;rf;args 0;args 1];
             method~`hybrid;           maxSharpeHybrid[R;rf;args 0;args 1;args 2];
             method~`median;           maxMedianSharpe[R;rf;args 0];
             method~`riskParity;       riskParity[R;args 0;args 1];
             method~`minVariance;      minVariance[R;args 0];
             method~`minCDaR;          minCDaR[R;args 0;args 1];
             method~`minCDaRConstr;    minCDaRConstrained[R;args 0;args 1;args 2;args 3];
             method~`maxCER;           maxCER[R;args 0;args 1];
             method~`maxCERAnalytical; maxCERAnalytical[R;args 0];
             method~`maxCERConstr;     maxCERConstrained[R;args 0;args 1;args 2;args 3];
             method~`minEntropic;      minEntropicRisk[R;args 0;args 1];
             method~`minEntropicConstr;minEntropicRiskConstrained[R;args 0;args 1;args 2;args 3];
             method~`maxKappa;         maxKappa[R;args 0;args 1;args 2];
             method~`maxKappaConstr;   maxKappaConstrained[R;args 0;args 1;args 2;args 3;args 4];
             method~`maxSortino;       maxSortino[R;args 0];
             method~`maxSortinoConstr; maxSortinoConstrained[R;args 0;args 1;args 2];
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

// Vectorized CER: E[r] - 0.5*lambda*Var[r] for all portfolios
cersVec:{[W;mu;C;lambda]
    rets:annFactor * sum each W *\: mu;
    vars:annFactor * portVarsVec[W;C];
    rets - 0.5 * lambda * vars}

// Vectorized portfolio returns matrix: nIter x T
// W: nIter x n, R: n x T -> returns nIter x T matrix
portRetsMatrix:{[W;R] .kdbtools.mm[W;R]}

// Vectorized entropic risk for all portfolios
// Returns nIter vector of entropic risks
entropicRisksVec:{[W;R;theta]
    PR:portRetsMatrix[W;R];               // nIter x T
    (1 % theta) * log each avg each exp neg theta * PR}

// Vectorized LPM (Lower Partial Moment) for all portfolios
// Returns nIter vector of LPMs
lpmsVec:{[W;R;order;mar]
    PR:portRetsMatrix[W;R];               // nIter x T
    shortfalls:mar - PR;                   // nIter x T
    avg each (0f | shortfalls) xexp order}

// Vectorized Kappa ratios for all portfolios
kappasVec:{[W;R;order;mar]
    PR:portRetsMatrix[W;R];
    excessRets:(avg each PR) - mar;
    lpms:lpmsVec[W;R;order;mar];
    // Handle division by zero
    excessRets % lpms xexp 1 % order}

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

    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

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
// rf is annual, convert to daily for gradient computation
sharpeGradAnalytical:{[mu;C;rf;w]
    Cw:.kdbtools.mm[C;w];
    variance:sum w * Cw;
    sigma:sqrt variance;
    ret:sum w * mu;
    rfDaily:rf % annFactor;
    sharpe:(ret - rfDaily) % sigma;
    (mu - sharpe * Cw % sigma) % sigma}

// Compute Sharpe for weight vector (daily scale, for optimization)
// rf is annual, convert to daily for comparison with daily returns
sharpeRaw:{[mu;C;rf;w] (sum[w*mu] - rf % annFactor) % sqrt sum w * .kdbtools.mm[C;w]}

// Project weights onto constraint set: sum=1, lo<=w<=hi
// Uses iterative clipping (Dykstra-like projection)
projectConstrained:{[lo;hi;w]
    // First project to simplex (sum=1, w>=0)
    w:projectSimplex w;
    // Then iteratively enforce bounds while maintaining sum=1
    iter:{[lo;hi;w]
        w:lo | w & hi;           // Clip to bounds
        slack:1 - sum w;         // How much we need to redistribute
        if[abs[slack] < 1e-10; :w];
        // Find assets that can absorb slack
        canInc:w < hi;           // Can increase
        canDec:w > lo;           // Can decrease
        $[slack > 0;
            // Need to add weight - distribute to assets below hi
            [room:hi - w; room:room * canInc; w + slack * room % sum room];
            // Need to remove weight - take from assets above lo
            [room:w - lo; room:room * canDec; w + slack * room % sum room]]};
    10 iter[lo;hi]/w}

// SLSQP-style optimizer with backtracking line search
// Matches scipy.optimize.minimize behavior
maxSharpeSLSQP:{[R;rf;lo;hi;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

    // Pack parameters into dict to avoid 8-param limit
    p:`mu`C`rf`lo`hi!(mu;C;rf;loVec;hiVec);

    // Initialize with multiple starting points, keep best
    starts:(n#1f%n;                              // Equal weight
            loVec + (hiVec-loVec) * n?1f;        // Random in bounds
            loVec + 0.5 * hiVec - loVec);        // Midpoint
    starts:projectConstrained[loVec;hiVec] each starts;
    startObjs:{[p;w] neg sharpeRaw[p`mu;p`C;p`rf;w]}[p] each starts;
    w:starts startObjs?min startObjs;

    // Optimization state
    state:`w`obj`iter`converged!(w;min startObjs;0;0b);

    // Single optimization step with backtracking line search
    step:{[p;s]
        if[s`converged; :s];

        wCur:s`w;
        objCur:s`obj;
        mu:p`mu; C:p`C; rf:p`rf; lo:p`lo; hi:p`hi;

        // Gradient (negative because we minimize negative Sharpe)
        grad:neg sharpeGradAnalytical[mu;C;rf;wCur];

        // Check convergence (gradient small)
        gradNorm:sqrt sum grad*grad;
        if[gradNorm < 1e-8; :`w`obj`iter`converged!(wCur;objCur;s`iter;1b)];

        // Search direction (steepest descent)
        dir:neg grad;

        // Backtracking line search
        alpha:1f;
        wNew:projectConstrained[lo;hi;wCur + alpha * dir];
        objNew:neg sharpeRaw[mu;C;rf;wNew];

        // Armijo condition threshold
        armijoThresh:objCur + 1e-4 * alpha * sum grad * dir;

        cnt:0;
        while[(objNew > armijoThresh) and (alpha > 1e-10) and (cnt < 20);
            alpha:0.5 * alpha;
            wNew:projectConstrained[lo;hi;wCur + alpha * dir];
            objNew:neg sharpeRaw[mu;C;rf;wNew];
            armijoThresh:objCur + 1e-4 * alpha * sum grad * dir;
            cnt+:1];

        // Update if improved, else mark converged
        $[objNew < objCur;
            `w`obj`iter`converged!(wNew;objNew;s[`iter]+1;0b);
            `w`obj`iter`converged!(wCur;objCur;s`iter;1b)]};

    // Run optimizer
    state:nIter step[p]/state;
    w:state`w;

    `weights`sharpe`return`volatility`bounds`iterations`converged!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w];
        `lo`hi!(loVec;hiVec);
        state`iter;
        state`converged)}

// Unconstrained analytical solution (allows negative weights)
// w* = C^-1 * (μ - rf) / sum(C^-1 * (μ - rf))
maxSharpeAnalytical:{[R;rf]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    // Solve C * w = (mu - rf/252) using LU decomposition approximation
    // Since we don't have matrix inverse, use gradient descent on quadratic
    excessMu:mu - rf % annFactor;

    // Iteratively solve C*w = excessMu
    w:n#1f%n;
    solveStep:{[C;b;w]
        Cw:.kdbtools.mm[C;w];
        resid:b - Cw;
        // Gradient of ||Cw - b||^2 is 2*C'*(Cw-b) = 2*C*(Cw-b) for symmetric C
        grad:.kdbtools.mm[C;resid];
        // Step size from exact line search on quadratic
        Cg:.kdbtools.mm[C;grad];
        alpha:(sum grad * resid) % sum grad * Cg;
        w + alpha * grad};

    w:500 solveStep[C;excessMu]/w;

    // Normalize to sum to 1
    w:w % sum w;

    `weights`sharpe`return`volatility!(
        w;
        portSharpe[w;R;C;rf];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w])}

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

    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

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

    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

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
// ADVANCED RISK MEASURES
// -----------------------------------------------------------------------------

// Drawdown series from cumulative returns
drawdowns:{[cumRets] cumRets - maxs cumRets}

// Maximum drawdown
maxDD:{[cumRets] min drawdowns cumRets}

// Conditional Drawdown at Risk (average of worst alpha% drawdowns)
CDaR:{[alpha;cumRets]
    dd:drawdowns cumRets;
    threshold:dd (floor alpha * count dd);  // Alpha quantile
    avg dd where dd <= threshold}

// Portfolio cumulative returns
portCumRet:{[w;R] sums sum w * R}

// Certainty Equivalent Return: E[r] - 0.5 * lambda * Var[r]
CER:{[lambda;w;R;C]
    ret:annFactor * sum w * avg each R;
    pvar:annFactor * sum w * .kdbtools.mm[C;w];
    ret - 0.5 * lambda * pvar}

// Entropic risk measure: (1/theta) * log(E[exp(-theta * returns)])
entropicRisk:{[theta;portRets]
    (1 % theta) * log avg exp neg theta * portRets}

// Lower Partial Moment of order n below MAR
LPM:{[n;mar;rets]
    shortfall:mar - rets;
    avg (0f | shortfall) xexp n}

// Kappa ratio: (E[r] - MAR) / LPM(n)^(1/n)
kappa:{[n;mar;rets]
    excessRet:(avg rets) - mar;
    lpm:LPM[n;mar;rets];
    excessRet % lpm xexp 1 % n}

// -----------------------------------------------------------------------------
// MIN CDAR OPTIMIZER
// -----------------------------------------------------------------------------

// Minimize Conditional Drawdown at Risk
// alpha: confidence level (e.g., 0.05 for worst 5% of drawdowns)
minCDaR:{[R;alpha;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;

    // Generate random portfolios
    W:randWeightMatrix[nIter;n];

    // Compute CDaR for each portfolio
    cdars:{[alpha;R;w] CDaR[alpha;portCumRet[w;R]]}[alpha;R] each W;

    // Find minimum (least negative = best)
    bestIdx:cdars?max cdars;
    bestW:W bestIdx;

    `weights`cdar`sharpe`return`volatility`maxDD!(
        bestW;
        cdars bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        maxDD portCumRet[bestW;R])}

// Min CDaR with constraints
minCDaRConstrained:{[R;alpha;lo;hi;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

    W:randWeightMatrixConstrained[nIter;n;loVec;hiVec];
    cdars:{[alpha;R;w] CDaR[alpha;portCumRet[w;R]]}[alpha;R] each W;

    bestIdx:cdars?max cdars;
    bestW:W bestIdx;

    `weights`cdar`sharpe`return`volatility`maxDD`bounds!(
        bestW;
        cdars bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        maxDD portCumRet[bestW;R];
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// MAX CER (CERTAINTY EQUIVALENT RETURN) OPTIMIZER
// -----------------------------------------------------------------------------

// Maximize Certainty Equivalent Return (VECTORIZED)
// lambda: risk aversion coefficient (higher = more risk averse)
// Typical values: 1-10, where 2-4 is moderate
maxCER:{[R;lambda;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    W:randWeightMatrix[nIter;n];
    cers:cersVec[W;mu;C;lambda];          // Vectorized!

    bestIdx:cers?max cers;
    bestW:W bestIdx;

    `weights`cer`sharpe`return`volatility`lambda!(
        bestW;
        cers bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * mu;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        lambda)}

// Max CER with gradient descent (analytical solution exists)
maxCERAnalytical:{[R;lambda]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;

    // Analytical: w* = (1/lambda) * C^-1 * mu (then normalize for long-only)
    // Approximate with gradient ascent
    w:n#1f%n;  // Equal weight start

    step:{[mu;C;lambda;w]
        // Gradient of CER: mu - lambda * C * w
        grad:mu - lambda * .kdbtools.mm[C;w];
        gNorm:grad % sqrt sum grad*grad;
        wNew:w + 0.01 * gNorm;
        wNew:0f | wNew;
        wNew % sum wNew};

    w:500 step[mu;C;lambda]/w;

    `weights`cer`sharpe`return`volatility`lambda!(
        w;
        CER[lambda;w;R;C];
        portSharpe[w;R;C;0f];
        annFactor * sum w * mu;
        sqrt[annFactor] * sqrt sum w * .kdbtools.mm[C;w];
        lambda)}

// Max CER with constraints (VECTORIZED)
maxCERConstrained:{[R;lambda;lo;hi;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    mu:avg each R;
    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

    W:randWeightMatrixConstrained[nIter;n;loVec;hiVec];
    cers:cersVec[W;mu;C;lambda];          // Vectorized!

    bestIdx:cers?max cers;
    bestW:W bestIdx;

    `weights`cer`sharpe`return`volatility`lambda`bounds!(
        bestW;
        cers bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * mu;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        lambda;
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// MIN ENTROPIC RISK OPTIMIZER
// -----------------------------------------------------------------------------

// Minimize Entropic Risk Measure (VECTORIZED)
// theta: risk aversion parameter (higher = more risk averse to tail losses)
// Typical values: 1-20
minEntropicRisk:{[R;theta;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;

    W:randWeightMatrix[nIter;n];
    eRisks:entropicRisksVec[W;R;theta];   // Vectorized!

    // Minimize (find most negative = lowest risk)
    bestIdx:eRisks?min eRisks;
    bestW:W bestIdx;

    `weights`entropicRisk`sharpe`return`volatility`theta!(
        bestW;
        eRisks bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        theta)}

// Min Entropic Risk with constraints (VECTORIZED)
minEntropicRiskConstrained:{[R;theta;lo;hi;nIter]
    n:count R;
    C:.kdbtools.covmat flip R;
    loVec:$[0 > type lo; n#lo; lo];
    hiVec:$[0 > type hi; n#hi; hi];

    W:randWeightMatrixConstrained[nIter;n;loVec;hiVec];
    eRisks:entropicRisksVec[W;R;theta];   // Vectorized!

    bestIdx:eRisks?min eRisks;
    bestW:W bestIdx;

    `weights`entropicRisk`sharpe`return`volatility`theta`bounds!(
        bestW;
        eRisks bestIdx;
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        theta;
        `lo`hi!(loVec;hiVec))}

// -----------------------------------------------------------------------------
// MAX KAPPA (GENERALIZED SHARPE) OPTIMIZER
// -----------------------------------------------------------------------------

// Maximize Kappa ratio (generalized Sharpe using LPM) - VECTORIZED
// n: LPM order (1=shortfall prob, 2=semi-variance, 3=skewness-aware)
// mar: Minimum Acceptable Return (daily, e.g., 0 or rf/252)
maxKappa:{[R;order;mar;nIter]
    nAssets:count R;
    C:.kdbtools.covmat flip R;

    W:randWeightMatrix[nIter;nAssets];
    kappas:kappasVec[W;R;order;mar];      // Vectorized!
    // Handle infinities (replace with large negative for argmax)
    kappas:@[kappas;where kappas=0w;:;neg 0w];

    // Find maximum
    bestIdx:kappas?max kappas;
    bestW:W bestIdx;

    portRets:sum bestW * R;

    `weights`kappa`lpm`sharpe`return`volatility`order`mar!(
        bestW;
        kappas bestIdx;
        LPM[order;mar;portRets];
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        order;
        mar)}

// Max Kappa with constraints (VECTORIZED)
maxKappaConstrained:{[R;order;mar;lo;hi;nIter]
    nAssets:count R;
    C:.kdbtools.covmat flip R;
    loVec:$[0h = type lo; nAssets#lo; lo];
    hiVec:$[0h = type hi; nAssets#hi; hi];

    W:randWeightMatrixConstrained[nIter;nAssets;loVec;hiVec];
    kappas:kappasVec[W;R;order;mar];      // Vectorized!
    kappas:@[kappas;where kappas=0w;:;neg 0w];

    bestIdx:kappas?max kappas;
    bestW:W bestIdx;

    portRets:sum bestW * R;

    `weights`kappa`lpm`sharpe`return`volatility`order`mar`bounds!(
        bestW;
        kappas bestIdx;
        LPM[order;mar;portRets];
        portSharpe[bestW;R;C;0f];
        annFactor * sum bestW * avg each R;
        sqrt[annFactor] * sqrt sum bestW * .kdbtools.mm[C;bestW];
        order;
        mar;
        `lo`hi!(loVec;hiVec))}

// Sortino ratio is Kappa with order=2, mar=0
maxSortino:{[R;nIter] maxKappa[R;2;0f;nIter]}
maxSortinoConstrained:{[R;lo;hi;nIter] maxKappaConstrained[R;2;0f;lo;hi;nIter]}

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
    -1 "3. ADVANCED RISK OPTIMIZERS";
    -1 "";

    // minCDaR - Minimize Conditional Drawdown at Risk
    -1 "   minCDaR[R;alpha;nIter]";
    -1 "   - alpha: confidence level (e.g., 0.05 = worst 5% of drawdowns)";
    -1 "   - Minimizes expected drawdown in tail scenarios";
    r8:minCDaR[R;0.05;10000];
    -1 "   Result: CDaR=",string[r8`cdar],", MaxDD=",string[r8`maxDD];
    -1 "            Sharpe=",string[r8`sharpe];
    -1 "";

    // maxCER - Maximize Certainty Equivalent Return
    -1 "   maxCER[R;lambda;nIter]";
    -1 "   - lambda: risk aversion (1-10, higher=more conservative)";
    -1 "   - Balances return vs variance based on risk preference";
    r9:maxCER[R;3;10000];
    -1 "   Result: CER=",string[r9`cer],", Sharpe=",string[r9`sharpe];
    -1 "";

    // maxCERAnalytical - Faster gradient-based CER
    -1 "   maxCERAnalytical[R;lambda]";
    -1 "   - Gradient descent (faster, no nIter needed)";
    r10:maxCERAnalytical[R;3];
    -1 "   Result: CER=",string[r10`cer],", Sharpe=",string[r10`sharpe];
    -1 "";

    // minEntropicRisk - Minimize Entropic Risk
    -1 "   minEntropicRisk[R;theta;nIter]";
    -1 "   - theta: tail aversion (1-20, higher=more tail-averse)";
    -1 "   - Penalizes large losses exponentially";
    r11:minEntropicRisk[R;10;10000];
    -1 "   Result: EntropicRisk=",string[r11`entropicRisk],", Sharpe=",string[r11`sharpe];
    -1 "";

    // maxKappa - Maximize Kappa (generalized Sharpe)
    -1 "   maxKappa[R;order;mar;nIter]";
    -1 "   - order: LPM order (2=semi-variance, 3=skewness-aware)";
    -1 "   - mar: minimum acceptable return (e.g., 0 or rf/252)";
    r12:maxKappa[R;2;0f;10000];
    -1 "   Result: Kappa=",string[r12`kappa],", LPM=",string[r12`lpm];
    -1 "            Sharpe=",string[r12`sharpe];
    -1 "";

    // maxSortino - Maximize Sortino ratio
    -1 "   maxSortino[R;nIter]";
    -1 "   - Sortino = Kappa with order=2, mar=0";
    -1 "   - Only penalizes downside volatility";
    r13:maxSortino[R;10000];
    -1 "   Result: Sortino=",string[r13`kappa],", Sharpe=",string[r13`sharpe];
    -1 "";

    // Constrained example
    -1 "   Constrained variants (add lo;hi before nIter):";
    -1 "   minCDaRConstrained, maxCERConstrained, minEntropicRiskConstrained";
    -1 "   maxKappaConstrained, maxSortinoConstrained";
    r14:maxSortinoConstrained[R;0.05;0.4;10000];
    -1 "   maxSortinoConstrained[R;0.05;0.4;10000]: Sortino=",string[r14`kappa];
    -1 "";

    // ---------------------------------------------------------------------
    -1 "4. RESULT DICTIONARY KEYS";
    -1 "";
    -1 "   All functions return a dictionary with:";
    -1 "   - weights: portfolio weights (sum to 1)";
    -1 "   - sharpe: annualized Sharpe ratio";
    -1 "   - return: annualized expected return";
    -1 "   - volatility: annualized volatility";
    -1 "   - bounds: (constrained only) lo/hi weight limits";
    -1 "   Advanced optimizers also return their objective (cdar, cer, etc.)";
    -1 "";

    // ---------------------------------------------------------------------
    -1 "5. RECOMMENDED USAGE";
    -1 "";
    -1 "   For diversified portfolios:";
    -1 "     .optimizer.maxSharpeConstrained[R;rf;0.05;0.3;20000]";
    -1 "";
    -1 "   For robust optimization:";
    -1 "     .optimizer.maxSharpeHybridConstrained[R;rf;0.05;0.3;10000;500;0.02]";
    -1 "";
    -1 "   For drawdown control:";
    -1 "     .optimizer.minCDaRConstrained[R;0.05;0.05;0.3;10000]";
    -1 "";
    -1 "   For downside risk focus (Sortino):";
    -1 "     .optimizer.maxSortinoConstrained[R;0.05;0.3;10000]";
    -1 "";
    -1 "   For tail risk aversion:";
    -1 "     .optimizer.minEntropicRiskConstrained[R;10;0.05;0.3;10000]";
    -1 "";
    -1 "   For mean-variance with risk aversion:";
    -1 "     .optimizer.maxCERAnalytical[R;3]  // fast gradient-based";
    -1 "";

    // ---------------------------------------------------------------------
    -1 "6. WORKING WITH TABLES";
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
    -1 "     `maxSharpe         enlist nIter              e.g. enlist 10000";
    -1 "     `constrained       (lo;hi;nIter)             e.g. (0.05;0.4;10000)";
    -1 "     `slsqp             (lo;hi;nIter)             e.g. (0.05;0.4;500)";
    -1 "     `hillClimb         (nIter;lr)                e.g. (500;0.02)";
    -1 "     `hybrid            (nRandom;nClimb;lr)       e.g. (5000;200;0.02)";
    -1 "     `median            enlist nIter              e.g. enlist 10000";
    -1 "     `riskParity        (nIter;lr)                e.g. (100;0.01)";
    -1 "     `minVariance       enlist nIter              e.g. enlist 500";
    -1 "     `minCDaR           (alpha;nIter)             e.g. (0.05;5000)";
    -1 "     `minCDaRConstr     (alpha;lo;hi;nIter)       e.g. (0.05;0.05;0.3;5000)";
    -1 "     `maxCER            (lambda;nIter)            e.g. (2;5000)";
    -1 "     `maxCERAnalytical  enlist lambda             e.g. enlist 2";
    -1 "     `maxCERConstr      (lambda;lo;hi;nIter)      e.g. (2;0.05;0.3;5000)";
    -1 "     `minEntropic       (theta;nIter)             e.g. (5;5000)";
    -1 "     `minEntropicConstr (theta;lo;hi;nIter)       e.g. (5;0.05;0.3;5000)";
    -1 "     `maxKappa          (order;mar;nIter)         e.g. (2;0f;5000)";
    -1 "     `maxKappaConstr    (order;mar;lo;hi;nIter)   e.g. (2;0f;0.05;0.3;5000)";
    -1 "     `maxSortino        enlist nIter              e.g. enlist 5000";
    -1 "     `maxSortinoConstr  (lo;hi;nIter)             e.g. (0.05;0.3;5000)";
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
    ([] func:`maxSharpe`maxSharpeConstrained`maxSharpeHillClimb`maxSharpeHybrid`maxMedianSharpe`riskParity`minVariance`minCDaR`maxCER`maxCERAnalytical`minEntropicRisk`maxKappa`maxSortino;
       sharpe:r1[`sharpe],r2[`sharpe],r3[`sharpe],r4[`sharpe],r5[`sharpe],r6[`sharpe],r7[`sharpe],r8[`sharpe],r9[`sharpe],r10[`sharpe],r11[`sharpe],r12[`sharpe],r13[`sharpe])}

// -----------------------------------------------------------------------------
// ROBUST OPTIMIZER (handles sparse, non-normal alphas)
// -----------------------------------------------------------------------------

// Symmetric winsorize at percentile bounds (e.g., 0.05 = clip at 5th and 95th)
pctWinsorize:{[pct;x] v:x where not null x; lo:v (iasc v) `long$pct*n:count v; hi:v (iasc v) `long$(1-pct)*n; lo|x&hi}

// Rank normalize to 0-1
rankNorm:{[x] v:x; nonnull:where not null v; v[nonnull]:(iasc iasc v nonnull) % count nonnull; v}

// Fill rate (fraction non-null)
fillRate:{1 - (sum null x) % count x}

// Pairwise correlation (only overlapping non-null)
pairwiseCor:{[x;y] valid:where not null[x] & not null y; $[2>count valid;0f;cor[x valid;y valid]]}

// Spearman correlation (rank-based, robust to outliers)
spearmanCor:{[x;y] valid:where not null[x] & not null y; $[2>count valid;0f;cor[iasc iasc x valid;iasc iasc y valid]]}

// Build correlation matrix using pairwise overlapping periods
pairwiseCorMat:{[R] n:count R; raze {[R;n;i] {[R;i;j] $[i=j;1f;pairwiseCor[R i;R j]]}[R;i] each til n}[R;n] each til n}

// Build Spearman correlation matrix (returns n x n matrix)
spearmanCorMat:{[R] n:count R; (n;n)#raze {[R;i;n] {[R;i;j] $[i=j;1f;spearmanCor[R i;R j]]}[R;i] each til n}[R;;n] each til n}

// Shrink correlation toward identity (reduces estimation error)
shrinkCor:{[shrinkage;corMat] n:count corMat; I:{x=y}'[til n;til n]; ((1-shrinkage)*corMat) + shrinkage*I}

// MAD-based volatility (robust to outliers)
madVol:{[x] v:x where not null x; 1.4826 * med abs v - med v}

// Robust covariance from Spearman cor + MAD-based vol
robustCov:{[R;shrinkage] vols:madVol each R; corMat:spearmanCorMat R; shrunkenCor:shrinkCor[shrinkage;corMat]; vols *\: vols * shrunkenCor}

// Robust optimizer for alpha tables
// t: table with alpha columns (and optional non-alpha columns to exclude)
// exclude: list of column names to exclude (e.g., `date`sym)
// params: dictionary with optional settings
//   `minFill    - minimum fill rate to include alpha (default 0.5)
//   `winsorPct  - winsorization percentile (default 0.05)
//   `rankNorm   - rank normalize signals (default 1b)
//   `shrinkage  - correlation shrinkage (default 0.2)
//   `method     - optimization method: `sharpe`sortino`riskParity`minVar (default `sortino)
//   `lo         - min weight (default 0.0)
//   `hi         - max weight (default 1.0)
//   `nIter      - iterations (default 5000)
robustOptimize:{[t;exclude;params]
    defaults:`minFill`winsorPct`rankNorm`shrinkage`method`lo`hi`nIter!(0.5;0.05;1b;0.2;`sortino;0f;1f;5000);
    p:defaults,params;
    allCols:cols t;
    alphaCols:allCols except exclude;
    alphas:t alphaCols;
    nAlphas:count alphas;
    alphaNames:alphaCols;
    fr:fillRate each alphas;
    validIdx:where fr >= p`minFill;
    if[0=count validIdx; '"No alphas meet minimum fill rate"];
    R:alphas validIdx;
    validNames:alphaNames validIdx;
    R:{0f^fills x} each R;
    R:pctWinsorize[p`winsorPct] each R;
    R:$[p`rankNorm; rankNorm each R; R];
    loVec:$[0>type p`lo; (count R)#p`lo; p`lo];
    hiVec:$[0>type p`hi; (count R)#p`hi; p`hi];
    lo1:first loVec; hi1:first hiVec;
    result:$[p[`method]~`sharpe;maxSharpeConstrained[R;0f;loVec;hiVec;p`nIter];p[`method]~`sortino;maxSortinoConstrained[R;loVec;hiVec;p`nIter];p[`method]~`riskParity;riskParity[R;p`nIter;0.01];p[`method]~`minVar;minVarianceConstrained[R;loVec;hiVec;p`nIter];p[`method]~`cdar;minCDaRConstrained[R;0.05;lo1;hi1;p`nIter];maxSortinoConstrained[R;loVec;hiVec;p`nIter]];
    fullWeights:nAlphas#0f;
    fullWeights[validIdx]:result`weights;
    weightsByAlpha:alphaNames!fullWeights;
    result,`weightsByAlpha`validAlphas`excludedAlphas`fillRates`params!(weightsByAlpha;validNames;alphaNames except validNames;alphaNames!fr;p)}

// Simplified interface for common case
robustOptimizeSimple:{[t;exclude]
    robustOptimize[t;exclude;()!()]}

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
    -1 "  maxSharpeSLSQP[R;rf;lo;hi;nIter]   - Gradient-based (like scipy)";
    -1 "  maxSharpeHillClimb[R;rf;nIter;lr]  - Gradient descent";
    -1 "  maxSharpeHybrid[R;rf;nR;nC;lr]     - Random + gradient";
    -1 "  maxMedianSharpe[R;rf;nIter]        - Median-based (robust)";
    -1 "";
    -1 "RISK-BASED:";
    -1 "  riskParity[R;nIter;lr]             - Equal risk contribution";
    -1 "  minVariance[R;nIter]               - Minimum volatility";
    -1 "";
    -1 "ADVANCED RISK OPTIMIZERS:";
    -1 "  minCDaR[R;alpha;nIter]             - Min Conditional Drawdown at Risk";
    -1 "  maxCER[R;lambda;nIter]             - Max Certainty Equivalent Return";
    -1 "  maxCERAnalytical[R;lambda]         - CER via gradient (faster)";
    -1 "  minEntropicRisk[R;theta;nIter]     - Min Entropic Risk Measure";
    -1 "  maxKappa[R;order;mar;nIter]        - Max Kappa (generalized Sharpe)";
    -1 "  maxSortino[R;nIter]                - Max Sortino (Kappa order=2, mar=0)";
    -1 "";
    -1 "CONSTRAINED VARIANTS (add lo;hi before nIter):";
    -1 "  minCDaRConstrained, maxCERConstrained, minEntropicRiskConstrained";
    -1 "  maxKappaConstrained, maxSortinoConstrained";
    -1 "  maxSharpeHillClimbConstrained, maxSharpeHybridConstrained";
    -1 "";
    -1 "RISK MEASURE HELPERS:";
    -1 "  drawdowns[cumRets]                 - Drawdown series";
    -1 "  maxDD[cumRets]                     - Maximum drawdown";
    -1 "  CDaR[alpha;cumRets]                - Conditional Drawdown at Risk";
    -1 "  CER[lambda;w;R;C]                  - Certainty Equivalent Return";
    -1 "  entropicRisk[theta;portRets]       - Entropic risk measure";
    -1 "  LPM[n;mar;rets]                    - Lower Partial Moment";
    -1 "  kappa[n;mar;rets]                  - Kappa ratio";
    -1 "";
    -1 "ROBUST OPTIMIZER (handles sparse, non-normal alphas):";
    -1 "  robustOptimize[t;exclude;params]   - Full robust pipeline";
    -1 "  robustOptimizeSimple[t;exclude]    - With default params";
    -1 "";
    -1 "  params dictionary keys:";
    -1 "    `minFill    - min fill rate (default 0.5)";
    -1 "    `winsorPct  - winsorization pct (default 0.05)";
    -1 "    `rankNorm   - rank normalize (default 1b)";
    -1 "    `shrinkage  - correlation shrinkage (default 0.2)";
    -1 "    `method     - `sharpe`sortino`riskParity`minVar`cdar";
    -1 "    `lo`hi      - weight bounds (default 0,1)";
    -1 "    `nIter      - iterations (default 5000)";
    -1 "";
    -1 "UTILITIES:";
    -1 "  example[]                          - Run example with sample data";
    -1 "  help[]                             - Show this help";
    -1 "";
    -1 "PARAMETERS:";
    -1 "  R      - Return matrix: n_assets rows x n_periods columns";
    -1 "  rf     - Annual risk-free rate (e.g., 0.02 for 2%)";
    -1 "  nIter  - Number of iterations";
    -1 "  lr     - Learning rate for gradient descent (0.01-0.05)";
    -1 "  lo/hi  - Min/max weight bounds (scalar or vector)";
    -1 "  alpha  - CDaR confidence level (e.g., 0.05 for worst 5%)";
    -1 "  lambda - Risk aversion for CER (1-10, higher=more conservative)";
    -1 "  theta  - Entropic risk aversion (1-20, higher=more tail-averse)";
    -1 "  order  - LPM order (2=semi-variance, 3=skewness-aware)";
    -1 "  mar    - Minimum acceptable return (e.g., 0 or rf/252)";
    -1 "";
    -1 "Run .optimizer.example[] for a complete demonstration.";
    -1 "";}

\d .

// Load message
-1 "Loaded .optimizer namespace v0.2.0";
-1 "Functions: maxSharpe, maxSharpeConstrained, maxSharpeSLSQP, maxSharpeHybrid, maxMedianSharpe";
-1 "          riskParity, minVariance, minCDaR, maxCER, minEntropicRisk, maxKappa, maxSortino";
-1 "Run .optimizer.help[] for usage or .optimizer.example[] for demo";
