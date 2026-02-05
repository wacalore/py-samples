// =============================================================================
// PC RISK - Principal Component Risk Management
// =============================================================================
// Factor-based portfolio risk decomposition and management
// Version: 0.2.0
//
// Key concepts:
//   - Decompose asset returns into orthogonal principal components
//   - Map portfolio weights to PC exposure space
//   - Constrain or optimize in PC space for factor-aware portfolios
//
// Typical PC interpretation:
//   PC1 = Market/Beta (60-80% of variance)
//   PC2 = Value/Growth or Sector rotation
//   PC3+ = Increasingly idiosyncratic factors

\d .pcrisk

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

// Covariance matrix (avoid dependency on kdbtools.q)
covmat:{[X] n:count X; mu:avg each flip X; Xc:X -\: mu; (flip Xc) mmu Xc % n-1}

// =============================================================================
// CORE PCA COMPUTATION
// =============================================================================

// Compute PCA on asset returns
// @param R - return matrix (T x n) or table with asset columns
// @param k - number of components (0 = all)
// @param cfg - config dict: `scale (1b=correlation, 0b=covariance), `excludeCols
// @return dict: loadings, eigenvalues, explainedVar, scores, mu, sigma, assets
pca:{[R;k;cfg]
    cfg:(`scale`excludeCols!(1b;`dt`date`time)),cfg;

    // Handle table input
    isTable:98h = type R;
    assets:$[isTable; cols[R] except cfg`excludeCols; `$"A",/:string til count first R];
    X:$[isTable; flip value flip (cols[R] except cfg`excludeCols)#R; R];
    X:`float$X;

    T:count X;
    n:count first X;
    k:$[k=0; n; k&n];

    // Center and optionally scale
    mu:avg each flip X;
    sigma:$[cfg`scale; {$[0<x;x;1e-10]} each dev each flip X; n#1f];
    Xc:flip ((flip X) - mu) % sigma;

    // Covariance/correlation matrix
    C:((flip Xc) mmu Xc) % T-1;

    // Eigendecomposition via power iteration with deflation
    loadings:();
    eigenvals:();
    Cwork:C;

    do[k;
        // Power iteration
        v:n?1.0;
        v:v % sqrt sum v*v;
        do[100; v2:Cwork mmu v; nrm:sqrt sum v2*v2; v:$[nrm>1e-10;v2%nrm;v]];
        ev:sum v * Cwork mmu v;
        eigenvals,:ev;
        loadings,:enlist v;
        // Deflate for next component
        Cwork:Cwork - ev * v */: v
    ];

    // Compute scores (projection of data onto PCs)
    L:flip loadings;  // n x k
    scores:Xc mmu L;  // T x k

    totalVar:sum eigenvals;
    cumVar:sums eigenvals;

    `loadings`eigenvalues`explainedVar`cumExplainedVar`scores`mu`sigma`assets`k`T`n!(
        L;eigenvals;eigenvals%totalVar;cumVar%totalVar;scores;mu;sigma;assets;k;T;n)}

// PCA from pre-computed covariance matrix
// @param C - covariance matrix (n x n)
// @param k - number of components
// @param assets - asset names (optional)
pcaFromCov:{[C;k;assets]
    n:count C;
    k:$[k=0; n; k&n];
    assets:$[0=count assets; `$"A",/:string til n; assets];

    loadings:();
    eigenvals:();
    Cwork:C;

    do[k;
        v:n?1.0;
        v:v % sqrt sum v*v;
        do[100; v2:Cwork mmu v; nrm:sqrt sum v2*v2; v:$[nrm>1e-10;v2%nrm;v]];
        ev:sum v * Cwork mmu v;
        eigenvals,:ev;
        loadings,:enlist v;
        Cwork:Cwork - ev * v */: v
    ];

    L:flip loadings;
    totalVar:sum eigenvals;

    `loadings`eigenvalues`explainedVar`cumExplainedVar`assets`k`n`cov!(
        L;eigenvals;eigenvals%totalVar;(sums eigenvals)%totalVar;assets;k;n;C)}

// Rolling PCA with expanding or fixed window
// @param R - return table with dt column
// @param k - number of PCs
// @param window - window size (0 = expanding)
// @param cfg - config dict
// @return table with dt and PCA results per date
pcaRolling:{[R;k;window;cfg]
    cfg:(`scale`excludeCols`minObs!(1b;`dt`date`time;30)),cfg;
    dates:asc distinct R`dt;
    n:count dates;

    results:{[R;k;window;cfg;dates;i]
        dt:dates i;
        startIdx:$[window=0; 0; (i-window+1)|0];
        sub:select from R where dt <= dates[i], dt >= dates[startIdx];

        if[(count sub) < cfg`minObs; :([] dt:enlist dt; valid:0b)];

        p:pca[sub;k;cfg];
        ([] dt:enlist dt; valid:1b;
            loadings:enlist p`loadings;
            eigenvalues:enlist p`eigenvalues;
            explainedVar:enlist p`explainedVar)
    }[R;k;window;cfg;dates] each til n;

    raze results}

// =============================================================================
// EXPOSURE MAPPING
// =============================================================================

// Map portfolio weights to PC exposures
// @param w - weights (dict asset->weight or vector)
// @param p - PCA result from pca[] or pcaFromCov[]
// @return dict: pcExposure, pcRisk, totalRisk, riskPct
pcExposure:{[w;p]
    // Convert dict to ordered vector
    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;

    L:p`loadings;       // n x k
    ev:p`eigenvalues;   // k

    // PC exposure = L' * w
    pcExp:(flip L) mmu wVec;

    // Risk in PC space: sqrt(sum(pcExp^2 * eigenvalue))
    pcVar:pcExp * pcExp * ev;
    pcRisk:sqrt pcVar;
    totalRisk:sqrt sum pcVar;

    // Percentage of total risk per PC
    riskPct:pcVar % sum pcVar;

    `pcExposure`pcRisk`pcVar`totalRisk`riskPct`eigenvalues!(pcExp;pcRisk;pcVar;totalRisk;riskPct;ev)}

// Detailed risk decomposition report
// @param w - weights
// @param p - PCA result
// @return table with per-PC analysis
pcRiskReport:{[w;p]
    expos:pcExposure[w;p];
    k:count expos`pcExposure;

    ([] pc:`$"PC",/:string 1+til k;
        exposure:expos`pcExposure;
        eigenvalue:expos`eigenvalues;
        riskContrib:expos`pcRisk;
        varianceContrib:expos`pcVar;
        pctOfRisk:100*expos`riskPct;
        pctOfAssetVar:100*p`explainedVar;
        cumPctAssetVar:100*p`cumExplainedVar)}

// =============================================================================
// MULTI-STRATEGY ANALYSIS
// =============================================================================

// Aggregate PC exposures across multiple strategies
// @param strategies - dict of (strategyName -> weights)
// @param allocations - dict of (strategyName -> capital allocation)
// @param p - PCA result
// @return dict with aggregate and per-strategy exposures
aggregateExposure:{[strategies;allocations;p]
    strats:key strategies;
    allocs:strats#allocations;

    // Per-strategy exposures
    perStrat:{[p;name;w]
        expos:pcExposure[w;p];
        `strategy`pcExposure`totalRisk!(name;expos`pcExposure;expos`totalRisk)
    }[p]'[strats;value strategies];

    // Weighted aggregate
    wgtExp:sum each (value allocs) *' perStrat`pcExposure;
    aggExp:`pcExposure`pcRisk`totalRisk!(wgtExp; sqrt wgtExp*wgtExp*p`eigenvalues; sqrt sum wgtExp*wgtExp*p`eigenvalues);

    `aggregate`perStrategy`allocations!(aggExp;perStrat;allocs)}

// Measure strategy overlap/correlation in PC space
// @param strategies - dict of (strategyName -> weights)
// @param p - PCA result
// @return correlation matrix of strategies in PC space
strategyCorrelation:{[strategies;p]
    strats:key strategies;
    n:count strats;

    // Get PC exposures as matrix
    exps:{[p;w] (pcExposure[w;p])`pcExposure}[p] each value strategies;
    E:flip exps;  // k x nStrats

    // Correlation in PC space (weighted by eigenvalues for risk-based)
    ev:p`eigenvalues;
    Ew:E * sqrt ev;  // Weight by sqrt(eigenvalue) for risk

    // Correlation matrix
    norms:sqrt each sum each Ew * Ew;
    calcCorr:{[Ew;norms;i;j]
        $[i=j; 1f; (sum Ew[;i]*Ew[;j]) % norms[i]*norms[j]]};
    corr:{[f;Ew;norms;n;i] f[Ew;norms;i] each til n}[calcCorr;Ew;norms;n] each til n;

    `strategies`correlation!(strats;strats!corr)}

// =============================================================================
// CONSTRAINTS & OPTIMIZATION
// =============================================================================

// Constrain weights to satisfy PC exposure limits
// Uses iterative projection
// @param w - original weights
// @param p - PCA result
// @param limits - PC limits: scalar (all PCs), list (per PC), or dict (`lo`hi)
// @param cfg - config: `maxIter`tol
// @return adjusted weights
pcConstrain:{[w;p;limits;cfg]
    cfg:(`maxIter`tol!(100;1e-6)),cfg;
    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;

    L:p`loadings;
    k:p`k;  // Number of PCs

    // Parse limits: scalar, list, or dict `lo`hi
    limVec:$[-9h=type limits; k#limits; limits];  // Broadcast scalar to k
    lims:$[99h=type limVec;
        (k#limVec`lo; k#limVec`hi);
        (neg abs limVec; abs limVec)];
    lo:lims 0;
    hi:lims 1;

    // Iterative projection
    wAdj:wVec;
    i:0;
    while[i < cfg`maxIter;
        pcExp:(flip L) mmu wAdj;
        pcExpClip:lo | hi & pcExp;

        if[(max abs pcExp - pcExpClip) < cfg`tol; :$[99h=type w; (p`assets)!wAdj; wAdj]];

        correction:L mmu (pcExpClip - pcExp);
        wAdj:wAdj + correction;
        i+:1
    ];

    $[99h=type w; (p`assets)!wAdj; wAdj]}

// Neutralize portfolio to specific PCs (set exposure to zero)
// @param w - weights
// @param p - PCA result
// @param pcs - list of PC indices to neutralize (0-indexed)
// @return neutralized weights
pcNeutralize:{[w;p;pcs]
    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;

    L:p`loadings;
    k:count first flip L;

    // Set limits to 0 for specified PCs, large for others
    limits:(k#1e10);
    limits[pcs]:0f;

    pcConstrain[wVec;p;limits;()!()]}

// Compute hedge weights to neutralize specific PC exposure
// @param w - current weights
// @param p - PCA result
// @param targetPCs - PCs to hedge (indices)
// @param hedgeAssets - assets available for hedging
// @return hedge weights to add
pcHedge:{[w;p;targetPCs;hedgeAssets]
    wVec:$[99h=type w; w (p`assets); w];

    // Current PC exposures
    expos:pcExposure[wVec;p];
    targetExp:expos[`pcExposure] targetPCs;

    // Loadings for target PCs and hedge assets
    L:p`loadings;
    hedgeIdx:(p`assets)?hedgeAssets;
    Lhedge:L[hedgeIdx;targetPCs];

    // Solve: Lhedge' * h = -targetExp (least squares)
    // h = (Lhedge * Lhedge')^-1 * Lhedge * (-targetExp)
    LLt:Lhedge mmu flip Lhedge;
    // Add regularization for stability
    LLtReg:LLt + 0.001 * {x[y;y]:1f;x}[count[LLt]#count[LLt]#0f] each til count LLt;
    h:.qml.minv[LLtReg] mmu Lhedge mmu neg targetExp;

    hedgeAssets!h}

// =============================================================================
// OPTIMIZATION WITH PC CONSTRAINTS
// =============================================================================

// Maximum Sharpe with PC exposure constraints
// @param R - return matrix (n x T) or table
// @param rf - risk-free rate
// @param p - PCA result (or null to compute)
// @param pcLimits - PC exposure limits
// @param cfg - config: `nIter`lo`hi (asset weight bounds)
maxSharpePC:{[R;rf;p;pcLimits;cfg]
    cfg:(`nIter`lo`hi`excludeCols!(10000;0f;1f;`dt`date`time)),cfg;

    // Convert table if needed
    isTable:98h = type R;
    Rmat:$[isTable; .optimizer.fromTableEx[cfg`excludeCols;R]; R];
    assets:$[isTable; .optimizer.assetNames[cfg`excludeCols;R]; `$"A",/:string til count Rmat];

    // Compute PCA if not provided
    p:$[99h=type p; p; pca[R;0;cfg]];

    n:count Rmat;
    C:covmat flip Rmat;
    mu:avg each Rmat;
    k:count p`eigenvalues;
    pcLimits:k#pcLimits;

    // Random search with PC constraint projection
    bestW:n#1f%n;
    bestSharpe:-1e10;

    i:0;
    while[i < cfg`nIter;
        // Random weights
        w:(cfg`lo) + ((cfg`hi)-(cfg`lo)) * n?1f;
        w:w % sum w;

        // Project to PC constraints
        w:pcConstrain[w;p;pcLimits;()!()];

        // Enforce weight bounds
        w:(cfg`lo) | (cfg`hi) & w;
        w:w % sum w;

        // Compute Sharpe
        ret:.optimizer.annFactor * sum w * mu;
        vol:sqrt[.optimizer.annFactor] * .optimizer.portVol[w;C];
        sharpe:(ret - rf) % vol;

        if[sharpe > bestSharpe;
            bestSharpe:sharpe;
            bestW:w
        ];
        i+:1
    ];

    // Final stats
    pcExp:pcExposure[bestW;p];

    `weights`sharpe`return`volatility`pcExposure`pcRiskPct`assets!(
        bestW;
        bestSharpe;
        .optimizer.annFactor * sum bestW * mu;
        sqrt[.optimizer.annFactor] * .optimizer.portVol[bestW;C];
        pcExp`pcExposure;
        100*pcExp`riskPct;
        assets!bestW)}

// Risk parity in PC space
// Equal risk contribution from each PC
// @param R - returns
// @param p - PCA result
// @param cfg - config
riskParityPC:{[R;p;cfg]
    cfg:(`nIter`excludeCols!(1000;`dt`date`time)),cfg;

    isTable:98h = type R;
    Rmat:$[isTable; .optimizer.fromTableEx[cfg`excludeCols;R]; R];
    assets:$[isTable; .optimizer.assetNames[cfg`excludeCols;R]; `$"A",/:string til count Rmat];

    p:$[99h=type p; p; pca[R;0;cfg]];

    n:count Rmat;
    k:count p`eigenvalues;
    L:p`loadings;
    ev:p`eigenvalues;
    C:covmat flip Rmat;

    // Target: equal PC risk contribution
    targetPCRisk:k#1f%k;

    // Iterative optimization
    w:n#1f%n;
    lr:0.01;

    do[cfg`nIter;
        pcExp:(flip L) mmu w;
        pcVar:pcExp * pcExp * ev;
        totalVar:sum pcVar;
        pcRiskPct:pcVar % totalVar;

        // Gradient: push towards equal risk
        riskGap:pcRiskPct - targetPCRisk;

        // Gradient in weight space
        // d(pcVar[i])/dw = 2 * ev[i] * pcExp[i] * L[;i]
        grad:sum each flip {[L;ev;pcExp;riskGap;i]
            2 * ev[i] * pcExp[i] * L[;i] * riskGap[i]
        }[L;ev;pcExp;riskGap] each til k;

        w:w - lr * grad;
        w:0f | w;
        w:w % sum w
    ];

    pcExp:pcExposure[w;p];

    `weights`pcExposure`pcRiskPct`assets!(w;pcExp`pcExposure;100*pcExp`riskPct;assets!w)}

// =============================================================================
// ALPHA PORTFOLIO OPTIMIZATION
// =============================================================================
// Optimize allocation across multiple alpha strategies with PC exposure limits
// Each alpha is a dict of asset weights; we optimize the allocation to alphas

// Main alpha optimization function
// @param alphas - dict of (alphaName -> asset weights dict)
// @param R - returns table or matrix
// @param p - PCA result (or null to compute)
// @param pcLimits - PC exposure limits: scalar, vector, or dict with `lo`hi
// @param cfg - config: `objective`nIter`tol`lo`hi`lambda`targetRet
//              objective: `maxSharpe`minVar`riskParity`maxRet`minTE
// @return dict with optimal alpha weights and diagnostics
alphaOptimize:{[alphas;R;p;pcLimits;cfg]
    defaults:`objective`nIter`tol`lo`hi`lambda`targetRet`rf`excludeCols!(
        `maxSharpe;5000;1e-6;0f;1f;0.01;0n;0f;`dt`date`time);
    cfg:defaults,cfg;

    // Extract alpha names and weights
    alphaNames:key alphas;
    nAlphas:count alphaNames;

    // Store alpha weights - may be table or list of dicts
    alphaWeights:value alphas;

    // Get returns and compute PCA if needed
    isTable:98h=type R;
    Rmat:$[isTable; flip value flip (cols[R] except cfg`excludeCols)#R; R];
    assets:$[isTable; cols[R] except cfg`excludeCols; `$"A",/:string til count first Rmat];
    p:$[99h=type p; p; pca[R;0;cfg]];

    // Compute alpha returns: each alpha's return = sum(alpha_weights * asset_returns)
    // alphaRets[t;a] = return of alpha a at time t
    alphaRets:{[Rmat;aw;assets;i]
        w:aw i; wVec:`float$w assets;
        Rmat mmu wVec
    }[Rmat;alphaWeights;assets] each til nAlphas;
    alphaRets:flip alphaRets;  // T x nAlphas

    // Alpha statistics
    alphaMu:avg each flip alphaRets;
    alphaC:covmat alphaRets;

    // PC exposures of each alpha - result is nAlphas x k (list of k-vectors)
    // alphaPCExp[i] = PC exposures for alpha i
    // alphaPCExp[;j] = all alphas' exposure to PC j
    L:p`loadings;
    alphaPCExp:{[L;aw;assets;i]
        w:aw i; wVec:`float$w assets;
        (flip L) mmu wVec
    }[L;alphaWeights;assets] each til nAlphas;

    k:p`k;
    ev:p`eigenvalues;

    // Parse PC limits (empty/null = unconstrained)
    pcLo:pcHi:k#0n;
    $[(99h=type pcLimits) & 0=count pcLimits;
        ::;  // ()!() - no constraints
      (99h=type pcLimits) & 0<count pcLimits;
        [lo:pcLimits`lo; hi:pcLimits`hi;
         if[0<count lo; pcLo:k#lo]; if[0<count hi; pcHi:k#hi]];
      -9h=type pcLimits;
        [pcLo:k#neg abs pcLimits; pcHi:k#abs pcLimits];
      0<count pcLimits;
        [pcLo:neg abs pcLimits; pcHi:abs pcLimits];
      ::];

    // Objective function based on config
    objective:cfg`objective;

    // Initialize with equal weights
    allocBest:nAlphas#1f%nAlphas;
    bestScore:-1e10;

    // Random search with projection
    nIter:cfg`nIter;
    lo:cfg`lo; hi:cfg`hi;

    i:0;
    while[i < nIter;
        // Generate candidate allocation
        alloc:$[i=0; nAlphas#1f%nAlphas; lo + (hi-lo) * nAlphas?1f];
        alloc:alloc % sum alloc;

        // Project to PC constraints
        alloc:alphaProjectPC[alloc;alphaPCExp;pcLo;pcHi;10];

        // Enforce bounds and renormalize
        alloc:lo | hi & alloc;
        if[0 < sum alloc; alloc:alloc % sum alloc];

        // Compute objective
        ret:sum alloc * alphaMu;
        variance:alloc mmu alphaC mmu alloc;
        vol:sqrt variance;

        score:$[objective=`maxSharpe; (ret - cfg`rf) % vol;
                objective=`minVar; neg variance;
                objective=`maxRet; ret;
                objective=`riskParity; neg sum abs (alloc * alphaC mmu alloc) - variance%nAlphas;
                (ret - cfg`rf) % vol];  // default to Sharpe

        if[score > bestScore;
            bestScore:score;
            allocBest:alloc
        ];
        i+:1
    ];

    // Compute final portfolio characteristics
    finalRet:sum allocBest * alphaMu;
    finalVar:allocBest mmu alphaC mmu allocBest;
    finalVol:sqrt finalVar;
    finalSharpe:(finalRet - cfg`rf) % finalVol;

    // Combined asset weights
    // Convert alpha weights to matrix (nAlphas x nAssets), then weight by allocation
    awMat:{[aw;assets;i] w:aw i; `float$w assets}[alphaWeights;assets] each til nAlphas;
    combinedW:assets!allocBest mmu awMat;

    // Final PC exposures
    finalPCExp:pcExposure[combinedW;p];

    `alphaWeights`combinedAssetWeights`return`volatility`sharpe`pcExposure`pcRiskPct`alphaNames`objective!(
        alphaNames!allocBest;
        combinedW;
        finalRet;
        finalVol;
        finalSharpe;
        finalPCExp`pcExposure;
        100*finalPCExp`riskPct;
        alphaNames;
        objective)}

// Project alpha allocation to satisfy PC constraints
// @param alloc - current alpha allocation (sums to 1)
// @param alphaPCExp - PC exposures of each alpha (nAlphas x k)
// @param pcLo - lower bounds on PC exposure
// @param pcHi - upper bounds on PC exposure
// @param maxIter - max projection iterations
// @return adjusted allocation
alphaProjectPC:{[alloc;alphaPCExp;pcLo;pcHi;maxIter]
    nAlphas:count alloc;
    k:count first alphaPCExp;
    getPCExp:{[alloc;alphaPCExp;i] sum alloc * alphaPCExp[;i]};

    // Helper to adjust allocation for one PC
    adjustPC:{[alloc;alphaPCExp;pcLo;pcHi;j]
        pcExp:sum alloc * alphaPCExp[;j];
        loadings:alphaPCExp[;j];
        posIdx:where loadings > 0;
        violated:0b;
        if[(pcExp < pcLo[j]) & 0 < count posIdx;
            gap:pcLo[j] - pcExp;
            lpos:loadings[posIdx];
            adj:0.1 * gap % 0.001 | max lpos;
            alloc[posIdx]:alloc[posIdx] + adj * lpos % 0.001 | sum lpos;
            violated:1b];
        if[(pcExp > pcHi[j]) & 0 < count posIdx;
            gap:pcExp - pcHi[j];
            lpos:loadings[posIdx];
            adj:0.1 * gap % 0.001 | max lpos;
            alloc[posIdx]:alloc[posIdx] - adj * lpos % 0.001 | sum lpos;
            violated:1b];
        (alloc;violated)};

    iter:0;
    while[iter < maxIter;
        violated:0b;
        j:0;
        while[j < k;
            res:adjustPC[alloc;alphaPCExp;pcLo;pcHi;j];
            alloc:res 0;
            if[res 1; violated:1b];
            j+:1];
        alloc:0f | alloc;
        if[0 < sum alloc; alloc:alloc % sum alloc];
        if[not violated; :alloc];
        iter+:1];
    alloc}

// Factor-neutral alpha optimization
// Optimize alphas while being neutral to specified PCs
// @param alphas - dict of alpha weights
// @param R - returns
// @param p - PCA result
// @param neutralPCs - list of PC indices to neutralize (0-indexed)
// @param cfg - config
alphaFactorNeutral:{[alphas;R;p;neutralPCs;cfg]
    cfg:(`objective`nIter!(`maxSharpe;5000)),cfg;

    k:p`k;
    // Create limits: zero for neutral PCs, large for others
    pcLimits:`lo`hi!(k#-1e10; k#1e10);
    pcLimits[`lo;neutralPCs]:0f;
    pcLimits[`hi;neutralPCs]:0f;

    // Add small tolerance for numerical stability
    pcLimits[`lo;neutralPCs]:-0.001;
    pcLimits[`hi;neutralPCs]:0.001;

    alphaOptimize[alphas;R;p;pcLimits;cfg]}

// Alpha optimization with target PC exposures
// @param alphas - dict of alpha weights
// @param R - returns
// @param p - PCA result
// @param pcTargets - target PC exposures (dict or vector)
// @param tolerance - tolerance around targets
// @param cfg - config
alphaTilted:{[alphas;R;p;pcTargets;tolerance;cfg]
    k:p`k;
    targets:$[99h=type pcTargets;
        value (`$"PC",/:string 1+til k)#pcTargets;
        k#pcTargets];
    tol:$[-9h=type tolerance; k#tolerance; k#tolerance];

    pcLimits:`lo`hi!(targets - tol; targets + tol);
    alphaOptimize[alphas;R;p;pcLimits;cfg]}

// Risk parity across alphas with PC constraints
// @param alphas - dict of alpha weights
// @param R - returns
// @param p - PCA result
// @param pcLimits - PC exposure limits
// @param cfg - config
alphaRiskParity:{[alphas;R;p;pcLimits;cfg]
    cfg:(`objective`nIter!(`riskParity;5000)),cfg;
    alphaOptimize[alphas;R;p;pcLimits;cfg]}

// Generate efficient frontier with PC constraints
// @param alphas - dict of alpha weights
// @param R - returns
// @param p - PCA result
// @param pcLimits - PC exposure limits
// @param nPoints - number of points on frontier
// @param cfg - config
// @return table with frontier points
alphaEfficientFrontier:{[alphas;R;p;pcLimits;nPoints;cfg]
    cfg:(`nIter!2000),cfg;

    // First find min variance and max return portfolios
    minVarResult:alphaOptimize[alphas;R;p;pcLimits;cfg,enlist[`objective]!enlist`minVar];
    maxRetResult:alphaOptimize[alphas;R;p;pcLimits;cfg,enlist[`objective]!enlist`maxRet];

    minRet:minVarResult`return;
    maxRet:maxRetResult`return;

    // Generate target returns
    targetRets:minRet + (til nPoints) * (maxRet - minRet) % nPoints - 1;

    // For each target, minimize variance subject to return >= target
    // Approximate by using maxSharpe with varying risk-free rate
    frontier:{[alphas;R;p;pcLimits;cfg;targetRet]
        // Use penalty method: maximize ret - lambda * var, adjusting lambda
        result:alphaOptimize[alphas;R;p;pcLimits;cfg,`objective`targetRet!(`maxSharpe;targetRet)];
        `targetReturn`actualReturn`volatility`sharpe`pcExposure`alphaWeights!(
            targetRet;
            result`return;
            result`volatility;
            result`sharpe;
            result`pcExposure;
            result`alphaWeights)
    }[alphas;R;p;pcLimits;cfg] each targetRets;

    flip `targetReturn`actualReturn`volatility`sharpe!(
        frontier[;`targetReturn];
        frontier[;`actualReturn];
        frontier[;`volatility];
        frontier[;`sharpe])}

// Comprehensive alpha optimization report
// @param result - output from alphaOptimize
// @param alphas - original alpha dict
// @param p - PCA result
alphaReport:{[result;alphas;p]
    -1 "================================================================================";
    -1 "  ALPHA PORTFOLIO OPTIMIZATION REPORT";
    -1 "================================================================================";
    -1 "";
    -1 "OBJECTIVE: ",string result`objective;
    -1 "";
    -1 "PORTFOLIO STATISTICS";
    -1 "  Expected Return:   ",string result`return;
    -1 "  Volatility:        ",string result`volatility;
    -1 "  Sharpe Ratio:      ",string result`sharpe;
    -1 "";
    -1 "ALPHA ALLOCATIONS";
    {-1 "  ",string[x],": ",string[100*y],"%"}' [key result`alphaWeights; value result`alphaWeights];
    -1 "";
    -1 "PC EXPOSURES (combined portfolio)";
    k:count result`pcExposure;
    {-1 "  PC",string[x+1],": exposure=",string[y`expos]," risk%=",string[y`riskPct]}' [til k; ([] expos:result`pcExposure; riskPct:result`pcRiskPct)];
    -1 "";
    -1 "TOP ASSET WEIGHTS (combined)";
    w:result`combinedAssetWeights;
    topIdx:5#idesc abs value w;
    {-1 "  ",string[x],": ",string[100*y],"%"}' [(key w) topIdx; (value w) topIdx];
    -1 "";
    result}

// Quick alpha optimization example
alphaOptExample:{[]
    -1 "=== Alpha Portfolio Optimization Example ===\n";

    // Generate sample data
    system "S 42";
    T:252; n:5;
    assets:`SPY`QQQ`IWM`TLT`GLD;

    mkt:T?0.01;
    R:([] dt:.z.d-reverse til T;
        SPY:mkt+T?0.005;
        QQQ:1.2*mkt+T?0.006;
        IWM:0.9*mkt+T?0.007;
        TLT:-0.3*mkt+T?0.004;
        GLD:0.1*mkt+T?0.005);

    p:pca[R;3;()!()];

    // Define 3 alpha strategies
    alpha1:assets!0.5 0.3 0.2 0.0 0.0;    // Momentum: heavy equities
    alpha2:assets!0.1 0.1 0.1 0.4 0.3;    // Defensive: bonds + gold
    alpha3:assets!0.25 0.25 0.25 0.15 0.1; // Balanced

    alphas:`Momentum`Defensive`Balanced!(alpha1;alpha2;alpha3);

    -1 "Individual Alpha PC Exposures:";
    {[p;name;w]
        expos:(pcExposure[w;p])`pcExposure;
        -1 "  ",string[name],": ",", " sv string expos
    }[p]' [key alphas; value alphas];
    -1 "";

    // Optimize with PC1 (market) exposure limit of 0.15
    -1 "Optimizing with PC1 limit of +/-0.15...";
    result:alphaOptimize[alphas;R;p;`lo`hi!(-0.15 -1 -1f; 0.15 1 1f);()!()];

    -1 "Optimal Allocations:";
    {-1 "  ",string[x],": ",string[100*y],"%"}' [key result`alphaWeights; value result`alphaWeights];
    -1 "";
    -1 "Combined Portfolio:";
    -1 "  Return: ",string result`return;
    -1 "  Vol:    ",string result`volatility;
    -1 "  Sharpe: ",string result`sharpe;
    -1 "";
    -1 "PC Exposures (constrained):";
    -1 "  PC1: ",string[result[`pcExposure;0]]," (limit: +/-0.15)";
    -1 "  PC2: ",string result[`pcExposure;1];
    -1 "  PC3: ",string result[`pcExposure;2];
    -1 "";

    // Compare to factor-neutral
    -1 "Factor-Neutral (PC1=0) Optimization:";
    resultNeutral:alphaFactorNeutral[alphas;R;p;enlist 0;()!()];
    -1 "  Allocations: ",", " sv {string[x],"=",string[100*y],"%"}' [key resultNeutral`alphaWeights; value resultNeutral`alphaWeights];
    -1 "  PC1 Exposure: ",string resultNeutral[`pcExposure;0];
    -1 "";

    -1 "Done!";}

// =============================================================================
// ALPHA LIST WRAPPER (for signal-based alpha tables)
// =============================================================================

// Optimize allocation across a list of alpha signal tables
// Each alpha table has columns: dt, sym, sig, prevSig, pxDiff
//   dt      - date/timestamp
//   sym     - symbol
//   sig     - current signal (position/weight)
//   prevSig - prior period signal (what was held)
//   pxDiff  - price change / return for the period
//
// @param alphaList   - list of alpha tables
// @param alphaNames  - symbol list of alpha names (e.g. `Mom`MR`Val)
// @param pcLimits    - PC limits: scalar, or `lo`hi dict
// @param cfg         - config dict with optional keys:
//                        `weightMethod - `avg`vol`gross`last (default `avg)
//                        `sigCol       - signal column name (default `sig)
//                        `retCol       - return column name (default `pxDiff)
//                        `symCol       - symbol column name (default `sym)
//                        `dtCol        - date column name (default `dt)
//                        `k            - number of PCs (default 0 = auto)
//                        `objective    - `maxSharpe`minVar`riskParity
//                        `nIter        - optimizer iterations
// @return optimization result dict
alphaListOptimize:{[alphaList;alphaNames;pcLimits;cfg]
    defaults:`sigCol`retCol`symCol`dtCol`prevSigCol`k`objective`nIter!(
        `sig;`pxDiff;`sym;`dt;`prevSig;0;`maxSharpe;5000);
    cfg:defaults,cfg;

    sc:cfg`sigCol; rc:cfg`retCol; yc:cfg`symCol; dc:cfg`dtCol; pc:cfg`prevSigCol;

    n:count alphaList;
    if[n<>count alphaNames; '"alphaList and alphaNames must have same length"];

    // Get all unique symbols across all alphas
    allSyms:asc distinct raze {[yc;t] distinct t yc}[yc] each alphaList;

    // Compute characteristic weights for each alpha: avg signal by symbol
    computeWeights:{[sc;yc;allSyms;t]
        // functional: select avg sc by yc from t
        r:0!?[t;();(enlist yc)!enlist yc;(enlist`v)!enlist(avg;sc)];
        w:r[yc]!r`v;
        w:allSyms#(allSyms!count[allSyms]#0f),w;
        w%sum abs w};
    charWeights:computeWeights[sc;yc;allSyms] each alphaList;
    alphas:alphaNames!charWeights;

    // Compute alpha returns: sum(prevSig * pxDiff) by dtCol, named after alpha
    computeAlphaRet:{[dc;pc;rc;t;nm] ?[t;();(enlist dc)!enlist dc;(enlist nm)!enlist(sum;(*;pc;rc))]};
    alphaRetTables:computeAlphaRet[dc;pc;rc]'[alphaList;alphaNames];

    // Build asset returns: pivot retCol by symCol
    combined:raze alphaList;
    // functional: select last rc by dc, yc from combined
    assetRetLong:0!?[combined;();(dc,yc)!(dc;yc);(enlist rc)!enlist(last;rc)];

    // Pivot to wide format using group-based approach
    allDates:asc distinct assetRetLong dc;
    gidx:group assetRetLong dc;
    pivotRow:{[t;yc;rc;allSyms;idx]
        r:t[yc][idx]!t[rc][idx];
        allSyms#(allSyms!count[allSyms]#0n),r};
    dicts:pivotRow[assetRetLong;yc;rc;allSyms] each gidx allDates;
    R:flip (enlist[dc]!enlist allDates),allSyms!flip value each dicts;

    // Run PCA (dc column is excluded automatically by pca)
    p:pca[R;cfg`k;()!()];

    // Run optimization
    optCfg:`objective`nIter!(cfg`objective;cfg`nIter);
    result:alphaOptimize[alphas;R;p;pcLimits;optCfg];

    // Join alpha return tables into wide format
    alphaRetWide:flip (enlist dc)!enlist allDates;
    i:0; while[i<count alphaNames; alphaRetWide:alphaRetWide lj alphaRetTables[i]; i+:1];

    // Compute stats for each alpha's return column
    computeStats:{[t;nm] ret:t nm; `ret`vol`sharpe!(avg ret;dev ret;avg[ret]%dev[ret]+1e-10)};
    alphaStatsDict:alphaNames!computeStats[alphaRetWide] each alphaNames;

    result,`alphaStats`alphaReturns`pca`charWeights!(alphaStatsDict;alphaRetWide;p;alphas)}

// Quick helper to view alpha signal table structure
// cfg is optional - pass ()!() or empty dict for defaults
alphaListInfo:{[alphaList;alphaNames;cfg]
    defaults:`symCol`dtCol!(`sym;`dt);
    cfg:defaults,$[99h=type cfg;cfg;()!()];
    yc:cfg`symCol; dc:cfg`dtCol;
    -1 "=== Alpha List Info ===";
    {[yc;dc;t;nm;i]
        -1 "";
        -1 string[nm]," (alpha ",string[i],"):";
        -1 "  Rows:    ",string count t;
        -1 "  Columns: ",", " sv string cols t;
        -1 "  Symbols: ",", " sv string asc distinct t yc;
        -1 "  Dates:   ",string[min t dc]," to ",string max t dc;
    }[yc;dc]'[alphaList;alphaNames;til count alphaList];
    -1 "";}

// Example with signal tables
alphaListExample:{[]
    -1 "=== Alpha List Optimization Example ===";
    -1 "";

    // Generate sample alpha signal tables
    system "S 42";
    dates:2024.01.01+til 100;
    syms:`SPY`QQQ`IWM`TLT`GLD;

    // Cross product of dates and syms
    crossData:dates cross syms;
    n:count crossData;

    // Alpha 1: Momentum - tends to be long equities
    alpha1:([] dt:crossData[;0]; sym:crossData[;1]);
    alpha1:update sig:0.3+n?0.5, prevSig:0.2+n?0.5, pxDiff:n?0.02-0.01 from alpha1;
    alpha1:update sig:sig-0.2, prevSig:prevSig-0.2 from alpha1 where sym in `TLT`GLD;

    // Alpha 2: Mean reversion - more balanced
    alpha2:([] dt:crossData[;0]; sym:crossData[;1]);
    alpha2:update sig:n?0.4-0.2, prevSig:n?0.4-0.2, pxDiff:n?0.02-0.01 from alpha2;

    // Alpha 3: Defensive - tends to be long bonds/gold
    alpha3:([] dt:crossData[;0]; sym:crossData[;1]);
    alpha3:update sig:-0.1+n?0.3, prevSig:-0.1+n?0.3, pxDiff:n?0.02-0.01 from alpha3;
    alpha3:update sig:sig+0.3, prevSig:prevSig+0.3 from alpha3 where sym in `TLT`GLD;

    alphaList:(alpha1;alpha2;alpha3);
    alphaNames:`Momentum`MeanRev`Defensive;

    // Show info
    alphaListInfo[alphaList;alphaNames;()!()];

    // Run optimization with PC1 (market) constraint
    -1 "Running optimization with PC1 limit +/-0.2...";
    result:alphaListOptimize[alphaList;alphaNames;`lo`hi!(-0.2 -0.5 -0.5f;0.2 0.5 0.5f);()!()];

    -1 "";
    -1 "=== Results ===";
    -1 "";
    -1 "Characteristic Weights per Alpha:";
    show result`charWeights;

    -1 "";
    -1 "Alpha Statistics:";
    show result`alphaStats;

    -1 "";
    -1 "Optimal Allocation:";
    show result`alphaWeights;

    -1 "";
    -1 "Combined Portfolio:";
    -1 "  Return: ",string result`return;
    -1 "  Vol:    ",string result`volatility;
    -1 "  Sharpe: ",string result`sharpe;

    -1 "";
    -1 "PC Exposures:";
    -1 "  ",", " sv {"PC",string[1+x],"=",string y}'[til count result`pcExposure;result`pcExposure];

    -1 "";
    -1 "Done!";
    result}

// =============================================================================
// REPORTING & VISUALIZATION
// =============================================================================

// Comprehensive PC risk report
// @param w - weights (dict or vector)
// @param p - PCA result
// @param name - portfolio name
report:{[w;p;name]
    expos:pcExposure[w;p];

    -1 "================================================================================";
    -1 "  PC RISK REPORT: ",string name;
    -1 "================================================================================";
    -1 "";
    -1 "PORTFOLIO SUMMARY";
    -1 "  Total PC Risk:     ",string[expos`totalRisk];
    -1 "  Number of Assets:  ",string p`n;
    -1 "  PCs Retained:      ",string p`k;
    -1 "";
    -1 "PC DECOMPOSITION";
    show pcRiskReport[w;p];
    -1 "";
    -1 "TOP LOADINGS PER PC";
    {[p;i]
        L:p`loadings;
        loadings:L[;i];
        ord:idesc abs loadings;
        top5:ord til 5&count ord;
        -1 "  PC",string[i+1],": ",", " sv {x[0],"(",x[1],")"}each flip (string p[`assets]top5;{$[x>0;"+";"-"],string abs x}each loadings top5)
    }[p] each til p`k;
    -1 "";
    `report`exposure`pca!(name;expos;p)}

// Track PC exposures over time
// @param weights - table with dt and weight columns, or dict of dt->weights
// @param rollingPCA - output from pcaRolling
// @return table with PC exposures over time
trackExposure:{[weights;rollingPCA]
    // Get common dates
    dates:exec dt from rollingPCA where valid;

    results:{[weights;rp;dt]
        pca:first select loadings, eigenvalues, assets from rp where dt=d, valid;
        if[0=count pca; :()];

        w:$[98h=type weights;
            first exec w from weights where dt=d;
            weights dt];
        if[0=count w; :()];

        // Build mini pca dict
        p:`loadings`eigenvalues`assets!(first pca`loadings;first pca`eigenvalues;first pca`assets);
        expos:pcExposure[w;p];

        ([] dt:enlist dt),flip (`$"PC",/:string 1+til count expos`pcExposure)!enlist each expos`pcExposure
    }[weights;rollingPCA] each dates;

    raze results}

// =============================================================================
// CROWDING DETECTION
// =============================================================================

// Compute crowding score for a set of strategies
// High crowding = many strategies loading on same PCs
// @param strategies - dict of (strategyName -> weights)
// @param p - PCA result
// @return dict with crowding metrics
crowdingScore:{[strategies;p]
    strats:key strategies;
    n:count strats;
    if[n<2; :(enlist`error)!enlist"Need at least 2 strategies"];

    // Extract weights as list of dicts (strategies value may be table if same keys)
    // When all strategy dicts have same keys, q converts to table
    // Table indexing t[i] returns a dict, so we just need to iterate
    weights:$[98h=type value strategies;
        {[t;i] t i}[value strategies] each til n;
        value strategies];

    // Get normalized PC exposures (unit vector in PC space)
    exps:{[p;w]
        pcExp:(pcExposure[w;p])`pcExposure;
        nrm:sqrt sum pcExp*pcExp;
        $[nrm>1e-10; pcExp%nrm; pcExp]
    }[p] each weights;

    // Pairwise cosine similarity (dot product of normalized vectors)
    cosSim:{[exps;n;i;j] $[i=j; 1f; sum exps[i]*exps[j]]};
    simMat:{[f;exps;n;i] f[exps;n;i] each til n}[cosSim;exps;n] each til n;

    // Off-diagonal average = average pairwise similarity
    offDiag:raze {[m;i] m[i] where not (til count m)=i}[simMat] each til n;
    avgSim:avg offDiag;

    // Herding index: how much strategies move together (0=diverse, 1=identical)
    herding:avgSim;

    // PC concentration: which PCs are most crowded
    expMat:flip exps;  // k x n_strats
    pcConcentration:avg each abs expMat;  // Average absolute exposure per PC

    // Dominant PC: the PC with highest average absolute exposure
    dominantPC:first idesc pcConcentration;

    // Diversification ratio: effective number of independent strategies
    // Using eigenvalues of strategy correlation matrix
    stratCorr:simMat;
    stratEig:eigenvaluesSimple stratCorr;
    effStrats:(sum stratEig) xexp 2 % sum stratEig*stratEig;

    `herdingIndex`avgPairwiseSim`dominantPC`pcConcentration`effectiveStrategies`correlationMatrix!(
        herding;avgSim;dominantPC;pcConcentration;effStrats;strats!simMat)}

// Simple eigenvalue computation for crowding (symmetric matrix)
eigenvaluesSimple:{[M]
    n:count M;
    evs:();
    Mwork:M;
    do[n;
        v:n?1.0; v:v%sqrt sum v*v;
        do[50; v2:Mwork mmu v; nrm:sqrt sum v2*v2; v:$[nrm>1e-10;v2%nrm;v]];
        ev:sum v * Mwork mmu v;
        evs,:ev;
        Mwork:Mwork - ev * v */: v
    ];
    evs}

// Identify crowded positions across strategies
// @param strategies - dict of (strategyName -> weights)
// @param p - PCA result
// @param threshold - correlation threshold for "crowded" (default 0.7)
// @return table of strategy pairs with high correlation
crowdedPairs:{[strategies;p;threshold]
    cs:crowdingScore[strategies;p];
    strats:key strategies;
    n:count strats;
    corr:cs`correlationMatrix;

    // Find pairs above threshold
    pairs:raze {[strats;corr;threshold;i]
        js:where (corr[i;] > threshold) & (til count corr) > i;
        if[0=count js; :()];
        ([] strat1:count[js]#strats[i]; strat2:strats js; correlation:corr[i;js])
    }[strats;value corr;threshold] each til n;

    `$[0=count pairs; ([] strat1:(); strat2:(); correlation:()); pairs]}

// Crowding report with recommendations
crowdingReport:{[strategies;p]
    cs:crowdingScore[strategies;p];
    strats:key strategies;

    -1 "================================================================================";
    -1 "  CROWDING ANALYSIS";
    -1 "================================================================================";
    -1 "";
    -1 "SUMMARY METRICS";
    -1 "  Herding Index:           ",$[cs[`herdingIndex]>0.7;"HIGH ";cs[`herdingIndex]>0.4;"MEDIUM ";"LOW    "],string cs`herdingIndex;
    -1 "  Effective Strategies:    ",string cs`effectiveStrategies," of ",string count strats;
    -1 "  Dominant PC:             PC",string 1+cs`dominantPC;
    -1 "";
    -1 "PC CONCENTRATION (avg absolute exposure)";
    {-1 "  PC",string[x+1],": ",string y}' [til count cs`pcConcentration; cs`pcConcentration];
    -1 "";
    -1 "STRATEGY CORRELATION MATRIX";
    show ([] strategy:strats),flip strats!value cs`correlationMatrix;
    -1 "";

    crowded:crowdedPairs[strategies;p;0.7];
    if[0 < count crowded;
        -1 "CROWDED PAIRS (correlation > 0.7):";
        show crowded
    ];

    cs}

// =============================================================================
// FACTOR ATTRIBUTION
// =============================================================================

// Compute PC returns from asset returns
// @param R - returns (T x n table or matrix)
// @param p - PCA result
// @return table with date and PC returns
pcReturns:{[R;p]
    isTable:98h=type R;
    Rmat:$[isTable; flip value flip (cols[R] except `dt`date`time)#R; R];
    dates:$[isTable; R`dt; til count Rmat];

    L:p`loadings;   // n x k
    mu:p`mu;
    sigma:p`sigma;

    // Standardize returns
    Rstd:flip (((flip Rmat) - mu) % sigma);

    // Project onto PCs: pcRet = Rstd * L
    pcRet:Rstd mmu L;  // T x k

    k:count p`eigenvalues;
    pcNames:`$"PC",/:string 1+til k;

    ([] dt:dates),'flip pcNames!flip pcRet}

// Attribute portfolio P&L to principal components
// @param R - returns table with dt column
// @param w - portfolio weights (dict or vector)
// @param p - PCA result
// @return table with total P&L and per-PC attribution
attributePnL:{[R;w;p]
    isTable:98h=type R;
    if[not isTable; '"R must be a table with dt column"];

    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;
    dates:R`dt;
    Rmat:flip value flip (cols[R] except `dt`date`time)#R;

    // Total portfolio return per period
    portRet:Rmat mmu wVec;

    // PC exposure of portfolio
    L:p`loadings;
    pcExp:(flip L) mmu wVec;  // k-vector

    // PC returns (standardized)
    mu:p`mu;
    sigma:p`sigma;
    Rstd:flip (((flip Rmat) - mu) % sigma);  // standardize before final flip
    pcRetStd:Rstd mmu L;  // T x k

    // Attributed P&L per PC
    k:count p`eigenvalues;
    attrib:pcRetStd *\: pcExp;  // T x k

    // Residual
    totalAttrib:sum each attrib;
    residual:portRet - totalAttrib;

    pcNames:`$"PC",/:string 1+til k;
    result:([] dt:dates; portReturn:portRet; totalAttributed:totalAttrib; residual:residual);
    result,'flip pcNames!flip attrib}

// Summarize attribution over period
// @param attrib - output from attributePnL
// @return dict with summary statistics
attributionSummary:{[attrib]
    pcCols:cols[attrib] except `dt`portReturn`totalAttributed`residual;

    total:sum attrib`portReturn;
    perPC:{[t;c] sum t c}[attrib] each pcCols;
    resid:sum attrib`residual;

    pctPC:perPC % total;
    pctResid:resid % total;

    `totalReturn`pcContribution`residualContribution`pcPct`residualPct!(
        total;pcCols!perPC;resid;pcCols!100*pctPC;100*pctResid)}

// Rolling attribution (attribution in rolling windows)
// @param R - returns
// @param w - weights
// @param p - PCA result
// @param window - window size
// @return table with rolling attribution sums
rollingAttribution:{[R;w;p;window]
    attrib:attributePnL[R;w;p];
    pcCols:cols[attrib] except `dt`portReturn`totalAttributed`residual;

    // Rolling sums
    rollSum:{[w;x] msum[w;x] - (w-1)#0n,msum[w-1;(w-1)#x]};

    ([] dt:attrib`dt;
        portReturn:window msum attrib`portReturn;
        residual:window msum attrib`residual),
    flip pcCols!{[w;t;c] w msum t c}[window;attrib] each pcCols}

// Attribution report
attributionReport:{[R;w;p;name]
    attrib:attributePnL[R;w;p];
    summary:attributionSummary attrib;
    pcCols:cols[attrib] except `dt`portReturn`totalAttributed`residual;

    -1 "================================================================================";
    -1 "  FACTOR ATTRIBUTION: ",string name;
    -1 "================================================================================";
    -1 "";
    -1 "PERIOD SUMMARY";
    -1 "  Total Return:      ",string summary`totalReturn;
    -1 "  Attributed:        ",string sum value summary`pcContribution;
    -1 "  Residual:          ",string summary`residualContribution;
    -1 "  Residual %:        ",string[summary`residualPct],"%";
    -1 "";
    -1 "PC CONTRIBUTIONS";
    {-1 "  ",string[x],": ",string[y]," (",string[z],"%)"}' [pcCols; value summary`pcContribution; value summary`pcPct];
    -1 "";

    // Time series stats
    -1 "TIME SERIES STATISTICS";
    -1 "  Periods:           ",string count attrib;
    -1 "  Best PC1 day:      ",string max attrib`PC1;
    -1 "  Worst PC1 day:     ",string min attrib`PC1;
    -1 "";

    summary}

// =============================================================================
// STRESS TESTING
// =============================================================================

// Apply PC shock and compute portfolio impact
// @param w - weights
// @param p - PCA result
// @param pcShocks - dict or vector of PC shocks (in stddev units)
// @return dict with stress results
pcStress:{[w;p;pcShocks]
    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;

    k:p`k;
    // Handle dict with PC names, vector, or scalar
    shocks:$[99h=type pcShocks;
        value (`$"PC",/:string 1+til k)#pcShocks;  // Extract by PC1, PC2, etc. keys
        k#pcShocks];  // Broadcast scalar or truncate/pad vector
    shocks:`float$shocks;

    L:p`loadings;
    ev:p`eigenvalues;
    sigma:p`sigma;

    // PC exposure
    pcExp:(flip L) mmu wVec;

    // Stressed PC returns = shock * sqrt(eigenvalue)
    pcStressedRet:shocks * sqrt ev;

    // Portfolio impact from each PC
    pcImpact:pcExp * pcStressedRet;

    // Total stressed return
    totalImpact:sum pcImpact;

    // Translate back to asset space for detailed view
    assetImpact:L mmu pcImpact;
    assetImpact:assetImpact * sigma;

    `totalImpact`pcImpact`pcShocks`pcExposure`assetImpact!(
        totalImpact;pcImpact;shocks;pcExp;p[`assets]!assetImpact)}

// Generate stress matrix: impact of 1-sigma shock to each PC
// @param w - weights
// @param p - PCA result
// @return table with sensitivities
stressMatrix:{[w;p]
    k:p`k;

    // Shock each PC individually
    stressOnePos:{[w;p;k;i] shocks:k#0f; shocks[i]:1f; s:pcStress[w;p;shocks]; `pc`shock`impact!(`$"PC",string i+1; 1f; s`totalImpact)};
    results:stressOnePos[w;p;k] each til k;

    // Combine shocks
    combined:pcStress[w;p;k#1f];
    allShock:`pc`shock`impact!(`ALL; 1f; combined`totalImpact);

    // Negative shocks
    stressOneNeg:{[w;p;k;i] shocks:k#0f; shocks[i]:-1f; s:pcStress[w;p;shocks]; `pc`shock`impact!(`$"PC",string[i+1],"_neg"; -1f; s`totalImpact)};
    resultsNeg:stressOneNeg[w;p;k] each til k;

    flip `pc`shock`impact!(
        results[;`pc],resultsNeg[;`pc],`ALL_neg`ALL;
        results[;`shock],resultsNeg[;`shock],-1 1f;
        results[;`impact],resultsNeg[;`impact],(neg combined`totalImpact),combined`totalImpact)}

// Historical stress: apply historical worst PC moves
// @param w - weights
// @param p - PCA result
// @param R - returns used for PCA (to find historical moves)
// @param percentile - percentile for worst case (default 0.01 = 1%)
// @return dict with historical stress scenarios
historicalStress:{[w;p;R;percentile]
    percentile:$[null percentile; 0.01; percentile];

    // Get PC returns
    pcRet:pcReturns[R;p];
    pcCols:cols[pcRet] except `dt;
    k:count pcCols;

    // Find percentile worst moves for each PC
    worstMoves:{[pct;col]
        sorted:asc col;
        n:count sorted;
        idx:"j"$pct*n;
        sorted idx
    }[percentile] each flip value flip pcCols#pcRet;

    // Find worst combined day
    scores:pcRet[pcCols];
    ev:p`eigenvalues;
    // Weighted score by eigenvalue
    combinedScore:sum each (flip value flip scores) *\: sqrt ev;
    worstIdx:first idesc neg combinedScore;
    worstDay:pcRet[worstIdx];
    worstDate:worstDay`dt;
    worstPCMoves:{[d;c] d c}[worstDay] each pcCols;

    // Apply stress scenarios
    pctileStress:pcStress[w;p;worstMoves % sqrt p`eigenvalues];
    historicStress:pcStress[w;p;worstPCMoves % sqrt p`eigenvalues];

    `percentile`percentileStress`worstDate`worstMoves`historicStress!(
        percentile;pctileStress;worstDate;pcCols!worstPCMoves;historicStress)}

// VaR and CVaR in PC space
// @param w - weights
// @param p - PCA result
// @param R - returns
// @param confidence - VaR confidence level (default 0.95)
// @return dict with VaR metrics
pcVaR:{[w;p;R;confidence]
    confidence:$[null confidence; 0.95; confidence];

    // Get portfolio returns
    wVec:$[99h=type w; w (p`assets); w];
    wVec:`float$wVec;
    Rmat:$[98h=type R; flip value flip (cols[R] except `dt`date`time)#R; R];
    portRet:Rmat mmu wVec;

    // Historical VaR
    sorted:asc portRet;
    n:count sorted;
    varIdx:"j"$(1-confidence)*n;
    histVaR:neg sorted varIdx;
    histCVaR:neg avg sorted til varIdx+1;

    // Parametric VaR (assuming normal)
    mu:avg portRet;
    sigma:dev portRet;
    zScore:$[confidence=0.95; 1.645; confidence=0.99; 2.326; 1.645];
    paramVaR:neg (mu - zScore*sigma);
    paramCVaR:neg (mu - sigma * {[z] (exp neg 0.5*z*z) % sqrt[2*3.14159] * 1-z}[zScore]);

    // Component VaR by PC
    pcRet:pcReturns[R;p];
    pcCols:cols[pcRet] except `dt;
    pcExp:(flip p`loadings) mmu wVec;

    // Marginal contribution to VaR
    pcVolContrib:pcExp * sqrt p`eigenvalues;
    mcVaR:zScore * pcVolContrib;

    `confidence`historicalVaR`historicalCVaR`parametricVaR`parametricCVaR`componentVaR!(
        confidence;histVaR;histCVaR;paramVaR;paramCVaR;pcCols!mcVaR)}

// Comprehensive stress report
stressReport:{[w;p;R;name]
    -1 "================================================================================";
    -1 "  STRESS TESTING: ",string name;
    -1 "================================================================================";
    -1 "";

    // Stress matrix
    -1 "SENSITIVITY TO 1-SIGMA PC SHOCKS";
    sm:stressMatrix[w;p];
    show select pc, shock, impact from sm where shock=1f;
    -1 "";

    // Historical stress
    hs:historicalStress[w;p;R;0.01];
    -1 "HISTORICAL STRESS (1% worst)";
    -1 "  Percentile stress impact: ",string hs[`percentileStress;`totalImpact];
    -1 "  Worst day: ",string hs`worstDate;
    -1 "  Worst day impact:         ",string hs[`historicStress;`totalImpact];
    -1 "";

    // VaR
    varResult:pcVaR[w;p;R;0.95];
    -1 "VALUE AT RISK (95%)";
    -1 "  Historical VaR:   ",string varResult`historicalVaR;
    -1 "  Historical CVaR:  ",string varResult`historicalCVaR;
    -1 "  Parametric VaR:   ",string varResult`parametricVaR;
    -1 "";
    -1 "COMPONENT VaR BY PC";
    {-1 "  ",string[x],": ",string y}'[key varResult`componentVaR; value varResult`componentVaR];
    -1 "";

    `stressMatrix`historicalStress`VaR!(sm;hs;varResult)}

// Scenario analysis: custom multi-factor shocks
// @param w - weights
// @param p - PCA result
// @param scenarios - dict of (scenarioName -> pcShocks dict)
// @return table with scenario impacts
scenarioAnalysis:{[w;p;scenarios]
    scenNames:key scenarios;

    results:{[w;p;name;shocks]
        s:pcStress[w;p;shocks];
        `scenario`totalImpact`pcImpact!(name;s`totalImpact;s`pcImpact)
    }[w;p]'[scenNames;value scenarios];

    flip `scenario`totalImpact!(results[;`scenario];results[;`totalImpact])}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Convert optimizer result to pcrisk-compatible format
fromOptimizer:{[result;assets]
    $[`assets in key result;
        result`assets;
        assets!result`weights]}

// Print loadings matrix nicely
showLoadings:{[p]
    k:p`k;
    pcNames:`$"PC",/:string 1+til k;
    ([] asset:p`assets),flip pcNames!flip p`loadings}

// Get top N assets by loading for each PC
topLoadings:{[p;n]
    k:p`k;
    {[p;n;i]
        L:p`loadings;
        loadings:L[;i];
        ord:idesc abs loadings;
        topN:ord til n&count ord;
        ([] pc:`$"PC",string i+1; asset:p[`assets]topN; loading:loadings topN)
    }[p;n] each til k}

// Quick usage example
example:{[]
    -1 "=== PC Risk Example ===\n";

    // Generate correlated returns
    system "S 42";
    n:5; T:252;
    assets:`SPY`QQQ`IWM`TLT`GLD;

    // Base returns with correlation structure
    mkt:T?0.01;
    R:([] dt:.z.d-reverse til T; SPY:mkt+T?0.005; QQQ:1.2*mkt+T?0.006; IWM:0.9*mkt+T?0.007; TLT:-0.3*mkt+T?0.004; GLD:0.1*mkt+T?0.005);

    // Compute PCA
    p:pca[R;3;()!()];
    -1 "PCA Results:";
    -1 "  Explained variance: ",", " sv string 100*p`explainedVar;
    -1 "";

    // Sample portfolio
    w:assets!0.3 0.25 0.2 0.15 0.1;
    -1 "Portfolio weights: ",", " sv {(string x),"=",(string y)}'[key w;value w];
    -1 "";

    // PC exposure
    expos:pcExposure[w;p];
    -1 "PC Exposures: ",", " sv string expos`pcExposure;
    -1 "PC Risk %:    ",", " sv string 100*expos`riskPct;
    -1 "";

    // Constrain PC1 (market) to 0.1
    wAdj:pcConstrain[w;p;0.1;()!()];
    expAdj:pcExposure[wAdj;p];
    -1 "After constraining PC1 to 0.1:";
    -1 "  Adjusted weights: ",", " sv {(string x),"=",(string y)}'[key wAdj;value wAdj];
    -1 "  New PC Exposures: ",", " sv string expAdj`pcExposure;
    -1 "";

    // --- CROWDING DETECTION ---
    -1 "=== Crowding Detection ===\n";

    // Create multiple strategies
    strat1:assets!0.4 0.35 0.25 0.0 0.0;   // Momentum: heavy equities
    strat2:assets!0.2 0.1 0.15 0.3 0.25;   // Value: more diversified
    strat3:assets!0.2 0.2 0.2 0.2 0.2;     // Balanced: equal weight
    strategies:`MomLong`ValLong`Balanced!(strat1;strat2;strat3);

    cs:crowdingScore[strategies;p];
    -1 "Herding Index: ",string[cs`herdingIndex]," (0=diverse, 1=identical)";
    -1 "Effective Strategies: ",string cs`effectiveStrategies;
    -1 "";

    // --- FACTOR ATTRIBUTION ---
    -1 "=== Factor Attribution ===\n";

    attrib:attributePnL[R;w;p];
    summary:attributionSummary attrib;
    -1 "Total Return: ",string summary`totalReturn;
    -1 "PC Contributions:";
    {-1 "  ",string[x],": ",string y}' [key summary`pcContribution; value summary`pcContribution];
    -1 "Residual: ",string summary`residualContribution;
    -1 "";

    // --- STRESS TESTING ---
    -1 "=== Stress Testing ===\n";

    // 1-sigma shock to each PC
    sm:stressMatrix[w;p];
    -1 "Sensitivity to 1-sigma PC shocks:";
    show select pc, impact from sm where shock=1f, not pc like "*neg*", not pc=`ALL;
    -1 "";

    // VaR
    varResult:pcVaR[w;p;R;0.95];
    -1 "Value at Risk (95%):";
    -1 "  Historical VaR:  ",string varResult`historicalVaR;
    -1 "  Parametric VaR:  ",string varResult`parametricVaR;
    -1 "";

    // Custom scenario
    crash:`PC1`PC2`PC3!-3 -1 0f;     // 3-sigma market drop
    flight:`PC1`PC2`PC3!-1 0 2f;     // Safety assets rally
    scenarios:`MarketCrash`FlightToQuality!(crash;flight);
    scen:scenarioAnalysis[w;p;scenarios];
    -1 "Custom Scenarios:";
    show scen;
    -1 "";

    -1 "Done!";}

// Usage/help
usage:{[]
    -1 "=============================================================================";
    -1 "  .pcrisk - Principal Component Risk Management";
    -1 "=============================================================================";
    -1 "";
    -1 "CORE PCA:";
    -1 "  pca[R;k;cfg]           - Compute PCA (k=0 for all, cfg=`scale`excludeCols)";
    -1 "  pcaFromCov[C;k;assets] - PCA from covariance matrix";
    -1 "  pcaRolling[R;k;w;cfg]  - Rolling/expanding PCA";
    -1 "";
    -1 "EXPOSURE MAPPING:";
    -1 "  pcExposure[w;pca]      - Map weights to PC space";
    -1 "  pcRiskReport[w;pca]    - Detailed per-PC risk table";
    -1 "";
    -1 "MULTI-STRATEGY:";
    -1 "  aggregateExposure[strats;allocs;pca] - Aggregate multiple strategies";
    -1 "  strategyCorrelation[strats;pca]      - Correlation matrix in PC space";
    -1 "";
    -1 "CROWDING DETECTION:";
    -1 "  crowdingScore[strats;pca]          - Herding index, effective strategies";
    -1 "  crowdedPairs[strats;pca;thresh]    - Find correlated strategy pairs";
    -1 "  crowdingReport[strats;pca]         - Full crowding analysis report";
    -1 "";
    -1 "FACTOR ATTRIBUTION:";
    -1 "  pcReturns[R;pca]                   - Compute PC returns from assets";
    -1 "  attributePnL[R;w;pca]              - Decompose P&L by PC";
    -1 "  attributionSummary[attrib]         - Summarize period attribution";
    -1 "  rollingAttribution[R;w;pca;window] - Rolling attribution";
    -1 "  attributionReport[R;w;pca;name]    - Full attribution report";
    -1 "";
    -1 "STRESS TESTING:";
    -1 "  pcStress[w;pca;shocks]             - Apply PC shocks";
    -1 "  stressMatrix[w;pca]                - 1-sigma sensitivity to each PC";
    -1 "  historicalStress[w;pca;R;pctl]     - Historical worst-case scenarios";
    -1 "  pcVaR[w;pca;R;conf]                - VaR/CVaR with PC decomposition";
    -1 "  scenarioAnalysis[w;pca;scenarios]  - Custom multi-factor scenarios";
    -1 "  stressReport[w;pca;R;name]         - Comprehensive stress report";
    -1 "";
    -1 "CONSTRAINTS:";
    -1 "  pcConstrain[w;pca;limits;cfg] - Constrain PC exposures";
    -1 "  pcNeutralize[w;pca;pcs]       - Zero out specific PC exposures";
    -1 "  pcHedge[w;pca;pcs;hedgeAssets]- Compute hedge for specific PCs";
    -1 "";
    -1 "OPTIMIZATION:";
    -1 "  maxSharpePC[R;rf;pca;limits;cfg] - Max Sharpe with PC constraints";
    -1 "  riskParityPC[R;pca;cfg]          - Risk parity in PC space";
    -1 "";
    -1 "ALPHA PORTFOLIO OPTIMIZATION:";
    -1 "  alphaOptimize[alphas;R;pca;pcLimits;cfg] - Optimize alpha allocation with PC limits";
    -1 "    cfg`objective: `maxSharpe`minVar`riskParity`maxRet";
    -1 "  alphaFactorNeutral[alphas;R;pca;neutralPCs;cfg] - Neutral to specified PCs";
    -1 "  alphaTilted[alphas;R;pca;pcTargets;tol;cfg]     - Target specific PC exposures";
    -1 "  alphaRiskParity[alphas;R;pca;pcLimits;cfg]      - Risk parity across alphas";
    -1 "  alphaEfficientFrontier[alphas;R;pca;pcLimits;n;cfg] - Efficient frontier";
    -1 "  alphaReport[result;alphas;pca]                  - Optimization report";
    -1 "  alphaOptExample[]                               - Alpha optimization example";
    -1 "";
    -1 "ALPHA LIST WRAPPER (for signal tables):";
    -1 "  alphaListOptimize[alphaList;names;pcLimits;cfg] - Optimize list of alpha tables";
    -1 "    Each alpha table has: dt, sym, sig, prevSig, pxDiff";
    -1 "    cfg`weightMethod: `avg`vol`gross`last (how to compute characteristic weights)";
    -1 "  alphaListInfo[alphaList;names;cfg]               - Display alpha table info";
    -1 "  alphaListExample[]                              - Signal table example";
    -1 "";
    -1 "REPORTING:";
    -1 "  report[w;pca;name]    - Comprehensive PC risk report";
    -1 "  showLoadings[pca]     - Display loadings matrix";
    -1 "  topLoadings[pca;n]    - Top N assets per PC";
    -1 "";
    -1 "EXAMPLE:";
    -1 "  .pcrisk.example[]";
    -1 "";}

help:usage

// Startup message
-1 "Loaded .pcrisk namespace v0.2.0";
-1 "PC-based portfolio risk decomposition and management";
-1 "Features: PCA, exposure mapping, crowding detection, factor attribution, stress testing";
-1 "Run .pcrisk.usage[] for function list";

\d .

// =============================================================================
// END
// =============================================================================
