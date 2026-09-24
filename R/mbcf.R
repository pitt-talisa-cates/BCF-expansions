.ident <- function(...){
  # courtesy https://stackoverflow.com/questions/19966515/how-do-i-test-if-three-variables-are-equal-r
  args <- c(...)
  if( length( args ) > 2L ){
    #  recursively call ident()
    out <- c( identical( args[1] , args[2] ) , .ident(args[-1]))
  }else{
    out <- identical( args[1] , args[2] )
  }
  return( all( out ) )
}

.cp_quantile = function(x, num=10000, cat_levels=8){
  nobs = length(x)
  nuniq = length(unique(x))
  
  if(nuniq==1) {
    ret = x[1]
    warning("A supplied covariate contains a single distinct value.")
  } else if(nuniq < cat_levels) {
    xx = sort(unique(x))
    ret = xx[-length(xx)] + diff(xx)/2
  } else {
    q = approxfun(sort(x),quantile(x,p = 0:(nobs-1)/nobs))
    ind = seq(min(x),max(x),length.out=num)
    ret = q(ind)
  }
  
  return(ret)
}

# Draw n samples from N(mu, sigma^2) truncated to the interval (lo, hi)
# Uses the inverse-CDF method; lo and hi may be -Inf / +Inf.
.rtnorm_interval <- function(n, mu = 0, sigma = 1, lo = -Inf, hi = Inf) {
  plo <- pnorm(lo, mean = mu, sd = sigma)
  phi <- pnorm(hi, mean = mu, sd = sigma)
  u   <- runif(n, min = plo, max = phi)
  qnorm(u, mean = mu, sd = sigma)
}

#' Fit Bayesian Causal Forests
#'
#' @references Hahn, Murray, and Carvalho(2017). Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects.
#'  https://arxiv.org/abs/1706.09523. (Call citation("bcf") from the
#' command line for citation information in Bibtex format.)
#'
#' @details Fits the Bayesian Causal Forest model (Hahn et. al. 2018): For a response
#' variable y, binary treatment z, and covariates x,
#' \deqn{y_i = \mu(x_i, \pi_i) + \tau(x_i, \pi_i)z_i + \epsilon_i}
#' where \eqn{\pi_i} is an (optional) estimate of the propensity score \eqn{\Pr(Z_i=1 | X_i=x_i)} and
#' \eqn{\epsilon_i \sim N(0,\sigma^2)}
#'
#' Some notes:
#' \itemize{
#'    \item x_control and x_moderate must be numeric matrices. See e.g. the makeModelMatrix function in the
#'    dbarts package for appropriately constructing a design matrix from a data.frame
#'    \item sd_control and sd_moderate are the prior SD(mu(x)) and SD(tau(x)) at a given value of x (respectively). If
#'    use_muscale = FALSE, then this is the parameter \eqn{\sigma_\mu} from the original BART paper, where the leaf parameters
#'    have prior distribution \eqn{N(0, \sigma_\mu/m)}, where m is the number of trees.
#'    If use_muscale=TRUE then sd_control is the prior median of a half Cauchy prior for SD(mu(x)). If use_tauscale = TRUE,
#'    then sd_moderate is the prior median of a half Normal prior for SD(tau(x)).
#'    \item By default the prior on \eqn{\sigma^2} is calibrated as in Chipman, George and McCulloch (2008).
#'
#'
#' }
#' @param y Response variable
#' @param z Treatment variable
#' @param x_control Design matrix for the "prognostic" function mu(x)
#' @param x_moderate Design matrix for the covariate-dependent treatment effects tau(x)
#' @param pihat Length n estimates of
#' @param nburn Number of burn-in MCMC iterations
#' @param nsim Number of MCMC iterations to save after burn-in
#' @param nthin Save every nthin'th MCMC iterate. The total number of MCMC iterations will be nsim*nthin + nburn.
#' @param update_interval Print status every update_interval MCMC iterations
#' @param ntree_control Number of trees in mu(x)
#' @param sd_control SD(mu(x)) marginally at any covariate value (or its prior median if use_muscale=TRUE)
#' @param base_control Base for tree prior on mu(x) trees (see details)
#' @param power_control Power for the tree prior on mu(x) trees
#' @param ntree_moderate Number of trees in tau(x)
#' @param sd_moderate SD(tau(x)) marginally at any covariate value (or its prior median if use_tauscale=TRUE)
#' @param base_moderate Base for tree prior on tau(x) trees (see details)
#' @param power_moderate Power for the tree prior on tau(x) trees (see details)
#' @param nu Degrees of freedom in the chisq prior on \eqn{sigma^2}
#' @param lambda Scale parameter in the chisq prior on \eqn{sigma^2}
#' @param sigq Calibration quantile for the chisq prior on \eqn{sigma^2}
#' @param sighat Calibration estimate for the chisq prior on \eqn{sigma^2}
#' @param include_pi Takes values "control", "moderate", "both" or "none". Whether to
#' include pihat in mu(x) ("control"), tau(x) ("moderate"), both or none. Values of "control"
#' or "both" are HIGHLY recommended with observational data.
#' @param use_muscale Use a half-Cauchy hyperprior on the scale of mu.
#' @param use_tauscale Use a half-Normal prior on the scale of tau.
#' @param compute_shap Master switch for mTreeSHAP. Default FALSE, since it adds
#'   runtime for users who do not need attributions.
#' @param shap_mod Compute SHAP for the treatment effect function tau(x). Default TRUE.
#' @param shap_con Compute SHAP for the prognostic function mu(x). Default FALSE.
#'   The two are computed independently and returned separately, so either or
#'   both can be requested.
#' @param shap_thin Compute SHAP every \code{shap_thin}-th post-burn-in iteration.
#'   This is NOT \code{nthin}: \code{nthin} controls which draws of the chain are
#'   saved, \code{shap_thin} controls how often SHAP is calculated from the chain,
#'   and the two grids need not agree. Node cover is recomputed inside the SHAP
#'   call and is never read by the sampler, so the chain is unaffected by the
#'   value of \code{shap_thin}. Default 10.
#' @param shap_var Also accumulate sums of squares, giving the posterior variance
#'   of each SHAP value. Costs one extra N x p matrix per requested function.
#' @param shap_store_draws Keep every computed SHAP draw (N x p x number of SHAP
#'   draws) for credible intervals. Off by default; the memory cost is reported
#'   as a warning when it exceeds 1 GB.
#' @param shap_check At every SHAP draw, record the worst local-accuracy error
#'   max_i |sum_j phi_ij + baseline - f(x_i)|. Should be ~1e-10; useful as a
#'   regression test after changes to the tree code.
#' @param max_ordinal_levels Maximum number of unique values allowed for a variable declared as
#'   "ordinal" in na_reg_type. If a variable exceeds this count the function stops with an
#'   informative message suggesting the user collapse categories. Default 25.
#' @return A list with elements
#' \item{tau}{\code{nsim} by \code{n} matrix of posterior samples of individual treatment effects}
#' \item{mu}{\code{nsim} by \code{n} matrix of posterior samples of individual treatment effects}
#' \item{sigma}{Length \code{nsim} vector of posterior samples of sigma}
#' \item{shap_mod, shap_con}{Present when \code{compute_shap=TRUE}. Each is a
#'   list with \code{mean} and \code{mean_abs} (N x p posterior mean SHAP and
#'   mean |SHAP|, in the units of \code{tau_post} / \code{mu_post}, rows and
#'   columns in the original input order), \code{var} (if \code{shap_var}),
#'   \code{mean_abs_obs} and \code{mean_abs_mis} (mean |SHAP| split by whether
#'   that covariate is observed for that row),
#'   \code{baseline} (one value per SHAP draw), \code{draws} (if
#'   \code{shap_store_draws}), \code{max_err} (if \code{shap_check}), plus
#'   \code{n_draws}, \code{thin} and \code{iters}. See \code{shap_summary()},
#'   \code{plot_shap_beeswarm()}, \code{plot_shap_importance()} and
#'   \code{plot_shap_waterfall()}.}
#' @examples
#'\donttest{
#'
#' # data generating process
#' p = 3 #two control variables and one moderator
#' n = 250
#' #
#' set.seed(1)
#'
#' x = matrix(rnorm(n*p), nrow=n)
#'
#' # create targeted selection
#' q = -1*(x[,1]>(x[,2])) + 1*(x[,1]<(x[,2]))
#'
#' # generate treatment variable
#' pi = pnorm(q)
#' z = rbinom(n,1,pi)
#'
#' # tau is the true (homogeneous) treatment effect
#' tau = (0.5*(x[,3] > -3/4) + 0.25*(x[,3] > 0) + 0.25*(x[,3]>3/4))
#'
#' # generate the response using q, tau and z
#' mu = (q + tau*z)
#'
#' # set the noise level relative to the expected mean function of Y
#' sigma = diff(range(q + tau*pi))/8
#'
#' # draw the response variable with additive error
#' y = mu + sigma*rnorm(n)
#'
#' # If you didn't know pi, you would estimate it here
#' pihat = pnorm(q)
#'
#' bcf_fit = bcf(y, z, x, x, pihat, nburn=2000, nsim=2000)
#'
#' # Get posterior of treatment effects
#' tau_post = bcf_fit$tau
#' tauhat = colMeans(tau_post)
#' plot(tau, tauhat); abline(0,1)
#'
#'}
#'\dontshow{
#'
#' # data generating process
#' p = 3 #two control variables and one moderator
#' n = 250
#' #
#' set.seed(1)
#'
#' x = matrix(rnorm(n*p), nrow=n)
#'
#' # create targeted selection
#' q = -1*(x[,1]>(x[,2])) + 1*(x[,1]<(x[,2]))
#'
#' # generate treatment variable
#' pi = pnorm(q)
#' z = rbinom(n,1,pi)
#'
#' # tau is the true (homogeneous) treatment effect
#' tau = (0.5*(x[,3] > -3/4) + 0.25*(x[,3] > 0) + 0.25*(x[,3]>3/4))
#'
#' # generate the response using q, tau and z
#' mu = (q + tau*z)
#'
#' # set the noise level relative to the expected mean function of Y
#' sigma = diff(range(q + tau*pi))/8
#'
#' # draw the response variable with additive error
#' y = mu + sigma*rnorm(n)
#'
#' pihat = pnorm(q)
#'
#' bcf_fit = bcf(y, z, x, x, pihat, nburn=10, nsim=10)
#'
#' # Get posterior of treatment effects
#' tau_post = bcf_fit$tau
#' tauhat = colMeans(tau_post)
#' plot(tau, tauhat); abline(0,1)
#'
#'}

mbcf <- function(y, 
                 z, 
                 x, 
                 mod_cols = 1:ncol(x), 
                 x_est=NULL,
                 
                 nburn, 
                 nsim, 
                 nthin = 1, 
                 update_interval = 100,
                 ntree_control = 200,
                 
                 trt_init = 1.0,   #default initial value for tau
                 
                 sd_control = 2*sd(y, na.rm=TRUE),
                 base_control = 0.95,
                 power_control = 2,
                 ntree_moderate = 50,
                 sd_moderate = sd(y, na.rm=TRUE),
                 
                 min_node=5,
                 
                 base_moderate = 0.25,
                 power_moderate = 3,
                 nu = 3, 
                 lambda = NULL, 
                 sigq = .9, 
                 
                 sighat = NULL,
                 use_muscale=TRUE, 
                 use_tauscale=FALSE,
                 na_mia=TRUE,
                 na_seqreg=FALSE,
                 check_node_missing=TRUE,
                 
                 na_reg_type=rep("lm",ncol(x)),
                 na_prob_pm_con=0.25,
                 na_prob_pm_mod=0.25,
                 num_breaks=10000,
                 max_ordinal_levels=25,
                 
                 compute_shap=FALSE,
                 shap_mod=TRUE,
                 shap_con=FALSE,
                 shap_thin=10,
                 shap_var=FALSE,
                 shap_store_draws=FALSE,
                 shap_check=FALSE,
                 
                 bugtest=FALSE) {
  #browser()
  if (!is.matrix(x)) stop("x must be a matrix")
  if (!is.null(x_est) && !is.matrix(x_est)) stop("x_est must be a matrix when provided")
  
  #------------------------------------------------------------
  # Missing outcomes are not supported. Drawing y from its fully conditional
  # posterior was tried and gave very poor interval coverage for tau, so the
  # feature was dropped rather than left available as a trap. yrr is kept (all
  # zeros) because mbcf_cpp() still takes it; the imputation branch in
  # bcf_clean_overpar.cpp is therefore never entered.
  #------------------------------------------------------------
  if (any(is.na(y))) {
    stop("Missing values in y. mbcf() does not support missing outcomes: ",
         sum(is.na(y)), " of ", length(y), " training outcomes are NA. ",
         "Drop these rows (they can be supplied through x_est if you want ",
         "predictions for them) before calling mbcf().")
  }
  yrr <- integer(length(y))
  n_ymiss <- 0L
  
  rr = 1*is.na(x)
  if(!is.null(x_est)){
    rr_est = 1*is.na(x_est)
  } else {
    rr_est <- matrix(0, 0, 0)
  }
  
  if (is.null(x_est)) {
    x_est_ord  <- matrix(0, 0, 0)  # true empty
    rr_est_ord <- matrix(0, 0, 0)
  } else {
    x_est_ord  <- x_est
    rr_est_ord <- rr_est
  }
  
  wmv<-rep(FALSE,ncol(x))
  wmv[mod_cols]<-TRUE
  
  #VBT: Pi is input; not estimated using BART that I can see
  #pihat = as.matrix(pihat)
  
  # VBT: Check that the dimensions are congruent
  if( !.ident(length(y),
              length(z),
              nrow(x)
              #nrow(pihat)
  )
  ) {
    stop("Data size mismatch. The following should all be equal:
         length(y): ", length(y), "\n",
         "length(z): ", length(z), "\n",
         "nrow(x): ", nrow(x), "\n"
         #"nrow(pihat): ", nrow(pihat),"\n"
    )
  }
  
  if (!is.null(x_est) && !.ident(ncol(x[, mod_cols, drop=FALSE]), ncol(x_est))) {
    stop("Data size mismatch. ncol(x[,mod_cols]): ",ncol(x[,mod_cols]),"\n",
         "ncol(x_est): ",ncol(x_est),"\n")
  }
  
  # VBT: Assorted checks
  if(any(is.na(z))) stop("Missing values in z")
  if((sum(rr) + sum(rr_est))>0 & (na_mia==FALSE & na_seqreg==FALSE)) {
    #browser()
    stop("Missing values in training and/or test datasets, but method for handling missing data not chosen")
  }
  if((sum(rr) + sum(rr_est))==0 & (na_mia==TRUE | na_seqreg==TRUE)) {
    warning("na_mia or na_seqreg will be ignored; no missing datapoints found in train or test datasets")
    na_mia=FALSE
    na_seqreg=FALSE
  }
  #if(any(is.na(pihat))) stop("Missing values in pihat")
  if(any(!is.finite(y[!is.na(y)]))) stop("Non-numeric values in y")
  if(any(!is.finite(z))) stop("Non-numeric values in z")
  #if(any(!is.finite(x))) stop("Non-numeric values in x")
  #if(any(!is.finite(pihat))) stop("Non-numeric values in pihat")
  if(!all(sort(unique(z)) == c(0,1))) stop("z must be a vector of 0's and 1's, with at least one of each")
  if(length(unique(y[!is.na(y)]))<5) warning("y appears to be discrete")
  if(nburn<0) stop("nburn must be non-negative")
  if(nsim<1) stop("nsim must be positive")
  if(nthin<1) stop("nthin must be positive")
  if(nthin>nsim) stop("nthin must be <= nsim")
  #if(nburn<100) warning("A low (<100) value for nburn was supplied")
  if(na_seqreg==TRUE & is.null(na_reg_type)) warning("Default regression types used for all missing covariates")
  
  #------------------------------------------------------------
  # mTreeSHAP argument checks
  #------------------------------------------------------------
  if(compute_shap){
    if(!shap_mod & !shap_con) stop("compute_shap=TRUE but both shap_mod and shap_con are FALSE")
    if(shap_thin < 1) stop("shap_thin must be a positive integer")
    shap_thin <- as.integer(shap_thin)
    n_shap <- ceiling((nsim*nthin)/shap_thin)
    if(n_shap < 1) stop("shap_thin is larger than nsim*nthin; no SHAP draws would be computed")
    if(shap_store_draws){
      p_store <- (if(shap_mod) length(mod_cols) else 0) + (if(shap_con) ncol(x) else 0)
      gb <- 8 * nrow(x) * p_store * n_shap / 2^30
      if(gb > 1) warning("shap_store_draws=TRUE will allocate about ",
                         round(gb,1), " GB of SHAP draws (", nrow(x), " rows x ",
                         p_store, " columns x ", n_shap, " draws). Consider a larger ",
                         "shap_thin, or shap_var=TRUE instead, which gives posterior ",
                         "means and variances without storing draws.")
    }
  } else {
    n_shap <- 0
  }
  
  #------------------------------------------------------------
  # Global collinearity check for sequential regression imputation
  # Runs before any imputation; uses only observed data.
  # TODO: extend to per-variable sequential collinearity diagnostics
  #       (condition number of each left-hand submatrix as j increases)
  #------------------------------------------------------------
  if (na_seqreg) {
    
    # Primary check: pairwise Spearman correlation on observed data.
    # Robust to mixed variable types (continuous, binary, ordinal) and
    # works with arbitrary missingness patterns via pairwise complete obs.
    cor_pw <- cor(x, method = "spearman", use = "pairwise.complete.obs")
    
    if (any(is.na(cor_pw))) {
      warning("Some variable pairs have no overlapping observed values; ",
              "pairwise Spearman collinearity check is incomplete.")
    } else {
      high_cor <- which(abs(cor_pw) > 0.9 & lower.tri(cor_pw), arr.ind = TRUE)
      if (nrow(high_cor) > 0) {
        pairs_str <- apply(high_cor, 1, function(idx) {
          nm <- colnames(x)
          if (is.null(nm)) nm <- paste0("x", seq_len(ncol(x)))
          paste0(nm[idx[1]], "-", nm[idx[2]])
        })
        warning("High pairwise Spearman correlation (|r| > 0.9) detected among: ",
                paste(pairs_str, collapse = ", "),
                ". Sequential regression imputation models may be unstable. ",
                "Consider removing or combining collinear covariates before calling mbcf().")
      }
    }
    
    # Secondary check: condition number on marginally-imputed complete matrix.
    # Missing values are filled by random draws from each variable's empirical
    # marginal distribution the least informative imputation possible, used
    # only so kappa() can run. Conservative: random marginal imputation tends
    # to dilute correlations, so a warning here means true collinearity is at
    # least as bad.
    x_marg_imp <- x
    for (jj in seq_len(ncol(x))) {
      miss_jj <- is.na(x_marg_imp[, jj])
      if (any(miss_jj)) {
        obs_vals <- x[!is.na(x[, jj]), jj]
        x_marg_imp[miss_jj, jj] <- sample(obs_vals, size = sum(miss_jj), replace = TRUE)
      }
    }
    cn <- kappa(cor(x_marg_imp, method = "spearman"), exact = FALSE)
    if (cn > 30) {
      warning("Condition number of marginally-imputed Spearman correlation matrix is ",
              round(cn, 1), ". This suggests multivariate collinearity beyond individual ",
              "pairs and may destabilize sequential regression imputation models.")
    }
    
  }
  
  na_first_j<-0
  x_ord<-x
  rr_ord<-rr
  wmv_ord<-wmv
  na_xparams<-list()
  
  if(!is.null(x_est)){
    x_est_ord<-x_est
    rr_est_ord<-rr_est
  }
  
  na_seqreg_modtyp <-rep(0,ncol(x))
  
  # identity unless the missingness-ordering branch below overwrites it; having it
  # always defined lets the vi_mod / SHAP un-permutation use one code path
  perm_miss <- seq_len(ncol(x))
  
  if(na_seqreg | na_mia){
    
    #VBT: Always perform initial imputation, even if all we're doing is mia
    # This way I avoid problems when I pack x into storage
    
    # VBT: permute columns so that missingness increases from left to right.
    # If there are variables in x_est with missing (but are complete in x), 
    # we want to set up imputation models for those too
    
    nmiss<-apply(x,2,function(x){sum(is.na(x))})
    anymiss<-1*(nmiss>0)
    
    if (!is.null(x_est)){
      nmiss_est<-apply(x_est,2,function(x){sum(is.na(x))})
      wn<-which(c(nmiss>0) != c(nmiss_est>0) )
      if(length(wn)>0){  
        if(any(anymiss[wn] !=1)){
          getimp<-which(anymiss[wn] != 1)
          for(v in getimp){
            nmiss[v]<-1
            nmiss[anymiss==1]<-nmiss[anymiss==1]+1
          }
        }
      }
    }
    
    perm_miss<-order(nmiss,decreasing=FALSE)
    x_ord<-x[,perm_miss, drop=FALSE]
    rr_ord<-rr[,perm_miss, drop=FALSE]
    wmv_ord<-wmv[perm_miss]
    nmiss_ord<-nmiss[perm_miss]
    anymiss_ord<-anymiss[perm_miss]
    na_reg_type_ord<-na_reg_type[perm_miss]
    xds<-as.data.frame(x_ord)
    
    na_row_list<-list()
    k=1
    for(j in which(nmiss_ord>0)){
      na_row_list[[j]]<-is.na(xds[,j])
    }
    
    #------------------------------------------------------------
    # Test set category check for ordinal variables:
    # All values observed in x_est must be present in the training
    # set for ordinal variables, because cutpoints are estimated
    # from the training distribution only.
    #------------------------------------------------------------
    if (!is.null(x_est)) {
      for (j in which(na_reg_type_ord == "ordinal")) {
        train_levels <- sort(unique(x_ord[, j][!is.na(x_ord[, j])]))
        test_vals    <- unique(x_est_ord[, j][!is.na(x_est_ord[, j])])
        novel        <- setdiff(test_vals, train_levels)
        if (length(novel) > 0) {
          vname <- if (!is.null(colnames(x))) colnames(x_ord)[j] else paste0("x", j)
          stop("Ordinal variable '", vname, "' has values in x_est not present in the ",
               "training set: ", paste(novel, collapse = ", "), ". ",
               "Please recode or collapse categories so that x_est contains only ",
               "values observed in x before calling mbcf().")
        }
      }
    }
    
    #------------------------------------------------------------
    # Rough pre-fill of NAs before sequential lm fits.
    # For lm: draw from N(mean, sd) of observed values.
    # For probit: draw from Bernoulli with p = observed mean.
    # For ordinal: sample uniformly from observed category values
    #   (marginal empirical distribution, no structure assumed).
    # This fill is only to make lm() run without NA issues;
    # proper initialization happens in the loop below.
    #------------------------------------------------------------
    for(j in which(nmiss_ord>0)){
      nm0 <- xds[!na_row_list[[j]], j]
      mean0 <- mean(nm0)
      sd0   <- sd(nm0)
      #cat("j=", j, colnames(xds)[j], "mean0=", mean0, "sd0=", sd0, "\n")  # ADD THIS
      if (na_reg_type_ord[j] == "lm"){
        xds[na_row_list[[j]], j] <- rnorm(sum(na_row_list[[j]]), mean0, sd0)
      } else if (na_reg_type_ord[j] == "probit"){
        xds[na_row_list[[j]], j] <- rbinom(sum(na_row_list[[j]]), 1, prob = mean0)
      } else if (na_reg_type_ord[j] == "ordinal"){
        xds[na_row_list[[j]], j] <- sample(nm0, size = sum(na_row_list[[j]]), replace = TRUE)
      }
    }
    
    if (!is.null(x_est)) {
      x_est_ord  <- x_est[, perm_miss, drop = FALSE]
      rr_est_ord <- rr_est[,  perm_miss, drop = FALSE]
      nmiss_est_ord <- nmiss_est[perm_miss]
    } else {
      x_est_ord     <- matrix(0, 0, 0)
      rr_est_ord    <- matrix(0, 0, 0)
      nmiss_est_ord <- integer(0)
    }
    xds_est<-as.data.frame(x_est_ord)
    
    
    #------------------------------------------------------------
    # Initialize missing x: stochastic draws from sequential fits
    # lm:      Gaussian sequential regression on raw values
    # probit:  Albert-Chib latent normal, lm on latent scale
    # ordinal: Albert-Chib latent normal with K-1 truncation
    #          intervals, lm on latent scale, cutpoints from
    #          empirical normal quantiles of observed categories
    #------------------------------------------------------------
    k=1
    for (j in which(nmiss_ord>0)){
      cn<-colnames(xds)
      xy<-cn[j]
      if(j==1){
        xx<-"1"
      } else {
        xx<-cn[1:(j-1)]
      }
      
      if(na_reg_type_ord[j]=="lm"){
        
        form <- as.formula(paste0(xy, "~", paste(xx, collapse = "+")))
        fit  <- lm(form, data = xds[!na_row_list[[j]], ])
        na_xparams[[k]] <- c(summary(fit)$sigma, fit$coefficients)
        
      } else if (na_reg_type_ord[j]=="probit") { 
        
        xxds <- xds
        xxds[na_row_list[[j]], j] <- rnorm(sum(na_row_list[[j]]))
        if(sum(xds[,j]==1, na.rm=TRUE) > 0){
          xxds[xds[,j]==1, j] <- unlist(lapply(1:sum(xds[,j]==1, na.rm=TRUE),
                                               function(x){ rtnormlo0(0) }))
        }
        if(sum(xds[,j]==0, na.rm=TRUE) > 0){
          xxds[xds[,j]==0, j] <- unlist(lapply(1:sum(xds[,j]==0, na.rm=TRUE),
                                               function(x){ rtnormhi1(0, 0) }))
        }
        form <- as.formula(paste0(xy, "~", paste(xx, collapse = "+")))
        fit  <- lm(form, data = xxds[!na_row_list[[j]], ])
        na_xparams[[k]] <- c(1, fit$coefficients)
        na_seqreg_modtyp[j] <- 1
        
      } else if (na_reg_type_ord[j]=="ordinal") {
        
        #----------------------------------------------------------
        # Ordinal branch: ordered probit via Albert-Chib latent normals
        #----------------------------------------------------------
        
        # Observed values for this variable (non-missing rows only)
        obs_vals <- xds[!na_row_list[[j]], j]
        cats     <- sort(unique(obs_vals))   # sorted unique observed categories
        K        <- length(cats)
        
        # Check that the number of categories is within the allowed limit
        if (K > max_ordinal_levels) {
          vname <- if (!is.null(colnames(x))) colnames(x_ord)[j] else paste0("x", j)
          stop("Ordinal variable '", vname, "' has ", K, " unique observed values, ",
               "which exceeds max_ordinal_levels = ", max_ordinal_levels, ". ",
               "Please collapse categories before calling mbcf(). The cumulative probit ",
               "imputation model is intended for variables with a limited number of ordered ",
               "categories.")
        }
        
        # Compute cutpoints from empirical cumulative proportions of observed categories.
        tab      <- table(factor(obs_vals, levels = cats))
        cum_prop <- cumsum(as.numeric(tab)) / length(obs_vals)
        kappa    <- qnorm(cum_prop[-K])   # length K-1; last boundary is +Inf
        
        # Build latent normal matrix xxds:
        # - Missing rows: initialise to standard normal draws (will be imputed below)
        # - Observed rows in category cats[m]: draw from N(0,1) truncated to
        #   (kappa[m-1], kappa[m]), where kappa[0]=-Inf and kappa[K]=+Inf
        xxds <- xds
        xxds[na_row_list[[j]], j] <- rnorm(sum(na_row_list[[j]]))
        
        for (m in seq_len(K)) {
          in_cat_m <- (!na_row_list[[j]]) & (xds[, j] == cats[m])
          n_cat_m  <- sum(in_cat_m)
          if (n_cat_m > 0) {
            lo_m <- if (m == 1) -Inf else kappa[m - 1]
            hi_m <- if (m == K)  Inf else kappa[m]
            xxds[in_cat_m, j] <- .rtnorm_interval(n_cat_m, mu = 0, sigma = 1,
                                                  lo = lo_m, hi = hi_m)
          }
        }
        
        # Fit lm of latent values on left-hand columns (observed rows only)
        form <- as.formula(paste0(xy, "~", paste(xx, collapse = "+")))
        fit  <- lm(form, data = xxds[!na_row_list[[j]], ])
        
        # Store beta and kappa for reference (not passed to C++; C++ recomputes from x)
        na_xparams[[k]] <- list(beta = fit$coefficients, kappa = kappa)
        na_seqreg_modtyp[j] <- 2
        
      }
      
      #------------------------------------------------------------
      # Impute missing training values using fitted model
      #------------------------------------------------------------
      if (sum(na_row_list[[j]]) > 0){
        n_miss_j <- sum(na_row_list[[j]])
        
        if(na_reg_type_ord[j] == "lm"){
          xds[na_row_list[[j]], j] <- predict(fit, newdata = xds[na_row_list[[j]], ]) +
            rnorm(n_miss_j, 0, summary(fit)$sigma)
          
        } else if (na_reg_type_ord[j] == "probit"){
          xds[na_row_list[[j]], j] <- 1*(predict(fit, newdata = xds[na_row_list[[j]], ]) > 0)
          
        } else if (na_reg_type_ord[j] == "ordinal"){
          xbeta_miss <- predict(fit, newdata = xds[na_row_list[[j]], ])
          kappa_ext <- c(-Inf, kappa, Inf)   # length K+1
          cat_probs <- matrix(0, nrow = n_miss_j, ncol = K)
          for (m in seq_len(K)) {
            cat_probs[, m] <- pnorm(kappa_ext[m + 1] - xbeta_miss) -
              pnorm(kappa_ext[m]     - xbeta_miss)
          }
          cat_probs <- cat_probs / rowSums(cat_probs)  # guard against rounding/floatpoint errors
          cat_draws <- apply(cat_probs, 1, function(p) sample(seq_len(K), size = 1, prob = p))
          xds[na_row_list[[j]], j] <- cats[cat_draws]
        }
      }
      
      #------------------------------------------------------------
      # Impute missing test set values using fitted model
      #------------------------------------------------------------
      if(!is.null(x_est)){
        na_rows_est <- is.na(xds_est[, j])
        if (any(na_rows_est)){
          n_miss_est_j <- sum(na_rows_est)
          
          if(na_reg_type_ord[j] == "lm"){
            xds_est[na_rows_est, j] <- predict(fit, newdata = xds_est[na_rows_est, ]) +
              rnorm(n_miss_est_j, 0, summary(fit)$sigma)
            
          } else if (na_reg_type_ord[j] == "probit"){
            xds_est[na_rows_est, j] <- 1*(predict(fit, newdata = xds_est[na_rows_est, ]) > 0)
            
          } else if (na_reg_type_ord[j] == "ordinal"){
            xbeta_est  <- predict(fit, newdata = xds_est[na_rows_est, ])
            kappa_ext  <- c(-Inf, kappa, Inf)
            cat_probs_est <- matrix(0, nrow = n_miss_est_j, ncol = K)
            for (m in seq_len(K)) {
              cat_probs_est[, m]<-pnorm(kappa_ext[m+1]-xbeta_est) - 
                                  pnorm(kappa_ext[m]-xbeta_est)
            }
            cat_probs_est <- cat_probs_est / rowSums(cat_probs_est)
            cat_draws_est <- apply(cat_probs_est, 1,
                                   function(p) sample(seq_len(K), size = 1, prob = p))
            xds_est[na_rows_est, j] <- cats[cat_draws_est]
          }
        }
      }
      
      k<-k+1
    }
    
    x_ord<-as.matrix(xds)
    na_reg_j1<-min(which(nmiss_ord>0))
    na_train_j1<-min(which(anymiss_ord>0))
    
    if(!is.null(x_est)){
      x_est_ord<-as.matrix(xds_est)
      na_test_js<-which(nmiss_est_ord>0)
    } else {
      na_test_js <- integer(0)
    }
    
  } else {
    na_reg_j1  <- 0
    na_train_j1<- 0
    na_test_js <- integer(0)
    anymiss_ord<- rep(0L, ncol(x))
    if (is.null(x_est)) {
      x_est_ord  <- matrix(0, 0, 0)
      rr_est_ord <- matrix(0, 0, 0)
    }
  }
  
  # y has no missing values (enforced above), so no initialisation is needed
  y_init <- y
  
  #VBT: Uses their (sort of confusing) function to spit out a vector of potential split points to choose from
  cutpoint_list_c = lapply(1:ncol(x_ord), function(i) .cp_quantile(x_ord[,i],num=num_breaks))
  
  #VBT: Center and divide by sample sd use y_init so missing slots are filled
  yscale = scale(y_init)
  
  #VBT: calculate standard deviation and mean for Y (unscaled) from y_init
  sdy = sd(y_init)
  muy = mean(y_init)
  
  #VBT: lambda, nu, sigq, sighat are for the prior on sigma (noise SD)
  # If the user doesn't supply scale parameter for sigma chisq, then it is calibrated to empirical Y distribution
  # sighat estimated after y initialization so that all n obs contribute
  if(is.null(lambda)) {
    if(is.null(sighat)) {
      lmf = lm(yscale~z+x_ord)
      sighat = summary(lmf)$sigma #sd(y) #summary(lmf)$sigma
    }
    qchi = qchisq(1.0-sigq,nu)
    lambda = (sighat*sighat*qchi)/nu
  }
  
  dir = tempdir()
  
  # VBT: everything must be permuted so that Z=1 observations come before Z=0
  perm = order(z, decreasing=TRUE)
  
  #VBT: the defaults are sd_control=2*sdy and sd_moderate=sdy
  csd<-ifelse(abs(2*sdy - sd_control)<1e-6, 2, sd_control/sdy)
  
  #VBT: if half-normal hyperprior on sd_moderate, default value for msd is 1/0.674 = 1.48368
  msd<-ifelse(abs(sdy - sd_moderate)<1e-6, 1, sd_moderate/sdy)/ifelse(use_tauscale,0.674,1) # if HalfNormal make sd_moderate the prior median
  
  na_stuff<-list(na_reg_j1,na_train_j1,na_test_js)
  
  mod_cols_ord <- which(wmv_ord)
  
  #browser()
  
  stopifnot(
    is.matrix(x_ord), is.numeric(x_ord),
    is.matrix(rr_ord), is.numeric(rr_ord),
    nrow(x_ord) == nrow(rr_ord),
    length(anymiss_ord) == ncol(x_ord),
    
    is.matrix(x_est_ord), is.numeric(x_est_ord),
    is.matrix(rr_est_ord), is.numeric(rr_est_ord),
    ncol(x_est_ord) == ncol(rr_est_ord),
    
    all(mod_cols_ord %in% seq_len(ncol(x_ord)))
  )
  
  
  fitbcf = mbcf_cpp(yscale[perm], 
                    z[perm], 
                    t(x_ord[perm,]),              
                    t(rr_ord[perm,]),
                    mod_cols_ord,
                    
                    cutpoint_list_c,
                    anymiss_ord,
                    
                    t(x_est_ord),              
                    t(rr_est_ord),
                    
                    random_des = matrix(1),     #VBT: what are these? they seem to be "on" by default
                    random_var = matrix(1),
                    random_var_ix = matrix(1),
                    random_var_df = 3,
                    
                    nburn, 
                    nsim, 
                    nthin,
                    ntree_moderate, 
                    ntree_control, 
                    
                    bugtest,
                    
                    lambda, 
                    nu,
                    con_sd = csd,
                    mod_sd = msd,
                    
                    base_control,
                    power_control, 
                    base_moderate, 
                    power_moderate, 
                    
                    tempdir(), 
                    
                    na_stuff,
                    na_seqreg_modtyp,
                    na_prob_pm_con,
                    na_prob_pm_mod,
                    yrr[perm],        # missingness indicator for y, permuted to match treatment ordering
                    na_mia,
                    na_seqreg,
                    check_node_missing,
                    
                    trt_init,
                    status_interval = update_interval,
                    use_mscale = use_muscale, 
                    use_bscale = use_tauscale, 
                    b_half_normal = TRUE,
                    
                    min_node,
                    
                    compute_shap      = compute_shap,
                    shap_mod          = shap_mod,
                    shap_con          = shap_con,
                    shap_thin         = shap_thin,
                    shap_var          = shap_var,
                    shap_store_draws  = shap_store_draws,
                    shap_check        = shap_check
  )  
  if (bugtest){
    fitbcf
  } else {
    yhat_post = muy + sdy*fitbcf$yhat_post[,order(perm)]
    mu_post = muy + sdy*fitbcf$m_post[,order(perm)]
    tau_post = sdy*fitbcf$b_post[,order(perm)]
    if(!is.null(x_est)){
      tau_post_est <- sdy*fitbcf$b_est_post
    } else {
      tau_post_est <- NULL
    }
    sigma_post = sdy*fitbcf$sigma
    mu_scale_post = sdy*fitbcf$msd
    tau_scale_post = sdy*fitbcf$bsd
    gamma_post = sdy*fitbcf$gamma
    random_var_post = sdy*fitbcf$random_var_post
    
    # --- Variable importance post-processing ---
    # vi_mod from C++ is p_mod x 4, rows in permuted column order.
    # Steps:
    #   1. Expand to full p_con space (inserting zeros for non-mod cols)
    #   2. Un-permute rows using order(perm_miss) to restore original column order
    #   3. Retain only mod_cols rows in original order
    #   4. Normalise rows to proportions
    #   5. Attach row/col names
    
    vi_raw <- fitbcf$vi_mod  # p_mod x 4 matrix
    
    if (na_seqreg | na_mia) {
      p_con <- ncol(x)
      vi_full_perm <- matrix(0, nrow = p_con, ncol = 4)
      vi_full_perm[which(wmv_ord), ] <- vi_raw
      vi_full_orig <- vi_full_perm[order(perm_miss), , drop = FALSE]
      vi_mod_orig <- vi_full_orig[mod_cols, , drop = FALSE]
      row_nms <- colnames(x)[mod_cols]
    } else {
      # no permutation was applied
      vi_mod_orig <- vi_raw
      row_nms <- colnames(x)[mod_cols]
    }
    
    colnames(vi_mod_orig) <- c("value_seqreg", "mia1", "mia2", "pattern")
    rownames(vi_mod_orig) <- if (!is.null(row_nms)) row_nms else paste0("x", mod_cols)
    
    # normalise to proportions within each variable (guard against all-zero rows)
    row_totals <- rowSums(vi_mod_orig)
    vi_mod_prop <- vi_mod_orig
    nonzero <- row_totals > 0
    vi_mod_prop[nonzero, ] <- vi_mod_orig[nonzero, ] / row_totals[nonzero]
    
    #------------------------------------------------------------
    # mTreeSHAP post-processing
    #
    # C++ returns SHAP on the standardised-y scale, with rows in the
    # treatment-ordered row order and columns in the missingness-ordered
    # covariate order. Undo both, then multiply by sdy to get the same units as
    # tau_post / mu_post. The additive constant muy belongs to the prognostic
    # baseline only; it does not affect any SHAP value.
    #------------------------------------------------------------
    shap_mod_out <- NULL
    shap_con_out <- NULL
    
    if(compute_shap && !is.null(fitbcf$shap)){
      
      sh <- fitbcf$shap
      K  <- sh$n_draws
      row_order <- order(perm)
      
      .shap_pp <- function(acc, col_map, col_names, mask, base_offset = 0){
        if(is.null(acc)) return(NULL)
        if(K < 1) return(NULL)
        take <- function(M) {
          M <- M[row_order, col_map, drop = FALSE]
          dimnames(M) <- list(NULL, col_names)
          M
        }
        out <- list()
        out$mean     <- sdy * take(acc$sum / K)
        out$mean_abs <- sdy * take(acc$sum_abs / K)
        
        # Stratum fields of section 4.2 of the plan. These are DERIVED here
        # rather than accumulated in C++: for a given cell (i,j) the
        # missingness status r_ij is fixed across MCMC iterations, so every
        # cell belongs entirely to one stratum and the mis matrix is either
        # mean_abs[i,j] or exactly 0. Computing them in R gives the documented
        # output with no extra accumulator in the sampler. The per-variable
        # aggregates, which are what these are actually used for, are in
        # shap_summary().
        if(!is.null(mask)){
          colnames(mask) <- col_names
          out$mean_abs_mis <- out$mean_abs * mask
          out$mean_abs_obs <- out$mean_abs * (!mask)
        }
        if(!is.null(acc$sum_sq)){
          v <- acc$sum_sq/K - (acc$sum/K)^2
          v[v < 0] <- 0                       # floating-point guard
          if(K > 1) v <- v * K/(K-1)
          out$var <- (sdy^2) * take(v)
        }
        out$baseline <- base_offset + sdy * as.numeric(acc$baseline)[seq_len(K)]
        if(!is.null(acc$draws)){
          D <- sdy * acc$draws[row_order, col_map, seq_len(K), drop = FALSE]
          dimnames(D) <- list(NULL, col_names, NULL)
          out$draws <- D
        }
        if(!is.null(acc$max_err)) out$max_err <- as.numeric(acc$max_err)[seq_len(K)]
        out$n_draws <- K
        out$thin    <- sh$thin
        out$iters   <- as.integer(sh$iters)[seq_len(K)]
        out
      }
      
      var_names <- colnames(x)
      if(is.null(var_names)) var_names <- paste0("x", seq_len(ncol(x)))
      
      if(shap_mod && !is.null(sh$mod)){
        # C++ moderate-tree features are the mod columns of x_ord, in x_ord order;
        # their original indices are perm_miss[which(wmv_ord)]
        orig_idx <- perm_miss[which(wmv_ord)]
        shap_mod_out <- .shap_pp(sh$mod,
                                 col_map   = match(mod_cols, orig_idx),
                                 col_names = var_names[mod_cols],
                                 mask      = (rr[, mod_cols, drop = FALSE] == 1))
      }
      if(shap_con && !is.null(sh$con)){
        shap_con_out <- .shap_pp(sh$con,
                                 col_map     = order(perm_miss),
                                 col_names   = var_names,
                                 mask        = (rr == 1),
                                 base_offset = muy)
      }
    }
    
    list(yhat = yhat_post,
         mu_post = mu_post,
         tau_post = tau_post,
         tau_post_est = tau_post_est,
         sigma_post = sigma_post,
         mu_scale_post = mu_scale_post,
         tau_scale_post = tau_scale_post,
         gamma_post = gamma_post,
         random_var_post = random_var_post,
         perm = perm,
         vi_mod = vi_mod_orig,       # raw coverage-weighted counts, original col order
         vi_mod_prop = vi_mod_prop,  # row-normalised proportions
         shap_mod = shap_mod_out,    # SHAP for tau(x); NULL unless requested
         shap_con = shap_con_out,    # SHAP for mu(x);  NULL unless requested
         na_mask = if(compute_shap) (rr == 1) else NULL,
         mod_cols = mod_cols)
  }
  
}