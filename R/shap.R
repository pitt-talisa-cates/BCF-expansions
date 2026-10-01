#------------------------------------------------------------------------------
# mTreeSHAP summaries and plots
#
# All of these operate on the list returned by mbcf() and take `which` to pick
# the function being explained: "mod" for tau(x) (the default) or "con" for the
# prognostic mu(x). Base graphics only, so the package gains no dependencies.
#
# A note on the stratified importance summaries of section 4.2 of the plan.
# sum_abs_shap_obs and sum_abs_shap_mis were specified as N x p accumulators
# inside C++, but for a given cell (i,j) the missingness status r_ij is fixed
# across MCMC iterations, so the cell belongs entirely to one stratum and the
# mis accumulator is either equal to sum_abs_shap[i,j] or exactly zero. The
# informative object is the per-variable average within stratum, which is what
# shap_summary() computes here from shap_mean_abs and the missingness mask, at
# no memory or runtime cost in the sampler.
#------------------------------------------------------------------------------

.shap_names <- function(M, var_names = NULL) {
  if (!is.null(var_names)) {
    if (length(var_names) != ncol(M)) {
      stop("var_names has length ", length(var_names), " but this SHAP matrix has ",
           ncol(M), " columns")
    }
    return(as.character(var_names))
  }
  nm <- colnames(M)
  if (is.null(nm)) nm <- paste0("x", seq_len(ncol(M)))
  nm
}

.shap_which <- c("mod", "con", "mod_est", "con_est")

.shap_label <- function(which) if (grepl("^mod", which)) "tau" else "mu"

.shap_get <- function(fit, which = .shap_which) {
  which <- match.arg(which, .shap_which)
  out <- switch(which,
                mod     = fit$shap_mod,
                con     = fit$shap_con,
                mod_est = fit$shap_mod_est,
                con_est = fit$shap_con_est)
  if (is.null(out)) {
    stop("No ", which, " SHAP values in this fit. Refit with compute_shap=TRUE, ",
         if (grepl("^mod", which)) "shap_mod=TRUE" else "shap_con=TRUE",
         if (grepl("_est$", which)) ", shap_est=TRUE, and x_est supplied" else "",
         ".")
  }
  out
}

# missingness mask restricted to the columns the requested SHAP matrix uses
.shap_mask <- function(fit, which = .shap_which, shp) {
  which <- match.arg(which, .shap_which)
  msk <- if (grepl("_est$", which)) fit$na_mask_est else fit$na_mask
  if (is.null(msk)) return(NULL)
  m <- if (grepl("^mod", which)) msk[, fit$mod_cols, drop = FALSE] else msk
  colnames(m) <- colnames(shp$mean_abs)
  m
}

#' Per-variable SHAP importance, overall and stratified by missingness
#'
#' @param fit Object returned by \code{mbcf()}
#' @param var_names Optional covariate labels, overriding the column names
#'   carried by the SHAP matrices. Length must match the number of covariates
#'   for the requested function (\code{length(mod_cols)} for "mod",
#'   \code{ncol(x)} for "con").
#' @param which "mod" or "con" for tau(x) or mu(x) on the training rows;
#'   "mod_est" or "con_est" for the same on the x_est rows
#' @return A data frame, one row per covariate, sorted by decreasing importance:
#'   \item{variable}{covariate name}
#'   \item{mean_abs}{mean over observations of the posterior mean |SHAP|}
#'   \item{mean_signed}{mean over observations of the posterior mean SHAP}
#'   \item{post_sd}{mean over observations of the posterior SD of SHAP, if
#'     \code{shap_var=TRUE} was used, otherwise NA}
#'   \item{mean_abs_obs, mean_abs_mis}{the same importance restricted to rows
#'     where the covariate is observed / missing}
#'   \item{n_mis}{number of rows where the covariate is missing}
#' @export
shap_summary <- function(fit, var_names = NULL, which = .shap_which) {
  which <- match.arg(which, .shap_which)
  shp <- .shap_get(fit, which)
  A   <- shp$mean_abs
  S   <- shp$mean
  msk <- .shap_mask(fit, which, shp)

  vars <- .shap_names(A, var_names)

  sd_col <- if (!is.null(shp$var)) colMeans(sqrt(shp$var)) else rep(NA_real_, ncol(A))

  obs_col <- mis_col <- rep(NA_real_, ncol(A))
  n_mis <- rep(0L, ncol(A))
  if (!is.null(msk)) {
    for (j in seq_len(ncol(A))) {
      mj <- msk[, j]
      n_mis[j] <- sum(mj)
      if (any(!mj)) obs_col[j] <- mean(A[!mj, j])
      if (any(mj))  mis_col[j] <- mean(A[mj, j])
    }
  }

  out <- data.frame(variable     = vars,
                    mean_abs     = colMeans(A),
                    mean_signed  = colMeans(S),
                    post_sd      = sd_col,
                    mean_abs_obs = obs_col,
                    mean_abs_mis = mis_col,
                    n_mis        = n_mis,
                    stringsAsFactors = FALSE)
  out <- out[order(-out$mean_abs), ]
  rownames(out) <- NULL
  out
}

#' Beeswarm plot of posterior mean SHAP values
#'
#' One row per covariate, one point per observation, jittered vertically.
#' Points are shaded by the covariate value (light = low, dark = high); rows
#' where the covariate is missing are drawn in grey.
#'
#' @param fit Object returned by \code{mbcf()}
#' @param var_names Optional covariate labels
#' @param x Covariate matrix used to colour the points, i.e. the same \code{x}
#'   passed to \code{mbcf()}. Optional; without it all points are grey.
#' @param which "mod" or "con" for tau(x) or mu(x) on the training rows;
#'   "mod_est" or "con_est" for the same on the x_est rows
#' @param max_vars Show at most this many covariates, by decreasing importance
#' @export
plot_shap_beeswarm <- function(fit, var_names = NULL, x = NULL,
                               which = .shap_which,
                               max_vars = 20, cex = 0.4) {
  which <- match.arg(which, .shap_which)
  shp <- .shap_get(fit, which)
  S   <- shp$mean
  colnames(S) <- .shap_names(S, var_names)
  msk <- .shap_mask(fit, which, shp)

  ord  <- order(colMeans(abs(S)))
  keep <- rev(rev(ord)[seq_len(min(max_vars, length(ord)))])
  S    <- S[, keep, drop = FALSE]
  if (!is.null(msk)) msk <- msk[, keep, drop = FALSE]

  xc <- NULL
  if (!is.null(x)) {
    if (nrow(x) != nrow(S))
      stop("x has ", nrow(x), " rows but these SHAP values have ", nrow(S),
           " rows; for this `which` pass the matrix those rows came from")
    cols_used <- if (grepl("^mod", which)) fit$mod_cols else seq_len(ncol(x))
    xc <- x[, cols_used[keep], drop = FALSE]
  }

  p <- ncol(S)
  op <- par(mar = c(4.5, 8, 2, 1)); on.exit(par(op))
  plot(range(S, na.rm = TRUE), c(0.5, p + 0.5), type = "n", yaxt = "n",
       xlab = paste0("posterior mean SHAP (", .shap_label(which), ")"),
       ylab = "", main = "mTreeSHAP")
  axis(2, at = seq_len(p), labels = colnames(S), las = 1)
  abline(v = 0, col = "grey70")

  for (j in seq_len(p)) {
    yj <- j + runif(nrow(S), -0.28, 0.28)
    if (!is.null(xc)) {
      v  <- xc[, j]
      rk <- rank(v, na.last = "keep") / sum(!is.na(v))
      shade <- grey(pmax(0.05, pmin(0.85, 0.85 - 0.8 * rk)))
      shade[is.na(v)] <- "grey80"
    } else {
      shade <- "grey50"
    }
    if (!is.null(msk)) shade[msk[, j]] <- "grey80"
    points(S[, j], yj, pch = 16, cex = cex, col = shade)
  }
  invisible(NULL)
}

#' Paired bar chart of SHAP importance by missingness stratum
#'
#' @param fit Object returned by \code{mbcf()}
#' @param var_names Optional covariate labels
#' @param which "mod" or "con" for tau(x) or mu(x) on the training rows;
#'   "mod_est" or "con_est" for the same on the x_est rows
#' @param max_vars Show at most this many covariates, by decreasing importance
#' @export
plot_shap_importance <- function(fit, var_names = NULL, which = .shap_which,
                                 max_vars = 20) {
  which <- match.arg(which, .shap_which)
  tab <- shap_summary(fit, var_names = var_names, which = which)
  tab <- tab[seq_len(min(max_vars, nrow(tab))), ]
  tab <- tab[rev(seq_len(nrow(tab))), ]

  M <- rbind(observed = ifelse(is.na(tab$mean_abs_obs), 0, tab$mean_abs_obs),
             missing  = ifelse(is.na(tab$mean_abs_mis), 0, tab$mean_abs_mis))
  colnames(M) <- tab$variable

  op <- par(mar = c(4.5, 8, 2, 1)); on.exit(par(op))
  barplot(M, beside = TRUE, horiz = TRUE, las = 1,
          col = c("grey35", "grey75"), border = NA,
          xlab = "posterior mean |SHAP|",
          main = paste0("SHAP importance by missingness (",
                        .shap_label(which), ")"))
  legend("bottomright", legend = c("observed", "missing"),
         fill = c("grey35", "grey75"), border = NA, bty = "n")
  invisible(tab)
}

#' Waterfall plot for a single observation
#'
#' Only meaningful for observations with no missing covariates, since for a row
#' with missing values the attribution is averaged over imputation draws as well
#' as over trees.
#'
#' @param fit Object returned by \code{mbcf()}
#' @param i Row index (in the original, unpermuted row order)
#' @param var_names Optional covariate labels
#' @param which "mod" or "con" for tau(x) or mu(x) on the training rows;
#'   "mod_est" or "con_est" for the same on the x_est rows
#' @param max_vars Show at most this many covariates, by decreasing |SHAP|
#' @export
plot_shap_waterfall <- function(fit, i, var_names = NULL, which = .shap_which,
                                max_vars = 12) {
  which <- match.arg(which, .shap_which)
  shp <- .shap_get(fit, which)
  colnames(shp$mean) <- .shap_names(shp$mean, var_names)
  msk <- .shap_mask(fit, which, shp)
  if (!is.null(msk) && any(msk[i, ])) {
    warning("observation ", i, " has missing covariates; its SHAP values also ",
            "average over imputation draws")
  }

  phi <- shp$mean[i, ]
  ord <- order(abs(phi), decreasing = TRUE)
  keep <- ord[seq_len(min(max_vars, length(ord)))]
  phi <- phi[rev(keep)]
  base <- mean(shp$baseline)

  op <- par(mar = c(4.5, 8, 3, 1)); on.exit(par(op))
  barplot(phi, horiz = TRUE, las = 1, border = NA,
          col = ifelse(phi > 0, "grey35", "grey65"),
          xlab = "posterior mean SHAP contribution",
          main = sprintf("observation %d:  baseline %.3f  ->  %s %.3f",
                         i, base, .shap_label(which),
                         base + sum(shp$mean[i, ])))
  abline(v = 0, col = "grey70")
  invisible(phi)
}
