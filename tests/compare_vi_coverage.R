#-------------------------------------------------------------------------------
# compare_vi_coverage.R
#
#   source("tests/compare_vi_coverage.R")
#
# Does the n0 staleness actually move the reported variable importance?
#
# vi_mod weights each internal node by getn0(). For an internal node that is
# whatever birth() wrote when the node was last a leaf (see the comment at
# tree.cpp line 494), so it is correct at the moment of the split and then
# drifts as the imputed x moves. vi_mod_cov weights the same counts by coverage
# recomputed from the current x at every save.
#
# The two can only differ when x moves, i.e. under na_seqreg = TRUE. With
# na_mia = TRUE and na_seqreg = FALSE the imputation is done once in R and
# frozen, so n0 never drifts and the two agree exactly -- verify_shap.R shows
# that, and it is a control rather than an answer.
#
# This script runs the same data both ways so the control and the real
# comparison appear side by side.
#-------------------------------------------------------------------------------

library(mbcf)

set.seed(11)
n <- 500; p <- 6
x <- matrix(rnorm(n * p), nrow = n)
colnames(x) <- paste0("x", 1:p)

# missingness in three covariates, at three different rates, so drift has room
# to differ across variables
x[sample(n, 0.30 * n), 2] <- NA
x[sample(n, 0.15 * n), 3] <- NA
x[sample(n, 0.05 * n), 5] <- NA

q <- -1 * (x[, 1] > x[, 2]) + 1 * (x[, 1] < x[, 2]); q[is.na(q)] <- 0
z <- rbinom(n, 1, pnorm(q))
tau <- 0.5 * (x[, 4] > -0.75) + 0.25 * (x[, 4] > 0) + 0.3 * is.na(x[, 2])
y <- q + tau * z + 0.25 * rnorm(n)

run <- function(label, ...) {
  cat("\n================ ", label, " ================\n", sep = "")
  set.seed(42)
  fit <- mbcf(y, z, x, mod_cols = 1:p,
              nburn = 1000, nsim = 1000, nthin = 1,
              update_interval = 1e9,
              vi_coverage = TRUE, ...)

  cat("\nvi_mod proportions, n0-weighted (as currently reported):\n")
  print(round(fit$vi_mod_prop, 4))
  cat("\nvi_mod proportions, coverage-weighted:\n")
  print(round(fit$vi_mod_cov_prop, 4))

  dprop <- abs(fit$vi_mod_prop - fit$vi_mod_cov_prop)
  cat("\nabsolute difference in proportions:\n")
  print(round(dprop, 4))

  # raw counts: relative difference per cell, guarding empty cells
  denom <- pmax(fit$vi_mod, 1)
  drel <- abs(fit$vi_mod_cov - fit$vi_mod) / denom

  cat(sprintf("\nmax |difference| in proportions:       %.6f\n", max(dprop)))
  cat(sprintf("max |relative difference| raw counts:  %.6f\n", max(drel)))

  # does either version change the RANKING of variables by total split weight?
  r_n0  <- rank(-rowSums(fit$vi_mod))
  r_cov <- rank(-rowSums(fit$vi_mod_cov))
  cat("ranking by total split weight, n0:       ",
      paste(rownames(fit$vi_mod)[order(r_n0)], collapse = " > "), "\n")
  cat("ranking by total split weight, coverage: ",
      paste(rownames(fit$vi_mod)[order(r_cov)], collapse = " > "), "\n")
  cat("ranking identical: ", identical(order(r_n0), order(r_cov)), "\n")

  # and does the dominant split TYPE per variable change? that is the claim the
  # paper would actually be making
  t_n0  <- colnames(fit$vi_mod)[apply(fit$vi_mod, 1, which.max)]
  t_cov <- colnames(fit$vi_mod)[apply(fit$vi_mod_cov, 1, which.max)]
  cat("dominant split type per variable identical: ", identical(t_n0, t_cov), "\n")
  if (!identical(t_n0, t_cov)) {
    print(data.frame(variable = rownames(fit$vi_mod), n0 = t_n0, coverage = t_cov))
  }
  invisible(fit)
}

# control: x frozen after one imputation, so n0 cannot drift
f1 <- run("na_mia = TRUE, na_seqreg = FALSE  (control: x frozen)",
          na_mia = TRUE, na_seqreg = FALSE)

# the real comparison: x redrawn every iteration
f2 <- run("na_seqreg = TRUE  (x redrawn every iteration)",
          na_mia = TRUE, na_seqreg = TRUE)

cat("\n--------------------------------------------------------------\n")
cat("Read the control first. If the frozen-x run shows any difference\n")
cat("at all, the coverage accumulation is wrong and the second table\n")
cat("means nothing. Only if the control is exactly zero does the\n")
cat("na_seqreg difference tell you about the n0 staleness.\n")
