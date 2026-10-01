#-------------------------------------------------------------------------------
# benchmark_shap.R
#
# How much does mTreeSHAP add to the cost of an mBCF run?
#
#   Rscript tests/benchmark_shap.R
#
# Each configuration is run on the same data with the same seed, so the chains
# are identical and the only difference in wall time is the SHAP work. mbcf()
# also reports the CPU time spent inside the SHAP block itself (shap_seconds),
# which separates the attribution cost from everything else.
#
# The cost of one SHAP draw is O(ntree * n * L * D^2), where L is leaves per tree
# and D is depth, so it scales linearly in n and in the number of trees, and
# quadratically in tree depth. It does NOT scale with the number of MCMC
# iterations except through how many draws are taken, which is what shap_thin
# controls: halving shap_thin doubles the SHAP cost. Note the per-draw cost is
# the same whether or not that iteration is a saved draw, so the comparison
# below is against nthin, not against nsim.
#-------------------------------------------------------------------------------

library(mbcf)

make_data <- function(n, p, pmiss = 0.15, seed = 1) {
  set.seed(seed)
  x <- matrix(rnorm(n * p), nrow = n)
  colnames(x) <- paste0("x", seq_len(p))
  for (j in seq_len(min(3L, p))) {
    x[sample(n, round(pmiss * n)), j] <- NA
  }
  q <- -1 * (x[, 1] > 0) + 1 * (x[, 1] < 0)
  q[is.na(q)] <- 0
  z <- rbinom(n, 1, pnorm(q))
  tau <- 0.5 * (x[, p] > -0.75) + 0.25 * (x[, p] > 0)
  tau[is.na(tau)] <- 0.25
  y <- q + tau * z + 0.25 * rnorm(n)
  list(y = y, z = z, x = x, p = p)
}

run_one <- function(d, label, seed = 99, ...) {
  set.seed(seed)
  t0 <- proc.time()
  fit <- mbcf(d$y, d$z, d$x, mod_cols = 1:d$p,
              nburn = 500, nsim = 500, nthin = 1,
              na_mia = TRUE, update_interval = 1e9, ...)
  el <- (proc.time() - t0)[["elapsed"]]
  shap_cpu <- if (is.null(fit$shap_seconds)) 0 else fit$shap_seconds
  nd <- if (is.null(fit$shap_mod$n_draws)) {
    if (is.null(fit$shap_con$n_draws)) 0L else fit$shap_con$n_draws
  } else fit$shap_mod$n_draws
  data.frame(config = label, elapsed = round(el, 2),
             shap_cpu = round(shap_cpu, 2),
             shap_draws = nd,
             sec_per_draw = round(ifelse(nd > 0, shap_cpu / nd, NA), 4),
             stringsAsFactors = FALSE)
}

bench <- function(n, p) {
  d <- make_data(n, p)
  cat("\n=== n =", n, " p =", p, "===\n")
  res <- rbind(
    run_one(d, "no SHAP",                compute_shap = FALSE),
    run_one(d, "tau only, thin 10",      compute_shap = TRUE, shap_mod = TRUE,
                                         shap_con = FALSE, shap_thin = 10),
    run_one(d, "tau + mu, thin 10",      compute_shap = TRUE, shap_mod = TRUE,
                                         shap_con = TRUE,  shap_thin = 10),
    run_one(d, "tau + mu, thin 1",       compute_shap = TRUE, shap_mod = TRUE,
                                         shap_con = TRUE,  shap_thin = 1),
    run_one(d, "tau + mu, thin 10, var", compute_shap = TRUE, shap_mod = TRUE,
                                         shap_con = TRUE,  shap_thin = 10,
                                         shap_var = TRUE)
  )
  base <- res$elapsed[res$config == "no SHAP"]
  res$overhead <- paste0(round(100 * (res$elapsed / base - 1)), "%")
  print(res, row.names = FALSE)
  invisible(res)
}

all_res <- list()
for (cfg in list(c(250, 5), c(1000, 10), c(2500, 20))) {
  all_res[[length(all_res) + 1]] <- bench(cfg[1], cfg[2])
}

cat("\n=== vi_coverage overhead (one coverage pass per saved draw) ===\n")
d <- make_data(1000, 10)
print(rbind(run_one(d, "vi_coverage FALSE", vi_coverage = FALSE),
            run_one(d, "vi_coverage TRUE",  vi_coverage = TRUE)),
      row.names = FALSE)

cat("\n=== out-of-sample SHAP (shap_est) ===\n")
d <- make_data(1000, 10)
dte <- make_data(500, 10, seed = 7)
set.seed(99); t0 <- proc.time()
f1 <- mbcf(d$y, d$z, d$x, mod_cols = 1:d$p, x_est = dte$x,
           nburn = 500, nsim = 500, nthin = 1, na_mia = TRUE,
           update_interval = 1e9,
           compute_shap = TRUE, shap_mod = TRUE, shap_con = TRUE,
           shap_thin = 10, shap_est = FALSE)
t_in <- (proc.time() - t0)[["elapsed"]]
set.seed(99); t0 <- proc.time()
f2 <- mbcf(d$y, d$z, d$x, mod_cols = 1:d$p, x_est = dte$x,
           nburn = 500, nsim = 500, nthin = 1, na_mia = TRUE,
           update_interval = 1e9,
           compute_shap = TRUE, shap_mod = TRUE, shap_con = TRUE,
           shap_thin = 10, shap_est = TRUE)
t_est <- (proc.time() - t0)[["elapsed"]]
cat("in-sample only:    ", round(t_in, 2), "s   SHAP CPU", round(f1$shap_seconds, 2), "\n")
cat("plus 500 test rows:", round(t_est, 2), "s   SHAP CPU", round(f2$shap_seconds, 2), "\n")
cat("undefined test rows (tau):",
    sum(f2$shap_mod_est$n_undef > 0), "of", nrow(dte$x), "\n")
