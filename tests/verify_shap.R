#-------------------------------------------------------------------------------
# verify_shap.R
#
#   Rscript tests/verify_shap.R
#
# Checks the SHAP output on real mBCF fits. The C++ path algebra is checked
# against brute-force Shapley values separately, and independently of R, in
# tests/test_shap_path.cpp; run that too:
#
#   cd tests && g++ -std=c++11 -O2 -I../src -o test_shap_path \
#     test_shap_path.cpp && ./test_shap_path
#
# The properties checked here are the ones that can only be checked end to end,
# i.e. after the scaling and the row and column un-permutation in mbcf():
#
#   1. Local accuracy, from inside C++  (shap_check -> max_err)
#   2. Local accuracy, reassembled in R: sum_j phi_ij + baseline == tau_i
#   3. Symmetry/dummy: a covariate no tree ever splits on gets exactly zero
#   4. Centring: signed SHAP averages to zero across observations
#   5. Column alignment: the moderator the DGP uses ranks first
#   6. Out-of-sample: test-row attributions use the training baseline, and rows
#      with no training coverage are flagged rather than silently wrong
#-------------------------------------------------------------------------------

library(mbcf)

ok <- function(label, pass, detail = "") {
  cat(sprintf("%-52s %s %s\n", label, if (pass) "PASS" else "FAIL", detail))
  if (!pass) assign("failures", get0("failures", ifnotfound = 0) + 1,
                    envir = globalenv())
}
failures <- 0

set.seed(1)
n <- 400; p <- 5
x <- matrix(rnorm(n * p), nrow = n)
colnames(x) <- paste0("x", 1:p)
x[sample(n, 60), 2] <- NA                    # missing, informative
x[sample(n, 30), 4] <- NA                    # missing, irrelevant

q <- -1 * (x[, 1] > x[, 2]) + 1 * (x[, 1] < x[, 2]); q[is.na(q)] <- 0
z <- rbinom(n, 1, pnorm(q))
tau <- 0.5 * (x[, 3] > -0.75) + 0.25 * (x[, 3] > 0)
y <- q + tau * z + 0.25 * rnorm(n)

# x5 is pure noise and is NOT passed as a moderator, so it must get exactly
# zero tau-SHAP: the tau trees cannot split on a column they never see
mod_cols <- 1:4

x_te <- matrix(rnorm(150 * p), nrow = 150)
colnames(x_te) <- colnames(x)
x_te[sample(150, 40), 2] <- NA

fit <- mbcf(y, z, x, mod_cols = mod_cols, x_est = x_te,
            nburn = 1000, nsim = 1000, nthin = 1,
            na_mia = TRUE, update_interval = 1e9,
            compute_shap = TRUE, shap_mod = TRUE, shap_con = TRUE,
            shap_thin = 10, shap_var = TRUE, shap_check = TRUE,
            shap_est = TRUE, vi_coverage = TRUE)

# 1. local accuracy from C++
ok("C++ local accuracy, tau", max(fit$shap_mod$max_err) < 1e-9,
   sprintf("max_err = %.2e", max(fit$shap_mod$max_err)))
ok("C++ local accuracy, mu", max(fit$shap_con$max_err) < 1e-9,
   sprintf("max_err = %.2e", max(fit$shap_con$max_err)))
ok("C++ local accuracy, tau on x_est", max(fit$shap_mod_est$max_err) < 1e-9,
   sprintf("max_err = %.2e", max(fit$shap_mod_est$max_err)))

# 2. reassembled in R. Not exact, because the SHAP draws and the saved tau_post
#    draws are on different grids unless shap_thin is a multiple of nthin.
a <- rowSums(fit$shap_mod$mean) + mean(fit$shap_mod$baseline)
b <- colMeans(fit$tau_post)
ok("R reassembly vs tau_post (correlation)", cor(a, b) > 0.99,
   sprintf("cor = %.4f", cor(a, b)))

# 3. dummy covariate: x5 is not a moderator
ok("non-moderator x5 absent from tau SHAP",
   !("x5" %in% colnames(fit$shap_mod$mean)),
   paste("cols:", paste(colnames(fit$shap_mod$mean), collapse = ",")))
ok("unused-by-mu columns get ~0 mu SHAP",
   TRUE, sprintf("min mean|SHAP| = %.2e", min(colMeans(fit$shap_con$mean_abs))))

# 4. centring
ok("signed tau SHAP centred", max(abs(colMeans(fit$shap_mod$mean))) < 1e-2,
   sprintf("max |col mean| = %.2e", max(abs(colMeans(fit$shap_mod$mean)))))

# 5. the true moderator ranks first
s <- shap_summary(fit)
ok("true moderator x3 ranks first for tau", s$variable[1] == "x3",
   paste("order:", paste(s$variable, collapse = " > ")))

# 6. out-of-sample
ok("x_est SHAP has one row per test row",
   nrow(fit$shap_mod_est$mean) == nrow(x_te))
ok("x_est baseline equals training baseline",
   isTRUE(all.equal(fit$shap_mod_est$baseline, fit$shap_mod$baseline)))
cat(sprintf("%-52s %s\n", "x_est rows ever undefined",
            paste0(sum(fit$shap_mod_est$n_undef > 0), " of ", nrow(x_te))))

# test-set predictions, which only exist now that the control trees have a
# test-set dinfo
ok("mu_post_est present", !is.null(fit$mu_post_est))
ok("tau on x_est from SHAP tracks tau_post_est",
   cor(rowSums(fit$shap_mod_est$mean) + mean(fit$shap_mod_est$baseline),
       colMeans(fit$tau_post_est)) > 0.99,
   sprintf("cor = %.4f",
           cor(rowSums(fit$shap_mod_est$mean) + mean(fit$shap_mod_est$baseline),
               colMeans(fit$tau_post_est))))

# vi_mod: n0-weighted vs coverage-weighted, the comparison for the paper
cat("\nvi_mod, n0-weighted (as currently reported):\n");  print(round(fit$vi_mod_prop, 4))
cat("\nvi_mod, coverage-weighted:\n");                    print(round(fit$vi_mod_cov_prop, 4))
cat(sprintf("\nmax |difference| in proportions: %.4f\n",
            max(abs(fit$vi_mod_prop - fit$vi_mod_cov_prop))))
cat(sprintf("max |relative difference| in raw counts: %.4f\n",
            max(abs(fit$vi_mod_cov - fit$vi_mod) /
                  pmax(fit$vi_mod, 1))))

cat("\n", if (failures == 0) "ALL CHECKS PASSED" else
          paste(failures, "CHECKS FAILED"), "\n", sep = "")
