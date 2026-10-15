#-------------------------------------------------------------------------------
# test_prognostic_only.R
#
#   source("tests/test_prognostic_only.R")
#
# Checks estimate_cate = FALSE, the prediction-only mode.
#
# The thing most likely to go wrong is not that the mode fails loudly, but that
# it produces NaN quietly: with no moderate forest, allfit_mod is identically
# zero and both bscale values are zero, so every place that DIVIDES by one of
# them is a 0/0 waiting to happen. There are four such places in the sampler
# (the bscale draw, the delta_mod update, r_mod, and the b_post recovery) and
# each is either gated or sits inside a loop over zero trees. These checks exist
# to catch a regression in that gating.
#-------------------------------------------------------------------------------

library(mbcf)

ok <- function(label, pass, detail = "") {
  cat(sprintf("%-54s %s %s\n", label, if (isTRUE(pass)) "PASS" else "FAIL", detail))
  if (!isTRUE(pass)) assign("failures", get0("failures", ifnotfound = 0) + 1,
                            envir = globalenv())
}
failures <- 0

set.seed(3)
n <- 300; p <- 5
x <- matrix(rnorm(n * p), nrow = n)
colnames(x) <- paste0("x", 1:p)
x[sample(n, 45), 2] <- NA
mu_true <- 2 * x[, 1] - 1.5 * (x[, 3] > 0)
mu_true[is.na(mu_true)] <- 0
y <- mu_true + 0.3 * rnorm(n)

x_te <- matrix(rnorm(100 * p), nrow = 100)
colnames(x_te) <- colnames(x)
x_te[sample(100, 15), 2] <- NA

# ---- argument handling -------------------------------------------------------
e1 <- tryCatch({ mbcf(y, z = rbinom(n, 1, 0.5), x, estimate_cate = FALSE,
                      nburn = 10, nsim = 10); NA },
               error = function(e) conditionMessage(e))
ok("supplying z with estimate_cate=FALSE is an error", grepl("not", e1))

e2 <- tryCatch({ mbcf(y, x = x, nburn = 10, nsim = 10); NA },
               error = function(e) conditionMessage(e))
ok("omitting z with estimate_cate=TRUE is an error", grepl("required", e2))

# ---- the actual fit ----------------------------------------------------------
fit <- mbcf(y, x = x, x_est = x_te, estimate_cate = FALSE,
            nburn = 500, nsim = 500, nthin = 1,
            na_mia = TRUE, na_seqreg = TRUE, update_interval = 1e9,
            compute_shap = TRUE, shap_con = TRUE, shap_mod = FALSE,
            shap_thin = 10, shap_check = TRUE, shap_est = TRUE)

# ---- nothing is NaN ----------------------------------------------------------
ok("mu_post has no NaN/Inf", all(is.finite(fit$mu_post)))
ok("mu_post_est has no NaN/Inf", all(is.finite(fit$mu_post_est)))
ok("sigma_post has no NaN/Inf", all(is.finite(fit$sigma_post)))
ok("mu_scale_post has no NaN/Inf", all(is.finite(fit$mu_scale_post)))

# ---- tau-related output is absent, not zero-filled ---------------------------
ok("tau_post is NULL", is.null(fit$tau_post))
ok("tau_post_est is NULL", is.null(fit$tau_post_est))
ok("yhat_post is NULL (equals mu)", is.null(fit$yhat_post))
ok("shap_mod is NULL", is.null(fit$shap_mod))

# ---- the prognostic half still works -----------------------------------------
ok("mu recovers the true prognostic function",
   cor(colMeans(fit$mu_post), mu_true) > 0.9,
   sprintf("cor = %.3f", cor(colMeans(fit$mu_post), mu_true)))
ok("prognostic SHAP local accuracy", max(fit$shap_con$max_err) < 1e-9,
   sprintf("max_err = %.2e", max(fit$shap_con$max_err)))
ok("prognostic SHAP on x_est present", !is.null(fit$shap_con_est$mean))
ok("SHAP reassembles to mu",
   cor(rowSums(fit$shap_con$mean) + mean(fit$shap_con$baseline),
       colMeans(fit$mu_post)) > 0.99,
   sprintf("cor = %.4f",
           cor(rowSums(fit$shap_con$mean) + mean(fit$shap_con$baseline),
               colMeans(fit$mu_post))))

# ---- the same data WITH a treatment arm still behaves ------------------------
set.seed(3)
z <- rbinom(n, 1, 0.5)
y2 <- mu_true + 0.5 * z + 0.3 * rnorm(n)
fit2 <- mbcf(y2, z, x, mod_cols = 1:p, nburn = 500, nsim = 500, nthin = 1,
             na_mia = TRUE, na_seqreg = TRUE, update_interval = 1e9)
ok("estimate_cate=TRUE still returns tau", !is.null(fit2$tau_post))
ok("tau recovers roughly 0.5", abs(mean(fit2$tau_post) - 0.5) < 0.25,
   sprintf("mean tau = %.3f", mean(fit2$tau_post)))

cat("\n", if (failures == 0) "ALL CHECKS PASSED" else
          paste(failures, "CHECKS FAILED"), "\n", sep = "")
