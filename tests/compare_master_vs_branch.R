#-------------------------------------------------------------------------------
# compare_master_vs_branch.R
#
#   source("tests/compare_master_vs_branch.R")
#
# Does branch_treeshap reproduce master's chain EXACTLY?
#
# The claim made for the SHAP work is that the sampler is untouched: n0,
# n0_xnew and update_n0() run exactly when and how they did, tree::coverage is
# written only by update_coverage() and read by nothing in the MCMC, and no new
# code consumes random numbers. If that claim holds, then for the same seed and
# the same data the two versions must produce bit-identical draws -- not close,
# identical.
#
# Victor reported slightly different fits on ENRICHD between the two branches.
# There are only two possibilities:
#
#   (a) the two runs did not start from the same RNG state, in which case any
#       two MCMC runs differ and the observation says nothing; or
#   (b) something in the branch really does perturb the sampler, in which case
#       the claim is wrong and we need to find it.
#
# This script distinguishes them. It clones the local repo at master into a
# temporary directory, installs it into a temporary library, and runs the same
# fixed-seed fit under each version in a separate R process, then compares the
# draws with identical().
#
# Requires callr and gert, both already present as devtools dependencies.
#-------------------------------------------------------------------------------

stopifnot(requireNamespace("callr", quietly = TRUE),
          requireNamespace("gert", quietly = TRUE))

proj <- normalizePath(".")
stopifnot(file.exists(file.path(proj, "DESCRIPTION")))

cat("current branch:", gert::git_branch(repo = proj), "\n")
st <- gert::git_status(repo = proj)
if (any(!st$staged & st$file %in% c("R/mbcf.R", "src/bcf_clean_overpar.cpp"))) {
  cat("NOTE: you have uncommitted changes to tracked source files. This script\n",
      "      compares the INSTALLED branch build against a fresh master build,\n",
      "      so uncommitted edits are only included if you have reinstalled.\n", sep = "")
}

# ---- 1. build master in its own library --------------------------------------
tmp      <- tempfile("mastercmp"); dir.create(tmp)
master_src <- file.path(tmp, "src_master")
master_lib <- file.path(tmp, "lib_master")
dir.create(master_lib)

cat("\ncloning local repo at master ...\n")
gert::git_clone(url = proj, path = master_src, branch = "master", verbose = FALSE)

# master has compiled objects committed to it, built on another platform (there
# is a .dll in there). R CMD INSTALL would try to LINK those rather than
# recompile, which fails with "unknown file type". Remove them from the clone.
unlink(Sys.glob(file.path(master_src, "src", c("*.o", "*.so", "*.dll"))))

# master also still carries the packageMaintainer{imbcf} typo that blocks the
# help step, and the duplicate \alias{mbcf}. Both are fixed on the branch;
# patch the clone so that master can be installed at all.
mf <- file.path(master_src, "man", "mbcf-package.Rd")
if (file.exists(mf)) {
  mx <- readLines(mf)
  mx <- sub("{imbcf}", "{mbcf}", mx, fixed = TRUE)
  mx <- mx[!grepl("^\\\\alias\\{mbcf\\}$", mx)]
  writeLines(mx, mf)
}

cat("installing master into a temporary library (compiles, a few minutes) ...\n")
callr::rcmd("INSTALL", c(paste0("--library=", master_lib), master_src),
            show = FALSE, fail_on_status = TRUE)

# ---- 2. the fit to compare ---------------------------------------------------
# Only arguments that exist on BOTH versions. No x_est with a moderator subset,
# since master's dimension check rejects that.
# NOTE: this function is serialized and run in a FRESH R process by callr, so it
# cannot see anything from this script's environment. Everything it needs must
# arrive through its arguments.
fit_code <- function(seed_data, seed_mcmc) {
    library(mbcf)
    set.seed(seed_data)
    n <- 300; p <- 5
    x <- matrix(rnorm(n * p), nrow = n)
    colnames(x) <- paste0("x", 1:p)
    x[sample(n, 45), 2] <- NA
    x[sample(n, 20), 4] <- NA
    q <- -1 * (x[, 1] > x[, 2]) + 1 * (x[, 1] < x[, 2]); q[is.na(q)] <- 0
    z <- rbinom(n, 1, pnorm(q))
    tau <- 0.5 * (x[, 3] > -0.75) + 0.25 * (x[, 3] > 0)
    y <- q + tau * z + 0.25 * rnorm(n)

    set.seed(seed_mcmc)
    f <- mbcf(y, z, x, mod_cols = 1:p,
              nburn = 300, nsim = 300, nthin = 1,
              na_mia = TRUE, na_seqreg = TRUE,     # x redrawn: the demanding case
              update_interval = 1e9)
    list(tau = f$tau_post, mu = f$mu_post, sigma = f$sigma_post,
         vi = f$vi_mod, ver = as.character(utils::packageVersion("mbcf")))
}

run_under <- function(lib, label) {
  cat("\nrunning under", label, "...\n")
  callr::r(fit_code, args = list(seed_data = 7, seed_mcmc = 101),
           libpath = c(lib, .libPaths()), show = FALSE)
}

a <- run_under(master_lib, "master")
b <- run_under(.libPaths()[1], "branch_treeshap (installed)")

# ---- 3. compare --------------------------------------------------------------
cmp <- function(nm) {
  x <- a[[nm]]; y <- b[[nm]]
  if (!identical(dim(x), dim(y))) {
    cat(sprintf("%-8s DIMENSIONS DIFFER: %s vs %s\n", nm,
                paste(dim(x), collapse = "x"), paste(dim(y), collapse = "x")))
    return(invisible(FALSE))
  }
  same <- identical(x, y)
  md   <- max(abs(as.numeric(x) - as.numeric(y)))
  cat(sprintf("%-8s identical: %-5s   max |difference|: %.3e\n", nm, same, md))
  invisible(same)
}

cat("\n================ master vs branch_treeshap, same seed ================\n")
ok <- c(cmp("tau"), cmp("mu"), cmp("sigma"), cmp("vi"))

cat("\n----------------------------------------------------------------------\n")
if (all(ok)) {
  cat("BIT-IDENTICAL. The branch does not perturb the sampler, so differing\n",
      "ENRICHD fits between branches came from different RNG states, not from\n",
      "the code. Any two unseeded mBCF runs differ by the same kind of amount.\n",
      "To confirm on ENRICHD: wrap the fit in set.seed(<same number>) and run\n",
      "it on each branch.\n", sep = "")
} else {
  cat("NOT IDENTICAL. The claim that the sampler is untouched is wrong and this\n",
      "needs to be found before anything else. The draw where they first diverge\n",
      "localises it: see below.\n", sep = "")
  d <- which(a$tau != b$tau)
  if (length(d)) {
    first_draw <- min(((d - 1) %% nrow(a$tau)) + 1)
    cat("first differing draw index:", first_draw, "of", nrow(a$tau), "\n")
    cat("if that is draw 1, the divergence is in setup or in the first\n",
        "iteration; if it is later, it is in something state-dependent.\n", sep = "")
  }
}

cat("\nmaster version string:", a$ver, " branch:", b$ver, "\n")
cat("temporary master build left at:", tmp, "\n")
