// shap.h : mTreeSHAP -- path-dependent TreeSHAP (Lundberg et al. 2020, Alg. 2)
//          adapted to the mBCF split rules (value / MIA1 / MIA2 / pattern-mixture).
//
// Design notes
// ------------
// 1. Node coverage. Path-dependent TreeSHAP needs, at every INTERNAL node, the
//    fraction of training rows that continue to each child. The sampler's n0
//    field cannot be used for this: update_n0() refills bottom nodes only, it is
//    called only inside the na_seqreg block, and internal-node n0 is whatever
//    birth() wrote when that node was last a leaf (see the comment at tree.cpp
//    line ~494). After a few imputation updates the invariant
//        n0(parent) == n0(left) + n0(right)
//    no longer holds, and TreeSHAP's local-accuracy property breaks with it.
//    So we add a separate tree::coverage field, refreshed by update_coverage() below.
//    n0 / n0_xnew and update_n0() are left exactly as they were, which means the
//    MCMC is bit-for-bit unaffected by anything in this file, and running
//    update_coverage() more or less often cannot change the chain.
//
// 2. Routing. The hot/cold child decision delegates to sendleft() in funs.cpp,
//    the same function the sampler and all the sufficient-statistic code use.
//    This is deliberate: the split-rule encoding must never be duplicated. In
//    the CODE the encoding is
//        na = 0 : value split      x_hat < c                (the plan calls this IMP)
//        na = 1 : MIA1             x_hat < c  OR  missing
//        na = 2 : MIA2             x_hat < c  AND observed
//        na = 3 : pattern mixture  missing                  (the plan calls this PM)
//    Note this is NOT the numbering in Table 2.2 of mtreeshap_plan_v2, which has
//    PM = 0 and IMP = 3 and swaps MIA1/MIA2. Delegating to sendleft() means the
//    implementation follows the code, not the table.
//
// 3. Credited feature. As specified in the plan, a split always credits the
//    covariate index j, whatever the split rule. A pattern-mixture split on x_j
//    therefore contributes to phi_j, not to a separate missingness pseudo-feature.
//
// 4. Units. mtreeshap_forest() returns phi and the baseline on the internal
//    (y-standardised) scale of the forest it was given, multiplied by the caller-
//    supplied `scale`:
//        tau trees : scale = (bscale1 - bscale0)   so sum_j phi_ij + base = tau_i
//        mu trees  : scale = mscale                so sum_j phi_ij + base = m_i
//    The remaining sdy (and muy, for mu) rescaling happens in R, as it does for
//    b_post / m_post.

#ifndef GUARD_shap_h
#define GUARD_shap_h

#include <RcppArmadillo.h>
#include <vector>
#include "tree.h"
#include "info.h"
#include "funs.h"

//--------------------------------------------------
// Recompute tree::coverage for every node of x by routing all di.n rows from the
// root to their leaf under the current x_hat / rr. Guarantees
// coverage(parent) == coverage(left) + coverage(right) at every internal node.
// Touches nothing the sampler reads.
void update_coverage(tree& x, xinfo& xi, dinfo& di);

//--------------------------------------------------
// SHAP values for one forest, for all di.n rows.
//   phi   : di.n x di.p, OVERWRITTEN (not accumulated) by this call
//   scale : multiplies every phi and the baseline (see note 4 above)
//   returns the baseline, i.e. scale * sum_over_trees(cover-weighted mean leaf)
// update_coverage() is called internally for every tree, so the caller does not
// need to.
double mtreeshap_forest(std::vector<tree>& t,
                        xinfo& xi,
                        dinfo& di,
                        double scale,
                        arma::mat& phi);

//--------------------------------------------------
// Largest |sum_j phi_ij + baseline - f(x_i)| over rows; f is recomputed from the
// forest with fit_i(). This is the local-accuracy (efficiency) check and should
// be ~1e-10. Cheap enough to run at every SHAP draw under bugtest.
double mtreeshap_max_efficiency_error(std::vector<tree>& t,
                                      xinfo& xi,
                                      dinfo& di,
                                      double scale,
                                      const arma::mat& phi,
                                      double baseline);

#endif
