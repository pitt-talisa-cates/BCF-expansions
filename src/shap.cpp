// shap.cpp : see shap.h for the design notes. The path algebra lives in
//            shap_path.h; this file supplies the mbcf-specific routing and the
//            forest-level bookkeeping.

#include "shap.h"
#include "shap_path.h"
#include <cmath>

using Rcpp::Rcout;

//==================================================================
// Node coverage
//==================================================================

void update_coverage(tree& x, xinfo& xi, dinfo& di)
{
  tree::npv nds;
  x.getnodes(nds);
  for(size_t k=0; k<nds.size(); ++k) nds[k]->setcoverage(0.0);

  for(size_t i=0; i<di.n; ++i) {
    double** xx = di.x  + i*di.p;
    double** rr = di.rr + i*di.p;
    tree::tree_p nd = x.getptr();              // root
    while(true) {
      nd->setcoverage(nd->getcoverage() + 1.0);
      if(nd->getl() == 0) break;               // leaf
      size_t v = nd->getv();
      bool left = sendleft(*(xx[v]), *(rr[v]), xi[v][nd->getc()], nd->getna());
      nd = left ? nd->getl() : nd->getr();
    }
  }
}

//==================================================================
// Routing for one observation
//==================================================================

namespace {

struct MbcfRouter {
  typedef tree::tree_p node_ptr;

  double** xx;
  double** rr;
  xinfo*   xi;

  bool   is_leaf(node_ptr n) const { return n->getl() == 0; }
  double value  (node_ptr n) const { return n->getm(); }
  int    var    (node_ptr n) const { return (int)n->getv(); }
  double coverage(node_ptr n) const { return n->getcoverage(); }

  // The single source of truth for the mbcf split rules is sendleft() in
  // funs.cpp, shared with the sampler and the sufficient-statistic code.
  bool goes_left(node_ptr n) const {
    const size_t v = n->getv();
    return sendleft(*(xx[v]), *(rr[v]), (*xi)[v][n->getc()], n->getna());
  }
  node_ptr hot (node_ptr n) const { return goes_left(n) ? n->getl() : n->getr(); }
  node_ptr cold(node_ptr n) const { return goes_left(n) ? n->getr() : n->getl(); }
};

// cover-weighted mean leaf value of one tree = E[f] under the cover-implied
// background distribution that path-dependent TreeSHAP conditions on
double tree_baseline(tree& t)
{
  tree::npv bots;
  t.getbots(bots);
  const double root = t.getptr()->getcoverage();
  if(root <= 0.0) return 0.0;
  double b = 0.0;
  for(size_t k=0; k<bots.size(); ++k) {
    b += (bots[k]->getcoverage() / root) * bots[k]->getm();
  }
  return b;
}

} // anonymous namespace

//==================================================================
// Forest-level driver
//==================================================================

double mtreeshap_forest(std::vector<tree>& t,
                        xinfo& xi,
                        dinfo& di,
                        double scale,
                        arma::mat& phi)
{
  double baseline = 0.0;

  // row-major scratch: arma matrices are column-major, so we cannot hand the
  // recursion a contiguous pointer into a row of phi
  std::vector<double> buf(di.n * di.p, 0.0);

  for(size_t k=0; k<t.size(); ++k) {

    update_coverage(t[k], xi, di);
    baseline += tree_baseline(t[k]);

    if(t[k].getl() == 0) continue;             // stump: all of it is baseline

    for(size_t i=0; i<di.n; ++i) {
      MbcfRouter tr;
      tr.xx = di.x  + i*di.p;
      tr.rr = di.rr + i*di.p;
      tr.xi = &xi;
      std::vector<mshap::PathElement> m;
      m.reserve(16);
      mshap::recurse<MbcfRouter>(t[k].getptr(), m, 1.0, 1.0, -1, tr, &buf[i*di.p]);
    }
  }

  phi.set_size(di.n, di.p);
  for(size_t i=0; i<di.n; ++i)
    for(size_t j=0; j<di.p; ++j)
      phi(i,j) = scale * buf[i*di.p + j];

  return scale * baseline;
}

double mtreeshap_max_efficiency_error(std::vector<tree>& t,
                                      xinfo& xi,
                                      dinfo& di,
                                      double scale,
                                      const arma::mat& phi,
                                      double baseline)
{
  double worst = 0.0;
  for(size_t i=0; i<di.n; ++i) {
    const double fx = scale * fit_i(i, t, xi, di);
    double s = baseline;
    for(size_t j=0; j<di.p; ++j) s += phi(i,j);
    const double err = std::fabs(s - fx);
    if(err > worst) worst = err;
  }
  return worst;
}
