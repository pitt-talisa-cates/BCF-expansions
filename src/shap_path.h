// shap_path.h : the path algebra of path-dependent TreeSHAP (Lundberg et al.
//               2020, Algorithm 2: EXTEND / UNWIND / UNWOUND_SUM / RECURSE).
//
// Deliberately free of Rcpp, Armadillo and the mbcf tree class, so that it can
// be compiled and unit-tested on its own (see test/test_shap.cpp, which checks
// it against brute-force Shapley values on small hand-built trees). The tree is
// reached only through a Router object supplied by the caller:
//
//   typedef ... node_ptr;
//   bool     is_leaf(node_ptr) const
//   double   value  (node_ptr) const     // leaf value
//   int      var    (node_ptr) const     // split feature index, 0-based
//   double   coverage(node_ptr) const     // training rows reaching this node
//   node_ptr hot    (node_ptr) const     // child the observation itself takes
//   node_ptr cold   (node_ptr) const     // the other child
//
// A Router is built per observation, which is what keeps the mbcf split rules
// (value / MIA1 / MIA2 / pattern mixture) out of this file entirely.

#ifndef GUARD_shap_path_h
#define GUARD_shap_path_h

#include <vector>
#include <cstddef>

namespace mshap {

struct PathElement {
  int    d;   // feature index; -1 for the root sentinel
  double z;   // zero fraction: P(path continues | feature absent)
  double o;   // one  fraction: 1 if the observation takes this branch, else 0
  double w;   // proportion of subsets of each cardinality
};

// EXTEND: grow the path by one feature
inline void extend(std::vector<PathElement>& m, double pz, double po, int pi)
{
  const std::size_t l = m.size();              // length BEFORE the new element
  PathElement e;
  e.d = pi; e.z = pz; e.o = po; e.w = (l == 0) ? 1.0 : 0.0;
  m.push_back(e);
  if(l == 0) return;
  for(std::size_t i = l; i-- > 0; ) {          // i = l-1 ... 0
    m[i+1].w += po * m[i].w * (double)(i+1) / (double)(l+1);
    m[i].w    = pz * m[i].w * (double)(l-i)  / (double)(l+1);
  }
}

// UNWIND: remove the feature at index idx from the path, in place
inline void unwind(std::vector<PathElement>& m, std::size_t idx)
{
  const std::size_t L = m.size();
  const double iz = m[idx].z;
  const double io = m[idx].o;
  double nw = m[L-1].w;

  for(std::size_t j = L-1; j-- > 0; ) {        // j = L-2 ... 0
    if(io != 0.0) {
      const double t = m[j].w;
      m[j].w = nw * (double)L / ((double)(j+1) * io);
      nw = t - m[j].w * iz * (double)(L-(j+1)) / (double)L;
    } else {
      m[j].w = m[j].w * (double)L / (iz * (double)(L-(j+1)));
    }
  }
  for(std::size_t j = idx; j+1 < L; ++j) {
    m[j].d = m[j+1].d;
    m[j].z = m[j+1].z;
    m[j].o = m[j+1].o;
  }
  m.pop_back();
}

// UNWOUND_SUM: total weight the path would carry with feature idx removed.
// Same recursion as unwind() but accumulating, so the path is left intact for
// the next feature at this leaf.
inline double unwound_sum(const std::vector<PathElement>& m, std::size_t idx)
{
  const std::size_t L = m.size();
  const double iz = m[idx].z;
  const double io = m[idx].o;
  double total = 0.0;
  double nw = m[L-1].w;

  if(io != 0.0) {
    for(std::size_t j = L-1; j-- > 0; ) {
      const double t = nw * (double)L / ((double)(j+1) * io);
      total += t;
      nw = m[j].w - t * iz * (double)(L-(j+1)) / (double)L;
    }
  } else {
    if(iz == 0.0) return 0.0;                  // unreachable branch
    for(std::size_t j = L-1; j-- > 0; ) {
      total += m[j].w * (double)L / (iz * (double)(L-(j+1)));
    }
  }
  return total;
}

// RECURSE: accumulate SHAP values for one observation into phi_row, which must
// have room for at least (max feature index + 1) doubles and is NOT zeroed here
// (so contributions add across the trees of a forest).
template <class Router>
void recurse(typename Router::node_ptr nd,
             std::vector<PathElement> m,       // by value: each child needs its own path
             double pz, double po, int pi,
             const Router& tr,
             double* phi_row)
{
  extend(m, pz, po, pi);

  if(tr.is_leaf(nd)) {
    const double v = tr.value(nd);
    for(std::size_t i = 1; i < m.size(); ++i) {
      if(m[i].d < 0) continue;                 // sentinel
      const double w = unwound_sum(m, i);
      phi_row[m[i].d] += w * (m[i].o - m[i].z) * v;
    }
    return;
  }

  const double cn = tr.coverage(nd);
  if(cn <= 0.0) return;                        // no training row reaches here

  const int vsplit = tr.var(nd);
  typename Router::node_ptr hot  = tr.hot(nd);
  typename Router::node_ptr cold = tr.cold(nd);

  // if this feature is already on the path, pull it off and carry its fractions
  double iz = 1.0, io = 1.0;
  for(std::size_t i = 1; i < m.size(); ++i) {
    if(m[i].d == vsplit) { iz = m[i].z; io = m[i].o; unwind(m, i); break; }
  }

  const double ch = tr.coverage(hot);
  const double cc = tr.coverage(cold);
  if(ch > 0.0) recurse<Router>(hot,  m, iz * ch / cn, io,  vsplit, tr, phi_row);
  if(cc > 0.0) recurse<Router>(cold, m, iz * cc / cn, 0.0, vsplit, tr, phi_row);
}

} // namespace mshap

#endif
