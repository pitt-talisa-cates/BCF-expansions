// test_shap_path.cpp
//
// Standalone check of src/shap_path.h against brute-force Shapley values.
// No R, no Rcpp, no Armadillo: compile and run with
//
//     g++ -std=c++11 -O2 -I../src -o test_shap_path test_shap_path.cpp && ./test_shap_path
//
// The reference is the definition of the Shapley value applied to the
// path-dependent (cover-conditional) value function
//
//     g(S) = expected prediction when only the features in S are known,
//            marginalising unknown features by the cover proportions
//
//     phi_j = sum_{S subseteq F\{j}} |S|!(|F|-|S|-1)!/|F|! * [g(S+j) - g(S)]
//
// which is exactly what path-dependent TreeSHAP is supposed to compute. The
// trees include all four mbcf split rules, repeated splits on one variable, and
// nodes with zero cover.

#include "shap_path.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

// ---------------------------------------------------------------- toy tree ---

struct Node {
  bool   leaf;
  double value;
  int    var;
  double cut;
  int    na;          // 0 value, 1 MIA1, 2 MIA2, 3 pattern mixture
  double cover;
  Node*  l;
  Node*  r;
  Node(double v) : leaf(true),  value(v), var(-1), cut(0), na(0), cover(0), l(0), r(0) {}
  Node(int j, double c, int rule, Node* L, Node* R)
    : leaf(false), value(0), var(j), cut(c), na(rule), cover(0), l(L), r(R) {}
};

// same rule set as sendleft() in src/funs.cpp
bool sendleft_ref(double xv, double rrv, double xcut, int NA)
{
  if(NA == 0)      return xv < xcut;
  else if(NA == 1) return xv < xcut || rrv == 1;
  else if(NA == 2) return xv < xcut && rrv == 0;
  else             return rrv == 1;
}

struct Row { std::vector<double> x, r; };

void accumulate_cover(Node* root, const std::vector<Row>& rows)
{
  // zero everything
  std::vector<Node*> stack; stack.push_back(root);
  while(!stack.empty()) {
    Node* n = stack.back(); stack.pop_back();
    n->cover = 0;
    if(!n->leaf) { stack.push_back(n->l); stack.push_back(n->r); }
  }
  for(size_t i=0;i<rows.size();++i) {
    Node* n = root;
    while(true) {
      n->cover += 1.0;
      if(n->leaf) break;
      n = sendleft_ref(rows[i].x[n->var], rows[i].r[n->var], n->cut, n->na) ? n->l : n->r;
    }
  }
}

// ------------------------------------------------------------ value function -

double g(Node* n, const Row& row, const std::set<int>& S)
{
  if(n->leaf) return n->value;
  if(n->cover <= 0) return 0.0;
  if(S.count(n->var)) {
    Node* c = sendleft_ref(row.x[n->var], row.r[n->var], n->cut, n->na) ? n->l : n->r;
    return g(c, row, S);
  }
  const double wl = n->l->cover / n->cover;
  const double wr = n->r->cover / n->cover;
  return wl*g(n->l, row, S) + wr*g(n->r, row, S);
}

double factorial(int k){ double f=1; for(int i=2;i<=k;++i) f*=i; return f; }

std::vector<double> brute_force(Node* root, const Row& row, int p)
{
  std::vector<double> phi(p, 0.0);
  const int nsub = 1 << p;
  for(int j=0;j<p;++j) {
    for(int mask=0; mask<nsub; ++mask) {
      if(mask & (1<<j)) continue;              // S must exclude j
      std::set<int> S;
      int sz = 0;
      for(int k=0;k<p;++k) if(mask & (1<<k)) { S.insert(k); ++sz; }
      const double w = factorial(sz)*factorial(p-sz-1)/factorial(p);
      std::set<int> Sj = S; Sj.insert(j);
      phi[j] += w * (g(root, row, Sj) - g(root, row, S));
    }
  }
  return phi;
}

// ------------------------------------------------------------------- router --

struct TestRouter {
  typedef Node* node_ptr;
  const Row* row;
  bool   is_leaf(node_ptr n) const { return n->leaf; }
  double value  (node_ptr n) const { return n->value; }
  int    var    (node_ptr n) const { return n->var; }
  double coverage(node_ptr n) const { return n->cover; }
  bool goes_left(node_ptr n) const {
    return sendleft_ref(row->x[n->var], row->r[n->var], n->cut, n->na);
  }
  node_ptr hot (node_ptr n) const { return goes_left(n) ? n->l : n->r; }
  node_ptr cold(node_ptr n) const { return goes_left(n) ? n->r : n->l; }
};

std::vector<double> treeshap(Node* root, const Row& row, int p)
{
  std::vector<double> phi(p, 0.0);
  TestRouter tr; tr.row = &row;
  std::vector<mshap::PathElement> m;
  mshap::recurse<TestRouter>(root, m, 1.0, 1.0, -1, tr, &phi[0]);
  return phi;
}

double predict(Node* n, const Row& row)
{
  while(!n->leaf) n = sendleft_ref(row.x[n->var], row.r[n->var], n->cut, n->na) ? n->l : n->r;
  return n->value;
}

double baseline(Node* n)
{
  // cover-weighted mean leaf value
  if(n->leaf) return n->value;
  const double root_cov = n->cover;
  // walk leaves
  std::vector<Node*> stack; stack.push_back(n);
  double b = 0.0;
  while(!stack.empty()) {
    Node* c = stack.back(); stack.pop_back();
    if(c->leaf) b += (c->cover/root_cov)*c->value;
    else { stack.push_back(c->l); stack.push_back(c->r); }
  }
  return b;
}

// -------------------------------------------------------------------- cases --

int failures = 0;
const double TOL = 1e-9;

// does this row's own path stay inside the support of the coverage rows?
bool path_defined(Node* n, const Row& row)
{
  while(true) {
    if(n->cover <= 0) return false;
    if(n->leaf) return true;
    n = sendleft_ref(row.x[n->var], row.r[n->var], n->cut, n->na) ? n->l : n->r;
  }
}

// cov_rows define node coverage (the training set); eval_rows are attributed
// (training or test). Passing the same vector twice is the in-sample case.
// cov_rows define node coverage (the training set); eval_rows are attributed
// (training or test). Passing the same vector twice is the in-sample case.
//
// Rows whose path leaves the support of cov_rows are excluded from both
// comparisons and counted instead. For such a row E[f | x_S] involves 0/0 at
// the unreachable node, so it has no value to be correct about: the brute-force
// reference happens to return the leaf value while the implementation
// contributes nothing, and neither convention is more right than the other.
// What matters, and what expect_undef checks, is that those rows are FLAGGED
// rather than reported as if they were fine.
void check_tree2(const char* name, Node* root,
                 const std::vector<Row>& cov_rows,
                 const std::vector<Row>& eval_rows, int p,
                 size_t expect_undef = 0)
{
  accumulate_cover(root, cov_rows);
  double worst_shap = 0.0, worst_eff = 0.0;
  size_t n_undef = 0, n_checked = 0;
  for(size_t i=0;i<eval_rows.size();++i) {
    if(!path_defined(root, eval_rows[i])) { ++n_undef; continue; }
    ++n_checked;
    std::vector<double> a = treeshap(root, eval_rows[i], p);
    std::vector<double> b = brute_force(root, eval_rows[i], p);
    double s = baseline(root);
    for(int j=0;j<p;++j) {
      worst_shap = std::max(worst_shap, std::fabs(a[j]-b[j]));
      s += a[j];
    }
    worst_eff = std::max(worst_eff, std::fabs(s - predict(root, eval_rows[i])));
  }
  const bool ok = (worst_shap < TOL) && (worst_eff < TOL) &&
                  (n_undef == expect_undef);
  if(!ok) ++failures;
  printf("%-34s  checked=%2zu p=%d   max|shap-brute| = %.3e   max eff err = %.3e   undef=%zu/%zu   %s\n",
         name, n_checked, p, worst_shap, worst_eff, n_undef, expect_undef,
         ok ? "PASS" : "FAIL");
}

void check_tree(const char* name, Node* root, const std::vector<Row>& rows, int p)
{
  check_tree2(name, root, rows, rows, p, 0);
}

std::vector<Row> make_rows(int p, int n, unsigned seed, double pmiss)
{
  std::vector<Row> rows(n);
  unsigned s = seed;
  for(int i=0;i<n;++i) {
    rows[i].x.resize(p); rows[i].r.resize(p);
    for(int j=0;j<p;++j) {
      s = s*1103515245u + 12345u;
      const double u = (double)((s>>16) & 0x7fff)/32767.0;
      s = s*1103515245u + 12345u;
      const double u2 = (double)((s>>16) & 0x7fff)/32767.0;
      rows[i].r[j] = (u2 < pmiss) ? 1.0 : 0.0;
      rows[i].x[j] = u;                        // imputed value present either way
    }
  }
  return rows;
}

int main()
{
  printf("path-dependent TreeSHAP vs brute-force Shapley\n");
  printf("----------------------------------------------\n");

  // 1. plain value splits, depth 2, three features
  {
    Node* t = new Node(0, 0.5, 0,
                new Node(1, 0.5, 0, new Node(1.0), new Node(-2.0)),
                new Node(2, 0.5, 0, new Node(3.0), new Node(0.5)));
    check_tree("value splits, depth 2", t, make_rows(3, 40, 1, 0.0), 3);
  }

  // 2. same shape, but with missing data present and an MIA1 split
  {
    Node* t = new Node(0, 0.5, 1,
                new Node(1, 0.5, 0, new Node(1.0), new Node(-2.0)),
                new Node(2, 0.5, 0, new Node(3.0), new Node(0.5)));
    check_tree("MIA1 at root", t, make_rows(3, 40, 2, 0.3), 3);
  }

  // 3. MIA2 and pattern-mixture splits together
  {
    Node* t = new Node(0, 0.4, 2,
                new Node(1, 0.6, 3, new Node(2.0), new Node(-1.0)),
                new Node(2, 0.5, 1, new Node(0.25), new Node(-0.75)));
    check_tree("MIA2 root + PM and MIA1 below", t, make_rows(3, 60, 3, 0.4), 3);
  }

  // 4. the same variable split twice on one path (exercises UNWIND)
  {
    Node* t = new Node(0, 0.7, 0,
                new Node(0, 0.3, 0, new Node(1.0), new Node(-1.0)),
                new Node(1, 0.5, 0, new Node(2.0), new Node(-2.0)));
    check_tree("repeated split on x0", t, make_rows(2, 40, 4, 0.0), 2);
  }

  // 5. variable split twice with different rules (value then pattern mixture)
  {
    Node* t = new Node(0, 0.5, 0,
                new Node(0, 0.5, 3, new Node(1.5), new Node(-0.5)),
                new Node(0, 0.5, 1, new Node(0.25), new Node(2.0)));
    check_tree("x0 three times, mixed rules", t, make_rows(2, 60, 5, 0.35), 2);
  }

  // 6. deeper tree, four features, heavy missingness
  {
    Node* t =
      new Node(0, 0.5, 1,
        new Node(1, 0.5, 0,
          new Node(2, 0.5, 3, new Node(1.0), new Node(-1.0)),
          new Node(3, 0.5, 2, new Node(0.5), new Node(2.5))),
        new Node(2, 0.3, 0,
          new Node(0, 0.8, 0, new Node(-2.0), new Node(0.75)),
          new Node(1.25)));
    check_tree("depth 4, four features", t, make_rows(4, 80, 6, 0.45), 4);
  }

  // 7. a branch no training row reaches (zero cover on the cold side)
  {
    Node* t = new Node(0, 0.5, 0,
                new Node(1, 0.5, 0, new Node(1.0), new Node(-2.0)),
                new Node(9.0));
    std::vector<Row> rows = make_rows(2, 30, 7, 0.0);
    for(size_t i=0;i<rows.size();++i) rows[i].x[0] = 0.1;   // everyone goes left
    check_tree("zero-cover right branch", t, rows, 2);
  }

  // 8. stump
  {
    Node* t = new Node(0.42);
    check_tree("stump", t, make_rows(2, 10, 8, 0.2), 2);
  }

  // ---- out-of-sample: coverage from the training rows, SHAP on test rows ----
  printf("\ncoverage from training rows, SHAP evaluated on test rows\n");
  printf("----------------------------------------------\n");

  // 9. test rows from the same distribution, no missingness
  {
    Node* t = new Node(0, 0.5, 0,
                new Node(1, 0.5, 0, new Node(1.0), new Node(-2.0)),
                new Node(2, 0.5, 0, new Node(3.0), new Node(0.5)));
    check_tree2("out-of-sample, value splits", t,
                make_rows(3, 200, 11, 0.0), make_rows(3, 40, 12, 0.0), 3);
  }

  // 10. test rows with missingness, all four split rules in the tree
  {
    Node* t =
      new Node(0, 0.5, 1,
        new Node(1, 0.5, 0,
          new Node(2, 0.5, 3, new Node(1.0), new Node(-1.0)),
          new Node(3, 0.5, 2, new Node(0.5), new Node(2.5))),
        new Node(2, 0.3, 0,
          new Node(0, 0.8, 0, new Node(-2.0), new Node(0.75)),
          new Node(1.25)));
    check_tree2("out-of-sample, all four rules", t,
                make_rows(4, 300, 13, 0.4), make_rows(4, 60, 14, 0.4), 4);
  }

  // 11. test rows with a DIFFERENT missingness rate from the training rows:
  //     the background distribution is the training one, which is the point
  {
    Node* t = new Node(0, 0.5, 1,
                new Node(1, 0.4, 3, new Node(2.0), new Node(-1.0)),
                new Node(1, 0.6, 0, new Node(0.25), new Node(-0.75)));
    check_tree2("out-of-sample, shifted missingness", t,
                make_rows(2, 300, 15, 0.15), make_rows(2, 60, 16, 0.6), 2);
  }

  // 12. test rows that reach a leaf no training row reaches: attribution is
  //     undefined there and must be FLAGGED, not silently wrong
  {
    Node* t = new Node(0, 0.5, 0,
                new Node(1, 0.5, 0, new Node(1.0), new Node(-2.0)),
                new Node(9.0));
    std::vector<Row> tr = make_rows(2, 100, 17, 0.0);
    for(size_t i=0;i<tr.size();++i) tr[i].x[0] = 0.1;        // training all left
    std::vector<Row> te = make_rows(2, 40, 18, 0.0);
    for(size_t i=0;i<te.size();++i) te[i].x[0] = 0.9;        // test all right
    check_tree2("out-of-sample, unreachable leaf", t, tr, te, 2, /*expect_undef=*/40);
  }

  printf("----------------------------------------------\n");
  printf("%s\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return failures == 0 ? 0 : 1;
}
