#ifndef RNG_H
#define RNG_H
#include <RcppArmadillo.h>

using std::vector;

class RNG
{
private:
public:
  // Continuous Distributions
  double uniform(double x = 0.0, double y = 1.0)
  { return R::runif(x, y); }
  double normal(double mu = 0.0, double sd = 1.0)
  { return R::rnorm(mu, sd); }
  double gamma(double shape = 1, double scale = 1)
  { return R::rgamma(shape, 1)*scale; }
  double chi_square(double df)
  { return R::rchisq(df); }//return gamma(df / 2.0, 0.5); }
  double beta(double a1, double a2)
  { const double x1 = gamma(a1, 1); return (x1 / (x1 + gamma(a2, 1))); }
  
  void uniform(vector<double>& res, double x = 0.0, double y = 1.0) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = uniform(x, y);
  }
  void normal(vector<double>& res, double mu = 0.0, double sd = 1.0) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = normal(mu, sd);
  }
  void gamma(vector<double>& res, double shape = 1, double scale = 1) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = gamma(shape, scale);
  }
  void chi_square(vector<double>& res, double df) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = chi_square(df);
  }
  void beta(vector<double>& res, double a1, double a2) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = beta(a1, a2);
  }
}; // class RNG


inline double znorm() {
  return R::rnorm(0.0, 1.0);
}


template<class T>
int rdisc_log(T &logweights, int n=-1) {
  if(n==-1) n = logweights.size();
  typename T::iterator itb = logweights.begin();
  typename T::iterator ite = logweights.begin() + n;
  
  double m = *std::max_element(itb, ite);
  double s = 0;
  std::vector<double> weights(n, 0.0);
  for(int i=0; i<n; ++i) {
    weights[i] = exp(logweights[i] - m);
    s += weights[i];
  }
  double u = s*R::runif(0.0, 1.0);
  double cs = weights[0];
  int i=0;
  while((u>cs) & (i<n)) {
    ++i;
    cs += weights[i];
  }
  return i;
}


template<class T>
int rdisc_log_inplace(T &logweights, int n=-1, double u=-1.0) {
  if(n==-1) n = logweights.size();
  if(u==-1) u = R::runif(0.0, 1.0);
  typename T::iterator itb = logweights.begin();
  typename T::iterator ite = logweights.begin() + n;
  
  double m = *std::max_element(itb, ite);
  double s = 0;
  //vector<double> weights(n, 0.0);
  for(int i=0; i<n; ++i) {
    logweights[i] = exp(logweights[i] - m);
    s += logweights[i];
  }
  u = s*u;
  double cs = logweights[0];
  int i=0;
  while((u>cs) & (i<n)) {
    ++i;
    cs += logweights[i];
  }
  return i;
}

double rtnormlo0(double lo);
double rtnormlo1(double mean, double lo);
double rtnormhi1(double mean, double lo);
double rtnormlo(double mean, double sd, double lo);

// Draw from N(mu, 1) truncated to the open interval (lo, hi).
// Uses the inverse-CDF method via R::pnorm / R::qnorm, consistent
// with the rest of the RNG infrastructure in this package.
// Either bound may be -Inf / +Inf (passed as e.g. R_NegInf, R_PosInf).
// The latent scale for ordinal variables is always sigma=1, so sigma
// is not a parameter here — standardise externally if needed.
inline double rtnorm_interval(double mu, double lo, double hi) {
  double plo = R::pnorm(lo, mu, 1.0, 1, 0);
  double phi = R::pnorm(hi, mu, 1.0, 1, 0);
  double u   = R::runif(plo, phi);
  return R::qnorm(u, mu, 1.0, 1, 0);
}


//sample from N(Phi^(-1)m, Phi^(-1))
arma::vec rmvnorm_post(arma::vec &m, arma::mat &Phi);


#endif // RNG_H