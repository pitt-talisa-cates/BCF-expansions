//VBT: Armadillo is the linear-algrbra package for cpp
#include <RcppArmadillo.h>    

#include <iostream>
#include <fstream>

//defines vector class container (from std library)
// see http://www.cplusplus.com/reference/vector/vector/
#include <vector>
#include <algorithm>
#include <cmath>

//a C library ctime, has functions for manipulating date/time variables
#include <ctime>

// header files from the BCF package
#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"
#include "shap.h"
#include <ctime>

using namespace Rcpp;


// y = m(x) + b(x)z + e, e~N(0, sigma^2_y

//x_con is the design matrix for m. It should have n = rows
//x_mod is the design matrix for b. It should have n = rows
//data should come in sorted with all trt first, then control cases

//note that X_mod_ and X-con_ are numeric vectors

// [[Rcpp::export]]

List  mbcf_cpp(NumericVector y_, 
               NumericVector z_,
               NumericVector x_, 
               NumericVector rr_,
               NumericVector vcol_mod_,
               
               List x_info_list, 
               NumericVector rr_info_list,
               
               NumericVector x_est_, 
               NumericVector rr_est_,
               
               arma::mat random_des,      //needs to come in with n rows no matter what(?) VBT: so confused...it is importing a 1x1 matrix
               arma::mat random_var,     
               arma::mat random_var_ix,   //random_var_ix*random_var = diag(Var(random effects))
               double random_var_df,
               
               int burn, 
               int nd, 
               int thin, //Draw nd*thin + burn samples, saving nd draws after burn-in
               int ntree_mod, 
               int ntree_con,
               
               bool bugtest,
               
               double lambda,      // Scale parameter in the chisq prior on sigma^2_y
               double nu,          // Degrees of freedom in the chisq prior on sigma^2_y
               double con_sd,      // SD(m(x)); Can either be fixed (see Chipman), or given a half-Cauchy distribution (when use_mscale=TRUE)
               double mod_sd,      // SD(b(x)): Can either be fixed, or given a half-normal hyperprior (when use_bscale=TRUE)
               
               double con_alpha,   // These are the Chipman prior parameters, specified separately for control and moderator
               double con_beta,    // alpha is equivalent to base, or eta (default is 0.95); 
               double mod_alpha,   // beta is equivalent to power (default is 2)
               double mod_beta,
               
               CharacterVector treef_name_,   //tempdir() in R...gives the temp directory on my machine
               
               List na_stuff,
               NumericVector na_seqreg_modtyp,
               double na_prob_pm_con,
               double na_prob_pm_mod,
               
               bool na_mia = true,
               bool na_seqreg =false,
               bool check_node_missing = true,
               
               double trt_init=1.0,
               
               int status_interval=100,       //after how many iterations should an update be printed to console?
               bool use_mscale=true, //Use a half-Cauchy hyperprior on the scale of mu.
               bool use_bscale=true, //Use a half-normal hyperprior on the scale of tau.
               bool b_half_normal=true,
               
               double min_node=5,
               
               bool compute_shap=false,
               bool shap_mod=true,
               bool shap_con=false,
               int  shap_thin=10,
               bool shap_var=false,
               bool shap_store_draws=false,
               bool shap_check=false,
               bool shap_est=false,
               bool vi_coverage=false,
               bool estimate_cate=true
)
{
  // ---- mTreeSHAP controls (see shap.h) ---------------------------------------
  // compute_shap      master switch
  // shap_mod          SHAP for the tau(x) / b(x) trees
  // shap_con          SHAP for the prognostic mu(x) / m(x) trees
  // shap_thin         compute SHAP every shap_thin-th POST-BURN-IN iteration.
  //                   Independent of the thin argument above, which only
  //                   controls which draws get SAVED.
  // shap_var          also accumulate the sum of squares
  // shap_store_draws  keep every computed SHAP draw (n x p x K)
  // shap_check        report the worst local-accuracy error per draw
  // shap_est          also attribute the x_est rows, using TRAINING coverage
  // vi_coverage       additionally accumulate vi_mod weighted by refreshed node
  //                   coverage, for comparison against the n0-weighted version
  //
  // estimate_cate     FALSE fits the prognostic function ONLY: the moderate
  //                   forest is not fitted at all and z is not used. This is the
  //                   prediction-only mode, not a way of hiding tau from the
  //                   output. It works because the control forest is already
  //                   independent of treatment: it is updated with the
  //                   homoskedastic bd()/drmu() and never reads z or bscale,
  //                   whereas only the moderate forest uses bdhet()/drmuhet()
  //                   with the precs weights. So this is a set of gates, not a
  //                   rewrite. Two things do need explicit handling rather than
  //                   just emptying loops, both marked [CATE GATE] below:
  //                     - taking a pointer into an empty xtp_mod/rrtp_mod is
  //                       undefined behaviour, so di_mod must not be built
  //                     - the bscale update divides by allfit_mod, which is
  //                       identically 0 here, so that block must be skipped
  //
  // NB keep the argument list above free of comments. Rcpp::compileAttributes()
  // flattens the signature onto one line, so a comment spanning more than one
  // line gets spliced into it and generates a bogus extra parameter.
  // ---------------------------------------------------------------------------
  
  
  bool randeff = true;
  if(random_var_ix.n_elem == 1) {
    randeff = false;
  }
  if(randeff) Rcout << "Using random effects." << std::endl;
  
  std::string treef_name = as<std::string>(treef_name_);
  Rcout << treef_name_ << std::endl;
  std::ofstream treef(treef_name.c_str());
  
  RNGScope scope;
  RNG gen; 
  
  //useful stuff before I get started
  size_t n = y_.length();
  size_t p_con = x_.length()/n;
  size_t p_mod = vcol_mod_.length();
  
  const bool have_est = (x_est_.size() > 0);
  if (have_est && (x_est_.size() % p_con != 0))
    Rcpp::stop("x_est_ length must be a multiple of p_con");
  
  
  /*****************************************************************************/
  // Read, format y 
  /*****************************************************************************/
  
  // sufficient stats (see info.h)
  sinfo allys;  
  allys.n = n;
  
  //storage for y: column vector
  arma::vec y(n);
  for(size_t i=0; i<n; i++) {
    *(y.memptr()+i)= y_[i];
    allys.sy += y_[i]; 
    allys.sy2 += (y_[i])*(y_[i]);
  }
  
  // set some useful quantities
  double ybar = allys.sy/n; //sample mean
  double shat = sqrt((allys.sy2-n*ybar*ybar)/(n-1)); //sample standard deviation
  
  // Missing outcomes are not supported: mbcf() stops before reaching here if y
  // has any NA. The fully-conditional y draw that used to live in this file has
  // been removed rather than left dormant -- interval coverage for tau was very
  // poor and the idea is not coming back.
  
  /*****************************************************************************/
  // Read, format x and rr for the training set */
  /*****************************************************************************/
  
  // x: storage for x: an n*(p_con+1) matrix
  arma::mat x(n, p_con+1);
  for(size_t i=0; i<n; ++i){
    x(i,0) = 1.0;
  }
  for(size_t j=1; j<(p_con+1); ++j){
    for(size_t i=0; i<n; ++i){
      x(i,j) = x_[0 + i * p_con +(j-1)];
    }
  }
  
  // xtp_con: store pointers to x in transpose fashion, without 1's column, for m(x)
  std::vector<double*> xtp_con;
  for(size_t i=0; i<n; ++i){
    for(size_t j=1; j<(p_con+1); ++j){
      xtp_con.push_back(x.memptr()+j*n + i);
    }
  }
  
  // xtp_mod: store pointers to x in transpose fashion, without 1's column, for b(x)
  std::vector<size_t> vcol_mod;
  for(NumericVector::iterator it=vcol_mod_.begin(); it!=vcol_mod_.end(); ++it){
    vcol_mod.push_back(*it);
  }
  std::vector<double*> xtp_mod;
  for(size_t i=0; i<n; ++i){
    for(size_t j=1; j<(p_con+1); ++j){
      if( std::count( vcol_mod.begin(), vcol_mod.end(), j) ){
        xtp_mod.push_back(x.memptr() + j*n + i);
      }
    }
  }
  
  
  Rcout << "Using " << p_con << " control variables." << std::endl;
  
  // xi_con and xi_mod: the lists of cutoff thresholds for m(x) amd b(x)
  xinfo xi_con;
  xinfo xi_mod;
  xi_con.resize(p_con);
  xi_mod.resize(p_mod);
  size_t v=0;
  for(size_t i=0; i<p_con; ++i) {
    NumericVector tmp = x_info_list[i];
    std::vector<double> tmp2;
    for(int j=0; j<tmp.length(); ++j) {
      tmp2.push_back(tmp[j]);
    }
    xi_con[i] = tmp2;
    if(std::count(vcol_mod.begin(), vcol_mod.end(), i+1)){
      xi_mod[v] = tmp2;
      v++;
    }
  }
  
  // rri_con and rri_mod: the vectors saying whether v has missing or not
  vec_s rri_con;
  vec_s rri_mod;
  for(size_t j=0; j<p_con; ++j){
    rri_con.push_back(rr_info_list[j]);
    if(std::count(vcol_mod.begin(), vcol_mod.end(), j+1)){
      rri_mod.push_back(rr_info_list[j]);
    }
  }
  
  // rr: store rr in n*(p_con+1) dim arma::mat
  arma::mat rr(n,p_con+1);
  for(size_t i=0; i<n; i++) {
    rr(i,0) = 0.0;
  }
  for(size_t j=1; j<(p_con+1); ++j){
    for(size_t i=0; i<n; ++i){
      rr(i,j) = rr_[0 + i * p_con +(j-1)];
    }
  }
  
  // rrtp_con: store rr pointers in transpose order for m(x)
  std::vector<double*> rrtp_con;
  for(size_t i=0; i<n; ++i){
    for(size_t j=1; j<(p_con+1); ++j){
      rrtp_con.push_back(rr.memptr()+j*n + i);
    }
  }
  
  // rrtp_mod; store rr pointers in transpose order for b(x)
  std::vector<double*> rrtp_mod;
  for(size_t i=0; i<n; ++i){
    for(size_t j=1; j<(p_con+1); ++j){
      if(std::count(vcol_mod.begin(), vcol_mod.end(), j)){
        rrtp_mod.push_back(rr.memptr()+j*n + i);
      }
    }
  }
  
  
  /*****************************************************************************/
  // Read, format X_est, the test set data 
  /*****************************************************************************/
  
  size_t n_est = have_est ? x_est_.size() / p_con : 0;
  arma::mat x_est;
  std::vector<double*> xtp_mod_est;
  arma::mat rr_est;
  std::vector<double*> rrtp_mod_est;
  std::vector<double*> xtp_con_est;
  std::vector<double*> rrtp_con_est;
  
  
  //When x_est_ is a vector of length 1, I'm not doing prediction
  
  if (have_est) {
    // x_est: an n_est*(p_con+1) matrix
    x_est.set_size(n_est, p_con+1);
    for (size_t i = 0; i < n_est; ++i) x_est(i,0) = 1.0;
    for (size_t j = 1; j < (p_con+1); ++j)
      for (size_t i = 0; i < n_est; ++i)
        x_est(i,j) = x_est_[0 + i * p_con + (j-1)];
    
    // xtp_mod_est: pointers to columns in x_est for b(x)
    for (size_t i = 0; i < n_est; ++i)
      for (size_t j = 1; j < (p_con+1); ++j)
        if (std::count(vcol_mod.begin(), vcol_mod.end(), j))
          xtp_mod_est.push_back(x_est.memptr() + j*n_est + i);
        
        // rr_est: arma::mat of dim n_est*(p_con+1)
        rr_est.set_size(n_est, p_con+1);
        for (size_t i = 0; i < n_est; ++i) rr_est(i,0) = 0.0;
        for (size_t j = 1; j < (p_con+1); ++j)
          for (size_t i = 0; i < n_est; ++i)
            rr_est(i,j) = rr_est_[0 + i * p_con + (j-1)];
        
        // rrtp_mod_est: pointers to rr_est for b(x)
        for (size_t i = 0; i < n_est; ++i)
          for (size_t j = 1; j < (p_con+1); ++j)
            if (std::count(vcol_mod.begin(), vcol_mod.end(), j))
              rrtp_mod_est.push_back(rr_est.memptr() + j*n_est + i);
    
    // xtp_con_est / rrtp_con_est: pointers for m(x) on the test set. These were
    // declared but never filled, which is why the control trees had no test-set
    // dinfo and mu(x) could not be evaluated at x_est at all. All p_con columns,
    // in the same order as the training xtp_con.
    for (size_t i = 0; i < n_est; ++i)
      for (size_t j = 1; j < (p_con+1); ++j)
        xtp_con_est.push_back(x_est.memptr() + j*n_est + i);
    for (size_t i = 0; i < n_est; ++i)
      for (size_t j = 1; j < (p_con+1); ++j)
        rrtp_con_est.push_back(rr_est.memptr() + j*n_est + i);
  }
  
  /*****************************************************************************/
  // Stuff for sequential regressions       
  /*****************************************************************************/
  
  // pull seqreg stuff
  size_t na_reg_j1(na_stuff[0]);
  size_t na_train_j1(na_stuff[1]);
  
  std::vector<size_t> na_test_js;
  if (na_stuff.size() > 2) {
    NumericVector nat = na_stuff[2];
    for (int i = 0; i < nat.size(); ++i)
      na_test_js.push_back((size_t)nat[i]);
  }
  
  const bool has_test_na = have_est && !na_test_js.empty();
  size_t na_test_j1 = 0;
  if (has_test_na) na_test_j1 = na_test_js.front();
  
  std::vector<size_t> xmodtyp;
  xmodtyp.push_back(0);
  for(int j=0; j<na_seqreg_modtyp.length(); ++j){
    xmodtyp.push_back(na_seqreg_modtyp[j]);
  }
  
  //I can have columns of xbeta that I only use in the test set
  size_t na_tr_xb_j1 = na_train_j1 - na_reg_j1;  
  size_t p_xb = p_con-na_reg_j1+1;
  
  xinfop na_tr_ptr;
  xinfo_s na_tr_i;
  xinfop na_te_ptr;
  xinfo_s na_te_i;
  
  std::vector<double> xsigs, sig2_xprop;
  // Smoothed acceptance rate per missing covariate, used to adapt sig2_xprop
  // for continuous variables during burn-in via Robbins-Monro. Initialised to
  // 0.44 (the target rate) so adaptation starts from a neutral position.
  // Only meaningful for xmodtyp==0; carried for all variables for index alignment.
  std::vector<double> xprop_accept_rate;
  arma::mat xbetas(p_con, p_xb, arma::fill::zeros);
  
  arma::mat zz(n, p_con+1, arma::fill::zeros);
  
  // kappa_ord: cutpoints for ordinal variables, one inner vector per ordinal variable
  // encountered in the loop na_train_j1..p_con, in order of encounter.
  // ord_cats:  sorted unique integer category values per ordinal variable (same order).
  // Both are indexed by a dedicated counter ord_ctr that increments only for xmodtyp==2.
  std::vector<std::vector<double>> kappa_ord;
  std::vector<std::vector<double>> ord_cats;
  
  if(na_seqreg){
    
    //store pointers to missing data
    
    for(size_t j=na_train_j1; j<(p_con+1); j++){
      std::vector<double*> tmp;
      std::vector<size_t> tmp2;
      for(size_t i=0; i<n; i++){
        if(rr(i,j)==1){
          tmp.push_back(x.memptr()+j*n + i);
          tmp2.push_back(i);
        }
      }
      na_tr_ptr.push_back(tmp);
      na_tr_i.push_back(tmp2);
    }
    
    //if (n_est>=1){ //Code111: Does this break if I have no missing data in x_est?
    //  //store pointers to missing data
    //  for(size_t j=na_test_j1; j<(p_con+1); j++){
    //    //for the columns in x_test with missing data
    //    if(std::count(na_test_js.begin(), na_test_js.end(), j)){
    //      std::vector<double*> tmp;
    //      std::vector<size_t> tmp2;
    //      for(size_t i=0; i<n_est; i++){
    //        if(rr_est(i,j) == 1){
    //          tmp.push_back(x_est.memptr()+j*n_est + i);
    //          tmp2.push_back(i);
    //        }
    //      }
    //      na_te_ptr.push_back(tmp);
    //      na_te_i.push_back(tmp2);
    //    }
    //  }
    //}
    if (has_test_na) {
      na_te_ptr.clear();
      na_te_i.clear();
      for (size_t jj : na_test_js) {
        std::vector<double*> tmp;
        std::vector<size_t>  tmp2;
        for (size_t i = 0; i < n_est; ++i) {
          if (rr_est(i, jj) == 1) {
            tmp.push_back(x_est.memptr() + jj * n_est + i);
            tmp2.push_back(i);
          }
        }
        na_te_ptr.push_back(tmp);
        na_te_i.push_back(tmp2);
      }
    }
    
    //zz: initialize latent normals (binary and ordinal x)
    // For binary:  observed rows draw from one-sided truncated N; missing rows draw N(0,1)
    // For ordinal: observed rows draw from two-sided truncated N within bracketing cutpoint
    //              interval; missing rows draw N(0,1). Cutpoints initialised from empirical
    //              category proportions of the latent draws (Albert-Chib style).
    size_t k=0;
    for(size_t j=na_train_j1; j<(p_con+1); ++j){
      if(xmodtyp[j]==1){
        // Binary: Albert-Chib latent normal
        for(size_t i=0; i<n; ++i){
          if(rr(i,j)==0){
            if(x(i,j)==0){
              zz(i,j) = rtnormhi1(0.0,0.0);
            } else {
              zz(i,j) = rtnormlo1(0.0,0.0);
            }
          } else {
            zz(i,j) = gen.normal();
            x(i,j) = (double) (zz(i,j)>0);
          }
        }
      } else if (xmodtyp[j]==2){
        // Ordinal: Albert-Chib latent normal with K-1 cutpoint intervals
        
        // Collect sorted unique category values from observed rows
        std::vector<double> cats_j;
        for(size_t i=0; i<n; ++i){
          if(rr(i,j)==0){
            double cv = x(i,j);
            if(std::find(cats_j.begin(), cats_j.end(), cv) == cats_j.end()){
              cats_j.push_back(cv);
            }
          }
        }
        std::sort(cats_j.begin(), cats_j.end());
        size_t K = cats_j.size();
        ord_cats.push_back(cats_j);
        
        // Count observations per category (observed only) to compute empirical proportions
        std::vector<double> counts_k(K, 0.0);
        double n_obs_j = 0.0;
        for(size_t i=0; i<n; ++i){
          if(rr(i,j)==0){
            for(size_t m=0; m<K; ++m){
              if(x(i,j)==cats_j[m]){ counts_k[m] += 1.0; break; }
            }
            n_obs_j += 1.0;
          }
        }
        
        // Cutpoints from empirical cumulative proportions mapped through qnorm.
        // kappa_j[m] separates cats_j[m] from cats_j[m+1], m=0..K-2.
        // Implicit boundaries: -Inf below cats_j[0], +Inf above cats_j[K-1].
        std::vector<double> kappa_j;
        double cum = 0.0;
        for(size_t m=0; m<K-1; ++m){
          cum += counts_k[m] / n_obs_j;
          kappa_j.push_back(R::qnorm(cum, 0.0, 1.0, 1, 0));
        }
        kappa_ord.push_back(kappa_j);
        
        // Draw latent normals:
        // - Observed rows: truncated N(0,1) within (kappa[m-1], kappa[m])
        //   where kappa[-1]=-Inf, kappa[K-1]=+Inf
        // - Missing rows: draw from N(0,1) and set x to category implied by
        //   which interval the draw falls in
        for(size_t i=0; i<n; ++i){
          if(rr(i,j)==0){
            // Find category index m for this observation
            size_t m = 0;
            for(size_t mm=0; mm<K; ++mm){
              if(x(i,j)==cats_j[mm]){ m=mm; break; }
            }
            double lo_m = (m==0)   ? R_NegInf : kappa_j[m-1];
            double hi_m = (m==K-1) ? R_PosInf : kappa_j[m];
            zz(i,j) = rtnorm_interval(0.0, lo_m, hi_m);
          } else {
            // Missing: unconditional draw; category assigned by cutpoint lookup
            zz(i,j) = gen.normal();
            // Map latent draw to category
            size_t cat_idx = K-1;
            for(size_t m=0; m<K-1; ++m){
              if(zz(i,j) <= kappa_j[m]){ cat_idx = m; break; }
            }
            x(i,j) = cats_j[cat_idx];
          }
        }
      }
      k++;
    }
    
    
    //xm residual variance for proposal distribution and posterior initial value
    k=0;
    for(size_t j=na_train_j1; j<(p_con+1); ++j){   //na_train_j1==4 means column 5 of matrix x
      arma::vec xy;
      if(xmodtyp[j]==0){
        xy=x.col(j);
      } else if (xmodtyp[j]==1){
        xy=zz.col(j);
      } else if (xmodtyp[j]==2){
        // Ordinal: use latent zz values as response, same as binary
        xy=zz.col(j);
      }
      arma::mat xx=x.submat(0,0,n-1,j-1);
      arma::mat Q,R;
      qr_econ(Q,R,xx);
      
      arma::vec bhat = solve(R,Q.t())*xy;
      arma::vec yhat = xx*bhat;
      double rss=0.0;
      for(size_t i=0; i<n; i++){
        rss+=(xy[i]-yhat[i])*(xy[i]-yhat[i]);
      }
      if(xmodtyp[j]==0){
        sig2_xprop.push_back(rss/(n-bhat.n_elem));
        xsigs.push_back(-0.5/(rss/(n-bhat.n_elem)));                 //xsig = -1/(2*sig_x)
      } else if(xmodtyp[j]==1){
        sig2_xprop.push_back(1.0);
        xsigs.push_back(-0.5);
      } else if(xmodtyp[j]==2){\
        // Ordinal: latent scale is standardised (sigma=1), same as binary
        sig2_xprop.push_back(1.0);
        xsigs.push_back(-0.5);
      }
      xprop_accept_rate.push_back(0.44);  // initialise to target acceptance rate
      
      for(size_t i=0; i<bhat.n_elem; ++i){
        xbetas(i,k)=bhat[i];
      }
      ++k;
    }
    
  }
  
  
  /*****************************************************************************/
  // Set up the bart models       
  /*****************************************************************************/
  
  //VBT: set ntrt to 0
  size_t ntrt = 0;
  
  //VBT: count up the number of treated observations
  for(size_t i=0; i<n; ++i) {
    if(z_[i]>0) ntrt += 1;
  }
  
  // centered treatment coding: z_centered in {-0.5, +0.5}
  // ensures mu(x) is the prognostic function blind to treatment assignment
  // matches the bscale0=-0.5, bscale1=0.5 initialization in Hahn et al. (2020) eq (5.6)
  std::vector<double> z_centered(n);
  for(size_t i=0; i<n; ++i) {
    z_centered[i] = z_[i] - 0.5;
  }
  
  // [CATE GATE] no moderate forest in prognostic-only mode. Setting ntree_mod to
  // zero makes every `for(k=0; k<ntree_mod; k++)` loop over the moderate trees a
  // no-op automatically, which covers most of the sampler for free.
  if(!estimate_cate) ntree_mod = 0;
  
  //initialize the trees
  std::vector<tree> t_mod(ntree_mod);
  for(int i=0; i<ntree_mod; i++) {
    t_mod[i].setm(1.0/(double)ntree_mod);
    t_mod[i].setn0((double)n);
    t_mod[i].setn0_xnew((double)n);
  }
  std::vector<tree> t_con(ntree_con);
  for(int i=0; i<ntree_con; i++) {
    t_con[i].setm(ybar/(double)ntree_con);
    t_con[i].setn0((double)n);
    t_con[i].setn0_xnew((double)n);
  }
  
  
  //--------------------------------------------------
  //prior parameters
  
  //the treatment effect parameterization variables; scale impacts scale of b(x) here
  double bscale_prec = 2;
  double bscale0 = -0.5;
  double bscale1 = 0.5;
  
  //the scale parameters for mu(x) 
  double mscale_prec = 1.0;
  double mscale = 1.0;
  
  //What is this again?
  double delta_con = 1.0;
  double delta_mod = 1.0;
  
  double sigma = shat;
  
  //b(x) tree building parameters
  pinfo pi_mod;
  pi_mod.pbd = 1.0;                                                               //prob of birth/death move
  pi_mod.pb = .5;                                                                 //prob of birth given  birth/death
  pi_mod.alpha = mod_alpha;                                                       //prob a bot node splits is alpha/(1+d)^beta, d is depth of node
  pi_mod.beta  = mod_beta;                                                        //2 for bart means it is harder to build big trees.
  // [CATE GATE] guard the division by sqrt(ntree_mod); the value is never used
  // with no moderate trees, but leaving it as inf would be a trap
  pi_mod.tau   = (ntree_mod > 0)
    ? mod_sd/(sqrt(delta_mod)*sqrt((double) ntree_mod))
    : 0.0;               //sigma_mu, variance on leaf parameters  
  pi_mod.sigma = shat;                                                            //resid variance is \sigma^2_y/bscale^2 in the backfitting update
  pi_mod.pr_pm = na_prob_pm_mod;
  
  //m(x) tree building parameters
  pinfo pi_con;
  pi_con.pbd = 1.0; 
  pi_con.pb = 0.5; 
  pi_con.alpha = con_alpha;
  pi_con.beta  = con_beta;
  pi_con.tau   = con_sd/(sqrt(delta_con)*sqrt((double) ntree_con));
  pi_con.sigma = shat/fabs(mscale); 
  pi_con.pr_pm = na_prob_pm_con;
  
  //initialize the portion of yhat coming from m(x)
  double* allfit_con = new double[n]; //sum of fit of all trees
  for(size_t i=0;i<n;i++) {
    allfit_con[i] = ybar;
  }
  
  //initialize the portion of yhat coming from b(x)
  double* allfit_mod = new double[n]; //sum of fit of all trees
  for(size_t i=0;i<n;i++) {
    // [CATE GATE] identically zero with no moderate forest, so it flows
    // harmlessly through r, allfit and the mscale update
    allfit_mod[i] = estimate_cate
      ? (z_[i]*bscale1 + (1-z_[i])*bscale0)*trt_init
      : 0.0;
  }
  
  
  
  //--------------------------------------------------
  // dinfo for control function m(x)
  
  //VBT: initialize residuals 
  double* r_con = new double[n];      // y-(allfit-ftemp) = y - allfit + ftemp
  
  dinfo di_con;
  di_con.n=n;
  di_con.p = p_con;
  di_con.x = &(xtp_con[0]);         //address of the pointer to x transpose
  di_con.rr = &(rrtp_con[0]);       //address of teh pointer to rr transpose
  di_con.y = r_con;
  
  
  //--------------------------------------------------
  // dinfo for tau function b(x)
  
  double* r_mod = new double[n];
  
  // [CATE GATE] &(xtp_mod[0]) on an empty vector is undefined behaviour, so in
  // prognostic-only mode di_mod stays empty rather than pointing at nothing.
  // Nothing reads it: every consumer is inside a loop over the (zero) moderate
  // trees or inside a block gated on estimate_cate.
  dinfo di_mod;
  di_mod.n = estimate_cate ? n : 0;
  di_mod.p = estimate_cate ? p_mod : 0;
  di_mod.x  = (estimate_cate && !xtp_mod.empty())  ? &(xtp_mod[0])  : nullptr;
  di_mod.rr = (estimate_cate && !rrtp_mod.empty()) ? &(rrtp_mod[0]) : nullptr;
  di_mod.y = r_mod;
  
  
  //--------------------------------------------------
  //dinfo for predictions
  
  dinfo di_mod_est;           // initialize to "empty"
  di_mod_est.n = 0;
  di_mod_est.p = 0;
  di_mod_est.x = nullptr;
  di_mod_est.rr = nullptr;
  di_mod_est.y = 0;
  
  dinfo di_con_est;           // initialize to "empty"
  di_con_est.n = 0;
  di_con_est.p = 0;
  di_con_est.x = nullptr;
  di_con_est.rr = nullptr;
  di_con_est.y = 0;
  
  if (have_est) {
    di_mod_est.n = n_est;
    di_mod_est.p = p_mod;
    di_mod_est.x = &(xtp_mod_est[0]);
    di_mod_est.rr = &(rrtp_mod_est[0]);
    di_mod_est.y = 0;
    
    di_con_est.n = n_est;
    di_con_est.p = p_con;
    di_con_est.x = &(xtp_con_est[0]);
    di_con_est.rr = &(rrtp_con_est[0]);
    di_con_est.y = 0;
  }
  
  //--------------------------------------------------
  //setup for random effects
  size_t random_dim = random_des.n_cols;
  int nr=1;
  if(randeff) nr = n;
  
  arma::vec r(nr); //working residuals
  arma::vec Wtr(random_dim); // W'r
  
  arma::mat WtW = random_des.t()*random_des; //W'W
  arma::mat Sigma_inv_random = diagmat(1/(random_var_ix*random_var));
  
  // PX parameters
  arma::vec eta(random_var_ix.n_cols); //random_var_ix is num random effects by num variance components
  eta.fill(1.0);
  
  for(int i=0; i<nr; ++i) {
    r(i) = y[i] - allfit_con[i] - allfit_mod[i];
  }
  
  Wtr = random_des.t()*r;
  arma::vec gamma = solve(WtW/(sigma*sigma)+Sigma_inv_random, Wtr/(sigma*sigma));
  arma::vec allfit_random = random_des*gamma;
  if(!randeff) allfit_random.fill(0);
  
  
  
  //--------------------------------------------------
  //storage for the fits
  double* allfit = new double[n]; //yhat
  for(size_t i=0;i<n;i++) {
    allfit[i] = allfit_mod[i] + allfit_con[i]; //VBT: allfit_mod already takes Z into account
    if(randeff) {
      allfit[i] += allfit_random[i]; 
    }
  }
  
  
  
  //fit of current tree
  double* ftemp  = new double[n]; 
  
  // temp storage for conditional ''precisions'' in heteroskedastic updates
  double* precs = new double[n]; 
  
  NumericVector sigma_post(nd);
  NumericVector msd_post(nd);
  NumericVector bsd_post(nd);
  NumericMatrix m_post(nd,n);
  NumericMatrix yhat_post(nd,n);
  NumericMatrix b_post(nd,n);
  NumericMatrix b_est_post(nd,n_est);
  NumericMatrix m_est_post(nd,n_est);   // mu(x) at x_est; gives yhat on the test set
  arma::mat gamma_post(nd,gamma.n_elem);
  arma::mat random_var_post(nd,random_var.n_elem);
  
  // Coverage-weighted variable importance for tau (b(x)) trees
  // Rows = variables (p_mod), cols = na type (0=value/SeqReg, 1=MIA1, 2=MIA2, 3=pattern)
  arma::mat vi_mod(p_mod, 4, arma::fill::zeros);
  // Companion to vi_mod, weighted by REFRESHED node coverage rather than by n0.
  // vi_mod weights each internal node by getn0(), which for an internal node is
  // whatever birth() wrote when that node was last a leaf, so it drifts as the
  // imputed x moves (see the comment at tree.cpp line ~494). This accumulates
  // the same counts weighted by coverage recomputed from the current x, so the
  // two can be compared directly on real data. Off by default: it costs one
  // update_coverage() pass per saved draw.
  arma::mat vi_mod_cov(p_mod, 4, arma::fill::zeros);
  
  /*****************************************************************************/
  // mTreeSHAP accumulators
  /*****************************************************************************/
  
  if(shap_thin < 1) shap_thin = 1;
  // [CATE GATE] no moderate forest means no tau attributions to compute
  const bool do_shap_mod = compute_shap && shap_mod && estimate_cate;
  const bool do_shap_con = compute_shap && shap_con;
  if(compute_shap && !do_shap_mod && !do_shap_con)
    Rcpp::stop("compute_shap is TRUE but both shap_mod and shap_con are FALSE");
  
  // number of SHAP draws: iterations mc in [burn, burn + nd*thin) with
  // (mc - burn) %% shap_thin == 0
  const int n_shap_max = compute_shap ? (int)std::ceil((double)(nd*thin)/(double)shap_thin) : 0;
  int shap_ctr = 0;
  
  // running accumulators. The N x p stratum accumulators of the plan (4.1) are
  // deliberately NOT allocated: for a given cell (i,j) the missingness status
  // r_ij is fixed across iterations, so every cell belongs entirely to one
  // stratum and sum_abs_shap_mis[i,j] is either sum_abs_shap[i,j] or 0. The
  // stratified per-variable summaries of 4.2 are therefore recovered exactly in
  // R from shap_mean_abs plus the missingness mask, at no memory cost.
  arma::mat shap_sum_mod, shap_sumabs_mod, shap_sumsq_mod, phi_mod;
  arma::mat shap_sum_con, shap_sumabs_con, shap_sumsq_con, phi_con;
  arma::vec shap_base_mod, shap_base_con, shap_err_mod, shap_err_con;
  arma::cube shap_draws_mod, shap_draws_con;
  IntegerVector shap_iters(n_shap_max);
  arma::uvec undef_mod, undef_con;
  
  // Out-of-sample attributions: the same accumulators over the x_est rows. Node
  // coverage still comes from the TRAINING rows, so the background distribution,
  // and therefore the baseline, is identical to the in-sample case. That is what
  // makes in-sample and out-of-sample attributions comparable, and it is why the
  // _est output shares the in-sample baseline vector rather than carrying its
  // own. n_undef counts, per test row, the draws at which that row's path
  // reached a node with no training coverage; see shap.h.
  const bool do_shap_mod_est = do_shap_mod && shap_est && have_est;
  const bool do_shap_con_est = do_shap_con && shap_est && have_est;
  arma::mat shap_sum_mod_est, shap_sumabs_mod_est, shap_sumsq_mod_est, phi_mod_est;
  arma::mat shap_sum_con_est, shap_sumabs_con_est, shap_sumsq_con_est, phi_con_est;
  arma::vec shap_err_mod_est, shap_err_con_est;
  arma::cube shap_draws_mod_est, shap_draws_con_est;
  arma::uvec shap_undef_mod_est, shap_undef_con_est;
  double shap_seconds = 0.0;
  
  if(do_shap_mod) {
    shap_sum_mod.zeros(n, p_mod);
    shap_sumabs_mod.zeros(n, p_mod);
    phi_mod.zeros(n, p_mod);
    shap_base_mod.zeros(n_shap_max);
    if(shap_var) shap_sumsq_mod.zeros(n, p_mod);
    if(shap_store_draws) shap_draws_mod.zeros(n, p_mod, n_shap_max);
    if(shap_check) shap_err_mod.zeros(n_shap_max);
  }
  if(do_shap_con) {
    shap_sum_con.zeros(n, p_con);
    shap_sumabs_con.zeros(n, p_con);
    phi_con.zeros(n, p_con);
    shap_base_con.zeros(n_shap_max);
    if(shap_var) shap_sumsq_con.zeros(n, p_con);
    if(shap_store_draws) shap_draws_con.zeros(n, p_con, n_shap_max);
    if(shap_check) shap_err_con.zeros(n_shap_max);
  }
  if(do_shap_mod_est) {
    shap_sum_mod_est.zeros(n_est, p_mod);
    shap_sumabs_mod_est.zeros(n_est, p_mod);
    shap_undef_mod_est.zeros(n_est);
    if(shap_var) shap_sumsq_mod_est.zeros(n_est, p_mod);
    if(shap_store_draws) shap_draws_mod_est.zeros(n_est, p_mod, n_shap_max);
    if(shap_check) shap_err_mod_est.zeros(n_shap_max);
  }
  if(do_shap_con_est) {
    shap_sum_con_est.zeros(n_est, p_con);
    shap_sumabs_con_est.zeros(n_est, p_con);
    shap_undef_con_est.zeros(n_est);
    if(shap_var) shap_sumsq_con_est.zeros(n_est, p_con);
    if(shap_store_draws) shap_draws_con_est.zeros(n_est, p_con, n_shap_max);
    if(shap_check) shap_err_con_est.zeros(n_shap_max);
  }
  if(compute_shap) {
    Rcout << "mTreeSHAP: " << n_shap_max << " draws (every " << shap_thin
          << " post-burn-in iterations)";
    if(do_shap_mod) Rcout << "; tau";
    if(do_shap_con) Rcout << "; mu";
    if(do_shap_mod_est || do_shap_con_est)
      Rcout << "; x_est (" << n_est << " rows, training coverage)";
    Rcout << std::endl;
  }
  //std::vector<double> xaccept;
  std::vector<double> xm1_post;
  std::vector<double> xm1_est_post;
  //std::vector<double> min_obs_node;
  
  
  
  
  
  //bool split00;
  std::vector<bool> split_post;
  std::vector<size_t> v_post,c_post,na_post;
  std::vector<int> L_post,U_post,Lna_post,Una_post;
  std::vector<double> alpha_post,mul_post;
  
  /*
   if(bugtest){
   
   
   std::vector<double> xtp_con_out;
   for(int i=0; i<(n*p_con); ++i){
   xtp_con_out.push_back(*(xtp_con[i]));
   }
   
   std::vector<double> xtp_mod_out;
   for(int i=0; i<(n*p_con); ++i){
   xtp_mod_out.push_back(*(xtp_mod[i]));
   }
   
   std::vector<double> rrtp_con_out;
   for(int i=0; i<(n*p_con); ++i){
   rrtp_con_out.push_back(*rrtp_con[i]);
   }
   
   std::vector<double> rrtp_mod_out;
   for(int i=0; i<(n*p_con); ++i){
   rrtp_mod_out.push_back(*rrtp_mod[i]);
   }
   
   
   return(List::create(_["rri_con"] = rri_con, 
   _["rri_mod"] = rri_mod,
   _["xtp_con"] = xtp_con_out,
   _["xtp_mod"] = xtp_mod_out,
   _["rrtp_con"] = rrtp_con_out,
   _["rrtp_mod"] = rrtp_mod_out,
   _["x"] = x));
   
   }
   
   */
  
  /*****************************************************************************/
  // MCMC
  // note: the allfit objects are all carrying the appropriate scales
  //
  /*****************************************************************************/
  
  Rcout << "\nBeginning MCMC:\n";
  time_t tp;
  int time1 = time(&tp);
  
  Rcout << "predictions for N_est=" << n_est << std::endl;
  
  size_t save_ctr = 0;
  for(int mc=0; mc<(nd*thin+burn); mc++) {
    
    if(mc%status_interval==0) {
      Rcout << "iteration: " << mc << " sigma: "<< sigma << endl;
    }
    
    /*
     if(bugtest){
     Rcout << "MC="<< mc << endl;
     Rcout << "TYPE=con"<<endl;
     for(size_t i=0; i<t_con.size(); ++i){
     tree::npv bots_this_k;
     t_con[i].getbots(bots_this_k);
     //Rcout << bots_this_k.size() << " ";
     for(tree::npv::size_type k=0; k<bots_this_k.size(); ++k){
     Rcout << "T" << i+1 << "N" << k+1  << ";; n0: " << bots_this_k[k]->getn0() <<", n0_xnew: " << bots_this_k[k]->getn0_xnew() << "    ";
     }
     Rcout << endl;
     }
     Rcout << "TYPE=mod"<<endl;
     for(size_t i=0; i<t_mod.size(); ++i){
     tree::npv bots_this_k;
     t_mod[i].getbots(bots_this_k);
     //Rcout << bots_this_k.size() << " ";
     for(tree::npv::size_type k=0; k<bots_this_k.size(); ++k){
     Rcout << "T" << i+1 << "N" << k+1  << ";; n0: " << bots_this_k[k]->getn0() <<", n0_xnew: " << bots_this_k[k]->getn0_xnew() << "    ";
     }
     Rcout << endl;
     }
     }
     */
    
    vec_dp ptr;
    vec_s ivec;
    
    double ysig = -1.0/(2*sigma*sigma);
    
    if (na_seqreg){
      
      ////////////
      //draw missing x
      ////////////
      
      //update node counts
      for(tree::vec_t::size_type k=0; k<t_con.size(); ++k){
        update_n0(t_con[k], xi_con, di_con);
      }
      for(tree::vec_t::size_type k=0; k<t_mod.size(); ++k){
        update_n0(t_mod[k], xi_mod, di_mod);
      }
      
      //loop through x's with missing values
      size_t ctrj_xb = na_tr_xb_j1;                           //use ctrj_xb to pick up col of objects with p_con-na_reg_j1 cols
      size_t ctrj_xm = 0;                                     //use ctrj_xm to pick up col of objects with p_con-na_train_j1 cols
      for(size_t jj=na_train_j1; jj<(p_con+1); ++jj){  //use jj to pick up col of objects with p_con+1 cols
        ptr = na_tr_ptr[ctrj_xm];
        ivec = na_tr_i[ctrj_xm];
        arma::vec xb = xbetas.submat(0, ctrj_xb, jj-1, ctrj_xb);  //I will always have jj-1 terms: 
        
        //need to get the trees that split on jj
        tree::npv t_con_splt_v;
        allsplitv(t_con, jj-1, t_con_splt_v);
        
        tree::npv t_mod_splt_v;
        allsplitv(t_mod, jj-1, t_mod_splt_v);
        
        
        //loop through the missing values for variable x
        for(size_t ii=0; ii<ptr.size(); ++ii){    //use ii to pick out row of objects with n rows
          
          size_t iobs = ivec[ii];
          
          arma::rowvec xi_old = x.row(iobs);
          arma::rowvec xxi = xi_old.subvec(0,jj-1);
          arma::rowvec xi_new = xi_old;
          double xij_old, xij_new;
          
          // Independence sampler: propose from sequential regression predictive
          // distribution N(xxi*xb, sig2) centred at the current regression mean,
          // rather than a random walk centred at the current value.
          // For binary/ordinal: proposal variance == likelihood variance (both = 1
          // on the latent scale), so the proposal ratio cancels exactly against the
          // sequential regression likelihood term in the acceptance ratio -- the
          // net acceptance ratio reduces to the outcome likelihood ratio alone.
          // For continuous: proposal variance is sig2_xprop (adaptively tuned during
          // burn-in via Robbins-Monro, see below) while likelihood variance is
          // sig2_x (current posterior draw); these differ in general, so a correction
          // term is added to loga below.
          double xmean = arma::as_scalar(xxi * xb);  // sequential regression mean for this obs
          
          if(xmodtyp[jj]==0){
            // Continuous: independence proposal N(xmean, sig2_xprop)
            xij_old = xi_old[jj];
            xij_new = xmean + gen.normal(0.0, 1.0)*sqrt(sig2_xprop[ctrj_xb]);
            xi_new[jj] = xij_new;
          } else if (xmodtyp[jj]==1){
            // Binary: independence proposal N(xmean, 1) on latent scale,
            // threshold at zero to get binary value (Albert-Chib style, no truncation
            // since x is missing -- observed value does not constrain which side of
            // zero the latent draw lives on)
            xij_old = zz(iobs,jj);
            xij_new = xmean + gen.normal(0.0, 1.0);
            xi_new[jj] = (double)(xij_new>0);
          } else if (xmodtyp[jj]==2){
            // Ordinal: independence proposal N(xmean, 1) on latent scale,
            // map through cutpoints to get ordinal category
            xij_old = zz(iobs,jj);
            xij_new = xmean + gen.normal(0.0, 1.0);
            size_t ord_ctr = 0;
            for(size_t jjj=na_train_j1; jjj<jj; ++jjj){
              if(xmodtyp[jjj]==2) ++ord_ctr;
            }
            const std::vector<double>& kappa_jj = kappa_ord[ord_ctr];
            const std::vector<double>& cats_jj   = ord_cats[ord_ctr];
            size_t K_jj = cats_jj.size();
            size_t cat_idx = K_jj - 1;
            for(size_t m=0; m<K_jj-1; ++m){
              if(xij_new <= kappa_jj[m]){ cat_idx = m; break; }
            }
            xi_new[jj] = cats_jj[cat_idx];
          }
          
          double yi = y[iobs];
          double bscl = (iobs<ntrt) ? bscale1 : bscale0;
          
          //NOTE: plugging in xi_old (or xi_new) to fit_it would not work if mod vars != con vars
          // this is because "v" of mod trees operate relative to dimensions of xtp_mod (need to confirm)
          // this means that sendleft() wouldnt access the right variables
          
          double yhat_old = allfit[iobs];
          double yhat_new_con = allfit_con[iobs];
          double yhat_new_mod = allfit_mod[iobs];
          
          //or, I have a situation where I draw mu from its prior distribution...
          
          double ftemp_old=0.0;
          double ftemp_new=0.0;
          tree::npv bot_npv;
          double min_n0_xnew=n;
          
          std::vector<double> bot_fits_con;
          std::vector<double> bot_fits_mod;
          
          bool same_cat = ((xmodtyp[jj]==1 || xmodtyp[jj]==2) && (xi_new[jj] == xi_old[jj]));
          
          //loop thru trees that split on v to get fits
          if(!same_cat){
            if(t_con_splt_v.size()>0){
              for(size_t k=0; k<t_con_splt_v.size(); ++k){
                //need to check if x_new leads to 1[valid split below bot]==0
                fit_xprop(iobs, t_con_splt_v[k], xi_old, xi_new, di_con, xi_con, 
                          ftemp_old, ftemp_new, bot_npv, min_n0_xnew);
                yhat_new_con = yhat_new_con - mscale*ftemp_old + mscale*ftemp_new;  //code111
              }
            }
            if(t_mod_splt_v.size()){
              for(size_t k=0; k<t_mod_splt_v.size(); ++k){
                fit_xprop(iobs, t_mod_splt_v[k], xi_old, xi_new, di_mod, xi_mod,
                          ftemp_old, ftemp_new, bot_npv, min_n0_xnew);
                yhat_new_mod = yhat_new_mod - bscl*ftemp_old + bscl*ftemp_new;  //code111
              }
            }
          }
          double yhat_new = yhat_new_con + yhat_new_mod;
          
          if(min_n0_xnew>=min_node){
            // MH acceptance ratio for independence sampler.
            //
            // For binary and ordinal (xmodtyp 1,2): proposal is N(xmean, 1) and
            // the sequential regression likelihood is also N(xmean, 1) on the latent
            // scale (xsigs = -0.5, sig2_x = 1). The proposal ratio log q(old)/q(new)
            // exactly cancels the sequential regression likelihood ratio
            // log p(xnew|seqreg)/p(xold|seqreg), so loga reduces to the outcome
            // likelihood ratio only (plus downstream seqreg terms for later covariates).
            //
            // For continuous (xmodtyp 0): proposal is N(xmean, sig2_xprop) but the
            // sequential regression likelihood uses sig2_x (current posterior draw,
            // stored as xsigs = -1/(2*sig2_x)). These differ in general, so the
            // cancellation is only partial. The residual proposal correction is:
            //   log q(old)/q(new) - log p(xnew|seqreg)/p(xold|seqreg)
            //   = 0.5*(1/sig2_xprop - 1/sig2_x) * [(xnew-xmean)^2 - (xold-xmean)^2]
            //   = 0.5*(1/sig2_xprop + 2*xsigs) * [(xnew-xmean)^2 - (xold-xmean)^2]
            // This is added to loga after the standard llnew-llold computation.
            //
            // The expensive part is re-evaluating the tree fit when x changes leaf;
            // this is only done when same_cat==false (i.e. the proposed x value
            // actually changes the discrete observed value for binary/ordinal, or
            // changes the continuous value for xmodtyp==0).
            double llold = arma::as_scalar(ysig*(yi-yhat_old)*(yi-yhat_old) + xsigs[ctrj_xb]*(xij_old-xmean)*(xij_old-xmean));
            double llnew = arma::as_scalar(ysig*(yi-yhat_new)*(yi-yhat_new) + xsigs[ctrj_xb]*(xij_new-xmean)*(xij_new-xmean));
            
            if(jj>na_train_j1){
              for(size_t it=0; it<(jj-na_train_j1); ++it){
                arma::vec xxb = xbetas.submat(0,ctrj_xb-1-it,jj-2-it,ctrj_xb-1-it);
                
                arma::rowvec xxi0 = xi_old.subvec(0,jj-2-it); //only predictors are updated
                arma::rowvec xxi1 = xi_new.subvec(0,jj-2-it);
                
                double xij = xi_old[jj-1-it]; //outcome won't have changed, could have used xi_new
                
                llold+=arma::as_scalar(xsigs[ctrj_xb-1-it]*(xij-xxi0*xxb)*(xij-xxi0*xxb));
                llnew+=arma::as_scalar(xsigs[ctrj_xb-1-it]*(xij-xxi1*xxb)*(xij-xxi1*xxb));
              }
            }
            
            //accept or reject
            double loga = llnew-llold;
            
            // Proposal correction for continuous independence sampler:
            // accounts for mismatch between proposal variance (sig2_xprop, adaptively
            // tuned during burn-in) and sequential regression likelihood variance
            // (sig2_x, current posterior draw). For binary/ordinal this correction is
            // zero (both variances == 1) and is omitted for clarity.
            if(xmodtyp[jj]==0){
              double prop_correction = 0.5*(1.0/sig2_xprop[ctrj_xb] + 2.0*xsigs[ctrj_xb]) *
                ((xij_new-xmean)*(xij_new-xmean) - (xij_old-xmean)*(xij_old-xmean));
              loga += prop_correction;
            }
            
            double u = gen.uniform(0.0,1.0);
            bool accepted = (log(u) <= std::min(0.0,loga));
            
            // Adaptive proposal variance for continuous x during burn-in.
            // Uses exponential smoothing of per-proposal accept/reject outcomes
            // (lambda=0.01, ~100-proposal effective window) feeding a Robbins-Monro
            // update of log(sig2_xprop) targeting 44% acceptance with step size
            // c/(mc+c), c=50. Robust to variable missingness rates: sparse variables
            // adapt across iterations, dense variables adapt within a single iteration.
            // Frozen once mc >= burn so posterior draws use a fixed proposal variance.
            if(xmodtyp[jj]==0 && mc < burn){
              const double lambda    = 0.01;
              const double c_rm      = 50.0;
              const double tgt_rate  = 0.44;
              xprop_accept_rate[ctrj_xb] = (1.0 - lambda) * xprop_accept_rate[ctrj_xb]
              + lambda * (accepted ? 1.0 : 0.0);
              double gamma_adapt = c_rm / ((double)mc + c_rm);
              sig2_xprop[ctrj_xb] = exp(log(sig2_xprop[ctrj_xb])
                                          + gamma_adapt * (xprop_accept_rate[ctrj_xb] - tgt_rate));
            }
            
            if(accepted){
              if(xmodtyp[jj]==0){
                *(ptr[ii]) = xij_new;
              } else if(xmodtyp[jj]==1){
                *(ptr[ii]) = (double)(xij_new>0);
                zz(iobs,jj) = xij_new;
              } else if(xmodtyp[jj]==2){
                // Accept: update x to the proposed category and zz to the proposed latent value
                *(ptr[ii]) = xi_new[jj];
                zz(iobs,jj) = xij_new;
              }
              //go thru bot_npv and update n0, allfit_con, allfit_mod
              if(bot_npv.size()){
                for(tree::npv::size_type kk=0; kk<bot_npv.size(); ++kk){
                  (bot_npv[kk])->setn0((bot_npv[kk])->getn0_xnew());
                }
                allfit_con[iobs] = yhat_new_con;
                allfit_mod[iobs] = yhat_new_mod;
                allfit[iobs] = yhat_new;
              }
            } else {
              //go thru bot_npv and reset n0_xnew
              if(bot_npv.size()){
                for(tree::npv::size_type kk=0; kk<bot_npv.size(); ++kk){
                  (bot_npv[kk])->setn0_xnew((bot_npv[kk])->getn0());
                }
              }
            }
            
          } else {
            //go thru bot_npv and reset n0_xnew
            if(bot_npv.size()){
              for(tree::npv::size_type kk=0; kk<bot_npv.size(); ++kk){
                (bot_npv[kk])->setn0_xnew( (bot_npv[kk])->getn0() );
              }
            }
          }
          
        }
        
        ++ctrj_xb;
        ++ctrj_xm;
      }
      
      //if(bugtest){
      //  stop("gothere");
      //}
      
      ////////////
      //draw parameters for x models
      ////////////
      
      ctrj_xb=0;
      for(size_t jj=na_reg_j1; jj<(p_con+1); ++jj){   //na_train_j1==4 means column 5 of matrix x
        
        //first update zz for binary and ordinal x (observed rows only;
        // missing rows were handled in the MH block above)
        arma::mat xx=x.submat(0,0,n-1,jj-1);
        arma::vec xy;
        if(xmodtyp[jj]==0){
          xy=x.col(jj);
        } else if (xmodtyp[jj]==1){
          arma::vec xb = xbetas.submat(0, ctrj_xb, jj-1, ctrj_xb);
          arma::vec xx_means = xx*xb;
          for(size_t i=0; i<n; ++i){
            if(rr(i,jj)==0){ //already updated zz for rr(i,jj)==1
              if(x(i,jj)==0){
                zz(i,jj) = rtnormhi1(xx_means[i],0.0);
              } else {
                zz(i,jj) = rtnormlo1(xx_means[i],0.0);
              }
            }
          }
          xy=zz.col(jj);
        } else if (xmodtyp[jj]==2){
          // Ordinal: update latent zz for observed rows using truncated N within
          // the bracketing cutpoint interval for each observation's category.
          // Then update cutpoints using the Albert-Chib uniform draw between
          // adjacent order statistics of the latent values.
          
          // Find ord_ctr index for this variable
          size_t ord_ctr = 0;
          for(size_t jjj=na_train_j1; jjj<jj; ++jjj){
            if(xmodtyp[jjj]==2) ++ord_ctr;
          }
          std::vector<double>& kappa_jj = kappa_ord[ord_ctr];
          const std::vector<double>& cats_jj = ord_cats[ord_ctr];
          size_t K_jj = cats_jj.size();
          
          arma::vec xb = xbetas.submat(0, ctrj_xb, jj-1, ctrj_xb);
          arma::vec xx_means = xx*xb;
          
          // Update zz for observed rows: draw from truncated N(xbeta_i, 1)
          // within the cutpoint interval for observation i's category
          for(size_t i=0; i<n; ++i){
            if(rr(i,jj)==0){
              // Find category index
              size_t m = 0;
              for(size_t mm=0; mm<K_jj; ++mm){
                if(x(i,jj)==cats_jj[mm]){ m=mm; break; }
              }
              double lo_m = (m==0)      ? R_NegInf : kappa_jj[m-1];
              double hi_m = (m==K_jj-1) ? R_PosInf : kappa_jj[m];
              zz(i,jj) = rtnorm_interval(xx_means[i], lo_m, hi_m);
            }
          }
          
          // Update cutpoints: Albert-Chib uniform draw.
          // For each interior cutpoint kappa_jj[m] (separating category m from m+1):
          //   lower bound = max(zz values in category m, kappa_jj[m-1])
          //   upper bound = min(zz values in category m+1, kappa_jj[m+1])
          for(size_t m=0; m<K_jj-1; ++m){
            double lo_kap = (m==0) ? R_NegInf : kappa_jj[m-1];
            double hi_kap = (m==K_jj-2) ? R_PosInf : kappa_jj[m+1];
            // tighten bounds using observed latent values
            for(size_t i=0; i<n; ++i){
              if(rr(i,jj)==0){
                if(x(i,jj)==cats_jj[m]   && zz(i,jj) > lo_kap) lo_kap = zz(i,jj);
                if(x(i,jj)==cats_jj[m+1] && zz(i,jj) < hi_kap) hi_kap = zz(i,jj);
              }
            }
            // Draw uniformly between lo_kap and hi_kap
            // Guard against degenerate interval (can occur if a category has no observed rows)
            if(lo_kap < hi_kap){
              kappa_jj[m] = R::runif(lo_kap, hi_kap);
            }
            // else: leave cutpoint unchanged
          }
          
          xy=zz.col(jj);
        }
        
        //now update beta
        arma::mat Q,R;
        qr_econ(Q,R,xx);
        arma::mat Rinv = inv(R);
        arma::mat Vb = Rinv*(Rinv.t());
        
        arma::vec xbhat = Rinv*(Q.t())*xy;
        arma::vec xyhat = xx*xbhat;
        
        //draw xbeta (multivariate normal)
        arma::vec Z(jj);
        for(size_t i=0; i<jj; ++i){
          Z[i] = gen.normal(0.0,1.0);
        }
        arma::vec mv_xbeta = xbhat + Rinv*sqrt(-1.0/(2.0*xsigs[ctrj_xb]))*Z; 
        for(size_t j=0; j<jj; ++j){
          xbetas(j,ctrj_xb) = mv_xbeta[j];
        }
        
        //draw sig_x
        if (xmodtyp[jj]==0){
          double rss=0.0;
          for(size_t i=0; i<n; i++){
            rss+=(xy[i]-xyhat[i])*(xy[i]-xyhat[i]);
          }
          double s2 = rss/(n-jj-1);
          xsigs[ctrj_xb] = (-1.0)*gen.gamma(0.5*(n-jj-1),1.0)/((n-jj-1)*s2);    //xsig = -1/(2*sig2_x)
        }
        // Binary and ordinal: latent scale is standardised (sigma=1); xsigs fixed at -0.5
        // (i.e. -1/(2*1^2)). No draw needed.
        
        
        ++ctrj_xb;
      }  
      
      // update x_est
      if (has_test_na) {
        size_t ctrj_xm=0;
        for(std::vector<size_t>::iterator itj=na_test_js.begin(); itj!=na_test_js.end(); ++itj){
          size_t jj = *itj;          //column of x_est
          size_t col_j = jj - na_reg_j1;   //slot for xsigs/xbetas
          
          if(ctrj_xm >= na_te_ptr.size()){
            break;
          }
          
          vec_dp ptr = na_te_ptr[ctrj_xm];
          vec_s ivec = na_te_i[ctrj_xm];
          
          arma::mat xxjj = x_est.submat(0,0,n_est-1,jj-1);
          //double xsig = sqrt(-1.0/(2.0*xsigs[jj-na_reg_j1]));  //xsigs has p_con-na_reg_j1 entries
          double xsig = sqrt(-1.0/(2.0*xsigs[col_j]));  //xsigs has p_con-na_reg_j1 entries
          arma::vec xb = xbetas.submat(0, jj-na_reg_j1, jj-1, jj-na_reg_j1);
          
          for(size_t ii=0; ii<ptr.size(); ++ii){
            size_t iobs = ivec[ii];
            arma::rowvec xxi = xxjj.row(iobs);
            if(xmodtyp[jj]==0){
              *(ptr[ii]) = arma::as_scalar(xxi*xb) + gen.normal(0.0, 1.0)*xsig; 
            } else if (xmodtyp[jj]==1){
              *(ptr[ii]) = (double)((arma::as_scalar(xxi*xb) + gen.normal(0.0, 1.0))>0);
            } else if (xmodtyp[jj]==2){
              // Ordinal test set: draw from predictive distribution.
              // Sample latent value from N(xbeta, 1), map to category via cutpoints.
              size_t ord_ctr = 0;
              for(size_t jjj=na_train_j1; jjj<jj; ++jjj){
                if(xmodtyp[jjj]==2) ++ord_ctr;
              }
              const std::vector<double>& kappa_jj = kappa_ord[ord_ctr];
              const std::vector<double>& cats_jj   = ord_cats[ord_ctr];
              size_t K_jj = cats_jj.size();
              double z_lat = arma::as_scalar(xxi*xb) + gen.normal(0.0, 1.0);
              size_t cat_idx = K_jj - 1;
              for(size_t m=0; m<K_jj-1; ++m){
                if(z_lat <= kappa_jj[m]){ cat_idx = m; break; }
              }
              *(ptr[ii]) = cats_jj[cat_idx];
            }
          }
          
          ++ctrj_xm;
        }
      }
      
      
    }
    
    
    ////////////
    //draw trees for m(x)
    ////////////
    
    //VBT: loop thru the trees
    for(int k=0;k<ntree_con;k++) {
      
      fit(t_con[k], xi_con, di_con, ftemp);  //doesn't need rr_con
      
      for(size_t i=0;i<n;i++) {
        if(ftemp[i] != ftemp[i]) { 
          Rcout << "control tree " << k <<" obs "<< i<<" "<< endl;
          Rcout << t_con[k] << endl;
          stop("nan in ftemp");
        }
        
        //get residuals
        allfit[i] = allfit[i]-mscale*ftemp[i];
        allfit_con[i] = allfit_con[i]-mscale*ftemp[i];
        r_con[i] = (y[i]-allfit[i])/mscale;
        
        //VBT: check that the update wound up ok
        if(r_con[i] != r_con[i]) {
          Rcout << y[i] << endl;
          Rcout << allfit[i] << endl;
          Rcout << mscale << endl;
          Rcout << r_con[i] << endl;
          stop("NaN in resid");
        }
      }
      
      //draw new tree via birth/death
      bd(t_con[k], xi_con, rri_con, di_con, pi_con, gen, na_mia, na_seqreg, min_node, check_node_missing);
      
      
      //draw node parameters
      drmu(t_con[k], xi_con, di_con, pi_con, gen);
      
      //get fits for the new tree
      fit(t_con[k], xi_con, di_con, ftemp);  //takes a vector
      for(size_t i=0; i<n; i++) {
        allfit[i] += mscale*ftemp[i];
        allfit_con[i] += mscale*ftemp[i];
      }
    }
    
    
    
    ////////////
    //draw trees for b(x)
    ////////////
    
    for(size_t i=0;i<ntrt;i++) {
      precs[i] = bscale1*bscale1/(sigma*sigma);
    }
    for(size_t i=ntrt; i<n; i++) {
      precs[i] = bscale0*bscale0/(sigma*sigma);
    }
    
    //Loop through b(x) trees
    for(int k=0; k<ntree_mod; k++) {
      
      //obtain residuals
      fit(t_mod[k], xi_mod, di_mod, ftemp);  //takes a vector
      
      for(size_t i=0; i<n; i++) {
        if(ftemp[i] != ftemp[i]) {
          Rcout << "moderator tree " << k <<" obs "<< i<<" "<< endl;
          Rcout << t_mod[k] << endl;
          stop("nan in ftemp");
        }
        double bscale = (i<ntrt) ? bscale1 : bscale0;
        allfit[i] = allfit[i]-bscale*ftemp[i];
        allfit_mod[i] = allfit_mod[i]-bscale*ftemp[i];
        r_mod[i] = (y[i]-allfit[i])/bscale;
      }
      
      //draw new trees for b(x)
      bdhet(t_mod[k], xi_mod, rri_mod, di_mod, precs, pi_mod, gen, na_mia, na_seqreg, min_node, check_node_missing);
      
      //draw new node parameters
      drmuhet(t_mod[k], xi_mod, di_mod, precs, pi_mod, gen);
      
      //update the fits
      fit(t_mod[k], xi_mod, di_mod, ftemp);  //takes a vector
      
      for(size_t i=0; i<ntrt; i++) {
        allfit[i] += bscale1*ftemp[i];
        allfit_mod[i] += bscale1*ftemp[i];
      }
      for(size_t i=ntrt; i<n; i++) {
        allfit[i] += bscale0*ftemp[i];
        allfit_mod[i] += bscale0*ftemp[i];
      }
    }
    
    
    ////////////
    //draw bscale parameters
    ////////////
    
    //update bscale0, bscale1, the PX parameters
    // bscale ~ N( [sum_i 1/w_i + bscale_prec]^{-1} \sum_i r_i/w_i, [sum_i 1/w_i + bscale_prec]^{-1})
    // where w_i = \sigma^2_y/b(x_i)^2
    //       r_i = (y_i-m(x_i))/b(x)
    // sums run over i:z_i=0 and i:z_i=1
    
    // [CATE GATE] this block CANNOT be left to run with empty loops: it divides
    // by allfit_mod, which is identically zero in prognostic-only mode, so it
    // would produce NaN rather than simply doing nothing. Both scales stay at 0
    // so that every (i<ntrt ? bscale1 : bscale0) elsewhere resolves consistently.
    if(!estimate_cate) {
      bscale0 = 0.0;
      bscale1 = 0.0;
    } else if(use_bscale) {
      
      double ww0 = 0.0, ww1 = 0.;
      double rw0 = 0.0, rw1 = 0.;
      double s2 = sigma*sigma;
      
      for(size_t i=0; i<n; ++i) {
        double bscale = (i<ntrt) ? bscale1 : bscale0;
        double w = s2*bscale*bscale/(allfit_mod[i]*allfit_mod[i]);
        
        if(w!=w) {
          Rcout << " w " << w << endl;
          stop("");
        }
        
        double randeff_contrib = randeff ? allfit_random[i] : 0.0;
        
        double r = (y[i] - allfit_con[i] - randeff_contrib)*bscale/allfit_mod[i];
        
        if(r!=r) {
          Rcout << "bscale " << i << " r " << r << " mscale " <<mscale<< " b*z " << allfit_mod[i]*z_[i] << " bscale " << bscale0 << " " <<bscale1 << endl;
          stop("");
        }
        if(i<ntrt) {
          ww1 += 1/w;
          rw1 += r/w;
        } else {
          ww0 += 1/w;
          rw0 += r/w;
        }
      }
      
      double bscale1_old = bscale1;
      double bscale_fc_var = 1/(ww1 + bscale_prec);
      bscale1 = bscale_fc_var*rw1 + gen.normal(0., 1.)*sqrt(bscale_fc_var);
      
      double bscale0_old = bscale0;
      bscale_fc_var = 1/(ww0 + bscale_prec);
      bscale0 = bscale_fc_var*rw0 + gen.normal(0., 1.)*sqrt(bscale_fc_var);
      
      for(size_t i=0; i<ntrt; ++i) {
        allfit_mod[i] = allfit_mod[i]*bscale1/bscale1_old;
      }
      for(size_t i=ntrt; i<n; ++i) {
        allfit_mod[i] = allfit_mod[i]*bscale0/bscale0_old;
      }
      
      
      // update delta_mod
      // if I'm not using half-normal, I use half-chi2
      if(!b_half_normal) {
        double ssq = 0.0;
        tree::npv bnv;
        typedef tree::npv::size_type bvsz;
        double endnode_count = 0.0;
        
        for(int k=0; k<ntree_mod; k++) {
          bnv.clear();
          t_mod[k].getbots(bnv);
          bvsz nb = bnv.size();
          for(bvsz ii = 0; ii<nb; ++ii) {
            double mm = bnv[ii]->getm(); //node parameter
            ssq += mm*mm/(pi_mod.tau*pi_mod.tau);
            endnode_count += 1.0;
          }
        }
        delta_mod = gen.gamma(0.5*(1.0 + endnode_count), 1.0)/(0.5*(1 + ssq));
      }
      pi_mod.tau   = mod_sd/(sqrt(delta_mod)*sqrt((double) ntree_mod));
      
    } else {
      bscale0 = -0.5;
      bscale1 =  0.5;
    }
    if(!estimate_cate) { bscale0 = 0.0; bscale1 = 0.0; }
    pi_mod.sigma = sigma;
    pi_mod.sigma = sigma;
    
    
    ////////////
    //draw mscale 
    ////////////
    
    //update mscale, the PX parameter
    // mscale ~ N( [sum_i 1/w_i + mscale_prec]^{-1} \sum_i r_i/w_i, [sum_i 1/w_i + mscale_prec]^{-1})
    // where w_i = \sigma^2_y/m(x_i)^2
    //       r_i = (y_i-b(x_i))/m(x_i)
    
    if(use_mscale) {
      
      double ww = 0.;
      double rw = 0.;
      double s2 = sigma*sigma;
      
      for(size_t i=0; i<n; ++i) {
        double w = s2*mscale*mscale/(allfit_con[i]*allfit_con[i]);
        if(w!=w) {
          Rcout << " w " << w << endl;
          stop("");
        }
        
        double randeff_contrib = randeff ? allfit_random[i] : 0.0;
        
        double r = (y[i] - allfit_mod[i]- randeff_contrib)*mscale/allfit_con[i];
        if(r!=r) {
          Rcout << "mscale " << i << " r " << r << " mscale " <<mscale<< " b*z_c " << allfit_mod[i] << endl;
          stop("");
        }
        ww += 1/w;
        rw += r/w;
      }
      
      double mscale_old = mscale;
      double mscale_fc_var = 1/(ww + mscale_prec);
      mscale = mscale_fc_var*rw + gen.normal(0.0, 1.0)*sqrt(mscale_fc_var);
      
      for(size_t i=0; i<n; ++i) {
        allfit_con[i] = allfit_con[i]*mscale/mscale_old;
      }
      
      // update delta_con
      
      double ssq = 0.0;
      tree::npv bnv;
      typedef tree::npv::size_type bvsz;
      double endnode_count = 0.0;
      
      for(int k=0; k<ntree_con; k++) {
        bnv.clear();
        t_con[k].getbots(bnv);
        bvsz nb = bnv.size();
        for(bvsz ii = 0; ii<nb; ++ii) {
          double mm = bnv[ii]->getm(); //node parameter
          ssq += mm*mm/(pi_con.tau*pi_con.tau);
          endnode_count += 1.0;
        }
      }
      
      delta_con = gen.gamma(0.5*(1.0 + endnode_count), 1.0)/(0.5*(1 + ssq));
      
      pi_con.tau   = con_sd/(sqrt(delta_con)*sqrt((double) ntree_con));
      
    } else {
      mscale = 1.0;
    }
    pi_con.sigma = sigma/fabs(mscale); //should be sigma/abs(mscale) for backfitting
    
    
    ////////////
    //Sync allfits after drawing allfit_con, allfit_mod, bscale, mscale
    ////////////
    //sync allfits after scale updates, if necessary. Could do smarter backfitting updates inline
    if(use_mscale || use_bscale) {
      for(size_t i=0; i<n; ++i) {
        double randeff_contrib = randeff ? allfit_random[i] : 0.0;
        allfit[i] = allfit_con[i] + allfit_mod[i] + randeff_contrib;
      }
    }
    
    
    ////////////
    //draw random effects 
    ////////////
    
    if(randeff) {
      //update random effects
      for(size_t i=0; i<n; ++i) {
        r(i) = y[i] - allfit_con[i] - allfit_mod[i];
        allfit[i] -= allfit_random[i];
      }
      
      Wtr = random_des.t()*r;
      
      arma::mat adj = diagmat(random_var_ix*eta);
      //    Rcout << adj << endl << endl;
      arma::mat Phi = adj*WtW*adj/(sigma*sigma) + Sigma_inv_random;
      arma::vec m = adj*Wtr/(sigma*sigma);
      //Rcout << m << Phi << endl << Sigma_inv_random;
      gamma = rmvnorm_post(m, Phi);
      
      //Rcout << "updated gamma";
      
      // Update px parameters eta
      
      arma::mat adj2 = diagmat(gamma)*random_var_ix;
      arma::mat Phi2 = adj2.t()*WtW*adj2/(sigma*sigma) + arma::eye(eta.size(), eta.size());
      arma::vec m2 = adj2.t()*Wtr/(sigma*sigma);
      //Rcout << m << Phi << endl << Sigma_inv_random;
      eta = rmvnorm_post(m2, Phi2);
      
      //Rcout << "updated eta";
      
      // Update variance parameters
      
      arma::vec ssqs   = random_var_ix.t()*(gamma % gamma);
      //Rcout << "A";
      arma::rowvec counts = sum(random_var_ix, 0);
      //Rcout << "B";
      for(size_t ii=0; ii<random_var_ix.n_cols; ++ii) {
        random_var(ii) = 1.0/gen.gamma(0.5*(random_var_df + counts(ii)), 1.0)*2.0/(random_var_df + ssqs(ii));
      }
      //Rcout << "updated vars" << endl;
      Sigma_inv_random = diagmat(1/(random_var_ix*random_var));
      
      allfit_random = random_des*diagmat(random_var_ix*eta)*gamma;
      
      //Rcout << "recom allfit vars" << endl;
      
      for(size_t i=0; i<n; ++i) {
        allfit[i] = allfit_con[i] + allfit_mod[i] + allfit_random(i); 
      }
    }
    
    
    ////////////
    //draw sigma: residual variance  
    ////////////
    
    double rss = 0.0;
    double restemp = 0.0;
    for(size_t i=0; i<n; i++) {
      restemp = y[i]-allfit[i];
      rss += restemp*restemp;
    }
    sigma = sqrt((nu*lambda + rss)/gen.chi_square(nu+n));
    pi_con.sigma = sigma/fabs(mscale);
    pi_mod.sigma = sigma;
    
    
    //if(mc>950){
    //  Rcpp::Rcout << mc << endl;
    //}
    
    
    ////////////
    //mTreeSHAP
    ////////////
    // Placed at the very end of the iteration, so the trees, the imputed x_hat,
    // the scales and sigma are all the current (post-update) state. Reads the
    // forests read-only and writes only to tree::coverage, which no part of the
    // sampler consults; the chain is therefore identical whether or not SHAP is
    // computed, and at any shap_thin.
    
    if( compute_shap && (mc>=burn) && (((mc-burn) % shap_thin)==0) && (shap_ctr<n_shap_max) )  {
      
      shap_iters(shap_ctr) = mc;
      const std::clock_t shap_t0 = std::clock();
      
      if(do_shap_mod) {
        // b(x) enters y as bscale*b(x); tau(x) = (bscale1-bscale0)*b(x), which is
        // exactly the scaling used for b_post below
        const double scale_mod = bscale1 - bscale0;
        const double base = mtreeshap_forest(t_mod, xi_mod, di_mod, di_mod,
                                             scale_mod, phi_mod, undef_mod);
        shap_base_mod(shap_ctr) = base;
        shap_sum_mod    += phi_mod;
        shap_sumabs_mod += arma::abs(phi_mod);
        if(shap_var) shap_sumsq_mod += phi_mod % phi_mod;
        if(shap_store_draws) shap_draws_mod.slice(shap_ctr) = phi_mod;
        if(shap_check) shap_err_mod(shap_ctr) =
          mtreeshap_max_efficiency_error(t_mod, xi_mod, di_mod, scale_mod,
                                         phi_mod, base, undef_mod);
        
        // x_est rows, TRAINING coverage, same baseline
        if(do_shap_mod_est) {
          arma::uvec ud;
          const double base_e = mtreeshap_forest(t_mod, xi_mod, di_mod, di_mod_est,
                                                 scale_mod, phi_mod_est, ud);
          shap_sum_mod_est    += phi_mod_est;
          shap_sumabs_mod_est += arma::abs(phi_mod_est);
          shap_undef_mod_est  += ud;
          if(shap_var) shap_sumsq_mod_est += phi_mod_est % phi_mod_est;
          if(shap_store_draws) shap_draws_mod_est.slice(shap_ctr) = phi_mod_est;
          if(shap_check) shap_err_mod_est(shap_ctr) =
            mtreeshap_max_efficiency_error(t_mod, xi_mod, di_mod_est, scale_mod,
                                           phi_mod_est, base_e, ud);
        }
      }
      
      if(do_shap_con) {
        // allfit_con = mscale*m(x), matching m_post below
        const double scale_con = mscale;
        const double base = mtreeshap_forest(t_con, xi_con, di_con, di_con,
                                             scale_con, phi_con, undef_con);
        shap_base_con(shap_ctr) = base;
        shap_sum_con    += phi_con;
        shap_sumabs_con += arma::abs(phi_con);
        if(shap_var) shap_sumsq_con += phi_con % phi_con;
        if(shap_store_draws) shap_draws_con.slice(shap_ctr) = phi_con;
        if(shap_check) shap_err_con(shap_ctr) =
          mtreeshap_max_efficiency_error(t_con, xi_con, di_con, scale_con,
                                         phi_con, base, undef_con);
        
        if(do_shap_con_est) {
          arma::uvec ud;
          const double base_e = mtreeshap_forest(t_con, xi_con, di_con, di_con_est,
                                                 scale_con, phi_con_est, ud);
          shap_sum_con_est    += phi_con_est;
          shap_sumabs_con_est += arma::abs(phi_con_est);
          shap_undef_con_est  += ud;
          if(shap_var) shap_sumsq_con_est += phi_con_est % phi_con_est;
          if(shap_store_draws) shap_draws_con_est.slice(shap_ctr) = phi_con_est;
          if(shap_check) shap_err_con_est(shap_ctr) =
            mtreeshap_max_efficiency_error(t_con, xi_con, di_con_est, scale_con,
                                           phi_con_est, base_e, ud);
        }
      }
      
      shap_seconds += (double)(std::clock() - shap_t0) / (double)CLOCKS_PER_SEC;
      shap_ctr += 1;
    }
    
    
    ////////////
    //save posterior draws  
    ////////////
    
    if( ((mc>=burn) & (mc % thin==0)) )  {
      
      msd_post(save_ctr) = fabs(mscale)*con_sd;
      bsd_post(save_ctr) = fabs(bscale1-bscale0)*mod_sd;
      
      gamma_post.row(save_ctr) = (diagmat(random_var_ix*eta)*gamma).t();
      random_var_post.row(save_ctr) = (sqrt( eta % eta % random_var)).t();
      
      sigma_post(save_ctr) = sigma;
      
      for(size_t i=0; i<n; i++) {
        m_post(save_ctr, i) = allfit_con[i];
        yhat_post(save_ctr, i) = allfit[i];
      }
      // recover tau(x): allfit_mod stores bscale*b(x), divide out bscale and multiply by (bscale1-bscale0)
      // [CATE GATE] another place that divides rather than just loops: with both
      // scales at 0 this is 0/0, so it has to be skipped rather than emptied.
      if(estimate_cate) {
        for(size_t i=0; i<n; i++) {
          double bscale = (i<ntrt) ? bscale1 : bscale0;
          b_post(save_ctr, i) = (bscale1-bscale0)*allfit_mod[i]/bscale;
        }
      }
      if (have_est) {
        for (size_t i = 0; i < n_est; ++i) {
          if(estimate_cate)
            b_est_post(save_ctr, i) = (bscale1 - bscale0) * fit_i(i, t_mod, xi_mod, di_mod_est);
          m_est_post(save_ctr, i) = mscale * fit_i(i, t_con, xi_con, di_con_est);
        }
      }
      if(bugtest){
        xm1_post.push_back(*(na_tr_ptr[0][0]));
        xm1_est_post.push_back(*(na_te_ptr[0][0]));
      }
      //xm1_post.push_back(*(na_tr_ptr[0][0]));
      //xm1_est_post.push_back(*(na_te_ptr[0][0]));
      
      // Accumulate coverage-weighted VI for tau trees. The loop is already a
      // no-op at ntree_mod = 0, so prognostic-only mode needs nothing here.
      for(int k=0; k<ntree_mod; k++) {
        tree::cnpv all_nodes;
        t_mod[k].getnodes(all_nodes);
        for(size_t ni=0; ni<all_nodes.size(); ++ni) {
          if(all_nodes[ni]->getl() != 0) { // internal nodes only
            size_t vv = all_nodes[ni]->getv();
            size_t na_type = all_nodes[ni]->getna();
            double cov = all_nodes[ni]->getn0();
            if(na_type > 3) na_type = 0; // safety clamp
            vi_mod(vv, na_type) += cov;
          }
        }
      }
      
      // Same counts, weighted by refreshed coverage instead of n0
      if(vi_coverage) {
        for(int k=0; k<ntree_mod; k++) {
          update_coverage(t_mod[k], xi_mod, di_mod);
          tree::npv cov_nodes;
          t_mod[k].getnodes(cov_nodes);
          for(size_t ni=0; ni<cov_nodes.size(); ++ni) {
            if(cov_nodes[ni]->getl() != 0) {
              size_t vv = cov_nodes[ni]->getv();
              size_t na_type = cov_nodes[ni]->getna();
              if(na_type > 3) na_type = 0;
              vi_mod_cov(vv, na_type) += cov_nodes[ni]->getcoverage();
            }
          }
        }
      }
      
      save_ctr += 1;
    }
  }
  
  int time2 = time(&tp);
  Rcout << "time for loop: " << time2 - time1 << endl;
  if(compute_shap)
    Rcout << "  of which mTreeSHAP: " << shap_seconds << " sec CPU over "
          << shap_ctr << " draws ("
          << (shap_ctr > 0 ? shap_seconds/shap_ctr : 0.0) << " sec/draw)" << endl;
  
  t_mod.clear(); 
  t_con.clear();
  delete[] allfit;
  delete[] allfit_mod;
  delete[] allfit_con;
  delete[] r_mod;
  delete[] r_con;
  delete[] ftemp;
  delete[] precs;
  
  treef.close();
  
  
  // ---- assemble the mTreeSHAP output ---------------------------------------
  // Rows are in the treatment-ordered (permuted) row order used inside C++ and
  // columns are in permuted covariate order; mbcf() un-permutes both, exactly as
  // it already does for vi_mod.
  SEXP shap_mod_out = R_NilValue;
  SEXP shap_con_out = R_NilValue;
  if(do_shap_mod) {
    shap_mod_out = List::create(
      _["sum"]      = shap_sum_mod,
      _["sum_abs"]  = shap_sumabs_mod,
      _["sum_sq"]   = shap_var         ? wrap(shap_sumsq_mod)  : R_NilValue,
      _["baseline"] = shap_base_mod,
      _["draws"]    = shap_store_draws ? wrap(shap_draws_mod)  : R_NilValue,
      _["max_err"]  = shap_check       ? wrap(shap_err_mod)    : R_NilValue);
  }
  if(do_shap_con) {
    shap_con_out = List::create(
      _["sum"]      = shap_sum_con,
      _["sum_abs"]  = shap_sumabs_con,
      _["sum_sq"]   = shap_var         ? wrap(shap_sumsq_con)  : R_NilValue,
      _["baseline"] = shap_base_con,
      _["draws"]    = shap_store_draws ? wrap(shap_draws_con)  : R_NilValue,
      _["max_err"]  = shap_check       ? wrap(shap_err_con)    : R_NilValue);
  }
  SEXP shap_mod_est_out = R_NilValue;
  SEXP shap_con_est_out = R_NilValue;
  if(do_shap_mod_est) {
    shap_mod_est_out = List::create(
      _["sum"]      = shap_sum_mod_est,
      _["sum_abs"]  = shap_sumabs_mod_est,
      _["sum_sq"]   = shap_var         ? wrap(shap_sumsq_mod_est) : R_NilValue,
      _["baseline"] = shap_base_mod,            // training coverage: same baseline
      _["draws"]    = shap_store_draws ? wrap(shap_draws_mod_est) : R_NilValue,
      _["max_err"]  = shap_check       ? wrap(shap_err_mod_est)   : R_NilValue,
      _["n_undef"]  = shap_undef_mod_est);
  }
  if(do_shap_con_est) {
    shap_con_est_out = List::create(
      _["sum"]      = shap_sum_con_est,
      _["sum_abs"]  = shap_sumabs_con_est,
      _["sum_sq"]   = shap_var         ? wrap(shap_sumsq_con_est) : R_NilValue,
      _["baseline"] = shap_base_con,
      _["draws"]    = shap_store_draws ? wrap(shap_draws_con_est) : R_NilValue,
      _["max_err"]  = shap_check       ? wrap(shap_err_con_est)   : R_NilValue,
      _["n_undef"]  = shap_undef_con_est);
  }
  SEXP shap_out = R_NilValue;
  if(compute_shap) {
    shap_out = List::create(
      _["n_draws"] = shap_ctr,
      _["thin"]    = shap_thin,
      _["iters"]   = shap_iters,
      _["seconds"] = shap_seconds,
      _["mod"]     = shap_mod_out,
      _["con"]     = shap_con_out,
      _["mod_est"] = shap_mod_est_out,
      _["con_est"] = shap_con_est_out);
  }
  
  if(bugtest){
    
    return(List::create(_["yhat_post"] = yhat_post, 
                        _["m_post"] = m_post,
                        _["b_post"] = b_post, 
                        _["b_est_post"] = b_est_post,
                        _["sigma"] = sigma_post, 
                        _["msd"] = msd_post, 
                        _["bsd"] = bsd_post,
                        _["gamma"] = gamma_post, 
                        _["random_var_post"] = random_var_post,
                        //_["xaccept"] = xaccept,
                        //_["min_obs_node"] = min_obs_node,
                        _["xm1_post"] = xm1_post,
                        _["xm1_post_est"] = xm1_est_post,
                        _["vi_mod"] = vi_mod,
                        _["vi_mod_cov"] = vi_coverage ? wrap(vi_mod_cov) : R_NilValue,
                        _["m_est_post"] = m_est_post,
                        _["shap"] = shap_out));
  }
  
  
  return(List::create(_["yhat_post"] = yhat_post, 
                      _["m_post"] = m_post,
                      _["b_post"] = b_post, 
                      _["b_est_post"] = b_est_post,
                      _["sigma"] = sigma_post, 
                      _["msd"] = msd_post, 
                      _["bsd"] = bsd_post,
                      _["gamma"] = gamma_post, 
                      _["random_var_post"] = random_var_post,
                      _["vi_mod"] = vi_mod,
                      _["vi_mod_cov"] = vi_coverage ? wrap(vi_mod_cov) : R_NilValue,
                      _["m_est_post"] = m_est_post,
                      _["shap"] = shap_out
  ));
}