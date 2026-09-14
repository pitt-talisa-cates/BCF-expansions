#ifndef GUARD_funs_h
#define GUARD_funs_h

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rng.h"

inline double logsumexp(const double &a, const double &b){
  return a < b ? b + log(1.0 + exp(a - b)) : a + log(1.0 + exp(b - a));
}

using std::cout;
using std::endl;

//pi and log(2*pi)
//#define PI 3.1415926535897931
#define LTPI 1.83787706640934536

//--------------------------------------------------
//send to left or right?
bool sendleft(double xv, double rrv, double xcut, size_t NA);
//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo& xi, vec_s& rri);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, vec_s& rri, pinfo& pi, tree::npv& goodbots);
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi, vec_s& rri, std::vector<size_t>& goodvars);

//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else a/(1+d)^b
double pgrow(tree::tree_p n, xinfo& xi, vec_s& rri, pinfo& pi);

//--------------------------------------------------
//get pointers to the trees that split on var
void allsplitv(std::vector<tree>& t, size_t var, tree::npv& fv);
//--------------------------------------------------
//Update sample size per node before proposing X
//void update_n0(tree& x, xinfo& xi, dinfo& di); 

//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<sinfo>& sv);
void allsuffhet(tree& x, xinfo& xi, dinfo& di, double* phi, tree::npv& bnv, std::vector<sinfo>& sv);

//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
void getsuff(tree& x, tree::tree_cp nx, size_t v, size_t c, size_t& na, xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr, bool check_node_missing);
//void getsuff_bot(tree& x, xinfo& xi, dinfo& di, std::vector<sinfo>& sbots);
void getsuffhet(tree& x, tree::tree_cp nx, size_t v, size_t c, size_t& na, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr, bool check_node_missing);

//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
void getsuff(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr);
void getsuffhet(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr);

//--------------------------------------------------
//log of the integreted likelihood
double lil(double n, double sy, double sy2, double sigma, double tau);
//lilhet drops constants that cancel in the mh ratio, lil/lilprec don't (legacy)
double lilhet(double n, double sy, double sy2, double sigma, double tau);

//--------------------------------------------------
//fit
void fit(tree& t, xinfo& xi, dinfo& di, std::vector<double>& fv);
void fit(tree& t, xinfo& xi, dinfo& di, double* fv);
double fit_i(size_t i, std::vector<tree>& t, xinfo& xi, dinfo& di);
double fit_it(size_t i, tree::tree_p t, arma::rowvec xxi, dinfo& di, xinfo& xi);

//--------------------------------------------------
// draw all the bottom node mu's

void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
void drmuhet(tree& t, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen);


#endif
