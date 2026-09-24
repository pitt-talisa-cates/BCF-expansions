// tree.h:  tree class and io streams to read/write 
//          trees and cutpoints
//--------------------------------------------------
#ifndef GUARD_tree_h
#define GUARD_tree_h

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstddef>
#include <vector>

#include "info.h"
#include "rng.h"

/*
Class is a bit confusing because sometimes you
want to think about an instance as a node and
sometimes as the whole tree including the node
and all its children.
Fundamentally, we have a tree not a node.
*/

/*
three ways to access a node:
(i) node id: is the integer assigned by the node numbering system 
assuming they are all there
(ii) node ind: is the index into the array of 
node(tree) pointers returned by getnodes or getbots or getnogs
which means you go left to right across the bottom of the tree
(iii) by its pointer (should only "give out" const pointers)
*/

//info contained in a node, used by input operator
struct node_info {
  std::size_t id; //node id
  std::size_t v;  //variable
  std::size_t c;  //cut point
  std::size_t na;
  double m;       //mu
};

// COnsider changing xinfo to be a vector of vec_d, where vec_d is now a class with two members: c and na
// c would be cutpoint, na would be what to do with missing.
// instead of just having one covariate matrix, we'd have X and R
// Splits would all be determined as answers to the control statement:
//    if (x<c | r==na) [go left]
//    else [go right]
// c would take vallues going from min(x)-1 to max(x), so that the answer to x<c could result in TRUE for all,
// leaving it up to the answer to r==na to split the node (pattern missing)
// na could take values -1, 0 or 1. when na==-1, it means the split ignored missing and only split on the 
// imputed variable.


//xinfo xi, then xi[v][c] is the c^{th} cutpoint for variable v.
// left if x[v] < xi[v][c]
typedef std::vector<double> vec_d;
typedef std::vector<vec_d> xinfo;
typedef std::vector<size_t> vec_s;
typedef std::vector<vec_s> xinfo_s;
typedef std::vector<double*> vec_dp;
typedef std::vector<vec_dp> xinfop;

class tree {
public:
  //------------------------------
  //typedefs
  typedef tree* tree_p;
  typedef const tree* tree_cp;      //VBT: pointer to a const tree type; can mod pointer but not the tree it points to
  typedef std::vector<tree_p> npv;   //Node Pointer Vector
  typedef std::vector<tree_cp> cnpv; //const Node Pointer Vector
  typedef std::vector<tree> vec_t;
  
  //------------------------------
  //friends
  friend std::istream& operator>>(std::istream&, tree&);
  friend bool bd(tree& x, xinfo& xi, vec_s& rri, dinfo& di, pinfo& pi, RNG& gen);
  friend bool bdhet(tree& x, xinfo& xi, vec_s& rri, dinfo& di, double* phi, pinfo& pi, RNG& gen);
  friend void fit_xprop(size_t i, tree::tree_p t, arma::rowvec xxi_old,arma::rowvec xxi_new,
                        dinfo& di, xinfo& xi, double& fold, double& fnew, 
                        tree::npv& bot_ptrs, double& min_n0_xnew);
  friend void update_n0(tree& x, xinfo& xi, dinfo& di); 
  
  //------------------------------
  //tree constructors--see tree.cpp for the definition of these constructors
  tree();
  tree(const tree&);
  tree(double);
  //tree destructor
  ~tree() {tonull();}
  
  //------------------------------
  //operators
  tree& operator=(const tree&);
  
  //------------------------------
  //node access
  //you are freely allowed to change mu, v, c
  //set----------
  void setm(double mu) {this->mu=mu;}
  void setv(size_t v) {this->v = v;}
  void setc(size_t c) {this->c = c;}
  void setna(size_t na) {this->na = na;}   
  void setn0(double n0) {this->n0 = n0;}
  void setn0_xnew(double n0) {this->n0_xnew = n0;}
  void setcoverage(double coverage) {this->coverage = coverage;}  //SHAP: see shap.h
  //get----------
  double getm() const {return mu;} 
  size_t getv() const {return v;}
  size_t getc() const {return c;}
  size_t getna() const {return na;}
  double getn0() const {return n0;}  //return 
  double getn0_xnew() const {return n0_xnew;}
  double getcoverage() const {return coverage;}                  //SHAP: see shap.h
  
  tree_p getp() const {return p;}  //should this be tree_cp? 
  tree_p getl() const {return l;}
  tree_p getr() const {return r;}
  tree_p getptr() {return this;}
  
  //------------------------------
  //tree functions
  //stuff about the tree----------
  size_t treesize() const;                               //number of nodes in tree
  size_t nnogs() const;                                  //number of nog nodes (no grandchildren nodes)
  size_t nbots() const;                                  //number of bottom nodes

  //birth death using nid----------
  bool birth(size_t nid, size_t v, size_t c, size_t na, double ml, double mr, double n0l,double n0r);   //need to add na
  bool death(size_t nid, double mu, double n); 
  
  //vectors of node pointers----------
  void getbots(npv& bv);                                 //get bottom nodes
  void getbots(cnpv& bv) const;
  void getnogs(npv& nv);                                  //get nog nodes (no granchildren)
  void getnodes(npv& v);                                 //get vector of all nodes
  void getnodes(cnpv& v) const;                          //get all nodes
  
  //find node from x and region for var----------
  tree_cp bn(double** x, double** rr, xinfo& xi);                       //find bottom node for x
  tree_p bn2(double** x, double** rr, xinfo& xi);
  tree_p bnarm(arma::rowvec x, double** rr, xinfo& xi);                       //find bottom node for x
  void rg(size_t v, int* L, int* U, int* Una, int* Lna);                     //find region [L,U] for var v.
  void splitv(size_t var,bool& answer);
  
  //------------------------------
  //node functions
  size_t depth() const;      //depth of a node
  size_t nid() const;        //node id
  tree_p getptr(size_t nid); //get node pointer from node id, 0 if not there.
  bool isnog() const;
  void tonull();             //like a "clear", null tree has just one node

private:
  //------------------------------
  //parameter for node
  double mu; 
  //------------------------------
  //rule: left if sendleft(...)
  size_t v; 
  size_t c; 
  size_t na;   
  //node size (for checking validity of new x proposals)
  double n0;
  double n0_xnew;
  //------------------------------
  //SHAP node coverage: number of training rows routed through THIS node under
  //the current imputed x. Maintained ONLY by update_coverage() (see shap.cpp),
  //which is called from the SHAP block of the sampler. Nothing in the MCMC reads
  //it, so the chain is unaffected by whether or how often it is refreshed.
  //Deliberately kept separate from n0/n0_xnew, which ARE read by the sampler
  //(min-node checks in bd/bdhet, x-proposal validity in fit_xprop) and whose
  //internal-node values are only correct as of the last birth/death at that node.
  //NB "coverage" here is TreeSHAP node coverage, not interval coverage.
  double coverage;
  //------------------------------
  //tree structure
  tree_p p; //parent
  tree_p l; //left child
  tree_p r; //right child
  //------------------------------
  //utiity functions
  void cp(tree_p n,  tree_cp o); //copy tree o to n
  
};
std::istream& operator>>(std::istream&, tree&);
std::ostream& operator<<(std::ostream&, const tree&);
std::istream& operator>>(std::istream&, xinfo&);
std::ostream& operator<<(std::ostream&, const xinfo&);


#endif