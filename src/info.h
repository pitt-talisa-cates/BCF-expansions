#ifndef GUARD_info_h
#define GUARD_info_h

#include<vector>

//data
class dinfo {
public:
  dinfo() {p=0;
           n=0;
           x=0;
           rr=0;
           y=0;}
  size_t p;  
  size_t n;  
  double** x;   // address of r1 c1 of pointers to x transpose
  double** rr;  // address of r1 c1 of pointers to rr transpose
  double* y;   // residuals to use as outcome
};

//prior and mcmc
class pinfo
{
public:
  //default constructor
  pinfo() {
    pbd=1.0;
    pb=.5;
    pr_pm=0.0;
    alpha=.95;
    beta=.5;
    tau=1.0;
    sigma=1.0;
    minperbot=5;
    taccept=0;
    tavgd=0;
    tmaxd=0;
  }
  //mcmc info
  double pbd; //prob of birth/death
  double pb;  //prob of birth
  double pr_pm; //prob of pattern mixture split
  //prior info
  double alpha;
  double beta;
  //tau: variance of variance of node parameters
  double tau;
  //sigma: variance of node parameters
  double sigma;
  //tree min obs
  size_t minperbot;
  //mcmc stuffs
  std::vector< std::vector<double> > cvpm;  // Change of variable proposal probability matrix
  unsigned int taccept; //acceptance count of tree proposals
  unsigned int tproposal; //number of tree proposals
  unsigned int tavgd; //average tree depth
  unsigned int tmaxd; //maximum tree depth
};



//sufficient statistics for 1 node
class sinfo
{
public:
  // Default constructor (initializes to these values)
  sinfo() {
    n0=0.0;
    n=0;
    sy=0.0;
    sy2=0.0;
  }
  double n0; //unweighted sample counts
  double n;
  double sy;
  double sy2;
};

#endif