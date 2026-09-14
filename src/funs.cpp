#include <cmath>
#include "funs.h"
#include <map>
#ifdef MPIBART
#include "mpi.h"
#include "tree.h"
#endif

using Rcpp::Rcout;


// NEW: replace x[v] < xi[v][c] with something more general
// default is to send missing to the left
bool sendleft(double xv,
              double rrv,
              double xcut,
              size_t NA)
{
  if (NA==0){ 
    return xv < xcut;              //Impute: If observed or imputed x<c
  } else if (NA==1){
    return xv < xcut || rrv == 1;  //MIA1: if missing, or when x<c when nonmissing
  } else if (NA==2){
    return xv < xcut && rrv == 0;  //MIA2: if missing and x<c, otherwise send right
  } else {
    return rrv==1;                 //pattern mixture split
  }
}


//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, 
              xinfo& xi,
              vec_s& rri)
{
  int L,U,Lna,Una;
  bool v_found = false; //have you found a variable you can split on
  size_t v=0;
  while(!v_found && (v < xi.size())) {               //invar: splitvar not found, vars left
    L = 0; 
    U = xi[v].size()-1;
    Lna = 0;
    Una = rri[v];
    n->rg(v,&L,&U,&Lna,&Una);               //find the rule 1(L+1<X_v<U-1) at least partially defining node n
    if(U>=L || Una>Lna){                           //can split if U>=L, or 
      v_found=true;
    }             
    v++;
  }
  return v_found;
}


//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, 
             xinfo& xi, 
             vec_s& rri,
             pinfo& pi, 
             tree::npv& goodbots)
{
  double pb;                                //prob of birth to be returned
  tree::npv bnv;                            //node pointer vector
  t.getbots(bnv);                           //VBT: packs the vector bnv with the pointers to bottom nodes of t
  for(size_t i=0; i<bnv.size(); i++) {
    if(cansplit(bnv[i],xi,rri)) {
      goodbots.push_back(bnv[i]);           //vBT: add the splittable node to a growing list
    }                          
    if(goodbots.size()==0) {
      pb=0.0;                               //if I can't split 
    } else { 
      if(t.treesize()==1) {
        pb=1.0;          //is there just one node?  //VBT: translation: is this a root node (death imposs?)
      }
      else pb=pi.pb;                       //VBT: default value is 0.5
    }
  }
  return pb;
}


//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, 
                 xinfo& xi,  
                 vec_s& rri,
                 std::vector<size_t>& goodvars) //,
                 //bool na_mia)
{
  int L,U,Lna,Una;
  for(size_t v=0; v!=xi.size(); v++) {//try each variable
    L=0; 
    U = xi[v].size()-1;
    Lna=0;
    Una = rri[v];
    n->rg(v,&L,&U,&Lna,&Una);
    //bool pm=0;
    //if(na_mia && Lna==Una){
    //  pm = pmdesc<-(n,v)
    //}
    if(U>=L || Una>Lna) {
      goodvars.push_back(v);
    }
  }
}



//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, 
             xinfo& xi, 
             vec_s& rri,
             pinfo& pi)
{
  if(cansplit(n,xi,rri)) {
    return pi.alpha/pow(1.0+n->depth(),pi.beta);
  } else {
    return 0.0;
  }
}



//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(tree& x, 
             xinfo& xi, 
             dinfo& di, 
             tree::npv& bnv, 
             std::vector<sinfo>& sv)  //the vector of sinfo (sufficient stats) class objects
{
  tree::tree_cp tbn; 
  size_t ni;         
  double** xx;        
  double** rr;
  double y;          
  
  bnv.clear();            //clear the contents of bnv
  x.getbots(bnv);         //pack bnv with the bottom nodes of tree x
  
  typedef tree::npv::size_type bvsz;   //bvsz is a type suitable for holding the size of npv object
  bvsz nb = bnv.size();
  sv.resize(nb);                       //make sure I have room in sv for all notes' sufficient stats
  
  std::map<tree::tree_cp, size_t> bnmap;     //initialize a map object (a sorted key), where each node in the tree is associated with a unique index
  for(bvsz i=0; i!=bnv.size(); i++) {    //set the indices of the key
    bnmap[bnv[i]]=i;                     
  }
  
  for(size_t i=0; i<di.n; i++) {      //di.n is the number of observations in the datset
    xx = di.x + i*di.p;               //di.x is a pointer to the first element of our x matrix
    rr = di.rr + i*di.p;             
    y=di.y[i];
    
    tbn = x.bn(xx,rr,xi);               //run xx down tree x (need xi because treex only has the cutpoint index, not the actual value)
    ni = bnmap[tbn];                  //need this to find the corresponding node in sv vector
    
    ++(sv[ni].n);                     //pack sv with sufficient stats
    sv[ni].sy += y;
    sv[ni].sy2 += y*y;
  }
}

void allsuffhet(tree& x, 
                xinfo& xi, 
                dinfo& di, 
                double* phi,                //what is this?
                tree::npv& bnv, 
                std::vector<sinfo>& sv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double** xx;        //current x
  double** rr;
  double y;          //current y
  
  bnv.clear();
  x.getbots(bnv);
  
  typedef tree::npv::size_type bvsz;
  bvsz nb = bnv.size();
  sv.resize(nb);
  
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) {
    bnmap[bnv[i]]=i;       
  }
  
  for(size_t i=0; i<di.n; i++) {
    xx = di.x + i*di.p;
    rr = di.rr + i*di.p;
    y=di.y[i];
    
    tbn = x.bn(xx,rr,xi);
    ni = bnmap[tbn];

    sv[ni].n0 += 1;                                
    sv[ni].n += phi[i];
    sv[ni].sy += phi[i]*y;
    sv[ni].sy2 += phi[i]*y*y;
    
  }
}



//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
void getsuff(tree& x, 
             tree::tree_cp nx, 
             size_t v, 
             size_t c, 
             size_t& na,           // by reference so caller sees corrected na
             xinfo& xi, 
             dinfo& di, 
             sinfo& sl, 
             sinfo& sr,
             bool check_node_missing)
{
  double** xx;//current x
  double** rr;
  double y;  //current y
  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;

  if(check_node_missing && na > 0) {
    if(na == 3) {
      // first pass: check for missing only, break early on first missing found
      bool any_missing = false;
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          if(*(rr[v])) {
            any_missing = true;
            break;
          }
        }
      }
      if(!any_missing) na = 0;
      // second pass: accumulate under confirmed na (0 or 3)
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          y = di.y[i];
          if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)){
            sl.n++;
            sl.sy += y;
            sl.sy2 += y*y;
          } else {
            sr.n++;
            sr.sy += y;
            sr.sy2 += y*y;
          }
        }
      }
    } else {
      // na=1 or na=2: single pass, accumulate and check for missing simultaneously
      // if no missing found, correct na=0; sl/sr already correct since MIA==value when no missing
      bool any_missing = false;
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          if(*(rr[v])) any_missing = true;
          y = di.y[i];
          if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)){
            sl.n++;
            sl.sy += y;
            sl.sy2 += y*y;
          } else {
            sr.n++;
            sr.sy += y;
            sr.sy2 += y*y;
          }
        }
      }
      if(!any_missing) na = 0;
    }
  } else {
    // na=0 or check_node_missing=false: original loop unchanged
    for(size_t i=0;i<di.n;i++) {
      xx = di.x + i*di.p;   //pointer to the address of variable 1 for observation i
      rr = di.rr + i*di.p;
      if(nx==x.bn(xx,rr,xi)) { //does the bottom node = xx's bottom node
        y = di.y[i];
        if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)){
          sl.n++;
          sl.sy += y;
          sl.sy2 += y*y;
        } else {
          sr.n++;
          sr.sy += y;
          sr.sy2 += y*y;
        }
      }
    }
  }
}

/*
void getsuff_bot(tree& x, 
                 xinfo& xi, 
                 dinfo& di, 
                 std::vector<sinfo*>& sbots)
{
  double** xx;//current x
  double** rr;
  double y;  //current y
  tree::npv bots;
  x.getbots(bots);
  
  for(int k=0; k<bots.size(); ++k){
    for(size_t i=0; i<di.n; i++) {
      xx = di.x + i*di.p;   //pointer to the address of variable 1 for observation i
      rr = di.rr + i*di.p;
      if(bots[k]==x.bn(xx,rr,xi)) { //does xx fall in the k'th bottom node?
        y = di.y[i];
        (sbots[k]).n++;
        (sbots[k]).sy += y;
        (sbots[k]).sy2 += y*y;
      }
    }
  }
}
 */


//for het, n = sum_i phi_i, sumy = \sum \phi_iy_i, sumy^2 \sum \phi_iy_i^2
void getsuffhet(tree& x, 
                tree::tree_cp nx, 
                size_t v, 
                size_t c, 
                size_t& na,           // by reference so caller sees corrected na
                xinfo& xi, 
                dinfo& di, 
                double* phi, 
                sinfo& sl, 
                sinfo& sr,
                bool check_node_missing)
{
  double** xx;//current x
  double** rr;
  double y;  //current y
  sl.n=0; sl.sy=0.0; sl.sy2=0.0; sl.n0=0;
  sr.n=0; sr.sy=0.0; sr.sy2=0.0; sr.n0=0;

  if(check_node_missing && na > 0) {
    if(na == 3) {
      // first pass: check for missing only, break early on first missing found
      bool any_missing = false;
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          if(*(rr[v])) {
            any_missing = true;
            break;
          }
        }
      }
      if(!any_missing) na = 0;
      // second pass: accumulate under confirmed na (0 or 3)
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          y = di.y[i];
          if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)) {
            sl.n0 += 1;
            sl.n += phi[i];
            sl.sy += phi[i]*y;
            sl.sy2 += phi[i]*y*y;
          } else {
            sr.n0 += 1;
            sr.n += phi[i];
            sr.sy += phi[i]*y;
            sr.sy2 += phi[i]*y*y;
          }
        }
      }
    } else {
      // na=1 or na=2: single pass, accumulate and check for missing simultaneously
      // if no missing found, correct na=0; sl/sr already correct since MIA==value when no missing
      bool any_missing = false;
      for(size_t i=0;i<di.n;i++) {
        xx = di.x + i*di.p;
        rr = di.rr + i*di.p;
        if(nx==x.bn(xx,rr,xi)) {
          if(*(rr[v])) any_missing = true;
          y = di.y[i];
          if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)) {
            sl.n0 += 1;
            sl.n += phi[i];
            sl.sy += phi[i]*y;
            sl.sy2 += phi[i]*y*y;
          } else {
            sr.n0 += 1;
            sr.n += phi[i];
            sr.sy += phi[i]*y;
            sr.sy2 += phi[i]*y*y;
          }
        }
      }
      if(!any_missing) na = 0;
    }
  } else {
    // na=0 or check_node_missing=false: original loop unchanged
    for(size_t i=0;i<di.n;i++) {
      xx = di.x + i*di.p;
      rr = di.rr + i*di.p;
      if(nx==x.bn(xx,rr,xi)) { //does the bottom node = xx's bottom node
        y = di.y[i];
        if(sendleft(*(xx[v]),*(rr[v]),xi[v][c],na)) {
          sl.n0 += 1;
          sl.n += phi[i];
          sl.sy += phi[i]*y;
          sl.sy2 += phi[i]*y*y;
        } else {
          sr.n0 += 1;
          sr.n += phi[i];
          sr.sy += phi[i]*y;
          sr.sy2 += phi[i]*y*y;
        }
      }
    }
  }
}


//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
void getsuff(tree& x, 
             tree::tree_cp nl, 
             tree::tree_cp nr, 
             xinfo& xi, 
             dinfo& di, 
             sinfo& sl, 
             sinfo& sr)
{
  double** xx;//current x
  double** rr;
  double y;  //current y
  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    rr = di.rr + i*di.p;
    tree::tree_cp bn = x.bn(xx,rr,xi);
    if(bn==nl) {
      y = di.y[i];
      sl.n++;
      sl.sy += y;
      sl.sy2 += y*y;
    }
    if(bn==nr) {
      y = di.y[i];
      sr.n++;
      sr.sy += y;
      sr.sy2 += y*y;
    }
  }
}
void getsuffhet(tree& x, 
                tree::tree_cp nl, 
                tree::tree_cp nr, 
                xinfo& xi, 
                dinfo& di, 
                double* phi, 
                sinfo& sl, 
                sinfo& sr)
{
  double** xx;//current x
  double** rr;
  double y;  //current y
  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    rr = di.rr + i*di.p;
    tree::tree_cp bn = x.bn(xx,rr,xi);
    if(bn==nl) {
      y = di.y[i];
      sl.n += phi[i];
      sl.sy += phi[i]*y;
      sl.sy2 += phi[i]*y*y;
    }
    if(bn==nr) {
      y = di.y[i];
      sr.n += phi[i];
      sr.sy += phi[i]*y;
      sr.sy2 += phi[i]*y*y;
    }
  }
}



//--------------------------------------------------
//log of the integrated likelihood
double lil(double n, double sy, double sy2, double sigma, double tau)
{
  double yb,yb2,S,sig2,d;
  double sum, rv;
  
  yb = sy/n;
  yb2 = yb*yb;
  S = sy2 - (n*yb2);
  sig2 = sigma*sigma;
  d = n*tau*tau + sig2;
  sum = S/sig2 + (n*yb2)/d;
  rv = -(n*LTPI/2.0) - (n-1)*log(sigma) -log(d)/2.0;
  rv = rv -sum/2.0;
  return rv;
}

double lilhet(double n, double sy, double sy2, double sigma, double tau)
{
  double d = 1/(tau*tau) + n;// n is \sum phi_i for het
  
  double out = -log(tau) - 0.5*log(d);
  out += 0.5*sy*sy/d - 0.5*sy2;
  return out;
}


//--------------------------------------------------
//fit
void fit(tree& t, 
         xinfo& xi, 
         dinfo& di, 
         std::vector<double>& fv)
{
  double** xx;
  double** rr;
  tree::tree_cp bn;
  fv.resize(di.n);
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;   //the value of di.ctptr is an address
    rr = di.rr + i*di.p;
    bn = t.bn(xx,rr,xi);
    fv[i] = bn->getm();
  }
}

void fit(tree& t, 
         xinfo& xi, 
         dinfo& di, 
         double* fv)
{
  double** xx;
  double** rr;
  tree::tree_cp bn;               // returns //VBT: tree_cp is a const tree* declaration (unmodifiable tree structure)
  for(size_t i=0; i<di.n; i++) {
    xx = di.x + i*di.p;
    rr = di.rr + i*di.p;
    bn = t.bn(xx,rr,xi);             //bn is all the info about the bottom nodes (tree structs without child nodes) for each person 
    fv[i] = bn->getm();           //a->b() is like saying a.b(), except you use -> when a is a pointer. 
  }
}

double fit_i(size_t i, 
             std::vector<tree>& t, 
             xinfo& xi, 
             dinfo& di)
{
  double** xx;
  double** rr;
  double fv = 0.0;
  tree::tree_cp bn;
  xx = di.x + i*di.p;      
  rr = di.rr + i*di.p;
  for (size_t j=0; j<t.size(); ++j) {
    bn = t[j].bn(xx,rr,xi);
    fv += bn->getm();
  }
  return fv;
}

double fit_it(size_t i, 
              tree::tree_p t, 
              arma::rowvec xxi,
              dinfo& di,
              xinfo& xi)
{
  double fv = 0.0;
  double** rri;
  rri = di.rr + i*di.p;
  tree::tree_cp bot;
  bot = t->bnarm(xxi,rri,xi);
  fv = bot->getm();
  return fv;
}

//new version: 
void fit_xprop(size_t i, 
              tree::tree_p t, 
              arma::rowvec xxi_old,
              arma::rowvec xxi_new,
              dinfo& di,
              xinfo& xi,
              double& fold,
              double& fnew,
              tree::npv& bot_ptrs,
              double& min_n0_xnew)
{
  double** rri;
  rri = di.rr + i*di.p;
  tree::tree_p bot_old, bot_new;
  bot_old = t->bnarm(xxi_old,rri,xi);
  bot_new = t->bnarm(xxi_new,rri,xi);
  fold = bot_old->mu;
  fnew = bot_new->mu;
  if (bot_old!=bot_new){
    bot_old->n0_xnew --;
    bot_new->n0_xnew ++;
    bot_ptrs.push_back(bot_old);  //so that I can set n0=n0_xnew if x_new is accepted, or n0_xnew=n0 if not
    bot_ptrs.push_back(bot_new);
    if(bot_old->n0_xnew < min_n0_xnew){
      min_n0_xnew = bot_old->n0_xnew;
    }
  }
}

//--------------------------------------------------
//find trees splitting on v
void allsplitv(tree::vec_t& t, 
               size_t var,
               tree::npv& fv)
{
  fv.resize(0);
  for(tree::vec_t::size_type j=0; j<t.size(); ++j){
    bool a=false;
    t[j].splitv(var,a);
    if(a){
      fv.push_back(t[j].getptr());
    }
  }
}


void update_n0(tree& x, 
               xinfo& xi, 
               dinfo& di)  
{
  tree::tree_p tbn; 
  double** xx;        
  double** rr;

  tree::npv bnv;
  x.getbots(bnv);         
  for(tree::npv::size_type kk=0; kk<bnv.size(); ++kk){
    (bnv[kk]->n0)=0;
    (bnv[kk]->n0_xnew)=0;
  }
  
  for(size_t i=0; i<di.n; i++) {      
    xx = di.x + i*di.p;
    rr = di.rr + i*di.p;
    tbn = x.bn2(xx,rr,xi);
    (tbn->n0)++;                    //= (tbn->getn0()+1);
    (tbn->n0_xnew)++;
  }
}



//--------------------------------------------------
// draw all the bottom node mu's

void drmu(tree& t, 
          xinfo& xi, 
          dinfo& di, 
          pinfo& pi, 
          RNG& gen)
{
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuff(t, xi, di, bnv, sv);
  
  double a = 1.0/(pi.tau * pi.tau);
  double sig2 = pi.sigma * pi.sigma;
  double b, ybar;
  
  for(size_t i=0; i<bnv.size();i++) {
    b = sv[i].n/sig2;
    ybar = sv[i].sy/sv[i].n;  
    bnv[i]->setm(b*ybar/(a+b) + gen.normal()/sqrt(a+b));
    if(bnv[i]->getm() != bnv[i]->getm()) {
      //for(size_t ii=0; ii<di.n; ++ii) {
      //  Rcout << **(di.x + ii*di.p) <<" ";        
      //}
      for(size_t ii=0; ii<bnv.size();ii++) {
        Rcout << endl<<" a "<< a<<" b "<<b<<" svi[n] "<<sv[ii].n<< "sigma " << pi.sigma << " i "<<ii;
      }
      Rcout << endl << t;
      
      Rcpp::stop("drmu failed");
    }
  }
}

void drmuhet(tree& t, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen)
{
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuffhet(t,xi,di,phi,bnv,sv);
  
  double a = 1.0/(pi.tau * pi.tau);
  //double sig2 = pi.sigma * pi.sigma;
  //double b,ybar;
  
  for(size_t i=0; i<bnv.size(); i++) {
    double fcvar = 1.0/(a+sv[i].n);
    double fcmean = sv[i].sy*fcvar;
    
    bnv[i]->setm(fcmean + gen.normal()*sqrt(fcvar));
    if(bnv[i]->getm() != bnv[i]->getm()) {
      for(size_t ii=0; ii<di.n; ++ii) {
        Rcout << **(di.x + ii*di.p) <<" "; 
      }
      Rcout << endl<<" a "<< a<<" svi[n] "<<sv[i].n<<" i "<<i;
      Rcout << endl<<" svi[n0] " << sv[i].n0 << endl;
      Rcout << endl << t;
      Rcpp::stop("drmuhet failed");
    }
  }
}


