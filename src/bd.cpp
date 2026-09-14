#include <iostream>

#include "info.h"
#include "tree.h"
#include "bd.h"
#include "funs.h"

using std::cout;
using std::endl;

/*
notation: (as in old code): going from state x to state y (eg, incoming tree is x).

note: rather than have x and making a tree y
we just figure out what we need from x, the drawn bottom node,the drawn (v,c).
note sure what the right thing to do is.
Could make y (using a birth) and figure stuff out from y.
That is how the old code works.
*/

bool bd(tree& x, 
        xinfo& xi, 
        vec_s& rri,
        dinfo& di, 
        pinfo& pi, 
        RNG& gen,
        size_t na_mia,
        size_t na_seqreg,
        double min_node,
        bool check_node_missing)
{ 
  tree::npv goodbots;  //nodes we could birth at (split on)
  
  //VBT: for each bottom node in x, return probability of birth
  double PBx = getpb(x, xi, rri, pi, goodbots); //prob of a birth at x
  
  if(gen.uniform() < PBx) { //do birth or death
    
    //--------------------------------------------------
    //draw proposal
    
    //draw bottom node, choose node index ni from list in goodbots 
    size_t ni = floor(gen.uniform()*goodbots.size()); 
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,rri,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    
    //draw c and na
    int L,U,Lna,Una;
    size_t c,na;
    L=0;
    U=xi[v].size()-1;
    Lna=0;
    Una=rri[v];
    nx->rg(v,&L,&U,&Lna,&Una);
    if(!na_mia){                //(!na_mia && na_seqreg) || (!na_mia && !na_seqreg)
      na=0; 
      c = L + floor(gen.uniform()*(U-L+1));
    } else {                    //(na_mia && na_seqreg) || (na_mia && !na_seqreg)
      if(U>=L && Una==Lna){     //can split on x, but not rr
        na=0;            
        c = L + floor(gen.uniform()*(U-L+1));
      } else if (U>=L && Una>Lna){
        if(gen.uniform()<pi.pr_pm){     //PM_prob of pattern mixture split
          na = 3;
          c = 0;
        } else if (!na_seqreg){
          na = floor(gen.uniform() + 1.5);
          c = L + floor(gen.uniform()*(U-L+1));
        } else {
          if(gen.uniform()<0.5){       //0.25*(1-PM_prob) probabiliyt of splitting via MIA1 ot MIA2
            na = floor(gen.uniform() + 1.5);
            c = L + floor(gen.uniform()*(U-L+1));
          } else {                     
            na = 0;
            c = L + floor(gen.uniform()*(U-L+1));
          }
        }
      } else {  //PM split is the only option available
        na = 3;
        c =0;
      }
    }

    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double Pbotx = 1.0/goodbots.size();               //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = pi.alpha/pow(1.0 + dnx,pi.beta);    //prior prob of growing at nx
    
    double PGly, PGry;                                 //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) {                            //know there are variables we could split l and r on
      PGly = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);      //depth of new nodes would be one more
      PGry = PGly;
    } else if (Una==Lna) {                //only had v, couldn't split on na; if it is exhausted at either child need PG=0
      if((int)(c-1)<L) {             //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
      if(U < (int)(c+1)) {  //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
    } else {                           //only had v, could split on na
      if((int)(c-1)<L && (((int)na==1) || ((int)na==3)) ) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
      if(U < (int)(c+1) && (((int)na==0) || ((int)na==3)) ) {  //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
    }
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - pi.pb;
      }
    }
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl,sr; //sl for left from nx and sr for right from nx (using rule (v,c))

    getsuff(x, nx, v, c, na, xi, di, sl, sr, check_node_missing);

    //--------------------------------------------------
    //compute alpha
    
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    if((sl.n>=min_node) && (sr.n>=min_node)) {            //cludge?
      lill = lil(sl.n,sl.sy,sl.sy2,pi.sigma,pi.tau);
      lilr = lil(sr.n,sr.sy,sr.sy2,pi.sigma,pi.tau);
      lilt = lil(sl.n+sr.n,sl.sy+sr.sy,sl.sy2+sr.sy2,pi.sigma,pi.tau);
      
      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt);
      alpha = std::min(1.0,alpha2);
    } else {
      alpha=0.0;
    }
    

    //--------------------------------------------------
    //finally ready to try metrop
    double a,b,s2,yb;
    double mul,mur; //means for new bottom nodes, left and right
    if(gen.uniform() < alpha) {
      //draw mul, mean for left node
      a= 1.0/(pi.tau*pi.tau); //a = 1/tau^2
      s2 = pi.sigma*pi.sigma; // sigma^2
      //left mean
      yb = sl.sy/sl.n;
      b = sl.n/s2; // b=n/sigma^2
      mul = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //draw mul, mean for left node
      yb = sr.sy/sr.n;
      b = sr.n/s2; // b=n/sigma^2
      mur = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //do birth
      if(sl.n<min_node || sr.n<min_node){
        Rcpp::Rcout << "TRIP UP;; sln: " << sl.n <<"; srn: " << sr.n << endl;
      }
      x.birth(nx->nid(),v,c,na,mul,mur,sl.n,sr.n);

      return true;
      
    } else {
      //Rcpp::Rcout <<"NM,";
      return false;
    }
  } else {
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size()); 
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
    tree::tree_cp nxp = nx->getp();
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = pi.alpha/pow(1.0+dny,pi.beta);
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,rri,pi);
    double PGrx = pgrow(nx->getr(),xi,rri,pi);
    
    double PBy;  //prob of birth move at y
    if(nxp==0) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi,rri)) --ngood; //if can split at left child, lose this one 
    if(cansplit(nx->getr(),xi,rri)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl,sr; //sl for left from nx and sr for right from nx (using sendleft rule)

    getsuff(x, nx->getl(), nx->getr(), xi, di, sl, sr);

    //--------------------------------------------------
    //compute alpha
    
    double lill = lil(sl.n,sl.sy,sl.sy2,pi.sigma,pi.tau);
    double lilr = lil(sr.n,sr.sy,sr.sy2,pi.sigma,pi.tau);
    double lilt = lil(sl.n+sr.n,sl.sy+sr.sy,sl.sy2+sr.sy2,pi.sigma,pi.tau);
    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr);
    double alpha = std::min(1.0,alpha2);
    
    //--------------------------------------------------
    //finally ready to try metrop
    double a,b,s2,yb;
    double mu;
    double n;
    if(gen.uniform()<alpha) {
      //draw mu for nog (which will be bot)
      n = sl.n + sr.n;
      a= 1.0/(pi.tau*pi.tau); //a = 1/tau^2
      s2 = pi.sigma*pi.sigma; // sigma^2
      yb = (sl.sy+sr.sy)/n;
      b = n/s2; // b=n/sigma^2
      mu = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //do death
      x.death(nx->nid(),mu,n);

      return true;
    } else {
      //Rcpp::Rcout <<"NM,";
      return false;
    }
  }
}



//////////////

bool bdhet(tree& x, 
           xinfo& xi, 
           vec_s& rri,
           dinfo& di, 
           double* phi, 
           pinfo& pi, 
           RNG& gen,
           size_t na_mia,
           size_t na_seqreg,
           double min_node,
           bool check_node_missing)
{
  tree::npv goodbots;  //nodes we could birth at (split on)
  double PBx = getpb(x,xi,rri,pi,goodbots); //prob of a birth at x
  
  if(gen.uniform() < PBx) { //do birth or death
    
    //--------------------------------------------------
    //draw proposal
    
    //draw bottom node, choose node index ni from list in goodbots 
    size_t ni = floor(gen.uniform()*goodbots.size()); 
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,rri,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    
    //draw c and na
    int L,U,Lna,Una;
    size_t c, na;
    L=0;
    U=xi[v].size()-1;
    Lna=0;
    Una=rri[v];
    nx->rg(v,&L,&U,&Lna,&Una);
    if(!na_mia){                //(!na_mia && na_seqreg) || (!na_mia && !na_seqreg)
      na=0; 
      c = L + floor(gen.uniform()*(U-L+1));
    } else {                    //(na_mia && na_seqreg) || (na_mia && !na_seqreg)
      if(U>=L && Una==Lna){     //can split on x, but not rr
        na=0;            
        c = L + floor(gen.uniform()*(U-L+1));
      } else if (U>=L && Una>Lna){
        if(gen.uniform()<pi.pr_pm){     //PM_prob of pattern mixture split
          na = 3;
          c = 0;
        } else if (!na_seqreg){
          na = floor(gen.uniform() + 1.5);
          c = L + floor(gen.uniform()*(U-L+1));
        } else {
          if(gen.uniform()<0.5){       //0.25*(1-PM_prob) probability of splitting via MIA1 ot MIA2
            na = floor(gen.uniform() + 1.5);
            c = L + floor(gen.uniform()*(U-L+1));
          } else {                     
            na = 0;
            c = L + floor(gen.uniform()*(U-L+1));
          }
        }
      } else {  //PM split is the only option available
        na = 3;
        c =0;
      }
    }
    
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = pi.alpha/pow(1.0 + dnx,pi.beta); //prior prob of growing at nx
    
    double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) { //know there are variables we could split l and r on
      PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta); //depth of new nodes would be one more
      PGry = PGly;
    } else if (Una==Lna) {                //only had v, couldn't split on na; if it is exhausted at either child need PG=0
      if((int)(c-1)<L) {             //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
      if(U < (int)(c+1)) {  //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
    } else {                           //only had v, could split on na
      if((int)(c-1)<L && (((int)na==1) || ((int)na==3)) ) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
      if(U < (int)(c+1) && (((int)na==0) || ((int)na==3)) ) {  //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0, pi.beta);
      }
    }
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - pi.pb;
      }
    }
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl,sr; //sl for left from nx and sr for right from nx (using rule (v,c))
    getsuffhet(x, nx, v, c, na, xi, di, phi, sl, sr, check_node_missing);
    
    //--------------------------------------------------
    //compute alpha
    
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    if((sl.n0>=min_node) && (sr.n0>=min_node)) { //do we actually want this? yep
      lill = lilhet(sl.n,sl.sy,sl.sy2,pi.sigma,pi.tau);
      lilr = lilhet(sr.n,sr.sy,sr.sy2,pi.sigma,pi.tau);
      lilt = lilhet(sl.n+sr.n,sl.sy+sr.sy,sl.sy2+sr.sy2,pi.sigma,pi.tau);
      
      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt);
      alpha = std::min(1.0,alpha2);
    } else {
      alpha=0.0;
    }
    
    //--------------------------------------------------
    //finally ready to try metrop
    double a,b,s2,yb;
    double mul,mur; //means for new bottom nodes, left and right
    
    //all wrong but whatever, they get resampled immediately
    if(gen.uniform() < alpha) {
      //draw mul, mean for left node
      a= 1.0/(pi.tau*pi.tau); //a = 1/tau^2
      s2 = pi.sigma*pi.sigma; // sigma^2
      //left mean
      yb = sl.sy/sl.n;
      b = sl.n/s2; // b=n/sigma^2
      mul = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //draw mul, mean for left node
      yb = sr.sy/sr.n;
      b = sr.n/s2; // b=n/sigma^2
      mur = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //do birth
      x.birth(nx->nid(),v,c,na,mul,mur,sl.n0,sr.n0);
      return true;
    } else {
      return false;
    }
  } else {
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size()); 
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
    tree::tree_cp nxp = nx->getp();
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = pi.alpha/pow(1.0+dny,pi.beta);
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,rri,pi);
    double PGrx = pgrow(nx->getr(),xi,rri,pi);
    
    double PBy;  //prob of birth move at y
    //if(nx->ntype()=='t') { //is the nog node nx the top node
    if(nxp==0) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi,rri)) --ngood; //if can split at left child, lose this one 
    if(cansplit(nx->getr(),xi,rri)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl,sr; //sl for left from nx and sr for right from nx (using rule (v,c))

    getsuffhet(x, nx->getl(), nx->getr(), xi, di, phi, sl, sr);

    //--------------------------------------------------
    //compute alpha
    
    double lill = lilhet(sl.n,sl.sy,sl.sy2,pi.sigma,pi.tau);
    double lilr = lilhet(sr.n,sr.sy,sr.sy2,pi.sigma,pi.tau);
    double lilt = lilhet(sl.n+sr.n,sl.sy+sr.sy,sl.sy2+sr.sy2,pi.sigma,pi.tau);
    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr);
    double alpha = std::min(1.0,alpha2);
    
    //--------------------------------------------------
    //finally ready to try metrop
    double a,b,s2,yb;
    double mu;
    double n;
    
    //all wrong, but updated immediately so whatever
    if(gen.uniform()<alpha) {
      //draw mu for nog (which will be bot)
      n = sl.n + sr.n;
      a= 1.0/(pi.tau*pi.tau); //a = 1/tau^2
      s2 = pi.sigma*pi.sigma; // sigma^2
      yb = (sl.sy+sr.sy)/n;
      b = n/s2; // b=n/sigma^2
      mu = b*yb/(a+b) + gen.normal()/sqrt(a+b);
      //do death
      //cout << "death, mu=" << mu << endl;
      //x.deathp(nx,mu);
      x.death(nx->nid(), mu, sl.n0+sr.n0);

      return true;
    } else {

      return false;
    }
  }
}
