// tree.cpp:  tree class and io streams to read/write
//            trees and cutpoints
//--------------------------------------------------
#include <string>
#include <vector>
#include <map>
#include "tree.h"
#include "funs.h"

using std::string;
using std::cout;
using std::endl;

//--------------------------------------------------
// Constructors
tree::tree(): mu(0.0),v(0),c(0), na(0), n0(0.0), n0_xnew(0.0),  p(0),l(0),r(0) {}
tree::tree(double m): mu(m),v(0),c(0),na(0), n0(0.0), n0_xnew(0.0), p(0),l(0),r(0) {}
tree::tree(const tree& n): mu(0.0),v(0),c(0),na(0), n0(0.0), n0_xnew(0.0),p(0),l(0),r(0) { cp(this,&n); }

//--------------------------------------------------
// Operators
tree& tree::operator=(const tree& rhs)
{
  if(&rhs != this) {       //if tree1 != tree2
    tonull();              //kill left hand side (this)
    cp(this,&rhs);         //copy right hand side to left hand side
  }
  return *this;
}

//--------------------
// Output operator for a tree t
std::ostream& operator<<(std::ostream& os, const tree& t)
{
  tree::cnpv nds,bnds;
  t.getnodes(nds);
  t.getbots(bnds);
  
  os << "All nodes: " << nds.size() << endl;
  for(size_t i=0;i<nds.size();i++) {
    os << nds[i]->nid() << " ";
    os << nds[i]->getv() << " ";
    os << nds[i]->getc() << " ";
    os << nds[i]->getna() << " ";
    os << nds[i]->getn0() << " ";
    os << nds[i]->getn0_xnew() << " ";
    os << nds[i]->getm() << endl;
  }
  os << "Bots: " << bnds.size() << endl;
  for(size_t i=0;i<bnds.size();i++) {
    os << bnds[i]->nid() << " ";
    os << bnds[i]->getv() << " ";
    os << bnds[i]->getc() << " ";
    os << bnds[i]->getna() << " ";
    os << bnds[i]->getn0() << " ";
    os << bnds[i]->getn0_xnew() << " ";
    os << bnds[i]->getm() << endl;
  }
  return os;
}

//--------------------
// Input operator for a tree t
std::istream& operator>>(std::istream& is, tree& t)
{
  size_t tid,pid;                     //tid: id of current node, pid: parent's id
  std::map<size_t,tree::tree_p> pts;  //pointers to nodes indexed by node id
  size_t nn;                          //number of nodes
  
  t.tonull(); // obliterate old tree (if there)
  
  //read number of nodes----------
  is >> nn;
  if(!is) {
    Rcpp::Rcout << ">> error: unable to read number of nodes" << endl;
    return is;
  }
  
  //read in vector of node information----------
  std::vector<node_info> nv(nn);
  for(size_t i=0;i!=nn;i++) {
    is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].na >> nv[i].m;
    if(!is) {
      Rcpp::Rcout << ">> error: unable to read node info, on node  " << i+1 << endl;
      return is;
    }
  }
  
  //first node has to be the top one
  pts[1] = &t; //careful! this is not the first pts, it is pointer of id 1.
  t.setv(nv[0].v); 
  t.setc(nv[0].c); 
  t.setna(nv[0].na);
  t.setm(nv[0].m);
  t.p=0;
  
  //now loop through the rest of the nodes knowing parent is already there.
  for(size_t i=1;i!=nv.size();i++) {
    tree::tree_p np = new tree;
    np->v = nv[i].v; 
    np->c=nv[i].c; 
    np->na=nv[i].na; 
    np->mu=nv[i].m;
    tid = nv[i].id;
    pts[tid] = np;
    pid = tid/2;
    // set pointers
    if(tid % 2 == 0) { //left child has even id
      pts[pid]->l = np;
    } else {
      pts[pid]->r = np;
    }
    np->p = pts[pid];
  }
  return is;
}

//--------------------
// Output operator for cutpoint info xi
std::ostream& operator<<(std::ostream& os, const xinfo& xi)
{
  os << xi.size() << endl;
  for(size_t i=0;i<xi.size();i++) {
    os << xi[i].size() << endl;
    for(size_t j=0;j<xi[i].size();j++)
      os << xi[i][j]<<endl;
    os << endl;
  }
  
  return os;
}

//--------------------
// Input operator for cutpoint info xi
std::istream& operator>>(std::istream& is, xinfo& xi)
{
  size_t xin;
  size_t vecdn;
  
  xi.resize(0); // reset old xinfo (if there)
  
  is >> xin;
  if(!is) {
    //cout << ">> error: unable to read size of xinfo" << endl;
    return is;
  }
  
  std::vector<double> vec_d;
  double vecdelem;
  for(size_t i=0;i<xin;i++) {
    is >> vecdn;
    for(size_t j=0;j<vecdn;j++) {
      is >> vecdelem;
      vec_d.push_back(vecdelem);
    }
    xi.push_back(vec_d);
    vec_d.resize(0);
  }
  
  return is;
}



//--------------------------------------------------
// Public functions


// return true if I split at least once on var
// To calculate liklihood of y, I'll need to run proposals of new x[v] down the trees,
// but the fits only change for the trees that split on v
void tree::splitv(size_t var, bool& answer)
{
  if(l){                  //if I have children
    if(this->v == var){
      if(this->na == 3){
        l->splitv(var,answer);
        r->splitv(var,answer);
      } else {
        answer=true;
      }
    } else {
      l->splitv(var,answer);
      r->splitv(var,answer);
    }
  }
}


//--------------------
// Get all nodes
// Recursion down the tree
// VBT: packs a vector of tree_p objects 
//      with all of the addresses of all of the nodes of teh current tree
void tree::getnodes(npv& v)
{
  v.push_back(this);
  if(l) {
    l->getnodes(v);
    r->getnodes(v);
  }
}
void tree::getnodes(cnpv& v)  const
{
  v.push_back(this);
  if(l) {
    l->getnodes(v);
    r->getnodes(v);
  }
}


//--------------------
// Find bottom node pointer given x
// returns an object of type tree_cp
tree::tree_cp tree::bn(double** x, 
                       double** rr,
                       xinfo& xi)
{ //members l, r, v, c, and na are all from this tree
  if(l==0) return this; 
  if(sendleft(*(x[this->v]),*(rr[this->v]),xi[this->v][this->c],this->na)){
    return l->bn(x,rr,xi);      
  } else {
    return r->bn(x,rr,xi);
  }
}

tree::tree_p tree::bn2(double** x, 
                       double** rr,
                       xinfo& xi)
{ //members l, r, v, c, and na are all from this tree
  if(l==0) return this; 
  if(sendleft(*(x[this->v]),*(rr[this->v]),xi[this->v][this->c],this->na)){
    return l->bn2(x,rr,xi);      
  } else {
    return r->bn2(x,rr,xi);
  }
}

tree::tree_p tree::bnarm(arma::rowvec x, 
                        double** rr,
                        xinfo& xi)
{ //members l, r, v, c, and na are all from this tree
  if(l==0) return this; 
  if(sendleft(x[1+(this->v)], *(rr[this->v]), xi[this->v][this->c],this->na)){
    return l->bnarm(x,rr,xi);      
  } else {
    return r->bnarm(x,rr,xi);
  }
}



//--------------------
// Find region for a given variable
void tree::rg(size_t v, 
              int* L, 
              int* U,
              int* Lna,
              int* Una)
{
  if(p==0)  { //no parent
    return;
  }
  if(p->v == v) { //does my parent use v?
    if(this == p->l) {                      //am I left or right child
      if ((int)(p->na) < 3){
        if ((int)(p->na) < 2){
          if ((int)(p->c) <= (*U)){
            *U = (p->c)-1;                      
          }
        } else {
          *Una=0; //I only have people with nonmissing
          if ((int)(p->c) <= (*U)){
            *U = (p->c)-1;                      
          }
        }
      } else {
        *Lna=1; //I only have people with missing
      }
    } else {
      if ((int)(p->na) < 3){
        if((int)(p->na) != 1){
          if((int)(p->c) >= *L){
            *L = (p->c)+1;
          }
        } else {
          *Una=0; //I only have people with nonmissing
          if((int)(p->c) >= *L){
            *L = (p->c)+1;
          }
        }
      } else {
        *Una=0; //I only have people with nonmissing
      }
    }
  }
  p->rg(v,L,U,Lna,Una);
}



//--------------------
// Tree size
// total number of nodes (internal + terminal)
size_t tree::treesize() const
{
  if(l==0) return 1;  //if bottom node, tree size is 1
  else return (1+l->treesize()+r->treesize());
}


//--------------------
// Number of nog nodes
size_t tree::nnogs() const
{
  if(!l) return 0; //if no l exists, I am a bottom node
  if(l->l || r->l) { //if either my right or my left child has a left child, I am not a nog
    return (l->nnogs() + r->nnogs());
  } else { //is a nog (no grandchildren?)
    return 1;
  }
}


//--------------------
// Depth of node
size_t tree::depth() const
{
  if(!p) return 0; //no parents
  else return (1+p->depth());
}

//--------------------
// Node id
// Recursion up the tree
size_t tree::nid() const
{
  if(!p) return 1;                      //if you don't have a parent, you are the top
  if(this==p->l) return 2*(p->nid());   //if you are a left child
  else return 2*(p->nid())+1;           //else you are a right child
}




//--------------------
// Get bottom nodes
// Recursion down the tree
void tree::getbots(npv& bv) 
{
  if(l) { //have children
    l->getbots(bv);
    r->getbots(bv);
  } else {
    bv.push_back(this);
  }
}

void tree::getbots(cnpv& bv) const
{
  if(l) { //have children
    l->getbots(bv);
    r->getbots(bv);
  } else {
    bv.push_back(this);
  }
}



//--------------------
// Get nog nodes
// Recursion down the tree
void tree::getnogs(npv& nv)
{
  if(l) { //have children
    if((l->l) || (r->l)) {  //have grandchildren
      if(l->l) l->getnogs(nv);
      if(r->l) r->getnogs(nv);
    } else {
      nv.push_back(this);
    }
  }
}





//--------------------
// Get pointer for node from its nid
// VBT: If I'm not the node nid, then I look through my descendents
tree::tree_p tree::getptr(size_t nid)
{
  if(this->nid() == nid) return this;       //found it
  if(l==0) return 0;                        //no children, did not find it
  tree_p lp = l->getptr(nid);
  if(lp) return lp;                         //found on left
  tree_p rp = r->getptr(nid);
  if(rp) return rp;                         //found on right
  return 0;                                 //never found it
}


//--------------------
// Add children to  bot node nid splitting on var v at cutpoint c
// with mu values ml, mr for left, right children
bool tree::birth(size_t nid,
                 size_t v, 
                 size_t c, 
                 size_t na,
                 double ml, 
                 double mr,
                 double n0l,
                 double n0r)
{
  tree_p np = getptr(nid);
  if(np==0) {
    Rcpp::Rcout << "error in birth: bottom node not found\n";
    return false; //did not find note with that nid
  }
  if(np->l) {
    Rcpp::Rcout << "error in birth: found node has children\n";
    return false; //node is not a bottom node
  }
  
  //add children to bottom node np
  tree_p l = new tree;  //l lives in automatic memory, is just a new tree
  l->mu=ml;             //assign the mu of l as ml
  l->n0=n0l;
  l->n0_xnew=n0l;
  tree_p r = new tree;
  r->mu=mr;
  r->n0=n0r;
  r->n0_xnew=n0r;
  //Rcpp::Rcout << "B:(" << l->n0 <<","<< r->n0 << "),";
  if(n0l==0.0 || n0r==0.0){
    Rcpp::Rcout << "birthed a node with 0 samples; n0l=" << n0l <<", n0r="<<n0r<<endl;
  }
  
  np->l=l;              //tack l and r onto teh exitsing tree (why not do this to begin with?)
  np->r=r;
  np->v = v; 
  np->c=c;
  np->na=na;
  np->n0 = (n0l+n0r);
  np->n0_xnew = (n0l+n0r);
  l->p = np;            //we are not making copies, right? just assigning l->p (a pointer) as the pointer np
  r->p = np;
  
  return true;
}




//--------------------
// Is the node a nog node
bool tree::isnog() const
{
  bool isnog=true;
  if(l) {
    if(l->l || r->l) isnog=false; //one of the children has children.
  } else {
    isnog=false; //no children
  }
  return isnog;
}




//--------------------
// Kill children of nog node nid and update value to mu
bool tree::death(size_t nid, 
                 double mu,
                 double n0)
{
  tree_p nb = getptr(nid);
  if(nb==0) {
    Rcpp::Rcout << "error in death, nid invalid\n";
    return false;
  }
  if(nb->isnog()) {
    //Rcpp::Rcout << "K:(" << nb->l->n0 <<","<< nb->r->n0 << "),";
    delete nb->l;
    delete nb->r;
    nb->l=0;
    nb->r=0;
    nb->v=0;
    nb->c=0;
    nb->na=0;
    nb->n0=n0;          //internal nodes' n0 are not kept up; we must bring it up to date
    nb->n0_xnew=n0;
    nb->mu=mu;
    return true;
  } else {
    Rcpp::Rcout << "error in death, node is not a nog node\n";
    return false;
  }
}


//--------------------------------------------------
// Private functions
//VBT: can be accessed only by the functions inside the class

//--------------------
// Copy tree o to tree n
// Assume n has no children (so we don't have to kill them)
// Recursion down
void tree::cp(tree_p n, 
              tree_cp o)
{
  if(n->l) {
    Rcpp::Rcout << "cp:error node has children\n";
    return;
  }
  
  n->mu = o->mu;
  n->v = o->v;
  n->c = o->c;
  n->na = o->na;
  n->n0 = o->n0;
  n->n0_xnew = o->n0_xnew;
  
  if(o->l) { //if o has children
    n->l = new tree;
    (n->l)->p = n;
    cp(n->l,o->l);    //recursion left
    
    n->r = new tree;
    (n->r)->p = n;
    cp(n->r,o->r);    //recursion right
  }
}

//--------------------
// Cut back to one node
void tree::tonull()
{
  size_t ts = treesize();
  while(ts>1) { //if false ts=1
    npv nv;
    getnogs(nv);
    for(size_t i=0;i<nv.size();i++) {
      delete nv[i]->l;
      delete nv[i]->r;
      nv[i]->l=0;
      nv[i]->r=0;
    }
    ts = treesize();
  }
  mu=0.0;
  v=0;
  c=0;
  na=0;
  p=0;
  l=0;
  r=0;
}


