#ifndef GUARD_bd_h
#define GUARD_bd_h

#include "rng.h"
#include "info.h"
#include "tree.h"

bool bd(tree& x, xinfo& xi, vec_s& rri, dinfo& di, pinfo& pi, RNG& gen, size_t na_mia, size_t na_seqreg, double min_node, bool check_node_missing);
bool bdhet(tree& x, xinfo& xi, vec_s& rri, dinfo& di, double* phi, pinfo& pi, RNG& gen, size_t na_mia, size_t na_seqreg, double min_node, bool check_node_missing);

#endif