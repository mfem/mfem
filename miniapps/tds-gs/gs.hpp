#ifndef GS
#define GS

#include "mfem.hpp"
#include <set>
#include <limits>
#include <iostream>
#include <math.h>

#include "test.hpp"
#include "exact.hpp"
#include "initial_coefficient.hpp"
#include "plasma_model.hpp"
#include "sys_operator.hpp"
#include "boundary.hpp"
#include "diffusion_term.hpp"

using namespace std;
using namespace mfem;

const int attr_r_eq_0_bdr = 831;
const int attr_ff_bdr = 900;
const int attr_lim = 1000;
const int attr_ext = 2000;


double gs(const char * mesh_file, const char * data_file, int order, int d_refine);
  
#endif
