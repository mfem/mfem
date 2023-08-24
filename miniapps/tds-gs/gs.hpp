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
#include<stdio.h>

using namespace std;
using namespace mfem;

const int attr_r_eq_0_bdr = 900;
const int attr_ff_bdr = 831;
const int attr_lim = 1000;
const int attr_ext = 2000;

const int num_currents = 11;


double gs(const char * mesh_file, const char * data_file, int order, int d_refine,
          int model,
          double & alpha, double & beta, double & gamma, double & mu, double & Ip,
          double & r0, double & rho_gamma, int max_krylov_iter, int max_newton_iter,
          double & krylov_tol, double & newton_tol,
          double & c1, double & c2, double & c3, double & c4, double & c5, double & c6, double & c7,
          double & c8, double & c9, double & c10, double & c11,
          double & ur_coeff,
          int do_control, int N_control,
          double & weight_solenoids,
          double & weight_coils,
          double & weight_obj, int obj_option, bool optimize_alpha,
          bool do_manufactured_solution,
          bool do_initial);

#endif
