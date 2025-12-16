#pragma once
#include "mfem.hpp"

using namespace mfem;
using namespace std;

extern double delta;
extern double a0;
extern double a1;

std::complex<real_t> pfunc(const Vector &x);
std::complex<real_t> sfunc(const Vector &x);

real_t pfunc_r(const Vector &x);
real_t pfunc_i(const Vector &x);
real_t sfunc_r(const Vector &x);
real_t sfunc_i(const Vector &x);
void bfunc(const Vector &x, Vector &b);
void bcrossb(const Vector &x, DenseMatrix &bb);
void epsilon_func_r(const Vector &x, DenseMatrix &eps);
void epsilon_func_i(const Vector &x, DenseMatrix &eps);