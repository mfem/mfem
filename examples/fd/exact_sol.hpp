
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double lshape_exact(const Vector & x);
void lshape_grad(const Vector & x, Vector & grad);
double lshape_rhs(const Vector & x);

double wavefront_exact(const Vector & x);
void wavefront_grad(const Vector & x, Vector & grad);
double wavefront_rhs(const Vector & x);