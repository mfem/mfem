#ifndef HPTEST_EXACT_HPP
#define HPTEST_EXACT_HPP

namespace mfem
{

// classic L-shape domain problem with a singularity in reentrant corner
double lshape_exsol(const Vector &p);
void   lshape_exgrad(const Vector &p, Vector &grad);
double lshape_laplace(const Vector &p);

double fichera_exsol(const Vector &p);
void   fichera_exgrad(const Vector &p, Vector &grad);
double fichera_laplace(const Vector &p);

// shock-like "inner layer" problem
double layer2_exsol(const Vector &p);
void   layer2_exgrad(const Vector &p, Vector &grad);
double layer2_laplace(const Vector &p);

// shock-like "inner layer" problem generalized to 3D
double layer3_exsol(const Vector &p);
void   layer3_exgrad(const Vector &p, Vector &grad);
double layer3_laplace(const Vector &p);

void hcurl_exsol(const Vector &x, Vector &E);
void hcurl_exrhs(const Vector &x, Vector &f);

void hdiv_exsol(const Vector &x, Vector &E);
void hdiv_exrhs(const Vector &x, Vector &f);


} // namespace mfem

#endif // HPTEST_EXACT_HPP
