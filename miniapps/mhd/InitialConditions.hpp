#include "mfem.hpp"

using namespace std;
using namespace mfem;

extern double beta; //a global value of magnetude for the pertubation
extern double Lx;  //size of x domain
extern double lambda;
extern double resiG;
extern double ep;

//initial condition
double InitialPhi(const Vector &x);
double InitialW(const Vector &x);
double InitialJ(const Vector &x);
double InitialPsi(const Vector &x);
double BackPsi(const Vector &x);

double InitialJ2(const Vector &x);
double InitialPsi2(const Vector &x);
double BackPsi2(const Vector &x);

double E0rhs(const Vector &x);
double E0rhs3(const Vector &x);
double E0rhs5(const Vector &x);

double InitialJ3(const Vector &x);
double InitialPsi3(const Vector &x);
double BackPsi3(const Vector &x);

double InitialJ6(const Vector &x);
double InitialPsi6(const Vector &x);

double resiVari(const Vector &x);

double InitialJ4(const Vector &x);
double InitialPsi4(const Vector &x);

