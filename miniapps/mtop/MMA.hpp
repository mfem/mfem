#ifndef __MMA__HPP
#define __MMA__HPP

#include "mfem.hpp"

class MMA {
  public:
    // Construct using defaults subproblem penalization
    MMA(int n, int m, double * x, int sx){};

    MMA(MPI_Comm Comm,int n, int m, double * x, int sx){};


    // Set and solve a subproblem: return new xval
    void Update(double* xval, double* dfdx, double* gx, double* dgdx, double* xmin, double xmax){};


    // Return KKT residual norms (norm2 and normInf)
    void KKTresidual(double* xval, double* dfdx, double* gx, double* dgdx, double* xmin, double* xmax, double* norm2,
                               double* normInf);


    // Options
    // Return necessary data for possible restart
    void Restart(double* xo1, double* xo2, double* U, double* L);

    // Set the aggresivity of the moving asymptotes
    void SetAsymptotes(doible init, double decrease, double increase);


  private:
    

    // Local vectors: elastic variables
    double* y;
    int  z;

    // Local vectors: Lagrange multipliers:
    double *lam, *mu, *s;

    // Global: Asymptotes, bounds, objective approx., constraint approx.
    double* L, U, alpha, beta, p0, q0, pij, qij;

    // Local: subproblem constant terms, dual gradient, dual hessian
    double *b, *grad, *Hess;

    // Global: Old design variables
    double* xo1, xo2;

};





#endif