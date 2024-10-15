#ifndef __MMA__HPP
#define __MMA__HPP

#include "mfem.hpp"
#include <petsc.h>

/* -----------------------------------------------------------------------------
Authors: Niels Aage
 Copyright (C) 2013-2020,
This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */
class MMA
{
public:
   // Construct using defaults subproblem penalization
   MMA(PetscInt n, PetscInt m, Vec x);
   // User defined subproblem penalization
   MMA(MPI_Comm comm_,PetscInt n, PetscInt m, Vec x, PetscScalar* a,
       PetscScalar* c, PetscScalar* d);
   // Initialize with restart from itr
   MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L);
   // Initialize with restart and specify subproblem parameters
   MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L,
       PetscScalar* a, PetscScalar* c,
       PetscScalar* d);
   // Destructor
   ~MMA();

   // Set and solve a subproblem: return new xval
   PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin,
                         Vec xmax);

   // Return necessary data for possible restart
   PetscErrorCode Restart(Vec xo1, Vec xo2, Vec U, Vec L);

   // Set the aggresivity of the moving asymptotes
   PetscErrorCode SetAsymptotes(PetscScalar init, PetscScalar decrease,
                                PetscScalar increase);

   // do/don't add convexity approx to constraints: default=false
   PetscErrorCode ConstraintModification(PetscBool conMod)
   {
      constraintModification = conMod;
      return 0;
   };

   // val=0: default, val=1: increase robustness, i.e
   // control the spacing between L < alp < x < beta < U,
   PetscErrorCode SetRobustAsymptotesType(PetscInt val);

   // Sets outer movelimits on all primal design variables
   // This is often requires to prevent the solver from oscilating
   PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
                                    PetscScalar movelim, Vec x, Vec xmin,
                                    Vec xmax);

   // Return KKT residual norms (norm2 and normInf)
   PetscErrorCode KKTresidual(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx,
                              Vec xmin, Vec xmax, PetscScalar* norm2,
                              PetscScalar* normInf);

   // Inf norm on diff between two vectors: SHOULD NOT BE HERE - USE BASIC
   // PETSc!!!!!
   PetscScalar DesignChange(Vec x, Vec xold);

private:
   // Set up the MMA subproblem based on old x's and xval
   PetscErrorCode GenSub(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax);

   // Interior point solver for the subproblem
   PetscErrorCode SolveDIP(Vec xval);

   // Compute primal vars based on dual solution
   PetscErrorCode XYZofLAMBDA(Vec x);

   // Dual gradient
   PetscErrorCode DualGrad(Vec x);

   // Dual Hessian
   PetscErrorCode DualHess(Vec x);

   // Dual line search
   PetscErrorCode DualLineSearch();

   // Dual residual
   PetscScalar DualResidual(Vec x, PetscScalar epsi);

   // Problem size and iteration counter
   PetscInt n, m, k;

   // "speed-control" for the asymptotes
   PetscScalar asyminit, asymdec, asyminc;

   // do/don't add convexity constraint approximation in subproblem
   PetscBool constraintModification; // default = FALSE

   // Bool specifying if non lin constraints are included or not
   PetscBool NonLinConstraints;

   // 0: (default) span between alp L x U beta,
   // 1: increase the span for further robustness
   PetscInt RobustAsymptotesType;

   // Local vectors: penalty numbers for subproblem
   PetscScalar *a, *c, *d;

   // Local vectors: elastic variables
   PetscScalar* y;
   PetscScalar  z;

   // Local vectors: Lagrange multipliers:
   PetscScalar *lam, *mu, *s;

   // Global: Asymptotes, bounds, objective approx., constraint approx.
   Vec L, U, alpha, beta, p0, q0, *pij, *qij;

   // Local: subproblem constant terms, dual gradient, dual hessian
   PetscScalar *b, *grad, *Hess;

   // Global: Old design variables
   Vec xo1, xo2;

   // Math helpers
   PetscErrorCode Factorize(PetscScalar* K, PetscInt nn);
   PetscErrorCode Solve(PetscScalar* K, PetscScalar* x, PetscInt nn);
   PetscScalar    Min(PetscScalar d1, PetscScalar d2);
   PetscScalar    Max(PetscScalar d1, PetscScalar d2);
   PetscInt       Min(PetscInt d1, PetscInt d2);
   PetscInt       Max(PetscInt d1, PetscInt d2);
   PetscScalar    Abs(PetscScalar d1);

   // Communicator
   MPI_Comm mma_comm;
};


namespace mfem
{

class NativeMMA
{
public:
   // User defined subproblem penalization
   NativeMMA(MPI_Comm comm_, int m, mfem::Vector& x, double* a, double* c,
             double* d)
   {

      comm=comm_;
      num_con=m;
      Vec pv;
      VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&pv);

      PetscScalar* ap=new PetscScalar[m];
      PetscScalar* cp=new PetscScalar[m];
      PetscScalar* dp=new PetscScalar[m];
      for (int i=0; i<m; i++)
      {
         ap[i]=a[i];
         cp[i]=c[i];
         dp[i]=d[i];
      }

      PetscInt nn;
      VecGetSize(pv,&nn);

      mma=new MMA(comm_,nn, m, pv, ap, cp, dp);

      delete [] ap;
      delete [] cp;
      delete [] dp;

      VecDestroy(&pv);

      //allocate the native PETSc objects necessary for the subproblems

      VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xval);
      VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&dfdx);
      VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xmin);
      VecCreateMPI(comm,x.Size(),PETSC_DETERMINE,&xmax);

      VecDuplicateVecs(xval,num_con, &dgdx);
   }

   ~NativeMMA()
   {
      delete mma;

      VecDestroy(&xval);
      VecDestroy(&dfdx);
      VecDestroy(&xmin);
      VecDestroy(&xmax);
      VecDestroyVecs(num_con,&dgdx);
   }

   // Set and solve a subproblem: return new xval
   void Update(Vector& xval_, Vector& dfdx_, double* gx_, Vector* dgdx_,
               Vector& xmin_, Vector& xmax_)
   {
      //copy data
      double* data;
      VecGetArray(xval,&data);
      for (int i=0; i<xval_.Size(); i++) {data[i]=xval_[i];}
      VecRestoreArray(xval,&data);
      //dfdx
      VecGetArray(dfdx,&data);
      for (int i=0; i<xval_.Size(); i++) {data[i]=dfdx_[i];}
      VecRestoreArray(dfdx,&data);
      //dgdx
      for (int j=0; j<num_con; j++)
      {
         VecGetArray(dgdx[j],&data);
         for (int i=0; i<xval_.Size(); i++) {data[i]=(dgdx_[j])[i];}
         VecRestoreArray(dgdx[j],&data);
      }
      //xmin
      VecGetArray(xmin,&data);
      for (int i=0; i<xval_.Size(); i++) {data[i]=xmin_[i];}
      VecRestoreArray(xmin,&data);
      //xmax
      VecGetArray(xmax,&data);
      for (int i=0; i<xval_.Size(); i++) {data[i]=xmax_[i];}
      VecRestoreArray(xmax,&data);

      mma->Update(xval,dfdx,gx_,dgdx,xmin,xmax);

      VecGetArray(xval,&data);
      for (int i=0; i<xval_.Size(); i++) {xval_[i]=data[i];}
      VecRestoreArray(xval,&data);
   }
private:
   MMA* mma;
   int num_con;
   MPI_Comm comm;

   Vec xval;
   Vec dfdx;
   Vec* dgdx;
   Vec xmin;
   Vec xmax;


};

}


#endif
