// This file is part of the RBVMS application. For more information and source
// code availability visit https://idoakkerman.github.io/
//
// RBVMS is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.
//------------------------------------------------------------------------------

#ifndef RBVMS_NAVSTO_HPP
#define RBVMS_NAVSTO_HPP

#include "mfem.hpp"

using namespace mfem;

/** This class defines the time-dependent integrator for the
    Residual-based Variational multiscale formulation
    for incompressible Navier-Stokes flow.
*/
class RBVMSIntegrator: public BlockTimeDepNonlinearFormIntegrator
{
private:

   // Physical parameters
   Coefficient &c_rho;
   Coefficient &c_mu;
   VectorCoefficient &c_force;

   /// Numerical parameters
   DenseMatrix Gij;

   /// Dimension data
   int dim = -1;
   Array2D<int> hmap;

   /// Physical values
   Vector u, dudt, f, grad_p, res_m, up;
   DenseMatrix flux;

   /// Solution & Residual vector
   DenseMatrix elf_u, elf_du, elv_u;

   /// Shape function data
   Vector sh_u, ushg_u, sh_p, dupdu;
   DenseMatrix shg_u, shh_u, shg_p, grad_u, hess_u;

   /// Compute RBVMS stabilisation parameters
   void GetTau(real_t &tau_m, real_t &tau_c, real_t &cfl2,
               real_t &rho, real_t &mu, Vector &u,
               ElementTransformation &Tr);

   real_t cfl = 0;

public:
   /// Constructor
   RBVMSIntegrator(Coefficient &rho_,
                   Coefficient &mu_,
                   VectorCoefficient &force_);

   /// Set the time @a t
   void SetTime(const real_t &t) override
   {
      c_rho.SetTime(t);
      c_mu.SetTime(t);
      c_force.SetTime(t);
      cfl = 0;
   };

   /// Assemble the element constant artifical diffusion
   real_t GetElemArtDiff(const Array<const FiniteElement *> &el,
                         ElementTransformation &Tr,
                         const Array<const Vector *> &elsol,
                         const Array<const Vector *> &elrate);

   /// Assemble the local energy
   real_t GetElementEnergy(const Array<const FiniteElement *>&el,
                           ElementTransformation &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<const Vector *> &elrate) override;

   /// Assemble the element interior residual vectors
   void AssembleElementVector(const Array<const FiniteElement *> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector *> &elsol,
                              const Array<const Vector *> &elrate,
                              const Array<Vector *> &elvec) override;

   /// Assemble the element interior gradient matrices
   void AssembleElementGrad(const Array<const FiniteElement*> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector *> &elsol,
                            const Array<const Vector *> &elrate,
                            const Array2D<DenseMatrix *> &elmats) override;
};

/** This class defines the time-dependent integrator for the
    Residual-based Variational multiscale formulation
    for incompressible Navier-Stokes flow.
*/
class OutflowIntegrator: public BlockTimeDepNonlinearFormIntegrator
{
private:

   // Physical parameters
   Coefficient &c_rho;

   /// Numerical parameters
   DenseMatrix Gij;
   Vector hn;

   /// Dimension data
   int dim = -1;

   /// Physical values
   Vector u, nor, traction;
   DenseMatrix flux;

   /// Solution & Residual vector
   DenseMatrix elf_u, elf_du, elv_u;

   /// Shape function data
   Vector sh_u, ushg_u, sh_p;
   DenseMatrix shg_u, shh_u, shg_p, grad_u, hess_u;

   /// Set dimension
   void SetDim (int dim_);

public:
   /// Constructor
   OutflowIntegrator(Coefficient &rho_)
      : c_rho(rho_) {};


   /// Set the time @a t
   void SetTime(const real_t &t) override
   {
      c_rho.SetTime(t);
   };

   /// Assemble the outflow boundary residual vectors
   void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                           const Array<const FiniteElement *> &el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<const Vector *> &elrate,
                           const Array<Vector *> &elvect) override;

   /// Assemble the outflow boundary gradient matrices
   void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                         const Array<const FiniteElement *>&el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *> &elfun,
                         const Array<const Vector *> &elrate,
                         const Array2D<DenseMatrix *> &elmats) override;
};

/** This class defines the time-dependent integrator for the
    Residual-based Variational multiscale formulation
    for incompressible Navier-Stokes flow.
*/
class WeakBCIntegrator: public BlockTimeDepNonlinearFormIntegrator
{
private:

   // Physical parameters
   Coefficient &c_rho;
   Coefficient &c_mu;
   VectorCoefficient &c_sol;

   /// Numerical parameters
   DenseMatrix Gij;
   Vector hn;

   /// Dimension data
   int dim = -1;

   /// Physical values
   Vector u, ug, ud,  nor, traction;
   DenseMatrix flux;

   /// Solution & Residual vector
   DenseMatrix elf_u, elf_du, elv_u;

   /// Shape function data
   Vector sh_u, ushg_u, sh_p, dupdu;
   DenseMatrix shg_u, shh_u, shg_p, grad_u, hess_u;

   /// Set dimension
   void SetDim (int dim_);

   /// Compute Weak Dirichlet stabilisation parameters
   void GetTauB(real_t &tau_b, real_t &tau_n,
                real_t &mu, Vector &u,
                Vector &nor,
                FaceElementTransformations &Tr);

public:
   /// Constructor
   WeakBCIntegrator(Coefficient &rho_,
                    Coefficient &mu_,
                    VectorCoefficient &sol_)
      : c_rho(rho_), c_mu(mu_), c_sol(sol_) {};

   /// Set the time @a d
   void SetTime(const real_t &t) override
   {
      c_rho.SetTime(t);
      c_mu.SetTime(t);
      c_sol.SetTime(t);
   };

   /// Assemble the outflow boundary residual vectors
   void AssembleFaceVector(const Array<const FiniteElement *> &el1,
                           const Array<const FiniteElement *> &el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *> &elfun,
                           const Array<const Vector *> &elrate,
                           const Array<Vector *> &elvect) override;

   /// Assemble the outflow boundary gradient matrices
   void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                         const Array<const FiniteElement *>&el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *> &elfun,
                         const Array<const Vector *> &elrate,
                         const Array2D<DenseMatrix *> &elmats) override;
};

/// Newton's method for solving F(x)=b for a given operator F.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class NewtonSystemSolver : public NewtonSolver
{
private:
   Array<int> &bOffsets;
   int nvar;

   // Compute the norms for each component
   void Norms(const Vector &r,Vector& norm) const;

   bool parallel = false;

public:
   // Constructor, provide MPI context and component subdivision
   NewtonSystemSolver( Array<int> &offsets)
      : bOffsets(offsets)
   {
      nvar = bOffsets.Size()-1;
      parallel = false;

   }

   NewtonSystemSolver(MPI_Comm comm_, Array<int> &offsets)
      : NewtonSolver(comm_), bOffsets(offsets)
   {
      nvar = bOffsets.Size()-1;
      parallel = true;
   }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const Vector &b, Vector &x) const;
};


#endif
