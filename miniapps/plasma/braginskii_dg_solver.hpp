// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BRAGINSKII_DG_TRANSPORT_SOLVER
#define MFEM_BRAGINSKII_DG_TRANSPORT_SOLVER

#include "../common/pfem_extras.hpp"
#include "braginskii_coefs.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

struct DGParams
{
   double sigma;
   double kappa;
};

class TwoFluidDiffusion;
class TwoFluidAdvection;

class TwoFluidTransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   DGParams & dg_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   Array<int>  & offsets_;
   BlockVector & nBV_;
   BlockVector & uBV_;
   BlockVector & TBV_;

   ParGridFunction & B_;

   double ion_mass_;
   double ion_charge_;

   TwoFluidDiffusion * tfDiff_;

   void initDiffusion();

public:
   TwoFluidTransportSolver(ODESolver * implicitSolver,
                           ODESolver * explicitSolver,
                           DGParams & dg,
                           ParFiniteElementSpace & sfes,
                           ParFiniteElementSpace & vfes,
                           ParFiniteElementSpace & ffes,
                           Array<int> & offsets,
                           BlockVector & nBV,
                           BlockVector & uBV,
                           BlockVector & TBV,
                           ParGridFunction & B,
                           double ion_mass,
                           double ion_charge);
   ~TwoFluidTransportSolver();

   void Update();

   void Step(Vector &x, double &t, double &dt);
};

class TwoFluidDiffusion : public TimeDependentOperator
{
private:
   int dim_;

   DGParams & dg_;

   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   Array<int>  & offsets_;
   BlockVector & nBV_;
   BlockVector & uBV_;
   BlockVector & TBV_;

   ParGridFunction & B_;

   double ion_mass_;
   double ion_charge_;

   std::vector<ParGridFunction> nGF_;
   std::vector<ParGridFunction> uGF_;
   std::vector<ParGridFunction> TGF_;

   std::vector<GridFunctionCoefficient>       nCoef_;
   std::vector<VectorGridFunctionCoefficient> uCoef_;
   std::vector<GridFunctionCoefficient>       TCoef_;

   std::vector<Coefficient *> dndnCoef_;

   std::vector<dpdnCoefficient *> dpdnCoef_;
   std::vector<dpduCoefficient *> dpduCoef_;

   std::vector<dEdnCoefficient *> dEdnCoef_;
   std::vector<dEduCoefficient *> dEduCoef_;
   std::vector<dEdTCoefficient *> dEdTCoef_;

   // std::vector<DiffCoefficient *> diffCoef_;
   std::vector<ChiCoefficient *>  chiCoef_;
   std::vector<EtaCoefficient *>  etaCoef_;

   // std::vector<ScalarMatrixProductCoefficient *> dtDiffCoef_;
   std::vector<ScalarMatrixProductCoefficient *> dtChiCoef_;
   std::vector<ScalarMatrixProductCoefficient *> dtEtaCoef_;

   // Bilinear Forms for particle equation
   std::vector<ParBilinearForm *> a_dndn_;
   // std::vector<ParBilinearForm *> stiff_D_;

   // Bilinear Forms for momentum equation
   std::vector<ParBilinearForm *> a_dpdn_;
   std::vector<ParBilinearForm *> a_dpdu_;
   std::vector<ParBilinearForm *> stiff_eta_;

   // Bilinear Forms for energy equation
   std::vector<ParBilinearForm *> a_dEdn_;
   std::vector<ParBilinearForm *> a_dEdu_;
   std::vector<ParBilinearForm *> a_dEdT_;
   std::vector<ParBilinearForm *> stiff_chi_;

   BlockOperator block_A_;
   BlockOperator block_B_;
   BlockVector block_rhs_;
   BlockDiagonalPreconditioner block_amg_;
   std::vector<HypreSolver *> amg_;

   GMRESSolver gmres_;

   void initCoefficients();
   void initBilinearForms();
   void initSolver();

   void deleteCoefficients();
   void deleteBilinearForms();

   void setTimeStep(double dt);

public:
   TwoFluidDiffusion(DGParams & dg,
                     ParFiniteElementSpace & sfes,
                     ParFiniteElementSpace & vfes,
                     Array<int> & offsets,
                     BlockVector & nBV,
                     BlockVector & uBV,
                     BlockVector & TBV,
                     ParGridFunction & B,
                     double ion_mass,
                     double ion_charge);

   ~TwoFluidDiffusion();

   void Assemble();

   void Update();

   void ImplicitSolve(const double dt, const Vector &x, Vector &y);
};


// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the diffusion term. (modified from ex14p)
class DiffusionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   double dt_;
   double dg_sigma_;
   double dg_kappa_;

   ParFiniteElementSpace &fes_;
   ParFiniteElementSpace &dfes_;
   ParFiniteElementSpace &vfes_;

   ParBilinearForm m_;
   ParBilinearForm d_;

   ParLinearForm rhs_;
   ParGridFunction x_;

   HypreParMatrix * M_;
   HypreParMatrix * D_;

   Vector RHS_;
   Vector X_;

   HypreSolver * solver_;
   HypreSolver * amg_;

   MatrixCoefficient &nuCoef_;
   ScalarMatrixProductCoefficient dtNuCoef_;

   void initSolver(double dt);

public:
   DiffusionTDO(ParFiniteElementSpace &fes,
                ParFiniteElementSpace &dfes,
                ParFiniteElementSpace &_vfes,
                MatrixCoefficient & nuCoef,
                double dg_sigma,
                double dg_kappa);

   // virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~DiffusionTDO() { }
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the advection term.
class AdvectionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   const int num_equation_;
   const double specific_heat_ratio_;

   mutable double max_char_speed_;

   ParFiniteElementSpace &vfes_;
   Operator &A_;
   SparseMatrix &Aflux_;
   DenseTensor Me_inv_;

   mutable Vector state_;
   mutable DenseMatrix f_;
   mutable DenseTensor flux_;
   mutable Vector z_;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   AdvectionTDO(ParFiniteElementSpace &_vfes,
                Operator &A, SparseMatrix &Aflux, int num_equation,
                double specific_heat_ratio);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~AdvectionTDO() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   int num_equation_;
   double specific_heat_ratio_;
   Vector flux1_;
   Vector flux2_;

public:
   RiemannSolver(int num_equation, double specific_heat_ratio);
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};


// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape_;
   DenseMatrix flux_;
   DenseMatrix dshapedr_;
   DenseMatrix dshapedx_;

public:
   DomainIntegrator(const int dim, const int num_equation);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   int num_equation_;
   double max_char_speed_;
   RiemannSolver rsolver_;
   Vector shape1_;
   Vector shape2_;
   Vector funval1_;
   Vector funval2_;
   Vector nor_;
   Vector fluxN_;
   IntegrationPoint eip1_;
   IntegrationPoint eip2_;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim,
                  const int num_equation);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_BRAGINSKII_DG_TRANSPORT_SOLVER
