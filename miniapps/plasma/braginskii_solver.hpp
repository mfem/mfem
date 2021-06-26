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

#ifndef MFEM_BRAGINSKII_TRANSPORT_SOLVER
#define MFEM_BRAGINSKII_TRANSPORT_SOLVER

#include "../common/pfem_extras.hpp"
#include "braginskii_coefs.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace plasma
{

class TwoFluidDiffusion;
class TwoFluidAdvection;

class TwoFluidTransportSolver : public ODESolver
{
private:
   ODESolver * impSolver_;
   ODESolver * expSolver_;

   ParFiniteElementSpace & sfes_; // Scalar fields
   ParFiniteElementSpace & vfes_; // Vector fields
   ParFiniteElementSpace & ffes_; // Full system

   Array<int>  & offsets_;
   Array<int>  & toffsets_;
   BlockVector & nBV_;
   BlockVector & uBV_;
   BlockVector & TBV_;

   ParGridFunction & B_;

   double ion_mass_;
   double ion_charge_;

   TwoFluidDiffusion * tfDiff_;
   TwoFluidAdvection * tfAdvc_;

   void initDiffusion();
   void initAdvection();

public:
   TwoFluidTransportSolver(ODESolver * implicitSolver,
                           ODESolver * explicitSolver,
                           ParFiniteElementSpace & sfes,
                           ParFiniteElementSpace & vfes,
                           ParFiniteElementSpace & ffes,
                           Array<int> & offsets,
                           Array<int> & toffsets,
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

   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   Array<int>  & offsets_;
   Array<int>  & toffsets_;
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
   BlockVector block_RHS_;
   BlockVector block_dXdt_;
   BlockDiagonalPreconditioner block_amg_;

   GMRESSolver gmres_;

   void initCoefficients();
   void initBilinearForms();
   void initSolver();

   void deleteCoefficients();
   void deleteBilinearForms();

   void setTimeStep(double dt);

public:
   TwoFluidDiffusion(ParFiniteElementSpace & sfes,
                     ParFiniteElementSpace & vfes,
                     Array<int> & offsets,
                     Array<int> & toffsets,
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

class TwoFluidAdvection : public TimeDependentOperator
{
private:
   int dim_;

   ParFiniteElementSpace &sfes_;
   ParFiniteElementSpace &vfes_;

   Array<int>  & offsets_;
   Array<int>  & toffsets_;
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
   std::vector<pAdvectionCoefficient *> pAdvcCoef_;

   std::vector<dEdnCoefficient *> dEdnCoef_;
   std::vector<dEduCoefficient *> dEduCoef_;
   std::vector<dEdTCoefficient *> dEdTCoef_;
   std::vector<TAdvectionCoefficient *> TAdvcCoef_;

   // Bilinear Forms for particle equation
   std::vector<ParBilinearForm *> a_dndn_;
   std::vector<ParBilinearForm *> advc_n_;

   // Bilinear Forms for momentum equation
   std::vector<ParBilinearForm *> a_dpdn_;
   std::vector<ParBilinearForm *> a_dpdu_;
   std::vector<ParBilinearForm *> advc_p_;

   // Bilinear Forms for energy equation
   std::vector<ParBilinearForm *> a_dEdn_;
   std::vector<ParBilinearForm *> a_dEdu_;
   std::vector<ParBilinearForm *> a_dEdT_;
   std::vector<ParBilinearForm *> advc_T_;

   BlockOperator block_A_;
   BlockOperator block_B_;
   mutable  BlockVector block_rhs_;
   mutable BlockVector block_RHS_;
   mutable BlockVector block_dXdt_;
   BlockDiagonalPreconditioner block_ds_;
   // std::vector<HypreSolver *> diag_scale_;

   GMRESSolver gmres_;

   void initCoefficients();
   void initBilinearForms();
   void initSolver();

   void deleteCoefficients();
   void deleteBilinearForms();

   void setTimeStep(double dt);

public:
   TwoFluidAdvection(ParFiniteElementSpace & sfes,
                     ParFiniteElementSpace & vfes,
                     Array<int> & offsets,
                     Array<int> & toffsets,
                     BlockVector & nBV,
                     BlockVector & uBV,
                     BlockVector & TBV,
                     ParGridFunction & B,
                     double ion_mass,
                     double ion_charge);

   ~TwoFluidAdvection();

   void Assemble();

   void Update();

   void Mult(const Vector &x, Vector &y) const;
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form for the diffusion term. (modified from ex14p)
class DiffusionTDO : public TimeDependentOperator
{
private:
   const int dim_;
   double dt_;

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
                MatrixCoefficient & nuCoef);

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

class AdvectionDiffusionTDO : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &fes_;

   Coefficient       & mCoef_;
   Coefficient       * dCoef_;
   MatrixCoefficient * DCoef_;
   VectorCoefficient * VCoef_;
   Coefficient       * SCoef_;

   Coefficient       * dudtBCCoef_;
   VectorCoefficient * dvdtBCCoef_; // where v = Grad(u)

   ProductCoefficient             * dtdCoef_;
   ScalarMatrixProductCoefficient * dtDCoef_;
   ScalarVectorProductCoefficient * dtVCoef_;

   Array<int> dudt_bc_bdr_;
   Array<int> dudt_bc_dofs_;
   Array<int> dvdt_bc_bdr_;

   bool td_M_;
   bool td_D_;
   bool td_V_;
   bool td_S_;
   bool td_dudt_;
   bool td_dvdt_;

   ParBilinearForm a0_;
   ParBilinearForm * s0_;
   ParBilinearForm * c0_;

   ParLinearForm   * s_;

   ParGridFunction dudt_;

   void initBLFs();

public:
   AdvectionDiffusionTDO(ParFiniteElementSpace &fes,
                         Coefficient &mCoef, bool td,
                         Coefficient *dCoef, bool td_D,
                         VectorCoefficient *VCoef, bool td_V);
   AdvectionDiffusionTDO(ParFiniteElementSpace &fes,
                         Coefficient &mCoef, bool td,
                         MatrixCoefficient *DCoef, bool td_D,
                         VectorCoefficient *VCoef, bool td_V);
   ~AdvectionDiffusionTDO();

   // In each of the following "Set" methods the "td" argument indicates
   // that the coefficient is time-dependent.
   // void SetMassCoef(Coefficient &M, bool td);
   // void SetDiffusionCoef(Coefficient &D, bool td);
   // void SetDiffusionCoef(MatrixCoefficient &D, bool td);
   // void SetAdvectionCoef(VectorCoefficient &V, bool td);
   void SetSourceCoef(Coefficient &S, bool td);
   void SetDirichletBC(const Array<int> &bdr_attr, Coefficient &dudt, bool td);
   void SetNeumannBC(const Array<int> &bdr_attr,
                     VectorCoefficient &dvdt, bool td);

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);
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

#endif // MFEM_BRAGINSKII_TRANSPORT_SOLVER
