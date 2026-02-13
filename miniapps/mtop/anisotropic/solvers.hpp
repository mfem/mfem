// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include <memory>

#include "mfem.hpp"
#include "linear_anisotropic_elasticity.hpp"

using real_t = mfem::real_t;

// gibbs function y <- exp(x) / sum(exp(x))
inline void gibbs(const mfem::Vector &x, mfem::Vector &y)
{
   y = x;
   const real_t maxx = x.Max();
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      y[i] = exp(x[i] - maxx);
      sum += y[i];
   }
   y /= sum;
}
// in-place gibbs function x <- exp(x) / sum(exp(x))
inline void gibbs(mfem::Vector &x)
{
   const real_t maxx = x.Max();
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      x[i] = exp(x[i] - maxx);
      sum += x[i];
   }
   x /= sum;
}
class PolytopeMirrorCF : public mfem::VectorCoefficient
{
   const mfem::DenseMatrix &V;
   VectorCoefficient &psi_cf;
   mutable mfem::Vector psi;
   mutable mfem::Vector Vtpsi; // V^T * psi
   mutable mfem::Vector lambda; // intermediate varicentric coordinates
public:
   PolytopeMirrorCF(const mfem::DenseMatrix &V_, mfem::VectorCoefficient &psi_)
      : VectorCoefficient(V_.Height())
      , V(V_), psi_cf(psi_)
      , psi(V_.Height()), Vtpsi(V_.Width()), lambda(V_.Width())
   {
      MFEM_VERIFY(V.Height() == psi_cf.GetVDim(),
                  "V and psi_cf should have the same height");
   }
   void Eval(mfem::Vector &p, mfem::ElementTransformation &T,
             const mfem::IntegrationPoint &ip) override
   {
      psi_cf.Eval(psi, T, ip);
      V.MultTranspose(psi, lambda);
      gibbs(lambda);
      V.Mult(lambda, p);
   }
};


///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticyLambdaCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Lame's lambda coefficient
class IsoElasticyLambdaCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticyLambdaCoeff(mfem::Coefficient *E,
                          mfem::Coefficient *nu):
      E(E), nu(nu) { }

   /// Evaluates the Lame's lambda coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Lambda = [](const real_t E, const real_t nu)
      {
         return E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
      };
      return Lambda(EE, nn);
   }
};

///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoElasticySchearCoeff class converts E modulus of elasticity
/// and Poisson's ratio to Shear coefficient
///
class IsoElasticyShearCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticyShearCoeff(mfem::Coefficient *E_, mfem::Coefficient *nu_):
      E(E_), nu(nu_) { }

   /// Evaluates the shear coefficient coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Shear = [](const real_t E, const real_t nu)
      {
         return E / (2.0 * (1.0 + nu));
      };
      return Shear(EE, nn);
   }
};

///////////////////////////////////////////////////////////////////////////////
/// \brief The AnisoLinElasticSolver class provides a solver for
/// linear anisotropic elasticity. The solver provides options for
/// dFEM based integrators and full assembly.
/// The preconditioners are based on block LOR approximations.
class AnisoLinElasticSolver : public mfem::Operator
{
public:
   /// Construct the solver for a given mesh and discretization
   /// order.
   AnisoLinElasticSolver(mfem::ParMesh *mesh, int vorder = 1);

   /// Destructor of the solver.
   virtual
   ~AnisoLinElasticSolver();

   /// Sets the linear solver relative tolerance (rtol),
   /// absolute tolerance (atol) and maximum number of
   /// iterations miter.
   void SetLinearSolver(real_t rtol = 1e-8,
                        real_t atol = 1e-12,
                        int miter = 200);

   /// Solves the forward problem.
   void FSolve();

   /// Adds displacement BC in direction 0(x), 1(y), 2(z), or -1(all).
   void AddDispBC(int id, int dir, real_t val);

   /// Adds displacement BC in direction 0(x), 1(y), 2(z), or -1(all).
   void AddDispBC(int id, int dir, mfem::Coefficient &val);

   /// Clear all displacement BC
   void DelDispBC();

   /// Set the values of the volumetric force.
   void SetVolForce(real_t fx, real_t fy, real_t fz = 0.0);

   /// Associates coefficient to the volumetric force.
   void SetVolForce(mfem::VectorCoefficient &ff);

   /// Returns the displacements.
   mfem::ParGridFunction &GetDisplacements()
   {
      fdisp.SetFromTrueDofs(sol);
      return fdisp;
   }

   /// Returns the adjoint displacements.
   mfem::ParGridFunction &GetADisplacements()
   {
      adisp.SetFromTrueDofs(adj);
      return adisp;
   }

   /// Returns the solution vector.
   mfem::Vector &GetSolutionVector() { return sol; }

   /// Returns the adjoint solution vector.
   mfem::Vector &GetAdjointSolutionVector() { return adj; }

   /// Returns the displacements.
   void GetSol(mfem::ParGridFunction &sgf)
   {
      sgf.SetSpace(vfes);
      sgf.SetFromTrueDofs(sol);
   }

   /// Returns the adjoint displacements.
   void GetAdj(mfem::ParGridFunction &agf)
   {
      agf.SetSpace(vfes);
      agf.SetFromTrueDofs(adj);
   }

   /// Sets BC dofs, bilinear form, preconditioner and solver.
   /// Should be called before calling Mult of MultTranspose
   virtual void Assemble();

   /// Forward solve with given RHS. x is the RHS vector.
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Adjoint solve with given RHS. x is the RHS vector.
   /// The essential BCs are set to zero.
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

   void SetDesignField(mfem::VectorCoefficient &design_field)
   {
      eta = std::make_unique<mfem::CoefficientVector>(*qs);
      p_eta = &design_field;
      eta->Project(design_field);
   }

   /// Set materials
   void SetIsoMaterials(mfem::Coefficient &E1_, mfem::Coefficient &nu1_,
                        mfem::Coefficient &E2_, mfem::Coefficient &nu2_)
   {
      p_E1 = &E1_;
      p_nu1 = &nu1_;
      p_E2 = &E2_;
      p_nu2 = &nu2_;


      lambda1.reset(new IsoElasticyLambdaCoeff(p_E1, p_nu1));
      mu1.reset(new IsoElasticyShearCoeff(p_E1, p_nu1));
      lambda2.reset(new IsoElasticyLambdaCoeff(p_E2, p_nu2));
      mu2.reset(new IsoElasticyShearCoeff(p_E2, p_nu2));

      E1.reset(); E2.reset();
      nu1.reset(); nu2.reset();

      l1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));
      l2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));
      m1.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));
      m2.reset(new mfem::CoefficientVector(*qs, mfem::CoefficientStorage::FULL));
      l1->Project(*lambda1);
      l2->Project(*lambda2);
      m1->Project(*mu1);
      m2->Project(*mu2);

   }

   /// Set materials - transfer the ounership
   void SetIsoMaterials(std::shared_ptr<mfem::Coefficient> E1_,
                        std::shared_ptr<mfem::Coefficient> nu1_,
                        std::shared_ptr<mfem::Coefficient> E2_,
                        std::shared_ptr<mfem::Coefficient> nu2_)
   {
      p_E1 = E1_.get();
      p_nu1 = nu1_.get();
      p_E2 = E2_.get();
      p_nu2 = nu2_.get();


      lambda1.reset(new IsoElasticyLambdaCoeff(p_E1, p_nu1));
      mu1.reset(new IsoElasticyShearCoeff(p_E1, p_nu1));
      lambda2.reset(new IsoElasticyLambdaCoeff(p_E2, p_nu2));
      mu2.reset(new IsoElasticyShearCoeff(p_E2, p_nu2));

      E1=E1_;
      E2=E2_;
      nu1=nu1_;
      nu2=nu2_;
   }

   /// Add surface load.
   void AddSurfLoad(int id, real_t fx, real_t fy, real_t fz = 0.0)
   {
      mfem::Vector vec;
      vec.SetSize(spaceDim);
      vec[0] = fx;
      vec[1] = fy;
      if (spaceDim == 3) { vec[2] = fz; }
      auto *vc = new mfem::VectorConstantCoefficient(vec);
      if (load_coeff.find(id) != load_coeff.end()) { delete load_coeff[id]; }
      load_coeff[id] = vc;
   }

   /// Add surface load
   void AddSurfLoad(int id, mfem::VectorCoefficient &ff)
   {
      surf_loads[id] = &ff;
   }

   // creates a list with essential dofs
   // sets the values in the bsol vector
   // the list is written in ess_dofs
   // The 'nvcc' compiler needs these SetEssTDofs functions to be public.
   void SetEssTDofs(mfem::Vector &bsol, mfem::Array<int> &ess_dofs);
   void SetEssTDofs(const int j, mfem::ParFiniteElementSpace& scalar_space,
                    mfem::Array<int> &ess_dofs);

   void SetAnisotropicTensor2D(real_t E, real_t nu, real_t Ex,
                               bool lower_bound=true)
   {
      if (lower_bound)
      {
         //lower bound of the stiffness tensor
         auto iC=mfem::voigt::IsotropicCompliance2D_PlaneStress(E,nu);
         iC(0,0)=iC(0,0)+1.0/Ex;
         auto C=mfem::future::inv(iC);
         aniso_tensor[0]=C(0,0);
         aniso_tensor[1]=C(0,1);
         aniso_tensor[2]=C(0,2);
         aniso_tensor[3]=C(1,1);
         aniso_tensor[4]=C(1,2);
         aniso_tensor[5]=C(2,2);
      }
      else
      {
         //uper bound of the stiffness tensor
         auto C=mfem::voigt::IsotropicStiffness2D_PlaneStress(E,nu);
         C(0,0)=C(0,0)+Ex;
         aniso_tensor[0]=C(0,0);
         aniso_tensor[1]=C(0,1);
         aniso_tensor[2]=C(0,2);
         aniso_tensor[3]=C(1,1);
         aniso_tensor[4]=C(1,2);
         aniso_tensor[5]=C(2,2);
      }
   }

private:
   mfem::ParMesh *pmesh;
   const int dim, spaceDim;

   // finite element collection for linear elasticity
   mfem::FiniteElementCollection *vfec;

   // finite element space for linear elasticity
   mfem::ParFiniteElementSpace *vfes;

   // solution true vector
   mutable mfem::Vector sol;
   // adjoint true vector
   mutable mfem::Vector adj;
   // RHS
   mutable mfem::Vector rhs;

   // forward solution
   mfem::ParGridFunction fdisp;
   // adjoint solution
   mfem::ParGridFunction adisp;

   // Linear solver parameters
   real_t linear_rtol;
   real_t linear_atol;
   int linear_iter;

   const mfem::FiniteElement *fe;
   mfem::ParGridFunction *nodes;
   mfem::ParFiniteElementSpace *mfes;
   mfem::Array<int> domain_attributes;
   const mfem::IntegrationRule &ir;
   std::unique_ptr<mfem::QuadratureSpace> qs;
   std::unique_ptr<mfem::future::UniformParameterSpace> ups;

   static constexpr int FDispl = 0; //grid function displacement
   // elasticity Coefficient Vectors
   static constexpr int Lambda1 = 2, Lambda2 = 3, Mu1 = 4, Mu2 = 5;
   // polytopal design variable
   static constexpr int Indicator = 6;
   // static constexpr int DensA = 6, DensB = 7;
   // // density for topology optimization
   // static constexpr int Density = 14; // coefficient vector
   static constexpr int Coords = 15; // coordinates grid function

   // DFEM operators definitions
   //evaluates the contribution to the residual
   std::unique_ptr<mfem::future::DifferentiableOperator> drhs;
   //tangent matrix
   std::shared_ptr<mfem::future::DerivativeOperator> dr_du;

   mfem::HypreParMatrix* K;

   // density coefficients in dFEM form for material 1 and 2
   std::unique_ptr<mfem::CoefficientVector> eta;
   // linear elasticty coefficients in dFEM form
   // l1, m1 - material 1
   // l2, m2 - material 2
   std::unique_ptr<mfem::CoefficientVector> l1, l2;
   std::unique_ptr<mfem::CoefficientVector> m1, m2;





   std::unique_ptr<mfem::HypreBoomerAMG> prec; // preconditioner
   std::unique_ptr<mfem::CGSolver> ls;         // linear solver

   /// Volumetric force created by the solver.
   mfem::VectorConstantCoefficient *lvforce;
   /// Volumetric force coefficient can point to the one
   /// created by the solver or to external vector coefficient.
   mfem::VectorCoefficient *volforce;

   // surface loads
   using VectorCoefficientPtrMap = std::map<int, mfem::VectorCoefficient *>;
   VectorCoefficientPtrMap load_coeff; // internaly generated load
   VectorCoefficientPtrMap surf_loads; // external vector coeeficients

   class SurfaceLoad;
   std::unique_ptr<SurfaceLoad> lcsurf_load; // localy generated surface loads
   std::unique_ptr<SurfaceLoad> glsurf_load; // global surface loads

   // boundary conditions for x,y, and z directions
   using ConstantCoefficientMap = std::map<int, mfem::ConstantCoefficient>;
   ConstantCoefficientMap bcx, bcy, bcz;

   // holds BC in coefficient form
   using CoefficientPtrMap = std::map<int, mfem::Coefficient*>;
   CoefficientPtrMap bccx, bccy, bccz;

   // holds the displacement contrained DOFs
   mfem::Array<int> ess_tdofv;

   //isotropic materials
   mfem::Coefficient *p_E1, *p_nu1;
   mfem::Coefficient *p_E2, *p_nu2;
   mfem::VectorCoefficient *p_eta;

   std::shared_ptr<mfem::Coefficient> E1, E2;
   std::shared_ptr<mfem::Coefficient> nu1, nu2;
   std::unique_ptr<mfem::Coefficient> lambda1, mu1;
   std::unique_ptr<mfem::Coefficient> lambda2, mu2;

   //anisotrpic material
   //full stiffness tensor in Voigt notations [3x3 in 2D, 6x6 in 3D]
   mfem::Vector aniso_tensor;

   class SurfaceLoad: public mfem::VectorCoefficient
   {
      VectorCoefficientPtrMap *map;
   public:
      SurfaceLoad(int dim, VectorCoefficientPtrMap &cmap):
         mfem::VectorCoefficient(dim)
      {
         map = &cmap;
      }
      using mfem::VectorCoefficient::Eval;

      void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override
      {
         V.SetSize(GetVDim());
         V = 0.0;
         auto it = map->find(T.Attribute);
         if (it != map->end()) { it->second->Eval(V, T, ip); }
      }
   };

};




