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

using real_t = mfem::real_t;

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
class IsoElasticySchearCoeff : public mfem::Coefficient
{
   mfem::Coefficient *E, *nu;

public:
   /// Constructor - takes as inputs E modulus and Poisson's ratio
   IsoElasticySchearCoeff(mfem::Coefficient *E_, mfem::Coefficient *nu_):
      E(E_), nu(nu_) { }

   /// Evaluates the shear coefficient coefficient
   real_t Eval(mfem::ElementTransformation &T,
               const mfem::IntegrationPoint &ip) override
   {
      const real_t EE = E->Eval(T, ip);
      const real_t nn = nu->Eval(T, ip);
      constexpr auto Schear = [](const real_t E, const real_t nu)
      {
         return E / (2.0 * (1.0 + nu));
      };
      return Schear(EE, nn);
   }
};

///////////////////////////////////////////////////////////////////////////////
/// \brief The IsoLinElasticSolver class provides a solver for
/// linear isotropic elasticity. The solver provides options for
/// partial assembly, dFEM based integrators, and full assembly.
/// The preconditioners are based on block LOR approximations.
class IsoLinElasticSolver : public mfem::Operator
{
public:
   /// Construct the solver for a given mesh and discretization
   /// order. The parameters pa and dfem define the utilization of
   /// partial assembly or dFEM based integrators.
   IsoLinElasticSolver(mfem::ParMesh *mesh, int vorder = 1,
                       bool pa = false, bool dfem = false);

   /// Destructor of the solver.
   ~IsoLinElasticSolver();

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

   /// Set material
   void SetMaterial(mfem::Coefficient &E_, mfem::Coefficient &nu_)
   {
      E = &E_;
      nu = &nu_;

      delete lambda;
      delete mu;
      delete bf; bf=nullptr;
      dop.release();

      lambda = new IsoElasticyLambdaCoeff(E, nu);
      mu = new IsoElasticySchearCoeff(E, nu);
   }

   class NqptUniformParameterSpace : public
      mfem::future::UniformParameterSpace
   {
   public:
      NqptUniformParameterSpace(mfem::ParMesh &mesh,
                                const mfem::IntegrationRule &ir,
                                int vdim) :
         mfem::future::UniformParameterSpace(mesh, ir, vdim, false)
      {
         dtq.nqpt = ir.GetNPoints();
      }
   };

   // creates a list with essential dofs
   // sets the values in the bsol vector
   // the list is written in ess_dofs
   // The 'nvcc' compiler needs these SetEssTDofs functions to be public.
   void SetEssTDofs(mfem::Vector &bsol, mfem::Array<int> &ess_dofs);
   void SetEssTDofs(const int j, mfem::ParFiniteElementSpace& scalar_space,
                    mfem::Array<int> &ess_dofs);
private:
   mfem::ParMesh *pmesh;
   const bool pa, dfem; // partial assembly, dFEM operator
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

   mfem::HypreBoomerAMG *prec; // preconditioner
   mfem::CGSolver *ls;         // linear solver

   // PA LOR preconditioner
   mfem::Array<int> lor_block_offsets;
   std::unique_ptr<mfem::Solver> lor_pa_prec;
   std::unique_ptr<mfem::ParLORDiscretization> lor_disc;
   std::unique_ptr<mfem::ElasticityIntegrator> lor_integrator;
   std::unique_ptr<mfem::ParFiniteElementSpace> lor_scalar_fespace;
   std::unique_ptr<mfem::BlockDiagonalPreconditioner> lor_blockDiag;
   std::vector<std::unique_ptr<mfem::ParBilinearForm>> lor_bilinear_forms;
   std::vector<std::unique_ptr<mfem::HypreParMatrix>> lor_block;
   std::vector<std::unique_ptr<mfem::HypreBoomerAMG>> lor_amg_blocks;

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

   mfem::Coefficient *E, *nu;
   mfem::Coefficient *lambda, *mu;

   mfem::ParBilinearForm *bf;
   mfem::ConstrainedOperator *Kc;
   std::unique_ptr<mfem::OperatorHandle> Kh;
   std::unique_ptr<mfem::HypreParMatrix> K, Ke;

   // begining of dFEM defintions
   // U - displacements, Coords - nodal coordinates
   // E modulud sampled on integration points
   // Nu Poisson's ratio sampled on integration points
   static constexpr int U = 0, Coords = 1, LCoeff = 2, MuCoeff = 3;
   const mfem::FiniteElement *fe;
   mfem::ParGridFunction *nodes;
   mfem::ParFiniteElementSpace *mfes;
   mfem::Array<int> domain_attributes;
   const mfem::IntegrationRule &ir;
   mfem::QuadratureSpace qs;
   NqptUniformParameterSpace Lambda_ps, Mu_ps;
   std::unique_ptr<mfem::CoefficientVector> Lambda_cv, Mu_cv;
   std::unique_ptr<mfem::future::DifferentiableOperator> dop;
   // end of dFEM definitions

   mfem::ParLinearForm *lf;

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

class PDEFilter : public mfem::Operator
{
public:
   /// Construct the PDE filter for a given mesh and discretization
   /// order.
   PDEFilter(mfem::ParMesh *mesh, real_t r = 1.0, int order = 1);

   /// Construct the PDE filter for a given input finite element space,
   /// discretization order and filter radius.
   PDEFilter(mfem::ParFiniteElementSpace *fespace, real_t r = 1.0, int order = 1); 
   
   /// Destructor of the filter.
   virtual ~PDEFilter();

   /// Set the filter radius
   void SetFilterRadius(real_t r);

   /// Set the linear solver relative tolerance (rtol),
   /// absolute tolerance (atol) and maximum number of
   /// iterations miter.
   void SetLinearSolver(real_t rtol = 1e-8,
                        real_t atol = 1e-12,
                        int miter = 200);

   /// Assemble the filter operator
   void Assemble();

   /// Apply the filter to input x, output y.
   virtual
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Adjoint apply the filter to input x, output y.
   virtual
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Returns the filtered field.
   mfem::ParGridFunction &GetFilteredField()
   {
      return filtered_field;
   }

   /// Returns the filtered finite element space
   mfem::ParFiniteElementSpace &GetFilteredFESpace()
   {
      return *ffes;
   }

   /// Returns the input finite element space
   mfem::ParFiniteElementSpace &GetInputFESpace()
   {
      return *ifes;
   }

   class NqptUniformParameterSpace : public
      mfem::future::UniformParameterSpace
   {
   public:
      NqptUniformParameterSpace(mfem::ParMesh &mesh,
                                const mfem::IntegrationRule &ir,
                                int vdim) :
         mfem::future::UniformParameterSpace(mesh, ir, vdim, false)
      {
         dtq.nqpt = ir.GetNPoints();
      }
   };

private:
   mfem::ParMesh *pmesh;
   const int dim; 
   const int spaceDim;

   real_t filter_radius;

   mfem::FiniteElementCollection *ffec;
   mfem::ParFiniteElementSpace *ffes;

   mfem::FiniteElementCollection *ifec;
   mfem::ParFiniteElementSpace *ifes;

   mutable mfem::ParGridFunction filtered_field;
   mutable mfem::ParGridFunction input_field;
   mutable mfem::Vector rhs;

   // H1 gradient of the input field: for more info see the paper
   // "A Simple Introduciton to the SiMPL-method for density-based topology optimization"
   // by D. Kim, B. Lazarov, T. Surowiec, B. Keith, 
   // Structural and Multidisciplinary Optimization, 2025, 68.
   mutable mfem::ParGridFunction h1_gradient;

   // Linear solver parameters
   real_t linear_rtol;
   real_t linear_atol;
   int linear_iter;

   mfem::HypreBoomerAMG *prec; // preconditioner
   mfem::CGSolver *ls;         // linear solver

   // holds the displacement contrained DOFs
   mfem::Array<int> ess_tdofv;

   // boundary conditions
   std::map<int, mfem::ConstantCoefficient> bcr;

   // DFEM related definitions
   std::unique_ptr<mfem::future::DifferentiableOperator> dop;
   // RHS of the PDE filter
   std::unique_ptr<mfem::future::DifferentiableOperator> drh;
   const mfem::FiniteElement *fe;
   mfem::ParGridFunction *nodes;
   mfem::ParFiniteElementSpace *mfes;
   const mfem::IntegrationRule &ir;
   mfem::QuadratureSpace qs;
   mfem::Array<int> domain_attributes;
   NqptUniformParameterSpace diff_ps;
   std::unique_ptr<mfem::CoefficientVector> diff_cv;
   // Fsol - filtered solution, USol - unfiltered solution
   // Coords - nodal coordinates
   static constexpr int FSol = 0, USol=1, Coords = 2, DiffCoeff = 3;

   mfem::HypreParMatrix *K;
   mfem::ConstrainedOperator *Kc;
   std::unique_ptr<mfem::OperatorHandle> Kh;
   // RHS operators
   mfem::ConstrainedOperator *Rc;
   std::unique_ptr<mfem::OperatorHandle> Rh;
};


// A dFEM-only version of the mtop IsoLinElasticSolver:
// - removes partial assembly option
// - removes full/classical assembly option
// - always builds the operator using mfem::future::DifferentiableOperator

class DFEMLinElasticSolver : public mfem::Operator
{
public:
   DFEMLinElasticSolver(mfem::ParMesh *mesh, int vorder = 1);
   ~DFEMLinElasticSolver();

   void SetLinearSolver(mfem::real_t rtol = 1e-8,
                        mfem::real_t atol = 1e-12,
                        int miter = 200);


   // Volumetric force
   void SetVolForce(mfem::real_t fx, mfem::real_t fy, mfem::real_t fz = 0.0);
   void SetVolForce(mfem::VectorCoefficient &ff);

   // Displacement BCs
   void AddDispBC(int bdr_attr, int dir, mfem::real_t val);
   void AddDispBC(int bdr_attr, int dir, mfem::Coefficient &val);
   void DelDispBC();

   // Surface loads (same interface pattern as mtop)
   void AddSurfLoad(int bdr_attr, mfem::real_t fx, mfem::real_t fy, mfem::real_t fz = 0.0);
   void AddSurfLoad(int bdr_attr, mfem::VectorCoefficient &ff);

   // Build operator/preconditioner (call after setting material and BCs).
   void Assemble();

   // Solve Ku = f (forward)
   void FSolve();

   // Operator interface: y = K^{-1} x (solve with x as RHS)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;
   void MultTranspose(const mfem::Vector &x, mfem::Vector &y) const override;

   mfem::ParGridFunction &GetDisplacements(){
         fdisp.SetFromTrueDofs(sol); return fdisp;
   }

   mfem::Vector          &GetSolutionVector() { return sol; }

private:
   // Essential dofs helper
   void SetEssTDofs(mfem::Vector &bsol, mfem::Array<int> &ess_dofs) const;
  

   mfem::ParMesh *pmesh = nullptr;
   int dim = 0;
   int spaceDim = 0;

   std::unique_ptr<mfem::FiniteElementCollection> vfec;
   std::unique_ptr<mfem::ParFiniteElementSpace>   vfes;

   // State solutions
   mfem::ParGridFunction fdisp;
   mfem::Vector sol, rhs;

   // Body force
   mfem::Vector vol_force_vec;
   std::unique_ptr<mfem::VectorCoefficient> volforce_owned;
   mfem::VectorCoefficient *volforce = nullptr;


   // Dirichlet BC storage (attr -> per-direction coeff)
   struct DispBC
   {
      int dir = -1; // 0,1,2 or -1(all)
      mfem::Coefficient *coeff = nullptr; // may be owned or external
      bool owned = false;
   };

   std::multimap<int, DispBC> disp_bcs;

   mfem::Array<int> ess_tdofv;

   // dFEM operator objects (future::DifferentiableOperator path)
   static constexpr int U = 0, Coords = 1, LCoeff = 2, MuCoeff = 3;

   // quadrature / coefficient sampling
   mfem::QuadratureSpace *qs = nullptr;
   std::unique_ptr<mfem::CoefficientVector> Lambda_cv;
   std::unique_ptr<mfem::CoefficientVector> Mu_cv;

   // The differentiable operator itself
   std::unique_ptr<mfem::future::DifferentiableOperator> dop;

   // Constrained operator handle (for eliminate RHS)
   std::unique_ptr<mfem::OperatorHandle> Kh;
   mfem::ConstrainedOperator *Kc = nullptr;

   // Linear solver + preconditioner
   mfem::CGSolver *ls = nullptr;

   // LOR preconditioner 
   std::unique_ptr<mfem::ParLORDiscretization> lor_disc;
   std::unique_ptr<mfem::ParFiniteElementSpace> lor_scalar_fespace;

   std::unique_ptr<mfem::HypreParMatrix>  lor_mat;
   std::unique_ptr<mfem::HypreBoomerAMG>  lor_amg;

   // Solver params
   mfem::real_t linear_rtol = 1e-8;
   mfem::real_t linear_atol = 1e-12;
   int linear_iter = 200;
};
