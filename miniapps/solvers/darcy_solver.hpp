// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DARCY_SOLVER_HPP
#define MFEM_DARCY_SOLVER_HPP

#include "mfem.hpp"
#include <memory>
#include <vector>

namespace mfem
{
namespace blocksolvers
{

// Exact solution, u and p, and r.h.s., f and g.
void u_exact(const Vector & x, Vector & u);
double p_exact(const Vector & x);
void f_exact(const Vector & x, Vector & f);
double g_exact(const Vector & x);
double natural_bc(const Vector & x);

/// Check if using Neumann BC
bool IsAllNeumannBoundary(const Array<int>& ess_bdr_attr);

/// Parameters for iterative solver
struct IterSolveParameters
{
   int print_level = 0;
   int max_iter = 500;
   double abs_tol = 1e-12;
   double rel_tol = 1e-9;
};

/// Set standard options for general solvers
void SetOptions(IterativeSolver& solver, const IterSolveParameters& param);

SparseMatrix ElemToDof(const ParFiniteElementSpace& fes);

/// DFS classes and structs
/// Parameters for the divergence free solver
struct DFSParameters : IterSolveParameters
{
   /** There are three components in the solver: a particular solution
       satisfying the divergence constraint, the remaining div-free component of
       the flux, and the pressure. When coupled_solve == false, the three
       components will be solved one by one in the aforementioned order.
       Otherwise, they will be solved at the same time. */
   bool coupled_solve = false;
   bool verbose = false;
   IterSolveParameters coarse_solve_param;
   IterSolveParameters BBT_solve_param;
};

/// Data for the divergence free solver
struct DFSData
{
   std::vector<OperatorPtr> agg_hdivdof;  // agglomerates to H(div) dofs table
   std::vector<OperatorPtr> agg_l2dof;    // agglomerates to L2 dofs table
   std::vector<OperatorPtr> P_hdiv;   // Interpolation matrix for H(div) space
   std::vector<OperatorPtr> P_l2;     // Interpolation matrix for L2 space
   std::vector<OperatorPtr> P_hcurl;  // Interpolation for kernel space of div
   std::vector<OperatorPtr> Q_l2;     // Q_l2[l] = (W_{l+1})^{-1} P_l2[l]^T W_l
   Array<int> coarsest_ess_hdivdofs;  // coarsest level essential H(div) dofs
   std::vector<OperatorPtr> C;        // discrete curl: ND -> RT, map to Null(B)
   DFSParameters param;
};

/// Finite element spaces concerning divergence free solver.
/// The main usage of this class is to collect data needed for the solver.
class DFSSpaces
{
   RT_FECollection hdiv_fec_;
   L2_FECollection l2_fec_;
   std::unique_ptr<FiniteElementCollection> hcurl_fec_;
   L2_FECollection l2_0_fec_;

   std::unique_ptr<ParFiniteElementSpace> coarse_hdiv_fes_;
   std::unique_ptr<ParFiniteElementSpace> coarse_l2_fes_;
   std::unique_ptr<ParFiniteElementSpace> coarse_hcurl_fes_;
   std::unique_ptr<ParFiniteElementSpace> l2_0_fes_;

   std::unique_ptr<ParFiniteElementSpace> hdiv_fes_;
   std::unique_ptr<ParFiniteElementSpace> l2_fes_;
   std::unique_ptr<ParFiniteElementSpace> hcurl_fes_;

   std::vector<SparseMatrix> el_l2dof_;
   const Array<int>& ess_bdr_attr_;
   Array<int> all_bdr_attr_;

   int level_;
   DFSData data_;

   void MakeDofRelationTables(int level);
   void DataFinalize();
public:
   DFSSpaces(int order, int num_refine, ParMesh *mesh,
             const Array<int>& ess_attr, const DFSParameters& param);

   /** This should be called each time when the mesh (where the FE spaces are
       defined) is refined. The spaces will be updated, and the prolongation for
       the spaces and other data needed for the div-free solver are stored. */
   void CollectDFSData();

   const DFSData& GetDFSData() const { return data_; }
   ParFiniteElementSpace* GetHdivFES() const { return hdiv_fes_.get(); }
   ParFiniteElementSpace* GetL2FES() const { return l2_fes_.get(); }
};

/** Wrapper for assembling the discrete Darcy problem (ex5p)
                     [ M  B^T ] [u] = [f]
                     [ B   0  ] [p] = [g]
    where:
       M = \int_\Omega (k u_h) \cdot v_h dx,
       B = -\int_\Omega (div_h u_h) q_h dx,
       f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
       g = \int_\Omega g_exact q_h dx,
       u_h, v_h \in R_h (Raviart-Thomas finite element space),
       q_h \in W_h (piecewise discontinuous polynomials),
       D: subset of the boundary where natural boundary condition is imposed. */
class DarcyProblem
{
   OperatorPtr M_;
   OperatorPtr B_;
   Vector rhs_;
   Vector ess_data_;
   ParGridFunction u_;
   ParGridFunction p_;
   ParMesh mesh_;
   std::shared_ptr<ParBilinearForm> mVarf_;
   std::shared_ptr<ParMixedBilinearForm> bVarf_;
   VectorFunctionCoefficient ucoeff_;
   FunctionCoefficient pcoeff_;
   DFSSpaces dfs_spaces_;
   PWConstCoefficient mass_coeff;
   const IntegrationRule *irs_[Geometry::NumGeom];
public:
   DarcyProblem(Mesh &mesh, int num_refines, int order, const char *coef_file,
                Array<int> &ess_bdr, DFSParameters param);

   HypreParMatrix& GetM() { return *M_.As<HypreParMatrix>(); }
   HypreParMatrix& GetB() { return *B_.As<HypreParMatrix>(); }
   const Vector& GetRHS() { return rhs_; }
   const Vector& GetEssentialBC() { return ess_data_; }
   const DFSData& GetDFSData() const { return dfs_spaces_.GetDFSData(); }
   void ShowError(const Vector &sol, bool verbose);
   void VisualizeSolution(const Vector &sol, std::string tag);
   std::shared_ptr<ParBilinearForm> GetMform() const { return mVarf_; }
   std::shared_ptr<ParMixedBilinearForm> GetBform() const { return bVarf_; }
};

/// Abstract solver class for Darcy's flow
class DarcySolver : public Solver
{
protected:
   Array<int> offsets_;
public:
   DarcySolver(int size0, int size1) : Solver(size0 + size1), offsets_(3)
   { offsets_[0] = 0; offsets_[1] = size0; offsets_[2] = height; }
   virtual int GetNumIterations() const = 0;
};

/// Wrapper for the block-diagonal-preconditioned MINRES defined in ex5p.cpp
class BDPMinresSolver : public DarcySolver
{
   BlockOperator op_;
   BlockDiagonalPreconditioner prec_;
   OperatorPtr BT_;
   OperatorPtr S_;   // S_ = B diag(M)^{-1} B^T
   MINRESSolver solver_;
   Array<int> ess_zero_dofs_;
public:
   BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                   IterSolveParameters param);
   virtual void Mult(const Vector & x, Vector & y) const;
   virtual void SetOperator(const Operator &op) { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   virtual int GetNumIterations() const { return solver_.GetNumIterations(); }
};

} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_DARCY_SOLVER_HPP
