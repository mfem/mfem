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

#ifndef MFEM_LOR
#define MFEM_LOR

#include "bilinearform.hpp"

namespace mfem
{

/// @brief Abstract base class for LORDiscretization and ParLORDiscretization
/// classes, which construct low-order refined versions of bilinear forms.
class LORBase
{
private:
   using GetIntegratorsFn = Array<BilinearFormIntegrator*> *(BilinearForm::*)();
   using GetMarkersFn = Array<Array<int>*> *(BilinearForm::*)();
   using AddIntegratorFn = void (BilinearForm::*)(BilinearFormIntegrator*);
   using AddIntegratorMarkersFn =
      void (BilinearForm::*)(BilinearFormIntegrator*, Array<int>&);

   IntegrationRules irs;
   const IntegrationRule *ir_el, *ir_face;
   std::map<BilinearFormIntegrator*, const IntegrationRule*> ir_map;

   /// Adds all the integrators from the BilinearForm @a a_from to @a a_to. If
   /// the mesh consists of tensor product elements, temporarily changes the
   /// integration rules of the integrators to use collocated quadrature for
   /// better conditioning of the LOR system.
   void AddIntegrators(BilinearForm &a_from,
                       BilinearForm &a_to,
                       GetIntegratorsFn get_integrators,
                       AddIntegratorFn add_integrator,
                       const IntegrationRule *ir);

   /// Adds all the integrators from the BilinearForm @a a_from to @a a_to,
   /// using the same marker lists (e.g. for integrators only applied to
   /// certain boundary attributes). @sa LORBase::AddIntegrators
   void AddIntegratorsAndMarkers(BilinearForm &a_from,
                                 BilinearForm &a_to,
                                 GetIntegratorsFn get_integrators,
                                 GetMarkersFn get_markers,
                                 AddIntegratorMarkersFn add_integrator_marker,
                                 AddIntegratorFn add_integrator,
                                 const IntegrationRule *ir);

   /// Resets the integration rules of the integrators of @a a to their original
   /// values (after temporarily changing them for LOR assembly).
   void ResetIntegrationRules(GetIntegratorsFn get_integrators);

   static inline int absdof(int i) { return i < 0 ? -1-i : i; }

protected:
   enum FESpaceType { H1, ND, RT, L2, INVALID };

   FiniteElementSpace &fes_ho;
   Mesh *mesh;
   FiniteElementCollection *fec;
   FiniteElementSpace *fes;
   BilinearForm *a;
   OperatorHandle A;
   mutable Array<int> perm;

   /// Constructs the local DOF (ldof) permutation. In parallel this is used as
   /// an intermediate step in computing the DOF permutation (see
   /// ConstructDofPermutation and GetDofPermutation).
   void ConstructLocalDofPermutation(Array<int> &perm_) const;

   /// Construct the permutation that maps LOR DOFs to high-order DOFs. See
   /// GetDofPermutation.
   void ConstructDofPermutation() const;

   /// Returns true if the LOR space and HO space have the same DOF numbering
   /// (H1 or L2 spaces), false otherwise (ND or RT spaces).
   bool HasSameDofNumbering() const;

   /// Sets up the prolongation and restriction operators required in the case
   /// of different DOF numberings (ND or RT spaces) or nonconforming spaces.
   void SetupProlongationAndRestriction();

   /// Returns the type of finite element space: H1, ND, RT or L2.
   FESpaceType GetFESpaceType() const;

   /// Returns the order of the LOR space. 1 for H1 or ND, 0 for L2 or RT.
   int GetLOROrder() const;

   /// Assembles the LOR system (used internally by
   /// LORDiscretization::AssembleSystem and
   /// ParLORDiscretization::AssembleSystem).
   void AssembleSystem_(BilinearForm &a_ho, const Array<int> &ess_dofs);

   LORBase(FiniteElementSpace &fes_ho_);

public:
   /// Returns the assembled LOR system.
   const OperatorHandle &GetAssembledSystem() const;

   /// @brief Returns the permutation that maps LOR DOFs to high-order DOFs.
   ///
   /// This permutation is constructed the first time it is requested, and then
   /// is cached. For H1 and L2 finite element spaces (or for nonconforming
   /// spaces) this is the identity. In these cases, RequiresDofPermutation will
   /// return false. However, if the DOF permutation is requested, an identity
   /// permutation will be built and returned.
   ///
   /// For vector finite element spaces (ND and RT), the DOF permutation is
   /// nontrivial. Returns an array @a perm such that, given an index @a i of a
   /// LOR dof, @a perm[i] is the index of the corresponding HO dof.
   const Array<int> &GetDofPermutation() const;

   /// Returns the low-order refined finite element space.
   FiniteElementSpace &GetFESpace() const { return *fes; }

   ~LORBase();
};

/// Create and assemble a low-order refined version of a BilinearForm.
class LORDiscretization : public LORBase
{
public:
   /// @brief Construct the low-order refined version of @a a_ho using the given
   /// list of essential DOFs.
   ///
   /// The mesh is refined using the refinement type specified by @a ref_type
   /// (see Mesh::MakeRefined).
   LORDiscretization(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
                     int ref_type=BasisType::GaussLobatto);

   /// @brief Construct a low-order refined version of the FiniteElementSpace @a
   /// fes_ho.
   ///
   /// The mesh is refined using the refinement type specified by @a ref_type
   /// (see Mesh::MakeRefined).
   LORDiscretization(FiniteElementSpace &fes_ho,
                     int ref_type=BasisType::GaussLobatto);

   /// Assembles the LOR system corresponding to @a a_ho.
   void AssembleSystem(BilinearForm &a_ho, const Array<int> &ess_dofs);

   /// Return the assembled LOR operator as a SparseMatrix.
   SparseMatrix &GetAssembledMatrix() const;
};

#ifdef MFEM_USE_MPI

/// Create and assemble a low-order refined version of a ParBilinearForm.
class ParLORDiscretization : public LORBase
{
public:
   /// @brief Construct the low-order refined version of @a a_ho using the given
   /// list of essential DOFs.
   ///
   /// The mesh is refined using the refinement type specified by @a ref_type
   /// (see ParMesh::MakeRefined).
   ParLORDiscretization(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
                        int ref_type=BasisType::GaussLobatto);

   /// @brief Construct a low-order refined version of the ParFiniteElementSpace
   /// @a pfes_ho.
   ///
   /// The mesh is refined using the refinement type specified by @a ref_type
   /// (see ParMesh::MakeRefined).
   ParLORDiscretization(ParFiniteElementSpace &fes_ho,
                        int ref_type=BasisType::GaussLobatto);

   /// Assembles the LOR system corresponding to @a a_ho.
   void AssembleSystem(ParBilinearForm &a_ho, const Array<int> &ess_dofs);

   /// Return the assembled LOR operator as a HypreParMatrix.
   HypreParMatrix &GetAssembledMatrix() const;

   /// Return the LOR ParFiniteElementSpace.
   ParFiniteElementSpace &GetParFESpace() const;
};

#endif

/// @brief Represents a solver of type @a SolverType created using the low-order
/// refined version of the given BilinearForm or ParBilinearForm.
///
/// @note To achieve good solver performance, the high-order finite element
/// space should use BasisType::GaussLobatto for H1 discretizations, and basis
/// pair (BasisType::GaussLobatto, BasisType::IntegratedGLL) for Nedelec and
/// Raviart-Thomas elements.
template <typename SolverType>
class LORSolver : public Solver
{
protected:
   LORBase *lor;
   bool own_lor = true;
   SolverType solver;
   mutable Vector px, py;
public:
   /// @brief Create a solver of type @a SolverType, formed using the assembled
   /// SparseMatrix of the LOR version of @a a_ho. @see LORDiscretization
   LORSolver(BilinearForm &a_ho, const Array<int> &ess_tdof_list,
             int ref_type=BasisType::GaussLobatto)
   {
      lor = new LORDiscretization(a_ho, ess_tdof_list, ref_type);
      SetOperator(*lor->GetAssembledSystem());
   }

#ifdef MFEM_USE_MPI
   /// @brief Create a solver of type @a SolverType, formed using the assembled
   /// HypreParMatrix of the LOR version of @a a_ho. @see ParLORDiscretization
   LORSolver(ParBilinearForm &a_ho, const Array<int> &ess_tdof_list,
             int ref_type=BasisType::GaussLobatto)
   {
      lor = new ParLORDiscretization(a_ho, ess_tdof_list, ref_type);
      SetOperator(*lor->GetAssembledSystem());
   }
#endif

   /// @brief Create a solver of type @a SolverType using Operator @a op and
   /// arguments @a args.
   template <typename... Args>
   LORSolver(const Operator &op, LORBase &lor_, Args&&... args) : solver(args...)
   {
      lor = &lor_;
      own_lor = false;
      SetOperator(op);
   }

   /// @brief Create a solver of type @a SolverType using the assembled LOR
   /// operator represented by @a lor_.
   ///
   /// The given @a args will be used as arguments to the solver constructor.
   template <typename... Args>
   LORSolver(LORBase &lor_, Args&&... args)
      : LORSolver(*lor_.GetAssembledSystem(), lor_, args...) { }

   void SetOperator(const Operator &op)
   {
      solver.SetOperator(op);
      width = solver.Width();
      height = solver.Height();
   }

   void Mult(const Vector &x, Vector &y) const { solver.Mult(x, y); }

   /// Access the underlying solver.
   SolverType &GetSolver() { return solver; }

   /// Access the underlying solver.
   const SolverType &GetSolver() const { return solver; }

   /// Access the LOR discretization object.
   const LORBase &GetLOR() const { return *lor; }

   ~LORSolver() { if (own_lor) { delete lor; } }
};

} // namespace mfem

#endif
