// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MULTIGRID
#define MFEM_MULTIGRID

#include "fespacehierarchy.hpp"
#include "bilinearform.hpp"

#include "../linalg/operator.hpp"
#include "../linalg/handle.hpp"

namespace mfem
{

/// Abstract base class for Multigrid solvers
class MultigridBase : public Solver
{
public:
   enum class CycleType
   {
      VCYCLE,
      WCYCLE
   };

protected:
   Array<Operator*> operators;
   Array<Solver*> smoothers;
   Array<bool> ownedOperators;
   Array<bool> ownedSmoothers;

   CycleType cycleType;
   int preSmoothingSteps;
   int postSmoothingSteps;

   mutable Array2D<Vector*> X, Y, R, Z;
   mutable int nrhs;

public:
   /// Constructs an empty multigrid hierarchy
   MultigridBase();

   /// Constructs a multigrid hierarchy from the given inputs
   /** Inputs include operators and smoothers on all levels, and ownership of
       the given operators and smoothers */
   MultigridBase(const Array<Operator*>& operators_,
                 const Array<Solver*>& smoothers_,
                 const Array<bool>& ownedOperators_,
                 const Array<bool>& ownedSmoothers_);

   /// Destructor
   virtual ~MultigridBase();

   /// Adds a level to the multigrid operator hierarchy
   /** The ownership of the operators and solvers/smoothers may be transferred
       to the Multigrid by setting the according boolean variables */
   void AddLevel(Operator* op, Solver* smoother, bool ownOperator,
                 bool ownSmoother);

   /// Returns the number of levels
   int NumLevels() const { return operators.Size(); }

   /// Returns the index of the finest level
   int GetFinestLevelIndex() const { return NumLevels() - 1; }

   /// Returns operator at given level
   const Operator* GetOperatorAtLevel(int level) const
   {
      return operators[level];
   }
   Operator* GetOperatorAtLevel(int level)
   {
      return operators[level];
   }

   /// Returns operator at finest level
   const Operator* GetOperatorAtFinestLevel() const
   {
      return GetOperatorAtLevel(GetFinestLevelIndex());
   }
   Operator* GetOperatorAtFinestLevel()
   {
      return GetOperatorAtLevel(GetFinestLevelIndex());
   }

   /// Returns smoother at given level
   const Solver* GetSmootherAtLevel(int level) const
   {
      return smoothers[level];
   }
   Solver* GetSmootherAtLevel(int level)
   {
      return smoothers[level];
   }

   /// Set cycle type and number of pre- and post-smoothing steps used by Mult
   void SetCycleType(CycleType cycleType_, int preSmoothingSteps_,
                     int postSmoothingSteps_);

   /// Application of the multigrid as a preconditioner
   void Mult(const Vector& x, Vector& y) const override;
   void ArrayMult(const Array<const Vector*>& X_,
                  Array<Vector*>& Y_) const override;

   /// Not supported for multigrid
   void SetOperator(const Operator& op) override
   {
      MFEM_ABORT("SetOperator is not supported in Multigrid!");
   }

private:
   /// Application of a multigrid cycle at particular level
   void Cycle(int level) const;

   /// Application of a pre-/post-smoothing step at particular level
   void SmoothingStep(int level, bool zero, bool transpose) const;

   /// Allocate or destroy temporary storage
   void InitVectors() const;
   void EraseVectors() const;

   /// Returns prolongation operator at given level
   virtual const Operator* GetProlongationAtLevel(int level) const = 0;
};

/// Multigrid solver class
class Multigrid : public MultigridBase
{
protected:
   Array<Operator*> prolongations;
   Array<bool> ownedProlongations;

public:
   /// Constructs an empty multigrid hierarchy
   Multigrid() = default;

   /// Constructs a multigrid hierarchy from the given inputs
   /** Inputs include operators and smoothers on all levels, prolongation
       operators that go from coarser to finer levels, and ownership of the
       given operators, smoothers, and prolongations */
   Multigrid(const Array<Operator*>& operators_, const Array<Solver*>& smoothers_,
             const Array<Operator*>& prolongations_, const Array<bool>& ownedOperators_,
             const Array<bool>& ownedSmoothers_, const Array<bool>& ownedProlongations_);

   /// Destructor
   virtual ~Multigrid();

private:
   /// Returns prolongation operator at given level
   const Operator* GetProlongationAtLevel(int level) const override
   {
      return prolongations[level];
   }
};

/// Geometric multigrid associated with a hierarchy of finite element spaces
class GeometricMultigrid : public Multigrid
{
protected:
   const FiniteElementSpaceHierarchy& fespaces;
   Array<Array<int>*> essentialTrueDofs;
   Array<BilinearForm*> bfs;

public:
   /// @brief Deprecated.
   ///
   /// Construct an empty geometric multigrid object for the given finite
   /// element space hierarchy @a fespaces_.
   ///
   /// @deprecated Use GeometricMultigrid::GeometricMultigrid(const
   /// FiniteElementSpaceHierarchy&, const Array<int>&) instead. This version
   /// constructs prolongation and restriction operators without eliminated
   /// essential boundary conditions.
   MFEM_DEPRECATED
   GeometricMultigrid(const FiniteElementSpaceHierarchy& fespaces_);

   /// @brief Construct a geometric multigrid object for the given finite
   /// element space hierarchy @a fespaces_, where @a ess_bdr is a list of
   /// mesh boundary element attributes that define the essential DOFs.
   ///
   /// If @a ess_bdr is empty, or all its entries are 0, then no essential
   /// boundary conditions are imposed and the protected array essentialTrueDofs
   /// remains empty.
   GeometricMultigrid(const FiniteElementSpaceHierarchy& fespaces_,
                      const Array<int> &ess_bdr);

   /// Destructor
   virtual ~GeometricMultigrid();

   /** Form the linear system A X = B, corresponding to the operator on the
       finest level of the geometric multigrid hierarchy */
   void FormFineLinearSystem(Vector& x, Vector& b, OperatorHandle& A, Vector& X,
                             Vector& B);

   /// Recover the solution of a linear system formed with FormFineLinearSystem()
   void RecoverFineFEMSolution(const Vector& X, const Vector& b, Vector& x);
};

} // namespace mfem

#endif
