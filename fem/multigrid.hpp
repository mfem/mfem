// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

/// Multigrid solver class
class Multigrid : public Solver
{
public:
   enum class CycleType
   {
      VCYCLE,
      WCYCLE
   };

protected:
   const FiniteElementSpaceHierarchy& fespaces;
   Array<Array<int>*> essentialTrueDofs;
   Array<BilinearForm*> bfs;

private:
   Array<Operator*> operators;
   Array<Solver*> smoothers;

   Array<bool> ownedOperators;
   Array<bool> ownedSmoothers;

   CycleType cycleType;
   int preSmoothingSteps;
   int postSmoothingSteps;

   mutable Array<Vector*> X;
   mutable Array<Vector*> Y;
   mutable Array<Vector*> R;
   mutable Array<Vector*> Z;

public:
   /// Constructs an empty multigrid for the given FiniteElementSpaceHierarchy
   Multigrid(const FiniteElementSpaceHierarchy& fespaces_);

   /// Destructor
   virtual ~Multigrid();

   /// Adds a level to the multigrid operator hierarchy.
   /** The ownership of the operators and solvers/smoothers may be transferred
       to the Multigrid by setting the according boolean variables. */
   void AddLevel(Operator* opr, Solver* smoother, bool ownOperator,
                 bool ownSmoother);

   /// Returns the number of levels
   int NumLevels() const;

   /// Returns the index of the finest level
   int GetFinestLevelIndex() const;

   /// Returns operator at given level
   const Operator* GetOperatorAtLevel(int level) const;

   /// Returns operator at given level
   Operator* GetOperatorAtLevel(int level);

   /// Returns operator at finest level
   const Operator* GetOperatorAtFinestLevel() const;

   /// Returns operator at finest level
   Operator* GetOperatorAtFinestLevel();

   /// Returns smoother at given level
   Solver* GetSmootherAtLevel(int level) const;

   /// Returns smoother at given level
   Solver* GetSmootherAtLevel(int level);

   /// Set the cycle type and number of pre- and post-smoothing steps used by Mult
   void SetCycleType(CycleType cycleType_, int preSmoothingSteps_,
                     int postSmoothingSteps_);

   /// Application of the multigrid as a preconditioner
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Not supported for multigrid
   virtual void SetOperator(const Operator& op) override;

   /// Form the linear system A X = B, corresponding to the operator on the finest level
   void FormFineLinearSystem(Vector& x, Vector& b, OperatorHandle& A, Vector& X,
                             Vector& B);

   /// Recover the solution of a linear system formed with FormFineLinearSystem()
   void RecoverFineFEMSolution(const Vector& X, const Vector& b, Vector& x);

private:
   /// Application of a smoothing step at particular level
   void SmoothingStep(int level) const;

   /// Application of a cycle at particular level
   void Cycle(int level) const;
};

} // namespace mfem

#endif
