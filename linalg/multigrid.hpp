// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_MULTIGRID
#define MFEM_MULTIGRID

#include "../general/tic_toc.hpp"
#include "operator.hpp"
#include <map>
#include <tuple>

namespace mfem
{

/// Abstract multigrid operator
class MultigridOperator : public Operator
{
protected:
   Array<Operator*> operators;
   Array<Solver*> smoothers;
   Array<const Operator*> prolongations;

   Array<bool> ownedOperators;
   Array<bool> ownedSmoothers;
   Array<bool> ownedProlongations;

public:
   /// Empty constructor
   MultigridOperator();

   /// Constructor adding the coarse operator and solver. See AddCoarseLevel for
   /// a description.
   MultigridOperator(Operator* opr, Solver* coarseSolver, bool ownOperator,
                     bool ownSolver);

   /// Destructor
   virtual ~MultigridOperator();

   /// This method adds the first coarse grid level to the hierarchy. Only after
   /// the call to this method additional levels may be added with AddLevel. The
   /// ownership of the operators and solvers may be transferred to the
   /// MultigridOperator by setting the according boolean variables.
   void AddCoarsestLevel(Operator* opr, Solver* solver, bool ownOperator,
                         bool ownSolver);

   /// Adds a level to the multigrid operator hierarchy. This method may only be
   /// called after the coarse level has been added. The ownership of the
   /// operators and solvers may be transferred to the MultigridOperator by
   /// setting the according boolean variables.
   void AddLevel(Operator* opr, Solver* smoother, const Operator* prolongation,
                 bool ownOperator, bool ownSmoother, bool ownProlongation);

   /// Returns the number of levels
   int NumLevels() const;

   /// Returns the index of the finest level
   int GetFinestLevelIndex() const;

   /// Matrix vector multiplication at given level
   virtual void MultAtLevel(int level, const Vector& x, Vector& y) const;

   /// Matrix vector multiplication with the operator at the finest level
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Restrict vector \p x from \p level + 1 to \p level. This method uses the
   /// transposed of the interpolation
   virtual void RestrictTo(int level, const Vector& x, Vector& y) const;

   /// Interpolate vector \p x from \p level to \p level + 1
   virtual void InterpolateFrom(int level, const Vector& x, Vector& y) const;

   /// Apply Smoother at given level
   virtual void ApplySmootherAtLevel(int level, const Vector& x,
                                     Vector& y) const;

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
};

/// Abstract multigrid operator with timing support
class TimedMultigridOperator : public MultigridOperator
{
public:
   enum class Operation
   {
      OPERATOR,
      PROLONGATION,
      RESTRICTION,
      SMOOTHER
   };

   enum class Statistics
   {
      NUMAPPLICATIONS,
      TOTALTIME
   };

private:
   mutable StopWatch sw;
   mutable std::map<std::tuple<Statistics, Operation, int>, double> stats;

public:
   /// See documentation of MultigridOperator
   TimedMultigridOperator();
   TimedMultigridOperator(Operator* opr, Solver* coarseSolver, bool ownOperator,
                          bool ownSolver);
   ~TimedMultigridOperator();

   void MultAtLevel(int level, const Vector& x, Vector& y) const override;
   void RestrictTo(int level, const Vector& x, Vector& y) const override;
   void InterpolateFrom(int level, const Vector& x, Vector& y) const override;
   void ApplySmootherAtLevel(int level, const Vector& x,
                             Vector& y) const override;

   /// Writes the collected staticistics for a specific \p operation to \p out
   void PrintStats(Operation operation, std::ostream& out) const;
};

// Multigrid solver
class MultigridSolver : public Solver
{
public:
   enum class CycleType
   {
      VCYCLE,
      WCYCLE
   };

private:
   const MultigridOperator* opr;
   CycleType cycleType;

   mutable Array<int> preSmoothingSteps;
   mutable Array<int> postSmoothingSteps;

   mutable Array<Vector*> X;
   mutable Array<Vector*> Y;
   mutable Array<Vector*> R;
   mutable Array<Vector*> Z;

public:
   /// Constructs the multigrid solver and helper vectors
   MultigridSolver(const MultigridOperator* opr_,
                   CycleType cycleType_ = CycleType::VCYCLE,
                   int preSmoothingSteps_ = 3, int postSmoothingSteps_ = 3);

   /// Destructor deleting the allocated memory of helper vectors
   ~MultigridSolver();

   /// Set the cycle type used by Mult
   void SetCycleType(CycleType cycleType_);

   /// Set the number of pre-smoothing steps on all levels, excluding the coarse
   /// grid
   void SetPreSmoothingSteps(int steps);

   /// Set the number of pre-smoothing steps on all levels, excluding the coarse
   /// grid
   void SetPreSmoothingSteps(const Array<int>& steps);

   /// Set the number of post-smoothing steps on all levels, excluding the
   /// coarse grid
   void SetPostSmoothingSteps(int steps);

   /// Set the number of post-smoothing steps on all levels, excluding the
   /// coarse grid
   void SetPostSmoothingSteps(const Array<int>& steps);

   /// Set the number of pre- and post-smoothing steps on all levels, excluding
   /// the coarse grid
   void SetSmoothingSteps(int steps);

   /// Set the number of pre- and post-smoothing steps on all levels, excluding
   /// the coarse grid. Only the pre-smoothing
   void SetSmoothingSteps(const Array<int>& steps);

   /// Application of a cycle
   virtual void Mult(const Vector& x, Vector& y) const override;

   /// Set/update the solver for the given operator. The operator must be of
   /// type MultigridOperator
   virtual void SetOperator(const Operator& op) override;

private:
   /// Application of a smoothing step at particular level
   void SmoothingStep(int level) const;

   /// Application of a cycle at particular level
   void Cycle(int level) const;

   /// Setup used by constructor and SetOperator which allocates memory for
   /// helper vectors on coarser levels
   void Setup(int preSmoothingSteps_ = 3, int postSmoothingSteps_ = 3);

   /// Frees the allocated memory
   void Reset();
};

} // namespace mfem

#endif
