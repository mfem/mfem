#ifndef MFEM_MULTIGRID
#define MFEM_MULTIGRID

#include "../../general/forall.hpp"
#include "../../linalg/operator.hpp"

namespace mfem
{

/// Class bundling a hierarchy of meshes and finite element spaces
template<typename M, typename FES>
class GeneralSpaceHierarchy
{
private:
   Array<M*> meshes;
   Array<FES*> fespaces;
   Array<bool> ownedMeshes;
   Array<bool> ownedFES;

public:
   GeneralSpaceHierarchy(M* mesh, FES* fespace, bool ownM, bool ownFES)
   {
      AddLevel(mesh, fespace, ownM, ownFES);
   }

   ~GeneralSpaceHierarchy()
   {
      for (int i = meshes.Size() - 1; i >= 0; --i)
      {
         if (ownedFES[i]) { delete fespaces[i]; }
         if (ownedMeshes[i]) { delete meshes[i]; }
      }

      fespaces.DeleteAll();
      meshes.DeleteAll();
   }

   unsigned GetNumLevels() const
   {
      return meshes.Size();
   }

   unsigned GetFinestLevelIndex() const
   {
      return GetNumLevels() - 1;
   }

   void AddLevel(M* mesh, FES* fespace, bool ownM, bool ownFES)
   {
      meshes.Append(mesh);
      fespaces.Append(fespace);
      ownedMeshes.Append(ownM);
      ownedFES.Append(ownFES);
   }

   void AddUniformlyRefinedLevel()
   {
      M* mesh = new M(*meshes[GetFinestLevelIndex()]);
      mesh->UniformRefinement();
      FES* coarseFEspace = fespaces[GetFinestLevelIndex()];
      FES* fineFEspace = new FES(mesh, coarseFEspace->FEColl());
      AddLevel(mesh, fineFEspace, true, true);
   }

   void AddOrderRefinedLevel(FiniteElementCollection* fec)
   {
      M* mesh = &GetFinestMesh();
      FES* newFEspace = new FES(mesh, fec);
      AddLevel(mesh, newFEspace, false, true);
   }

   const M& GetMeshAtLevel(unsigned level) const
   {
      MFEM_ASSERT(level < meshes.Size(), "Mesh at given level does not exist.");
      return *meshes[level];
   }

   M& GetMeshAtLevel(unsigned level)
   {
      return const_cast<M&>(const_cast<const GeneralSpaceHierarchy*>(this)->GetMeshAtLevel(level));
   }

   const M& GetFinestMesh() const
   {
      return GetMeshAtLevel(GetFinestLevelIndex());
   }

   M& GetFinestMesh()
   {
      return const_cast<M&>(const_cast<const GeneralSpaceHierarchy*>(this)->GetFinestMesh());
   }

   const FES& GetFESpaceAtLevel(unsigned level) const
   {
      MFEM_ASSERT(level < fespaces.Size(), "FE space at given level does not exist.");
      return *fespaces[level];
   }

   FES& GetFESpaceAtLevel(unsigned level)
   {
      return const_cast<FES&>(const_cast<const GeneralSpaceHierarchy*>(this)->GetFESpaceAtLevel(level));
   }

   const FES& GetFinestFESpace() const
   {
      return GetFESpaceAtLevel(GetFinestLevelIndex());
   }

   FES& GetFinestFESpace()
   {
      return const_cast<FES&>(const_cast<const GeneralSpaceHierarchy*>(this)->GetFinestFESpace());
   }
};

using SpaceHierarchy = GeneralSpaceHierarchy<Mesh, FiniteElementSpace>;
using ParSpaceHierarchy = GeneralSpaceHierarchy<ParMesh, ParFiniteElementSpace>;

/// Abstract multigrid operator
class MultigridOperator : public Operator
{
protected:
   Array<Operator*> operators;
   Array<Solver*> smoothers;
   Array<Operator*> prolongations;

   Array<bool> ownedOperators;
   Array<bool> ownedSmoothers;
   Array<bool> ownedProlongations;

public:
   /// Constructor
   MultigridOperator(Operator* opr, Solver* coarseSolver, bool ownOperator, bool ownSolver)
   {
      operators.Append(opr);
      smoothers.Append(coarseSolver);
      ownedOperators.Append(ownOperator);
      ownedSmoothers.Append(ownSolver);
   }

   ~MultigridOperator()
   {
      for (int i = operators.Size() - 1; i >= 0; --i)
      {
         if (ownedOperators[i]) { delete operators[i]; }
         if (ownedSmoothers[i]) { delete smoothers[i]; }
      }

      for (int i = prolongations.Size() - 1; i >= 0; --i)
      {
         if (ownedProlongations[i]) { delete prolongations[i]; }
      }

      operators.DeleteAll();
      smoothers.DeleteAll();
   }

   virtual void AddLevel(Operator* opr, Solver* smoother, Operator* prolongation, bool ownOperator, bool ownSmoother, bool ownProlongation)
   {
      operators.Append(opr);
      smoothers.Append(smoother);
      prolongations.Append(prolongation);
      ownedOperators.Append(ownOperator);
      ownedSmoothers.Append(ownSmoother);
      ownedProlongations.Append(ownProlongation);
   }

   /// Returns the number of levels
   unsigned NumLevels() const
   {
      return operators.Size();
   }

   /// Returns the index of the finest level
   unsigned GetFinestLevelIndex() const
   {
      return NumLevels() - 1;
   }

   /// Matrix vector multiplication at given level
   void MultAtLevel(unsigned level, const Vector &x, Vector &y) const
   {
      MFEM_ASSERT(level < NumLevels(), "Level does not exist.");
      operators[level]->Mult(x, y);
   }

   /// Matrix vector multiplication on finest level
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_ASSERT(NumLevels() > 0, "At least one level needs to exist.");
      MultAtLevel(NumLevels() - 1, x, y);
   }

   void RestrictTo(unsigned level, const Vector &x, Vector &y) const
   {
      prolongations[level]->MultTranspose(x, y);
   }

   void InterpolateFrom(unsigned level, const Vector &x, Vector &y) const
   {
      prolongations[level]->Mult(x, y);
   }

   /// Apply Smoother at given level
   void ApplySmoother(unsigned level, const Vector &x, Vector &y) const
   {
      smoothers[level]->Mult(x, y);
   }

   /// Apply coarse solver
   void ApplyCoarseSolver(const Vector &x, Vector &y) const
   {
      ApplySmoother(0, x, y);
   }

   /// Returns operator at given level
   const Operator* GetOperatorAtLevel(unsigned level) const
   {
      return operators[level];
   }

   /// Returns operator at given level
   Operator* GetOperatorAtLevel(unsigned level)
   {
      return operators[level];
   }

   /// Returns operator at finest level
   const Operator* GetOperatorAtFinestLevel() const
   {
      return GetOperatorAtLevel(operators.Size() - 1);
   }

   /// Returns operator at finest level
   Operator* GetOperatorAtFinestLevel()
   {
      return GetOperatorAtLevel(operators.Size() - 1);
   }

   /// Returns smoother at given level
   const Solver* GetSmootherAtLevel(unsigned level) const
   {
      return smoothers[level];
   }

   /// Returns smoother at given level
   Solver* GetSmootherAtLevel(unsigned level)
   {
      return smoothers[level];
   }
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
   MultigridOperator& opr;
   CycleType cycleType;

   mutable Array<unsigned> preSmoothingSteps;
   mutable Array<unsigned> postSmoothingSteps;

   mutable Array<Vector*> X;
   mutable Array<Vector*> Y;
   mutable Array<Vector*> R;

   void Cycle(unsigned level) const
   {
      if (level == 0)
      {
         opr.ApplyCoarseSolver(*X[level], *Y[level]);
         return;
      }

      // Pre-smooth
      SLI(*opr.GetOperatorAtLevel(level), *opr.GetSmootherAtLevel(level), *X[level], *Y[level], -1, preSmoothingSteps[level]);

      // Compute residual
      opr.GetOperatorAtLevel(level)->Mult(*Y[level], *R[level]);
      subtract(*X[level], *R[level], *R[level]);

      // Restrict residual
      opr.RestrictTo(level - 1, *R[level], *X[level - 1]);

      // Init zeros
      *Y[level - 1] = 0.0;

      // Corrections
      unsigned corrections = 1;
      if (cycleType == CycleType::WCYCLE) { corrections = 2; }
      for (unsigned correction = 0; correction < corrections; ++correction)
      {
         Cycle(level - 1);
      }

      // Prolongate
      opr.InterpolateFrom(level - 1, *Y[level - 1], *R[level]);

      // Add update
      *Y[level] += *R[level];

      // Post-smooth
      SLI(*opr.GetOperatorAtLevel(level), *opr.GetSmootherAtLevel(level), *X[level], *Y[level], -1, postSmoothingSteps[level]);
   }

public:
   MultigridSolver(MultigridOperator &opr_,
                   CycleType cycleType_ = CycleType::VCYCLE,
                   unsigned preSmoothingSteps_ = 3,
                   unsigned postSmoothingSteps_ = 3)
      : opr(opr_), cycleType(cycleType_)
   {
     for (unsigned level = 0; level < opr.NumLevels(); ++level)
     {
        int vectorSize = opr.GetOperatorAtLevel(level)->Height();
        X.Append(new Vector(vectorSize));
        Y.Append(new Vector(vectorSize));
        R.Append(new Vector(vectorSize));
     }

     preSmoothingSteps.SetSize(opr.NumLevels());
     postSmoothingSteps.SetSize(opr.NumLevels());

     preSmoothingSteps = preSmoothingSteps_;
     postSmoothingSteps = postSmoothingSteps_;
   }

   ~MultigridSolver()
   {
      Reset();
   }

   void SetCycleType(CycleType cycleType_)
   {
      cycleType = cycleType_;
   }

   void Reset()
   {
      for (unsigned i = 0; i < X.Size(); ++i)
      {
         delete X[i];
         X[i] = nullptr;
         delete Y[i];
         Y[i] = nullptr;
         delete R[i];
         R[i] = nullptr;
      }

      X.DeleteAll();
      Y.DeleteAll();
      R.DeleteAll();
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      *X.Last() = x;
      *Y.Last() = y;
      Cycle(opr.NumLevels() - 1);
      y = *Y.Last();
   }

   /// Set/update the solver for the given operator.
   virtual void SetOperator(const Operator &op) override
   {
      if (!dynamic_cast<const MultigridOperator*>(&op))
      {
         MFEM_ABORT("Unsupported operator for MultigridSolver");
      }

      // TODO
   }

};

} // namespace mfem

#endif
