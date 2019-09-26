#ifndef MFEM_MULTIGRID
#define MFEM_MULTIGRID

#include "../../general/forall.hpp"
#include "../../linalg/operator.hpp"

namespace mfem
{

/// Class bundling a hierarchy of meshes and finite element spaces
template <typename M, typename FES> class GeneralSpaceHierarchy
{
 private:
   Array<M *> meshes;
   Array<FES *> fespaces;
   Array<bool> ownedMeshes;
   Array<bool> ownedFES;

 public:
   GeneralSpaceHierarchy(M *mesh, FES *fespace, bool ownM, bool ownFES)
   {
      AddLevel(mesh, fespace, ownM, ownFES);
   }

   ~GeneralSpaceHierarchy()
   {
      for (int i = meshes.Size() - 1; i >= 0; --i)
      {
         if (ownedFES[i])
         {
            delete fespaces[i];
         }
         if (ownedMeshes[i])
         {
            delete meshes[i];
         }
      }

      fespaces.DeleteAll();
      meshes.DeleteAll();
   }

   unsigned GetNumLevels() const { return meshes.Size(); }

   unsigned GetFinestLevelIndex() const { return GetNumLevels() - 1; }

   void AddLevel(M *mesh, FES *fespace, bool ownM, bool ownFES)
   {
      meshes.Append(mesh);
      fespaces.Append(fespace);
      ownedMeshes.Append(ownM);
      ownedFES.Append(ownFES);
   }

   void AddUniformlyRefinedLevel(int dim = 1, int ordering = Ordering::byVDIM)
   {
      M *mesh = new M(*meshes[GetFinestLevelIndex()]);
      mesh->UniformRefinement();
      FES *coarseFEspace = fespaces[GetFinestLevelIndex()];
      FES *fineFEspace = new FES(mesh, coarseFEspace->FEColl(), dim, ordering);
      AddLevel(mesh, fineFEspace, true, true);
   }

   void AddOrderRefinedLevel(FiniteElementCollection *fec, int dim = 1,
                             int ordering = Ordering::byVDIM)
   {
      M *mesh = &GetFinestMesh();
      FES *newFEspace = new FES(mesh, fec, dim, ordering);
      AddLevel(mesh, newFEspace, false, true);
   }

   const M &GetMeshAtLevel(unsigned level) const
   {
      MFEM_ASSERT(level < meshes.Size(), "Mesh at given level does not exist.");
      return *meshes[level];
   }

   M &GetMeshAtLevel(unsigned level)
   {
      return const_cast<M &>(
          const_cast<const GeneralSpaceHierarchy *>(this)->GetMeshAtLevel(
              level));
   }

   const M &GetFinestMesh() const
   {
      return GetMeshAtLevel(GetFinestLevelIndex());
   }

   M &GetFinestMesh()
   {
      return const_cast<M &>(
          const_cast<const GeneralSpaceHierarchy *>(this)->GetFinestMesh());
   }

   const FES &GetFESpaceAtLevel(unsigned level) const
   {
      MFEM_ASSERT(level < fespaces.Size(),
                  "FE space at given level does not exist.");
      return *fespaces[level];
   }

   FES &GetFESpaceAtLevel(unsigned level)
   {
      return const_cast<FES &>(
          const_cast<const GeneralSpaceHierarchy *>(this)->GetFESpaceAtLevel(
              level));
   }

   const FES &GetFinestFESpace() const
   {
      return GetFESpaceAtLevel(GetFinestLevelIndex());
   }

   FES &GetFinestFESpace()
   {
      return const_cast<FES &>(
          const_cast<const GeneralSpaceHierarchy *>(this)->GetFinestFESpace());
   }
};

using SpaceHierarchy = GeneralSpaceHierarchy<Mesh, FiniteElementSpace>;
using ParSpaceHierarchy = GeneralSpaceHierarchy<ParMesh, ParFiniteElementSpace>;

/// Abstract multigrid operator
class MultigridOperator : public Operator
{
 protected:
   Array<Operator *> operators;
   Array<Solver *> smoothers;
   Array<const Operator *> prolongations;

   Array<bool> ownedOperators;
   Array<bool> ownedSmoothers;
   Array<bool> ownedProlongations;

 public:
   MultigridOperator() {}

   MultigridOperator(Operator *opr, Solver *coarseSolver, bool ownOperator,
                     bool ownSolver)
   {
      AddCoarseLevel(opr, coarseSolver, ownOperator, ownSolver);
   }

   ~MultigridOperator()
   {
      for (int i = operators.Size() - 1; i >= 0; --i)
      {
         if (ownedOperators[i])
         {
            delete operators[i];
         }
         if (ownedSmoothers[i])
         {
            delete smoothers[i];
         }
      }

      for (int i = prolongations.Size() - 1; i >= 0; --i)
      {
         if (ownedProlongations[i])
         {
            delete prolongations[i];
         }
      }

      operators.DeleteAll();
      smoothers.DeleteAll();
   }

   void AddCoarseLevel(Operator *opr, Solver *solver, bool ownOperator,
                       bool ownSolver)
   {
      MFEM_VERIFY(NumLevels() == 0, "Coarse level already exists");
      operators.Append(opr);
      smoothers.Append(solver);
      ownedOperators.Append(ownOperator);
      ownedSmoothers.Append(ownSolver);
      width = opr->Width();
      height = opr->Height();
   }

   void AddLevel(Operator *opr, Solver *smoother, const Operator *prolongation,
                 bool ownOperator, bool ownSmoother, bool ownProlongation)
   {
      MFEM_VERIFY(NumLevels() > 0, "Please add a coarse level first");
      operators.Append(opr);
      smoothers.Append(smoother);
      prolongations.Append(prolongation);
      ownedOperators.Append(ownOperator);
      ownedSmoothers.Append(ownSmoother);
      ownedProlongations.Append(ownProlongation);
      width = opr->Width();
      height = opr->Height();
   }

   /// Returns the number of levels
   unsigned NumLevels() const { return operators.Size(); }

   /// Returns the index of the finest level
   unsigned GetFinestLevelIndex() const { return NumLevels() - 1; }

   /// Matrix vector multiplication at given level
   void MultAtLevel(unsigned level, const Vector &x, Vector &y) const
   {
      MFEM_ASSERT(level < NumLevels(), "Level does not exist.");
      operators[level]->Mult(x, y);
   }

   /// Matrix vector multiplication on finest level
   void Mult(const Vector &x, Vector &y) const override
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
   void ApplySmootherAtLevel(unsigned level, const Vector &x, Vector &y) const
   {
      smoothers[level]->Mult(x, y);
   }

   /// Returns operator at given level
   const Operator *GetOperatorAtLevel(unsigned level) const
   {
      return operators[level];
   }

   /// Returns operator at given level
   Operator *GetOperatorAtLevel(unsigned level) { return operators[level]; }

   /// Returns operator at finest level
   const Operator *GetOperatorAtFinestLevel() const
   {
      return GetOperatorAtLevel(operators.Size() - 1);
   }

   /// Returns operator at finest level
   Operator *GetOperatorAtFinestLevel()
   {
      return GetOperatorAtLevel(operators.Size() - 1);
   }

   /// Returns smoother at given level
   Solver *GetSmootherAtLevel(unsigned level) const { return smoothers[level]; }

   /// Returns smoother at given level
   Solver *GetSmootherAtLevel(unsigned level) { return smoothers[level]; }
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
   const MultigridOperator *opr;
   CycleType cycleType;

   mutable Array<unsigned> preSmoothingSteps;
   mutable Array<unsigned> postSmoothingSteps;

   mutable Array<Vector *> X;
   mutable Array<Vector *> Y;
   mutable Array<Vector *> R;
   mutable Array<Vector *> Z;

   void SmoothingStep(int level) const
   {
      opr->MultAtLevel(level, *Y[level], *R[level]);          // r = A x
      subtract(*X[level], *R[level], *R[level]);              // r = b - A x
      opr->ApplySmootherAtLevel(level, *R[level], *Z[level]); // z = S r
      add(*Y[level], 1.0, *Z[level], *Y[level]); // x = x + S (b - A x)
   }

   void Cycle(unsigned level) const
   {
      if (level == 0)
      {
         opr->ApplySmootherAtLevel(level, *X[level], *Y[level]);
         return;
      }

      for (int i = 0; i < preSmoothingSteps[level]; i++)
      {
         SmoothingStep(level);
      }

      // Compute residual
      opr->GetOperatorAtLevel(level)->Mult(*Y[level], *R[level]);
      subtract(*X[level], *R[level], *R[level]);

      // Restrict residual
      opr->RestrictTo(level - 1, *R[level], *X[level - 1]);

      // Init zeros
      *Y[level - 1] = 0.0;

      // Corrections
      unsigned corrections = 1;
      if (cycleType == CycleType::WCYCLE)
      {
         corrections = 2;
      }
      for (unsigned correction = 0; correction < corrections; ++correction)
      {
         Cycle(level - 1);
      }

      // Prolongate
      opr->InterpolateFrom(level - 1, *Y[level - 1], *R[level]);

      // Add update
      *Y[level] += *R[level];

      // Post-smooth
      for (int i = 0; i < postSmoothingSteps[level]; i++)
      {
         SmoothingStep(level);
      }
   }

   void Setup(unsigned preSmoothingSteps_ = 3, unsigned postSmoothingSteps_ = 3)
   {
      for (unsigned level = 0; level < opr->NumLevels() - 1; ++level)
      {
         int vectorSize = opr->GetOperatorAtLevel(level)->Height();
         X.Append(new Vector(vectorSize));
         *X.Last() = 0.0;
         Y.Append(new Vector(vectorSize));
         *Y.Last() = 0.0;
         R.Append(new Vector(vectorSize));
         *R.Last() = 0.0;
         Z.Append(new Vector(vectorSize));
         *Z.Last() = 0.0;
      }

      // X and Y at the finest level will be filled by Mult
      X.Append(nullptr);
      Y.Append(nullptr);
      R.Append(new Vector(opr->GetOperatorAtFinestLevel()->Height()));
      *R.Last() = 0.0;
      Z.Append(new Vector(opr->GetOperatorAtFinestLevel()->Height()));
      *Z.Last() = 0.0;

      preSmoothingSteps.SetSize(opr->NumLevels());
      postSmoothingSteps.SetSize(opr->NumLevels());

      preSmoothingSteps = preSmoothingSteps_;
      postSmoothingSteps = postSmoothingSteps_;
   }

   void Reset()
   {
      for (unsigned i = 0; i < X.Size(); ++i)
      {
         delete X[i];
         delete Y[i];
         delete R[i];
         delete Z[i];
      }

      X.DeleteAll();
      Y.DeleteAll();
      R.DeleteAll();
      Z.DeleteAll();

      preSmoothingSteps.DeleteAll();
      postSmoothingSteps.DeleteAll();
   }

 public:
   MultigridSolver(const MultigridOperator *opr_,
                   CycleType cycleType_ = CycleType::VCYCLE,
                   unsigned preSmoothingSteps_ = 3,
                   unsigned postSmoothingSteps_ = 3)
       : opr(opr_), cycleType(cycleType_)
   {
      Setup(preSmoothingSteps_, postSmoothingSteps_);
   }

   ~MultigridSolver() { Reset(); }

   void SetCycleType(CycleType cycleType_) { cycleType = cycleType_; }

   void SetPreSmoothingSteps(unsigned steps) { preSmoothingSteps = steps; }

   void SetPreSmoothingSteps(const Array<unsigned> &steps)
   {
      MFEM_VERIFY(
          steps.Size() == preSmoothingSteps.Size(),
          "Number of step sizes needs to be the same as the number of levels");
      preSmoothingSteps = steps;
   }

   void SetPostSmoothingSteps(unsigned steps) { postSmoothingSteps = steps; }

   void SetPostSmoothingSteps(const Array<unsigned> &steps)
   {
      MFEM_VERIFY(
          steps.Size() == postSmoothingSteps.Size(),
          "Number of step sizes needs to be the same as the number of levels");
      postSmoothingSteps = steps;
   }

   void SetSmoothingSteps(unsigned steps)
   {
      SetPreSmoothingSteps(steps);
      SetPostSmoothingSteps(steps);
   }

   void SetSmoothingSteps(const Array<unsigned> &steps)
   {
      SetPreSmoothingSteps(steps);
      SetPostSmoothingSteps(steps);
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      // Safe const_cast, since x at the finest level will never be modified
      X.Last() = const_cast<Vector *>(&x);
      y = 0.0;
      Y.Last() = &y;
      Cycle(opr->NumLevels() - 1);
      X.Last() = nullptr;
      Y.Last() = nullptr;
   }

   /// Set/update the solver for the given operator.
   virtual void SetOperator(const Operator &op) override
   {
      if (!dynamic_cast<const MultigridOperator *>(&op))
      {
         MFEM_ABORT("Unsupported operator for MultigridSolver");
      }

      Reset();
      opr = static_cast<const MultigridOperator *>(&op);
      Setup();
   }
};

} // namespace mfem

#endif
