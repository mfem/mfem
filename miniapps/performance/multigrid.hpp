#ifndef MFEM_MULTIGRID
#define MFEM_MULTIGRID

#include "../../general/forall.hpp"
#include "../../linalg/operator.hpp"

namespace mfem
{

/// Class bundling a hierarchy of meshes, finite element spaces, and essential dof lists
template<typename M, typename FES>
class GeneralSpaceHierarchy
{
private:
   Array<M*> meshes;
   Array<FES*> fespaces;

public:
   GeneralSpaceHierarchy(M* mesh, FES* fespace)
   {
      addLevel(mesh, fespace);
   }

   unsigned GetNumLevels() const
   {
      return meshes.Size();
   }

   unsigned GetFinestLevel() const
   {
      return GetNumLevels() - 1;
   }

   void addLevel(M* mesh, FES* fespace)
   {
      meshes.Append(mesh);
      fespaces.Append(fespace);
   }

   M* GetMesh(unsigned level) const
   {
      MFEM_ASSERT(meshes.Size() > level, "Mesh at given level does not exist.");
      return meshes[level];
   }

   M* GetFinestMesh() const
   {
      return GetMesh(GetFinestLevel());
   }

   FES* GetFESpace(unsigned level) const
   {
      MFEM_ASSERT(fespaces.Size() > level, "FE space at given level does not exist.");
      return fespaces[level];
   }

   FES* GetFinestFESpace() const
   {
      return GetFESpace(GetFinestLevel());
   }
};

using SpaceHierarchy = GeneralSpaceHierarchy<Mesh, FiniteElementSpace>;
using ParSpaceHierarchy = GeneralSpaceHierarchy<ParMesh, ParFiniteElementSpace>;

/// Multigrid operator
class OperatorMultigrid : public Operator
{
private:
   Array<Operator*> forms;
   Array<Operator*> operators;
   Array<Solver*> smoothers;
   Array<Operator*> prolongations;
   Array<Array<int>*> ess_tdof_lists;

public:
   /// Constructor
   OperatorMultigrid(Operator* form, Operator* opr, Solver* coarseSolver, Array<int>* ess_tdof_list)
   {
      forms.Append(form);
      operators.Append(opr);
      smoothers.Append(coarseSolver);
      ess_tdof_lists.Append(ess_tdof_list);
   }

   void AddLevel(Operator* form, Operator* opr, Solver* smoother, Operator* prolongation, Array<int>* ess_tdof_list)
   {
      forms.Append(form);
      operators.Append(opr);
      smoothers.Append(smoother);
      prolongations.Append(prolongation);
      ess_tdof_lists.Append(ess_tdof_list);
   }

   /// Returns the number of levels
   int NumLevels() const
   {
      return operators.Size();
   }

   /// Matrix vector multiplication at given level
   virtual void MultAtLevel(unsigned level, const Vector &x, Vector &y) const
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

   /// Apply Smoother on given level
   void ApplySmoother(unsigned level, const Vector &x, Vector &y) const
   {
      smoothers[level]->Mult(x, y);
   }

   /// Apply coarse solver
   void ApplyCoarseSolver(const Vector &x, Vector &y) const
   {
      ApplySmoother(0, x, y);
   }

   /// Returns form at given level
   Operator* GetFormAtLevel(unsigned level)
   {
      MFEM_ASSERT(forms.Size() > level, "Form at given level do not exist.");
      return forms[level];
   }

   /// Returns form at finest level
   Operator* GetFormAtFinestLevel()
   {
      return GetFormAtLevel(forms.Size() - 1);
   }

   /// Returns operator at given level
   const Operator* GetOperatorAtLevel(unsigned level) const
   {
      return operators[level];
   }

   /// Returns operator at given level
   Solver* GetSmootherAtLevel(unsigned level)
   {
      return smoothers[level];
   }

   Array<int>* GetEssentialDoFsAtLevel(unsigned level) const
   {
      MFEM_ASSERT(ess_tdof_lists.Size() > level, "Ess. dofs at given level do not exist.");
      return ess_tdof_lists[level];
   }

   Array<int>* GetEssentialDoFsAtFinestLevel() const
   {
      return GetEssentialDoFsAtLevel(ess_tdof_lists.Size() - 1);
   }
};

// Multigrid solver
class SolverMultigrid : public Solver
{
private:
   OperatorMultigrid& opr;

   mutable Array<Vector*> X;
   mutable Array<Vector*> Y;
   mutable Array<Vector*> R;

   void cycle(unsigned level) const
   {
      if (level == 0)
      {
         opr.ApplyCoarseSolver(*X[level], *Y[level]);
         return;
      }

      // Pre-smooth
      SLI(*opr.GetOperatorAtLevel(level), *opr.GetSmootherAtLevel(level), *X[level], *Y[level], -1, 3);

      // Compute residual
      opr.GetOperatorAtLevel(level)->Mult(*Y[level], *R[level]);
      subtract(*X[level], *R[level], *R[level]);

      // Restrict residual
      opr.RestrictTo(level - 1, *R[level], *X[level - 1]);

      // Init zeros
      *Y[level - 1] = 0.0;

      // Corrections
      for (int correction = 0; correction < 1; ++correction)
      {
         cycle(level - 1);
      }

      // Prolongate
      opr.InterpolateFrom(level - 1, *Y[level - 1], *R[level]);

      Array<int>& essentialDofs = *opr.GetEssentialDoFsAtLevel(level);
      auto I = essentialDofs.Read();
      auto T = R[level]->Write();
      MFEM_FORALL(i, essentialDofs.Size(), T[I[i]] = 0.0; );

      // Add update
      *Y[level] += *R[level];

      // Post-smooth
      SLI(*opr.GetOperatorAtLevel(level), *opr.GetSmootherAtLevel(level), *X[level], *Y[level], -1, 3);
   }

public:
   SolverMultigrid(OperatorMultigrid &opr_)
      : opr(opr_)
   {
      for (unsigned level = 0; level < opr.NumLevels(); ++level)
      {
         int vectorSize = opr.GetOperatorAtLevel(level)->Height();
         X.Append(new Vector(vectorSize));
         Y.Append(new Vector(vectorSize));
         R.Append(new Vector(vectorSize));
      }
   }

   ~SolverMultigrid()
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
   }

   /// Matrix vector multiplication on finest level
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      *X.Last() = x;
      *Y.Last() = y;
      cycle(opr.NumLevels() - 1);
      y = *Y.Last();
   }

   /// Set/update the solver for the given operator.
   virtual void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("Not implemented.");
   }

};

} // namespace mfem

#endif
