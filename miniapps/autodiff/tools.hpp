#pragma once
#include "mfem.hpp"

namespace mfem
{
class MappedGridFunctionCoefficient : public Coefficient
{
private:
   GridFunction *gf;
   std::function<real_t(const real_t)> map_func;
public:
   MappedGridFunctionCoefficient(GridFunction *gf_,
                                 std::function<real_t(const real_t)> map_func_)
      : gf(gf_), map_func(map_func_) {  }
   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return map_func(gf->GetValue(T.ElementNo, T.GetIntPoint()));
   }
};
class VectorGradientGridFunction : public MatrixCoefficient
{
private:
   GridFunction &gf;
public:
   VectorGradientGridFunction(GridFunction &gf)
      : MatrixCoefficient(gf.FESpace()->GetVDim(),
                          gf.FESpace()->GetMesh()->SpaceDimension()), gf(gf)
   {}

   void Eval(DenseMatrix &grad, ElementTransformation &T,
             const IntegrationPoint &ip) override
   { gf.GetVectorGradient(T, grad); }
};

inline std::unique_ptr<GridFunction>
NewGridFunction(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParGridFunction>(pfes);
   }
#endif
   return std::make_unique<GridFunction>(&fes);
}

inline std::unique_ptr<LinearForm>
NewLinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParLinearForm>(pfes);
   }
#endif
   return std::make_unique<LinearForm>(&fes);
}

inline std::unique_ptr<BilinearForm>
NewBilinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParBilinearForm>(pfes);
   }
#endif
   return std::make_unique<BilinearForm>(&fes);
}

inline std::unique_ptr<MixedBilinearForm>
NewMixedBilinearForm(FiniteElementSpace &trial_fes,
                     FiniteElementSpace &test_fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *trial_pfes =
          dynamic_cast<ParFiniteElementSpace*>(&trial_fes))
   {
      ParFiniteElementSpace *test_pfes = dynamic_cast<ParFiniteElementSpace*>
                                         (&test_fes);
      MFEM_VERIFY(test_pfes != nullptr,
                  "NewMixedBilinearForm: Trial is parallel, but test is not.");
      return std::make_unique<ParMixedBilinearForm>(trial_pfes, test_pfes);
   }
   MFEM_VERIFY(dynamic_cast<ParFiniteElementSpace*>(&test_fes) == nullptr,
               "NewMixedBilinearForm: Trial is not parallel, but test is.");
#endif
   return std::make_unique<MixedBilinearForm>(&trial_fes, &test_fes);
}
inline std::unique_ptr<NonlinearForm>
NewNonlinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParNonlinearForm>(pfes);
   }
#endif
   return std::make_unique<NonlinearForm>(&fes);
}
inline std::unique_ptr<BlockNonlinearForm>
NewBlockNonlinearForm(Array<FiniteElementSpace*> &fes)
{
#ifdef MFEM_USE_MPI
   int numParallel = 0;

   Array<ParFiniteElementSpace*> pfes;
   for (auto *space : fes)
   {
      pfes.Append(dynamic_cast<ParFiniteElementSpace*>(space));
      numParallel += pfes.Last() != nullptr;
   }
   MFEM_VERIFY(numParallel == 0 || numParallel == fes.Size(),
               "NewBlockNonlinearForm: either all or none of the spaces must be parallel");
   if (numParallel == fes.Size())
   {
      return std::make_unique<ParBlockNonlinearForm>(pfes);
   }
#endif
   return std::make_unique<BlockNonlinearForm>(fes);
}

#ifdef MFEM_USE_PETSC
// Monolithic direct solver for block system
class MUMPSMonoSolver : public MUMPSSolver
{
private:
   std::unique_ptr<HypreParMatrix> mono;
public:
   MUMPSMonoSolver(MPI_Comm comm) : MUMPSSolver(comm) {}

   void SetOperator(const Operator &op)
   {
      const BlockOperator *bop = dynamic_cast<const BlockOperator*>(&op);
      MFEM_VERIFY(bop != nullptr, "Not a BlockOperator");
      Array2D<const HypreParMatrix*> blocks(bop->NumRowBlocks(), bop->NumColBlocks());
      for (int j=0; j<bop->NumColBlocks(); j++)
      {
         for (int i=0; i<bop->NumRowBlocks(); i++)
         {
            if (bop->IsZeroBlock(i,j)) { continue; }
            const HypreParMatrix *m =
               dynamic_cast<const HypreParMatrix*>(&bop->GetBlock(i,j));
            MFEM_VERIFY(m != nullptr, "Not a HypreParMatrix");
            blocks(i,j) = m;
         }
      }
      mono.reset(HypreParMatrixFromBlocks(blocks));
      MUMPSSolver::SetOperator(*mono);
   };
};
#endif

inline std::tuple<std::unique_ptr<FiniteElementSpace>, std::unique_ptr<L2_FECollection>>
QSpaceToFESpace(QuadratureSpace &qs)
{
   Mesh *mesh = qs.GetMesh();
   const int dim = mesh->Dimension();
   Geometry::Type geom = mesh->GetTypicalElementGeometry();
   MFEM_VERIFY(geom != Geometry::TRIANGLE &&
               geom != Geometry::TETRAHEDRON &&
               geom != Geometry::PRISM &&
               geom != Geometry::PYRAMID,
               "QSpaceToFESpace: only support tensor product elements");
   std::unique_ptr<L2_FECollection> fec
      = std::make_unique<L2_FECollection> (qs.GetOrder()/2, dim);

   std::unique_ptr<FiniteElementSpace> fes;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(qs.GetMesh());
   if (pmesh) { fes = std::make_unique<ParFiniteElementSpace>(pmesh, fec.get()); }
#endif
   if (!fes) { fes = std::make_unique<FiniteElementSpace>(mesh, fec.get()); }
   return std::make_tuple(std::move(fes), std::move(fec));
}

inline Array<int> GetOffsets(const Array<FiniteElementSpace*> &fespaces)
{
   Array<int> offsets(fespaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.Size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetVSize();
   }
   return std::move(offsets);
}
inline Array<int> GetTrueOffsets(const Array<FiniteElementSpace*> &fespaces)
{
   Array<int> offsets(fespaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.Size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetTrueVSize();
   }
   return std::move(offsets);
}

class VectorNormCoefficient : public Coefficient
{
private:
   VectorCoefficient &vc;
   Vector v;
public:
   VectorNormCoefficient(VectorCoefficient &vc): vc(vc), v(vc.GetVDim()) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      vc.Eval(v, T, ip);
      return std::sqrt(v*v);
   }
};

class BooleanCoefficient : public Coefficient
{
private:
   Coefficient &cf;
   std::function<bool(real_t)> func;
public:
   BooleanCoefficient(Coefficient &cf, std::function<bool(real_t)> func)
      : cf(cf), func(func) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return func(cf.Eval(T, ip));
   }
};

};
