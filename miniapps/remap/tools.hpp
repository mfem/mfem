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


};
