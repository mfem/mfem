#include "mfem.hpp"
#include "util.hpp"

namespace mfem
{

GridFunction* ProlongToMaxOrder(const GridFunction *x)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // find the max order in the space
   int max_order = 1;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      max_order = std::max(fespace->GetElementOrder(i), max_order);
   }

   // create a visualization space of max order for all elements
   FiniteElementCollection *l2fec =
      new L2_FECollection(max_order, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace *l2space = new FiniteElementSpace(mesh, l2fec);

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *prolonged_x = new GridFunction(l2space);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementDofs(i, dofs);
      Vector elemvect, l2vect;
      x->GetSubVector(dofs, elemvect);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *l2fe = l2fec->GetFE(geom, max_order);

      l2fe->GetTransferMatrix(*fe, T, I);
      l2space->GetElementDofs(i, dofs);
      l2vect.SetSize(dofs.Size());

      I.Mult(elemvect, l2vect);
      prolonged_x->SetSubVector(dofs, l2vect);
   }

   prolonged_x->MakeOwner(l2fec);
   return prolonged_x;
}


} // namespace mfem
