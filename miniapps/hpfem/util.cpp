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


const char vishost[] = "localhost";
const int  visport   = 19916;

void VisualizeField(socketstream &sock, GridFunction &gf, const char *title,
                    const char *keys, int w, int h, int x, int y, bool vec)
{
   Mesh &mesh = *gf.FESpace()->GetMesh();

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (!sock.is_open() || !sock)
      {
         sock.open(vishost, visport);
         sock.precision(8);
         newly_opened = true;
      }
      sock << "solution\n";

      mesh.Print(sock);
      gf.Save(sock);

      if (newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";

         if (keys) { sock << "keys " << keys << "\n"; }
         else { sock << "keys mAc\n"; }

         if (vec) { sock << "vvv"; }
         sock << std::endl;
      }

      connection_failed = !sock && !newly_opened;
   }
   while (connection_failed);
}


} // namespace mfem
