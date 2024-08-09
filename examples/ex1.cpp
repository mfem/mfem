#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{

   Mesh mesh_macro = Mesh::MakeCartesian2D(2, 2, mfem::Element::Type::QUADRILATERAL,
                                           true, 2.0, 2.0);

   Mesh mesh_micro = Mesh::MakeCartesian2D(4, 4, mfem::Element::Type::QUADRILATERAL,
                                           true, 0.8, 0.8);

   int dim = mesh_macro.Dimension();

   mesh_micro.SetCurvature(1, false, dim, 0);
   mesh_macro.SetCurvature(1, false, dim, 0);

   FiniteElementCollection *fec = new H1_FECollection(1, dim);

   FiniteElementSpace fespace_macro(&mesh_macro, fec);
   FiniteElementSpace fespace_micro(&mesh_micro, fec);

   Array<int> dofs; //to store edge dofs

   set<int> dof_set, dof_set_cpy;

   for (int i = 0; i < mesh_micro.GetNBE(); i++)
   {
      const FiniteElement * elem = fespace_micro.GetBE(i);
      const IntegrationRule & ir = elem->GetNodes();

      fespace_micro.GetBdrElementDofs(i, dofs);

      for (int j = 0; j < dofs.Size(); j++)
      {
         dof_set.insert(dofs[j]);
      }
   }

   std::cout << "Bounday DOF Size " << dof_set.size() << std::endl;
   std::cout << "nbe micro: " << fespace_micro.GetNBE() << std::endl;

   int bnd_pts =  dof_set.size();
   Vector vxy(bnd_pts * dim);
   int *dof_ids = new int[bnd_pts];
   int count = 0;

   for (int i = 0; i < dof_set.size(); i++)
   {
      const FiniteElement * elem = fespace_micro.GetBE(i);
      const IntegrationRule & ir = elem->GetNodes();

      fespace_micro.GetBdrElementDofs(i, dofs);

      FaceElementTransformations * T = mesh_micro.GetBdrFaceTransformations(i);
      DenseMatrix P;
      T->Transform(ir, P);

      assert(("The size of dof should be same ir", dofs.Size()  == ir.Size()));

      for (int j = 0; j < ir.Size(); j++)
      {
         // std::cout << "DOF# " << dofs[j] << "  "  << P(0, j) << "  " << P(1, j) << std::endl;

         if(dof_set_cpy.find(dofs[j]) == dof_set_cpy.end())
         {
            vxy(count)             = P(0, j);
            vxy(bnd_pts + (count)) = P(1, j);
            dof_set_cpy.insert(dofs[j]);
            dof_ids[count] = dofs[j];
            count = count + 1;
         }
      }
   }

   FindPointsGSLIB finder_pos;
   finder_pos.FindPoints(mesh_macro, vxy);

   auto obj_elem = finder_pos.GetGSLIBElem();
   auto obj_ref_pos = finder_pos.GetReferencePosition();
   //auto obj_ref_pos = finder_pos.GetGSLIBReferencePosition();

   //std::cout << "Boundary points:" << bnd_pts << " Count = " << count << std::endl;

   IntegrationPoint ip;
   BiLinear2DFiniteElement bilinear_elem;
   Vector shape(dim*dim);

   for (int i = 0; i < bnd_pts; i++)
   {
      ip.Set2(obj_ref_pos(i*dim + 0), obj_ref_pos(i*dim + 1));
      bilinear_elem.CalcShape(ip, shape);

      //std::cout << obj_elem[i] << " DOF # " << dof_ids[i]  << "  " << obj_ref_pos(i*dim + 0) << "  " << obj_ref_pos(i*dim + 1) << "  " << shape(0) << std::endl;
      std::cout << " DOF # " << dof_ids[i]  << "    " << shape(0) << std::endl;
   }


   return 0;
}
