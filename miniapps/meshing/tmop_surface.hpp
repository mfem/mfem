#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mesh-optimizer.hpp"
using namespace std;
using namespace mfem;

void ModifyAttributeForMarkingDOFS(Mesh *mesh, GridFunction &mat,
                                   int attr_to_switch)
{
   // Switch attribute if all but 1 of the faces of an element will be marked?
   Array<int> element_attr(mesh->GetNE());
   element_attr = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      Array<int> faces, ori;
      if (mesh->Dimension() == 2)
      {
         mesh->GetElementEdges(e, faces, ori);
      }
      else
      {
         mesh->GetElementFaces(e, faces, ori);
      }
      int inf1, inf2;
      int elem1, elem2;
      int diff_attr_count = 0;
      int attr1;
      int attr2;
      attr1 = mat(e);
      bool bdr_element = false;
      element_attr[e] = attr1;
      int target_attr = -1;
      for (int f = 0; f < faces.Size(); f++)
      {
         mesh->GetFaceElements(faces[f], &elem1, &elem2);
         if (elem2 >= 0)
         {
            attr2 = elem1 == e ? (int)(mat(elem2)) : (int)(mat(elem1));
            if (attr1 != attr2 && attr1 == attr_to_switch)
            {
               diff_attr_count += 1;
               target_attr = attr2;
            }
         }
         else
         {
            mesh->GetFaceInfos(faces[f], &inf1, &inf2);
            if (inf2 >= 0)
            {
               Vector dof_vals;
               Array<int> dofs;
               mat.GetElementDofValues(mesh->GetNE() + (-1-elem2), dof_vals);
               attr2 = (int)(dof_vals(0));
               if (attr1 != attr2 && attr1 == attr_to_switch)
               {
                  diff_attr_count += 1;
                  target_attr = attr2;
               }
            }
            else
            {
               bdr_element = true;
            }
         }
      }

      if (diff_attr_count == faces.Size()-1 && !bdr_element)
      {
         element_attr[e] = target_attr;
      }
   }
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mat(e) = element_attr[e];
      mesh->SetAttribute(e, element_attr[e]+1);
   }
   mesh->SetAttributes();
}

Mesh* TrimMesh(Mesh &mesh, FunctionCoefficient &ls_coeff, int order,
               int attr_to_trim)
{
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes_s(&mesh, &fec);
   GridFunction distance_s(&fes_s);
   distance_s.ProjectCoefficient(ls_coeff);
   L2_FECollection mat_coll(0, dim);
   FiniteElementSpace mat_fes(&mesh, &mat_coll);
   GridFunction mat(&mat_fes);

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.SetAttribute(e, 1);
   }
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mat(e) = material_id(e, distance_s);
      mesh.SetAttribute(e, mat(e) + 1);
   }

   ModifyAttributeForMarkingDOFS(&mesh, mat, 0);
   ModifyAttributeForMarkingDOFS(&mesh, mat, 1);

   mesh.SetAttributes();

   Array<int> attr(1);
   attr[0] = attr_to_trim;
   Array<int> bdr_attr;

   int max_attr     = mesh.attributes.Max();
   int max_bdr_attr = mesh.bdr_attributes.Max();

   if (bdr_attr.Size() == 0)
   {
      bdr_attr.SetSize(attr.Size());
      for (int i=0; i<attr.Size(); i++)
      {
         bdr_attr[i] = max_bdr_attr + attr[i];
      }
   }
   MFEM_VERIFY(attr.Size() == bdr_attr.Size(),
               "Size mismatch in attribute arguments.");

   Array<int> marker(max_attr);
   Array<int> attr_inv(max_attr);
   marker = 0;
   attr_inv = 0;
   for (int i=0; i<attr.Size(); i++)
   {
      marker[attr[i]-1] = 1;
      attr_inv[attr[i]-1] = i;
   }

   // Count the number of elements in the final mesh
   int num_elements = 0;
   for (int e=0; e<mesh.GetNE(); e++)
   {
      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1]) { num_elements++; }
   }

   // Count the number of boundary elements in the final mesh
   int num_bdr_elements = 0;
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 == 0 || a2 == 0)
      {
         if (a1 == 0 && !marker[a2-1]) { num_bdr_elements++; }
         else if (a2 == 0 && !marker[a1-1]) { num_bdr_elements++; }
      }
      else
      {
         if (marker[a1-1] && !marker[a2-1]) { num_bdr_elements++; }
         else if (!marker[a1-1] && marker[a2-1]) { num_bdr_elements++; }
      }
   }

   cout << "Number of Elements:          " << mesh.GetNE() << " -> "
        << num_elements << endl;
   cout << "Number of Boundary Elements: " << mesh.GetNBE() << " -> "
        << num_bdr_elements << endl;

   Mesh *trimmed_mesh = new Mesh(mesh.Dimension(), mesh.GetNV(),
                                 num_elements, num_bdr_elements, mesh.SpaceDimension());
   //   Mesh trimmed_mesh(mesh.Dimension(), mesh.GetNV(),
   //                     num_elements, num_bdr_elements, mesh.SpaceDimension());

   // Copy vertices
   for (int v=0; v<mesh.GetNV(); v++)
   {
      trimmed_mesh->AddVertex(mesh.GetVertex(v));
   }

   // Copy elements
   for (int e=0; e<mesh.GetNE(); e++)
   {
      Element * el = mesh.GetElement(e);
      int elem_attr = el->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nel = mesh.NewElement(el->GetGeometryType());
         nel->SetAttribute(elem_attr);
         nel->SetVertices(el->GetVertices());
         trimmed_mesh->AddElement(nel);
      }
   }

   // Copy selected boundary elements
   for (int be=0; be<mesh.GetNBE(); be++)
   {
      int e, info;
      mesh.GetBdrElementAdjacentElement(be, e, info);

      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nbel = mesh.GetBdrElement(be)->Duplicate(trimmed_mesh);
         trimmed_mesh->AddBdrElement(nbel);
      }
   }

   // Create new boundary elements
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int i1 = -1, i2 = -1;
      mesh.GetFaceInfos(f, &i1, &i2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 != 0 && a2 != 0)
      {
         if (marker[a1-1] && !marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(trimmed_mesh);
            //bel->SetAttribute(bdr_attr[attr_inv[a1-1]]);
            bel->SetAttribute(3);
            trimmed_mesh->AddBdrElement(bel);
         }
         else if (!marker[a1-1] && marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(trimmed_mesh);
            //bel->SetAttribute(bdr_attr[attr_inv[a2-1]]);
            bel->SetAttribute(3);
            trimmed_mesh->AddBdrElement(bel);
         }
      }
   }

   trimmed_mesh->FinalizeTopology();
   trimmed_mesh->Finalize();
   trimmed_mesh->RemoveUnusedVertices();

   // Check for curved or discontinuous mesh
   if (mesh.GetNodes())
   {
      // Extract Nodes GridFunction and determine its type
      const GridFunction * Nodes = mesh.GetNodes();
      const FiniteElementSpace * fes = Nodes->FESpace();

      Ordering::Type ordering = fes->GetOrdering();
      int order = fes->FEColl()->GetOrder();
      int sdim = mesh.SpaceDimension();
      bool discont =
         dynamic_cast<const L2_FECollection*>(fes->FEColl()) != NULL;

      // Set curvature of the same type as original mesh
      trimmed_mesh->SetCurvature(order, discont, sdim, ordering);

      const FiniteElementSpace * trimmed_fes = trimmed_mesh->GetNodalFESpace();
      GridFunction * trimmed_nodes = trimmed_mesh->GetNodes();

      Array<int> vdofs;
      Array<int> trimmed_vdofs;
      Vector loc_vec;

      // Copy nodes to trimmed mesh
      int te = 0;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Element * el = mesh.GetElement(e);
         int elem_attr = el->GetAttribute();
         if (!marker[elem_attr-1])
         {
            fes->GetElementVDofs(e, vdofs);
            Nodes->GetSubVector(vdofs, loc_vec);

            trimmed_fes->GetElementVDofs(te, trimmed_vdofs);
            trimmed_nodes->SetSubVector(trimmed_vdofs, loc_vec);
            te++;
         }
      }
   }

   return trimmed_mesh;
}

void ModifyBoundaryAttributesForNodeMovement(ParMesh *pmesh, ParGridFunction &x)
{
   const int dim = pmesh->Dimension();
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      mfem::Array<int> dofs;
      pmesh->GetNodalFESpace()->GetBdrElementDofs(i, dofs);
      mfem::Vector bdr_xy_data;
      mfem::Vector dof_xyz(dim);
      mfem::Vector dof_xyz_compare;
      mfem::Array<int> xyz_check(dim);
      for (int j = 0; j < dofs.Size(); j++)
      {
         for (int d = 0; d < dim; d++)
         {
            dof_xyz(d) = x(pmesh->GetNodalFESpace()->DofToVDof(dofs[j], d));
         }
         if (j == 0)
         {
            dof_xyz_compare = dof_xyz;
            xyz_check = 1;
         }
         else
         {
            for (int d = 0; d < dim; d++)
            {
               if (std::fabs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
               {
                  xyz_check[d] += 1;
               }
            }
         }
      }
      if (dim == 2)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
      }
      else if (dim == 3)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else if (xyz_check[2] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
   }
}

void ModifyAttributeForMarkingDOFS(ParMesh *pmesh, ParGridFunction &mat,
                                   int attr_to_switch)
{
   mat.ExchangeFaceNbrData();
   // Switch attribute if all but 1 of the faces of an element will be marked?
   Array<int> element_attr(pmesh->GetNE());
   element_attr = 0;
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      Array<int> faces, ori;
      if (pmesh->Dimension() == 2)
      {
         pmesh->GetElementEdges(e, faces, ori);
      }
      else
      {
         pmesh->GetElementFaces(e, faces, ori);
      }
      int inf1, inf2;
      int elem1, elem2;
      int diff_attr_count = 0;
      int attr1;
      int attr2;
      attr1 = mat(e);
      bool bdr_element = false;
      element_attr[e] = attr1;
      int target_attr = -1;
      for (int f = 0; f < faces.Size(); f++)
      {
         pmesh->GetFaceElements(faces[f], &elem1, &elem2);
         if (elem2 >= 0)
         {
            attr2 = elem1 == e ? (int)(mat(elem2)) : (int)(mat(elem1));
            if (attr1 != attr2 && attr1 == attr_to_switch)
            {
               diff_attr_count += 1;
               target_attr = attr2;
            }
         }
         else
         {
            pmesh->GetFaceInfos(faces[f], &inf1, &inf2);
            if (inf2 >= 0)
            {
               Vector dof_vals;
               Array<int> dofs;
               mat.GetElementDofValues(pmesh->GetNE() + (-1-elem2), dof_vals);
               attr2 = (int)(dof_vals(0));
               if (attr1 != attr2 && attr1 == attr_to_switch)
               {
                  diff_attr_count += 1;
                  target_attr = attr2;
               }
            }
            else
            {
               bdr_element = true;
            }
         }
      }

      if (diff_attr_count == faces.Size()-1 && !bdr_element)
      {
         element_attr[e] = target_attr;
      }
   }
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      mat(e) = element_attr[e];
      pmesh->SetAttribute(e, element_attr[e]+1);
   }
   mat.ExchangeFaceNbrData();
   pmesh->SetAttributes();
}
