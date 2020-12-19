
#include "MeshPart.hpp"

// remove/leave elements with attributes given by attr
Mesh * GetPartMesh(const Mesh * mesh0, const Array<int> & attr_, bool complement)
{
   Array<int> bdr_attr;
   bool visualization = 1;
   int max_attr     = mesh0->attributes.Max();
   int min_attr     = mesh0->attributes.Min();

   Array<int> attr;
   
   Array<int> all_attr(max_attr); all_attr = 0;
   for (int i = 0; i<attr_.Size(); i++)
   {
      all_attr[attr_[i]-1] = 1;
   }
   for (int i = min_attr; i<=max_attr; i++)
   {
      
      if (complement && all_attr[i-1]==0) attr.Append(i);
      if (!complement && all_attr[i-1]==1) attr.Append(i);
   }


   int max_bdr_attr = mesh0->bdr_attributes.Max();

   bdr_attr.SetSize(attr.Size());
   for (int i=0; i<attr.Size(); i++)
   {
      bdr_attr[i] = max_bdr_attr + attr[i];
   }

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
   for (int e=0; e<mesh0->GetNE(); e++)
   {
      int elem_attr = mesh0->GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1]) { num_elements++; }
   }

   // Count the number of boundary elements in the final mesh
   int num_bdr_elements = 0;
   for (int f=0; f<mesh0->GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh0->GetFaceElements(f, &e1, &e2);
      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh0->GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh0->GetElement(e2)->GetAttribute(); }

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

   Mesh * mesh = new Mesh(mesh0->Dimension(), mesh0->GetNV(),
                          num_elements, num_bdr_elements, mesh0->SpaceDimension());

   // Mesh * mesh = new Mesh(mesh0->Dimension(), mesh0->GetNV(), num_elements);
   // Copy vertices
   for (int v=0; v<mesh0->GetNV(); v++)
   {
      mesh->AddVertex(mesh0->GetVertex(v));
   }

   // Copy elements
   int k = 0;
   for (int e=0; e<mesh0->GetNE(); e++)
   {
      const Element * el = mesh0->GetElement(e);
      
      int elem_attr = el->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nel = mesh->NewElement(el->GetGeometryType());
         nel->SetAttribute(elem_attr);
         nel->SetVertices(el->GetVertices());
         mesh->AddElement(nel);
      }
   }

   // Copy selected boundary elements
   for (int be=0; be<mesh0->GetNBE(); be++)
   {
      int e, info;
      mesh0->GetBdrElementAdjacentElement(be, e, info);
      int elem_attr = mesh0->GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nbel = mesh0->GetBdrElement(be)->Duplicate(mesh);
         mesh->AddBdrElement(nbel);
      }
   }

   // Create new boundary elements
   for (int f=0; f<mesh0->GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh0->GetFaceElements(f, &e1, &e2);

      int i1 = -1, i2 = -1;
      mesh0->GetFaceInfos(f, &i1, &i2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh0->GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh0->GetElement(e2)->GetAttribute(); }

      if (a1 != 0 && a2 != 0)
      {
         if (marker[a1-1] && !marker[a2-1])
         {
            Element * bel = (mesh0->Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh0->GetFace(f)->Duplicate(mesh);
            bel->SetAttribute(bdr_attr[attr_inv[a1-1]]);
            mesh->AddBdrElement(bel);
         }
         else if (!marker[a1-1] && marker[a2-1])
         {
            Element * bel = (mesh0->Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh0->GetFace(f)->Duplicate(mesh);
            bel->SetAttribute(bdr_attr[attr_inv[a2-1]]);
            mesh->AddBdrElement(bel);
         }
      }
   }

   mesh->FinalizeTopology();
   mesh->RemoveUnusedVertices();

   const GridFunction * nodes0 = mesh0->GetNodes();

   int order = nodes0->FESpace()->GetOrder(0);
   if (order > 1)
   {
      mesh->SetCurvature(order, false, 3, Ordering::byVDIM);
   }

   GridFunction * nodes = mesh->GetNodes();
   int nel = mesh0->GetNE();
   // copy nodes
   int jel = 0;
   for (int iel = 0; iel< nel; iel++)
   {
      int elem_attr = mesh0->GetElement(iel)->GetAttribute();
      if (!marker[elem_attr-1]) 
      {
         Array<int> vdofs0,vdofs;
         nodes0->FESpace()->GetElementVDofs(iel,vdofs0);
         Vector x;
         nodes0->GetSubVector(vdofs0,x);
         nodes->FESpace()->GetElementVDofs(jel++,vdofs);
         nodes->SetSubVector(vdofs,x);
      }
   }
   
   if (visualization)
   {
      // GLVis server to visualize to
      char vishost[] = "localhost";
      int  visport   = 19916;

      socketstream mesh0_sock(vishost, visport);
      mesh0_sock.precision(8);
      mesh0_sock << "mesh\n" << *mesh0 << flush;

      socketstream mesh_sock(vishost, visport);
      mesh_sock.precision(8);
      mesh_sock << "mesh\n" << *mesh << flush;
   }

   return mesh;
}
