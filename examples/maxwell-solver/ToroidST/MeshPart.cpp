
#include "MeshPart.hpp"

// remove/leave elements with attributes given by attr
Mesh * GetPartMesh(const Mesh * mesh0, const Array<int> & attr_, Array<int> & elem_map,
 bool complement)
{
   Array<int> bdr_attr;
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


   Mesh * mesh = new Mesh(mesh0->Dimension(), mesh0->GetNV(), num_elements);
   // Copy vertices
   for (int v=0; v<mesh0->GetNV(); v++)
   {
      mesh->AddVertex(mesh0->GetVertex(v));
   }

   // Copy elements
   elem_map.SetSize(num_elements);
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
         elem_map[k++] = e;
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
   
   return mesh;
}
