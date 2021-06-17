
#include "mesh_partition.hpp"


SubMesh::SubMesh(const Mesh & mesh0, const Array<int> & elems) 
{
   int dim = mesh0.Dimension();
   int sdim = mesh0.SpaceDimension();
   // if mesh nodes are defined we use them for the vertices
   // otherwise we use the vertices them selfs
   int nv = mesh0.GetNV();
   int subelems = elems.Size();
   Array<int> vmarker(nv); vmarker = 0;
   int numvertices = 0;
   for (int ie = 0; ie<subelems; ie++)
   {
      int el = elems[ie];
      Array<int> vertices;
      mesh0.GetElementVertices(el,vertices);
      for (int iv=0; iv<vertices.Size(); iv++)
      {
         int v = vertices[iv];
         if (vmarker[v]) continue;
         vmarker[v] = 1;
         numvertices++;
      }
   }
   cout << "Num of new vertices: " << numvertices << endl;

   // Construct new mesh
   mesh = new Mesh(dim,numvertices, subelems, 0, sdim);

   const GridFunction * nodes0 = mesh0.GetNodes();
   int vk = 0;
   if (nodes0)
   {
      vmarker = 0;
      for (int ie = 0; ie<subelems; ie++)
      {
         int el = elems[ie];
         Array<int> vertices;
         mesh0.GetElementVertices(el,vertices);
         DenseMatrix val(sdim,vertices.Size());
         for (int d=0; d<sdim; d++)
         {
            Array<double> values;
            nodes0->GetNodalValues(el,values,d+1);
            val.SetRow(d,values.GetData());
         }

         for (int iv=0; iv<vertices.Size(); iv++)
         {
            int v = vertices[iv];
            if (vmarker[v]) continue;
            double * coords = val.GetColumn(iv);
            mesh->AddVertex(coords);
            vmarker[v] = ++vk;
         }
      }
   }
   else
   {
      for (int iv = 0; iv<mesh0.GetNV(); ++iv)
      {
         if (!vmarker[iv]) continue;
         mesh->AddVertex(mesh0.GetVertex(iv));
         vmarker[iv] = ++vk;
      }
   }

   // Add elements
   for (int ie = 0; ie<subelems; ie++)
   {
      const Element * el = mesh0.GetElement(elems[ie]);
      Element * nel = mesh->NewElement(el->GetGeometryType());
      int nv0 = el->GetNVertices();
      const int * v0 = el->GetVertices();
      int v1[nv0];
      for (int i=0; i<nv0; i++)
      {
         v1[i] = vmarker[v0[i]]-1;
      }   
      nel->SetVertices(v1);
      mesh->AddElement(nel);
   }
   mesh->FinalizeTopology();
   element_map = elems;


   if (nodes0)
   {
      // Extract Nodes GridFunction and determine its type
      const FiniteElementSpace * fes0 = nodes0->FESpace();

      Ordering::Type ordering = fes0->GetOrdering();
      int order = fes0->FEColl()->GetOrder();
      bool discont =
         dynamic_cast<const L2_FECollection*>(fes0->FEColl()) != NULL;

      // Set curvature of the same type as original mesh
      mesh->SetCurvature(order, discont, sdim, ordering);

      const FiniteElementSpace * fes = mesh->GetNodalFESpace();
      GridFunction * nodes = mesh->GetNodes();

      Array<int> vdofs0;
      Array<int> vdofs;
      Vector loc_vec;
      // Copy nodes to submesh
      for (int e = 0; e < element_map.Size(); e++)
      {
         fes0->GetElementVDofs(element_map[e], vdofs0);
         nodes0->GetSubVector(vdofs0, loc_vec);
         fes->GetElementVDofs(e, vdofs);
         nodes->SetSubVector(vdofs, loc_vec);
      }
   }
}
