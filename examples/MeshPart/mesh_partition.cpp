
#include "mesh_partition.hpp"


Subdomain::Subdomain(const Mesh & mesh0_) 
: mesh0(&mesh0_), dim(mesh0->Dimension()), sdim(mesh0->SpaceDimension()) 
{ 
   // MFEM_VERIFY(dim == 3, "Only 3D domains for now are supported");
   if(mesh0->NURBSext)
   {
      MFEM_ABORT("Nurbs meshes are not supported yet");
   }
}

void Subdomain::BuildSubMesh(const Array<int> & elems, const entity_type & etype)
{
   // if mesh nodes are defined we use them for the vertices
   // otherwise we use the vertices them selfs
   cout << "entity type " << etype << endl;


   int nv = mesh0->GetNV();
   int subelems = elems.Size();
   Array<int> vmarker(nv); vmarker = 0;
   int numvertices = 0;
   for (int ie = 0; ie<subelems; ie++)
   {
      int el = elems[ie];
      Array<int> vertices;
      switch (etype)
      {
      case 0: mesh0->GetElementVertices(el,vertices); break;
      case 1: 
         {
            // mesh0->GetBdrElementVertices(el,vertices); 
            int face = mesh0->GetBdrFace(el);
            mesh0->GetFaceVertices(face,vertices);
         }
         break;
      case 2: mesh0->GetFaceVertices(el,vertices); break;
      default:
         MFEM_ABORT("Wrong entity type choice");
         break;
      }
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
   Mesh * meshptr = nullptr;
   switch (etype)
   {
   case 0: 
      mesh = new Mesh(dim,numvertices, subelems); 
      element_map = elems;
      meshptr = mesh;
      break;
   case 1: 
      bdr_mesh = new Mesh(dim-1,numvertices, subelems, 0, sdim);
      bdr_element_map = elems;
      meshptr = bdr_mesh;
      break;   
   case 2: 
      surface_mesh = new Mesh(dim-1,numvertices, subelems,0,sdim);
      face_element_map = elems;
      meshptr = surface_mesh;
      break;
      default:
         MFEM_ABORT("Wrong entity type choice");
         break;
   }

   Vector values;
   const GridFunction * nodes0 = mesh0->GetNodes();
   if (nodes0)
   {
      vcoords.SetSize(sdim, mesh0->GetNV());
      for (int i = 0; i< sdim; i++)
      {
         nodes0->GetNodalValues(values,i+1);
         vcoords.SetRow(i,values);
         cout << "values size = " << values.Size() << endl;
      }
   }
   int vk = 0;
   if (nodes0) 
   {
      for (int iv = 0; iv<mesh0->GetNV(); ++iv)
      {
         if (!vmarker[iv]) continue;
         meshptr->AddVertex(vcoords.GetColumn(iv));
         vmarker[iv] = ++vk;
      }
   }
   else
   {
      for (int iv = 0; iv<mesh0->GetNV(); ++iv)
      {
         if (!vmarker[iv]) continue;
         meshptr->AddVertex(mesh0->GetVertex(iv));
         vmarker[iv] = ++vk;
      }
   }
   // Add elements
   for (int ie = 0; ie<subelems; ie++)
   {
      const Element * el = nullptr;
      switch (etype)
      {
         case 0: el = mesh0->GetElement(elems[ie]); break;
         case 1: 
            {
               // el = mesh0->GetBdrElement(elems[ie]);
               int face = mesh0->GetBdrFace(elems[ie]);
               el = mesh0->GetFace(face);
               // int face = mesh0->GetBdrFace(el);
               // mesh0->GetFaceVertices(face,vertices);
            }
            break;
         case 2: el = mesh0->GetFace(elems[ie]); break;
         default: MFEM_ABORT("Wrong entity type choice"); break;
      }
      Element * nel = meshptr->NewElement(el->GetGeometryType());
      int nv0 = el->GetNVertices();
      const int * v0 = el->GetVertices();
      Array<int> v1(nv0);
      for (int i=0; i<nv0; i++)
      {
         v1[i] = vmarker[v0[i]]-1;
      }   
      nel->SetVertices(v1.GetData());
      meshptr->AddElement(nel);
   }
   meshptr->FinalizeTopology();
   
   if (nodes0)
   {
      cout << "nodes not null" << endl;
      // Extract Nodes GridFunction and determine its type
      const FiniteElementSpace * fes0 = nodes0->FESpace();

      Ordering::Type ordering = fes0->GetOrdering();
      int order = fes0->FEColl()->GetOrder();
      bool discont = fes0->IsDGSpace();   

      cout << "discont = " << discont << endl;
      // Set curvature of the same type as original mesh
      meshptr->SetCurvature(order, discont, sdim, ordering);

      const FiniteElementSpace * fes1 = meshptr->GetNodalFESpace();
      GridFunction * nodes = meshptr->GetNodes();

      Array<int> vdofs0;
      Array<int> vdofs;
      Vector loc_vec;
      
      // Copy nodes to submesh
      for (int e = 0; e < elems.Size(); e++)
      {
         switch (etype)
         {
         case 0: 
            fes0->GetElementVDofs(elems[e], vdofs0); 
            nodes0->GetSubVector(vdofs0, loc_vec);
            fes1->GetElementVDofs(e, vdofs);
            nodes->SetSubVector(vdofs, loc_vec);
            break;
         case 1: 
            if (!discont && false)
            {
               fes0->GetBdrElementVDofs(elems[e], vdofs0); 
               nodes0->GetSubVector(vdofs0, loc_vec);
               fes1->GetElementVDofs(e, vdofs);
               nodes->SetSubVector(vdofs, loc_vec);
            }
            else
            {
               const FiniteElement * el = fes1->GetFE(e);
               const IntegrationRule & ir = el->GetNodes(); 
               int np = ir.GetNPoints();
               FaceElementTransformations * Tr = const_cast<Mesh *>(mesh0)->GetBdrFaceTransformations(elems[e]);

               int el1 = Tr->Elem1No;
               Array<int> vdofs1;
               fes1->GetElementVDofs(e,vdofs1);
               loc_vec.SetSize(vdofs1.Size());
               for (int i = 0; i<np; i++)
               {
                  Tr->SetAllIntPoints(&ir[i]);
                  const IntegrationPoint & ip = Tr->GetElement1IntPoint();
                  Vector val;
                  nodes0->GetVectorValue(el1,ip,val);
                  for (int j = 0; j<val.Size(); j++)
                  {
                     // (*nodes)[vdofs1[i+j*np]] = val[j];
                     // (*nodes)[vdofs1[i+j*np]] = val[j];
                     loc_vec[i+j*np] = val[j];
                  }
               }
               nodes->SetSubVector(vdofs1, loc_vec);
            }
            break;
         case 2: fes0->GetFaceVDofs(elems[e], vdofs0); break;
         default:
            MFEM_ABORT("Wrong entity type choice");
            break;
         }

      }
   }
}


void Subdomain::BuildDofMap(const entity_type & etype)
{
   Array<int> elems;
   FiniteElementSpace * fesptr = nullptr;
   const FiniteElementCollection *fec = fes0->FEColl();
   switch(etype)
   {
   case 0: 
      fesptr = new FiniteElementSpace(mesh,fec);
      elems = element_map;
      break;
   case 1: 
      fesptr = new FiniteElementSpace(bdr_mesh,fec);
      elems = bdr_element_map;
      break;
   case 2: 
      fesptr = new FiniteElementSpace(surface_mesh,fec);
      elems = face_element_map;
      break;
   default:
      MFEM_ABORT("Wrong entity type choice");
      break;
   }
   
   Array<int> dofs(fesptr->GetTrueVSize());
   for (int iel = 0; iel<elems.Size(); ++iel)
   {
      // index in the global mesh
      int iel_idx = elems[iel];
      // get the dofs of this element
      Array<int> ldofs;
      Array<int> gdofs;
      switch(etype)
      {
         case 0: fes0->GetElementVDofs(iel_idx,gdofs); break;
         case 1: 
            {
               // fes0->GetBdrElementVDofs(iel_idx,gdofs); 
               int face = mesh0->GetBdrFace(iel_idx);
               fes0->GetFaceVDofs(face,gdofs);
            }
            break;
         case 2: fes0->GetFaceVDofs(iel_idx,gdofs); break;
         default:  MFEM_ABORT("Wrong entity type");  break;
      }
      fesptr->GetElementDofs(iel,ldofs);
         // the sizes have to match
      MFEM_VERIFY(gdofs.Size() == ldofs.Size(),
                  "Size inconsistency");
     // loop through the dofs and take into account the signs;
      int ndof = ldofs.Size();
      for (int i = 0; i<ndof; ++i)
      {
         int ldof_ = ldofs[i];
         int gdof_ = gdofs[i];
         int ldof = (ldof_ >= 0) ? ldof_ : abs(ldof_) - 1;
         int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
         dofs[ldof] = gdof;
      }
   }
   switch(etype)
   {
   case 0: 
      dof_map = dofs; 
      fes = fesptr;
      break;
   case 1: 
      bdr_dof_map = dofs; 
      bdr_fes = fesptr;
      break;
   case 2: 
      face_dof_map = dofs; 
      surface_fes = fesptr;
      break;
   default:  
      MFEM_ABORT("Wrong entity type");  break;
   }
}

void Subdomain::BuildProlongationMatrix(const entity_type & etype)
{
   Array<int> dofs;
   SparseMatrix * Ptr = nullptr;
   switch (etype)
   {
   case 0: 
      if (!dof_map.Size()) BuildDofMap(etype);
      dofs = dof_map; 
      Ptr = P;
      break;
   case 1: 
      if (!bdr_dof_map.Size()) BuildDofMap(etype);
      dofs = bdr_dof_map; 
      Ptr = Pb;
      break;
   case 2: 
      if (!face_dof_map.Size()) BuildDofMap(etype);
      dofs = face_dof_map; 
      Ptr = Pf;
      break;
   default: 
      MFEM_ABORT("Wrong entity type"); 
      break;
   }

   int height = fes0->GetTrueVSize();
   int width = dofs.Size();
   Ptr = new SparseMatrix(height,width);
   for (int i = 0; i< dofs.Size(); i++)
   {
      int j = dofs[i];
      Ptr->Set(j,i,1.);
   }
   Ptr->Finalize();

   switch (etype)
   {
      case 0: P = Ptr; break;
      case 1: Pb = Ptr; break;
      case 2: Pf = Ptr; break;
      default: MFEM_ABORT("Wrong entity type"); break;
   }

}

Subdomain::~Subdomain()
{
   delete mesh;
   delete bdr_mesh;
   delete surface_mesh;
   delete fes; 
   delete bdr_fes; 
   delete surface_fes;
   delete P; 
   delete Pb; 
   delete Pf; 
}
