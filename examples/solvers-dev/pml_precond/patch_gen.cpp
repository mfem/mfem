
#include "patch_gen.hpp"


// constructor
mesh_partition::mesh_partition(Mesh *mesh_) : mesh(mesh_)
{
   nrpatch = mesh->GetNV();
   int dim = mesh->Dimension();
   element_map.SetSize(nrpatch);
   //every element will contribute to the the patches of its vertices
   // loop through the elements
   int nrelems = mesh->GetNE();
   for (int iel=0; iel<nrelems; ++iel)
   {
      // get element vertex index
      Array<int> vertices;
      mesh->GetElementVertices(iel,vertices);
      int nrvert = vertices.Size();
      // fill in the element contribution lists
      for (int iv = 0; iv< nrvert; ++iv)
      {
         int ip = vertices[iv];
         element_map[ip].Append(iel);
      }
   }
   // Print patch element list
   // for (int ip=0; ip<nrpatch; ++ip)
   // {
   //    cout << "ip: " << ip << ", element numbers: "; element_map[ip].Print(cout, 10);
   // }

   // Compute and store vertices coordinates of the global mesh
   mesh->EnsureNodes();
   GridFunction * nodes = mesh->GetNodes();
   int ndofs = nodes->FESpace()->GetNDofs();
   Array2D<double> coords(ndofs,dim);
   for (int comp = 0; comp < dim; comp++)
   {
      // cout << comp << endl;
      for (int i = 0; i < ndofs; i++)
      {
         coords(i,comp) = (*nodes)[nodes->FESpace()->DofToVDof(i, comp)];
      }
   }

   patch_mesh.SetSize(nrpatch);
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int patch_nrelems = element_map[ip].Size();
      element_map[ip].SetSize(patch_nrelems);
      // need to ensure that a vertex is not added more than once
      // and that the ordering of vertices is known for when the element is added
      // create a list of for this patch including possible repetitions
      // loop through elements in the patch 
      Array<int> patch_vertices;
      for (int iel=0; iel<patch_nrelems; ++iel)
      {
         // get the vertices list for the element
         Array<int> elem_vertices;
         int iel_idx = element_map[ip][iel];
         mesh->GetElementVertices(iel_idx,elem_vertices);
         patch_vertices.Append(elem_vertices);
      }
      patch_vertices.Sort();   
      patch_vertices.Unique();   
      int patch_nrvertices = patch_vertices.Size();

      // create the mesh
      patch_mesh[ip] = new Mesh(dim,patch_nrvertices,patch_nrelems);
      // Add the vertices
      double vert[dim];
      for (int iv = 0; iv<patch_nrvertices; ++iv)
      {
         int vert_idx = patch_vertices[iv];
         for (int comp=0; comp<dim; ++comp) vert[comp] = coords(vert_idx,comp);
         patch_mesh[ip]->AddVertex(vert);
      }

      // Add the elements (for now a search through all the vertices in the patch is need)
      for (int iel=0; iel<patch_nrelems; ++iel)
      {
         // get the vertices list for the element
         Array<int> elem_vertices;
         int iel_idx = element_map[ip][iel];
         mesh->GetElementVertices(iel_idx,elem_vertices);
         int nrvert = elem_vertices.Size();
         int ind[nrvert];
         for (int iv = 0; iv<nrvert; ++iv)
         {
            ind[iv] = patch_vertices.FindSorted(elem_vertices[iv]);   
         }
         mfem::Element::Type elem_type = mesh->GetElementType(element_map[ip][iel]);
         
         AddElementToMesh(patch_mesh[ip],elem_type,ind);
         
      }
      patch_mesh[ip]->FinalizeTopology();
   }
   // print_element_map();
   // save_mesh_partition();
}

void mesh_partition::AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,int * ind)
{
   switch (elem_type)
   {
   case Element::QUADRILATERAL:
      mesh->AddQuad(ind);
      break;
   case Element::TRIANGLE :
      mesh->AddTri(ind);
      break;
   case Element::HEXAHEDRON :
      mesh->AddHex(ind);
      break;
   case Element::TETRAHEDRON :
      mesh->AddTet(ind);
      break;
   default:
      MFEM_ABORT("Unknown element type");
      break;
   }
}

void mesh_partition::print_element_map()
{
   mfem::out << "Element map" << endl;
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      mfem::out << "Patch No: " << ip;
      mfem::out << ", element map: " ; element_map[ip].Print(cout,element_map[ip].Size()); 
   }
}

void mesh_partition::save_mesh_partition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      ostringstream mesh_name;
      mesh_name << "output/mesh." << setfill('0') << setw(6) << ip;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      patch_mesh[ip]->Print(mesh_ofs);
   }
}
mesh_partition::~mesh_partition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete patch_mesh[ip]; patch_mesh[ip] = nullptr;
   }
   patch_mesh.DeleteAll();
}



// constructor
patch_assembly::patch_assembly(FiniteElementSpace *fespace_) : fespace(fespace_)
{
   Mesh * mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   mesh_partition * p = new mesh_partition(mesh); 
   nrpatch = p->nrpatch;
   
   patch_fespaces.SetSize(nrpatch);
   // create finite element spaces for each patch
   // Table element_table = fespace->GetElementDofs()
   patch_dof_map.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ++ip)
   {
      patch_fespaces[ip] = new FiniteElementSpace(p->patch_mesh[ip],fec);

      // construct the patch tdof to global tdof map
      int nrdof = patch_fespaces[ip]->GetTrueVSize();
      patch_dof_map[ip].SetSize(nrdof);
      // loop through the elements in the patch
      for (int iel = 0; iel<p->element_map[ip].Size(); ++iel)
      {
         // index in the global mesh
         int iel_idx = p->element_map[ip][iel];
         // get the dofs of this element
         Array<int> patch_elem_dofs;
         Array<int> global_elem_dofs;
         patch_fespaces[ip]->GetElementDofs(iel,patch_elem_dofs);
         fespace->GetElementDofs(iel_idx,global_elem_dofs);
         // the sizes have to match
         MFEM_VERIFY(patch_elem_dofs.Size() == global_elem_dofs.Size(), "Size inconsistency");
         // loop through the dofs and take into account the signs;
         int ndof = patch_elem_dofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = patch_elem_dofs[i];
            int gdof_ = global_elem_dofs[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            patch_dof_map[ip][pdof] = gdof;
         }
         // TODO: remove the essential dofs
      }
   }

   print_patch_dof_map();
   delete p;
}

void patch_assembly::print_patch_dof_map()
{
   mfem::out << "Patch dof map" << endl;
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      mfem::out << "Patch No: " << ip;
      mfem::out << ", dof map: " ; patch_dof_map[ip].Print(cout,patch_dof_map[ip].Size()); 
   }
}

patch_assembly::~patch_assembly()
{
   for (int ip=0; ip<nrpatch; ++ip)
   {
      delete patch_fespaces[ip]; patch_fespaces[ip]=nullptr;
   }
   patch_fespaces.DeleteAll();

}