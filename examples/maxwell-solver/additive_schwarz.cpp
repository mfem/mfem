#include "additive_schwarz.hpp"

// constructor
CartesianMeshPartition::CartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
{
   int dim = mesh->Dimension();
   // int nx = sqrt(mesh->GetNE());
   int nx = 4;
   int ny = 4;
   int nz = 1;
   int nxyz[3] = {nx,ny,nz};
   nrpatch = nx*ny*nz;
   // double pmin[3] = { infinity(), infinity(), infinity() };
   // double pmax[3] = { -infinity(), -infinity(), -infinity() };

   // // find a bounding box using the vertices
   // for (int vi = 0; vi < mesh->GetNV(); vi++)
   // {
   //    const double *p = mesh->GetVertex(vi);
   //    for (int i = 0; i < dim; i++)
   //    {
   //       if (p[i] < pmin[i])
   //       {
   //          pmin[i] = p[i];
   //       }
   //       if (p[i] > pmax[i])
   //       {
   //          pmax[i] = p[i];
   //       }
   //    }
   // }
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);

   int nrelem = mesh->GetNE();
   int partitioning[nrelem];

   // determine the partitioning using the centers of the elements
   double ppt[dim];
   Vector pt(ppt, dim);
   for (int el = 0; el < nrelem; el++)
   {
      mesh->GetElementTransformation(el)->Transform(
         Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
      int part = 0;
      for (int i = dim-1; i >= 0; i--)
      {
         int idx = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
         if (idx < 0)
         {
            idx = 0;
         }
         if (idx >= nxyz[i])
         {
            idx = nxyz[i]-1;
         }
         part = part * nxyz[i] + idx;
      }
      partitioning[el] = part;
   }

   element_map.resize(nrpatch);
   for (int iel = 0; iel < nrelem; iel++)
   {
      int ip = partitioning[iel];
      element_map[ip].Append(iel);
   }
}

// constructor
VertexMeshPartition::VertexMeshPartition(Mesh *mesh_) : mesh(mesh_)
{
   nrpatch = mesh->GetNV();
   element_map.resize(nrpatch);
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
}


MeshPartition::MeshPartition(Mesh* mesh_, int part): mesh(mesh_)
{
   if (part)
   {
      cout << "Non Overlapping Cartesian Partition " << endl;
      CartesianMeshPartition partition(mesh);
      element_map = partition.element_map;
   }
   else
   {
      cout << "Overlapping Vertex based partition " << endl;
      VertexMeshPartition partition(mesh);
      element_map = partition.element_map;
   }

   nrpatch = element_map.size();
   int dim = mesh->Dimension();

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
      for (int iv = 0; iv<patch_nrvertices; ++iv)
      {
         int vert_idx = patch_vertices[iv];
         patch_mesh[ip]->AddVertex(mesh->GetVertex(vert_idx));
      }

      // Add the elements (for now search through all the vertices in the patch is need)
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
}


void MeshPartition::AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,
                                     int * ind)
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

void MeshPartition::PrintElementMap()
{
   mfem::out << "Element map" << endl;
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      mfem::out << "Patch No: " << ip;
      mfem::out << ", element map: " ;
      element_map[ip].Print(cout,element_map[ip].Size());
   }
}

void MeshPartition::SaveMeshPartition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      cout << "saving mesh no " << ip << endl;
      ostringstream mesh_name;
      mesh_name << "output/mesh." << setfill('0') << setw(6) << ip;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      patch_mesh[ip]->Print(mesh_ofs);
   }
}
MeshPartition::~MeshPartition()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete patch_mesh[ip];
      patch_mesh[ip] = nullptr;
   }
   patch_mesh.DeleteAll();
}

// constructor
PatchAssembly::PatchAssembly(BilinearForm *bf_, Array<int> & ess_tdofs, int part) : bf(bf_)
{
   fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();

   // list of dofs to distiguish between interior/boundary and essential
   Array<int> global_tdofs(fespace->GetTrueVSize());
   Array<int> bdr_tdofs(fespace->GetTrueVSize());
   global_tdofs = 0;
   // Mark boundary dofs and ess_dofs
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, bdr_tdofs);
   }

   // mark boundary dofs
   for (int i = 0; i<bdr_tdofs.Size(); i++) global_tdofs[bdr_tdofs[i]] = 1;
   // overwrite flag for essential dofs
   for (int i = 0; i<ess_tdofs.Size(); i++) global_tdofs[ess_tdofs[i]] = 0;

   MeshPartition * p = new MeshPartition(mesh, part);
   p->SaveMeshPartition();
   nrpatch = p->nrpatch;
   patch_fespaces.SetSize(nrpatch);
   patch_dof_map.resize(nrpatch);
   patch_mat.SetSize(nrpatch);
   patch_mat_inv.SetSize(nrpatch);
   ess_tdof_list.resize(nrpatch);
   ess_int_tdofs.resize(nrpatch);
   for (int ip=0; ip<nrpatch; ++ip)
   {
      // create finite element spaces for each patch
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
         MFEM_VERIFY(patch_elem_dofs.Size() == global_elem_dofs.Size(),
                     "Size inconsistency");
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
      }
      // Define the patch bilinear form and apply boundary conditions (only the LHS)
      Array <int> ess_temp_list;
      if (p->patch_mesh[ip]->bdr_attributes.Size())
      {
         Array<int> ess_bdr(p->patch_mesh[ip]->bdr_attributes.Max());
         ess_bdr = 1;
         patch_fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_temp_list);
      }

      // Adjust the essential tdof list for each patch
      for (int i=0; i<ess_temp_list.Size(); i++)
      {
         int ldof = ess_temp_list[i];
         int tdof = patch_dof_map[ip][ldof];
         // check the kind of this tdof
         if (!global_tdofs[tdof]) ess_tdof_list[ip].Append(ldof);
      }

      BilinearForm a(patch_fespaces[ip], bf);
      a.Assemble();
      OperatorPtr Alocal;
      a.FormSystemMatrix(ess_tdof_list[ip],Alocal);
      delete patch_fespaces[ip];
      patch_mat[ip] = new SparseMatrix((SparseMatrix&)(*Alocal));
      patch_mat[ip]->Threshold(0.0);
      // Save the inverse
      patch_mat_inv[ip] = new KLUSolver;
      patch_mat_inv[ip]->SetOperator(*patch_mat[ip]);
   }
   delete p;
}

void PatchAssembly::print_patch_dof_map()
{
   mfem::out << "Patch dof map" << endl;
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      mfem::out << "Patch No: " << ip;
      mfem::out << ", dof map: " ;
      patch_dof_map[ip].Print(cout,patch_dof_map[ip].Size());
   }
}

PatchAssembly::~PatchAssembly()
{
   for (int ip=0; ip<nrpatch; ++ip)
   {
      // delete patch_fespaces[ip]; patch_fespaces[ip]=nullptr;
      delete patch_mat_inv[ip];
      patch_mat_inv[ip]=nullptr;
      delete patch_mat[ip];
      patch_mat[ip]=nullptr;
   }
   patch_fespaces.DeleteAll();
   patch_mat.DeleteAll();
   patch_mat_inv.DeleteAll();
}

AddSchwarz::AddSchwarz(BilinearForm * bf_, Array<int> & global_ess_tdof_list, int i)
   : Solver(bf_->FESpace()->GetTrueVSize(), bf_->FESpace()->GetTrueVSize()),
     part(i)
{
   p = new PatchAssembly(bf_, global_ess_tdof_list,  part);
   nrpatch = p->nrpatch;
}

void AddSchwarz::Mult(const Vector &r, Vector &z) const
{
   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      for (int ip = 0; ip < nrpatch; ip++)
      {
         Array<int> * dof_map = &p->patch_dof_map[ip];
         int ndofs = dof_map->Size();
         res_local.SetSize(ndofs);
         sol_local.SetSize(ndofs);

         rnew.GetSubVector(*dof_map, res_local);
         Array<int> ess_bdr_indices = p->ess_tdof_list[ip];
         // for the overlapping case
         // zero out the entries corresponding to the ess_bdr
         p->patch_mat_inv[ip]->Mult(res_local, sol_local);
         if (!part) { sol_local.SetSubVector(ess_bdr_indices,0.0); }
         znew.AddElementVector(*dof_map,sol_local);
      }
      // Relaxation parameter
      znew *= theta;
      z += znew;
      // Update residual
      if (iter + 1 < maxit)
      {
         A->Mult(znew, raux);
         rnew -= raux;
      }
   }
}

AddSchwarz::~AddSchwarz()
{
   delete p;
}

Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions)
{
    // extrute on one dimension
   // flag = 1 +x, -1 -x, 2 +y, -2 +y , 3 +z, -3, -z

   // copy the original mesh;
   Mesh * mesh_orig = new Mesh(*mesh);
   int dim = mesh_orig->Dimension();

   Mesh * mesh_ext;

   for (int j=0; j<directions.Size(); j++)
   {
      int d = directions[j];
      MFEM_VERIFY(abs(d)<= dim, "Cannot Extend in dimension " << d << ". Dim = " << dim << endl);
      int nrelem = mesh_orig->GetNE();


      Vector pmin;
      Vector pmax;
      mesh_orig->GetBoundingBox(pmin,pmax);

      DenseMatrix J(dim);
      double hmin, hmax;
      hmin = infinity();
      hmax = -infinity();
      Vector attr(nrelem);
      // element size

      for (int iel=0; iel<nrelem; ++iel)
      {
         int geom = mesh_orig->GetElementBaseGeometry(iel);
         ElementTransformation *T = mesh_orig->GetElementTransformation(iel);
         T->SetIntPoint(&Geometries.GetCenter(geom));
         Geometries.JacToPerfJac(geom, T->Jacobian(), J);
         attr(iel) = J.Det();
         attr(iel) = pow(abs(attr(iel)), 1.0/double(dim));
         hmin = min(hmin, attr(iel));
         hmax = max(hmax, attr(iel));
      }
      MFEM_VERIFY(hmin==hmax, "Case not supported yet")

      double val;
      // find the vertices on the specific boundary
      switch (d)
      {
      case 1:
         val = pmax[0];
         break;
      case -1:
         val = pmin[0];
         hmax = -hmax;
         break;    
      case 2:
         val = pmax[1];
         break;
      case -2:
         val = pmin[1];
         hmax = -hmax;
         break;   
      case 3:
         val = pmax[2];
         break;
      case -3:
         val = pmin[2];
         hmax = -hmax;
         break;      
      }
      int k = 0;
      for (int i = 0; i<mesh_orig->GetNV(); ++i)
      {
         double * coords = mesh_orig->GetVertex(i);
         switch (abs(d))
         {
         case 1:
            if (coords[0] == val) k++;
            break;
         case 2:
            if (coords[1] == val) k++;
            break;   
         case 3:
            if (coords[2] == val) k++;
            break;      
         }
      }
      int nrvertices = mesh_orig->GetNV() + k;
      int nrelements = mesh_orig->GetNE() + pow(pow(k,1.0/(dim-1))-1.0,dim-1);

      mesh_ext = new Mesh(dim, nrvertices, nrelements);

      // Add existing vertices
      Array<int> vmap(mesh_orig->GetNV()); vmap = 0;
      k = mesh_orig->GetNV();
      for (int i=0; i<mesh_orig->GetNV(); ++i)
      {
         double * vert = mesh_orig->GetVertex(i);
         mesh_ext->AddVertex(vert);
         switch (abs(d))
         {
         case 1:
            if (vert[0] == val) 
            {
               vmap[i] = k;
               k++;
            }
            break;
         case 2:
            if (vert[1] == val)
            {
               vmap[i] = k;
               k++;
            }
            break;   
         case 3:
            if (vert[2] == val)
            {
               vmap[i] = k;
               k++;
            }
            break;      
         }
      } 
      // Add existing elements
      for (int i=0; i<mesh_orig->GetNE(); ++i)
      {
         Array<int>ind;
         mesh_orig->GetElementVertices(i,ind);
         if (dim == 2)
         {
            mesh_ext->AddQuad(ind);
         }
         else if (dim == 3)
         {
            mesh_ext->AddHex(ind);
         }
      } 
      // Add new vertices
      k = mesh_orig->GetNV();
      for (int i=0; i<mesh_orig->GetNV(); ++i)
      {
         double * vert = mesh_orig->GetVertex(i);
         switch (abs(d))
         {
         case 1:
            if (vert[0] == val) 
            {
               double coords[dim];
               coords[0] = vert[0] + hmax;
               coords[1] = vert[1];
               if (dim == 3) coords[2] = vert[2];
               mesh_ext->AddVertex(coords);
            }
            break;
         case 2:
            if (vert[1] == val)
            {
               double coords[dim];
               coords[0] = vert[0];
               coords[1] = vert[1] + hmax;
               if (dim == 3) coords[2] = vert[2];
               mesh_ext->AddVertex(coords);
            }
            break;   
         case 3:
            if (vert[2] == val)
            {
               double coords[dim];
               coords[0] = vert[0];
               coords[1] = vert[1];
               coords[2] = vert[2] + hmax;
               mesh_ext->AddVertex(coords);
            }
            break;      
         }
      }    
      // loop through boundary elements and extend in the given direction
      for (int i=0; i<mesh_orig->GetNBE(); ++i) 
      {
         Array<int> vertices;
         mesh_orig->GetBdrElementVertices(i,vertices);
         if (dim == 2)
         {
            int ind[4];
            if (vmap[vertices[0]] && vmap[vertices[1]])
            {
               ind[0] = vmap[vertices[0]];
               ind[1] = vmap[vertices[1]];
               ind[2] = vertices[1];
               ind[3] = vertices[0];
               mesh_ext->AddQuad(ind);
            }
         }
         else if (dim == 3)
         {
            int ind[8];
            if (vmap[vertices[0]] && vmap[vertices[1]] && vmap[vertices[2]] && vmap[vertices[3]])
            {
               ind[0] = vmap[vertices[0]];
               ind[1] = vmap[vertices[1]];
               ind[2] = vmap[vertices[2]];
               ind[3] = vmap[vertices[3]];
               ind[4] = vertices[0];
               ind[5] = vertices[1];
               ind[6] = vertices[2];
               ind[7] = vertices[3];
               mesh_ext->AddHex(ind);
            }
         }   
      }
      mesh_ext->FinalizeTopology();

      if (j<directions.Size()-1)
      {
         delete mesh_orig;
         mesh_orig = mesh_ext;
      }
   }
   delete mesh_orig;
   return mesh_ext;
}
