#include "additive_schwarz.hpp"


// constructor
OverlappingCartesianMeshPartition::OverlappingCartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
{  // default overlap size is 2 elements 
   int dim = mesh->Dimension();
   int n = pow(mesh->GetNE(), 1.0/(double)dim);
   nx = 16;
   ny = 1;
   nz = 1;
   if (nx > n) 
   {
      nx = n;
      MFEM_WARNING("Changed partition in the x direction to nx = " << n << endl);
   }
   if (ny > n) 
   {
      ny = n;
      MFEM_WARNING("Changed partition in the y direction to ny = " << n << endl);
   }
   if (nz > n) 
   {
      nz = n;
      MFEM_WARNING("Changed partition in the z direction to nz = " << n << endl);
   } 
   if (dim == 2) nz = 1;
   int nxyz[3] = {nx,ny,nz};
   nrpatch = nx*ny*nz;
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(mesh);

   element_map.resize(nrpatch);

   double ppt[dim];
   Vector pt(ppt, dim);
   int nrelem = mesh->GetNE();

   for (int el = 0; el < nrelem; el++)
   {
      mesh->GetElementTransformation(el)->Transform(
         Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
      // Given the center coordinates determine the patches that this element contributes to
      Array<int> idx0(dim);
      Array<int> idx1(dim);
      Array<int> idx2(dim);
      vector<Array<int>> idx(3);
      if (dim == 2) idx[2].Append(0);

      for (int i = 0; i<dim; i++)
      {
         idx0[i]  = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
         idx1[i] = (int)floor(nxyz[i]*((pt(i)-2*h - pmin[i])/(pmax[i] - pmin[i])));
         idx2[i] = (int)floor(nxyz[i]*((pt(i)-h - pmin[i])/(pmax[i] - pmin[i])));

         if (idx0[i] < 0) idx0[i] = 0;
         if (idx0[i] >= nxyz[i]) idx0[i] = nxyz[i]-1;
         
         if (idx1[i] < 0) idx1[i] = 0;
         if (idx1[i] >= nxyz[i]) idx1[i] = nxyz[i]-1;

         if (idx2[i] < 0) idx2[i] = 0;
         if (idx2[i] >= nxyz[i]) idx2[i] = nxyz[i]-1;
         // convenient to put in one list
         idx[i].Append(idx0[i]);
         if (idx1[i] != idx0[i]) idx[i].Append(idx1[i]);
         if (idx2[i] != idx0[i] && idx2[i] != idx1[i]) idx[i].Append(idx2[i]);
      }
      // Now loop through all the combinations according to the idx above
      // in case of dim = 2 then kk = 0
      for (int k=0; k<idx[2].Size(); k++)
      {
         int kk = idx[2][k];
         for (int j=0; j<idx[1].Size(); j++)
         {
            int jj = idx[1][j];
            for (int i=0; i<idx[0].Size(); i++)
            {
               int ii = idx[0][i];
               int ip = kk*nxyz[0]*nxyz[1] + jj*nxyz[0]+ii;
               element_map[ip].Append(el);
            }
         }
      }
   }
}


// constructor
CartesianMeshPartition::CartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
{
   int dim = mesh->Dimension();
   nx = 5;
   ny = 1;
   nz = 1;
   int nxyz[3] = {nx,ny,nz};
   nrpatch = nx*ny*nz;

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


STPOverlappingCartesianMeshPartition::STPOverlappingCartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
{
   int dim = mesh->Dimension();
   nx = 9;
   ny = 1;
   nz = 1;
   int nxyz[3] = {nx,ny,nz};
   // nrpatch = nx*ny*nz;

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
   
   // element_map.resize(nrpatch);
   // for (int iel = 0; iel < nrelem; iel++)
   // {
   //    int ip = partitioning[iel];
   //    element_map[ip].Append(iel);
   // }
   // // Append the next subdomain to the previous
   // for (int ip = 0; ip<nrpatch-1; ip++)
   // {
   //    element_map[ip].Append(element_map[ip+1]);
   // }


   std::vector<Array<int>> elem_map;
   int npatch = nx*ny*nz;
   elem_map.resize(npatch);
   for (int iel = 0; iel < nrelem; iel++)
   {
      int ip = partitioning[iel];
      elem_map[ip].Append(iel);
   }
   // Append the next subdomain to the previous
   nrpatch = nx*ny*nz-1;
   element_map.resize(nrpatch);
   for (int ip = 0; ip<nrpatch; ip++)
   {
      element_map[ip].Append(elem_map[ip]);
      element_map[ip].Append(elem_map[ip+1]);
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
   partition_kind = part;
   if (part == 1)
   {
      cout << "Non Overlapping Cartesian Partition " << endl;
      CartesianMeshPartition partition(mesh);
      element_map = partition.element_map;
      nx = partition.nx;
      ny = partition.ny;
      nz = partition.nz;
   }
   // else if (part == 3 || part == 4)
   else if (part == 2)
   {
      cout << "Overlapping Cartesian Partition " << endl;
      OverlappingCartesianMeshPartition partition(mesh);
      element_map = partition.element_map;
      nx = partition.nx;
      ny = partition.ny;
      nz = partition.nz;
   }
   else if (part == 3 || part == 4)
   // else if (part == 2)
   {
      cout << "STP Overlapping Cartesian Partition " << endl;
      STPOverlappingCartesianMeshPartition partition(mesh);
      element_map = partition.element_map;
      nx = partition.nx;
      ny = partition.ny;
      nz = partition.nz;
   }
   else
   {
      cout << "Overlapping Vertex based partition " << endl;
      VertexMeshPartition partition(mesh);
      element_map = partition.element_map;
      partition_kind = 0;
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

      // Add the elements (for now search through all the vertices in the patch is needed)
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

void SaveMeshPartition(Array<Mesh *> meshes, string mfilename, string sfilename)
{
   int nrmeshes = meshes.Size();
   for (int ip = 0; ip<nrmeshes; ++ip)
   {
      cout << "saving mesh no " << ip << endl;
      ostringstream mesh_name;
      mesh_name << mfilename << setfill('0') << setw(6) << ip;
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      meshes[ip]->Print(mesh_ofs);
      L2_FECollection L2fec(1,meshes[ip]->Dimension());
      FiniteElementSpace L2fes(meshes[ip], &L2fec);
      GridFunction x(&L2fes); 
      
      ConstantCoefficient alpha((double)ip);
      x.ProjectCoefficient(alpha);
      ostringstream sol_name;
      sol_name << sfilename << setfill('0') << setw(6) << ip;
      ofstream sol_ofs(sol_name.str().c_str());
      x.Save(sol_ofs);
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
   // SaveMeshPartition(p->patch_mesh);
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



double GetUniformMeshElementSize(Mesh * mesh)
{
   int dim = mesh->Dimension();
   int nrelem = mesh->GetNE();

   DenseMatrix J(dim);
   double hmin, hmax;
   hmin = infinity();
   hmax = -infinity();
   Vector attr(nrelem);

   for (int iel=0; iel<nrelem; ++iel)
   {
      int geom = mesh->GetElementBaseGeometry(iel);
      ElementTransformation *T = mesh->GetElementTransformation(iel);
      T->SetIntPoint(&Geometries.GetCenter(geom));
      Geometries.JacToPerfJac(geom, T->Jacobian(), J);
      attr(iel) = J.Det();
      attr(iel) = pow(abs(attr(iel)), 1.0/double(dim));
      hmin = min(hmin, attr(iel));
      hmax = max(hmax, attr(iel));
   }

   MFEM_VERIFY(abs(hmin-hmax) < 1e-12, "Case not supported yet")

   return hmax;
}



Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions)
{
    // extrute on one dimension
   // flag = 1 +x, -1 -x, 2 +y, -2 +y , 3 +z, -3, -z

   // copy the original mesh;
   Mesh * mesh_orig = new Mesh(*mesh);
   if (!directions.Size()) return mesh_orig;
   
   int dim = mesh_orig->Dimension();

   Mesh * mesh_ext=nullptr;

   for (int j=0; j<directions.Size(); j++)
   {
      int d = directions[j];
      MFEM_VERIFY(abs(d)<= dim, "Cannot Extend in dimension " << d << ". Dim = " << dim << endl);

      Vector pmin;
      Vector pmax;
      mesh_orig->GetBoundingBox(pmin,pmax);

      // DenseMatrix J(dim);
      // double hmin, hmax;
      // hmin = infinity();
      // hmax = -infinity();
      // Vector attr(nrelem);
      // // element size

      // for (int iel=0; iel<nrelem; ++iel)
      // {
      //    int geom = mesh_orig->GetElementBaseGeometry(iel);
      //    ElementTransformation *T = mesh_orig->GetElementTransformation(iel);
      //    T->SetIntPoint(&Geometries.GetCenter(geom));
      //    Geometries.JacToPerfJac(geom, T->Jacobian(), J);
      //    attr(iel) = J.Det();
      //    attr(iel) = pow(abs(attr(iel)), 1.0/double(dim));
      //    hmin = min(hmin, attr(iel));
      //    hmax = max(hmax, attr(iel));
      // }
      // MFEM_VERIFY(hmin==hmax, "Case not supported yet")
      double h = GetUniformMeshElementSize(mesh_orig);
      double val;
      // find the vertices on the specific boundary
      switch (d)
      {
      case 1:
         val = pmax[0];
         break;
      case -1:
         val = pmin[0];
         h = -h;
         break;    
      case 2:
         val = pmax[1];
         break;
      case -2:
         val = pmin[1];
         h = -h;
         break;   
      case 3:
         val = pmax[2];
         break;
      case -3:
         val = pmin[2];
         h = -h;
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
               coords[0] = vert[0] + h;
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
               coords[1] = vert[1] + h;
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
               coords[2] = vert[2] + h;
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
