#include "MeshPartition.hpp"

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
   // cout << "hmin , hmax = " << hmin << ", " << hmax << endl;
   if (nrelem)
   {
      MFEM_VERIFY(abs(hmin-hmax) < 1e-12, "Case not supported yet")
   }   
   else
   {
      hmin = hmax = 0.0;
   }

   // cout << hmax << endl;
#ifdef MFEM_USE_MPI
   ParMesh * pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh)
      MPI_Allreduce(MPI_IN_PLACE, &hmax,1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());

#endif
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



      // cout << "nrvertices : " << mesh_orig->GetNV() << endl;
      // cout << "nrvertices : " << nrvertices << endl;
      // cout << "nrelements : " << nrelements << endl;

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
               ind[0] = vertices[1];
               ind[1] = vertices[0];
               ind[2] = vmap[vertices[0]];
               ind[3] = vmap[vertices[1]];
               mesh_ext->AddQuad(ind);
            }
         }
         else if (dim == 3)
         {
            int ind[8];
            if (vmap[vertices[0]] && vmap[vertices[1]] && vmap[vertices[2]] && vmap[vertices[3]])
            {
               ind[0] = vertices[0];
               ind[1] = vertices[1];
               ind[2] = vertices[2];
               ind[3] = vertices[3];
               ind[4] = vmap[vertices[0]];
               ind[5] = vmap[vertices[1]];
               ind[6] = vmap[vertices[2]];
               ind[7] = vmap[vertices[3]];
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

// constructor
   
OverlappingCartesianMeshPartition::OverlappingCartesianMeshPartition(Mesh *mesh_,int & nx,int & ny,int & nz) : mesh(mesh_)
{  // default overlap size is 2 elements 
   int dim = mesh->Dimension();
   int n = pow(mesh->GetNE(), 1.0/(double)dim);
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
   subdomains.SetSize(nx,ny,nz);
   nxyz[0] = nx; nxyz[1]=ny; nxyz[2] = nz;
   nrpatch = nx*ny*nz;
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(mesh);
   MeshSize = h;

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
         idx1[i] = (int)floor(nxyz[i]*((pt(i)+h - pmin[i])/(pmax[i] - pmin[i])));
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
   for (int k = 0; k<nz; k++)
   {
      for (int j = 0; j<ny; j++)
      {
         for (int i = 0; i<nx; i++)
         {
            subdomains(i,j,k) = k*ny*nx + j*nx + i;
         }
      }   
   }
}


OverlappingCartesianMeshPartition::OverlappingCartesianMeshPartition
(Mesh *mesh_,int & nx,int & ny,int & nz, int ovlp_nlayers) : mesh(mesh_)
{  // default overlap size is 2 elements 
   int dim = mesh->Dimension();
   // int n = pow(mesh->GetNE(), 1.0/(double)dim);
   // if (nx > n) 
   // {
   //    nx = n;
   //    MFEM_WARNING("Changed partition in the x direction to nx = " << n << endl);
   // }
   // if (ny > n) 
   // {
   //    ny = n;
   //    MFEM_WARNING("Changed partition in the y direction to ny = " << n << endl);
   // }
   // if (nz > n) 
   // {
   //    nz = n;
   //    MFEM_WARNING("Changed partition in the z direction to nz = " << n << endl);
   // } 
   if (dim == 2) nz = 1;
   subdomains.SetSize(nx,ny,nz);
   nxyz[0] = nx; nxyz[1]=ny; nxyz[2] = nz;
   nrpatch = nx*ny*nz;
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(mesh);
   MeshSize = h;
   cout << "h = " << h << endl;

   // Check that ovlp_size does not exit subdomain size
   MFEM_VERIFY((pmax[0]-pmin[0])/nx >= h*ovlp_nlayers, 
                "Check ovlp size in partition"); 
                cout << "pmax[0]-pmin[0])/nx = " << (pmax[0]-pmin[0])/nx << endl;
                cout << "ovlp_nlayers = " << ovlp_nlayers << endl;
                cout << "h*ovlp_nlayers = " << h*ovlp_nlayers << endl;
   MFEM_VERIFY((pmax[1]-pmin[1])/ny >= h*ovlp_nlayers, 
               "Check ovlp size in partition"); 
   if (dim == 3)
   {
      MFEM_VERIFY((pmax[2]-pmin[2])/nz >= h*ovlp_nlayers, 
               "Check ovlp size in partition"); 
   }
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
         idx1[i] = (int)floor(nxyz[i]*((pt(i)+ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));
         idx2[i] = (int)floor(nxyz[i]*((pt(i)-ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));

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
   for (int k = 0; k<nz; k++)
   {
      for (int j = 0; j<ny; j++)
      {
         for (int i = 0; i<nx; i++)
         {
            subdomains(i,j,k) = k*ny*nx + j*nx + i;
         }
      }   
   }
}

// constructor
CartesianMeshPartition::CartesianMeshPartition(Mesh *mesh_,int & nx, int & ny, int & nz) : mesh(mesh_)
{
   int dim = mesh->Dimension();
   nxyz[0] = nx;
   nxyz[1] = ny;
   nxyz[2] = nz;
   nrpatch = nx*ny*nz;
   subdomains.SetSize(nx,ny,nz);
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
   for (int k = 0; k<nz; k++)
   {
      for (int j = 0; j<ny; j++)
      {
         for (int i = 0; i<nx; i++)
         {
            subdomains(i,j,k) = k*ny*nx + j*nx + i;
         }
      }   
   }
}

STPOverlappingCartesianMeshPartition::STPOverlappingCartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
{
   int dim = mesh->Dimension();
   nx = 5;
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

MeshPartition::MeshPartition(Mesh* mesh_, int part,int nx, int ny, int nz, int nrlayers): mesh(mesh_)
{
   partition_kind = part;
   if (part == 1)
   {
      cout << "Non Overlapping Cartesian Partition " << endl;
      CartesianMeshPartition partition(mesh,nx, ny, nz);
      element_map = partition.element_map;
      // subdomains = partition.subdomains;
   }
   // else if (part == 3 || part == 4)
   else if (part == 2)
   {
      cout << "Overlapping Cartesian Partition " << endl;
      OverlappingCartesianMeshPartition partition(mesh,nx, ny, nz,nrlayers);
      element_map = partition.element_map;
      subdomains = partition.subdomains;
      nxyz[0] = partition.nxyz[0];
      nxyz[1] = partition.nxyz[1];
      nxyz[2] = partition.nxyz[2];
      MeshSize = partition.MeshSize;
   }
   else if (part == 3 || part == 4)
   // else if (part == 2)
   {
      cout << "STP Overlapping Cartesian Partition " << endl;
      STPOverlappingCartesianMeshPartition partition(mesh);
      element_map = partition.element_map;
   }
   else
   {
      MFEM_ABORT("Overlapping Vertex based partition not supprorted")
   }

   nrpatch = element_map.size();

   int dim = mesh->Dimension();

   patch_mesh.SetSize(nrpatch);
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      int patch_nrelems = element_map[ip].Size();
      element_map[ip].SetSize(patch_nrelems);
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

void SaveMesh(Mesh * mesh, string mfilename)
{
   cout << "saving global mesh " << endl;
   ostringstream mesh_name;
   mesh_name << mfilename;
   ofstream mesh_ofs(mesh_name.str().c_str());
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
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


#ifdef MFEM_USE_MPI


CartesianParMeshPartition::CartesianParMeshPartition(ParMesh * pmesh_,
                                                     int & nx, 
                                                     int & ny, 
                                                     int & nz,
                                                     int ovlp_nlayers) : pmesh(pmesh_)
{
   int num_procs,myid;
   MPI_Comm comm = pmesh->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);


   int dim = pmesh->Dimension();
   if (dim == 2 ) nz = 1;
   subdomains.SetSize(nx,ny,nz);
   nxyz[0] = nx; nxyz[1]=ny; nxyz[2] = nz;
   nrsubdomains = nx*ny*nz;
   cout << "nrsubdomains = " << nrsubdomains << endl;
   Vector pmin, pmax;
   pmesh->GetBoundingBox(pmin, pmax);
   double h = pmesh->GetNE() ? pmesh->GetElementSize(0) : 0.0;
   MeshSize = h;

   // Check that ovlp_size does not exit subdomain size
   for (int d = 0; d<dim; d++)
   {
      MFEM_VERIFY((pmax[d]-pmin[d])/nxyz[d] >= h*ovlp_nlayers, 
                  "Check ovlp size in partition"); 
   }
   local_element_map.resize(nrsubdomains);

   double ppt[dim];
   Vector pt(ppt, dim);
   int mynrelem = pmesh->GetNE();

   int myelem_offset;
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, comm);
   myelem_offset -= mynrelem;

   for (int el = 0; el < mynrelem; el++)
   {
      pmesh->GetElementTransformation(el)->Transform(
         Geometries.GetCenter(pmesh->GetElementBaseGeometry(el)), pt);
      // Given the center coordinates determine the patches that this element contributes to
      Array<int> idx0(dim);
      Array<int> idx1(dim);
      Array<int> idx2(dim);
      vector<Array<int>> idx(3);
      if (dim == 2) idx[2].Append(0);

      for (int i = 0; i<dim; i++)
      {
         idx0[i]  = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
         idx1[i] = (int)floor(nxyz[i]*((pt(i)+ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));
         idx2[i] = (int)floor(nxyz[i]*((pt(i)-ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));

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
               local_element_map[ip].Append(el+myelem_offset);
            }
         }
      }
   }

   Array<int>subdomain_size(nrsubdomains);
   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      subdomain_size[ip] = local_element_map[ip].Size();
   }

   Array<int>subdomain_ranks(nrsubdomains*num_procs);
   MPI_Allgather(subdomain_size, nrsubdomains, MPI_INT, 
                 subdomain_ranks, nrsubdomains, MPI_INT, comm);

   Array<int> max(nrsubdomains);
   max = -1;
   subdomain_rank.SetSize(nrsubdomains);
   subdomain_rank = -1;
   // loop through the patches and determine the rank with the max number of elements
   for (int irank = 0; irank < num_procs; ++irank)
   {
      int offset = irank*nrsubdomains;
      for (int ip = 0; ip<nrsubdomains; ++ip)
      {
         if (subdomain_ranks[ip+offset]>= max[ip])
         {
            max[ip] = subdomain_ranks[ip+offset];
            subdomain_rank[ip] = irank;
         }
      }
   }
   
   subdomains.SetSize(nx,ny,nz);
   for (int k = 0; k<nz; k++)
   {
      for (int j = 0; j<ny; j++)
      {
         for (int i = 0; i<nx; i++)
         {
            subdomains(i,j,k) = k*ny*nx + j*nx + i;
         }
      }   
   }
}

ParMeshPartition::ParMeshPartition(ParMesh* pmesh_,
         int nx, int ny, int nz, int nrlayers) : pmesh(pmesh_), OvlpNlayers(nrlayers)
{

   int num_procs,myid;
   comm = pmesh->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   int dim = pmesh->Dimension();
   CartesianParMeshPartition partition(pmesh,nx, ny, nz,nrlayers);

   local_element_map = partition.local_element_map;
   subdomains = partition.subdomains;
   nxyz[0] = partition.nxyz[0];
   nxyz[1] = partition.nxyz[1];
   nxyz[2] = partition.nxyz[2];
   MeshSize = partition.MeshSize;
   subdomain_rank = partition.subdomain_rank;

   nrsubdomains = partition.local_element_map.size();
   int mynrelem = pmesh->GetNE();
   MPI_Scan(&mynrelem, &myelem_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   myelem_offset -= mynrelem;

   // communicate the element map to every processor that is involved
   element_map.resize(nrsubdomains);
   // Alternative way to construct element_map on the host rank
   // Each contributing process sends the contents of local_element_map
   // to the host rank
   Array<int> send_count(num_procs); send_count = 0;
   Array<int> send_displ(num_procs); send_displ = 0;
   Array<int> recv_count(num_procs); recv_count = 0;
   Array<int> recv_displ(num_procs); recv_displ = 0;
   
   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      // a) subdomain number
      // b) subdomain size (local) number of elements
      // c) the list of elements
      if (local_element_map[ip].Size())
      {
         send_count[subdomain_rank[ip]] += 1+1+local_element_map[ip].Size();
      }
   }
   
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();
   Array<int> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs);     soffs = 0;

   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      // a) subdomain number
      // b) subdomain size (local) number of elements
      // c) the list of elements
      if (local_element_map[ip].Size())
      {
         int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
         sendbuf[j] = ip;
         sendbuf[j+1] = local_element_map[ip].Size();
         for (int i=0; i<local_element_map[ip].Size(); i++)
         {
            sendbuf[j+2+i] = local_element_map[ip][i];
         }
         soffs[subdomain_rank[ip]] += 1 + 1 + local_element_map[ip].Size();
      }
   }


   Array<int> recvbuf(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);

   // recv_count.Print();

   // Extract from recvbuff
   int k=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k]; 
      int ipsize = recvbuf[k+1];
      for (int i=0; i<ipsize; i++)
      {
         element_map[ip].Append(recvbuf[k+2+i]);
      }
      k += 1+1+ipsize;
   }

   // for (int ip = 0; ip < nrsubdomains; ++ip)
   // for (int ip = 0; ip < 1; ++ip)
   // {
   //    if (myid == subdomain_rank[ip]) 
   //    {
   //       cout << "myid, ip " << myid << ", " << ip 
   //       << ", elems = " ; element_map[ip].Print(cout,20); 
   //    }
   // }


   // Now each process sends the vertex coords and elements 
   // to the subdomain's host rank
   send_count = 0;
   send_displ = 0;
   recv_count = 0;
   recv_displ = 0;

   // send buffer for coordinates
   Array<int> send_count_d(num_procs); send_count_d = 0;
   Array<int> send_displ_d(num_procs); send_displ_d = 0;
   Array<int> recv_count_d(num_procs); recv_count_d = 0;
   Array<int> recv_displ_d(num_procs); recv_displ_d = 0;

   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      // a) patch no
      // b) element global number
      // c) type of the element (int)
      // d) number of vertices
      // e) global index of vertices
      // f) the coordinates of the vertices (x,y,z) 
      //---------------------------------------------
      // get local element_map size
      int subdomain_local_nelems = local_element_map[ip].Size();
      if (subdomain_local_nelems) // the rank is contributing to the subdomain ip
      {
         // loop through the elements
         for (int iel=0; iel<subdomain_local_nelems; ++iel)
         {
            // get the vertices list for the element
            Array<int> elem_vertices;
            int iel_idx = local_element_map[ip][iel]-myelem_offset;
            pmesh->GetElementVertices(iel_idx,elem_vertices);
            int nrvert = elem_vertices.Size();
            send_count[subdomain_rank[ip]] += 1 + 1 + 1 + 1 + nrvert;
            send_count_d[subdomain_rank[ip]] += dim * nrvert;
         }
      }
   }

   // communicate so that recv_count is constructed
   MPI_Alltoall(send_count,1,MPI_INT,recv_count,1,MPI_INT,comm);
   MPI_Alltoall(send_count_d,1,MPI_INT,recv_count_d,1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
      send_displ_d[k+1] = send_displ_d[k] + send_count_d[k];
      recv_displ_d[k+1] = recv_displ_d[k] + recv_count_d[k];
   }
   sbuff_size = send_count.Sum();
   rbuff_size = recv_count.Sum();
   int sbuff_size_d = send_count_d.Sum();
   int rbuff_size_d = recv_count_d.Sum();

   // now allocate space for the send buffer
   sendbuf.SetSize(sbuff_size); sendbuf = 0;
   soffs = 0;

   Array<double> sendbuf_d(sbuff_size_d);  sendbuf_d = 0.0;
   Array<int> soffs_d(num_procs);  soffs_d = 0;

   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      // The send_buffer contains the following:
      // a) patch no
      // b) element global number
      // c) number of vertices
      // d) global index of vertices
      // e) the coordinates of the vertices (x,y,z)
      // f) The type of the element
      //---------------------------------------------
      // get local element_map size
      int subdomain_local_nelems = local_element_map[ip].Size();
      if (subdomain_local_nelems !=0) // the rank is contributing to the patch ip
      {
         // loop through the elements
         for (int iel=0; iel<subdomain_local_nelems; ++iel)
         {
            // get the vertex list for the element
            Array<int> elem_vertices;
            int iel_idx = local_element_map[ip][iel]-myelem_offset;
            pmesh->GetElementVertices(iel_idx,elem_vertices);
            int nrvert = elem_vertices.Size();
            int j = send_displ[subdomain_rank[ip]] + soffs[subdomain_rank[ip]];
            int j_d = send_displ_d[subdomain_rank[ip]] + soffs_d[subdomain_rank[ip]];
            sendbuf[j] = ip;
            sendbuf[j+1] = iel_idx + myelem_offset;
            sendbuf[j+2] = pmesh->GetElementType(iel_idx);
            sendbuf[j+3] = nrvert;
            for (int iv = 0; iv<nrvert; ++iv)
            {
               sendbuf[j+4+iv] = aux_fespace->GetGlobalTDofNumber(elem_vertices[iv]);
               for (int comp=0; comp<dim; ++comp)
               {
                  sendbuf_d[j_d+iv+comp] = pmesh->GetVertex(elem_vertices[iv])[comp];
               }
               j_d += dim-1;
            }
            soffs[subdomain_rank[ip]] += 1 + 1 + 1 + 1 + nrvert;
            soffs_d[subdomain_rank[ip]] += dim * nrvert;
         }
      }
   }
   delete aux_fespace;
   delete aux_fec;

   // Communication
   recvbuf.SetSize(rbuff_size);
   MPI_Alltoallv(sendbuf, send_count, send_displ, MPI_INT, recvbuf,
                 recv_count, recv_displ, MPI_INT, comm);

   Array<double> recvbuf_d(rbuff_size_d);
   MPI_Alltoallv(sendbuf_d, send_count_d, send_displ_d, MPI_DOUBLE, recvbuf_d,
                 recv_count_d, recv_displ_d, MPI_DOUBLE, comm);


   // Extract from the recv_buffer
   std::vector<Array<int>> subdomain_elements(nrsubdomains);
   std::vector<Array<int>> subdomain_elements_type(nrsubdomains);
   std::vector<Array<int>> subdomain_vertices(nrsubdomains);
   std::vector<Array<double>> subdomain_vertex_xcoord(nrsubdomains);
   std::vector<Array<double>> subdomain_vertex_ycoord(nrsubdomains);
   std::vector<Array<double>> subdomain_vertex_zcoord(nrsubdomains);

   k=0;
   int kd=0;
   while (k<rbuff_size)
   {
      int ip = recvbuf[k]; k++;
      subdomain_elements[ip].Append(recvbuf[k]); k++;
      subdomain_elements_type[ip].Append(recvbuf[k]); k++;
      int nrvert = recvbuf[k]; k++;
      int id = 0;
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vertid = recvbuf[k+iv];
         // if (sets[ip]->insert(vertid).second) continue;
         subdomain_vertices[ip].Append(vertid);
         subdomain_vertex_xcoord[ip].Append(recvbuf_d[kd+iv+id]);
         subdomain_vertex_ycoord[ip].Append(recvbuf_d[kd+iv+1+id]);
         if (dim == 3) { subdomain_vertex_zcoord[ip].Append(recvbuf_d[kd+iv+2+id]); }
         id += dim-1;
      }
      k += nrvert;
      kd += dim* nrvert;
   }
   subdomain_mesh.SetSize(nrsubdomains);
   for (int ip = 0; ip < nrsubdomains; ++ip)
   {
      subdomain_mesh[ip] = nullptr;
      if (myid == subdomain_rank[ip])
      {
         // Array<int> vertices_local_id(subdomain_vertices[ip].Size());
         // loop through the patch vertices;
         // UniqueIndexGenerator gen;
         // gen.Reset();
         Array<int> sorted_vertices = subdomain_vertices[ip];
         sorted_vertices.Sort();
         sorted_vertices.Unique();
         // for (int iv = 0; iv< subdomain_vertices[ip].Size(); ++iv)
         // {
         //    int global_idx = subdomain_vertices[ip][iv];
         //    int local_idx = gen.Get(global_idx);
         //    vertices_local_id[iv] = local_idx;
         // }
         // int subdomain_nrvertices = gen.counter;
         int subdomain_nrvertices = sorted_vertices.Size();
         int subdomain_nrelems = subdomain_elements[ip].Size();
         subdomain_mesh[ip] = new Mesh(dim,subdomain_nrvertices,subdomain_nrelems);
         // Add the vertices
         // int k = -1;
         // for (int iv = 0; iv<subdomain_vertices[ip].Size(); ++iv)
         // {
         //    int vert_local_idx = vertices_local_id[iv];
         //    if (vert_local_idx > k)
         //    {
         //       double vert[dim];
         //       vert[0] = subdomain_vertex_xcoord[ip][iv];
         //       vert[1] = subdomain_vertex_ycoord[ip][iv];
         //       if (dim == 3) { vert[2] = subdomain_vertex_zcoord[ip][iv]; }
         //       subdomain_mesh[ip]->AddVertex(vert);
         //       k++;
         //    }
         // }
         for (int iv = 0; iv<sorted_vertices.Size(); ++iv)
         {
            int vert_idx = sorted_vertices[iv];
            int jv = subdomain_vertices[ip].Find(vert_idx);
            double vert[dim];
            vert[0] = subdomain_vertex_xcoord[ip][jv];
            vert[1] = subdomain_vertex_ycoord[ip][jv];
            if (dim == 3) { vert[2] = subdomain_vertex_zcoord[ip][jv]; }
            subdomain_mesh[ip]->AddVertex(vert);
         }

         int l = 0;
         // Array<int> sorted_elements = subdomain_elements[ip];
         // sorted_elements.Sort();
         for (int iel=0; iel<subdomain_nrelems; ++iel)
         {
            // int iel = subdomain_elements[ip].Find(sorted_elements[jel]);
            // int iel = jel;
            mfem::Element::Type elem_type;
            int type = subdomain_elements_type[ip][iel];
            int nrvert;
            GetNumVertices(type, elem_type, nrvert);
            // get the vertices list for the element
            int ind[nrvert];
            for (int iv = 0; iv<nrvert; ++iv)
            {
               int jv = subdomain_vertices[ip][iv+l];
               // ind[iv] = vertices_local_id[iv+l];
               ind[iv] = sorted_vertices.FindSorted(jv);
            }
            l += nrvert;
            AddElementToMesh(subdomain_mesh[ip],elem_type,ind);
         }
         subdomain_mesh[ip]->FinalizeTopology();
      }
   }
   // SaveMeshPartition();
}


void ParMeshPartition::AddElementToMesh(Mesh * mesh,
                                        mfem::Element::Type elem_type,int * ind)
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
      case Element::WEDGE :
         mesh->AddWedge(ind);
         break;
      default:
         MFEM_ABORT("Unknown element type");
         break;
   }
}

void ParMeshPartition::GetNumVertices(int type, mfem::Element::Type & elem_type,
                                      int & nrvert)
{
   switch (type)
   {
      case 0:
         elem_type = Element::POINT;
         nrvert = 1;
         break;
      case 1:
         elem_type = Element::SEGMENT;
         nrvert = 2;
         break;
      case 2:
         elem_type = Element::TRIANGLE;
         nrvert = 3;
         break;
      case 3:
         elem_type = Element::QUADRILATERAL;
         nrvert = 4;
         break;
      case 4:
         elem_type = Element::TETRAHEDRON;
         nrvert = 4;
         break;
      case 5:
         elem_type = Element::HEXAHEDRON;
         nrvert = 8;
         break;
      case 6:
         elem_type = Element::WEDGE;
         nrvert = 6;
         break;
      default:
         MFEM_ABORT("Unknown element type");
         break;
   }
}

void ParMeshPartition::SaveMeshPartition()
{
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      if (subdomain_mesh[ip])
      {
         ostringstream mesh_name;
         mesh_name << "output/mesh." << setfill('0') << setw(6) << ip;
         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         subdomain_mesh[ip]->Print(mesh_ofs);
      }
   }
}

ParMeshPartition::~ParMeshPartition()
{
   for (int ip = 0; ip<nrsubdomains; ++ip)
   {
      delete subdomain_mesh[ip];
      subdomain_mesh[ip] = nullptr;
   }
   subdomain_mesh.DeleteAll();
}


#endif