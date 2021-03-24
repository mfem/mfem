// #include "MeshPartition2D.hpp"

// double GetUniformMeshElementSize(Mesh * mesh)
// {
//    int dim = mesh->Dimension();
//    int nrelem = mesh->GetNE();

//    DenseMatrix J(dim);
//    double hmin, hmax;
//    hmin = infinity();
//    hmax = -infinity();
//    Vector attr(nrelem);
//    for (int iel=0; iel<nrelem; ++iel)
//    {
//       int geom = mesh->GetElementBaseGeometry(iel);
//       ElementTransformation *T = mesh->GetElementTransformation(iel);
//       T->SetIntPoint(&Geometries.GetCenter(geom));
//       Geometries.JacToPerfJac(geom, T->Jacobian(), J);
//       attr(iel) = J.Det();
//       attr(iel) = pow(abs(attr(iel)), 1.0/double(dim));
//       hmin = min(hmin, attr(iel));
//       hmax = max(hmax, attr(iel));
//    }
//    MFEM_VERIFY(abs(hmin-hmax) < 1e-12, "Case not supported yet")

//    return hmax;
// }

// Mesh * ExtendMesh(Mesh * mesh, const Array<int> & directions)
// {
//     // extrute on one dimension
//    // flag = 1 +x, -1 -x, 2 +y, -2 +y , 3 +z, -3, -z

//    // copy the original mesh;
//    Mesh * mesh_orig = new Mesh(*mesh);
//    if (!directions.Size()) return mesh_orig;
   
//    int dim = mesh_orig->Dimension();

//    Mesh * mesh_ext=nullptr;

//    for (int j=0; j<directions.Size(); j++)
//    {
//       int d = directions[j];
//       MFEM_VERIFY(abs(d)<= dim, "Cannot Extend in dimension " << d << ". Dim = " << dim << endl);

//       Vector pmin;
//       Vector pmax;
//       mesh_orig->GetBoundingBox(pmin,pmax);
//       double h = GetUniformMeshElementSize(mesh_orig);
//       double val;
//       // find the vertices on the specific boundary
//       switch (d)
//       {
//       case 1:
//          val = pmax[0];
//          break;
//       case -1:
//          val = pmin[0];
//          h = -h;
//          break;    
//       case 2:
//          val = pmax[1];
//          break;
//       case -2:
//          val = pmin[1];
//          h = -h;
//          break;   
//       case 3:
//          val = pmax[2];
//          break;
//       case -3:
//          val = pmin[2];
//          h = -h;
//          break;      
//       }
//       int k = 0;
//       for (int i = 0; i<mesh_orig->GetNV(); ++i)
//       {
//          double * coords = mesh_orig->GetVertex(i);
//          switch (abs(d))
//          {
//          case 1:
//             if (coords[0] == val) k++;
//             break;
//          case 2:
//             if (coords[1] == val) k++;
//             break;   
//          case 3:
//             if (coords[2] == val) k++;
//             break;      
//          }
//       }
//       int nrvertices = mesh_orig->GetNV() + k;
//       int nrelements = mesh_orig->GetNE() + pow(pow(k,1.0/(dim-1))-1.0,dim-1);

//       mesh_ext = new Mesh(dim, nrvertices, nrelements);

//       // Add existing vertices
//       Array<int> vmap(mesh_orig->GetNV()); vmap = 0;
//       k = mesh_orig->GetNV();
//       for (int i=0; i<mesh_orig->GetNV(); ++i)
//       {
//          double * vert = mesh_orig->GetVertex(i);
//          mesh_ext->AddVertex(vert);
//          switch (abs(d))
//          {
//          case 1:
//             if (vert[0] == val) 
//             {
//                vmap[i] = k;
//                k++;
//             }
//             break;
//          case 2:
//             if (vert[1] == val)
//             {
//                vmap[i] = k;
//                k++;
//             }
//             break;   
//          case 3:
//             if (vert[2] == val)
//             {
//                vmap[i] = k;
//                k++;
//             }
//             break;      
//          }
//       } 
//       // Add existing elements
//       for (int i=0; i<mesh_orig->GetNE(); ++i)
//       {
//          Array<int>ind;
//          mesh_orig->GetElementVertices(i,ind);
//          if (dim == 2)
//          {
//             mesh_ext->AddQuad(ind);
//          }
//          else if (dim == 3)
//          {
//             mesh_ext->AddHex(ind);
//          }
//       } 
//       // Add new vertices
//       k = mesh_orig->GetNV();
//       for (int i=0; i<mesh_orig->GetNV(); ++i)
//       {
//          double * vert = mesh_orig->GetVertex(i);
//          switch (abs(d))
//          {
//          case 1:
//             if (vert[0] == val) 
//             {
//                double coords[dim];
//                coords[0] = vert[0] + h;
//                coords[1] = vert[1];
//                if (dim == 3) coords[2] = vert[2];
//                mesh_ext->AddVertex(coords);
//             }
//             break;
//          case 2:
//             if (vert[1] == val)
//             {
//                double coords[dim];
//                coords[0] = vert[0];
//                coords[1] = vert[1] + h;
//                if (dim == 3) coords[2] = vert[2];
//                mesh_ext->AddVertex(coords);
//             }
//             break;   
//          case 3:
//             if (vert[2] == val)
//             {
//                double coords[dim];
//                coords[0] = vert[0];
//                coords[1] = vert[1];
//                coords[2] = vert[2] + h;
//                mesh_ext->AddVertex(coords);
//             }
//             break;      
//          }
//       }    
//       // loop through boundary elements and extend in the given direction
//       for (int i=0; i<mesh_orig->GetNBE(); ++i) 
//       {
//          Array<int> vertices;
//          mesh_orig->GetBdrElementVertices(i,vertices);
//          if (dim == 2)
//          {
//             int ind[4];
//             if (vmap[vertices[0]] && vmap[vertices[1]])
//             {
//                ind[0] = vmap[vertices[0]];
//                ind[1] = vmap[vertices[1]];
//                ind[2] = vertices[1];
//                ind[3] = vertices[0];
//                mesh_ext->AddQuad(ind);
//             }
//          }
//          else if (dim == 3)
//          {
//             int ind[8];
//             if (vmap[vertices[0]] && vmap[vertices[1]] && vmap[vertices[2]] && vmap[vertices[3]])
//             {
//                ind[0] = vmap[vertices[0]];
//                ind[1] = vmap[vertices[1]];
//                ind[2] = vmap[vertices[2]];
//                ind[3] = vmap[vertices[3]];
//                ind[4] = vertices[0];
//                ind[5] = vertices[1];
//                ind[6] = vertices[2];
//                ind[7] = vertices[3];
//                mesh_ext->AddHex(ind);
//             }
//          }   
//       }
//       mesh_ext->FinalizeTopology();

//       if (j<directions.Size()-1)
//       {
//          delete mesh_orig;
//          mesh_orig = mesh_ext;
//       }
//    }
//    delete mesh_orig;
//    return mesh_ext;
// }

// // constructor
   
// OverlappingCartesianMeshPartition::OverlappingCartesianMeshPartition(Mesh *mesh_,int & nx,int & ny,int & nz) : mesh(mesh_)
// {  // default overlap size is 2 elements 
//    int dim = mesh->Dimension();
//    int n = pow(mesh->GetNE(), 1.0/(double)dim);
//    if (nx > n) 
//    {
//       nx = n;
//       MFEM_WARNING("Changed partition in the x direction to nx = " << n << endl);
//    }
//    if (ny > n) 
//    {
//       ny = n;
//       MFEM_WARNING("Changed partition in the y direction to ny = " << n << endl);
//    }
//    if (nz > n) 
//    {
//       nz = n;
//       MFEM_WARNING("Changed partition in the z direction to nz = " << n << endl);
//    } 
//    if (dim == 2) nz = 1;
//    subdomains.SetSize(nx,ny,nz);
//    nxyz[0] = nx; nxyz[1]=ny; nxyz[2] = nz;
//    nrpatch = nx*ny*nz;
//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);
//    double h = GetUniformMeshElementSize(mesh);

//    element_map.resize(nrpatch);

//    double ppt[dim];
//    Vector pt(ppt, dim);
//    int nrelem = mesh->GetNE();

//    for (int el = 0; el < nrelem; el++)
//    {
//       mesh->GetElementTransformation(el)->Transform(
//          Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
//       // Given the center coordinates determine the patches that this element contributes to
//       Array<int> idx0(dim);
//       Array<int> idx1(dim);
//       Array<int> idx2(dim);
//       vector<Array<int>> idx(3);
//       if (dim == 2) idx[2].Append(0);

//       for (int i = 0; i<dim; i++)
//       {
//          idx0[i]  = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
//          idx1[i] = (int)floor(nxyz[i]*((pt(i)+h - pmin[i])/(pmax[i] - pmin[i])));
//          idx2[i] = (int)floor(nxyz[i]*((pt(i)-h - pmin[i])/(pmax[i] - pmin[i])));

//          if (idx0[i] < 0) idx0[i] = 0;
//          if (idx0[i] >= nxyz[i]) idx0[i] = nxyz[i]-1;
         
//          if (idx1[i] < 0) idx1[i] = 0;
//          if (idx1[i] >= nxyz[i]) idx1[i] = nxyz[i]-1;

//          if (idx2[i] < 0) idx2[i] = 0;
//          if (idx2[i] >= nxyz[i]) idx2[i] = nxyz[i]-1;
//          // convenient to put in one list
//          idx[i].Append(idx0[i]);
//          if (idx1[i] != idx0[i]) idx[i].Append(idx1[i]);
//          if (idx2[i] != idx0[i] && idx2[i] != idx1[i]) idx[i].Append(idx2[i]);
//       }
//       // Now loop through all the combinations according to the idx above
//       // in case of dim = 2 then kk = 0
//       for (int k=0; k<idx[2].Size(); k++)
//       {
//          int kk = idx[2][k];
//          for (int j=0; j<idx[1].Size(); j++)
//          {
//             int jj = idx[1][j];
//             for (int i=0; i<idx[0].Size(); i++)
//             {
//                int ii = idx[0][i];
//                int ip = kk*nxyz[0]*nxyz[1] + jj*nxyz[0]+ii;
//                element_map[ip].Append(el);
//             }
//          }
//       }
//    }
//    for (int k = 0; k<nz; k++)
//    {
//       for (int j = 0; j<ny; j++)
//       {
//          for (int i = 0; i<nx; i++)
//          {
//             subdomains(i,j,k) = k*ny*nx + j*nx + i;
//          }
//       }   
//    }
// }


// OverlappingCartesianMeshPartition::OverlappingCartesianMeshPartition(Mesh *mesh_,int & nx,int & ny,int & nz, int ovlp_nlayers) : mesh(mesh_)
// {  // default overlap size is 2 elements 
//    int dim = mesh->Dimension();
//    int n = pow(mesh->GetNE(), 1.0/(double)dim);
//    if (nx > n) 
//    {
//       nx = n;
//       MFEM_WARNING("Changed partition in the x direction to nx = " << n << endl);
//    }
//    if (ny > n) 
//    {
//       ny = n;
//       MFEM_WARNING("Changed partition in the y direction to ny = " << n << endl);
//    }
//    if (nz > n) 
//    {
//       nz = n;
//       MFEM_WARNING("Changed partition in the z direction to nz = " << n << endl);
//    } 
//    if (dim == 2) nz = 1;
//    subdomains.SetSize(nx,ny,nz);
//    nxyz[0] = nx; nxyz[1]=ny; nxyz[2] = nz;
//    nrpatch = nx*ny*nz;
//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);
//    double h = GetUniformMeshElementSize(mesh);
//    cout << "h = " << h << endl;

//    // Check that ovlp_size does not exit subdomain size
//    MFEM_VERIFY((pmax[0]-pmin[0])/nx >= h*ovlp_nlayers, 
//                 "Check ovlp size in partition"); 
//                 cout << "pmax[0]-pmin[0])/nx = " << (pmax[0]-pmin[0])/nx << endl;
//                 cout << "ovlp_nlayers = " << ovlp_nlayers << endl;
//                 cout << "h*ovlp_nlayers = " << h*ovlp_nlayers << endl;
//    MFEM_VERIFY((pmax[1]-pmin[1])/ny >= h*ovlp_nlayers, 
//                "Check ovlp size in partition"); 
//    if (dim == 3)
//    {
//       MFEM_VERIFY((pmax[2]-pmin[2])/nz >= h*ovlp_nlayers, 
//                "Check ovlp size in partition"); 
//    }
//    element_map.resize(nrpatch);

//    double ppt[dim];
//    Vector pt(ppt, dim);
//    int nrelem = mesh->GetNE();

//    for (int el = 0; el < nrelem; el++)
//    {
//       mesh->GetElementTransformation(el)->Transform(
//          Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
//       // Given the center coordinates determine the patches that this element contributes to
//       Array<int> idx0(dim);
//       Array<int> idx1(dim);
//       Array<int> idx2(dim);
//       vector<Array<int>> idx(3);
//       if (dim == 2) idx[2].Append(0);

//       for (int i = 0; i<dim; i++)
//       {
//          idx0[i]  = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
//          idx1[i] = (int)floor(nxyz[i]*((pt(i)+ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));
//          idx2[i] = (int)floor(nxyz[i]*((pt(i)-ovlp_nlayers*h - pmin[i])/(pmax[i] - pmin[i])));

//          if (idx0[i] < 0) idx0[i] = 0;
//          if (idx0[i] >= nxyz[i]) idx0[i] = nxyz[i]-1;
         
//          if (idx1[i] < 0) idx1[i] = 0;
//          if (idx1[i] >= nxyz[i]) idx1[i] = nxyz[i]-1;

//          if (idx2[i] < 0) idx2[i] = 0;
//          if (idx2[i] >= nxyz[i]) idx2[i] = nxyz[i]-1;
//          // convenient to put in one list
//          idx[i].Append(idx0[i]);
//          if (idx1[i] != idx0[i]) idx[i].Append(idx1[i]);
//          if (idx2[i] != idx0[i] && idx2[i] != idx1[i]) idx[i].Append(idx2[i]);
//       }
//       // Now loop through all the combinations according to the idx above
//       // in case of dim = 2 then kk = 0
//       for (int k=0; k<idx[2].Size(); k++)
//       {
//          int kk = idx[2][k];
//          for (int j=0; j<idx[1].Size(); j++)
//          {
//             int jj = idx[1][j];
//             for (int i=0; i<idx[0].Size(); i++)
//             {
//                int ii = idx[0][i];
//                int ip = kk*nxyz[0]*nxyz[1] + jj*nxyz[0]+ii;
//                element_map[ip].Append(el);
//             }
//          }
//       }
//    }
//    for (int k = 0; k<nz; k++)
//    {
//       for (int j = 0; j<ny; j++)
//       {
//          for (int i = 0; i<nx; i++)
//          {
//             subdomains(i,j,k) = k*ny*nx + j*nx + i;
//          }
//       }   
//    }
// }

// // constructor
// CartesianMeshPartition::CartesianMeshPartition(Mesh *mesh_,int & nx, int & ny, int & nz) : mesh(mesh_)
// {
//    int dim = mesh->Dimension();
//    nxyz[0] = nx;
//    nxyz[1] = ny;
//    nxyz[2] = nz;
//    nrpatch = nx*ny*nz;
//    subdomains.SetSize(nx,ny,nz);
//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);

//    int nrelem = mesh->GetNE();
//    int partitioning[nrelem];

//    // determine the partitioning using the centers of the elements
//    double ppt[dim];
//    Vector pt(ppt, dim);
//    for (int el = 0; el < nrelem; el++)
//    {
//       mesh->GetElementTransformation(el)->Transform(
//          Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
//       int part = 0;
//       for (int i = dim-1; i >= 0; i--)
//       {
//          int idx = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
//          if (idx < 0)
//          {
//             idx = 0;
//          }
//          if (idx >= nxyz[i])
//          {
//             idx = nxyz[i]-1;
//          }
//          part = part * nxyz[i] + idx;
//       }
//       partitioning[el] = part;
//    }

//    element_map.resize(nrpatch);
//    for (int iel = 0; iel < nrelem; iel++)
//    {
//       int ip = partitioning[iel];
//       element_map[ip].Append(iel);
//    }
//    for (int k = 0; k<nz; k++)
//    {
//       for (int j = 0; j<ny; j++)
//       {
//          for (int i = 0; i<nx; i++)
//          {
//             subdomains(i,j,k) = k*ny*nx + j*nx + i;
//          }
//       }   
//    }
// }

// STPOverlappingCartesianMeshPartition::STPOverlappingCartesianMeshPartition(Mesh *mesh_) : mesh(mesh_)
// {
//    int dim = mesh->Dimension();
//    nx = 5;
//    ny = 1;
//    nz = 1;
//    int nxyz[3] = {nx,ny,nz};
//    // nrpatch = nx*ny*nz;

//    Vector pmin, pmax;
//    mesh->GetBoundingBox(pmin, pmax);

//    int nrelem = mesh->GetNE();
//    int partitioning[nrelem];

//    // determine the partitioning using the centers of the elements
//    double ppt[dim];
//    Vector pt(ppt, dim);
//    for (int el = 0; el < nrelem; el++)
//    {
//       mesh->GetElementTransformation(el)->Transform(
//          Geometries.GetCenter(mesh->GetElementBaseGeometry(el)), pt);
//       int part = 0;
//       for (int i = dim-1; i >= 0; i--)
//       {
//          int idx = (int)floor(nxyz[i]*((pt(i) - pmin[i])/(pmax[i] - pmin[i])));
//          if (idx < 0)
//          {
//             idx = 0;
//          }
//          if (idx >= nxyz[i])
//          {
//             idx = nxyz[i]-1;
//          }
//          part = part * nxyz[i] + idx;
//       }
//       partitioning[el] = part;
//    }

//    std::vector<Array<int>> elem_map;
//    int npatch = nx*ny*nz;
//    elem_map.resize(npatch);
//    for (int iel = 0; iel < nrelem; iel++)
//    {
//       int ip = partitioning[iel];
//       elem_map[ip].Append(iel);
//    }
//    // Append the next subdomain to the previous
//    nrpatch = nx*ny*nz-1;
//    element_map.resize(nrpatch);
//    for (int ip = 0; ip<nrpatch; ip++)
//    {
//       element_map[ip].Append(elem_map[ip]);
//       element_map[ip].Append(elem_map[ip+1]);
//    }
// }

// MeshPartition::MeshPartition(Mesh* mesh_, int part,int nx, int ny, int nz, int nrlayers): mesh(mesh_)
// {
//    partition_kind = part;
//    if (part == 1)
//    {
//       cout << "Non Overlapping Cartesian Partition " << endl;
//       CartesianMeshPartition partition(mesh,nx, ny, nz);
//       element_map = partition.element_map;
//       // subdomains = partition.subdomains;
//    }
//    // else if (part == 3 || part == 4)
//    else if (part == 2)
//    {
//       cout << "Overlapping Cartesian Partition " << endl;
//       OverlappingCartesianMeshPartition partition(mesh,nx, ny, nz,nrlayers);
//       element_map = partition.element_map;
//       subdomains = partition.subdomains;
//       nxyz[0] = partition.nxyz[0];
//       nxyz[1] = partition.nxyz[1];
//       nxyz[2] = partition.nxyz[2];
//    }
//    else if (part == 3 || part == 4)
//    // else if (part == 2)
//    {
//       cout << "STP Overlapping Cartesian Partition " << endl;
//       STPOverlappingCartesianMeshPartition partition(mesh);
//       element_map = partition.element_map;
//    }
//    else
//    {
//       MFEM_ABORT("Overlapping Vertex based partition not supprorted")
//    }

//    nrpatch = element_map.size();

//    int dim = mesh->Dimension();

//    patch_mesh.SetSize(nrpatch);
//    for (int ip = 0; ip<nrpatch; ++ip)
//    {
//       int patch_nrelems = element_map[ip].Size();
//       element_map[ip].SetSize(patch_nrelems);
//       Array<int> patch_vertices;
//       for (int iel=0; iel<patch_nrelems; ++iel)
//       {
//          // get the vertices list for the element
//          Array<int> elem_vertices;
//          int iel_idx = element_map[ip][iel];
//          mesh->GetElementVertices(iel_idx,elem_vertices);
//          patch_vertices.Append(elem_vertices);
//       }
//       patch_vertices.Sort();
//       patch_vertices.Unique();
//       int patch_nrvertices = patch_vertices.Size();

//       // create the mesh
//       patch_mesh[ip] = new Mesh(dim,patch_nrvertices,patch_nrelems);
//       // Add the vertices
//       for (int iv = 0; iv<patch_nrvertices; ++iv)
//       {
//          int vert_idx = patch_vertices[iv];
//          patch_mesh[ip]->AddVertex(mesh->GetVertex(vert_idx));
//       }

//       // Add the elements (for now search through all the vertices in the patch is needed)
//       for (int iel=0; iel<patch_nrelems; ++iel)
//       {
//          // get the vertices list for the element
//          Array<int> elem_vertices;
//          int iel_idx = element_map[ip][iel];
//          mesh->GetElementVertices(iel_idx,elem_vertices);
//          int nrvert = elem_vertices.Size();
//          int ind[nrvert];
//          for (int iv = 0; iv<nrvert; ++iv)
//          {
//             ind[iv] = patch_vertices.FindSorted(elem_vertices[iv]);
//          }
//          mfem::Element::Type elem_type = mesh->GetElementType(element_map[ip][iel]);

//          AddElementToMesh(patch_mesh[ip],elem_type,ind);

//       }
//       patch_mesh[ip]->FinalizeTopology();
//    }
// }

// void MeshPartition::AddElementToMesh(Mesh * mesh,mfem::Element::Type elem_type,
//                                      int * ind)
// {
//    switch (elem_type)
//    {
//       case Element::QUADRILATERAL:
//          mesh->AddQuad(ind);
//          break;
//       case Element::TRIANGLE :
//          mesh->AddTri(ind);
//          break;
//       case Element::HEXAHEDRON :
//          mesh->AddHex(ind);
//          break;
//       case Element::TETRAHEDRON :
//          mesh->AddTet(ind);
//          break;
//       default:
//          MFEM_ABORT("Unknown element type");
//          break;
//    }
// }

// void MeshPartition::PrintElementMap()
// {
//    mfem::out << "Element map" << endl;
//    for (int ip = 0; ip<nrpatch; ++ip)
//    {
//       mfem::out << "Patch No: " << ip;
//       mfem::out << ", element map: " ;
//       element_map[ip].Print(cout,element_map[ip].Size());
//    }
// }

// void SaveMeshPartition(Array<Mesh *> meshes, string mfilename, string sfilename)
// {
//    int nrmeshes = meshes.Size();
//    for (int ip = 0; ip<nrmeshes; ++ip)
//    {
//       cout << "saving mesh no " << ip << endl;
//       ostringstream mesh_name;
//       mesh_name << mfilename << setfill('0') << setw(6) << ip;
//       ofstream mesh_ofs(mesh_name.str().c_str());
//       mesh_ofs.precision(8);
//       meshes[ip]->Print(mesh_ofs);
//       L2_FECollection L2fec(1,meshes[ip]->Dimension());
//       FiniteElementSpace L2fes(meshes[ip], &L2fec);
//       GridFunction x(&L2fes); 
      
//       ConstantCoefficient alpha((double)ip);
//       x.ProjectCoefficient(alpha);
//       ostringstream sol_name;
//       sol_name << sfilename << setfill('0') << setw(6) << ip;
//       ofstream sol_ofs(sol_name.str().c_str());
//       x.Save(sol_ofs);
//    }
// }

// void SaveMesh(Mesh * mesh, string mfilename)
// {
//    cout << "saving global mesh " << endl;
//    ostringstream mesh_name;
//    mesh_name << mfilename;
//    ofstream mesh_ofs(mesh_name.str().c_str());
//    mesh_ofs.precision(8);
//    mesh->Print(mesh_ofs);
// }

// MeshPartition::~MeshPartition()
// {
//    for (int ip = 0; ip<nrpatch; ++ip)
//    {
//       delete patch_mesh[ip];
//       patch_mesh[ip] = nullptr;
//    }
//    patch_mesh.DeleteAll();
// }
