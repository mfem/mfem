// #include "Utilities2D.hpp"

// double CutOffFncn(const Vector &x, const Vector & pmin, const Vector & pmax, const Array2D<double> & h_)
// {
//    int dim = pmin.Size();
//    Vector h0(dim);
//    Vector h1(dim);
//    for (int i=0; i<dim; i++)
//    {
//       h0(i) = h_[i][0];
//       h1(i) = h_[i][1];
//    }
//    Vector x0(dim);
//    Vector x1(dim);
//    x0 = pmin; x0+=h0;
//    x1 = pmax; x1-=h1;

//    double f = 1.0;
//    for (int i = 0; i<dim; i++)
//    {
//       double val = 1.0;
//       if( x(i) >= pmax(i) || x(i) <= pmin(i))
//       {
//          val = 0.0;
//       }  
//       else if (x(i) < pmax(i) && x(i) >= x1(i))
//       {
//          if(h1(i) != 0.0)
//             // val = (x(i)-pmax(i))/(x1(i)-pmax(i)); 
//             val = pow((x(i)-pmax(i))/(x1(i)-pmax(i)),1.0); 
//       }
//       else if (x(i) > pmin(i) && x(i) <= x0(i))
//       {
//          if (h0(i) != 0.0)
//             // val = (x(i)-pmin(i))/(x0(i)-pmin(i)); 
//             val = pow((x(i)-pmin(i))/(x0(i)-pmin(i)),1.0); 
//       }

//       if (h0(i) == 0 && x(i) <= x1(i))
//       {
//          val = 1.0;
//       }
//       if (h1(i) == 0 && x(i) >= x0(i))
//       {
//          val = 1.0;
//       }
//       f *= val;
//    }
//    return f;
// }

// double ChiFncn(const Vector &x, const Vector & pmin, const Vector & pmax, const Array2D<double> & h_)
// {
//    int dim = pmin.Size();
//    Vector h0(dim);
//    Vector h1(dim);
//    for (int i=0; i<dim; i++)
//    {
//       h0(i) = h_[i][0];
//       h1(i) = h_[i][1];
//    }
//    Vector x0(dim);
//    Vector x1(dim);
//    x0 = pmin; x0+=h0;
//    x1 = pmax; x1-=h1;

//    double f = 1.0;
//    for (int i = 0; i<dim; i++)
//    {
//       double val = 1.0;
//       if( x(i) >= pmax(i) || x(i) <= pmin(i))
//       {
//          val = 0.0;
//       }  
//       else if (x(i) < pmax(i) && x(i) >= x1(i))
//       {
//          if(h1(i) != 0.0)
//             // val = (x(i)-pmax(i))/(x1(i)-pmax(i)); 
//             // This function has to be changed to smth more reasonable
//             val = pow((x(i)-pmax(i))/(x1(i)-pmax(i)),100.0); 
//       }
//       else if (x(i) > pmin(i) && x(i) <= x0(i))
//       {
//          if (h0(i) != 0.0)
//             // val = (x(i)-pmin(i))/(x0(i)-pmin(i)); 
//             val = pow((x(i)-pmin(i))/(x0(i)-pmin(i)),100.0); 
//       }

//       if (h0(i) == 0 && x(i) <= x1(i))
//       {
//          val = 1.0;
//       }
//       if (h1(i) == 0 && x(i) >= x0(i))
//       {
//          val = 1.0;
//       }
//       f *= val;
//    }
//    return f;
// }


// DofMap::DofMap(SesquilinearForm * bf_ , MeshPartition * partition_) 
//                : bf(bf_), partition(partition_)
// {
//    // int partition_kind = partition->partition_kind;
//    // MFEM_VERIFY(partition_kind == 1, "Check Partition kind");
//    fespace = bf->FESpace();
//    // Mesh * mesh = fespace->GetMesh();
//    const FiniteElementCollection * fec = fespace->FEColl();
//    nrpatch = partition->nrpatch;

//    fespaces.SetSize(nrpatch);

//    Dof2GlobalDof.resize(nrpatch);

//    for (int ip=0; ip<nrpatch; ++ip)
//    {
//       // create finite element spaces for each patch 
//       fespaces[ip] = new FiniteElementSpace(partition->patch_mesh[ip],fec);

//       // construct the patch tdof to global tdof map
//       int nrdof = fespaces[ip]->GetTrueVSize();
//       Dof2GlobalDof[ip].SetSize(2*nrdof);

//       // loop through the elements in the patch
//       for (int iel = 0; iel<partition->element_map[ip].Size(); ++iel)
//       {
//          // index in the global mesh
//          int iel_idx = partition->element_map[ip][iel];
//          // get the dofs of this element
//          Array<int> ElemDofs;
//          Array<int> GlobalElemDofs;
//          fespaces[ip]->GetElementDofs(iel,ElemDofs);
//          fespace->GetElementDofs(iel_idx,GlobalElemDofs);
//          // the sizes have to match
//          MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
//                      "Size inconsistency");
//          // loop through the dofs and take into account the signs;
//          int ndof = ElemDofs.Size();
//          for (int i = 0; i<ndof; ++i)
//          {
//             int pdof_ = ElemDofs[i];
//             int gdof_ = GlobalElemDofs[i];
//             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
//             int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
//             Dof2GlobalDof[ip][pdof] = gdof;
//             Dof2GlobalDof[ip][pdof+nrdof] = gdof+fespace->GetTrueVSize();
//          }
//       }
//    }
// }

// DofMap::DofMap(SesquilinearForm * bf_ , MeshPartition * partition_, int nrlayers) 
//                : bf(bf_), partition(partition_)
// {

//    nx = partition->nxyz[0];
//    ny = partition->nxyz[1];
//    nz = partition->nxyz[2];

//    int partition_kind = partition->partition_kind;
//    fespace = bf->FESpace();
//    // Mesh * mesh = fespace->GetMesh();
//    const FiniteElementCollection * fec = fespace->FEColl();
//    nrpatch = partition->nrpatch;

//    fespaces.SetSize(nrpatch);
//    PmlMeshes.SetSize(nrpatch);
//    // Extend patch meshes to include pml

//    for  (int ip = 0; ip<nrpatch; ip++)
//    {
//       int k = ip/(nx*ny);
//       int j = (ip-k*nx*ny)/nx;
//       int i = (ip-k*nx*ny)%nx;
      
//       Array<int> directions;
//       if (i > 0)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             directions.Append(-1);
//          }
//       }
//       if (j > 0)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             directions.Append(-2);
//          }
//       }
//       if (k > 0)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             directions.Append(-3);
//          }
//       }
//       if (i < nx-1)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             if (partition_kind == 3 || partition_kind == 2) directions.Append(1);
//          }
//       }
//       if (j < ny-1)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             if (partition_kind == 3 || partition_kind == 2) directions.Append(2);
//          }
//       }
//       if (k < nz-1)
//       {
//          for (int i=0; i<nrlayers; i++)
//          {
//             if (partition_kind == 3 || partition_kind == 2) directions.Append(1);
//          }
//       }
//       PmlMeshes[ip] = ExtendMesh(partition->patch_mesh[ip],directions);
//    }

//    // Save PML_meshes
//    string meshpath;
//    string solpath;
//    if (partition_kind == 3 || partition_kind == 2) 
//    {
//       meshpath = "output/mesh_ovlp_pml.";
//       solpath = "output/sol_ovlp_pml.";
//    }
//    else if (partition_kind == 4)
//    {
//       meshpath = "output/mesh_novlp_pml.";
//       solpath = "output/sol_novlp_pml.";
//    }
//    else
//    {
//       MFEM_ABORT("This partition kind not supported yet");
//    }
   
//    // SaveMeshPartition(PmlMeshes, meshpath, solpath);

//    PmlFespaces.SetSize(nrpatch);
//    Dof2GlobalDof.resize(nrpatch);
//    Dof2PmlDof.resize(nrpatch);

//    for (int ip=0; ip<nrpatch; ++ip)
//    {
//       // create finite element spaces for each patch 
//       fespaces[ip] = new FiniteElementSpace(partition->patch_mesh[ip],fec);
//       PmlFespaces[ip] = new FiniteElementSpace(PmlMeshes[ip],fec);

//       // construct the patch tdof to global tdof map
//       int nrdof = fespaces[ip]->GetTrueVSize();
//       Dof2GlobalDof[ip].SetSize(2*nrdof);
//       Dof2PmlDof[ip].SetSize(2*nrdof);

//       // build dof maps between patch and extended patch
//       // loop through the patch elements and constract the dof map
//       // The same elements in the extended mesh have the same ordering (but not the dofs)

//       // loop through the elements in the patch
//       for (int iel = 0; iel<partition->element_map[ip].Size(); ++iel)
//       {
//          // index in the global mesh
//          int iel_idx = partition->element_map[ip][iel];
//          // get the dofs of this element
//          Array<int> ElemDofs;
//          Array<int> PmlElemDofs;
//          Array<int> GlobalElemDofs;
//          fespaces[ip]->GetElementDofs(iel,ElemDofs);
//          PmlFespaces[ip]->GetElementDofs(iel,PmlElemDofs);
//          fespace->GetElementDofs(iel_idx,GlobalElemDofs);
//          // the sizes have to match
//          MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
//                      "Size inconsistency");
//          MFEM_VERIFY(ElemDofs.Size() == PmlElemDofs.Size(),
//                      "Size inconsistency");            
//          // loop through the dofs and take into account the signs;
//          int ndof = ElemDofs.Size();
//          for (int i = 0; i<ndof; ++i)
//          {
//             int pdof_ = ElemDofs[i];
//             int gdof_ = GlobalElemDofs[i];
//             int pmldof_ = PmlElemDofs[i];
//             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
//             int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
//             int pmldof = (pmldof_ >= 0) ? pmldof_ : abs(pmldof_) - 1;

//             Dof2GlobalDof[ip][pdof] = gdof;
//             Dof2GlobalDof[ip][pdof+nrdof] = gdof+fespace->GetTrueVSize();
//             Dof2PmlDof[ip][pdof] = pmldof;
//             Dof2PmlDof[ip][pdof+nrdof] = pmldof+PmlFespaces[ip]->GetTrueVSize();
//          }
//       }
//    }
// }


// LocalDofMap::LocalDofMap(const FiniteElementCollection * fec_, MeshPartition * part1_, 
//                MeshPartition * part2_):fec(fec_), part1(part1_), part2(part2_)
// {
//    // Each overlapping patch has 2 non-overlapping subdomains
//    // Thre are n non-overlapping and and n-1 overlapping subdomains
//    int nrpatch = part2->nrpatch;
//    MFEM_VERIFY(part1->nrpatch-1 == part2->nrpatch, "Check number of subdomains");

//    cout << "Constructing local dof maps" << endl; 
//    map1.resize(nrpatch);
//    map2.resize(nrpatch);
//    for (int ip=0; ip<nrpatch; ip++)
//    {
//       // Get the 3 meshes involved
//       Mesh * mesh = part2->patch_mesh[ip];
//       Mesh * mesh1 = part1->patch_mesh[ip];
//       Mesh * mesh2 = part1->patch_mesh[ip+1];

//       // Define the fespaces
//       FiniteElementSpace fespace(mesh, fec);
//       FiniteElementSpace fespace1(mesh1, fec);
//       FiniteElementSpace fespace2(mesh2, fec);

//       int ndof1 = fespace1.GetTrueVSize();
//       int ndof2 = fespace2.GetTrueVSize();

//       map1[ip].SetSize(2*ndof1); // times 2 because it's complex
//       map2[ip].SetSize(2*ndof2); // times 2 because it's complex

//       // loop through the elements in the patches
//       // map 1 is constructed by the first half of elements
//       // map 2 is constructed by the second half of elements

//       for (int iel = 0; iel<part1->element_map[ip].Size(); ++iel)
//       {
//          // index in the overlapping mesh
//          int iel_idx = iel;
//          Array<int> ElemDofs;
//          Array<int> GlobalElemDofs;
//          fespace1.GetElementDofs(iel,ElemDofs);
//          fespace.GetElementDofs(iel_idx,GlobalElemDofs);
//          // the sizes have to match
//          MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
//                      "Size inconsistency");
//          // loop through the dofs and take into account the signs;
//          int ndof = ElemDofs.Size();
//          for (int i = 0; i<ndof; ++i)
//          {
//             int pdof_ = ElemDofs[i];
//             int gdof_ = GlobalElemDofs[i];
//             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
//             int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
//             map1[ip][pdof] = gdof;
//             map1[ip][pdof+ndof1] = gdof+fespace.GetTrueVSize();
//          }
//       }
//       for (int iel = 0; iel<part1->element_map[ip+1].Size(); ++iel)
//       {
//          // index in the overlapping mesh
//          int k = part1->element_map[ip].Size();
//          int iel_idx = iel+k;
//          Array<int> ElemDofs;
//          Array<int> GlobalElemDofs;
//          fespace2.GetElementDofs(iel,ElemDofs);
//          fespace.GetElementDofs(iel_idx,GlobalElemDofs);
//          // the sizes have to match
//          MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
//                      "Size inconsistency");
//          // loop through the dofs and take into account the signs;
//          int ndof = ElemDofs.Size();
//          for (int i = 0; i<ndof; ++i)
//          {
//             int pdof_ = ElemDofs[i];
//             int gdof_ = GlobalElemDofs[i];
//             int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
//             int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
//             map2[ip][pdof] = gdof;
//             map2[ip][pdof+ndof2] = gdof+fespace.GetTrueVSize();
//          }
//       }
//    }
// }