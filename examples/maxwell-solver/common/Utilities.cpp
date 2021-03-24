#include "Utilities.hpp"

Sweep::Sweep(int dim_) : dim(dim_)
{
   nsweeps = pow(2,dim);
   sweeps.resize(nsweeps);

   for (int is = 0; is<nsweeps; is++)
   {
      sweeps[is].SetSize(dim);
   }

   switch(dim)
   {
      case 1: 
         sweeps[0][0] =  1; 
         sweeps[1][0] = -1; 
         break;
      case 2: 
         sweeps[0][0] =  1; sweeps[0][1] =  1; 
         sweeps[1][0] = -1; sweeps[1][1] =  1; 
         sweeps[2][0] =  1; sweeps[2][1] = -1; 
         sweeps[3][0] = -1; sweeps[3][1] = -1;
         break;
      default:
         sweeps[0][0] =  1; sweeps[0][1] =  1; sweeps[0][2] =  1; 
         sweeps[1][0] = -1; sweeps[1][1] =  1; sweeps[1][2] =  1; 
         sweeps[2][0] =  1; sweeps[2][1] = -1; sweeps[2][2] =  1; 
         sweeps[3][0] = -1; sweeps[3][1] = -1; sweeps[3][2] =  1;
         sweeps[4][0] =  1; sweeps[4][1] =  1; sweeps[4][2] = -1; 
         sweeps[5][0] = -1; sweeps[5][1] =  1; sweeps[5][2] = -1; 
         sweeps[6][0] =  1; sweeps[6][1] = -1; sweeps[6][2] = -1; 
         sweeps[7][0] = -1; sweeps[7][1] = -1; sweeps[7][2] = -1;
         break;
   }
}

Sweep::~Sweep()
{
   for (int i = 0; i<nsweeps; i++)
   {
      sweeps[i].DeleteAll();
   }
}


double CutOffFncn(const Vector &x, const Vector & pmin, const Vector & pmax, const Array2D<double> & h_)
{
   int dim = pmin.Size();
   Vector h0(dim);
   Vector h1(dim);
   for (int i=0; i<dim; i++)
   {
      h0(i) = h_[i][0];
      h1(i) = h_[i][1];
   }
   Vector x0(dim);
   Vector x1(dim);
   x0 = pmin; x0+=h0;
   x1 = pmax; x1-=h1;

   double f = 1.0;
   for (int i = 0; i<dim; i++)
   {
      double val = 1.0;
      if( x(i) >= pmax(i) || x(i) <= pmin(i))
      {
         val = 0.0;
      }  
      else if (x(i) < pmax(i) && x(i) >= x1(i))
      {
         if(h1(i) != 0.0)
            // val = (x(i)-pmax(i))/(x1(i)-pmax(i)); 
            val = pow((x(i)-pmax(i))/(x1(i)-pmax(i)),1.0); 
      }
      else if (x(i) > pmin(i) && x(i) <= x0(i))
      {
         if (h0(i) != 0.0)
            // val = (x(i)-pmin(i))/(x0(i)-pmin(i)); 
            val = pow((x(i)-pmin(i))/(x0(i)-pmin(i)),1.0); 
      }

      if (h0(i) == 0 && x(i) <= x1(i))
      {
         val = 1.0;
      }
      if (h1(i) == 0 && x(i) >= x0(i))
      {
         val = 1.0;
      }
      f *= val;
   }
   return f;
}

double ChiFncn(const Vector &x, const Vector & pmin, const Vector & pmax, const Array2D<double> & h_)
{
   int dim = pmin.Size();
   Vector h0(dim);
   Vector h1(dim);
   for (int i=0; i<dim; i++)
   {
      h0(i) = h_[i][0];
      h1(i) = h_[i][1];
   }
   Vector x0(dim);
   Vector x1(dim);
   x0 = pmin; x0+=h0;
   x1 = pmax; x1-=h1;

   double f = 1.0;
   for (int i = 0; i<dim; i++)
   {
      double val = 1.0;
      if( x(i) >= pmax(i) || x(i) <= pmin(i))
      {
         val = 0.0;
      }  
      else if (x(i) < pmax(i) && x(i) >= x1(i))
      {
         if(h1(i) != 0.0)
            val = (x(i)-pmax(i))/(x1(i)-pmax(i)); 
            // This function has to be changed to smth more reasonable
            // val = pow((x(i)-pmax(i))/(x1(i)-pmax(i)),100.0); 
      }
      else if (x(i) > pmin(i) && x(i) <= x0(i))
      {
         if (h0(i) != 0.0)
            val = (x(i)-pmin(i))/(x0(i)-pmin(i)); 
            // val = pow((x(i)-pmin(i))/(x0(i)-pmin(i)),100.0); 
      }

      if (h0(i) == 0 && x(i) <= x1(i))
      {
         val = 1.0;
      }
      if (h1(i) == 0 && x(i) >= x0(i))
      {
         val = 1.0;
      }
      f *= val;
   }
   return f;
}


DofMap::DofMap(FiniteElementSpace * fes , MeshPartition * partition) 
{
   const FiniteElementCollection * fec = fes->FEColl();
   nrpatch = partition->nrpatch;

   fespaces.SetSize(nrpatch);

   Dof2GlobalDof.resize(nrpatch);

   for (int ip=0; ip<nrpatch; ++ip)
   {
      // create finite element spaces for each patch 
      fespaces[ip] = new FiniteElementSpace(partition->patch_mesh[ip],fec);

      // construct the patch tdof to global tdof map
      int nrdof = fespaces[ip]->GetTrueVSize();
      Dof2GlobalDof[ip].SetSize(2*nrdof);

      // loop through the elements in the patch
      for (int iel = 0; iel<partition->element_map[ip].Size(); ++iel)
      {
         // index in the global mesh
         int iel_idx = partition->element_map[ip][iel];
         // get the dofs of this element
         Array<int> ElemDofs;
         Array<int> GlobalElemDofs;
         fespaces[ip]->GetElementDofs(iel,ElemDofs);
         fes->GetElementDofs(iel_idx,GlobalElemDofs);
         // the sizes have to match
         MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
                     "Size inconsistency");
         // loop through the dofs and take into account the signs;
         int ndof = ElemDofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = ElemDofs[i];
            int gdof_ = GlobalElemDofs[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            Dof2GlobalDof[ip][pdof] = gdof;
            Dof2GlobalDof[ip][pdof+nrdof] = gdof+fes->GetTrueVSize();
         }
      }
   }
}

DofMap::DofMap(FiniteElementSpace * fes , MeshPartition * partition, int nrlayers) 
{

   nx = partition->nxyz[0];
   ny = partition->nxyz[1];
   nz = partition->nxyz[2];

   int partition_kind = partition->partition_kind;
   // Mesh * mesh = fespace->GetMesh();
   const FiniteElementCollection * fec = fes->FEColl();
   nrpatch = partition->nrpatch;

   fespaces.SetSize(nrpatch);
   PmlMeshes.SetSize(nrpatch);
   // Extend patch meshes to include pml

   for  (int ip = 0; ip<nrpatch; ip++)
   {
      int k = ip/(nx*ny);
      int j = (ip-k*nx*ny)/nx;
      int i = (ip-k*nx*ny)%nx;
      
      Array<int> directions;
      if (i > 0)
      {
         for (int i=0; i<nrlayers; i++)
         {
            directions.Append(-1);
         }
      }
      if (j > 0)
      {
         for (int i=0; i<nrlayers; i++)
         {
            directions.Append(-2);
         }
      }
      if (k > 0)
      {
         for (int i=0; i<nrlayers; i++)
         {
            directions.Append(-3);
         }
      }
      if (i < nx-1)
      {
         for (int i=0; i<nrlayers; i++)
         {
            if (partition_kind == 3 || partition_kind == 2) directions.Append(1);
         }
      }
      if (j < ny-1)
      {
         for (int i=0; i<nrlayers; i++)
         {
            if (partition_kind == 3 || partition_kind == 2) directions.Append(2);
         }
      }
      if (k < nz-1)
      {
         for (int i=0; i<nrlayers; i++)
         {
            if (partition_kind == 3 || partition_kind == 2) directions.Append(1);
         }
      }
      PmlMeshes[ip] = ExtendMesh(partition->patch_mesh[ip],directions);
   }

   // Save PML_meshes
   string meshpath;
   string solpath;
   if (partition_kind == 3 || partition_kind == 2) 
   {
      meshpath = "output/mesh_ovlp_pml.";
      solpath = "output/sol_ovlp_pml.";
   }
   else if (partition_kind == 4)
   {
      meshpath = "output/mesh_novlp_pml.";
      solpath = "output/sol_novlp_pml.";
   }
   else
   {
      MFEM_ABORT("This partition kind not supported yet");
   }
   
   // SaveMeshPartition(PmlMeshes, meshpath, solpath);

   PmlFespaces.SetSize(nrpatch);
   Dof2GlobalDof.resize(nrpatch);
   Dof2PmlDof.resize(nrpatch);

   for (int ip=0; ip<nrpatch; ++ip)
   {
      // create finite element spaces for each patch 
      fespaces[ip] = new FiniteElementSpace(partition->patch_mesh[ip],fec);
      PmlFespaces[ip] = new FiniteElementSpace(PmlMeshes[ip],fec);

      // construct the patch tdof to global tdof map
      int nrdof = fespaces[ip]->GetTrueVSize();
      Dof2GlobalDof[ip].SetSize(2*nrdof);
      Dof2PmlDof[ip].SetSize(2*nrdof);

      // build dof maps between patch and extended patch
      // loop through the patch elements and constract the dof map
      // The same elements in the extended mesh have the same ordering (but not the dofs)

      // loop through the elements in the patch
      for (int iel = 0; iel<partition->element_map[ip].Size(); ++iel)
      {
         // index in the global mesh
         int iel_idx = partition->element_map[ip][iel];
         // get the dofs of this element
         Array<int> ElemDofs;
         Array<int> PmlElemDofs;
         Array<int> GlobalElemDofs;
         fespaces[ip]->GetElementDofs(iel,ElemDofs);
         PmlFespaces[ip]->GetElementDofs(iel,PmlElemDofs);
         fes->GetElementDofs(iel_idx,GlobalElemDofs);
         // the sizes have to match
         MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
                     "Size inconsistency");
         MFEM_VERIFY(ElemDofs.Size() == PmlElemDofs.Size(),
                     "Size inconsistency");            
         // loop through the dofs and take into account the signs;
         int ndof = ElemDofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = ElemDofs[i];
            int gdof_ = GlobalElemDofs[i];
            int pmldof_ = PmlElemDofs[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            int pmldof = (pmldof_ >= 0) ? pmldof_ : abs(pmldof_) - 1;

            Dof2GlobalDof[ip][pdof] = gdof;
            Dof2GlobalDof[ip][pdof+nrdof] = gdof+fes->GetTrueVSize();
            Dof2PmlDof[ip][pdof] = pmldof;
            Dof2PmlDof[ip][pdof+nrdof] = pmldof+PmlFespaces[ip]->GetTrueVSize();
         }
      }
   }
}


LocalDofMap::LocalDofMap(const FiniteElementCollection * fec_, MeshPartition * part1_, 
               MeshPartition * part2_):fec(fec_), part1(part1_), part2(part2_)
{
   // Each overlapping patch has 2 non-overlapping subdomains
   // Thre are n non-overlapping and and n-1 overlapping subdomains
   int nrpatch = part2->nrpatch;
   MFEM_VERIFY(part1->nrpatch-1 == part2->nrpatch, "Check number of subdomains");

   cout << "Constructing local dof maps" << endl; 
   map1.resize(nrpatch);
   map2.resize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      // Get the 3 meshes involved
      Mesh * mesh = part2->patch_mesh[ip];
      Mesh * mesh1 = part1->patch_mesh[ip];
      Mesh * mesh2 = part1->patch_mesh[ip+1];

      // Define the fespaces
      FiniteElementSpace fespace(mesh, fec);
      FiniteElementSpace fespace1(mesh1, fec);
      FiniteElementSpace fespace2(mesh2, fec);

      int ndof1 = fespace1.GetTrueVSize();
      int ndof2 = fespace2.GetTrueVSize();

      map1[ip].SetSize(2*ndof1); // times 2 because it's complex
      map2[ip].SetSize(2*ndof2); // times 2 because it's complex

      // loop through the elements in the patches
      // map 1 is constructed by the first half of elements
      // map 2 is constructed by the second half of elements

      for (int iel = 0; iel<part1->element_map[ip].Size(); ++iel)
      {
         // index in the overlapping mesh
         int iel_idx = iel;
         Array<int> ElemDofs;
         Array<int> GlobalElemDofs;
         fespace1.GetElementDofs(iel,ElemDofs);
         fespace.GetElementDofs(iel_idx,GlobalElemDofs);
         // the sizes have to match
         MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
                     "Size inconsistency");
         // loop through the dofs and take into account the signs;
         int ndof = ElemDofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = ElemDofs[i];
            int gdof_ = GlobalElemDofs[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            map1[ip][pdof] = gdof;
            map1[ip][pdof+ndof1] = gdof+fespace.GetTrueVSize();
         }
      }
      for (int iel = 0; iel<part1->element_map[ip+1].Size(); ++iel)
      {
         // index in the overlapping mesh
         int k = part1->element_map[ip].Size();
         int iel_idx = iel+k;
         Array<int> ElemDofs;
         Array<int> GlobalElemDofs;
         fespace2.GetElementDofs(iel,ElemDofs);
         fespace.GetElementDofs(iel_idx,GlobalElemDofs);
         // the sizes have to match
         MFEM_VERIFY(ElemDofs.Size() == GlobalElemDofs.Size(),
                     "Size inconsistency");
         // loop through the dofs and take into account the signs;
         int ndof = ElemDofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = ElemDofs[i];
            int gdof_ = GlobalElemDofs[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            map2[ip][pdof] = gdof;
            map2[ip][pdof+ndof2] = gdof+fespace.GetTrueVSize();
         }
      }
   }
};


NeighborDofMaps::NeighborDofMaps(MeshPartition * part_, FiniteElementSpace * fes_,
                                DofMap * dmap_, 
                                int ovlp_layers_) : part(part_), fes(fes_), 
                                dmap(dmap_),
                                ovlp_layers(ovlp_layers_)
{

   nrsubdomains = part->nrpatch;
   nxyz.SetSize(3);
   mesh = fes->GetMesh();
   dim = mesh->Dimension();
   for (int d=0; d<3; d++)  nxyz[d] = part->nxyz[d];
   MarkOvlpElements();
   ComputeNeighborDofMaps();
}

void NeighborDofMaps::MarkOvlpElements()
{
   // Lists of elements
   // x,y,z = +/- 1 ovlp 
   OvlpElems.resize(nrsubdomains);

   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      int i0,j0,k0;
      Getijk(ip,i0,j0,k0);
      int ijk[dim]; ijk[0] = i0; ijk[1]=j0;
      if (dim==3) ijk[2] = k0;

      FiniteElementSpace * sub_fes = dmap->fespaces[ip];
      Mesh * sub_mesh = sub_fes->GetMesh();
      // OvlpElems[ip].resize(2*dim);
      OvlpElems[ip].resize(pow(3,dim)); 

      Vector pmin, pmax;
      sub_mesh->GetBoundingBox(pmin,pmax);
      double h = part->MeshSize;
      // Loop through elements
      for (int iel=0; iel<sub_mesh->GetNE(); iel++)
      {
         // Get element center
         Vector center(dim);
         int geom = sub_mesh->GetElementBaseGeometry(iel);
         ElementTransformation * tr = sub_mesh->GetElementTransformation(iel);
         tr->Transform(Geometries.GetCenter(geom),center);

         // loop through dimensions   
         Array<bool> pos(dim); pos = 0;
         Array<bool> neg(dim); neg = 0;

         for (int d=0;d<dim; d++)
         {
            if (ijk[d]>0 && center[d] < pmin[d]+2.0*h*ovlp_layers)
            {
               neg[d] = true;
            }

            if (ijk[d]<nxyz[d]-1 && center[d] > pmax[d]-2.0*h*ovlp_layers) 
            {
               pos[d] = true;
            }
         }
         SetElementToOverlap(ip,iel,neg,pos);        
      }   
   }
}

void NeighborDofMaps::ComputeNeighborDofMaps()
{
   OvlpDofMaps.resize(nrsubdomains);

   // Array<UniqueIndexGen * > Gen(nrsubdomains);
   // // construct unique number generator for the elements of a patch
   // for (int ip = 0; ip<nrsubdomains; ip++)
   // {
   //    Gen[ip] = new UniqueIndexGen;
   //    // register the elements
   //    int nel = part->element_map[ip].Size();
   //    for (int iel=0; iel<nel; iel++)
   //    {
   //       int iel_idx = part->element_map[ip][iel];
   //       Gen[ip]->Set(iel_idx);
   //    }
   // }

   // construct dof maps
   int nrneighbors = pow(3,dim); // including its self

   for (int ip0 = 0; ip0<nrsubdomains; ip0++)
   {
      OvlpDofMaps[ip0].resize(nrneighbors);

      FiniteElementSpace * fes0 = dmap->fespaces[ip0];
      int tdofs0 = fes0->GetTrueVSize();
      Array<int> marker0(tdofs0); marker0 = 0;
      int i0, j0, k0;
      Array<int> ijk(dim);
      Getijk(ip0, i0,j0,k0);

      int kbeg = (dim == 2) ? 0 : -1;
      int kend = (dim == 2) ? 1 :  2;
      for (int k=kbeg; k<kend; k++)
      {
         int k1 = k0 + k;
         if (k1 <0 || k1>=nxyz[2]) continue;
         int kk = (dim == 2) ? -1 : k;
         for (int j=-1; j<2; j++)
         {
            int j1 = j0 + j;
            if (j1 <0 || j1>=nxyz[1]) continue;
            for (int i=-1; i<2; i++)
            {
               int i1 = i0 + i;
               if (i1 <0 || i1>=nxyz[0]) continue;
     
               Array<int> ip0list;  marker0 = 0;
               int directionId = GetDirectionId(i,j,kk);

               Array<int> Elems = OvlpElems[ip0][directionId];
               int nel = Elems.Size();

               for (int iel = 0; iel<nel; ++iel)
               {
                  int iel0 = Elems[iel];   
                  Array<int> ElemDofs0;

                  fes0->GetElementDofs(iel0,ElemDofs0);
                  int ndof = ElemDofs0.Size();
                  // since the elements are added to the subdomain meshes 
                  // in the same ordered fashion (as they come from the
                  // original mesh) then the ordering of elements in each 
                  // subdomain is the same. Hence the dof ovlp lists
                  // can be computed for each subdomain independendly
                  for (int l = 0; l<ndof; ++l)
                  {
                     int dof0_ = ElemDofs0[l];
                     int dof0 = (dof0_ >= 0) ? dof0_ : abs(dof0_) - 1;
                     if (!marker0[dof0])
                     {
                        ip0list.Append(dof0); // dofs of ip0 in ovlp
                        marker0[dof0] = 1;
                     }
                  }
               }
               
               OvlpDofMaps[ip0][directionId].Append(ip0list);
               int tsize = fes0->GetTrueVSize();
               // Imaginary part
               for (int l=0;l<ip0list.Size(); l++) { ip0list[l] += tsize; }
               OvlpDofMaps[ip0][directionId].Append(ip0list);
            }
         }
      }
   }
}


void NeighborDofMaps::GetNeighborDofMap(const int ip,
                                        const Array<int> & directions, 
                                        Array<int> & dofmap)
{
   int k = (dim == 2) ? -1 : directions[2];
   int directionid = GetDirectionId(directions[0],directions[1],k);
   dofmap = OvlpDofMaps[ip][directionid];
}


void NeighborDofMaps::SetElementToOverlap(int ip, int iel,
                        const Array<bool> & neg, 
                        const Array<bool> & pos)
{
   int kbeg = (dim == 2) ? 0 : -1;
   int kend = (dim == 2) ? 0 :  1;
   for (int k = kbeg; k<=kend; k++)
   {
      if (dim == 3)
      {
         if (k == -1 && !neg[2]) continue;   
         if (k ==  1 && !pos[2]) continue;   
      }   
      for (int j = -1; j<=1; j++)
      {
         if (j== -1 && !neg[1]) continue;   
         if (j==  1 && !pos[1]) continue;   
         for (int i = -1; i<=1; i++)
         {
            // cases to skip 
            if (i==-1 && !neg[0]) continue;   
            if (i== 1 && !pos[0]) continue;   

            if (i==0 && j==0 && k == 0) continue;
            int kk  = (dim==2)?-1 : k;  
            int DirId = GetDirectionId(i,j,kk);
            OvlpElems[ip][DirId].Append(iel);
         }
      }
   }
}