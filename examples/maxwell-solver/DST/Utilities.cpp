
#include "Utilities.hpp"

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
   x0 = pmax; x0-=h1;
   Vector x1(dim);
   x1 = pmin; x1+=h0;

   double f = 1.0;

   for (int i = 0; i<dim; i++)
   {
      double val = 1.0;
      if( x(i) > pmax(i) || x(i) < pmin(i))
      {
         val = 0.0;
      }  
      else if (x(i) <= pmax(i) && x(i) >= x0(i))
      {
         if(x0(i)-pmax(i) != 0.0)
            val = (x(i)-pmax(i))/(x0(i)-pmax(i)); 
      }
      else if (x(i) >= pmin(i) && x(i) <= x1(i))
      {
         if (x1(i)-pmin(i) != 0.0)
            val = (x(i)-pmin(i))/(x1(i)-pmin(i)); 
      }
      else
      {
         val = 1.0;
      }
      if (h_[i][0] == 0 && x(i) < pmin(i))
      {
         val = 1.0;
      }
      if (h_[i][1] == 0 && x(i) > pmax(i))
      {
         val = 1.0;
      }
      f *= val;
   }
   return f;
}


DofMap::DofMap(SesquilinearForm * bf_ , MeshPartition * partition_) 
               : bf(bf_), partition(partition_)
{
   int partition_kind = partition->partition_kind;
   MFEM_VERIFY(partition_kind == 1, "Check Partition kind");
   fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   const FiniteElementCollection * fec = fespace->FEColl();
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
         fespace->GetElementDofs(iel_idx,GlobalElemDofs);
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
            Dof2GlobalDof[ip][pdof+nrdof] = gdof+fespace->GetTrueVSize();
         }
      }
   }
}

DofMap::DofMap(SesquilinearForm * bf_ , MeshPartition * partition_, int nrlayers) 
               : bf(bf_), partition(partition_)
{
   int partition_kind = partition->partition_kind;
   fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   const FiniteElementCollection * fec = fespace->FEColl();
   nrpatch = partition->nrpatch;

   fespaces.SetSize(nrpatch);
   PmlMeshes.SetSize(nrpatch);
   // Extend patch meshes to include pml
   for  (int ip = 0; ip<nrpatch; ip++)
   {
      Array<int> directions;
      if (ip > 0)
      {
         for (int i=0; i<nrlayers; i++)
         {
            directions.Append(-1);
         }
      }
      if (ip < nrpatch-1)
      {
         for (int i=0; i<nrlayers; i++)
         {
            if (partition_kind == 3) directions.Append(1);
         }
      }
      PmlMeshes[ip] = ExtendMesh(partition->patch_mesh[ip],directions);
   }

   // Save PML_meshes
   string meshpath;
   string solpath;
   if (partition_kind == 3) 
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
         fespace->GetElementDofs(iel_idx,GlobalElemDofs);
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
            Dof2GlobalDof[ip][pdof+nrdof] = gdof+fespace->GetTrueVSize();
            Dof2PmlDof[ip][pdof] = pmldof;
            Dof2PmlDof[ip][pdof+nrdof] = pmldof+PmlFespaces[ip]->GetTrueVSize();
         }
      }
   }
}