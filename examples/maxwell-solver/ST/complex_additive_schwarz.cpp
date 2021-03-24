#include "complex_additive_schwarz.hpp"

ComplexPatchAssembly::ComplexPatchAssembly(SesquilinearForm * bf_, Array<int> & ess_tdofs, int part) : bf(bf_)
{
   fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   int dim = mesh->Dimension();
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
   nx = p->nx;
   ny = p->ny;
   nz = p->nz;
   // SaveMeshPartition(p->patch_mesh);
   nrpatch = p->nrpatch;
   patch_fespaces.SetSize(nrpatch);
   patch_meshes_ext.SetSize(nrpatch);
   patch_fespaces_ext.SetSize(nrpatch);
   dof2extdof_map.resize(nrpatch);
   patch_dof_map.resize(nrpatch);
   patch_mat.SetSize(nrpatch);
   patch_mat_ext.SetSize(nrpatch);
   patch_mat_inv.SetSize(nrpatch);
   patch_mat_inv_ext.SetSize(nrpatch);
   ess_tdof_list.resize(nrpatch);

   // construct extended meshes
   int ip = -1;
   int nrlayers = 0;
   if (!part)
   {
      for (int ip = 0; ip<nrpatch; ip++)
      {
         patch_meshes_ext[ip] = new Mesh(*p->patch_mesh[ip]);
      }
   }
   else
   {   
      for (int kz = 0; kz<nz; kz++)
      {
         for (int ky = 0; ky<ny; ky++)
         {
            for (int kx = 0; kx<nx; kx++)
            {
               ip++;
               Array<int> ext_directions;
               for (int j=0; j<nrlayers; ++j)
               {
                  for (int comp=0; comp<dim; ++comp)
                  {
                     if (comp == 0 && kx != 0)
                     {
                        ext_directions.Append(-comp-1);
                     }
                     if (comp == 0 && kx != nx-1)
                     {
                        ext_directions.Append(comp+1);
                     }
                     if (comp == 1 && ky != 0)
                     {
                        ext_directions.Append(-comp-1);
                     }
                     if (comp == 1 && ky != ny-1)
                     {
                        ext_directions.Append(comp+1);
                     }
                     if (comp == 2 && kz != 0)
                     {
                        ext_directions.Append(-comp-1);
                     }
                     if (comp == 2 && kz != ny-1)
                     {
                        ext_directions.Append(comp+1);
                     }
                  }
               }
               patch_meshes_ext[ip] = ExtendMesh(p->patch_mesh[ip],ext_directions);
            }
         }
      }
   }

   
   // SaveMeshPartition(patch_meshes_ext, "output/ext_mesh.", "output/ext_sol.");

   for (int ip=0; ip<nrpatch; ++ip)
   {
      // create finite element spaces for each patch // This might be avoided
      patch_fespaces[ip] = new FiniteElementSpace(p->patch_mesh[ip],fec);
      // create finite element spaces on the extented (PML) meshes
      patch_fespaces_ext[ip] = new FiniteElementSpace(patch_meshes_ext[ip],fec);

      // construct the patch tdof to global tdof map
      int nrdof = patch_fespaces[ip]->GetTrueVSize();
      patch_dof_map[ip].SetSize(2*nrdof);
      dof2extdof_map[ip].SetSize(2*nrdof);

      // build dof maps between patch and extended patch
      //loop through the patch elements and constract the dof map
      // The same elements in the extended mesh have the same ordering (but not the dofs)

      // loop through the elements in the patch
      for (int iel = 0; iel<p->element_map[ip].Size(); ++iel)
      {
         // index in the global mesh
         int iel_idx = p->element_map[ip][iel];
         // get the dofs of this element
         Array<int> patch_elem_dofs;
         Array<int> patch_elem_dofs_ext;
         Array<int> global_elem_dofs;
         patch_fespaces[ip]->GetElementDofs(iel,patch_elem_dofs);
         patch_fespaces_ext[ip]->GetElementDofs(iel,patch_elem_dofs_ext);
         fespace->GetElementDofs(iel_idx,global_elem_dofs);
         // the sizes have to match
         MFEM_VERIFY(patch_elem_dofs.Size() == global_elem_dofs.Size(),
                     "Size inconsistency");
         MFEM_VERIFY(patch_elem_dofs.Size() == patch_elem_dofs_ext.Size(),
                     "Size inconsistency");            
         // loop through the dofs and take into account the signs;
         int ndof = patch_elem_dofs.Size();
         for (int i = 0; i<ndof; ++i)
         {
            int pdof_ = patch_elem_dofs[i];
            int gdof_ = global_elem_dofs[i];
            int extdof_ = patch_elem_dofs_ext[i];
            int pdof = (pdof_ >= 0) ? pdof_ : abs(pdof_) - 1;
            int gdof = (gdof_ >= 0) ? gdof_ : abs(gdof_) - 1;
            int extdof = (extdof_ >= 0) ? extdof_ : abs(extdof_) - 1;
            patch_dof_map[ip][pdof] = gdof;
            patch_dof_map[ip][pdof+nrdof] = gdof+fespace->GetTrueVSize();
            dof2extdof_map[ip][pdof] = extdof;
            dof2extdof_map[ip][pdof+nrdof] = extdof+patch_fespaces_ext[ip]->GetTrueVSize();
         }
      }
      // Define the patch bilinear form and apply boundary conditions (only the LHS)
      Array <int> ess_temp_list;
      if (p->patch_mesh[ip]->bdr_attributes.Size())
      {
         Array<int> ess_bdr(p->patch_mesh[ip]->bdr_attributes.Max());
         ess_bdr = 0;
         patch_fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_temp_list);
      }

      Array <int> ess_list_ext;
      if (patch_meshes_ext[ip]->bdr_attributes.Size())
      {
         Array<int> ess_bdr(patch_meshes_ext[ip]->bdr_attributes.Max());
         ess_bdr = 0;
         patch_fespaces_ext[ip]->GetEssentialTrueDofs(ess_bdr, ess_list_ext);
      }

      // Adjust the essential tdof list for each patch
      for (int i=0; i<ess_temp_list.Size(); i++)
      {
         int ldof = ess_temp_list[i];
         int tdof = patch_dof_map[ip][ldof];
         // check the kind of this tdof
         if (!global_tdofs[tdof]) ess_tdof_list[ip].Append(ldof);
      }

      SesquilinearForm a(patch_fespaces[ip], &bf->real(), &bf->imag());
      SesquilinearForm a_ext(patch_fespaces_ext[ip], &bf->real(), &bf->imag());
      
      a.Assemble();
      a_ext.Assemble();
      OperatorPtr Alocal;
      a.FormSystemMatrix(ess_tdof_list[ip],Alocal);
      ComplexSparseMatrix * AZ = Alocal.As<ComplexSparseMatrix>();
      patch_mat[ip] = AZ->GetSystemMatrix();
      patch_mat[ip]->Threshold(0.0);
      // Save the inverse
      patch_mat_inv[ip] = new KLUSolver;
      patch_mat_inv[ip]->SetOperator(*patch_mat[ip]);


      OperatorPtr Alocal_ext;
      a_ext.FormSystemMatrix(ess_list_ext,Alocal_ext);
      ComplexSparseMatrix * AZ_ext = Alocal_ext.As<ComplexSparseMatrix>();
      patch_mat_ext[ip] = AZ_ext->GetSystemMatrix();
      patch_mat_ext[ip]->Threshold(0.0);
      patch_mat_inv_ext[ip] = new KLUSolver;
      patch_mat_inv_ext[ip]->SetOperator(*patch_mat_ext[ip]);


      delete patch_fespaces[ip];
      delete patch_fespaces_ext[ip];
   }
   delete p;
}

ComplexPatchAssembly::~ComplexPatchAssembly()
{
   for (int ip=0; ip<nrpatch; ++ip)
   {
      // delete patch_fespaces[ip]; patch_fespaces[ip]=nullptr;
      delete patch_meshes_ext[ip];
      patch_meshes_ext[ip]=nullptr;
      delete patch_mat_inv[ip];
      patch_mat_inv[ip]=nullptr;
      delete patch_mat[ip];
      patch_mat[ip]=nullptr;
   }
   patch_fespaces.DeleteAll();
   patch_meshes_ext.DeleteAll();
   patch_mat.DeleteAll();
   patch_mat_inv.DeleteAll();
}



ComplexAddSchwarz::ComplexAddSchwarz(SesquilinearForm * bf_, Array<int> & ess_tdofs, int i)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), bf(bf_),
     part(i)
{
   p = new ComplexPatchAssembly(bf_, ess_tdofs, part);
   nrpatch = p->nrpatch;
}

void  ComplexAddSchwarz::Mult(const Vector &r, Vector &z) const
{
   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   Array<int> visit(znew.Size());
   // char vishost[] = "localhost";
   // int  visport   = 19916;

   // socketstream sol_sock(vishost, visport);
   // sol_sock.precision(8);
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      visit = 0;
      for (int ip = 0; ip < nrpatch; ip++)
      {
         Array<int> * dof_map = &p->patch_dof_map[ip];
         int ndofs = dof_map->Size();
         res_local.SetSize(ndofs);
         sol_local.SetSize(ndofs);

         rnew.GetSubVector(*dof_map, res_local);

         //-----------------------------------------------
         // Extend by zero to the extended mesh
         int nrdof_ext = p->patch_mat_ext[ip]->Height();
         
         Vector res_ext(nrdof_ext); res_ext = 0.0;
         Vector sol_ext(nrdof_ext); sol_ext = 0.0;

         res_ext.SetSubVector(p->dof2extdof_map[ip],res_local.GetData());

         p->patch_mat_inv_ext[ip]->Mult(res_ext, sol_ext);

         sol_ext.GetSubVector(p->dof2extdof_map[ip],sol_local);


         //-----------------------------------------------

         // p->patch_mat_inv[ip]->Mult(res_local, sol_local);

         // for the overlapping case
         // zero out the entries corresponding to the ess_bdr
         Array<int> ess_bdr_indices_re = p->ess_tdof_list[ip]; // real part
         Array<int> ess_bdr_indices(2*ess_bdr_indices_re.Size()); //imag part

         for (int i = 0; i< ess_bdr_indices_re.Size(); i++)
         {
            ess_bdr_indices[i] = ess_bdr_indices_re[i];
            ess_bdr_indices[i+ess_bdr_indices_re.Size()] = ess_bdr_indices_re[i]+ndofs/2;
         }
         if (!part) 
         { 
            sol_local.SetSubVector(ess_bdr_indices,0.0); 
         }
         if (type == 1) znew = 0.0;

         znew.AddElementVector(*dof_map,sol_local);
         // zero out the contributions to the dofs which are already updated
         if (type == 1)
         {
            for (int i = 0; i<ndofs; i++)
            {
               int j = (*dof_map)[i];
               if (visit[j])
               {
                  znew(j) = 0.0;
               }
               else
               {
                  visit[j] = 1;
               }
            }
            z.Add(theta, znew);
            A->Mult(znew, raux);
            rnew -= raux;
         }
         // PlotSolution(z, sol_sock, ip); cin.get();
      }
      if (type == 0)
      {
         z.Add(theta, znew);
         A->Mult(znew, raux);
         rnew -= raux;
      }
      // Update residual
      if (iter + 1 < maxit)
      {
         A->Mult(znew, raux);
         rnew -= raux;
      }
   }
   // PlotSolution(z, sol_sock, 0); cin.get();
}


void ComplexAddSchwarz::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   ComplexGridFunction gf(fespace);
   bf->RecoverFEMSolution(sol,B,gf);
   
   string keys;
   if (ip == 0) keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf.real() << keys << flush;
}

ComplexAddSchwarz::~ComplexAddSchwarz(){ delete p;}
