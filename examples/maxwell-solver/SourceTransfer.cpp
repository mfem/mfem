#include "SourceTransfer.hpp"

STPmlPatchAssembly::STPmlPatchAssembly(SesquilinearForm * bf_, Array<int> & ess_tdofs, 
                                   double omega_, int nrlayers_, int part) 
                                   : bf(bf_), omega(omega_), nrlayers(nrlayers_)
{
   fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   int dim = mesh->Dimension();
   const FiniteElementCollection *fec = fespace->FEColl();

   p = new MeshPartition(mesh, part);
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
   ess_tdof_list_ext.resize(nrpatch);

   // construct extended meshes for the pml

   int ip = -1;
   for (int kz = 0; kz<nz; kz++)
   {
      for (int ky = 0; ky<ny; ky++)
      {
         for (int kx = 0; kx<nx; kx++)
         {
            ip++;
            Array<int> ext_directions;
            for (int j=0; j<nrlayers; ++j)// one more layer of extension (epsilon layer)
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
                     // ext_directions.Append(-comp-1);
                  }
                  if (comp == 2 && kz != nz-1)
                  {
                     // ext_directions.Append(comp+1);
                  }
               }
            }
            if (ip < nrpatch-1)
            {
               // ext_directions.Append(1);
               // ext_directions.Append(1);
               // ext_directions.Append(1);
               // ext_directions.Append(1);
            }
            patch_meshes_ext[ip] = ExtendMesh(p->patch_mesh[ip],ext_directions);
         }
      }
   }
   // SaveMeshPartition(patch_meshes_ext, "output/ext_mesh.", "output/ext_sol.");
   // // cout << p->patch_mesh[0]->GetNE() << endl;


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
      // loop through the patch elements and constract the dof map
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
      // // Define the patch bilinear form and apply boundary conditions (only the LHS)
      // Array <int> ess_temp_list;
      // if (p->patch_mesh[ip]->bdr_attributes.Size())
      // {
      //    Array<int> ess_bdr(p->patch_mesh[ip]->bdr_attributes.Max());
      //    ess_bdr = 0;
      //    patch_fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_temp_list);
      // }

      Array <int> ess_list_ext;
      if (patch_meshes_ext[ip]->bdr_attributes.Size())
      {
         Array<int> ess_bdr(patch_meshes_ext[ip]->bdr_attributes.Max());
         ess_bdr = 1;
         patch_fespaces_ext[ip]->GetEssentialTrueDofs(ess_bdr, ess_list_ext);
      }
      ess_tdof_list_ext[ip] = ess_list_ext;

      // // Adjust the essential tdof list for each patch
      // for (int i=0; i<ess_temp_list.Size(); i++)
      // {
      //    int ldof = ess_temp_list[i];
      //    int tdof = patch_dof_map[ip][ldof];
      //    // check the kind of this tdof
      //    if (!global_tdofs[tdof]) ess_tdof_list[ip].Append(ldof);
      // }

      // SesquilinearForm a(patch_fespaces[ip], &bf->real(), &bf->imag());
      
      //-----------------PML FORMULATION----------------------------
      Array2D<double> length(dim,2);
      double h = GetUniformMeshElementSize(patch_meshes_ext[ip]);
      length = h*(nrlayers);

      if (ip < nrpatch-1)
      {
         // length(0,1) = h*(nrlayers+4);
      }
      // if (ip != 0) 
      // {
      //    length(0,0) = 0.0;
      // }
      // // length = h * nrlayers;
      // // if (ip != 0)
      // // {
      // //    length(0,0) = 0.0;
      // //    length(1,0) = 0.0;
      // // }
      // // length(0,1) = h * nrlayers;
      // // length(1,1) = h * nrlayers;
      // // if (ip == 1 || ip == 2 || ip == 3) length(1,0) = h * nrlayers;
      // // if (ip == 4 || ip == 8 || ip == 12) length(0,0) = h * nrlayers;


      CartesianPML pml(patch_meshes_ext[ip], length);
      pml.SetOmega(omega);

      ConstantCoefficient one(1.0);
      ConstantCoefficient sigma(-pow(omega, 2));

      PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
      PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

      PmlCoefficient detJ_re(pml_detJ_Re,&pml);
      PmlCoefficient detJ_im(pml_detJ_Im,&pml);

      ProductCoefficient c2_re(sigma, detJ_re);
      ProductCoefficient c2_im(sigma, detJ_im);

      SesquilinearForm a_ext(patch_fespaces_ext[ip],ComplexOperator::HERMITIAN);

      a_ext.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                                new DiffusionIntegrator(c1_im));
      a_ext.AddDomainIntegrator(new MassIntegrator(c2_re),
                                new MassIntegrator(c2_im));

      //------------------------------------------------------------

      // a.Assemble();
      a_ext.Assemble();
      // OperatorPtr Alocal;
      // a.FormSystemMatrix(ess_tdof_list[ip],Alocal);
      // ComplexSparseMatrix * AZ = Alocal.As<ComplexSparseMatrix>();
      // patch_mat[ip] = AZ->GetSystemMatrix();
      // patch_mat[ip]->Threshold(0.0);
      // // Save the inverse
      // patch_mat_inv[ip] = new KLUSolver;
      // patch_mat_inv[ip]->SetOperator(*patch_mat[ip]);


      OperatorPtr Alocal_ext;
      a_ext.FormSystemMatrix(ess_list_ext,Alocal_ext);
      ComplexSparseMatrix * AZ_ext = Alocal_ext.As<ComplexSparseMatrix>();
      patch_mat_ext[ip] = AZ_ext->GetSystemMatrix();
      patch_mat_ext[ip]->Threshold(0.0);
      patch_mat_inv_ext[ip] = new KLUSolver;
      patch_mat_inv_ext[ip]->SetOperator(*patch_mat_ext[ip]);


      // delete patch_fespaces[ip];
      // delete patch_fespaces_ext[ip];
   }
   // delete p;
}

STPmlPatchAssembly::~STPmlPatchAssembly()
{
   for (int ip=0; ip<nrpatch; ++ip)
   {
      // delete patch_fespaces[ip]; patch_fespaces[ip]=nullptr;
      delete patch_fespaces[ip];
      delete patch_fespaces_ext[ip];
      delete patch_meshes_ext[ip];
      patch_meshes_ext[ip]=nullptr;
      // delete patch_mat_inv[ip];
      delete patch_mat_inv_ext[ip];
      // patch_mat_inv[ip]=nullptr;
      patch_mat_inv_ext[ip]=nullptr;
      // delete patch_mat[ip];
      delete patch_mat_ext[ip];
      // patch_mat[ip]=nullptr;
      patch_mat_ext[ip]=nullptr;
   }
   // patch_fespaces.DeleteAll();
   patch_meshes_ext.DeleteAll();
   patch_mat_ext.DeleteAll();
   // patch_mat.DeleteAll();
   // patch_mat_inv.DeleteAll();
   // patch_mat_inv.DeleteAll();
   // delete p;
}




void SourceTransferPrecond::GetCutOffSolution(Vector & sol, int ip) const
{

   Mesh * mesh = p->patch_fespaces[ip]->GetMesh();
   int n = p->patch_fespaces[ip]->GetTrueVSize();
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   int dim = mesh->Dimension();
   double hl = GetUniformMeshElementSize(mesh);
   Array2D<double> h(dim,2);
   h[0][0] = 0.0;
   h[0][1] = hl;
   h[1][0] = 0.0;
   h[1][1] = 0.0;
   CutOffFunctionCoefficient cf(CutOffFn, pmin, pmax, h);

   double * data = sol.GetData();

   GridFunction solgf_re(p->patch_fespaces[ip], data);
   GridFunction solgf_im(p->patch_fespaces[ip], &data[n]);


   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(p->patch_fespaces[ip]);
   gf.ProjectCoefficient(prod_re,prod_im);

   sol = gf;
}






SourceTransferPrecond::SourceTransferPrecond(SesquilinearForm * bf_, Array<int> & ess_tdofs, double omega_, int nrlayers_, int i)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), bf(bf_), omega(omega_), nrlayers(nrlayers_),
     part(i)
{
   p = new STPmlPatchAssembly(bf_, ess_tdofs, omega, nrlayers, part);
   nrpatch = p->nrpatch;
}

void SourceTransferPrecond::Mult(const Vector &r, Vector &z) const
{
   z = 0.0;
   Vector rnew(r);
   Vector znew(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   Array<int> visit(znew.Size());
   char vishost[] = "localhost";
   int  visport   = 19916;

   // zero out sources from other subdomains
   // save the first subdomain
   // rnew = 0.0;
   // Array<int> * dof_map0 = &p->patch_dof_map[0];
   // int ndofs = dof_map0->Size();
   // res_local.SetSize(ndofs);
   // r.GetSubVector(*dof_map0, res_local);
   // rnew.SetSubVector(*dof_map0,res_local.GetData());


   // socketstream sol_sock(vishost, visport);
   // sol_sock.precision(8);
   // socketstream res_sock(vishost, visport);
   // res_sock.precision(8);
   // cout << "nrpatch = " << nrpatch << endl; 
   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      visit = 0;
      for (int ip = 0; ip < nrpatch; ip++)
      {
         // cout << "ip = " << ip << endl;
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

         // Smooth the solution before transfer
         // if (ip < nrpatch-1) GetCutOffSolution(sol_local, ip);

         if (type == 1) znew = 0.0;
         znew.AddElementVector(*dof_map,sol_local);
         // zero out the contributions to the dofs which are already updated
         // for (int i = 0; i<ndofs; i++)
         // {
         //    int j = (*dof_map)[i];
         //    if (visit[j])
         //    {
         //       znew(j) = 0.0;
         //    }
         //    else
         //    {
         //       visit[j] = 1;
         //    }
         // }
         if (type == 1)
         {
            z.Add(theta, znew);
            A->Mult(znew, raux);
            rnew -= raux;
         }
         // PlotSolution(z, sol_sock, ip); cin.get();
         // PlotSolution(rnew, res_sock, ip); cin.get();
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
   // PlotSolution(rnew, sol_sock, 0); cin.get();
}


void SourceTransferPrecond::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   ComplexGridFunction gf(fespace);
   bf->RecoverFEMSolution(sol,B,gf);
   
   string keys;
   if (ip == 0) keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf.imag() << keys << flush;
}

SourceTransferPrecond::~SourceTransferPrecond(){ }



double CutOffFn(const Vector &x, const Vector & pmin, const Vector & pmax, const Array2D<double> & h_)
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
      f *= val;
   }
   return f;
}
