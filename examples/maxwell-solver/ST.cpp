#include "ST.hpp"



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
      if (ip < nrpatch-2)
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


STP::STP(SesquilinearForm * bf_, Array<int> & ess_tdofs, 
         double omega_, FunctionCoefficient * ws_,  int nrlayers_)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
     bf(bf_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   Mesh * mesh = bf->FESpace()->GetMesh();
   dim = mesh->Dimension();

   // ----------------- Step 1 --------------------
   // Introduce 2 layered partitios of the domain 
   // 
   int partition_kind;
   // 1. Non ovelapping 
   partition_kind = 4; // Ovelapping partition for the halfspace problem
   pnovlp = new MeshPartition(mesh, partition_kind);

   // 2. Overlapping to the right
   partition_kind = 3; // Ovelapping partition for the full space
   povlp = new MeshPartition(mesh, partition_kind);

   // nrpatch = pnovlp->nrpatch-1;
   nrpatch = pnovlp->nrpatch;
   //
   // ----------------- Step 1a -------------------
   // Save the partition for visualization
   // SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");
   // SaveMeshPartition(pnovlp->patch_mesh, "output/mesh_novlp.", "output/sol_novlp.");

   // ------------------Step 2 --------------------
   // Construct the dof maps from subdomains to global (for the extended and not)
   // The non ovelapping is extended on the left by pml (halfspace problem)
   // The overlapping is extended left and right by pml (unbounded domain problem)
   novlp_prob = new DofMap(bf,pnovlp,nrlayers);
   ovlp_prob  = new DofMap(bf,povlp,nrlayers); 

   // ------------------Step 3 --------------------
   // Assemble the PML Problem matrices and factor them

   PmlMat.SetSize(nrpatch);
   PmlMatInv.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      PmlMat[ip] = GetPmlSystemMatrix(ip);
      PmlMatInv[ip] = new KLUSolver;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
   }
}

SparseMatrix * STP::GetPmlSystemMatrix(int ip)
{
   double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);
   // if (ip == nrpatch-1) length[0][1] = 0.0;
   // if (ip == 0) length[0][0] = 0.0;
   // length[1][0] = 0.0;
   // length[1][1] = 0.0;
   // length[0][0] = 0.0;
   CartesianPML pml(ovlp_prob->PmlMeshes[ip], length);
   pml.SetOmega(omega);

   Array <int> ess_tdof_list;
   if (ovlp_prob->PmlMeshes[ip]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(ovlp_prob->PmlMeshes[ip]->bdr_attributes.Max());
      ess_bdr = 1;
      ovlp_prob->PmlFespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

   PmlCoefficient detJ_re(pml_detJ_Re,&pml);
   PmlCoefficient detJ_im(pml_detJ_Im,&pml);

   ProductCoefficient c2_re0(sigma, detJ_re);
   ProductCoefficient c2_im0(sigma, detJ_im);

   ProductCoefficient c2_re(c2_re0, *ws);
   ProductCoefficient c2_im(c2_im0, *ws);

   SesquilinearForm a(ovlp_prob->PmlFespaces[ip],ComplexOperator::HERMITIAN);

   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();

   OperatorPtr Alocal;
   a.FormSystemMatrix(ess_tdof_list,Alocal);
   ComplexSparseMatrix * AZ_ext = Alocal.As<ComplexSparseMatrix>();
   SparseMatrix * Mat = AZ_ext->GetSystemMatrix();
   Mat->Threshold(0.0);
   return Mat;
}



void STP::SolveHalfSpaceLinearSystem(int ip, Vector &x, Vector & load) const
{
   double h = GetUniformMeshElementSize(novlp_prob->PmlMeshes[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);
   length[0][1] = 0.0;
   // length[1][1] = 0.0;
   // length[1][0] = 0.0;
   // length[0][0] = 0.0;
   // length[1][1] = 0.0;
   // if (ip == 0) length[0][0] = 0.0;

   CartesianPML pml(novlp_prob->PmlMeshes[ip], length);
   pml.SetOmega(omega);

   Array <int> ess_tdof_list;
   if (novlp_prob->PmlMeshes[ip]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(ovlp_prob->PmlMeshes[ip]->bdr_attributes.Max());
      ess_bdr = 1;
      novlp_prob->PmlFespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

   PmlCoefficient detJ_re(pml_detJ_Re,&pml);
   PmlCoefficient detJ_im(pml_detJ_Im,&pml);

   ProductCoefficient c2_re0(sigma, detJ_re);
   ProductCoefficient c2_im0(sigma, detJ_im);

   ProductCoefficient c2_re(c2_re0, *ws);
   ProductCoefficient c2_im(c2_im0, *ws);

   SesquilinearForm a(novlp_prob->PmlFespaces[ip],ComplexOperator::HERMITIAN);

   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();

   OperatorHandle Ah;
   Vector X;
   Vector Modload;

   a.FormLinearSystem(ess_tdof_list, x, load, Ah, X, Modload);

   ComplexSparseMatrix * AZ = Ah.As<ComplexSparseMatrix>();

   KLUSolver * HSmatInv = new KLUSolver;
   HSmatInv->SetOperator(*AZ->GetSystemMatrix());
   HSmatInv->Mult(Modload,X);
   delete HSmatInv;
   a.RecoverFEMSolution(X,Modload,x);
}


void STP::Mult(const Vector &r, Vector &z) const
{
   z = 0.0; 
   res.SetSize(nrpatch);
   Vector rnew(r);
   Vector znew(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   znew = 0.0;
   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream subsol_sock(vishost, visport);

   // source transfer algorithm
   for (int ip = 0; ip < nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
      int ndofs = Dof2GlobalDof->Size();
      res_local.SetSize(ndofs);
      sol_local.SetSize(ndofs);

      rnew.GetSubVector(*Dof2GlobalDof, res_local);

      // store residuals for the non overlapping partition
      Array<int> * nDof2GlobalDof;
      if (ip == nrpatch-1 )
      {
         nDof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      }
      else
      {
         nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
      }
      int mdofs = nDof2GlobalDof->Size();
      res[ip] = new Vector(mdofs);
      rnew.GetSubVector(*nDof2GlobalDof, *res[ip]);
      if (ip == nrpatch-1) continue;

      //-----------------------------------------------
      // Extend by zero to the extended mesh
      int nrdof_ext = PmlMat[ip]->Height();
         
      Vector res_ext(nrdof_ext); res_ext = 0.0;
      Vector sol_ext(nrdof_ext); sol_ext = 0.0;

      res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
      PmlMatInv[ip]->Mult(res_ext, sol_ext);

      sol_ext.GetSubVector(*Dof2PmlDof,sol_local);

      znew = 0.0;
      znew.SetSubVector(*Dof2GlobalDof,sol_local);

      // PlotSolution(znew, subsol_sock,ip); cin.get();

      GetCutOffSolution(znew, ip);

      // PlotSolution(znew, subsol_sock,1); cin.get();

      A->Mult(znew, raux);
      rnew -= raux;
      // PlotSolution(rnew, subsol_sock,ip); cin.get();

   }

   // solution stage
   // First solve the nrpatch-1 problem (last subdomain)
   // extend residual to all around pml
   // cout << "ip = " << nrpatch-1 << endl;
   int nrdof_ext = PmlMat[nrpatch-1]->Height();
   Vector res_ext(nrdof_ext); res_ext = 0.0;
   Vector sol_ext(nrdof_ext); sol_ext = 0.0;
   Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[nrpatch-1];
   Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[nrpatch-1];
   res_ext.SetSubVector(*Dof2PmlDof,*res[nrpatch-1]);
   PmlMatInv[nrpatch-1]->Mult(res_ext, sol_ext);
   int ndofs = Dof2GlobalDof->Size();
   sol_local.SetSize(ndofs);
   sol_ext.GetSubVector(*Dof2PmlDof,sol_local);
   znew = 0.0;
   znew.SetSubVector(*Dof2GlobalDof,sol_local);
   z.SetSubVector(*Dof2GlobalDof,sol_local);

   // PlotSolution(z, subsol_sock,0); cin.get();

   // backward sweep for half space problems

   Vector z_loc(z.Size());
   for (int ip = nrpatch-2; ip >= 0; ip--)
   {
      // cout << "ip = " << ip<< endl;
      // Get solution from previous layer
      Array<int> * Dof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &novlp_prob->Dof2PmlDof[ip];
      int ndof = Dof2GlobalDof->Size();
      Vector sol_loc(ndof);      
      znew.GetSubVector(* Dof2GlobalDof, sol_loc);

      // extend by zero to the halfspace pml problem
      FiniteElementSpace * subfespace = novlp_prob->PmlFespaces[ip];
      int mdof = 2*subfespace->GetTrueVSize();
      Vector sol_pml(mdof); sol_pml = 0.0;
      sol_pml.SetSubVector(* Dof2PmlDof, sol_loc);
      Mesh * submesh = subfespace->GetMesh();

      // Set to zero the non boundary dofs
      Array<int> ess_tdof_list;
      Array<int> ess_bdr(submesh->bdr_attributes.Max());
      ess_bdr = 1;
      subfespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      int n = ess_tdof_list.Size();
      for (int i=0; i<n; i++)
      {
         ess_tdof_list.Append(ess_tdof_list[i]+mdof/2);
      }
      sol_pml.SetSubVectorComplement(ess_tdof_list,0.0);
      
      // Set up the halfspace problem
      // extend the residual by zero to pml region
      Vector pmlres(sol_pml.Size()); pmlres = 0.0;
      pmlres.SetSubVector(* Dof2PmlDof,*res[ip]);
      SolveHalfSpaceLinearSystem(ip, sol_pml, pmlres);

      sol_loc = 0.0;
      sol_pml.GetSubVector(* Dof2PmlDof, sol_loc);

      z_loc = 0.0;
      z_loc.SetSubVector(* Dof2GlobalDof, sol_loc);
      znew = z_loc;
      z.SetSubVector(* Dof2GlobalDof, sol_loc);
      // PlotSolution(z, subsol_sock,1); cin.get();
   }

}

void STP::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   GridFunction gf(fespace);
   double * data = sol.GetData();
   gf.SetData(data);
   
   string keys;
   if (ip == 0) keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf << keys << flush;
}

void STP::GetCutOffSolution(Vector & sol, int ip) const
{
   Mesh * novlp_mesh = novlp_prob->fespaces[ip+1]->GetMesh();
   Mesh * ovlp_mesh = ovlp_prob->fespaces[ip]->GetMesh();
   Vector novlpmin, novlpmax;
   Vector ovlpmin, ovlpmax;
   novlp_mesh->GetBoundingBox(novlpmin, novlpmax);
   ovlp_mesh->GetBoundingBox(ovlpmin, ovlpmax);

   Array2D<double> h(dim,2);
   h[0][0] = 0.0;
   h[0][1] = ovlpmax[0] - novlpmin[0];
   h[1][0] = ovlpmin[1] - novlpmin[1];
   h[1][1] = ovlpmax[1] - novlpmax[1];


   CutOffFnCoefficient cf(CutOffFncn, ovlpmin, ovlpmax, h);

   double * data = sol.GetData();

   
   FiniteElementSpace * fespace = bf->FESpace();
   int n = fespace->GetTrueVSize();

   GridFunction solgf_re(fespace, data);
   GridFunction solgf_im(fespace, &data[n]);


   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fespace);
   gf.ProjectCoefficient(prod_re,prod_im);

   sol = gf;
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
