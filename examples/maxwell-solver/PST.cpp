// Pure Source Transfer Preconditioner

#include "PST.hpp"

PSTP::PSTP(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   Mesh * mesh = bf->FESpace()->GetMesh();
   dim = mesh->Dimension();

   // ----------------- Step 1 --------------------
   // Introduce 2 layered partitios of the domain 
   // 
   int partition_kind;
   // 1. Non ovelapping 
   partition_kind = 1; // Non Ovelapping partition 
   pnovlp = new MeshPartition(mesh, partition_kind);

   // 2. Overlapping to the right
   partition_kind = 3; // Overlapping partition for the full space
   povlp = new MeshPartition(mesh, partition_kind);

   nrpatch = povlp->nrpatch;
   MFEM_VERIFY(povlp->nrpatch+1 == pnovlp->nrpatch,"Check nrpatch");
   //
   // ----------------- Step 1a -------------------
   // Save the partition for visualization
   // SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");
   // SaveMeshPartition(pnovlp->patch_mesh, "output/mesh_novlp.", "output/sol_novlp.");

   // ------------------Step 2 --------------------
   // Construct the dof maps from subdomains to global (for the extended and not)
   // The non ovelapping is extended on the left by pml (halfspace problem)
   // The overlapping is extended left and right by pml (unbounded domain problem)
   novlp_prob = new DofMap(bf,pnovlp);
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

SparseMatrix * PSTP::GetPmlSystemMatrix(int ip)
{
   double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);
   if (ip == nrpatch-1 || ip == 0) 
   {
      length[0][0] = Pmllength[0][0];
      length[0][1] = Pmllength[0][1];
   }
   length[1][0] = Pmllength[1][0];
   length[1][1] = Pmllength[1][1];

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

void PSTP::Mult(const Vector &r, Vector &z) const
{
   z = 0.0; 
   res.SetSize(nrpatch);
   Vector rnew(r);
   Vector rnew2(r);
   Vector znew(z);
   Vector znew1(z);
   Vector znew2(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   znew = 0.0;
   znew1= 0.0;
   znew2= 0.0;
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream subsol_sock(vishost, visport);

   // source transfer algorithm
   for (int ip = 0; ip < nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
      int ndofs = Dof2GlobalDof->Size();
      res_local.SetSize(ndofs);
      sol_local.SetSize(ndofs);
      rnew.GetSubVector(*Dof2GlobalDof, res_local);

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
      int direction = 1;
      if (ip < nrpatch-1) GetCutOffSolution(znew, ip, direction);

      A->Mult(znew, raux);
      rnew -= raux;
      znew1 += znew;
   }

   PlotSolution(znew1, subsol_sock,0); cin.get();

   for (int ip = nrpatch-1; ip >=0; ip--)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
      int ndofs = Dof2GlobalDof->Size();
      res_local.SetSize(ndofs);
      sol_local.SetSize(ndofs);
      rnew2.GetSubVector(*Dof2GlobalDof, res_local);

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
      int direction = -1;
      if (ip > 0) GetCutOffSolution(znew, ip-1, direction);

      A->Mult(znew, raux);
      rnew2 -= raux;
      znew2 += znew;
   }

   PlotSolution(znew2, subsol_sock,0); cin.get();


}

void PSTP::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = bf->FESpace();
   Mesh * mesh = fespace->GetMesh();
   GridFunction gf(fespace);
   double * data = sol.GetData();
   gf.SetData(data);
   
   string keys = "keys z\n";
   if (ip ==0) keys = "keys rRljc\n";
   sol_sock << "solution\n" << *mesh << gf << flush;
   
}

void PSTP::GetCutOffSolution(Vector & sol, int ip, int direction) const
{

   int l,k;

   l=(direction == 1)? ip+1: ip;
   k=(direction == 1)? ip: ip+1;

   Mesh * mesh1 = ovlp_prob->fespaces[l]->GetMesh();
   Mesh * mesh2 = ovlp_prob->fespaces[k]->GetMesh();
   
   Vector pmin1, pmax1;
   Vector pmin2, pmax2;
   mesh1->GetBoundingBox(pmin1, pmax1);
   mesh2->GetBoundingBox(pmin2, pmax2);

   Array2D<double> h(dim,2);
   
   h[0][0] = pmin2[0] - pmin1[0];
   h[0][1] = pmax2[0] - pmin1[0];
   h[1][0] = pmin2[1] - pmin1[1];
   h[1][1] = pmax2[1] - pmax1[1];

   if (direction == 1)
   {
      h[0][0] = 0.0;
   }
   else if (direction == -1)
   {
      h[0][1] = 0.0;
   }
   CutOffFnCoefficient cf(CutOffFncn, pmin2, pmax2, h);
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

PSTP::~PSTP()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete PmlMatInv[ip];
      delete PmlMat[ip];
   }
   PmlMat.DeleteAll();
   PmlMatInv.DeleteAll();
}

