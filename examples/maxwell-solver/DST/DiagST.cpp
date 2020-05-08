//Diagonal Source Transfer Preconditioner

#include "DiagST.hpp"

DiagST::DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
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
   partition_kind = 3; // Ovelapping partition for the full space
   povlp = new MeshPartition(mesh, partition_kind);
   nrpatch = povlp->nrpatch;
   cout<< povlp->nrpatch << endl;
   cout<< pnovlp->nrpatch << endl;
   MFEM_VERIFY(povlp->nrpatch+1 == pnovlp->nrpatch,"Check nrpatch");

   lmap = new LocalDofMap(bf->FESpace()->FEColl(),pnovlp,povlp);

   //
   // ----------------- Step 1a -------------------
   // Save the partition for visualization
   // SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");

   // ------------------Step 2 --------------------
   // Construct the dof maps from subdomains to global (for the extended and not)
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

SparseMatrix * DiagST::GetPmlSystemMatrix(int ip)
{
   double h = GetUniformMeshElementSize(ovlp_prob->PmlMeshes[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);
   // if (ip == nrpatch-1 || ip == 0) 
   // {
   //    length[0][0] = Pmllength[0][0];
   //    length[0][1] = Pmllength[0][1];
   // }
   // length[1][0] = Pmllength[1][0];
   // length[1][1] = Pmllength[1][1];

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


void DiagST::Mult(const Vector &r, Vector &z) const
{
   z = 0.0; 
   res.SetSize(nrpatch);
   Vector rnew(r);
   Vector znew(z);
   Vector z1(z);
   Vector z2(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   znew = 0.0;
   Array<Vector> fsol(nrpatch+1);
   Array<Vector> bsol(nrpatch+1);

   Array<Vector> fn(nrpatch+1);
   Array<Vector> ftransf(nrpatch+1);
   for (int ip=0; ip<=nrpatch; ip++)
   {
      Array<int> *Dof2GDof = &novlp_prob->Dof2GlobalDof[ip];
      fn[ip].SetSize(Dof2GDof->Size()); fn[ip]=0.0;
      ftransf[ip].SetSize(Dof2GDof->Size()); ftransf[ip]=0.0;
      r.GetSubVector(*Dof2GDof,fn[ip]);
   }


   char vishost[] = "localhost";
   int  visport   = 19916;
   // forward source transfer algorithm
   for (int ip = 0; ip < nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
      int ndofs = Dof2GlobalDof->Size();
      res_local.SetSize(ndofs); res_local = 0.0;
      sol_local.SetSize(ndofs);
      rnew.GetSubVector(*Dof2GlobalDof, res_local);

      // if (ip == 0)  ftransf[ip] = fn[ip];

      // res_local = 0.0;
      // res_local.SetSubVector(lmap->map1[ip],ftransf[ip]);
      // res_local.SetSubVector(lmap->map2[ip],fn[ip+1]);
      

      //-----------------------------------------------
      // Extend by zero to the PML mesh
      int nrdof_ext = PmlMat[ip]->Height();
         
      Vector res_ext(nrdof_ext); res_ext = 0.0;
      Vector sol_ext(nrdof_ext); sol_ext = 0.0;

      res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
      PmlMatInv[ip]->Mult(res_ext, sol_ext);

      //-----------------------------------------------
      //          FOR PURE ST
      //-----------------------------------------------
      if (ip != nrpatch-1) 
      {
         int direction = 1;
         GetCutOffSolution(sol_ext, ip, direction, true);
         // find the residual
         // Vector respml(sol_ext.Size());
         // PmlMat[ip]->Mult(sol_ext,respml);

         // restrict to non-pml problem
         // Vector res(ndofs); 
         // respml.GetSubVector(*Dof2PmlDof, res);
         // source to be transfered
         // res.GetSubVector(lmap->map2[ip],ftransf[ip+1]);

      }
      //-----------------------------------------------
      //-----------------------------------------------



      sol_ext.GetSubVector(*Dof2PmlDof,sol_local);
      znew = 0.0;
      znew.SetSubVector(*Dof2GlobalDof,sol_local);

      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
      fsol[ip+1].SetSize(nDof2GlobalDof->Size());
      znew.GetSubVector(*nDof2GlobalDof,fsol[ip+1]);



      // Find residual;
      Vector respml(sol_ext.Size());
      Vector reslocal(sol_local.Size());
      PmlMat[ip]->Mult(sol_ext,respml);
      respml *= -1.0;

      respml.GetSubVector(*Dof2PmlDof,reslocal);
      // source to be transfered
      // reslocal.GetSubVector(lmap->map2[ip],ftransf[ip+1]);
      // ftransf[ip+1]+= fn[ip+1];
      rnew.AddElementVector(*Dof2GlobalDof,reslocal);

      // Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
      // fsol[ip+1].SetSize(nDof2GlobalDof->Size());
      // znew.GetSubVector(*nDof2GlobalDof,fsol[ip+1]);

      // if (ip != nrpatch-1) 
      // {
      //    int direction = 1;
      //    GetCutOffSolution(znew, ip, direction);
      //    // find the residual
      // }
      
      // socketstream sockznew(vishost, visport);
      // PlotSolution(znew,sockznew,0); cin.get();
      z1+=znew;

      // A->Mult(znew, raux);
      // rnew -= raux;
   }

   // backward source transfer algorithm
   // for (int ip=0; ip<=nrpatch; ip++)
   // {
   //    Array<int> *Dof2GDof = &novlp_prob->Dof2GlobalDof[ip];
   //    fn[ip].SetSize(Dof2GDof->Size()); fn[ip] = 0.0;
   //    ftransf[ip].SetSize(Dof2GDof->Size()); ftransf[ip] = 0.0;
   //    r.GetSubVector(*Dof2GDof,fn[ip]);
   // }
   // rnew = r;
   // for (int ip = nrpatch-1; ip >=0; ip--)
   // {
   //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndofs = Dof2GlobalDof->Size();
   //    res_local.SetSize(ndofs); res_local = 0.0;
   //    sol_local.SetSize(ndofs);
   //    rnew.GetSubVector(*Dof2GlobalDof, res_local);

   //    // if (ip == nrpatch-1) ftransf[ip+1] = fn[ip+1];
   //    // res_local = 0.0;
   //    // res_local.SetSubVector(lmap->map1[ip],ftransf[ip]);
   //    // res_local.SetSubVector(lmap->map2[ip],fn[ip+1]);

   //    //-----------------------------------------------
   //    // Extend by zero to the extended mesh
   //    int nrdof_ext = PmlMat[ip]->Height();
         
   //    Vector res_ext(nrdof_ext); res_ext = 0.0;
   //    Vector sol_ext(nrdof_ext); sol_ext = 0.0;

   //    res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
   //    PmlMatInv[ip]->Mult(res_ext, sol_ext);

   //    //-----------------------------------------------
   //    //          FOR PURE ST
   //    //-----------------------------------------------
   //    if (ip != 0) 
   //    {
   //       int direction = -1;
   //       GetCutOffSolution(sol_ext, ip, direction, true);
   //       // find the residual
   //       // Vector respml(sol_ext.Size());
   //       // PmlMat[ip]->Mult(sol_ext,respml);

   //       // restrict to non-pml problem
   //       // Vector res(ndofs); 
   //       // respml.GetSubVector(*Dof2PmlDof, res);
   //       // source to be transfered
   //       // res.GetSubVector(lmap->map2[ip],ftransf[ip+1]);

   //    }
   //    //-----------------------------------------------
   //    //-----------------------------------------------

   //    sol_ext.GetSubVector(*Dof2PmlDof,sol_local);

   //    znew = 0.0;
   //    znew.SetSubVector(*Dof2GlobalDof,sol_local);

   //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   //    bsol[ip].SetSize(nDof2GlobalDof->Size());
   //    znew.GetSubVector(*nDof2GlobalDof,bsol[ip]);

   //    // Find residual;
   //    Vector respml(sol_ext.Size());
   //    Vector reslocal(sol_local.Size());
   //    PmlMat[ip]->Mult(sol_ext,respml);
   //    respml *= -1.0;

   //    respml.GetSubVector(*Dof2PmlDof,reslocal);
   //    // source to be transfered
   //    // reslocal.GetSubVector(lmap->map1[ip],ftransf[ip]);
   //    // ftransf[ip]+= fn[ip];
   //    rnew.AddElementVector(*Dof2GlobalDof,reslocal);

   // //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   // //    bsol[ip].SetSize(nDof2GlobalDof->Size());
   // //    znew.GetSubVector(*nDof2GlobalDof,bsol[ip]);
   // //    if (ip != 0) 
   // //    {
   // //       int direction = -1;
   // //       GetCutOffSolution(znew, ip, direction);
   // //    }
   // //    if (ip != nrpatch-1) z2+=znew;
   // //    A->Mult(znew, raux);
   // //    rnew -= raux;
   //    z2+=znew;
   // }

   // z = z2;
   // Array<Vector> gsol(nrpatch+1);
   // // socketstream subsol5_sock(vishost, visport);

   // for (int ip = 0; ip<=nrpatch; ip++)
   // {
   //    if (ip == 0) 
   //    {
   //       gsol[ip].SetSize(bsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       gsol[ip] = bsol[ip];
   //    }
   //    else if (ip == nrpatch)
   //    {
   //       gsol[ip].SetSize(fsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       // gsol[ip] = fsol[ip];
   //    }
   //    else
   //    {
   //       gsol[ip].SetSize(fsol[ip].Size());
   //       gsol[ip] = 0.0;
   //       gsol[ip] += bsol[ip];
   //       // gsol[ip] += fsol[ip];
   //    }
      
   //    znew = 0.0;
   //    Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
   //    znew.SetSubVector(*nDof2GlobalDof,gsol[ip]);
   //    // PlotSolution(znew, subsol5_sock,0); cin.get();
   //    z.SetSubVector(*nDof2GlobalDof,gsol[ip]);
   //    // z.AddElementVector(*nDof2GlobalDof,gsol[ip]);
   // }
   z = z1;

}

void DiagST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const
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

void DiagST::GetCutOffSolution(Vector & sol, int ip, int direction, bool local) const
{
   int l,k;
   k=(direction == 1)? ip: ip-1;
   l=(direction == 1)? ip+1: ip;

   Mesh * mesh1 = ovlp_prob->fespaces[k]->GetMesh();
   Mesh * mesh2 = ovlp_prob->fespaces[l]->GetMesh();
   
   Vector pmin1, pmax1;
   Vector pmin2, pmax2;
   mesh1->GetBoundingBox(pmin1, pmax1);
   mesh2->GetBoundingBox(pmin2, pmax2);

   Array2D<double> h(dim,2); h = 0.0;
   
   Vector pmin, pmax;
   if (direction == 1)
   {
      h[0][1] = pmax1[0] - pmin2[0];
      CutOffFnCoefficient cf(CutOffFncn, pmin1, pmax1, h);
      pmin = pmin1;
      pmax = pmax1;
   }
   else if (direction == -1)
   {
      h[0][0] = pmax1[0] - pmin2[0];
      pmin = pmin2;
      pmax = pmax2;
   }
   CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, h);

   double * data = sol.GetData();

   FiniteElementSpace * fespace;
   if (!local)
   {
      fespace = bf->FESpace();
   }
   else
   {
      if (direction == 1)
      {
         fespace = ovlp_prob->PmlFespaces[k];
      }
      else
      {
         fespace = ovlp_prob->PmlFespaces[l];
      }
   }
   
   int n = fespace->GetTrueVSize();
   GridFunction cutF(fespace);
   cutF.ProjectCoefficient(cf);
   // char vishost[] = "localhost";
   // int  visport   = 19916;

   

   // socketstream sub_sock1(vishost, visport);
   // sub_sock1 << "solution\n" << *fespace->GetMesh() << cutF << flush;
   // cin.get();


   GridFunction solgf_re(fespace, data);

   // socketstream sub_sock(vishost, visport);
   // sub_sock << "solution\n" << *fespace->GetMesh() << solgf_re << flush;
   // cin.get();

   GridFunction solgf_im(fespace, &data[n]);

   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fespace);
   gf.ProjectCoefficient(prod_re,prod_im);

   sol = gf;
   // socketstream sub_sock2(vishost, visport);
   // sub_sock2 << "solution\n" << *fespace->GetMesh() << gf.real() << flush;
   // cin.get();
}

DiagST::~DiagST()
{
   for (int ip = 0; ip<nrpatch; ++ip)
   {
      delete PmlMatInv[ip];
      delete PmlMat[ip];
   }
   PmlMat.DeleteAll();
   PmlMatInv.DeleteAll();
}



