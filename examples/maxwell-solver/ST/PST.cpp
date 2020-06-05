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


   lmap = new LocalDofMap(bf->FESpace()->FEColl(),pnovlp,povlp);

   // Given the two partitions create a dof map between the non-ovelapping 
   // subdomain dofs and the overlapping ones


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

   // Given 

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
   Vector znew(z);
   Vector z1(z);
   Vector z2(z);
   Vector raux(znew.Size());
   Vector res_local, sol_local;
   znew = 0.0;
   char vishost[] = "localhost";
   int  visport   = 19916;
   Array<Vector> fsol(nrpatch+1);
   Array<Vector> bsol(nrpatch+1);

   // source transfer algorithm
   for (int ip = 0; ip < nrpatch; ip++)
   {
      // cout << "ip = " << ip << endl;
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
      int ndofs = Dof2GlobalDof->Size();
      res_local.SetSize(ndofs);
      sol_local.SetSize(ndofs);

      rnew.GetSubVector(*Dof2GlobalDof, res_local);

      int nrdof_ext = PmlMat[ip]->Height();
         
      Vector res_ext(nrdof_ext); res_ext = 0.0;
      Vector sol_ext(nrdof_ext); sol_ext = 0.0;

      res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
      PmlMatInv[ip]->Mult(res_ext, sol_ext);

      sol_ext.GetSubVector(*Dof2PmlDof,sol_local);

      znew = 0.0;
      znew.SetSubVector(*Dof2GlobalDof,sol_local);

      // cout << "ip+1 = " << ip+1 << endl;
      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
      fsol[ip+1].SetSize(nDof2GlobalDof->Size());
      znew.GetSubVector(*nDof2GlobalDof,fsol[ip+1]);

      socketstream subsol_sock(vishost, visport);
      // PlotSolution(znew, subsol_sock,ip); cin.get();

      // z.AddElementVector(*Dof2GlobalDof,sol_local);
      int direction = 1;
      if (ip <nrpatch-1) GetCutOffSolution(znew, ip, direction);
      

      if (ip != 0) z1+=znew;
      // PlotSolution(z, subsol_sock,1); cin.get();

      A->Mult(znew, raux);
      rnew -= raux;
      // PlotSolution(rnew, subsol_sock,ip); cin.get();

   }

   // socketstream subsol1_sock(vishost, visport);
   // PlotSolution(z1, subsol1_sock,0); 

   rnew = r;
   for (int ip = nrpatch-1; ip >=0; ip--)
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


      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
      bsol[ip].SetSize(nDof2GlobalDof->Size());
      znew.GetSubVector(*nDof2GlobalDof,bsol[ip]);
      // cout << "ip = " << ip << endl;

      // PlotSolution(znew, subsol_sock,ip); cin.get();

      // z.AddElementVector(*Dof2GlobalDof,sol_local);


      int direction = -1;
      if (ip>0) GetCutOffSolution(znew, ip-1, direction);

      if (ip != nrpatch-1) z2+=znew;
      // PlotSolution(z, subsol_sock,1); cin.get();

      A->Mult(znew, raux);
      rnew -= raux;
      // PlotSolution(rnew, subsol_sock,ip); cin.get();

   }
   // socketstream subsol2_sock(vishost, visport);
   // PlotSolution(z2, subsol2_sock,0); cin.get();


   // construct solution z by z1 and z2
   // vizualize solutions 
   // Forward solutions
   // socketstream subsol3_sock(vishost, visport);
   for (int ip = 0; ip<nrpatch; ip++)
   {
      // cout << "ip = " << ip << endl;
      znew = 0.0;
      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
      znew.SetSubVector(*nDof2GlobalDof,fsol[ip+1]);
      // PlotSolution(znew, subsol3_sock,0); cin.get();
   }

      // Backward solutions
   // socketstream subsol4_sock(vishost, visport);
   for (int ip = 0; ip<nrpatch; ip++)
   {
      znew = 0.0;
      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
      znew.SetSubVector(*nDof2GlobalDof,bsol[ip]);
      // PlotSolution(znew, subsol4_sock,0); cin.get();
   }


   Array<Vector> gsol(nrpatch+1);
   // socketstream subsol5_sock(vishost, visport);

   for (int ip = 0; ip<=nrpatch; ip++)
   {
      if (ip == 0) 
      {
         gsol[ip].SetSize(bsol[ip].Size());
         gsol[ip] = bsol[ip];
      }
      else if (ip == nrpatch)
      {
         gsol[ip].SetSize(fsol[ip].Size());
         gsol[ip] = fsol[ip];
      }
      else
      {
         gsol[ip].SetSize(fsol[ip].Size());
         gsol[ip] = 0.0;
         gsol[ip] += bsol[ip];
         gsol[ip] += fsol[ip];
      }
      
      znew = 0.0;
      Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip];
      znew.SetSubVector(*nDof2GlobalDof,gsol[ip]);
      // PlotSolution(znew, subsol5_sock,0); cin.get();
      z.SetSubVector(*nDof2GlobalDof,gsol[ip]);
   }


   // required for visualization
   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream subsol_sock(vishost, visport);
   // socketstream subsol1_sock(vishost, visport);
   // socketstream subsol2_sock(vishost, visport);
   // socketstream subsol3_sock(vishost, visport);

   // // Initialize correction
   // z = 0.0; 
   // Vector fpml;
   // Vector zpml;
   // Vector z1(z);
   // Vector res(z);

   // // Construct the sources in each non-overlapping subdomain by restricting 
   // // the global source
   // Array<Vector> fn(nrpatch+1);
   // Array<Vector> ftransf(nrpatch+1);
   // for (int ip=0; ip<=nrpatch; ip++)
   // {
   //    Array<int> *Dof2GDof = &novlp_prob->Dof2GlobalDof[ip];
   //    fn[ip].SetSize(Dof2GDof->Size());
   //    ftransf[ip].SetSize(Dof2GDof->Size());
   //    r.GetSubVector(*Dof2GDof,fn[ip]);
   // }

   // // source transfer algorithm 1 (forward sweep)
   // Vector f;
   // for (int ip = 0; ip < nrpatch; ip++)
   // {
   //    // construct the source in the overlapping PML problem
   //    if (ip == 0) ftransf[ip] = fn[ip];

   //    int ndof = ovlp_prob->Dof2GlobalDof[ip].Size();
   //    f.SetSize(ndof); f = 0.0;
   //    f.SetSubVector(lmap->map1[ip],ftransf[ip]);
   //    f.SetSubVector(lmap->map2[ip],fn[ip+1]);

   //    // Extend to the pml problem and solve for the local pml solution
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndof_pml = PmlMat[ip]->Height();
   //    fpml.SetSize(ndof_pml); fpml=0.0;
   //    zpml.SetSize(ndof_pml); zpml=0.0;
   //    fpml.SetSubVector(*Dof2PmlDof,f);
   //    // Solve the pml problem
   //    PmlMatInv[ip]->Mult(fpml, zpml);
   //    // PlotLocalSolution(zpml,subsol_sock,ip); cin.get();

   //    //--------------------------------------------------
   //    // Save the solution to the global solution
   //     // restrict to non-pml problem
   //    Vector sol(ndof); 
   //    zpml.GetSubVector(*Dof2PmlDof, sol);
   //    // restrict to the non-ovlp subdomain
   //    // z1.AddElementVector(ovlp_prob->Dof2GlobalDof[ip],sol);

   //    int m = lmap->map2[ip].Size();
   //    Vector soll(m);
   //    sol.GetSubVector(lmap->map2[ip],soll);
   //    // prolong to the global solution
   //    z.SetSubVector(novlp_prob->Dof2GlobalDof[ip+1],soll);

   //    // PlotSolution(z,subsol1_sock,0); 
   //    // PlotSolution(z1,subsol2_sock,0); 

   //    //--------------------------------------------------

   //    if (ip == nrpatch-1) continue;

   //    int direction = 1;
   //    GetCutOffSol(zpml, ip, direction);
   //    // PlotLocalSolution(zpml,subsol3_sock,ip); cin.get();

   //    // Calculate source to be trasfered to the pml mesh
   //    Vector respml(zpml.Size());
   //    PmlMat[ip]->Mult(zpml,respml);

   //    // PlotLocalSolution(respml,subsol_sock,ip); cin.get();
   //    // restrict to non-pml problem
   //    Vector res(ndof); 
   //    respml.GetSubVector(*Dof2PmlDof, res);
   //    // source to be transfered
   //    res.GetSubVector(lmap->map2[ip],ftransf[ip+1]);

   //    // restrict to nonpml problem
   //    // restrict to non-pml problem
   //    // Vector sol1(ndof); 
   //    // zpml.GetSubVector(*Dof2PmlDof, sol1);
   //    // // prolong to global sol
   //    // z1 = 0.0;
   //    // Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //    // z1.SetSubVector(*Dof2GlobalDof, sol1);
   //    // // calculate new source
   //    // A->Mult(z1,res);

   //    // //restrict to subdomain ip+1
   //    // Array<int> * nDof2GlobalDof = &novlp_prob->Dof2GlobalDof[ip+1];
   //    // res.GetSubVector(*nDof2GlobalDof,ftransf[ip+1]);
   // }

   // // source transfer algorithm 2 (backward sweep)
   // for (int ip = nrpatch-1; ip >= 0; ip--)
   // {
   //    // construct the source in the overlapping PML problem
   //    if (ip == nrpatch-1) ftransf[ip+1] = fn[ip+1];

   //    int ndof = ovlp_prob->Dof2GlobalDof[ip].Size();
   //    f.SetSize(ndof); f = 0.0;
   //    f.SetSubVector(lmap->map1[ip],fn[ip]);
   //    f.SetSubVector(lmap->map2[ip],ftransf[ip+1]);

   //    // Extend to the pml problem and solve for the local pml solution
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndof_pml = PmlMat[ip]->Height();
   //    fpml.SetSize(ndof_pml); fpml=0.0;
   //    zpml.SetSize(ndof_pml); zpml=0.0;
   //    fpml.SetSubVector(*Dof2PmlDof,f);
   //    // Solve the pml problem
   //    PmlMatInv[ip]->Mult(fpml, zpml);
   //    PlotLocalSolution(zpml,subsol_sock,ip); cin.get();

   //    //--------------------------------------------------
   //    // Save the solution to the global solution
   //    // restrict to non-pml problem
   //    Vector sol(ndof); 
   //    zpml.GetSubVector(*Dof2PmlDof, sol);
   //    // restrict to the non-ovlp subdomain
   //    int m = lmap->map1[ip].Size();
   //    Vector soll(m);
   //    sol.GetSubVector(lmap->map1[ip],soll);
   //    // prolong to the global solution
   //    z.AddElementVector(novlp_prob->Dof2GlobalDof[ip],soll);

   //    // PlotSolution(z,subsol_sock,0); cin.get();

   //    //--------------------------------------------------

   //    if (ip == 0) continue;

   //    int direction = -1;
   //    GetCutOffSol(zpml, ip-1, direction);
   //    PlotLocalSolution(zpml,subsol_sock,ip); cin.get();

   //    // Calculate source to be trasfered to the pml mesh
   //    Vector respml(zpml.Size());
   //    PmlMat[ip]->Mult(zpml,respml);

   //    // PlotLocalSolution(respml,subsol_sock,ip); cin.get();
   //    // restrict to non-pml problem
   //    Vector res(ndof); 
   //    respml.GetSubVector(*Dof2PmlDof, res);
   //    // source to be transfered
   //    res.GetSubVector(lmap->map2[ip],ftransf[ip]);
   // }


   // PlotSolution(z,subsol_sock,0); cin.get();


   // res.SetSize(nrpatch);
   // Vector rnew(r);
   // Vector rnew2(r);
   // Vector znew(z);
   // Vector znew1(z);
   // Vector znew2(z);
   // Vector raux(znew.Size());
   // Vector res_local, sol_local;
   // znew = 0.0;
   // znew1= 0.0;
   // znew2= 0.0;

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream subsol_sock(vishost, visport);

   // std::vector<Vector*> zloc;
   // zloc.resize(nrpatch+1);

   // // allocate memory and initialize
   // for (int ip = 0; ip <= nrpatch; ip++)
   // {
   //    int n = novlp_prob->Dof2GlobalDof[ip].Size();
   //    zloc[ip] = new Vector(n); *zloc[ip]=0.0;
   // }


   // // source transfer algorithm 1 (forward sweep)
   // for (int ip = 0; ip < nrpatch; ip++)
   // {
   //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndofs = Dof2GlobalDof->Size();
   //    res_local.SetSize(ndofs);
   //    sol_local.SetSize(ndofs);

   //    rnew.GetSubVector(*Dof2GlobalDof, res_local);

   //    //-----------------------------------------------
   //    // Extend by zero to the extended mesh
   //    int nrdof_ext = PmlMat[ip]->Height();
   //    Vector res_ext(nrdof_ext); res_ext = 0.0;
   //    Vector sol_ext(nrdof_ext); sol_ext = 0.0;
   //    res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
   //    PmlMatInv[ip]->Mult(res_ext, sol_ext);
   //    sol_ext.GetSubVector(*Dof2PmlDof,sol_local);
   //    znew = 0.0;
   //    znew.SetSubVector(*Dof2GlobalDof,sol_local);

   //    Array<int> * Dof2GDof = &novlp_prob->Dof2GlobalDof[ip+1];
   //    int n = Dof2GDof->Size();
   //    Vector nsol(n);
   //    znew.GetSubVector(*Dof2GDof, nsol);
   //    *zloc[ip+1] += nsol;
      
   //    int direction = 1;
      // if (ip < nrpatch-1) GetCutOffSolution(znew, ip, direction);

   //    A->Mult(znew, raux);
   //    rnew -= raux;
   //    znew1 += znew;


   // }

   // PlotSolution(znew1, subsol_sock,0); cin.get();

   // // source transfer algorithm 2 (backward sweep)
   // for (int ip = nrpatch-1; ip >=0; ip--)
   // {
   //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //    Array<int> * Dof2PmlDof = &ovlp_prob->Dof2PmlDof[ip];
   //    int ndofs = Dof2GlobalDof->Size();
   //    res_local.SetSize(ndofs);
   //    sol_local.SetSize(ndofs);
   //    rnew2.GetSubVector(*Dof2GlobalDof, res_local);

   //    //-----------------------------------------------
   //    // Extend by zero to the extended mesh
   //    int nrdof_ext = PmlMat[ip]->Height();
   //    Vector res_ext(nrdof_ext); res_ext = 0.0;
   //    Vector sol_ext(nrdof_ext); sol_ext = 0.0;
   //    res_ext.SetSubVector(*Dof2PmlDof,res_local.GetData());
   //    PmlMatInv[ip]->Mult(res_ext, sol_ext);
   //    sol_ext.GetSubVector(*Dof2PmlDof,sol_local);
   //    znew = 0.0;
   //    znew.SetSubVector(*Dof2GlobalDof,sol_local);



   //    Array<int> * Dof2GDof = &novlp_prob->Dof2GlobalDof[ip];
   //    int n = Dof2GDof->Size();
   //    Vector nsol(n);
   //    znew.GetSubVector(*Dof2GDof, nsol);
   //    *zloc[ip] += nsol;
      
   //    int direction = -1;
   //    if (ip > 0) GetCutOffSolution(znew, ip-1, direction);

   //    A->Mult(znew, raux);
   //    rnew2 -= raux;
   //    znew2 += znew;
   // }

   // // PlotSolution(znew2, subsol_sock,0); cin.get();

   // // propagate to global dofs 
   // z = 0.0;
   // for (int ip = 0; ip <= nrpatch; ip++)
   // {
   //    Array<int> Dof2GDof = novlp_prob->Dof2GlobalDof[ip];
   //    z.AddElementVector(Dof2GDof,*zloc[ip]);
   // }

   // PlotSolution(z, subsol_sock,0); cin.get();



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


void PSTP::PlotLocalSolution(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fespace = ovlp_prob->PmlFespaces[ip];
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

void PSTP::GetCutOffSol(Vector & sol, int ip, int direction) const
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

   int m = (direction == 1) ? ip : ip+1;
   FiniteElementSpace * fespace = ovlp_prob->PmlFespaces[m];
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
}