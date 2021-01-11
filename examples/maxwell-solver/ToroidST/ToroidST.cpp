
#include "ToroidST.hpp"


void ToroidST::SetupSubdomainProblems()
{
   // Sesquilinear forms and Operator
   sqf.SetSize(nrsubdomains);
   Optr.SetSize(nrsubdomains);
   // Subdomain Matrix and its LU factorization
   PmlMat.SetSize(nrsubdomains);
   PmlMatInv.SetSize(nrsubdomains);
   // Right hand sides
   f_orig.SetSize(nrsubdomains);
   forward_transf.SetSize(nrsubdomains);
   backward_transf.SetSize(nrsubdomains);

   for (int ip=0; ip<nrsubdomains; ip++)
   {
      cout << "Ip = " << ip << endl;
      SetMaxwellPmlSystemMatrix(ip);
      PmlMat[ip] = Optr[ip]->As<ComplexSparseMatrix>();
      // PmlMat[ip]->PrintMatlab(cout);
      PmlMatInv[ip] = new ComplexUMFPackSolver;
      PmlMatInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      cout << "ComplexUMFPack: size = " << PmlMat[ip]->Height() << endl;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
      int ndofs = fespaces[ip]->GetTrueVSize();
      f_orig[ip] = new Vector(2*ndofs);
      forward_transf[ip] = new Vector(2*ndofs);
      backward_transf[ip] = new Vector(2*ndofs);
   }
}

void ToroidST::SetMaxwellPmlSystemMatrix(int ip)
{
   Mesh * mesh = fespaces[ip]->GetMesh();
   // Mesh * mesh = fes->GetMesh();
   MFEM_VERIFY(mesh, "Null mesh pointer");
   int dim = mesh->Dimension();
   ToroidPML tpml(mesh);
   Vector zlim, rlim, alim;
   tpml.GetDomainBdrs(zlim,rlim,alim);
   Vector zpml(2); zpml = 0.0;
   Vector rpml(2); rpml = 0.0;
   Vector apml(2); apml = 0.0; 
   bool zstretch = false;
   bool astretch = true;
   bool rstretch = false;
   apml = aPmlThickness[1]; // just for this test (toroid waveguide)
   if (ip == 0) 
   {
      apml[0] = aPmlThickness[0];
   }
   if (ip == nrsubdomains-1)
   {
      apml[1] = aPmlThickness[1];
   }
   tpml.SetPmlAxes(zstretch,rstretch,astretch);
   tpml.SetPmlWidth(zpml,rpml,apml);
   tpml.SetOmega(omega); 



   ComplexOperator::Convention conv = bf->GetConvention();
   tpml.SetAttributes(mesh);
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 1;
   }
   fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   Array<int> attr;
   Array<int> attrPML;
   if (mesh->attributes.Size())
   {
      attr.SetSize(mesh->attributes.Max());
      attrPML.SetSize(mesh->attributes.Max());
      attr = 0;   attr[0] = 1;
      attrPML = 0;
      if (mesh->attributes.Max() > 1)
      {
         attrPML[1] = 1;
      }
   }
   ConstantCoefficient one(1.0);
   ConstantCoefficient omeg(-pow(omega, 2));
   RestrictedCoefficient restr_one(one,attr);
   RestrictedCoefficient restr_omeg(omeg,attr);

     // Integrators inside the computational domain (excluding the PML region)
   sqf[ip] = new SesquilinearForm(fespaces[ip], conv);
   // sqf[ip] = new SesquilinearForm(fes, conv);
   sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(restr_one),NULL);
   sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),NULL);

   PMLMatrixCoefficient pml_c1_Re(dim,detJ_inv_JT_J_Re, &tpml);
   PMLMatrixCoefficient pml_c1_Im(dim,detJ_inv_JT_J_Im, &tpml);
   ScalarMatrixProductCoefficient c1_Re(one,pml_c1_Re);
   ScalarMatrixProductCoefficient c1_Im(one,pml_c1_Im);
   MatrixRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
   MatrixRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

   PMLMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&tpml);
   PMLMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&tpml);
   ScalarMatrixProductCoefficient c2_Re(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im(omeg,pml_c2_Im);
   MatrixRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
   MatrixRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

   // Integrators inside the PML region
   sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                        new CurlCurlIntegrator(restr_c1_Im));
   sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                        new VectorFEMassIntegrator(restr_c2_Im));
   sqf[ip]->Assemble(0);

   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);
   // SparseMatrix * SpMat = (*Optr[ip]->As<ComplexSparseMatrix>()).GetSystemMatrix();
   // SpMat->PrintMatlab(cout);
}





ToroidST::ToroidST(SesquilinearForm * bf_, const Vector & aPmlThickness_, 
       double omega_, int nrsubdomains_)
: bf(bf_), aPmlThickness(aPmlThickness_), omega(omega_), nrsubdomains(nrsubdomains_)       
{
   fes = bf->FESpace();
   cout << "In ToroidST" << endl;

   // overlap = 2.5;
   overlap = 1.25;
   ovlp = overlap + aPmlThickness[1]; // for now
   //-------------------------------------------------------
   // Step 0: Generate Mesh and FiniteElementSpace Partition
   // ------------------------------------------------------
   Array<Array<int> *> ElemMaps;
   PartitionFE(fes,nrsubdomains,ovlp,fespaces, ElemMaps, 
               DofMaps0, DofMaps1, OvlpMaps0, OvlpMaps1);
   for (int i = 0; i<nrsubdomains; i++) delete ElemMaps[i];

   //-------------------------------------------------------
   // Step 1: Setup local Maxwell Problems PML
   // ------------------------------------------------------
   cout << "Setting up local problems " << endl;
   SetupSubdomainProblems();
   cout << "Done "<< endl;


   // Test local to global dof Maps
   // cout << "Testing local to global maps " << endl;
   // for (int i = 0; i<nrsubdomains; i++)
   // {
   //    DofMapTests(*fespaces[i],*fes,*DofMaps0[i], *DofMaps1[i]);
   //    // DofMapTests(*fes,*fespaces[i], *DofMaps1[i], *DofMaps0[i]);
   // }

   // cout << "Testing local to neighbor maps " << endl;
   // for (int i = 0; i<nrsubdomains-1; i++)
   // {
   //    DofMapTests(*fespaces[i],*fespaces[i+1],*OvlpMaps0[i], *OvlpMaps1[i]);
   //    DofMapTests(*fespaces[i+1],*fespaces[i],*OvlpMaps1[i], *OvlpMaps0[i]);
   // }

   // cout << "Testing local to overlap maps " << endl;
   // for (int i = 0; i<nrsubdomains; i++)
   // {
      // Array<int> rdofs;
      // GetRestrictionDofs(*fespaces[i],1,ovlp,rdofs);
      // GetRestrictionDofs(*fespaces[i],-1,ovlp,rdofs);
      // DofMapOvlpTest(*fespaces[i],rdofs);
   // }
}


void ToroidST::Mult(const Vector & r, Vector & z) const 
{
   cout << "ToroidST::Mult " << endl;
   cout << "r norm = " << r.Norml2() << endl;

   z = 0.0;
   // Step 0;
   // Initialize transfered residuals to 0.0 and 
   // restrict Source to subdomains
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      *forward_transf[ip] = 0.0;
      *backward_transf[ip] = 0.0;
      MapDofs(*DofMaps1[ip], *DofMaps0[ip],r,*f_orig[ip]);
      // cout << "0:f_orig[ip] norm = " << f_orig[ip]->Norml2() << endl; 
      // cout << "ovlp = " << ovlp << endl;
      int direction = 0;
      if (ip == 0) direction = 1;
      if (ip == nrsubdomains-1) direction = -1;
      if (nrsubdomains == 1) continue;
      Array<int> rdofs;
      GetRestrictionDofs(*fespaces[ip],direction,ovlp,rdofs);
      // cout << "direction = " << direction << endl;
      // DofMapOvlpTest(*fespaces[ip],rdofs);
      // cin.get();
      // rdofs.Print(cout, 20);
      RestrictDofs(rdofs,f_orig[ip]->Size()/2,*f_orig[ip]);
      // cout << "1:f_orig[ip] norm = " << f_orig[ip]->Norml2() << endl; 
      // cin.get();
   }

   // Step 1; "forward sweep"
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      int n = fespaces[ip]->GetTrueVSize();
      Vector res(2*n); res = 0.0;
      res += *f_orig[ip];
      res += *forward_transf[ip];
      Vector sol(2*n);
      PmlMatInv[ip]->Mult(res,sol);
      // accumulate for the global correction;
      // AddMapDofs(*DofMaps1[ip],*DofMaps0[ip],sol,z);
      // Transfer source to (forward) neighbor
      int sweep = 1;
      SourceTransfer(ip,sol, sweep);
      // cout << "res norm = " << res.Norml2() << endl;
      // cout << "sol norm = " << sol.Norml2() << endl;
      AddMapDofs(*DofMaps0[ip],*DofMaps1[ip],sol,z);
      // cout << "z norm = " << z.Norml2() << endl;
   }
   // Step 2: "Backward Sweep"
   for (int ip=nrsubdomains-1; ip>=0; ip--)
   {
      int n = fespaces[ip]->GetTrueVSize();
      Vector res(2*n); res = 0.0;
      res += *backward_transf[ip];
      Vector sol(2*n);
      PmlMatInv[ip]->Mult(res,sol);
      int sweep = -1;
      SourceTransfer(ip,sol,sweep);
      AddMapDofs(*DofMaps0[ip],*DofMaps1[ip],sol,z);
   }
}

void ToroidST::SourceTransfer(int ip, const Vector & sol, int sweep) const
{
   // Transfer to ip+1 and ip-1
   int ip0 = ip-1;
   int ip1 = ip+1;

   // sweep : 1  - forward
   // sweep : -1 - forward
   // direction : 0  - both

   if (ip0 >= 0)
   {  // map sol from ip to ip0
      int n = fespaces[ip0]->GetTrueVSize();
      Vector sol0(2*n); sol0 = 0.0;
      Vector Psi0(2*n);
      MapDofs(*OvlpMaps1[ip0], *OvlpMaps0[ip0],sol,sol0);
      PmlMat[ip0]->Mult(sol0,Psi0);
      int direction = 1;
      Array<int> rdofs;
      GetRestrictionDofs(*fespaces[ip0],direction,ovlp,rdofs);
      RestrictDofs(rdofs,backward_transf[ip0]->Size()/2,Psi0);
      *backward_transf[ip0]-= Psi0;
   }

   if (sweep == 1)
   {
      if (ip1 <= nrsubdomains-1)
      {
         int n = fespaces[ip1]->GetTrueVSize();
         Vector sol1(2*n); sol1 = 0.0;
         Vector Psi1(2*n);
         MapDofs(*OvlpMaps0[ip], *OvlpMaps1[ip],sol,sol1);
         PmlMat[ip1]->Mult(sol1,Psi1);
         int direction = -1;
         Array<int> rdofs;
         GetRestrictionDofs(*fespaces[ip1],direction,ovlp,rdofs);
         RestrictDofs(rdofs,forward_transf[ip1]->Size()/2,Psi1);
         *forward_transf[ip1]-= Psi1;
      } 
   }


}


ToroidST::~ToroidST()
{
   for (int i = 0; i<nrsubdomains-1; i++)
   {
      delete DofMaps0[i];
      delete DofMaps1[i];
      delete OvlpMaps0[i];
      delete OvlpMaps1[i];
   }
   delete DofMaps0[nrsubdomains-1];     
   delete DofMaps1[nrsubdomains-1];     
}