
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
   f_transf.SetSize(nrsubdomains);

   for (int ip=0; ip<nrsubdomains; ip++)
   {
      SetMaxwellPmlSystemMatrix(ip);
      PmlMat[ip] = Optr[ip]->As<ComplexSparseMatrix>();
      PmlMatInv[ip] = new ComplexUMFPackSolver;
      PmlMatInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
      int ndofs = fespaces[ip]->GetTrueVSize();
      f_orig[ip] = new Vector(2*ndofs);
      f_transf[ip] = new Vector(2*ndofs);
   }
}

void ToroidST::SetMaxwellPmlSystemMatrix(int ip)
{
   
}





ToroidST::ToroidST(SesquilinearForm * bf_, const Vector & aPmlThickness_, 
       double omega_, int nrsubdomains_)
: bf(bf_), aPmlThickness(aPmlThickness_), omega(omega_), nrsubdomains(nrsubdomains_)       
{
   fes = bf->FESpace();
   cout << "In ToroidST" << endl;

   double overlap = 12.0;
   //-------------------------------------------------------
   // Step 0: Generate Mesh and FiniteElementSpace Partition
   // ------------------------------------------------------
   Array<Array<int> *> ElemMaps;
   PartitionFE(fes,nrsubdomains,overlap,fespaces, ElemMaps, 
               DofMaps0, DofMaps1, OvlpMaps0, OvlpMaps1);
   for (int i = 0; i<nrsubdomains; i++) delete ElemMaps[i];

   //-------------------------------------------------------
   // Step 1: Setup local Maxwell Problems PML
   // ------------------------------------------------------
   SetupSubdomainProblems();



   // Test local to global dof Maps
   // cout << "Testing local to global maps " << endl;
   // for (int i = 0; i<nrsubdomains; i++)
   // {
   //    DofMapTests(*fespaces[i],*fes,*DofMaps0[i], *DofMaps1[i]);
   //    // DofMapTests(*fes,*fespaces[i], *DofMaps1[i], *DofMaps0[i]);
   // }

   // Test local to neighbor dof Maps
   // cout << "Testing local to neighbor maps " << endl;
   // for (int i = 0; i<nrsubdomains-1; i++)
   // {
   //    DofMapTests(*fespaces[i],*fespaces[i+1],*OvlpMaps0[i], *OvlpMaps1[i]);
   //    DofMapTests(*fespaces[i+1],*fespaces[i],*OvlpMaps1[i], *OvlpMaps0[i]);
   // }

   // Test local to overlap dof maps
   // cout << "Testing local to overlap maps " << endl;
   // for (int i = 0; i<nrsubdomains; i++)
   // {
   //    Array<int> rdofs;
   //    RestrictDofs(*fespaces[i],1,overlap,rdofs);
   //    RestrictDofs(*fespaces[i],-1,overlap,rdofs);
   //    DofMapOvlpTest(*fespaces[i],rdofs);
   // }
}


void ToroidST::Mult(const Vector & r, Vector & z) const 
{

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