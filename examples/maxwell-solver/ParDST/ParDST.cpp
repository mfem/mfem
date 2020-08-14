//Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"

ParDST::ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_ , int nx_, int ny_, int nz_)
   : Solver(2*bf_->ParFESpace()->GetTrueVSize(), 2*bf_->ParFESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   pfes = bf->ParFESpace();
   fec = pfes->FEColl();
 

   comm = pfes->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   //1. Indentify problem ... Helmholtz or Maxwell
   prob_kind = fec->GetContType();
   if (myid == 0)
   {
      cout << " 1. Indentify problem to be solved ... " << endl;
      if (prob_kind == 0) cout << "    Helmholtz" << endl;
      if (prob_kind == 1) cout << "    Maxwell" << endl; 
   }

   //2. Create the parallel mesh partition
   pmesh = pfes->GetParMesh();
   dim = pmesh->Dimension();
   if (myid == 0)
   {
      cout << "\n 2. Generating ParMesh partitioning ... " << endl;
   }
   part = new ParMeshPartition(pmesh,nx_,ny_,nz_,nrlayers);
   nxyz.SetSize(3);
   nxyz[0] = nx = part->nxyz[0]; 
   nxyz[1] = ny = part->nxyz[1];  
   nxyz[2] = nz = part->nxyz[2];

   nrsubdomains = part->nrsubdomains;
   SubdomainRank = part->subdomain_rank;
   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }

   //3. Setup info for sweeps
   if (myid == 0)
   {
      cout << "\n 3. Computing sweeps info ..." << endl; 
   }
   sweeps = new Sweep(dim);

   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }

   //4. Create LocalToGlobal maps 
   //   (local GridFunctions/Vector to Global ParGridFunction/Vector) 
   if (myid == 0)
   {
      cout << "\n 4. Computing true dofs maps ..." << endl; 
   }
   dmaps = new DofMaps(pfes,part);

   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }

   // 4. Setting up the local problems 
   if (myid == 0)
   {
      cout << "\n 5. Setting up the subdomain problems ..." << endl; 
   }

   SetupSubdomainProblems();
   
   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }

}

void ParDST::Mult(const Vector &r, Vector &z) const
{
}



void ParDST::SetupSubdomainProblems()
{
   sqf.SetSize(nrsubdomains);
   Optr.SetSize(nrsubdomains);
   PmlMat.SetSize(nrsubdomains);
   PmlMatInv.SetSize(nrsubdomains);

   for (int ip=0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      // cout << "Setting up patch ip = " << ip << endl;
      if (prob_kind == 0)
      {
         SetHelmholtzPmlSystemMatrix(ip);
      }
      else if (prob_kind == 1)
      {
         SetMaxwellPmlSystemMatrix(ip);
      }
      PmlMat[ip] = Optr[ip]->As<ComplexSparseMatrix>();

      // cout << "Factorizing patch ip = " << ip << endl;

      PmlMatInv[ip] = new ComplexUMFPackSolver;
      PmlMatInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);

      // int ndofs = dmaps->Dof2GlobalDof[ip].Size();
      // f_orig[ip] = new Vector(ndofs); 
      // f_transf[ip].SetSize(swp->nsweeps);
      // for (int i=0;i<swp->nsweeps; i++)
      // {
         // f_transf[ip][i] = new Vector(ndofs);
      // }
   }


}



void ParDST::SetHelmholtzPmlSystemMatrix(int ip)
{
   MFEM_VERIFY(part->subdomain_mesh[ip], "Null mesh pointer");
   Mesh * mesh = part->subdomain_mesh[ip];
   double h = part->MeshSize;
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   Array<int> ijk;
   GetSubdomainijk(ip,nxyz,ijk);
   int i = ijk[0];
   int j = ijk[1];
   int k = ijk[2];

   if (i == 0 )    length[0][0] = Pmllength[0][0];
   if (i == nx-1 ) length[0][1] = Pmllength[0][1];
   if (dim  > 1)
   {
      if (j == 0 )    length[1][0] = Pmllength[1][0];
      if (j == ny-1 ) length[1][1] = Pmllength[1][1];
   }
   if (dim  == 3)
   {
      if (k == 0 )    length[2][0] = Pmllength[2][0];
      if (k == nz-1 ) length[2][1] = Pmllength[2][1];
   }
   
   CartesianPML pml(mesh, length);
   pml.SetOmega(omega);

   Array <int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      dmaps->fes[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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
   sqf[ip] = new SesquilinearForm (dmaps->fes[ip],bf->GetConvention());

   sqf[ip]->AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   sqf[ip]->AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   sqf[ip]->Assemble();

   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);
}

void ParDST::SetMaxwellPmlSystemMatrix(int ip)
{
   MFEM_VERIFY(part->subdomain_mesh[ip], "Null mesh pointer");
   Mesh * mesh = part->subdomain_mesh[ip];
   double h = part->MeshSize;
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   Array<int> ijk;
   GetSubdomainijk(ip,nxyz,ijk);
   int i = ijk[0];
   int j = ijk[1];
   int k = ijk[2];

   if (i == 0 )    length[0][0] = Pmllength[0][0];
   if (i == nx-1 ) length[0][1] = Pmllength[0][1];
   if (dim  > 1)
   {
      if (j == 0 )    length[1][0] = Pmllength[1][0];
      if (j == ny-1 ) length[1][1] = Pmllength[1][1];
   }
   if (dim  == 3)
   {
      if (k == 0 )    length[2][0] = Pmllength[2][0];
      if (k == nz-1 ) length[2][1] = Pmllength[2][1];
   }
   
   CartesianPML pml(mesh, length);
   pml.SetOmega(omega);
   Array <int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      dmaps->fes[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient omeg(-pow(omega, 2));
   int cdim = (dim == 2) ? 1 : dim;

   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &pml);

   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&pml);
   ScalarMatrixProductCoefficient c2_Re0(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im0(omeg,pml_c2_Im);
   ScalarMatrixProductCoefficient c2_Re(*ws,c2_Re0);
   ScalarMatrixProductCoefficient c2_Im(*ws,c2_Im0);

   sqf[ip] = new SesquilinearForm(dmaps->fes[ip],bf->GetConvention());

   sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                         new CurlCurlIntegrator(pml_c1_Im));
   sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));
   sqf[ip]->Assemble();

   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);
}

ParDST::~ParDST() {}
