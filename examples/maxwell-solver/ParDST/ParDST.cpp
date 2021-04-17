//Parallel Diagonal Source Transfer Preconditioner

#include "ParDST.hpp"

ParDST::ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, int nrlayers_ , 
         Coefficient * Qc_, Coefficient * Qm_ ,
         MatrixCoefficient * MQc_, MatrixCoefficient * MQm_,
         int nx_, int ny_, int nz_, 
         BCType bc_type_, Coefficient * LossCoeff_)
   : Solver(2*bf_->ParFESpace()->GetTrueVSize(), 2*bf_->ParFESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), 
     nrlayers(nrlayers_),
     Qc(Qc_), Qm(Qm_), MQc(MQc_), MQm(MQm_),
     bc_type(bc_type_), LossCoeff(LossCoeff_)
{
   nx = nx_; ny = ny_; nz = nz_;
   Init();
}

void ParDST::Init()
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
      cout << " 1. Identify problem to be solved ... " << endl;
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
   ovlpnrlayers = nrlayers+1;
   part = new ParMeshPartition(pmesh,nx,ny,nz,ovlpnrlayers);
   nxyz.SetSize(3);
   nxyz[0] = nx = part->nxyz[0]; 
   nxyz[1] = ny = part->nxyz[1];  
   nxyz[2] = nz = part->nxyz[2];

   nrsubdomains = part->nrsubdomains;
   SubdomainRank = part->subdomain_rank;

   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      if (myid == SubdomainRank[ip])
      {
         RankSubdomains.Append(ip);
      }
   }

   cout << " myid: " << myid 
        << ", nrsubdomains: " << RankSubdomains.Size() << endl;

   MPI_Barrier(comm);

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

   // if (myid == SubdomainRank[0])
   // {
   //    cout << "myid = " << myid << endl;
   //    char vishost[] = "localhost";
   //    int visport = 19916;
   //    socketstream mesh_sock1(vishost, visport);
   //    mesh_sock1.precision(8);
   //    mesh_sock1 << "mesh\n"
   //               << *part->subdomain_mesh[0] << "window_title 'Subdomain'" << flush;
   //    part->subdomain_mesh[0]->Print();
   
   // }
   bool comp = true;

   dmaps = new DofMaps(pfes,part, comp);
   
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
   if (myid == 0)
   {
      cout << "\n 6. Mark subdomain overlap truedofs ..." << endl; 
   }
   MarkSubdomainOverlapDofs(comp);
   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }
}

void ParDST::Mult(const Vector &r, Vector &z) const
{
   // Initialize transfered residuals to 0.0;
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      for (int i=0;i<sweeps->nsweeps; i++)
      {
         *f_transf[ip][i] = 0.0;
      }
   }

   // restrict given residual to subdomains
   dmaps->GlobalToSubdomains(r,f_orig);

   for (int ip=0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      Array<int> ijk(3);
      GetSubdomainijk(ip,nxyz,ijk);
      Array2D<int> direct(dim,2); direct = 0;
      for (int d=0;d<dim; d++)
      {
         if (ijk[d] > 0) direct[d][0] = 1; 
         if (ijk[d] < part->nxyz[d]-1) direct[d][1] = 1; 
      }
      GetChiRes(*f_orig[ip],ip,direct);
   }

   z = 0.0; 
   int nsweeps = sweeps->nsweeps;
   // 1. Loop through sweeps
   if (dim == 3 && nz == 1) { nsweeps = 4; } // x-y partition only;
   for (int l=0; l<nsweeps; l++)
   {  
      // cout << "sweep = " << l << endl;
      int nsteps = GetSweepNumSteps(l);
      // 2. loop through diagonals/steps of each sweep   
      for (int s = 0; s<nsteps; s++)
      {
         // cout << "step = " << s << endl;
         Array2D<int> subdomains;
         GetStepSubdomains(l,s,subdomains);
         // cout << "subdomains = " << endl;
         // subdomains.Print(cout, subdomains.NumCols());
         // cin.get();

         int nsubdomains = subdomains.NumRows();

         // 3. Loop through the subdomains on the diagonal
         Array<int> subdomain_ids;
         for (int sb=0; sb < nsubdomains; sb++)
         {
            Array<int> ijk(dim); ijk = 0;
            for (int d=0; d<dim; d++) ijk[d] = subdomains[sb][d]; 
            int ip = GetSubdomainId(nxyz,ijk);
            subdomain_ids.Append(ip);
            if (myid != SubdomainRank[ip]) continue;

            int n = dmaps->fes[ip]->GetTrueVSize();
            Vector res_local(2*n); res_local = 0.0;

            if (l==0)  { res_local += *f_orig[ip]; }
            res_local += *f_transf[ip][l];
            if (res_local.Norml2() < 1e-12) 
            {
               *subdomain_sol[ip] = 0.0;
               continue;
            }
            
            // char vishost[] = "localhost";
            // int visport = 19916;

            // socketstream res_sock(vishost, visport);
            // PlotLocal(res_local,res_sock,ip);

            PmlMatInv[ip]->Mult(res_local, *subdomain_sol[ip]);
            // GetSubdomainijk(ip,nxyz,ijk);
            // Array2D<int> direct(dim,2); direct = 0;
            // for (int d=0;d<dim; d++)
            // {
            //    if (ijk[d] > 0) direct[d][0] = 1; 
            //    if (ijk[d] < part->nxyz[d]-1) direct[d][1] = 1; 
            // }
            // cout << "direct = " ; direct.Print();
            // GetChiRes(*subdomain_sol[ip],ip,direct);

            // socketstream sol_sock1(vishost, visport);
            // PlotLocal(*subdomain_sol[ip],sol_sock1,ip);
            // cout << "ip = " << ip << endl;
            // cin.get();
         }
         // 4. Transfer solutions to neighbors so that the subdomain
         // residuals are updated
         TransferSources(l,subdomain_ids);
      }
      // 5. Update the global solution 
      dmaps->SubdomainsToGlobal(subdomain_sol,z);
      // char vishost[] = "localhost";
      // int visport = 19916;
      // socketstream sol_sock1(vishost, visport);
      // PlotGlobal(z,sol_sock1);
      // cin.get();
   }
   
}

void ParDST::SetupSubdomainProblems()
{
   sqf.SetSize(nrsubdomains);
   Optr.SetSize(nrsubdomains);
   PmlMat.SetSize(nrsubdomains);
   PmlMatInv.SetSize(nrsubdomains);
   f_orig.SetSize(nrsubdomains);
   f_transf.SetSize(nrsubdomains);
   subdomain_sol.SetSize(nrsubdomains);
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      sqf[ip] = nullptr;
      f_orig[ip] = nullptr;
      subdomain_sol[ip] = nullptr;
      PmlMat[ip] = nullptr;
      PmlMatInv[ip] = nullptr;
      Optr[ip] = nullptr;

      if (myid != SubdomainRank[ip]) continue;
      subdomain_sol[ip] = new Vector(2*dmaps->fes[ip]->GetTrueVSize());
      if (prob_kind == 0)
      {
         SetHelmholtzPmlSystemMatrix(ip);
      }
      else if (prob_kind == 1)
      {
         SetMaxwellPmlSystemMatrix(ip);
      }
      PmlMat[ip] = Optr[ip]->As<ComplexSparseMatrix>();

      PmlMatInv[ip] = new ComplexUMFPackSolver;
      PmlMatInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);

      // HYPRE_Int rowstarts[2]; rowstarts[0] = 0;
      // rowstarts[1] = dmaps->fes[ip]->GetTrueVSize();
      // HypreParMatrix * HypreMat_r =
      //    new HypreParMatrix(MPI_COMM_SELF,rowstarts[1],rowstarts,
      //                       &(PmlMat[ip]->real()));
      // HypreParMatrix * HypreMat_i =
      //    new HypreParMatrix(MPI_COMM_SELF,rowstarts[1],rowstarts,
      //                       &(PmlMat[ip]->imag()));
      // ComplexHypreParMatrix * HypreMat =
      //    new ComplexHypreParMatrix(HypreMat_r,HypreMat_i,true,true);
      // PmlMatInv[ip] = new ComplexMUMPSSolver;
      // PmlMatInv[ip]->SetOperator(*HypreMat);
      // delete HypreMat;
      int ndofs = dmaps->fes[ip]->GetTrueVSize();
      f_transf[ip].SetSize(sweeps->nsweeps);
      for (int i=0;i<sweeps->nsweeps; i++)
      {
         f_transf[ip][i] = new Vector(2*ndofs);
      }
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
      ess_bdr = (bc_type == BCType::DIRICHLET) ? 1 : 0;
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
   ProductCoefficient c2_re(c2_re0, *Qm);
   ProductCoefficient c2_im(c2_im0, *Qm);
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
   pml.SetAttributes(mesh);

   Array <int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = (bc_type == BCType::DIRICHLET) ? 1 : 0;
      dmaps->fes[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

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

   int cdim = (dim == 2) ? 1 : dim;

   // Integrators inside the computational domain (excluding the PML region)
   double mu = 1.0;
   double epsilon = 1.0;
   ConstantCoefficient muinv(1.0/mu);
   ConstantCoefficient omeg(-pow(omega, 2)* epsilon);

   Coefficient * amu = nullptr;
   RestrictedCoefficient * restr_amu = nullptr;
   ScalarMatrixProductCoefficient * Amu = nullptr;
   MatrixRestrictedCoefficient * restr_Amu = nullptr; 

   if (Qc)
   {
      amu = new ProductCoefficient(muinv,*Qc);
      restr_amu = new RestrictedCoefficient(*amu,attr);
   }
   else if (MQc)
   {
      Amu = new ScalarMatrixProductCoefficient(muinv,*MQc);
      restr_Amu = new MatrixRestrictedCoefficient(*Amu,attr);
   }
   else
   {
      amu = &muinv;
      restr_amu = new RestrictedCoefficient(*amu,attr);
   }

   Coefficient * ws_re = nullptr;
   // Coefficient * ws_im = nullptr;
   MatrixCoefficient * Mws_re = nullptr;
   // MatrixCoefficient * Mws_im = nullptr;
   RestrictedCoefficient * restr_wsomeg_re = nullptr;
   MatrixRestrictedCoefficient * restr_Mwsomeg_re = nullptr;
   RestrictedCoefficient * restr_loss = nullptr;
   if (Qm)
   {
      ws_re = new ProductCoefficient(*Qm,omeg);
      restr_wsomeg_re = new RestrictedCoefficient(*ws_re,attr);
   }
   else if (MQm)
   {
      Mws_re = new ScalarMatrixProductCoefficient(omeg,*MQm);
      restr_Mwsomeg_re = new MatrixRestrictedCoefficient(*Mws_re,attr);
   }
   else
   {
      ws_re = &omeg;
      restr_wsomeg_re = new RestrictedCoefficient(*ws_re,attr);
   }


   sqf[ip] = new SesquilinearForm(dmaps->fes[ip],bf->GetConvention());
   sqf[ip]->SetDiagonalPolicy(mfem::Matrix::DIAG_ONE);

   if (MQc)
   {
      sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(*restr_Amu),NULL);
   }
   else
   {
      sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(*restr_amu),NULL);
   }
   if (MQm)
   {
      sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(*restr_Mwsomeg_re),NULL);
   }
   else
   {
      sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(*restr_wsomeg_re),NULL);
   }

   if (LossCoeff) 
   {
      restr_loss = new RestrictedCoefficient(*LossCoeff,attr);
      sqf[ip]->AddDomainIntegrator(NULL, new VectorFEMassIntegrator(*LossCoeff));      
   }

   PmlMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, &pml);
   PmlMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, &pml);


   MatrixCoefficient * c1_Re = nullptr;
   MatrixCoefficient * c1_Im = nullptr;

   if (MQc)
   {
      c1_Re = new MatrixMatrixProductCoefficient(*Amu,pml_c1_Re);
      c1_Im = new MatrixMatrixProductCoefficient(*Amu,pml_c1_Im);
   }
   else
   {
      c1_Re = new ScalarMatrixProductCoefficient(*amu,pml_c1_Re);
      c1_Im = new ScalarMatrixProductCoefficient(*amu,pml_c1_Im);
   }

   MatrixRestrictedCoefficient restr_c1_Re(*c1_Re,attrPML);
   MatrixRestrictedCoefficient restr_c1_Im(*c1_Im,attrPML);

   PmlMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,&pml);
   ScalarMatrixProductCoefficient c2_Re0(omeg,pml_c2_Re);
   ScalarMatrixProductCoefficient c2_Im0(omeg,pml_c2_Im);

   MatrixCoefficient * c2_Re=nullptr;
   MatrixCoefficient * c2_Im=nullptr;

   if (Qm)
   {
      c2_Re = new ScalarMatrixProductCoefficient(*Qm,c2_Re0);
      c2_Im = new ScalarMatrixProductCoefficient(*Qm,c2_Im0);
   }
   else if (MQm)
   {
      c2_Re = new MatrixMatrixProductCoefficient(*MQm,c2_Re0);
      c2_Im = new MatrixMatrixProductCoefficient(*MQm,c2_Im0);
   }
   else
   {
      c2_Re = &c2_Re0;
      c2_Im = &c2_Im0;
   }
   
   MatrixRestrictedCoefficient restr_c2_Re(*c2_Re,attrPML);
   MatrixRestrictedCoefficient restr_c2_Im(*c2_Im,attrPML);


   sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                                new CurlCurlIntegrator(restr_c1_Im));
   sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                                new VectorFEMassIntegrator(restr_c2_Im));
   sqf[ip]->Assemble();
   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);

   if (Qc)
   {
      delete restr_amu;
      delete amu;
   }
   else if (MQc)
   {
      delete restr_Amu;
      delete Amu;
   }

   delete c1_Re;
   delete c1_Im;

   if (Qm)
   {
      delete restr_wsomeg_re;
      delete ws_re;
   }
   else if (MQm)
   {
      delete restr_Mwsomeg_re;
      delete Mws_re;
   }
   if (Qm || MQm)
   {
      delete c2_Re;
      delete c2_Im;
   }
   if (LossCoeff)  delete restr_loss;
}


void ParDST::MarkSubdomainOverlapDofs(const bool comp)
{
   // First mark the elements
         // cout<< "Compute Overlap Elements (in each possible direction) " << endl;
      // Lists of elements
      // x,y,z = +/- 1 ovlp 
   NovlpElems.resize(nrsubdomains);

   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      Array<int> ijk;
      GetSubdomainijk(ip,nxyz,ijk);

      Mesh * mesh = dmaps->fes[ip]->GetMesh();
      NovlpElems[ip].resize(2*dim);

      Vector pmin, pmax;
      mesh->GetBoundingBox(pmin,pmax);
      double h = part->MeshSize;
      // Loop through elements
      for (int iel=0; iel<mesh->GetNE(); iel++)
      {
         // Get element center
         Vector center(dim);
         int geom = mesh->GetElementBaseGeometry(iel);
         ElementTransformation * tr = mesh->GetElementTransformation(iel);
         tr->Transform(Geometries.GetCenter(geom),center);

         // Assign elements to the appropriate lists
         for (int d=0;d<dim; d++)
         {
            if (ijk[d]>0)
            {
               if (center[d] >= pmin[d]+h*ovlpnrlayers) 
               {
                  NovlpElems[ip][d].Append(iel);
               }
            }
            else
            {
               NovlpElems[ip][d].Append(iel);
            }
               
            if (ijk[d]<nxyz[d]-1)
            {
               if (center[d] <= pmax[d]-h*ovlpnrlayers) 
               {
                  NovlpElems[ip][dim+d].Append(iel);
               }
            }
            else
            {
               NovlpElems[ip][dim+d].Append(iel);
            }
         }
      }
   }

      // mark dofs
   NovlpDofs.resize(nrsubdomains);
   int mm = (comp) ? 2 : 1; // complex or real valued
   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      FiniteElementSpace * fes = dmaps->fes[ip];
      // Loop through the marked elements
      NovlpDofs[ip].resize(2*dim);
      int n = fes->GetTrueVSize();
      Array<int> marker(n);
      for (int d=0;d<2*dim; d++)
      {
         marker = 0;
         int m = 0;
         int melems = NovlpElems[ip][d].Size(); 
         for (int iel=0; iel<melems; iel++)
         {
            Array<int> ElemDofs;
            int el =  NovlpElems[ip][d][iel];
            fes->GetElementDofs(el,ElemDofs);
            int ndof = ElemDofs.Size();
            for (int i = 0; i<ndof; ++i)
            {
               int eldof = ElemDofs[i];
               int tdof = (eldof >= 0) ? eldof : abs(eldof) - 1; 
               if (marker[tdof] == 1) continue;
               marker[tdof] = 1;
               m++;
            }
         }
         int k = mm*(n-m);
         NovlpDofs[ip][d].SetSize(k);
         int l = 0;
         for (int i = 0; i<n; i++)
         {
            if (marker[i]==0) 
            {
               NovlpDofs[ip][d][l] = i;  // real dofs
               if (comp) 
               {
                  NovlpDofs[ip][d][l+k/2] = i+fes->GetTrueVSize();
               }
               l++;
            }
         }
      }
   }
}

void ParDST::GetChiRes(Vector & res, int ip, Array2D<int> direct) const
{
   for (int d=0; d<dim; d++)
   {
      // negative direction
      if (direct[d][0]==1) res.SetSubVector(NovlpDofs[ip][d],0.0);
      // possitive direction
      if (direct[d][1]==1) res.SetSubVector(NovlpDofs[ip][d+dim],0.0);
   }
}



void ParDST::PlotLocal(Vector & sol, socketstream & sol_sock, int ip) const
{
   FiniteElementSpace * fes = dmaps->fes[ip];
   Mesh * mesh = fes->GetMesh();
   GridFunction gf(fes);
   double * data = sol.GetData();
   gf.SetData(data);
   
   string keys;
   keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf << keys << flush;
}

void ParDST::PlotGlobal(Vector & sol, socketstream & sol_sock) const
{
   ParMesh * pmesh = pfes->GetParMesh();
   ParGridFunction pgf(pfes);
   double * data = sol.GetData();
   pgf.SetData(data);
   string keys;
   keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *pmesh << pgf << keys << flush;
}


double ParDST::GetSweepNumSteps(const int sweep) const
{
   int nsteps;
   switch(dim)
   {
      case 1: nsteps = nx; break;
      case 2: nsteps = nx+ny-1; break;
      default: nsteps = nx+ny+nz-2; break;
   }
   return nsteps;
}

void ParDST::GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const
{
   Array<int> aux;

   switch(dim)
   {
      case 2: 
      for (int i=nx-1;i>=0; i--)
      {  
         int j;
         switch (sweep)
         {
            case 0:  j = step-i;         break;
            case 1:  j = step-nx+i+1;    break;
            case 2:  j = nx+i-step-1;    break;
            default: j = nx+ny-i-step-2; break;
         }
         if (j<0 || j>=ny) continue;
         aux.Append(i); aux.Append(j);
      }
      break; 
      default:
      for (int i=nx-1;i>=0; i--)
      {  
         for (int j=ny-1;j>=0; j--)
         {
            int k;
            switch (sweep)
            {
               case 0:  k = step-i-j;            break;
               case 1:  k = step-nx+i+1-j;       break;
               case 2:  k = step-ny+j+1-i;       break;
               case 3:  k = step-nx-ny+i+j+2;    break;
               case 4:  k = i+j+nz-1-step;       break;
               case 5:  k = nx+nz-i+j-step-2;    break;
               case 6:  k = ny+nz+i-j-step-2;    break;
               default: k = nx+ny+nz-i-j-step-3; break;
            }
            if (k<0 || k>=nz) continue;   
            aux.Append(i); aux.Append(j); aux.Append(k); 
         }
      }
      break;
   }

   int nrows = aux.Size()/dim;
   int ncols = dim;

   subdomains.SetSize(nrows,ncols);
   for (int r=0;r<nrows; r++)
   {
      for (int c=0; c<ncols; c++)
      {
         int k = r*ncols + c;
         subdomains[r][c] = aux[k];
      }
   }
}


void ParDST::TransferSources(int sweep, const Array<int> & subdomain_ids) const
{
   OvlpSol.resize(nrsubdomains);
   int nrneighbors = pow(3,dim);
   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      if (myid == SubdomainRank[ip])
      {
         OvlpSol[ip].resize(nrneighbors);
      }
   }
   int m = subdomain_ids.Size();
   Array<Vector *> x(m);
   for (int i = 0; i<m; i++)
   {
      x[i] = nullptr;
      int ip = subdomain_ids[i];
      if (myid != SubdomainRank[ip]) continue;
      x[i] = new Vector(subdomain_sol[ip]->GetData(),subdomain_sol[ip]->Size());
   }
   dmaps->TransferToNeighbors(subdomain_ids,x,OvlpSol);
   for (int i = 0; i<m; i++)
   {
      delete x[i]; x[i] = nullptr;
   }
   // Update residuals
//   Find all neighbors of patch ip0
   for (int is = 0; is<m; is++)
   {
      int ip0 = subdomain_ids[is];
      Array<int> ijk;
      Array<int> ijk1(3);
      GetSubdomainijk(ip0,nxyz,ijk);
      // cout << "Subdomain to transfer its sources: " << "(" <<ijk[0] << "," << ijk[1] << ")" <<endl;
      Array<int> directions(3);   
      for (int i=-1; i<2; i++)
      {
         int i1 = ijk[0] + i;
         if (i1 <0 || i1>=nx) continue;
         directions[0] = i;
         ijk1[0] = i1;
         for (int j=-1; j<2; j++)
         {
            int j1 = ijk[1] + j;
            if (j1 <0 || j1>=ny) continue;
            directions[1] = j;
            ijk1[1] = j1;
            int kbeg = (dim == 2) ? 0 : -1;
            int kend = (dim == 2) ? 1 :  2;
            for (int k=kbeg; k<kend; k++)
            {
               int k1 = ijk[2] + k;
               if (k1 <0 || k1>=nz) continue;
               directions[2] = (dim == 3) ? k : -1 ;
               if (i==0 && j==0 && k==0) continue;

               int l = GetSweepToTransfer(sweep,directions);
               // cout << "in the direction " ; directions.Print(); 
               // cout << "sweep of transfer = " << l << endl;
               if (l == -1) continue;
               ijk1[2] = k1;
               int ip1 = GetSubdomainId(nxyz,ijk1);

               if (myid != SubdomainRank[ip1]) continue;
               Array<int>directions1(3); directions1 = -1;
               for (int i = 0; i<dim; i++) directions1[i] = -directions[i];
               int dir = GetDirectionId(directions1);
               int n = dmaps->fes[ip1]->GetTrueVSize();
               Vector res(2*n);
               PmlMat[ip1]->Mult(*OvlpSol[ip1][dir],res);

               Array2D<int> direct(dim,2); direct = 0;
               for (int d = 0; d<dim; d++)
               {
                  if (directions[d]==1)  direct[d][0] = 1;
                  if (directions[d]==-1) direct[d][1] = 1;
               }   
               GetChiRes(res,ip1,direct);
               *f_transf[ip1][l] -= res;
            }
         }  
      }
      // cin.get();
   }

   for (int ip = 0; ip<nrsubdomains; ip++)
   {
      if (myid == SubdomainRank[ip])
      {
         for (int i = 0; i<nrneighbors; i++)
         {
            if (OvlpSol[ip][i])
            {
               delete OvlpSol[ip][i];
            }
         }
         OvlpSol[ip].clear();
      }
   }
}

int ParDST::GetSweepToTransfer(const int s, Array<int> directions) const
{
   int l1=-1;
   int nsweeps = sweeps->nsweeps;
   Array<int> sweep0;   
   sweeps->GetSweep(s,sweep0);
   switch (dim)
   {
      case 2:
      for (int l=s; l<nsweeps; l++)
      {
         // Rule 1: the transfer source direction has to be similar with 
         // the sweep direction
         Array<int> sweep1;   
         sweeps->GetSweep(l,sweep1);
         int ddot = 0;
         for (int d=0; d<dim; d++) ddot+= sweep1[d] * directions[d];
         if (ddot <= 0) continue;

         // Rule 2: The horizontal or vertical transfer source cannot be used
         // Case of horizontal or vertical transfer source 
         // (it can't be both 0 cause it's skipped)
         if (directions[0]==0 || directions[1] == 0) 
         {
            if (sweep0[0] == -sweep1[0] && sweep0[1] == -sweep1[1]) continue;
         }
         l1 = l;
         break;
      }
      break;
      default:
      for (int l=s; l<nsweeps; l++)
      {
         // Rule 1: (similar directions) the transfer source direction has to be similar with 
         // the sweep direction
         Array<int> sweep1;   
         sweeps->GetSweep(l,sweep1);
         int ddot = 0;
         bool similar = true;
         for (int d=0; d<dim; d++) 
         {
            if (sweep1[d] * directions[d] < 0) similar = false;
            ddot+= sweep1[d] * directions[d];
         }
         if (!similar || ddot<=0) continue; // not similar

         // Rule 2: (oposite directions) the transfer source direction has to be similar with 
         // the sweep direction
         // 
         // check any of the projections onto the planes
         // (xy, xz, yz)

         if ( (directions[0]==0 && directions[1] != 0) || 
              (directions[0]!=0 && directions[1] == 0) ||
              (directions[0]==0 && directions[2] != 0) || 
              (directions[0]!=0 && directions[2] == 0) ||
              (directions[2]==0 && directions[1] != 0) || 
              (directions[2]!=0 && directions[1] == 0) ) 
         {
            if (sweep0[0] == -sweep1[0] && 
                sweep0[1] == -sweep1[1] && 
                sweep0[2] == -sweep1[2]) continue;
         }
         l1 = l;
         break;
      }
      break;
   }

   return l1;   
}

void ParDST::CorrectOrientation(int ip,Vector &x) const
{
   FiniteElementSpace * fespace = dmaps->fes[ip];
   Mesh * mesh = fespace->GetMesh();
   int nrelems = mesh->GetNE();
   // GridFunction test; 
   // test.SetFromTrueDofs(x)
   Array<int> signs(fespace->GetTrueVSize()); signs = 0;
   for (int iel=0; iel<nrelems; iel++)
   {
      Array<int> ElemDofs;
      fespace->GetElementDofs(iel,ElemDofs);
      int ndofs = ElemDofs.Size();
      ElemDofs.Print();
      for (int i = 0; i< ndofs; i++)
      {
         int pdof_ = ElemDofs[i];
         if (pdof_ < 0) 
         {
            signs[abs(pdof_)-1] += 1.0 ; 
         }
         else
         {
            signs[pdof_] -= 1.0 ; 
         }
      }
   }

   cout << "signs = " ; signs.Print();
   for (int i = 0; i<fespace->GetTrueVSize(); i++)
   {
      if (signs[i]<0) 
      {
         x(i) *= -1.0;
         x(i+fespace->GetTrueVSize()) *= -1.0;
      }
   }
}


ParDST::~ParDST()
{

   for (int ip=0; ip<nrsubdomains; ip++)
   {
      delete Optr[ip];
      delete subdomain_sol[ip];
      delete PmlMatInv[ip];
      delete sqf[ip];
      if (myid != SubdomainRank[ip]) continue;
      for (int i=0;i<sweeps->nsweeps; i++)
      {
         delete f_transf[ip][i];
      }
      delete f_orig[ip];
   }
   f_orig.DeleteAll();
   delete dmaps;
   delete sweeps;
   delete part;

}
