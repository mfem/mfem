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
   ovlpnrlayers = nrlayers+1;
   part = new ParMeshPartition(pmesh,nx_,ny_,nz_,ovlpnrlayers);
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

   if (myid == 0)
   {
      cout << "\n 6. Mark subdomain overlap truedofs ..." << endl; 
   }

   MarkSubdomainOverlapDofs();
   
   if (myid == 0)
   {
      cout << "    Done ! " << endl;
   }



}

void ParDST::Mult(const Vector &r, Vector &z) const
{
   // Initialize transfered residuals to 0.0;
   // if (myid == 0) cout << "In Mult " << endl;
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      if (myid != SubdomainRank[ip]) continue;
      for (int i=0;i<sweeps->nsweeps; i++)
      {
         *f_transf_re[ip][i] = 0.0;
         *f_transf_im[ip][i] = 0.0;
      }
   }

   // restrict given residual to subdomains
   Vector r_re;
   double * data = r.GetData();
   int n = pfes->GetTrueVSize();
   r_re.SetDataAndSize(data,n);
   Vector r_im;
   r_im.SetDataAndSize(&data[n],n);


   dmaps->GlobalToSubdomains(r_re,f_orig_re);
   dmaps->GlobalToSubdomains(r_im,f_orig_im);

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
      GetChiRes(*f_orig_re[ip],ip,direct);
      GetChiRes(*f_orig_im[ip],ip,direct);
   }

   char vishost[] = "localhost";
   int  visport   = 19916;

   // for (int ip = 0; ip<nrsubdomains; ip++)
   // {
   //    if (myid == SubdomainRank[ip])
   //    {
   //       socketstream sol_sock_re(vishost, visport);
   //       PlotLocal(*f_orig_re[ip],sol_sock_re,ip);
   //       socketstream sol_sock_im(vishost, visport);
   //       PlotLocal(*f_orig_im[ip],sol_sock_im,ip);
   //    }
   // }


   z = 0.0; 
   int nsteps;
   switch(dim)
   {
      case 1: nsteps = nx; break;
      case 2: nsteps = nx+ny-1; break;
      default: nsteps = nx+ny+nz-2; break;
   }
   int nsweeps = sweeps->nsweeps;
   // for (int l=0; l<nsweeps; l++)
   // 1. Loop through sweeps
   for (int l=0; l<1; l++)
   {  
      // 2. loop through diagonals/steps of each sweep   
      for (int s = 0; s<nsteps; s++)
      {
         Array2D<int> subdomains;
         GetStepSubdomains(l,s,subdomains);
         int nsubdomains = subdomains.NumRows();

         // 3. Loop through the subdomains on the diagonal
         for (int sb=0; sb < nsubdomains; sb++)
         {
            Array<int> ijk(dim); ijk = 0;
            for (int d=0; d<dim; d++) ijk[d] = subdomains[sb][d]; 
            int ip = GetSubdomainId(nxyz,ijk);
            if (myid != SubdomainRank[ip]) continue;

            int n = dmaps->fes[ip]->GetTrueVSize();
            Vector res_re(n); res_re = 0.0;
            Vector res_im(n); res_im = 0.0;
            if (l==0) 
            {
               res_re += *f_orig_re[ip];
               res_im += *f_orig_im[ip];
            }
            res_re += *f_transf_re[ip][l];
            res_im += *f_transf_im[ip][l];

            Vector res_local(2*n);
            Vector sol_local(2*n);
            res_local.SetVector(res_re,0);
            res_local.SetVector(res_im,n);
            PmlMatInv[ip]->Mult(res_local, *subdomain_sol[ip]);
         }
         // 4. Transfer solutions to neighbors so that the subdomain
         // residuals are updated
      }
      // 5. Update the global solution 
      Array<Vector * > sol_re(nrsubdomains);
      Array<Vector * > sol_im(nrsubdomains);
      for (int ip = 0; ip<nrsubdomains; ip++)
      {
         if (myid != SubdomainRank[ip]) continue;
         int n = dmaps->fes[ip]->GetTrueVSize();
         sol_re[ip] = new Vector(n);
         sol_im[ip] = new Vector(n);
         
         sol_re[ip]->SetDataAndSize(subdomain_sol[ip]->GetData(),n);
         sol_im[ip]->SetDataAndSize(&(subdomain_sol[ip]->GetData())[n],n);
      }
      Vector z_re;
      int n = pfes->GetTrueVSize();
      z_re.SetDataAndSize(z.GetData(),n);
      Vector z_im;
      z_im.SetDataAndSize(&(z.GetData())[n],n);
      dmaps->SubdomainsToGlobal(sol_re,z_re);
      dmaps->SubdomainsToGlobal(sol_im,z_im);
   }
}

void ParDST::SetupSubdomainProblems()
{
   sqf.SetSize(nrsubdomains);
   Optr.SetSize(nrsubdomains);
   PmlMat.SetSize(nrsubdomains);
   PmlMatInv.SetSize(nrsubdomains);
   f_orig_re.SetSize(nrsubdomains);
   f_orig_im.SetSize(nrsubdomains);
   f_transf_re.SetSize(nrsubdomains);
   f_transf_im.SetSize(nrsubdomains);
   subdomain_sol.SetSize(nrsubdomains);
   for (int ip=0; ip<nrsubdomains; ip++)
   {
      subdomain_sol[ip] = nullptr;
      PmlMat[ip] = nullptr;
      PmlMatInv[ip] = nullptr;
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
      int ndofs = dmaps->fes[ip]->GetTrueVSize();
      f_transf_re[ip].SetSize(sweeps->nsweeps);
      f_transf_im[ip].SetSize(sweeps->nsweeps);
      for (int i=0;i<sweeps->nsweeps; i++)
      {
         f_transf_re[ip][i] = new Vector(ndofs);
         f_transf_im[ip][i] = new Vector(ndofs);
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


void ParDST::MarkSubdomainOverlapDofs()
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
         int k = n-m;
         NovlpDofs[ip][d].SetSize(k);
         int l = 0;
         for (int i = 0; i<n; i++)
         {
            if (marker[i]==0) 
            {
               NovlpDofs[ip][d][l] = i;  // real dofs
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

ParDST::~ParDST() {}
