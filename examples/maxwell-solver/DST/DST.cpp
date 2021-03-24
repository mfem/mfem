//Diagonal Source Transfer Preconditioner

#include "DST.hpp"

DST::DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_ , int nx_, int ny_, int nz_)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{

   // Indentify problem ... Helmholtz or Maxwell
   int prob_kind = bf->FESpace()->FEColl()->GetContType();

   Mesh * mesh = bf->FESpace()->GetMesh();
   dim = mesh->Dimension();
   int partition_kind = 2;
   nx=nx_; ny=ny_;  nz=nz_;
   ovlpnrlayers = nrlayers+1;
   part = new MeshPartition(mesh, partition_kind,nx,ny,nz, ovlpnrlayers);
   nx = part->nxyz[0]; ny = part->nxyz[1];  nz = part->nxyz[2];
   nrpatch = part->nrpatch;

   // partition_kind = 1;
   // MeshPartition * part1 = new MeshPartition(mesh, partition_kind,nx,ny,nz);
   // SaveMeshPartition(part1->patch_mesh, "output/mesh3x3.", "output/sol3x3.");
   // SaveMeshPartition(part->patch_mesh, "output/mesh3x3.", "output/sol3x3.");
   swp = new Sweep(dim);

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();
   dmap  = new DofMap(bf->FESpace(),part);
   chrono.Stop();
   cout << "Computing subdomain to global maps: " 
        << chrono.RealTime() <<" s" << endl; 


   chrono.Clear();
   chrono.Start();
   NeighborMap = new NeighborDofMaps(part,bf->FESpace(),dmap,ovlpnrlayers); 
   chrono.Stop();
   cout << "Computing subdomain to neighbor maps: " 
        << chrono.RealTime() <<" s" << endl; 

   chrono.Clear();
   chrono.Start();
   MarkOverlapElements();
   MarkOverlapDofs();
   chrono.Stop();
   cout << "Computing subdomain overlap dofs: " 
        << chrono.RealTime() <<" s" << endl; 

   // Set up the local patch problems
   sqf.SetSize(nrpatch);
   Optr.SetSize(nrpatch);
   PmlMat.SetSize(nrpatch);
   PmlMatInv.SetSize(nrpatch);
   f_orig.SetSize(nrpatch);
   f_transf.SetSize(nrpatch);
   cout << "nrsubdomain = " << nrpatch << endl;

   chrono.Clear();
   chrono.Start();
   for (int ip=0; ip<nrpatch; ip++)
   {
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

      int ndofs = dmap->Dof2GlobalDof[ip].Size();
      f_orig[ip] = new Vector(ndofs); 
      f_transf[ip].SetSize(swp->nsweeps);
      for (int i=0;i<swp->nsweeps; i++)
      {
         f_transf[ip][i] = new Vector(ndofs);
      }
   }
   chrono.Stop();
   cout << "Computing and factoring subdomain matrices: " 
        << chrono.RealTime() <<" s" << endl; 

   zaux.SetSize(2*bf->FESpace()->GetTrueVSize());

   // char vishost[] = "localhost";
   // int visport = 19916;
   // socketstream mesh_sock1(vishost, visport);
   // mesh_sock1.precision(8);
   // mesh_sock1 << "mesh\n"
   //           << *part->patch_mesh[0] << "window_title 'Subdomain'" << flush;

}

void DST::Mult(const Vector &r, Vector &z) const
{

   for (int ip=0; ip<nrpatch; ip++)
   {
      *f_orig[ip] = 0.0;
      for (int i=0;i<swp->nsweeps; i++)
      {
         *f_transf[ip][i] = 0.0;
      }
   }
   for (int ip=0; ip<nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &dmap->Dof2GlobalDof[ip];
      r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);

      int i,j,k;
      Getijk(ip,i,j,k);
      Array<int> ijk(dim);
      ijk[0] = i;
      ijk[1] = j;
      if (dim == 3) ijk[2] = k;
      Array2D<int> direct(dim,2); direct = 0;
      for (int d=0;d<dim; d++)
      {
         if (ijk[d] > 0) direct[d][0] = 1; 
         if (ijk[d] < part->nxyz[d]-1) direct[d][1] = 1; 
      }
      GetChiRes(*f_orig[ip],ip,direct);
   }

   z = 0.0; 
   int nsteps;
   switch(dim)
   {
      case 1: nsteps = nx; break;
      case 2: nsteps = nx+ny-1; break;
      default: nsteps = nx+ny+nz-2; break;
   }
   int nsweeps = swp->nsweeps;

   for (int l=0; l<nsweeps; l++)
   {  
      for (int s = 0; s<nsteps; s++)
      {
         Array2D<int> subdomains;
         GetStepSubdomains(l,s,subdomains);

         int nsubdomains = subdomains.NumRows();
         for (int sb=0; sb< nsubdomains; sb++)
         {
            Array<int> ijk(dim);
            for (int d=0; d<dim; d++) ijk[d] = subdomains[sb][d]; 
            int ip = GetPatchId(ijk);

            Array<int> * Dof2GlobalDof = &dmap->Dof2GlobalDof[ip];
            int ndofs = Dof2GlobalDof->Size();

            Vector sol_local(ndofs); 
            Vector res_local(ndofs); res_local = 0.0;
            if (l==0) res_local += *f_orig[ip];
            res_local += *f_transf[ip][l];
            if (res_local.Norml2() < 1e-8) continue;
            PmlMatInv[ip]->Mult(res_local, sol_local);
            TransferSources(l,ip, sol_local);
            z.AddElementVector(*Dof2GlobalDof, sol_local);
         }
      }
   }
}


void DST::Getijk(int ip, int & i, int & j, int & k) const
{
   k = ip/(nx*ny);
   j = (ip-k*nx*ny)/nx;
   i = (ip-k*nx*ny)%nx;
}

int DST::GetPatchId(const Array<int> & ijk) const
{
   int d=ijk.Size();
   int z = (d==2)? 0 : ijk[2];
   return part->subdomains(ijk[0],ijk[1],z);
}


void DST::TransferSources(int s, int ip0, Vector & sol0) const
{
//   Find all neighbors of patch ip0
   int i0, j0, k0;
   Getijk(ip0, i0,j0,k0);
   Array<int> directions(dim);   
   for (int i=-1; i<2; i++)
   {
      int i1 = i0 + i;
      if (i1 <0 || i1>=nx) continue;
      directions[0] = i;
      for (int j=-1; j<2; j++)
      {
         int j1 = j0 + j;
         if (j1 <0 || j1>=ny) continue;
         directions[1] = j;
         int kbeg = (dim == 2) ? 0 : -1;
         int kend = (dim == 2) ? 1 :  2;
         for (int k=kbeg; k<kend; k++)
         {
            int k1 = k0 + k;
            if (k1 <0 || k1>=nz) continue;
            if (dim == 3) directions[2] = k;

            if (i==0 && j==0 && k==0) continue;

            int l = GetSweepToTransfer(s,directions);
            if (l == -1) continue;

            Vector raux;
            int ip1 = SourceTransfer(sol0,directions,ip0,raux);
            *f_transf[ip1][l]-=raux;
         }
      }  
   }
}

void DST::SetHelmholtzPmlSystemMatrix(int ip)
{
   Mesh * mesh = part->patch_mesh[ip];
   double h = part->MeshSize;
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   int i,j,k;
   Getijk(ip,i,j,k);
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
      dmap->fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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
   sqf[ip] = new SesquilinearForm (dmap->fespaces[ip],bf->GetConvention());

   sqf[ip]->AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   sqf[ip]->AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   sqf[ip]->Assemble();

   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);
}

void DST::SetMaxwellPmlSystemMatrix(int ip)
{
   Mesh * mesh = part->patch_mesh[ip];
   double h = part->MeshSize;
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   int i,j,k;
   Getijk(ip,i,j,k);
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
      dmap->fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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

   sqf[ip] = new SesquilinearForm(dmap->fespaces[ip],bf->GetConvention());

   sqf[ip]->AddDomainIntegrator(new CurlCurlIntegrator(pml_c1_Re),
                         new CurlCurlIntegrator(pml_c1_Im));
   sqf[ip]->AddDomainIntegrator(new VectorFEMassIntegrator(c2_Re),
                         new VectorFEMassIntegrator(c2_Im));
   sqf[ip]->Assemble();

   Optr[ip] = new OperatorPtr;
   sqf[ip]->FormSystemMatrix(ess_tdof_list,*Optr[ip]);
}

int DST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
{
   int i0,j0,k0;
   Getijk(ip0,i0,j0,k0);

   int i1 = i0+direction[0];   
   int j1 = j0+direction[1];   
   int k1;
   if (dim==3) k1 = k0+direction[2];   
   Array<int> ijk(dim); ijk[0]=i1; ijk[1]=j1; 
   if (dim == 3 ) ijk[2]=k1;
   int ip1 = GetPatchId(ijk);

   // Array<int> * Dof2GlobalDof0 = &dmap->Dof2GlobalDof[ip0];
   Array<int> * Dof2GlobalDof1 = &dmap->Dof2GlobalDof[ip1];
   // zaux.SetSubVector(*Dof2GlobalDof1,0.0);
   // zaux.SetSubVector(*Dof2GlobalDof0,Psi0);
   Psi1.SetSize(Dof2GlobalDof1->Size());
   Vector zloc(Psi1.Size()); zloc = 0.0;
   // zaux.GetSubVector(*Dof2GlobalDof1,zloc);


   Array<int> test_list0;
   Array<int> test_list1;
   Array<int>direction1(dim);

   for (int i = 0; i<dim; i++) direction1[i] = -direction[i];

   NeighborMap->GetNeighborDofMap(ip0,direction,test_list0);
   NeighborMap->GetNeighborDofMap(ip1,direction1,test_list1);

   Vector test1(zloc.Size()); test1 = 0.0;
   for (int i = 0; i<test_list0.Size(); i++)
   {
      // pick up input possition
      int j = test_list0[i];
      // destination
      int k = test_list1[i];
      zloc[k] = Psi0[j];
   }

   PmlMat[ip1]->Mult(zloc,Psi1);

   Array2D<int> direct(dim,2); direct = 0;
   for (int d = 0; d<dim; d++)
   {
      if (direction[d]==1) direct[d][0] = 1;
      if (direction[d]==-1) direct[d][1] = 1;
   }

   GetChiRes(Psi1,ip1,direct);
   return ip1;
}

void DST::GetChiRes(Vector & res, int ip, Array2D<int> direct) const
{
   for (int d=0; d<dim; d++)
   {
      // negative direction
      if (direct[d][0]==1) res.SetSubVector(NovlpDofs[ip][d],0.0);
      // possitive direction
      if (direct[d][1]==1) res.SetSubVector(NovlpDofs[ip][d+dim],0.0);
   }
}


void DST::GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const
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


int DST::GetSweepToTransfer(const int s, Array<int> directions) const
{
   int l1=-1;
   int nsweeps = swp->nsweeps;
   Array<int> sweep0;   
   swp->GetSweep(s,sweep0);


   switch (dim)
   {
      case 2:
      for (int l=s; l<nsweeps; l++)
      {
         // Rule 1: the transfer source direction has to be similar with 
         // the sweep direction
         Array<int> sweep1;   
         swp->GetSweep(l,sweep1);
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
         swp->GetSweep(l,sweep1);
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


DST::~DST()
{
   for (int ip=0; ip<nrpatch; ip++)
   {
      for (int i=0;i<swp->nsweeps; i++)
      {
         delete f_transf[ip][i];
      }
      delete f_orig[ip];
      delete PmlMatInv[ip];
      delete Optr[ip];
      delete sqf[ip];
      // delete PmlMat[ip];
   }
   delete dmap;
   delete part; 
}

void DST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip, 
                  bool localdomain) const
{
   FiniteElementSpace * fes;
   if (!localdomain)
   {
      fes = bf->FESpace();
   }
   else
   {
      fes = dmap->fespaces[ip];
   }
   Mesh * mesh = fes->GetMesh();
   GridFunction gf(fes);
   double * data = sol.GetData();
   gf.SetData(data);
   
   string keys;
   // if (ip == 0) 
   keys = "keys mrRljc\n";
   sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.05 0.05 \n"  << flush;
   // sol_sock << "solution\n" << *mesh << gf << keys << flush;
}


   void DST::MarkOverlapElements()
   {
      // cout<< "Compute Overlap Elements (in each possible direction) " << endl;
      
      // Lists of elements
      // x,y,z = +/- 1 ovlp 
      NovlpElems.resize(nrpatch);

      for (int ip = 0; ip<nrpatch; ip++)
      {
         int i,j,k;
         Getijk(ip,i,j,k);
         int ijk[dim]; ijk[0] = i; ijk[1]=j;
         if (dim==3) ijk[2] = k;
         int nxyz[dim]; nxyz[0] = nx; nxyz[1]=ny; nxyz[2]=nz;

         FiniteElementSpace * fes = dmap->fespaces[ip];
         Mesh * mesh = fes->GetMesh();
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
   }



void DST::MarkOverlapDofs()
{
   // cout<< "Compute Overlap dofs (in each possible direction) " << endl;
   NovlpDofs.resize(nrpatch);
   for (int ip = 0; ip<nrpatch; ip++)
   {
      FiniteElementSpace * fes = dmap->fespaces[ip];
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
         NovlpDofs[ip][d].SetSize(2*k);
         int l = 0;
         for (int i = 0; i<n; i++)
         {
            if (marker[i]==0) 
            {
               NovlpDofs[ip][d][l] = i;  // real dofs
               NovlpDofs[ip][d][l+k] = i+n;  // imag dofs
               l++;
            }
         }
      }
   }
}
