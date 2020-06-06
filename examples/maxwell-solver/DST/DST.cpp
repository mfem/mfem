//Diagonal Source Transfer Preconditioner

#include "DST.hpp"

Sweep::Sweep(int dim_) : dim(dim_)
{
   nsweeps = pow(2,dim);
   directions.SetSize(2);
   directions[0] = 1;
   directions[1] = -1;
   sweeps.resize(nsweeps);

   for (int is = 0; is<nsweeps; is++)
   {
      sweeps[is].SetSize(dim);
   }

   switch(dim)
   {
      case 1: 
         sweeps[0][0] =  1; 
         sweeps[1][0] = -1; 
         break;
      case 2: 
         sweeps[0][0] =  1; sweeps[0][1] =  1; 
         sweeps[1][0] = -1; sweeps[1][1] =  1; 
         sweeps[2][0] =  1; sweeps[2][1] = -1; 
         sweeps[3][0] = -1; sweeps[3][1] = -1;
         break;
      default:
         sweeps[0][0] =  1; sweeps[0][1] =  1; sweeps[0][2] =  1; 
         sweeps[1][0] = -1; sweeps[1][1] =  1; sweeps[1][2] =  1; 
         sweeps[2][0] =  1; sweeps[2][1] = -1; sweeps[2][2] =  1; 
         sweeps[3][0] = -1; sweeps[3][1] = -1; sweeps[3][2] =  1;
         sweeps[4][0] =  1; sweeps[4][1] =  1; sweeps[4][2] = -1; 
         sweeps[5][0] = -1; sweeps[5][1] =  1; sweeps[5][2] = -1; 
         sweeps[6][0] =  1; sweeps[6][1] = -1; sweeps[6][2] = -1; 
         sweeps[7][0] = -1; sweeps[7][1] = -1; sweeps[7][2] = -1;
         break;
   }
}



DST::DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
         double omega_, Coefficient * ws_,  int nrlayers_)
   : Solver(2*bf_->FESpace()->GetTrueVSize(), 2*bf_->FESpace()->GetTrueVSize()), 
     bf(bf_), Pmllength(Pmllength_), omega(omega_), ws(ws_), nrlayers(nrlayers_)
{
   Mesh * mesh = bf->FESpace()->GetMesh();
   dim = mesh->Dimension();

   int partition_kind = 2;

   nx=3;
   ny=3; 
   nz=1;
   ovlpnrlayers = nrlayers+2;
   part = new MeshPartition(mesh, partition_kind,nx,ny,nz, ovlpnrlayers);
   nx = part->nxyz[0];
   ny = part->nxyz[1];
   nz = part->nxyz[2];

   nrpatch = part->nrpatch;

   // Sweeps info
   swp = new Sweep(dim);

   dmap  = new DofMap(bf->FESpace(),part); 

   // Set up the local patch problems
   PmlMat.SetSize(nrpatch);
   PmlMatInv.SetSize(nrpatch);
   f_orig.SetSize(nrpatch);
   f_transf.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      PmlMat[ip] = GetPmlSystemMatrix(ip);
      PmlMatInv[ip] = new KLUSolver;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);

      int ndofs = PmlMat[ip]->Height();
      f_orig[ip] = new Vector(ndofs); 
      f_transf[ip].SetSize(swp->nsweeps);
      for (int i=0;i<swp->nsweeps; i++)
      {
         f_transf[ip][i] = new Vector(ndofs);
      }
   }

   cout << "DST: Constructor Done" << endl;
}

void DST::Mult(const Vector &r, Vector &z) const
{
   // Init
   cout << "DST: in mult " << endl;
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
      Array2D<int> direct(dim,2); direct = 0;
      if (i>0) direct[0][0] = 1;
      if (i+1<nx) direct[0][1] = 1;
      if (j>0) direct[1][0] = 1;
      if (j+1<ny) direct[1][1] = 1;

      Vector faux(*f_orig[ip]);
      *f_orig[ip] = 0.0;
      GetChiRes(faux,*f_orig[ip],ip,direct,ovlpnrlayers);
   }

   z = 0.0; 
   Vector znew(z);


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

            // cout << "ip = " << ip << endl;
            Array<int> * Dof2GlobalDof = &dmap->Dof2GlobalDof[ip];
            int ndofs = Dof2GlobalDof->Size();

            Vector sol_local(ndofs); sol_local = 0.0;
            Vector res_local(ndofs); res_local = 0.0;
            if (l==0) res_local += *f_orig[ip];
            res_local += *f_transf[ip][l];
            if (res_local.Norml2() < 1e-12) continue;
            PmlMatInv[ip]->Mult(res_local, sol_local);
            TransferSources(l,ip, sol_local);
            Array2D<int> direct(dim,2); direct = 0;
            for (int d=0;d<dim; d++)
            {
               if (ijk[d] > 0) direct[d][0] = 1; 
               if (ijk[d] < part->nxyz[d]-1) direct[d][1] = 1; 
            }

            Vector cfsol_local;
            GetCutOffSolution(sol_local,cfsol_local,ip,direct,ovlpnrlayers,true);
            znew = 0.0;
            znew.SetSubVector(*Dof2GlobalDof, cfsol_local);
            z+=znew;
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
  // Find all neighbors of patch ip0
   int i0, j0, k0;
   Getijk(ip0, i0,j0,k0);
   Array<int> directions(dim);   
   for (int i=-1; i<2; i++)
   {
      int i1 = i0 + i;
      directions[0] = i;
      if (i1 <0 || i1>=nx) continue;
      for (int j=-1; j<2; j++)
      {
         if (i==0 && j==0) continue;

         int j1 = j0 + j;
         if (j1 <0 || j1>=ny) continue;
         Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
         int ip1 = GetPatchId(ij1);

         directions[1] = j;

         int l = GetSweepToTransfer(s,directions);

         if (l == -1) continue;
         Array2D<int> direct(dim,2); direct = 0;
         for (int d=0; d<dim; d++)
         {
            if (directions[d] == -1) direct[d][0] = 1;
            if (directions[d] ==  1) direct[d][1] = 1;
         }

         Vector cfsol0;
         GetCutOffSolution(sol0,cfsol0,ip0,direct,ovlpnrlayers,true);

         Vector raux;
         SourceTransfer(cfsol0,directions,ip0,raux);
         *f_transf[ip1][l]+=raux;
      }  
   }
}

SparseMatrix * DST::GetPmlSystemMatrix(int ip)
{
   Mesh * mesh = part->patch_mesh[ip];
   double h = GetUniformMeshElementSize(mesh);
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
   SesquilinearForm a(dmap->fespaces[ip],ComplexOperator::HERMITIAN);

   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();

   OperatorPtr Alocal;
   a.FormSystemMatrix(ess_tdof_list,Alocal);
   ComplexSparseMatrix * AZ_ext = Alocal.As<ComplexSparseMatrix>();
   SparseMatrix * Mat = AZ_ext->GetSystemMatrix();
   Mat->Threshold(1e-13);
   return Mat;
}


void DST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
{
   int i0,j0,k0;
   Getijk(ip0,i0,j0,k0);

   int i1 = i0+direction[0];   
   int j1 = j0+direction[1];   
   Array<int> ij(2); ij[0]=i1; ij[1]=j1;
   int ip1 = GetPatchId(ij);

   MFEM_VERIFY(i1 < nx && i1>=0, "SourceTransfer: i1 out of bounds");
   MFEM_VERIFY(j1 < ny && j1>=0, "SourceTransfer: j1 out of bounds");

   Array<int> * Dof2GlobalDof0 = &dmap->Dof2GlobalDof[ip0];
   Array<int> * Dof2GlobalDof1 = &dmap->Dof2GlobalDof[ip1];
   Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
   Vector r(2*bf->FESpace()->GetTrueVSize());
   r = 0.0;
   r.SetSubVector(*Dof2GlobalDof0,Psi0);
   Vector zloc(Psi1.Size()); zloc = 0.0;
   r.GetSubVector(*Dof2GlobalDof1,zloc);
   Vector Psi(Dof2GlobalDof1->Size()); Psi=0.0;
   PmlMat[ip1]->Mult(zloc,Psi);
   Psi *=-1.0;

   Array2D<int> direct(dim,2); direct = 0;
   if (direction[0]==1) direct[0][0] = 1;
   if (direction[0]==-1) direct[0][1] = 1;
   if (direction[1]==1) direct[1][0] = 1;
   if (direction[1]==-1) direct[1][1] = 1;
   GetChiRes(Psi, Psi1,ip1,direct, ovlpnrlayers);
}


void DST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
                  int ip, Array2D<int> direct, int nlayers, bool local) const
{

   Mesh * mesh = dmap->fespaces[ip]->GetMesh();
   
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(mesh);


   int i, j, k;
   Getijk(ip,i,j,k);

   Array2D<double> pmlh(dim,2); pmlh = 0.0;
   for (int i=0; i<dim; i++)
   {
      if (direct[i][0]==1) pmin[i] += h*nrlayers; 
      if (direct[i][1]==1) pmax[i] -= h*nrlayers; 
      for (int j=0; j<2; j++)
      {
         if (direct[i][j]==1)
         {
            pmlh[i][j] = h*(nlayers-nrlayers-1);
         }
      }  
   }

   CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);
   double * data = sol.GetData();
   FiniteElementSpace * fes;
   if (!local)
   {
      fes = bf->FESpace();
   }
   else
   {
      fes = dmap->fespaces[ip];
   }
   int n = fes->GetTrueVSize();
   GridFunction solgf_re(fes, data);
   GridFunction solgf_im(fes, &data[n]);

   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fes);
   gf.ProjectCoefficient(prod_re,prod_im);

   cfsol.SetSize(sol.Size());
   cfsol = gf;
}








void DST::GetChiRes(const Vector & res, Vector & cfres, 
                    int ip, Array2D<int> direct, int nlayers) const
{

   FiniteElementSpace * fes = dmap->fespaces[ip];
   Mesh * mesh = fes->GetMesh();
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(mesh);
   
   Array2D<double> pmlh(dim,2); pmlh = 0.0;

   for (int i=0; i<dim; i++)
   {
      if (direct[i][0]==1) pmin[i] += h*(nlayers-1); 
      if (direct[i][1]==1) pmax[i] -= h*(nlayers-1); 
      for (int j=0; j<2; j++)
      {
         if (direct[i][j]==1)
         {
            pmlh[i][j] = h;
         }
      }  
   }

   CutOffFnCoefficient cf(ChiFncn, pmin, pmax, pmlh);

   double * data = res.GetData();
   
   int n = fes->GetTrueVSize();

   GridFunction solgf_re(fes, data);
   GridFunction solgf_im(fes, &data[n]);

   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fes);
   gf.ProjectCoefficient(prod_re,prod_im);

   cfres.SetSize(res.Size());
   cfres = gf;
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
      MFEM_ABORT("dimension not supported yet" );
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
   for (int l=s; l<nsweeps; l++)
   {
      // Conditions on sweeps
      // Rule 1: the transfer source direction has to be similar with 
      // the sweep direction
      Array<int> sweep1;   
      swp->GetSweep(l,sweep1);
      int ddot = 0;
      for (int d=0; d<dim; d++) ddot+= sweep1[d] * directions[d];
      if (ddot <= 0) continue;

      // Rule 2: The horizontal or vertical transfer source cannot be used
      if (directions[0]==0 || directions[1] == 0) // Case of horizontal or vertical transfer source
      {
         if (sweep0[0] == -sweep1[0] && sweep0[1] == -sweep1[1]) continue;
      }
      l1 = l;
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
      delete PmlMat[ip];
   }
   delete dmap;
   delete part; 
}

// void DST::PlotSolution(Vector & sol, socketstream & sol_sock, int ip, 
//                   bool localdomain) const
// {
//    FiniteElementSpace * fes;
//    if (!localdomain)
//    {
//       fes = bf->FESpace();
//    }
//    else
//    {
//       fes = ovlp_prob->fespaces[ip];
//    }
//    Mesh * mesh = fes->GetMesh();
//    GridFunction gf(fes);
//    double * data = sol.GetData();
//    gf.SetData(data);
   
//    string keys;
//    // if (ip == 0) 
//    keys = "keys mrRljc\n";
//    // sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
//    sol_sock << "solution\n" << *mesh << gf << keys << flush;
// }