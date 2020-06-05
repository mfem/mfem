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

   cout << "Constructor Done" << endl;
   cin.get();

}

void DST::Mult(const Vector &r, Vector &z) const
{
   // Init
   for (int ip=0; ip<nrpatch; ip++)
   {
      *f_orig[ip] = 0.0;
      for (int i=0;i<swp->nsweeps; i++)
      {
         *f_transf[ip][i] = 0.0;
      }
   }
   // for (int ip=0; ip<nrpatch; ip++)
   // {
   //    Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      
   //    r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);

   //    // make sure that f_ij is compactly supported in \Omega_ij (non overlapping)
   //    int i,j,k;
   //    Getijk(ip,i,j,k);
   //    Array<int> directions(2); directions = 0;
   //    int nx = nxyz[0];
   //    int ny = nxyz[1];
   //    if (i+1<nx) directions[0] = 1;
   //    if (j+1<ny) directions[1] = 1;
   //    Vector faux(f_orig[ip]->Size());
   //    GetChiRes(*f_orig[ip],faux,ip,directions,ovlpnrlayers);
   //    directions = 0.0;
   //    if (i>0) directions[0] = -1;
   //    if (j>0) directions[1] = -1;
   //    *f_orig[ip] = 0.0;
   //    GetChiRes(faux,*f_orig[ip],ip,directions,ovlpnrlayers);
   // }

   // z = 0.0; 
   // Vector znew(z);

   // // --------------------------------------------
   // //       Sweep in the direction (1,1)
   // // --------------------------------------------
   // int nx = nxyz[0];
   // int ny = nxyz[1];

   // int nsteps = nx + ny - 1;

   // for (int l=0; l<nsweeps; l++)
   // {
   //    for (int s = 0; s<nsteps; s++)
   //    {
   //       for (int i=nx-1;i>=0; i--)
   //       {
   //          int j;
   //          switch (l)
   //          {
   //             case 0:  j = s-i;         break;
   //             case 1:  j = s-nx+i+1;    break;
   //             case 2:  j = nx+i-s-1;    break;
   //             default: j = nx+ny-i-s-2; break;
   //          }
   //          if (j<0 || j>=ny) continue;

   //          Array<int> ij(2); ij[0] = i; ij[1]=j;
   //          int ip = GetPatchId(ij);

   //          // Solve the PML problem in patch ip with all sources
   //          // Original and all transfered 
   //          Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
   //          int ndofs = Dof2GlobalDof->Size();

   //          Vector sol_local(ndofs); sol_local = 0.0;
   //          Vector res_local(ndofs); res_local = 0.0;
   //          if (l==0) res_local += *f_orig[ip];
   //          res_local += *f_transf[ip][l];
   //          if (res_local.Norml2() < 1e-12) continue;
   //          PmlMatInv[ip]->Mult(res_local, sol_local);
   //          TransferSources(l,ip, sol_local);
   //          Array<int>directx(2); directx = 0; 
   //          Array<int>directy(2); directy = 0; 
   //          if (i+1<nx) directx[1] = 1;
   //          if (j+1<ny) directy[1] = 1;
   //          if (i>0) directx[0] = 1;
   //          if (j>0) directy[0] = 1;
   //          Vector cfsol_local;
   //          GetCutOffSolution(sol_local,cfsol_local,ip,directx, directy,ovlpnrlayers,true);
            
   //          znew = 0.0;
   //          znew.SetSubVector(*Dof2GlobalDof, cfsol_local);
   //          z+=znew;
   //       }
   //    }
   // }
}

// void DST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
                  // int ip, Array<int> directx, Array<int> directy, int nlayers, bool local) const
// {


   // Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
   
   // Vector pmin, pmax;
   // mesh->GetBoundingBox(pmin, pmax);
   // double h = GetUniformMeshElementSize(povlp->patch_mesh[ip]);

   // int i, j, k;
   // Getijk(ip,i,j,k);
   // if (directx[1]==1) pmax[0] -= h*nrlayers; 
   // if (directy[1]==1) pmax[1] -= h*nrlayers; 

   // if (directx[0]==1) pmin[0] += h*nrlayers; 
   // if (directy[0]==1) pmin[1] += h*nrlayers; 

   // Array2D<double> pmlh(dim,2); pmlh = 0.0;
   
   // if (directx[1]==1)
   // {
   //    pmlh[0][1] = h*(nlayers-nrlayers-1);
   // }
   // if (directx[0]==1)
   // {
   //    pmlh[0][0] = h*(nlayers-nrlayers-1);
   // }
   // if (directy[1]==1)
   // {
   //    pmlh[1][1] = h*(nlayers-nrlayers-1);
   // }
   // if (directy[0]==1)
   // {
   //    pmlh[1][0] = h*(nlayers-nrlayers-1);
   // }

   // CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);
   // double * data = sol.GetData();
   // FiniteElementSpace * fes;
   // if (!local)
   // {
   //    fes = bf->FESpace();
   // }
   // else
   // {
   //    fes = ovlp_prob->fespaces[ip];
   // }
   // int n = fes->GetTrueVSize();
   // GridFunction solgf_re(fes, data);
   // GridFunction solgf_im(fes, &data[n]);

   // GridFunctionCoefficient coeff1_re(&solgf_re);
   // GridFunctionCoefficient coeff1_im(&solgf_im);

   // ProductCoefficient prod_re(coeff1_re, cf);
   // ProductCoefficient prod_im(coeff1_im, cf);

   // ComplexGridFunction gf(fes);
   // gf.ProjectCoefficient(prod_re,prod_im);

   // cfsol.SetSize(sol.Size());
   // cfsol = gf;
// }



void DST::Getijk(int ip, int & i, int & j, int & k) const
{
   k = ip/(nx*ny);
   j = (ip-k*nx*ny)/nx;
   i = (ip-k*nx*ny)%nx;
}

// int DST::GetPatchId(const Array<int> & ijk) const
// {
//    int d=ijk.Size();
//    int z = (d==2)? 0 : ijk[2];
//    return povlp->subdomains(ijk[0],ijk[1],z);
// }


// void DST::TransferSources(int sweep, int ip0, Vector & sol0) const
// {
 // Find all neighbors of patch ip0
   // int nx = nxyz[0];
   // int ny = nxyz[1];
   // int i0, j0, k0;
   // Getijk(ip0, i0,j0,k0);
   // int is = sweeps(sweep,0);
   // int js = sweeps(sweep,1);
   // for (int i=-1; i<2; i++)
   // {
   //    int i1 = i0 + i;
   //    if (i1 <0 || i1>=nx) continue;
   //    for (int j=-1; j<2; j++)
   //    {
   //       if (i==0 && j==0) continue;

   //       int j1 = j0 + j;
   //       if (j1 <0 || j1>=ny) continue;
   //       Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
   //       int ip1 = GetPatchId(ij1);

   //       for (int l=sweep; l<nsweeps; l++)
   //       {
   //          // Conditions on sweeps
   //          // Rule 1: the transfer source direction has to be similar with 
   //          // the sweep direction
   //          int il = sweeps(l,0); 
   //          int jl = sweeps(l,1);
   //          int ddot = il*i + jl * j;
   //          if (ddot <= 0) continue;

   //          // Rule 2: The horizontal or vertical transfer source cannot be used
   //          // in a later sweep that with opposite directions

   //          if (i==0 || j == 0) // Case of horizontal or vertical transfer source
   //          {
   //             // skip if the two sweeps have opposite direction
   //             if (is == -il && js == -jl) continue;
   //          }

   //          Array<int> directions(2);
   //          directions[0] = i;
   //          directions[1] = j;
   //          // Vector cfsol0;
   //          // GetCutOffSolution(sol0,cfsol0,ip0,directions,ovlpnrlayers,true);
   //          Array<int> directx(2); directx = 0;
   //          Array<int> directy(2); directy = 0;
   //          if (i==-1) directx[0] = 1;
   //          if (i==1) directx[1] = 1;
   //          if (j==-1) directy[0] = 1;
   //          if (j==1) directy[1] = 1;
   //          Vector cfsol0;
   //          GetCutOffSolution(sol0,cfsol0,ip0,directx,directy,ovlpnrlayers,true);

   //          Vector raux;
   //          SourceTransfer(cfsol0,directions,ip0,raux);
   //          *f_transf[ip1][l]+=raux;
   //          break;
   //       }
   //    }  
   // }
// }

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
   
   CartesianPML pml(part->patch_mesh[ip], length);
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


// void DST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
// {
   // int i0,j0,k0;
   // Getijk(ip0,i0,j0,k0);

   // int i1 = i0+direction[0];   
   // int j1 = j0+direction[1];   
   // Array<int> ij(2); ij[0]=i1; ij[1]=j1;
   // int ip1 = GetPatchId(ij);

   // MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "SourceTransfer: i1 out of bounds");
   // MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "SourceTransfer: j1 out of bounds");

   // Array<int> * Dof2GlobalDof0 = &ovlp_prob->Dof2GlobalDof[ip0];
   // Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
   // Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
   // Vector r(2*bf->FESpace()->GetTrueVSize());
   // r = 0.0;
   // r.SetSubVector(*Dof2GlobalDof0,Psi0);
   // Vector zloc(Psi1.Size()); zloc = 0.0;
   // r.GetSubVector(*Dof2GlobalDof1,zloc);

   // Vector Psi(Dof2GlobalDof1->Size()); Psi=0.0;

   // PmlMat[ip1]->Mult(zloc,Psi);
   // Psi *=-1.0;

   // Array<int> direct(2); direct = 0;
   // direct[0] = -direction[0];
   // direct[1] = -direction[1];
   // GetChiRes(Psi, Psi1,ip1,direct, ovlpnrlayers);

// }

// void DST::GetChiRes(const Vector & res, Vector & cfres, 
                  //   int ip, Array<int> directions, int nlayers) const
// {

   // Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
   // double h = GetUniformMeshElementSize(mesh);
   
   // Vector pmin, pmax;
   // mesh->GetBoundingBox(pmin, pmax);
   // Array2D<double> pmlh(dim,2); pmlh = 0.0;
   // int i,j,k;
   // Getijk(ip,i,j,k);
   // if (directions[0]==-1)
   // {
   //    pmlh[0][0] = h;
   //    pmin[0] += h*(nlayers-1);
   // }
   // if (directions[0]==1)
   // {
   //    pmlh[0][1] = h;
   //    pmax[0] -= h*(nlayers-1);

   // }
   // if (directions[1]==-1)
   // {
   //    pmlh[1][0] = h;
   //    pmin[1] += h*(nlayers-1);
   // }
   // if (directions[1]==1)
   // {
   //    pmlh[1][1] = h;
   //    pmax[1] -= h*(nlayers-1);
   // }
   // CutOffFnCoefficient cf(ChiFncn, pmin, pmax, pmlh);

   // double * data = res.GetData();

   // FiniteElementSpace * fespace;
   // fespace = ovlp_prob->fespaces[ip];
   
   // int n = fespace->GetTrueVSize();

   // GridFunction solgf_re(fespace, data);
   // GridFunction solgf_im(fespace, &data[n]);

   // GridFunctionCoefficient coeff1_re(&solgf_re);
   // GridFunctionCoefficient coeff1_im(&solgf_im);

   // ProductCoefficient prod_re(coeff1_re, cf);
   // ProductCoefficient prod_im(coeff1_im, cf);

   // ComplexGridFunction gf(fespace);
   // gf.ProjectCoefficient(prod_re,prod_im);

   // cfres.SetSize(res.Size());
   // cfres = gf;
// }




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