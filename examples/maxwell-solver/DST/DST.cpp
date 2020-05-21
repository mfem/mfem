//Diagonal Source Transfer Preconditioner

#include "DST.hpp"


DST::DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
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

   // 1. Ovelapping partition with overlap = 2h 
   partition_kind = 2; // Non Overlapping partition 
   int nx=4;
   int ny=4; 
   int nz=1;
   ovlpnrlayers = 2*nrlayers;
   povlp = new MeshPartition(mesh, partition_kind,nx,ny,nz, ovlpnrlayers);

   partition_kind = 1;
   novlp = new MeshPartition(mesh, partition_kind,nx,ny,nz);

   nxyz[0] = povlp->nxyz[0];
   nxyz[1] = povlp->nxyz[1];
   nxyz[2] = povlp->nxyz[2];
   nrpatch = povlp->nrpatch;
   subdomains = povlp->subdomains;

   //
   // ----------------- Step 1a -------------------
   // Save the partition for visualization
   SaveMeshPartition(povlp->patch_mesh, "output/mesh_ovlp.", "output/sol_ovlp.");

   ovlp_prob  = new DofMap(bf,povlp); 
   nvlp_prob  = new DofMap(bf,novlp); 
   PmlMat.SetSize(nrpatch);
   PmlMatInv.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      PmlMat[ip] = GetPmlSystemMatrix(ip);
      PmlMatInv[ip] = new KLUSolver;
      PmlMatInv[ip]->SetOperator(*PmlMat[ip]);
   }
   nsweeps = pow(2,dim);
   sweeps.SetSize(nsweeps,dim);
   // 2D
   sweeps(0,0) =  1; sweeps(0,1) = 1;
   sweeps(1,0) = -1; sweeps(1,1) = 1;
   sweeps(2,0) =  1; sweeps(2,1) =-1;
   sweeps(3,0) = -1; sweeps(3,1) =-1;

   // Set up src arrays size
   f_orig.SetSize(nrpatch);
   f_transf.SetSize(nrpatch);
   // Construct a simple map used for directions of transfer
   for (int ip=0; ip<nrpatch; ip++)
   {
      int n = 2*ovlp_prob->fespaces[ip]->GetTrueVSize(); // (x 2 for complex ) 
      f_orig[ip] = new Vector(n); *f_orig[ip] = 0.0;
      f_transf[ip].SetSize(nsweeps);
      for (int i=0;i<nsweeps; i++)
      {
         f_transf[ip][i] = new Vector(n);
      }
   }
}



void DST::Mult(const Vector &r, Vector &z) const
{
   for (int ip=0; ip<nrpatch; ip++)
   {
      *f_orig[ip] = 0.0;
      for (int i=0;i<nsweeps; i++)
      {
         *f_transf[ip][i] = 0.0;
      }
   }
   Vector res(r.Size());
   for (int ip=0; ip<nrpatch; ip++)
   {
      Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
      r.GetSubVector(*Dof2GlobalDof,*f_orig[ip]);
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   z = 0.0; 
   Vector znew(z);
   Vector z1(z);
   Vector z2(z);

   // --------------------------------------------
   //       Sweep in the direction (1,1)
   // --------------------------------------------
   int nx = nxyz[0];
   int ny = nxyz[1];

   int nsteps = nx + ny - 1;

   for (int l=0; l<1; l++)
   {
      for (int s = 0; s<nsteps; s++)
      {
         // the patches involved are the ones such that
         // i+j = s
         // cout << "Step no: " << s << endl;
         for (int i=0;i<nx; i++)
         {
            int j;
            switch (l)
            {
               case 0:  j = s-i;         break;
               case 1:  j = s-nx+i+1;    break;
               case 2:  j = nx+i-s-1;    break;
               default: j = nx+ny-i-s-2; break;
            }
            if (j<0 || j>=ny) continue;
            // cout << "Patch no: (" << i <<"," << j << ")" << endl; 

            // find patch id
            Array<int> ij(2); ij[0] = i; ij[1]=j;
            int ip = GetPatchId(ij);
            // cout << "ip = " << ip << endl;

            // Solve the PML problem in patch ip with all sources
            // Original and all transfered (maybe some of them)
            Array<int> * Dof2GlobalDof = &ovlp_prob->Dof2GlobalDof[ip];
            int ndofs = Dof2GlobalDof->Size();

            Vector sol_local(ndofs); sol_local = 0.0;
            Vector res_local(ndofs); res_local = 0.0;
            if (l==0) res_local += *f_orig[ip];
            // res_local += *f_orig[ip];
            res_local += *f_transf[ip][l];
            // Extend by zero to the PML mesh
            // if (res_local.Norml2() < 1e-11) continue;
            PmlMatInv[ip]->Mult(res_local, sol_local);

            TransferSources(l,ip, sol_local);

            // cut off the ip solution to all possible directions
            Array<int>directions(2); directions = 0; 
            
            if (i+1<nx) directions[0] = 1;
            if (j+1<ny) directions[1] = 1;
            Vector cfsol_local;
            GetCutOffSolution(sol_local,cfsol_local,ip,directions,ovlpnrlayers,true);
            sol_local = cfsol_local;
            directions = 0.0;
            if (i>0) directions[0] = -1;
            if (j>0) directions[1] = -1;
            GetCutOffSolution(sol_local,cfsol_local,ip,directions,ovlpnrlayers,true);
            znew = 0.0;
            znew.SetSubVector(*Dof2GlobalDof, cfsol_local);
            z+=znew;


            // socketstream zsock(vishost, visport);
            // PlotSolution(cfsol_local,zsock,ip,true); cin.get();
         }
      }
   }
}


void DST::GetCutOffSolution(const Vector & sol, Vector & cfsol, 
                  int ip, Array<int> directions, int nlayers, bool local) const
{

   int d = directions.Size();
   int directx = directions[0]; // 1,0,-1
   int directy = directions[1]; // 1,0,-1
   int directz;
   if (d ==3) directz = directions[2];

   Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
   
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   double h = GetUniformMeshElementSize(povlp->patch_mesh[ip]);

   int i, j, k;
   Getijk(ip,i,j,k);
   int nx = nxyz[0];
   int ny = nxyz[1];
   if (directions[0]==1) pmax[0] -= h*nrlayers; 
   if (directions[1]==1) pmax[1] -= h*nrlayers; 

   if (directions[0]==-1) pmin[0] += h*nrlayers; 
   if (directions[1]==-1) pmin[1] += h*nrlayers; 

   Array2D<double> pmlh(dim,2); pmlh = 0.0;
   
   if (directions[0]==1)
   {
      pmlh[0][1] = h*(nlayers-nrlayers);
   }
   if (directions[0]==-1)
   {
      pmlh[0][0] = h*(nlayers-nrlayers);
   }
   if (directions[1]==1)
   {
      pmlh[1][1] = h*(nlayers-nrlayers);
   }
   if (directions[1]==-1)
   {
      pmlh[1][0] = h*(nlayers-nrlayers);
   }

   // cout << "pmin = " ; pmin.Print(); 
   // cout << "pmax = " ; pmax.Print(); 

   // cout << "nlayers = " << nlayers << endl;
   // cout << "nrlayers = " << nrlayers << endl;

   CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);

   double * data = sol.GetData();

   FiniteElementSpace * fes;
   if (!local)
   {
      fes = bf->FESpace();
   }
   else
   {
      fes = ovlp_prob->fespaces[ip];
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

   // GridFunction cgf(fes);
   // cgf.ProjectCoefficient(cf);
   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream cfsock(vishost, visport);
   // PlotSolution(cgf,cfsock,ip,true); cin.get();

}


DST::~DST()
{
}


void DST::Getijk(int ip, int & i, int & j, int & k) const
{
   k = ip/(nxyz[0]*nxyz[1]);
   j = (ip-k*nxyz[0]*nxyz[1])/nxyz[0];
   i = (ip-k*nxyz[0]*nxyz[1])%nxyz[0];
}

int DST::GetPatchId(const Array<int> & ijk) const
{
   int d=ijk.Size();
   int z = (d==2)? 0 : ijk[2];
   return subdomains(ijk[0],ijk[1],z);
}


void DST::TransferSources(int sweep, int ip0, Vector & sol0) const
{
 // Find all neighbors of patch ip0
   int nx = nxyz[0];
   int ny = nxyz[1];
   int i0, j0, k0;
   Getijk(ip0, i0,j0,k0);
   for (int i=-1; i<2; i++)
   {
      int i1 = i0 + i;
      if (i1 <0 || i1>=nx) continue;
      for (int j=-1; j<2; j++)
      {
         if (i==0 && j==0) continue;
         int j1 = j0 + j;
         if (j1 <0 || j1>=ny) continue;
         Array<int> ij1(2); ij1[0] = i1; ij1[1]=j1;
         int ip1 = GetPatchId(ij1);
         Array<int> directions(2);
         directions[0] = i;
         directions[1] = j;
         Vector cfsol0;
         GetCutOffSolution(sol0,cfsol0,ip0,directions,ovlpnrlayers,true);

         // char vishost[] = "localhost";
         // int  visport   = 19916;

         // std::cout << std::fixed;
         // std::cout << std::setprecision(10);
         // cout << "sol0 norm = " << sol0.Norml2() << endl;
         // cout << "cfsol0 norm = " << cfsol0.Norml2() << endl;
         // socketstream solsock(vishost, visport);
         // PlotSolution(cfsol0,solsock,ip0,true);  cin.get();


         // cout << "cfsol  norm = " << cfsol0.Norml2() << endl;
         // socketstream cfsock(vishost, visport);
         // PlotSolution(cfsol0,cfsock,ip0,true); cin.get();
         // Find the minumum sweep number that to transfer the source that 
         // satisfies the two rules
         for (int l=sweep; l<nsweeps; l++)
         {
            // Conditions on sweeps
            // Rule 1: the transfer source direction has to be similar with 
            // the sweep direction
            int is = sweeps(l,0); 
            int js = sweeps(l,1);
            int ddot = is*i + js * j;
            if (ddot <= 0) continue;

            // Rule 2: The horizontal or vertical transfer source cannot be used
            // in a later sweep that with opposite directions

            if (i==0 || j == 0) // Case of horizontal or vertical transfer source
            {
               int il = sweeps(l,0);
               int jl = sweeps(l,1);
               // skip if the two sweeps have opposite direction
               if (is == -il && js == -jl) continue;
            }
            Vector raux;
            int jp1 = SourceTransfer(cfsol0,directions,ip0,raux);

            MFEM_VERIFY(ip1 == jp1, "Check SourceTransfer patch id");
            MFEM_VERIFY(f_transf[ip1][l]->Size()==raux.Size(), 
                        "Transfer Sources: inconsistent size");
            *f_transf[ip1][l]+=raux;
            break;
         }
      }  
   }
}




SparseMatrix * DST::GetPmlSystemMatrix(int ip)
{
   double h = GetUniformMeshElementSize(povlp->patch_mesh[ip]);
   Array2D<double> length(dim,2);
   length = h*(nrlayers);

   CartesianPML pml(povlp->patch_mesh[ip], length);
   pml.SetOmega(omega);

   Array <int> ess_tdof_list;
   if (povlp->patch_mesh[ip]->bdr_attributes.Size())
   {
      Array<int> ess_bdr(povlp->patch_mesh[ip]->bdr_attributes.Max());
      ess_bdr = 1;
      ovlp_prob->fespaces[ip]->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
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
   SesquilinearForm a(ovlp_prob->fespaces[ip],ComplexOperator::HERMITIAN);

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
      fes = ovlp_prob->fespaces[ip];
   }
   Mesh * mesh = fes->GetMesh();
   GridFunction gf(fes);
   double * data = sol.GetData();
   gf.SetData(data);
   
   string keys;
   // if (ip == 0) 
   keys = "keys mrRljc\n";
   // sol_sock << "solution\n" << *mesh << gf << keys << "valuerange -0.1 0.1 \n"  << flush;
   sol_sock << "solution\n" << *mesh << gf << keys << flush;
}



int DST::SourceTransfer(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const
{
   int i0,j0,k0;
   Getijk(ip0,i0,j0,k0);

   int i1 = i0+direction[0];   
   int j1 = j0+direction[1];   
   Array<int> ij(2); ij[0]=i1; ij[1]=j1;
   int ip1 = GetPatchId(ij);

   MFEM_VERIFY(i1 < nxyz[0] && i1>=0, "SourceTransfer: i1 out of bounds");
   MFEM_VERIFY(j1 < nxyz[1] && j1>=0, "SourceTransfer: j1 out of bounds");

   Array<int> * Dof2GlobalDof0 = &ovlp_prob->Dof2GlobalDof[ip0];
   Array<int> * Dof2GlobalDof1 = &ovlp_prob->Dof2GlobalDof[ip1];
   Psi1.SetSize(Dof2GlobalDof1->Size()); Psi1=0.0;
   Vector r(2*bf->FESpace()->GetTrueVSize());
   r = 0.0;
   r.SetSubVector(*Dof2GlobalDof0,Psi0);
   Vector zloc(Psi1.Size()); zloc = 0.0;
   r.GetSubVector(*Dof2GlobalDof1,zloc);
   Vector Psi(Dof2GlobalDof1->Size()); Psi=0.0;
   PmlMat[ip1]->Mult(zloc,Psi);
   Psi *=-1.0;

   Array<int> direct(2); direct = 0;
   direct[0] = -direction[0];
   direct[1] = -direction[1];
   GetChiRes(Psi, Psi1,ip1,direct, ovlpnrlayers);

   // char vishost[] = "localhost";
   // int  visport   = 19916;
   // socketstream zsock(vishost, visport);
   // PlotSolution(Psi1,zsock,ip1,true); cin.get();

   return ip1;
}


void DST::GetChiRes(const Vector & res, Vector & cfres, 
                    int ip, Array<int> directions, int nlayers) const
{
   // int l,k;
   int d = directions.Size();
   int directx = directions[0]; // 1,0,-1
   int directy = directions[1]; // 1,0,-1
   int directz;
   if (d ==3) directz = directions[2];

   Mesh * mesh = ovlp_prob->fespaces[ip]->GetMesh();
   double h = GetUniformMeshElementSize(mesh);
   
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   // pmin.Print();
   // pmax.Print();
   // directions.Print();
   Array2D<double> pmlh(dim,2); pmlh = 0.0;
   int i,j,k;
   Getijk(ip,i,j,k);
   if (directions[0]==-1)
   {
      // pmlh[0][1] = h*nlayers;
      pmlh[0][0] = h;
      pmin[0] += h*(nrlayers);
   }
   if (directions[0]==1)
   {
      // pmlh[0][0] = h*nlayers;
      pmlh[0][1] = h;
      pmax[0] -= h*(nrlayers);

   }
   if (directions[1]==-1)
   {
      // pmlh[1][1] = h*nlayers;
      pmlh[1][0] = h;
      pmin[1] += h*(nrlayers);
   }
   if (directions[1]==1)
   {
      // pmlh[1][0] = h*nlayers;
      pmlh[1][1] = h;
      pmax[1] -= h*(nrlayers);
   }
   // pmin.Print();
   // pmax.Print();
   // CutOffFnCoefficient cf(ChiFncn, pmin, pmax, pmlh);
   CutOffFnCoefficient cf(CutOffFncn, pmin, pmax, pmlh);

   double * data = res.GetData();

   FiniteElementSpace * fespace;
   fespace = ovlp_prob->fespaces[ip];
   
   int n = fespace->GetTrueVSize();

   GridFunction solgf_re(fespace, data);
   GridFunction solgf_im(fespace, &data[n]);

   GridFunctionCoefficient coeff1_re(&solgf_re);
   GridFunctionCoefficient coeff1_im(&solgf_im);

   ProductCoefficient prod_re(coeff1_re, cf);
   ProductCoefficient prod_im(coeff1_im, cf);

   ComplexGridFunction gf(fespace);
   gf.ProjectCoefficient(prod_re,prod_im);

   cfres.SetSize(res.Size());
   cfres = gf;
}