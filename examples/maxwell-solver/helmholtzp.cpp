//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 p = 1 with impedance boundary condition.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ParDST/ParDST.hpp"
#include "common/PML.hpp"

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double f_exact_Re(const Vector &x);
double f_exact_Im(const Vector &x);

double wavespeed(const Vector &x);

double funccoeff_re(const Vector & x);
double funccoeff_im(const Vector & x);


int dim;
double omega;
int sol = 1;
double length = 1.0;
double pml_length = 0.25;
Array2D<double>comp_bdr;

// #ifndef MFEM_USE_SUPERLU
// #error This example requires that MFEM is built with MFEM_USE_PETSC=YES
// #endif

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // 2. Parse command-line options.
   // geometry file
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   int bc_type = 1;
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of serial refinements
   int ser_ref_levels = 1;
   // number of parallel refinements
   int par_ref_levels = 2;
   // dimension
   int nd = 2;
   int nx=2;
   int ny=2;
   int nz=2;
   bool herm_conv = true;

   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&nd, "-nd", "--dim","Problem space dimension");
   args.AddOption(&nx, "-nx", "--nx","Number of subdomains in x direction");
   args.AddOption(&ny, "-ny", "--ny","Number of subdomains in y direction");
   args.AddOption(&nz, "-nz", "--nz","Number of subdomains in z direction");
   args.AddOption(&sol, "-sol", "--exact",
                  "Exact solution flag - 0:polynomial, 1: plane wave, -1: unknown exact");
   args.AddOption(&bc_type, "-bct", "--bc-type",
                  "BC type - 0:Neumann, 1: Dirichlet");                  
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&pml_length, "-pml_length", "--pml_length",
                  "Length of the PML region in each direction");
   args.AddOption(&length, "-length", "--length",
                  "length of the domain in each direction.");
   args.AddOption(&ser_ref_levels, "-sr", "--ser_ref_levels",
                  "Number of Serial Refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par_ref_levels",
                  "Number of Parallel Refinements.");                  
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");                  
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   // check if the inputs are correct
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   // Angular frequency
   omega = 2.0 * M_PI * k;

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh;

   if (nd == 2)
   {
      // mesh = new Mesh(mesh_file,1,1);
      mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, length, length, false);
   }
   else
   {
      mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length,false);
   }

   // 3. Executing uniform h-refinement
   dim = mesh->Dimension();
   for (int i = 0; i < ser_ref_levels; i++ )
   {
      mesh->UniformRefinement();
   }



   // 4. Define a parallel mesh by a partitioning of the serial mesh.
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   int nprocs;
   int nprocsx;
   int nprocsy;
   int nprocsz;
   if (dim == 2)
   {
      nprocs = sqrt(num_procs); 
      // nprocsx = nprocs;
      // nprocsy = nprocs;
      nprocsx = 1;
      nprocsy = num_procs;
      nprocsz = 1;
   }    
   else
   {
      nprocs = cbrt(num_procs); 
      // nprocsx = nprocs;
      // nprocsy = nprocs;
      // nprocsz = nprocs;
      nprocsx = 1;
      if (nz != 1)
      {
         nprocsy = sqrt(num_procs);
         nprocsz = nprocsy;
      }
      else
      {
         nprocsy = num_procs;
         nprocsz = 1;
      }
   }
   // MFEM_VERIFY(nprocs*nprocs == num_procs, "Check MPI partitioning");
   // int nxyz[3] = {num_procs,1,1};
   // int nxyz[3] = {nprocs,nprocs,1};
   // int nxyz[3] = {1,num_procs,1};
   
   int nxyz[3] = {nprocsx,nprocsy,nprocsz};
   // int nxyz[3] = {num_procs,1,1};
   int * part = mesh->CartesianPartitioning(nxyz);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh,part);
   // ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD,*mesh);
   delete [] part;
   delete mesh;

   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }


   double hl = GetUniformMeshElementSize(pmesh);
   int nrlayers = 3;

   Array2D<double> lengths(dim,2);
   lengths = hl*nrlayers;
   // lengths[0][1] = 0.0;
   // lengths[1][1] = 0.0;
   // lengths[1][0] = 0.0;
   // lengths[0][0] = 0.0;
   // CartesianPML pml(mesh,lengths);
   CartesianPML pml(pmesh,lengths);
   pml.SetOmega(omega);
   comp_bdr.SetSize(dim,2);
   comp_bdr = pml.GetCompDomainBdr(); 

   int basis = BasisType::GetType('G');

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim,basis);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }
   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient f_Im(f_exact_Im);

   // 8. Setup Complex Operator convention
   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

   // ParLinearForm *b_Re(new ParLinearForm);
   ParComplexLinearForm b(fespace, conv);
   b.AddDomainIntegrator(new DomainLFIntegrator(f_Re),
                         new DomainLFIntegrator(f_Im));
   b.real().Vector::operator=(0.0);
   b.imag().Vector::operator=(0.0);
   b.Assemble();

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   FunctionCoefficient ws(wavespeed);

   PmlMatrixCoefficient c1_re(dim,pml_detJ_JT_J_inv_Re,&pml);
   PmlMatrixCoefficient c1_im(dim,pml_detJ_JT_J_inv_Im,&pml);

   PmlCoefficient detJ_re(pml_detJ_Re,&pml);
   PmlCoefficient detJ_im(pml_detJ_Im,&pml);

   ProductCoefficient c2_re0(sigma, detJ_re);
   ProductCoefficient c2_im0(sigma, detJ_im);

   ProductCoefficient c2_re(c2_re0, ws);
   ProductCoefficient c2_im(c2_im0, ws);

   ParSesquilinearForm a(fespace,conv);
   a.AddDomainIntegrator(new DiffusionIntegrator(c1_re),
                         new DiffusionIntegrator(c1_im));
   a.AddDomainIntegrator(new MassIntegrator(c2_re),
                         new MassIntegrator(c2_im));
   a.Assemble();
   a.Finalize();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = bc_type;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Solution grid function
   ParComplexGridFunction p_gf(fespace); p_gf = 0.0;
   OperatorHandle Ah;
   Vector X, B;

   a.FormLinearSystem(ess_tdof_list, p_gf, b, Ah, X, B);

   // // lor preconditioner
   // ParMesh *pmesh_lor = NULL;
   // FiniteElementCollection *fec_lor = NULL;
   // ParFiniteElementSpace *fespace_lor = NULL;
   // int basis_lor = basis;
   // cout << order << endl;
   // pmesh_lor = new ParMesh(pmesh, order, basis_lor);
   
   // CartesianPML pml_lor(pmesh_lor,lengths);
   // pml_lor.SetOmega(omega);
   
   
   // fec_lor = new H1_FECollection(1, dim);
   // fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);
   // ParSesquilinearForm a_lor(fespace_lor,conv);

   // PmlMatrixCoefficient c1_re_lor(dim,pml_detJ_JT_J_inv_Re,&pml_lor);
   // PmlMatrixCoefficient c1_im_lor(dim,pml_detJ_JT_J_inv_Im,&pml_lor);

   // PmlCoefficient detJ_re_lor(pml_detJ_Re,&pml_lor);
   // PmlCoefficient detJ_im_lor(pml_detJ_Im,&pml_lor);

   // ProductCoefficient c2_re0_lor(sigma, detJ_re_lor);
   // ProductCoefficient c2_im0_lor(sigma, detJ_im_lor);

   // ProductCoefficient c2_re_lor(c2_re0_lor, ws);
   // ProductCoefficient c2_im_lor(c2_im0_lor, ws);


   // a_lor.AddDomainIntegrator(new DiffusionIntegrator(c1_re_lor),
                           //   new DiffusionIntegrator(c1_im_lor));
   // a_lor.AddDomainIntegrator(new MassIntegrator(c2_re_lor),
                           //   new MassIntegrator(c2_im_lor));
   // a_lor.Assemble();
   // a_lor.Finalize();

   // Solution grid function
   // OperatorHandle Ah_lor;
   // a_lor.FormSystemMatrix(ess_tdof_list, Ah_lor);

   // ComplexMUMPSSolver prec;

   // StopWatch chrono;
   // chrono.Clear();
   // chrono.Start();
   // prec.SetOperator(*Ah.As<ComplexHypreParMatrix>());
   // prec.SetOperator(*Ah_lor.As<ComplexHypreParMatrix>());
   // chrono.Stop();
   // cout << " myid: " << myid 
      //   << ", lor time: " << chrono.RealTime() << endl;

   {
      StopWatch chrono;
      chrono.Clear();
      chrono.Start();
      ParDST::BCType bct = (bc_type == 1)? ParDST::BCType::DIRICHLET : ParDST::BCType::NEUMANN;
      // ParDST S(&a,lengths,omega, &ws,nrlayers,nx,ny,nz, bct);

      ParDST S(&a,lengths, omega, nrlayers,&one, &ws,nullptr,nullptr, nx, ny, nz, bct);


      // ParDST Slor(&a_lor,lengths,omega, &ws,nrlayers,nx,ny,nz);
      chrono.Stop();
      double t1 = chrono.RealTime();

      chrono.Clear();
      chrono.Start();
      // X = 0.0;
      GMRESSolver gmres(MPI_COMM_WORLD);
      // gmres.SetPreconditioner(Slor);
      gmres.SetOperator(*Ah);
      gmres.SetPreconditioner(S);
      // gmres.SetPreconditioner(prec);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(200);
      gmres.SetPrintLevel(1);
      gmres.Mult(B, X);
      chrono.Stop();

      double t2 = chrono.RealTime();
      
      MPI_Barrier(MPI_COMM_WORLD);


      cout << " myid: " << myid 
           << ", setup time: " << t1
           << ", solution time: " << t2 << endl; 


      // chrono.Clear();
      // chrono.Start();
      // X = 0.0;
      // SLISolver sli(MPI_COMM_WORLD);
      // sli.iterative_mode = true;
      // sli.SetPreconditioner(S);
      // sli.SetOperator(*Ah);
      // sli.SetRelTol(1e-6);
      // sli.SetMaxIter(20);
      // sli.SetPrintLevel(1);
      // sli.Mult(B,X);


      // chrono.Stop();
      // double t3 = chrono.RealTime();

      // cout << " myid: " << myid 
      //      << ", SLI solution time: " << t3 << endl; 

         //    cout << " myid: " << myid 
         //   << ", setup time: " << t1
         //   << ", SLI solution time: " << t3 << endl; 
         //   << ", solution time: " << t2 << endl; 

      a.RecoverFEMSolution(X,B,p_gf);
      if (visualization)
      {
         char vishost[] = "localhost";
         int  visport   = 19916;
         string keys;
         if (dim ==2 )
         {
            keys = "keys mrRljc\n";
         }
         else
         {
            keys = "keys mc\n";
         }
         // socketstream mesh_sock(vishost, visport);
         // mesh_sock.precision(8);
         // mesh_sock << "parallel " << num_procs << " " << myid << "\n"
         //             << "mesh\n" << *pmesh  << flush;
         socketstream sol_sock_re(vishost, visport);
         sol_sock_re.precision(8);
         sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << p_gf.real() << keys 
                     << "window_title 'Numerical Pressure: Real Part' " << flush;                     

         socketstream sol_sock_im(vishost, visport);
         sol_sock_im.precision(8);
         sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << p_gf.imag() << keys 
                     << "window_title 'Numerical Pressure: Imag Part' " << flush;                     
      }
   }




//    // solve
// {
//    HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
//    SuperLURowLocMatrix SA(*A);
//    SuperLUSolver superlu(MPI_COMM_WORLD);
//    superlu.SetPrintStatistics(false);
//    superlu.SetSymmetricPattern(false);
//    superlu.SetColumnPermutation(superlu::PARMETIS);
//    superlu.SetOperator(SA);
//    superlu.Mult(B, X);
//    delete A;
// }
//    a.RecoverFEMSolution(X,B,p_gf);

//    if (visualization)
//    {
//       char vishost[] = "localhost";
//       int  visport   = 19916;
//       string keys;
//       if (dim ==2 )
//       {
//          keys = "keys mrRljc\n";
//       }
//       else
//       {
//          keys = "keys mc\n";
//       }
//       socketstream sol_sock_re(vishost, visport);
//       sol_sock_re.precision(8);
//       sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
//                   << "solution\n" << *pmesh << p_gf.real() <<
//                   "window_title 'Numerical Pressure' "
//                   // << keys << "valuerange -0.08 0.08 \n" << flush;
//                   << keys << flush;
//    }
   delete fespace;
   delete fec;
	delete pmesh;
   MPI_Finalize();
   return 0;
}


double f_exact_Re(const Vector &x)
{
   
   // int nrsources = (dim == 2) ? 4 : 8;
   int nrsources = 1;
   Vector x0(nrsources);
   Vector y0(nrsources);
   Vector z0(nrsources);
   x0(0) = 0.25; y0(0) = 0.25; z0(0) = 0.25;
   // x0(1) = 0.75; y0(1) = 0.25; z0(1) = 0.25;
   // x0(2) = 0.25; y0(2) = 0.75; z0(2) = 0.25;
   // x0(3) = 0.75; y0(3) = 0.75; z0(3) = 0.25;
   if (dim == 3)
   {
      // x0(4) = 0.25; y0(4) = 0.25; z0(4) = 0.75;
      // x0(5) = 0.75; y0(5) = 0.25; z0(5) = 0.75;
      // x0(6) = 0.25; y0(6) = 0.75; z0(6) = 0.75;
      // x0(7) = 0.75; y0(7) = 0.75; z0(7) = 0.75;
   }
  
   double n = 4.0*omega/M_PI;
   double coeff = 16.0*omega*omega/M_PI/M_PI/M_PI;

   double f_re = 0.0;
   // for (int i = 0; i<1; i++)
   for (int i = 0; i<nrsources; i++)
   {
      double beta = pow(x0(i)-x(0),2) + pow(y0(i)-x(1),2);
      if (dim == 3) { beta += pow(z0(i)-x(2),2); }
      double alpha = -pow(n,2) * beta;
      f_re += coeff*exp(alpha);
   }

   bool in_pml = false;
   for (int i = 0; i<dim; i++)
   {
      if (x(i)<=comp_bdr(i,0) || x(i)>=comp_bdr(i,1))
      {
         in_pml = true;
         break;
      }
   }
   if (in_pml) f_re = 0.0;

   return f_re;

}
double f_exact_Im(const Vector &x)
{
   double f_im;
   f_im = 0.0;
   return f_im;
}

double wavespeed(const Vector &x)
{
   double ws;
   ws = 1.0;
   return ws;
}


double funccoeff_re(const Vector & x)
{
   return sin(3*M_PI*(x.Sum()));
}

double funccoeff_im(const Vector & x)
{
   return cos(10*M_PI*(x.Sum()));
}























