//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// #define DEFINITE

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void f_exact(const Vector & x, Vector & f);
void get_maxwell_solution(const Vector & x, double E[], double curl2E[]);

int dim;
double omega;
int isol = 1;


int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Initialise MPI
   int num_procs, myid;
   MPI_Init(&argc, &argv); // Initialise MPI
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //total number of processors available
   MPI_Comm_rank(MPI_COMM_WORLD, &myid); // Determine process identifier
   // 1. Parse command-line options.
   // geometry file
   // const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   // static condensation flag
   bool static_cond = false;
   // visualization flag
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of mg levels
   int maxref = 1;
   // number of initial ref
   int initref = 1;
   // PETSC
   // const char *petscrc_file = "petscrc_direct";
   const char *petscrc_file = "petscrc_mult_options";
   // optional command line inputs
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&k, "-k", "--wavelengths",
                  "Number of wavelengths.");
   args.AddOption(&maxref, "-maxref", "--maxref",
                  "Number of Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&isol, "-isol", "--exact",
                  "Exact solution flag - "
                  " 1:sinusoidal, 2: point source, 3: plane wave");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
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
   omega = 2.0*k*M_PI;
   // omega = k;

   // 2b. We initialize PETSc
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);


   // 2. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // 4. Define a finite element space on the mesh.
   FiniteElementCollection *fec   = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);


   std::vector<HypreParMatrix*>  P(maxref);

   for (int i = 0; i < maxref; i++)
   {
      const ParFiniteElementSpace cfespace(*fespace);
      pmesh->UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      Tr.Get(P[i]);
   }

   ConstantCoefficient muinv(1.0);
#ifdef DEFINITE
   ConstantCoefficient sigma(pow(omega, 2));
#else
   ConstantCoefficient sigma(-pow(omega, 2));
#endif
   // 6. Linear form (i.e RHS b = (f,v) = (1,v))
   ParLinearForm *b = new ParLinearForm(fespace);
   VectorFunctionCoefficient f(sdim, f_exact);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 7. Bilinear form a(.,.) on the finite element space
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv)); // one is the coeff
   a->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a->Assemble();

   Array<int> ess_tdof_list;
   if (pmesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParGridFunction x(fespace);
   VectorFunctionCoefficient E(sdim, E_exact);
   x.ProjectCoefficient(E);

   HypreParMatrix * A = new HypreParMatrix;
   Vector B, X;

   a->FormLinearSystem(ess_tdof_list, x, *b, *A, X, B);

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << A->GetGlobalNumRows() << " x " << A->GetGlobalNumCols() << endl;
   }

   chrono.Clear();
   chrono.Start();
   GMGSolver M(A, P);
   M.SetTheta(0.5);
   M.SetSmootherType(HypreSmoother::Jacobi);
   chrono.Stop();

   if (myid == 0)
   {
      cout << "Construction of MG precond time: " << chrono.RealTime() << endl;
   }

   int maxit(1000);
   double rtol(1.e-6);
   double atol(0.0);

   chrono.Clear();
   chrono.Start();

   X = 0.0;
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(atol);
   gmres.SetRelTol(rtol);
   gmres.SetMaxIter(maxit);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(M);
   gmres.SetPrintLevel(1);
   gmres.Mult(B, X);

   chrono.Stop();
   if (myid == 0)
   {
      cout << " GMRES time: " << chrono.RealTime() << endl;
   }


   // chrono.Clear();
   // chrono.Start();

   // MINRESSolver minres(MPI_COMM_WORLD);
   // minres.SetAbsTol(atol);
   // minres.SetRelTol(rtol);
   // minres.SetMaxIter(maxit);
   // minres.SetOperator(*A);
   // minres.SetPreconditioner(M);
   // minres.SetPrintLevel(1);
   // minres.Mult(B, X);

   // chrono.Stop();
   // if (myid == 0)
   // {
   //    cout << "MINRES time: " << chrono.RealTime() << endl;
   // }
   a->RecoverFEMSolution(X, *b, x);

   int order_quad = max(2, 2 * order + 1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i = 0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double L2Error = x.ComputeL2Error(E, irs);
   double norm_E = ComputeGlobalLpNorm(2, E, *pmesh, irs);

   if (myid == 0)
   {
      cout << "\n GMRES:  || E_h - E || / ||E|| = " << L2Error / norm_E << '\n' << endl;
   }



   // Compare with direct solve
   // X = 0.0;

   // PetscLinearSolver *invA;
   // invA = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   // invA->SetOperator(PetscParMatrix(A, Operator::PETSC_MATAIJ));
   // invA->Mult(B, X);

   // a->RecoverFEMSolution(X, *b, x);
   // double L2Error_d = x.ComputeL2Error(E);
   // double norm_E_d = ComputeGlobalLpNorm(2, E, *pmesh, irs);

   // if (myid == 0)
   // {
   //    cout << "\n Direct: || E_h - E || / ||E|| = " << L2Error_d / norm_E_d << '\n' << endl;
   // }


   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }
   // ---------------------------------------------------------------------

   for (int i = 0 ; i < maxref; i++)
   {
      delete P[i];
   }
   delete A;
   delete a;
   delete b;
   delete fec;
   delete fespace;
   delete pmesh;
   MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}


//define exact solution
void E_exact(const Vector &x, Vector &E)
{
   if (dim != 3)
   {
      cout << "2D not set up yet: " << endl;
      exit(0);
   }
   double curl2E[3];
   get_maxwell_solution(x, E, curl2E);
}

//calculate RHS from exact solution
// f = curl (mu curl E ) + omega^2*E
void f_exact(const Vector &x, Vector &f)
{
   if (dim != 3)
   {
      cout << "2D not set up yet: " << endl;
      exit(0);
   }
   double E[3], curl2E[3];

   get_maxwell_solution(x, E, curl2E);

   // curl ( curl E) + omega^2 E = f
   double coeff;
#ifdef DEFINITE
   coeff = omega * omega;
#else
   coeff = -omega * omega;
#endif

   f(0) = curl2E[0] + coeff * E[0];
   f(1) = curl2E[1] + coeff * E[1];
   f(2) = curl2E[2] + coeff * E[2];
}

void get_maxwell_solution(const Vector & x, double E[], double curl2E[])
{

   if (isol == 0) // polynomial
   {
      // Polynomial vanishing on the boundary
      E[0] = x[1] * x[2]      * (1.0 - x[1]) * (1.0 - x[2]);
      E[1] = x[0] * x[1] * x[2] * (1.0 - x[0]) * (1.0 - x[2]);
      E[2] = x[0] * x[1]      * (1.0 - x[0]) * (1.0 - x[1]);
      //
      curl2E[0] = 2.0 * x[1] * (1.0 - x[1]) - (2.0 * x[0] - 3.0) * x[2] * (1 - x[2]);
      curl2E[1] = 2.0 * x[1] * (x[0] * (1.0 - x[0]) + (1.0 - x[2]) * x[2]);
      curl2E[2] = 2.0 * x[1] * (1.0 - x[1]) + x[0] * (3.0 - 2.0 * x[2]) * (1.0 - x[0]);

   }

   else if (isol == 1) // sinusoidal
   {
      E[0] = sin(omega * x[1]);
      E[1] = sin(omega * x[2]);
      E[2] = sin(omega * x[0]);

      curl2E[0] = omega * omega * E[0];
      curl2E[1] = omega * omega * E[1];
      curl2E[2] = omega * omega * E[2];

   }
   else if (isol == 2) //point source
   {
      // shift to avoid singularity
      double x0 = x(0) + 0.1;
      double x1 = x(1) + 0.1;
      double x2 = x(2) + 0.1;
      //
      double r = sqrt(x0 * x0 + x1 * x1 + x2 * x2);

      E[0] = cos(omega * r);
      E[1] = 0.0;
      E[2] = 0.0;

      double r_x = x0 / r;
      double r_y = x1 / r;
      double r_z = x2 / r;
      double r_xy = -(r_x / r) * r_y;
      double r_xz = -(r_x / r) * r_z;
      double r_yx = r_xy;
      double r_yy = (1.0 / r) * (1.0 - r_y * r_y);
      double r_zx = r_xz;
      double r_zz = (1.0 / r) * (1.0 - r_z * r_z);

      curl2E[0] = omega * ((r_yy + r_zz) * sin(omega * r) +
                           (omega * r_y * r_y + omega * r_z * r_z) * cos(omega * r));
      curl2E[1] = -omega * (r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
      curl2E[2] = -omega * (r_zx * sin(omega * r) + omega * r_z * r_x * cos(omega * r));

   }
   else if (isol == 3) // plane wave
   {
      E[0] = cos(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
      E[1] = 0.0;
      E[2] = 0.0;

      curl2E[0] = 2.0 * omega * omega * E[0] / 3.0;
      curl2E[1] = -omega * omega * E[0] / 3.0;
      curl2E[2] = -omega * omega * E[0] / 3.0;
   }

}