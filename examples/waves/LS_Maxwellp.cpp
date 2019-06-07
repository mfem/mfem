//                                MFEM Example multigrid-grid Cycle
//
// Compile with: make mg_maxwellp
//
// Sample runs:  mg_maxwellp -m ../data/one-hex.mesh

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ddoper.hpp"
#include "AMS_LS.hpp"

using namespace std;
using namespace mfem;

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void H_exact(const Vector & x, Vector & H);
void scaledf_exact_E(const Vector & x, Vector & f_E);
void scaledf_exact_H(const Vector & x, Vector & f_H);
void f_exact_E(const Vector & x, Vector & f_E);
void f_exact_H(const Vector & x, Vector & f_H);
void get_maxwell_solution(const Vector & x, double E[], double curlE[], double curl2E[]);

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
   double k = 1.0;
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
   args.AddOption(&maxref, "-ref", "--maxref",
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
   // omega = 2.0*k*M_PI;
   omega = k;

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
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

   std::vector<HypreParMatrix*>  P(maxref);

   for (int i = 0; i < maxref; i++)
   {
      const ParFiniteElementSpace cfespace(*fespace);
      pmesh->UniformRefinement();
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(cfespace, Tr);
      HypreParMatrix * Paux;
      Tr.Get(Paux);
      P[i] = new HypreParMatrix(*Paux);
   }


   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   Array<int> block_offsets(3);
   block_offsets[0] = 0;
   block_offsets[1] = fespace->GetVSize();
   block_offsets[2] = fespace->GetVSize();
   block_offsets.PartialSum();

   Array<int> block_trueOffsets(3);
   block_trueOffsets[0] = 0;
   block_trueOffsets[1] = fespace->TrueVSize();
   block_trueOffsets[2] = fespace->TrueVSize();
   block_trueOffsets.PartialSum();

   //    _           _    _  _       _  _
   //   |             |  |    |     |    |
   //   |  A00   A01  |  | E  |     |F_E |
   //   |             |  |    |  =  |    |
   //   |  A10   A11  |  | H  |     |F_G |
   //   |_           _|  |_  _|     |_  _|
   //
   // A00 = (curl E, curl F) + \omega^2 (E,F)
   // A01 = - \omega *( (curl E, F) + (E,curl F)
   // A10 = - \omega *( (curl H, G) + (H,curl G)
   // A11 = (curl H, curl H) + \omega^2 (H,G)

   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

   x = 0.0;
   rhs = 0.0;
   trueX = 0.0;
   trueRhs = 0.0;

   VectorFunctionCoefficient Eex(sdim, E_exact);
   ParGridFunction * E_gf = new ParGridFunction;
   E_gf->MakeRef(fespace, x.GetBlock(0));
   E_gf->ProjectCoefficient(Eex);

   VectorFunctionCoefficient Hex(sdim, H_exact);
   ParGridFunction * H_gf = new ParGridFunction;   
   H_gf->MakeRef(fespace, x.GetBlock(1));
   H_gf->ProjectCoefficient(Hex);

   // // 6. Set up the linear form
   VectorFunctionCoefficient sf_E(sdim,scaledf_exact_E);
   VectorFunctionCoefficient sf_H(sdim,scaledf_exact_H);
   VectorFunctionCoefficient f_E(sdim,f_exact_H);
   VectorFunctionCoefficient f_H(sdim,f_exact_H);

   ParLinearForm *b_E = new ParLinearForm;
   b_E->Update(fespace, rhs.GetBlock(0), 0);
   b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_H));
   b_E->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_E));
   b_E->Assemble();


   ParLinearForm *b_H = new ParLinearForm;
   b_H->Update(fespace, rhs.GetBlock(1), 0);
   b_H->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_E));
   b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
   b_H->Assemble();



   // 7. Bilinear form a(.,.) on the finite element space
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(pow(omega, 2));
   ConstantCoefficient neg(-abs(omega));
   ConstantCoefficient pos(abs(omega));
   //
   ParBilinearForm *a_EE = new ParBilinearForm(fespace);
   a_EE->AddDomainIntegrator(new CurlCurlIntegrator(one)); 
   a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_EE->Assemble();
   a_EE->EliminateEssentialBC(ess_bdr,x.GetBlock(0), rhs.GetBlock(0));
   a_EE->Finalize();
   HypreParMatrix *A_EE = a_EE->ParallelAssemble();

   ParMixedBilinearForm *a_EH = new ParMixedBilinearForm(fespace,fespace);
   a_EH->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
   a_EH->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(neg)); 
   a_EH->Assemble();
   a_EH->EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
   a_EH->Finalize();
   HypreParMatrix *A_EH = a_EH->ParallelAssemble();

   HypreParMatrix *A_HE = A_EH->Transpose();

   ParBilinearForm *a_HH = new ParBilinearForm(fespace);
   a_HH->AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
   a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_HH->Assemble();
   a_HH->Finalize();
   HypreParMatrix *A_HH = a_HH->ParallelAssemble();

   BlockOperator *LS_Maxwellop = new BlockOperator(block_trueOffsets);
   LS_Maxwellop->SetBlock(0, 0, A_EE);
   LS_Maxwellop->SetBlock(0, 1, A_HE);
   LS_Maxwellop->SetBlock(1, 0, A_EH);
   LS_Maxwellop->SetBlock(1, 1, A_HH);
   

   HypreSolver *amsE = new HypreAMS(*A_EE,fespace);
   HypreSolver *amsH = new HypreAMS(*A_HH,fespace);

   GMGSolver * gmgE = new GMGSolver(A_EE, P);
   gmgE->SetTheta(0.5);
   gmgE->SetSmootherType(HypreSmoother::Jacobi);

   GMGSolver * gmgH = new GMGSolver(A_HH, P);
   gmgH->SetTheta(0.5);
   gmgH->SetSmootherType(HypreSmoother::Jacobi);


   PetscLinearSolver * invE = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invE->SetOperator(PetscParMatrix(A_EE, Operator::PETSC_MATAIJ));

   PetscLinearSolver * invH = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invH->SetOperator(PetscParMatrix(A_HH, Operator::PETSC_MATAIJ));

   BlockDiagonalPreconditioner *LSprec1 = new BlockDiagonalPreconditioner(block_trueOffsets);
   BlockDiagonalPreconditioner *LSprec2 = new BlockDiagonalPreconditioner(block_trueOffsets);
   BlockDiagonalPreconditioner *LSprec3 = new BlockDiagonalPreconditioner(block_trueOffsets);
   LSprec1->SetDiagonalBlock(0, amsE);
   LSprec1->SetDiagonalBlock(1, amsH);

   LSprec2->SetDiagonalBlock(0, gmgE);
   LSprec2->SetDiagonalBlock(1, gmgH);

   LSprec3->SetDiagonalBlock(0, invE);
   LSprec3->SetDiagonalBlock(1, invH);

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << 2.0 * A_EE->GetGlobalNumRows() << " x " << 2.0* A_EE->GetGlobalNumCols() << endl;
   }
   int maxit(5000);
   double rtol(1.e-6);
   double atol(0.0);
   trueX = 0.0;

   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetAbsTol(atol);
   pcg.SetRelTol(rtol);
   pcg.SetMaxIter(maxit);
   pcg.SetOperator(*LS_Maxwellop);
   pcg.SetPreconditioner(*LSprec1);
   pcg.SetPrintLevel(1);
   pcg.Mult(trueRhs, trueX);

   if (myid == 0)
   {
      cout << "PCG with AMS finished" << endl;
   }
   trueX = 0.0;

   pcg.SetPreconditioner(*LSprec2);
   pcg.Mult(trueRhs, trueX);

   if (myid == 0)
   {
      cout << "PCG with GMG finished" << endl;
   }

   trueX = 0.0;

   pcg.SetPreconditioner(*LSprec3);
   pcg.Mult(trueRhs, trueX);

   if (myid == 0)
   {
      cout << "PCG with Petsc exact inverse finished" << endl;
   }


   *E_gf = 0.0;
   *H_gf = 0.0;

   E_gf->Distribute(&(trueX.GetBlock(0)));
   H_gf->Distribute(&(trueX.GetBlock(1)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double Error_E = E_gf->ComputeL2Error(Eex, irs);
   double norm_E = ComputeGlobalLpNorm(2, Eex, *pmesh, irs);

   double Error_H = H_gf->ComputeL2Error(Hex, irs);
   double norm_H = ComputeGlobalLpNorm(2, Hex , *pmesh, irs);

   if (myid == 0)
   {
      // cout << "|| E_h - E || / || E || = " << Error_E / norm_E << "\n";
      // cout << "|| H_h - H || / || H || = " << Error_H / norm_H << "\n";
      cout << "|| E_h - E || = " << Error_E  << "\n";
      cout << "|| H_h - H || = " << Error_H  << "\n";

      cout << "Total Relative error = " <<  Error_E / norm_E + Error_H / norm_H  << "\n";

      // cout << "|| E || = " << norm_E  << "\n";
      // cout << "|| H || = " << norm_H  << "\n";
   }

   if (visualization)
   {
      // ParGridFunction * Eex_gf = new ParGridFunction;   
      // Eex_gf->MakeRef(fespace, x.GetBlock(0));
      // Eex_gf->ProjectCoefficient(Eex);
      // ParGridFunction * Hex_gf = new ParGridFunction;   
      // Hex_gf->MakeRef(fespace, x.GetBlock(1));
      // Hex_gf->ProjectCoefficient(Hex);

      // 8. Connect to GLVis.
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock << "parallel " << num_procs << " " << myid << "\n";
      E_sock.precision(8);
      E_sock << "solution\n" << *pmesh << *E_gf << "window_title 'Electric field'" << endl;
      // MPI_Barrier(pmesh->GetComm());
      // socketstream Eex_sock(vishost, visport);
      // Eex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Eex_sock.precision(8);
      // Eex_sock << "solution\n" << *pmesh << *Eex_gf << "window_title 'Exact Electric Field'" << endl;
      MPI_Barrier(pmesh->GetComm());
      socketstream H_sock(vishost, visport);
      H_sock << "parallel " << num_procs << " " << myid << "\n";
      H_sock.precision(8);
      H_sock << "solution\n" << *pmesh << *H_gf << "window_title 'Magnetic field'" << endl;
      // MPI_Barrier(pmesh->GetComm());
      // socketstream Hex_sock(vishost, visport);
      // Hex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Hex_sock.precision(8);
      // Hex_sock << "solution\n" << *pmesh << *Hex_gf << "window_title 'Exact Magnetic field'" << endl;
   }


   delete a_EE;
   delete a_EH;
   delete a_HH;
   delete b_E;
   delete b_H;
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
   double curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
}

void H_exact(const Vector &x, Vector &H)
{
   double E[3], curlE[3], curl2E[3];
   get_maxwell_solution(x, E, curlE, curl2E);
   for (int i = 0; i<3; i++) {H(i) = curlE[i]/omega;}
}

//calculate RHS from exact solution
void f_exact_E(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl E - omega H = 0 
   f(0) = curlE[0] - omega * (curlE[0]/omega); // = 0
   f(1) = curlE[1] - omega * (curlE[1]/omega); // = 0
   f(2) = curlE[2] - omega * (curlE[2]/omega); // = 0
}

void f_exact_H(const Vector &x, Vector &f)
{
   if (dim != 3)
   {
      cout << "2D not set up yet: " << endl;
      exit(0);
   }
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f 
   // = curl (curl E / omega) - omega E 
   f(0) = curl2E[0]/omega - omega * E[0];
   f(1) = curl2E[1]/omega - omega * E[1];
   f(2) = curl2E[2]/omega - omega * E[2];
}

void scaledf_exact_E(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   //  - omega *( curl E - omega H) = 0 
   f(0) =-omega * (curlE[0] - omega * (curlE[0]/omega)); // = 0
   f(1) =-omega * (curlE[1] - omega * (curlE[1]/omega)); // = 0
   f(2) =-omega * (curlE[2] - omega * (curlE[2]/omega)); // = 0
}

void scaledf_exact_H(const Vector &x, Vector &f)
{
   double E[3], curlE[3], curl2E[3];

   get_maxwell_solution(x, E, curlE, curl2E);

   // curl H - omega E = f 
   // = - omega *( curl (curl E / omega) - omega E) 
   
   f(0) = -omega * (curl2E[0]/omega - omega * E[0]);
   f(1) = -omega * (curl2E[1]/omega - omega * E[1]);
   f(2) = -omega * (curl2E[2]/omega - omega * E[2]);
}

void get_maxwell_solution(const Vector & X, double E[], double curlE[], double curl2E[])
{
   const double x = X[0];
   const double y = X[1];
   const double z = X[2];

   if (isol == 0) // polynomial
   {
      // Polynomial vanishing on the boundary
      E[0] = y * z * (1.0 - y) * (1.0 - z);
      E[1] = (1.0 - x) * x * y * (1.0 - z) * z;
      E[2] = (1.0 - x) * x * (1.0 - y) * y;
      //

      curlE[0] = -(-1.0 + x) * x * (1.0 + y * (-3.0 + 2.0 * z));
      curlE[1] = -2.0 * (-1.0 + y) * y * (x - z);
      curlE[2] = (1.0 + (-3.0 + 2.0 * x) * y) * (-1.0 + z) * z; 

      curl2E[0] = -2.0 * (-1.0 + y) * y + (-3.0 + 2.0 * x) * (-1.0 + z) * z;
      curl2E[1] = -2.0 * y * (-x + x*x + (-1.0 + z) * z);
      curl2E[2] = -2.0 * (-1.0 + y) * y + (-1.0 + x) * x * (-3.0 + 2.0 * z);

   }

   else if (isol == 1) // sinusoidal
   {
      E[0] = sin(omega * y);
      E[1] = sin(omega * z);
      E[2] = sin(omega * x);

      curlE[0] = -omega * cos(omega * z);
      curlE[1] = -omega * cos(omega * x);;
      curlE[2] = -omega * cos(omega * y);; 

      curl2E[0] = omega * omega * E[0];
      curl2E[1] = omega * omega * E[1];
      curl2E[2] = omega * omega * E[2];

   }
   else if (isol == 2) //simple polynomial
   {
      E[0] = y;
      E[1] = z;
      E[2] = x;

      curlE[0] = -1.0;
      curlE[1] = -1.0;
      curlE[2] = -1.0; 

      curl2E[0] =0.0;
      curl2E[1] =0.0;
      curl2E[2] =0.0;
   }
      else if (isol == 4) //constant
   {
      E[0] = 1.0;
      E[1] = 1.0;
      E[2] = 1.0;

      curlE[0] = 0.0;
      curlE[1] = 0.0;
      curlE[2] = 0.0; 

      curl2E[0] =0.0;
      curl2E[1] =0.0;
      curl2E[2] =0.0;
   }
   else if (isol == 3) // plane wave
   {
      double coeff = omega / sqrt(3.0);
      E[0] = cos(coeff * (x + y + z));
      E[1] = 0.0;
      E[2] = 0.0;


      curlE[0] = 0.0;
      curlE[1] = -coeff * sin(coeff * (x+y+z));
      curlE[2] = coeff * sin(coeff * (x+y+z)); 

      curl2E[0] = 2.0 * coeff * coeff * E[0];
      curl2E[1] = -coeff * coeff * E[0];
      curl2E[2] = -coeff * coeff * E[0];
   }

}

