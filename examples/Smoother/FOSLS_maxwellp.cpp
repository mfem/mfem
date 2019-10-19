
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <boost/math/special_functions/airy.hpp>
#include "blkams.hpp"
#include "multigrid.hpp"

using namespace std;
using namespace mfem;
using namespace boost;

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void H_exact(const Vector & x, Vector & H);
void f_exact_H(const Vector & x, Vector & f_H);
void get_maxwell_solution(const Vector & x, double E[], double curlE[], double curl2E[]);
void epsilon_func(const Vector &x, DenseMatrix &M);
void epsilon2_func(const Vector &x, DenseMatrix &M);

int dim;
double omega;
int sol = 1;


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
   int ref_levels = 1;
   // number of initial ref
   int initref = 1;

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
   args.AddOption(&ref_levels, "-ref", "--ref_levels",
                  "Number of Refinements.");
   args.AddOption(&initref, "-initref", "--initref",
                  "Number of initial refinements.");
   args.AddOption(&sol, "-sol", "--exact",
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

   // 2. Read the mesh from the given mesh file.
   Mesh *mesh;
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   double length;
   length = (sol == 4) ? 0.5: 1.0;
   mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, length, length, length, false);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   ParMesh *cpmesh = new ParMesh(*pmesh);
   
// 4. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   // ParFiniteElementSpace *fespace = new ParFiniteElementSpace(mesh, fec);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   std::vector<ParFiniteElementSpace * > fespaces(ref_levels+1);
   std::vector<ParMesh * > ParMeshes(ref_levels+1);
   std::vector<HypreParMatrix*>  P(ref_levels);

   for (int i = 0; i < ref_levels; i++)
   {
      ParMeshes[i] =new ParMesh(*pmesh);
      fespaces[i] = new ParFiniteElementSpace(*fespace, *ParMeshes[i]);
      pmesh->UniformRefinement();
      // Update fespace
      fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      fespace->GetTrueTransferOperator(*fespaces[i], Tr);
      Tr.SetOperatorOwner(false);
      Tr.Get(P[i]);
   }
   fespaces[ref_levels] = new ParFiniteElementSpace(*fespace);

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
   ParGridFunction * Exact_gf = new ParGridFunction(fespace);
   E_gf->MakeRef(fespace, x.GetBlock(0));
   E_gf->ProjectCoefficient(Eex);
   Exact_gf->ProjectCoefficient(Eex);

   VectorFunctionCoefficient Hex(sdim, H_exact);
   ParGridFunction * H_gf = new ParGridFunction;   
   H_gf->MakeRef(fespace, x.GetBlock(1));
   // H_gf->ProjectCoefficient(Hex);


   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(pow(omega, 2));
   ConstantCoefficient neg(-abs(omega));
   ConstantCoefficient pos(abs(omega));

   // // 6. Set up the linear form
   VectorFunctionCoefficient f_H(sdim,f_exact_H);
   ScalarVectorProductCoefficient sf_H(neg,f_H);



   ParLinearForm *b_E = new ParLinearForm;
   b_E->Update(fespace, rhs.GetBlock(0), 0);
   b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(sf_H));
   b_E->Assemble();


   ParLinearForm *b_H = new ParLinearForm;
   b_H->Update(fespace, rhs.GetBlock(1), 0);
   b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
   b_H->Assemble();



   // 7. Bilinear form a(.,.) on the finite element space


   MatrixFunctionCoefficient epsilon(dim,epsilon_func);
   MatrixFunctionCoefficient epsilon2(dim,epsilon2_func);
   ScalarMatrixProductCoefficient coeff(neg,epsilon);
   ScalarMatrixProductCoefficient coeff2(sigma,epsilon2);


   //
   ParBilinearForm *a_EE = new ParBilinearForm(fespace);
   a_EE->AddDomainIntegrator(new CurlCurlIntegrator(one)); 
   a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(coeff2));
   a_EE->Assemble();
   a_EE->EliminateEssentialBC(ess_bdr,x.GetBlock(0), rhs.GetBlock(0));
   a_EE->Finalize();
   HypreParMatrix *A_EE = a_EE->ParallelAssemble();

   ParMixedBilinearForm *a_HE = new ParMixedBilinearForm(fespace,fespace);
   a_HE->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
   a_HE->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(coeff)); 
   a_HE->Assemble();
   a_HE->EliminateTrialDofs(ess_bdr, x.GetBlock(0), rhs.GetBlock(1));
   a_HE->Finalize();

   HypreParMatrix *A_HE = a_HE->ParallelAssemble();

   HypreParMatrix *A_EH = A_HE->Transpose();

   ParBilinearForm *a_HH = new ParBilinearForm(fespace);
   a_HH->AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
   a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
   a_HH->Assemble();
   a_HH->Finalize();
   HypreParMatrix *A_HH = a_HH->ParallelAssemble();

   BlockOperator *LS_Maxwellop = new BlockOperator(block_trueOffsets);
   LS_Maxwellop->SetBlock(0, 0, A_EE);
   LS_Maxwellop->SetBlock(0, 1, A_EH);
   LS_Maxwellop->SetBlock(1, 0, A_HE);
   LS_Maxwellop->SetBlock(1, 1, A_HH);
   

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << 2.0 * A_EE->GetGlobalNumRows() << " x " << 2.0* A_EE->GetGlobalNumCols() << endl;
   }

   // Set up the preconditioner
   Array2D<HypreParMatrix*> blockA(2,2);
   blockA(0,0) = A_EE;
   blockA(0,1) = A_EH;
   blockA(1,0) = A_HE;
   blockA(1,1) = A_HH;


   int maxit(100);
   double rtol(1.e-6);
   double atol(0.0);
   trueX = 0.0;
   
   CGSolver pcg(MPI_COMM_WORLD);
   pcg.SetAbsTol(atol);
   pcg.SetRelTol(rtol);
   pcg.SetMaxIter(maxit);
   pcg.SetOperator(*LS_Maxwellop);
   pcg.SetPrintLevel(1);

   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
   chrono.Clear();
   chrono.Start();
   BlockMGSolver * precMG = new BlockMGSolver(blockA,P,fespaces);
   chrono.Stop();
   if(myid == 0)
      cout << "MG Setup time: " << chrono.RealTime() << endl;

   

   chrono.Clear();
   chrono.Start();
   pcg.SetPreconditioner(*precMG);
   pcg.Mult(trueRhs, trueX);
   chrono.Stop();
   delete precMG;
   MFEMFinalizePetsc();

    if(myid == 0)
      cout << "MG prec Solution time: " << chrono.RealTime() << endl;


   chrono.Clear();
   chrono.Start();
   Block_AMSSolver * precAMS = new Block_AMSSolver(block_trueOffsets,fespaces);
   precAMS->SetSmootherType(Block_AMS::BlkSmootherType::SCHWARZ);
   // precAMS->SetSmootherType(Block_AMS::BlkSmootherType::HYPRE);
   precAMS->SetOperator(blockA);
   precAMS->SetTheta(0.2);
   // 0-Smoother, 1-Grad, 2,3,4-Pix,Piy,Piz
   precAMS->SetCycleType("023414320");
   precAMS->SetNumberofCycles(1);
   chrono.Stop();
   if(myid == 0)
      cout << "BlkAMS Setup time: " << chrono.RealTime() << endl;

   // resolve with block AMS
   trueX = 0; 
   chrono.Clear();
   chrono.Start();
   pcg.SetPreconditioner(*precAMS);
   pcg.Mult(trueRhs, trueX);
   chrono.Stop();
   delete precAMS;

   if(myid == 0)
      cout << "BlockAMS Solution time: " << chrono.RealTime() << endl;


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
      cout << "|| E_h - E || = " << Error_E  << "\n";
      cout << "|| E_h - E ||/||E|| = " << Error_E/norm_E  << "\n";
      // cout << "|| H_h - H || = " << Error_H  << "\n";
      // cout << "Total error = " << sqrt(Error_H*Error_H+Error_E*Error_E) << "\n";
   }

   ParGridFunction ExactE(fespace);

   if (visualization)
   {
      // 8. Connect to GLVis.
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream E_sock(vishost, visport);
      E_sock << "parallel " << num_procs << " " << myid << "\n";
      E_sock.precision(8);
      E_sock << "solution\n" << *pmesh << *E_gf << "window_title 'Electric field'" << endl;
      // socketstream Exact_sock(vishost, visport);
      // Exact_sock << "parallel " << num_procs << " " << myid << "\n";
      // Exact_sock.precision(8);
      // Exact_sock << "solution\n" << *pmesh << *Exact_gf << "window_title 'Electric field'" << endl;
 
    // MPI_Barrier(pmesh->GetComm());
      // socketstream Eex_sock(vishost, visport);
      // Eex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Eex_sock.precision(8);
      // Eex_sock << "solution\n" << *pmesh << *Exact_gf << "window_title 'Exact Electric field'" << endl;
   }

   delete a_EE;
   delete a_HE;
   delete a_HH;
   delete b_E;
   delete b_H;
   delete fec;
   delete fespace;
   delete pmesh;
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
   for (int i = 0; i<3; i++) H(i) = curlE[i]/omega;
}


void f_exact_H(const Vector &x, Vector &f)
{
   // curl H - omega E = f
   // = curl (curl E / omega) - omega E
   f = 0.0;
   if (sol !=4)
   {
      double E[3], curlE[3], curl2E[3];
      get_maxwell_solution(x, E, curlE, curl2E);
      f(0) = curl2E[0] / omega - omega * E[0];
      f(1) = curl2E[1] / omega - omega * E[1];
      f(2) = curl2E[2] / omega - omega * E[2];
   }
}
 
   void get_maxwell_solution(const Vector &X, double E[], double curlE[], double curl2E[])
{
   double x = X[0];
   double y = X[1];
   double z = X[2];



   if (sol ==-1)
   {
      E[0] = y * z * (1.0 - y) * (1.0 - z);
      E[1] = x * y * z * (1.0 - x) * (1.0 - z);
      E[2] = x * y * (1.0 - x) * (1.0 - y);
      
      curlE[0] = -(x-1.0) * x * (y*(2.0*z-3.0)+1.0);
      curlE[1] = -2.0*(y-1.0)*y*(x-z);
      curlE[2] = (z-1)*z*(1.0+y*(2.0*x-3.0));
      
      curl2E[0] = 2.0 * y * (1.0 - y) - (2.0 * x - 3.0) * z * (1 - z);
      curl2E[1] = 2.0 * y * (x * (1.0 - x) + (1.0 - z) * z);
      curl2E[2] = 2.0 * y * (1.0 - y) + x * (3.0 - 2.0 * z) * (1.0 - x);
   }
   else if (sol == 0) // polynomial
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
      curl2E[1] = -2.0 * y * (-x + x * x + (-1.0 + z) * z);
      curl2E[2] = -2.0 * (-1.0 + y) * y + (-1.0 + x) * x * (-3.0 + 2.0 * z);
   }
   else if (sol == 1) // sinusoidal
   {
      E[0] = sin(omega * y);
      E[1] = sin(omega * z);
      E[2] = sin(omega * x);

      curlE[0] = -omega * cos(omega * z);
      curlE[1] = -omega * cos(omega * x);
      curlE[2] = -omega * cos(omega * y);

      curl2E[0] = omega * omega * E[0];
      curl2E[1] = omega * omega * E[1];
      curl2E[2] = omega * omega * E[2];
   }
   else if (sol == 2) // point source
   {
      MFEM_ABORT("Case unfinished");
   }
   else if (sol == 3) // plane wave
   {
      double coeff = omega / sqrt(3.0);
      E[0] = cos(coeff * (x + y + z));
      E[1] = 0.0;
      E[2] = 0.0;

      curlE[0] = 0.0;
      curlE[1] = -coeff * sin(coeff * (x + y + z));
      curlE[2] = coeff * sin(coeff * (x + y + z));

      curl2E[0] = 2.0 * coeff * coeff * E[0];
      curl2E[1] = -coeff * coeff * E[0];
      curl2E[2] = -coeff * coeff * E[0];
   }
   else if (sol == -1) 
   {
      E[0] = cos(omega * y);
      E[1] = 0.0;

      curlE[0] = 0.0;
      curlE[1] = 0.0;
      curlE[2] = -omega * sin(omega * y);

      curl2E[0] = omega*omega * cos(omega*y);  
      curl2E[1] = 0.0;
      curl2E[2] = 0.0;
   }
   else if (sol == 4) // Airy function
   {
      E[0] = 0;
      E[1] = 0;
      // double b = -pow(omega/4.0,2.0/3.0)*(4.0*x(0)-1.0);
      double b = -pow(omega/4.0,2.0/3.0)*(4.0*x-1.0);
      E[2] = boost::math::airy_ai(b);

      //  not used
      curl2E[0] = 0.0;
      curl2E[1] = 0.0;  
      curl2E[2] = 0.0;
   }
}


void epsilon_func(const Vector &x, DenseMatrix &M)
{
   M.SetSize(3);

   M = 0.0;
   M(0,0) = 1.0;
   M(1,1) = 1.0;
   if (sol != 4)
   {
      M(2,2) = 1.0;
   }
   else
   {
      M(2,2) = 4.0*x(0)-1.0;
      // M(2,2) = 2.0;
   }
}

void epsilon2_func(const Vector &x, DenseMatrix &M)
{
   M.SetSize(3);

   M = 0.0;
   M(0,0) = 1.0;
   M(1,1) = 1.0;
   if (sol != 4)
   {
      M(2,2) = 1.0;
   }
   else
   {
      M(2,2) = pow(4.0*x(0)-1.0,2.0);
      // M(2,2) = 4.0;
   }
}