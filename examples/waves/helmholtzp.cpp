//
// Compile with: make helmholtz
//
// Sample runs:  helmholtz -m ../data/one-hex.mesh
//               helmholtz -m ../data/fichera.mesh
//               helmholtz -m ../data/fichera-mixed.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Helmholtz problem
//               -Delta p - omega^2 p = 1 with impedance boundary conditiones.
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution and r.h.s., see below for implementation.
double p_exact_Re(const Vector &x);
double p_exact_Im(const Vector &x);
double f_exact_Re(const Vector &x);
double f_exact_Im(const Vector &x);
double g_exact_Re(const Vector &x);
double g_exact_Im(const Vector &x);
void grad_exact_Re(const Vector &x, Vector &grad_Re);
void grad_exact_Im(const Vector &x, Vector &grad_Im);

int dim;
double omega;

// flag for definite or indefinite
// #define DEFINITE

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

int main(int argc, char *argv[])
{

   // 1. Initialise MPI
   int num_procs, myid;
   MPI_Init(&argc, &argv);                    // Initialise MPI
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //total number of processors available
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);      // Determine process identifier

   //-----------------------------------------------------------------------------

   // 2. Parse command-line options.
   // geometry file
   const char *mesh_file = "../../data/one-hex.mesh";
   // finite element order of approximation
   int order = 1;
   // static condensation flag
   bool static_cond = false;
   bool visualization = 1;
   // number of wavelengths
   double k = 0.5;
   // number of mg levels
   int maxref = 1;
   // number of initial ref
   int initref = 1;
   // PETSC
   // const char *petscrc_file = "";
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
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
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
   // omega = 2.0 * M_PI * k;
   omega = k;

   //-----------------------------------------------------------------------------

   // 3. Read the mesh from the given mesh file.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();

   // ----------------------------------------------------------------------------

   // 3. Executing uniform h-refinement
   for (int i = 0; i < initref; i++ )
   {
      mesh->UniformRefinement();
   }

   // ----------------------------------------------------------------------------

   // 5. Define a parallel mesh and delete the serial mesh.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // ----------------------------------------------------------------------------

   // 6. Define a finite element space on the mesh.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
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

   // 2b. We initialize PETSc
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);

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
   //   |  A      B   |  |x_Re|     |b_Re|
   //   |             |  |    |  =  |    |
   //   |  C      D   |  |x_Im|     |b_Im|
   //   |_           _|  |_  _|     |_  _|

   FunctionCoefficient p_Re(p_exact_Re);
   FunctionCoefficient p_Im(p_exact_Im);
   BlockVector x(block_offsets), rhs(block_offsets);
   BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

   // 6. Set up the linear form (Real and Imaginary part)
   FunctionCoefficient f_Re(f_exact_Re);
   FunctionCoefficient g_Re(g_exact_Re);
   VectorFunctionCoefficient grad_Re(sdim, grad_exact_Re);
   ParLinearForm *b_Re(new ParLinearForm);
   b_Re->Update(fespace, rhs.GetBlock(0), 0);
   b_Re->AddDomainIntegrator(new DomainLFIntegrator(f_Re));
   b_Re->AddBoundaryIntegrator(new BoundaryLFIntegrator(g_Re));
   b_Re->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_Re));
   b_Re->Assemble();
   b_Re->ParallelAssemble(trueRhs.GetBlock(0));

   FunctionCoefficient f_Im(f_exact_Im);
   FunctionCoefficient g_Im(g_exact_Im);
   VectorFunctionCoefficient grad_Im(sdim, grad_exact_Im);
   ParLinearForm *b_Im(new ParLinearForm);
   b_Im->Update(fespace, rhs.GetBlock(1), 0);
   b_Im->AddDomainIntegrator(new DomainLFIntegrator(f_Im));
   b_Im->AddBoundaryIntegrator(new BoundaryLFIntegrator(g_Im));
   b_Im->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(grad_Im));
   b_Im->Assemble();
   b_Im->ParallelAssemble(trueRhs.GetBlock(1));

   // 7. Set up the bilinear form (Real and Imaginary part)
   ConstantCoefficient one(1.0);
   ConstantCoefficient sigma(-pow(omega, 2));

   ParBilinearForm *a_rr(new ParBilinearForm(fespace));
   a_rr->AddDomainIntegrator(new DiffusionIntegrator(one));
   a_rr->AddDomainIntegrator(new MassIntegrator(sigma));
   a_rr->Assemble();
   a_rr->Finalize();
   HypreParMatrix *A_rr = a_rr->ParallelAssemble();

   ParBilinearForm *a_ri = new ParBilinearForm(fespace);
   ConstantCoefficient negimpedance(-omega);
   a_ri->AddBoundaryIntegrator(new BoundaryMassIntegrator(negimpedance));
   a_ri->Assemble();
   a_ri->Finalize();
   HypreParMatrix *A_ri = a_ri->ParallelAssemble();

   ParBilinearForm *a_ir = new ParBilinearForm(fespace);
   ConstantCoefficient impedance(omega);
   a_ir->AddBoundaryIntegrator(new BoundaryMassIntegrator(impedance));
   a_ir->Assemble();
   a_ir->Finalize();
   HypreParMatrix *A_ir = a_ir->ParallelAssemble();

   ParBilinearForm *a_ii = new ParBilinearForm(fespace);
   a_ii->AddDomainIntegrator(new DiffusionIntegrator(one));
   a_ii->AddDomainIntegrator(new MassIntegrator(sigma));
   a_ii->Assemble();
   a_ii->Finalize();
   HypreParMatrix *A_ii = a_ii->ParallelAssemble();








   BlockOperator *HelmholtzOp = new BlockOperator(block_trueOffsets);
   HelmholtzOp->SetBlock(0, 0, A_rr);
   HelmholtzOp->SetBlock(0, 1, A_ri);
   HelmholtzOp->SetBlock(1, 0, A_ir);
   HelmholtzOp->SetBlock(1, 1, A_ii);

   PetscLinearSolver * invRe = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invRe->SetOperator(PetscParMatrix(A_rr, Operator::PETSC_MATAIJ));
   PetscLinearSolver * invIm = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invIm->SetOperator(PetscParMatrix(A_ii, Operator::PETSC_MATAIJ));
   BlockDiagonalPreconditioner *precInv = new BlockDiagonalPreconditioner(block_trueOffsets);
   precInv->SetDiagonalBlock(0, invRe);
   precInv->SetDiagonalBlock(1, invIm);



   GMGSolver * gmgRe = new GMGSolver(A_rr, P);
   gmgRe->SetTheta(0.5);
   gmgRe->SetSmootherType(HypreSmoother::Jacobi);
   GMGSolver * gmgIm = new GMGSolver(A_ii, P);
   gmgIm->SetTheta(0.5);
   gmgIm->SetSmootherType(HypreSmoother::Jacobi);
   BlockDiagonalPreconditioner *precGMG = new BlockDiagonalPreconditioner(block_trueOffsets);
   precGMG->SetDiagonalBlock(0, gmgRe);
   precGMG->SetDiagonalBlock(1, gmgIm);


   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(0), trueX.GetBlock(0));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(0),trueRhs.GetBlock(0));

   fespace->GetRestrictionMatrix()->Mult(x.GetBlock(1), trueX.GetBlock(1));
   fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(1),trueRhs.GetBlock(1));

   if (myid == 0)
   {
      cout << "Size of fine grid system: "
           << 2.0 * A_rr->GetGlobalNumRows() << " x " << 2.0* A_rr->GetGlobalNumCols() << endl;
   }
   int maxit(5000);
   double rtol(1.e-6);
   double atol(0.0);
   trueX = 0.0;
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetAbsTol(atol);
   gmres.SetRelTol(rtol);
   gmres.SetMaxIter(maxit);
   gmres.SetOperator(*HelmholtzOp);
   // gmres.SetOperator(*cSysMat);
   gmres.SetPreconditioner(*precGMG);
   gmres.SetPrintLevel(1);
   gmres.Mult(trueRhs, trueX);

   if (myid == 0)
   {
      cout << "GMRES with GMG finished" << endl;
   }
   trueX = 0.0;
   gmres.SetPreconditioner(*precInv);
   gmres.Mult(trueRhs, trueX);

   if (myid == 0)
   {
      cout << "GMRES with exact Inv finished" << endl;
   }

   // Direct solve
   ComplexHypreParMatrix chpm(A_rr, A_ir, false, false);
   HypreParMatrix *CA = chpm.GetSystemMatrix();
   PetscLinearSolver * invCA = new PetscLinearSolver(MPI_COMM_WORLD, "direct");
   invCA->SetOperator(PetscParMatrix(CA, Operator::PETSC_MATAIJ));
   trueX = 0.0;
   invCA->Mult(trueRhs,trueX);

   if (myid == 0)
   {
      cout << "Direct solver finished" << endl;
   }


   ParGridFunction *x_Re(new ParGridFunction);
   ParGridFunction *x_Im(new ParGridFunction);

   x_Re->MakeRef(fespace, x.GetBlock(0), 0);
   x_Im->MakeRef(fespace, x.GetBlock(1), 0);

   x_Re->Distribute(&(trueX.GetBlock(0)));
   x_Im->Distribute(&(trueX.GetBlock(1)));

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   const int h1_norm_type = 1;
   double L2error;
   double H1error;

   double L2err_Re  = x_Re->ComputeL2Error(p_Re);
   double L2err_Im  = x_Im->ComputeL2Error(p_Im);

   double loc_H1err_Re = x_Re->ComputeH1Error(&p_Re, &grad_Re, &one, 1.0, h1_norm_type);
   double loc_H1err_Im = x_Im->ComputeH1Error(&p_Im, &grad_Im, &one, 1.0, h1_norm_type);
   double H1err_Re = GlobalLpNorm(2.0, loc_H1err_Re, MPI_COMM_WORLD);
   double H1err_Im = GlobalLpNorm(2.0, loc_H1err_Im, MPI_COMM_WORLD);
   // double norm_Re = ComputeGlobalLpNorm(2, p_Re, *pmesh, irs);
   // double norm_Im = ComputeGlobalLpNorm(2, p_Im, *pmesh, irs);
      
   L2error = sqrt(L2err_Re*L2err_Re + L2err_Im*L2err_Im);
   H1error = sqrt(H1err_Re*H1err_Re + H1err_Im*H1err_Im);
   // double L2norm = sqrt(norm_Re*norm_Re + norm_Im*norm_Im);
   if (myid == 0)
   {
      cout << " || p_h - p ||_{H^1} = " << H1error <<  endl;
      cout << " || p_h - p ||_{L^2} = " << L2error <<  endl;
      // cout << " || p_h - p ||_{L^2}/||p||_{L^2} = " << L2error/L2norm <<  endl;
   }

   // release memory
   delete b_Re;
   delete b_Im;
   delete a_rr;
   delete a_ri;
   delete a_ir;
   delete a_ii;

   delete fespace;
   delete fec;
   delete pmesh;
   MFEMFinalizePetsc();
   MPI_Finalize();
   return 0;
}

//define exact solution plane wave
double p_exact_Re(const Vector &x)
{
   return cos(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
}
double p_exact_Im(const Vector &x)
{
   return -sin(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
}

//calculate RHS from exact solution f = - \Delta u
double f_exact_Re(const Vector &x)
{
   return 0.0;
}
double f_exact_Im(const Vector &x)
{
   return 0.0;
}

void grad_exact_Re(const Vector &x, Vector &grad_Re)
{
   grad_Re[0] = -omega / sqrt(3.0) * sin(omega / sqrt(3.0) * (x(0) + x(1) + x(2)));
   grad_Re[1] = grad_Re[0];
   grad_Re[2] = grad_Re[1];
}
void grad_exact_Im(const Vector &x, Vector &grad_Im)
{

   grad_Im[0] = -omega / sqrt(3.0) * cos(omega / sqrt(3.0) * (x(0) + x(1) + x(2)));
   grad_Im[1] = grad_Im[0];
   grad_Im[2] = grad_Im[1];
}

//define impedence coefficient: i omega p
double g_exact_Re(const Vector &x)
{
   return omega * sin(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
}
double g_exact_Im(const Vector &x)
{
   return omega * cos(omega * (x(0) + x(1) + x(2)) / sqrt(3.0));
}
