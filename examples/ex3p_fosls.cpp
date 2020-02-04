#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "gsl_sf_airy.h"

#include "multigrid.hpp"

using namespace std;
using namespace mfem;

#define COMPLEX_IMPEDANCE  // Whether to use impedance BC and solve a complex system

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void H_exact(const Vector & x, Vector & H);
void f_exact_H(const Vector & x, Vector & f_H);
void get_maxwell_solution(const Vector & x, double E[], double curlE[], double curl2E[]);
void epsilon_func(const Vector &x, DenseMatrix &M);
void epsilon2_func(const Vector &x, DenseMatrix &M);

int dim;
double omega;
int sol = 4;

int main(int argc, char *argv[])
{
  StopWatch chrono;

  // 1. Initialize MPI
  int num_procs, myid;
  MPI_Init(&argc, &argv); // Initialize MPI
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs); //total number of processors available
  MPI_Comm_rank(MPI_COMM_WORLD, &myid); // Determine process identifier
  // 1. Parse command-line options.
  // geometry file
  const char *mesh_file = "../data/inline-hex.mesh";
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
  //omega = 2.0*k*M_PI;
  omega = k;

  // 2. Read the mesh from the given mesh file.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);

  if (sol == 4)
    {
      mesh->EnsureNodes();
      GridFunction *nodes = mesh->GetNodes();
      (*nodes) *= 0.5;
    }
  
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
  std::vector<ParFiniteElementSpace * > fespaces(ref_levels+1);
  std::vector<ParMesh * > ParMeshes(ref_levels+1);
  std::vector<HypreParMatrix*> P(ref_levels);

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

  Array<int> ess_tdof_listE;
  Array<int> ess_tdof_listH;
  Array<int> ess_bdrE(pmesh->bdr_attributes.Max());
  Array<int> ess_bdrH(pmesh->bdr_attributes.Max());
#ifdef COMPLEX_IMPEDANCE
  ess_bdrE = 0;
  ess_bdrH = 0;  // Neumann
#else
  ess_bdrE = 1;
  ess_bdrH = 0;  // Neumann
#endif
  fespace->GetEssentialTrueDofs(ess_bdrE, ess_tdof_listE);
  fespace->GetEssentialTrueDofs(ess_bdrH, ess_tdof_listH);

#ifdef COMPLEX_IMPEDANCE
  Array<int> bdr_attr(pmesh->bdr_attributes.Max());
  bdr_attr = 1;
  ConstantCoefficient impedance(omega);
  RestrictedCoefficient imp_rest(impedance, bdr_attr);
  Array<int> block_offsets(5);
  Array<int> block_trueOffsets(5);
#else
  Array<int> block_offsets(3);
  Array<int> block_trueOffsets(3);
#endif
  block_offsets[0] = 0;
  block_offsets[1] = fespace->GetVSize();
  block_offsets[2] = fespace->GetVSize();
#ifdef COMPLEX_IMPEDANCE
  block_offsets[3] = fespace->GetVSize();
  block_offsets[4] = fespace->GetVSize();  
#endif
  block_offsets.PartialSum();

  block_trueOffsets[0] = 0;
  block_trueOffsets[1] = fespace->TrueVSize();
  block_trueOffsets[2] = fespace->TrueVSize();
#ifdef COMPLEX_IMPEDANCE
  block_trueOffsets[3] = fespace->TrueVSize();
  block_trueOffsets[4] = fespace->TrueVSize();
#endif
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
  // A11 = (curl H, curl G) + \omega^2 (H,G)

  BlockVector x(block_offsets), rhs(block_offsets);
  BlockVector trueX(block_trueOffsets), trueRhs(block_trueOffsets);

  x = 0.0;
  rhs = 0.0;
  trueX = 0.0;
  trueRhs = 0.0;

  VectorFunctionCoefficient Eex(sdim, E_exact);
  ConstantCoefficient negOne(-1.0);
  VectorFunctionCoefficient mEex(sdim, E_exact, &negOne);
    
  ParGridFunction * E_gf = new ParGridFunction;
  ParGridFunction * Exact_gf = new ParGridFunction(fespace);
  E_gf->MakeRef(fespace, x.GetBlock(0));
  E_gf->ProjectCoefficient(Eex);
  Exact_gf->ProjectCoefficient(Eex);

  VectorFunctionCoefficient Hex(sdim, H_exact);
  ParGridFunction * H_gf = new ParGridFunction;   
  H_gf->MakeRef(fespace, x.GetBlock(1));
  H_gf->ProjectCoefficient(Hex);

  ConstantCoefficient one(1.0);
  ConstantCoefficient sigma(pow(omega, 2));
  ConstantCoefficient neg(-abs(omega));
  ConstantCoefficient pos(abs(omega));

  MatrixFunctionCoefficient epsilon(dim,epsilon_func);
  TransposeMatrixCoefficient epsilonT(epsilon);  // transpose of epsilon
  MatrixFunctionCoefficient epsilon2(dim,epsilon2_func);
  ScalarMatrixProductCoefficient coeff(neg,epsilon);
  ScalarMatrixProductCoefficient coeff2(sigma,epsilon2);

  // 6. Set up the linear form
  VectorFunctionCoefficient f_H(sdim,f_exact_H);
  ScalarVectorProductCoefficient sf_H(neg,f_H);

  MatVecCoefficient epsT_sf_H(epsilonT, sf_H);
    
  ParLinearForm *b_E = new ParLinearForm;
  b_E->Update(fespace, rhs.GetBlock(0), 0);
  b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(epsT_sf_H));
#ifdef COMPLEX_IMPEDANCE
  b_E->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Eex));  // <g_Im, n x Q x n> = <n x E_Re x n, n x Q x n>

  ParLinearForm *b_E_Im = new ParLinearForm;
  b_E_Im->Update(fespace, rhs.GetBlock(2), 0);
  b_E_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(Hex));  // -<g_Re, n x Q x n> = <n x H_Re, n x Q x n>
  b_E_Im->Assemble();
#endif
  b_E->Assemble();

  ParLinearForm *b_H = new ParLinearForm;
  b_H->Update(fespace, rhs.GetBlock(1), 0);
  b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));
#ifdef COMPLEX_IMPEDANCE
  b_H->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Hex));  // -<g_Re, n x R> = <n x H_Re, n x R>

  ParLinearForm *b_H_Im = new ParLinearForm;
  b_H_Im->Update(fespace, rhs.GetBlock(3), 0);
  b_H_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(mEex));  // -<g_Im, n x R> = -<n x E_Re x n, n x R>
  b_H_Im->Assemble();
#endif
  b_H->Assemble();

#ifdef COMPLEX_IMPEDANCE
  fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(2), trueRhs.GetBlock(2));
  fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(3), trueRhs.GetBlock(3));
#endif
  
  // 7. Bilinear form a(.,.) on the finite element space
  ParBilinearForm *a_EE = new ParBilinearForm(fespace);
  a_EE->AddDomainIntegrator(new CurlCurlIntegrator(one)); 
  a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(coeff2));

#ifdef COMPLEX_IMPEDANCE
  a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator());
  //a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator(imp_rest));
  //a_EE->AddBoundaryIntegrator(new BoundaryMassIntegrator(imp_rest));
  //a_EE->AddBdrFaceIntegrator(new BoundaryMassIntegrator(imp_rest));
#endif
  
  a_EE->Assemble();
  a_EE->Finalize();
  HypreParMatrix *A_EE = new HypreParMatrix;
  a_EE->FormLinearSystem(ess_tdof_listE, x.GetBlock(0), rhs.GetBlock(0), *A_EE, trueX.GetBlock(0), trueRhs.GetBlock(0));

  ParBilinearForm *a_HH = new ParBilinearForm(fespace);
  a_HH->AddDomainIntegrator(new CurlCurlIntegrator(one)); // one is the coeff
  a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));

#ifdef COMPLEX_IMPEDANCE
  a_HH->AddBoundaryIntegrator(new VectorFEMassIntegrator());
  //a_HH->AddBoundaryIntegrator(new VectorFEMassIntegrator(imp_rest));
  //a_HH->AddBoundaryIntegrator(new BoundaryMassIntegrator(imp_rest));
#endif

  a_HH->Assemble();
  a_HH->Finalize();
  HypreParMatrix *A_HH = new HypreParMatrix;
  a_HH->FormLinearSystem(ess_tdof_listH, x.GetBlock(1), rhs.GetBlock(1), *A_HH, trueX.GetBlock(1), trueRhs.GetBlock(1));

  ParMixedBilinearForm *a_HE = new ParMixedBilinearForm(fespace,fespace);
  a_HE->AddDomainIntegrator(new MixedVectorCurlIntegrator(neg));
  a_HE->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(coeff)); 
  a_HE->Assemble();
  a_HE->Finalize();
  HypreParMatrix *A_HE = new HypreParMatrix;
  a_HE->FormColLinearSystem(ess_tdof_listE,x.GetBlock(0),rhs.GetBlock(1),*A_HE,trueX.GetBlock(0),trueRhs.GetBlock(1));

  HypreParMatrix *A_EH = A_HE->Transpose();

#ifdef COMPLEX_IMPEDANCE
  ParBilinearForm *a_EH_Im = new ParBilinearForm(fespace);
  //a_EH_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator(imp_rest));
  //a_EH_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator(impedance));
  //a_EH_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator(omega));
  a_EH_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator());
  a_EH_Im->Assemble();
  a_EH_Im->Finalize();

  OperatorHandle A_EH_Im_ptr;
  a_EH_Im->FormSystemMatrix(ess_tdof_listE, A_EH_Im_ptr);  // empty ess_tdof_list for impedance

  HypreParMatrix *A_EH_Im = A_EH_Im_ptr.As<HypreParMatrix>();
  
  HypreParMatrix *A_HE_Im = A_EH_Im->Transpose();
#endif

  BlockOperator *LS_Maxwellop = new BlockOperator(block_trueOffsets);
#ifdef COMPLEX_IMPEDANCE
  const int numBlocks = 4;
#else
  const int numBlocks = 2;
#endif
  
  LS_Maxwellop->SetBlock(0, 0, A_EE);
  LS_Maxwellop->SetBlock(0, 1, A_EH);
  LS_Maxwellop->SetBlock(1, 0, A_HE);
  LS_Maxwellop->SetBlock(1, 1, A_HH);

#ifdef COMPLEX_IMPEDANCE
  LS_Maxwellop->SetBlock(0, 3, A_EH_Im);
  LS_Maxwellop->SetBlock(1, 2, A_HE_Im, -1.0);
  
  LS_Maxwellop->SetBlock(2, 1, A_EH_Im, -1.0);
  LS_Maxwellop->SetBlock(3, 0, A_HE_Im);
  
  LS_Maxwellop->SetBlock(2, 2, A_EE);
  LS_Maxwellop->SetBlock(2, 3, A_EH);
  LS_Maxwellop->SetBlock(3, 2, A_HE);
  LS_Maxwellop->SetBlock(3, 3, A_HH);
#endif
  
  if (myid == 0)
  {
     cout << "Size of fine grid system: "
          << 2.0 * A_EE->GetGlobalNumRows() << " x " << 2.0* A_EE->GetGlobalNumCols() << endl;
  }

  // Set up the preconditioner  
  Array2D<HypreParMatrix*> blockA(numBlocks, numBlocks);
  Array2D<double> blockAcoef(numBlocks, numBlocks);

  for (int i=0; i<numBlocks; ++i)
  {
     for (int j=0; j<numBlocks; ++j)
     {
       if (LS_Maxwellop->IsZeroBlock(i,j) == 0)
	 {
	   blockA(i,j) = static_cast<HypreParMatrix *>(&LS_Maxwellop->GetBlock(i,j));
	   blockAcoef(i,j) = LS_Maxwellop->GetBlockCoef(i,j);
	 }
       else
	 {
	   blockA(i,j) = NULL;
	   blockAcoef(i,j) = 1.0;
	 }
     }
  }

  // // double nnz = A_HH->NNZ();
  // // double ndof = A_HH->GetGlobalNumRows();
  // // double est_mem_b = nnz*12.0 + (ndof+1.0)*4; 
  // // double gb = est_mem_b*4.0/pow(1024.0,3);

  // // mfem::out << "Estimated memory taken by the global matrix: " <<  gb << endl;

  int maxit(2000);
  double rtol(1.e-8);
  double atol(1.e-12);

  // trueX = 0.0;
  CGSolver pcg(MPI_COMM_WORLD);
  pcg.SetAbsTol(atol);
  pcg.SetRelTol(rtol);
  pcg.SetMaxIter(maxit);
  pcg.SetOperator(*LS_Maxwellop);
  pcg.SetPrintLevel(1);
  chrono.Clear();
  chrono.Start();
  BlockMGSolver * precMG = new BlockMGSolver(LS_Maxwellop->Height(), LS_Maxwellop->Width(), blockA, blockAcoef, P);
  //precMG->SetTheta(0.5);
  // // int lv_coarse = min(ref_levels,ref_levels-1);
  // // int levels = ref_levels - lv_coarse; 
  // // BlkParSchwarzSmoother * precAS = new BlkParSchwarzSmoother(fespaces[lv_coarse]->GetParMesh(),levels,fespaces[ref_levels],LS_Maxwellop);
  chrono.Stop();
  if (myid == 0)
  {
     cout << "MG Setup time: " << chrono.RealTime() << endl;
  }

  // Randomize H RHS
  //trueRhs = 1.0;
  /*
  for (int i=block_trueOffsets[1]; i<block_trueOffsets[2]; ++i)
    trueRhs[i] = i % 53;
  */
  
  chrono.Clear();
  chrono.Start();
  pcg.SetPreconditioner(*precMG);
  // // pcg.SetPreconditioner(*precAS);
  pcg.Mult(trueRhs, trueX);
  chrono.Stop();
  delete precMG;
  // // delete precAS;
   
  // // trueX = 0.0;
  // // invA->Mult(trueRhs,trueX);

  if (myid == 0)
  {
     cout << "MG Solution time time: " << chrono.RealTime() << endl;
  }   
  // // cin.get();
  // //  if(myid == 0)
  // //    cout << "MG prec Solution time: " << chrono.RealTime() << endl;


  // // chrono.Clear();
  // // chrono.Start();
  // // Block_AMSSolver * precAMS = new Block_AMSSolver(block_trueOffsets,fespaces);
  // // precAMS->SetSmootherType(Block_AMS::BlkSmootherType::SCHWARZ);
  // // precAMS->SetSmootherType(Block_AMS::BlkSmootherType::HYPRE);
  // // precAMS->SetOperator(LS_Maxwellop);
  // // precAMS->SetTheta(1.0/5.0);
  // // // 0-Smoother, 1-Grad, 2,3,4-Pix,Piy,Piz
  // // precAMS->SetCycleType("023414320");
  // // precAMS->SetNumberofCycles(1);
  // // chrono.Stop();
  // // if(myid == 0)
  // //    cout << "BlkAMS Setup time: " << chrono.RealTime() << endl;

  // // // resolve with block AMS
  // // trueX = 0; 
  // // chrono.Clear();
  // // chrono.Start();
  // // pcg.SetPreconditioner(*precAMS);
  // // pcg.Mult(trueRhs, trueX);
  // // chrono.Stop();
  // // delete precAMS;

  // // if(myid == 0)
  // //    cout << "BlockAMS Solution time: " << chrono.RealTime() << endl;


  a_EE->RecoverFEMSolution(trueX.GetBlock(0), *b_E, *E_gf);
  a_HH->RecoverFEMSolution(trueX.GetBlock(1), *b_H, *H_gf);


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
     cout << "|| H_h - H || = " << Error_H  << "\n";
     cout << "|| H_h - H ||/||H|| = " << Error_H/norm_H  << "\n";
     cout << "Total error = " << setprecision(15) << sqrt(Error_H*Error_H+Error_E*Error_E) << "\n";
  }

  {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      E_gf->Save(sol_ofs);
   }
  
  //ParGridFunction ExactE(fespace);
  /*
  if (visualization)
    {
      // 8. Connect to GLVis.
      char vishost[] = "localhost";
      int  visport   = 19916;
      // socketstream E_sock(vishost, visport);
      // E_sock << "parallel " << num_procs << " " << myid << "\n";
      // E_sock.precision(8);
      // E_sock << "solution\n" << *pmesh << *E_gf << "window_title 'Electric field'" << endl;
      socketstream Exact_sock(vishost, visport);
      Exact_sock << "parallel " << num_procs << " " << myid << "\n";
      Exact_sock.precision(8);
      Exact_sock << "solution\n" << *pmesh << *Exact_gf << "window_title 'Electric field'" << endl;
 
      // MPI_Barrier(pmesh->GetComm());
      // socketstream Eex_sock(vishost, visport);
      // Eex_sock << "parallel " << num_procs << " " << myid << "\n";
      // Eex_sock.precision(8);
      // Eex_sock << "solution\n" << *pmesh << *Exact_gf << "window_title 'Exact Electric field'" << endl;
    }
  */
  // delete A_EE;
  // delete A_HE;
  // delete A_EH;
  // delete A_HH;
  // delete LS_Maxwellop;
  // delete a_EE;
  // delete a_HE;
  // delete a_HH;
  // delete b_E;
  // delete b_H;
  delete E_gf;
  delete Exact_gf;
  for (auto p: ParMeshes) delete p;
  for (auto p: fespaces)  delete p;
  for (auto p: P)  delete p;
  ParMeshes.clear();
  fespaces.clear();
  P.clear();
  delete fec;
  delete fespace;
  delete pmesh;
   
  // cout << "Freed memory: " << endl;
  // cin.get();
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
      // shift to avoid singularity
      double x0 = x + 0.1;
      double x1 = y + 0.1;
      double x2 = z + 0.1;
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

      curlE[0] = 0.0;
      curlE[1] = -omega * r_z * sin(omega * r);
      curlE[2] =  omega * r_y * sin(omega * r);

      curl2E[0] = omega * ((r_yy + r_zz) * sin(omega * r) +
                           (omega * r_y * r_y + omega * r_z * r_z) * cos(omega * r));
      curl2E[1] = -omega * (r_yx * sin(omega * r) + omega * r_y * r_x * cos(omega * r));
      curl2E[2] = -omega * (r_zx * sin(omega * r) + omega * r_z * r_x * cos(omega * r));
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
      //E[2] = boost::math::airy_ai(b);
      E[2] = gsl_sf_airy_Ai(b, GSL_PREC_DOUBLE);

      curlE[0] = 0.0;
      curlE[1] = 4.0 * pow(omega/4.0,2.0/3.0) * gsl_sf_airy_Ai_deriv(b, GSL_PREC_DOUBLE);
      curlE[2] = 0.0;
      
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
      M(2,2) = (4.0*x(0)-1.0) * (4.0*x(0)-1.0);
      // M(2,2) = 4.0;
    }
}
