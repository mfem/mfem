#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "gsl_sf_airy.h"

#include "multigrid.hpp"

using namespace std;
using namespace mfem;


//#define DIRECT_SOLVER

// Define exact solution
void E_exact(const Vector & x, Vector & E);
void H_exact(const Vector & x, Vector & H);
void f_exact_H(const Vector & x, Vector & f_H);
void get_maxwell_solution(const Vector & x, double E[], double curlE[], double curl2E[]);
void epsilon_func(const Vector &x, Vector &M);
void epsilon2_func(const Vector &x, Vector &M);
void epsilon_func_mat(const Vector &x, DenseMatrix &M);

int dim;
double omega;
int sol = 3;


class FOSLSSolver : public Solver
{
public:
  FOSLSSolver(ParFiniteElementSpace *fespace_, std::vector<HypreParMatrix*>& P, const int myid_)
    : Solver(2 * fespace_->GetTrueVSize()), M_inv(MPI_COMM_WORLD), fespace(fespace_),
      n(fespace_->GetTrueVSize()), nfull(fespace_->GetVSize()), LSpcg(MPI_COMM_WORLD), myid(myid_)
  {
    z.SetSize(n);
    Minv_x.SetSize(n);
    
    ParMesh *pmesh = fespace->GetParMesh();
    int dim = pmesh->Dimension();
    int sdim = pmesh->SpaceDimension();
    
    VectorFunctionCoefficient epsilon(dim, epsilon_func);
    VectorFunctionCoefficient epsilonT(epsilon);  // transpose of epsilon
    VectorFunctionCoefficient epsilon2(dim,epsilon2_func);
    ConstantCoefficient pos(omega);
    ConstantCoefficient sigma(omega*omega);
    ScalarVectorProductCoefficient coeff(pos,epsilon);
    ScalarVectorProductCoefficient coeffT(pos,epsilonT);
    ScalarVectorProductCoefficient coeff2(sigma,epsilon2);

    bM = new ParBilinearForm(fespace);
    bM->AddDomainIntegrator(new VectorFEMassIntegrator());
    bM->Assemble();
    bM->Finalize();

    bM_eps = new ParBilinearForm(fespace);
    bM_eps->AddDomainIntegrator(new VectorFEMassIntegrator(epsilonT));
    bM_eps->Assemble();
    bM_eps->Finalize();

    bM_curl = new ParMixedBilinearForm(fespace,fespace);
    bM_curl->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator()); 
    bM_curl->Assemble();
    bM_curl->Finalize();
    
    Array<int> ess_tdof_list;  // empty

    bM->FormSystemMatrix(ess_tdof_list, M);
    bM_eps->FormSystemMatrix(ess_tdof_list, M_eps);
    bM_curl->FormColSystemMatrix(ess_tdof_list, M_curl);

    M_inv.SetAbsTol(1.0e-12);
    M_inv.SetRelTol(1.0e-12);
    M_inv.SetMaxIter(100);
    M_inv.SetOperator(M);
    M_inv.SetPrintLevel(0);

    block_trueOffsets.SetSize(5);
    
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = n;
    block_trueOffsets[2] = n;
    block_trueOffsets[3] = n;
    block_trueOffsets[4] = n;
    block_trueOffsets.PartialSum();

    trueRhs = new BlockVector(block_trueOffsets);
    trueSol = new BlockVector(block_trueOffsets);

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

    ParBilinearForm *a_EE = new ParBilinearForm(fespace);
    a_EE->AddDomainIntegrator(new CurlCurlIntegrator());
    a_EE->AddDomainIntegrator(new VectorFEMassIntegrator(coeff2));
    a_EE->AddBoundaryIntegrator(new VectorFEMassIntegrator());
    a_EE->Assemble();
    a_EE->Finalize();
    HypreParMatrix *A_EE = new HypreParMatrix;
    a_EE->FormSystemMatrix(ess_tdof_list, *A_EE);
  
    ParBilinearForm *a_HH = new ParBilinearForm(fespace);
    a_HH->AddDomainIntegrator(new CurlCurlIntegrator());
    a_HH->AddDomainIntegrator(new VectorFEMassIntegrator(sigma));
    a_HH->AddBoundaryIntegrator(new VectorFEMassIntegrator());
    a_HH->Assemble();
    a_HH->Finalize();
    HypreParMatrix *A_HH = new HypreParMatrix;
    a_HH->FormSystemMatrix(ess_tdof_list, *A_HH);

    ParBilinearForm *a_tang = new ParBilinearForm(fespace);
    a_tang->AddBoundaryIntegrator(new VectorFEBoundaryTangentIntegrator(1.0));
    a_tang->Assemble();
    a_tang->Finalize();
    OperatorHandle A_tang_ptr;
    a_tang->FormSystemMatrix(ess_tdof_list, A_tang_ptr);
    HypreParMatrix *A_tang = A_tang_ptr.As<HypreParMatrix>();
    
    // (k curl u, eps v) + (k u, curl v)
    ParMixedBilinearForm *a_mix1 = new ParMixedBilinearForm(fespace,fespace);
    a_mix1->AddDomainIntegrator(new MixedVectorCurlIntegrator(coeffT));
    a_mix1->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(pos));
    a_mix1->Assemble();
    a_mix1->Finalize();
    HypreParMatrix *A_mix1 = new HypreParMatrix;
    a_mix1->FormColSystemMatrix(ess_tdof_list, *A_mix1);
    
    // (k curl u, v) + (k eps u, curl v)
    ParMixedBilinearForm *a_mix2 = new ParMixedBilinearForm(fespace,fespace);
    a_mix2->AddDomainIntegrator(new MixedVectorCurlIntegrator(pos));
    a_mix2->AddDomainIntegrator(new MixedVectorWeakCurlIntegrator(coeff));
    a_mix2->Assemble();
    a_mix2->Finalize();
    HypreParMatrix *A_mix2 = new HypreParMatrix;
    a_mix2->FormColSystemMatrix(ess_tdof_list, *A_mix2);
    
    BlockOperator *LS_Maxwellop = new BlockOperator(block_trueOffsets);
    const int numBlocks = 4;

    LS_Maxwellop->SetBlock(0, 0, A_EE);
    LS_Maxwellop->SetBlock(1, 0, A_mix2, -1.0); // no bc
    LS_Maxwellop->SetBlock(3, 0, A_tang, -1.0);
    LS_Maxwellop->SetBlock(0, 1, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(1, 1, A_HH);
    LS_Maxwellop->SetBlock(2, 1, A_tang, -1.0);  // other rotation
    LS_Maxwellop->SetBlock(1, 2, A_tang);
    LS_Maxwellop->SetBlock(2, 2, A_EE);
    LS_Maxwellop->SetBlock(3, 2, A_mix2, -1.0);  // no bc
    LS_Maxwellop->SetBlock(0, 3, A_tang);  // other rotation
    LS_Maxwellop->SetBlock(2, 3, A_mix1, -1.0);  // no bc
    LS_Maxwellop->SetBlock(3, 3, A_HH);
    
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

    LSpcg.SetAbsTol(1.0e-12);
    LSpcg.SetRelTol(1.0e-8);
    LSpcg.SetMaxIter(2000);
    LSpcg.SetOperator(*LS_Maxwellop);
    LSpcg.SetPrintLevel(1);
    
    BlockMGSolver * precMG = NULL;

#ifdef DIRECT_SOLVER
    std::vector<std::vector<int> > blockProcOffsets(numBlocks);
    std::vector<std::vector<int> > all_block_num_loc_rows(numBlocks);
    Array2D<SparseMatrix*> Asp;
    Asp.SetSize(numBlocks,numBlocks);

    {
      int nprocs, rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

      std::vector<int> allnumrows(nprocs);
      const int blockNumRows = n;
      MPI_Allgather(&blockNumRows, 1, MPI_INT, allnumrows.data(), 1, MPI_INT, MPI_COMM_WORLD);

      for (int b=0; b<numBlocks; ++b)
	{
	  blockProcOffsets[b].resize(nprocs);
	  all_block_num_loc_rows[b].resize(nprocs);

	  for (int j=0; j<numBlocks; ++j)
	    Asp(b,j) = NULL;
	}
     
      blockProcOffsets[0][0] = 0;
      for (int i=0; i<nprocs-1; ++i)
	blockProcOffsets[0][i+1] = blockProcOffsets[0][i] + allnumrows[i];

      for (int i=0; i<nprocs; ++i)
	{
	  for (int b=0; b<numBlocks; ++b)
	    all_block_num_loc_rows[b][i] = allnumrows[i];
	  
	  for (int b=1; b<numBlocks; ++b)
	    blockProcOffsets[b][i] = blockProcOffsets[0][i];
	}
    }
   
    LSH = CreateHypreParMatrixFromBlocks(MPI_COMM_WORLD, block_trueOffsets, blockA, Asp,
					 blockAcoef, blockProcOffsets, all_block_num_loc_rows);

    invLSH = CreateStrumpackSolver(new STRUMPACKRowLocMatrix(*LSH), MPI_COMM_WORLD);
#else
    precMG = new BlockMGSolver(LS_Maxwellop->Height(), LS_Maxwellop->Width(), blockA, blockAcoef, P);
    LSpcg.SetPreconditioner(*precMG);
#endif
  }
  
  void SetOperator(const Operator &op) { }

  void Mult(const Vector &x, Vector &y) const
  {
    // Solve (curl E, curl v) - k^2 (eps E, v) + ik <pi(u), pi(v)> = (x, v), with no BC,
    // where x is complex, using FOSLS. This is the Galerkin discretization of
    // curl curl u - k^2 eps u = x, with ik n x u x n - n x curl u = 0 on the boundary.

    MFEM_VERIFY(x.Size() == 2*n, "");

    (*trueRhs) = 0.0;
    
    for (int i=0; i<n; ++i)
      z[i] = x[i];  // Set z = x_Re
    
    M_inv.Mult(z, Minv_x);
    M_eps.Mult(Minv_x, z);

    trueRhs->GetBlock(0) -= z;

    M_curl.Mult(Minv_x, z);
    z *= 1.0 / omega;
    
    trueRhs->GetBlock(1) = z;

    for (int i=0; i<n; ++i)
      z[i] = x[n + i];  // Set z = x_Im
    
    M_inv.Mult(z, Minv_x);
    M_eps.Mult(Minv_x, z);

    trueRhs->GetBlock(2) -= z;

    M_curl.Mult(Minv_x, z);
    z *= 1.0 / omega;

    trueRhs->GetBlock(3) += z;
    
#ifdef DIRECT_SOLVER
    invLSH->Mult(*trueRhs, *trueSol);
#else
    LSpcg.Mult(*trueRhs, *trueSol);
#endif
    
    for (int i=0; i<n; ++i)
      y[i] = trueSol->GetBlock(0)[i];  // Set y_Re = E_Re

    for (int i=0; i<n; ++i)
      y[n + i] = trueSol->GetBlock(2)[i];  // Set y_Im = E_Im
  }

  void ComplexFOSLSTest()
  {
    Array<int> block_offsets;

    block_offsets.SetSize(5);
    
    block_offsets[0] = 0;
    block_offsets[1] = nfull;
    block_offsets[2] = nfull;
    block_offsets[3] = nfull;
    block_offsets[4] = nfull;
    block_offsets.PartialSum();

    BlockVector rhs(block_offsets);
    BlockVector rhsIm(block_offsets);

    rhs = 0.0;
    rhsIm = 0.0;

    const double ci = 3.3;

    // Exact complex solution: E = Er + i Ei = Epw + ci i Epw, where Epw is E_exact.
    
    // Set up the linear form with the real part Fr only.
    const int sdim = 3;
    VectorFunctionCoefficient Eex(sdim, E_exact);
    VectorFunctionCoefficient Hex(sdim, H_exact);
    ConstantCoefficient negOne(-1.0);
    VectorFunctionCoefficient mEex(sdim, E_exact, &negOne);
    VectorFunctionCoefficient mHex(sdim, H_exact, &negOne);

    ConstantCoefficient neg(-omega);
    ConstantCoefficient pos(omega);
    VectorFunctionCoefficient f_H(3,f_exact_H);  // f / omega
    ScalarVectorProductCoefficient sf_H(neg,f_H);
    ScalarVectorProductCoefficient spf_H(pos,f_H);
    ScalarVectorProductCoefficient mf_H(negOne,f_H);

    VectorFunctionCoefficient epsilon(3, epsilon_func);
    VectorFunctionCoefficient epsilonT(3, epsilon_func);  // transpose of epsilon
    MatrixFunctionCoefficient epsilonTmat(3, epsilon_func_mat);  // transpose of epsilon

    MatVecCoefficient epsT_spf_H(epsilonTmat, spf_H);
    MatVecCoefficient epsT_sf_H(epsilonTmat, sf_H);

    ParLinearForm *b_E = new ParLinearForm;
    b_E->Update(fespace, rhs.GetBlock(0), 0);
    b_E->AddDomainIntegrator(new VectorFEDomainLFIntegrator(epsT_sf_H)); // (k^{-1} Fr, -k eps Qr)
    b_E->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Eex));  // <n x E_Re x n, n x Q_Re x n>
    b_E->Assemble();

    ParLinearForm *b_H = new ParLinearForm;
    b_H->Update(fespace, rhs.GetBlock(1), 0);
    b_H->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H));  // (k^{-1} Fr, curl Rr)
    b_H->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Hex));  // <n x H_Re, n x R_Re>
    b_H->Assemble();

    ParLinearForm *b_E_Im = new ParLinearForm;
    b_E_Im->Update(fespace, rhs.GetBlock(2), 0);
    b_E_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(mHex));  // <n x H_Re, n x Q_Im x n>
    b_E_Im->Assemble();

    ParLinearForm *b_H_Im = new ParLinearForm;
    b_H_Im->Update(fespace, rhs.GetBlock(3), 0);
    b_H_Im->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(mEex));  // -<n x E_Re x n, n x R_Im>
    b_H_Im->Assemble();

    // Add the imaginary part Fi.

    ParLinearForm *b_Ei = new ParLinearForm;
    b_Ei->Update(fespace, rhsIm.GetBlock(0), 0);
    b_Ei->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(Hex));  // -<n x H_Im, n x Q_Re x n>
    b_Ei->Assemble();

    ParLinearForm *b_Hi = new ParLinearForm;
    b_Hi->Update(fespace, rhsIm.GetBlock(1), 0);
    b_Hi->AddBoundaryIntegrator(new VectorFEBoundaryTangentLFIntegrator(Eex));  // <n x E_Im x n, n x R_Re>
    b_Hi->Assemble();
    
    ParLinearForm *b_Ei_Im = new ParLinearForm;
    b_Ei_Im->Update(fespace, rhsIm.GetBlock(2), 0);
    b_Ei_Im->AddDomainIntegrator(new VectorFEDomainLFIntegrator(epsT_sf_H)); // -(k^{-1} Fi, k eps Qi)
    b_Ei_Im->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Eex));  // <n x E_Im x n, n x Q_Im x n>
    b_Ei_Im->Assemble();

    ParLinearForm *b_Hi_Im = new ParLinearForm;
    b_Hi_Im->Update(fespace, rhsIm.GetBlock(3), 0);
    b_Hi_Im->AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(f_H)); // (k^{-1} Fi, curl Ri)
    b_Hi_Im->AddBoundaryIntegrator(new VectorFEDomainLFIntegrator(Hex));  // <n x H_Im, n x R_Im>
    b_Hi_Im->Assemble();

    rhsIm *= ci;
    rhs += rhsIm;

    for (int i=0; i<4; ++i)
      fespace->GetProlongationMatrix()->MultTranspose(rhs.GetBlock(i), trueRhs->GetBlock(i));

#ifdef DIRECT_SOLVER
    invLSH->Mult(*trueRhs, *trueSol);
#else
    LSpcg.Mult(*trueRhs, *trueSol);
#endif
    
    // Check error
    ParGridFunction E_gf(fespace);
    int order = 2;
    int order_quad = std::max(2, 2*order+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
      {
	irs[i] = &(IntRules.Get(i, order_quad));
      }
    
    ParMesh *pmesh = fespace->GetParMesh();
    
    // Check error of real part
    
    E_gf.SetFromTrueDofs(trueSol->GetBlock(0));
    double Error_E = E_gf.ComputeL2Error(Eex, irs);
    double norm_E = ComputeGlobalLpNorm(2, Eex, *pmesh, irs);

    cout << myid << ": real error " << Error_E << " relative to " << norm_E << endl;

    {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      E_gf.Save(sol_ofs);
    }
    
    // Check error of imaginary part
    E_gf.SetFromTrueDofs(trueSol->GetBlock(2));
    E_gf *= (1.0 / ci);
    Error_E = E_gf.ComputeL2Error(Eex, irs);

    cout << myid << ": imag error " << Error_E << " relative to " << norm_E << endl;
  }
  
private:

  BlockVector *trueRhs, *trueSol;

  Array<int> block_trueOffsets;

  ParBilinearForm *bM, *bM_eps;
  ParMixedBilinearForm *bM_curl;
  
  HypreParMatrix M, M_eps;
  HypreParMatrix M_curl;

  CGSolver M_inv;

  const int n;
  const int nfull;
  const int myid;
  
  mutable Vector z, Minv_x;

  CGSolver LSpcg;
  
  STRUMPACKSolver *invLSH;
  HypreParMatrix *LSH;

  ParFiniteElementSpace *fespace;
};

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
  
  FOSLSSolver fosls(fespace, P, myid);
  
  fosls.ComplexFOSLSTest();
    
  for (auto p: ParMeshes) delete p;
  for (auto p: fespaces)  delete p;
  for (auto p: P)  delete p;
  ParMeshes.clear();
  fespaces.clear();
  P.clear();
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
      // shift to avoid singularity
      double x0 = x + 0.1;
      double x1 = y + 0.1;
      double x2 = z + 0.1;
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
      
      // not used
      curl2E[0] = 0.0;
      curl2E[1] = 0.0;  
      curl2E[2] = 0.0;
    }
}

void epsilon_func(const Vector &x, Vector &M)
{
  M.SetSize(3);

  M = 1.0;
  if (sol == 4)
    {
      M[2] = 4.0*x(0)-1.0;
    }
}

void epsilon2_func(const Vector &x, Vector &M)
{
  M.SetSize(3);

  M = 1.0;
  if (sol == 4)
    {
      M[2] = (4.0*x(0)-1.0) * (4.0*x(0)-1.0);
    }
}

void epsilon_func_mat(const Vector &x, DenseMatrix &M)
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
    }
}
