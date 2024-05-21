//                                MFEM Maxwell example (with imaginary mass coeff)
//
//                   ∇×(1/μ ∇×E) - (ω^2 ϵ + i ω σ) E = J ,   in Ω   (by default ϵ=0)
//                                               E×n = E_0, on ∂Ω
//
// Compile with: pmaxwell
//
// Sample runs:


// mpirun -np 8 ./pmaxwell -o 1 -sref 0  -m ../data/inline-hex.mesh -rnum 3.0  -pref 4 -sigma 1.0 -no-herm -vis
/*
  Ref |    Dofs    |    ω    | H(curl) Error |  Rate  | Solv it |
-----------------------------------------------------------------
    0 |        300 |  6.0 π  |     2.697e+01 |   0.00 |  28 (20)|
    1 |       1944 |  6.0 π  |     2.183e+01 |  -0.34 |  28 (19)|
    2 |      13872 |  6.0 π  |     1.227e+01 |  -0.88 |  30 (19)|
    3 |     104544 |  6.0 π  |     6.341e+00 |  -0.98 |  36 (19)|
    4 |     811200 |  6.0 π  |     3.197e+00 |  -1.00 |  46 (19)|
*/

// mpirun -np 8 ./pmaxwell -o 2 -sref 0  -m ../data/inline-hex.mesh -rnum 3.0  -pref 3 -sigma 1.0 -vis
/*
  Ref |    Dofs    |    ω    | H(curl) Error |  Rate  | Solv it |
-----------------------------------------------------------------
    0 |       1944 |  6.0 π  |     2.121e+01 |   0.00 |  28 (24)|
    1 |      13872 |  6.0 π  |     7.291e+00 |  -1.63 |  34 (19)|
    2 |     104544 |  6.0 π  |     1.926e+00 |  -1.98 |  48 (19)|
    3 |     811200 |  6.0 π  |     4.862e-01 |  -2.02 |  60 (19)|
*/


// mpirun -np 8 ./pmaxwell -o 3 -sref 0  -m ../data/inline-hex.mesh -rnum 3.0  -pref 2 -sigma 1.0 -vis
/*
  Ref |    Dofs    |    ω    | H(curl) Error |  Rate  | Solv it |
-----------------------------------------------------------------
    0 |       6084 |  6.0 π  |     9.797e+00 |   0.00 |  36 (22)|
    1 |      45000 |  6.0 π  |     1.475e+00 |  -2.84 |  56 (19)|
    2 |     345744 |  6.0 π  |     1.918e-01 |  -3.00 |  78 (19)|
*/
// Note: (*) indicates a priconditioner with exact inverse of the diagonal blocks

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void  rhs_func_r(const Vector &x, Vector & J_r);
void  rhs_func_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);

void maxwell_solution(const Vector & X,
                      std::vector<complex<real_t>> &E);

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<real_t>> &curlE);

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<real_t>> &curlcurlE);

int dim;
int dimc;
real_t omega;
real_t mu = 1.0;
real_t epsilon = 0.0;
real_t sigma = 0.1;

complex<real_t> zi = complex<real_t>(0., 1.);

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   const char *mesh_file = "../data/inline-hex.mesh";

   int order = 1;
   bool visualization = false;
   real_t rnum=1.0;
   int sr = 0;
   int pr = 0;
   bool paraview = false;
   bool mumps_solver = false;
   bool herm_conv = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&sigma, "-sigma", "--conductivity",
                  "Conductivity");
   args.AddOption(&sr, "-sref", "--serial_ref",
                  "Number of serial refinements.");
   args.AddOption(&pr, "-pref", "--parallel_ref",
                  "Number of parallel refinements.");
   args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
                  "--no-hermitian", "Use convention for Hermitian operators.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.AddOption(&mumps_solver, "-mumps-solver", "--mumps-solver",
                  "-no-mumps-solver",
                  "--no-mumps-solver",
                  "Enable or disable mumps solver");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   ComplexOperator::Convention conv =
      herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;


   socketstream E_out_r;
   socketstream E_out_i;

   omega = real_t(2.0 * M_PI)*rnum;

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   dimc = (dim == 3) ? 3 : 1;

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   FiniteElementCollection *fec = new ND_FECollection(order, dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh, fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient muinv(1./mu);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient eps(epsilon);

   ConstantCoefficient negomeg2eps(-omega*omega*epsilon);
   ConstantCoefficient omegsigma(omega*sigma);

   VectorFunctionCoefficient Jr(dim,rhs_func_r);
   VectorFunctionCoefficient Ji(dim,rhs_func_i);

   VectorFunctionCoefficient Er(dim,E_exact_r);
   VectorFunctionCoefficient Ei(dim,E_exact_i);

   VectorFunctionCoefficient CurlEr(dim,curlE_exact_r);
   VectorFunctionCoefficient CurlEi(dim,curlE_exact_i);

   ParComplexLinearForm *b = new ParComplexLinearForm(E_fes, conv);
   b->Vector::operator=(0.0);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(Jr),
                          new VectorFEDomainLFIntegrator(Ji));


   ParSesquilinearForm *a = new ParSesquilinearForm(E_fes, conv);
   a->AddDomainIntegrator(new CurlCurlIntegrator(muinv),nullptr);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(negomeg2eps),
                          new VectorFEMassIntegrator(omegsigma));



   ParBilinearForm *prec = new ParBilinearForm(E_fes);
   prec->AddDomainIntegrator(new CurlCurlIntegrator(muinv));
   prec->AddDomainIntegrator(new VectorFEMassIntegrator(omegsigma));


   ParComplexGridFunction E_gf(E_fes);
   E_gf.real() = 0.0;
   E_gf.imag() = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(mesh_file, &pmesh);
      paraview_dc->SetPrefixPath("ParaView");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_gf.real());
      paraview_dc->RegisterField("E_i",&E_gf.imag());
   }

   if (Mpi::Root())
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "    ω    |"
                << " H(curl) Error |"
                << "  Rate  |"
                << " Solv it |" << endl;
      std::cout << std::string(65,'-')
                << endl;
   }

   real_t err0 = 0.;
   int dof0;

   for (int it = 0; it<=pr; it++)
   {
      b->Assemble();
      a->Assemble();
      prec->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;

      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      E_gf.real() = 0.0;
      E_gf.imag() = 0.0;
      E_gf.ProjectBdrCoefficientTangent(Er,Ei, ess_bdr);

      OperatorPtr Ah;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, E_gf, *b, Ah, X, B);

      HypreParMatrix M;
      prec->FormSystemMatrix(ess_tdof_list, M);

      HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = E_fes->TrueVSize();
      offsets[2] = E_fes->TrueVSize();
      offsets.PartialSum();
      BlockDiagonalPreconditioner BlockPrec(offsets);

      std::unique_ptr<Operator> pc_r;
      std::unique_ptr<Operator> pc_i;
      int s = (conv == ComplexOperator::HERMITIAN) ? -1 : 1;

#ifdef MFEM_USE_MUMPS
      if (mumps_solver)
      {
         pc_r.reset(new MUMPSSolver(M));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }
#else
      mumps_solver = false;
#endif
      if (!mumps_solver)
      {
         pc_r.reset(new HypreAMS(M,E_fes));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }

      BlockPrec.SetDiagonalBlock(0,pc_r.get());
      BlockPrec.SetDiagonalBlock(1,pc_i.get());

      std::unique_ptr<IterativeSolver> solver;
      if (conv == ComplexOperator::HERMITIAN)
      {
         solver.reset(new FGMRESSolver(MPI_COMM_WORLD));
      }
      else
      {
         solver.reset(new MINRESSolver(MPI_COMM_WORLD));
      }
      solver.get()->SetRelTol(1e-12);
      solver.get()->SetMaxIter(2000);
      solver.get()->SetPrintLevel(0);
      solver.get()->SetPreconditioner(BlockPrec);
      solver.get()->SetOperator(*A);
      solver.get()->Mult(B, X);
      int num_iter = solver.get()->GetNumIterations();

      a->RecoverFEMSolution(X, *b, E_gf);

      real_t err_r = E_gf.real().ComputeHCurlError(&Er,&CurlEr);
      real_t err_i = E_gf.imag().ComputeHCurlError(&Ei,&CurlEi);

      real_t totalerr = std::sqrt(err_r*err_r + err_i*err_i);

      int dofs = E_fes->GlobalTrueVSize();

      real_t rate_err = (it) ? dim*log(err0/totalerr)/log((real_t)dof0/dofs) : 0.0;

      err0 = totalerr;
      dof0 = dofs;

      if (Mpi::Root())
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2.0*rnum << " π  | "
                   << std::setprecision(3);
         std::cout << std::setw(13) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | "
                   << std::setw(7) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         // Define visualization keys for GLVis (see GLVis documentation)
         string keys;
         keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";

         char vishost[] = "localhost";
         int visport = 19916;

         {
            socketstream sol_sock_re(vishost, visport);
            sol_sock_re.precision(8);
            sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                        << "solution\n" << pmesh << E_gf.real() << keys
                        << "window_title 'Solution real part'" << flush;
         }

         {
            socketstream sol_sock_im(vishost, visport);
            sol_sock_im.precision(8);
            sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                        << "solution\n" << pmesh << E_gf.imag() << keys
                        << "window_title 'Solution imag part'" << flush;
         }
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((real_t)it);
         paraview_dc->Save();
      }

      pmesh.UniformRefinement();

      E_fes->Update();
      E_gf.Update();
      a->Update();
      b->Update();
      prec->Update();
   }
   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete b;
   delete E_fes;
   delete fec;
   return 0;
}

void maxwell_solution(const Vector & X, std::vector<complex<real_t>> &E)
{
   E.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
   E[0] = exp(zi * omega * (X.Sum()));
}

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<real_t>> &curlE)
{
   curlE.resize(dimc);
   for (int i = 0; i < dimc; ++i)
   {
      curlE[i] = 0.0;
   }
   std::complex<real_t> pw = exp(zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlE[0] = 0.0;
      curlE[1] = zi * omega * pw;
      curlE[2] = -zi * omega * pw;
   }
   else
   {
      curlE[0] = -zi * omega * pw;
   }
}

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<real_t>> &curlcurlE)
{
   curlcurlE.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      curlcurlE[i] = 0.0;;
   }
   std::complex<real_t> pw = exp(zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlcurlE[0] = 2_r * omega * omega * pw;
      curlcurlE[1] = - omega * omega * pw;
      curlcurlE[2] = - omega * omega * pw;
   }
   else
   {
      curlcurlE[0] = omega * omega * pw;
      curlcurlE[1] = -omega * omega * pw;
   }
}

void E_exact_r(const Vector &x, Vector & E_r)
{
   std::vector<std::complex<real_t>> E;
   maxwell_solution(x,E);
   E_r.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_r[i]= E[i].real();
   }
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   std::vector<std::complex<real_t>> E;
   maxwell_solution(x, E);
   E_i.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_i[i]= E[i].imag();
   }
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   std::vector<std::complex<real_t>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_r.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_r[i]= curlE[i].real();
   }
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   std::vector<std::complex<real_t>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_i.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_i[i]= curlE[i].imag();
   }
}

void  rhs_func_r(const Vector &x, Vector & J_r)
{
   std::vector<std::complex<real_t>> curlcurlE;
   std::vector<std::complex<real_t>> E;
   maxwell_solution(x,E);
   maxwell_solution_curlcurl(x, curlcurlE);
   J_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      const complex<real_t> tmp = 1_r/mu*curlcurlE[i]
                                  + zi * omega *(sigma+zi*omega*epsilon) *E[i];
      J_r(i) = tmp.real();
   }
}

void  rhs_func_i(const Vector &x, Vector & J_i)
{
   std::vector<std::complex<real_t>> curlcurlE;
   std::vector<std::complex<real_t>> E;
   maxwell_solution(x,E);
   maxwell_solution_curlcurl(x, curlcurlE);
   J_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      const complex<real_t> tmp = 1_r/mu*curlcurlE[i]
                                  + zi * omega *(sigma+zi*omega*epsilon) *E[i];
      J_i(i) = tmp.imag();
   }
}

