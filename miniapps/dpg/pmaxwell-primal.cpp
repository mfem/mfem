//                   MFEM Ultraweak DPG Maxwell parallel example
//
// Compile with: make pmaxwell
//
// sample run
// mpirun -np 4 pmaxwell -m ../../data/star.mesh -o 2 -sref 0 -pref 3 -rnum 1.0
// mpirun -np 4 pmaxwell -m ../../data/inline-quad.mesh -o 3 -sref 0 -pref 3 -rnum 4.8 -sc
// mpirun -np 4 pmaxwell -m ../../data/inline-hex.mesh -o 2 -sref 0 -pref 1 -rnum 0.8 -sc

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Maxwell problem

//      ∇×(1/μ ∇×E) - ω² ϵ E = J ,   in Ω
//                       E×n = E₀ , on ∂Ω

// It solves the following kinds of problems
// 1) Known exact solutions with error convergence rates
//    a) A manufactured solution problem where E is a plane wave

#include "mfem.hpp"
#include "util/pcomplexweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

void E_exact_r(const Vector &x, Vector & E_r);
void E_exact_i(const Vector &x, Vector & E_i);

void  rhs_func_r(const Vector &x, Vector & J_r);
void  rhs_func_i(const Vector &x, Vector & J_i);

void curlE_exact_r(const Vector &x, Vector &curlE_r);
void curlE_exact_i(const Vector &x, Vector &curlE_i);

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r);
void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i);

void maxwell_solution(const Vector & X,
                      std::vector<complex<double>> &E);

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE);

void maxwell_solution_curlcurl(const Vector & X,
                               std::vector<complex<double>> &curlcurlE);

int dim;
int dimc;
double omega;
double mu = 1.0;
double epsilon = 1.0;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   double rnum=1.0;
   bool static_cond = false;
   int sr = 0;
   int pr = 1;
   bool visualization = true;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&rnum, "-rnum", "--number-of-wavelengths",
                  "Number of wavelengths");
   args.AddOption(&mu, "-mu", "--permeability",
                  "Permeability of free space (or 1/(spring constant)).");
   args.AddOption(&epsilon, "-eps", "--permittivity",
                  "Permittivity of free space (or mass constant).");
   args.AddOption(&delta_order, "-do", "--delta-order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sr, "-sref", "--serial-ref",
                  "Number of parallel refinements.");
   args.AddOption(&pr, "-pref", "--parallel-ref",
                  "Number of parallel refinements.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview", "-no-paraview",
                  "--no-paraview",
                  "Enable or disable ParaView visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   omega = 2.*M_PI*rnum;

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   dimc = (dim == 3) ? 3 : 1;

   for (int i = 0; i<sr; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   enum TrialSpace { E_space = 0, hatE_space = 1 };
   enum TestSpace  { F_space = 0 };

   // H(curl) space for E
   FiniteElementCollection *E_fec = new ND_FECollection(order,dim);
   ParFiniteElementSpace *E_fes = new ParFiniteElementSpace(&pmesh,E_fec);

   // H^-1/2 (curl) space for Ê
   FiniteElementCollection * hatE_fec = nullptr;
   int test_order = order+delta_order;
   if (dim == 2)
   {
      hatE_fec = new H1_Trace_FECollection(order,dim);
   }
   else
   {
      hatE_fec = new ND_Trace_FECollection(order,dim);
   }
   ParFiniteElementSpace *hatE_fes = new ParFiniteElementSpace(&pmesh,hatE_fec);
   FiniteElementCollection * F_fec = new ND_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;
   trial_fes.Append(E_fes);
   trial_fes.Append(hatE_fes);
   test_fec.Append(F_fec);

   // Bilinear form coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient negepsomeg2(-epsilon*omega*omega);
   ConstantCoefficient muinv(1./mu);
   // for the 2D case

   ParComplexDPGWeakForm * a = new ParComplexDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(); // needed for AMR

   // (∇ × E,∇ × F)
   a->AddTrialIntegrator(new CurlCurlIntegrator(muinv), nullptr,
                         TrialSpace::E_space, TestSpace::F_space);
   // -ω² ϵ (E , F)
   a->AddTrialIntegrator(new VectorFEMassIntegrator(negepsomeg2), nullptr,
                         TrialSpace::E_space,TestSpace::F_space);
   // < n×Ê ,F>
   if (dim == 3)
   {
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::F_space);
   }
   else
   {
      a->AddTrialIntegrator(new TangentTraceIntegrator,nullptr,
                            TrialSpace::hatE_space, TestSpace::F_space);
   }
   // test integrators
   // (∇×F ,∇×δF)
   a->AddTestIntegrator(new CurlCurlIntegrator(one),nullptr,
                        TestSpace::F_space,TestSpace::F_space);
   // (F,δF)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,
                        TestSpace::F_space,TestSpace::F_space);

   // RHS
   VectorFunctionCoefficient f_rhs_r(dim,rhs_func_r);
   VectorFunctionCoefficient f_rhs_i(dim,rhs_func_i);
   a->AddDomainLFIntegrator(new VectorFEDomainLFIntegrator(f_rhs_r),
                            new VectorFEDomainLFIntegrator(f_rhs_i),
                            TestSpace::F_space);

   VectorFunctionCoefficient Eex_r(dim,E_exact_r);
   VectorFunctionCoefficient Eex_i(dim,E_exact_i);

   socketstream E_out_r;
   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "    ω    |" ;
      std::cout  << "  H(curl) Error  |"
                 << "  Rate  |" ;
      std::cout << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string(87,'-')
                << endl;
   }

   double res0 = 0.;
   double err0 = 0.;
   int dof0;
   ParGridFunction E_r, E_i;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection("Plane", &pmesh);
      paraview_dc->SetPrefixPath("ParaViewPrimal/Maxwell");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("E_r",&E_r);
      paraview_dc->RegisterField("E_i",&E_i);
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pr; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         E_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = E_fes->GetVSize();
      offsets[2] = hatE_fes->GetVSize();
      offsets.PartialSum();

      Vector x(2*offsets.Last());
      x = 0.;

      ParGridFunction E_gf_r(E_fes, x, offsets[0]);
      ParGridFunction E_gf_i(E_fes, x, offsets.Last() + offsets[0]);
      E_gf_r.ProjectBdrCoefficientTangent(Eex_r, ess_bdr);
      E_gf_i.ProjectBdrCoefficientTangent(Eex_i, ess_bdr);

      OperatorPtr Ah;
      Vector X,B;
      a->FormLinearSystem(ess_tdof_list,x,Ah, X,B);

      ComplexOperator * Ahc = Ah.As<ComplexOperator>();

      BlockOperator * BlockA_r = dynamic_cast<BlockOperator *>(&Ahc->real());
      BlockOperator * BlockA_i = dynamic_cast<BlockOperator *>(&Ahc->imag());

      int num_blocks = BlockA_r->NumRowBlocks();
      Array<int> tdof_offsets(2*num_blocks+1);

      tdof_offsets[0] = 0;
      for (int i=0; i<num_blocks; i++)
      {
         const int h = BlockA_r->GetBlock(i,i).Height();
         tdof_offsets[i+1] = h;
         tdof_offsets[num_blocks+i+1] = h;
      }
      tdof_offsets.PartialSum();

      BlockOperator blockA(tdof_offsets);
      for (int i = 0; i<num_blocks; i++)
      {
         for (int j = 0; j<num_blocks; j++)
         {
            blockA.SetBlock(i,j,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i,j+num_blocks,&BlockA_i->GetBlock(i,j), -1.0);
            blockA.SetBlock(i+num_blocks,j+num_blocks,&BlockA_r->GetBlock(i,j));
            blockA.SetBlock(i+num_blocks,j,&BlockA_i->GetBlock(i,j));
         }
      }

      X = 0.;
      BlockDiagonalPreconditioner M(tdof_offsets);

      ParFiniteElementSpace *ams_fes = nullptr;
      if (static_cond)
      {
         ams_fes = new ParFiniteElementSpace(&pmesh,
                                             E_fes->FEColl()->GetTraceCollection());
      }
      HypreAMS * solver_E = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(0,0),
                                         (static_cond) ? ams_fes : E_fes);
      solver_E->SetPrintLevel(0);

      HypreSolver * solver_hatE = nullptr;

      if (dim == 2)
      {
         solver_hatE = new HypreBoomerAMG((HypreParMatrix &)BlockA_r->GetBlock(1,1));
         dynamic_cast<HypreBoomerAMG*>(solver_hatE)->SetPrintLevel(0);
      }
      else
      {
         solver_hatE = new HypreAMS((HypreParMatrix &)BlockA_r->GetBlock(1,1),
                                    hatE_fes);
         dynamic_cast<HypreAMS*>(solver_hatE)->SetPrintLevel(0);
      }

      M.SetDiagonalBlock(0,solver_E);
      M.SetDiagonalBlock(1,solver_hatE);
      M.SetDiagonalBlock(num_blocks,solver_E);
      M.SetDiagonalBlock(num_blocks+1,solver_hatE);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(10000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(blockA);
      cg.Mult(B, X);
      delete ams_fes;
      for (int i = 0; i<num_blocks; i++)
      {
         delete &M.GetDiagonalBlock(i);
      }

      int num_iter = cg.GetNumIterations();

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();
      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;
      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      E_r.MakeRef(E_fes,x, 0);
      E_i.MakeRef(E_fes,x, offsets.Last());

      int dofs = 0;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         dofs += trial_fes[i]->GlobalTrueVSize();
      }

      double HcurlError = 0.0;
      double rate_err = 0.0;

      VectorFunctionCoefficient curlEex_r(dim,curlE_exact_r);
      VectorFunctionCoefficient curlEex_i(dim,curlE_exact_i);

      double E_err_r = E_r.ComputeHCurlError(&Eex_r,&curlEex_r);
      double E_err_i = E_i.ComputeHCurlError(&Eex_i,&curlEex_i);

      HcurlError = sqrt(  E_err_r*E_err_r + E_err_i*E_err_i);
      rate_err = (it) ? dim*log(err0/HcurlError)/log((double)dof0/dofs) : 0.0;
      err0 = HcurlError;

      double rate_res =
         (it) ? dim*log(res0/globalresidual)/log((double)dof0/dofs) : 0.0;

      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(1) << std::fixed
                   << std::setw(4) <<  2.0*rnum << " π  | "
                   << std::setprecision(3);
         std::cout << std::setw(15) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | " ;
         std::cout << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << num_iter << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcml\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;
         VisualizeField(E_out_r,vishost, visport, E_r,
                        "Numerical Electric field (real part)", 0, 0, 500, 500, keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pr)
      {
         break;
      }

      pmesh.UniformRefinement();

      for (int i =0; i<trial_fes.Size(); i++)
      {
         trial_fes[i]->Update(false);
      }
      a->Update();
   }

   if (paraview)
   {
      delete paraview_dc;
   }

   delete a;
   delete F_fec;
   delete hatE_fes;
   delete hatE_fec;
   delete E_fec;
   delete E_fes;

   return 0;
}

void E_exact_r(const Vector &x, Vector & E_r)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x,E);
   E_r.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_r[i]= E[i].real();
   }
}

void E_exact_i(const Vector &x, Vector & E_i)
{
   std::vector<std::complex<double>> E;
   maxwell_solution(x, E);
   E_i.SetSize(E.size());
   for (unsigned i = 0; i < E.size(); i++)
   {
      E_i[i]= E[i].imag();
   }
}

void curlE_exact_r(const Vector &x, Vector &curlE_r)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_r.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_r[i]= curlE[i].real();
   }
}

void curlE_exact_i(const Vector &x, Vector &curlE_i)
{
   std::vector<std::complex<double>> curlE;
   maxwell_solution_curl(x, curlE);
   curlE_i.SetSize(curlE.size());
   for (unsigned i = 0; i < curlE.size(); i++)
   {
      curlE_i[i]= curlE[i].imag();
   }
}

void curlcurlE_exact_r(const Vector &x, Vector & curlcurlE_r)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_r.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_r[i]= curlcurlE[i].real();
   }
}

void curlcurlE_exact_i(const Vector &x, Vector & curlcurlE_i)
{
   std::vector<std::complex<double>> curlcurlE;
   maxwell_solution_curlcurl(x, curlcurlE);
   curlcurlE_i.SetSize(curlcurlE.size());
   for (unsigned i = 0; i < curlcurlE.size(); i++)
   {
      curlcurlE_i[i]= curlcurlE[i].imag();
   }
}

void  rhs_func_r(const Vector &x, Vector & J_r)
{
   Vector E_r, curlcurlE_r;
   E_exact_r(x,E_r);
   curlcurlE_exact_r(x,curlcurlE_r);
   J_r.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_r(i) = 1./mu*curlcurlE_r(i) - omega *omega * epsilon * E_r(i);
   }
}

void  rhs_func_i(const Vector &x, Vector & J_i)
{
   Vector E_i, curlcurlE_i;
   E_exact_i(x,E_i);
   curlcurlE_exact_i(x,curlcurlE_i);
   J_i.SetSize(dim);
   for (int i = 0; i<dim; i++)
   {
      J_i(i) = 1./mu*curlcurlE_i(i) - omega *omega * epsilon * E_i(i);
   }
}

void maxwell_solution(const Vector & X, std::vector<complex<double>> &E)
{
   complex<double> zi = complex<double>(0., 1.);
   E.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      E[i] = 0.0;
   }
   E[0] = exp(zi * omega * (X.Sum()));
}

void maxwell_solution_curl(const Vector & X,
                           std::vector<complex<double>> &curlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlE.resize(dimc);
   for (int i = 0; i < dimc; ++i)
   {
      curlE[i] = 0.0;
   }
   std::complex<double> pw = exp(zi * omega * (X.Sum()));
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
                               std::vector<complex<double>> &curlcurlE)
{
   complex<double> zi = complex<double>(0., 1.);
   curlcurlE.resize(dim);
   for (int i = 0; i < dim; ++i)
   {
      curlcurlE[i] = 0.0;;
   }
   std::complex<double> pw = exp(zi * omega * (X.Sum()));
   if (dim == 3)
   {
      curlcurlE[0] = 2.0 * omega * omega * pw;
      curlcurlE[1] = - omega * omega * pw;
      curlcurlE[2] = - omega * omega * pw;
   }
   else
   {
      curlcurlE[0] = omega * omega * pw;
      curlcurlE[1] = -omega * omega * pw;
   }
}
