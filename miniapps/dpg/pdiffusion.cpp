//                 MFEM Ultraweal DPG parallel example for diffusion
//
// Compile with: make pdiffusion
//
// Sample runs
// mpirun -np 4 pdiffusion -m ../../data/inline-quad.mesh -o 3 -sref 1 -pref 2 -theta 0.0 -prob 0
// mpirun -np 4 pdiffusion -m ../../data/inline-hex.mesh -o 2 -sref 0 -pref 1 -theta 0.0 -prob 0 -sc
// mpirun -np 4 pdiffusion -m ../../data/beam-tet.mesh -o 3 -sref 0 -pref 2 -theta 0.0 -prob 0 -sc

// L-shape runs
// Note: uniform ref are expected to give sub-optimal rate for the L-shape problem (rate = 2/3)
// mpirun -np 4 pdiffusion -o 2 -sref 1 -pref 5 -theta 0.0 -prob 1

// L-shape AMR runs
// mpirun -np 4 pdiffusion -o 1 -sref 1 -pref 20 -theta 0.8 -prob 1
// mpirun -np 4 pdiffusion -o 2 -sref 1 -pref 20 -theta 0.75 -prob 1 -sc
// mpirun -np 4 pdiffusion -o 3 -sref 1 -pref 20 -theta 0.75 -prob 1 -sc -do 2

// Description:
// This example code demonstrates the use of MFEM to define and solve
// the "ultraweak" (UW) DPG formulation for the Poisson problem in parallel

//       - Δ u = f,   in Ω
//           u = u₀, on ∂Ω
//
// It solves two kinds of problems
// a) A manufactured solution problem where u_exact = sin(π * (x + y + z)).
//    This example computes and prints out convergence rates for the L2 error.
// b) The L-shape benchmark problem with AMR. The AMR process is driven by the
//    DPG built-in residual indicator.

// The DPG UW deals with the First Order System
//   ∇ u - σ = 0, in Ω
// - ∇⋅σ     = f, in Ω
//        u  = u₀, in ∂Ω

// Ultraweak-DPG is obtained by integration by parts of both equations and the
// introduction of trace unknowns on the mesh skeleton

// u ∈ L²(Ω), σ ∈ (L²(Ω))ᵈⁱᵐ
// û ∈ H^1/2, σ̂ ∈ H^-1/2
// -(u , ∇⋅τ) + < û, τ⋅n> - (σ , τ) = 0,      ∀ τ ∈ H(div,Ω)
//  (σ , ∇ v) - < σ̂, v  >           = (f,v)   ∀ v ∈ H¹(Ω)
//                                û = u₀        on ∂Ω

// Note:
// û := u and σ̂ := -σ on the mesh skeleton

// -------------------------------------------------------------
// |   |     u     |     σ     |    û      |    σ̂    |  RHS    |
// -------------------------------------------------------------
// | τ | -(u,∇⋅τ)  |  -(σ,τ)   | < û, τ⋅n> |         |    0    |
// |   |           |           |           |         |         |
// | v |           |  (σ,∇ v)  |           | -<σ̂,v>  |  (f,v)  |

// where (τ,v) ∈  H(div,Ω) × H¹(Ω)

// For more information see https://doi.org/10.1007/978-3-319-01818-8_6

#include "mfem.hpp"
#include "util/pweakform.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mfem::common;

enum prob_type
{
   manufactured,
   lshape
};

static const char *enum_str[] =
{
   "manufactured",
   "lshape"
};

prob_type prob;

double exact_u(const Vector & X);
void exact_gradu(const Vector & X, Vector &gradu);
double exact_laplacian_u(const Vector & X);
void exact_sigma(const Vector & X, Vector & sigma);
double exact_hatu(const Vector & X);
void exact_hatsigma(const Vector & X, Vector & hatsigma);
double f_exact(const Vector & X);

int main(int argc, char *argv[])
{
   // 0. Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   int sref = 0; // initial uniform mesh refinements
   int pref = 0; // parallel mesh refinements for AMR
   int iprob = 0;
   bool static_cond = false;
   double theta = 0.7;
   bool visualization = true;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");
   args.AddOption(&sref, "-sref", "--num-serial-refinements",
                  "Number of initial serial uniform refinements");
   args.AddOption(&pref, "-pref", "--num-parallel-refinements",
                  "Number of AMR refinements");
   args.AddOption(&theta, "-theta", "--theta-factor",
                  "Refinement factor (0 indicates uniform refinements) ");
   args.AddOption(&iprob, "-prob", "--problem", "Problem case"
                  " 0: manufactured, 1: L-shape");
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

   if (iprob > 1) { iprob = 1; }
   prob = (prob_type)iprob;

   if (prob == prob_type::lshape)
   {
      mesh_file = "../../data/l-shape.mesh";
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   MFEM_VERIFY(dim > 1, "Dimension = 1 is not supported in this example");

   if (prob == prob_type::lshape)
   {
      /** rotate mesh to be consistent with l-shape benchmark problem
          See https://doi.org/10.1016/j.amc.2013.05.068 */
      mesh.EnsureNodes();
      GridFunction *nodes = mesh.GetNodes();
      int size = nodes->Size()/2;
      for (int i = 0; i<size; i++)
      {
         double x = (*nodes)[2*i];
         (*nodes)[2*i] =  2*(*nodes)[2*i+1]-1;
         (*nodes)[2*i+1] = -2*x+1;
      }
   }


   for (int i = 0; i<sref; i++)
   {
      mesh.UniformRefinement();
   }

   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // Define spaces
   enum TrialSpace
   {
      u_space        = 0,
      sigma_space    = 1,
      hatu_space     = 2,
      hatsigma_space = 3
   };
   enum TestSpace
   {
      tau_space = 0,
      v_space   = 1
   };
   // L2 space for u
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *u_fes = new ParFiniteElementSpace(&pmesh,u_fec);

   // Vector L2 space for σ
   FiniteElementCollection *sigma_fec = new L2_FECollection(order-1,dim);
   ParFiniteElementSpace *sigma_fes = new ParFiniteElementSpace(&pmesh,sigma_fec,
                                                                dim);

   // H^1/2 space for û
   FiniteElementCollection * hatu_fec = new H1_Trace_FECollection(order,dim);
   ParFiniteElementSpace *hatu_fes = new ParFiniteElementSpace(&pmesh,hatu_fec);

   // H^-1/2 space for σ̂
   FiniteElementCollection * hatsigma_fec = new RT_Trace_FECollection(order-1,dim);
   ParFiniteElementSpace *hatsigma_fes = new ParFiniteElementSpace(&pmesh,
                                                                   hatsigma_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * tau_fec = new RT_FECollection(test_order-1, dim);
   FiniteElementCollection * v_fec = new H1_FECollection(test_order, dim);

   Array<ParFiniteElementSpace * > trial_fes;
   Array<FiniteElementCollection * > test_fec;

   trial_fes.Append(u_fes);
   trial_fes.Append(sigma_fes);
   trial_fes.Append(hatu_fes);
   trial_fes.Append(hatsigma_fes);
   test_fec.Append(tau_fec);
   test_fec.Append(v_fec);

   // Required coefficients for the weak formulation
   ConstantCoefficient one(1.0);
   ConstantCoefficient negone(-1.0);
   FunctionCoefficient f(f_exact); // rhs for the manufactured solution problem

   // Required coefficients for the exact solutions
   FunctionCoefficient uex(exact_u);
   VectorFunctionCoefficient sigmaex(dim,exact_sigma);
   FunctionCoefficient hatuex(exact_hatu);

   ParDPGWeakForm * a = new ParDPGWeakForm(trial_fes,test_fec);
   a->StoreMatrices(true); // this is needed for estimation of residual

   //  -(u,∇⋅τ)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),
                         TrialSpace::u_space,TestSpace::tau_space);

   // -(σ,τ)
   a->AddTrialIntegrator(new TransposeIntegrator(new VectorFEMassIntegrator(
                                                    negone)), TrialSpace::sigma_space, TestSpace::tau_space);

   // (σ,∇ v)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(one)),
                         TrialSpace::sigma_space,TestSpace::v_space);

   //  <û,τ⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,
                         TrialSpace::hatu_space,TestSpace::tau_space);

   // -<σ̂,v> (sign is included in σ̂)
   a->AddTrialIntegrator(new TraceIntegrator,
                         TrialSpace::hatsigma_space, TestSpace::v_space);

   // test integrators (space-induced norm for H(div) × H1)
   // (∇⋅τ,∇⋅δτ)
   a->AddTestIntegrator(new DivDivIntegrator(one),
                        TestSpace::tau_space, TestSpace::tau_space);
   // (τ,δτ)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),
                        TestSpace::tau_space, TestSpace::tau_space);
   // (∇v,∇δv)
   a->AddTestIntegrator(new DiffusionIntegrator(one),
                        TestSpace::v_space, TestSpace::v_space);
   // (v,δv)
   a->AddTestIntegrator(new MassIntegrator(one),
                        TestSpace::v_space, TestSpace::v_space);

   // RHS
   if (prob == prob_type::manufactured)
   {
      a->AddDomainLFIntegrator(new DomainLFIntegrator(f),TestSpace::v_space);
   }

   // GridFunction for Dirichlet bdr data
   ParGridFunction hatu_gf;

   // Visualization streams
   socketstream u_out;
   socketstream sigma_out;

   if (myid == 0)
   {
      std::cout << "\n  Ref |"
                << "    Dofs    |"
                << "  L2 Error  |"
                << "  Rate  |"
                << "  Residual  |"
                << "  Rate  |"
                << " PCG it |" << endl;
      std::cout << std::string(72,'-') << endl;
   }

   Array<int> elements_to_refine; // for AMR
   double err0 = 0.;
   int dof0=0.;
   double res0=0.0;

   ParGridFunction u_gf(u_fes);
   ParGridFunction sigma_gf(sigma_fes);
   u_gf = 0.0;
   sigma_gf = 0.0;

   ParaViewDataCollection * paraview_dc = nullptr;

   if (paraview)
   {
      paraview_dc = new ParaViewDataCollection(enum_str[prob], &pmesh);
      paraview_dc->SetPrefixPath("ParaView/Diffusion");
      paraview_dc->SetLevelsOfDetail(order);
      paraview_dc->SetCycle(0);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetTime(0.0); // set the time
      paraview_dc->RegisterField("u",&u_gf);
      paraview_dc->RegisterField("sigma",&sigma_gf);
   }

   if (static_cond) { a->EnableStaticCondensation(); }
   for (int it = 0; it<=pref; it++)
   {
      a->Assemble();

      Array<int> ess_tdof_list;
      Array<int> ess_bdr;
      if (pmesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh.bdr_attributes.Max());
         ess_bdr = 1;
         hatu_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      // shift the ess_tdofs
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         ess_tdof_list[i] += u_fes->GetTrueVSize() + sigma_fes->GetTrueVSize();
      }

      Array<int> offsets(5);
      offsets[0] = 0;
      offsets[1] = u_fes->GetVSize();
      offsets[2] = sigma_fes->GetVSize();
      offsets[3] = hatu_fes->GetVSize();
      offsets[4] = hatsigma_fes->GetVSize();
      offsets.PartialSum();
      BlockVector x(offsets);
      x = 0.0;
      hatu_gf.MakeRef(hatu_fes,x.GetBlock(2),0);
      hatu_gf.ProjectBdrCoefficient(uex,ess_bdr);

      Vector X,B;
      OperatorPtr Ah;
      a->FormLinearSystem(ess_tdof_list,x,Ah,X,B);

      BlockOperator * A = Ah.As<BlockOperator>();

      BlockDiagonalPreconditioner M(A->RowOffsets());
      M.owns_blocks = 1;
      int skip = 0;
      if (!static_cond)
      {
         HypreBoomerAMG * amg0 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(0,0));
         HypreBoomerAMG * amg1 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(1,1));
         amg0->SetPrintLevel(0);
         amg1->SetPrintLevel(0);
         M.SetDiagonalBlock(0,amg0);
         M.SetDiagonalBlock(1,amg1);
         skip=2;
      }
      HypreBoomerAMG * amg2 = new HypreBoomerAMG((HypreParMatrix &)A->GetBlock(skip,
                                                                               skip));
      amg2->SetPrintLevel(0);
      M.SetDiagonalBlock(skip,amg2);
      HypreSolver * prec;
      if (dim == 2)
      {
         // AMS preconditioner for 2D H(div) (trace) space
         prec = new HypreAMS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      else
      {
         // ADS preconditioner for 3D H(div) (trace) space
         prec = new HypreADS((HypreParMatrix &)A->GetBlock(skip+1,skip+1), hatsigma_fes);
      }
      M.SetDiagonalBlock(skip+1,prec);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(0);
      cg.SetPreconditioner(M);
      cg.SetOperator(*A);
      cg.Mult(B, X);

      a->RecoverFEMSolution(X,x);

      Vector & residuals = a->ComputeResidual(x);

      double residual = residuals.Norml2();

      double maxresidual = residuals.Max();
      double globalresidual = residual * residual;

      MPI_Allreduce(MPI_IN_PLACE,&maxresidual,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&globalresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      globalresidual = sqrt(globalresidual);

      u_gf.MakeRef(u_fes,x.GetBlock(0),0);
      sigma_gf.MakeRef(sigma_fes,x.GetBlock(1),0);

      int dofs = u_fes->GlobalTrueVSize() + sigma_fes->GlobalTrueVSize()
                 + hatu_fes->GlobalTrueVSize() + hatsigma_fes->GlobalTrueVSize();

      double u_err = u_gf.ComputeL2Error(uex);
      double sigma_err = sigma_gf.ComputeL2Error(sigmaex);
      double L2Error = sqrt(u_err*u_err + sigma_err*sigma_err);
      double rate_err = (it) ? dim*log(err0/L2Error)/log((double)dof0/dofs) : 0.0;
      double rate_res = (it) ? dim*log(res0/globalresidual)/log((
                                                                   double)dof0/dofs) : 0.0;
      err0 = L2Error;
      res0 = globalresidual;
      dof0 = dofs;

      if (myid == 0)
      {
         std::ios oldState(nullptr);
         oldState.copyfmt(std::cout);
         std::cout << std::right << std::setw(5) << it << " | "
                   << std::setw(10) <<  dof0 << " | "
                   << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  err0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_err << " | "
                   << std::setprecision(3)
                   << std::setw(10) << std::scientific <<  res0 << " | "
                   << std::setprecision(2)
                   << std::setw(6) << std::fixed << rate_res << " | "
                   << std::setw(6) << std::fixed << cg.GetNumIterations() << " | "
                   << std::endl;
         std::cout.copyfmt(oldState);
      }

      if (visualization)
      {
         const char * keys = (it == 0 && dim == 2) ? "jRcm\n" : nullptr;
         char vishost[] = "localhost";
         int  visport   = 19916;

         VisualizeField(u_out,vishost,visport,u_gf,
                        "Numerical u", 0,0,500,500,keys);
         VisualizeField(sigma_out,vishost,visport,sigma_gf,
                        "Numerical flux", 500,0,500,500,keys);
      }

      if (paraview)
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime((double)it);
         paraview_dc->Save();
      }

      if (it == pref) { break; }

      elements_to_refine.SetSize(0);
      for (int iel = 0; iel<pmesh.GetNE(); iel++)
      {
         if (residuals[iel] >= theta * maxresidual)
         {
            elements_to_refine.Append(iel);
         }
      }

      pmesh.GeneralRefinement(elements_to_refine);

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
   delete tau_fec;
   delete v_fec;
   delete hatsigma_fes;
   delete hatsigma_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete sigma_fec;
   delete sigma_fes;
   delete u_fec;
   delete u_fes;

   return 0;
}

double exact_u(const Vector & X)
{
   switch (prob)
   {
      case prob_type::lshape:
      {
         double x = X[0];
         double y = X[1];
         double r = sqrt(x*x + y*y);
         double alpha = 2./3.;
         double phi = atan2(y,x);
         if (phi < 0) { phi += 2*M_PI; }
         return pow(r,alpha) * sin(alpha * phi);
      }
      break;
      default:
      {
         double alpha = M_PI * (X.Sum());
         return sin(alpha);
      }
      break;
   }
}

void exact_gradu(const Vector & X, Vector & du)
{
   du.SetSize(X.Size());
   switch (prob)
   {
      case prob_type::lshape:
      {
         double x = X[0];
         double y = X[1];
         double r = sqrt(x*x + y*y);
         double alpha = 2./3.;
         double phi = atan2(y,x);
         if (phi < 0) { phi += 2*M_PI; }

         double r_x = x/r;
         double r_y = y/r;
         double phi_x = - y / (r*r);
         double phi_y = x / (r*r);
         double beta = alpha * pow(r,alpha - 1.);
         du[0] = beta*(r_x * sin(alpha*phi) + r * phi_x * cos(alpha*phi));
         du[1] = beta*(r_y * sin(alpha*phi) + r * phi_y * cos(alpha*phi));
      }
      break;
      default:
      {
         double alpha = M_PI * (X.Sum());
         du.SetSize(X.Size());
         for (int i = 0; i<du.Size(); i++)
         {
            du[i] = M_PI * cos(alpha);
         }
      }
      break;
   }
}

double exact_laplacian_u(const Vector & X)
{
   switch (prob)
   {
      case prob_type::manufactured:
      {
         double alpha = M_PI * (X.Sum());
         double u = sin(alpha);
         return - M_PI*M_PI * u * X.Size();
      }
      break;
      default:
         MFEM_ABORT("Should be unreachable");
         return 1;
         break;
   }
}

void exact_sigma(const Vector & X, Vector & sigma)
{
   // σ = ∇ u
   exact_gradu(X,sigma);
}

double exact_hatu(const Vector & X)
{
   return exact_u(X);
}

void exact_hatsigma(const Vector & X, Vector & hatsigma)
{
   exact_sigma(X,hatsigma);
   hatsigma *= -1.;
}

double f_exact(const Vector & X)
{
   MFEM_VERIFY(prob!=prob_type::lshape,
               "f_exact should not be called for l-shape benchmark problem, i.e., f = 0")
   return -exact_laplacian_u(X);
}
