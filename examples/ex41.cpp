//                                MFEM Example 41
//
// Compile with: make ex41
//
// Sample runs: ex41
//              ex41 -p 0 -r 2 -dt 0.01 -tf 10
//              ex41 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.005 -tf 10
//              ex41 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//              ex41 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//              ex41 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//              ex41 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.001 -tf 9
//              ex41 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//              ex41 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//              ex41 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//              ex41 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//              ex41 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.01 -tf 8
//
// Device sample runs:
//
// Description:  This example code solves the time-dependent advection-diffusion
//               equation du/dt + v.grad(u) - a div(grad(u)) = 0, where v is a
//               given fluid velocity, a is the diffusion coefficient, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), and the use of IMEX ODE time integrators.

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Mesh bounding box
Vector bb_min, bb_max;

// Velocity coefficient
template<int problem=0>
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }
   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const real_t w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const real_t w = M_PI/2;
         real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
template<int problem=0>
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

class DG_Solver : public Solver
{
private:
   SparseMatrix &M, &S, A;
   CGSolver linear_solver;
   BlockILU prec;
   real_t dt;
public:
   DG_Solver(SparseMatrix &M_, SparseMatrix &S_,
             const FiniteElementSpace &fes)
      : M(M_),
        S(S_),
        prec(fes.GetTypicalFE()->GetDof(),
             BlockILU::Reordering::MINIMUM_DISCARDED_FILL),
        dt(1.0)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // Form operator A = M + dt*S
         A = S;
         A *= dt;
         A += M;

         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of the advection-diffusion equation is (M + dt S) du/dt = Su - K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. In the case of IMEX evolution, the diffusion term is treated
    implicitly, and the advection term is treated explicitly.  */
class IMEX_Evolution : public SplitTimeDependentOperator
{
private:
   BilinearForm &M, &K, &S;
   const Vector &b;
   unique_ptr<Solver> M_prec;
   CGSolver M_solver;
   unique_ptr<DG_Solver> dg_solver;

   mutable Vector z;

public:
   IMEX_Evolution(BilinearForm &M_, BilinearForm &K_, BilinearForm &S_,
                  const Vector &b_);

   void Mult1(const Vector &x, Vector &y) const override;
   void ImplicitSolve2(const real_t dt, const Vector &x, Vector &k) override;
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 0;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 2;
   int order = 3;
   const char *device_config = "cpu";
   int ode_solver_type = 58; //55 - Forward Backward Euler
   //56 - IMEXRK2(2,2,2)
   //57 - IMEXRK2(2,3,2)
   //58 - IMEXRK3(3,4,3)
   real_t t_final = 10.0;
   real_t dt = 0.001;
   bool paraview = false;
   int vis_steps = 50;
   real_t diffusion_term = 0.01;
   real_t kappa = -1.0;
   real_t sigma = -1.0;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int precision = 8;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order", "Order of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(cout);
   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();

   // 3. Define the Split ODE solver used for time integration. The IMEX solvers currently
   // available are: 55 - Forward Backward Euler, 56 - IMEXRK2(2,2,2), 57 - IMEXRK2(2,3,2), and
   // 58 - IMEX_DIRK_RK3.
   unique_ptr<SplitODESolver> ode_solver = SplitODESolver::Select(ode_solver_type);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++) {mesh.UniformRefinement();}
   if (mesh.NURBSext) {mesh.SetCurvature(max(order, 1));}
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   std::unique_ptr<VectorFunctionCoefficient> velocity;
   if (0==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<0>));
   }
   else if (1==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<1>));
   }
   else if (2==problem)
   {
      velocity.reset(new VectorFunctionCoefficient(dim, velocity_function<2>));
   }

   ConstantCoefficient diff_coeff(diffusion_term);

   BilinearForm m(&fes);
   BilinearForm k(&fes);
   BilinearForm s(&fes);

   Vector b(fes.GetTrueVSize());
   b=0.0; //The inflow on the boundaries is set to zero.

   m.AddDomainIntegrator(new MassIntegrator);

   constexpr real_t alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(*velocity, alpha));
   k.AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity,
                                                                    alpha));
   k.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(*velocity, alpha));

   s.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   s.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
                                                         kappa));
   s.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));


   int skip_zeros = 0;
   m.Assemble(skip_zeros);
   k.Assemble(skip_zeros);
   s.Assemble(skip_zeros);

   m.Finalize(skip_zeros);
   k.Finalize(skip_zeros);
   s.Finalize(skip_zeros);

   // 7. Define the initial conditions.
   std::unique_ptr<FunctionCoefficient> u0;
   if (0==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<0>));
   }
   else if (1==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<1>));
   }
   else if (2==problem)
   {
      u0.reset(new FunctionCoefficient(u0_function<2>));
   }

   GridFunction u(&fes);
   u.ProjectCoefficient(*u0);

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example41", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example41", &mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   // 8. Set up paraview visualization, if desired.
   unique_ptr<ParaViewDataCollection> pv;
   if (paraview)
   {
      pv = make_unique<ParaViewDataCollection>("Example41", &mesh);
      pv->SetPrefixPath("ParaView");
      pv->RegisterField("solution", &u);
      pv->SetLevelsOfDetail(order);
      pv->SetDataFormat(VTKFormat::BINARY);
      pv->SetHighOrderOutput(true);
      pv->SetCycle(0);
      pv->SetTime(0.0);
      pv->Save();
   }
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 9. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   IMEX_Evolution adv(m, k, s, b);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         if (paraview)
         {
            pv->SetCycle(ti);
            pv->SetTime(t);
            pv->Save();
         }
         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }
         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

      }
   }

   {
      ofstream osol("ex41-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   return 0;
}


// Implementation of class IMEX_Evolution
IMEX_Evolution::IMEX_Evolution(BilinearForm &M_, BilinearForm &K_,
                               BilinearForm &S_, const Vector &b_)
   : SplitTimeDependentOperator(M_.FESpace()->GetTrueVSize()),
     M(M_), K(K_), S(S_), b(b_), z(height)
{
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      M_prec = make_unique<DSmoother>(M.SpMat());
      M_solver.SetOperator(M.SpMat());
      dg_solver = make_unique<DG_Solver>(M.SpMat(), S.SpMat(),
                                         *M.FESpace());
   }
   else
   {
      M_prec = make_unique<OperatorJacobiSmoother>(M, ess_tdof_list);
      M_solver.SetOperator(M);
      dg_solver = NULL;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void IMEX_Evolution::Mult1(const Vector &x, Vector &y) const
{
   // Perform the explicit step
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void IMEX_Evolution::ImplicitSolve2(const real_t dt, const Vector &x, Vector &k)
{
   // Perform the implicit step
   // solve for k, k = -(M+dt S)^{-1} S x
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   S.Mult(x, z);
   z*= -1.0;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}
