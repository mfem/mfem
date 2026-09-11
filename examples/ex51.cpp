//                                MFEM Example 51
//
// Compile with: make ex51
//
// Sample runs:
//    ex51 -m ../data/periodic-square.mesh
//
// Description:  This examples solves a time dependent nonlinear burgers equation

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Mesh bounding box
Vector bb_min, bb_max;

// Initial condition
void u0_function(const Vector &x, Vector &v)
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
               v = exp(-40.*pow(X(0)-0.5,2));
               break;
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
               v = (std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                    std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
               break;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         v = pow(sin(M_PI*rho),2)*sin(3*phi);
         break;
      }
      case 3:
      {
         const real_t f = M_PI;
         v = sin(f*X(0))*sin(f*X(1));
         break;
      }
   }
}

//
real_t Tau (ElementTransformation& Tr,
            const real_t& dt,
            const Vector& u,
            const Vector& dudt,
            const DenseMatrix& dudx,
            const Vector& res)
{
   DenseMatrix Metric(u.Size());
   MultAtB(Tr.InverseJacobian(), Tr.InverseJacobian(), Metric);
   return 1.0/sqrt(1.0/(dt*dt) + u.Norm2(Metric));
   //return 1.0/sqrt(1.0/(dt*dt) + Tr.Metric().Mult(u));
}

//
real_t Kdc (ElementTransformation& Tr,
            const real_t& dt,
            const Vector& u,
            const Vector& dudt,
            const DenseMatrix& dudx,
            const Vector& res)
{
   DenseMatrix Metric(u.Size());
   MultAtB(Tr.InverseJacobian(), Tr.InverseJacobian(), Metric);
   return 0.1*res.Norml2()/(sqrt(Metric.Trace()*dudx.FNorm2()) + 1e-10);
   //return 0.1*res.Norml2()/(sqrt(Tr.Metric().Trace()*dudx.FNorm2()) + 1e-10);
}

DenseMatrix mat1;
void Kmat (ElementTransformation& Tr,
           const real_t& dt,
           const Vector& u,
           const Vector& dudt,
           const DenseMatrix& dudx,
           const Vector& res,
           DenseMatrix& Kij)
{
   MultAtB(Tr.Jacobian(), Tr.Jacobian(), Kij);
   //MultAtB(Tr.PerfectJacobian(), Tr.PerfectJacobian(), Kij);
   mat1.SetSize(dudx.Size());
   //Mult(Tr.PerfectJacobian(), dudx, mat1);
   Mult(Tr.Jacobian(), dudx, mat1);
   Kij *= 0.01*res.Norml2()/(mat1.FNorm() + 1e-10);
}

//
int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *device_config = "cpu";
   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);

   // Setup parameters
   problem = 0;
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");

   // Discretization parameters
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 1;
   int order = 2;
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");

   // Time integration parameters
   int ode_solver_type = 32;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");

   // Solver parameters
   double GMRES_RelTol = 1e-3;
   int    GMRES_MaxIter = 250;
   double Newton_RelTol = 1e-3;
   int    Newton_MaxIter = 10;

   args.AddOption(&GMRES_RelTol, "-lt", "--linear-tolerance",
                  "Relative tolerance for the GMRES solver.");
   args.AddOption(&GMRES_MaxIter, "-li", "--linear-itermax",
                  "Maximum iteration count for the GMRES solver.");
   args.AddOption(&Newton_RelTol, "-nt", "--newton-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&Newton_MaxIter, "-ni", "--newton-itermax",
                  "Maximum iteration count for the Newton solver.");

   // Visualization parameters
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 5;
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient u0(dim, u0_function);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex51.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex51-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example51", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example51", &mesh);
         dc->SetPrecision(precision);
      }
      dc->SetPrefixPath("Example51");
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example51", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", &u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
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

   // 8.
   StabilizedVectorConvectionNLFIntegrator::TauFunc_t tau;
   StabilizedVectorConvectionNLFIntegrator::KappaFunc_t kdc;
   StabilizedVectorConvectionNLFIntegrator::KappaMatFunc_t kmat;
   tau = Tau;
   kdc = Kdc;
   kmat = Kmat;

   TimeDepNonlinearForm form(&fes);
   form.AddTimeDepDomainIntegrator(new StabilizedVectorConvectionNLFIntegrator(
                                      &tau, &kmat));

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FGMRESSolver gmres;
   gmres.iterative_mode = false;
   gmres.SetRelTol(GMRES_RelTol);
   gmres.SetAbsTol(1e-12);
   gmres.SetMaxIter(GMRES_MaxIter);
   gmres.SetPrintLevel(-1);

   // Set up the Newton solver
   NewtonSolver newton_solver;
   newton_solver.iterative_mode = true;
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(Newton_RelTol);
   newton_solver.SetAbsTol(1e-12);
   newton_solver.SetMaxIter(Newton_MaxIter );
   newton_solver.SetSolver(gmres);

   Evolution adv(form, newton_solver);

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

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex51.mesh -g ex51-final.gf".
   {
      ofstream osol("ex51-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete pd;
   delete dc;

   return 0;
}






