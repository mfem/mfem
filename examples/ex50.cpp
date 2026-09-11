//                                MFEM Example 50
//
// Compile with: make ex50
//
// Sample runs:
//    ex50 -m ../data/periodic-square.mesh
//
// Description: This examples solves the unsteady convection-diffusion-reaction
//              using a stabilized formulation.

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
      case 4:
      {
         if (fabs(X(0)) < 0.1 && fabs(X(1)) < 0.1) { return 1; }
         return 0;
      }
   }
   return 0.0;
}

//
real_t Tau (ElementTransformation& Tr,
            const real_t& dt,
            const Vector& a,
            const real_t& dudt,
            const Vector& dudx,
            const real_t& res)
{
   DenseMatrix Metric(a.Size());
   MultAtB(Tr.InverseJacobian(), Tr.InverseJacobian(), Metric);

   return 1.0/sqrt(1.0/(dt*dt) + a.Norm2(Metric));

   //return 1.0/sqrt(1.0/(dt*dt) + Tr.Metric().Mult(a));
}

//
real_t Kdc (ElementTransformation& Tr,
            const real_t& dt,
            const Vector& a,
            const real_t& dudt,
            const Vector& dudx,
            const real_t& res)
{
   DenseMatrix Metric(a.Size());
   MultAtB(Tr.InverseJacobian(), Tr.InverseJacobian(), Metric);

   return 0.01*fabs(res)/(sqrt(Metric.Trace())*dudx.Norml2() + 1e-10);
   //   return 0.01*fabs(res)/(sqrt(Tr.Metric().Trace())*dudx.Norml2() + 1e-10);
}

//
Vector vec1;
void Kmat (ElementTransformation& Tr,
           const real_t& dt,
           const Vector& a,
           const real_t& dudt,
           const Vector& dudx,
           const real_t& res,
           DenseMatrix& Kij)
{
   // MultAtB(Tr.PerfectJacobian(), Tr.PerfectJacobian(), Kij);
   MultAtB(Tr.Jacobian(), Tr.Jacobian(), Kij);
   vec1.SetSize(dudx.Size());
   Tr.Jacobian().Mult(dudx, vec1);
   //Tr.PerfectJacobian().Mult(dudx, vec1);
   Kij *= 0.01*fabs(res)/(vec1.Norml2() + 1e-10);
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
   problem = 4;
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");

   // Discretization parameters
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
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
   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   FunctionCoefficient u0(u0_function);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex50.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex50-init.gf");
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
         dc = new SidreDataCollection("Example50", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example50", &mesh);
         dc->SetPrecision(precision);
      }
      dc->SetPrefixPath("Example50");
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example50", &mesh);
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
   ConstantCoefficient mass(1.0);
   Vector a(dim); a = 1.0;
   VectorConstantCoefficient conv(a);
   ConstantCoefficient diff(0.0);
   ConstantCoefficient react(0.0);
   ConstantCoefficient force(0.0);

   StabilizedCDRIntegrator::TauFunc_t tau;
   StabilizedCDRIntegrator::KappaFunc_t kdc;
   StabilizedCDRIntegrator::KappaMatFunc_t kdc_mat;
   tau = Tau;
   kdc = Kdc;
   kdc_mat = Kmat;

   TimeDepNonlinearForm form(&fes);
   form.AddTimeDepDomainIntegrator(
      new StabilizedCDRIntegrator(&mass, &conv, &diff, &react, &force,
                                  &tau, &kdc));


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
   newton_solver.SetMaxIter(Newton_MaxIter);
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
   //    "glvis -m ex50.mesh -g ex50-final.gf".
   {
      ofstream osol("ex50-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete pd;
   delete dc;

   return 0;
}






