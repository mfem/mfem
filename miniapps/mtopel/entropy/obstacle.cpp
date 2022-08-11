//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   FunctionCoefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, FunctionCoefficient &obst_, double min_val_=1e-12)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, double max_val_=1e12)
      : u(&u_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ReciprocalGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u;
   FunctionCoefficient *obstacle;
   double min_val;
   double max_val;

public:
   ReciprocalGridFunctionCoefficient(GridFunction &u_, FunctionCoefficient &obst_, double min_val_=1e-12, double max_val_=1e12)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;
   int max_it = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels = 4;
      // int ref_levels =
      //    (int)floor(log(50000./mesh.GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   // ess_tdof_list.Print();

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   auto func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      // return -1.0;
      // return -x.Size()*pow(M_PI,2) * val;
      return x.Size()*pow(M_PI,2) * val;
   };
   ConstantCoefficient f(-1.0);
   // FunctionCoefficient f(func);
   ConstantCoefficient one(1.0);
   FunctionCoefficient obstacle(spherical_obstacle);

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction u(&fespace);
   GridFunction delta_u(&fespace);
   GridFunction u_old(&fespace);
   GridFunction tmp_gf(&fespace);
   u = 1.0;
   delta_u = 0.0;
   u_old = 1.0;
   tmp_gf = 0.5;

   OperatorPtr A;
   Vector B, X;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   // 12. Iterate
   double alpha0 = 0.1;
   for (int k = 1; k <= max_it; k++)
   {
      // double alpha = alpha0 / sqrt(k);
      double alpha = alpha0 * sqrt(k);
      // alpha *= 2;
      for (int j = 1; j <= 10; j++)
      {
         // A. Assembly
         
         // MD
         // double c1 = 1.0;
         // double c2 = 1.0 - alpha;

         // IMD
         double c1 = 1.0 + alpha;
         double c2 = 1.0;

         BilinearForm a(&fespace);
         ConstantCoefficient alpha_cf(alpha);
         ConstantCoefficient c1_cf(c1);
         ReciprocalGridFunctionCoefficient one_over_u(u, obstacle);

         a.AddDomainIntegrator(new DiffusionIntegrator(c1_cf));
         // a.AddDomainIntegrator(new MassIntegrator(alpha_cf));
         a.AddDomainIntegrator(new MassIntegrator(one_over_u));
         a.Assemble();

         LinearForm b(&fespace);
         GradientGridFunctionCoefficient grad_u(&u);
         GradientGridFunctionCoefficient grad_u_old(&u_old);
         VectorSumCoefficient gradient_term_RHS(grad_u, grad_u_old, -c1, c2);
         b.AddDomainIntegrator(new DomainLFGradIntegrator(gradient_term_RHS));
         // ScalarVectorProductCoefficient minus_alpha_grad_u_old(-alpha, grad_u_old);
         // b.AddDomainIntegrator(new DomainLFGradIntegrator(minus_alpha_grad_u_old));
         // GridFunctionCoefficient u_cf(&u);
         // ProductCoefficient minus_alpha_u(-alpha, u_cf);
         // b.AddDomainIntegrator(new DomainLFIntegrator(minus_alpha_u));
         ProductCoefficient alpha_f(alpha, f);
         b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         LogarithmGridFunctionCoefficient log_u_old(u_old, obstacle);
         b.AddDomainIntegrator(new DomainLFIntegrator(log_u_old));
         LogarithmGridFunctionCoefficient log_u(u, obstacle);
         ProductCoefficient minus_log_u(-1.0, log_u);
         b.AddDomainIntegrator(new DomainLFIntegrator(minus_log_u));
         b.Assemble();

         // B. Solve state equation
         a.FormLinearSystem(ess_tdof_list, delta_u, b, A, X, B);
         GSSmoother S((SparseMatrix&)(*A));
         PCG(*A, S, B, X, 0, 800, 1e-12, 0.0);

         // C. Recover state variable
         a.RecoverFEMSolution(X, b, delta_u);

         double gamma = 0.5;
         // while (true)
         // {
         //    tmp_gf = delta_u;
         //    tmp_gf *= gamma;
         //    tmp_gf += u;
         //    double min_u = tmp_gf.Min();
         //    if (min_u <= -1e-8)
         //    {
         //       gamma /= 2.0;
         //    }
         //    else
         //    {
         //       u = tmp_gf;
         //       break;
         //    }
         // }
         delta_u *= gamma;
         u += delta_u;
      }
      u_old = u;

      // 14. Send the solution by socket to a GLVis server.
      if (visualization)
      {
         sol_sock << "solution\n" << mesh << u << flush;
      }
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, exp(val));
}

double ReciprocalGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   if (val < 0)
   {
      return max_val;
   }
   else
   {
      return min(max_val, max(min_val, 1.0/val) );
   }
}

double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;

   if (r > r0)
   {
      return 0.0;
   }
   else
   {
      // return 0.0;
      return r0 + sqrt(r0*r0-r*r);
   }
}

   // for (int k = 1; k <= max_it; k++)
   // {
   //    // A. Assembly
   //    a.AddDomainIntegrator(new MassIntegrator(one));
   //    a.Assemble();

   //    LinearForm b(&fespace);
   //    GradientGridFunctionCoefficient grad_u_prev(&u_old);
   //    ScalarVectorProductCoefficient minus_alpha_grad_u_prev(-alpha, grad_u_prev);
   //    b.AddDomainIntegrator(new DomainLFGradIntegrator(minus_alpha_grad_u_prev));
   //    GridFunctionCoefficient u_prev(&u_old);
   //    ProductCoefficient minus_alpha_u_prev(-alpha, u_prev);
   //    b.AddDomainIntegrator(new DomainLFIntegrator(minus_alpha_u_prev));
   //    ProductCoefficient alpha_f(alpha, f);
   //    b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
   //    LogarithmGridFunctionCoefficient log_u_prev(u_old);
   //    b.AddDomainIntegrator(new DomainLFIntegrator(log_u_prev));
   //    b.Assemble();

   //    // B. Solve state equation
   //    a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);
   //    GSSmoother S((SparseMatrix&)(*A));
   //    PCG(*A, S, B, X, 0, 800, 1e-12, 0.0);

   //    // C. Recover state variable
   //    a.RecoverFEMSolution(X, b, u);

   //    for (int i = 0; i < u.Size(); i++)
   //    {
   //       if (ess_tdof_list.Find(i) > -1)
   //       {
   //          mfem::out << "u[i]     = " << u[i] << endl;
   //          mfem::out << "u_old[i] = " << u_old[i] << endl;
   //          continue;
   //       }
   //       u_old[i] = exp(u[i]);
   //    }
   // }
   // u = u_old;