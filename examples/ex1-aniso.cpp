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
//               ex1 -fa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//               ex1 -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
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

// Define the analytical solution and forcing terms / boundary conditions
typedef std::function<real_t(const Vector &, real_t)> TFunc;
typedef std::function<void(const Vector &, Vector &)> VecFunc;
typedef std::function<void(const Vector &, real_t, Vector &)> VecTFunc;
typedef std::function<void(const Vector &, DenseMatrix &)> MatFunc;

enum Problem
{
   SteadyDiffusion = 1,
   MFEMLogo,
   DiffusionRing,
   DiffusionRingGauss,
   DiffusionRingSine,
   BoundaryLayer,
   SteadyPeak,
   SteadyVaryingAngle,
   Sovinec,
   Umansky,
};

struct ProblemParams
{
   Problem prob;
   int nx, ny;
   real_t x0, y0, sx, sy;
   int order;
   real_t k, ks, ka;
   real_t t_0;
   real_t a;
   real_t c;
};

MatFunc GetKFun(const ProblemParams &params);
TFunc GetTFun(const ProblemParams &params);
TFunc GetFFun(const ProblemParams &params);

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   int iproblem = Problem::SteadyDiffusion;
   ProblemParams pars;
   pars.nx = 0;
   pars.ny = 0;
   pars.x0 = 0.;
   pars.y0 = 0.;
   pars.sx = 1.;
   pars.sy = 1.;
   pars.order = 1;
   pars.t_0 = 1.;
   pars.k = 1.;
   pars.ks = 1.;
   pars.ka = 0.;
   pars.a = 0.;
   pars.c = 1.;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   bool algebraic_ceed = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&iproblem, "-p", "--problem",
                  "Problem to solve:\n\t\t"
                  "1=sine diffusion\n\t\t"
                  "2=MFEM logo\n\t\t"
                  "3=diffusion ring\n\t\t"
                  "4=diffusion ring - Gauss source\n\t\t"
                  "5=diffusion ring - sine source\n\t\t"
                  "6=boundary layer\n\t\t"
                  "7=steady peak\n\t\t"
                  "8=steady varying angle\n\t\t"
                  "9=Sovinec\n\t\t"
                  "10=Umansky\n\t\t");
   args.AddOption(&pars.k, "-k", "--kappa",
                  "Heat conductivity");
   args.AddOption(&pars.ks, "-ks", "--kappa_sym",
                  "Symmetric anisotropy of the heat conductivity tensor");
   args.AddOption(&pars.a, "-a", "--heat_capacity",
                  "Heat capacity coefficient (0=indefinite problem)");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
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

   pars.prob = (Problem)iproblem;
   pars.order = order;

   Vector x_min(2), x_max(2);
   mesh.GetBoundingBox(x_min, x_max);
   pars.x0 = x_min(0);
   pars.y0 = x_min(1);
   pars.sx = x_max(0) - x_min(0);
   pars.sy = x_max(1) - x_min(1);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh.GetNE())/log(2.)/dim);
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
   //    the external boundary attributes from the mesh as essential (Dirichlet)
   //    and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      // Apply boundary conditions on all external boundaries:
      mesh.MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // mesh.MarkNamedBoundaries(set_name, ess_bdr)

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   auto fFun = GetFFun(pars);
   FunctionCoefficient fcoeff(fFun); //temperature rhs

   LinearForm b(&fespace);
   //ConstantCoefficient one(1.0);
   b.AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   b.Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.

   auto kFun = GetKFun(pars);
   MatrixFunctionCoefficient kcoeff(dim, kFun); //tensor conductivity
   ConstantCoefficient acoeff(pars.a); //heat capacity

   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(kcoeff));

   if (pars.a > 0)
   {
      a.AddDomainIntegrator(new MassIntegrator(acoeff));
   }

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 1000, 1e-8, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else
   {
      if (UsesTensorBasis(fespace))
      {
         if (algebraic_ceed)
         {
            ceed::AlgebraicSolver M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
         else
         {
            OperatorJacobiSmoother M(a, ess_tdof_list);
            PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
         }
      }
      else
      {
         CG(*A, B, X, 1, 400, 1e-12, 0.0);
      }
   }

   // 12. Recover the solution as a finite element grid function.
   a.RecoverFEMSolution(X, b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   /*ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);*/
   /*auto tFun = GetTFun(pars);
   FunctionCoefficient tcoeff(tFun); //temperature
   x.ProjectCoefficient(tcoeff);*/

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << endl;
      sol_sock << "window_title 'Solution'" << endl;
      sol_sock << "keys Rljmmc" << endl;
   }

   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}

MatFunc GetKFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   const real_t &ka = params.ka;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::BoundaryLayer:
      case Problem::SteadyPeak:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            kappa.Diag(k, ndim);
            kappa(0,0) *= ks;
            kappa(0,1) = +ka * k;
            kappa(1,0) = -ka * k;
            if (ndim > 2)
            {
               kappa(0,2) = +ka * k;
               kappa(2,0) = -ka * k;
            }
         };
      case Problem::MFEMLogo:
         MFEM_ABORT("N/A");
         break;
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            b(0) = (r>0.)?(-dx(1) / r):(1.);
            b(1) = (r>0.)?(+dx(0) / r):(0.);

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Sovinec:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            b = 0.;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            //const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_x = M_PI * sin(M_PI * dx(0)) * cos(M_PI * dx(1));
            const real_t psi_y = M_PI * cos(M_PI * dx(0)) * sin(M_PI * dx(1));
            const real_t psi_norm = hypot(psi_x, psi_y);
            if (psi_norm > 0.)
            {
               b(0) = -psi_y / psi_norm;
               b(1) = +psi_x / psi_norm;
            }
            else
            {
               b = 0.;
            }

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
      case Problem::Umansky:
         return [=](const Vector &x, DenseMatrix &kappa)
         {
            const int ndim = x.Size();
            Vector b(ndim);
            const real_t s = hypot(sx, sy);
            b(0) = +sx / s;
            b(1) = +sy / s;

            kappa.Diag(ks * k, ndim);
            if (ks != 1.)
            {
               AddMult_a_VVt((1. - ks) * k, b, kappa);
            }
         };
   }
   return MatFunc();
}

TFunc GetTFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &t_0 = params.t_0;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;
   const real_t hx = sx / params.nx;
   const real_t hy = sy / params.ny;
   const real_t &order = params.order;

   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
         return [=](const Vector &x, real_t t) -> real_t
         {
            const int ndim = x.Size();
            real_t t0 = t_0 * sin(M_PI*x(0)) * sin(M_PI*x(1));
            if (ndim > 2)
            {
               t0 *= sin(M_PI*x(2));
            }

            if (a <= 0.) { return t0; }

            Vector ddT((ndim<=2)?(2):(4));
            ddT(0) = -t_0 * M_PI*M_PI * sin(M_PI*x(0)) * sin(M_PI*x(1));//xx,yy
            ddT(1) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * cos(M_PI*x(1));//xy
            if (ndim > 2)
            {
               ddT(0) *= sin(M_PI*x(2));//xx,yy,zz
               ddT(1) *= sin(M_PI*x(2));//xy
               //xz
               ddT(2) = +t_0 * M_PI*M_PI * cos(M_PI*x(0)) * sin(M_PI*x(1)) * cos(M_PI*x(2));
               //yz
               ddT(3) = +t_0 * M_PI*M_PI * sin(M_PI*x(0)) * cos(M_PI*x(1)) * cos(M_PI*x(2));

            }

            DenseMatrix kappa;
            kFun(x, kappa);

            real_t div = -(kappa(0,0) + kappa(1,1)) * ddT(0) - (kappa(0,1) + kappa(1,0)) * ddT(1);
            if (ndim > 2)
            {
               div += -kappa(2,2) * ddT(0) - (kappa(0,2) + kappa(2,0)) * ddT(2)
               - (kappa(1,2) + kappa(2,1)) * ddT(3);
            }
            return t0 - div / a * t;
         };
      case Problem::MFEMLogo:
         MFEM_ABORT("N/A");
         break;
      case Problem::DiffusionRing:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.25;
            constexpr real_t r1 = 0.35;
            constexpr real_t dr01 = 0.025;
            constexpr real_t theta0 = 11./12. * M_PI;
            constexpr real_t dtheta0 = 1./48. * M_PI;

            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t theta = fabs(atan2(dx(1), dx(0)));

            if (r < r0 - dr01 || r > r1 + dr01 || theta < theta0 - dtheta0)
            {
               return 0.;
            }

            const real_t dr = min(r - r0 + dr01, r1 + dr01 - r) / dr01;
            const real_t dth = (theta - theta0 + dtheta0) / dtheta0;
            return min(1., dr) * min(1., dth) * t_0;
         };
      case Problem::DiffusionRingGauss:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.025;
            constexpr real_t x_c = 0.15;

            const real_t dx_l = x(0) - (x0       + x_c  * sx);
            const real_t dx_r = x(0) - (x0 + (1. - x_c) * sx);
            const real_t dy = x(1) - (y0 + 0.5*sy);
            const real_t r_l = hypot(dx_l, dy);
            const real_t r_r = hypot(dx_r, dy);

            return - exp(- r_l*r_l/(r0*r0)) + exp(- r_r*r_r/(r0*r0));
         };
      case Problem::DiffusionRingSine:
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t r0 = 0.05;
            constexpr real_t w0 = 16.;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            if (r <= 0.) { return 0.; }
            const real_t th = atan2(dx(1), dx(0));

            const real_t C = w0 / r;
            return 1. / (1. + t * k * C*C / a) * cos(w0*th) * sin(M_PI * r/r0);
         };
      case Problem::BoundaryLayer:
         // C. Vogl, I. Joseph and M. Holec, Mesh refinement for anisotropic
         // diffusion in magnetized plasmas, Computers and Mathematics with
         // Applications, 145, pp. 159-174 (2023).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t k_para = M_PI*M_PI * k * ks;
            const real_t k_perp = k;
            const real_t k_frac = sqrt(k_para/k_perp);
            const real_t denom = 1. + exp(-k_frac);
            const real_t e_down = exp(-k_frac * x(1));
            const real_t e_up = exp(- k_frac * (1. - x(1)));
            return - (e_down + e_up) / denom * sin(M_PI * x(0));
         };
      case Problem::SteadyPeak:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            return x(0)*x(1) * pow(arg, s);
         };
      case Problem::SteadyVaryingAngle:
         // B. van Es, B. Koern and Hugo de Blank, DISCRETIZATIONMETHODS
         // FOR EXTREMELY ANISOTROPIC DIFFUSION. In 7th International
         // Conference on Computational Fluid Dynamics (ICCFD 2012) (pp.
         // ICCFD7-1401)
         return [=](const Vector &x, real_t t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            return 1. - r*r*r;
         };
      case Problem::Sovinec:
         // C. R. Sovinec et al., Nonlinear magnetohydrodynamics simulation
         // using high-order finite elements. Journal of Computational Physics,
         // 195, pp. 355–386 (2004).
         return [=](const Vector &x, real_t t) -> real_t
         {
            const real_t &kappa_perp = k * ks;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return psi / kappa_perp;
         };
      case Problem::Umansky:
         // M. V. Umansky, M. S. Day and T. D. Rognlien, On Numerical Solution
         // of Strongly Anisotropic Diffusion Equation on Misaligned Grids,
         // Numerical Heat Transfer, Part B: Fundamentals, 47(6), pp. 533-554
         // (2005).
         // Adopted from plasma-dev:miniapps/plasma/transport2d.cpp
         return [=](const Vector &x, real_t t) -> real_t
         {
            if (x(0) < hx && x(1) < hy)
            {
               return 0.5 * (1.0 -
                             std::pow(1.0 - x(0) / hx, order) +
                             std::pow(1.0 - x(1) / hy, order));
            }
            else if (x(0) > sx - hx && x(1) > sy - hy)
            {
               return 0.5 * (1.0 +
                             std::pow(1.0 - (sx - x(0)) / hx, order) -
                             std::pow(1.0 - (sy - x(1)) / hy, order));
            }
            // else if (x_[0] > Lx_ - hx_ || x_[1] < hy_)
            else if (hx * (x(1) + hy) < hy * x(0))
            {
               return 1.0;
            }
            // else if (x_[0] < hx_ || x_[1] > Ly_ - hy_)
            else if (hx * x(1) > hy * (x(0) + hx))
            {
               return 0.0;
            }
            else
            {
               return 0.5 * (1.0 + std::tanh(M_LN10 * (x(0) / hx - x(1) / hy)));
            }
         };
   }
   return TFunc();
}

TFunc GetFFun(const ProblemParams &params)
{
   const real_t &k = params.k;
   const real_t &ks = params.ks;
   //const real_t &ka = params.ka;
   const real_t &a = params.a;
   const real_t &x0 = params.x0;
   const real_t &y0 = params.y0;
   const real_t &sx = params.sx;
   const real_t &sy = params.sy;

   auto TFun = GetTFun(params);
   auto kFun = GetKFun(params);

   switch (params.prob)
   {
      case Problem::SteadyDiffusion:
      case Problem::MFEMLogo:
      case Problem::DiffusionRing:
      case Problem::DiffusionRingGauss:
      case Problem::DiffusionRingSine:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t T = TFun(x, 0);
            return ((a > 0.)?(a):(1.)) * T;
         };
      case Problem::BoundaryLayer:
      case Problem::Umansky:
         return [=](const Vector &x, real_t) -> real_t
         {
            return 0.;
         };
      case Problem::SteadyPeak:
         return [=](const Vector &x, real_t) -> real_t
         {
            DenseMatrix kappa;
            kFun(x, kappa);
            constexpr real_t s = 10.;
            const real_t arg = sin(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_x = M_PI * cos(M_PI * x(0)) * sin(M_PI * x(1));
            const real_t arg_y = M_PI * cos(M_PI * x(1)) * sin(M_PI * x(0));
            const real_t T_xx = x(1) * pow(arg, s-2) * (2.*s * arg_x * arg + x(0) * s * ((s-1) * arg_x*arg_x - M_PI*M_PI * arg*arg));
            const real_t T_yy = x(0) * pow(arg, s-2) * (2.*s * arg_y * arg + x(1) * s * ((s-1) * arg_y*arg_y - M_PI*M_PI * arg*arg));
            return -kappa(0,0) * T_xx - kappa(1,1) * T_yy;
         };
      case Problem::SteadyVaryingAngle:
         return [=](const Vector &x, real_t) -> real_t
         {
            const real_t kappa_r = ks * k;
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t r = hypot(dx(0), dx(1));
            const real_t T_rr = - 9. * r;
            return -kappa_r * T_rr;
         };
      case Problem::Sovinec:
         return [=](const Vector &x, real_t) -> real_t
         {
            Vector dx(x);
            dx(0) -= x0 + 0.5*sx;
            dx(1) -= y0 + 0.5*sy;

            const real_t psi = cos(M_PI * dx(0)) * cos(M_PI * dx(1));
            return 2.*M_PI*M_PI * psi;
         };
   }
   return TFunc();
}
