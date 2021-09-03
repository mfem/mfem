//                         Test of ZZ error estimator
//
// Compile with: make ZZ_test
//

#include "mfem.hpp"
#include <fstream>

// #include "exact.hpp"

using namespace std;
using namespace mfem;

double lshape_exsol(const Vector &p);
void   lshape_exgrad(const Vector &p, Vector &grad);
double lshape_laplace(const Vector &p);

double sinsin_exsol(const Vector &p);
void   sinsin_exgrad(const Vector &p, Vector &grad);
double sinsin_laplace(const Vector &p);

int dim;
const char* keys = "Rjlmc*******";

int main(int argc, char *argv[])
{
   // Parse command-line options.
   int problem = 0;
   int order = 1;
   double ref_threshold = 0.8;
//    const char *elemerr_file = "elemerr.txt";
   int nc_limit = 1;
   const char *device_config = "cpu";
   bool visualization = false;
   int which_estimator = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type: 0 = canonical L-shaped solution, 1 = sinusoid.");
   args.AddOption(&order, "-o", "--order",
                  "Initial mesh finite element order (polynomial degree).");
   args.AddOption(&ref_threshold, "-rt", "--ref-threshold",
                  "Refine elements with error larger than threshold * max_error.");
   args.AddOption(&nc_limit, "-nc", "--nc-limit",
                  "Set maximum difference of refinement levels of adjacent elements.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&which_estimator, "-est", "--estimator",
                  "Which estimator to use: "
                  "0 = ZZ, 1 = Kelly. Defaults to ZZ.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Enable hardware devices such as GPUs, and programming models such as
   // CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   const char *mesh_file;
   // if (problem == 0)
   // {
      mesh_file = "l-shape-benchmark.mesh";
   // }

   // 2. Read the (serial) mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();
   mesh.EnsureNCMesh();
   mesh.UniformRefinement(); // ZZ doesn't work properly on the initial L-shaped mesh

   // Define a finite element space on the mesh.
   H1_FECollection fec(order, dim);
   L2_FECollection l2fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // Define the solution vector x as a finite element grid function
   // corresponding to fespace.
   GridFunction x(&fespace);

   // Define exact solutions
   FunctionCoefficient *exsol=nullptr;
   VectorFunctionCoefficient *exgrad=nullptr;
   FunctionCoefficient *rhs=nullptr;
   ConstantCoefficient one(1.0);

   switch (problem)
   {
      case 1:
      {
         exsol = new FunctionCoefficient(sinsin_exsol);
         exgrad = new VectorFunctionCoefficient(dim, sinsin_exgrad);
         rhs = new FunctionCoefficient(sinsin_laplace);
         break;
      }
      default:
      case 0:
      {
         exsol = new FunctionCoefficient(lshape_exsol);
         exgrad = new VectorFunctionCoefficient(dim, lshape_exgrad);
         rhs = new FunctionCoefficient(lshape_laplace);
         break;
      }
   }

   // Set up the linear form b(.) and the bilinear form a(.,.).
   LinearForm b(&fespace);
   BilinearForm a(&fespace);

   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs));
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   // All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");   

   // Connect to GLVis.
   socketstream sol_sock, ord_sock, dbg_sock[3], err_sock;

   ostringstream file_name;
   file_name << "conv_order" << order << ".csv";
   ofstream conv(file_name.str().c_str());
//    std::ofstream elemerr(elemerr_file);

   if (visualization)
   {
      cout << "\n Press enter to advance... " << endl;
   }

   cout << setw(4) << "\nRef." << setw(12) << "DOFs" << setw(21) << "H^1_0 error" << setw(21) << "error estimate" << setw(18) << "H^1_0 rate" << setw(18) << "estimator rate" <<  endl;
   conv << "DOFs " << ", " << "H^1_0 error" << ", " << "error estimate" << endl;

   double old_num_dofs = 0.0;
   double old_H10_error = 0.0;
   double old_ZZ_error = 0.0;
   const int max_dofs = 50000;
   for (int it = 0; ; it++)
   {
      int num_dofs = fespace.GetTrueVSize();

      // Set Dirichlet boundary values in the GridFunction x.
      // Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      x = 0.0;
      x.ProjectBdrCoefficient(*exsol, ess_bdr);
      Array<int> ess_tdof_list;
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   
      // Solve for the current mesh:
      b.Assemble();
      a.Assemble();
      a.Finalize();
      OperatorPtr A;
      Vector B, X;
   
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 2000, 1e-30, 0.0);

      a.RecoverFEMSolution(X, b, x);

      // Calculate the total error in the H^1_0 norm.
      double H10_error = x.ComputeGradError(exgrad);
      DiffusionIntegrator di;

      ErrorEstimator* estimator{nullptr};
      switch (which_estimator)
      {
         case 1:
         {
            int flux_order = 4;
            estimator = new NewZienkiewiczZhuEstimator(di, x, flux_order);
            break;
         }

         case 2:
         {
            auto flux_fes = new FiniteElementSpace(&mesh, &l2fec, dim);
            estimator = new KellyErrorEstimator(di, x, flux_fes);
            break;
         }

         default:
            std::cout << "Unknown estimator. Falling back to ZZ." << std::endl;
         case 0:
         {
            auto flux_fes = new FiniteElementSpace(&mesh, &fec, dim);
            estimator = new ZienkiewiczZhuEstimator(di, x, flux_fes);
            break;
         }
      }
      const Vector &zzerr = estimator->GetLocalErrors();
      double ZZ_error = estimator->GetTotalError();

      // estimate convergence rate
      double H10_rate = 0.0;
      double ZZ_rate = 0.0;
      if (old_H10_error > 0.0)
      {
          H10_rate = log(H10_error/old_H10_error) / log(old_num_dofs/num_dofs);
          ZZ_rate  = log(ZZ_error/old_ZZ_error)   / log(old_num_dofs/num_dofs);
      }

      cout << setw(4) << it << setw(12) << num_dofs << setw(21) << H10_error << setw(21) << ZZ_error << setw(18) << H10_rate << setw(18) << ZZ_rate << endl;

      // Send solution by socket to the GLVis server.
      if (visualization)
      {
         cin.get();
         const char vishost[] = "localhost";
         const int  visport   = 19916;
         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x;
         sol_sock << "keys ARjlm\n";
      }

      if (num_dofs > max_dofs)
      {
         cout << "\n Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // Save dofs and error for convergence plot
      conv << num_dofs << ", " << H10_error << ", " << ZZ_error << endl;

    //   for (int i = 0; i < mesh.GetNE(); i++)
    //   {
    //      elemerr << sqrt(elemError[i]) << ' ';
    //   }
    //   elemerr << endl;


      Array<Refinement> refinements;
      double err_max = zzerr.Max();
      for (int i = 0; i < mesh.GetNE(); i++)
      {
         if (zzerr[i] > ref_threshold * err_max)
         {
             refinements.Append(Refinement(i, 7));
         }
      }
      mesh.GeneralRefinement(refinements, -1, nc_limit);

      old_num_dofs = double(num_dofs);
      old_H10_error = H10_error;
      old_ZZ_error = ZZ_error;

      // Update the space, interpolate the solution.
      fespace.Update();
      a.Update();
      b.Update();
      x.Update();

      // Free the used memory.
      delete estimator;
   }

   // Free the used memory.
   delete exsol;
   delete exgrad;
   delete rhs;
   return 0;
}


// L-shape domain problem exact solution (2D)

double lshape_exsol(const Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y);
   double a = atan2(y, x);
   if (a < 0) { a += 2*M_PI; }
   return pow(r, 2.0/3.0) * sin(2.0*a/3.0);
}

void lshape_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double a = atan2(y, x);
   if (a < 0) { a += 2*M_PI; }
   double theta23 = 2.0/3.0*a;
   double r23 = pow(x*x + y*y, 2.0/3.0);
   grad(0) = 2.0/3.0*x*sin(theta23)/(r23) - 2.0/3.0*y*cos(theta23)/(r23);
   grad(1) = 2.0/3.0*y*sin(theta23)/(r23) + 2.0/3.0*x*cos(theta23)/(r23);
}

double lshape_laplace(const Vector &p)
{
   return 0;
}

double sinsin_exsol(const Vector &p)
{
   double x = p(0), y = p(1);
   return sin(M_PI * x) * sin(M_PI * y);
}

void sinsin_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   grad(0) = M_PI * cos(M_PI * x) * sin(M_PI * y);
   grad(1) = M_PI * sin(M_PI * x) * cos(M_PI * y);
}

double sinsin_laplace(const Vector &p)
{
   double x = p(0), y = p(1);
   return 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}