//                                MFEM Example 14
//
// Compile with: make ex14
//
// Sample runs:  ex14 -m ../data/inline-quad.mesh -o 0
//               ex14 -m ../data/star.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 4 -o 2
//               ex14 -m ../data/star-mixed.mesh -r 2 -o 2 -k 0 -e 1
//               ex14 -m ../data/escher.mesh -s 1
//               ex14 -m ../data/fichera.mesh -s 1 -k 1
//               ex14 -m ../data/fichera-mixed.mesh -s 1 -k 1
//               ex14 -m ../data/square-disc-p2.vtk -r 3 -o 2
//               ex14 -m ../data/square-disc-p3.mesh -r 2 -o 3
//               ex14 -m ../data/square-disc-nurbs.mesh -o 1
//               ex14 -m ../data/disc-nurbs.mesh -r 3 -o 2 -s 1 -k 0
//               ex14 -m ../data/pipe-nurbs.mesh -o 1
//               ex14 -m ../data/inline-segment.mesh -r 5
//               ex14 -m ../data/amr-quad.mesh -r 3
//               ex14 -m ../data/amr-hex.mesh
//               ex14 -m ../data/fichera-amr.mesh
//               ex14 -pa -r 1 -o 3
//               ex14 -pa -r 1 -o 3 -m ../data/fichera.mesh
//
// Device sample runs:
//               ex14 -pa -r 2 -d cuda -o 3
//               ex14 -pa -r 2 -d cuda -o 3 -m ../data/fichera.mesh
//
// Description:  This example code demonstrates the use of MFEM to define a
//               discontinuous Galerkin (DG) finite element discretization of
//               the Poisson problem -Delta u = 1 with homogeneous Dirichlet
//               boundary conditions. Finite element spaces of any order,
//               including zero on regular grids, are supported. The example
//               highlights the use of discontinuous spaces and DG-specific face
//               integrators.
//
//               We recommend viewing examples 1 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
double E_exact(const Vector &);
double f_exact(const Vector &);
int dim;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = -1;
   int order = 1;
   real_t sigma = -1.0;
   real_t kappa = -1.0;
   real_t eta = 0.0;
   bool pa = false;
   bool visualization = 1;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the three DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the three DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eta, "-e", "--eta", "BR2 penalty parameter.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
    
   if (kappa < 0)
   {
      kappa = 3*(order+1)*(order+2);
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   //    NURBS meshes are projected to second order meshes.
   Mesh mesh(mesh_file);
   dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
   //    we choose it to be the largest number that gives a final mesh with no
   //    more than 50,000 elements.
   
   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }

   // 5. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   //const auto bt = pa ? BasisType::GaussLobatto : BasisType::GaussLegendre;
   DG_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   cout << "Number of unknowns: " << fespace.GetVSize() << endl;
    
    
    Array<int> ess_tdof_list;
    if (mesh.bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

   // 6. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system.
   LinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   FunctionCoefficient f(f_exact);
   FunctionCoefficient E(E_exact);


   b.AddDomainIntegrator(new DomainLFIntegrator(f));
   b.AddBdrFaceIntegrator(
      new DGDirichletLFIntegrator(zero, one, sigma, kappa));
   b.Assemble();

   // 7a. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero.
   GridFunction x(&fespace);
   x.ProjectCoefficient(E);
    
    const IntegrationRule* irs[Geometry::NumGeom];
    for (int i = 0; i < Geometry::NumGeom; i++)
    {
        if (i == 4)
        {
            // Tet Int Rule
            irs[i] = &(IntRules.Get(i, 10));
        }else
        {
            // Everything else
            irs[i] = &(IntRules.Get(i, 16));
        }
    }

    // 7b. Compute and print the L^2 norm of the error.
   cout << "\n Initial || E_h - E ||_{L^2} = " << x.ComputeL2Error(E, irs) << '\n' << endl;

   // 8. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator and the interior and boundary DG face integrators.
   //    Note that boundary conditions are imposed weakly in the form, so there
   //    is no need for dof elimination. After assembly and finalizing we
   //    extract the corresponding sparse matrix A.
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   if (eta > 0)
   {
      MFEM_VERIFY(!pa, "BR2 not yet compatible with partial assembly.");
      a.AddInteriorFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
      a.AddBdrFaceIntegrator(new DGDiffusionBR2Integrator(fespace, eta));
   }
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.Assemble();
    
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B); // x = x_ess + X
    
    
    
//    // 11. Solve the linear system A X = B.
//    if (!pa)
//    {
// #ifndef MFEM_USE_SUITESPARSE
//       // Use a simple symmetric Gauss-Seidel preconditioner with PCG.   MADE CHANGE HERE TO ITER #
//       GSSmoother M((SparseMatrix&)(*A));
//       PCG(*A, M, B, X, 1, 5000, 1e-12, 0.0);
// #else
//       // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//       UMFPackSolver umf_solver;
//       umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//       umf_solver.SetOperator(*A);
//       umf_solver.Mult(B, X);
// #endif
//    }
//    else // Jacobi preconditioning in partial assembly mode
//    {
//        CG(*A, B, X, 1, 400, 1e-12, 0.0);
//    }
    
    // 11. Solve the linear system A X = B.
     if (!pa)
     {
         #ifndef MFEM_USE_SUITESPARSE
         // Use GMRES with a Gauss-Seidel preconditioner
         GSSmoother M((SparseMatrix&)(*A));  // Gauss-Seidel preconditioner
         GMRESSolver gmres;                  // Create GMRES solver
         gmres.SetKDim(9000);                 // Set the Krylov subspace dimension (restart size)
         gmres.SetMaxIter(50000);              // Set maximum iterations
         gmres.SetRelTol(1e-12);             // Set relative tolerance
         gmres.SetPrintLevel(1);             // Print convergence information
         gmres.SetPreconditioner(M);         // Set the preconditioner
         gmres.SetOperator(*A);              // Set the operator
         gmres.Mult(B, X);                   // Solve the system
         #else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
         #endif
     }

         
    
    // 12. Recover the solution as a finite element grid function.
    a.RecoverFEMSolution(X, b, x);
    
    
   //a.Finalize();
   // 9. Define a simple symmetric Gauss-Seidel preconditioner and use it to
   //    solve the system Ax=b with PCG in the symmetric case, and GMRES in the
   //    non-symmetric one. (Note that tolerances are squared: 1e-12 corresponds
   //    to a relative tolerance of 1e-6).
   //
   //    If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//   if (pa)
//   {
//      MFEM_VERIFY(sigma == -1.0,
//                  "The case of PA with sigma != -1 is not yet supported.");
//      CG(a, b, x, 1, 500, 1e-12, 0.0);
//   }
//   else
//   {
//      const SparseMatrix &A = a.SpMat();
//#ifndef MFEM_USE_SUITESPARSE
//      GSSmoother M(A);
//      if (sigma == -1.0)
//      {
//         PCG(A, M, b, x, 1, 5000, 1e-12, 0.0);
//      }
//      else
//      {
//         GMRES(A, M, b, x, 1, 500, 10, 1e-12, 0.0);
//      }
//#else
//      UMFPackSolver umf_solver;
//      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//      umf_solver.SetOperator(A);
//      umf_solver.Mult(b, x);
//#endif
//   }
    
//    // 11. Solve the linear system A X = B.
//     if (!pa)
//     {
//         #ifndef MFEM_USE_SUITESPARSE
//         // Use GMRES with a Gauss-Seidel preconditioner
//         const SparseMatrix &A = a.SpMat();
//         Vector B, X;
//         GSSmoother M(A);  // Gauss-Seidel preconditioner
//         GMRESSolver gmres;                  // Create GMRES solver
//         gmres.SetKDim(200);                 // Set the Krylov subspace dimension (restart size)
//         gmres.SetMaxIter(200);              // Set maximum iterations
//         gmres.SetRelTol(1e-12);             // Set relative tolerance
//         gmres.SetPrintLevel(1);             // Print convergence information
//         gmres.SetPreconditioner(M);         // Set the preconditioner
//         gmres.SetOperator(A);              // Set the operator
//         gmres.Mult(B, X);                   // Solve the system
//         #else
//         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//         UMFPackSolver umf_solver;
//         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//         umf_solver.SetOperator(*A);
//         umf_solver.Mult(B, X);
//         #endif
//     }

    
    cout << "\n Initial || E_h - E ||_{L^2} = " << x.ComputeL2Error(E, irs) << '\n' << endl;


   // 10. Save the refined mesh and the solution. This output can be viewed
   //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 11. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}


double E_exact(const Vector &x)
{
    double E_out;

    if (dim == 2)
    {
        E_out = x(0)*(1. - x(0))*x(1)*(1. - x(1));
    }
    
    if (dim == 3)
    {
        E_out = x(0)*(1. - x(0)) * x(1)*(1. - x(1)) * x(2)*(1. - x(2));
    }
    
    if (dim == 4)
    {
        E_out = x(0)*(1. - x(0)) * x(1)*(1. - x(1)) * x(2)*(1. - x(2)) * x(3)*(1. - x(3));
    }
    
    return E_out;
}



double f_exact(const Vector &x)
{
    double f_out;

    if (dim == 2)
    {
        f_out = 2.0*(x(0)*(1. - x(0))+x(1)*(1. - x(1)));
    }
    
    if (dim == 3)
    {
        f_out = 2.0*((x(1)*(1. - x(1))* x(2)*(1. - x(2))) + (x(0)*(1. - x(0))* x(2)*(1. - x(2))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1))));
    }
    
    if (dim == 4)
    {
        f_out = 2.0*((x(1)*(1. - x(1))* x(2)*(1. - x(2)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(2)*(1. - x(2)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1)) * x(2)*(1. - x(2))));
    }
    
    return f_out;
}


