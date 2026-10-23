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

// Exact solution, E, and r.h.s., f. See below for implementation.
double E_exact(const Vector &);
double f_exact(const Vector &);
double freq = 1.0, kappa;
int dim;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "../data/star.mesh";
    int order = 1;
    bool static_cond = false;
    bool pa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    int ref_levels = 0;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&ref_levels, "-ref", "--ref-levels", "");
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
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    dim = mesh->Dimension();
    int sdim = mesh->SpaceDimension();
    
    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 50,000
    //    elements.
    
    std::cout << "ref_levels = " << ref_levels << std::endl;
    for (int l = 0; l < ref_levels; l++)
    {
        mesh->UniformRefinement();
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
    else if (mesh->GetNodes())
    {
        fec = mesh->GetNodes()->OwnFEC();
        delete_fec = false;
        cout << "Using isoparametric FEs: " << fec->Name() << endl;
    }
    else
    {
        fec = new H1_FECollection(order = 1, dim);
        delete_fec = true;
    }
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
    cout << "Number of finite element unknowns: "
    << fespace->GetTrueVSize() << endl;
    
    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet) and
    //    converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size())
    {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
    
    // 7. Set up the linear form b(.) which corresponds to the right-hand side
    //    of the FEM linear system, which in this case is (f,phi_i) where f is
    //    given by the function f_exact and phi_i are the basis functions in the
    //    finite element fespace.
    FunctionCoefficient f(f_exact);
    LinearForm *b = new LinearForm(fespace);
    b->AddDomainIntegrator(new DomainLFIntegrator(f));
    b->Assemble();
    

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x by projecting the exact
    //    solution. Note that only values from the boundary edges will be used
    //    when eliminating the non-homogeneous boundary condition to modify the
    //    r.h.s. vector b.
    GridFunction x(fespace);
    ConstantCoefficient one(1.0);
    FunctionCoefficient E(E_exact);
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

 

    // 8a. Compute and print the L^2 norm of the error.
   cout << "\n Initial || E_h - E ||_{L^2} = " << x.ComputeL2Error(E, irs) << '\n' << endl;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(one)); // -Delta u
    
   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
    
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B); // x = x_ess + X
   // A*X = f - A*x_ess = B
   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.   MADE CHANGE HERE TO ITER #
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // Jacobi preconditioning in partial assembly mode
   {
       CG(*A, B, X, 1, 400, 1e-12, 0.0);
   }
    
//    // 11. Solve the linear system A X = B.
//     if (!pa)
//     {
//     #ifndef MFEM_USE_SUITESPARSE
//     // Use GMRES with a Gauss-Seidel preconditioner
//     GSSmoother M((SparseMatrix&)(*A));  // Gauss-Seidel preconditioner
//     GMRESSolver gmres;                  // Create GMRES solver
//     gmres.SetKDim(200);                 // Set the Krylov subspace dimension (restart size)
//     gmres.SetMaxIter(200);              // Set maximum iterations
//     gmres.SetRelTol(1e-12);             // Set relative tolerance
//     gmres.SetPrintLevel(1);             // Print convergence information
//     gmres.SetPreconditioner(M);         // Set the preconditioner
//     gmres.SetOperator(*A);              // Set the operator
//     gmres.Mult(B, X);                   // Solve the system
//     #else
//     // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
//     UMFPackSolver umf_solver;
//     umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//     umf_solver.SetOperator(*A);
//     umf_solver.Mult(B, X);
//     #endif
//     }
//     else // Jacobi preconditioning in partial assembly mode
//     {
//     GMRESSolver gmres;                  // Create GMRES solver
//     gmres.SetKDim(400);                 // Set the Krylov subspace dimension (restart size)
//     gmres.SetMaxIter(400);              // Set maximum iterations
//     gmres.SetRelTol(1e-12);             // Set relative tolerance
//     gmres.SetPrintLevel(1);             // Print convergence information
//     gmres.SetOperator(*A);              // Set the operator
//     gmres.Mult(B, X);                   // Solve the system
//     }


   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);
    
   // 13. Compute and print the L^2 norm of the error.
   cout << "\n|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E, irs) << '\n' << endl;


   // 14. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);

   // 15. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }
    
   // 16. Free the used memory.
   if(delete_fec)
   {
       delete a;
       delete b;
       delete fespace;
       delete fec;
       delete mesh;
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
        //f_out = -1.0*(((2.+12.*x(0)*(x(0) - 1.))*x(1)*x(1)*(1.-x(1))*(1. - x(1)))  + ((2.+12.*x(1)*(x(1) - 1.))*x(0)*x(0)*(1. - x(0))*(1. - x(0))));

    }
    
    if (dim == 3)
    {
        //f_out = 2.0*(x(0)*(1. - x(0)) + x(1)*(1. - x(1)) + x(2)*(1. - x(2)));
        f_out = 2.0*((x(1)*(1. - x(1))* x(2)*(1. - x(2))) + (x(0)*(1. - x(0))* x(2)*(1. - x(2))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1))));
    }
    
    if (dim == 4)
    {
        f_out = 2.0*((x(1)*(1. - x(1))* x(2)*(1. - x(2)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(2)*(1. - x(2)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1)) * x(3)*(1. - x(3))) + (x(0)*(1. - x(0))* x(1)*(1. - x(1)) * x(2)*(1. - x(2))));
    }
    
    return f_out;
}

