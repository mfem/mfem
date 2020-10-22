//                       MFEM
//
// Compile with: make shift
//
// Sample runs:  shift
// make shift;./shift -m ../data/inline-quad.mesh  -rs 2

#include "../mfem.hpp"
#include <fstream>
#include <iostream>
#include "shift.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   /// 1. Parse command-line options.
    const char *mesh_file = "../data/square-disc.mesh";
    int order = 2;
    bool static_cond = false;
    bool pa = false;
    const char *device_config = "cpu";
    bool visualization = true;
    int ser_ref_levels = 0;

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
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");

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
    for (int lev = 0; lev < ser_ref_levels; lev++) { mesh.UniformRefinement(); }

    // 5. Define a finite element space on the mesh. Here we use continuous
    //    Lagrange finite elements of the specified order. If order < 1, we
    //    instead use an isoparametric/isogeometric space.
    FiniteElementCollection *fec_mesh;
    if (order > 0)
    {
       fec_mesh = new H1_FECollection(order, dim);
    }
    else
    {
       fec_mesh = new H1_FECollection(order = 1, dim);
    }
    FiniteElementSpace *fespace_mesh = new FiniteElementSpace(&mesh, fec_mesh, dim);
    mesh.SetNodalFESpace(fespace_mesh);
    GridFunction x_mesh(fespace_mesh);
    mesh.SetNodalGridFunction(&x_mesh);

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
    else
    {
       fec = new H1_FECollection(order = 1, dim);
       delete_fec = true;
    }
    FiniteElementSpace fespace(&mesh, fec);
    cout << "Number of finite element unknowns: "
         << fespace.GetTrueVSize() << endl;

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    GridFunction x(&fespace);
    x = 0.0;

    FunctionCoefficient dist_fun_coef(dist_fun);
    x.ProjectCoefficient(dist_fun_coef);

    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << mesh << x << flush;
       sol_sock << "window_title 'Distance function'\n"
                << "window_geometry "
                << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
                << "keys Rjmpc" << endl;
    }

    // optimize the mesh
    optimize_mesh_with_distfun(mesh, x_mesh, x);
    // project dist_fun at original smoothed mesh
    x.ProjectCoefficient(dist_fun_coef);

    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << mesh << x << flush;
       sol_sock << "window_title 'Distance function'\n"
                << "window_geometry "
                << 0 << " " << 0 << " " << 350 << " " << 350 << "\n"
                << "keys Rjmpc" << endl;
    }

    int max_attr     = mesh.attributes.Max();
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

    // Set trim flag based on the distance field
    Array<int> trim_flag(mesh.GetNE());
    trim_flag = 0;
    Vector vals;
    for (int i = 0; i < mesh.GetNE(); i++) {
        ElementTransformation *Tr = mesh.GetElementTransformation(i);
        const IntegrationRule &ir =
           IntRulesLo.Get(mesh.GetElementBaseGeometry(i), 4*Tr->OrderJ());
        x.GetValues(i, ir, vals);
        double minv = vals.Min();
        if (minv < 0) {
            mesh.SetAttribute(i, max_attr+1);
            trim_flag[i] = 1;
        }
    }

    // Trim the mesh and define a gridfunction for dist_fun
    Mesh *mesh_t = trim_mesh(mesh, trim_flag);
    FiniteElementSpace fespace_t(mesh_t, fec);
    GridFunction x_t(&fespace_t);
    x_t.ProjectCoefficient(dist_fun_coef);

    // Set nodal gridfunction for trimmed mesh
    FiniteElementSpace *fespace_mesh_t = new FiniteElementSpace(mesh_t, fec_mesh, dim);
    mesh_t->SetNodalFESpace(fespace_mesh_t);
    GridFunction x_mesh_t(fespace_mesh_t);
    mesh_t->SetNodalGridFunction(&x_mesh_t);

    // Copy nodal optimized mesh from original mesh to trimmed mesh.
    int recv = 0;
    Array<int> vdofs_donor, vdofs_recv;
    for (int i = 0; i < mesh.GetNE(); i++) {
        if (trim_flag[i] != 1) {
            Vector x_donor;
            x_mesh.FESpace()->GetElementVDofs(i, vdofs_donor);
            x_mesh.GetSubVector(vdofs_donor, x_donor);
            x_mesh_t.FESpace()->GetElementVDofs(recv, vdofs_recv);
            x_mesh_t.SetSubVector(vdofs_recv, x_donor);
            recv++;
        }
    }
    x_t.ProjectCoefficient(dist_fun_coef);


    // Visualize stuff
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << *mesh_t << x_t << flush;
       sol_sock << "window_title 'Trim2'\n"
                << "window_geometry "
                << 350 << " " << 0 << " " << 350 << " " << 350 << "\n"
                << "keys Rjmpc" << endl;
    }


    // Get the distance field vector
    GridFunction x_dx_t(&fespace_t), x_dy_t(&fespace_t),
            x_dx_dy_t(fespace_mesh_t);
    x_t.GetDerivative(1, 0, x_dx_t);
    x_t.GetDerivative(1, 1, x_dy_t);
    // set vector magnitude
    for (int i = 0; i < x_dx_t.Size(); i++) {
        double dxv = x_dx_t(i),
               dyv = x_dy_t(i);
        double mag = dxv*dxv + dyv*dyv;
        if (mag > 0) { mag = pow(mag, 0.5); }
        x_dx_t(i) *= x_t(i)/mag;
        x_dy_t(i) *= x_t(i)/mag;
    }

    // copy to vector GridFunction
    for (int i = 0; i < x_dx_dy_t.Size()/dim; i++) {
        x_dx_dy_t(i) = x_dx_t(i);
        x_dx_dy_t(i + x_dx_dy_t.Size()/dim) = x_dy_t(i);
    }

    // Visualize vector GridFunction
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << *mesh_t << x_dx_dy_t << flush;
       sol_sock << "window_title 'Derivative distfun'\n"
                << "window_geometry "
                << 350 << " " << 350 << " " << 350 << " " << 350 << "\n"
                << "keys Rjmpc" << endl;
    }

    x = 0;
    x_t = 0;


    int bdr_attr_max = mesh_t->bdr_attributes.Max();
    // Check out surface normals
    for (int it = 0; it < fespace_t.GetNBE(); it++) {
        int bdr_attr = mesh_t->GetBdrAttribute(it);
        if (bdr_attr == bdr_attr_max) {
            Vector nor(dim);
            ElementTransformation *Trans = mesh_t->GetBdrElementTransformation(it);
            Trans->SetIntPoint(&Geometries.GetCenter(Trans->GetGeometryType()));
            CalcOrtho(Trans->Jacobian(), nor);
            std::cout << it << " BDR_ ELEMENT\n";
            nor.Print();
        }
    }

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet) and
    //    converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh_t->bdr_attributes.Size())
    {
       Array<int> ess_bdr(mesh_t->bdr_attributes.Max());
       ess_bdr = 1;
       fespace_t.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 7. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
    //    the basis functions in the finite element fespace.
    LinearForm b(&fespace_t);
    ConstantCoefficient one(1.0);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    // 9. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
    //    domain integrator.
    BilinearForm a(&fespace_t);
    if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    a.AddDomainIntegrator(new DiffusionIntegrator(one));

    // 10. Assemble the bilinear form and the corresponding linear system,
    //     applying any necessary transformations such as: eliminating boundary
    //     conditions, applying conforming constraints for non-conforming AMR,
    //     static condensation, etc.
    if (static_cond) { a.EnableStaticCondensation(); }
    a.Assemble();

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x_t, b, A, X, B);

    cout << "Size of linear system: " << A->Height() << endl;

    // 11. Solve the linear system A X = B.
    if (!pa)
    {
 #ifndef MFEM_USE_SUITESPARSE
       // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
       GSSmoother M((SparseMatrix&)(*A));
       PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
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
       if (UsesTensorBasis(fespace))
       {
          OperatorJacobiSmoother M(a, ess_tdof_list);
          PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
       }
       else
       {
          CG(*A, B, X, 1, 400, 1e-12, 0.0);
       }
    }

    // 12. Recover the solution as a finite element grid function.
    a.RecoverFEMSolution(X, b, x_t);

    // 13. Save the refined mesh and the solution. This output can be viewed later
    //     using GLVis: "glvis -m refined.mesh -g sol.gf".
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh_t->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x_t.Save(sol_ofs);

    // 14. Send the solution by socket to a GLVis server.
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << *mesh_t << x_t << flush;
       sol_sock << "window_title 'Solution'\n"
                << "window_geometry "
                << 700 << " " << 0 << " " << 350 << " " << 350 << "\n"
                << "keys Rj" << endl;
    }

    // 15. Free the used memory.
    if (delete_fec)
    {
       delete fec;
    }

    return 0;
}
