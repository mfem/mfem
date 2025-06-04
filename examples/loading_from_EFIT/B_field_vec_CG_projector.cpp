#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "vec_coeffs.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    const char *mesh_file = "mesh/2d_mesh.mesh";
    const char *gf_file = "output/B_pol_vec_CG.gf"; // default value
    const char *gf_name = "B_pol_vec_CG"; // default value
    bool visualization = true;

    // Parse command-line options
    OptionsParser args(argc, argv);
    args.AddOption(&gf_file, "-gf", "--grid-function",
                   "Grid function (.gf) input file.");
    args.AddOption(&gf_name, "-gn", "--grid-function-name",
                   "Grid function name to be used in ParaView output.");
    args.Parse();
    args.PrintOptions(cout);

    // Load the mesh
    Mesh mesh(mesh_file, 1, 1);

    // Load the grid function
    ifstream temp_log(gf_file);
    GridFunction field(&mesh, temp_log);

    cout << "Mesh loaded" << endl;

    const char *new_mesh_file = "mesh/2d_mesh.mesh";
    Mesh *projected_mesh = new Mesh(new_mesh_file, 1, 1);

    FiniteElementSpace fespace(projected_mesh, field.FESpace()->FEColl(), field.FESpace()->GetVDim());

    GridFunction field_projected(&fespace);

    LinearForm b(&fespace);
    BilinearForm a(&fespace);

    // 1. make the linear form
    FieldRVectorGridFunctionCoefficient field_r_coef(&field, false);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(field_r_coef));
    b.Assemble();

    // 2. make the bilinear form
    RGridFunctionCoefficient r_coef;
    a.AddDomainIntegrator(new VectorMassIntegrator(r_coef));
    a.Assemble();
    a.Finalize();

    // 3. solve the system
    CGSolver M_solver;
    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-24);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(1e5);
    M_solver.SetPrintLevel(1);
    M_solver.SetOperator(a.SpMat());

    Vector X(field_projected.Size());
    X = 0.0;
    M_solver.Mult(b, X);

    field_projected.SetFromTrueDofs(X);

    if (visualization)
    {
        char vishost[] = "localhost";
        int visport = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n"
                 << *projected_mesh << field_projected << flush;
    }

    // paraview
    {
        ParaViewDataCollection paraview_dc(gf_name, projected_mesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(1);
        paraview_dc.SetCycle(0);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.SetTime(0.0); // set the time
        paraview_dc.RegisterField(gf_name, &field_projected);
        paraview_dc.Save();
    }

    ofstream sol_ofs(gf_file);
    sol_ofs.precision(8);
    field_projected.Save(sol_ofs);

    delete projected_mesh;

    return 0;
}