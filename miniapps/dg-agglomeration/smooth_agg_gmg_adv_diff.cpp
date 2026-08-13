// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//                     ------------------------
//                     DG Smooth-Agg GMG Solver
//                     ------------------------

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "mg_agglom.hpp"

using namespace std;
using namespace mfem;

// RHS
real_t rhs_function(const Vector &x);

// true solution
real_t u_true(const Vector &x);

void velocity_func(const Vector &x, Vector &v)
{
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    v *= 1.0/sqrt(14);
    // v(0) = 0.0;
    // v(1) = 0.0;
    // v(2) = 0.0;
}

int main(int argc, char *argv[])
{
    const char *mesh_file = "../../data/inline-hex.mesh";
    int order = 1;
    real_t kappa_0 = 1.0;
    int num_levels = 2;
    real_t diff_c = 0.0;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
    // args.AddOption(&ref_levels, "-r", "--refine", "Refinement levels.");
    args.AddOption(&order, "-o", "--order", "Polynomial degree.");
    args.AddOption(&kappa_0, "-k", "--kappa", "DG penalty parameter.");
    args.AddOption(&diff_c, "-dc", "--diff_c", "Diffusion Coefficient.");
    // args.AddOption(&ncoarse, "-nc", "--ncoarse", "Number of Fine Elements per Coarse.");
    args.AddOption(&num_levels, "-nl", "--levels", "Number of Multigrid Levels.");
    args.ParseCheck();

    Mesh mesh(mesh_file);
    const int dim = mesh.Dimension();
    int ncoarse = pow(2, dim); 
    int ref_levels = num_levels - 2;


    for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }

    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace fespace(&mesh, &fec);
    cout << "Number of unknowns: " << fespace.GetVSize() << endl;

    if (diff_c != 0) {cout << "Peclet Number: " << 2.0 / diff_c << endl;}
    else {cout << "Peclet Number: inf" << endl;}

    int ne = mesh.GetNE();

    const real_t sigma = -1.0;
    const real_t kappa = kappa_0 * (order + 1.0) * (order + 1.0) / 2.0;

    //    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    //    ess_bdr = 0;
    //    ess_bdr[0] = 1;

    Array<int> dof_indices;
    for (int e = 0; e < ne; ++e)
    {
        fespace.GetElementDofs(e, dof_indices);
            std::cout << "Element " << e << " DoF indices: ";
        for (int j = 0; j < dof_indices.Size(); ++j) {
            std::cout << dof_indices[j] << " ";
        }
        std::cout << std::endl;
    }
    std::string file_name_t2t = "../../../adaptiveMG/t2t_mfem.txt";

    {
        const auto &e2e = mesh.ElementToElementTable();
        Array<int> fn;

        std::ofstream f(file_name_t2t);
        for (int e = 0; e < mesh.GetNE(); ++e)
        {
            e2e.GetRow(e, fn);
            int i = 0;
            for (; i < fn.Size(); ++i)
            {
                if (fn[i] != e)
                {
                f << (fn[i] + 1) << " ";
                }
            }
            const int nf = dim == 2 ? Geometry::NumEdges[mesh.GetElementGeometry(e)]
                            : Geometry::NumFaces[mesh.GetElementGeometry(e)];
            for (; i < nf; ++i)
            {
                f << -1 << " ";
            }
            f << '\n';
        }
    }

    //    mfem::Array<int> dbc_marker(6);
    //    dbc_marker = 1;
    //    dbc_marker[4] = 0;

    // LinearForm b(&fespace);
    ConstantCoefficient diff_coef(diff_c);
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);
    VectorFunctionCoefficient velocity(dim, velocity_func);

    GridFunction x(&fespace);
    x = 0.0;

    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(diff_coef));
    a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));
    a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coef, sigma, kappa));

    a.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
    a.AddInteriorFaceIntegrator(
        new NonconservativeDGTraceIntegrator(velocity, -1.0));
    a.AddBdrFaceIntegrator(
        new NonconservativeDGTraceIntegrator(velocity, -1.0));
    a.Assemble();
    a.Finalize();

    LinearForm b(&fespace);
    b.AddDomainIntegrator(new DomainLFIntegrator(zero));
    // b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(one, diff_coef, sigma, kappa));
    // b.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(one, velocity, -1.0));
    b.Assemble();

    SparseMatrix &A = a.SpMat();

    std::string file_name = "A_mfem.mtx";
    std::ofstream ofs1("../../../adaptiveMG/" + file_name);
    A.PrintMM(ofs1); 
    ofs1.close();

    // {
    //     std::ofstream f("A.txt");
    //     A.PrintMatlab(f);
    // }

    SmoothedAggregationGMG mg(fespace, A, ncoarse, num_levels, false);
    mg.SetCycleType(mfem::MultigridBase::CycleType::VCYCLE, 3, 3);

    GMRESSolver gmres;
    gmres.SetRelTol(1e-7);
    gmres.SetMaxIter(5000);
    gmres.SetPrintLevel(1);
    gmres.SetKDim(1000);
    gmres.SetOperator(A);
    gmres.SetPreconditioner(mg);
    // Vector b_vec(fespace.GetVSize());
    // b_vec = 1.0;
    Vector& b_vec = b;
    std::cout << "norm b = " << b_vec.Norml2() << std::endl;


    std::string file_name_b = "b_mfem.txt";
    std::ofstream ofsb("../../../adaptiveMG/" + file_name_b);
    b_vec.Print(ofsb); 
    ofsb.close();

    x = 1.0;
    gmres.Mult(b_vec, x);

    // FunctionCoefficient true_solution(u_true);
    // GridFunction true_sol_gf(&fespace);
    // true_sol_gf.ProjectCoefficient(true_solution);
    // double l2_error = x.ComputeL2Error(true_solution);

    // std::cout << "True  rel L2 Error: " << l2_error/true_sol_gf.Norml2() << std::endl;

    return 0;
}

// Initial condition
real_t rhs_function(const Vector &x)
{
   // int dim = x.Size();

   real_t px = M_PI*x(0);
   real_t py = M_PI*x(1);
   real_t pz = M_PI*x(2);

   real_t pi_s = M_PI*M_PI;

   real_t sss = sin(px)*sin(py)*sin(pz);

   real_t css = cos(px)*sin(py)*sin(pz)*cos(px)*sin(py)*sin(pz);
   real_t scs = sin(px)*cos(py)*sin(pz)*sin(px)*cos(py)*sin(pz);
   real_t ssc = sin(px)*sin(py)*cos(pz)*sin(px)*sin(py)*cos(pz);
   return -pi_s*exp(sss)*(-3*sss + css + scs + ssc);
}


// Initial condition
real_t u_true(const Vector &x)
{
   // int dim = x.Size();

   real_t px = M_PI*x(0);
   real_t py = M_PI*x(1);
   real_t pz = M_PI*x(2);

   real_t sss = sin(px)*sin(py)*sin(pz);
   return exp(sss) - 1;
}