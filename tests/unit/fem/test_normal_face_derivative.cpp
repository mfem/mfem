// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

double f(const Vector& x)
{
    return x[0] + 2 * x[1];
}

TEST_CASE("Normal Face Derivative", "[FaceRestriction]")
{
    // 1. f(x) == 1, setup, say, cartesian mesh and verify derivatives are zero
    // 2. f(x) == c * x, cartesian mesh, verify derivatives are n * c

    const int dim = 2;
    const int nx = 3;
    const int order = 4;

    // Mesh mesh = (dim == 2) ? Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true) : Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
    Mesh mesh("../../data/star.mesh");
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);

    FiniteElementSpace fes(&mesh, &fec);

    GridFunction gf(&fes);
    // gf = 1.0; // constant function 1

    IntegrationRules irs(0, Quadrature1D::GaussLobatto);
    const IntegrationRule& ir = irs.Get(mesh.GetFaceGeometry(0), 2 * order);
    FaceQuadratureSpace fqs(mesh, ir, FaceType::Interior);

    L2NormalDerivativeFaceRestriction dfr(fes, ElementDofOrdering::LEXICOGRAPHIC, fqs);

    QuadratureFunction qf(fqs, 2);
    QuadratureFunction qfref(fqs, 2);
    QuadratureFunction qff(fqs, 1);
    FunctionCoefficient coef(f);
    coef.Project(qff);
    gf.ProjectCoefficient(coef);

    auto geom = mesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::NORMALS, FaceType::Interior);
    auto ns = geom->normal.Read();
    // auto ns = Reshape(geom->normal.Read(), ir.Size(), 2, fqs.GetNumFaces());

    for (int f=0; f < fqs.GetNumFaces(); ++f)
    {
        for (int p=0; p < ir.Size(); ++p)
        {
            const double val = ns[p + ir.Size() * (0 + 2 * f)] + 2 * ns[p + ir.Size() * (1 + 2 * f)];
            // q vdim 2 nf

            qfref[p + ir.Size() * (0 + 2 * f)] = val;
            qfref[p + ir.Size() * (1 + 2 * f)] = val;
        }
    }

    dfr.Mult(gf, qf);

    qfref.Print(std::cout, 1);
    std::cout << "\n----------------------\n";
    qf.Print(std::cout, 1);

    qf -= qfref;
    
    REQUIRE(qf.Normlinf() == MFEM_Approx(0.0));
}
