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
    // Set grid function f(x) = dot(c, x) and verify that normal derivatives
    // computed by L2NormalDerivativeFaceRestriction are dot(c, n).

    const int dim = 2;
    const int nx = 3;
    const int order = 4;

    Mesh mesh = (dim == 2) ? Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL, true) : Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
    // Mesh mesh("../../data/star.mesh");
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);

    FiniteElementSpace fes(&mesh, &fec);

    GridFunction gf(&fes);
    // gf = 1.0; // constant function 1

    IntegrationRules irs(0, Quadrature1D::GaussLobatto);
    const IntegrationRule& ir = irs.Get(mesh.GetFaceGeometry(0), 2 * order - 1);
    FaceQuadratureSpace fqs(mesh, ir, FaceType::Interior);

    L2NormalDerivativeFaceRestriction dfr(fes, ElementDofOrdering::LEXICOGRAPHIC, FaceType::Interior);

    QuadratureFunction qf(fqs, 2);
    QuadratureFunction qfref(fqs, 2);
    QuadratureFunction qff(fqs, 1);
    FunctionCoefficient coef(f);
    coef.Project(qff);
    gf.ProjectCoefficient(coef);

    auto geom = mesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::NORMALS, FaceType::Interior);
    auto ns = geom->normal.Read();
    for (int f=0; f < fqs.GetNumFaces(); ++f)
    {
        for (int p=0; p < ir.Size(); ++p)
        {
            const double n_x = std::abs(ns[p + ir.Size() * (0 + 2 * f)]);
            const double n_y = std::abs(ns[p + ir.Size() * (1 + 2 * f)]);
            const double val = n_x + 2*n_y;

            qfref[p + ir.Size() * (0 + 2 * f)] = val/nx;
            qfref[p + ir.Size() * (1 + 2 * f)] = val/nx;
        }
    }

    dfr.Mult(gf, qf);

    qf -= qfref;

    REQUIRE(qf.Normlinf() == MFEM_Approx(0.0));
}
