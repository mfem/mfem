// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "catch.hpp"
#include "mfem.hpp"

using namespace mfem;

int dimension;
double coeff(const Vector& x)
{
   if (dimension == 2)
   {
      return 1.1 * x[0] + 2.0 * x[1];
   }
   else
   {
      return 1.1 * x[0] + 2.0 * x[1] + 3.0 * x[2];
   }
}

TEST_CASE("ptransfer")
{

   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int ne = 1; ne <= 3; ++ne)
      {
         for (int order = 1; order <= 4; order *= 2)
         {
            std::cout << "Testing transfer:\n"
                      << "  Dimension:    " << dimension << "\n"
                      << "  Elements:     " << std::pow(ne, dimension) << "\n"
                      << "  Coarse order: " << order << "\n"
                      << "  Fine order:   " << 2 * order << "\n";

            Mesh* mesh;
            if (dimension == 2)
            {
               mesh = new Mesh(ne, ne, Element::QUADRILATERAL, 1, 1.0, 1.0);
            }
            else
            {
               mesh =
                  new Mesh(ne, ne, ne, Element::HEXAHEDRON, 1, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection* c_h1_fec =
               new H1_FECollection(order, dimension);
            FiniteElementCollection* f_h1_fec =
               new H1_FECollection(2 * order, dimension);
            FiniteElementSpace c_h1_fespace(mesh, c_h1_fec);
            FiniteElementSpace f_h1_fespace(mesh, f_h1_fec);

            PRefinementTransferOperator transferOperator(c_h1_fespace,
                                                         f_h1_fespace);
            TensorProductPRefinementTransferOperator tpTransferOperator(
               c_h1_fespace, f_h1_fespace);
            GridFunction X(&c_h1_fespace);
            GridFunction X_tp(&c_h1_fespace);
            GridFunction Y_exact(&f_h1_fespace);
            GridFunction Y_std(&f_h1_fespace);
            GridFunction Y_tp(&f_h1_fespace);

            FunctionCoefficient funcCoeff(&coeff);

            X.ProjectCoefficient(funcCoeff);
            Y_exact.ProjectCoefficient(funcCoeff);
            Y_std = 0.0;
            Y_tp = 0.0;

            transferOperator.Mult(X, Y_std);
            tpTransferOperator.Mult(X, Y_tp);

            Y_tp -= Y_exact;
            REQUIRE(Y_tp.Norml2() < 1e-12);

            Y_std -= Y_exact;
            REQUIRE(Y_std.Norml2() < 1e-12);

            transferOperator.MultTranspose(Y_exact, X);
            tpTransferOperator.MultTranspose(Y_exact, X_tp);

            X -= X_tp;
            REQUIRE(X.Norml2() < 1e-12);

            delete f_h1_fec;
            delete c_h1_fec;
            delete mesh;
         }
      }
   }
}
