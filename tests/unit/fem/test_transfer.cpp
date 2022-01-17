// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

int RandomPRefinement(FiniteElementSpace & fes)
{
   Mesh * mesh = fes.GetMesh();
   int maxorder = 0;
   int order;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      order = fes.GetElementOrder(i);
      maxorder = std::max(maxorder,order);
      if ((double) rand() / RAND_MAX < 0.5)
      {
         fes.SetElementOrder(i,order+1);
         maxorder = std::max(maxorder,order+1);
      }
   }
   fes.Update(false);
   return maxorder;
}


int dimension;
int coeff_order;
double coeff(const Vector& X)
{
   double x = X[0];
   double y = X[1];
   double z = 0.;
   if (dimension == 2)
   {
      if (coeff_order == 1)
      {
         return 1.1 * x + 2.0 * y;
      }
      else
      {
         return (1.-x)*x*(1.-y)*y;
      }
   }
   else
   {
      z = X[2];
      if (coeff_order == 1)
      {
        return 1.1 * x + 2.0 * y + 3.0 * z;
      }
      else  
      {
         return (1.-x)*x*(1.-y)*y*(1.-z)*z;
      }
   }
}

void vectorcoeff(const Vector& x, Vector& y)
{
   y(0) = coeff(x);
   y(1) = -coeff(x);
   if (dimension == 3)
   {
      y(2) = 2.0 * coeff(x);
   }
}

TEST_CASE("transfer")
{
   for (int vectorspace = 0; vectorspace <= 3; ++vectorspace)
   {
      for (dimension = 2; dimension <= 3; ++dimension)
      {
         for (int elementType = 0; elementType <= 1; ++elementType)
         {
            for (int ne = 1; ne <= 3; ++ne)
            {
               for (int order = 1; order <= 4; order *= 2)
               {
                  for (int geometric = 0; geometric <= 1; ++geometric)
                  {
                     int fineOrder = (geometric == 1) ? order : 2 * order;

                     std::cout << "Testing transfer:\n"
                               << "  Vectorspace:  " << vectorspace << "\n"
                               << "  Dimension:    " << dimension << "\n"
                               << "  Element type: " << elementType << "\n"
                               << "  Elements:     " << std::pow(ne, dimension) << "\n"
                               << "  Coarse order: " << order << "\n"
                               << "  Fine order:   " << fineOrder << "\n"
                               << "  Geometric:    " << geometric << "\n";

                     Mesh mesh;
                     if (dimension == 2)
                     {
                        Element::Type type = Element::QUADRILATERAL;
                        if (elementType != 0)
                        {
                           type = Element::TRIANGLE;
                        }
                        mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
                     }
                     else
                     {
                        Element::Type type = Element::HEXAHEDRON;
                        if (elementType != 0)
                        {
                           type = Element::TETRAHEDRON;
                        }
                        mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
                     }
                     FiniteElementCollection* c_fec = nullptr;
                     FiniteElementCollection* f_fec = nullptr;

                     if (vectorspace < 2)
                     {
                        c_fec = new H1_FECollection(order, dimension);
                        f_fec = (geometric == 1) ? c_fec : new
                                H1_FECollection(fineOrder, dimension);
                     }
                     else if (vectorspace == 2)
                     {
                        c_fec = new ND_FECollection(order+1, dimension);
                        f_fec = (geometric == 1) ? c_fec : new
                                ND_FECollection(fineOrder, dimension);
                     }
                     else
                     {
                        c_fec = new RT_FECollection(order, dimension);
                        f_fec = (geometric == 1) ? c_fec : new
                                RT_FECollection(fineOrder, dimension);
                     }

                     Mesh fineMesh(mesh);
                     if (geometric)
                     {
                        fineMesh.UniformRefinement();
                     }

                     int spaceDimension = 1;

                     if (vectorspace == 1)
                     {
                        spaceDimension = dimension;
                     }

                     FiniteElementSpace* c_fespace =
                        new FiniteElementSpace(&mesh, c_fec, spaceDimension);
                     FiniteElementSpace* f_fespace =
                        new FiniteElementSpace(&fineMesh, f_fec,spaceDimension);


                     Operator* referenceOperator = nullptr;

                     if (geometric == 0)
                     {
                        referenceOperator = new PRefinementTransferOperator(*c_fespace,
                                                                            *f_fespace);
                     }
                     else
                     {
                        OperatorPtr P(Operator::ANY_TYPE);
                        f_fespace->GetTransferOperator(*c_fespace, P);
                        P.SetOperatorOwner(false);
                        referenceOperator = P.Ptr();
                     }

                     TransferOperator testTransferOperator(*c_fespace, *f_fespace);
                     GridFunction X(c_fespace);
                     GridFunction X_cmp(c_fespace);
                     GridFunction Y_exact(f_fespace);
                     GridFunction Y_std(f_fespace);
                     GridFunction Y_test(f_fespace);
                     coeff_order = 1;
                     if (vectorspace == 0)
                     {
                        FunctionCoefficient funcCoeff(&coeff);
                        X.ProjectCoefficient(funcCoeff);
                        Y_exact.ProjectCoefficient(funcCoeff);
                     }
                     else
                     {
                        VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
                        X.ProjectCoefficient(funcCoeff);
                        Y_exact.ProjectCoefficient(funcCoeff);
                     }

                     Y_std = 0.0;
                     Y_test = 0.0;

                     referenceOperator->Mult(X, Y_std);

                     Y_std -= Y_exact;
                     REQUIRE(Y_std.Norml2() < 1e-12 * Y_exact.Norml2());

                     testTransferOperator.Mult(X, Y_test);

                     Y_test -= Y_exact;
                     REQUIRE(Y_test.Norml2() < 1e-12 * Y_exact.Norml2());

                     referenceOperator->MultTranspose(Y_exact, X);
                     testTransferOperator.MultTranspose(Y_exact, X_cmp);

                     X -= X_cmp;
                     REQUIRE(X.Norml2() < 1e-12 * X_cmp.Norml2());

                     delete referenceOperator;
                     delete f_fespace;
                     delete c_fespace;
                     if (geometric == 0)
                     {
                        delete f_fec;
                     }
                     delete c_fec;
                  }
               }
            }
         }
      }
   }
}


TEST_CASE("variable_order_transfer")
{
   for (int vectorspace = 0; vectorspace <= 3; ++vectorspace)
   {
      for (dimension = 2; dimension <= 3; ++dimension)
      {
         for (int ne = 1; ne <= 3; ++ne)
         {
            for (int order = 1; order <= 4; order *= 2)
            {
               std::cout << "Testing variable order transfer:\n"
                         << "  Vectorspace:    " << vectorspace << "\n"
                         << "  Dimension:      " << dimension << "\n"
                         << "  Elements:       " << std::pow(ne, dimension) << "\n"
                         << "  Coarse order:   " << order << "\n";

               Mesh mesh;
               if (dimension == 2)
               {
                  Element::Type type = Element::QUADRILATERAL;
                  mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
               }
               else
               {
                  Element::Type type = Element::HEXAHEDRON;
                  mesh = Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
               }
               FiniteElementCollection* c_fec = nullptr;
               FiniteElementCollection* f_fec = nullptr;
               if (vectorspace < 2)
               {
                  c_fec = new H1_FECollection(order, dimension);
                  f_fec = new H1_FECollection(order, dimension);
               }
               else if (vectorspace == 2)
               {
                  c_fec = new ND_FECollection(order+1, dimension);
                  f_fec = new ND_FECollection(order+1, dimension);
               }
               else
               {
                  c_fec = new RT_FECollection(order, dimension);
                  f_fec = new RT_FECollection(order, dimension);
               }

               mesh.EnsureNCMesh();
               mesh.RandomRefinement(0.5);

               int spaceDimension = 1;

               if (vectorspace == 1)
               {
                  spaceDimension = dimension;
               }

               FiniteElementSpace* c_fespace =
                  new FiniteElementSpace(&mesh, c_fec, spaceDimension);
               FiniteElementSpace* f_fespace =
                  new FiniteElementSpace(&mesh, f_fec,spaceDimension);
               int maxorder = RandomPRefinement(*f_fespace);

               std::cout  << "  Max fine order: " << maxorder << "\n";

               Operator* referenceOperator = nullptr;

               referenceOperator = new PRefinementTransferOperator(*c_fespace,
                                                                   *f_fespace);

               TransferOperator testTransferOperator(*c_fespace, *f_fespace);
               GridFunction X(c_fespace); X = 0.;
               GridFunction X_cmp(c_fespace); X_cmp = 0.;
               GridFunction Y_exact(f_fespace); Y_exact = 0.;
               GridFunction Y_std(f_fespace); Y_std = 0.;
               GridFunction Y_test(f_fespace); Y_test = 0.;
               coeff_order = std::min(2,order);
               if (vectorspace == 0)
               {
                  FunctionCoefficient funcCoeff(&coeff);
                  X.ProjectCoefficient(funcCoeff);
                  Y_exact.ProjectCoefficient(funcCoeff);
               }
               else
               {
                  VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff);
                  X.ProjectCoefficient(funcCoeff);
                  Y_exact.ProjectCoefficient(funcCoeff);
               }

               Y_std = 0.0;
               Y_test = 0.0;

               referenceOperator->Mult(X, Y_std);
               Y_std -= Y_exact;
               REQUIRE(Y_std.Norml2() < 1e-12 * Y_exact.Norml2());

               testTransferOperator.Mult(X, Y_test);

               Y_test -= Y_exact;
               REQUIRE(Y_test.Norml2() < 1e-12 * Y_exact.Norml2());

               referenceOperator->MultTranspose(Y_exact, X);
               testTransferOperator.MultTranspose(Y_exact, X_cmp);

               X -= X_cmp;
               REQUIRE(X.Norml2() < 1e-12 * X_cmp.Norml2());

               delete referenceOperator;
               delete f_fespace;
               delete c_fespace;
               delete f_fec;
               delete c_fec;
            }
         }
      }
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("partransfer", "[Parallel]")
{
   for (dimension = 2; dimension <= 3; ++dimension)
   {
      for (int elementType = 0; elementType <= 1; ++elementType)
      {
         for (int ne = 4; ne <= 5; ++ne)
         {
            for (int order = 1; order <= 4; order *= 2)
            {
               for (int geometric = 0; geometric <= 1; ++geometric)
               {
                  int fineOrder = (geometric == 1) ? order : 2 * order;

                  int num_procs;
                  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
                  int myid;
                  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

                  if (myid == 0)
                  {
                     std::cout << "Testing parallel transfer:\n"
                               << "  Dimension:    " << dimension << "\n"
                               << "  Element type: " << elementType << "\n"
                               << "  Elements:     " << std::pow(ne, dimension) << "\n"
                               << "  Coarse order: " << order << "\n"
                               << "  Fine order:   " << fineOrder << "\n"
                               << "  Geometric:    " << geometric << "\n";
                  }
                  coeff_order = std::min(1,order);

                  Mesh mesh;
                  if (dimension == 2)
                  {
                     Element::Type type = Element::QUADRILATERAL;
                     if (elementType != 0)
                     {
                        type = Element::TRIANGLE;
                     }
                     mesh = Mesh::MakeCartesian2D(ne, ne, type, 1, 1.0, 1.0);
                  }
                  else
                  {
                     Element::Type type = Element::HEXAHEDRON;
                     if (elementType != 0)
                     {
                        type = Element::TETRAHEDRON;
                     }
                     mesh =
                        Mesh::MakeCartesian3D(ne, ne, ne, type, 1.0, 1.0, 1.0);
                  }

                  Mesh fineMesh(mesh);
                  if (geometric)
                  {
                     fineMesh.UniformRefinement();
                  }

                  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
                  ParMesh pfineMesh(MPI_COMM_WORLD, mesh);
                  if (geometric)
                  {
                     pfineMesh.UniformRefinement();
                  }

                  FiniteElementCollection* c_h1_fec =
                     new H1_FECollection(order, dimension);
                  FiniteElementCollection* f_h1_fec = (geometric == 1) ? c_h1_fec : new
                                                      H1_FECollection(fineOrder, dimension);

                  int spaceDimension = 1;

                  double referenceRestrictionValue = 0.0;

                  // Compute reference values in serial
                  {
                     FiniteElementSpace* c_h1_fespace = new FiniteElementSpace(&mesh, c_h1_fec,
                                                                               spaceDimension);
                     FiniteElementSpace* f_h1_fespace = new FiniteElementSpace(&fineMesh, f_h1_fec,
                                                                               spaceDimension);

                     Operator* transferOperator = new TransferOperator(*c_h1_fespace,
                                                                       *f_h1_fespace);
                     GridFunction X(c_h1_fespace);
                     GridFunction Y(f_h1_fespace);

                     FunctionCoefficient funcCoeff(&coeff);
                     Y.ProjectCoefficient(funcCoeff);
                     X = 0.0;

                     transferOperator->MultTranspose(Y, X);

                     referenceRestrictionValue = std::sqrt(InnerProduct(X, X));

                     delete transferOperator;
                     delete f_h1_fespace;
                     delete c_h1_fespace;
                  }

                  ParFiniteElementSpace* c_h1_fespace = new ParFiniteElementSpace(pmesh, c_h1_fec,
                                                                                  spaceDimension);
                  ParFiniteElementSpace* f_h1_fespace = new ParFiniteElementSpace(&pfineMesh,
                                                                                  f_h1_fec,
                                                                                  spaceDimension);

                  Operator* transferOperator = new TrueTransferOperator(*c_h1_fespace,
                                                                        *f_h1_fespace);
                  ParGridFunction X(c_h1_fespace);
                  ParGridFunction Y_exact(f_h1_fespace);
                  ParGridFunction Y(f_h1_fespace);
                  FunctionCoefficient funcCoeff(&coeff);
                  X.ProjectCoefficient(funcCoeff);
                  Y_exact.ProjectCoefficient(funcCoeff);

                  Y = 0.0;

                  Vector X_true(c_h1_fespace->GetTrueVSize());
                  Vector Y_true(f_h1_fespace->GetTrueVSize());

                  c_h1_fespace->GetRestrictionMatrix()->Mult(X, X_true);
                  transferOperator->Mult(X_true, Y_true);
                  f_h1_fespace->GetProlongationMatrix()->Mult(Y_true, Y);

                  Y -= Y_exact;
                  REQUIRE(Y.Norml2() < 1e-12 * Y_exact.Norml2());

                  f_h1_fespace->GetRestrictionMatrix()->Mult(Y_exact, Y_true);
                  transferOperator->MultTranspose(Y_true, X_true);

                  double restrictionValue = std::sqrt(InnerProduct(MPI_COMM_WORLD, X_true,
                                                                   X_true));
                  REQUIRE(std::abs(restrictionValue - referenceRestrictionValue) < 1e-12 *
                          std::abs(referenceRestrictionValue));

                  delete transferOperator;
                  delete f_h1_fespace;
                  delete c_h1_fespace;
                  if (geometric == 0)
                  {
                     delete f_h1_fec;
                  }
                  delete c_h1_fec;
                  delete pmesh;
               }
            }
         }
      }
   }
}

#endif
