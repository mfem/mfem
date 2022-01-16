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
         maxorder = std::max(maxorder,order);
      }
   }
   fes.Update(false);
   return maxorder;
}


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

void vectorcoeff(const Vector& x, Vector& y)
{
   y(0) = coeff(x);
   y(1) = -coeff(x);
   if (dimension == 3)
   {
      y(2) = 2.0 * coeff(x);
   }
}

double coeff2(const Vector & X)
{
   double x = X[0];
   double y = X[1];
   return (1.-x)*x*(1.-y)*y;
}

void vectorcoeff2(const Vector& x, Vector& y)
{
   y(0) = coeff2(x);
   y(1) = -coeff2(x);
   if (dimension == 3)
   {
      y(2) = 2.0 * coeff2(x);
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
               GridFunction X(c_fespace);
               GridFunction X_cmp(c_fespace);
               GridFunction Y_exact(f_fespace);
               GridFunction Y_std(f_fespace);
               GridFunction Y_test(f_fespace);

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

TEST_CASE("variable_order_true_transfer")
{
   for (int vectorspace = 0; vectorspace <= 1; ++vectorspace)
   {
      for (dimension = 2; dimension <= 3; ++dimension)
      {
         for (int order = 1; order <= 3; order++)
         {
            std::cout << "Testing variable order true transfer:\n"
                      << "  Vectorspace:    " << vectorspace << "\n"
                      << "  Dimension:      " << dimension << "\n"
                      << "  Coarse order:   " << order << "\n";

            Mesh mesh;
            if (dimension == 2)
            {
               Element::Type type = Element::QUADRILATERAL;
               mesh = Mesh::MakeCartesian2D(3, 3, type, 1, 1.0, 1.0);
            }
            else
            {
               Element::Type type = Element::HEXAHEDRON;
               mesh = Mesh::MakeCartesian3D(3, 3, 3, type, 1.0, 1.0, 1.0);
            }
            FiniteElementCollection* c_fec = nullptr;
            FiniteElementCollection* f_fec = nullptr;
            c_fec = new H1_FECollection(order, dimension);
            f_fec = new H1_FECollection(order, dimension);
            mesh.EnsureNCMesh();

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


            const SparseMatrix * Rc = c_fespace->GetRestrictionMatrix();
            TrueTransferOperator T(*c_fespace, *f_fespace);
            GridFunction xc(c_fespace);
            Vector Xc(c_fespace->GetTrueVSize());
            Vector Diff(c_fespace->GetTrueVSize());
            Vector Yc(c_fespace->GetTrueVSize());
            Vector Xf(f_fespace->GetTrueVSize());
            Vector Yf(f_fespace->GetTrueVSize());


            if (vectorspace == 0)
            {
               FunctionCoefficient funcCoeff(&coeff2);
               xc.ProjectCoefficient(funcCoeff);
            }
            else
            {
               VectorFunctionCoefficient funcCoeff(dimension, &vectorcoeff2);
               xc.ProjectCoefficient(funcCoeff);
            }
            if (Rc)
            {
               Rc->Mult(xc,Xc);
            }
            else
            {
               Xc = xc;
            }
            T.Mult(Xc, Xf);

            BilinearFormIntegrator * massc=nullptr;
            BilinearFormIntegrator * massf=nullptr;

            switch (vectorspace)
            {
               case 0:
                  massc = new MassIntegrator;
                  massf = new MassIntegrator;
                  break;
               default:
                  massc = new VectorMassIntegrator;
                  massf = new VectorMassIntegrator;
                  break;
            }


            BilinearForm mc(c_fespace);
            mc.AddDomainIntegrator(massc);
            mc.Assemble();
            SparseMatrix Mc;
            Array<int> empty;
            mc.FormSystemMatrix(empty, Mc);

            BilinearForm mf(f_fespace);
            mf.AddDomainIntegrator(massf);
            mf.Assemble();
            SparseMatrix Mf;
            mf.FormSystemMatrix(empty, Mf);

            Mf.Mult(Xf,Yf);

            T.MultTranspose(Yf,Yc);

            GSSmoother M(Mc);
            Diff = 0.;
            PCG(Mc, M, Yc, Diff, 0, 500, 1e-24, 0.0);

            Diff -= Xc;

            REQUIRE(Diff.Norml2() < 1e-10);

            delete f_fespace;
            delete c_fespace;
            delete f_fec;
            delete c_fec;
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
