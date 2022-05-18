// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
using namespace std;

namespace project_gf
{

int dimension;


int Ww = 400, Wh = 350;
int Wx = 0, Wy = 0;
int offx = Ww+5, offy = Wh+25;

void Visualize(Mesh& mesh, GridFunction& gf, const string &title, const string& caption, int x, int y)
{
    int w = 400, h = 350;

    char vishost[] = "localhost";
    int  visport   = 19916;

    socketstream sol_sockL2(vishost, visport);
    sol_sockL2.precision(10);
    sol_sockL2 << "solution\n" << mesh << gf
               << "window_geometry " << x << " " << y << " " << w << " " << h
               << "window_title '" << title << "'"
               << "plot_caption '" << caption << "'" << flush;
}

struct PolyCoeff
{
   static int order_;

   static double poly_coeff(const Vector& x)
      {
         int& o = order_;
         double f = 0.0;
         for (int d = 0; d < dimension; d++) {
            f += pow(x[d],o);
         }
         return f;
      }
};
int PolyCoeff::order_ = -1;

void verify_exact_project(int order, Element::Type el_type, int basis_type)
{
   Mesh mesh;
   if (dimension == 1) {
      mesh = Mesh::MakeCartesian1D(1, 1.0);
   }
   if (dimension == 2) {
      mesh = Mesh::MakeCartesian2D(1, 1, el_type, 1.0, 1.0);
   }
   if (dimension == 3) {
      mesh = Mesh::MakeCartesian3D(1, 1, 1, el_type, 1.0, 1.0, 1.0);
   }      
   mesh.EnsureNCMesh();

   L2_FECollection fec(order, dimension, basis_type);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   PolyCoeff pcoeff;
   pcoeff.order_ = order;
   FunctionCoefficient c(PolyCoeff::poly_coeff);

   bool use_L2 = true; // this is what we are testing
   x.ProjectCoefficient(c, use_L2);

   //Visualize(mesh, x, "proj", "proj", Wx, Wy); Wx += offx;
   
   Vector l2_err(mesh.GetNE());

   x.ComputeElementL2Errors(c, l2_err);

   // verify projection is exact
   for (int i = 0; i < l2_err.Size(); i++) {
      REQUIRE( fabs(l2_err(i)) < 1.e-14 );
   }
}

TEST_CASE("L2 Projection Tests", "[GridFunction][Project][L2Project]")
{
   vector<int> orders;
   orders.push_back(0);
   orders.push_back(1);
   orders.push_back(2);
   orders.push_back(3);

   vector<int> basis_types;
   basis_types.push_back(BasisType::GaussLegendre);
   basis_types.push_back(BasisType::GaussLobatto);
   basis_types.push_back(BasisType::Positive);

   vector<Element::Type> el_types_1d; 
   el_types_1d.push_back(Element::SEGMENT);

   vector<Element::Type> el_types_2d;
   el_types_2d.push_back(Element::QUADRILATERAL);
   el_types_2d.push_back(Element::TRIANGLE);

   vector<Element::Type> el_types_3d;
   el_types_3d.push_back(Element::HEXAHEDRON);
   el_types_3d.push_back(Element::TETRAHEDRON);
   el_types_3d.push_back(Element::WEDGE);
   //el_types_3d.push_back(Element::PYRAMID); // not impl. for pos basis
   
   
   SECTION("1D Gridfunctions")
   {
      dimension = 1;
      for (auto order: orders) {
         for (auto basis_type: basis_types) {
            for (auto el_type: el_types_1d) {
               verify_exact_project(order, el_type, basis_type);
            }
         }
      }
   }

   SECTION("2D Gridfunctions")
   {
      dimension = 2;
      for (auto order: orders) {
         for (auto basis_type: basis_types) {
            for (auto el_type: el_types_2d) {
               verify_exact_project(order, el_type, basis_type);
            }
         }
      }
   }

   SECTION("3D Gridfunctions")
   {
      dimension = 3;
      for (auto order: orders) {
         for (auto basis_type: basis_types) {
            for (auto el_type: el_types_3d) {
               verify_exact_project(order, el_type, basis_type);
            }
         }
      }
   }
}

} // namespace project_gf
