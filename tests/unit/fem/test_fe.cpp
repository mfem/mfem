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

#include "mfem.hpp"
#include "catch.hpp"

using namespace mfem;

TEST_CASE("H1 Hexahedron Finite Element",
          "[H1_HexahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   H1_HexahedronElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 3                     );
      REQUIRE( fe.GetGeomType()       == Geometry::CUBE        );
      REQUIRE( fe.GetDof()            == (int)pow(p+1,3)       );
      REQUIRE( fe.GetOrder()          == p                     );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
   }
}

TEST_CASE("Nedelec Hexahedron Finite Element",
          "[ND_HexahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   ND_HexahedronElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 3                     );
      REQUIRE( fe.GetGeomType()       == Geometry::CUBE        );
      REQUIRE( fe.GetDof()            == 3*p*(int)pow(p+1,2)   );
      REQUIRE( fe.GetOrder()          == p                     );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_CURL );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::CURL   );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_DIV  );
   }
}

TEST_CASE("Raviart-Thomas Hexahedron Finite Element",
          "[RT_HexahedronElement]"
          "[VectorFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   RT_HexahedronElement fe(p-1);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 3                       );
      REQUIRE( fe.GetGeomType()       == Geometry::CUBE          );
      REQUIRE( fe.GetDof()            == 3*(p+1)*(int)pow(p,2)   );
      REQUIRE( fe.GetOrder()          == p                       );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Qk       );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::VECTOR   );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::H_DIV    );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::DIV      );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::SCALAR   );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::INTEGRAL );
   }
}

TEST_CASE("L2 Hexahedron Finite Element",
          "[L2_HexahedronElement]"
          "[NodalFiniteElement]"
          "[ScalarFiniteElement]"
          "[FiniteElement]")
{
   int p = 1;

   L2_HexahedronElement fe(p);

   SECTION("Attributes")
   {
      REQUIRE( fe.GetDim()            == 3                     );
      REQUIRE( fe.GetGeomType()       == Geometry::CUBE        );
      REQUIRE( fe.GetDof()            == (int)pow(p+1,3)       );
      REQUIRE( fe.GetOrder()          == p                     );
      REQUIRE( fe.Space()             == (int) FunctionSpace::Qk     );
      REQUIRE( fe.GetRangeType()      == (int) FiniteElement::SCALAR );
      REQUIRE( fe.GetMapType()        == (int) FiniteElement::VALUE  );
      REQUIRE( fe.GetDerivType()      == (int) FiniteElement::GRAD   );
      REQUIRE( fe.GetDerivRangeType() == (int) FiniteElement::VECTOR );
      REQUIRE( fe.GetDerivMapType()   == (int) FiniteElement::H_CURL );
   }
}
