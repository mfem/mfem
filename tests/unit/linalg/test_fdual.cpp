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


template<typename tbase>
tbase exprp02(tbase x,tbase y)
{
   return sin(x)*cos(y)+tan(x*y);
}

template<typename tbase>
tbase exprp02x(tbase x,tbase y)
{
   return cos(x)*cos(y)+y*(1.0+pow(tan(x*y),2.0));
}

template<typename tbase>
tbase exprp02y(tbase x,tbase y)
{
   return -sin(x)*sin(y)+x*(1.0+pow(tan(x*y),2.0));
}


TEST_CASE("Simple AD tests", "[Simple_AD_tests]")
{
   SECTION("sin")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::sin(xx);
      d = std::cos(x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("cos")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::cos(xx);
      d = -std::sin(x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("tan")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::tan(xx);
      d = 1.0+std::tan(x)*std::tan(x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("exp")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::exp(xx);
      d = exp(x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("log")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::log(xx);
      d = 1.0/x;
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("pow")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::pow(xx,1.5);
      d = 1.5*std::pow(x,0.5);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());

   }

   SECTION("atan")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::atan(xx);
      d = 1.0/(1.0+x*x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("asin")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::asin(xx);
      d = 1.0/std::sqrt(1.0-x*x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("acos")
   {
      double x = 0.5;
      double d;
      ad::FDual<double> xx(x,1.0);
      ad::FDual<double> lrez;
      lrez = ad::acos(xx);
      d = -1.0/std::sqrt(1.0-x*x);
      REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("general")
   {
      double x = 1.0;
      double y = 1.5;

      double pr = exprp02(x,y);
      double dx = exprp02x(x,y);
      double dy = exprp02y(x,y);

      {
         mfem::ad::FDual<double> xx(x,1.0);
         mfem::ad::FDual<double> yy(y,0.0);
         mfem::ad::FDual<double> rr=exprp02(xx,yy);
         REQUIRE(std::abs(rr.real()-pr)<std::numeric_limits<double>::epsilon());
         REQUIRE(std::abs(rr.dual()-dx)<std::numeric_limits<double>::epsilon());
      }

      {
         mfem::ad::FDual<double> xx(x,0.0);
         mfem::ad::FDual<double> yy(y,1.0);
         mfem::ad::FDual<double> rr=exprp02(xx,yy);
         REQUIRE(std::abs(rr.real()-pr)<std::numeric_limits<double>::epsilon());
         REQUIRE(std::abs(rr.dual()-dy)<std::numeric_limits<double>::epsilon());
      }
   }

   SECTION("second_derivative")
   {

      double x = 0.5;
      double d;
      mfem::ad::FDual<mfem::ad::FDual<double>> xxx(mfem::ad::FDual<double>(x,1.0),
                                                   mfem::ad::FDual<double>(1.0,0.0));
      mfem::ad::FDual<mfem::ad::FDual<double>> drez=mfem::ad::exp(xxx);
      d=exp(x);
      REQUIRE(std::abs(d-drez.dual().dual())<std::numeric_limits<double>::epsilon());

      drez = mfem::ad::log(xxx);
      d = -1.0/(x*x);
      REQUIRE(std::abs(d-drez.dual().dual())<std::numeric_limits<double>::epsilon());

      drez = mfem::ad::sin(xxx);
      d = -sin(x);
      REQUIRE(std::abs(d-drez.dual().dual())<std::numeric_limits<double>::epsilon());

      drez = mfem::ad::cos(xxx);
      d = -cos(x);
      REQUIRE(std::abs(d-drez.dual().dual())<std::numeric_limits<double>::epsilon());

   }




}

