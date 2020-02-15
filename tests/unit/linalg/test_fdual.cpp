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

#ifdef MFEM_USE_EXCEPTIONS

using namespace mfem;

TEST_CASE("Simple AD tests", "[Simple_AD_tests]")
{
   SECTION("sin")
   {
   	double x=0.5;
   	double d;
	ad::FDual<double> xx(x,1.0);
	ad::FDual<double> lrez;	
	lrez=ad::sin(xx);
    	d=std::cos(x);
	REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("cos")
   {
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::cos(xx);
        d=-std::sin(x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("tan")
   {
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::tan(xx);
        d=1.0+std::tan(x)*std::tan(x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("exp")
   {
	double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::exp(xx);
        d=exp(x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());   
   }

   SECTION("log")
   {
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::log(xx);
        d=1.0/x;
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("pow")
   {
	double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::pow(xx,1.5);
        d=1.5*std::pow(x,0.5);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());

   }

   SECTION("atan")
   {
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::atan(xx);
        d=1.0/(1.0+x*x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("asin")
   {
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::asin(xx);
        d=1.0/std::sqrt(1.0-x*x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }

   SECTION("acos")
   {
	std::cout<<"Test acos!"<<std::endl;
        double x=0.5;
        double d;
        ad::FDual<double> xx(x,1.0);
        ad::FDual<double> lrez;
        lrez=ad::acos(xx);
        d=-1.0/std::sqrt(1.0-x*x);
        REQUIRE(std::abs(d-lrez.dual())<std::numeric_limits<double>::epsilon());
   }




}

#endif  // MFEM_USE_EXCEPTIONS
