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

#include "test_linearform_ext.hpp"

using namespace mfem;
using namespace linearform_ext_tests;

namespace mfem
{

namespace linearform_ext_tests
{

void LinearFormExtTest::Description()
{
   mfem::out << "[LinearFormExt]"
             << " p=" << p
             << " q=" << q
             << " "<< dim << "D"
             << " "<< vdim << "-"
             << (problem%2?"Scalar":"Vector")
             << (problem>2?"Grad":"")
             //<< (gll ? "GLL" : "GL")
             << std::endl;
}

void LinearFormExtTest::Run()
{
   Description();
   AssembleBoth();
   REQUIRE((lf_full*lf_full) == MFEM_Approx(lf_legacy*lf_legacy));
}

} // namespace linearform_ext_tests

} // namespace mfem

TEST_CASE("Linear Form Extension", "[LinearformExt], [CUDA]")
{
   const auto N = GENERATE(2,3,4);
   const auto dim = GENERATE(2,3);
   const auto order = GENERATE(1,2,3);
   const auto gll = GENERATE(false,true); // q=p+2, q=p+1

   SECTION("Scalar")
   {
      const auto vdim = 1;
      const auto problem = GENERATE(LinearFormExtTest::DomainLF,
                                    LinearFormExtTest::DomainLFGrad);
      LinearFormExtTest(N, dim, vdim, gll, problem, order, true).Run();
   }

   SECTION("Vector")
   {
      const auto vdim = GENERATE(1,2,24);
      const auto problem = GENERATE(LinearFormExtTest::VectorDomainLF,
                                    LinearFormExtTest::VectorDomainLFGrad);
      LinearFormExtTest(N, dim, vdim, gll, problem, order, true).Run();
   }

} // test case

