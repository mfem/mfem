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

#include "fem/test_linearform_ext.hpp"

using namespace mfem;
using namespace linearform_ext_tests;

namespace mfem
{

namespace linearform_ext_tests
{

void LinearFormExtTest::Description()
{
   const bool scalar = problem==LinearFormExtTest::DomainLF ||
                       problem==LinearFormExtTest::DomainLFGrad;
   if (scalar) { assert(vdim==1); }
   const bool grad = problem==LinearFormExtTest::DomainLFGrad ||
                     problem==LinearFormExtTest::VectorDomainLFGrad;

   mfem::out << "[LinearFormExt]"
             << " p=" << p
             << " q=" << q
             << " "<< dim << "D"
             << " "<< vdim << "-"
             << (scalar?"Scalar":"Vector")
             << (grad?"Grad":"")
             //<< (gll ? ", GLL" : ", GL")
             << (ordering==Ordering::byNODES?", byNODES":", byVDIM")
             << std::endl;
}

void LinearFormExtTest::Run()
{
   Description();
   AssembleBoth();
   REQUIRE((lf_full*lf_full) == MFEM_Approx(lf_legacy*lf_legacy));
   // Test also the diffs to verify the orderings
   lf_legacy -= lf_full;
   REQUIRE(0.0 == MFEM_Approx(lf_legacy*lf_legacy));
}

} // namespace linearform_ext_tests

} // namespace mfem

TEST_CASE("Linear Form Extension", "[LinearformExt], [CUDA]")
{
   const auto p = GENERATE(1,2,3);
   const auto N = GENERATE(2,3,4);
   const auto dim = GENERATE(2,3);
   const auto gll = GENERATE(false,true); // q=p+2, q=p+1

   SECTION("Scalar")
   {
      const auto vdim = 1;
      const auto ordering = GENERATE(Ordering::byNODES); // default
      const auto problem = GENERATE(LinearFormExtTest::DomainLF,
                                    LinearFormExtTest::DomainLFGrad);
      LinearFormExtTest(N, dim, vdim, ordering, gll, problem, p, true).Run();
   }

   SECTION("Vector")
   {
      const auto vdim = GENERATE(1,5,7);
      const auto ordering = GENERATE(Ordering::byVDIM, Ordering::byNODES);
      const auto problem = GENERATE(LinearFormExtTest::VectorDomainLF,
                                    LinearFormExtTest::VectorDomainLFGrad);
      LinearFormExtTest(N, dim, vdim, ordering, gll, problem, p, true).Run();
   }

} // test case

