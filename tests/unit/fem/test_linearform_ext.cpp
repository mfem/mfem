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
   const bool scalar = problem == LinearFormExtTest::DomainLF ||
                       problem == LinearFormExtTest::DomainLFGrad;
   if (scalar) { MFEM_VERIFY(vdim == 1, "VDIM should be 1"); }
   const bool grad = problem == LinearFormExtTest::DomainLFGrad ||
                     problem == LinearFormExtTest::VectorDomainLFGrad;

   mfem::out << "[LinearFormExt]"
             << " p=" << p
             << " q=" << q
             << (ordering==Ordering::byNODES ? " byNODES" : " byVDIM ")
             << " "<< dim << "D"
             << " "<< vdim << "-"
             << (scalar ? "Scalar" : "Vector")
             << (grad ? "Grad" : "")
             << std::endl;
}

void LinearFormExtTest::Run()
{
   Description();

   AssembleBoth();

   // Test the difference to verify the orderings
   Vector difference = lf_legacy;
   difference -= lf_full;
   REQUIRE(0.0 == MFEM_Approx(difference * difference));

   REQUIRE(lf_full * lf_full == MFEM_Approx(lf_legacy * lf_legacy));
}

} // namespace linearform_ext_tests

} // namespace mfem

TEST_CASE("Linear Form Extension", "[LinearformExt], [CUDA]")
{
   const auto N = GENERATE(3,4);
   const auto p = GENERATE(1,3,6); // limitations: 2D:11, 3D:6
   const auto dim = GENERATE(2,3);
   const auto gll = GENERATE(false,true); // q=p+2, q=p+1

   SECTION("Scalar")
   {
      constexpr auto vdim = 1;
      const auto ordering = Ordering::byNODES;
      const auto problem = GENERATE(LinearFormExtTest::DomainLF,
                                    LinearFormExtTest::DomainLFGrad);
      LinearFormExtTest(N, dim, vdim, ordering, gll, problem, p, true).Run();
   }

   SECTION("Vector")
   {
      const auto vdim = GENERATE(1,5);
      const auto ordering = GENERATE(Ordering::byVDIM, Ordering::byNODES);
      const auto problem = GENERATE(LinearFormExtTest::VectorDomainLF,
                                    LinearFormExtTest::VectorDomainLFGrad);
      LinearFormExtTest(N, dim, vdim, ordering, gll, problem, p, true).Run();
   }
} // test case

