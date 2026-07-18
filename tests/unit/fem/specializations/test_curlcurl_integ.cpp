/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_hcurl_kernels.hpp"

TEST_CASE("CurlCurl Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   CurlCurlIntegrator::AddSpecialization<3, 2, 4>();
}
