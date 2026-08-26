/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/dgmassinv_kernels.hpp"

TEST_CASE("DGMassInverse Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   DGMassInverse::CGKernels::Specialization<2, 1, 2>::Add();
}
