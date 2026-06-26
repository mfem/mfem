/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/qinterp/det.hpp"

TEST_CASE("QInterp Det Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   QuadratureInterpolator::AddDetSpecializations<2,3,2,2>();
}
