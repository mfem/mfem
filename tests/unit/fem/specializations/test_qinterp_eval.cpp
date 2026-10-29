/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/qinterp/eval.hpp"

TEST_CASE("QInterp Eval Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   QuadratureInterpolator::AddEvalSpecializations<2, 1, 1, 2>();
}
