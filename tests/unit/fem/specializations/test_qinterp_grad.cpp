/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/qinterp/grad.hpp"

TEST_CASE("QInterp Grad Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   QuadratureInterpolator::AddGradSpecializations<2, QVectorLayout::byNODES, false,
                          1, 3, 3, 1>();
   QuadratureInterpolator::AddGradSpecializations<2, QVectorLayout::byNODES, true,
                          1, 3, 3, 1>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<
   2, QVectorLayout::byNODES, false, 1, 2, 1>();
   QuadratureInterpolator::AddCollocatedGradSpecializations<
   2, QVectorLayout::byNODES, true, 1, 2, 1>();
}
