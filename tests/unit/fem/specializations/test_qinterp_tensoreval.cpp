/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/qinterp/eval.hpp"

TEST_CASE("QInterp TensorEval Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   QuadratureInterpolator::AddTensorEvalSpecializations<
   2, QVectorLayout::byNODES, 1, 3, 3, 2>();
}
