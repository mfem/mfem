/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/qinterp/eval_hdiv.hpp"

TEST_CASE("QInterp Eval HDiv Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   QuadratureInterpolator::TensorEvalHDivKernels::Specialization<
      2, QVectorLayout::byNODES, QuadratureInterpolator::PHYSICAL_VALUES, 2,
      4>::Add();
   QuadratureInterpolator::TensorEvalHDivKernels::Specialization<
      2, QVectorLayout::byNODES, QuadratureInterpolator::PHYSICAL_MAGNITUDES, 2,
      4>::Add();
}
