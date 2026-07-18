/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_dgdiffusion_kernels.hpp"

TEST_CASE("DGDiffusion Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   DGDiffusionIntegrator::AddSpecialization<2, 2, 4>();
}
