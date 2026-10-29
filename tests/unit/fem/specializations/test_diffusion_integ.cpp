/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_diffusion_kernels.hpp"

TEST_CASE("Diffusion Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   DiffusionIntegrator::AddSpecialization<2, 1, 5>();
   DiffusionIntegrator::AddSimplexSpecialization<2, 2, 3>();
}
