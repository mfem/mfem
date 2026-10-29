/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_vecdiffusion_pa.hpp"

TEST_CASE("VectorDiffusion Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   VectorDiffusionIntegrator::AddSpecialization<2, 2, 2, 3>();
}
