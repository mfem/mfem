/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_mass_kernels.hpp"

TEST_CASE("Mass Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   MassIntegrator::AddSpecialization<2, 1, 3>();
   MassIntegrator::AddSimplexSpecialization<2, 2, 4>();
}
