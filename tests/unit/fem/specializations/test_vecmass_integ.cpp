/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_vecmass_pa.hpp"

TEST_CASE("Vector Mass Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   VectorMassIntegrator::VectorMassAddMultPA::Specialization<2, 2, 4>::Add();
}
