/// Tests which make sure adding user-defined kernel specializations work.
/// These tests are compile/link-only tests

#include "mfem.hpp"
#include "unit_tests.hpp"

#include "fem/integ/bilininteg_dgtrace_kernels.hpp"

TEST_CASE("DGTrace Kernel Specializations", "[Specializations]")
{
   using namespace mfem;

   DGTraceIntegrator::AddSpecialization<2, 2, 3>();
}
