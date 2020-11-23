#include "mfem.hpp"
#include "tensor.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace mfem;

TEST_CASE("Dual number tensor tests", "[DualNumber]")
{
   SECTION("cos")
   {
      double x = 0.5;
      auto xd = cos(derivative_wrt(x));
      REQUIRE(std::abs(-std::sin(x) - xd.gradient)
              < std::numeric_limits<double>::epsilon());
   }
}
