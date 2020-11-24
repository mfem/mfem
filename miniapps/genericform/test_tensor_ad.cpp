#include "mfem.hpp"
#include "tensor.hpp"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace mfem;

TEST_CASE("Tensor tests", "[tensor]")
{
   SECTION("norm")
   {
      tensor<double, 5> a = {{1.0, 2.0, 3.0, 4.0, 5.0}};
      REQUIRE(norm(a) - sqrt(55) < std::numeric_limits<double>::epsilon());
   }
}

TEST_CASE("Dual number tensor tests", "[DualNumber]")
{
   double x = 0.5;

   SECTION("cos")
   {
      auto xd = cos(derivative_wrt(x));
      REQUIRE(abs(-sin(x) - xd.gradient)
              < std::numeric_limits<double>::epsilon());
   }

   SECTION("exp")
   {
      auto xd = exp(derivative_wrt(x));
      REQUIRE(abs(exp(x) - xd.gradient)
              < std::numeric_limits<double>::epsilon());
   }

   SECTION("log")
   {
      auto xd = log(derivative_wrt(x));
      REQUIRE(abs(1.0 / x - xd.gradient)
              < std::numeric_limits<double>::epsilon());
   }

   SECTION("pow")
   {
      // f(x) = x^3/2
      auto xd = pow(derivative_wrt(x), 1.5);
      REQUIRE(abs(1.5 * pow(x, 0.5) - xd.gradient)
              < std::numeric_limits<double>::epsilon());
   }
}
