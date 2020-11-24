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
      REQUIRE(norm(a) - sqrt(55) == Approx(0.0));
   }
}

TEST_CASE("Dual number tensor tests", "[DualNumber]")
{
   auto eps = std::numeric_limits<double>::epsilon();
   double x = 0.5;

   SECTION("cos")
   {
      auto xd = cos(derivative_wrt(x));
      REQUIRE(abs(-sin(x) - xd.gradient) == Approx(0.0));
   }

   SECTION("exp")
   {
      auto xd = exp(derivative_wrt(x));
      REQUIRE(abs(exp(x) - xd.gradient) == Approx(0.0));
   }

   SECTION("log")
   {
      auto xd = log(derivative_wrt(x));
      REQUIRE(abs(1.0 / x - xd.gradient) == Approx(0.0));
   }

   SECTION("pow")
   {
      // f(x) = x^3/2
      auto xd = pow(derivative_wrt(x), 1.5);
      REQUIRE(abs(1.5 * pow(x, 0.5) - xd.gradient) == Approx(0.0));
   }

   SECTION("mixed operations")
   {
      auto xd = derivative_wrt(x);
      auto r = cos(xd) * cos(xd);
      REQUIRE(abs(-2.0 * sin(x) * cos(x) - r.gradient) == Approx(0.0));

      r = exp(xd) * cos(xd);
      REQUIRE(abs(exp(x) * (cos(x) - sin(x)) - r.gradient) < eps);

      r = log(xd) * cos(xd);
      REQUIRE(abs((cos(x) / x - log(x) * sin(x)) - r.gradient) < eps);

      r = exp(xd) * pow(xd, 1.5);
      REQUIRE(abs((exp(x) * (pow(x, 1.5) + 1.5 * pow(x, 0.5))) - r.gradient)
              < eps);

      tensor<double, 2> vx = {{0.5, 0.25}};
      tensor<double, 2> vre = {{0.894427190999916, 0.4472135954999579}};
      auto vr = norm(derivative_wrt(vx));
      REQUIRE(norm(vr.gradient - vre) < eps);
   }
}
