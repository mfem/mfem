#include "mfem.hpp"
#include "unit_tests.hpp"

#ifdef MFEM_USE_ENZYME

#include "../../../general/enzyme.hpp"

template<typename VectorT>
void square(const VectorT& v, double& y)
{
   for (int i = 0; i < 4; i++)
   {
      y += v[i]*v[i];
   }
}

template<typename VectorT>
void dsquare(const VectorT& v, double& y, VectorT& dydv)
{
   double seed = 1.0;
   __enzyme_autodiff<void>(square<VectorT>, &v, &dydv, &y, &seed);
}

template<typename VectorT>
void run_test()
{
   VectorT v(4);
   v[0] = 2.0;
   v[1] = 3.0;
   v[2] = 1.0;
   v[3] = 7.0;

   double yy = 0;
   VectorT dydv(4);
   dydv[0] = 0;
   dydv[1] = 0;
   dydv[2] = 0;
   dydv[3] = 0;
   dsquare(v, yy, dydv);

   REQUIRE(dydv[0] == MFEM_Approx(4.0));
   REQUIRE(dydv[1] == MFEM_Approx(6.0));
   REQUIRE(dydv[2] == MFEM_Approx(2.0));
   REQUIRE(dydv[3] == MFEM_Approx(14.0));
}

TEST_CASE("AD Vector implementation", "[Enzyme]")
{
   run_test<mfem::Vector>();
   run_test<std::vector<double>>();
}

#endif