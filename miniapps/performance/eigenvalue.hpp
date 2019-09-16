#ifndef MFEM_EIGENVALUE
#define MFEM_EIGENVALUE

#include "../../linalg/operator.hpp"

namespace mfem
{

namespace PowerMethod
{

double EstimateLargestEigenvalue(MPI_Comm comm, Operator& opr, Vector& v0, int numSteps = 10, double tolerance = 1e-8, int seed = 12345)
{
   Vector v1(v0.Size());

   v0.Randomize(seed);

   double eigenvalue = 1.0;

   for (int iter = 0; iter < numSteps; ++iter)
   {
      double normV0 = InnerProduct(comm, v0, v0);
      v0 /= sqrt(normV0);
      
      opr.Mult(v0, v1);

      double eigenvalueNew = InnerProduct(comm, v0, v1);
      double diff = std::abs((eigenvalueNew - eigenvalue) / eigenvalue);

      eigenvalue = eigenvalueNew;
      std::swap(v0, v1);

      if (diff < tolerance)
      {
         break;
      }
   }

   return eigenvalue;
}

}
}
#endif