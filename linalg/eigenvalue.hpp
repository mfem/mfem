#ifndef MFEM_EIGENVALUE
#define MFEM_EIGENVALUE

#include "operator.hpp"

namespace mfem
{

class PowerMethod
{
   Vector v1;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:

#ifdef MFEM_USE_MPI
   PowerMethod() : comm(MPI_COMM_NULL) {}
#else
   PowerMethod() {}
#endif

#ifdef MFEM_USE_MPI
   PowerMethod(MPI_Comm _comm) : comm(_comm) {}
#endif

   /// Returns an estimate on the largest eigenvalue of the operator \p opr
   /// using the power method
   double EstimateLargestEigenvalue(Operator& opr, Vector& v0,
                                    int numSteps = 10, double tolerance = 1e-8,
                                    int seed = 12345)
   {
      v1.SetSize(v0.Size());
      v0.Randomize(seed);

      double eigenvalue = 1.0;

      for (int iter = 0; iter < numSteps; ++iter)
      {
         double normV0;

#ifdef MFEM_USE_MPI
         if (comm != MPI_COMM_NULL)
         {
            normV0 = InnerProduct(comm, v0, v0);
         }
         else
         {
            normV0 = InnerProduct(v0, v0);
         }
#else
         normV0 = InnerProduct(v0, v0);
#endif

         v0 /= sqrt(normV0);
         opr.Mult(v0, v1);

         double eigenvalueNew;
#ifdef MFEM_USE_MPI
         if (comm != MPI_COMM_NULL)
         {
            eigenvalueNew = InnerProduct(comm, v0, v1);
         }
         else
         {
            eigenvalueNew = InnerProduct(v0, v1);
         }
#else
         eigenvalueNew = InnerProduct(v0, v1);
#endif
         double diff = std::abs((eigenvalueNew - eigenvalue) / eigenvalue);

         eigenvalue = eigenvalueNew;
         std::swap(v0, v1);

         if (diff < tolerance)
         {
            break;
         }
      }

      return eigenvalue;
   };
};

} // namespace mfem
#endif
