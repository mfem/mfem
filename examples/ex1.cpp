#include "mfem.hpp"

using namespace std;
using namespace mfem;

void axpy(Vector *ymem, const Vector *amem, const Vector *xmem)
{
   const int n = xmem->Size();
   const auto alpha = amem->Read();
   const auto x = xmem->Read();
   auto y = ymem->ReadWrite();

   MFEM_FORALL(i, n,
   {
      y[i] = alpha[0] * x[i] + y[i];
   });
}

int main(int argc, char *argv[])
{
   const char *device_config = "raja-cpu";
   Device device(device_config);
   device.Print();

   const int n = 1000;

   Vector y(n);
   Vector alpha(1);
   Vector x(n);
   y = 2.0;
   x = 1.0;

   // Regular call
   axpy(&y, &alpha, &x);

   y.HostRead();
   for (const auto &v : y)
   {
      MFEM_ASSERT(v == alpha[0] * 1.0 + 2.0, "error");
   }

   // Enzyme fwddiff call
   Vector dy(n);
   Vector dalpha(1);
   Vector dx(n);

   __enzyme_fwddiff<void>(
      (void*)axpy,
      enzyme_dup, y, dy,
      enzyme_dup, alpha, dalpha,
      enzyme_dup, x, dx);

   return 0;
}
