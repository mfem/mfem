#ifndef MFEM_SCHURLSC
#define MFEM_SCHURLSC

#ifndef MFEM_USE_MPI
#endif

#include "../../linalg/linalg.hpp"
#include "../../general/error.hpp"

namespace mfem
{

class SchurLSC : public Operator
{
public:
   SchurLSC(HypreParMatrix *C, HypreParMatrix *B);

   void SetA(HypreParMatrix *A)
   {
      A_ = A;
   }

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void MultTranspose(const Vector &x, Vector &y) const
   {
      mfem_error("Wrong code path.");
   }

   ~SchurLSC();

private:
   HypreParMatrix *A_;
   HypreParMatrix *B_;
   HypreParMatrix *C_;
   HypreParMatrix *CB_;
   HypreBoomerAMG *amgCB_;
   mutable Vector x0, y0, x1;
};

}
#endif
