#ifndef MFEM_SCHURLSC
#define MFEM_SCHURLSC

#ifndef MFEM_USE_MPI
#endif

#include "../../general/error.hpp"
#include "../../linalg/linalg.hpp"

namespace mfem {

class SchurLSC : public Operator
{
public:
   SchurLSC(BlockOperator *op);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void MultTranspose(const Vector &, Vector &) const
   {
      mfem_error("Wrong code path.");
   }

   ~SchurLSC();

private:
   BlockOperator *op_;
   HypreParMatrix *B_;
   HypreParMatrix *C_;
   HypreParMatrix *CB_;
   HypreBoomerAMG *amgCB_;
   mutable Vector x0, y0, x1;
};

} // namespace mfem
#endif
