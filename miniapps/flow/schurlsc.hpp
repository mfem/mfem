#ifndef MFEM_SCHURLSC
#define MFEM_SCHURLSC

#ifndef MFEM_USE_MPI
#endif

#include "../../general/error.hpp"
#include "../../linalg/linalg.hpp"

namespace mfem
{
/**
 * @brief SchurLSC approximates the Schur complement of a Saddle point problem.
 *
 * Based on a BlockOperator with saddle point structure, this Operator constructs
 * an approximation of the Schur complement \f$S = -C A^{-1} B\f$. Reviewing the
 * references a good approximation for \f$S\f$ is
 *
 * \f$\hat{S}^{-1} = (CB)^{-1}(CAB)(CB)^{-1}\f$.
 *
 * The matrix product \f$CB\f$ is automatically computed and the inverse is approximated
 * by one V-cycle of HypreBoomerAMG.
 *
 * This class uses the BlockOperator to keep track of possible changing matrices.
 *
 * [1] Elman, Howle, Shadid, Shuttleworth, and Tuminaro Block preconditioners
 * based on approximate commutators, 2006
 *
 * [2] Silvester, Elman, Kay, Wathen, Efficient preconditioning of the linearized
 * Navier Stokes equations for incompressible flow, 2001
 */
class SchurLSC : public Operator
{
public:
   SchurLSC(BlockOperator *op);

   /**
    * @brief Apply \f$(CB)^{-1}(CAB)(CB)^{-1}\f$ to the residual @a x.
    */
   void Mult(const Vector &x, Vector &y) const override;

   void MultTranspose(const Vector &, Vector &) const override
   {
      mfem_error("SchurLSC: Not supported.");
   }

   ~SchurLSC() override;

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
