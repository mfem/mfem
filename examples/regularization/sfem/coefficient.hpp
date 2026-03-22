#include "mfem.hpp"

namespace mfem
{

/// Matrix coefficient defined as the Stress of a vector GridFunction
/// Here, we use $ \sigma(u) = \lambda div(u) I + 2 \mu \epsilon(u) $
/// where $ \epsilon(u) = 1/2 (\nabla u + (\nabla u)^T) $ and
/// $ \lambda, \mu $ are the Lamé parameters.
class StressGridFunctionCoefficient : public MatrixCoefficient
{
protected:
   real_t L, M;
   const GridFunction *GridFunc;

public:
   /** @brief Construct the coefficient with a vector grid function @a gf. The
       grid function is not owned by the coefficient. */
   StressGridFunctionCoefficient(real_t lambda, real_t mu,
      const GridFunction *gf);

   /// Set the vector grid function.
   void SetGridFunction(const GridFunction *gf);

   /// Get the vector grid function.
   const GridFunction * GetGridFunction() const { return GridFunc; }

   /// Evaluate the stress grid function coefficient at @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~StressGridFunctionCoefficient() { }
};

/// Vector coefficient defined as a (row-wise) flattened matrix coefficient
class FlatVectorCoefficient : public VectorCoefficient
{
protected:
   MatrixCoefficient *a;
   real_t alpha;

public:
   /** @brief Construct the coefficient with a matrix coefficient @a A. */
   FlatVectorCoefficient(MatrixCoefficient &A, real_t _alpha = 1.0);

   /// Evaluate the vector coefficient at @a ip.
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;
};

} // namespace mfem
