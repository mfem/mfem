#include "mfem.hpp"

namespace mfem
{

/// Matrix coefficient defined as the Stress of a vector GridFunction
/// Here, we use $ \sigma(u) = \lambda div(u) I + 2 \mu \epsilon(u) $
/// where $ \epsilon(u) = 1/2 (\nabla u + (\nabla u)^T) $ and
/// $ \lambda, \mu $ are the Lamé parameters. The Lamé parameters may be
/// supplied either as constants or as (possibly spatially varying)
/// Coefficient objects.
class StressGridFunctionCoefficient : public MatrixCoefficient
{
protected:
   Coefficient *lambda, *mu;
   bool own_coefs;
   const GridFunction *GridFunc;

public:
   /** @brief Construct with constant Lamé parameters @a lambda and @a mu, and a
       vector grid function @a gf. The grid function is not owned by the
       coefficient. */
   StressGridFunctionCoefficient(real_t lambda, real_t mu,
      const GridFunction *gf);

   /** @brief Construct with Lamé parameter coefficients @a lambda and @a mu
       (which may vary in space), and a vector grid function @a gf. Neither the
       coefficients nor the grid function are owned by this object. */
   StressGridFunctionCoefficient(Coefficient &lambda, Coefficient &mu,
      const GridFunction *gf);

   /// Set the vector grid function.
   void SetGridFunction(const GridFunction *gf);

   /// Get the vector grid function.
   const GridFunction * GetGridFunction() const { return GridFunc; }

   /// Set the Lamé parameters to constants (replaces any existing ones).
   void SetLameParameters(real_t lambda, real_t mu);

   /// Set the Lamé parameters to coefficients (not owned by this object).
   void SetLameParameters(Coefficient &lambda, Coefficient &mu);

   /// Evaluate the stress grid function coefficient at @a ip.
   void Eval(DenseMatrix &K, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   virtual ~StressGridFunctionCoefficient();
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
