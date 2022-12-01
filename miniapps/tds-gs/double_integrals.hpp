#ifndef DOUBLE_INTEGRALS_HPP
#define DOUBLE_INTEGRALS_HPP

#include <mfem.hpp>

using namespace std;
using namespace mfem;

/// Abstract double-integral bilinear form integrator
class DoubleIntegralBFIntegrator
{
public:
   /// The destructor is pure virtual.
   virtual ~DoubleIntegralBFIntegrator() { }

   /** @brief Given a pair of FiniteElement%s and ElementTransformation%s,
       compute the element matrix @a elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el1,
                                      const FiniteElement &el2,
                                      ElementTransformation &Trans1,
                                      ElementTransformation &Trans2,
                                      DenseMatrix &elmat) = 0;
};

/// Double-integral boundary bilinear form integrator
/** Compute the double boundary integral
    \f[
        \int_{\Gamma_2} \int_{\Gamma_1}
            [u(x1)-u(x2)] M(x1,x2) [v(x1)-v(x2)] dx1 dx2
    \f]
    where the test and trial functions, u(x) and v(x) respectively, are in the
    same scalar FE space, H1 or L2.
*/
class DoubleBoundaryBFIntegrator : public DoubleIntegralBFIntegrator
{
protected:
   /// Kernel function \f$ M(x,y) \f$.
   /** TODO: replace with a Coefficient-like class? */
   std::function<double(const Vector &, const Vector &)> M;

public:
   /// Constructor from a given kernel function @a M_.
   DoubleBoundaryBFIntegrator(
      std::function<double(const Vector &, const Vector &)> M_)
      : M(std::move(M_))
   { }

   virtual ~DoubleBoundaryBFIntegrator() { }

   /** @brief Given a pair of FiniteElement%s and ElementTransformation%s,
       compute the element matrix @a elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el1,
                                      const FiniteElement &el2,
                                      ElementTransformation &Trans1,
                                      ElementTransformation &Trans2,
                                      DenseMatrix &elmat);
};

/// Assemble the given DoubleIntegralBFIntegrator into the given BilinearForm
/** TODO: make this part of the BilinearForm class? */
void AssembleDoubleBoundaryIntegrator(BilinearForm &a,
                                      DoubleIntegralBFIntegrator &di_bfi,
                                      int attribute = NULL);





#endif
