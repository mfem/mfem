#ifndef MFEM_HESS_AD_HPP
#define MFEM_HESS_AD_HPP
#include "mfem.hpp"
#include "miniapps/autodiff/admfem.hpp"
namespace mfem
{
/// The class provides an evaluation of the Jacobian of a templated vector
/// function provided in the constructor. The Jacobian is evaluated with the
/// help of automatic differentiation (AD). The template parameters specify the
/// size of the return vector (vector_size), the size of the input vector
/// (state_size), and the size of the parameters supplied to the function.
template<int state_size=1, int param_size=0>
class HessAD : public VectorFuncAutoDiff<1, state_size, param_size>
{
   typedef std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)>
   ADFuncType;
   typedef std::function<void(mfem::Vector&, ad::ADVectorType&, ad::ADVectorType&)>
   ADGradType;
public:
   /// F_ is user implemented function to be differentiated by
   /// VectorFuncAutoDiff. The signature of the function is: F_(mfem::Vector&
   /// parameters, ad::ADVectroType& state_vector, ad::ADVectorType& result).
   /// The parameters vector should have size param_size. The state_vector
   /// should have size state_size, and the result vector should have size
   /// vector_size. All size parameters are teplate parameters in
   /// VectorFuncAutoDiff.
   HessAD(): F(nullptr) {}
   HessAD(ADFuncType F_)
   { SetFunction(F_); }
   void SetFunction(ADFuncType F_)
   {
      F=F_;
      G = [&](mfem::Vector &vparam, ad::ADVectorType &u,
              ad::ADVectorType &grad_out)
      {
         // Result from original function
         ad::ADVectorType uu(u);
         ad::ADVectorType result(1);
         F(vparam, uu, result);

         for (int i = 0; i < u.Size(); ++i)
         {
            uu[i].gradient = 1.0;
            F(vparam, uu, result);
            grad_out[i] = result[0].gradient;
            uu[i].gradient = 0.0;
         }
      };
   }

   /// Evaluates the Jacobian of the vector function F_ for a set of parameters
   /// (vparam) and state vector uu. The Jacobian (jac) has dimensions
   /// [vector_size x state_size].
   void Gradient(mfem::Vector &vparam, mfem::Vector &uu,
                 mfem::Vector &jac) const
   {
      jac.SetSize(state_size);
      jac = 0.0;
      {
         ad::ADVectorType aduu(uu); // all dual numbers are initialized to zero
         ad::ADVectorType rr(1);

         for (int ii = 0; ii < state_size; ii++)
         {
            aduu[ii].gradient = 1.0;
            F(vparam,aduu,rr);
            jac(ii) = rr[0].gradient;
            aduu[ii].gradient = 0.0;
         }
      }
   }

   void Hessian(mfem::Vector &vparam, mfem::Vector &uu,
                mfem::DenseMatrix &hess) const
   {
      hess.SetSize(state_size, state_size);
      hess = 0.0;
      {
         ad::ADVectorType aduu(uu); // all dual numbers are initialized to zero
         ad::ADVectorType rr(state_size);

         for (int ii = 0; ii < state_size; ii++)
         {
            aduu[ii].gradient = 1.0;
            G(vparam,aduu,rr);
            for (int jj = 0; jj < state_size; jj++)
            {
               hess(jj, ii) = rr[jj].gradient;
            }
            aduu[ii].gradient = 0.0;
         }
      }
   }

private:
   ADFuncType F;
   ADGradType G;
};
} // namespace mfem
#endif
