#include "ad_native.hpp"

namespace mfem
{
int Evaluator::GetSize(const param_t &param)
{
   return std::visit([](auto arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, real_t>)
      {
         return 1;
      }
      if constexpr (std::is_same_v<T, Vector>)
      {
         return 1;
      }
      if constexpr (std::is_same_v<T, DenseMatrix>)
      {
         return 1;
      }
      if constexpr (std::is_same_v<T, const real_t*>)
      {
         return 1;
      }
      if constexpr (std::is_same_v<T, const Vector*>)
      {
         return arg->Size();
      }
      if constexpr (std::is_same_v<T, const DenseMatrix*>)
      {
         return arg->Height()*arg->Width();
      }
      if constexpr (std::is_same_v<T, Coefficient*>)
      {
         return 1;
      }
      if constexpr (std::is_same_v<T, VectorCoefficient*>)
      {
         return arg->GetVDim();
      }
      if constexpr (std::is_same_v<T, MatrixCoefficient*>)
      {
         return arg->GetHeight() * arg->GetWidth();
      }
      if constexpr (std::is_same_v<T, const GridFunction*> ||
                    std::is_same_v<T, const ParGridFunction*>)
      {
         return arg->FESpace()->GetVDim();
      }
      if constexpr (std::is_same_v<T, const QuadratureFunction*>)
      {
         return arg->GetVDim();
      }
      MFEM_ABORT("Evaluator: Unsupported parameter type");
      return 0;
   }, param);
}

Evaluator::~Evaluator()
{
   for (int i=0; i<params.size(); i++)
   {
      if (owns[i])
      {
         std::visit([](auto &arg)
         {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_pointer_v<T>)
            {
               delete arg;
            }
         }, params[i]);
      }
   }
}
int Evaluator::Add(param_t param, bool eval_owns)
{
   int idx = params.size();
   params.push_back(param);
   offsets.Append(offsets.Last() + GetSize(param));
   val.Update(offsets);
   owns.Append(eval_owns);
   std::visit([&](auto arg)
   {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, real_t>)
      {
         MFEM_VERIFY(eval_owns==false,
                     "Evaluator::Add: real_t parameter cannot own the value");
         val.GetBlock(idx) = arg;
      }
      if constexpr (std::is_same_v<T, Vector>)
      {
         MFEM_VERIFY(eval_owns==false,
                     "Evaluator::Add: real_t parameter cannot own the value");
         val.GetBlock(idx) = arg;
      }
      if constexpr (std::is_same_v<T, DenseMatrix>)
      {
         MFEM_VERIFY(eval_owns==false,
                     "Evaluator::Add: real_t parameter cannot own the value");
         Vector v(arg.GetData(), arg.TotalSize());
         val.GetBlock(idx) = v;
      }
   }, param);
   return idx;
}
void Evaluator::Replace(size_t i, param_t param)
{
   MFEM_VERIFY(i < params.size(),
               "Evaluator::Set: index out of range");
   params[i] = param;
   int size = GetSize(param);
   MFEM_VERIFY(size == offsets[i+1] - offsets[i],
               "Evaluator::Set: size mismatch for parameter at index " << i
               << ": expected " << (offsets[i+1] - offsets[i]) << ", got " << size);
}

const Vector& Evaluator::Eval(int i, ElementTransformation &Tr,
                              const IntegrationPoint &ip) const
{
   std::visit([&](auto arg)
   {
      Vector &v = this->val.GetBlock(i);
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, real_t> ||
                    std::is_same_v<T, Vector> ||
                    std::is_same_v<T, DenseMatrix>)
      {
         // Already stored, do nothing
         return;
      }
      if constexpr (std::is_same_v<T, const real_t*>)
      {
         v = *arg;
         return;
      }
      if constexpr (std::is_same_v<T, const Vector*>)
      {
         v = *arg;
         return;
      }
      if constexpr (std::is_same_v<T, const DenseMatrix*>)
      {
         DenseMatrix m(v.GetData(), arg->Height(), arg->Width());
         m = *arg;
         return;
      }
      if constexpr (std::is_same_v<T, Coefficient*>)
      {
         v(0) = arg->Eval(Tr, ip);
         return;
      }
      if constexpr (std::is_same_v<T, VectorCoefficient*>)
      {
         arg->Eval(v, Tr, ip);
         return;
      }
      if constexpr (std::is_same_v<T, MatrixCoefficient*>)
      {
         DenseMatrix m(v.GetData(), arg->GetHeight(), arg->GetWidth());
         arg->Eval(m, Tr, ip);
         return;
      }
      if constexpr (std::is_same_v<T, const GridFunction*> ||
                    std::is_same_v<T, const ParGridFunction*>)
      {
         arg->GetVectorValue(Tr, ip, v);
         return;
      }
      if constexpr (std::is_same_v<T, const QuadratureFunction*>)
      {
         arg->GetValues(Tr.ElementNo, ip.index, v);
         return;
      }
   }, params[i]);
   return this->val.GetBlock(i);
}

void ADFunction::Gradient(const Vector &x, ElementTransformation &Tr,
                          const IntegrationPoint &ip,
                          Vector &J) const
{
   ProcessParameters(Tr, ip);
   Gradient(x, J);
}
void ADFunction::Gradient(const Vector &x, Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Gradient: x.Size() must match n_input");
   J.SetSize(x.Size());
   ADVector x_ad(x);
   for (int i=0; i < n_input; i++)
   {
      x_ad[i].gradient = 1.0;
      ADReal_t result = (*this)(x_ad);
      J[i] = result.gradient;
      x_ad[i].gradient = 0.0;
   }
}

void ADFunction::Hessian(const Vector &x, ElementTransformation &Tr,
                         const IntegrationPoint &ip,
                         DenseMatrix &H) const
{
   ProcessParameters(Tr, ip);
   Hessian(x, H);
}

void ADFunction::Hessian(const Vector &x, DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Hessian: x.Size() must match n_input");
   H.SetSize(x.Size(), x.Size());
   AD2Vector x_ad(x);
   for (int i=0; i<n_input; i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient.value = 1.0;
         AD2Real_t result = (*this)(x_ad);
         H(j, i) = result.gradient.gradient;
         H(i, j) = result.gradient.gradient;
         x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}

void ADVectorFunction::Gradient(const Vector &x, DenseMatrix &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADVectorFunction::Gradient: x.Size() must match n_input");
   ADVector x_ad(x);
   ADVector Fx(n_output);
   J.SetSize(n_output, n_input);
   for (int i=0; i<n_input; i++)
   {
      x_ad[i].gradient = 1.0;
      Fx = ADReal_t();
      (*this)(x_ad, Fx);
      for (int j=0; j<n_output; j++)
      {
         J(j,i) = Fx[j].gradient;
      }
      x_ad[i].gradient = 0.0; // Reset gradient for next iteration
   }
}

void ADVectorFunction::Hessian(const Vector &x, DenseTensor &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADVectorFunction::Gradient: x.Size() must match n_input");
   AD2Vector x_ad(x);
   AD2Vector Fx(n_output);
   H.SetSize(n_input, n_input, n_output);
   for (int i=0; i<n_input; i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient.value = 1.0;
         Fx = AD2Real_t();
         (*this)(x_ad, Fx);
         for (int k=0; k<n_output; k++)
         {
            H(j, i, k) = Fx[k].gradient.gradient;
            H(i, j, k) = Fx[k].gradient.gradient;
         }
         x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}


Lagrangian Lagrangian::AddEqConstraint(ADFunction &constraint,
                                       real_t target)
{
   eq_con.push_back(&constraint);
   int numCon = eq_con.size();
   eq_rhs.SetSize(numCon);
   eq_rhs[numCon - 1] = target;
   return *this;
}

void Lagrangian::ProcessParameters(ElementTransformation &Tr,
                                   const IntegrationPoint &ip) const
{
   objective.ProcessParameters(Tr, ip);
   for (auto *con : eq_con) { con->ProcessParameters(Tr, ip); }
}

ALFunctional ALFunctional::AddEqConstraint(ADFunction &constraint,
      real_t target)
{
   eq_con.push_back(&constraint);
   int numCon = eq_con.size();
   eq_rhs.SetSize(numCon);
   lambda.SetSize(numCon);

   eq_rhs[numCon - 1] = target;
   lambda[numCon - 1] = 0.0;
   return *this;
}

void ALFunctional::SetLambda(const Vector &lambda)
{
   MFEM_VERIFY(lambda.Size() == this->lambda.Size(),
               "ALFunctional: lambda size mismatch");
   this->lambda = lambda;
}

void ALFunctional::SetPenalty(real_t mu)
{
   MFEM_VERIFY(mu >= 0.0, "ALFunctional: mu must be non-negative");
   this->penalty = mu;
}

void ALFunctional::ProcessParameters(ElementTransformation &Tr,
                                     const IntegrationPoint &ip) const
{
   objective.ProcessParameters(Tr, ip);
   for (auto *con : eq_con) { con->ProcessParameters(Tr, ip); }
}

} // namespace mfem
