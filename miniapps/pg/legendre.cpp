#include "legendre.hpp"

namespace mfem
{

// void CoefficientScaledLegendreFunction::update_scale_shift(
//    ElementTransformation &Tr) const
// {
//    MFEM_ASSERT(scale_cf != nullptr && shift_cf != nullptr,
//                "Scale and shift Coefficients must be set.");
//    scale = scale_cf->Eval(Tr, Tr.GetIntPoint());
//    shift = shift_cf->Eval(Tr, Tr.GetIntPoint());
// }
// Vector& CoefficientScaledLegendreFunction::transform_vector(const Vector &x,
//                                                             ElementTransformation &Tr) const
// {
//    update_scale_shift(Tr);
//    y.SetSize(x.Size());
//    for (int i=0; i<x.Size(); i++)
//    {
//       y[i] = (x[i] - shift) / scale;
//    }
//    return y;
// }
//
// // @brief Evaluate the inverse gradient of the function at a point x
// // @param[in] x The point at which to evaluate the inverse gradient
// // @param[out] invg The vector to store the inverse gradient
// void CoefficientScaledLegendreFunction::gradinv(
//    const Vector &x, ElementTransformation &Tr, Vector &invg) const
// {
//    update_scale_shift(Tr);
//    y = x;
//    y *= scale;
//    f->gradinv(y, invg);
//    invg *= scale;
//    invg += shift;
// }
//
//
// Vector& TransformedLegendreFunction::transform_vector(const Vector &x) const
// {
//    MFEM_ASSERT(!gradinv_only,
//                "Cannot evaluate forward transform when gradinv_only is true.");
//    MFEM_ASSERT(x.Size() == A.Width(),
//                "Input vector size does not match transformation matrix.");
//    MFEM_ASSERT(b.Size() == x.Size(),
//                "Translation vector size does not match input vector.");
//    x_shifted.SetSize(x.Size());
//    add(x, -1.0, b, x_shifted);
//
//    x_transformed.SetSize(A.Width());
//    Ainv.Mult(x_shifted, x_transformed);
//
//    return x_transformed;
// }
// @brief Construct a TransformedLegendreFunction with given A and b
// @param[in] func The base Legendre function g
// @param[in] A_ The transformation matrix A
// @param[in] b_ The translation vector b
// @param[in] gradinv_only If true, only the gradinv method is supported
//
// NOTE: gradinv_only is useful when A is not invertible or avoid computing
// the inverse for performance reasons
// TransformedLegendreFunction::TransformedLegendreFunction(
//    std::unique_ptr<LegendreFunction> &func,
//    const DenseMatrix A_, const Vector b_,
//    bool gradinv_only)
//    : LegendreFunction()
//    , f(std::move(func))
//    , A(A_), b(b_)
//    , gradinv_only(gradinv_only)
// {
//    if (!gradinv_only)
//    {
//       DenseMatrix At(A.Width(), A.Height());
//       A.Transpose(At);
//       Ainv.Factor(A);
//       Atinv.Factor(At);
//    }
// }
//
// void TransformedLegendreFunction::SetAffine(const DenseMatrix A_,
//                                             const Vector b_)
// {
//    A = A_;
//    if (!gradinv_only)
//    {
//       DenseMatrix At(A.Width(), A.Height());
//       A.Transpose(At);
//       Ainv.Factor(A);
//       Atinv.Factor(At);
//    }
//    b = b_;
// }
//
// // @brief Evaluate the gradient of the function at a point x
// // @param[in] x The point at which to evaluate the gradient
// // @param[out] g The vector to store the gradient
// void TransformedLegendreFunction::grad(const Vector &x,
//                                        Vector &g) const
// {
//    f->grad(transform_vector(x), aux_g);
//
//    g.SetSize(x.Size());
//    Atinv.Mult(aux_g, g);
// }
//
// // @brief Evaluate the inverse gradient of the function at a point x
// // @param[in] x The point at which to evaluate the inverse gradient
// // @param[out] invg The vector to store the inverse gradient
// void TransformedLegendreFunction::gradinv(const Vector &x,
//                                           Vector &invg) const
// {
//    Atx.SetSize(A.Width());
//    A.MultTranspose(x, Atx);
//
//    f->gradinv(Atx, aux_invg);
//
//    invg.SetSize(x.Size());
//    A.Mult(aux_invg, invg);
//    invg += b;
// }

real_t Gibbs::operator()(const Vector &x) const
{
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      sum += x[i] > tol ? x[i] * std::log(x[i]) : 0.0;
   }
   return sum;
}
void Gibbs::grad(const Vector &x, Vector &g) const
{
   g.SetSize(x.Size());
   for (int i=0; i<x.Size(); i++)
   {
      g[i] = x[i] > tol ? std::log(x[i]) + 1.0 : std::log(tol) + 1.0;
   }
}

void Gibbs::gradinv(const Vector &x, Vector &invg) const
{
   real_t maxval = x.Max();
   invg.SetSize(x.Size());
   real_t sum = 0.0;
   for (int i=0; i<x.Size(); i++)
   {
      invg[i] = std::exp(x[i] - maxval);
      sum += invg[i];
   }
   invg /= sum;
}

void Gibbs::hessinv(const Vector &x, DenseMatrix &H) const
{
   y.SetSize(x.Size());
   gradinv(x, y);
   H.Diag(y.GetData(), y.Size());
   AddMult_a_VVt(-1.0, y, H);
}

PointwiseGibbs::PointwiseGibbs(const Array<DenseMatrix*> &Vs_, real_t tol)
   : Gibbs(tol), Vs(Vs_)
{
   int max_siz = 0;
   for (auto &V : Vs) { max_siz = std::max(max_siz, V->Width()); }
   y_buffer.SetSize(max_siz);
   invgy_buffer.SetSize(max_siz);
   H_buffer.SetSize(max_siz*max_siz);
   VH_buffer.SetSize(max_siz*max_siz);
}
void PointwiseGibbs::gradinv(const Vector &x, int i, Vector &invg) const
{
   DenseMatrix &V = *Vs[i];
   const int dim = V.Width();
   y.MakeRef(y_buffer, dim);

   V.MultTranspose(x, y);
   invgy.MakeRef(invgy_buffer, dim);
   Gibbs::gradinv(y, invgy);
   invg.SetSize(x.Size());
   V.Mult(invgy, invg);
}
void PointwiseGibbs::hessinv(const Vector &x, int i, DenseMatrix &H) const
{
   DenseMatrix &V = *Vs[i];
   const int dim = V.Width();
   y.MakeRef(y_buffer, dim);

   V.MultTranspose(x, y);
   invhessy.UseExternalData(H_buffer.GetData(), dim, dim);
   Gibbs::hessinv(y, invhessy);
   Vinvhessy.UseExternalData(VH_buffer.GetData(), V.Height(), dim);
   // V * invhessy * V^T
   Mult(V, invhessy, Vinvhessy);
   H.SetSize(V.Height(), V.Height());
   MultABt(Vinvhessy, V, H);
}


} // namespace mfem
