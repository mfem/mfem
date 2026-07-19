#include "mfem.hpp"
#include "coefficient.hpp"

namespace mfem
{

StressGridFunctionCoefficient::StressGridFunctionCoefficient( real_t lambda,
   real_t mu, const GridFunction *gf) : MatrixCoefficient((gf) ? gf ->
   FESpace() -> GetMesh() -> SpaceDimension() : 0)
{
   this->lambda = new ConstantCoefficient(lambda);
   this->mu     = new ConstantCoefficient(mu);
   own_coefs = true;
   GridFunc = gf;
}

StressGridFunctionCoefficient::StressGridFunctionCoefficient( Coefficient &lambda,
   Coefficient &mu, const GridFunction *gf) : MatrixCoefficient((gf) ? gf ->
   FESpace() -> GetMesh() -> SpaceDimension() : 0)
{
   this->lambda = &lambda;
   this->mu     = &mu;
   own_coefs = false;
   GridFunc = gf;
}

void StressGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   GridFunc = gf;
   height = (gf) ? gf->FESpace()->GetMesh()->SpaceDimension() : 0;
   width = height;
}

void StressGridFunctionCoefficient::SetLameParameters(real_t lambda, real_t mu)
{
   if (own_coefs) { delete this->lambda; delete this->mu; }
   this->lambda = new ConstantCoefficient(lambda);
   this->mu     = new ConstantCoefficient(mu);
   own_coefs = true;
}

void StressGridFunctionCoefficient::SetLameParameters(Coefficient &lambda,
                                                      Coefficient &mu)
{
   if (own_coefs) { delete this->lambda; delete this->mu; }
   this->lambda = &lambda;
   this->mu     = &mu;
   own_coefs = false;
}

void StressGridFunctionCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   T.SetIntPoint(&ip);

   Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
   if (T.mesh->GetNE() == gf_mesh->GetNE())
   {
      GridFunc->GetVectorGradient(T, K);
   }
   else
   {
      MFEM_ABORT("StressGridFunctionCoefficient::Eval not implemented for "
                 "refined meshes.");
   }

   const real_t L = lambda->Eval(T, ip);
   const real_t M = mu->Eval(T, ip);

   real_t c = L * GridFunc->GetDivergence(T);

   for (int i = 0; i < height; i++)
   {
      for (int j = i; j < width; j++)
      {
         real_t k = c * ((i == j) ? 1.0 : 0.0) + M * (K(i, j) + K(j, i));
         K(j,i) = K(i,j) = k;
      }
   }
}

StressGridFunctionCoefficient::~StressGridFunctionCoefficient()
{
   if (own_coefs)
   {
      delete lambda;
      delete mu;
   }
}

FlatVectorCoefficient::FlatVectorCoefficient(MatrixCoefficient &A, real_t _alpha)
   : VectorCoefficient(A.GetHeight() * A.GetWidth())
{
   a = &A;
   alpha = _alpha;
}


void FlatVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   const int height = a->GetHeight();
   const int width = a->GetWidth();

   DenseMatrix mat;
   V.SetSize(height * width);
   mat.SetSize(height, width);

   T.SetIntPoint(&ip);
   a->Eval(mat, T, ip);

   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         V(i * width + j) = alpha * mat(i, j);
      }
   }
}

} // namespace mfem
