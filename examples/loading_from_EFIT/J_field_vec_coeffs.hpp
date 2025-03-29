#include "mfem.hpp"
#include <iostream>
#include <functional>

using namespace std;
using namespace mfem;

class FindPointsGSLIBOneByOne : public FindPointsGSLIB
{
public:
   FindPointsGSLIBOneByOne(const GridFunction *gf) : FindPointsGSLIB()
   {
      gf->FESpace()->GetMesh()->EnsureNodes();
      Setup(*gf->FESpace()->GetMesh());
   }

   void InterpolateOneByOne(const Vector &point_pos,
                            const GridFunction &field_in,
                            Vector &field_out,
                            int point_pos_ordering = Ordering::byNODES)
   {
      FindPoints(point_pos, point_pos_ordering);
      // gsl_mfem_elem (element number) and gsl_mfem_ref (this is the integration point location)
      int element_number = gsl_mfem_elem[0];
      IntegrationPoint ip;
      ip.Set2(gsl_mfem_ref.GetData());
      field_in.GetVectorValue(element_number, ip, field_out);
   }

   ~FindPointsGSLIBOneByOne()
   {
      FreeData();
   }
};
/// @brief Input $\Psi$ and return $1/r \nabla \Psi_r$
class GradPsiOverRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_psi_coef;

public:
   GradPsiOverRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), grad_psi_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      grad_psi_coef.Eval(V, T, ip);
      V /= (1e-14 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_tor$ and return $B_tor n^\perp$ if v is 2D and $B_tor$ if v is 1D
class BTorVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   BTorVectorGridFunctionCoefficient() = delete;

   BTorVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      if (V.Size() == 2)
      {
         Vector normal(2);
         // get normal vector
         CalcOrtho(T.Jacobian(), normal);
         // normalize
         normal /= normal.Norml2();

         V(0) = -normal(1);
         V(1) = normal(0);

         V *= interp_val[0] * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val[0] * (flip_sign ? -1 : 1);
   }
};
/// @brief Input $B_tor$ and return $B_tor/r$
class BTorOverRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   BTorOverRVectorGridFunctionCoefficient() = delete;

   BTorOverRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);

      V(0) = 0;
      V(1) = interp_val[0] / (1e-14 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_tor$ and return $B_tor r n^\perp$ if v is 2D and $B_tor r$ if v is 1D
class BTorRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   BTorRVectorGridFunctionCoefficient() = delete;

   BTorRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      if (V.Size() == 2)
      {
         Vector normal(2);
         // get normal vector
         CalcOrtho(T.Jacobian(), normal);
         // normalize
         normal /= normal.Norml2();

         V(0) = -normal(1);
         V(1) = normal(0);

         V *= interp_val[0] * (1e-14 + r) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val[0] * (1e-14 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Return $r$
class RGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   RGridFunctionCoefficient(bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      return (1e-14 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Return $1/r$
class OneOverRGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   OneOverRGridFunctionCoefficient(bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      return 1 / (1e-14 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Return $(1/r, 0)$
class OneOverRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   OneOverRVectorGridFunctionCoefficient(bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign)
   {
   }
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      counter++;
      V(0) = 1 / (1e-14 + r) * (flip_sign? -1 : 1);
      V(1) = 0;
   }
};

enum WeightedHarmonicType
{
   POLYNOMIAL,
   EXPONENTIAL,
   TRIGONOMETRIC
};

/// @brief Return weighted harmonic function
class WeightedHarmonicGridFunctionCoefficient : public Coefficient
{
private:
   WeightedHarmonicType type;

public:
   WeightedHarmonicGridFunctionCoefficient(WeightedHarmonicType type)
       : Coefficient(), type(type)
   {
   }
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0], z = x[1];
      switch (type)
      {
      case POLYNOMIAL:
         // Polynomial solution: f(r, z) = A r²z + B r² + C z + D
         {
            real_t A = 1.0; // Default coefficient for r²z
            real_t B = 1.0; // Default coefficient for r²
            real_t C = 1.0; // Default coefficient for z
            real_t D = 1.0; // Default constant term
            return A * r * r * z + B * r * r + C * z + D;
         }
      case EXPONENTIAL:
         // Exponential solution: f(r, z) = √r * I₁(√(2k) r) * exp(k z)
         // (Using an approximation for I₁)
         {
            real_t k = 1.0;
            real_t arg = sqrt(2 * k) * r;
            real_t I1 = (arg / 2) + (pow(arg, 3) / 16); // Approximate I₁
            return sqrt(r) * I1 * exp(k * z);
         }

      case TRIGONOMETRIC:
         // Trigonometric solution: f(r, z) = √r * J₁(√(2k) r) * cos(k z)
         // (Using an approximation for J₁)
         {
            real_t k = 1.0;
            real_t arg = sqrt(2 * k) * r;
            real_t J1 = (arg / 2) - (pow(arg, 3) / 16); // Approximate J₁
            return sqrt(r) * J1 * cos(k * z);
         }
      }
   }
};