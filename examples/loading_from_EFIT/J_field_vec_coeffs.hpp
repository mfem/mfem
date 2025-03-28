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
class GradPsiOverRRComponentGridFunctionCoefficient : public Coefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_psi_coef;

public:
   GradPsiOverRRComponentGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), grad_psi_coef(gf)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];
      Vector grad_psi;
      grad_psi_coef.Eval(grad_psi, T, ip);
      return grad_psi[0] / (1e-14 + r) * (flip_sign ? -1 : 1);
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

   BTorVectorGridFunctionCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(dim), gf(gf), flip_sign(flip_sign), finder(gf)
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

   BTorOverRVectorGridFunctionCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(dim), gf(gf), flip_sign(flip_sign), finder(gf)
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

   BTorRVectorGridFunctionCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(dim), gf(gf), flip_sign(flip_sign), finder(gf)
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