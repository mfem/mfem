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

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class JTorFromFGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   // disable default constructor
   JTorFromFGridFunctionCoefficient() = delete;

   JTorFromFGridFunctionCoefficient(const GridFunction *gf)
       : Coefficient(), gf(gf), finder(gf)
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
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      return interp_val[0] / (1e-14 + r);
   }
};

/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class JPerpBGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   JPerpBGridFunctionCoefficient() = delete;

   JPerpBGridFunctionCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
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
/// @brief Returns f(u(x)) where u is a scalar GridFunction and f:R → R
class JPerpBOverRGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   JPerpBOverRGridFunctionCoefficient() = delete;

   JPerpBOverRGridFunctionCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
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
