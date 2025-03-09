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

/// @brief Input $J_perp$ and return $J_perp*r$
class JPerpRVectorGridFunctionCoefficient : public VectorGridFunctionCoefficient
{
public:
   JPerpRVectorGridFunctionCoefficient() : VectorGridFunctionCoefficient() {}

   JPerpRVectorGridFunctionCoefficient(const GridFunction *gf) : VectorGridFunctionCoefficient(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x[0];

      VectorGridFunctionCoefficient::Eval(V, T, ip);
      V *= r;
   }
};

/// @brief Return $r$
class RGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;
   real_t scale;
public:
   int counter = 0;
   RGridFunctionCoefficient(real_t scale = 1.0, bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), scale(scale)
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
      return r * scale * (flip_sign ? -1 : 1);
   }
};

/// @brief Return $r^2$
class RSquareGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   RSquareGridFunctionCoefficient(bool flip_sign = false)
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
      return r * r * (flip_sign ? -1 : 1);
   }
};

/// @brief Compute r*(curl B_perp × B_perp_perp)
class RCurlBPerpBPerpPerpVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   VectorGridFunctionCoefficient B_perp;
   GridFunctionCoefficient R_Curl_B_perp;

public:
   RCurlBPerpBPerpPerpVectorGridFunctionCoefficient(GridFunction *R_Curl_B_perp, GridFunction *B_perp)
       : VectorCoefficient(2), B_perp(B_perp), R_Curl_B_perp(R_Curl_B_perp) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x;
      Vector B_perp_val;
      B_perp.Eval(B_perp_val, T, ip);
      real_t R_Curl_B_perp_val = R_Curl_B_perp.Eval(T, ip);
      V(0) = -R_Curl_B_perp_val * B_perp_val(1);
      V(1) = R_Curl_B_perp_val * B_perp_val(0);
   }
};

/// @brief Input $f$ and return $f$
class PRGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   // disable default constructor
   PRGridFunctionCoefficient() = delete;

   PRGridFunctionCoefficient(const GridFunction *gf)
       : Coefficient(), gf(gf), finder(gf)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      return interp_val[0];
   }
};