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

/// @brief Input $J_pol$ and return $J_pol*r$
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

/// @brief Compute r*(curl B_pol × B_pol_perp)
class RCurlBPerpBPerpPerpVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   VectorGridFunctionCoefficient B_pol;
   GridFunctionCoefficient R_Curl_B_pol;

public:
   RCurlBPerpBPerpPerpVectorGridFunctionCoefficient(GridFunction *R_Curl_B_pol, GridFunction *B_pol)
       : VectorCoefficient(2), B_pol(B_pol), R_Curl_B_pol(R_Curl_B_pol) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x;
      Vector B_pol_val;
      B_pol.Eval(B_pol_val, T, ip);
      real_t R_Curl_B_pol_val = R_Curl_B_pol.Eval(T, ip);
      V(0) = -R_Curl_B_pol_val * B_pol_val(1);
      V(1) = R_Curl_B_pol_val * B_pol_val(0);
   }
};

/// @brief Input $p$ and return $p*r$
class PRGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   FindPointsGSLIBOneByOne finder;
   bool flip_sign;
   real_t scale;

public:
   int counter = 0;

   // disable default constructor
   PRGridFunctionCoefficient() = delete;

   PRGridFunctionCoefficient(const GridFunction *gf, real_t scale = 1.0, bool flip_sign = false)
       : Coefficient(), gf(gf), finder(gf), flip_sign(flip_sign), scale(scale)
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
      return interp_val[0] * r * scale * (flip_sign ? -1 : 1);
   }
};