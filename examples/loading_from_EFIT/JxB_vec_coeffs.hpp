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
class JPolRVectorGridFunctionCoefficient : public VectorGridFunctionCoefficient
{
public:
   JPolRVectorGridFunctionCoefficient() : VectorGridFunctionCoefficient() {}

   JPolRVectorGridFunctionCoefficient(const GridFunction *gf) : VectorGridFunctionCoefficient(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);

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
      real_t r = x(0);
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
      real_t r = x(0);
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
      real_t r = x(0);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      return interp_val(0) * r * scale * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_tor$ and return $[[0, -rB_tor], [rB_tor, 0]]
class BTorRPerpMatrixGridFunctionCoefficient : public MatrixCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;
   BTorRPerpMatrixGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : MatrixCoefficient(2, 2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }
   using MatrixCoefficient::Eval;
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);

      M(0, 0) = 0;
      M(0, 1) = -interp_val(0) * r * (flip_sign ? -1 : 1);
      M(1, 0) = interp_val(0) * r * (flip_sign ? -1 : 1);
      M(1, 1) = 0;
   }
};

/// @brief Input $B_tor$ and return $B_tor Grad rB_tor$
class BTorGradRBTorVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_B_tor_coef;
   GridFunctionCoefficient B_tor_coef;

public:
   int counter = 0;

   BTorGradRBTorVectorGridFunctionCoefficient() = delete;

   BTorGradRBTorVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), grad_B_tor_coef(gf), B_tor_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // Grad rB_tor = r Grad B_tor + (1, 0) B_tor

      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);

      grad_B_tor_coef.Eval(V, T, ip);
      V *= r;
      // now, V = r * Grad B_tor
      real_t B_tor = B_tor_coef.Eval(T, ip);
      V(0) += B_tor;
      // now, V = Grad rB_tor

      V *= B_tor * (flip_sign ? -1 : 1);
   }
};

// @brief Input $B_pol$ and return $r Curl B_pol B_pol^perp$
class CurlBPolBPolPerpRGridFunctionCoefficient : public VectorCoefficient
{
private:
   bool flip_sign;
   const GridFunction *gf;
   VectorGridFunctionCoefficient B_pol_coef;

public:
   int counter = 0;
   CurlBPolBPolPerpRGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), gf(gf), B_pol_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector x1;
      T.Transform(ip, x1);
      real_t r = x1(0);
      Vector x2;
      gf->GetCurl(T, x2);
      B_pol_coef.Eval(V, T, ip);
      swap(V(0), V(1));
      V(0) = -V(0);
      V *= x2(0) * r * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_pol$ and $B_tor$ and return $B_pol dot Grad rB_tor$
class BPolGradRBTorGridFunctionCoefficient : public Coefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_B_tor_coef;
   VectorGridFunctionCoefficient B_pol_coef;
   GridFunctionCoefficient B_tor_coef;

public:
   int counter = 0;

   BPolGradRBTorGridFunctionCoefficient() = delete;

   BPolGradRBTorGridFunctionCoefficient(const GridFunction *B_pol, const GridFunction *B_tor, 
                                        bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), 
         grad_B_tor_coef(B_tor), B_pol_coef(B_pol), B_tor_coef(B_tor)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // Grad rB_tor = r Grad B_tor + (1, 0) B_tor

      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);

      Vector Grad_rB_tor;
      grad_B_tor_coef.Eval(Grad_rB_tor, T, ip);
      Grad_rB_tor *= r;
      // now, r * Grad B_tor
      real_t B_tor = B_tor_coef.Eval(T, ip);
      Grad_rB_tor(0) += B_tor;
      // now, Grad rB_tor

      Vector B_pol;
      B_pol_coef.Eval(B_pol, T, ip);
      // now, B_pol

      return (B_pol(0) * Grad_rB_tor(0) + B_pol(1) * Grad_rB_tor(1)) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $J_tor$ and return $[[0, -rJ_tor], [rJ_tor, 0]]
class JTorRPerpMatrixGridFunctionCoefficient : public MatrixCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;
   JTorRPerpMatrixGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : MatrixCoefficient(2, 2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }
   using MatrixCoefficient::Eval;
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);

      M(0, 0) = 0;
      M(0, 1) = -interp_val(0) * r * (flip_sign ? -1 : 1);
      M(1, 0) = interp_val(0) * r * (flip_sign ? -1 : 1);
      M(1, 1) = 0;
   }
};
