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

/// @brief Input $f$ and return $f r$
class FRGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   const bool from_psi;
   real_t alpha, f_x, psi_x, z_x;
   real_t z_min, z_max, r_min, r_max;
   FindPointsGSLIBOneByOne finder;
   real_t fFun(const real_t psi, const real_t z, const real_t r)
   {
      if (z < z_min || z > z_max || r < r_min || r > r_max)
         return f_x;
      return min(f_x + alpha * (psi - psi_x), f_x);
   };

public:
   int counter = 0;

   // disable default constructor
   FRGridFunctionCoefficient() = delete;

   FRGridFunctionCoefficient(const GridFunction *gf, bool from_psi = false, bool flip_sign = false)
       : Coefficient(), gf(gf), flip_sign(flip_sign), from_psi(from_psi), finder(gf)
   {
      if (from_psi)
      {
         ifstream infile("input/f_from_psi_coefficients.txt");
         if (!infile)
         {
            cerr << "Error: could not open input file" << endl;
            exit(1);
         }
         infile >> alpha >> f_x >> psi_x >> z_x;
         infile >> r_min >> r_max >> z_min >> z_max;
         cout << "r_min: " << r_min << ", r_max: " << r_max << ", z_min: " << z_min << ", z_max: " << z_max << endl;
      }
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0), z = x(1);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      if (from_psi)
         interp_val(0) = fFun(interp_val(0), z, r);
      return interp_val(0) * r * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $f$ and return $f/r$
class FOverRGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   // disable default constructor
   FOverRGridFunctionCoefficient() = delete;

   FOverRGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), gf(gf), flip_sign(flip_sign), finder(gf)
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
      return interp_val(0) / (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $f$ and return $f$
class FGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   const bool from_psi;
   real_t alpha, f_x, psi_x, z_x;
   real_t z_min, z_max, r_min, r_max;
   FindPointsGSLIBOneByOne finder;
   real_t fFun(const real_t psi, const real_t z, const real_t r)
   {
      if (z < z_min || z > z_max || r < r_min || r > r_max)
         return f_x;
      return min(f_x + alpha * (psi - psi_x), f_x);
   };

public:
   int counter = 0;

   // disable default constructor
   FGridFunctionCoefficient() = delete;

   FGridFunctionCoefficient(const GridFunction *gf, bool from_psi = false, bool flip_sign = false)
       : Coefficient(), gf(gf), flip_sign(flip_sign), from_psi(from_psi), finder(gf)
   {
      if (from_psi)
      {
         ifstream infile("input/f_from_psi_coefficients.txt");
         if (!infile)
         {
            cerr << "Error: could not open input file" << endl;
            exit(1);
         }
         infile >> alpha >> f_x >> psi_x >> z_x;
         infile >> r_min >> r_max >> z_min >> z_max;
         cout << "r_min: " << r_min << ", r_max: " << r_max << ", z_min: " << z_min << ", z_max: " << z_max << endl;
      }
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0), z = x(1);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      if (from_psi)
         interp_val(0) = fFun(interp_val(0), z, r);
      return interp_val(0) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi r$
class PsiRGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   // disable default constructor
   PsiRGridFunctionCoefficient() = delete;

   PsiRGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), gf(gf), flip_sign(flip_sign), finder(gf)
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
      return interp_val(0) * r * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi / r n^\perp$ if v is 2D and $\Psi / r$ if v is 1D
class PsiOverRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiOverRVectorGridFunctionCoefficient() = delete;

   PsiOverRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
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

         V *= interp_val(0) / (1e-10 + r) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val(0) / (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi / r^2$
class PsiOverRSquareVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiOverRSquareVectorGridFunctionCoefficient() = delete;

   PsiOverRSquareVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), gf(gf), flip_sign(flip_sign), finder(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);

      V(0) = 0;
      V(1) = interp_val(0) / (1e-10 + r * r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi$
class PsiGridFunctionCoefficient : public Coefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiGridFunctionCoefficient() = delete;

   PsiGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), gf(gf), flip_sign(flip_sign), finder(gf)
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
      return interp_val(0) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi$
class PsiVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiVectorGridFunctionCoefficient() = delete;

   PsiVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
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

         V *= interp_val(0) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val(0) * (flip_sign ? -1 : 1);
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
      real_t r = x(0);
      grad_psi_coef.Eval(V, T, ip);
      V /= (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $curl \Psi$
class CurlPsiVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_psi_coef;

public:
   int counter = 0;

   CurlPsiVectorGridFunctionCoefficient() = delete;

   CurlPsiVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), grad_psi_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      grad_psi_coef.Eval(V, T, ip);
      swap(V(0), V(1));
      V(0) = -V(0);
      V *= (flip_sign ? -1 : 1);
   }
};



/// @brief Input $Grad_Psi$ and return $Grad_Psi^\perp$
class GradPsiPerpVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   GradPsiPerpVectorGridFunctionCoefficient() = delete;

   GradPsiPerpVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
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
      Vector interp_val(2);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      V(0) = -interp_val(1);
      V(1) = interp_val(0);
      V *= (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_pol$ and return $rB_pol$
class BPolRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   VectorGridFunctionCoefficient B_pol_coef;

public:
   BPolRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), B_pol_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
      B_pol_coef.Eval(V, T, ip);
      V *= r * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_tor$ and return $Div rB_pol$
class DivRBPolGridFunctionCoefficient : public Coefficient
{
private:
   const bool flip_sign;
   DivergenceGridFunctionCoefficient div_B_pol_coef;
   VectorGridFunctionCoefficient B_pol_coef;

public:
   DivRBPolGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), div_B_pol_coef(gf), B_pol_coef(gf)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x1;
      T.Transform(ip, x1);
      real_t r = x1(0);

      real_t div_B_pol = div_B_pol_coef.Eval(T, ip);

      Vector x2;
      B_pol_coef.Eval(x2, T, ip);

      return (r * div_B_pol + x2(0)) * (flip_sign ? -1 : 1);
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

         V *= interp_val(0) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val(0) * (flip_sign ? -1 : 1);
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
      real_t r = x(0);
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);

      V(0) = 0;
      V(1) = interp_val(0) / (1e-10 + r) * (flip_sign ? -1 : 1);
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
      real_t r = x(0);
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

         V *= interp_val(0) * (1e-10 + r) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val(0) * (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_pol$ and return $rB_pol^\perp$
class BPolPerpRVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   VectorGridFunctionCoefficient B_pol_coef;

public:
   BPolPerpRVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), B_pol_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);
      B_pol_coef.Eval(V, T, ip);
      swap(V(0), V(1));
      V(0) = -V(0);
      V *= (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

// @brief Input $B_pol$ and return $r Curl B_pol$
class CurlBPolRGridFunctionCoefficient : public Coefficient
{
private:
   bool flip_sign;
   const GridFunction *gf;

public:
   int counter = 0;
   CurlBPolRGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : Coefficient(), flip_sign(flip_sign), gf(gf)
   {
   }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      Vector x1;
      T.Transform(ip, x1);
      real_t r = x1(0);
      Vector x2;
      gf->GetCurl(T, x2);
      return x2(0) * r * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $B_tor$ and return $Curl rB_tor$
class CurlRBTorVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   const bool flip_sign;
   GradientGridFunctionCoefficient grad_B_tor_coef;
   GridFunctionCoefficient B_tor_coef;

public:
   int counter = 0;

   CurlRBTorVectorGridFunctionCoefficient() = delete;

   CurlRBTorVectorGridFunctionCoefficient(const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(2), flip_sign(flip_sign), grad_B_tor_coef(gf), B_tor_coef(gf)
   {
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      // Curl rB_tor = r Curl B_tor + (0, 1) B_tor

      // get r, z coordinates
      Vector x;
      T.Transform(ip, x);
      real_t r = x(0);

      grad_B_tor_coef.Eval(V, T, ip);
      swap(V(0), V(1));
      V(0) = -V(0);
      V *= r;
      // now, V = r * Curl B_tor

      V(1) += B_tor_coef.Eval(T, ip);
      V *= (flip_sign ? -1 : 1);
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


/// @brief Return $[[0, -1/r], [1/r, 0]]$
class OneOverRPerpMatrixGridFunctionCoefficient : public MatrixCoefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   OneOverRPerpMatrixGridFunctionCoefficient(bool flip_sign = false)
       : MatrixCoefficient(2, 2), flip_sign(flip_sign)
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
      M(0, 0) = 0;
      M(0, 1) = -1.0 / (1e-10 + r) * (flip_sign ? -1 : 1);
      M(1, 0) = 1.0 / (1e-10 + r) * (flip_sign ? -1 : 1);
      M(1, 1) = 0;
   }
};

/// @brief Return $[[0, -r], [r, 0]]$
class RPerpMatrixGridFunctionCoefficient : public MatrixCoefficient
{
private:
   bool flip_sign;

public:
   int counter = 0;
   RPerpMatrixGridFunctionCoefficient(bool flip_sign = false)
       : MatrixCoefficient(2, 2), flip_sign(flip_sign)
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
      M(0, 0) = 0;
      M(0, 1) = -r * (flip_sign ? -1 : 1);
      M(1, 0) = r * (flip_sign ? -1 : 1);
      M(1, 1) = 0;
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
      real_t r = x(0);
      counter++;
      return 1 / (1e-10 + r) * (flip_sign ? -1 : 1);
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
      real_t r = x(0);
      counter++;
      V(0) = 1 / (1e-10 + r) * (flip_sign ? -1 : 1);
      V(1) = 0;
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
      real_t r = x(0), z = x(1);
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