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
      real_t r = x[0];
      counter++;
      Vector interp_val(1);
      finder.InterpolateOneByOne(x, *gf, interp_val, 0);
      return interp_val[0] / (1e-10 + r) * (flip_sign ? -1 : 1);
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
         ifstream infile("input/psi_coefficients.txt");
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
         interp_val[0] = fFun(interp_val[0], z, r);
      return interp_val[0] * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi / r n^\perp$ if v is 2D and $\Psi / r$ if v is 1D
class PsiOverRGridFunctionVectorCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiOverRGridFunctionVectorCoefficient() = delete;

   PsiOverRGridFunctionVectorCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
       : VectorCoefficient(dim), gf(gf), flip_sign(flip_sign), finder(gf)
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

         V *= interp_val[0] / (1e-10 + r) * (flip_sign ? -1 : 1);
      }
      else
         V(0) = interp_val[0] / (1e-10 + r) * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi / r^2$
class PsiOverRSquareGridFunctionVectorCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiOverRSquareGridFunctionVectorCoefficient() = delete;

   PsiOverRSquareGridFunctionVectorCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
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
      V(1) = interp_val[0] / (1e-10 + r * r) * (flip_sign ? -1 : 1);
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
      return interp_val[0] * (flip_sign ? -1 : 1);
   }
};

/// @brief Input $\Psi$ and return $\Psi$
class PsiGridFunctionVectorCoefficient : public VectorCoefficient
{
private:
   const GridFunction *gf;
   const bool flip_sign;
   FindPointsGSLIBOneByOne finder;

public:
   int counter = 0;

   PsiGridFunctionVectorCoefficient() = delete;

   PsiGridFunctionVectorCoefficient(int dim, const GridFunction *gf, bool flip_sign = false)
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
      return r * (flip_sign ? -1 : 1);
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
      V(0) = -interp_val[1];
      V(1) = interp_val[0];
      V *= (flip_sign ? -1 : 1);
   }
};

/// @brief Return $[[0, 1/r], [-1/r, 0]]$
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
      real_t r = x[0];
      counter++;
      M(0, 0) = 0;
      M(0, 1) = 1.0 / (1e-10 + r) * (flip_sign ? -1 : 1);
      M(1, 0) = - 1.0 / (1e-10 + r) * (flip_sign ? -1 : 1);
      M(1, 1) = 0;
   }
};
