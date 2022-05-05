#ifndef INTEG_ALGOIM_HPP
#define INTEG_ALGOIM_HPP

#include <mfem.hpp>

#ifdef MFEM_USE_ALGOIM
#include <algoim_quad.hpp>


namespace mfem
{


//define templated element bases
namespace TmplPoly_1D
{

/// Templated version of CalcBinomTerms
template<typename float_type>
void CalcBinomTerms(const int p, const float_type x, const float_type y,
                    float_type* u)
{
   if (p == 0)
   {
      u[0] = float_type(1.);
   }
   else
   {
      int i;
      const int *b = Poly_1D::Binom(p);
      float_type z = x;
      for (i = 1; i < p; i++)
      {
         u[i] = b[i]*z;
         z *= x;
      }
      u[p] = z;
      z = y;
      for (i--; i > 0; i--)
      {
         u[i] *= z;
         z *= y;
      }
      u[0] = z;
   }
}

/// Templated version of CalcBinomTerms
template<typename float_type>
void CalcBinomTerms(const int p, const float_type x, const float_type y,
                    float_type* u, float_type* d)
{
   if (p == 0)
   {
      u[0] = float_type(1.);
      d[0] = float_type(0.);
   }
   else
   {
      int i;
      const int *b = Poly_1D::Binom(p);
      const float_type xpy = x + y, ptx = p*x;
      float_type z = float_type(1.);

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
         u[i] = b[i]*z;
      }
      d[p] = p*z;
      u[p] = z*x;
      z = float_type(1.);
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
         u[i] *= z;
      }
      d[0] = -p*z;
      u[0] = z*y;
   }

}

/// Templated version of CalcDBinomTerms
template <typename float_type>
void CalcDBinomTerms(const int p, const float_type x,
                     const float_type y, float_type* d)
{
   if (p == 0)
   {
      d[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Poly_1D::Binom(p);
      const float_type xpy = x + y, ptx = p*x;
      float_type z = float_type(1.);

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
      }
      d[p] = p*z;
      z = float_type(1.);
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
      }
      d[0] = -p*z;
   }
}

/// Templated evaluation of Bernstein basis
template <typename float_type>
void CalcBernstein(const int p, const float_type x, float_type *u)
{
   CalcBinomTerms(p, x, 1. - x, u);
}


/// Templated evaluation of Bernstein basis
template <typename float_type>
void CalcBernstein(const int p, const float_type x,
                   float_type *u, float_type *d)
{
   CalcBinomTerms(p, x, 1. - x, u, d);
}


}

/// Construct volumetric and surface integration rules for a given element
/// using the Algoim library. The volume is define as the possitive part of
/// a level-set function(LSF) (lsfun argument in the constructor). The surface
/// is defined as the zero level-set of the LSF.
class AlgoimIntegrationRule
{
public:

   /// Construct Algoim object using a finite element, its transformation
   /// and level-set function defined over the element using Lagrangian
   /// bases. The argument o provides the number of the integration points
   /// of the 1D Gaussian integration used for deriving the vol/surface
   /// integration rules.
   AlgoimIntegrationRule(int o, const FiniteElement &el,
                         ElementTransformation &trans, const Vector &lsfun)
   {
      int_order=o;
      vir=nullptr;
      sir=nullptr;

      if (el.GetGeomType()==Geometry::Type::SQUARE)
      {
         pe=new H1Pos_QuadrilateralElement(el.GetOrder());
      }
      else if (el.GetGeomType()==Geometry::Type::CUBE)
      {
         pe=new H1Pos_HexahedronElement(el.GetOrder());
      }
      else
      {
         MFEM_ABORT("AlgoimIntegrationRule: The element type is not supported by Algoim!\n");
      }

      //change the basis of the level-set function
      //from Lagrangian to Bernstein (possitive)
      lsvec.SetSize(pe->GetDof());
      DenseMatrix T; T.SetSize(pe->GetDof());
      pe->Project(el,trans,T);
      T.Mult(lsfun,lsvec);
   }


   /// Destructor of the Algoim object
   ~AlgoimIntegrationRule()
   {
      delete pe;
      delete vir;
      delete sir;
   }

   /// Returns volumetric integration rule based on the provided
   /// level-set function.
   const IntegrationRule* GetVolumeIntegrationRule();

   /// Returns surface integration rule based on the provided
   /// level-set function.
   const IntegrationRule* GetSurfaceIntegrationRule();



private:

   /// 3D level-set function object required by Algoim.
   struct LevelSet3D
   {
      /// Constructor for 3D level-set function object required by Algoim.
      LevelSet3D(PositiveTensorFiniteElement* el_, Vector& lsfun_):el(el_),
         lsfun(lsfun_)
      {

      }

      /// Returns the value of the LSF for point x.
      template<typename T>
      T operator() (const blitz::TinyVector<T,3>& x) const
      {
         int el_order=el->GetOrder();
         T u1[el_order+1];
         T u2[el_order+1];
         T u3[el_order+1];
         TmplPoly_1D::CalcBernstein(el_order, x[0], u1);
         TmplPoly_1D::CalcBernstein(el_order, x[1], u2);
         TmplPoly_1D::CalcBernstein(el_order, x[2], u3);

         const Array<int>& dof_map=el->GetDofMap();

         T res=T(0.0);
         for (int oo = 0, kk = 0; kk <= el_order; kk++)
            for (int jj = 0; jj <= el_order; jj++)
               for (int ii = 0; ii <= el_order; ii++)
               {
                  res=res-u1[ii]*u2[jj]*u3[kk]*lsfun(dof_map[oo++]);
               }
         return res;
      }

      /// Returns the gradients of the LSF for point x.
      template<typename T>
      blitz::TinyVector<T,3> grad(const blitz::TinyVector<T,3>& x) const
      {
         int el_order=el->GetOrder();
         T u1[el_order+1];
         T u2[el_order+1];
         T u3[el_order+1];
         T d1[el_order+1];
         T d2[el_order+1];
         T d3[el_order+1];

         TmplPoly_1D::CalcBernstein(el_order,x[0], u1, d1);
         TmplPoly_1D::CalcBernstein(el_order,x[1], u2, d2);
         TmplPoly_1D::CalcBernstein(el_order,x[2], u3, d3);

         blitz::TinyVector<T,3> res(T(0.0),T(0.0),T(0.0));

         const Array<int>& dof_map=el->GetDofMap();

         for (int oo = 0, kk = 0; kk <= el_order; kk++)
            for (int jj = 0; jj <= el_order; jj++)
               for (int ii = 0; ii <= el_order; ii++)
               {
                  res[0]=res[0]-d1[ii]*u2[jj]*u3[kk]*lsfun(dof_map[oo]);
                  res[1]=res[1]-u1[ii]*d2[jj]*u3[kk]*lsfun(dof_map[oo]);
                  res[2]=res[2]-u1[ii]*u2[jj]*d3[kk]*lsfun(dof_map[oo]);
                  oo++;
               }

         return res;
      }

   private:
      PositiveTensorFiniteElement* el;
      Vector& lsfun;
   };

   /// 2D level-set function object required by Algoim.
   struct LevelSet2D
   {
      /// Constructor for 2D level-set function object required by Algoim.
      LevelSet2D(PositiveTensorFiniteElement* el_, Vector& lsfun_):el(el_),
         lsfun(lsfun_)
      {

      }

      /// Returns the value of the LSF for point x.
      template<typename T>
      T operator() (const blitz::TinyVector<T,2>& x) const
      {
         int el_order=el->GetOrder();
         T u1[el_order+1];
         T u2[el_order+1];
         TmplPoly_1D::CalcBernstein(el_order, x[0], u1);
         TmplPoly_1D::CalcBernstein(el_order, x[1], u2);

         const Array<int>& dof_map=el->GetDofMap();

         T res=T(0.0);

         for (int oo = 0, jj = 0; jj <= el_order; jj++)
            for (int ii = 0; ii <= el_order; ii++)
            {
               res=res-u1[ii]*u2[jj]*lsfun(dof_map[oo++]);
            }
         return res;
      }

      /// Returns the gradients of the LSF for point x.
      template<typename T>
      blitz::TinyVector<T,2> grad(const blitz::TinyVector<T,2>& x) const
      {
         int el_order=el->GetOrder();
         T u1[el_order+1];
         T u2[el_order+1];
         T d1[el_order+1];
         T d2[el_order+1];

         TmplPoly_1D::CalcBernstein(el_order,x[0], u1, d1);
         TmplPoly_1D::CalcBernstein(el_order,x[1], u2, d2);

         blitz::TinyVector<T,2> res(T(0.0),T(0.0));

         const Array<int>& dof_map=el->GetDofMap();

         for (int oo = 0, jj = 0; jj <= el_order; jj++)
            for (int ii = 0; ii <= el_order; ii++)
            {
               res[0]=res[0]-(d1[ii]*u2[jj])*lsfun(dof_map[oo]);
               res[1]=res[1]-(u1[ii]*d2[jj])*lsfun(dof_map[oo]);
               oo++;
            }

         return res;
      }


   private:
      PositiveTensorFiniteElement* el;
      Vector& lsfun;
   };


   IntegrationRule* sir; //surface integration rule
   IntegrationRule* vir; //volumetric integration rule
   PositiveTensorFiniteElement *pe;
   Vector lsvec; //level-set in Bernstein bases
   int int_order; //integration order

};



}
#endif

#endif
