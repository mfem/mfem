// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// This file contains operator-based bilinear form integrators used
// with BilinearFormOperator.


#ifndef MFEM_SOLVERKERNELS
#define MFEM_SOLVERKERNELS

#include "dalg.hpp"
#include "dgpabilininteg.hpp"
#include "tensorialfunctions.hpp"
#include "partialassemblykernel.hpp"

namespace mfem
{

/**
*  A class that implement the BtDB partial assembly Kernel
*/
template <typename Equation>
class CGSolverDG<Equation,PAOp::BtDB>: private Equation
{
public:
   static const int dimD = 2;
   const double treshold;
   typedef Tensor<dimD,double> DTensor;
   typedef DenseMatrix Tensor2d;

protected:
   FiniteElementSpace *fes;
   Tensor2d shape1d;
   DTensor D;

public:
   CGSolverDG(FiniteElementSpace* _fes, int order, const typename Equation::Args& args, const double tr = 1e-10)
   : fes(_fes), D(), treshold(tr)
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d);
   }

   CGSolverDG(FiniteElementSpace* _fes, int order, const double tr = 1e-10)
   : fes(_fes), D(), treshold(tr)
   {
      ComputeBasis1d(fes->GetFE(0), order, shape1d);
   }

   void InitD(const int dim, const int quads, const int nb_elts){
      this->D.setSize(quads,nb_elts);
   }

   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const typename Equation::Args& args)
   {
      double res = 0.0;
      this->evalD(res, Tr, ip, args);
      this->D(k,e) = res;
   }

   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, const typename Equation::Args& args)
   {
      double res = 0.0;
      this->evalD(res, Tr, ip, J, args);
      this->D(k,e) = res;
   }

   template <typename... Args>
   void evalEq(const int dim, const int k, const int e, ElementTransformation * Tr,
               const IntegrationPoint & ip, const Tensor<2>& J, Args... args)
   {
      double res = 0.0;
      this->evalD(res, Tr, ip, J, args...);
      this->D(k,e) = res;
   }

protected:
   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U);
   void Mult2d(const Vector &V, Vector &U);
   void Mult3d(const Vector &V, Vector &U);  

};

/**
*  A class that implement the BtDB partial assembly Kernel
*/
class DiagSolverDG: public Operator
{
private:
   const int nbelts;
   const int dofs;
   const int size;
   Tensor<2> D;

public:
   template <typename Mass>
   DiagSolverDG(FiniteElementSpace* fes, Mass& op)
   : nbelts( fes->GetNE() ), dofs( fes->GetFE(0)->GetDof() ), size(nbelts*dofs), D(dofs,nbelts),
     Operator(fes->GetVSize())
   {
      Vector U(nbelts*dofs);
      for (int i = 0; i < size; ++i)
      {
         U(i) = 1.0;
      }
      //If the op is diagonal, then we obtain the diagonal by multiplying by a vector of 1.
      Vector V(D.getData(),size);
      op.AddMult(U,V);
   }

   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   virtual void Mult(const Vector &U, Vector &V) const
   {
      const Tensor<2> Ut(U.GetData(), dofs, nbelts);
      Tensor<2> Vt(V.GetData(), dofs, nbelts);
      for (int e = 0; e < nbelts; e++)
      {
         for (int k = 0; k < dofs; ++k)
         {
            Vt(k,e) =  Ut(k,e)/D(k,e);
         }
      }     
   }
};

template<typename Equation>
void CGSolverDG<Equation,PAOp::BtDB>::Mult1d(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;
   const int dofs = dofs1d;

   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<1> V0mat(V.GetData() + e*dofs, dofs1d);
      Tensor<1> b(dofs);
      b = V0mat;
      Tensor<1> x(U.GetData() + e*dofs, dofs1d);
      Tensor<1> r(dofs);
      r = b;
      Tensor<1> p(dofs);
      p = r;
      double rsold = norm2sq(r);

      for(int i=0; i<dofs; i++){
         Tensor<1> Ap(dofs),tmp(quads1d);
         // Ap = A * p
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            tmp(k1) = 0.0;
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               tmp(k1) += shape1d(i1,k1) * p(i1);
            }
         }
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            tmp(k1) = D(k1,e) * tmp(k1);
         }
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            Ap(j1) = 0.0;
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               Ap(j1) += shape1d(j1,k1) * tmp(k1);
            }
         }
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) - alpha * Ap(i);
         }
         const double rsnew = norm2sq(r);
         if (sqrt(rsnew)<treshold) break;
         // p = r + (rsnew/rsold) * p
         for (int i = 0; i < dofs1d; ++i)
         {
            p(i) = r(i) + (rsnew/rsold) * p(i);
         }
         rsold = rsnew;
      }
   }
}

template<typename Equation>
void CGSolverDG<Equation,PAOp::BtDB>::Mult2d(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d * quads1d;
   const int dofs = dofs1d * dofs1d;

   Tensor<1> r(dofs);
   Tensor<1> p(dofs);
   Tensor<2> pT(p.getData(),dofs1d,dofs1d);
   Tensor<1> Ap(dofs);
   Tensor<2> ApT(Ap.getData(),dofs1d,dofs1d);
   Tensor<2> tmp1(dofs1d,quads1d), tmp2(quads1d,quads1d), tmp3(quads1d,dofs1d);
   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      Tensor<1> x(U.GetData() + e*dofs, dofs);
      // Tensor<1> x(dofs);
      x.zero();
      // r = b - A x
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = b(i);
      }
      // p = r;
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = r(i);
      }
      double rsold = norm2sq(r);
      for(int i=0; i<dofs; i++){
         // Ap = A * p
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               tmp1(i2,k1) = 0.0;
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  tmp1(i2,k1) += shape1d(i1,k1) * pT(i1,i2);
               }
            }
         }
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               tmp2(k1,k2) = 0.0;
               for (int i2 = 0; i2 < dofs1d; ++i2)
               {
                  tmp2(k1,k2) += shape1d(i2,k2) * tmp1(i2,k1);
               }
            }
         }
         for (int k2 = 0, k = 0; k2 < quads1d; ++k2)
         {
            for (int k1 = 0; k1 < quads1d; ++k1, ++k)
            {
               tmp2(k1,k2) = D(k,e) * tmp2(k1,k2);
            }
         }
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               tmp3(k2,j1) = 0.0;
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  tmp3(k2,j1) += shape1d(j1,k1) * tmp2(k1,k2);
               }
            }
         }
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               ApT(j1,j2) = 0.0;
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  ApT(j1,j2) += shape1d(j2,k2) * tmp3(k2,j1);
               }
            }
         }
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) - alpha * Ap(i);
         }
         const double rsnew = norm2sq(r);
         // cout << "elem " << e << " iter " << i << endl;
         // cout << "residual = " << rsnew << endl;
         if (sqrt(rsnew)<treshold) break;
         // p = r + (rsnew/rsold) * p
         for (int i = 0; i < dofs; ++i)
         {
            p(i) = r(i) + (rsnew/rsold) * p(i);
         }
         rsold = rsnew;
      }
   }
}

template<typename Equation>
void CGSolverDG<Equation,PAOp::BtDB>::Mult3d(const Vector &V, Vector &U)
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d * quads1d * quads1d;
   const int dofs = dofs1d * dofs1d * dofs1d;

   for (int e = 0; e < fes->GetNE(); e++)
   {
      const Tensor<1> V0mat(V.GetData() + e*dofs, dofs);
      Tensor<1> b(dofs);
      b = V0mat;
      Tensor<1> x(U.GetData() + e*dofs, dofs);
      Tensor<1> r(dofs);
      r = b;
      Tensor<1> p(dofs);
      p = r;
      double rsold = norm2sq(r);

      for(int i=0; i<dofs; i++){
         Tensor<3> pT(p.getData(),dofs1d,dofs1d,dofs1d);
         Tensor<1> Ap(dofs);
         Tensor<3> ApT(Ap.getData(),dofs1d,dofs1d,dofs1d);
         Tensor<3> tmp1(dofs1d,dofs1d,quads1d), tmp2(dofs1d,quads1d,quads1d), tmp3(quads1d,quads1d,quads1d),
                     tmp4(quads1d,quads1d,dofs1d), tmp5(quads1d,dofs1d,dofs1d);
         // Ap = A * p
         for (int k1 = 0; k1 < quads1d; ++k1)
         {
            for (int i3 = 0; i3 < dofs1d; ++i3)
            {
               for (int i2 = 0; i2 < dofs1d; ++i2)
               {
                  tmp1(i2,i3,k1) = 0.0;
                  for (int i1 = 0; i1 < dofs1d; ++i1)
                  {
                     tmp1(i2,i3,k1) += shape1d(i1,k1) * pT(i1,i2,i3);
                  }
               }
            }
         }
         for (int k2 = 0; k2 < quads1d; ++k2)
         {
            for (int k1 = 0; k1 < quads1d; ++k1)
            {
               for (int i3 = 0; i3 < dofs1d; ++i3)
               {
                  tmp2(i3,k1,k2) = 0.0;
                  for (int i2 = 0; i2 < dofs1d; ++i2)
                  {
                     tmp2(i3,k1,k2) += shape1d(i2,k2) * tmp1(i2,i3,k1);
                  }
               }
            }
         }
         for (int k3 = 0; k3 < quads1d; ++k3)
         {
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               for (int k1 = 0; k1 < quads1d; ++k1)
               {
                  tmp3(k1,k2,k3) = 0.0;
                  for (int i3 = 0; i3 < dofs1d; ++i3)
                  {
                     tmp3(k1,k2,k3) += shape1d(i3,k3) * tmp2(i3,k1,k2);
                  }
               }
            }
         }
         for (int k3 = 0, k = 0; k3 < quads1d; ++k3)
         {
            for (int k2 = 0; k2 < quads1d; ++k2)
            {
               for (int k1 = 0; k1 < quads1d; ++k1, ++k)
               {
                  tmp3(k1,k2,k3) = D(k,e) * tmp3(k1,k2,k3);
               }
            }
         }
         for (int j1 = 0; j1 < dofs1d; ++j1)
         {
            for (int k3 = 0; k3 < quads1d; ++k3)
            {
               for (int k2 = 0; k2 < quads1d; ++k2)
               {
                  tmp4(k2,k3,j1) = 0.0;
                  for (int k1 = 0; k1 < quads1d; ++k1)
                  {
                     tmp4(k2,k3,j1) += shape1d(j1,k1) * tmp3(k1,k2,k3);
                  }
               }
            }
         }
         for (int j2 = 0; j2 < dofs1d; ++j2)
         {
            for (int j1 = 0; j1 < dofs1d; ++j1)
            {
               for (int k3 = 0; k3 < quads1d; ++k3)
               {
                  tmp5(k3,j1,j2) = 0.0;
                  for (int k2 = 0; k2 < quads1d; ++k2)
                  {
                     tmp5(k3,j1,j2) += shape1d(j2,k2) * tmp4(k2,k3,j1);
                  }
               }
            }
         }
         for (int j3 = 0; j3 < dofs1d; ++j3)
         {
            for (int j2 = 0; j2 < dofs1d; ++j2)
            {
               for (int j1 = 0; j1 < dofs1d; ++j1)
               {
                  ApT(j1,j2,j3) = 0.0;
                  for (int k3 = 0; k3 < quads1d; ++k3)
                  {
                     ApT(j1,j2,j3) += shape1d(j3,k3) * tmp5(k3,j1,j2);
                  }
               }
            }
         }
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) - alpha * Ap(i);
         }
         const double rsnew = norm2sq(r);
         if (sqrt(rsnew)<treshold) break;
         // p = r + (rsnew/rsold) * p
         for (int i = 0; i < dofs1d; ++i)
         {
            p(i) = r(i) + (rsnew/rsold) * p(i);
         }
         rsold = rsnew;
      }
   }
}

}

#endif //MFEM_SOLVERKERNELS