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

template <typename Op>
class CGSolverDG: public Operator
{
public:
   typedef Tensor<2> Tensor2d;

protected:
   FiniteElementSpace& fes;
   const Tensor2d& D;
   Tensor2d shape1d;
   const double treshold;

public:
   CGSolverDG(FiniteElementSpace& fes, int order, const Op& op, const double tr = 1e-10)
   : Operator(fes.GetVSize()), fes(fes), D(op.getD()),
     shape1d(fes.GetNDofs1d(),fes.GetNQuads1d(order)), treshold(tr)
   {
      ComputeBasis1d(fes.GetFE(0), order, shape1d);
   }

   virtual void Mult(const Vector &U, Vector &V) const
   {
      switch(fes.GetFE(0)->GetDim())
      {
         case 1:
            Mult1d(U, V);
            break;
         case 2:
            Mult2d(U, V);
            break;
         case 3:
            Mult3d(U, V);
            break;
      }
   }

protected:
   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;  

};

template <typename Mass, typename Prec>
class PrecCGSolverDG: public Operator
{
public:
   typedef Tensor<2> Tensor2d;

protected:
   FiniteElementSpace& fes;
   const Tensor2d& D;
   Prec& prec;
   Tensor2d shape1d;
   const double treshold;

public:
   PrecCGSolverDG(FiniteElementSpace& fes, int order, Mass& mass, Prec& prec, const double tr = 1e-10)
   : Operator(fes.GetVSize()), fes(fes), D(mass.getD()), prec(prec),
     shape1d(fes.GetNDofs1d(),fes.GetNQuads1d(order)), treshold(tr)
   {
      ComputeBasis1d(fes.GetFE(0), order, shape1d);
   }

   virtual void Mult(const Vector &U, Vector &V) const
   {
      switch(fes.GetFE(0)->GetDim())
      {
         case 1:
            Mult1d(U, V);
            break;
         case 2:
            Mult2d(U, V);
            break;
         case 3:
            Mult3d(U, V);
            break;
      }
   }

protected:
   /**
   *  The domain Kernels for BtDB in 1d,2d and 3d.
   */
   void Mult1d(const Vector &V, Vector &U) const;
   void Mult2d(const Vector &V, Vector &U) const;
   void Mult3d(const Vector &V, Vector &U) const;  

};

// class CGSolverDG: public Operator
// {
// private:
//    FiniteElementSpace* fes;
//    Operator& op;

// public:
//    CGSolverDG(FiniteElementSpace* _fes, const Operator& _op, const double tr = 1e-10)
//    {

//    }


//    virtual void Mult(const Vector &U, Vector &V) const
//    {
//       const int dofs1d = shape1d.Height();
//       const int quads1d = shape1d.Width();
//       const int quads = quads1d;
//       const int dofs = dofs1d;

//       for (int e = 0; e < fes->GetNE(); e++)
//       {
//          const Tensor<1> V0mat(V.GetData() + e*dofs, dofs);
//          Tensor<1> b(dofs);
//          b = V0mat;
//          Tensor<1> x(U.GetData() + e*dofs, dofs);
//          Tensor<1> r(dofs);
//          r = b;
//          Tensor<1> p(dofs);
//          p = r;
//          double rsold = norm2sq(r);

//          for(int i=0; i<dofs; i++){
//             Tensor<1> Ap(dofs),tmp(quads1d);
//             // Ap = A * p
//             op.Mult(e,p,Ap);
//             const double alpha = rsold / dot(p,Ap);
//             // x = x + alpha * p
//             // r = r - alpha * Ap
//             for (int i = 0; i < dofs; ++i)
//             {
//                x(i) = x(i) + alpha * p(i);
//                r(i) = r(i) - alpha * Ap(i);
//             }
//             const double rsnew = norm2sq(r);
//             if (sqrt(rsnew)<treshold) break;
//             // p = r + (rsnew/rsold) * p
//             for (int i = 0; i < dofs; ++i)
//             {
//                p(i) = r(i) + (rsnew/rsold) * p(i);
//             }
//             rsold = rsnew;
//          }
//    }
//    }
// };

template <typename Mass>
class PACGSolver: public Operator
{
private:
   const int dofs;
   Mass& mass;
   const double treshold;

public:
   PACGSolver(const FiniteElementSpace* fes, Mass& mass, const double tr = 1e-10)
   : Operator(fes->GetVSize()), dofs(fes->GetNDofs()), mass(mass), treshold(tr)
   {

   }

   virtual void Mult(const Vector &U, Vector &V) const
   {
      int iter = 0;

      Vector b(dofs);
      b = U;
      Vector& x = V;
      x = 0.0;
      Vector r(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = -b(i);
      }
      // Precondition My=r
      // r = b;
      Vector p(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = - r(i);
      }
      // p = r;
      // rsold = y*r
      double rsold = r*r;
      for(iter=0; iter<dofs; iter++){
         Vector Ap(dofs);
         Ap = 0.0;
         // Ap = A * p
         mass.AddMult(p,Ap);
         const double alpha = rsold / (p*Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) + alpha * Ap(i);
         }
         // rsnew = y*r;
         const double rsnew = r*r;
         if (sqrt(rsnew)<treshold) break;
         // p = r + (rsnew/rsold) * p
         for (int i = 0; i < dofs; ++i)
         {
            //p = -y + Beta * p;
            p(i) = -r(i) + (rsnew/rsold) * p(i);
         }
         rsold = rsnew;
      }
      // cout << "residual=" << rsold << endl;
      // cout << "iter=" << iter << endl;
   }
};

template <typename Mass, typename Prec>
class PAPrecCGSolver: public Operator
{
private:
   const int dofs;
   Mass& mass;
   Prec& prec;
   const double treshold;

public:
   PAPrecCGSolver(const FiniteElementSpace* fes, Mass& mass, Prec& prec, const double tr = 1e-10)
   : Operator(fes->GetVSize()), dofs(fes->GetNDofs()), mass(mass), prec(prec), treshold(tr)
   {

   }

   virtual void Mult(const Vector &U, Vector &V) const
   {
      int iter = 0;

      Vector b(dofs);
      b = U;
      Vector& x = V;
      x = 0.0;
      Vector r(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = -b(i);
      }
      Vector y(dofs);
      prec.Mult(r,y);
      Vector p(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = - y(i);
      }
      double rsold = y*r;
      for(iter=0; iter<dofs; iter++){
         Vector Ap(dofs);
         Ap = 0.0;
         mass.AddMult(p,Ap);
         const double alpha = rsold / (p*Ap);
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) + alpha * Ap(i);
         }
         prec.Mult(r,y);
         const double rsnew = y*r;
         if (sqrt(rsnew)<treshold){
            // cout << "residual=" << rsnew << endl;
            // cout << "iter=" << iter << endl;
            break;  
         }
         const double beta = rsnew/rsold;
         for (int i = 0; i < dofs; ++i)
         {
            p(i) = -y(i) + beta * p(i);
         }
         rsold = rsnew;
      }
   }
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
   template <typename Op>
   DiagSolverDG(FiniteElementSpace fes, int order, Op& op, bool fast_eval = false)
   : nbelts( fes.GetNE() ), dofs( fes.GetFE(0)->GetDof() ), size(nbelts*dofs), D(dofs,nbelts),
     Operator(fes.GetVSize())
   {
      if(fast_eval)// FIXME: For some reason does not work in 2d...
      {
         Vector U(nbelts*dofs);
         for (int i = 0; i < size; ++i)
         {
            U(i) = 1.0;
         }
         //If the op is diagonal, then we obtain the diagonal by multiplying by a vector of 1.
         Vector V(D.getData(),size);
         op.AddMult(U,V);
      }else{
         GetDiag(fes,order,op,D);
      }
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

   template <typename Vector>
   void Mult(const int elt, const Vector &U, Vector &V) const
   {
      for (int k = 0; k < dofs; ++k)
      {
         V(k) =  U(k)/D(k,elt);
      }
   }

};

template<typename Op>
void CGSolverDG<Op>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;
   const int dofs = dofs1d;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<1> eD(D.getData() + e*quads, quads1d);
      Tensor<1> x(U.GetData() + e*dofs, dofs1d);
      x.zero();
      Tensor<1> r(dofs);
      r = b;
      Tensor<1> p(dofs);
      p = r;
      double rsold = norm2sq(r);

      for(int i=0; i<dofs; i++){
         Tensor<1> Ap(dofs),tmp(quads1d);
         // Ap = A * p
         contract(shape1d,p,tmp);
         cWiseMult(eD,tmp,tmp);
         contract(shape1d,tmp,Ap);
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

template<typename Op>
void CGSolverDG<Op>::Mult2d(const Vector &V, Vector &U) const
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
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<2> eD(D.getData() + e*quads, quads1d, quads1d);
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
         contract(shape1d,pT,tmp1);
         contract(shape1d,tmp1,tmp2);
         cWiseMult(eD,tmp2,tmp2);
         contractT(shape1d,tmp2,tmp3);
         contractT(shape1d,tmp3,ApT);
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

template<typename Op>
void CGSolverDG<Op>::Mult3d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d * quads1d * quads1d;
   const int dofs = dofs1d * dofs1d * dofs1d;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<3> eD(D.getData() + e*quads, quads1d, quads1d, quads1d);
      Tensor<1> x(U.GetData() + e*dofs, dofs);
      x.zero();
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
         contract(shape1d,pT,tmp1);
         contract(shape1d,tmp1,tmp2);
         contract(shape1d,tmp2,tmp3);
         cWiseMult(eD,tmp3,tmp3);
         contract(shape1d,tmp3,tmp4);
         contract(shape1d,tmp4,tmp5);
         contract(shape1d,tmp5,ApT);
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
         for (int i = 0; i < dofs; ++i)
         {
            p(i) = r(i) + (rsnew/rsold) * p(i);
         }
         rsold = rsnew;
      }
   }
}


template <typename Mass, typename Prec>
void PrecCGSolverDG<Mass,Prec>::Mult1d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d;
   const int dofs = dofs1d;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<1> eD(D.getData() + e*quads, quads1d);
      Tensor<1> x(U.GetData() + e*dofs, dofs1d);
      x.zero();
      Tensor<1> r(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = -b(i);
      }
      Tensor<1> y(dofs);
      prec.Mult(e,r,y);
      Tensor<1> p(dofs);
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = -y(i);
      }
      double rsold = dot(r,y);
      for(int iter=0; iter<dofs; iter++){
         Tensor<1> Ap(dofs),tmp(quads1d);
         // Ap = A * p
         contract(shape1d,p,tmp);
         cWiseMult(eD,tmp,tmp);
         contract(shape1d,tmp,Ap);
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) + alpha * Ap(i);
         }
         prec.Mult(e,r,y);
         const double rsnew = dot(r,y);
         if (sqrt(rsnew)<treshold){
            // cout << "residual=" << rsnew << endl;
            // cout << "iter=" << iter << endl;            
            break;
         }
         // p = r + (rsnew/rsold) * p
         const double beta = rsnew/rsold;
         for (int i = 0; i < dofs1d; ++i)
         {
            p(i) = -y(i) + beta * p(i);
         }
         rsold = rsnew;
      }
   }
}

template <typename Mass, typename Prec>
void PrecCGSolverDG<Mass,Prec>::Mult2d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d * quads1d;
   const int dofs = dofs1d * dofs1d;

   Tensor<1> r(dofs);
   Tensor<1> y(dofs);
   Tensor<1> p(dofs);
   Tensor<2> pT(p.getData(),dofs1d,dofs1d);
   Tensor<1> Ap(dofs);
   Tensor<2> ApT(Ap.getData(),dofs1d,dofs1d);
   Tensor<2> tmp1(dofs1d,quads1d), tmp2(quads1d,quads1d), tmp3(quads1d,dofs1d);
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<2> eD(D.getData() + e*quads, quads1d, quads1d);
      Tensor<1> x(U.GetData() + e*dofs, dofs);
      // Tensor<1> x(dofs);
      x.zero();
      // r = b - A x
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = -b(i);
      }
      // p = r;
      prec.Mult(e,r,y);
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = -y(i);
      }
      double rsold = dot(y,r);
      for(int iter=0; iter<dofs; iter++){
         // Ap = A * p
         contract(shape1d,pT,tmp1);
         contract(shape1d,tmp1,tmp2);
         cWiseMult(eD,tmp2,tmp2);
         contractT(shape1d,tmp2,tmp3);
         contractT(shape1d,tmp3,ApT);
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) + alpha * Ap(i);
         }
         prec.Mult(e,r,y);
         const double rsnew = dot(y,r);
         if (sqrt(rsnew)<treshold){
            // cout << "residual=" << rsnew << endl;
            // cout << "iter=" << iter << endl;
            break;  
         }
         // p = r + (rsnew/rsold) * p
         const double beta = (rsnew/rsold);
         for (int i = 0; i < dofs; ++i)
         {
            p(i) = -y(i) + beta * p(i);
         }
         rsold = rsnew;
      }
   }
}

template <typename Mass, typename Prec>
void PrecCGSolverDG<Mass,Prec>::Mult3d(const Vector &V, Vector &U) const
{
   const int dofs1d = shape1d.Height();
   const int quads1d = shape1d.Width();
   const int quads = quads1d * quads1d * quads1d;
   const int dofs = dofs1d * dofs1d * dofs1d;

   // Tensor<1> r(dofs);
   // Tensor<1> y(dofs);
   // Tensor<1> p(dofs);
   // Tensor<3> pT(p.getData(),dofs1d,dofs1d,dofs1d);
   // Tensor<1> Ap(dofs);
   // Tensor<3> ApT(Ap.getData(),dofs1d,dofs1d,dofs1d);
   // Tensor<3> tmp1(dofs1d,dofs1d,quads1d), tmp2(dofs1d,quads1d,quads1d), tmp3(quads1d,quads1d,quads1d),
               // tmp4(quads1d,quads1d,dofs1d), tmp5(quads1d,dofs1d,dofs1d);
   for (int e = 0; e < fes.GetNE(); e++)
   {
      const Tensor<1> b(V.GetData() + e*dofs, dofs);
      const Tensor<3> eD(D.getData() + e*quads, quads1d, quads1d, quads1d);
      Tensor<1> r(dofs);
      Tensor<1> y(dofs);
      Tensor<1> p(dofs);
      Tensor<1> x(U.GetData() + e*dofs, dofs);
      x.zero();
      for (int i = 0; i < dofs; ++i)
      {
         r(i) = -b(i);
      }
      // p = r;
      prec.Mult(e,r,y);
      for (int i = 0; i < dofs; ++i)
      {
         p(i) = -y(i);
      }
      double rsold = dot(y,r);
      for(int i=0; i<dofs; i++){
         Tensor<3> pT(p.getData(),dofs1d,dofs1d,dofs1d);
         Tensor<1> Ap(dofs);
         Tensor<3> ApT(Ap.getData(),dofs1d,dofs1d,dofs1d);
         Tensor<3> tmp1(dofs1d,dofs1d,quads1d), tmp2(dofs1d,quads1d,quads1d), tmp3(quads1d,quads1d,quads1d),
                     tmp4(quads1d,quads1d,dofs1d), tmp5(quads1d,dofs1d,dofs1d);
         // Ap = A * p
         contract(shape1d,pT,tmp1);
         contract(shape1d,tmp1,tmp2);
         contract(shape1d,tmp2,tmp3);
         cWiseMult(eD,tmp3,tmp3);
         contract(shape1d,tmp3,tmp4);
         contract(shape1d,tmp4,tmp5);
         contract(shape1d,tmp5,ApT);
         const double alpha = rsold / dot(p,Ap);
         // x = x + alpha * p
         // r = r - alpha * Ap
         for (int i = 0; i < dofs; ++i)
         {
            x(i) = x(i) + alpha * p(i);
            r(i) = r(i) + alpha * Ap(i);
         }
         prec.Mult(e,r,y);
         const double rsnew = dot(y,r);
         if (sqrt(rsnew)<treshold){
            // cout << "residual=" << rsnew << endl;
            // cout << "iter=" << iter << endl;
            break;  
         }
         // p = r + (rsnew/rsold) * p
         const double beta = (rsnew/rsold);
         for (int i = 0; i < dofs; ++i)
         {
            p(i) = -y(i) + beta * p(i);
         }
         rsold = rsnew;
      }
   }
}



}

#endif //MFEM_SOLVERKERNELS