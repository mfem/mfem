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


// This file contains a prototype version for Discontinuous Galerkin Partial assembly

#ifndef MFEM_DGPABILININTEG
#define MFEM_DGPABILININTEG

// #include "bilininteg.hpp"
#include "tensor.hpp"

// #include "fem.hpp"
// #include <cmath>
// #include <algorithm>
// #include "../linalg/vector.hpp"

namespace mfem
{

namespace pa
{

/**
*  The different operators available for the Kernels
*/
enum PAOp { BtDB, BtDG, GtDB, GtDG };


class DiffusionEquation
{
public:
   static const PAOp OpName = GtDG;

   struct Args {
      Args(Coefficient& q) : q(q) {}
      Coefficient& q;
   };

   void evalD(Tensor<2>& res, ElementTransformation *Tr, const IntegrationPoint& ip,
                  const Tensor<2>& Jac, const Args& args)
   {
      const int dim = res.size(0);
      Tensor<2> Adj(dim,dim);
      adjugate(Jac,Adj);
      double val = 0.0;
      double qval = 1.0;
      double detJ = det(Jac);
      qval = 1.0;//args.q.Eval(*Tr, ip);//FIXME
      for (int i = 0; i < dim; ++i)
      {
         for (int j = 0; j < dim; ++j)
         {
            val = 0.0;
            for (int k = 0; k < dim; ++k)
            {
               val += Adj(i,k)*Adj(j,k); //Adj*Adj^T
            }
            res(i,j) = ip.weight * qval / detJ * val;
         }
      }
   }
};

/**
*	A class that describes the Convection Equation using DG for Partial Assembly.
*/
class DGConvectionEquation
{
public:
   /**
   *  Defines the Kernel to apply to the Domain
   */
   static const PAOp OpName = BtDG;

   /**
   *  Defines the variables needed to build D for the Domain kernel
   */
   struct Args {
      Args(VectorCoefficient& _q, double _a, double _b = 0.0) : q(_q), a(_a), b(_b) {}
      VectorCoefficient& q;
      double a;
      double b;
   };

   /**
   *  Returns the values of the D tensor at a given integration Point.
   */
   void evalD(Tensor<1>& res, ElementTransformation *Tr, const IntegrationPoint& ip,
                  const Tensor<2>& Jac, const Args& args)
   {
      const int dim = res.size(0);
      mfem::Vector qvec(dim);
      args.q.Eval(qvec, *Tr, ip);
      Tensor<2> Adj(dim,dim);
      adjugate(Jac,Adj);
      for (int i = 0; i < dim; ++i)
      {
         double val = 0.0;
         for (int j = 0; j < dim; ++j)
         {
            val += Adj(i,j) * qvec(j);
         }
         res(i) = ip.weight * args.a * val;
      }
   }

   /**
   *  Defines the Kernel to apply to the Faces
   */
   static const PAOp FaceOpName = BtDB;

   /**
   *  Returns the values of the Dint and Dext tensors at a given integration Point for
   *  each element over a face.
   */
   void evalFaceD(double& res11, double& res21, double& res22, double& res12,
      const FaceElementTransformations* face_tr, const mfem::Vector& normal,
      const IntegrationPoint& ip1, const IntegrationPoint& ip2,
      const Tensor<2>& Jac1, const Tensor<2>& Jac2,
      const Args& args)
   {
      const int dim = normal.Size();
      mfem::Vector qvec(dim);
      // FIXME: qvec might be discontinuous if not constant with a periodic mesh
      // We should then use the evaluation on Elem2 and eip2
      args.q.Eval( qvec, *(face_tr->Elem1), ip1 );
      const double res = qvec * normal;
      const double a = -args.a, b = args.b;
      res11 = ip1.weight * (   a/2 * res + b * abs(res) );
      res21 = ip1.weight * (   a/2 * res - b * abs(res) );
      res22 = ip1.weight * ( - a/2 * res + b * abs(res) );
      res12 = ip1.weight * ( - a/2 * res - b * abs(res) );
   }  
};

class FCTEquation
{
public:
   struct Args {
      VectorCoefficient& q;
      mfem::Vector& d_e;
      double a;
      double b;
   };

   static const PAOp FaceOpName = BtDB;

   void evalFaceD(double& res11, double& res21, double& res22, double& res12,
      const FaceElementTransformations* face_tr, const mfem::Vector& normal,
      const IntegrationPoint& ip1, const IntegrationPoint& ip2,
      const Tensor<2>& Jac1, const Tensor<2>& Jac2,
      const Args& args)
   {
      const int dim = normal.Size();
      mfem::Vector qvec(dim);
      // FIXME: qvec might be discontinuous if not constant with a periodic mesh
      // We should then use the evaluation on Elem2 and eip2
      args.q.Eval( qvec, *(face_tr->Elem1), ip1 );
      const double res = qvec * normal;
      const double a = -args.a, b = args.b;
      res11 = 0.0; //ip1.weight * (   a/2 * res + b * abs(res) );
      res21 = 0.0; //ip1.weight * (   a/2 * res - b * abs(res) );
      res22 = 0.0; //ip1.weight * ( - a/2 * res + b * abs(res) );
      res12 = 0.0; //ip1.weight * ( - a/2 * res - b * abs(res) );      
   }
};

/**
*  A class that describes a Mass Equation using Partial Assembly
*/
class MassEquation
{
public:
   static const PAOp OpName = BtDB;

   struct ArgsEmpty{};

   void evalD(double& res, ElementTransformation* Tr, const IntegrationPoint& ip,
               const Tensor<2>& Jac, const ArgsEmpty& args)
   {
      res = ip.weight * det(Jac);
   }

   struct ArgsCoeff
   {
      ArgsCoeff(Coefficient& coeff): coeff(coeff) {}
      Coefficient& coeff;
   };

   void evalD(double& res, ElementTransformation* Tr, const IntegrationPoint& ip,
               const Tensor<2>& Jac, const ArgsCoeff& args)
   {
      res = args.coeff.Eval(*Tr, ip) * ip.weight * det(Jac);
   }

   void evalD(double& res, ElementTransformation* Tr, const IntegrationPoint& ip,
               const Tensor<2>& Jac, Coefficient& coeff)
   {
      res = coeff.Eval(*Tr, ip) * ip.weight * det(Jac);
   }
};

}

}

#endif //MFEM_DGPABILININTEG