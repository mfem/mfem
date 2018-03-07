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

#include "../config/config.hpp"
#include "bilininteg.hpp"
#include "partialassemblykernel.hpp"
#include "dgfacefunctions.hpp"

#include "fem.hpp"
#include <cmath>
#include <algorithm>
#include "../linalg/vector.hpp"

namespace mfem
{

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
   *  Returns the values of the D tensor at a given integration Point.
   */
   void evalD(Tensor<1>& res, ElementTransformation *Tr, const IntegrationPoint& ip,
                  VectorCoefficient& q, double a = 1.0)
   {
      int dim = res.size(0);
      Vector qvec(dim);
      const DenseMatrix& locD = Tr->AdjugateJacobian();
      q.Eval(qvec, *Tr, ip);
      for (int i = 0; i < dim; ++i)
      {
         double val = 0.0;
         for (int j = 0; j < dim; ++j)
         {
            val += locD(i,j) * qvec(j);
         }
         res(i) = ip.weight * a * val;
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
      const FaceElementTransformations* face_tr, const Vector& normal,
      const IntegrationPoint& ip1, const IntegrationPoint& ip2,
      VectorCoefficient &q, double a = 1.0, double b = 1.0)
   {
      int dim = normal.Size();
      Vector qvec(dim);
      double res = 0.0;
      //FIXME: qvec might be discontinuous if not constant with a periodic mesh
      // We should then use the evaluation on Elem2 and eip2
      q.Eval( qvec, *(face_tr->Elem1), ip1 );
      res = qvec * normal;
      res11 = ip1.weight * (   a/2 * res + b * abs(res) );
      res21 = ip1.weight * (   a/2 * res - b * abs(res) );
      res22 = ip1.weight * ( - a/2 * res + b * abs(res) );
      res12 = ip1.weight * ( - a/2 * res - b * abs(res) );
   }  

};

}

#endif //MFEM_DGPABILININTEG