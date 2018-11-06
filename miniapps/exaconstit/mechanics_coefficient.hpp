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

#ifndef MECHANICS_COEF
#define MECHANICS_COEF

#include "mfem.hpp"

namespace mfem
{
class QuadratureFunction;

/// Quadrature function coefficient
class QuadratureVectorFunctionCoefficient : public VectorCoefficient
{
private:
   QuadratureFunction *QuadF;

public:
   // constructor with a quadrature function as input
   QuadratureVectorFunctionCoefficient(QuadratureFunction *qf) 
      : VectorCoefficient() { QuadF = qf; }

   // constructor with a null qf
   QuadratureVectorFunctionCoefficient() : VectorCoefficient(0) { QuadF = NULL; }

   void SetQuadratureFunction(QuadratureFunction *qf) { QuadF = qf; }

   QuadratureFunction * GetQuadFunction() const { return QuadF; }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);

//   virtual void EvalQ(Vector &V, ElementTransformation &T,
//                      const int ip_num);

   virtual ~QuadratureVectorFunctionCoefficient() { };
};

}

#endif
