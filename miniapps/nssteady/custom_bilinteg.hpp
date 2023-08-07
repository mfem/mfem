// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"

namespace mfem
{

/** Class for integrating the trilinear form for the convective operator 
    
    c(w,u,v) := alpha (w . grad u, v) = alpha * w . grad u
    
    for vector field FE spaces, where alpha is a scalar, w is a known vector field (VectorCoefficient),
    u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined by scalar FE through standard transformation.
    The resulting local element matrix is square, of size <tt> dim*dof </tt>,
    where \c dim is the vector dimension space and \c dof is the local degrees
    of freedom. 
*/

class VectorConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *W;
   double alpha;
   bool SkewSym;
   // PA extension // Not supported yet
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

private:
   DenseMatrix dshape, adjJ, W_ir, pelmat, pelmat_T;
   Vector shape, vec1, vec2, vec3;

public:
   VectorConvectionIntegrator(VectorCoefficient &w, double alpha = 1.0, bool SkewSym_ = false)
      : W(&w), alpha(alpha), SkewSym(SkewSym_) {}

   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                         ElementTransformation &T);
                                         
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

};


/** Class for integrating the trilinear form for the convective operator 
    
    c(u,w,v) := alpha (u . grad w, v) = alpha * u . grad w
    
    for vector field FE spaces, where alpha is a scalar, w is a known vector field (VectorGridFunctionCoefficient),
    u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined by scalar FE through standard transformation.
    The resulting local element matrix is square, of size <tt> dim*dof </tt>,
    where \c dim is the vector dimension space and \c dof is the local degrees
    of freedom. 
*/
class VectorGradCoefficientIntegrator : public BilinearFormIntegrator
{
protected:
   VectorGridFunctionCoefficient *W;
   const GridFunction* W_gf;
   double alpha;
   // PA extension // Not supported yet
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

private:
   DenseMatrix pelmat, gradW;
   Vector shape;

public:
   VectorGradCoefficientIntegrator(VectorGridFunctionCoefficient &w, double alpha = 1.0)
      : W(&w), alpha(alpha) {}


   static const IntegrationRule &GetRule(const FiniteElement &fe,
                                       ElementTransformation &T);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};

}