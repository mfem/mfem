// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SBM_SOLVER_HPP
#define MFEM_SBM_SOLVER_HPP

#include "mfem.hpp"

namespace mfem
{

/// ShiftedFunctionCoefficient, similar to FunctionCoefficient, but also takes
/// into account a displacement vector if specified.
class ShiftedFunctionCoefficient : public Coefficient
{
protected:
   std::function<double(const Vector &)> Function;

public:
   ShiftedFunctionCoefficient(std::function<double(const Vector &v)> F)
      : Function(std::move(F)) { }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector D(1);
      D = 0.;
      return (this)->Eval(T, ip, D);
   }

   /// Evaluate the coefficient at @a ip + @a D.
   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip,
               const Vector &D);
};

/// BilinearFormIntegrator for the high-order extension of shifted boundary
/// method.
/// A(u, w) = -<nabla u.n, w>
///           -<u + nabla u.d + h.o.t, nabla w.n>
///           -<alpha h^{-1} (u + nabla u.d + h.o.t), w + nabla w.d + h.o.t>
/// where h.o.t include higher-order derivatives (nabla^k u) due to Taylor
/// expansion. Since this interior face integrator is applied to the surrogate
/// boundary (see marking.hpp for notes on how the surrogate faces are
/// determined and elements are marked), this integrator adds contribution to
/// only the element that is adjacent to that face (Trans.Elem1 or Trans.Elem2)
/// and is part of the surrogate domain.
class SBM2DirichletIntegrator : public BilinearFormIntegrator
{
protected:
   double alpha;
   VectorCoefficient *vD;     // Distance function coefficient
   Array<int> *elem_marker;   // marker indicating whether element is inside,
   //cut, or outside the domain.
   bool include_cut_cell;     // include element cut by true boundary
   int nterms;                // Number of terms in addition to the gradient
   // term from Taylor expansion that should be included. (0 by default).
   int NEproc;                //Number of elements on the current MPI rank
   int par_shared_face_count; //

   // these are not thread-safe!
   Vector shape, dshapedn, dshapephysdn, nor, nh, ni;
   DenseMatrix jmat, dshape, dshapephys, adjJ;


public:
   SBM2DirichletIntegrator(const ParMesh *pmesh,
                           const double a,
                           VectorCoefficient &vD_,
                           Array<int> &elem_marker_,
                           bool include_cut_cell_ = false,
                           int nterms_ = 0)
      : alpha(a), vD(&vD_),
        elem_marker(&elem_marker_),
        include_cut_cell(include_cut_cell_),
        nterms(nterms_),
        NEproc(pmesh->GetNE()),
        par_shared_face_count(0) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   virtual ~SBM2DirichletIntegrator() { }
};

/// LinearFormIntegrator for the high-order extension of shifted boundary
/// method.
/// (u, w) = -<u_D, nabla w.n >
///          -<alpha h^{-1} u_D, w + nabla w.d + h.o.t>
/// where h.o.t include higher-order derivatives (nabla^k u) due to Taylor
/// expansion. Since this interior face integrator is applied to the surrogate
/// boundary (see marking.hpp for notes on how the surrogate faces are
/// determined and elements are marked), this integrator adds contribution to
/// only the element that is adjacent to that face (Trans.Elem1 or Trans.Elem2)
/// and is part of the surrogate domain.
/// Note that u_D is evaluated at the true boundary using the distance function
/// and ShiftedFunctionCoefficient, i.e. u_D(x_true) = u_D(x_surrogate + D),
/// where x_surrogate is the location of the integration point on the surrogate
/// boundary and D is the distance vector from the surrogate boundary to the
/// true boundary.
class SBM2DirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   ShiftedFunctionCoefficient *uD;
   double alpha;              // Nitsche parameter
   VectorCoefficient *vD;     // Distance function coefficient
   Array<int> *elem_marker;   //marker indicating whether element is inside,
   //cut, or outside the domain.
   bool include_cut_cell;     // include element cut by true boundary
   int nterms;                // Number of terms in addition to the gradient
   // term from Taylor expansion that should be included. (0 by default).
   int NEproc;                //Number of elements on the current MPI rank
   int par_shared_face_count; //

   // these are not thread-safe!
   Vector shape, dshape_dd, dshape_dn, nor, nh, ni;
   DenseMatrix dshape, mq, adjJ;

public:
   SBM2DirichletLFIntegrator(const ParMesh *pmesh,
                             ShiftedFunctionCoefficient &u,
                             const double a,
                             VectorCoefficient &vD_,
                             Array<int> &elem_marker_,
                             bool include_cut_cell_ = false,
                             int nterms_ = 0)
      : uD(&u), alpha(a), vD(&vD_),
        elem_marker(&elem_marker_),
        include_cut_cell(include_cut_cell_),
        nterms(nterms_),
        NEproc(pmesh->GetNE()),
        par_shared_face_count(0) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};

} // namespace mfem

#endif
