// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "marking.hpp"

namespace mfem
{

/// ShiftedFunctionCoefficient, similar to FunctionCoefficient, but also takes
/// into account a displacement vector if specified.
class ShiftedFunctionCoefficient : public Coefficient
{
protected:
   std::function<double(const Vector &)> Function;
   double constant = 0.0;
   bool constantcoefficient;

public:
   ShiftedFunctionCoefficient(std::function<double(const Vector &v)> F)
      : Function(std::move(F)), constantcoefficient(false) { }
   ShiftedFunctionCoefficient(double constant_)
      : constant(constant_), constantcoefficient(true) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      if (constantcoefficient) { return constant; }

      Vector D(T.GetSpaceDim());
      D = 0.;
      return (this)->Eval(T, ip, D);
   }

   /// Evaluate the coefficient at @a ip + @a D.
   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip,
               const Vector &D);
};

class ShiftedVectorFunctionCoefficient : public VectorCoefficient
{
protected:
   std::function<void(const Vector &, Vector &)> Function;

public:
   ShiftedVectorFunctionCoefficient(int dim,
                                    std::function<void(const Vector &, Vector &)> F)
      : VectorCoefficient(dim), Function(std::move(F)) { }

   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      Vector D(vdim);
      D = 0.;
      return (this)->Eval(V, T, ip, D);
   }

   /// Evaluate the coefficient at @a ip + @a D.
   void Eval(Vector &V,
             ElementTransformation &T,
             const IntegrationPoint &ip,
             const Vector &D);
};

/// BilinearFormIntegrator for the high-order extension of shifted boundary
/// method.
/// \f[
///  A(u, w) = -\langle \nabla u \cdot n, w \rangle
///            -\langle u + \nabla u \cdot d + h.o.t, \nabla w.n \rangle
///            +\langle \alpha h^{-1} (u + \nabla u \cdot d + h.o.t), w + \nabla w \cdot d + h.o.t \rangle
/// \f]
/// where \f$h.o.t\f$ include higher-order derivatives (\f$\nabla^k u\f$) due to Taylor
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
   int NEproc;                // Number of elements on the current MPI rank
   int par_shared_face_count; //
   Array<int> cut_marker;     // Array with marker values for cut-cell
   // corresponding to the level set that BilinearForm applies to.

   // these are not thread-safe!
   Vector shape, dshapedn, nor, nh, ni;
   DenseMatrix dshape, dshapephys, adjJ;


public:
   SBM2DirichletIntegrator(const ParMesh *pmesh,
                           const double a,
                           VectorCoefficient &vD_,
                           Array<int> &elem_marker_,
                           Array<int> &cut_marker_,
                           bool include_cut_cell_ = false,
                           int nterms_ = 0)
      : alpha(a), vD(&vD_),
        elem_marker(&elem_marker_),
        include_cut_cell(include_cut_cell_),
        nterms(nterms_),
        NEproc(pmesh->GetNE()),
        par_shared_face_count(0),
        cut_marker(cut_marker_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   virtual ~SBM2DirichletIntegrator() { }
};

/// LinearFormIntegrator for the high-order extension of shifted boundary
/// method.
/// \f[
///   (u, w) = -\langle u_D, \nabla w \cdot n  \rangle
///            +\langle \alpha h^{-1} u_D, w + \nabla w \cdot d + h.o.t \rangle
/// \f]
/// where \f$h.o.t\f$ include higher-order derivatives (\f$\nabla^k u\f$) due to Taylor
/// expansion. Since this interior face integrator is applied to the surrogate
/// boundary (see marking.hpp for notes on how the surrogate faces are
/// determined and elements are marked), this integrator adds contribution to
/// only the element that is adjacent to that face (Trans.Elem1 or Trans.Elem2)
/// and is part of the surrogate domain.
/// Note that \f$u_D\f$ is evaluated at the true boundary using the distance function
/// and ShiftedFunctionCoefficient, i.e. \f$u_D(x_{true}) = u_D(x_{surrogate} + D)\f$,
/// where \f$x_{surrogate}\f$ is the location of the integration point on the surrogate
/// boundary and \f$D\f$ is the distance vector from the surrogate boundary to the
/// true boundary.
class SBM2DirichletLFIntegrator : public LinearFormIntegrator
{
protected:
   ShiftedFunctionCoefficient *uD;
   double alpha;              // Nitsche parameter
   VectorCoefficient *vD;     // Distance function coefficient
   Array<int> *elem_marker;   // marker indicating whether element is inside,
   //cut, or outside the domain.
   bool include_cut_cell;     // include element cut by true boundary
   int nterms;                // Number of terms in addition to the gradient
   // term from Taylor expansion that should be included. (0 by default).
   int NEproc;                // Number of elements on the current MPI rank
   int par_shared_face_count; //
   int ls_cut_marker;         // Flag used for the cut-cell corresponding to the
   // level set.

   // these are not thread-safe!
   Vector shape, dshape_dd, dshape_dn, nor, nh, ni;
   DenseMatrix dshape, adjJ;

public:
   SBM2DirichletLFIntegrator(const ParMesh *pmesh,
                             ShiftedFunctionCoefficient &u,
                             const double alpha_,
                             VectorCoefficient &vD_,
                             Array<int> &elem_marker_,
                             bool include_cut_cell_ = false,
                             int nterms_ = 0,
                             int ls_cut_marker_ = ShiftedFaceMarker::SBElementType::CUT)
      : uD(&u), alpha(alpha_), vD(&vD_),
        elem_marker(&elem_marker_),
        include_cut_cell(include_cut_cell_),
        nterms(nterms_),
        NEproc(pmesh->GetNE()),
        par_shared_face_count(0),
        ls_cut_marker(ls_cut_marker_) { }

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


/// BilinearFormIntegrator for Neumann boundaries using the shifted boundary
/// method.
/// \f[
///   A(u,w) = \langle [\nabla u + \nabla(\nabla u) \cdot d + h.o.t.] \cdot \hat{n} \, (n \cdot \hat{n}),w ⟩ - \langle \nabla u \cdot n,w \rangle
/// \f]
/// where h.o.t are the high-order terms due to Taylor expansion for \f$\nabla u\f$,
/// \f$\hat{n}\f$ is the normal vector at the true boundary, \f$n\f$ is the normal vector at
/// the surrogate boundary. Since this interior face integrator is applied to
/// the surrogate boundary (see marking.hpp for notes on how the surrogate faces
/// are determined and elements are marked), this integrator adds contribution
/// to only the element that is adjacent to that face (Trans.Elem1 or
/// Trans.Elem2) and is part of the surrogate domain.
class SBM2NeumannIntegrator : public BilinearFormIntegrator
{
protected:
   ShiftedVectorFunctionCoefficient *vN; // Normal function coefficient
   VectorCoefficient *vD;     // Distance function coefficient
   Array<int> *elem_marker;   // Marker indicating whether element is inside,
   //cut, or outside the domain.
   bool include_cut_cell;
   int nterms;                // Number of terms in addition to the gradient
   // term from Taylor expansion that should be included. (0 by default).
   int NEproc;                // Number of elements on the current MPI rank
   int par_shared_face_count; //
   Array<int> cut_marker;


   // these are not thread-safe!
   Vector shape, dshapedn, nor, nh, ni;
   DenseMatrix dshape, adjJ;


public:
   SBM2NeumannIntegrator(const ParMesh *pmesh,
                         VectorCoefficient &vD_,
                         ShiftedVectorFunctionCoefficient &vN_,
                         Array<int> &elem_marker_,
                         Array<int> &cut_marker_,
                         bool include_cut_cell_ = false,
                         int nterms_ = 1)
      : vN(&vN_), vD(&vD_),
        elem_marker(&elem_marker_),
        include_cut_cell(include_cut_cell_),
        nterms(nterms_),
        NEproc(pmesh->GetNE()),
        par_shared_face_count(0),
        cut_marker(cut_marker_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   bool GetTrimFlag() const { return include_cut_cell; }

   virtual ~SBM2NeumannIntegrator() { }
};

/// LinearFormIntegrator for Neumann boundaries using the shifted boundary
/// method.
/// \f[
///   (u, w) = \langle \hat{n} \cdot n \, t_n, w \rangle
/// \f]
/// where \f$\hat{n}\f$ is the normal vector at the true boundary, \f$n\f$ is the normal vector
/// at the surrogate boundary, and \f$t_n\f$ is the traction boundary condition.
/// Since this interior face integrator is applied to the surrogate boundary
/// (see marking.hpp for notes on how the surrogate faces are determined and
/// elements are marked), this integrator adds contribution to only the element
/// that is adjacent to that face (Trans.Elem1 or Trans.Elem2) and is part of
/// the surrogate domain.
/// Note that \f$t_n\f$ is evaluated at the true boundary using the distance function
/// and ShiftedFunctionCoefficient, i.e. \f$t_n(x_{true}) = t_N(x_{surrogate} + D)\f$,
/// where \f$x_{surrogate}\f$ is the location of the integration point on the surrogate
/// boundary and \f$D\f$ is the distance vector from the surrogate boundary to the
/// true boundary.
class SBM2NeumannLFIntegrator : public LinearFormIntegrator
{
protected:
   ShiftedVectorFunctionCoefficient *vN; // Normal function coefficient
   ShiftedFunctionCoefficient *uN; // Neumann condition on true boundary
   VectorCoefficient *vD;     // Distance function coefficient
   Array<int> *elem_marker;   // Marker indicating whether element is inside,
   int nterms;                // Number of terms in addition to the gradient
   // term from Taylor expansion that should be included. (0 by default).
   bool include_cut_cell;
   int NEproc;                // Number of elements on the current MPI rank
   int par_shared_face_count;
   int ls_cut_marker;

   // these are not thread-safe!
   Vector shape, nor;

public:
   SBM2NeumannLFIntegrator(const ParMesh *pmesh,
                           ShiftedFunctionCoefficient &u,
                           VectorCoefficient &vD_,
                           ShiftedVectorFunctionCoefficient &vN_,
                           Array<int> &elem_marker_,
                           int nterms_ = 0,
                           bool include_cut_cell_ = false,
                           int ls_cut_marker_ = ShiftedFaceMarker::SBElementType::CUT)
      :  vN(&vN_), uN(&u), vD(&vD_),
         elem_marker(&elem_marker_),
         nterms(nterms_),
         include_cut_cell(include_cut_cell_),
         NEproc(pmesh->GetNE()),
         par_shared_face_count(0),
         ls_cut_marker(ls_cut_marker_) { }

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
   bool GetTrimFlag() const { return include_cut_cell; }
};

} // namespace mfem

#endif
