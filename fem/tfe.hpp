// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TEMPLATE_FINITE_ELEMENTS
#define MFEM_TEMPLATE_FINITE_ELEMENTS

#include "../config/tconfig.hpp"
#include "fe_coll.hpp"

namespace mfem
{

// Templated finite element classes, cf. fe.?pp

/** @brief Store mass-like matrix B for each integration point on the reference
    element.
    For tensor product evaluation, this is only called on the 1D reference
    element, and higher dimensions are put together from that.
    The element mass matrix can be written \f$ M_E = B^T D_E B \f$ where the B
    built here is the B, and is unchanging across the mesh. The diagonal matrix
    \f$ D_E \f$ then contains all the element-specific geometry and physics data.
    @param fe the element we are calculating on
    @param ir the integration rule to calculate the shape matrix on
    @param B must be (nip x dof) with column major storage
    @param dof_map the inverse of dof_map is applied to reorder local dofs.
*/
template <typename real_t>
void CalcShapeMatrix(const FiniteElement &fe, const IntegrationRule &ir,
                     real_t *B, const Array<int> *dof_map = NULL)
{
   // - B must be (nip x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   Vector shape(dof);

   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcShape(ir.IntPoint(ip), shape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         B[ip+nip*id] = shape(orig_id);
      }
   }
}

/** @brief store gradient matrix G for each integration point on the reference
    element.
    For tensor product evaluation, this is only called on the 1D reference
    element, and higher dimensions are put together from that.
    The element stiffness matrix can be written
    \f[
       S_E = \sum_{k=1}^{nq} G_{k,i}^T (D_E^G)_{k,k} G_{k,j}
    \f]
    where \f$ nq \f$ is the number of quadrature points, \f$ D_E^G \f$ contains
    all the information about the element geometry and coefficients (Jacobians
    etc.), and \f$ G \f$ is the matrix built in this routine, which is the same
    for all elements in a mesh.
    @param fe the element we are calculating on
    @param ir the integration rule to calculate the gradients on
    @param[out] G must be (nip x dim x dof) with column major storage
    @param[in] dof_map the inverse of dof_map is applied to reorder local dofs.
*/
template <typename real_t>
void CalcGradTensor(const FiniteElement &fe, const IntegrationRule &ir,
                    real_t *G, const Array<int> *dof_map = NULL)
{
   // - G must be (nip x dim x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int dim = fe.GetDim();
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   DenseMatrix dshape(dof, dim);

   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcDShape(ir.IntPoint(ip), dshape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         for (int d = 0; d < dim; d++)
         {
            G[ip+nip*(d+dim*id)] = dshape(orig_id, d);
         }
      }
   }
}

template <typename real_t>
void CalcShapes(const FiniteElement &fe, const IntegrationRule &ir,
                real_t *B, real_t *G, const Array<int> *dof_map)
{
   if (B) { mfem::CalcShapeMatrix(fe, ir, B, dof_map); }
   if (G) { mfem::CalcGradTensor(fe, ir, G, dof_map); }
}

// H1 finite elements

template <Geometry::Type G, int P>
class H1_FiniteElement;

template <int P>
class H1_FiniteElement<Geometry::SEGMENT, P>
{
public:
   static const Geometry::Type geom = Geometry::SEGMENT;
   static const int dim    = 1;
   static const int degree = P;
   static const int dofs   = P+1;

   static const bool tensor_prod = true;
   static const int  dofs_1d     = P+1;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type
   void Init(const parameter_type type_)
   {
      type = type_;
      if (type == BasisType::Positive)
      {
         H1Pos_SegmentElement *fe = new H1Pos_SegmentElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
      }
      else
      {
         int pt_type = BasisType::GetQuadrature1D(type);
         H1_SegmentElement *fe = new H1_SegmentElement(P, pt_type);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
      }
   }

public:
   H1_FiniteElement(const parameter_type type_ = BasisType::GaussLobatto)
   {
      Init(type_);
   }
   H1_FiniteElement(const FiniteElementCollection &fec)
   {
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(&fec);
      MFEM_ASSERT(h1_fec, "invalid FiniteElementCollection");
      Init(h1_fec->GetBasisType());
   }
   ~H1_FiniteElement() { delete my_fe; }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      CalcShapes(ir, B, G);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};

template <int P>
class H1_FiniteElement<Geometry::TRIANGLE, P>
{
public:
   static const Geometry::Type geom = Geometry::TRIANGLE;
   static const int dim    = 2;
   static const int degree = P;
   static const int dofs   = ((P + 1)*(P + 2))/2;

   static const bool tensor_prod = false;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe;
   parameter_type type; // run-time specified basis type
   void Init(const parameter_type type_)
   {
      type = type_;
      if (type == BasisType::Positive)
      {
         my_fe = new H1Pos_TriangleElement(P);
      }
      else
      {
         int pt_type = BasisType::GetQuadrature1D(type);
         my_fe = new H1_TriangleElement(P, pt_type);
      }
   }

public:
   H1_FiniteElement(const parameter_type type_ = BasisType::GaussLobatto)
   {
      Init(type_);
   }
   H1_FiniteElement(const FiniteElementCollection &fec)
   {
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(&fec);
      MFEM_ASSERT(h1_fec, "invalid FiniteElementCollection");
      Init(h1_fec->GetBasisType());
   }
   ~H1_FiniteElement() { delete my_fe; }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};

template <int P>
class H1_FiniteElement<Geometry::SQUARE, P>
{
public:
   static const Geometry::Type geom = Geometry::SQUARE;
   static const int dim     = 2;
   static const int degree  = P;
   static const int dofs    = (P+1)*(P+1);

   static const bool tensor_prod = true;
   static const int dofs_1d = P+1;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type
   void Init(const parameter_type type_)
   {
      type = type_;
      if (type == BasisType::Positive)
      {
         H1Pos_QuadrilateralElement *fe = new H1Pos_QuadrilateralElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2Pos_SegmentElement(P);
      }
      else
      {
         int pt_type = BasisType::GetQuadrature1D(type);
         H1_QuadrilateralElement *fe = new H1_QuadrilateralElement(P, pt_type);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2_SegmentElement(P, pt_type);
      }
   }

public:
   H1_FiniteElement(const parameter_type type_ = BasisType::GaussLobatto)
   {
      Init(type_);
   }
   H1_FiniteElement(const FiniteElementCollection &fec)
   {
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(&fec);
      MFEM_ASSERT(h1_fec, "invalid FiniteElementCollection");
      Init(h1_fec->GetBasisType());
   }
   ~H1_FiniteElement() { delete my_fe; delete my_fe_1d; }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe_1d, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};

template <int P>
class H1_FiniteElement<Geometry::TETRAHEDRON, P>
{
public:
   static const Geometry::Type geom = Geometry::TETRAHEDRON;
   static const int dim    = 3;
   static const int degree = P;
   static const int dofs   = ((P + 1)*(P + 2)*(P + 3))/6;

   static const bool tensor_prod = false;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe;
   parameter_type type; // run-time specified basis type
   void Init(const parameter_type type_)
   {
      type = type_;
      if (type == BasisType::Positive)
      {
         my_fe = new H1Pos_TetrahedronElement(P);
      }
      else
      {
         int pt_type = BasisType::GetQuadrature1D(type);
         my_fe = new H1_TetrahedronElement(P, pt_type);
      }
   }

public:
   H1_FiniteElement(const parameter_type type_ = BasisType::GaussLobatto)
   {
      Init(type_);
   }
   H1_FiniteElement(const FiniteElementCollection &fec)
   {
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(&fec);
      MFEM_ASSERT(h1_fec, "invalid FiniteElementCollection");
      Init(h1_fec->GetBasisType());
   }
   ~H1_FiniteElement() { delete my_fe; }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};

template <int P>
class H1_FiniteElement<Geometry::CUBE, P>
{
public:
   static const Geometry::Type geom = Geometry::CUBE;
   static const int dim     = 3;
   static const int degree  = P;
   static const int dofs    = (P+1)*(P+1)*(P+1);

   static const bool tensor_prod = true;
   static const int dofs_1d = P+1;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   const Array<int> *my_dof_map;
   parameter_type type; // run-time specified basis type

   void Init(const parameter_type type_)
   {
      type = type_;
      if (type == BasisType::Positive)
      {
         H1Pos_HexahedronElement *fe = new H1Pos_HexahedronElement(P);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2Pos_SegmentElement(P);
      }
      else
      {
         int pt_type = BasisType::GetQuadrature1D(type);
         H1_HexahedronElement *fe = new H1_HexahedronElement(P, pt_type);
         my_fe = fe;
         my_dof_map = &fe->GetDofMap();
         my_fe_1d = new L2_SegmentElement(P, pt_type);
      }
   }

public:
   H1_FiniteElement(const parameter_type type_ = BasisType::GaussLobatto)
   {
      Init(type_);
   }
   H1_FiniteElement(const FiniteElementCollection &fec)
   {
      const H1_FECollection *h1_fec =
         dynamic_cast<const H1_FECollection *>(&fec);
      MFEM_ASSERT(h1_fec, "invalid FiniteElementCollection");
      Init(h1_fec->GetBasisType());
   }
   ~H1_FiniteElement() { delete my_fe; delete my_fe_1d; }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe, ir, B, G, my_dof_map);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcShapes(*my_fe_1d, ir, B, G, NULL);
   }
   const Array<int> *GetDofMap() const { return my_dof_map; }
};


// L2 finite elements

template <Geometry::Type G, int P, typename L2_FE_type, typename L2Pos_FE_type,
          int DOFS, bool TP>
class L2_FiniteElement_base
{
public:
   static const Geometry::Type geom = G;
   static const int dim    = Geometry::Constants<G>::Dimension;
   static const int degree = P;
   static const int dofs   = DOFS;

   static const bool tensor_prod = TP;
   static const int  dofs_1d     = P+1;

   // Type for run-time parameter for the constructor
   typedef int parameter_type;

protected:
   const FiniteElement *my_fe, *my_fe_1d;
   parameter_type type; // run-time specified basis type

   void Init(const parameter_type type_)
   {
      type = type_;
      switch (type)
      {
         case BasisType::Positive:
            my_fe = new L2Pos_FE_type(P);
            my_fe_1d = (TP && dim != 1) ? new L2Pos_SegmentElement(P) : NULL;
            break;

         default:
            int pt_type = BasisType::GetQuadrature1D(type);
            my_fe = new L2_FE_type(P, pt_type);
            my_fe_1d = (TP && dim != 1) ? new L2_SegmentElement(P, pt_type) :
                       NULL;
      }
   }

   L2_FiniteElement_base(const parameter_type type)
   { Init(type); }

   L2_FiniteElement_base(const FiniteElementCollection &fec)
   {
      const L2_FECollection *l2_fec =
         dynamic_cast<const L2_FECollection *>(&fec);
      MFEM_ASSERT(l2_fec, "invalid FiniteElementCollection");
      Init(l2_fec->GetBasisType());
   }

   ~L2_FiniteElement_base() { delete my_fe; delete my_fe_1d; }

public:
   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *Grad) const
   {
      mfem::CalcShapes(*my_fe, ir, B, Grad, NULL);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *Grad) const
   {
      mfem::CalcShapes(dim == 1 ? *my_fe : *my_fe_1d, ir, B, Grad, NULL);
   }
   const Array<int> *GetDofMap() const { return NULL; }
};


template <Geometry::Type G, int P>
class L2_FiniteElement;


template <int P>
class L2_FiniteElement<Geometry::SEGMENT, P>
   : public L2_FiniteElement_base<
     Geometry::SEGMENT,P,L2_SegmentElement,L2Pos_SegmentElement,P+1,true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::SEGMENT,P,L2_SegmentElement,
           L2Pos_SegmentElement,P+1,true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = BasisType::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::TRIANGLE, P>
   : public L2_FiniteElement_base<Geometry::TRIANGLE,P,L2_TriangleElement,
     L2Pos_TriangleElement,((P+1)*(P+2))/2,false>
{
protected:
   typedef L2_FiniteElement_base<Geometry::TRIANGLE,P,L2_TriangleElement,
           L2Pos_TriangleElement,((P+1)*(P+2))/2,false> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = BasisType::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::SQUARE, P>
   : public L2_FiniteElement_base<Geometry::SQUARE,P,L2_QuadrilateralElement,
     L2Pos_QuadrilateralElement,(P+1)*(P+1),true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::SQUARE,P,L2_QuadrilateralElement,
           L2Pos_QuadrilateralElement,(P+1)*(P+1),true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = BasisType::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::TETRAHEDRON, P>
   : public L2_FiniteElement_base<Geometry::TETRAHEDRON,P,L2_TetrahedronElement,
     L2Pos_TetrahedronElement,((P+1)*(P+2)*(P+3))/6,false>
{
protected:
   typedef L2_FiniteElement_base<Geometry::TETRAHEDRON,P,L2_TetrahedronElement,
           L2Pos_TetrahedronElement,((P+1)*(P+2)*(P+3))/6,false> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = BasisType::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};


template <int P>
class L2_FiniteElement<Geometry::CUBE, P>
   : public L2_FiniteElement_base<Geometry::CUBE,P,L2_HexahedronElement,
     L2Pos_HexahedronElement,(P+1)*(P+1)*(P+1),true>
{
protected:
   typedef L2_FiniteElement_base<Geometry::CUBE,P,L2_HexahedronElement,
           L2Pos_HexahedronElement,(P+1)*(P+1)*(P+1),true> base_class;
public:
   typedef typename base_class::parameter_type parameter_type;
   L2_FiniteElement(const parameter_type type_ = BasisType::GaussLegendre)
      : base_class(type_) { }
   L2_FiniteElement(const FiniteElementCollection &fec)
      : base_class(fec) { }
};

} // namespace mfem

#endif // MFEM_TEMPLATE_FINITE_ELEMENTS
