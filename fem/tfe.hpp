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

    @param B must be (nip x dof) with column major storage
    @param dof_map the inverse of dof_map is applied to reorder local dofs.
*/
template <typename real_t>
void CalcShapeMatrix(const FiniteElement &fe, const IntegrationRule &ir,
                     real_t *B, const Array<int> *dof_map = NULL)
{
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

    @param G must be (nip x dim x dof) with column major storage
    @param dof_map the inverse of dof_map is applied to reorder local dofs.
*/
template <typename real_t>
void CalcGradTensor(const FiniteElement &fe, const IntegrationRule &ir,
                    real_t *G, const Array<int> *dof_map = NULL)
{
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
bool absolute_comparison(const std::pair<real_t, int>& a,
                         const std::pair<real_t, int>& b)
{
   return (fabs(a.first) < fabs(b.first));
}

/**
   Find largest entries, normalize, put them in B1d

   This particular implementation is quite inefficient.
*/
template <typename real_t>
void CalcAlgSparsifiedShapeMatrix(
   const FiniteElement &fe, const IntegrationRule &ir, real_t *B,
   const Array<int> *dof_map = NULL, int sparsification_parameter = 2)
{
   // - B must be (nip x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   Vector shape(dof);

   // put high-order dense shape matrix in Btemp
   real_t * Btemp = new real_t[nip * dof];
   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcShape(ir.IntPoint(ip), shape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         Btemp[ip+nip*id] = shape(orig_id);
         B[ip+nip*id] = 0.0;
      }
   }

   int sc = std::min(sparsification_parameter, nip);
   for (int ip = 0; ip < nip; ip++)
   {
      std::vector<std::pair<real_t,int> > pairs;

      for (int id = 0; id < dof; id++)
      {
         real_t val = Btemp[ip+nip*id];
         pairs.push_back(std::make_pair(val, id));
      }
      std::sort(pairs.begin(), pairs.end(), absolute_comparison<real_t>);
      real_t sum = 0.0;
      for (int i=0; i<sc; ++i)
      {
         real_t val = pairs.at(dof-i-1).first;
         sum += val;
      }
      for (int i=0; i<sc; ++i)
      {
         real_t val = pairs.at(dof-i-1).first / sum;
         int id = pairs.at(dof-i-1).second;
         B[ip+nip*id] = val;
      }
   }

   delete [] Btemp;
}

template <typename real_t>
void CalcAlgSparsifiedGradTensor(
   const FiniteElement &fe, const IntegrationRule &ir, real_t *G,
   const Array<int> *dof_map = NULL, int sparsification_parameter = 2)
{
   // - G must be (nip x dim x dof) with column major storage
   // - The inverse of dof_map is applied to reorder the local dofs.
   int dim = fe.GetDim();
   int nip = ir.GetNPoints();
   int dof = fe.GetDof();
   MFEM_ASSERT(dim == 1,"Expect to use this in 1D!");
   DenseMatrix dshape(dof, dim);
   Vector shape(dof);

   // put *mass* tensor in Btemp, so sparsity is same as B1d
   real_t * Btemp = new real_t[nip * dof];
   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcShape(ir.IntPoint(ip), shape);
      for (int id = 0; id < dof; id++)
      {
         int orig_id = dof_map ? (*dof_map)[id] : id;
         Btemp[ip+nip*id] = shape(orig_id);
         for (int d = 0; d < dim; d++)
         {
            G[ip+nip*(d+dim*id)] = 0.0;
         }
      }
   }

   int sc = std::min(sparsification_parameter, nip);
   for (int ip = 0; ip < nip; ip++)
   {
      fe.CalcDShape(ir.IntPoint(ip), dshape);
      for (int d = 0; d < dim; d++)
      {
         std::vector<std::pair<real_t, int> > pairs;
         for (int id = 0; id < dof; id++)
         {
            pairs.push_back(std::make_pair(Btemp[ip+nip*(d+dim*id)], id));
         }
         std::sort(pairs.begin(), pairs.end(), absolute_comparison<real_t>);
         real_t sum_positive = 0.0;
         real_t sum_negative = 0.0;
         for (int i=0; i<sc; ++i)
         {
            int id = pairs.at(dof-i-1).second;
            int orig_id = dof_map ? (*dof_map)[id] : id;
            real_t val = dshape(orig_id, 0);
            if (val > 0.0)
            {
               sum_positive += val;
            }
            else
            {
               sum_negative += (-val);
            }
         }
         real_t positive_scale = (sum_positive + sum_negative) /
                                 (2.0 * sum_positive);
         if (sum_positive == 0.0)
            positive_scale = 1.0;
         real_t negative_scale = (sum_positive + sum_negative) /
                                 (2.0 * sum_negative);
         if (sum_negative == 0.0)
            negative_scale = 1.0;
         for (int i=0; i<sc; ++i)
         {
            int id = pairs.at(dof-i-1).second;
            int orig_id = dof_map ? (*dof_map)[id] : id;
            real_t val = dshape(orig_id, 0);
            if (val > 0.0)
            {
               G[ip+nip*(d+dim*id)] = positive_scale * val;
            }
            else
            {
               G[ip+nip*(d+dim*id)] = negative_scale * val;
            }
         }
      }
   }

   delete [] Btemp;
}

template <typename real_t>
void CalcShapes(const FiniteElement &fe, const IntegrationRule &ir,
                real_t *B, real_t *G, const Array<int> *dof_map)
{
   if (B) { mfem::CalcShapeMatrix(fe, ir, B, dof_map); }
   if (G) { mfem::CalcGradTensor(fe, ir, G, dof_map); }
}

template <typename real_t>
void CalcSparsifiedShapes(const FiniteElement &fe, const IntegrationRule &ir,
                          real_t *B, real_t *G, const Array<int> *dof_map,
                          int sp_strategy)
{
   MFEM_ASSERT(sp_strategy > 0, "sp_strategy must be positive!");

   if (B) { mfem::CalcAlgSparsifiedShapeMatrix(fe, ir, B, dof_map, sp_strategy); }
   if (G) { mfem::CalcAlgSparsifiedGradTensor(fe, ir, G, dof_map, sp_strategy); }
}

/// H1 finite elements, templated on geometry and order
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

/// generic Sparsified H1_FiniteElement
template <Geometry::Type GEOM, int P, int SP>
class Sparsified_H1_FE : public H1_FiniteElement<GEOM, P>
{
public:
   typedef H1_FiniteElement<GEOM, P> base_class;
   typedef typename base_class::parameter_type parameter_type;

   using base_class::my_fe;
   using base_class::my_fe_1d;
   using base_class::my_dof_map;

   Sparsified_H1_FE(const parameter_type type_ = BasisType::GaussLobatto) :
      base_class(type_) { }
   Sparsified_H1_FE(const FiniteElementCollection &fec) :
      base_class(fec) { }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcSparsifiedShapes(*my_fe, ir, B, G, my_dof_map, SP);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcSparsifiedShapes(*my_fe_1d, ir, B, G, NULL, SP);
   }
};

/// specialized for segment, because it doesn't have my_fe_1d
template <int P, int SP>
class Sparsified_H1_FE<Geometry::SEGMENT, P, SP> :
   public H1_FiniteElement<Geometry::SEGMENT, P>
{
public:
   typedef H1_FiniteElement<Geometry::SEGMENT, P> base_class;
   typedef typename base_class::parameter_type parameter_type;

   using base_class::my_fe;
   using base_class::my_dof_map;

   Sparsified_H1_FE(const parameter_type type_ = BasisType::GaussLobatto) :
      base_class(type_) { }
   Sparsified_H1_FE(const FiniteElementCollection &fec) :
      base_class(fec) { }

   template <typename real_t>
   void CalcShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      mfem::CalcSparsifiedShapes(*my_fe, ir, B, G, my_dof_map, SP);
   }
   template <typename real_t>
   void Calc1DShapes(const IntegrationRule &ir, real_t *B, real_t *G) const
   {
      CalcShapes(ir, B, G);
   }
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
