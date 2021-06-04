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

#ifndef MFEM_FE_COLLECTION
#define MFEM_FE_COLLECTION

#include "../config/config.hpp"
#include "geom.hpp"
#include "fe.hpp"

namespace mfem
{

/** Collection of finite elements from the same family in multiple dimensions.
    This class is used to match the degrees of freedom of a FiniteElementSpace
    between elements, and to provide the finite element restriction from an
    element to its boundary. */
class FiniteElementCollection
{
protected:
   template <Geometry::Type geom>
   static inline void GetNVE(int &nv, int &ne);

   template <Geometry::Type geom, typename v_t>
   static inline void GetEdge(int &nv, v_t &v, int &ne, int &e, int &eo,
                              const int edge_info);

   template <Geometry::Type geom, Geometry::Type f_geom,
             typename v_t, typename e_t, typename eo_t>
   static inline void GetFace(int &nv, v_t &v, int &ne, e_t &e, eo_t &eo,
                              int &nf, int &f, Geometry::Type &fg, int &fo,
                              const int face_info);

public:
   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const = 0;

   virtual int DofForGeometry(Geometry::Type GeomType) const = 0;

   /** @brief Returns an array, say p, that maps a local permuted index i to
       a local base index: base_i = p[i]. */
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const = 0;

   virtual const char * Name() const { return "Undefined"; }

   int HasFaceDofs(Geometry::Type GeomType) const;

   virtual const FiniteElement *TraceFiniteElementForGeometry(
      Geometry::Type GeomType) const
   {
      return FiniteElementForGeometry(GeomType);
   }

   virtual FiniteElementCollection *GetTraceCollection() const;

   virtual ~FiniteElementCollection() { }

   /** @brief Factory method: return a newly allocated FiniteElementCollection
       according to the given name. */
   static FiniteElementCollection *New(const char *name);

   /** @brief Get the local dofs for a given sub-manifold.

      Return the local dofs for a SDim-dimensional sub-manifold (0D - vertex,
      1D - edge, 2D - face) including those on its boundary. The local index of
      the sub-manifold (inside Geom) and its orientation are given by the
      parameter Info = 64 * SubIndex + SubOrientation. Naturally, it is assumed
      that 0 <= SDim <= Dim(Geom). */
   void SubDofOrder(Geometry::Type Geom, int SDim, int Info,
                    Array<int> &dofs) const;
};

/// Arbitrary order H1-conforming (continuous) finite elements.
class H1_FECollection : public FiniteElementCollection
{

protected:
   int b_type;
   char h1_name[32];
   FiniteElement *H1_Elements[Geometry::NumGeom];
   int H1_dof[Geometry::NumGeom];
   int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];

public:
   explicit H1_FECollection(const int p, const int dim = 3,
                            const int btype = BasisType::GaussLobatto);

   virtual const FiniteElement *FiniteElementForGeometry(
      Geometry::Type GeomType) const
   { return H1_Elements[GeomType]; }
   virtual int DofForGeometry(Geometry::Type GeomType) const
   { return H1_dof[GeomType]; }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char *Name() const { return h1_name; }
   FiniteElementCollection *GetTraceCollection() const;

   int GetBasisType() const { return b_type; }
   /// Get the Cartesian to local H1 dof map
   const int *GetDofMap(Geometry::Type GeomType) const;

   virtual ~H1_FECollection();
};

/** Arbitrary order H1-conforming (continuous) finite elements with positive
    basis functions. */
class H1Pos_FECollection : public H1_FECollection
{
public:
   explicit H1Pos_FECollection(const int p, const int dim = 3)
      : H1_FECollection(p, dim, BasisType::Positive) { }
};

/** Arbitrary order H1-conforming (continuous) serendipity finite elements;
    Current implementation works in 2D only; 3D version is in development. */
class H1Ser_FECollection : public H1_FECollection
{
public:
   explicit H1Ser_FECollection(const int p, const int dim = 2)
      : H1_FECollection(p, dim, BasisType::Serendipity) { };
};

/** Arbitrary order "H^{1/2}-conforming" trace finite elements defined on the
    interface between mesh elements (faces,edges,vertices); these are the trace
    FEs of the H1-conforming FEs. */
class H1_Trace_FECollection : public H1_FECollection
{
public:
   H1_Trace_FECollection(const int p, const int dim,
                         const int btype = BasisType::GaussLobatto);
};

/// Arbitrary order "L2-conforming" discontinuous finite elements.
class L2_FECollection : public FiniteElementCollection
{
private:
   int b_type; // BasisType
   char d_name[32];
   ScalarFiniteElement *L2_Elements[Geometry::NumGeom];
   ScalarFiniteElement *Tr_Elements[Geometry::NumGeom];
   int *SegDofOrd[2]; // for rotating segment dofs in 1D
   int *TriDofOrd[6]; // for rotating triangle dofs in 2D
   int *OtherDofOrd;  // for rotating other types of elements (for Or == 0)

public:
   L2_FECollection(const int p, const int dim,
                   const int btype = BasisType::GaussLegendre,
                   const int map_type = FiniteElement::VALUE);

   virtual const FiniteElement *FiniteElementForGeometry(
      Geometry::Type GeomType) const
   {
      return L2_Elements[GeomType];
   }
   virtual int DofForGeometry(Geometry::Type GeomType) const
   {
      if (L2_Elements[GeomType])
      {
         return L2_Elements[GeomType]->GetDof();
      }
      return 0;
   }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char *Name() const { return d_name; }

   virtual const FiniteElement *TraceFiniteElementForGeometry(
      Geometry::Type GeomType) const
   {
      return Tr_Elements[GeomType];
   }

   int GetBasisType() const { return b_type; }

   virtual ~L2_FECollection();
};

/// Declare an alternative name for L2_FECollection = DG_FECollection
typedef L2_FECollection DG_FECollection;

/// Arbitrary order H(div)-conforming Raviart-Thomas finite elements.
class RT_FECollection : public FiniteElementCollection
{
protected:
   int ob_type; // open BasisType
   char rt_name[32];
   FiniteElement *RT_Elements[Geometry::NumGeom];
   int RT_dof[Geometry::NumGeom];
   int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];

   // Initialize only the face elements
   void InitFaces(const int p, const int dim, const int map_type,
                  const bool signs);

   // Constructor used by the constructor of the RT_Trace_FECollection and
   // DG_Interface_FECollection classes
   RT_FECollection(const int p, const int dim, const int map_type,
                   const bool signs,
                   const int ob_type = BasisType::GaussLegendre);

public:
   RT_FECollection(const int p, const int dim,
                   const int cb_type = BasisType::GaussLobatto,
                   const int ob_type = BasisType::GaussLegendre);

   virtual const FiniteElement *FiniteElementForGeometry(
      Geometry::Type GeomType) const
   { return RT_Elements[GeomType]; }
   virtual int DofForGeometry(Geometry::Type GeomType) const
   { return RT_dof[GeomType]; }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char *Name() const { return rt_name; }
   FiniteElementCollection *GetTraceCollection() const;

   virtual ~RT_FECollection();
};

/** Arbitrary order "H^{-1/2}-conforming" face finite elements defined on the
    interface between mesh elements (faces); these are the normal trace FEs of
    the H(div)-conforming FEs. */
class RT_Trace_FECollection : public RT_FECollection
{
public:
   RT_Trace_FECollection(const int p, const int dim,
                         const int map_type = FiniteElement::INTEGRAL,
                         const int ob_type = BasisType::GaussLegendre);
};

/** Arbitrary order discontinuous finite elements defined on the interface
    between mesh elements (faces). The functions in this space are single-valued
    on each face and are discontinuous across its boundary. */
class DG_Interface_FECollection : public RT_FECollection
{
public:
   DG_Interface_FECollection(const int p, const int dim,
                             const int map_type = FiniteElement::VALUE,
                             const int ob_type = BasisType::GaussLegendre);
};

/// Arbitrary order H(curl)-conforming Nedelec finite elements.
class ND_FECollection : public FiniteElementCollection
{
protected:
   char nd_name[32];
   FiniteElement *ND_Elements[Geometry::NumGeom];
   int ND_dof[Geometry::NumGeom];
   int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];

public:
   ND_FECollection(const int p, const int dim,
                   const int cb_type = BasisType::GaussLobatto,
                   const int ob_type = BasisType::GaussLegendre);

   virtual const FiniteElement *FiniteElementForGeometry(Geometry::Type GeomType)
   const
   { return ND_Elements[GeomType]; }
   virtual int DofForGeometry(Geometry::Type GeomType) const
   { return ND_dof[GeomType]; }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char *Name() const { return nd_name; }
   FiniteElementCollection *GetTraceCollection() const;

   virtual ~ND_FECollection();
};

/** Arbitrary order H(curl)-trace finite elements defined on the interface
    between mesh elements (faces,edges); these are the tangential trace FEs of
    the H(curl)-conforming FEs. */
class ND_Trace_FECollection : public ND_FECollection
{
public:
   ND_Trace_FECollection(const int p, const int dim,
                         const int cb_type = BasisType::GaussLobatto,
                         const int ob_type = BasisType::GaussLegendre);
};

/// Arbitrary order non-uniform rational B-splines (NURBS) finite elements.
class NURBSFECollection : public FiniteElementCollection
{
private:
   NURBS1DFiniteElement *SegmentFE;
   NURBS2DFiniteElement *QuadrilateralFE;
   NURBS3DFiniteElement *ParallelepipedFE;

   mutable int mOrder; // >= 1 or VariableOrder
   // The 'name' can be:
   // 1) name = "NURBS" + "number", for fixed order, or
   // 2) name = "NURBS", for VariableOrder.
   // The name is updated before writing it to a stream, for example, see
   // FiniteElementSpace::Save().
   mutable char name[16];

public:
   enum { VariableOrder = -1 };

   /** @brief The parameter @a Order must be either a positive number, for fixed
      order, or VariableOrder (default). */
   explicit NURBSFECollection(int Order = VariableOrder);

   void Reset() const
   {
      SegmentFE->Reset();
      QuadrilateralFE->Reset();
      ParallelepipedFE->Reset();
   }

   /** @brief Get the order of the NURBS collection: either a positive number,
       when using fixed order, or VariableOrder. */
   int GetOrder() const { return mOrder; }

   /** @brief Set the order and the name, based on the given @a Order: either a
       positive number for fixed order, or VariableOrder. */
   void SetOrder(int Order) const;

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char *Name() const { return name; }

   FiniteElementCollection *GetTraceCollection() const;

   virtual ~NURBSFECollection();
};


/// Piecewise-(bi)linear continuous finite elements.
class LinearFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Linear1DFiniteElement SegmentFE;
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;
   const Linear3DFiniteElement TetrahedronFE;
   const TriLinear3DFiniteElement ParallelepipedFE;
   const H1_WedgeElement WedgeFE;
public:
   LinearFECollection() : WedgeFE(1) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "Linear"; }
};

/// Piecewise-(bi)quadratic continuous finite elements.
class QuadraticFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Quad1DFiniteElement SegmentFE;
   const Quad2DFiniteElement TriangleFE;
   const BiQuad2DFiniteElement QuadrilateralFE;
   const Quadratic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;
   const H1_WedgeElement WedgeFE;

public:
   QuadraticFECollection() : ParallelepipedFE(2), WedgeFE(2) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "Quadratic"; }
};

/// Version of QuadraticFECollection with positive basis functions.
class QuadraticPosFECollection : public FiniteElementCollection
{
private:
   const QuadPos1DFiniteElement   SegmentFE;
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "QuadraticPos"; }
};

/// Piecewise-(bi)cubic continuous finite elements.
class CubicFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Cubic1DFiniteElement SegmentFE;
   const Cubic2DFiniteElement TriangleFE;
   const BiCubic2DFiniteElement QuadrilateralFE;
   const Cubic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;
   const H1_WedgeElement WedgeFE;

public:
   CubicFECollection()
      : ParallelepipedFE(3), WedgeFE(3, BasisType::ClosedUniform) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "Cubic"; }
};

/// Crouzeix-Raviart nonconforming elements in 2D.
class CrouzeixRaviartFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE;
   const CrouzeixRaviartFiniteElement TriangleFE;
   const CrouzeixRaviartQuadFiniteElement QuadrilateralFE;
public:
   CrouzeixRaviartFECollection() : SegmentFE(1) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "CrouzeixRaviart"; }
};

/// Piecewise-linear nonconforming finite elements in 3D.
class LinearNonConf3DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P1TetNonConfFiniteElement TetrahedronFE;
   const P0QuadFiniteElement QuadrilateralFE;
   const RotTriLinearHexFiniteElement ParallelepipedFE;

public:
   LinearNonConf3DFECollection () { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "LinearNonConf3D"; }
};


/** First order Raviart-Thomas finite elements in 2D. This class is kept only
    for backward compatibility, consider using RT_FECollection instead. */
class RT0_2DFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE; // normal component on edge
   const RT0TriangleFiniteElement TriangleFE;
   const RT0QuadFiniteElement QuadrilateralFE;
public:
   RT0_2DFECollection() : SegmentFE(0) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RT0_2D"; }
};

/** Second order Raviart-Thomas finite elements in 2D. This class is kept only
    for backward compatibility, consider using RT_FECollection instead. */
class RT1_2DFECollection : public FiniteElementCollection
{
private:
   const P1SegmentFiniteElement SegmentFE; // normal component on edge
   const RT1TriangleFiniteElement TriangleFE;
   const RT1QuadFiniteElement QuadrilateralFE;
public:
   RT1_2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RT1_2D"; }
};

/** Third order Raviart-Thomas finite elements in 2D. This class is kept only
    for backward compatibility, consider using RT_FECollection instead. */
class RT2_2DFECollection : public FiniteElementCollection
{
private:
   const P2SegmentFiniteElement SegmentFE; // normal component on edge
   const RT2TriangleFiniteElement TriangleFE;
   const RT2QuadFiniteElement QuadrilateralFE;
public:
   RT2_2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RT2_2D"; }
};

/** Piecewise-constant discontinuous finite elements in 2D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class Const2DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P0QuadFiniteElement QuadrilateralFE;
public:
   Const2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "Const2D"; }
};

/** Piecewise-linear discontinuous finite elements in 2D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class LinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;

public:
   LinearDiscont2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "LinearDiscont2D"; }
};

/// Version of LinearDiscont2DFECollection with dofs in the Gaussian points.
class GaussLinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const CrouzeixRaviartFiniteElement TriangleFE;
   const GaussLinear2DFiniteElement TriangleFE;
   const GaussBiLinear2DFiniteElement QuadrilateralFE;

public:
   GaussLinearDiscont2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "GaussLinearDiscont2D"; }
};

/// Linear (P1) finite elements on quadrilaterals.
class P1OnQuadFECollection : public FiniteElementCollection
{
private:
   const P1OnQuadFiniteElement QuadrilateralFE;
public:
   P1OnQuadFECollection() { }
   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;
   virtual int DofForGeometry(Geometry::Type GeomType) const;
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char * Name() const { return "P1OnQuad"; }
};

/** Piecewise-quadratic discontinuous finite elements in 2D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class QuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Quad2DFiniteElement TriangleFE;
   const BiQuad2DFiniteElement QuadrilateralFE;

public:
   QuadraticDiscont2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "QuadraticDiscont2D"; }
};

/// Version of QuadraticDiscont2DFECollection with positive basis functions.
class QuadraticPosDiscont2DFECollection : public FiniteElementCollection
{
private:
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosDiscont2DFECollection() { }
   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;
   virtual int DofForGeometry(Geometry::Type GeomType) const;
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const
   { return NULL; }
   virtual const char * Name() const { return "QuadraticPosDiscont2D"; }
};

/// Version of QuadraticDiscont2DFECollection with dofs in the Gaussian points.
class GaussQuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const Quad2DFiniteElement TriangleFE;
   const GaussQuad2DFiniteElement TriangleFE;
   const GaussBiQuad2DFiniteElement QuadrilateralFE;

public:
   GaussQuadraticDiscont2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "GaussQuadraticDiscont2D"; }
};

/** Piecewise-cubic discontinuous finite elements in 2D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class CubicDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Cubic2DFiniteElement TriangleFE;
   const BiCubic2DFiniteElement QuadrilateralFE;

public:
   CubicDiscont2DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "CubicDiscont2D"; }
};

/** Piecewise-constant discontinuous finite elements in 3D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class Const3DFECollection : public FiniteElementCollection
{
private:
   const P0TetFiniteElement TetrahedronFE;
   const P0HexFiniteElement ParallelepipedFE;
   const L2_WedgeElement WedgeFE;

public:
   Const3DFECollection() : WedgeFE(0) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "Const3D"; }
};

/** Piecewise-linear discontinuous finite elements in 3D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class LinearDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Linear3DFiniteElement TetrahedronFE;
   const TriLinear3DFiniteElement ParallelepipedFE;

public:
   LinearDiscont3DFECollection () { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "LinearDiscont3D"; }
};

/** Piecewise-quadratic discontinuous finite elements in 3D. This class is kept
    only for backward compatibility, consider using L2_FECollection instead. */
class QuadraticDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Quadratic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;

public:
   QuadraticDiscont3DFECollection () : ParallelepipedFE(2) { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "QuadraticDiscont3D"; }
};

/// Finite element collection on a macro-element.
class RefinedLinearFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const RefinedLinear1DFiniteElement SegmentFE;
   const RefinedLinear2DFiniteElement TriangleFE;
   const RefinedBiLinear2DFiniteElement QuadrilateralFE;
   const RefinedLinear3DFiniteElement TetrahedronFE;
   const RefinedTriLinear3DFiniteElement ParallelepipedFE;

public:
   RefinedLinearFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RefinedLinear"; }
};

/** Lowest order Nedelec finite elements in 3D. This class is kept only for
    backward compatibility, consider using the new ND_FECollection instead. */
class ND1_3DFECollection : public FiniteElementCollection
{
private:
   const Nedelec1HexFiniteElement HexahedronFE;
   const Nedelec1TetFiniteElement TetrahedronFE;

public:
   ND1_3DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "ND1_3D"; }
};

/** First order Raviart-Thomas finite elements in 3D. This class is kept only
    for backward compatibility, consider using RT_FECollection instead. */
class RT0_3DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P0QuadFiniteElement QuadrilateralFE;
   const RT0HexFiniteElement HexahedronFE;
   const RT0TetFiniteElement TetrahedronFE;
public:
   RT0_3DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RT0_3D"; }
};

/** Second order Raviart-Thomas finite elements in 3D. This class is kept only
    for backward compatibility, consider using RT_FECollection instead. */
class RT1_3DFECollection : public FiniteElementCollection
{
private:
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;
   const RT1HexFiniteElement HexahedronFE;
public:
   RT1_3DFECollection() { }

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;

   virtual const char * Name() const { return "RT1_3D"; }
};

/// Discontinuous collection defined locally by a given finite element.
class Local_FECollection : public FiniteElementCollection
{
private:
   char d_name[32];
   Geometry::Type GeomType;
   FiniteElement *Local_Element;

public:
   Local_FECollection(const char *fe_name);

   virtual const FiniteElement *FiniteElementForGeometry(
      Geometry::Type _GeomType) const
   { return (GeomType == _GeomType) ? Local_Element : NULL; }
   virtual int DofForGeometry(Geometry::Type _GeomType) const
   { return (GeomType == _GeomType) ? Local_Element->GetDof() : 0; }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const
   { return NULL; }
   virtual const char *Name() const { return d_name; }

   virtual ~Local_FECollection() { delete Local_Element; }
};

/// Radial basis function collection
class KernelFECollection : public FiniteElementCollection
{
private:
   int maxDim;
   char d_name[32];
   ScalarFiniteElement *Tr_Elements[Geometry::NumGeom];
   ScalarFiniteElement *L2_Elements[Geometry::NumGeom];
   int *SegDofOrd[2]; // for rotating segment dofs in 1D
   int *OtherDofOrd; 
   bool ValidGeomType(Geometry::Type GeomType) const;
   
public:
   KernelFECollection(const int D,
                      const int numPointsD,
                      const double h,
                      const int rbfType,
                      const int distNorm,
                      const int order = -1,
                      const int intOrder = 2, // num integration points per 1d point
                      const int mapType = FiniteElement::VALUE);
   virtual ~KernelFECollection();
   
   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual const FiniteElement *
   TraceFiniteElementForGeometry(Geometry::Type GeomType) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const;

   virtual const int * DofOrderForOrientation(Geometry::Type GeomType, int Or) const;

   virtual const char * Name() const { return d_name; }
   // virtual int GetContType() const { return DISCONTINUOUS; }
};

} // namespace mfem

#endif
