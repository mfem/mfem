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

#ifndef MFEM_FE_COLLECTION
#define MFEM_FE_COLLECTION

#include "../config/config.hpp"
#include "geom.hpp"
#include "fe.hpp"

namespace mfem
{

/** @brief Collection of finite elements from the same family in multiple
    dimensions. This class is used to match the degrees of freedom of a
    FiniteElementSpace between elements, and to provide the finite element
    restriction from an element to its boundary. */
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
   /** @brief Enumeration for ContType: defines the continuity of the field
       across element interfaces. */
   enum { CONTINUOUS,   ///< Field is continuous across element interfaces
          TANGENTIAL,   ///< Tangential components of vector field
          NORMAL,       ///< Normal component of vector field
          DISCONTINUOUS ///< Field is discontinuous across element interfaces
        };

   virtual const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const = 0;

   /** @brief Returns the first non-NULL FiniteElement for the given dimension

       @note Repeatedly calls FiniteElementForGeometry in the order defined in
       the Geometry::Type enumeration.
   */
   virtual const FiniteElement *FiniteElementForDim(int dim) const;

   virtual int DofForGeometry(Geometry::Type GeomType) const = 0;

   /** @brief Returns a DoF transformation object compatible with this basis
       and geometry type.
   */
   virtual const StatelessDofTransformation *
   DofTransformationForGeometry(Geometry::Type GeomType) const
   { return NULL; }

   /** @brief Returns an array, say p, that maps a local permuted index i to a
       local base index: base_i = p[i].

       @note Only provides information about interior dofs. See
       FiniteElementCollection::SubDofOrder if interior \a and boundary dof
       order is needed. */
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const = 0;

   virtual const char *Name() const { return "Undefined"; }

   virtual int GetContType() const = 0;

   /** @note The following methods provide the same information as the
       corresponding methods of the FiniteElement base class.
       @{
   */
   int GetRangeType(int dim) const;
   int GetDerivRangeType(int dim) const;
   int GetMapType(int dim) const;
   int GetDerivType(int dim) const;
   int GetDerivMapType(int dim) const;
   int GetRangeDim(int dim) const;
   /** @} */

   int HasFaceDofs(Geometry::Type geom, int p) const;

   virtual const FiniteElement *TraceFiniteElementForGeometry(
      Geometry::Type GeomType) const
   {
      return FiniteElementForGeometry(GeomType);
   }

   virtual FiniteElementCollection *GetTraceCollection() const;

   virtual ~FiniteElementCollection();

   /** @brief Factory method: return a newly allocated FiniteElementCollection
       according to the given name. */
   /**
   | FEC Name | Space | Order | BasisType | FiniteElement::MapT | Notes |
   | :------: | :---: | :---: | :-------: | :-----: | :---: |
   | H1_[DIM]_[ORDER] | H1 | * | 1 | VALUE | H1 nodal elements |
   | H1@[BTYPE]_[DIM]_[ORDER] | H1 | * | * | VALUE | H1 nodal elements |
   | H1Pos_[DIM]_[ORDER] | H1 | * | 1 | VALUE | H1 nodal elements |
   | H1Pos_Trace_[DIM]_[ORDER] | H^{1/2} | * | 2 | VALUE | H^{1/2}-conforming trace elements for H1 defined on the interface between mesh elements (faces,edges,vertices) |
   | H1_Trace_[DIM]_[ORDER] | H^{1/2} | * | 1 | VALUE | H^{1/2}-conforming trace elements for H1 defined on the interface between mesh elements (faces,edges,vertices) |
   | H1_Trace@[BTYPE]_[DIM]_[ORDER] | H^{1/2} | * | 1 | VALUE | H^{1/2}-conforming trace elements for H1 defined on the interface between mesh elements (faces,edges,vertices) |
   | ND_[DIM]_[ORDER] | H(curl) | * | 1 / 0 | H_CURL | Nedelec vector elements |
   | ND@[CBTYPE][OBTYPE]_[DIM]_[ORDER] | H(curl) | * | * / * | H_CURL | Nedelec vector elements |
   | ND_Trace_[DIM]_[ORDER] | H^{1/2} | * | 1 / 0  | H_CURL | H^{1/2}-conforming trace elements for H(curl) defined on the interface between mesh elements (faces) |
   | ND_Trace@[CBTYPE][OBTYPE]_[DIM]_[ORDER] | H^{1/2} | * | 1 / 0 | H_CURL | H^{1/2}-conforming trace elements for H(curl) defined on the interface between mesh elements (faces) |
   | RT_[DIM]_[ORDER] | H(div) | * | 1 / 0 | H_DIV | Raviart-Thomas vector elements |
   | RT@[CBTYPE][OBTYPE]_[DIM]_[ORDER] | H(div) | * | * / * | H_DIV | Raviart-Thomas vector elements |
   | RT_Trace_[DIM]_[ORDER] | H^{1/2} | * | 1 / 0 | INTEGRAL | H^{1/2}-conforming trace elements for H(div) defined on the interface between mesh elements (faces) |
   | RT_ValTrace_[DIM]_[ORDER] | H^{1/2} | * | 1 / 0 | VALUE | H^{1/2}-conforming trace elements for H(div) defined on the interface between mesh elements (faces) |
   | RT_Trace@[BTYPE]_[DIM]_[ORDER] | H^{1/2} | * | 1 / 0 | INTEGRAL | H^{1/2}-conforming trace elements for H(div) defined on the interface between mesh elements (faces) |
   | RT_ValTrace@[BTYPE]_[DIM]_[ORDER] |  H^{1/2} | * | 1 / 0 | VALUE | H^{1/2}-conforming trace elements for H(div) defined on the interface between mesh elements (faces) |
   | L2_[DIM]_[ORDER] | L2 | * | 0 | VALUE | Discontinuous L2 elements |
   | L2_T[BTYPE]_[DIM]_[ORDER] | L2 | * | 0 | VALUE | Discontinuous L2 elements |
   | L2Int_[DIM]_[ORDER] | L2 | * | 0 | INTEGRAL | Discontinuous L2 elements |
   | L2Int_T[BTYPE]_[DIM]_[ORDER] | L2 | * | 0 | INTEGRAL | Discontinuous L2 elements |
   | DG_Iface_[DIM]_[ORDER] | - | * | 0 | VALUE | Discontinuous elements on the interface between mesh elements (faces) |
   | DG_Iface@[BTYPE]_[DIM]_[ORDER] | - | * | 0 | VALUE | Discontinuous elements on the interface between mesh elements (faces) |
   | DG_IntIface_[DIM]_[ORDER] | - | * | 0 | INTEGRAL | Discontinuous elements on the interface between mesh elements (faces) |
   | DG_IntIface@[BTYPE]_[DIM]_[ORDER] | - | * | 0 | INTEGRAL | Discontinuous elements on the interface between mesh elements (faces) |
   | NURBS[ORDER] | - | * | - | VALUE | Non-Uniform Rational B-Splines (NURBS) elements |
   | LinearNonConf3D | - | 1 | 1 | VALUE | Piecewise-linear nonconforming finite elements in 3D |
   | CrouzeixRaviart | - | - | - | - | Crouzeix-Raviart nonconforming elements in 2D |
   | Local_[FENAME] | - | - | - | - | Special collection that builds a local version out of the FENAME collection |
   |-|-|-|-|-|-|
   | Linear | H1 | 1 | 1 | VALUE | Left in for backward compatibility, consider using H1_ |
   | Quadratic | H1 | 2 | 1 | VALUE | Left in for backward compatibility, consider using H1_ |
   | QuadraticPos | H1 | 2 | 2 | VALUE | Left in for backward compatibility, consider using H1_ |
   | Cubic | H1 | 2 | 1 | VALUE | Left in for backward compatibility, consider using H1_ |
   | Const2D | L2 | 0 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | Const3D | L2 | 0 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | LinearDiscont2D | L2 | 1 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | GaussLinearDiscont2D | L2 | 1 | 0 | VALUE | Left in for backward compatibility, consider using L2_ |
   | P1OnQuad | H1 | 1 | 1 | VALUE | Linear P1 element with 3 nodes on a square |
   | QuadraticDiscont2D | L2 | 2 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | QuadraticPosDiscont2D | L2 | 2 | 2 | VALUE | Left in for backward compatibility, consider using L2_ |
   | GaussQuadraticDiscont2D | L2 | 2 | 0 | VALUE | Left in for backward compatibility, consider using L2_ |
   | CubicDiscont2D | L2 | 3 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | LinearDiscont3D | L2 | 1 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | QuadraticDiscont3D | L2 | 2 | 1 | VALUE | Left in for backward compatibility, consider using L2_ |
   | ND1_3D | H(Curl) | 1 | 1 / 0 | H_CURL | Left in for backward compatibility, consider using ND_ |
   | RT0_2D | H(Div) | 1 | 1 / 0 | H_DIV | Left in for backward compatibility, consider using RT_ |
   | RT1_2D | H(Div) | 2 | 1 / 0 | H_DIV | Left in for backward compatibility, consider using RT_ |
   | RT2_2D | H(Div) | 3 | 1 / 0 | H_DIV | Left in for backward compatibility, consider using RT_ |
   | RT0_3D | H(Div) | 1 | 1 / 0 | H_DIV | Left in for backward compatibility, consider using RT_ |
   | RT1_3D | H(Div) | 2 | 1 / 0 | H_DIV | Left in for backward compatibility, consider using RT_ |

   | Tag | Description |
   | :------: | :--------: |
   | [DIM]    | Dimension of the elements (1D, 2D, 3D) |
   | [ORDER]  | Approximation order of the elements (P0, P1, P2, ...) |
   | [BTYPE]  | BasisType of the element (0-GaussLegendre, 1 - GaussLobatto, 2-Bernstein, 3-OpenUniform, 4-CloseUniform, 5-OpenHalfUniform) |
   | [OBTYPE] | Open BasisType of the element for elements which have both types |
   | [CBTYPE] | Closed BasisType of the element for elements which have both types |

   [FENAME]  Is a special case for the Local FEC which generates a local version of a given
   FEC.  It is selected from one of (BiCubic2DFiniteElement, Quad_Q3, Nedelec1HexFiniteElement,
      Hex_ND1, H1_[DIM]_[ORDER],H1Pos_[DIM]_[ORDER], L2_[DIM]_[ORDER] )
   */
   static FiniteElementCollection *New(const char *name);

   /** @brief Get the local dofs for a given sub-manifold.

      Return the local dofs for a SDim-dimensional sub-manifold (0D - vertex, 1D
      - edge, 2D - face) including those on its boundary. The local index of the
      sub-manifold (inside Geom) and its orientation are given by the parameter
      Info = 64 * SubIndex + SubOrientation. Naturally, it is assumed that 0 <=
      SDim <= Dim(Geom). */
   void SubDofOrder(Geometry::Type Geom, int SDim, int Info,
                    Array<int> &dofs) const;

   /// Variable order version of FiniteElementForGeometry().
   /** The order parameter @a p represents the order of the highest-dimensional
       FiniteElement%s the fixed-order collection we want to query. In general,
       this order is different from the order of the returned FiniteElement. */
   const FiniteElement *GetFE(Geometry::Type geom, int p) const
   {
      if (p == base_p) { return FiniteElementForGeometry(geom); }
      if (p >= var_orders.Size() || !var_orders[p]) { InitVarOrder(p); }
      return var_orders[p]->FiniteElementForGeometry(geom);
   }

   /// Variable order version of TraceFiniteElementForGeometry().
   /** The order parameter @a p represents the order of the highest-dimensional
       FiniteElement%s the fixed-order collection we want to query. In general,
       this order is different from the order of the returned FiniteElement. */
   const FiniteElement *GetTraceFE(Geometry::Type geom, int p) const
   {
      if (p == base_p) { return TraceFiniteElementForGeometry(geom); }
      if (p >= var_orders.Size() || !var_orders[p]) { InitVarOrder(p); }
      return var_orders[p]->TraceFiniteElementForGeometry(geom);
   }

   /// Variable order version of DofForGeometry().
   /** The order parameter @a p represents the order of the highest-dimensional
       FiniteElement%s the fixed-order collection we want to query. In general,
       this order is different from the order of the element corresponding to
       @a geom in that fixed-order collection. */
   int GetNumDof(Geometry::Type geom, int p) const
   {
      if (p == base_p) { return DofForGeometry(geom); }
      if (p >= var_orders.Size() || !var_orders[p]) { InitVarOrder(p); }
      return var_orders[p]->DofForGeometry(geom);
   }

   /// Variable order version of DofOrderForOrientation().
   /** The order parameter @a p represents the order of the highest-dimensional
       FiniteElement%s the fixed-order collection we want to query. In general,
       this order is different from the order of the element corresponding to
       @a geom in that fixed-order collection. */
   const int *GetDofOrdering(Geometry::Type geom, int p, int ori) const
   {
      if (p == base_p) { return DofOrderForOrientation(geom, ori); }
      if (p >= var_orders.Size() || !var_orders[p]) { InitVarOrder(p); }
      return var_orders[p]->DofOrderForOrientation(geom, ori);
   }

   /** @brief Return the order (polynomial degree) of the FE collection,
       corresponding to the order/degree returned by FiniteElement::GetOrder()
       of the highest-dimensional FiniteElement%s defined by the collection. */
   int GetOrder() const { return base_p; }

   /// Instantiate a new collection of the same type with a different order.
   /** Generally, the order parameter @a p is NOT the same as the parameter @a p
       used by some of the constructors of derived classes. Instead, this @a p
       represents the order of the new FE collection as it will be returned by
       its GetOrder() method. */
   virtual FiniteElementCollection *Clone(int p) const;

protected:
   const int base_p; ///< Order as returned by GetOrder().

   FiniteElementCollection() : base_p(0) {}
   FiniteElementCollection(int p) : base_p(p) {}

   void InitVarOrder(int p) const;

   mutable Array<FiniteElementCollection*> var_orders;

   /// How to treat errors in FiniteElementForGeometry() calls.
   enum ErrorMode
   {
      RETURN_NULL,      ///< Return NULL on errors
      RAISE_MFEM_ERROR  /**< Raise an MFEM error (default in base class).
                             Sub-classes can ignore this and return NULL. */
   };

   /// How to treat errors in FiniteElementForGeometry() calls.
   /** The typical error in derived classes is that no FiniteElement is defined
       for the given Geometry, or the input is not a valid Geometry. */
   mutable ErrorMode error_mode = RAISE_MFEM_ERROR;
};

/// Arbitrary order H1-conforming (continuous) finite elements.
class H1_FECollection : public FiniteElementCollection
{
protected:
   int dim, b_type;
   char h1_name[32];
   FiniteElement *H1_Elements[Geometry::NumGeom];
   int H1_dof[Geometry::NumGeom];
   int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8], *TetDofOrd[24];

public:
   explicit H1_FECollection(const int p, const int dim = 3,
                            const int btype = BasisType::GaussLobatto);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override
   { return H1_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return h1_name; }

   int GetContType() const override { return CONTINUOUS; }

   int GetBasisType() const { return b_type; }

   FiniteElementCollection *GetTraceCollection() const override;

   /// Get the Cartesian to local H1 dof map
   const int *GetDofMap(Geometry::Type GeomType) const;
   /// Variable order version of GetDofMap
   const int *GetDofMap(Geometry::Type GeomType, int p) const;

   FiniteElementCollection *Clone(int p) const override
   { return new H1_FECollection(p, dim, b_type); }

   virtual ~H1_FECollection();
};

/** @brief Arbitrary order H1-conforming (continuous) finite elements with
    positive basis functions. */
class H1Pos_FECollection : public H1_FECollection
{
public:
   explicit H1Pos_FECollection(const int p, const int dim = 3)
      : H1_FECollection(p, dim, BasisType::Positive) {}
};

/** Arbitrary order H1-conforming (continuous) serendipity finite elements;
    Current implementation works in 2D only; 3D version is in development. */
class H1Ser_FECollection : public H1_FECollection
{
public:
   explicit H1Ser_FECollection(const int p, const int dim = 2)
      : H1_FECollection(p, dim, BasisType::Serendipity) {}
};

/** @brief Arbitrary order "H^{1/2}-conforming" trace finite elements defined on
    the interface between mesh elements (faces,edges,vertices); these are the
    trace FEs of the H1-conforming FEs. */
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
   int dim;
   int b_type; // BasisType
   int m_type; // map type
   char d_name[32];
   ScalarFiniteElement *L2_Elements[Geometry::NumGeom];
   ScalarFiniteElement *Tr_Elements[Geometry::NumGeom];
   int *SegDofOrd[2];  // for rotating segment dofs in 1D
   int *TriDofOrd[6];  // for rotating triangle dofs in 2D
   int *TetDofOrd[24]; // for rotating tetrahedron dofs in 3D
   int *OtherDofOrd;   // for rotating other types of elements (for Or == 0)

public:
   L2_FECollection(const int p, const int dim,
                   const int btype = BasisType::GaussLegendre,
                   const int map_type = FiniteElement::VALUE);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override
   {
      if (L2_Elements[GeomType])
      {
         return L2_Elements[GeomType]->GetDof();
      }
      return 0;
   }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return d_name; }

   int GetContType() const override { return DISCONTINUOUS; }

   const FiniteElement *
   TraceFiniteElementForGeometry(Geometry::Type GeomType) const override
   {
      return Tr_Elements[GeomType];
   }

   int GetBasisType() const { return b_type; }

   FiniteElementCollection *Clone(int p) const override
   { return new L2_FECollection(p, dim, b_type, m_type); }

   virtual ~L2_FECollection();
};

/// Declare an alternative name for L2_FECollection = DG_FECollection
typedef L2_FECollection DG_FECollection;

/// Arbitrary order H(div)-conforming Raviart-Thomas finite elements.
class RT_FECollection : public FiniteElementCollection
{
protected:
   int dim;
   int cb_type; // closed BasisType
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
   /// Construct an H(div)-conforming Raviart-Thomas FE collection, RT_p.
   /** The index @a p corresponds to the space RT_p, as typically denoted in the
       literature, which contains vector polynomials of degree up to (p+1).
       For example, the RT_0 collection contains vector-valued linear functions
       and, in particular, FiniteElementCollection::GetOrder() will,
       correspondingly, return order 1. */
   RT_FECollection(const int p, const int dim,
                   const int cb_type = BasisType::GaussLobatto,
                   const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override
   { return RT_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return rt_name; }

   int GetContType() const override { return NORMAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   int GetClosedBasisType() const { return cb_type; }
   int GetOpenBasisType() const { return ob_type; }

   FiniteElementCollection *Clone(int p) const override
   { return new RT_FECollection(p, dim, cb_type, ob_type); }

   virtual ~RT_FECollection();
};

/** @brief Arbitrary order "H^{-1/2}-conforming" face finite elements defined on
    the interface between mesh elements (faces); these are the normal trace FEs
    of the H(div)-conforming FEs. */
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
   int dim;
   int cb_type; // closed BasisType
   int ob_type; // open BasisType
   char nd_name[32];
   FiniteElement *ND_Elements[Geometry::NumGeom];
   int ND_dof[Geometry::NumGeom];
   int *SegDofOrd[2], *TriDofOrd[6], *QuadDofOrd[8];

public:
   ND_FECollection(const int p, const int dim,
                   const int cb_type = BasisType::GaussLobatto,
                   const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override
   { return ND_dof[GeomType]; }

   const StatelessDofTransformation *
   DofTransformationForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return nd_name; }

   int GetContType() const override { return TANGENTIAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   int GetClosedBasisType() const { return cb_type; }
   int GetOpenBasisType() const { return ob_type; }

   FiniteElementCollection *Clone(int p) const override
   { return new ND_FECollection(p, dim, cb_type, ob_type); }

   virtual ~ND_FECollection();
};

/** @brief Arbitrary order H(curl)-trace finite elements defined on the
    interface between mesh elements (faces,edges); these are the tangential
    trace FEs of the H(curl)-conforming FEs. */
class ND_Trace_FECollection : public ND_FECollection
{
public:
   ND_Trace_FECollection(const int p, const int dim,
                         const int cb_type = BasisType::GaussLobatto,
                         const int ob_type = BasisType::GaussLegendre);
};

/// Arbitrary order 3D H(curl)-conforming Nedelec finite elements in 1D.
class ND_R1D_FECollection : public FiniteElementCollection
{
protected:
   char nd_name[32];
   FiniteElement *ND_Elements[Geometry::NumGeom];
   int ND_dof[Geometry::NumGeom];

public:
   ND_R1D_FECollection(const int p, const int dim,
                       const int cb_type = BasisType::GaussLobatto,
                       const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override
   { return ND_Elements[GeomType]; }

   int DofForGeometry(Geometry::Type GeomType) const override
   { return ND_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return nd_name; }

   int GetContType() const override { return TANGENTIAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~ND_R1D_FECollection();
};

/// Arbitrary order 3D H(div)-conforming Raviart-Thomas finite elements in 1D.
class RT_R1D_FECollection : public FiniteElementCollection
{
protected:
   char rt_name[32];
   FiniteElement *RT_Elements[Geometry::NumGeom];
   int RT_dof[Geometry::NumGeom];

public:
   RT_R1D_FECollection(const int p, const int dim,
                       const int cb_type = BasisType::GaussLobatto,
                       const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override
   { return RT_Elements[GeomType]; }

   int DofForGeometry(Geometry::Type GeomType) const override
   { return RT_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return rt_name; }

   int GetContType() const override { return NORMAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~RT_R1D_FECollection();
};

/// Arbitrary order 3D H(curl)-conforming Nedelec finite elements in 2D.
class ND_R2D_FECollection : public FiniteElementCollection
{
protected:
   char nd_name[32];
   FiniteElement *ND_Elements[Geometry::NumGeom];
   int ND_dof[Geometry::NumGeom];
   int *SegDofOrd[2];

public:
   ND_R2D_FECollection(const int p, const int dim,
                       const int cb_type = BasisType::GaussLobatto,
                       const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override
   { return ND_Elements[GeomType]; }

   int DofForGeometry(Geometry::Type GeomType) const override
   { return ND_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return nd_name; }

   int GetContType() const override { return TANGENTIAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~ND_R2D_FECollection();
};

/** @brief Arbitrary order 3D H(curl)-trace finite elements in 2D defined on the
    interface between mesh elements (edges); these are the tangential
    trace FEs of the H(curl)-conforming FEs. */
class ND_R2D_Trace_FECollection : public ND_R2D_FECollection
{
public:
   ND_R2D_Trace_FECollection(const int p, const int dim,
                             const int cb_type = BasisType::GaussLobatto,
                             const int ob_type = BasisType::GaussLegendre);
};

/// Arbitrary order 3D H(div)-conforming Raviart-Thomas finite elements in 2D.
class RT_R2D_FECollection : public FiniteElementCollection
{
protected:
   int ob_type; // open BasisType
   char rt_name[32];
   FiniteElement *RT_Elements[Geometry::NumGeom];
   int RT_dof[Geometry::NumGeom];
   int *SegDofOrd[2];

   // Initialize only the face elements
   void InitFaces(const int p, const int dim, const int map_type,
                  const bool signs);

   // Constructor used by the constructor of the RT_R2D_Trace_FECollection
   RT_R2D_FECollection(const int p, const int dim, const int map_type,
                       const bool signs,
                       const int ob_type = BasisType::GaussLegendre);

public:
   RT_R2D_FECollection(const int p, const int dim,
                       const int cb_type = BasisType::GaussLobatto,
                       const int ob_type = BasisType::GaussLegendre);

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override
   { return RT_Elements[GeomType]; }

   int DofForGeometry(Geometry::Type GeomType) const override
   { return RT_dof[GeomType]; }

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return rt_name; }

   int GetContType() const override { return NORMAL; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~RT_R2D_FECollection();
};

/** @brief Arbitrary order 3D "H^{-1/2}-conforming" face finite elements defined on
    the interface between mesh elements (faces); these are the normal trace FEs
    of the H(div)-conforming FEs. */
class RT_R2D_Trace_FECollection : public RT_R2D_FECollection
{
public:
   RT_R2D_Trace_FECollection(const int p, const int dim,
                             const int map_type = FiniteElement::INTEGRAL,
                             const int ob_type = BasisType::GaussLegendre);
};

/// Arbitrary order non-uniform rational B-splines (NURBS) finite elements.
class NURBSFECollection : public FiniteElementCollection
{
protected:
   PointFiniteElement *PointFE;
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

   virtual void Reset() const
   {
      SegmentFE->Reset();
      QuadrilateralFE->Reset();
      ParallelepipedFE->Reset();
   }

   virtual void SetDim(const int dim) {};

   /** @brief Get the order of the NURBS collection: either a positive number,
       when using fixed order, or VariableOrder. */
   /** @note Not to be confused with FiniteElementCollection::GetOrder(). */
   int GetOrder() const { return mOrder; }

   /** @brief Set the order and the name, based on the given @a Order: either a
       positive number for fixed order, or VariableOrder. */
   virtual void SetOrder(int Order) const;

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return name; }

   int GetContType() const override { return CONTINUOUS; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~NURBSFECollection();
};

/// Arbitrary order H(div) NURBS finite elements.
class NURBS_HDivFECollection : public NURBSFECollection
{
private:

   NURBS1DFiniteElement *SegmentFE;
   NURBS2DFiniteElement *QuadrilateralFE;

   NURBS_HDiv2DFiniteElement *QuadrilateralVFE;
   NURBS_HDiv3DFiniteElement *ParallelepipedVFE;

   FiniteElement *sFE;
   FiniteElement *qFE;
   FiniteElement *hFE;

public:

   /** @brief The parameter @a Order must be either a positive number, for fixed
      order, or VariableOrder (default). */
   explicit NURBS_HDivFECollection(int Order = VariableOrder, const int vdim = -1);

   void Reset() const override
   {
      SegmentFE->Reset();
      QuadrilateralFE->Reset();
      QuadrilateralVFE->Reset();
      ParallelepipedVFE->Reset();
   }

   void SetDim(const int dim) override;

   /** @brief Set the order and the name, based on the given @a Order: either a
       positive number for fixed order, or VariableOrder. */
   void SetOrder(int Order) const override;

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return name; }

   int GetContType() const override { return CONTINUOUS; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~NURBS_HDivFECollection();
};

/// Arbitrary order H(curl) NURBS finite elements.
class NURBS_HCurlFECollection : public NURBSFECollection
{
private:
   NURBS1DFiniteElement *SegmentFE;
   NURBS2DFiniteElement *QuadrilateralFE;

   NURBS_HCurl2DFiniteElement *QuadrilateralVFE;
   NURBS_HCurl3DFiniteElement *ParallelepipedVFE;

   FiniteElement *sFE;
   FiniteElement *qFE;
   FiniteElement *hFE;
public:

   /** @brief The parameter @a Order must be either a positive number, for fixed
      order, or VariableOrder (default). */
   explicit NURBS_HCurlFECollection(int Order = VariableOrder,
                                    const int vdim = -1);

   void Reset() const override
   {
      SegmentFE->Reset();
      QuadrilateralFE->Reset();
      QuadrilateralVFE->Reset();
      ParallelepipedVFE->Reset();
   }

   void SetDim(const int dim) override;

   /** @brief Set the order and the name, based on the given @a Order: either a
       positive number for fixed order, or VariableOrder. */
   void SetOrder(int Order) const override;

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return name; }

   int GetContType() const override { return CONTINUOUS; }

   FiniteElementCollection *GetTraceCollection() const override;

   virtual ~NURBS_HCurlFECollection();
};

/// Piecewise-(bi/tri)linear continuous finite elements.
class LinearFECollection : public FiniteElementCollection
{
private:
   const PointFiniteElement PointFE;
   const Linear1DFiniteElement SegmentFE;
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;
   const Linear3DFiniteElement TetrahedronFE;
   const TriLinear3DFiniteElement ParallelepipedFE;
   const LinearWedgeFiniteElement WedgeFE;
   const LinearPyramidFiniteElement PyramidFE;
public:
   LinearFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "Linear"; }

   int GetContType() const override { return CONTINUOUS; }
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
   QuadraticFECollection()
      : FiniteElementCollection(2), ParallelepipedFE(2), WedgeFE(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "Quadratic"; }

   int GetContType() const override { return CONTINUOUS; }
};

/// Version of QuadraticFECollection with positive basis functions.
class QuadraticPosFECollection : public FiniteElementCollection
{
private:
   const QuadPos1DFiniteElement   SegmentFE;
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "QuadraticPos"; }

   int GetContType() const override { return CONTINUOUS; }
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
      : FiniteElementCollection(3),
        ParallelepipedFE(3), WedgeFE(3, BasisType::ClosedUniform)
   {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "Cubic"; }

   int GetContType() const override { return CONTINUOUS; }
};

/// Crouzeix-Raviart nonconforming elements in 2D.
class CrouzeixRaviartFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE;
   const CrouzeixRaviartFiniteElement TriangleFE;
   const CrouzeixRaviartQuadFiniteElement QuadrilateralFE;
public:
   CrouzeixRaviartFECollection() : FiniteElementCollection(1), SegmentFE(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "CrouzeixRaviart"; }

   int GetContType() const override { return DISCONTINUOUS; }
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
   LinearNonConf3DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "LinearNonConf3D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief First order Raviart-Thomas finite elements in 2D. This class is kept
    only for backward compatibility, consider using RT_FECollection instead. */
class RT0_2DFECollection : public FiniteElementCollection
{
private:
   const P0SegmentFiniteElement SegmentFE; // normal component on edge
   const RT0TriangleFiniteElement TriangleFE;
   const RT0QuadFiniteElement QuadrilateralFE;
public:
   RT0_2DFECollection() : FiniteElementCollection(1), SegmentFE(0) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RT0_2D"; }

   int GetContType() const override { return NORMAL; }
};

/** @brief Second order Raviart-Thomas finite elements in 2D. This class is kept
    only for backward compatibility, consider using RT_FECollection instead. */
class RT1_2DFECollection : public FiniteElementCollection
{
private:
   const P1SegmentFiniteElement SegmentFE; // normal component on edge
   const RT1TriangleFiniteElement TriangleFE;
   const RT1QuadFiniteElement QuadrilateralFE;
public:
   RT1_2DFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RT1_2D"; }

   int GetContType() const override { return NORMAL; }
};

/** @brief Third order Raviart-Thomas finite elements in 2D. This class is kept
    only for backward compatibility, consider using RT_FECollection instead. */
class RT2_2DFECollection : public FiniteElementCollection
{
private:
   const P2SegmentFiniteElement SegmentFE; // normal component on edge
   const RT2TriangleFiniteElement TriangleFE;
   const RT2QuadFiniteElement QuadrilateralFE;
public:
   RT2_2DFECollection() : FiniteElementCollection(3) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RT2_2D"; }

   int GetContType() const override { return NORMAL; }
};

/** @brief Piecewise-constant discontinuous finite elements in 2D. This class is
    kept only for backward compatibility, consider using L2_FECollection
    instead. */
class Const2DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P0QuadFiniteElement QuadrilateralFE;
public:
   Const2DFECollection() : FiniteElementCollection(0) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "Const2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-linear discontinuous finite elements in 2D. This class is
    kept only for backward compatibility, consider using L2_FECollection
    instead. */
class LinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;

public:
   LinearDiscont2DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "LinearDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/// Version of LinearDiscont2DFECollection with dofs in the Gaussian points.
class GaussLinearDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const CrouzeixRaviartFiniteElement TriangleFE;
   const GaussLinear2DFiniteElement TriangleFE;
   const GaussBiLinear2DFiniteElement QuadrilateralFE;

public:
   GaussLinearDiscont2DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "GaussLinearDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/// Linear (P1) finite elements on quadrilaterals.
class P1OnQuadFECollection : public FiniteElementCollection
{
private:
   const P1OnQuadFiniteElement QuadrilateralFE;
public:
   P1OnQuadFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "P1OnQuad"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-quadratic discontinuous finite elements in 2D. This class
    is kept only for backward compatibility, consider using L2_FECollection
    instead. */
class QuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Quad2DFiniteElement TriangleFE;
   const BiQuad2DFiniteElement QuadrilateralFE;

public:
   QuadraticDiscont2DFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "QuadraticDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/// Version of QuadraticDiscont2DFECollection with positive basis functions.
class QuadraticPosDiscont2DFECollection : public FiniteElementCollection
{
private:
   const BiQuadPos2DFiniteElement QuadrilateralFE;

public:
   QuadraticPosDiscont2DFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override
   { return NULL; }

   const char *Name() const override { return "QuadraticPosDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/// Version of QuadraticDiscont2DFECollection with dofs in the Gaussian points.
class GaussQuadraticDiscont2DFECollection : public FiniteElementCollection
{
private:
   // const Quad2DFiniteElement TriangleFE;
   const GaussQuad2DFiniteElement TriangleFE;
   const GaussBiQuad2DFiniteElement QuadrilateralFE;

public:
   GaussQuadraticDiscont2DFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "GaussQuadraticDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-cubic discontinuous finite elements in 2D. This class is
    kept only for backward compatibility, consider using L2_FECollection
    instead. */
class CubicDiscont2DFECollection : public FiniteElementCollection
{
private:
   const Cubic2DFiniteElement TriangleFE;
   const BiCubic2DFiniteElement QuadrilateralFE;

public:
   CubicDiscont2DFECollection() : FiniteElementCollection(3) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "CubicDiscont2D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-constant discontinuous finite elements in 3D. This class is
    kept only for backward compatibility, consider using L2_FECollection
    instead. */
class Const3DFECollection : public FiniteElementCollection
{
private:
   const P0TetFiniteElement TetrahedronFE;
   const P0HexFiniteElement ParallelepipedFE;
   const P0WdgFiniteElement WedgeFE;
   const P0PyrFiniteElement PyramidFE;

public:
   Const3DFECollection() : FiniteElementCollection(0) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "Const3D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-linear discontinuous finite elements in 3D. This class is
    kept only for backward compatibility, consider using L2_FECollection
    instead. */
class LinearDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Linear3DFiniteElement TetrahedronFE;
   const LinearPyramidFiniteElement PyramidFE;
   const LinearWedgeFiniteElement WedgeFE;
   const TriLinear3DFiniteElement ParallelepipedFE;

public:
   LinearDiscont3DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "LinearDiscont3D"; }

   int GetContType() const override { return DISCONTINUOUS; }
};

/** @brief Piecewise-quadratic discontinuous finite elements in 3D. This class
    is kept only for backward compatibility, consider using L2_FECollection
    instead. */
class QuadraticDiscont3DFECollection : public FiniteElementCollection
{
private:
   const Quadratic3DFiniteElement TetrahedronFE;
   const LagrangeHexFiniteElement ParallelepipedFE;

public:
   QuadraticDiscont3DFECollection()
      : FiniteElementCollection(2), ParallelepipedFE(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "QuadraticDiscont3D"; }

   int GetContType() const override { return DISCONTINUOUS; }
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
   RefinedLinearFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RefinedLinear"; }

   int GetContType() const override { return CONTINUOUS; }
};

/** @brief Lowest order Nedelec finite elements in 3D. This class is kept only
    for backward compatibility, consider using the new ND_FECollection
    instead. */
class ND1_3DFECollection : public FiniteElementCollection
{
private:
   const Nedelec1HexFiniteElement HexahedronFE;
   const Nedelec1TetFiniteElement TetrahedronFE;
   const Nedelec1WdgFiniteElement WedgeFE;
   const Nedelec1PyrFiniteElement PyramidFE;

public:
   ND1_3DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "ND1_3D"; }

   int GetContType() const override { return TANGENTIAL; }
};

/** @brief First order Raviart-Thomas finite elements in 3D. This class is kept
    only for backward compatibility, consider using RT_FECollection instead. */
class RT0_3DFECollection : public FiniteElementCollection
{
private:
   const P0TriangleFiniteElement TriangleFE;
   const P0QuadFiniteElement QuadrilateralFE;
   const RT0HexFiniteElement HexahedronFE;
   const RT0TetFiniteElement TetrahedronFE;
   const RT0WdgFiniteElement WedgeFE;
   const RT0PyrFiniteElement PyramidFE;
public:
   RT0_3DFECollection() : FiniteElementCollection(1) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RT0_3D"; }

   int GetContType() const override { return NORMAL; }
};

/** @brief Second order Raviart-Thomas finite elements in 3D. This class is kept
    only for backward compatibility, consider using RT_FECollection instead. */
class RT1_3DFECollection : public FiniteElementCollection
{
private:
   const Linear2DFiniteElement TriangleFE;
   const BiLinear2DFiniteElement QuadrilateralFE;
   const RT1HexFiniteElement HexahedronFE;
public:
   RT1_3DFECollection() : FiniteElementCollection(2) {}

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType) const override;

   int DofForGeometry(Geometry::Type GeomType) const override;

   const int *DofOrderForOrientation(Geometry::Type GeomType,
                                     int Or) const override;

   const char *Name() const override { return "RT1_3D"; }

   int GetContType() const override { return NORMAL; }
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

   const FiniteElement *
   FiniteElementForGeometry(Geometry::Type GeomType_) const override
   { return (GeomType == GeomType_) ? Local_Element : NULL; }

   int DofForGeometry(Geometry::Type GeomType_) const override
   { return (GeomType == GeomType_) ? Local_Element->GetDof() : 0; }

   const int *DofOrderForOrientation(Geometry::Type GeomType_,
                                     int Or) const override
   { return NULL; }

   const char *Name() const override { return d_name; }

   int GetContType() const override { return DISCONTINUOUS; }

   virtual ~Local_FECollection() { delete Local_Element; }
};

}

#endif
