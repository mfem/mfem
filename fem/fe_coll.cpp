// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#ifdef _WIN32
#define snprintf _snprintf_s
#endif

namespace mfem
{

using namespace std;

const FiniteElement *
FiniteElementCollection::FiniteElementForDim(int dim) const
{
   ErrorMode save_error_mode = error_mode;
   error_mode = RETURN_NULL;
   const FiniteElement *fe = nullptr;
   for (int g = Geometry::DimStart[dim]; g < Geometry::DimStart[dim+1]; g++)
   {
      fe = FiniteElementForGeometry((Geometry::Type)g);
      if (fe != nullptr) { break; }
   }
   error_mode = save_error_mode;
   return fe;
}

int FiniteElementCollection::GetRangeType(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetRangeType();
   }
   return FiniteElement::UNKNOWN_RANGE_TYPE;
}

int FiniteElementCollection::GetDerivRangeType(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetDerivRangeType();
   }
   return FiniteElement::UNKNOWN_RANGE_TYPE;
}

int FiniteElementCollection::GetMapType(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetMapType();
   }
   return FiniteElement::UNKNOWN_MAP_TYPE;
}

int FiniteElementCollection::GetDerivType(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetDerivType();
   }
   return FiniteElement::NONE;
}

int FiniteElementCollection::GetDerivMapType(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetDerivMapType();
   }
   return FiniteElement::UNKNOWN_MAP_TYPE;
}

int FiniteElementCollection::GetRangeDim(int dim) const
{
   const FiniteElement *fe = FiniteElementForDim(dim);
   if (fe)
   {
      return fe->GetRangeDim();
   }
   return 0;
}

int FiniteElementCollection::HasFaceDofs(Geometry::Type geom, int p) const
{
   switch (geom)
   {
      case Geometry::TETRAHEDRON:
         return GetNumDof(Geometry::TRIANGLE, p);
      case Geometry::CUBE:
         return GetNumDof(Geometry::SQUARE, p);
      case Geometry::PRISM:
      case Geometry::PYRAMID:
         return max(GetNumDof(Geometry::TRIANGLE, p),
                    GetNumDof(Geometry::SQUARE, p));
      default:
         MFEM_ABORT("unknown geometry type");
   }
   return 0;
}

FiniteElementCollection *FiniteElementCollection::GetTraceCollection() const
{
   MFEM_ABORT("this method is not implemented in this derived class!");
   return NULL;
}

FiniteElementCollection *FiniteElementCollection::New(const char *name)
{
   FiniteElementCollection *fec = NULL;

   if (!strcmp(name, "Linear"))
   {
      fec = new LinearFECollection;
   }
   else if (!strcmp(name, "Quadratic"))
   {
      fec = new QuadraticFECollection;
   }
   else if (!strcmp(name, "QuadraticPos"))
   {
      fec = new QuadraticPosFECollection;
   }
   else if (!strcmp(name, "Cubic"))
   {
      fec = new CubicFECollection;
   }
   else if (!strcmp(name, "Const3D"))
   {
      fec = new Const3DFECollection;
   }
   else if (!strcmp(name, "Const2D"))
   {
      fec = new Const2DFECollection;
   }
   else if (!strcmp(name, "LinearDiscont2D"))
   {
      fec = new LinearDiscont2DFECollection;
   }
   else if (!strcmp(name, "GaussLinearDiscont2D"))
   {
      fec = new GaussLinearDiscont2DFECollection;
   }
   else if (!strcmp(name, "P1OnQuad"))
   {
      fec = new P1OnQuadFECollection;
   }
   else if (!strcmp(name, "QuadraticDiscont2D"))
   {
      fec = new QuadraticDiscont2DFECollection;
   }
   else if (!strcmp(name, "QuadraticPosDiscont2D"))
   {
      fec = new QuadraticPosDiscont2DFECollection;
   }
   else if (!strcmp(name, "GaussQuadraticDiscont2D"))
   {
      fec = new GaussQuadraticDiscont2DFECollection;
   }
   else if (!strcmp(name, "CubicDiscont2D"))
   {
      fec = new CubicDiscont2DFECollection;
   }
   else if (!strcmp(name, "LinearDiscont3D"))
   {
      fec = new LinearDiscont3DFECollection;
   }
   else if (!strcmp(name, "QuadraticDiscont3D"))
   {
      fec = new QuadraticDiscont3DFECollection;
   }
   else if (!strcmp(name, "LinearNonConf3D"))
   {
      fec = new LinearNonConf3DFECollection;
   }
   else if (!strcmp(name, "CrouzeixRaviart"))
   {
      fec = new CrouzeixRaviartFECollection;
   }
   else if (!strcmp(name, "ND1_3D"))
   {
      fec = new ND1_3DFECollection;
   }
   else if (!strcmp(name, "RT0_2D"))
   {
      fec = new RT0_2DFECollection;
   }
   else if (!strcmp(name, "RT1_2D"))
   {
      fec = new RT1_2DFECollection;
   }
   else if (!strcmp(name, "RT2_2D"))
   {
      fec = new RT2_2DFECollection;
   }
   else if (!strcmp(name, "RT0_3D"))
   {
      fec = new RT0_3DFECollection;
   }
   else if (!strcmp(name, "RT1_3D"))
   {
      fec = new RT1_3DFECollection;
   }
   else if (!strncmp(name, "H1_Trace_", 9))
   {
      fec = new H1_Trace_FECollection(atoi(name + 13), atoi(name + 9));
   }
   else if (!strncmp(name, "H1_Trace@", 9))
   {
      fec = new H1_Trace_FECollection(atoi(name + 15), atoi(name + 11),
                                      BasisType::GetType(name[9]));
   }
   else if (!strncmp(name, "H1_", 3))
   {
      fec = new H1_FECollection(atoi(name + 7), atoi(name + 3));
   }
   else if (!strncmp(name, "H1Pos_Trace_", 12))
   {
      fec = new H1_Trace_FECollection(atoi(name + 16), atoi(name + 12),
                                      BasisType::Positive);
   }
   else if (!strncmp(name, "H1Pos_", 6))
   {
      fec = new H1Pos_FECollection(atoi(name + 10), atoi(name + 6));
   }
   else if (!strncmp(name, "H1Ser_", 6))
   {
      fec = new H1Ser_FECollection(atoi(name + 10), atoi(name + 6));
   }
   else if (!strncmp(name, "H1@", 3))
   {
      fec = new H1_FECollection(atoi(name + 9), atoi(name + 5),
                                BasisType::GetType(name[3]));
   }
   else if (!strncmp(name, "L2_T", 4))
      fec = new L2_FECollection(atoi(name + 10), atoi(name + 6),
                                atoi(name + 4));
   else if (!strncmp(name, "L2_", 3))
   {
      fec = new L2_FECollection(atoi(name + 7), atoi(name + 3));
   }
   else if (!strncmp(name, "L2Int_T", 7))
   {
      fec = new L2_FECollection(atoi(name + 13), atoi(name + 9),
                                atoi(name + 7), FiniteElement::INTEGRAL);
   }
   else if (!strncmp(name, "L2Int_", 6))
   {
      fec = new L2_FECollection(atoi(name + 10), atoi(name + 6),
                                BasisType::GaussLegendre,
                                FiniteElement::INTEGRAL);
   }
   else if (!strncmp(name, "RT_Trace_", 9))
   {
      fec = new RT_Trace_FECollection(atoi(name + 13), atoi(name + 9));
   }
   else if (!strncmp(name, "RT_ValTrace_", 12))
   {
      fec = new RT_Trace_FECollection(atoi(name + 16), atoi(name + 12),
                                      FiniteElement::VALUE);
   }
   else if (!strncmp(name, "RT_Trace@", 9))
   {
      fec = new RT_Trace_FECollection(atoi(name + 15), atoi(name + 11),
                                      FiniteElement::INTEGRAL,
                                      BasisType::GetType(name[9]));
   }
   else if (!strncmp(name, "RT_ValTrace@", 12))
   {
      fec = new RT_Trace_FECollection(atoi(name + 18), atoi(name + 14),
                                      FiniteElement::VALUE,
                                      BasisType::GetType(name[12]));
   }
   else if (!strncmp(name, "DG_Iface_", 9))
   {
      fec = new DG_Interface_FECollection(atoi(name + 13), atoi(name + 9));
   }
   else if (!strncmp(name, "DG_Iface@", 9))
   {
      fec = new DG_Interface_FECollection(atoi(name + 15), atoi(name + 11),
                                          FiniteElement::VALUE,
                                          BasisType::GetType(name[9]));
   }
   else if (!strncmp(name, "DG_IntIface_", 12))
   {
      fec = new DG_Interface_FECollection(atoi(name + 16), atoi(name + 12),
                                          FiniteElement::INTEGRAL);
   }
   else if (!strncmp(name, "DG_IntIface@", 12))
   {
      fec = new DG_Interface_FECollection(atoi(name + 18), atoi(name + 14),
                                          FiniteElement::INTEGRAL,
                                          BasisType::GetType(name[12]));
   }
   else if (!strncmp(name, "RT_R1D",6))
   {
      fec = new RT_R1D_FECollection(atoi(name+11),atoi(name + 7));
   }
   else if (!strncmp(name, "RT_R2D",6))
   {
      fec = new RT_R2D_FECollection(atoi(name+11),atoi(name + 7));
   }
   else if (!strncmp(name, "RT_", 3))
   {
      fec = new RT_FECollection(atoi(name + 7), atoi(name + 3));
   }
   else if (!strncmp(name, "RT@", 3))
   {
      fec = new RT_FECollection(atoi(name + 10), atoi(name + 6),
                                BasisType::GetType(name[3]),
                                BasisType::GetType(name[4]));
   }
   else if (!strncmp(name, "ND_Trace_", 9))
   {
      fec = new ND_Trace_FECollection(atoi(name + 13), atoi(name + 9));
   }
   else if (!strncmp(name, "ND_Trace@", 9))
   {
      fec = new ND_Trace_FECollection(atoi(name + 16), atoi(name + 12),
                                      BasisType::GetType(name[9]),
                                      BasisType::GetType(name[10]));
   }
   else if (!strncmp(name, "ND_R1D",6))
   {
      fec = new ND_R1D_FECollection(atoi(name+11),atoi(name + 7));
   }
   else if (!strncmp(name, "ND_R2D",6))
   {
      fec = new ND_R2D_FECollection(atoi(name+11),atoi(name + 7));
   }
   else if (!strncmp(name, "ND_", 3))
   {
      fec = new ND_FECollection(atoi(name + 7), atoi(name + 3));
   }
   else if (!strncmp(name, "ND@", 3))
   {
      fec = new ND_FECollection(atoi(name + 10), atoi(name + 6),
                                BasisType::GetType(name[3]),
                                BasisType::GetType(name[4]));
   }
   else if (!strncmp(name, "Local_", 6))
   {
      fec = new Local_FECollection(name + 6);
   }
   else if (!strncmp(name, "NURBS_HDiv", 10))
   {
      if (name[10] != '\0')
      {
         // "NURBS" + "number" --> fixed order nurbs collection
         fec = new NURBS_HDivFECollection(atoi(name + 10));
      }
      else
      {
         // "NURBS" --> variable order nurbs collection
         fec = new NURBS_HDivFECollection();
      }
   }
   else if (!strncmp(name, "NURBS_HCurl", 11))
   {
      if (name[11] != '\0')
      {
         // "NURBS" + "number" --> fixed order nurbs collection
         fec = new NURBS_HCurlFECollection(atoi(name + 11));
      }
      else
      {
         // "NURBS" --> variable order nurbs collection
         fec = new NURBS_HCurlFECollection();
      }
   }
   else if (!strncmp(name, "NURBS", 5))
   {
      if (name[5] != '\0')
      {
         // "NURBS" + "number" --> fixed order nurbs collection
         fec = new NURBSFECollection(atoi(name + 5));
      }
      else
      {
         // "NURBS" --> variable order nurbs collection
         fec = new NURBSFECollection();
      }
   }
   else
   {
      MFEM_ABORT("unknown FiniteElementCollection: " << name);
   }

   return fec;
}

FiniteElementCollection *FiniteElementCollection::Clone(int p) const
{
   // default implementation for collections that don't care about variable p
   MFEM_ABORT("Collection " << Name() << " does not support variable orders.");
   (void) p;
   return NULL;
}

void FiniteElementCollection::InitVarOrder(int p) const
{
   if (p >= var_orders.Size())
   {
      var_orders.SetSize(p+1, NULL);
   }
   var_orders[p] = Clone(p);
}

FiniteElementCollection::~FiniteElementCollection()
{
   for (int i = 0; i < var_orders.Size(); i++)
   {
      delete var_orders[i];
   }
}

template <Geometry::Type geom>
inline void FiniteElementCollection::GetNVE(int &nv, int &ne)
{
   typedef typename Geometry::Constants<geom> g_consts;

   nv = g_consts::NumVert;
   ne = g_consts::NumEdges;
}

template <Geometry::Type geom, typename v_t>
inline void FiniteElementCollection::
GetEdge(int &nv, v_t &v, int &ne, int &e, int &eo, const int edge_info)
{
   typedef typename Geometry::Constants<Geometry::SEGMENT> e_consts;
   typedef typename Geometry::Constants<geom> g_consts;

   nv = e_consts::NumVert;
   ne = 1;
   e = edge_info/64;
   eo = edge_info%64;
   MFEM_ASSERT(0 <= e && e < g_consts::NumEdges, "");
   MFEM_ASSERT(0 <= eo && eo < e_consts::NumOrient, "");
   v[0] = e_consts::Orient[eo][0];
   v[1] = e_consts::Orient[eo][1];
   v[0] = g_consts::Edges[e][v[0]];
   v[1] = g_consts::Edges[e][v[1]];
}

template <Geometry::Type geom, Geometry::Type f_geom,
          typename v_t, typename e_t, typename eo_t>
inline void FiniteElementCollection::
GetFace(int &nv, v_t &v, int &ne, e_t &e, eo_t &eo,
        int &nf, int &f, Geometry::Type &fg, int &fo, const int face_info)
{
   typedef typename Geometry::Constants<  geom> g_consts;
   typedef typename Geometry::Constants<f_geom> f_consts;

   nv = f_consts::NumVert;
   nf = 1;
   f = face_info/64;
   fg = f_geom;
   fo = face_info%64;
   MFEM_ASSERT(0 <= f && f < g_consts::NumFaces, "");
   MFEM_ASSERT(0 <= fo && fo < f_consts::NumOrient, "");
   for (int i = 0; i < f_consts::NumVert; i++)
   {
      v[i] = f_consts::Orient[fo][i];
      v[i] = g_consts::FaceVert[f][v[i]];
   }
   ne = f_consts::NumEdges;
   for (int i = 0; i < f_consts::NumEdges; i++)
   {
      int v0 = v[f_consts::Edges[i][0]];
      int v1 = v[f_consts::Edges[i][1]];
      int eor = 0;
      if (v0 > v1) { swap(v0, v1); eor = 1; }
      for (int j = g_consts::VertToVert::I[v0]; true; j++)
      {
         MFEM_ASSERT(j < g_consts::VertToVert::I[v0+1],
                     "internal error, edge not found");
         if (v1 == g_consts::VertToVert::J[j][0])
         {
            int en = g_consts::VertToVert::J[j][1];
            if (en < 0)
            {
               en = -1-en;
               eor = 1-eor;
            }
            e[i] = en;
            eo[i] = eor;
            break;
         }
      }
   }
}

void FiniteElementCollection::SubDofOrder(Geometry::Type Geom, int SDim,
                                          int Info,
                                          Array<int> &dofs) const
{
   // Info = 64 * SubIndex + SubOrientation
   MFEM_ASSERT(0 <= Geom && Geom < Geometry::NumGeom,
               "invalid Geom = " << Geom);
   MFEM_ASSERT(0 <= SDim && SDim <= Geometry::Dimension[Geom],
               "invalid SDim = " << SDim <<
               " for Geom = " << Geometry::Name[Geom]);

   const int nvd = DofForGeometry(Geometry::POINT);
   if (SDim == 0) // vertex
   {
      const int off = nvd*(Info/64);
      dofs.SetSize(nvd);
      for (int i = 0; i < nvd; i++)
      {
         dofs[i] = off + i;
      }
   }
   else
   {
      int v[4], e[4], eo[4], f[1], fo[1];
      int av = 0, nv = 0, ae = 0, ne = 0, nf = 0;
      Geometry::Type fg[1];

      switch (Geom)
      {
         case Geometry::SEGMENT:
         {
            GetNVE<Geometry::SEGMENT>(av, ae);
            GetEdge<Geometry::SEGMENT>(nv, v, ne, e[0], eo[0], Info);
            break;
         }

         case Geometry::TRIANGLE:
         {
            GetNVE<Geometry::TRIANGLE>(av, ae);
            switch (SDim)
            {
               case 1:
                  GetEdge<Geometry::TRIANGLE>(nv, v, ne, e[0], eo[0], Info);
                  break;
               case 2:
                  GetFace<Geometry::TRIANGLE,Geometry::TRIANGLE>(
                     nv, v, ne, e, eo, nf, f[0], fg[0], fo[0], Info);
                  break;
               default:
                  goto not_supp;
            }
            break;
         }

         case Geometry::SQUARE:
         {
            GetNVE<Geometry::SQUARE>(av, ae);
            switch (SDim)
            {
               case 1:
                  GetEdge<Geometry::SQUARE>(nv, v, ne, e[0], eo[0], Info);
                  break;
               case 2:
                  GetFace<Geometry::SQUARE,Geometry::SQUARE>(
                     nv, v, ne, e, eo, nf, f[0], fg[0], fo[0], Info);
                  break;
               default:
                  goto not_supp;
            }
            break;
         }

         case Geometry::TETRAHEDRON:
         {
            GetNVE<Geometry::TETRAHEDRON>(av, ae);
            switch (SDim)
            {
               case 1:
                  GetEdge<Geometry::TETRAHEDRON>(nv, v, ne, e[0], eo[0], Info);
                  break;
               case 2:
                  GetFace<Geometry::TETRAHEDRON,Geometry::TRIANGLE>(
                     nv, v, ne, e, eo, nf, f[0], fg[0], fo[0], Info);
                  break;
               default:
                  goto not_supp;
            }
            break;
         }

         case Geometry::CUBE:
         {
            GetNVE<Geometry::CUBE>(av, ae);
            switch (SDim)
            {
               case 1:
                  GetEdge<Geometry::CUBE>(nv, v, ne, e[0], eo[0], Info);
                  break;
               case 2:
                  GetFace<Geometry::CUBE,Geometry::SQUARE>(
                     nv, v, ne, e, eo, nf, f[0], fg[0], fo[0], Info);
                  break;
               default:
                  goto not_supp;
            }
            break;
         }

         default:
            MFEM_ABORT("invalid Geom = " << Geom);
      }

      int ned = (ne > 0) ? DofForGeometry(Geometry::SEGMENT) : 0;

      // add vertex dofs
      dofs.SetSize(nv*nvd+ne*ned);
      for (int i = 0; i < nv; i++)
      {
         for (int j = 0; j < nvd; j++)
         {
            dofs[i*nvd+j] = v[i]*nvd+j;
         }
      }
      int l_off = nv*nvd, g_off = av*nvd;

      // add edge dofs
      if (ned > 0)
      {
         for (int i = 0; i < ne; i++)
         {
            const int *ed = DofOrderForOrientation(Geometry::SEGMENT,
                                                   eo[i] ? -1 : 1);
            for (int j = 0; j < ned; j++)
            {
               dofs[l_off+i*ned+j] =
                  ed[j] >= 0 ?
                  g_off+e[i]*ned+ed[j] :
                  -1-(g_off+e[i]*ned+(-1-ed[j]));
            }
         }
         l_off += ne*ned;
         g_off += ae*ned;
      }

      // add face dofs
      if (nf > 0)
      {
         const int nfd = DofForGeometry(fg[0]); // assume same face geometry
         dofs.SetSize(dofs.Size()+nf*nfd);
         for (int i = 0; i < nf; i++)
         {
            const int *fd = DofOrderForOrientation(fg[i], fo[i]);
            for (int j = 0; j < nfd; j++)
            {
               dofs[l_off+i*nfd+j] =
                  fd[j] >= 0 ?
                  g_off+f[i]*nfd+fd[j] :
                  -1-(g_off+f[i]*nfd+(-1-fd[j]));
            }
         }
      }

      // add volume dofs ...
   }
   return;

not_supp:
   MFEM_ABORT("Geom = " << Geometry::Name[Geom] <<
              ", SDim = " << SDim << " is not supported");
}

const FiniteElement *
LinearFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return &PointFE;
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      case Geometry::PRISM:       return &WedgeFE;
      case Geometry::PYRAMID:     return &PyramidFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("LinearFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int LinearFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 1;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 0;
      case Geometry::PRISM:       return 0;
      case Geometry::PYRAMID:     return 0;
      default:
         mfem_error ("LinearFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *LinearFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return &PointFE;
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      case Geometry::PRISM:       return &WedgeFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("QuadraticFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int QuadraticFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 1;
      case Geometry::SEGMENT:     return 1;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 1;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 1;
      case Geometry::PRISM:       return 0;
      default:
         mfem_error ("QuadraticFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *QuadraticFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
QuadraticPosFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("QuadraticPosFECollection: unknown geometry type.");
   }
   return NULL; // Make some compilers happy
}

int QuadraticPosFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 1;
      case Geometry::SEGMENT:     return 1;
      case Geometry::SQUARE:      return 1;
      default:
         mfem_error ("QuadraticPosFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *QuadraticPosFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
CubicFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return &PointFE;
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      case Geometry::PRISM:       return &WedgeFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("CubicFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int CubicFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 1;
      case Geometry::SEGMENT:     return 2;
      case Geometry::TRIANGLE:    return 1;
      case Geometry::SQUARE:      return 4;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 8;
      case Geometry::PRISM:       return 2;
      default:
         mfem_error ("CubicFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *CubicFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                     int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      static int ind_pos[] = { 0, 1 };
      static int ind_neg[] = { 1, 0 };

      if (Or < 0)
      {
         return ind_neg;
      }
      return ind_pos;
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      static int indexes[] = { 0 };

      return indexes;
   }
   else if (GeomType == Geometry::SQUARE)
   {
      static int sq_ind[8][4] = {{0, 1, 2, 3}, {0, 2, 1, 3},
         {2, 0, 3, 1}, {1, 0, 3, 2},
         {3, 2, 1, 0}, {3, 1, 2, 0},
         {1, 3, 0, 2}, {2, 3, 0, 1}
      };
      return sq_ind[Or];
   }

   return NULL;
}


const FiniteElement *
CrouzeixRaviartFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("CrouzeixRaviartFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int CrouzeixRaviartFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 1;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      default:
         mfem_error ("CrouzeixRaviartFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *CrouzeixRaviartFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
RT0_2DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RT0_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT0_2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 1;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      default:
         mfem_error ("RT0_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int * RT0_2DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if (Or > 0)
   {
      return ind_pos;
   }
   return ind_neg;
}


const FiniteElement *
RT1_2DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RT1_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT1_2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 2;
      case Geometry::TRIANGLE:    return 2;
      case Geometry::SQUARE:      return 4;
      default:
         mfem_error ("RT1_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *RT1_2DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   static int ind_pos[] = {  0,  1 };
   static int ind_neg[] = { -2, -1 };

   if (Or > 0)
   {
      return ind_pos;
   }
   return ind_neg;
}

const FiniteElement *
RT2_2DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RT2_2DFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RT2_2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 3;
      case Geometry::TRIANGLE:    return 6;
      case Geometry::SQUARE:      return 12;
      default:
         mfem_error ("RT2_2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *RT2_2DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   static int ind_pos[] = { 0, 1, 2 };
   static int ind_neg[] = { -3, -2, -1 };

   if (Or > 0)
   {
      return ind_pos;
   }
   return ind_neg;
}


const FiniteElement *
Const2DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("Const2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int Const2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 1;
      case Geometry::SQUARE:      return 1;
      default:
         mfem_error ("Const2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *Const2DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   return NULL;
}


const FiniteElement *
LinearDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("LinearDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int LinearDiscont2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 3;
      case Geometry::SQUARE:      return 4;
      default:
         mfem_error ("LinearDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int * LinearDiscont2DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
GaussLinearDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("GaussLinearDiscont2DFECollection:"
                     " unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int GaussLinearDiscont2DFECollection::DofForGeometry(Geometry::Type GeomType)
const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 3;
      case Geometry::SQUARE:      return 4;
      default:
         mfem_error ("GaussLinearDiscont2DFECollection:"
                     " unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *GaussLinearDiscont2DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
P1OnQuadFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   if (GeomType != Geometry::SQUARE)
   {
      if (error_mode == RETURN_NULL) { return nullptr; }
      mfem_error ("P1OnQuadFECollection: unknown geometry type.");
   }
   return &QuadrilateralFE;
}

int P1OnQuadFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::SQUARE:      return 3;
      default:
         mfem_error ("P1OnQuadFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *P1OnQuadFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("QuadraticDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int QuadraticDiscont2DFECollection::DofForGeometry(Geometry::Type GeomType)
const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 6;
      case Geometry::SQUARE:      return 9;
      default:
         mfem_error ("QuadraticDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *QuadraticDiscont2DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticPosDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SQUARE:  return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("QuadraticPosDiscont2DFECollection: unknown geometry type.");
   }
   return NULL; // Make some compilers happy
}

int QuadraticPosDiscont2DFECollection::DofForGeometry(Geometry::Type GeomType)
const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::SQUARE:      return 9;
      default:
         mfem_error ("QuadraticPosDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}


const FiniteElement *
GaussQuadraticDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType)
const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("GaussQuadraticDiscont2DFECollection:"
                     " unknown geometry type.");
   }
   return &QuadrilateralFE; // Make some compilers happy
}

int GaussQuadraticDiscont2DFECollection::DofForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 6;
      case Geometry::SQUARE:      return 9;
      default:
         mfem_error ("GaussQuadraticDiscont2DFECollection:"
                     " unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *GaussQuadraticDiscont2DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
CubicDiscont2DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("CubicDiscont2DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int CubicDiscont2DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 10;
      case Geometry::SQUARE:      return 16;
      default:
         mfem_error ("CubicDiscont2DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *CubicDiscont2DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
LinearNonConf3DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("LinearNonConf3DFECollection: unknown geometry type.");
   }
   return &TriangleFE; // Make some compilers happy
}

int LinearNonConf3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 1;
      case Geometry::SQUARE:      return 1;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 0;
      default:
         mfem_error ("LinearNonConf3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *LinearNonConf3DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
Const3DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      case Geometry::PRISM:       return &WedgeFE;
      case Geometry::PYRAMID:     return &PyramidFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("Const3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int Const3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      case Geometry::TETRAHEDRON: return 1;
      case Geometry::CUBE:        return 1;
      case Geometry::PRISM:       return 1;
      case Geometry::PYRAMID:     return 1;
      default:
         mfem_error ("Const3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *Const3DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   return NULL;
}


const FiniteElement *
LinearDiscont3DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::PYRAMID:     return &PyramidFE;
      case Geometry::PRISM:       return &WedgeFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("LinearDiscont3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int LinearDiscont3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      case Geometry::TETRAHEDRON: return 4;
      case Geometry::PYRAMID:     return 5;
      case Geometry::PRISM:       return 6;
      case Geometry::CUBE:        return 8;
      default:
         mfem_error ("LinearDiscont3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *LinearDiscont3DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}


const FiniteElement *
QuadraticDiscont3DFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("QuadraticDiscont3DFECollection: unknown geometry type.");
   }
   return &TetrahedronFE; // Make some compilers happy
}

int QuadraticDiscont3DFECollection::DofForGeometry(Geometry::Type GeomType)
const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      case Geometry::TETRAHEDRON: return 10;
      case Geometry::CUBE:        return 27;
      default:
         mfem_error ("QuadraticDiscont3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *QuadraticDiscont3DFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   return NULL;
}

const FiniteElement *
RefinedLinearFECollection::FiniteElementForGeometry(
   Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return &PointFE;
      case Geometry::SEGMENT:     return &SegmentFE;
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::CUBE:        return &ParallelepipedFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RefinedLinearFECollection: unknown geometry type.");
   }
   return &SegmentFE; // Make some compilers happy
}

int RefinedLinearFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 1;
      case Geometry::SEGMENT:     return 1;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 1;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 1;
      default:
         mfem_error ("RefinedLinearFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *RefinedLinearFECollection::DofOrderForOrientation(
   Geometry::Type GeomType, int Or) const
{
   static int indexes[] = { 0 };

   return indexes;
}


const FiniteElement *
ND1_3DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::CUBE:        return &HexahedronFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::PRISM:       return &WedgeFE;
      case Geometry::PYRAMID:     return &PyramidFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("ND1_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int ND1_3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 1;
      case Geometry::TRIANGLE:    return 0;
      case Geometry::SQUARE:      return 0;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 0;
      case Geometry::PRISM:       return 0;
      case Geometry::PYRAMID:     return 0;
      default:
         mfem_error ("ND1_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *ND1_3DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if (Or > 0)
   {
      return ind_pos;
   }
   return ind_neg;
}


const FiniteElement *
RT0_3DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::CUBE:        return &HexahedronFE;
      case Geometry::TETRAHEDRON: return &TetrahedronFE;
      case Geometry::PRISM:       return &WedgeFE;
      case Geometry::PYRAMID:     return &PyramidFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RT0_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int RT0_3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 1;
      case Geometry::SQUARE:      return 1;
      case Geometry::TETRAHEDRON: return 0;
      case Geometry::CUBE:        return 0;
      case Geometry::PRISM:       return 0;
      case Geometry::PYRAMID:     return 0;
      default:
         mfem_error ("RT0_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *RT0_3DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   static int ind_pos[] = { 0 };
   static int ind_neg[] = { -1 };

   if ((GeomType == Geometry::TRIANGLE) || (GeomType == Geometry::SQUARE))
   {
      if (Or % 2 == 0)
      {
         return ind_pos;
      }
      return ind_neg;
   }
   return NULL;
}

const FiniteElement *
RT1_3DFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::TRIANGLE:    return &TriangleFE;
      case Geometry::SQUARE:      return &QuadrilateralFE;
      case Geometry::CUBE:        return &HexahedronFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("RT1_3DFECollection: unknown geometry type.");
   }
   return &HexahedronFE; // Make some compilers happy
}

int RT1_3DFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return 0;
      case Geometry::SEGMENT:     return 0;
      case Geometry::TRIANGLE:    return 2;
      case Geometry::SQUARE:      return 4;
      case Geometry::CUBE:        return 12;
      default:
         mfem_error ("RT1_3DFECollection: unknown geometry type.");
   }
   return 0; // Make some compilers happy
}

const int *RT1_3DFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                      int Or) const
{
   if (GeomType == Geometry::SQUARE)
   {
      static int sq_ind[8][4] =
      {
         {0, 1, 2, 3}, {-1, -3, -2, -4},
         {2, 0, 3, 1}, {-2, -1, -4, -3},
         {3, 2, 1, 0}, {-4, -2, -3, -1},
         {1, 3, 0, 2}, {-3, -4, -1, -2}
      };

      return sq_ind[Or];
   }
   else
   {
      return NULL;
   }
}


H1_FECollection::H1_FECollection(const int p, const int dim, const int btype,
                                 const int pyrtype)
   : FiniteElementCollection(p)
   , dim(dim)
{
   MFEM_VERIFY(p >= 1, "H1_FECollection requires order >= 1.");
   MFEM_VERIFY(dim >= 0 && dim <= 3, "H1_FECollection requires 0 <= dim <= 3.");

   const int pm1 = p - 1, pm2 = pm1 - 1, pm3 = pm2 - 1, pm4 = pm3 - 1;

   int pt_type = BasisType::GetQuadrature1D(btype);
   b_type = BasisType::Check(btype);
   switch (btype)
   {
      case BasisType::GaussLobatto:
      {
         snprintf(h1_name, 32, "H1_%dD_P%d", dim, p);
         break;
      }
      case BasisType::Positive:
      {
         snprintf(h1_name, 32, "H1Pos_%dD_P%d", dim, p);
         break;
      }
      case BasisType::Serendipity:
      {
         snprintf(h1_name, 32, "H1Ser_%dD_P%d", dim, p);
         break;
      }
      default:
      {
         MFEM_VERIFY(Quadrature1D::CheckClosed(pt_type) !=
                     Quadrature1D::Invalid,
                     "unsupported BasisType: " << BasisType::Name(btype));

         snprintf(h1_name, 32, "H1@%c_%dD_P%d",
                  (int)BasisType::GetChar(btype), dim, p);
      }
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      H1_dof[g] = 0;
      H1_Elements[g] = NULL;
   }
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }
   for (int i = 0; i < 6; i++)
   {
      TriDofOrd[i] = NULL;
   }
   for (int i = 0; i < 8; i++)
   {
      QuadDofOrd[i] = NULL;
   }
   for (int i = 0; i < 24; i++)
   {
      TetDofOrd[i] = NULL;
   }

   H1_dof[Geometry::POINT] = 1;
   H1_Elements[Geometry::POINT] = new PointFiniteElement;

   if (dim >= 1)
   {
      H1_dof[Geometry::SEGMENT] = pm1;
      if (b_type == BasisType::Positive)
      {
         H1_Elements[Geometry::SEGMENT] = new H1Pos_SegmentElement(p);
      }
      else
      {
         H1_Elements[Geometry::SEGMENT] = new H1_SegmentElement(p, btype);
      }

      SegDofOrd[0] = (pm1 > 0) ? new int[2*pm1] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + pm1;
      for (int i = 0; i < pm1; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = pm2 - i;
      }
   }

   if (dim >= 2)
   {
      H1_dof[Geometry::TRIANGLE] = (pm1*pm2)/2;
      H1_dof[Geometry::SQUARE] = pm1*pm1;
      if (b_type == BasisType::Positive)
      {
         H1_Elements[Geometry::TRIANGLE] = new H1Pos_TriangleElement(p);
         H1_Elements[Geometry::SQUARE] = new H1Pos_QuadrilateralElement(p);
      }
      else if (b_type == BasisType::Serendipity)
      {
         // Note: in fe_coll.hpp the DofForGeometry(Geometry::Type) method
         // returns H1_dof[GeomType], so we need to fix the value of H1_dof here
         // for the serendipity case.

         // formula for number of interior serendipity DoFs (when p>1)
         H1_dof[Geometry::SQUARE] = (pm3*pm2)/2;
         H1_Elements[Geometry::SQUARE] = new H1Ser_QuadrilateralElement(p);
         // allows for mixed tri/quad meshes
         H1_Elements[Geometry::TRIANGLE] = new H1Pos_TriangleElement(p);
      }
      else
      {
         H1_Elements[Geometry::TRIANGLE] = new H1_TriangleElement(p, btype);
         H1_Elements[Geometry::SQUARE] = new H1_QuadrilateralElement(p, btype);
      }

      const int &TriDof = H1_dof[Geometry::TRIANGLE];
      const int &QuadDof = H1_dof[Geometry::SQUARE];
      TriDofOrd[0] = (TriDof > 0) ? new int[6*TriDof] : nullptr;
      for (int i = 1; i < 6; i++)
      {
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      }
      // see Mesh::GetTriOrientation in mesh/mesh.cpp
      for (int j = 0; j < pm2; j++)
      {
         for (int i = 0; i + j < pm2; i++)
         {
            int o = TriDof - ((pm1 - j)*(pm2 - j))/2 + i;
            int k = pm3 - j - i;
            TriDofOrd[0][o] = o;  // (0,1,2)
            TriDofOrd[1][o] = TriDof - ((pm1-j)*(pm2-j))/2 + k;  // (1,0,2)
            TriDofOrd[2][o] = TriDof - ((pm1-i)*(pm2-i))/2 + k;  // (2,0,1)
            TriDofOrd[3][o] = TriDof - ((pm1-k)*(pm2-k))/2 + i;  // (2,1,0)
            TriDofOrd[4][o] = TriDof - ((pm1-k)*(pm2-k))/2 + j;  // (1,2,0)
            TriDofOrd[5][o] = TriDof - ((pm1-i)*(pm2-i))/2 + j;  // (0,2,1)
         }
      }

      QuadDofOrd[0] = (QuadDof > 0) ? new int[8*QuadDof] : nullptr;
      for (int i = 1; i < 8; i++)
      {
         QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
      }

      // For serendipity order >=4, the QuadDofOrd array must be re-defined. We
      // do this by computing the corresponding tensor product QuadDofOrd array
      // or two orders less, which contains enough DoFs for their serendipity
      // basis. This could be optimized.
      if (b_type == BasisType::Serendipity)
      {
         if (p < 4)
         {
            // no face dofs --> don't need to adjust QuadDofOrd
         }
         else  // p >= 4 --> have face dofs
         {
            // Exactly the same as tensor product case, but with all orders
            // reduced by 2 e.g. in case p=5 it builds a 2x2 array, even though
            // there are only 3 serendipity dofs.
            // In the tensor product case, the i and j index tensor directions,
            // and o index from 0 to (pm1)^2,

            for (int j = 0; j < pm3; j++)   // pm3 instead of pm1, etc
            {
               for (int i = 0; i < pm3; i++)
               {
                  int o = i + j*pm3;
                  QuadDofOrd[0][o] = i + j*pm3;  // (0,1,2,3)
                  QuadDofOrd[1][o] = j + i*pm3;  // (0,3,2,1)
                  QuadDofOrd[2][o] = j + (pm4 - i)*pm3;  // (1,2,3,0)
                  QuadDofOrd[3][o] = (pm4 - i) + j*pm3;  // (1,0,3,2)
                  QuadDofOrd[4][o] = (pm4 - i) + (pm4 - j)*pm3;  // (2,3,0,1)
                  QuadDofOrd[5][o] = (pm4 - j) + (pm4 - i)*pm3;  // (2,1,0,3)
                  QuadDofOrd[6][o] = (pm4 - j) + i*pm3;  // (3,0,1,2)
                  QuadDofOrd[7][o] = i + (pm4 - j)*pm3;  // (3,2,1,0)
               }
            }

         }
      }
      else // not serendipity
      {
         for (int j = 0; j < pm1; j++)
         {
            for (int i = 0; i < pm1; i++)
            {
               int o = i + j*pm1;
               QuadDofOrd[0][o] = i + j*pm1;  // (0,1,2,3)
               QuadDofOrd[1][o] = j + i*pm1;  // (0,3,2,1)
               QuadDofOrd[2][o] = j + (pm2 - i)*pm1;  // (1,2,3,0)
               QuadDofOrd[3][o] = (pm2 - i) + j*pm1;  // (1,0,3,2)
               QuadDofOrd[4][o] = (pm2 - i) + (pm2 - j)*pm1;  // (2,3,0,1)
               QuadDofOrd[5][o] = (pm2 - j) + (pm2 - i)*pm1;  // (2,1,0,3)
               QuadDofOrd[6][o] = (pm2 - j) + i*pm1;  // (3,0,1,2)
               QuadDofOrd[7][o] = i + (pm2 - j)*pm1;  // (3,2,1,0)
            }
         }
      }

      if (dim >= 3)
      {
         H1_dof[Geometry::TETRAHEDRON] = (TriDof*pm3)/3;
         H1_dof[Geometry::CUBE] = QuadDof*pm1;
         H1_dof[Geometry::PRISM] = TriDof*pm1;
         if (pyrtype == 0 || b_type == BasisType::Positive)
         {
            H1_dof[Geometry::PYRAMID] = pm2*pm1*(2*p-3)/6; // Bergot (JSC)
         }
         else if (pyrtype == 1)
         {
            H1_dof[Geometry::PYRAMID] = pm1*pm1*pm1; // Fuentes
         }
         else
         {
            H1_dof[Geometry::PYRAMID] = (p-1)*(p-2)/2;
         }
         if (b_type == BasisType::Positive)
         {
            H1_Elements[Geometry::TETRAHEDRON] = new H1Pos_TetrahedronElement(p);
            H1_Elements[Geometry::CUBE] = new H1Pos_HexahedronElement(p);
            H1_Elements[Geometry::PRISM] = new H1Pos_WedgeElement(p);
            H1_Elements[Geometry::PYRAMID] = new H1Pos_PyramidElement(p);
         }
         else
         {
            H1_Elements[Geometry::TETRAHEDRON] =
               new H1_TetrahedronElement(p, btype);
            H1_Elements[Geometry::CUBE] = new H1_HexahedronElement(p, btype);
            H1_Elements[Geometry::PRISM] = new H1_WedgeElement(p, btype);
            if (pyrtype == 0)
            {
               H1_Elements[Geometry::PYRAMID] = new H1_BergotPyramidElement(p, btype);
            }
            else
            {
               H1_Elements[Geometry::PYRAMID] = new H1_FuentesPyramidElement(p, btype);
            }
         }

         const int &TetDof = H1_dof[Geometry::TETRAHEDRON];
         TetDofOrd[0] = (TetDof > 0) ? new int[24*TetDof] : nullptr;
         for (int i = 1; i < 24; i++)
         {
            TetDofOrd[i] = TetDofOrd[i-1] + TetDof;
         }
         // see Mesh::GetTetOrientation in mesh/mesh.cpp
         for (int k = 0; k < pm3; k++)
         {
            for (int j = 0; j + k < pm3; j++)
            {
               for (int i = 0; i + j + k < pm3; i++)
               {
                  int l = pm4 - k - j - i;
                  int o   = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (j * (2 * p - 5 - j - 2 * k)) / 2 + i;
                  int o1  = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (k * (2 * p - 5 - k - 2 * j)) / 2 + i;
                  int o2  = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (k * (2 * p - 5 - k - 2 * i)) / 2 + j;
                  int o3  = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (i * (2 * p - 5 - i - 2 * k)) / 2 + j;
                  int o4  = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (i * (2 * p - 5 - i - 2 * j)) / 2 + k;
                  int o5  = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (j * (2 * p - 5 - j - 2 * i)) / 2 + k;
                  int o6  = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (l * (2 * p - 5 - l - 2 * k)) / 2 + j;
                  int o7  = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (k * (2 * p - 5 - k - 2 * l)) / 2 + j;
                  int o8  = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (j * (2 * p - 5 - j - 2 * l)) / 2 + k;
                  int o9  = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (l * (2 * p - 5 - l - 2 * j)) / 2 + k;
                  int o10 = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (k * (2 * p - 5 - k - 2 * j)) / 2 + l;
                  int o11 = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (j * (2 * p - 5 - j - 2 * k)) / 2 + l;
                  int o12 = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (l * (2 * p - 5 - l - 2 * i)) / 2 + k;
                  int o13 = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (i * (2 * p - 5 - i - 2 * l)) / 2 + k;
                  int o14 = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (i * (2 * p - 5 - i - 2 * k)) / 2 + l;
                  int o15 = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (k * (2 * p - 5 - k - 2 * i)) / 2 + l;
                  int o16 = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (k * (2 * p - 5 - k - 2 * l)) / 2 + i;
                  int o17 = TetDof - ((pm1 - k) * (pm2 - k) * (pm3 - k)) / 6
                            + (l * (2 * p - 5 - l - 2 * k)) / 2 + i;
                  int o18 = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (j * (2 * p - 5 - j - 2 * i)) / 2 + l;
                  int o19 = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (i * (2 * p - 5 - i - 2 * j)) / 2 + l;
                  int o20 = TetDof - ((pm1 - j) * (pm2 - j) * (pm3 - j)) / 6
                            + (l * (2 * p - 5 - l - 2 * j)) / 2 + i;
                  int o21 = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (j * (2 * p - 5 - j - 2 * l)) / 2 + i;
                  int o22 = TetDof - ((pm1 - l) * (pm2 - l) * (pm3 - l)) / 6
                            + (i * (2 * p - 5 - i - 2 * l)) / 2 + j;
                  int o23 = TetDof - ((pm1 - i) * (pm2 - i) * (pm3 - i)) / 6
                            + (l * (2 * p - 5 - l - 2 * i)) / 2 + j;
                  TetDofOrd[ 0][o] = o;   // (0,1,2,3)
                  TetDofOrd[ 1][o] = o1;  // (0,1,3,2)
                  TetDofOrd[ 2][o] = o2;  // (0,2,3,1)
                  TetDofOrd[ 3][o] = o3;  // (0,2,1,3)
                  TetDofOrd[ 4][o] = o4;  // (0,3,1,2)
                  TetDofOrd[ 5][o] = o5;  // (0,3,2,1)
                  TetDofOrd[ 6][o] = o6;  // (1,2,0,3)
                  TetDofOrd[ 7][o] = o7;  // (1,2,3,0)
                  TetDofOrd[ 8][o] = o8;  // (1,3,2,0)
                  TetDofOrd[ 9][o] = o9;  // (1,3,0,2)
                  TetDofOrd[10][o] = o10; // (1,0,3,2)
                  TetDofOrd[11][o] = o11; // (1,0,2,3)
                  TetDofOrd[12][o] = o12; // (2,3,0,1)
                  TetDofOrd[13][o] = o13; // (2,3,1,0)
                  TetDofOrd[14][o] = o14; // (2,0,1,3)
                  TetDofOrd[15][o] = o15; // (2,0,3,1)
                  TetDofOrd[16][o] = o16; // (2,1,3,0)
                  TetDofOrd[17][o] = o17; // (2,1,0,3)
                  TetDofOrd[18][o] = o18; // (3,0,2,1)
                  TetDofOrd[19][o] = o19; // (3,0,1,2)
                  TetDofOrd[20][o] = o20; // (3,1,0,2)
                  TetDofOrd[21][o] = o21; // (3,1,2,0)
                  TetDofOrd[22][o] = o22; // (3,2,1,0)
                  TetDofOrd[23][o] = o23; // (3,2,0,1)
               }
            }
         }
      }
   }
}

const FiniteElement *
H1_FECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   return H1_Elements[GeomType];
}

const int *H1_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                   int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   else if (GeomType == Geometry::TETRAHEDRON)
   {
      return TetDofOrd[Or%24];
   }
   return NULL;
}

FiniteElementCollection *H1_FECollection::GetTraceCollection() const
{
   int tr_p = H1_dof[Geometry::SEGMENT] + 1;
   int tr_dim = -1;
   if (!strncmp(h1_name, "H1_", 3))
   {
      tr_dim = atoi(h1_name + 3);
   }
   else if (!strncmp(h1_name, "H1Pos_", 6))
   {
      tr_dim = atoi(h1_name + 6);
   }
   else if (!strncmp(h1_name, "H1@", 3))
   {
      tr_dim = atoi(h1_name + 5);
   }
   return (dim < 0) ? NULL : new H1_Trace_FECollection(tr_p, tr_dim, b_type);
}

const int *H1_FECollection::GetDofMap(Geometry::Type GeomType) const
{
   const int *dof_map = NULL;
   const FiniteElement *fe = H1_Elements[GeomType];
   const NodalFiniteElement *nodal_fe =
      dynamic_cast<const NodalFiniteElement*>(fe);
   if (nodal_fe)
   {
      dof_map = nodal_fe->GetLexicographicOrdering().GetData();
   }
   else
   {
      MFEM_ABORT("Geometry type " << Geometry::Name[GeomType] << " is not "
                 "implemented");
   }
   return dof_map;
}

const int *H1_FECollection::GetDofMap(Geometry::Type GeomType, int p) const
{
   if (p == base_p) { return GetDofMap(GeomType); }
   if (p >= var_orders.Size() || !var_orders[p]) { InitVarOrder(p); }
   return ((H1_FECollection*) var_orders[p])->GetDofMap(GeomType);
}

H1_FECollection::~H1_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   delete [] TetDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete H1_Elements[g];
   }
}


H1_Trace_FECollection::H1_Trace_FECollection(const int p, const int dim,
                                             const int btype)
   : H1_FECollection(p, dim-1, btype)
{
   if (btype == BasisType::GaussLobatto)
   {
      snprintf(h1_name, 32, "H1_Trace_%dD_P%d", dim, p);
   }
   else if (btype == BasisType::Positive)
   {
      snprintf(h1_name, 32, "H1Pos_Trace_%dD_P%d", dim, p);
   }
   else // base class checks that type is closed
   {
      snprintf(h1_name, 32, "H1_Trace@%c_%dD_P%d",
               (int)BasisType::GetChar(btype), dim, p);
   }
}


L2_FECollection::L2_FECollection(const int p, const int dim, const int btype,
                                 const int map_type, const int pyr_type)
   : FiniteElementCollection(p)
   , dim(dim)
   , m_type(map_type)
{
   MFEM_VERIFY(p >= 0, "L2_FECollection requires order >= 0.");

   b_type = BasisType::Check(btype);
   const char *prefix = NULL;
   switch (map_type)
   {
      case FiniteElement::VALUE:    prefix = "L2";    break;
      case FiniteElement::INTEGRAL: prefix = "L2Int"; break;
      default:
         MFEM_ABORT("invalid map_type: " << map_type);
   }
   switch (btype)
   {
      case BasisType::GaussLegendre:
         snprintf(d_name, 32, "%s_%dD_P%d", prefix, dim, p);
         break;
      default:
         snprintf(d_name, 32, "%s_T%d_%dD_P%d", prefix, btype, dim, p);
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      L2_Elements[g] = NULL;
      Tr_Elements[g] = NULL;
   }
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }
   for (int i = 0; i < 6; i++)
   {
      TriDofOrd[i] = NULL;
   }
   for (int i = 0; i < 24; i++)
   {
      TetDofOrd[i] = NULL;
   }
   OtherDofOrd = NULL;

   if (dim == 0)
   {
      L2_Elements[Geometry::POINT] = new PointFiniteElement;
   }
   else if (dim == 1)
   {
      if (b_type == BasisType::Positive)
      {
         L2_Elements[Geometry::SEGMENT] = new L2Pos_SegmentElement(p);
      }
      else
      {
         L2_Elements[Geometry::SEGMENT] = new L2_SegmentElement(p, btype);
      }
      L2_Elements[Geometry::SEGMENT]->SetMapType(map_type);

      Tr_Elements[Geometry::POINT] = new PointFiniteElement;
      // No need to set the map_type for Tr_Elements.

      const int pp1 = p + 1;
      SegDofOrd[0] = (pp1 > 0) ? new int[2*pp1] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + pp1;
      for (int i = 0; i <= p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = p - i;
      }
   }
   else if (dim == 2)
   {
      if (b_type == BasisType::Positive)
      {
         L2_Elements[Geometry::TRIANGLE] = new L2Pos_TriangleElement(p);
         L2_Elements[Geometry::SQUARE] = new L2Pos_QuadrilateralElement(p);
      }
      else
      {
         L2_Elements[Geometry::TRIANGLE] = new L2_TriangleElement(p, btype);
         L2_Elements[Geometry::SQUARE] = new L2_QuadrilateralElement(p, btype);
      }
      L2_Elements[Geometry::TRIANGLE]->SetMapType(map_type);
      L2_Elements[Geometry::SQUARE]->SetMapType(map_type);
      // Trace element use the default Gauss-Legendre nodal points for positive basis
      if (b_type == BasisType::Positive)
      {
         Tr_Elements[Geometry::SEGMENT] = new L2Pos_SegmentElement(p);
      }
      else
      {
         Tr_Elements[Geometry::SEGMENT] = new L2_SegmentElement(p, btype);
      }

      const int TriDof = L2_Elements[Geometry::TRIANGLE]->GetDof();
      TriDofOrd[0] = (TriDof > 0) ? new int[6*TriDof] : nullptr;
      for (int i = 1; i < 6; i++)
      {
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      }
      const int pp1 = p + 1, pp2 = pp1 + 1;
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i + j <= p; i++)
         {
            int o = TriDof - ((pp2 - j)*(pp1 - j))/2 + i;
            int k = p - j - i;
            TriDofOrd[0][o] = o;  // (0,1,2)
            TriDofOrd[1][o] = TriDof - ((pp2-j)*(pp1-j))/2 + k;  // (1,0,2)
            TriDofOrd[2][o] = TriDof - ((pp2-i)*(pp1-i))/2 + k;  // (2,0,1)
            TriDofOrd[3][o] = TriDof - ((pp2-k)*(pp1-k))/2 + i;  // (2,1,0)
            TriDofOrd[4][o] = TriDof - ((pp2-k)*(pp1-k))/2 + j;  // (1,2,0)
            TriDofOrd[5][o] = TriDof - ((pp2-i)*(pp1-i))/2 + j;  // (0,2,1)
         }
      }
      const int QuadDof = L2_Elements[Geometry::SQUARE]->GetDof();
      OtherDofOrd = (QuadDof > 0) ? new int[QuadDof] : nullptr;
      for (int j = 0; j < QuadDof; j++)
      {
         OtherDofOrd[j] = j; // for Or == 0
      }
   }
   else if (dim == 3)
   {
      if (b_type == BasisType::Positive)
      {
         L2_Elements[Geometry::TETRAHEDRON] = new L2Pos_TetrahedronElement(p);
         L2_Elements[Geometry::CUBE] = new L2Pos_HexahedronElement(p);
         L2_Elements[Geometry::PRISM] = new L2Pos_WedgeElement(p);
         L2_Elements[Geometry::PYRAMID] = new L2Pos_PyramidElement(p);
      }
      else
      {
         L2_Elements[Geometry::TETRAHEDRON] =
            new L2_TetrahedronElement(p, btype);
         L2_Elements[Geometry::CUBE] = new L2_HexahedronElement(p, btype);
         L2_Elements[Geometry::PRISM] = new L2_WedgeElement(p, btype);
         if (pyr_type == 0)
         {
            L2_Elements[Geometry::PYRAMID] = new L2_BergotPyramidElement(p, btype);
         }
         else
         {
            L2_Elements[Geometry::PYRAMID] = new L2_FuentesPyramidElement(p, btype);
         }
      }

      L2_Elements[Geometry::TETRAHEDRON]->SetMapType(map_type);
      L2_Elements[Geometry::CUBE]->SetMapType(map_type);
      L2_Elements[Geometry::PRISM]->SetMapType(map_type);
      L2_Elements[Geometry::PYRAMID]->SetMapType(map_type);
      if (b_type != BasisType::Positive)
      {
         L2_Elements[Geometry::PYRAMID]->SetMapType(map_type);
      }
      // Trace element use the default Gauss-Legendre nodal points for positive basis
      if (b_type == BasisType::Positive)
      {
         Tr_Elements[Geometry::TRIANGLE] = new L2Pos_TriangleElement(p);
         Tr_Elements[Geometry::SQUARE] = new L2Pos_QuadrilateralElement(p);
      }
      else
      {
         Tr_Elements[Geometry::TRIANGLE] = new L2_TriangleElement(p, btype);
         Tr_Elements[Geometry::SQUARE] = new L2_QuadrilateralElement(p, btype);
      }

      const int TetDof = L2_Elements[Geometry::TETRAHEDRON]->GetDof();
      const int HexDof = L2_Elements[Geometry::CUBE]->GetDof();
      const int PriDof = L2_Elements[Geometry::PRISM]->GetDof();
      const int PyrDof = L2_Elements[Geometry::PYRAMID]->GetDof();
      const int MaxDof = std::max(std::max(TetDof, PyrDof),
                                  std::max(PriDof, HexDof));

      TetDofOrd[0] = (TetDof > 0) ? new int[24*TetDof] : nullptr;
      for (int i = 1; i < 24; i++)
      {
         TetDofOrd[i] = TetDofOrd[i-1] + TetDof;
      }
      // see Mesh::GetTetOrientation in mesh/mesh.cpp
      const int pp1 = p + 1, pp2 = pp1 + 1, pp3 = pp2 + 1;
      for (int k = 0; k <= p; k++)
      {
         for (int j = 0; j + k <= p; j++)
         {
            for (int i = 0; i + j + k <= p; i++)
            {
               int l = p - k - j - i;
               int o   = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (j * (2 * p + 3 - j - 2 * k)) / 2 + i;
               int o1  = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (k * (2 * p + 3 - k - 2 * j)) / 2 + i;
               int o2  = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (k * (2 * p + 3 - k - 2 * i)) / 2 + j;
               int o3  = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (i * (2 * p + 3 - i - 2 * k)) / 2 + j;
               int o4  = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (i * (2 * p + 3 - i - 2 * j)) / 2 + k;
               int o5  = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (j * (2 * p + 3 - j - 2 * i)) / 2 + k;
               int o6  = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (l * (2 * p + 3 - l - 2 * k)) / 2 + j;
               int o7  = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (k * (2 * p + 3 - k - 2 * l)) / 2 + j;
               int o8  = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (j * (2 * p + 3 - j - 2 * l)) / 2 + k;
               int o9  = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (l * (2 * p + 3 - l - 2 * j)) / 2 + k;
               int o10 = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (k * (2 * p + 3 - k - 2 * j)) / 2 + l;
               int o11 = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (j * (2 * p + 3 - j - 2 * k)) / 2 + l;
               int o12 = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (l * (2 * p + 3 - l - 2 * i)) / 2 + k;
               int o13 = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (i * (2 * p + 3 - i - 2 * l)) / 2 + k;
               int o14 = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (i * (2 * p + 3 - i - 2 * k)) / 2 + l;
               int o15 = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (k * (2 * p + 3 - k - 2 * i)) / 2 + l;
               int o16 = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (k * (2 * p + 3 - k - 2 * l)) / 2 + i;
               int o17 = TetDof - ((pp1 - k) * (pp2 - k) * (pp3 - k)) / 6
                         + (l * (2 * p + 3 - l - 2 * k)) / 2 + i;
               int o18 = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (j * (2 * p + 3 - j - 2 * i)) / 2 + l;
               int o19 = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (i * (2 * p + 3 - i - 2 * j)) / 2 + l;
               int o20 = TetDof - ((pp1 - j) * (pp2 - j) * (pp3 - j)) / 6
                         + (l * (2 * p + 3 - l - 2 * j)) / 2 + i;
               int o21 = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (j * (2 * p + 3 - j - 2 * l)) / 2 + i;
               int o22 = TetDof - ((pp1 - l) * (pp2 - l) * (pp3 - l)) / 6
                         + (i * (2 * p + 3 - i - 2 * l)) / 2 + j;
               int o23 = TetDof - ((pp1 - i) * (pp2 - i) * (pp3 - i)) / 6
                         + (l * (2 * p + 3 - l - 2 * i)) / 2 + j;
               TetDofOrd[ 0][o] = o;   // (0,1,2,3)
               TetDofOrd[ 1][o] = o1;  // (0,1,3,2)
               TetDofOrd[ 2][o] = o2;  // (0,2,3,1)
               TetDofOrd[ 3][o] = o3;  // (0,2,1,3)
               TetDofOrd[ 4][o] = o4;  // (0,3,1,2)
               TetDofOrd[ 5][o] = o5;  // (0,3,2,1)
               TetDofOrd[ 6][o] = o6;  // (1,2,0,3)
               TetDofOrd[ 7][o] = o7;  // (1,2,3,0)
               TetDofOrd[ 8][o] = o8;  // (1,3,2,0)
               TetDofOrd[ 9][o] = o9;  // (1,3,0,2)
               TetDofOrd[10][o] = o10; // (1,0,3,2)
               TetDofOrd[11][o] = o11; // (1,0,2,3)
               TetDofOrd[12][o] = o12; // (2,3,0,1)
               TetDofOrd[13][o] = o13; // (2,3,1,0)
               TetDofOrd[14][o] = o14; // (2,0,1,3)
               TetDofOrd[15][o] = o15; // (2,0,3,1)
               TetDofOrd[16][o] = o16; // (2,1,3,0)
               TetDofOrd[17][o] = o17; // (2,1,0,3)
               TetDofOrd[18][o] = o18; // (3,0,2,1)
               TetDofOrd[19][o] = o19; // (3,0,1,2)
               TetDofOrd[20][o] = o20; // (3,1,0,2)
               TetDofOrd[21][o] = o21; // (3,1,2,0)
               TetDofOrd[22][o] = o22; // (3,2,1,0)
               TetDofOrd[23][o] = o23; // (3,2,0,1)
            }
         }
      }
      OtherDofOrd = (MaxDof > 0) ? new int[MaxDof] : nullptr;
      for (int j = 0; j < MaxDof; j++)
      {
         OtherDofOrd[j] = j; // for Or == 0
      }
   }
   else
   {
      mfem::err << "L2_FECollection::L2_FECollection : dim = "
                << dim << endl;
      mfem_error();
   }
}

const FiniteElement *
L2_FECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   return L2_Elements[GeomType];
}

const int *L2_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                   int Or) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:
         return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];

      case Geometry::TRIANGLE:
         return TriDofOrd[Or%6];

      case Geometry::TETRAHEDRON:
         return TetDofOrd[Or%24];

      default:
         return (Or == 0) ? OtherDofOrd : NULL;
   }
}

L2_FECollection::~L2_FECollection()
{
   delete [] OtherDofOrd;
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] TetDofOrd[0];
   for (int i = 0; i < Geometry::NumGeom; i++)
   {
      delete L2_Elements[i];
      delete Tr_Elements[i];
   }
}


RT_FECollection::RT_FECollection(const int order, const int dim,
                                 const int cb_type, const int ob_type)
   : FiniteElementCollection(order + 1)
   , dim(dim)
   , cb_type(cb_type)
   , ob_type(ob_type)
{
   int p = order;
   MFEM_VERIFY(p >= 0, "RT_FECollection requires order >= 0.");

   int cp_type = BasisType::GetQuadrature1D(cb_type);
   int op_type = BasisType::GetQuadrature1D(ob_type);

   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type); // this may abort
      MFEM_ABORT("unknown closed BasisType: " << cb_name);
   }
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type); // this may abort
      MFEM_ABORT("unknown open BasisType: " << ob_name);
   }

   InitFaces(p, dim, FiniteElement::INTEGRAL, true);

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(rt_name, 32, "RT_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(rt_name, 32, "RT@%c%c_%dD_P%d", (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   const int pp1 = p + 1;
   if (dim == 2)
   {
      // TODO: cb_type, ob_type for triangles
      RT_Elements[Geometry::TRIANGLE] = new RT_TriangleElement(p);
      RT_dof[Geometry::TRIANGLE] = p*pp1;

      RT_Elements[Geometry::SQUARE] = new RT_QuadrilateralElement(p, cb_type,
                                                                  ob_type);
      // two vector components * n_unk_face *
      RT_dof[Geometry::SQUARE] = 2*p*pp1;
   }
   else if (dim == 3)
   {
      // TODO: cb_type, ob_type for tets
      RT_Elements[Geometry::TETRAHEDRON] = new RT_TetrahedronElement(p);
      RT_dof[Geometry::TETRAHEDRON] = p*pp1*(p + 2)/2;

      RT_Elements[Geometry::CUBE] = new RT_HexahedronElement(p, cb_type, ob_type);
      RT_dof[Geometry::CUBE] = 3*p*pp1*pp1;

      RT_Elements[Geometry::PRISM] = new RT_WedgeElement(p);
      RT_dof[Geometry::PRISM] = p*pp1*(3*p + 4)/2;

      RT_Elements[Geometry::PYRAMID] = new RT_FuentesPyramidElement(p);
      RT_dof[Geometry::PYRAMID] = 3*p*pp1*pp1;
   }
   else
   {
      MFEM_ABORT("invalid dim = " << dim);
   }
}

// This is a special protected constructor only used by RT_Trace_FECollection
// and DG_Interface_FECollection
RT_FECollection::RT_FECollection(const int p, const int dim,
                                 const int map_type, const bool signs,
                                 const int ob_type)
   : FiniteElementCollection(p + 1)
   , dim(dim)
   , ob_type(ob_type)
{
   if (Quadrature1D::CheckOpen(BasisType::GetQuadrature1D(ob_type)) ==
       Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type); // this may abort
      MFEM_ABORT("Invalid open basis type: " << ob_name);
   }
   InitFaces(p, dim, map_type, signs);
}

void RT_FECollection::InitFaces(const int p, const int dim_,
                                const int map_type,
                                const bool signs)
{
   int op_type = BasisType::GetQuadrature1D(ob_type);

   MFEM_VERIFY(Quadrature1D::CheckOpen(op_type) != Quadrature1D::Invalid,
               "invalid open point type");

   const int pp1 = p + 1, pp2 = p + 2;

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      RT_Elements[g] = NULL;
      RT_dof[g] = 0;
   }
   // Degree of Freedom orderings
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }
   for (int i = 0; i < 6; i++)
   {
      TriDofOrd[i] = NULL;
   }
   for (int i = 0; i < 8; i++)
   {
      QuadDofOrd[i] = NULL;
   }

   if (dim_ == 2)
   {
      L2_SegmentElement *l2_seg = new L2_SegmentElement(p, ob_type);
      l2_seg->SetMapType(map_type);
      RT_Elements[Geometry::SEGMENT] = l2_seg;
      RT_dof[Geometry::SEGMENT] = pp1;

      SegDofOrd[0] = (pp1 > 0) ? new int[2*pp1] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + pp1;
      for (int i = 0; i <= p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = signs ? (-1 - (p - i)) : (p - i);
      }
   }
   else if (dim_ == 3)
   {
      L2_TriangleElement *l2_tri = new L2_TriangleElement(p, ob_type);
      l2_tri->SetMapType(map_type);
      RT_Elements[Geometry::TRIANGLE] = l2_tri;
      RT_dof[Geometry::TRIANGLE] = pp1*pp2/2;

      L2_QuadrilateralElement *l2_quad = new L2_QuadrilateralElement(p, ob_type);
      l2_quad->SetMapType(map_type);
      RT_Elements[Geometry::SQUARE] = l2_quad;
      RT_dof[Geometry::SQUARE] = pp1*pp1;

      int TriDof = RT_dof[Geometry::TRIANGLE];
      TriDofOrd[0] = (TriDof > 0) ? new int[6*TriDof] : nullptr;
      for (int i = 1; i < 6; i++)
      {
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      }
      // see Mesh::GetTriOrientation in mesh/mesh.cpp,
      // the constructor of H1_FECollection
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i + j <= p; i++)
         {
            int o = TriDof - ((pp2 - j)*(pp1 - j))/2 + i;
            int k = p - j - i;
            TriDofOrd[0][o] = o;  // (0,1,2)
            TriDofOrd[1][o] = -1-(TriDof-((pp2-j)*(pp1-j))/2+k);  // (1,0,2)
            TriDofOrd[2][o] =     TriDof-((pp2-i)*(pp1-i))/2+k;   // (2,0,1)
            TriDofOrd[3][o] = -1-(TriDof-((pp2-k)*(pp1-k))/2+i);  // (2,1,0)
            TriDofOrd[4][o] =     TriDof-((pp2-k)*(pp1-k))/2+j;   // (1,2,0)
            TriDofOrd[5][o] = -1-(TriDof-((pp2-i)*(pp1-i))/2+j);  // (0,2,1)
            if (!signs)
            {
               for (int kk = 1; kk < 6; kk += 2)
               {
                  TriDofOrd[kk][o] = -1 - TriDofOrd[kk][o];
               }
            }
         }
      }

      int QuadDof = RT_dof[Geometry::SQUARE];
      QuadDofOrd[0] = (QuadDof > 0) ? new int[8*QuadDof] : nullptr;
      for (int i = 1; i < 8; i++)
      {
         QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
      }
      // see Mesh::GetQuadOrientation in mesh/mesh.cpp
      for (int j = 0; j <= p; j++)
      {
         for (int i = 0; i <= p; i++)
         {
            int o = i + j*pp1;
            QuadDofOrd[0][o] = i + j*pp1;                    // (0,1,2,3)
            QuadDofOrd[1][o] = -1 - (j + i*pp1);             // (0,3,2,1)
            QuadDofOrd[2][o] = j + (p - i)*pp1;              // (1,2,3,0)
            QuadDofOrd[3][o] = -1 - ((p - i) + j*pp1);       // (1,0,3,2)
            QuadDofOrd[4][o] = (p - i) + (p - j)*pp1;        // (2,3,0,1)
            QuadDofOrd[5][o] = -1 - ((p - j) + (p - i)*pp1); // (2,1,0,3)
            QuadDofOrd[6][o] = (p - j) + i*pp1;              // (3,0,1,2)
            QuadDofOrd[7][o] = -1 - (i + (p - j)*pp1);       // (3,2,1,0)
            if (!signs)
            {
               for (int k = 1; k < 8; k += 2)
               {
                  QuadDofOrd[k][o] = -1 - QuadDofOrd[k][o];
               }
            }
         }
      }
   }
}

const FiniteElement *
RT_FECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   return RT_Elements[GeomType];
}

const int *RT_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                   int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   return NULL;
}

FiniteElementCollection *RT_FECollection::GetTraceCollection() const
{
   int tr_dim, tr_p;
   if (!strncmp(rt_name, "RT_", 3))
   {
      tr_dim = atoi(rt_name + 3);
      tr_p = atoi(rt_name + 7);
   }
   else // rt_name = RT@.._.D_P*
   {
      tr_dim = atoi(rt_name + 6);
      tr_p = atoi(rt_name + 10);
   }
   return new RT_Trace_FECollection(tr_p, tr_dim, FiniteElement::INTEGRAL,
                                    ob_type);
}

RT_FECollection::~RT_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete RT_Elements[g];
   }
}


RT_Trace_FECollection::RT_Trace_FECollection(const int p, const int dim,
                                             const int map_type,
                                             const int ob_type)
   : RT_FECollection(p, dim, map_type, true, ob_type)
{
   const char *prefix =
      (map_type == FiniteElement::INTEGRAL) ? "RT_Trace" : "RT_ValTrace";
   char ob_str[3] = { '\0', '\0', '\0' };

   if (ob_type != BasisType::GaussLegendre)
   {
      ob_str[0] = '@';
      ob_str[1] = BasisType::GetChar(ob_type);
   }
   snprintf(rt_name, 32, "%s%s_%dD_P%d", prefix, ob_str, dim, p);

   MFEM_VERIFY(dim == 2 || dim == 3, "Wrong dimension, dim = " << dim);
}


DG_Interface_FECollection::DG_Interface_FECollection(const int p, const int dim,
                                                     const int map_type,
                                                     const int ob_type)
   : RT_FECollection(p, dim, map_type, false, ob_type)
{
   MFEM_VERIFY(dim == 2 || dim == 3, "Wrong dimension, dim = " << dim);

   const char *prefix =
      (map_type == FiniteElement::VALUE) ? "DG_Iface" : "DG_IntIface";
   if (ob_type == BasisType::GaussLegendre)
   {
      snprintf(rt_name, 32, "%s_%dD_P%d", prefix, dim, p);
   }
   else
   {
      snprintf(rt_name, 32, "%s@%c_%dD_P%d", prefix,
               (int)BasisType::GetChar(ob_type), dim, p);
   }
}

ND_FECollection::ND_FECollection(const int p, const int dim,
                                 const int cb_type, const int ob_type)
   : FiniteElementCollection(dim > 1 ? p : p - 1)
   , dim(dim)
   , cb_type(cb_type)
   , ob_type(ob_type)
{
   MFEM_VERIFY(p >= 1, "ND_FECollection requires order >= 1.");
   MFEM_VERIFY(dim >= 1 && dim <= 3, "ND_FECollection requires 1 <= dim <= 3.");

   const int pm1 = p - 1, pm2 = p - 2;

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(nd_name, 32, "ND_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(nd_name, 32, "ND@%c%c_%dD_P%d", (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      ND_Elements[g] = NULL;
      ND_dof[g] = 0;
   }
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }
   for (int i = 0; i < 6; i++)
   {
      TriDofOrd[i] = NULL;
   }
   for (int i = 0; i < 8; i++)
   {
      QuadDofOrd[i] = NULL;
   }

   int op_type = BasisType::GetQuadrature1D(ob_type);
   int cp_type = BasisType::GetQuadrature1D(cb_type);

   // Error checking
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type);
      MFEM_ABORT("Invalid open basis point type: " << ob_name);
   }
   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type);
      MFEM_ABORT("Invalid closed basis point type: " << cb_name);
   }

   if (dim >= 1)
   {
      ND_Elements[Geometry::SEGMENT] = new ND_SegmentElement(p, ob_type);
      ND_dof[Geometry::SEGMENT] = p;

      SegDofOrd[0] = (p > 0) ? new int[2*p] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + p;
      for (int i = 0; i < p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = -1 - (pm1 - i);
      }
   }

   if (dim >= 2)
   {
      ND_Elements[Geometry::SQUARE] = new ND_QuadrilateralElement(p, cb_type,
                                                                  ob_type);
      ND_dof[Geometry::SQUARE] = 2*p*pm1;

      // TODO: cb_type and ob_type for triangles
      ND_Elements[Geometry::TRIANGLE] = new ND_TriangleElement(p);
      ND_dof[Geometry::TRIANGLE] = p*pm1;

      int QuadDof = ND_dof[Geometry::SQUARE];
      QuadDofOrd[0] = (QuadDof > 0) ? new int[8*QuadDof] : nullptr;
      for (int i = 1; i < 8; i++)
      {
         QuadDofOrd[i] = QuadDofOrd[i-1] + QuadDof;
      }
      // see Mesh::GetQuadOrientation in mesh/mesh.cpp
      for (int j = 0; j < pm1; j++)
      {
         for (int i = 0; i < p; i++)
         {
            int d1 = i + j*p;            // x-component
            int d2 = p*pm1 + j + i*pm1;  // y-component
            // (0,1,2,3)
            QuadDofOrd[0][d1] = d1;
            QuadDofOrd[0][d2] = d2;
            // (0,3,2,1)
            QuadDofOrd[1][d1] = d2;
            QuadDofOrd[1][d2] = d1;
            // (1,2,3,0)
            // QuadDofOrd[2][d1] = p*pm1 + (pm2 - j) + i*pm1;
            // QuadDofOrd[2][d2] = -1 - ((pm1 - i) + j*p);
            QuadDofOrd[2][d1] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
            QuadDofOrd[2][d2] = i + (pm2 - j)*p;
            // (1,0,3,2)
            QuadDofOrd[3][d1] = -1 - ((pm1 - i) + j*p);
            QuadDofOrd[3][d2] = p*pm1 + (pm2 - j) + i*pm1;
            // (2,3,0,1)
            QuadDofOrd[4][d1] = -1 - ((pm1 - i) + (pm2 - j)*p);
            QuadDofOrd[4][d2] = -1 - (p*pm1 + (pm2 - j) + (pm1 - i)*pm1);
            // (2,1,0,3)
            QuadDofOrd[5][d1] = -1 - (p*pm1 + (pm2 - j) + (pm1 - i)*pm1);
            QuadDofOrd[5][d2] = -1 - ((pm1 - i) + (pm2 - j)*p);
            // (3,0,1,2)
            // QuadDofOrd[6][d1] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
            // QuadDofOrd[6][d2] = i + (pm2 - j)*p;
            QuadDofOrd[6][d1] = p*pm1 + (pm2 - j) + i*pm1;
            QuadDofOrd[6][d2] = -1 - ((pm1 - i) + j*p);
            // (3,2,1,0)
            QuadDofOrd[7][d1] = i + (pm2 - j)*p;
            QuadDofOrd[7][d2] = -1 - (p*pm1 + j + (pm1 - i)*pm1);
         }
      }

      int TriDof = ND_dof[Geometry::TRIANGLE];
      TriDofOrd[0] = (TriDof > 0) ? new int[6*TriDof] : nullptr;
      for (int i = 1; i < 6; i++)
      {
         TriDofOrd[i] = TriDofOrd[i-1] + TriDof;
      }
      // see Mesh::GetTriOrientation in mesh/mesh.cpp,
      // the constructor of H1_FECollection
      for (int j = 0; j <= pm2; j++)
      {
         for (int i = 0; i + j <= pm2; i++)
         {
            int k0 = p*pm1 - (p - j)*(pm1 - j) + 2*i;
            int k1 = 2*pm2 - 2*i + ((2*p-3)-j)*j;
            int k2 = 2*pm2 - 2*j + ((2*p-3)-i)*i;
            int k3 = p*pm1 - 2 - 3*j - i - (i+j)*(i+j);
            int k4 = p*pm1 - 2 - 3*i - j - (i+j)*(i+j);
            int k5 = p*pm1 - (p - i)*(pm1 - i) + 2*j;

            // (0,1,2)
            TriDofOrd[0][k0  ] = k0;
            TriDofOrd[0][k0+1] = k0 + 1;
            // (1,0,2)
            TriDofOrd[1][k0  ] = k1;
            TriDofOrd[1][k0+1] = k1 + 1;
            // (2,0,1)
            TriDofOrd[2][k0  ] = k2;
            TriDofOrd[2][k0+1] = k2 + 1;
            // (2,1,0)
            TriDofOrd[3][k0  ] = k3;
            TriDofOrd[3][k0+1] = k3 + 1;
            // (1,2,0)
            TriDofOrd[4][k0  ] = k4;
            TriDofOrd[4][k0+1] = k4 + 1;
            // (0,2,1)
            TriDofOrd[5][k0  ] = k5;
            TriDofOrd[5][k0+1] = k5 + 1;
         }
      }
   }

   if (dim >= 3)
   {
      ND_Elements[Geometry::CUBE] = new ND_HexahedronElement(p, cb_type, ob_type);
      ND_dof[Geometry::CUBE] = 3*p*pm1*pm1;

      // TODO: cb_type and ob_type for tets
      ND_Elements[Geometry::TETRAHEDRON] = new ND_TetrahedronElement(p);
      ND_dof[Geometry::TETRAHEDRON] = p*pm1*pm2/2;

      ND_Elements[Geometry::PRISM] = new ND_WedgeElement(p);
      ND_dof[Geometry::PRISM] = p*pm1*(3*p-4)/2;

      ND_Elements[Geometry::PYRAMID] = new ND_FuentesPyramidElement(p);
      ND_dof[Geometry::PYRAMID] = 3*p*pm1*pm1;
   }
}

const FiniteElement *
ND_FECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   return ND_Elements[GeomType];
}

const StatelessDofTransformation *
ND_FECollection::DofTransformationForGeometry(Geometry::Type GeomType) const
{
   if (!Geometry::IsTensorProduct(GeomType) && this->GetOrder() > 1)
   {
      return FiniteElementForGeometry(GeomType)->GetDofTransformation();
   }
   else
   {
      return NULL;
   }
}

const int *ND_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                   int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   else if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   else if (GeomType == Geometry::SQUARE)
   {
      return QuadDofOrd[Or%8];
   }
   return NULL;
}

FiniteElementCollection *ND_FECollection::GetTraceCollection() const
{
   int tr_p, tr_dim, tr_cb_type, tr_ob_type;

   tr_p = ND_dof[Geometry::SEGMENT];
   if (nd_name[2] == '_') // ND_
   {
      tr_dim = atoi(nd_name + 3);
      tr_cb_type = BasisType::GaussLobatto;
      tr_ob_type = BasisType::GaussLegendre;
   }
   else // ND@
   {
      tr_dim = atoi(nd_name + 6);
      tr_cb_type = BasisType::GetType(nd_name[3]);
      tr_ob_type = BasisType::GetType(nd_name[4]);
   }
   return new ND_Trace_FECollection(tr_p, tr_dim, tr_cb_type, tr_ob_type);
}

ND_FECollection::~ND_FECollection()
{
   delete [] SegDofOrd[0];
   delete [] TriDofOrd[0];
   delete [] QuadDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete ND_Elements[g];
   }
}


ND_Trace_FECollection::ND_Trace_FECollection(const int p, const int dim,
                                             const int cb_type,
                                             const int ob_type)
   : ND_FECollection(p, dim-1, cb_type, ob_type)
{
   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(nd_name, 32, "ND_Trace_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(nd_name, 32, "ND_Trace@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }
}


ND_R1D_FECollection::ND_R1D_FECollection(const int p, const int dim,
                                         const int cb_type, const int ob_type)
   : FiniteElementCollection(p)
{
   MFEM_VERIFY(p >= 1, "ND_R1D_FECollection requires order >= 1.");
   MFEM_VERIFY(dim == 1, "ND_R1D_FECollection requires dim == 1.");

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(nd_name, 32, "ND_R1D_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(nd_name, 32, "ND_R1D@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      ND_Elements[g] = NULL;
      ND_dof[g] = 0;
   }

   int op_type = BasisType::GetQuadrature1D(ob_type);
   int cp_type = BasisType::GetQuadrature1D(cb_type);

   // Error checking
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type);
      MFEM_ABORT("Invalid open basis point type: " << ob_name);
   }
   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type);
      MFEM_ABORT("Invalid closed basis point type: " << cb_name);
   }

   ND_Elements[Geometry::POINT] = new ND_R1D_PointElement(p);
   ND_dof[Geometry::POINT] = 2;

   ND_Elements[Geometry::SEGMENT] = new ND_R1D_SegmentElement(p,
                                                              cb_type,
                                                              ob_type);
   ND_dof[Geometry::SEGMENT] = 3 * p - 2;
}

const int *ND_R1D_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   return NULL;
}

FiniteElementCollection *ND_R1D_FECollection::GetTraceCollection() const
{
   return NULL;
}

ND_R1D_FECollection::~ND_R1D_FECollection()
{
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete ND_Elements[g];
   }
}


RT_R1D_FECollection::RT_R1D_FECollection(const int p, const int dim,
                                         const int cb_type, const int ob_type)
   : FiniteElementCollection(p + 1)
{
   MFEM_VERIFY(p >= 0, "RT_R1D_FECollection requires order >= 0.");
   MFEM_VERIFY(dim == 1, "RT_R1D_FECollection requires dim == 1.");

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(rt_name, 32, "RT_R1D_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(rt_name, 32, "RT_R1D@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      RT_Elements[g] = NULL;
      RT_dof[g] = 0;
   }

   int op_type = BasisType::GetQuadrature1D(ob_type);
   int cp_type = BasisType::GetQuadrature1D(cb_type);

   // Error checking
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type);
      MFEM_ABORT("Invalid open basis point type: " << ob_name);
   }
   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type);
      MFEM_ABORT("Invalid closed basis point type: " << cb_name);
   }

   RT_Elements[Geometry::POINT] = new PointFiniteElement;
   RT_dof[Geometry::POINT] = 1;

   RT_Elements[Geometry::SEGMENT] = new RT_R1D_SegmentElement(p,
                                                              cb_type,
                                                              ob_type);
   RT_dof[Geometry::SEGMENT] = 3 * p + 2;
}

const int *RT_R1D_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   return NULL;
}

FiniteElementCollection *RT_R1D_FECollection::GetTraceCollection() const
{
   MFEM_ABORT("this method is not implemented in RT_R1D_FECollection!");
   return NULL;
}

RT_R1D_FECollection::~RT_R1D_FECollection()
{
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete RT_Elements[g];
   }
}


ND_R2D_FECollection::ND_R2D_FECollection(const int p, const int dim,
                                         const int cb_type, const int ob_type)
   : FiniteElementCollection(p)
{
   MFEM_VERIFY(p >= 1, "ND_R2D_FECollection requires order >= 1.");
   MFEM_VERIFY(dim >= 1 && dim <= 2,
               "ND_R2D_FECollection requires 1 <= dim <= 2.");

   const int pm1 = p - 1, pm2 = p - 2;

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(nd_name, 32, "ND_R2D_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(nd_name, 32, "ND_R2D@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      ND_Elements[g] = NULL;
      ND_dof[g] = 0;
   }
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }

   int op_type = BasisType::GetQuadrature1D(ob_type);
   int cp_type = BasisType::GetQuadrature1D(cb_type);

   // Error checking
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type);
      MFEM_ABORT("Invalid open basis point type: " << ob_name);
   }
   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type);
      MFEM_ABORT("Invalid closed basis point type: " << cb_name);
   }

   ND_dof[Geometry::POINT] = 1;

   if (dim >= 1)
   {
      ND_Elements[Geometry::SEGMENT] = new ND_R2D_SegmentElement(p,
                                                                 cb_type,
                                                                 ob_type);
      ND_dof[Geometry::SEGMENT] = 2 * p - 1;

      SegDofOrd[0] = (4*p > 2) ? new int[4 * p - 2] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + 2 * p - 1;
      for (int i = 0; i < p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = -1 - (pm1 - i);
      }
      for (int i = 0; i < pm1; i++)
      {
         SegDofOrd[0][p+i] = p + i;
         SegDofOrd[1][p+i] = 2 * pm1 - i;
      }
   }

   if (dim >= 2)
   {
      ND_Elements[Geometry::SQUARE] = new ND_R2D_QuadrilateralElement(p,
                                                                      cb_type,
                                                                      ob_type);
      ND_dof[Geometry::SQUARE] = 2*p*pm1 + pm1*pm1;

      // TODO: cb_type and ob_type for triangles
      ND_Elements[Geometry::TRIANGLE] = new ND_R2D_TriangleElement(p, cb_type);
      ND_dof[Geometry::TRIANGLE] = p*pm1 + (pm1*pm2)/2;
   }
}

const int *ND_R2D_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   return NULL;
}

FiniteElementCollection *ND_R2D_FECollection::GetTraceCollection() const
{
   int p, dim, cb_type, ob_type;

   p = ND_dof[Geometry::SEGMENT];
   if (nd_name[5] == '_') // ND_R2D_
   {
      dim = atoi(nd_name + 6);
      cb_type = BasisType::GaussLobatto;
      ob_type = BasisType::GaussLegendre;
   }
   else // ND_R2D@
   {
      dim = atoi(nd_name + 9);
      cb_type = BasisType::GetType(nd_name[6]);
      ob_type = BasisType::GetType(nd_name[7]);
   }
   return new ND_R2D_Trace_FECollection(p, dim, cb_type, ob_type);
}

ND_R2D_FECollection::~ND_R2D_FECollection()
{
   delete [] SegDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete ND_Elements[g];
   }
}


ND_R2D_Trace_FECollection::ND_R2D_Trace_FECollection(const int p, const int dim,
                                                     const int cb_type,
                                                     const int ob_type)
   : ND_R2D_FECollection(p, dim-1, cb_type, ob_type)
{
   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(nd_name, 32, "ND_R2D_Trace_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(nd_name, 32, "ND_R2D_Trace@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }
}


RT_R2D_FECollection::RT_R2D_FECollection(const int p, const int dim,
                                         const int cb_type, const int ob_type)
   : FiniteElementCollection(p + 1),
     ob_type(ob_type)
{
   MFEM_VERIFY(p >= 0, "RT_R2D_FECollection requires order >= 0.");
   MFEM_VERIFY(dim >= 1 && dim <= 2,
               "RT_R2D_FECollection requires 1 <= dim <= 2.");

   int cp_type = BasisType::GetQuadrature1D(cb_type);
   int op_type = BasisType::GetQuadrature1D(ob_type);

   if (Quadrature1D::CheckClosed(cp_type) == Quadrature1D::Invalid)
   {
      const char *cb_name = BasisType::Name(cb_type); // this may abort
      MFEM_ABORT("unknown closed BasisType: " << cb_name);
   }
   if (Quadrature1D::CheckOpen(op_type) == Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type); // this may abort
      MFEM_ABORT("unknown open BasisType: " << ob_name);
   }

   InitFaces(p, dim, FiniteElement::INTEGRAL, true);

   if (cb_type == BasisType::GaussLobatto &&
       ob_type == BasisType::GaussLegendre)
   {
      snprintf(rt_name, 32, "RT_R2D_%dD_P%d", dim, p);
   }
   else
   {
      snprintf(rt_name, 32, "RT_R2D@%c%c_%dD_P%d",
               (int)BasisType::GetChar(cb_type),
               (int)BasisType::GetChar(ob_type), dim, p);
   }

   const int pp1 = p + 1;
   const int pp2 = p + 2;
   if (dim == 2)
   {
      // TODO: cb_type, ob_type for triangles
      RT_Elements[Geometry::TRIANGLE] = new RT_R2D_TriangleElement(p);
      RT_dof[Geometry::TRIANGLE] = p*pp1 + (pp1 * pp2) / 2;

      RT_Elements[Geometry::SQUARE] = new RT_R2D_QuadrilateralElement(p,
                                                                      cb_type,
                                                                      ob_type);
      // two vector components * n_unk_face *
      RT_dof[Geometry::SQUARE] = 2*p*pp1 + pp1*pp1;
   }
}

// This is a special protected constructor only used by RT_Trace_FECollection
// and DG_Interface_FECollection
RT_R2D_FECollection::RT_R2D_FECollection(const int p, const int dim,
                                         const int map_type,
                                         const bool signs, const int ob_type)
   : ob_type(ob_type)
{
   if (Quadrature1D::CheckOpen(BasisType::GetQuadrature1D(ob_type)) ==
       Quadrature1D::Invalid)
   {
      const char *ob_name = BasisType::Name(ob_type); // this may abort
      MFEM_ABORT("Invalid open basis type: " << ob_name);
   }
   InitFaces(p, dim, map_type, signs);
}

void RT_R2D_FECollection::InitFaces(const int p, const int dim,
                                    const int map_type,
                                    const bool signs)
{
   int op_type = BasisType::GetQuadrature1D(ob_type);

   MFEM_VERIFY(Quadrature1D::CheckOpen(op_type) != Quadrature1D::Invalid,
               "invalid open point type");

   const int pp1 = p + 1;

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      RT_Elements[g] = NULL;
      RT_dof[g] = 0;
   }
   // Degree of Freedom orderings
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
   }

   if (dim == 2)
   {
      L2_SegmentElement *l2_seg = new L2_SegmentElement(p, ob_type);
      l2_seg->SetMapType(map_type);
      RT_Elements[Geometry::SEGMENT] = l2_seg;
      RT_dof[Geometry::SEGMENT] = pp1;

      SegDofOrd[0] = (pp1 > 0) ? new int[2*pp1] : nullptr;
      SegDofOrd[1] = SegDofOrd[0] + pp1;
      for (int i = 0; i <= p; i++)
      {
         SegDofOrd[0][i] = i;
         SegDofOrd[1][i] = signs ? (-1 - (p - i)) : (p - i);
      }
   }
}

const int *RT_R2D_FECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                       int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   return NULL;
}

FiniteElementCollection *RT_R2D_FECollection::GetTraceCollection() const
{
   int dim, p;
   if (!strncmp(rt_name, "RT_R2D_", 7))
   {
      dim = atoi(rt_name + 7);
      p = atoi(rt_name + 11);
   }
   else // rt_name = RT_R2D@.._.D_P*
   {
      dim = atoi(rt_name + 10);
      p = atoi(rt_name + 14);
   }
   return new RT_R2D_Trace_FECollection(p, dim, FiniteElement::INTEGRAL, ob_type);
}

RT_R2D_FECollection::~RT_R2D_FECollection()
{
   delete [] SegDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete RT_Elements[g];
   }
}


RT_R2D_Trace_FECollection::RT_R2D_Trace_FECollection(const int p, const int dim,
                                                     const int map_type,
                                                     const int ob_type)
   : RT_R2D_FECollection(p, dim-1, map_type, true, ob_type)
{
   const char *prefix =
      (map_type == FiniteElement::INTEGRAL) ? "RT_R2D_Trace" : "RT_R2D_ValTrace";
   char ob_str[3] = { '\0', '\0', '\0' };

   if (ob_type != BasisType::GaussLegendre)
   {
      ob_str[0] = '@';
      ob_str[1] = BasisType::GetChar(ob_type);
   }
   snprintf(rt_name, 32, "%s%s_%dD_P%d", prefix, ob_str, dim, p);

   MFEM_VERIFY(dim == 2, "Wrong dimension, dim = " << dim);
}


Local_FECollection::Local_FECollection(const char *fe_name)
{
   snprintf(d_name, 32, "Local_%s", fe_name);

   Local_Element = NULL;

   if (!strcmp(fe_name, "BiCubic2DFiniteElement") ||
       !strcmp(fe_name, "Quad_Q3"))
   {
      GeomType = Geometry::SQUARE;
      Local_Element = new BiCubic2DFiniteElement;
   }
   else if (!strcmp(fe_name, "Nedelec1HexFiniteElement") ||
            !strcmp(fe_name, "Hex_ND1"))
   {
      GeomType = Geometry::CUBE;
      Local_Element = new Nedelec1HexFiniteElement;
   }
   else if (!strncmp(fe_name, "H1_", 3))
   {
      GeomType = Geometry::SQUARE;
      Local_Element = new H1_QuadrilateralElement(atoi(fe_name + 7));
   }
   else if (!strncmp(fe_name, "H1Pos_", 6))
   {
      GeomType = Geometry::SQUARE;
      Local_Element = new H1Pos_QuadrilateralElement(atoi(fe_name + 10));
   }
   else if (!strncmp(fe_name, "L2_", 3))
   {
      GeomType = Geometry::SQUARE;
      Local_Element = new L2_QuadrilateralElement(atoi(fe_name + 7));
   }
   else
   {
      mfem::err << "Local_FECollection::Local_FECollection : fe_name = "
                << fe_name << endl;
      mfem_error();
   }
}


NURBSFECollection::NURBSFECollection(int Order)
   : FiniteElementCollection((Order == VariableOrder) ? 1 : Order)
{
   const int order = (Order == VariableOrder) ? 1 : Order;
   PointFE          = new PointFiniteElement();
   SegmentFE        = new NURBS1DFiniteElement(order);
   QuadrilateralFE  = new NURBS2DFiniteElement(order);
   ParallelepipedFE = new NURBS3DFiniteElement(order);

   SetOrder(Order);
}

void NURBSFECollection::SetOrder(int Order) const
{
   mOrder = Order;
   if (Order != VariableOrder)
   {
      snprintf(name, 16, "NURBS%i", Order);
   }
   else
   {
      snprintf(name, 16, "NURBS");
   }
}

NURBSFECollection::~NURBSFECollection()
{
   delete PointFE;
   delete SegmentFE;
   delete QuadrilateralFE;
   delete ParallelepipedFE;
}

const FiniteElement *
NURBSFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::POINT:       return PointFE;
      case Geometry::SEGMENT:     return SegmentFE;
      case Geometry::SQUARE:      return QuadrilateralFE;
      case Geometry::CUBE:        return ParallelepipedFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("NURBSFECollection: unknown geometry type.");
   }
   return SegmentFE; // Make some compilers happy
}

int NURBSFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   mfem_error("NURBSFECollection::DofForGeometry");
   return 0; // Make some compilers happy
}

const int *NURBSFECollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                     int Or) const
{
   mfem_error("NURBSFECollection::DofOrderForOrientation");
   return NULL;
}

FiniteElementCollection *NURBSFECollection::GetTraceCollection() const
{
   MFEM_ABORT("NURBS finite elements can not be statically condensed!");
   return NULL;
}


NURBS_HDivFECollection::NURBS_HDivFECollection(int Order, const int dim)
   : NURBSFECollection((Order == VariableOrder) ? 1 : Order)
{
   const int order = (Order == VariableOrder) ? 1 : Order;

   SegmentFE       = new NURBS1DFiniteElement(order);
   QuadrilateralFE = new NURBS2DFiniteElement(order);

   QuadrilateralVFE  = new NURBS_HDiv2DFiniteElement(order);
   ParallelepipedVFE = new NURBS_HDiv3DFiniteElement(order);

   if (dim != -1) { SetDim(dim); }
   SetOrder(Order);
}

void NURBS_HDivFECollection::SetDim(int dim)
{
   if (dim == 2)
   {
      sFE = SegmentFE;
      qFE = QuadrilateralVFE;
      hFE = nullptr;
   }
   else if (dim == 3)
   {
      sFE = nullptr;
      qFE = QuadrilateralFE;
      hFE = ParallelepipedVFE;
   }
   else
   {
      mfem::err<<"Dimension = "<<dim<<endl;
      mfem_error ("NURBS_HDivFECollection: wrong dimension!");
   }
}

NURBS_HDivFECollection::~NURBS_HDivFECollection()
{
   delete SegmentFE;
   delete QuadrilateralFE;
   delete QuadrilateralVFE;
   delete ParallelepipedVFE;
}

const FiniteElement *
NURBS_HDivFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return sFE;
      case Geometry::SQUARE:      return qFE;
      case Geometry::CUBE:        return hFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("NURBS_HDivFECollection: unknown geometry type.");
   }
   return QuadrilateralFE; // Make some compilers happy
}

void NURBS_HDivFECollection::SetOrder(int Order) const
{
   mOrder = Order;
   if (Order != VariableOrder)
   {
      snprintf(name, 16, "NURBS_HDiv%i", Order);
   }
   else
   {
      snprintf(name, 16, "NURBS_HDiv");
   }
}

int NURBS_HDivFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   mfem_error("NURBS_HDivFECollection::DofForGeometry");
   return 0; // Make some compilers happy
}

const int *NURBS_HDivFECollection::DofOrderForOrientation(
   Geometry::Type GeomType,
   int Or) const
{
   mfem_error("NURBS_HDivFECollection::DofOrderForOrientation");
   return NULL;
}

FiniteElementCollection *NURBS_HDivFECollection::GetTraceCollection() const
{
   MFEM_ABORT("NURBS finite elements can not be statically condensed!");
   return NULL;
}

NURBS_HCurlFECollection::NURBS_HCurlFECollection(int Order, const int dim)
   : NURBSFECollection((Order == VariableOrder) ? 1 : Order)
{
   const int order = (Order == VariableOrder) ? 1 : Order;

   SegmentFE       = new NURBS1DFiniteElement(order+1);
   QuadrilateralFE = new NURBS2DFiniteElement(order+1);

   QuadrilateralVFE  = new NURBS_HCurl2DFiniteElement(order);
   ParallelepipedVFE = new NURBS_HCurl3DFiniteElement(order);
   if (dim != -1) { SetDim(dim); }
   SetOrder(Order);
}

void NURBS_HCurlFECollection::SetDim(int dim)
{
   if (dim == 2)
   {
      sFE = SegmentFE;
      qFE = QuadrilateralVFE;
      hFE = nullptr;
   }
   else if (dim == 3)
   {
      sFE = nullptr;
      qFE = QuadrilateralFE;
      hFE = ParallelepipedVFE;
   }
   else
   {
      mfem::err<<"Dimension = "<<dim<<endl;
      mfem_error ("NURBS_HCurlFECollection: wrong dimension!");
   }
}



NURBS_HCurlFECollection::~NURBS_HCurlFECollection()
{
   delete SegmentFE;
   delete QuadrilateralFE;
   delete QuadrilateralVFE;
   delete ParallelepipedVFE;
}

const FiniteElement *
NURBS_HCurlFECollection::FiniteElementForGeometry(Geometry::Type GeomType) const
{
   switch (GeomType)
   {
      case Geometry::SEGMENT:     return sFE;
      case Geometry::SQUARE:      return qFE;
      case Geometry::CUBE:        return hFE;
      default:
         if (error_mode == RETURN_NULL) { return nullptr; }
         mfem_error ("NURBS_HCurlFECollection: unknown geometry type.");
   }
   return QuadrilateralFE; // Make some compilers happy
}

void NURBS_HCurlFECollection::SetOrder(int Order) const
{
   mOrder = Order;
   if (Order != VariableOrder)
   {
      snprintf(name, 16, "NURBS_HCurl%i", Order);
   }
   else
   {
      snprintf(name, 16, "NURBS_HCurl");
   }
}

int NURBS_HCurlFECollection::DofForGeometry(Geometry::Type GeomType) const
{
   mfem_error("NURBS_HCurlFECollection::DofForGeometry");
   return 0; // Make some compilers happy
}

const int *NURBS_HCurlFECollection::DofOrderForOrientation(
   Geometry::Type GeomType,
   int Or) const
{
   mfem_error("NURBS_HCurlFECollection::DofOrderForOrientation");
   return NULL;
}

FiniteElementCollection *NURBS_HCurlFECollection::GetTraceCollection() const
{
   MFEM_ABORT("NURBS finite elements can not be statically condensed!");
   return NULL;
}



}
