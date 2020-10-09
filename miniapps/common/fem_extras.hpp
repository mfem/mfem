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

#ifndef MFEM_FEM_EXTRAS
#define MFEM_FEM_EXTRAS

#include "mfem.hpp"
#include <cstddef>

namespace mfem
{

namespace common
{

/** The H1_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an H1_FECollection object.
*/
class H1_FESpace : public FiniteElementSpace
{
public:
   H1_FESpace(Mesh *m,
              const int p, const int space_dim = 3,
              const int type = BasisType::GaussLobatto,
              int vdim = 1, int order = Ordering::byNODES);
   ~H1_FESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The ND_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an ND_FECollection object.
*/
class ND_FESpace : public FiniteElementSpace
{
public:
   ND_FESpace(Mesh *m, const int p, const int space_dim,
              int vdim = 1, int order = Ordering::byNODES);
   ~ND_FESpace();
private:
   const FiniteElementCollection *FEC_;
};

/** The RT_FESpace class is a FiniteElementSpace which automatically
    allocates and destroys its own FiniteElementCollection, in this
    case an RT_FECollection object.
*/
class RT_FESpace : public FiniteElementSpace
{
public:
   RT_FESpace(Mesh *m, const int p, const int space_dim,
              int vdim = 1, int order = Ordering::byNODES);
   ~RT_FESpace();
private:
   const FiniteElementCollection *FEC_;
};


/** Split a vector coefficient into a collection of non-overlapping
    partitions which may themselves be scalar or vector-valued.
*/
class PartitionedVectorCoefficient
{
private:
   VectorCoefficient & V;
   Array<unsigned int> & p;

   enum PartType {SCALAR_PART, VECTOR_PART};

   Array<PartType> pType;
   Array<unsigned int> pOffset;

   Array<Coefficient*> sCoefs;
   Array<VectorCoefficient*> vCoefs;

   class ComponentCoefficient : public Coefficient
   {
   private:
      VectorCoefficient &V;
      int c;

      mutable Vector val;

   public:
      ComponentCoefficient(VectorCoefficient &v, int comp)
         : V(v), c(comp), val(v.GetVDim())
      {
         MFEM_VERIFY(c >= 0 && c < V.GetVDim(),
                     "Invalid component specified in "
                     "ComponentCoefficient.");
      }

      double Eval(ElementTransformation &T, const IntegrationPoint &ip)
      {
         V.Eval(val, T, ip);
         return val[c];
      }
   };

   class SubVectorCoefficient : public VectorCoefficient
   {
   private:
      VectorCoefficient &V;
      int o;

      mutable Vector val;

   public:
      SubVectorCoefficient(VectorCoefficient &v, int offset, int dim)
         : VectorCoefficient(dim), V(v), o(offset), val(v.GetVDim())
      {
         MFEM_VERIFY(vdim > 1 && vdim < V.GetVDim(),
                     "Invalid subvector dimension "
                     "specified in SubVectorCoefficient.");
         MFEM_VERIFY(o >= 0 && o <= V.GetVDim()-vdim,
                     "Invalid offset specified "
                     "in SubVectorCoefficient.");
      }

      void Eval(Vector &sval, ElementTransformation &T,
                const IntegrationPoint &ip)
      {
         sval.SetSize(vdim);
         V.Eval(val, T, ip);
         for (int i=0; i<vdim; i++)
         {
            sval[i] = val[o + i];
         }
      }
   };

public:
   PartitionedVectorCoefficient(VectorCoefficient &v,
                                Array<unsigned int> &part);

   int GetNumPartitions() const { return p.Size(); }
   unsigned int GetPartitionSize(int i) const { return p[i]; }

   bool IsScalarPartition(int i) const { return p[i] == 1; }
   bool IsVectorPartition(int i) const { return p[i] > 1; }

   Coefficient & GetScalarPartition(int i)
   {
      MFEM_VERIFY(IsScalarPartition(i), "Scalar Coefficient requested for "
                  "vector valued partition.");
      return *sCoefs[pOffset[i]];
   }

   VectorCoefficient & GetVectorPartition(int i)
   {
      MFEM_VERIFY(IsVectorPartition(i), "Vector Coefficient requested for "
                  "scalar valued partition.");
      return *vCoefs[pOffset[i]];
   }
};


/// Visualize the given mesh object, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   Mesh &mesh, const char *title,
                   int x = 0, int y = 0, int w = 400, int h = 400,
                   const char *keys = NULL);

/// Visualize the given grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    GridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    const char *keys = NULL, bool vec = false);

} // namespace common

} // namespace mfem

#endif
