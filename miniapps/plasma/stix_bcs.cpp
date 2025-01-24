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

#include "stix_bcs.hpp"

using namespace std;

namespace mfem
{

using namespace common;

namespace plasma
{

// Used for combining scalar coefficients
double prodFunc(double a, double b) { return a * b; }

StixBCs::~StixBCs()
{
   for (int i=0; i<dbc.Size(); i++)
   {
      delete dbc[i];
   }
   for (int i=0; i<nbc.Size(); i++)
   {
      delete nbc[i];
   }
   for (int i=0; i<sbc.Size(); i++)
   {
      delete sbc[i];
   }
   for (int i=0; i<cyl_axis.Size(); i++)
   {
      delete cyl_axis[i];
   }
   for (int i=0; i<jsrc.Size(); i++)
   {
      delete jsrc[i];
   }
}

const char * StixBCs::GetBCTypeName(BCType bctype)
{
   switch (bctype)
   {
      case DIRICHLET_BC: return "Dirichlet";
      case NEUMANN_BC: return "Neumann";
      case SHEATH_BC: return "Sheath";
      case CYL_AXIS: return "Cylindrical Axis";
      case CURRENT_SRC: return "Current Density";
   }
   return "Unknown";
}

void StixBCs::AddDirichletBC(const Array<int> & bdr,
                             VectorCoefficient &real_val,
                             VectorCoefficient &imag_val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Dirichlet BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   ComplexVectorCoefficientByAttr *bc = new ComplexVectorCoefficientByAttr;
   bc->attr = bdr;
   bc->real = &real_val;
   bc->imag = &imag_val;

   AttrToMarker(bdr_attr.Max(), bc->attr, bc->attr_marker);

   dbc.Append(bc);
}

void StixBCs::AddNeumannBC(const Array<int> & bdr,
                           VectorCoefficient &real_val,
                           VectorCoefficient &imag_val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Neumann BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   ComplexVectorCoefficientByAttr *bc = new ComplexVectorCoefficientByAttr;
   bc->attr = bdr;
   bc->real = &real_val;
   bc->imag = &imag_val;

   AttrToMarker(bdr_attr.Max(), bc->attr, bc->attr_marker);

   nbc.Append(bc);
}

void StixBCs::AddSheathBC(const Array<int> & bdr,
                          Coefficient &real_val,
                          Coefficient &imag_val)
{
   for (int i=0; i<bdr.Size(); i++)
   {
      if (bc_attr.count(bdr[i]) == 0)
      {
         bc_attr.insert(bdr[i]);
      }
      else
      {
         MFEM_ABORT("Attempting to add a Sheath BC on boundary " << bdr[i]
                    << " which already has a boundary condition defined.");
      }
   }
   ComplexCoefficientByAttr *bc = new ComplexCoefficientByAttr;
   bc->attr = bdr;
   bc->real = &real_val;
   bc->imag = &imag_val;

   AttrToMarker(bdr_attr.Max(), bc->attr, bc->attr_marker);

   sbc.Append(bc);
}

void StixBCs::AddCylindricalAxis(const Array<int> & bdr)
{
   AttributeArrays *aa = new AttributeArrays;
   aa->attr = bdr;

   AttrToMarker(bdr_attr.Max(), aa->attr, aa->attr_marker);
   cyl_axis.Append(aa);
}

void StixBCs::AddCurrentSrc(const Array<int> & reg,
                            VectorCoefficient &real_val,
                            VectorCoefficient &imag_val)
{
   ComplexVectorCoefficientByAttr *src = new ComplexVectorCoefficientByAttr;
   src->attr = reg;
   src->real = &real_val;
   src->imag = &imag_val;

   AttrToMarker(reg_attr.Max(), src->attr, src->attr_marker);

   jsrc.Append(src);
}

} // namespace plasma

} // namespace mfem
