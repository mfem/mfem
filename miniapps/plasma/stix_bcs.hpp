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

#ifndef MFEM_STIX_BCS
#define MFEM_STIX_BCS

#include "mfem.hpp"
#include "../common/mesh_extras.hpp"

namespace mfem
{

namespace plasma
{

struct AttributeArrays
{
   Array<int> attr;
   Array<int> attr_marker;

};

struct ComplexCoefficientByAttr : public AttributeArrays
{
   Coefficient * real;
   Coefficient * imag;
};

struct ComplexVectorCoefficientByAttr : public AttributeArrays
{
   VectorCoefficient * real;
   VectorCoefficient * imag;
};

typedef Array<ComplexCoefficientByAttr*>       CmplxScaCoefArray;
typedef Array<ComplexVectorCoefficientByAttr*> CmplxVecCoefArray;

// Used for combining scalar coefficients
double prodFunc(double a, double b);

class StixBCs
{
public:
   enum BCType {DIRICHLET_BC, NEUMANN_BC, SHEATH_BC, CYL_AXIS, CURRENT_SRC};

private:
   Array<ComplexVectorCoefficientByAttr*>  dbc; // Dirichlet BC data
   Array<ComplexVectorCoefficientByAttr*>  nbc; // Neumann BC data
   Array<ComplexCoefficientByAttr*> sbc; // Sheath BC data

   Array<AttributeArrays*> cyl_axis; // Symmetry axis in cylindrical coords

   Array<ComplexVectorCoefficientByAttr*> jsrc; // Current Density Source data

   mutable Array<int>  hbc_attr; // Homogeneous Neumann BC boundary attributes
   Array<int>  dbc_attr; // Dirichlet BC boundary attributes

   std::set<int> bc_attr;

   const Array<int> & reg_attr;
   const Array<int> & bdr_attr;

public:
   StixBCs(const Array<int> & reg, const Array<int> & bdr)
      : reg_attr(reg), bdr_attr(bdr) {}

   ~StixBCs();

   static const char * GetBCTypeName(BCType bctype);

   // Enforce u = val on boundaries with attributes in bdr
   void AddDirichletBC(const Array<int> & bdr,
                       VectorCoefficient &real_val,
                       VectorCoefficient &imag_val);

   // Enforce du/dn = val on boundaries with attributes in bdr
   void AddNeumannBC(const Array<int> & bdr,
                     VectorCoefficient &real_val,
                     VectorCoefficient &imag_val);

   // Model a non-linear plasma sheath on boundaries with attributes in bdr
   void AddSheathBC(const Array<int> & bdr,
                    Coefficient &real_imped,
                    Coefficient &imag_imped);

   // Enforce phi component of u equal zero on cylindrical axis
   void AddCylindricalAxis(const Array<int> & bdr);

   // Enforce J = val on regions with attributes in reg
   void AddCurrentSrc(const Array<int> & reg,
                      VectorCoefficient &real_val,
                      VectorCoefficient &imag_val);

   const Array<ComplexVectorCoefficientByAttr*> & GetDirichletBCs() const
   { return dbc; }
   const Array<ComplexVectorCoefficientByAttr*> & GetNeumannBCs() const
   { return nbc; }
   const Array<ComplexCoefficientByAttr*> & GetSheathBCs() const { return sbc; }

   const Array<AttributeArrays*> & GetCylindricalAxis() const
   { return cyl_axis; }

   const Array<ComplexVectorCoefficientByAttr*> & GetCurrentSrcs() const
   { return jsrc; }

   const Array<int> & GetHomogeneousNeumannBDR() const;
   const Array<int> & GetDirichletBDR() const { return dbc_attr; }
};

} // namespace plasma

} // namespace mfem

#endif // MFEM_STIX_BCS
