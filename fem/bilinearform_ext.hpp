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

#ifndef MFEM_BILINEARFORM_EXT
#define MFEM_BILINEARFORM_EXT

#include "../config/config.hpp"
#include "fespace.hpp"

namespace mfem
{

class BilinearForm;

/// Element restriction operator
class ElemRestriction: public Operator
{
public:
   const FiniteElementSpace &fes;
   const int ne;
   const int vdim;
   const bool byvdim;
   const int ndofs;
   const int dof;
   const int nedofs;
   Array<int> offsets;
   Array<int> indices;
public:
   ElemRestriction(const FiniteElementSpace&);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

/// Data and methods for fully-assembled bilinear forms
class FABilinearFormExtension : public Operator
{
private:
   BilinearForm *a;
public:
   FABilinearFormExtension(BilinearForm *form);

   /// TODO
   void AddDomainIntegrator(BilinearFormIntegrator*) {}
   void Assemble() {}
   void FormSystemOperator(const Array<int> &ess_tdof_list, Operator *&A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator *&A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   ~FABilinearFormExtension() {}
};

/// Data and methods for element-assembled bilinear forms
class EABilinearFormExtension : public Operator
{
private:
   BilinearForm *a;
public:
   EABilinearFormExtension(BilinearForm *form);

   /// TODO
   void AddDomainIntegrator(BilinearFormIntegrator*) {}
   void Assemble() {}
   void FormSystemOperator(const Array<int> &ess_tdof_list, Operator *&A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator *&A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   ~EABilinearFormExtension() {}
};

/// Data and methods for partially-assembled bilinear forms
class PABilinearFormExtension : public Operator
{
private:
   BilinearForm *a;
   const FiniteElementSpace *trialFes, *testFes;
   Array<BilinearFormIntegrator*> integrators;
   mutable Vector localX, localY;
   ElemRestriction *elem_restrict;

public:
   PABilinearFormExtension(BilinearForm*);
   void AddDomainIntegrator(BilinearFormIntegrator*);
   // void AddBoundaryIntegrator(BilinearFormIntegrator*);
   // void AddInteriorFaceIntegrator(BilinearFormIntegrator*);
   // void AddBoundaryFaceIntegrator(BilinearFormIntegrator*);

   void Assemble();
   void FormSystemOperator(const Array<int> &ess_tdof_list, Operator *&A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator *&A, Vector &X, Vector &B,
                         int copy_interior = 0);
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x);

   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void Update(FiniteElementSpace *fes);

   ~PABilinearFormExtension();
};

/// Data and methods for matrix-free bilinear forms
class MFBilinearFormExtension : public Operator
{
private:
   BilinearForm *a;
public:
   MFBilinearFormExtension(BilinearForm *form);

   /// TODO
   void AddDomainIntegrator(BilinearFormIntegrator*) {}
   void Assemble() {}
   void FormSystemOperator(const Array<int> &ess_tdof_list, Operator *&A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         Operator *&A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void RecoverFEMSolution(const Vector &X, const Vector &b, Vector &x) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   ~MFBilinearFormExtension() {}
};

}

#endif
