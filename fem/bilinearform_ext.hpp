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


class BilinearFormExtension : public Operator
{
protected:
   BilinearForm *a; ///< Not owned

public:

   mutable DenseTensor Me;

   BilinearFormExtension(BilinearForm *form);

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const;

   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const;

   virtual void Assemble() = 0;
   virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A) = 0;
   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b,
                                 OperatorHandle &A, Vector &X, Vector &B,
                                 int copy_interior = 0) = 0;
   virtual void Update() = 0;
};

/// Data and methods for fully-assembled bilinear forms
class FABilinearFormExtension : public BilinearFormExtension
{
protected:

   const FiniteElementSpace *trialFes, *testFes;
   mutable Vector localX, localY;
   ElemRestriction *elem_restrict;

public:
   FABilinearFormExtension(BilinearForm *form);

   /// TODO
   void Assemble();
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   void Update () {};
   ~FABilinearFormExtension() {}
};

/// Data and methods for element-assembled bilinear forms
class EABilinearFormExtension : public BilinearFormExtension
{
public:
   EABilinearFormExtension(BilinearForm *form)
      : BilinearFormExtension(form) { }

   /// TODO
   void Assemble() {}
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   ~EABilinearFormExtension() {}
};

/// Data and methods for partially-assembled bilinear forms
class PABilinearFormExtension : public BilinearFormExtension
{
protected:
   const FiniteElementSpace *trialFes, *testFes;
   mutable Vector localX, localY;
   ElemRestriction *elem_restrict;

public:
   PABilinearFormExtension(BilinearForm*);

   void Assemble();
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);

   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void Update();

   ~PABilinearFormExtension();
};

/// Data and methods for matrix-free bilinear forms
class MFBilinearFormExtension : public BilinearFormExtension
{
public:
   MFBilinearFormExtension(BilinearForm *form)
      : BilinearFormExtension(form) { }

   /// TODO
   void Assemble() {}
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A) {}
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) {}
   void Mult(const Vector &x, Vector &y) const {}
   void MultTranspose(const Vector &x, Vector &y) const {}
   ~MFBilinearFormExtension() {}
};

}

#endif
