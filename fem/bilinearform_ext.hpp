// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILINEARFORM_EXT
#define MFEM_BILINEARFORM_EXT

#include "../config/config.hpp"
#include "fespace.hpp"
#include "../general/device.hpp"

namespace mfem
{

class BilinearForm;
class MixedBilinearForm;
class DiscreteLinearOperator;

/// Class extending the BilinearForm class to support different AssemblyLevels.
/**  FA - Full Assembly
     EA - Element Assembly
     PA - Partial Assembly
     MF - Matrix Free
*/
class BilinearFormExtension : public Operator
{
protected:
   BilinearForm *a; ///< Not owned

public:
   BilinearFormExtension(BilinearForm *form);

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetDeviceMemoryClass(); }

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const;

   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const;

   /// Assemble at the level given for the BilinearFormExtension subclass
   virtual void Assemble() = 0;

   virtual void AssembleDiagonal(Vector &diag) const
   {
      MFEM_ABORT("AssembleDiagonal not implemented for this assembly level!");
   }

   virtual void Update() = 0;
};

/// Data and methods for matrix-free bilinear forms
class MFBilinearFormExtension : public BilinearFormExtension
{
protected:
   const FiniteElementSpace *fes; // Not owned
   mutable Vector local_x, local_y, temp_y;
   mutable Vector int_face_x, int_face_y;
   mutable Vector bdr_face_x, bdr_face_y;
   const ElementRestriction *elem_restrict; // Not owned
   const FaceRestriction *int_face_restrict_lex; // Not owned
   const FaceRestriction *bdr_face_restrict_lex; // Not owned

public:
   MFBilinearFormExtension(BilinearForm *form);

   void Assemble();
   void AssembleDiagonal(Vector &diag) const;
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y, const double c = 1.0) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void AddMultTranspose(const Vector &x, Vector &y, const double c = 1.0) const;
   void Update();

protected:
   void SetupRestrictionOperators(const L2FaceValues m);
};

/// Data and methods for partially-assembled bilinear forms
class PABilinearFormExtension : public MFBilinearFormExtension
{
public:
   PABilinearFormExtension(BilinearForm *form);

   void Assemble();
   void AssembleDiagonal(Vector &diag) const;
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y, const double c = 1.0) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void AddMultTranspose(const Vector &x, Vector &y, const double c = 1.0) const;
};

/// Data and methods for element-assembled bilinear forms
class EABilinearFormExtension : public PABilinearFormExtension
{
protected:
   const bool factorize_face_terms;
   int ne, elem_dofs;
   Vector ea_data;  // The element matrices are stored row major
   int nf_int, nf_bdr, face_dofs;
   Vector ea_data_int, ea_data_ext, ea_data_bdr;

public:
   EABilinearFormExtension(BilinearForm *form);

   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

/// Data and methods for fully-assembled bilinear forms
class FABilinearFormExtension : public EABilinearFormExtension
{
private:
   SparseMatrix *mat;
   mutable Vector dg_x, dg_y;

public:
   FABilinearFormExtension(BilinearForm *form);

   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;

   /** DGMult and DGMultTranspose use the extended L-vector to perform the
       computation. */
   void DGMult(const Vector &x, Vector &y) const;
   void DGMultTranspose(const Vector &x, Vector &y) const;
};

/// Class extending the MixedBilinearForm class to support different AssemblyLevels.
/**  FA - Full Assembly
     EA - Element Assembly
     PA - Partial Assembly
     MF - Matrix Free
*/
class MixedBilinearFormExtension : public Operator
{
protected:
   MixedBilinearForm *a; ///< Not owned

public:
   MixedBilinearFormExtension(MixedBilinearForm *form);

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetDeviceMemoryClass(); }

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const;

   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const;

   /// Get the output finite element space restriction matrix
   virtual const Operator *GetOutputProlongation() const;

   /// Get the output finite element space restriction matrix
   virtual const Operator *GetOutputRestriction() const;

   /// Assemble at the level given for the BilinearFormExtension subclass
   virtual void Assemble() = 0;

   virtual void AssembleDiagonal_ADAt(const Vector &D, Vector &diag) const
   {
      MFEM_ABORT("AssembleDiagonal_ADAt not implemented for this assembly level!");
   }

   virtual void Update() = 0;
};

/// Data and methods for matrix-free mixed bilinear forms
class MFMixedBilinearFormExtension : public MixedBilinearFormExtension
{
protected:
   const FiniteElementSpace *trial_fes, *test_fes; // Not owned
   mutable Vector local_trial, local_test, temp_trial, temp_test;
   mutable Vector int_face_trial, int_face_test, int_face_y;
   mutable Vector bdr_face_trial, bdr_face_test, bdr_face_y;
   const ElementRestriction *elem_restrict_trial; // Not owned
   const ElementRestriction *elem_restrict_test;  // Not owned
   const FaceRestriction *int_face_restrict_lex_trial; // Not owned
   const FaceRestriction *int_face_restrict_lex_test;  // Not owned
   const FaceRestriction *bdr_face_restrict_lex_trial; // Not owned
   const FaceRestriction *bdr_face_restrict_lex_test;  // Not owned

public:
   MFMixedBilinearFormExtension(MixedBilinearForm *form);

   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y, const double c = 1.0) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void AddMultTranspose(const Vector &x, Vector &y, const double c = 1.0) const;
   void Update();

protected:
   void SetupRestrictionOperators(const L2FaceValues m);
};

/// Data and methods for partially-assembled mixed bilinear forms
class PAMixedBilinearFormExtension : public MFMixedBilinearFormExtension
{
public:
   PAMixedBilinearFormExtension(MixedBilinearForm *form);

   void Assemble();
   void AssembleDiagonal_ADAt(const Vector &D, Vector &diag) const;
   void AddMult(const Vector &x, Vector &y, const double c = 1.0) const;
   void AddMultTranspose(const Vector &x, Vector &y, const double c = 1.0) const;
};

/// Data and methods for partially-assembled discrete linear operators
class PADiscreteLinearOperatorExtension : public PAMixedBilinearFormExtension
{
private:
   Vector test_multiplicity;

public:
   PADiscreteLinearOperatorExtension(DiscreteLinearOperator *linop);

   void Assemble();
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y, const double c = 1.0) const;
   void AddMultTranspose(const Vector &x, Vector &y, const double c = 1.0) const;
};

}

#endif
