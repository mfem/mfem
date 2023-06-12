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
     PA - Partial Assembly
     EA - Element Assembly
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

   virtual void FormSystemMatrix(const Array<int> &ess_tdof_list,
                                 OperatorHandle &A) = 0;
   virtual void FormLinearSystem(const Array<int> &ess_tdof_list,
                                 Vector &x, Vector &b,
                                 OperatorHandle &A, Vector &X, Vector &B,
                                 int copy_interior = 0) = 0;
   virtual void Update() = 0;
};

/// Data and methods for partially-assembled bilinear forms
class PABilinearFormExtension : public BilinearFormExtension
{
protected:
   const FiniteElementSpace *trial_fes, *test_fes; // Not owned
   mutable Vector localX, localY;
   mutable Vector int_face_X, int_face_Y;
   mutable Vector bdr_face_X, bdr_face_Y;
   const Operator *elem_restrict; // Not owned
   const FaceRestriction *int_face_restrict_lex; // Not owned
   const FaceRestriction *bdr_face_restrict_lex; // Not owned

public:
   PABilinearFormExtension(BilinearForm*);

   void Assemble();
   void AssembleDiagonal(Vector &diag) const;
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void Update();

protected:
   void SetupRestrictionOperators(const L2FaceValues m);
};

/// Data and methods for element-assembled bilinear forms
class EABilinearFormExtension : public PABilinearFormExtension
{
protected:
   int ne;
   int elemDofs;
   // The element matrices are stored row major
   Vector ea_data;
   int nf_int, nf_bdr;
   int faceDofs;
   Vector ea_data_int, ea_data_ext, ea_data_bdr;
   bool factorize_face_terms;

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
   void RAP(OperatorHandle &A);
   /** @note Always does `DIAG_ONE` policy to be consistent with
       `Operator::FormConstrainedSystemOperator`. */
   void EliminateBC(const Array<int> &ess_dofs, OperatorHandle &A);
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;

   /** DGMult and DGMultTranspose use the extended L-vector to perform the
       computation. */
   void DGMult(const Vector &x, Vector &y) const;
   void DGMultTranspose(const Vector &x, Vector &y) const;
};

/// Data and methods for matrix-free bilinear forms
class MFBilinearFormExtension : public BilinearFormExtension
{
protected:
   const FiniteElementSpace *trial_fes, *test_fes; // Not owned
   mutable Vector localX, localY;
   mutable Vector int_face_X, int_face_Y;
   mutable Vector bdr_face_X, bdr_face_Y;
   const Operator *elem_restrict; // Not owned
   const FaceRestriction *int_face_restrict_lex; // Not owned
   const FaceRestriction *bdr_face_restrict_lex; // Not owned

public:
   MFBilinearFormExtension(BilinearForm *form);

   void Assemble();
   void AssembleDiagonal(Vector &diag) const;
   void FormSystemMatrix(const Array<int> &ess_tdof_list, OperatorHandle &A);
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
   void Update();
};

/// Class extending the MixedBilinearForm class to support different AssemblyLevels.
/**  FA - Full Assembly
     PA - Partial Assembly
     EA - Element Assembly
     MF - Matrix Free
*/
class MixedBilinearFormExtension : public Operator
{
protected:
   MixedBilinearForm *a; ///< Not owned

public:
   MixedBilinearFormExtension(MixedBilinearForm *form);

   virtual MemoryClass GetMemoryClass() const
   { return Device::GetMemoryClass(); }

   /// Get the finite element space prolongation matrix
   virtual const Operator *GetProlongation() const;

   /// Get the finite element space restriction matrix
   virtual const Operator *GetRestriction() const;

   /// Get the output finite element space restriction matrix
   virtual const Operator *GetOutputProlongation() const;

   /// Get the output finite element space restriction matrix
   virtual const Operator *GetOutputRestriction() const;

   virtual void Assemble() = 0;
   virtual void FormRectangularSystemOperator(const Array<int> &trial_tdof_list,
                                              const Array<int> &test_tdof_list,
                                              OperatorHandle &A) = 0;
   virtual void FormRectangularLinearSystem(const Array<int> &trial_tdof_list,
                                            const Array<int> &test_tdof_list,
                                            Vector &x, Vector &b,
                                            OperatorHandle &A, Vector &X, Vector &B) = 0;

   virtual void AssembleDiagonal_ADAt(const Vector &D, Vector &diag) const = 0;

   virtual void Update() = 0;
};

/// Data and methods for partially-assembled mixed bilinear forms
class PAMixedBilinearFormExtension : public MixedBilinearFormExtension
{
protected:
   const FiniteElementSpace *trial_fes, *test_fes; // Not owned
   mutable Vector localTrial, localTest, tempY;
   const Operator *elem_restrict_trial; // Not owned
   const Operator *elem_restrict_test;  // Not owned

   /// Helper function to set up inputs/outputs for Mult or MultTranspose
   void SetupMultInputs(const Operator *elem_restrict_x,
                        const Vector &x, Vector &localX,
                        const Operator *elem_restrict_y,
                        Vector &y, Vector &localY, const double c) const;

public:
   PAMixedBilinearFormExtension(MixedBilinearForm *form);

   /// Partial assembly of all internal integrators
   void Assemble();
   /**
      @brief Setup OperatorHandle A to contain constrained linear operator

      OperatorHandle A contains matrix-free constrained operator formed for RAP
      system where ess_tdof_list are in trial space and eliminated from
      "columns" of A.
   */
   void FormRectangularSystemOperator(const Array<int> &trial_tdof_list,
                                      const Array<int> &test_tdof_list,
                                      OperatorHandle &A);
   /**
      Setup OperatorHandle A to contain constrained linear operator and
      eliminate columns corresponding to essential dofs from system,
      updating RHS B vector with the results.
   */
   void FormRectangularLinearSystem(const Array<int> &trial_tdof_list,
                                    const Array<int> &test_tdof_list,
                                    Vector &x, Vector &b,
                                    OperatorHandle &A, Vector &X, Vector &B);
   /// y = A*x
   void Mult(const Vector &x, Vector &y) const;
   /// y += c*A*x
   void AddMult(const Vector &x, Vector &y, const double c=1.0) const;
   /// y = A^T*x
   void MultTranspose(const Vector &x, Vector &y) const;
   /// y += c*A^T*x
   void AddMultTranspose(const Vector &x, Vector &y, const double c=1.0) const;
   /// Assemble the diagonal of ADA^T for a diagonal vector D.
   void AssembleDiagonal_ADAt(const Vector &D, Vector &diag) const;

   /// Update internals for when a new MixedBilinearForm is given to this class
   void Update();
};


/**
   @brief Partial assembly extension for DiscreteLinearOperator

   This acts very much like PAMixedBilinearFormExtension, but its
   FormRectangularSystemOperator implementation emulates 'Set' rather than
   'Add' in the assembly case.
*/
class PADiscreteLinearOperatorExtension : public PAMixedBilinearFormExtension
{
public:
   PADiscreteLinearOperatorExtension(DiscreteLinearOperator *linop);

   /// Partial assembly of all internal integrators
   void Assemble();

   void AddMult(const Vector &x, Vector &y, const double c=1.0) const;

   void AddMultTranspose(const Vector &x, Vector &y, const double c=1.0) const;

   void FormRectangularSystemOperator(const Array<int>&, const Array<int>&,
                                      OperatorHandle& A);

   const Operator * GetOutputRestrictionTranspose() const;

private:
   Vector test_multiplicity;
};

}

#endif
