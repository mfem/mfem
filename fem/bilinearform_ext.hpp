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

   MemoryClass GetMemoryClass() const override
   { return Device::GetDeviceMemoryClass(); }

   /// Get the finite element space prolongation matrix
   const Operator *GetProlongation() const override;

   /// Get the finite element space restriction matrix
   const Operator *GetRestriction() const override;

   /// Assemble at the level given for the BilinearFormExtension subclass
   virtual void Assemble() = 0;

   void AssembleDiagonal(Vector &diag) const override
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
   /// Attributes of all mesh elements.
   const Array<int> *elem_attributes; // Not owned
   const Array<int> *bdr_face_attributes; // Not owned
   mutable Vector tmp_evec; // Work array
   mutable Vector localX, localY;
   mutable Vector int_face_X, int_face_Y;
   mutable Vector bdr_face_X, bdr_face_Y;
   mutable Vector int_face_dXdn, int_face_dYdn;
   mutable Vector bdr_face_dXdn, bdr_face_dYdn;
   const Operator *elem_restrict; // Not owned
   const FaceRestriction *int_face_restrict_lex; // Not owned
   const FaceRestriction *bdr_face_restrict_lex; // Not owned

public:
   PABilinearFormExtension(BilinearForm*);

   void Assemble() override;
   void AssembleDiagonal(Vector &diag) const override;
   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A) override;
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) override;
   void Mult(const Vector &x, Vector &y) const override
   { MultInternal(x,y); }
   void AbsMult(const Vector &x, Vector &y) const override
   { MultInternal(x,y, true); }
   void MultTranspose(const Vector &x, Vector &y) const override;
   void Update() override;

protected:
   void SetupRestrictionOperators(const L2FaceValues m);
   void MultInternal(const Vector &x, Vector &y,
                     const bool useAbs = false) const;

   /// @brief Accumulate the action (or transpose) of the integrator on @a x
   /// into @a y, taking into account the (possibly null) @a markers array.
   ///
   /// If @a markers is non-null, then only those elements or boundary elements
   /// whose attribute is marked in the markers array will be added to @a y.
   ///
   /// @param integ The integrator (domain, boundary, or boundary face).
   /// @param x Input E-vector.
   /// @param markers Marked attributes (possibly null, meaning all attributes).
   /// @param attributes Array of element or boundary element attributes.
   /// @param transpose Compute the action or transpose of the integrator .
   /// @param y Output E-vector
   /// @param useAbs Apply absolute-value operator
   void AddMultWithMarkers(const BilinearFormIntegrator &integ,
                           const Vector &x,
                           const Array<int> *markers,
                           const Array<int> &attributes,
                           const bool transpose,
                           Vector &y,
                           const bool useAbs = false) const;

   /// @brief Performs the same function as AddMultWithMarkers, but takes as
   /// input and output face normal derivatives.
   ///
   /// This is required when the integrator requires face normal derivatives,
   /// for example, DGDiffusionIntegrator.
   ///
   /// This is called when the integrator's member function
   /// BilinearFormIntegrator::RequiresFaceNormalDerivatives() returns true.
   void AddMultNormalDerivativesWithMarkers(
      const BilinearFormIntegrator &integ,
      const Vector &x,
      const Vector &dxdn,
      const Array<int> *markers,
      const Array<int> &attributes,
      Vector &y,
      Vector &dydn) const;
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

   void Assemble() override;

   void Mult(const Vector &x, Vector &y) const override
   { MultInternal(x, y, false); }
   void AbsMult(const Vector &x, Vector &y) const override
   { MultInternal(x, y, false, true); }
   void MultTranspose(const Vector &x, Vector &y) const override
   { MultInternal(x, y, true); }
   void AbsMultTranspose(const Vector &x, Vector &y) const override
   { MultInternal(x, y, true, true); }

   /// @brief Populates @a element_matrices with the element matrices.
   ///
   /// The element matrices are converted from row-major (how they are stored in
   /// @a ea_data) to column-major format.
   ///
   /// If @a ordering is ElementDofOrdering::NATIVE, then the matrices are
   /// reordered from the lexicographic ordering used internally.
   void GetElementMatrices(DenseTensor &element_matrices,
                           ElementDofOrdering ordering,
                           bool add_bdr);

   // This method needs to be public due to 'nvcc' restriction.
   void MultInternal(const Vector &x, Vector &y, const bool useTranspose,
                     const bool useAbs = false) const;
};

/// Data and methods for fully-assembled bilinear forms
class FABilinearFormExtension : public EABilinearFormExtension
{
private:
   SparseMatrix *mat;
   mutable Vector dg_x, dg_y;

public:
   FABilinearFormExtension(BilinearForm *form);

   void Assemble() override;
   void RAP(OperatorHandle &A);
   /** @note Always does `DIAG_ONE` policy to be consistent with
       `Operator::FormConstrainedSystemOperator`. */
   void EliminateBC(const Array<int> &ess_dofs, OperatorHandle &A);
   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A) override;
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) override;
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;

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

   void Assemble() override;
   void AssembleDiagonal(Vector &diag) const override;
   void FormSystemMatrix(const Array<int> &ess_tdof_list,
                         OperatorHandle &A) override;
   void FormLinearSystem(const Array<int> &ess_tdof_list,
                         Vector &x, Vector &b,
                         OperatorHandle &A, Vector &X, Vector &B,
                         int copy_interior = 0) override;
   void Mult(const Vector &x, Vector &y) const override;
   void MultTranspose(const Vector &x, Vector &y) const override;
   void Update() override;
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

   MemoryClass GetMemoryClass() const override
   { return Device::GetMemoryClass(); }

   /// Get the finite element space prolongation matrix
   const Operator *GetProlongation() const override;

   /// Get the finite element space restriction matrix
   const Operator *GetRestriction() const override;

   /// Get the output finite element space restriction matrix
   const Operator *GetOutputProlongation() const override;

   /// Get the output finite element space restriction matrix
   const Operator *GetOutputRestriction() const override;

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
                        Vector &y, Vector &localY, const real_t c) const;

public:
   PAMixedBilinearFormExtension(MixedBilinearForm *form);

   /// Partial assembly of all internal integrators
   void Assemble() override;
   /**
      @brief Setup OperatorHandle A to contain constrained linear operator

      OperatorHandle A contains matrix-free constrained operator formed for RAP
      system where ess_tdof_list are in trial space and eliminated from
      "columns" of A.
   */
   void FormRectangularSystemOperator(const Array<int> &trial_tdof_list,
                                      const Array<int> &test_tdof_list,
                                      OperatorHandle &A) override;
   /**
      Setup OperatorHandle A to contain constrained linear operator and
      eliminate columns corresponding to essential dofs from system,
      updating RHS B vector with the results.
   */
   void FormRectangularLinearSystem(const Array<int> &trial_tdof_list,
                                    const Array<int> &test_tdof_list,
                                    Vector &x, Vector &b,
                                    OperatorHandle &A, Vector &X, Vector &B) override;
   /// y = A*x
   void Mult(const Vector &x, Vector &y) const override;
   /// y += c*A*x
   void AddMult(const Vector &x, Vector &y, const real_t c=1.0) const override;
   /// y = A^T*x
   void MultTranspose(const Vector &x, Vector &y) const override;
   /// y += c*A^T*x
   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t c=1.0) const override;
   /// Assemble the diagonal of ADA^T for a diagonal vector D.
   void AssembleDiagonal_ADAt(const Vector &D, Vector &diag) const override;

   /// Update internals for when a new MixedBilinearForm is given to this class
   void Update() override;
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
   void Assemble() override;

   void AddMult(const Vector &x, Vector &y, const real_t c=1.0) const override;

   void AddMultTranspose(const Vector &x, Vector &y,
                         const real_t c=1.0) const override;

   void FormRectangularSystemOperator(const Array<int>&, const Array<int>&,
                                      OperatorHandle& A) override;

   const Operator * GetOutputRestrictionTranspose() const override;

private:
   Vector test_multiplicity;
};

}

#endif
