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

#ifndef MFEM_BACKENDS_OMP_BILINEARFORM_HPP
#define MFEM_BACKENDS_OMP_BILINEARFORM_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#include "fespace.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "../../fem/bilininteg.hpp"

namespace mfem
{

namespace omp
{

class TensorBilinearFormIntegrator
{
public:
   virtual ~TensorBilinearFormIntegrator() { }

   virtual void ReassembleOperator() = 0;

   virtual void ComputeElementMatrices(Vector &element_matrices)
   { mfem_error("TensorBilinaerFormIntegrator::ComputeElementMatrices is not overloaded"); }

   virtual void MultAdd(const Vector &x, Vector &y) const = 0;

   virtual void Mult(const Vector &x, Vector &y) const
   { y.Fill<double>(0.0); MultAdd(x, y); }
};

/// TODO: doxygen
class BilinearForm : public mfem::PBilinearForm, public mfem::Operator
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::BilinearForm *bform;

   mfem::Array<TensorBilinearFormIntegrator*> tbfi;
   bool has_assembled;

   mutable FiniteElementSpace *trial_fes, *test_fes;

   mutable Vector x_local, y_local;

   mfem::Vector *element_matrices;
   OperatorHandle mat_e;

   void TransferIntegrators();

   void ComputeElementMatrices();

   void InitRHS(const mfem::Array<int> &constraint_list,
                mfem::Vector &mfem_x, mfem::Vector &mfem_b,
                mfem::OperatorHandle &A,
                mfem::Vector &mfem_X, mfem::Vector &mfem_B,
                int copy_interior = 0) const;

public:
   /// TODO: doxygen
   BilinearForm(const Engine &e, mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf),
        // FIXME: for mixed bilinear forms
        mfem::Operator(*bf.FESpace()->GetVLayout().As<Layout>()),
        tbfi(),
        has_assembled(false),
        trial_fes(&bf.FESpace()->Get_PFESpace()->As<FiniteElementSpace>()),
        test_fes(&bf.FESpace()->Get_PFESpace()->As<FiniteElementSpace>()),
        x_local(trial_fes->GetELayout()),
        y_local(test_fes->GetELayout()),
        element_matrices(NULL),
        mat_e() { }

   /// Virtual destructor
   virtual ~BilinearForm();

   /// Return the engine as an OpenMP engine
   const Engine &OmpEngine() { return static_cast<const Engine&>(*engine); }

   /** @brief Prolongation operator from linear algebra (linear system) vectors,
       to input vectors for the operator. `NULL` means identity. */
   virtual const Operator *GetProlongation() const { return trial_fes->GetProlongation(); }

   /** @brief Restriction operator from input vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetRestriction() const  { return test_fes->GetRestriction(); }

   /// Assemble the PBilinearForm.
   /** This method is called from the method BilinearForm::Assemble() of the
       associated BilinearForm #bform.
       @returns True, if the host assembly should be skipped. */
   virtual bool Assemble();

   /// TODO: doxygen
   virtual void FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                 mfem::OperatorHandle &A);

   /// TODO: doxygen
   virtual void FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                 mfem::Vector &x, mfem::Vector &b,
                                 mfem::OperatorHandle &A, mfem::Vector &mfem_X, mfem::Vector &mfem_B,
                                 int copy_interior);

   /// TODO: doxygen
   virtual void RecoverFEMSolution(const mfem::Vector &mfem_X, const mfem::Vector &mfem_b,
                                   mfem::Vector &mfem_x);

   /// Operator application: `y=A(x)`.
   virtual void Mult(const mfem::Vector &mfem_x, mfem::Vector &mfem_y) const;

   /** @brief Action of the transpose operator: `y=A^t(x)`. The default behavior
       in class Operator is to generate an error. */
   virtual void MultTranspose(const mfem::Vector &mfem_x, mfem::Vector &mfem_y) const;
};

class ConstrainedOperator : public mfem::Operator
{
   const mfem::Operator *A;
   const bool own_A;
   const Array constraint_list;
   mutable Vector z, w;
   mutable mfem::Vector mfem_z, mfem_w;

public:
   ConstrainedOperator(mfem::Operator *A_,
                       const mfem::Array<int> &constraint_list_,
                       bool own_A_ = false);

   // Destructor: destroys the unconstrained Operator @a A if @a own_A is true.
   virtual ~ConstrainedOperator();

   /** @brief Eliminate "essential boundary condition" values specified in @a x
       from the given right-hand side @a b.

       Performs the following steps:

       z = A((0,x_b));  b_i -= z_i;  b_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   void EliminateRHS(const mfem::Vector &mfem_x, mfem::Vector &mfem_b) const;

   /** @brief Constrained operator action.

       Performs the following steps:

       z = A((x_i,0));  y_i = z_i;  y_b = x_b;

       where the "_b" subscripts denote the essential (boundary) indices/dofs of
       the vectors, and "_i" -- the rest of the entries. */
   virtual void Mult(const mfem::Vector &mfem_x, mfem::Vector &mfem_y) const;
};

} // namespace mfem::omp

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OMP)

#endif // MFEM_BACKENDS_OMP_BILINEAR_FORM_HPP
