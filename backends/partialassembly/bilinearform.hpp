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

#ifndef MFEM_BACKENDS_PA_BILINEARFORM_HPP
#define MFEM_BACKENDS_PA_BILINEARFORM_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#include "fespace.hpp"
#include "array.hpp"
#include "vector.hpp"
#include "../../fem/bilininteg.hpp"
#include "partialassemblykernel.hpp"
#include "integrator.hpp"

namespace mfem
{

namespace pa
{

/**
*  A backend BilinearForm for cpu that mostly duplicate the mfem::BilinearForm code...
*/
class BilinearForm : public mfem::PBilinearForm, public mfem::Operator
{
protected:
   //
   // Inherited fields
   //
   // SharedPtr<const mfem::Engine> engine;
   // mfem::BilinearForm *bform;

   mfem::Array<TensorBilinearFormIntegrator*> tbfi;
   mfem::Array<PAIntegrator<Vector<double>>*> pabfi;
   bool has_assembled;

   mutable FiniteElementSpace *trial_fes, *test_fes;

   mutable Vector<double> x_local, y_local;

   /**
   *  This function transfers the mfem::BilinearFormIntegrator to backend integrators.
   */
   void TransferIntegrators(mfem::Array<mfem::BilinearFormIntegrator*>& bfi);

   void InitRHS(const mfem::Array<int> &constraint_list,
                mfem::Vector &mfem_x, mfem::Vector &mfem_b,
                mfem::OperatorHandle& A,
                mfem::Vector &mfem_X, mfem::Vector &mfem_B,
                int copy_interior = 0) const;

   void AddIntegrator(TensorBilinearFormIntegrator* integrator) { tbfi.Append(integrator); }

public:
   BilinearForm(const Engine &e, mfem::BilinearForm &bf)
      : mfem::PBilinearForm(e, bf),
        // FIXME: for mixed bilinear forms
        mfem::Operator(*bf.FESpace()->GetVLayout().As<Layout>()),
        tbfi(),
        pabfi(),
        has_assembled(false),
        trial_fes(&bf.FESpace()->Get_PFESpace()->As<FiniteElementSpace>()),
        test_fes(&bf.FESpace()->Get_PFESpace()->As<FiniteElementSpace>()),
        x_local(trial_fes->GetELayout()),
        y_local(test_fes->GetELayout()) { }

   /// Virtual destructor
   virtual ~BilinearForm();

   /// Return the engine as a backend engine
   const Engine &GetEngine() { return static_cast<const Engine&>(*engine); }

   /** @brief Prolongation operator from linear algebra (linear system) vectors,
       to input vectors for the operator. `NULL` means identity. */
   virtual const Operator *GetProlongation() const { return bform->GetProlongation(); }

   /** @brief Restriction operator from input vectors for the operator to linear
       algebra (linear system) vectors. `NULL` means identity. */
   virtual const Operator *GetRestriction() const  { return bform->GetRestriction(); }

   /// Assemble the PBilinearForm.
   /** This method is called from the method BilinearForm::Assemble() of the
       associated BilinearForm #bform.
       @returns True, if the host assembly should be skipped. */
   virtual bool Assemble();

   virtual void FormSystemMatrix(const mfem::Array<int> &ess_tdof_list,
                                 mfem::OperatorHandle &A);

   virtual void FormLinearSystem(const mfem::Array<int> &ess_tdof_list,
                                 mfem::Vector &x, mfem::Vector &b,
                                 mfem::OperatorHandle &A, mfem::Vector &mfem_X, mfem::Vector &mfem_B,
                                 int copy_interior);

   virtual void RecoverFEMSolution(const mfem::Vector &mfem_X,
                                   const mfem::Vector &mfem_b,
                                   mfem::Vector &mfem_x);

   /// Operator application: `y=A(x)`.
   virtual void Mult(const mfem::Vector &mfem_x, mfem::Vector &mfem_y) const;

   /** @brief Action of the transpose operator: `y=A^t(x)`. The default behavior
       in class Operator is to generate an error. */
   virtual void MultTranspose(const mfem::Vector &mfem_x,
                              mfem::Vector &mfem_y) const;
};

/**
*  Duplicates the mfem::ConstrainedOperator in the backend
*/
class ConstrainedOperator : public mfem::Operator
{
   const mfem::Operator *A;
   const bool own_A;
   Array constraint_list;
   mutable Vector<double> z, w;
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

} // namespace mfem::pa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_PA)

#endif // MFEM_BACKENDS_PA_BILINEAR_FORM_HPP
