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

#ifndef NONLINEARFORM_EXT_HPP
#define NONLINEARFORM_EXT_HPP

#include "../config/config.hpp"
#include "fespace.hpp"

namespace mfem
{

class NonlinearForm;
class NonlinearFormIntegrator;

/** @brief Class extending the NonlinearForm class to support the different
    AssemblyLevel%s. */
/** This class represents the action of the NonlinearForm as an L-to-L operator,
    i.e. both the input and output Vectors are L-vectors (GridFunction-size
    vectors). Essential boundary conditions are NOT applied to the action of the
    operator. */
class NonlinearFormExtension : public Operator
{
protected:
   const NonlinearForm *nlf; ///< Not owned

public:
   NonlinearFormExtension(const NonlinearForm*);

   /// Assemble at the AssemblyLevel of the subclass.
   virtual void Assemble() = 0;

   /** @brief Return the gradient as an L-to-L Operator. The input @a x must be
       an L-vector (i.e. GridFunction-size vector). */
   /** Essential boundary conditions are NOT applied to the returned operator.

       The returned gradient Operator defines the virtual method GetProlongation
       which enables support for the method FormSystemOperator to define the
       matrix-free global true-dof gradient with imposed boundary conditions. */
   virtual Operator &GetGradient(const Vector &x) const = 0;

   /// Compute the local (to the MPI rank) energy of the L-vector state @a x.
   virtual real_t GetGridFunctionEnergy(const Vector &x) const = 0;

   /// Called by NonlinearForm::Update() to reflect changes in the FE space.
   virtual void Update() = 0;
};

/// Data and methods for partially-assembled nonlinear forms
class PANonlinearFormExtension : public NonlinearFormExtension
{
private:
   class PAGradient : public Operator
   {
   protected:
      const PANonlinearFormExtension &ext;

   public:
      /// Assumes that @a g is a ldof Vector.
      PAGradient(const PANonlinearFormExtension &ext);

      /// Assumes that @a x and @a y are ldof Vector%s.
      virtual void Mult(const Vector &x, Vector &y) const;

      /// Assumes that @a g is an ldof Vector.
      void AssembleGrad(const Vector &g);

      /// Assemble the diagonal of the gradient into the ldof Vector @a diag.
      virtual void AssembleDiagonal(Vector &diag) const;

      /** @brief Define the prolongation Operator for use with methods like
          FormSystemOperator. */
      virtual const Operator *GetProlongation() const
      {
         return ext.fes.GetProlongationMatrix();
      }

      /** @brief Called by PANonlinearFormExtension::Update to reflect changes
          in the FiniteElementSpace. */
      void Update();
   };

protected:
   mutable Vector xe, ye;
   const FiniteElementSpace &fes;
   const Array<NonlinearFormIntegrator*> &dnfi;
   const Operator *elemR; // not owned
   mutable PAGradient paGrad;
   const ElementDofOrdering edf;

public:
   PANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf = ElementDofOrdering::LEXICOGRAPHIC);

   /// Prepare the PANonlinearFormExtension for evaluation with Mult().
   /** This method must be called before the first call to Mult(), when the mesh
       coordinates are changed, or some coefficients in the integrators need to
       be re-evaluated (this is NonlinearFormIntegrator-dependent). */
   void Assemble();

   /// Perform the action of the PANonlinearFormExtension.
   /** Both the input, @a x, and output, @a y, vectors are L-vectors, i.e.
       GridFunction-size vectors. */
   void Mult(const Vector &x, Vector &y) const;

   /** @brief Return the gradient as an L-to-L Operator. The input @a x must be
       an L-vector (i.e. GridFunction-size vector). */
   /** Essential boundary conditions are NOT applied to the returned operator.

       The returned gradient Operator defines the virtual method GetProlongation
       which enables support for the method FormSystemOperator to define the
       matrix-free global true-dof gradient with imposed boundary conditions. */
   virtual Operator &GetGradient(const Vector &x) const;

   /// Compute the local (to the MPI rank) energy of the L-vector state @a x.
   real_t GetGridFunctionEnergy(const Vector &x) const override;

   /// Called by NonlinearForm::Update() to reflect changes in the FE space.
   void Update() override;
};

class DenseTensor;
class LibBatchMult;
/// Data and methods for element-assembled bilinear forms
class EANonlinearFormExtension : public PANonlinearFormExtension
{
#if defined(MFEM_USE_CUDA)
public:
#else
private:
#endif
   class EAGradient : public Operator
   {
   protected:
      const EANonlinearFormExtension &ext;

   public:
      /// Assumes that @a g is a ldof Vector.
      EAGradient(const EANonlinearFormExtension &ext);

      /// Assumes that @a x and @a y are ldof Vector%s.
      virtual void Mult(const Vector &x, Vector &y) const;

      /// Assumes that @a g is an ldof Vector.
      void AssembleGrad(const Vector &g);

      /// Assemble the diagonal of the gradient into the ldof Vector @a diag.
      virtual void AssembleDiagonal(Vector &diag) const;

      /** @brief Define the prolongation Operator for use with methods like
          FormSystemOperator. */
      virtual const Operator *GetProlongation() const
      {
         return ext.fes.GetProlongationMatrix();
      }

      /** @brief Called by PANonlinearFormExtension::Update to reflect changes
          in the FiniteElementSpace. */
      void Update();
   };

#if defined(MFEM_USE_CUDA)
public:
#else
protected:
#endif
   mutable int ne;
   mutable int elemDofs;
   // The element matrices are stored row major
   mutable Vector ea_data;
   mutable EAGradient eaGrad;
   mutable DenseTensor eaGradDT;
   LibBatchMult* batchMult = nullptr;

public:
   EANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf = ElementDofOrdering::LEXICOGRAPHIC);

   using PANonlinearFormExtension::Assemble;
   using PANonlinearFormExtension::Mult;
   using PANonlinearFormExtension::GetGridFunctionEnergy;


   /** @brief Return the gradient as an L-to-L Operator. The input @a x must be
       an L-vector (i.e. GridFunction-size vector). */
   /** Essential boundary conditions are NOT applied to the returned operator.

       The returned gradient Operator defines the virtual method GetProlongation
       which enables support for the method FormSystemOperator to define the
       matrix-free global true-dof gradient with imposed boundary conditions. */
   virtual Operator &GetGradient(const Vector &x) const;
   ~EANonlinearFormExtension();
};

/// Data and methods for fully-assembled bilinear forms
class FANonlinearFormExtension : public EANonlinearFormExtension
{

private:
   class FAGradient : public Operator
   {
   protected:
      const FANonlinearFormExtension &ext;

   public:
      /// Assumes that @a g is a ldof Vector.
      FAGradient(const FANonlinearFormExtension &ext);

      /// Assumes that @a x and @a y are ldof Vector%s.
      virtual void Mult(const Vector &x, Vector &y) const;

      /// Assumes that @a g is an ldof Vector.
      void AssembleGrad(const Vector &g);

      /// Assemble the diagonal of the gradient into the ldof Vector @a diag.
      virtual void AssembleDiagonal(Vector &diag) const;

      /** @brief Define the prolongation Operator for use with methods like
          FormSystemOperator. */
      virtual const Operator *GetProlongation() const
      {
         return ext.fes.GetProlongationMatrix();
      }

      /** @brief Called by PANonlinearFormExtension::Update to reflect changes
          in the FiniteElementSpace. */
      void Update();
   };

protected:
   mutable SparseMatrix *mat;
   mutable FAGradient faGrad;

public:
   FANonlinearFormExtension(const NonlinearForm *nlf, const ElementDofOrdering edf = ElementDofOrdering::LEXICOGRAPHIC);

   using PANonlinearFormExtension::Assemble;
   using PANonlinearFormExtension::Mult;
   using PANonlinearFormExtension::GetGridFunctionEnergy;

   /** @brief Return the gradient as an L-to-L Operator. The input @a x must be
       an L-vector (i.e. GridFunction-size vector). */
   /** Essential boundary conditions are NOT applied to the returned operator.

       The returned gradient Operator defines the virtual method GetProlongation
       which enables support for the method FormSystemOperator to define the
       matrix-free global true-dof gradient with imposed boundary conditions. */
   virtual Operator &GetGradient(const Vector &x) const;

};

/// Data and methods for unassembled nonlinear forms
class MFNonlinearFormExtension : public NonlinearFormExtension
{
protected:
   const FiniteElementSpace &fes; // Not owned
   mutable Vector localX, localY;
   const Operator *elem_restrict_lex; // Not owned
   const ElementDofOrdering edf;

public:
   MFNonlinearFormExtension(const NonlinearForm*, const ElementDofOrdering edf_ = ElementDofOrdering::LEXICOGRAPHIC);

   /// Prepare the MFNonlinearFormExtension for evaluation with Mult().
   void Assemble() override;

   /// Perform the action of the MFNonlinearFormExtension.
   /** Both the input, @a x, and output, @a y, vectors are L-vectors, i.e.
       GridFunction-size vectors. */
   void Mult(const Vector &x, Vector &y) const override;

   Operator &GetGradient(const Vector &x) const override
   {
      MFEM_ABORT("TODO");
      return *const_cast<MFNonlinearFormExtension*>(this);
   }

   real_t GetGridFunctionEnergy(const Vector &x) const override
   {
      MFEM_ABORT("TODO");
      return 0.0;
   }

   /// Called by NonlinearForm::Update() to reflect changes in the FE space.
   void Update() override;
};

}
#endif // NONLINEARFORM_EXT_HPP
