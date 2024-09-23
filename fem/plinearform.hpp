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

#ifndef MFEM_PLINEARFORM
#define MFEM_PLINEARFORM

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pgridfunc.hpp"
#include "linearform.hpp"

namespace mfem
{

/// Class for parallel linear form
class ParLinearForm : public LinearForm
{
protected:
   ParFiniteElementSpace *pfes; ///< Points to the same object as #fes

private:
   /// Copy construction is not supported; body is undefined.
   ParLinearForm(const ParLinearForm &);

public:
   /** @brief Create an empty ParLinearForm without an associated
       ParFiniteElementSpace.

       The associated ParFiniteElementSpace can be set later using one of the
       methods: Update(ParFiniteElementSpace *) or
       Update(ParFiniteElementSpace *, Vector &, int). */
   ParLinearForm() : LinearForm() { pfes = NULL; }

   /// Create a ParLinearForm on the FE space @a *pf.
   /** The pointer @a pf is not owned by the newly constructed object. */
   ParLinearForm(ParFiniteElementSpace *pf) : LinearForm(pf) { pfes = pf; }

   /// Construct a ParLinearForm using previously allocated array @a data.
   /** The ParLinearForm does not assume ownership of @a data which is assumed
       to be of size at least `pf->GetVSize()`. Similar to the LinearForm and
       Vector constructors for externally allocated array, the pointer @a data
       can be NULL. The data array can be replaced later using the method
       SetData(). */
   ParLinearForm(ParFiniteElementSpace *pf, real_t *data) :
      LinearForm(pf, data), pfes(pf) { }

   /** @brief Create a ParLinearForm on the ParFiniteElementSpace @a *pf, using
       the same integrators as the ParLinearForm @a *plf.

       The pointer @a pf is not owned by the newly constructed object.

       The integrators in @a plf are copied as pointers and they are not owned
       by the newly constructed LinearForm. */
   ParLinearForm(ParFiniteElementSpace *pf, ParLinearForm * plf)
      : LinearForm(pf, plf) { pfes = pf; }

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** It is assumed that this object and @a rhs use ParFiniteElementSpace%s
       that have the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   ParLinearForm &operator=(const ParLinearForm &rhs)
   { return operator=((const Vector &)rhs); }

   ParFiniteElementSpace *ParFESpace() const { return pfes; }

   /// Update the object according to the given new FE space @a *pf.
   /** If the pointer @a pf is NULL (this is the default value), the FE space
       already associated with this object is used.

       This method should be called when the associated FE space #fes has been
       updated, e.g. after its associated Mesh object has been refined.

       @note This method does not perform assembly. */
   void Update(ParFiniteElementSpace *pf = NULL);

   /** @brief Associate a new FE space, @a *pf, with this object and use the
       data of @a v, offset by @a v_offset, to initialize this object's
       Vector::data.

       @note This method does not perform assembly. */
   void Update(ParFiniteElementSpace *pf, Vector &v, int v_offset);


   /** @brief Make the ParLinearForm reference external data on a new
       FiniteElementSpace. */
   /** This method changes the FiniteElementSpace associated with the
       ParLinearForm to @a *f and sets the data of the Vector @a v (plus the @a
       v_offset) as external data in the ParLinearForm.

       @note This version of the method will also perform bounds checks when the
       build option MFEM_DEBUG is enabled. */
   void MakeRef(FiniteElementSpace *f, Vector &v, int v_offset) override;

   /** @brief Make the ParLinearForm reference external data on a new
       ParFiniteElementSpace. */
   /** This method changes the ParFiniteElementSpace associated with the
       ParLinearForm to @a *pf and sets the data of the Vector @a v (plus the @a
       v_offset) as external data in the ParLinearForm.

       @note This version of the method will also perform bounds checks when the
       build option MFEM_DEBUG is enabled. */
   void MakeRef(ParFiniteElementSpace *pf, Vector &v, int v_offset);

   /// Assembles the ParLinearForm i.e. sums over all domain/bdr integrators.
   /** When @ref LinearForm::UseFastAssembly "UseFastAssembly(true)" has been
       called and the linear form assembly is compatible with device execution,
       the assembly will be executed on the device. */
   void Assemble();

   /// Return true if assembly on device is supported, false otherwise.
   bool SupportsDevice() const override;

   void AssembleSharedFaces();

   /// Assemble the vector on the true dofs, i.e. P^t v.
   void ParallelAssemble(Vector &tv);

   /// Returns the vector assembled on the true dofs, i.e. P^t v.
   HypreParVector *ParallelAssemble();

   /// Return the action of the ParLinearForm as a linear mapping.
   /** Linear forms are linear functionals which map ParGridFunction%s to the
       real numbers. This method performs this mapping which in this case is
       equivalent as an inner product of the ParLinearForm and
       ParGridFunction. */
   real_t operator()(const ParGridFunction &gf) const
   {
      return InnerProduct(pfes->GetComm(), *this, gf);
   }

   /// Assign constant values to the ParLinearForm data.
   ParLinearForm &operator=(real_t value)
   { LinearForm::operator=(value); return *this; }

   /// Copy the data from a Vector to the ParLinearForm data.
   ParLinearForm &operator=(const Vector &v)
   { LinearForm::operator=(v); return *this; }
};

}

#endif // MFEM_USE_MPI

#endif
