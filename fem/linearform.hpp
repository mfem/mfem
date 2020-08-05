// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LINEARFORM
#define MFEM_LINEARFORM

#include "../config/config.hpp"
#include "lininteg.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Vector with associated FE space and LinearFormIntegrators.
class LinearForm : public Vector
{
protected:
   /// FE space on which the LinearForm lives. Not owned.
   FiniteElementSpace *fes;

   /** @brief Indicates the LinerFormIntegrator%s stored in #dlfi, #dlfi_delta,
       #blfi, and #flfi are owned by another LinearForm. */
   int extern_lfs;

   /// Set of Domain Integrators to be applied.
   Array<LinearFormIntegrator*> dlfi;

   /// Separate array for integrators with delta function coefficients.
   Array<DeltaLFIntegrator*> dlfi_delta;

   /// Set of Boundary Integrators to be applied.
   Array<LinearFormIntegrator*> blfi;
   Array<Array<int>*>           blfi_marker; ///< Entries are not owned.

   /// Set of Boundary Face Integrators to be applied.
   Array<LinearFormIntegrator*> flfi;
   Array<Array<int>*>           flfi_marker; ///< Entries are not owned.

   /// The element ids where the centers of the delta functions lie
   Array<int> dlfi_delta_elem_id;

   /// The reference coordinates where the centers of the delta functions lie
   Array<IntegrationPoint> dlfi_delta_ip;

   /// If true, the delta locations are not (re)computed during assembly.
   bool HaveDeltaLocations() { return (dlfi_delta_elem_id.Size() != 0); }

   /// Force (re)computation of delta locations.
   void ResetDeltaLocations() { dlfi_delta_elem_id.SetSize(0); }

private:
   /// Copy construction is not supported; body is undefined.
   LinearForm(const LinearForm &);

public:
   /// Creates linear form associated with FE space @a *f.
   /** The pointer @a f is not owned by the newly constructed object. */
   LinearForm(FiniteElementSpace *f) : Vector(f->GetVSize())
   { fes = f; extern_lfs = 0; UseDevice(true); }

   /** @brief Create a LinearForm on the FiniteElementSpace @a f, using the
       same integrators as the LinearForm @a lf.

       The pointer @a f is not owned by the newly constructed object.

       The integrators in @a lf are copied as pointers and they are not owned by
       the newly constructed LinearForm. */
   LinearForm(FiniteElementSpace *f, LinearForm *lf);

   /// Create an empty LinearForm without an associated FiniteElementSpace.
   /** The associated FiniteElementSpace can be set later using one of the
       methods: Update(FiniteElementSpace *) or
       Update(FiniteElementSpace *, Vector &, int). */
   LinearForm() { fes = NULL; extern_lfs = 0; UseDevice(true); }

   /// Construct a LinearForm using previously allocated array @a data.
   /** The LinearForm does not assume ownership of @a data which is assumed to
       be of size at least `f->GetVSize()`. Similar to the Vector constructor
       for externally allocated array, the pointer @a data can be NULL. The data
       array can be replaced later using the method SetData(). */
   LinearForm(FiniteElementSpace *f, double *data) : Vector(data, f->GetVSize())
   { fes = f; extern_lfs = 0; }

   /// Copy assignment. Only the data of the base class Vector is copied.
   /** It is assumed that this object and @a rhs use FiniteElementSpace%s that
       have the same size.

       @note Defining this method overwrites the implicitly defined copy
       assignment operator. */
   LinearForm &operator=(const LinearForm &rhs)
   { return operator=((const Vector &)rhs); }

   /// (DEPRECATED) Return the FE space associated with the LinearForm.
   /** @deprecated Use FESpace() instead. */
   MFEM_DEPRECATED FiniteElementSpace *GetFES() { return fes; }

   /// Read+write access to the associated FiniteElementSpace.
   FiniteElementSpace *FESpace() { return fes; }
   /// Read-only access to the associated FiniteElementSpace.
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator. Assumes ownership of @a lfi.
   void AddDomainIntegrator(LinearFormIntegrator *lfi);

   /// Adds new Boundary Integrator. Assumes ownership of @a lfi.
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi);

   /** @brief Add new Boundary Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi. The array @a bdr_attr_marker is stored
       internally as a pointer to the given Array<int> object. */
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi,
                              Array<int> &bdr_attr_marker);

   /// Adds new Boundary Face Integrator. Assumes ownership of @a lfi.
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi);

   /** @brief Add new Boundary Face Integrator, restricted to the given boundary
       attributes.

       Assumes ownership of @a lfi. The array @a bdr_attr_marker is stored
       internally as a pointer to the given Array<int> object. */
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                             Array<int> &bdr_attr_marker);

   /** @brief Access all integrators added with AddDomainIntegrator() which are
       not DeltaLFIntegrator%s or they are DeltaLFIntegrator%s with non-delta
       coefficients. */
   Array<LinearFormIntegrator*> *GetDLFI() { return &dlfi; }

   /** @brief Access all integrators added with AddDomainIntegrator() which are
       DeltaLFIntegrator%s with delta coefficients. */
   Array<DeltaLFIntegrator*> *GetDLFI_Delta() { return &dlfi_delta; }

   /// Access all integrators added with AddBoundaryIntegrator().
   Array<LinearFormIntegrator*> *GetBLFI() { return &blfi; }

   /// Access all integrators added with AddBdrFaceIntegrator().
   Array<LinearFormIntegrator*> *GetFLFI() { return &flfi; }

   /** @brief Access all boundary markers added with AddBdrFaceIntegrator().
       If no marker was specified when the integrator was added, the
       corresponding pointer (to Array<int>) will be NULL. */
   Array<Array<int>*> *GetFLFI_Marker() { return &flfi_marker; }

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assembles delta functions of the linear form
   void AssembleDelta();

   /// Update the object according to the associated FE space #fes.
   /** This method should be called when the associated FE space #fes has been
       updated, e.g. after its associated Mesh object has been refined.

       @note This method does not perform assembly. */
   void Update() { SetSize(fes->GetVSize()); ResetDeltaLocations(); }

   /// Associate a new FE space, @a *f, with this object and Update() it. */
   void Update(FiniteElementSpace *f)
   { fes = f; SetSize(f->GetVSize()); ResetDeltaLocations(); }

   /** @brief Associate a new FE space, @a *f, with this object and use the data
       of @a v, offset by @a v_offset, to initialize this object's Vector::data.

       @note This method does not perform assembly. */
   void Update(FiniteElementSpace *f, Vector &v, int v_offset);

   /// Return the action of the LinearForm as a linear mapping.
   /** Linear forms are linear functionals which map GridFunctions to
       the real numbers.  This method performs this mapping which in
       this case is equivalent as an inner product of the LinearForm
       and GridFunction. */
   double operator()(const GridFunction &gf) const { return (*this)*gf; }

   /// Redefine '=' for LinearForm = constant.
   LinearForm &operator=(double value);

   /// Copy the data from @a v.
   /** The size of @a v must be equal to the size of the associated
       FiniteElementSpace #fes. */
   LinearForm &operator=(const Vector &v);

   /// Destroys linear form.
   ~LinearForm();
};

}

#endif
