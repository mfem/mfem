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

#ifndef MFEM_LINEARFORM
#define MFEM_LINEARFORM

#include "../config/config.hpp"
#include "lininteg.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Class for linear form - Vector with associated FE space and LFIntegrators.
class LinearForm : public Vector
{
private:
   /// FE space on which LF lives.
   FiniteElementSpace * fes;

   /// Set of Domain Integrators to be applied.
   Array<LinearFormIntegrator*> dlfi;

   /// Separate array for integrators with delta function coefficients.
   Array<DeltaLFIntegrator*> dlfi_delta;

   /// Set of Boundary Integrators to be applied.
   Array<LinearFormIntegrator*> blfi;
   Array<Array<int>*>           blfi_marker;

   /// Set of Boundary Face Integrators to be applied.
   Array<LinearFormIntegrator*> flfi;
   Array<Array<int>*>           flfi_marker;

   /// The element ids where the centers of the delta functions lie
   Array<int> dlfi_delta_elem_id;

   /// The reference coordinates where the centers of the delta functions lie
   Array<IntegrationPoint> dlfi_delta_ip;

   /// If true, the delta locations are not (re)computed during assembly.
   bool HaveDeltaLocations() { return (dlfi_delta_elem_id.Size() != 0); }

   /// Force (re)computation of delta locations.
   void ResetDeltaLocations() { dlfi_delta_elem_id.SetSize(0); }

public:
   /// Creates linear form associated with FE space *f.
   LinearForm (FiniteElementSpace * f) : Vector (f -> GetVSize())
   { fes = f; }

   LinearForm() { fes = NULL; }

   /// (DEPRECATED) Return the FE space associated with the LinearForm.
   /** @deprecated Use FESpace() instead. */
   FiniteElementSpace * GetFES() { return fes; }

   /// Read+write access to the associated FiniteElementSpace.
   FiniteElementSpace *FESpace() { return fes; }
   /// Read-only access to the associated FiniteElementSpace.
   const FiniteElementSpace *FESpace() const { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator (LinearFormIntegrator * lfi);

   /** @brief Add new Boundary Integrator, restricted to the given boundary
       attributes. */
   void AddBoundaryIntegrator(LinearFormIntegrator *lfi,
                              Array<int> &bdr_attr_marker);

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator (LinearFormIntegrator * lfi);

   /** @brief Add new Boundary Face Integrator, restricted to the given boundary
       attributes. */
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                             Array<int> &bdr_attr_marker);

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Assembles delta functions of the linear form
   void AssembleDelta();

   void Update() { SetSize(fes->GetVSize()); ResetDeltaLocations(); }

   void Update(FiniteElementSpace *f)
   { fes = f; SetSize(f->GetVSize()); ResetDeltaLocations(); }

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
   /** The size of @a v must be equal to the size of the FiniteElementSpace
       @a fes. */
   LinearForm &operator=(const Vector &v);

   /// Destroys linear form.
   ~LinearForm();
};

}

#endif
