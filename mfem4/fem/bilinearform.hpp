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

#ifndef MFEM4_BILINEARFORM
#define MFEM4_BILINEARFORM

#include "linalg/linalg.hpp"
#include "fem/fespace.hpp"

namespace mfem4
{

using namespace mfem;

enum class AssemblyLevel
{
   PARTIAL, FULL
};

///
class BilinearForm : public Matrix
{
public:
   /// Creates bilinear form associated with FE space @a *fes.
   BilinearForm(const FiniteElementSpace *fes,
                AssemblyLevel level = AssemblyLevel::FULL);

   /// Adds new Domain Integrator.
   void AddDomainIntegrator(const BilinearFormIntegrator *bfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator(const BilinearFormIntegrator *bfi);

   /** @brief Adds new Boundary Integrator, restricted to specific boundary
       attributes. */
   void AddBoundaryIntegrator(const BilinearFormIntegrator * bfi,
                              const Array<int> &attr_list);

   /// Adds new interior Face Integrator.
   void AddInteriorFaceIntegrator(const BilinearFormIntegrator *bfi);

   /// Adds new boundary Face Integrator.
   void AddBdrFaceIntegrator(const BilinearFormIntegrator *bfi);

   /** @brief Adds new boundary Face Integrator, restricted to specific boundary
       attributes. */
   void AddBdrFaceIntegrator(const BilinearFormIntegrator *bfi,
                             const Array<int> &attr_list);

   void SetAssemblyLevel(AssemblyLevel level);

   /// Assembles the form i.e. sums over all domain/bdr integrators.
   void Assemble();

   /// Returns reference to a_{ij}.
   virtual double &Elem(int i, int j);

   /// Returns constant reference to a_{ij}.
   virtual const double &Elem(int i, int j) const;

   /// Matrix vector multiplication.
   virtual void Mult(const Vector &x, Vector &y) const;

   double InnerProduct(const Vector &x, const Vector &y) const;

   /// Returns a reference to the sparse matrix
   const SparseMatrix &SpMat() const;


protected:


};

} // namespace mfem4

#endif // MFEM4_BILINEARFORM
