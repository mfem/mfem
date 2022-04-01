// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LINEARFORM_EXT
#define MFEM_LINEARFORM_EXT

#include "../general/array.hpp"
#include "../linalg/vector.hpp"

namespace mfem
{

class Operator;
class LinearForm;

/// Class extending the LinearForm class to support different AssemblyLevels.
class LinearFormExtension
{
   /// Attributes of all mesh elements.
   Array<int> attributes;

   /// Temporary markers for device kernels.
   Array<int> markers;

   /// Linear form from which this extension depends. Not owned.
   LinearForm *lf;

   /// Operator that converts FiniteElementSpace L-vectors to E-vectors.
   const Operator *elem_restrict; // Not owned
   mutable Vector Ye;

public:
   LinearFormExtension(LinearForm *lf);

   ~LinearFormExtension() { }

   /// Updates the linear form extension
   void Update();

   /// Assemble at the level given for the linear form extension
   /// Fully assembles the linear form, compatible with device execution.
   /// Only integrators added with AddDomainIntegrator are supported.
   void Assemble();

   /// Assembles delta functions of the linear form extension
   /// Fully assembles the delta functions of the linear form.
   /// Not yet supported.
   void AssembleDelta() { MFEM_ABORT("AssembleDelta not implemented!"); }
};

} // namespace mfem

#endif // MFEM_LINEARFORM_EXT
