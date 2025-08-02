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

#ifndef MFEM_LINEARFORM_EXT
#define MFEM_LINEARFORM_EXT

#include "../general/array.hpp"
#include "../linalg/vector.hpp"

namespace mfem
{

class Operator;
class LinearForm;

/// Class extending the LinearForm class to support assembly on devices.
class LinearFormExtension
{
   /// Attributes of all mesh elements.
   const Array<int> *attributes; // Not owned
   const Array<int> *bdr_face_attributes; // Not owned

   /// Temporary markers for device kernels.
   Array<int> markers, bdr_markers;

   /// Linear form from which this extension depends. Not owned.
   LinearForm *lf;

   /// Operator that converts FiniteElementSpace L-vectors to E-vectors.
   const ElementRestrictionOperator *elem_restrict_lex; // Not owned

   /// Operator that converts L-vectors to boundary E-vectors.
   const FaceRestriction *bdr_restrict_lex; // Not owned

   /// Internal E-vectors.
   mutable Vector b, bdr_b;

public:

   /// \brief Create a LinearForm extension of @a lf.
   LinearFormExtension(LinearForm *lf);

   ~LinearFormExtension() { }

   /// Assemble the linear form, compatible with device execution.
   /// Only integrators added with AddDomainIntegrator are supported for now.
   void Assemble();

   /// Update the linear form extension.
   void Update();
};

} // namespace mfem

#endif // MFEM_LINEARFORM_EXT
