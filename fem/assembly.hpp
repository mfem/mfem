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

#ifndef MFEM_ASSEMBLY
#define MFEM_ASSEMBLY

namespace mfem
{

/** @brief Enumeration defining the assembly level for bilinear, nonlinear and
    linear form classes derived from Operator and Vector. For more details, see
    https://mfem.org/howto/assembly_levels */
enum class AssemblyLevel
{
   /// In the case of a BilinearForm LEGACY corresponds to a fully assembled
   /// form, i.e. a global sparse matrix in MFEM, Hypre or PETSC format.
   /// In the case of a NonlinearForm LEGACY corresponds to an operator that
   /// is fully evaluated on the fly.
   /// In the case of a LinearForm LEGACY corresponds to a fully assembled
   /// linear form, i.e. a global vector in MFEM, Hypre or PETSC format
   /// This assembly level is ALWAYS performed on the host.
   LEGACY = 0,
   /// @deprecated Use LEGACY instead.
   LEGACYFULL = 0,
   /// Fully assembled form, i.e. a global sparse matrix or a global vector in
   /// MFEM format. This assembly is compatible with device execution.
   FULL,
   /// Form assembled at element level, which computes and stores dense element
   /// matrices.
   ELEMENT,
   /// Partially-assembled form, which computes and stores data only at
   /// quadrature points.
   PARTIAL,
   /// "Matrix-free" form that computes all of its action on-the-fly without any
   /// substantial storage.
   NONE,
};

}

#endif
