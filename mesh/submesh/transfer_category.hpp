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

#ifndef MFEM_TRANSFER_CATEGORY
#define MFEM_TRANSFER_CATEGORY

namespace mfem
{

/**
 * @brief TransferCategory describes the type of transfer.
 *
 * Usually used with a TransferMap.
 */
enum TransferCategory
{
   ParentToSubMesh,
   SubMeshToParent,
   SubMeshToSubMesh
};

} // namespace mfem

#endif // MFEM_TRANSFER_CATEGORY
