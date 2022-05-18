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

#ifndef MFEM_PTRANSFERMAP
#define MFEM_PTRANSFERMAP

#include "../../fem/pgridfunc.hpp"
#include "transfer_category.hpp"

namespace mfem
{

class ParTransferMap
{
public:
   ParTransferMap(const ParGridFunction &src,
                  const ParGridFunction &dst);

   void Transfer(const ParGridFunction &src, ParGridFunction &dst) const;

   ~ParTransferMap();

private:
   void CommunicateIndicesSet(Array<int> &map, int dst_sz);

   void CommunicateSharedVdofs(Vector &f) const;

   detail::TransferCategory category_;
   Array<int> sub1_to_parent_map_, sub2_to_parent_map_, indices_set_local_,
         indices_set_global_;
   const ParFiniteElementSpace *root_fes_ = nullptr;
   const GroupCommunicator *root_gc_ = nullptr;
   mutable Vector z_;
};

} // namespace mfem

#endif // MFEM_PTRANSFERMAP