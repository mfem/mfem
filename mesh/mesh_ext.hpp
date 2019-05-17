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

#ifndef MFEM_MESH_EXT
#define MFEM_MESH_EXT

namespace mfem
{

/// Mesh Extension
class XTMesh
{
private:
   int XJID;
   long sequence;
public:
   enum Compute
   {
      _X_ = 1 << 0,
      _J_ = 1 << 1,
      _I_ = 1 << 2,
      _D_ = 1 << 3,
   };
   Array<double> X, J, invJ, detJ;
   XTMesh(const Mesh*,
          const IntegrationRule&,
          const int flags = (_X_|_J_|_I_|_D_));
   long& GetSequence() {return sequence;}
};

}

#endif
