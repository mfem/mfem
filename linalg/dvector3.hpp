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

#ifndef MFEM_DVECTOR3
#define MFEM_DVECTOR3

namespace mfem
{

// Simple vector classes on the device

/// Class for a simple Vector of size 3, appropriate for use on the GPU
class DeviceVector3
{
private:
   double data[3];
public:
   DeviceVector3() {}
   DeviceVector3(const double *r) { data[0]=r[0]; data[1]=r[1]; data[2]=r[2]; }
   DeviceVector3(const double r0, const double r1 = 0.0, const double r2 = 0.0)
   { data[0]=r0; data[1]=r1; data[2]=r2; }
   inline operator double* () { return data; }
   inline operator const double* () const { return data; }
   inline double& operator[](const size_t i) { return data[i]; }
   inline double& operator()(const int i) { return data[i]; }
   inline const double& operator()(const int i) const { return data[i]; }
};

} // mfem namespace

#endif // MFEM_DVECTOR3
