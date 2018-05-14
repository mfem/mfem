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
#include "../raja.hpp"

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{
   
namespace raja
{

  // ***************************************************************************
  bool dotTest(const int rs_levels){
    cuProfilerStart();
    struct timeval st, et;
    int size = 0x400;
    for (int lev = 0; lev < rs_levels; lev++) size<<=1;
    mfem::Vector h_a(size); h_a=1.0/M_PI;
    mfem::Vector h_b(size); h_b=M_PI;
    gettimeofday(&st, NULL);
    RajaVector a(size);a=1.0/M_PI;//h_a;//a.Print();
    RajaVector b(size);b=M_PI;//(h_b); //b.Print();
    //RajaVector c(size); c=0.0;
    gettimeofday(&et, NULL);
    const double setTime = ((et.tv_sec-st.tv_sec)*1000.0+(et.tv_usec-st.tv_usec)/1000.0);
    printf("\033[32m[laghos] Set in \033[1m%12.6e(s)\033[m\n",setTime/1000.0);
    gettimeofday(&st, NULL);
    //double dt = a*b;
    //c+=1.0;
    a+=b;
    gettimeofday(&et, NULL);
    //assert(dt == (double)size);
    const double alltime = ((et.tv_sec-st.tv_sec)*1000.0+(et.tv_usec-st.tv_usec)/1000.0);
    printf("\033[32m[laghos] Ker (%d) in \033[1m%12.6e(s)\033[m\n",size,alltime/1000.0);
    return true;
  }
   
} //  namespace raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
