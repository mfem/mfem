// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"
#include <sys/time.h>

namespace mfem {

  // ***************************************************************************
  bool dotTest(const int rs_levels){
    cuProfilerStart();
    struct timeval st, et;
    int size = 0x400;
    for (int lev = 0; lev < rs_levels; lev++) size<<=1;
    Vector h_a(size); h_a=1.0/M_PI;
    Vector h_b(size); h_b=M_PI;
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

} // namespace mfem
