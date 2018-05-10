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
#ifndef LAGHOS_RAJA_CONFIG
#define LAGHOS_RAJA_CONFIG

namespace mfem {

#ifndef __NVCC__
  typedef int CUstream;
#endif

  // ***************************************************************************
  // * Configuration class for RAJA
  // ***************************************************************************
  class rconfig{
  private:
    // *************************************************************************
    int mpi_rank=0;
    int mpi_size=0;
    bool aware=false;
    //  ************************************************************************
    bool mps=false;
    int gpu_count=0;
    int maxXGridSize=0;
    int maxXThreadsDim=0;
#ifdef __NVCC__
    CUdevice cuDevice;
    CUcontext cuContext;
    CUstream *hStream;
#endif
    // *************************************************************************
    bool cuda=false;
    bool dcg=false;
    bool uvm=false;
    bool share=false;
    bool share_env=false;
    // *************************************************************************
    bool occa=false;
    bool hcpo=false;
    bool sync=false;
    bool nvvp=false;
    // *************************************************************************
  private:
    rconfig(){}
    rconfig(rconfig const&);
    void operator=(rconfig const&);
    // *************************************************************************
  public:
    static rconfig& Get(){
      static rconfig rconfig_singleton;
      return rconfig_singleton;
    }
    // *************************************************************************
    void Setup(const int,const int,
               const bool cuda, const bool dcg,
               const bool uvm, const bool aware,
               const bool share, const bool occa, const bool hcpo,
               const bool sync, const bool dot, const int rs_levels);
    // *************************************************************************
    bool IAmAlone();
    bool GeomNeedsUpdate(const int);
    bool DoHostConformingProlongationOperator();
    // *************************************************************************
    inline int Rank() { return mpi_rank; }
    inline int Size() { return mpi_size; }
    inline bool Root() { return mpi_rank==0; }
    inline bool Aware() { return aware; }
    // *************************************************************************
    inline bool Mps() { return mps; }
    // *************************************************************************
    inline bool Uvm() { return uvm; }
    inline bool Cuda() { return cuda; }
    inline bool Dcg() { return dcg; }
    inline bool Share() { return share && !share_env; }
    inline bool ShareEnv() { return share_env; }
    inline bool Occa() { return occa; }
    inline bool Hcpo() { return hcpo; }
    inline bool Sync() { return sync; }
    inline bool Nvvp(bool toggle=false) { return toggle?nvvp=!nvvp:nvvp; }
    inline int MaxXGridSize() { return maxXGridSize; }
    inline int MaxXThreadsDim() { return maxXThreadsDim; }
   // *************************************************************************
#ifdef __NVCC__
    inline CUstream *Stream() { return hStream; }
#endif
  };
  
} // namespace mfem

#endif // LAGHOS_RAJA_CONFIG
