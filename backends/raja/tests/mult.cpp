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

#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{
  // ***************************************************************************
  // RajaPm1APOperator
  // ***************************************************************************
  class RajaPm1APOperator : public RajaOperator{
  private:
    const RajaOperator &Rt;
    const RajaOperator &A;
    const RajaOperator &P;
    mutable RajaVector Px;
    mutable RajaVector APx;
  public:
    /// Construct the RAP operator given R^T, A and P.
    RajaPm1APOperator(const RajaOperator &Rt_, const RajaOperator &A_, const RajaOperator &P_)
      : RajaOperator(Rt_.Width(), P_.Width()), Rt(Rt_), A(A_), P(P_),
        Px(P.Height()), APx(A.Height()) { }
    /// Operator application.
    void Mult(const RajaVector & x, RajaVector & y) const {
      push(Pm1AP,SkyBlue);
      P.Mult(x, Px);
      A.Mult(Px, APx);
      Rt.MultTranspose(APx, y);
      pop();
    }
    /// Application of the transpose.
    void MultTranspose(const RajaVector & x, RajaVector & y) const {
      push(Pm1APT,SkyBlue);
      Rt.Mult(x, APx);
      A.MultTranspose(APx, Px);
      P.MultTranspose(Px, y);
      pop();
    }
  };

  // ***************************************************************************
  // RajaIdOperator
  // ***************************************************************************
  class RajaIdOperator: public RajaOperator {
  public:
    RajaIdOperator(int s = 0) { height = width = s; }
    void Mult(const RajaVector &x, RajaVector &y) const  {
      push(Id=,DarkCyan);
      //usleep(1);
      y=x;
      pop();
    }
    void MultTranspose(const RajaVector &x, RajaVector &y) const {
      push(Id=T,DarkCyan);
      //usleep(1);
      y=x;
      pop();
    }
  };

  // ***************************************************************************
#ifdef __NVCC__
  __global__ void iniK(void){}
#endif

  // ***************************************************************************
  bool multTest(ParMesh *pmesh, const int order, const int max_step){
    struct timeval st, et;
    const int nb_step = (max_step>0)?max_step:100;
    assert(order>=1);
    
    const int dim = pmesh->Dimension();
    const H1_FECollection fec(order, dim);
    RajaFiniteElementSpace fes(pmesh, &fec, 1);
    
    const int lsize = fes.GetVSize();
    const int ltsize = fes.GetTrueVSize();
    const HYPRE_Int gsize = fes.GlobalTrueVSize();
    
    if (rconfig::Get().Root())
      mfem::out << "Number of global dofs: " << gsize << std::endl;
    if (rconfig::Get().Root())
      mfem::out << "Number of local dofs: " << lsize << std::endl;
    
    const RajaOperator &prolong = *fes.GetProlongationOperator();
    const RajaOperator &testP  = prolong;
    const RajaOperator &trialP = prolong;
    const RajaIdOperator Id(lsize);
    
    RajaPm1APOperator Pm1AP(testP,Id,trialP);
    RajaVector x(ltsize); x=1.0;
    RajaVector y(lsize);
    
    MPI_Barrier(pmesh->GetComm());
#ifdef __NVCC__
    cudaDeviceSynchronize();
#endif

    // Do one RAP Mult/MultTranspose to set MPI's buffers
    Pm1AP.Mult(x,y);
    Pm1AP.MultTranspose(x,y);
    
#ifdef __NVCC__
    MPI_Barrier(pmesh->GetComm());
    cudaDeviceSynchronize();
    // Now let go the markers
    rconfig::Get().Nvvp(true);
    cuProfilerStart();
#endif
    
    // Launch first dummy kernel for profiling overhead to start
#ifdef __NVCC__
    push(iniK,Green);
    iniK<<<128,1>>>();
    pop();
#endif
    
    MPI_Barrier(pmesh->GetComm());
#ifdef __NVCC__
    cudaDeviceSynchronize();
#endif

    gettimeofday(&st, NULL);
    for(int i=0;i<nb_step;i++){
#ifdef __NVCC__
      cudaDeviceSynchronize(); // used with nvvp
#endif
      push(SkyBlue);
      Pm1AP.Mult(x, y);
      pop();
      
      push(Wrk,LawnGreen);
      //usleep(1);
      pop();
      
      push(SkyBlue);
      Pm1AP.MultTranspose(x, y);
      pop();
    }
    // We MUST sync after to make sure every kernel has completed
    // or play with the -sync flag to enforce it with the push/pop
#ifdef __NVCC__
    cudaDeviceSynchronize();
#endif
    gettimeofday(&et, NULL);
    const float alltime = ((et.tv_sec-st.tv_sec)*1.0e3+(et.tv_usec - st.tv_usec)/1.0e3);
    if (rconfig::Get().Root())
      printf("\033[32m[laghos] Elapsed time = %f ms/step\33[m\n", alltime/nb_step);
    return true;
  }

} //  namespace raja
   
} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)
