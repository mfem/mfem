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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem {
  
namespace kernels {

#ifdef MFEM_USE_MPI

// ***************************************************************************
// * kConformingProlongationOperator
// ***************************************************************************
kConformingProlongationOperator::kConformingProlongationOperator
(ParFiniteElementSpace &pfes): kernels::Operator(*pfes.GetTrueVLayout().As<Layout>(), 
                                                 *pfes.GetVLayout().As<Layout>()),
                               external_ldofs(),
                               //d_external_ldofs(Height()-Width()),
                               gc(new kCommD(pfes)),
                               kMaxTh(0){
   push();
   mfem::Array<int> ldofs;
   assert((std::size_t)pfes.GetTrueVSize()==pfes.GetTrueVLayout()->Size());   
   dbg("\033[32;7m GetVSize()=%d, GetTrueVSize()=%d",pfes.GetVSize(), pfes.GetTrueVSize());
   dbg("\033[32;7m GetVLayout()=%d, GetTrueVLayout()=%d",pfes.GetVLayout()->Size(), pfes.GetTrueVLayout()->Size());
   dbg("\033[32;7m Height()=%d, Width()=%d",Height(),Width());
   MPI_Barrier(MPI_COMM_WORLD);
   
   assert(Height()>=Width());
   d_external_ldofs.allocate(Height()-Width());
      
   const std::size_t absHmW = abs(Height()-Width());
   
   Table &group_ldof = gc->GroupLDofTable();
   external_ldofs.Reserve(absHmW);
   for (int gr = 1; gr < group_ldof.Size(); gr++)
    {
      if (!gc->GetGroupTopology().IAmMaster(gr)) 
      {
        ldofs.MakeRef(group_ldof.GetRow(gr), group_ldof.RowSize(gr));
        external_ldofs.Append(ldofs);
      }
    }
    external_ldofs.Sort();
#ifdef __NVCC__
    const std::size_t HmW = absHmW;
    if (HmW>0){
       d_external_ldofs = external_ldofs;
    }
#endif
    assert((std::size_t)external_ldofs.Size() == absHmW);
    // *************************************************************************
    const int m = external_ldofs.Size();
    //printf("\n[kConformingProlongationOperator] m=%d\n",m);fflush(stdout);
    int j = 0;
    for (int i = 0; i < m; i++) {
      const int end = external_ldofs[i];
      const int size = end-j;
      if (size>kMaxTh) kMaxTh=size;
      //printf(" %d",size);
      j = end+1;
    }
    //printf("\n[kConformingProlongationOperator] kMaxTh=%d",kMaxTh);fflush(stdout);
    //gc->PrintInfo(); 
    //pfes.Dof_TrueDof_Matrix()->PrintCommPkg();
    pop();
  }

  // ***************************************************************************
  // * ~kConformingProlongationOperator
  // ***************************************************************************
  kConformingProlongationOperator::~kConformingProlongationOperator(){
    delete  gc;
  }

#ifdef __NVCC__
  // ***************************************************************************
  // * CUDA Error Status Check
  // ***************************************************************************
  void cuLastCheck(){
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
      exit(fprintf(stderr, "\n\t\033[31;1m[cuLastCheck] failed: %s\033[m\n",
                   cudaGetErrorString(cudaStatus)));
  }

  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
  static __global__
  void k_Mult(double *y,const double *x,const int *external_ldofs,const int m){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=m) return;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    for(int k=0;k<(end-j);k+=1)
      y[j+k]=x[j-i+k];
  }
  static __global__
  void k_Mult2(double *y,const double *x,const int *external_ldofs,
               const int m, const int base){
    const int i = base+threadIdx.x;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = blockIdx.x;
    if (k>=(end-j)) return;
    y[j+k]=x[j-i+k];
  }
#endif

  // ***************************************************************************
  // * Device Mult
  // ***************************************************************************
void kConformingProlongationOperator::d_Mult(const kernels::Vector &x,
                                                   kernels::Vector &y) const{
    push(Coral);
    MFEM_ASSERT(x.Size() == (std::size_t)Width(), "x.Size()=" << x.Size()<<", Width()="<<Width());
    MFEM_ASSERT(y.Size() == (std::size_t)Height(), "");
    const double *d_xdata = x.GetData();
    const int in_layout = 2; // 2 - input is ltdofs array
    
    push(d_BcastBegin,Coral);
    gc->d_BcastBegin(const_cast<double*>(d_xdata), in_layout);
    pop();
    
    push(d_Mult_Work,Coral);
    double *d_ydata = y.GetData(); 
#ifdef __NVCC__
    int j = 0;
    const int m = external_ldofs.Size();
    /* // Test with async rDtoD
       push(k_DtoDAsync,Coral);
       for (int i = 0; i < m; i++){
       const int end = external_ldofs[i];
       //printf("\n[k_Mult] rDtoD async size %d",end-j);
       rmemcpy::rDtoD(d_ydata+j,d_xdata+j-i,(end-j)*sizeof(double),true); // async
       j = end+1;
       }
       cudaDeviceSynchronize();
       pop();*/
    
    if (m>0){
      const int maxXThDim = mfem::kernels::config::Get().MaxXThreadsDim();
      dbg("maxXThDim=%d",maxXThDim);
      if (m>maxXThDim){
        const int kTpB=64;
        printf("\n[k_Mult] m=%d kMaxTh=%d",m,kMaxTh);
        k_Mult<<<(m+kTpB-1)/kTpB,kTpB>>>(d_ydata,d_xdata,d_external_ldofs,m);
        cuLastCheck();
      }else{
        assert((m/maxXThDim)==0);
        assert(kMaxTh<mfem::kernels::config::Get().MaxXGridSize());
        for(int of7=0;of7<m/maxXThDim;of7+=1){
          const int base = of7*maxXThDim;
          k_Mult2<<<kMaxTh,maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,base);
          cuLastCheck();
        }
        k_Mult2<<<kMaxTh,m%maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,0);
        cuLastCheck();
      }
      j = external_ldofs[m-1]+1;
    }else{
       dbg("m<0");
    }
    dbg("last mfem::kernels::kmemcpy::rDtoD");
    mfem::kernels::kmemcpy::rDtoD(d_ydata+j,d_xdata+j-m,(Width()+m-j)*sizeof(double));
#endif
    pop();
    
    push(d_BcastEnd,Coral);
    const int out_layout = 0; // 0 - output is ldofs array
    gc->d_BcastEnd(d_ydata, out_layout);
    pop();
    pop();
  }

  
  // ***************************************************************************
  // * k_Mult
  // ***************************************************************************
#ifdef __NVCC__
  static __global__
  void k_MultTranspose(double *y,const double *x,const int *external_ldofs,const int m){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>=m) return;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    for(int k=0;k<(end-j);k+=1)
      y[j-i+k]=x[j+k];
  }
  
  static __global__
  void k_MultTranspose2(double *y,const double *x,const int *external_ldofs,
                        const int m, const int base){
    const int i = base+threadIdx.x;
    const int j = (i>0)?external_ldofs[i-1]+1:0;
    const int end = external_ldofs[i];
    const int k = blockIdx.x;
    if (k>=(end-j)) return;
    y[j-i+k]=x[j+k];
  }
#endif
  
  // ***************************************************************************
  // * Device MultTranspose
  // ***************************************************************************
  void kConformingProlongationOperator::d_MultTranspose(const kernels::Vector &x,
                                                           kernels::Vector &y) const{
    push(Coral);
    const double *d_xdata = x.GetData();
    
    push(d_ReduceBegin,Coral);
    gc->d_ReduceBegin(d_xdata);
    pop();
    
    push(d_MultTranspose_Work,Coral);
    double *d_ydata = y.GetData();
#ifdef __NVCC__
    int j = 0;
    const int m = external_ldofs.Size();
    /*push(k_DtoDT,Coral);
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
      rmemcpy::rDtoD(d_ydata+j-i,d_xdata+j,(end-j)*sizeof(double));
      j = end+1;
    }
    pop();*/
    if (m>0){      
      const int maxXThDim = mfem::kernels::config::Get().MaxXThreadsDim();
      if (m>maxXThDim){
        const int kTpB=64;
        k_MultTranspose<<<(m+kTpB-1)/kTpB,kTpB>>>(d_ydata,d_xdata,d_external_ldofs,m);
        cuLastCheck();
      }else{
        const int TpB = mfem::kernels::config::Get().MaxXThreadsDim();
        assert(kMaxTh<mfem::kernels::config::Get().MaxXGridSize());
        for(int of7=0;of7<m/maxXThDim;of7+=1){
        const int base = of7*maxXThDim;
        k_MultTranspose2<<<kMaxTh,maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,base);
        cuLastCheck();
      }
      k_MultTranspose2<<<kMaxTh,m%maxXThDim>>>(d_ydata,d_xdata,d_external_ldofs,m,0);
      cuLastCheck();
      }
      j = external_ldofs[m-1]+1;
    }
    mfem::kernels::kmemcpy::rDtoD(d_ydata+j-m,d_xdata+j,(Height()-j)*sizeof(double));
#endif
    pop();
    push(d_ReduceEnd,Coral);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    gc->d_ReduceEnd<double>(d_ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Mult_ & MultTranspose_
  // ***************************************************************************
   void kConformingProlongationOperator::Mult_(const kernels::Vector &x,
                                               kernels::Vector &y) const{
      assert(false);
   }

   // **************************************************************************
   void kConformingProlongationOperator::MultTranspose_(const kernels::Vector &x,
                                                        kernels::Vector &y) const{
      assert(false);
   }

   // **************************************************************************
   // * Host Mult
   // **************************************************************************   
   void kConformingProlongationOperator::Mult(const mfem::Vector &x,
                                              mfem::Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData(); 
    const int m = external_ldofs.Size();
    const int in_layout = 2; // 2 - input is ltdofs array
    push(BcastBegin,Moccasin);
    gc->BcastBegin(const_cast<double*>(xdata), in_layout);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++){
      const int end = external_ldofs[i];
      std::copy(xdata+j-i, xdata+end-i, ydata+j);
      j = end+1;
    }
    std::copy(xdata+j-m, xdata+Width(), ydata+j);
    const int out_layout = 0; // 0 - output is ldofs array
    push(BcastEnd,PeachPuff);
    gc->BcastEnd(ydata, out_layout);
    pop();
    pop();
  }

  // ***************************************************************************
  // * Host MultTranspose
  // ***************************************************************************
   void kConformingProlongationOperator::MultTranspose(const mfem::Vector &x,
                                                       mfem::Vector &y) const{
    push(Coral);
    const double *xdata = x.GetData();
    double *ydata = y.GetData();
    const int m = external_ldofs.Size();
    push(ReduceBegin,PapayaWhip);
    gc->ReduceBegin(xdata);
    pop();
    int j = 0;
    for (int i = 0; i < m; i++)   {
      const int end = external_ldofs[i];
      std::copy(xdata+j, xdata+end, ydata+j-i);
      j = end+1;
    }
    std::copy(xdata+j, xdata+Height(), ydata+j-m);
    const int out_layout = 2; // 2 - output is an array on all ltdofs
    push(ReduceEnd,LavenderBlush);
    gc->ReduceEnd<double>(ydata, out_layout, GroupCommunicator::Sum);
    pop();
    pop();
  }

#endif
   
} // namespace mfem::kernels
   
} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
