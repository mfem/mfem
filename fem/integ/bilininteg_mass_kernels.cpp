// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../config/config.hpp"
#include "../../general/jit/jit.hpp" // for MFEM_JIT

MFEM_JIT
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{

MassIntegrator::Kernels::Kernels()
{
   // 2D
   MassIntegrator::AddSpecialization<2,2,2>();
   MassIntegrator::AddSpecialization<2,3,3>();
   MassIntegrator::AddSpecialization<2,4,4>();
   MassIntegrator::AddSpecialization<2,5,5>();
   MassIntegrator::AddSpecialization<2,6,6>();
   MassIntegrator::AddSpecialization<2,7,7>();
   MassIntegrator::AddSpecialization<2,8,8>();
   MassIntegrator::AddSpecialization<2,9,9>();
   // 3D
   MassIntegrator::AddSpecialization<3,2,2>();
   MassIntegrator::AddSpecialization<3,2,3>();
   MassIntegrator::AddSpecialization<3,3,4>();
   MassIntegrator::AddSpecialization<3,4,5>();
   MassIntegrator::AddSpecialization<3,4,6>();
   MassIntegrator::AddSpecialization<3,5,6>();
   MassIntegrator::AddSpecialization<3,5,8>();
   MassIntegrator::AddSpecialization<3,6,7>();
   MassIntegrator::AddSpecialization<3,7,8>();
   MassIntegrator::AddSpecialization<3,8,9>();
}

namespace internal
{

// PA Mass Diagonal 1D kernel
static void PAMassAssembleDiagonal1D(const int NE,
                                     const Array<double> &b,
                                     const Vector &d,
                                     Vector &y,
                                     const int D1D,
                                     const int Q1D)
{
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, NE);
   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int dx = 0; dx < D1D; ++dx)
      {
         Y(dx, e) = 0.0;
         for (int qx = 0; qx < Q1D; ++qx)
         {
            Y(dx, e) += B(qx, dx) * B(qx, dx) * D(qx, e);
         }
      }
   });
}

// Shared memory PA Mass Diagonal 2D kernel
MFEM_JIT template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPAMassAssembleDiagonal2D(const int NE,
                                         const Array<double> &b_,
                                         const Vector &d_,
                                         Vector &y_,
                                         const int d1d = 0,
                                         const int q1d = 0,
                                         const int nbz = 1)
{
   MFEM_CONTRACT_VAR(nbz);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   const auto b = Reshape(b_.Read(), Q1D, D1D);
   const auto D = Reshape(d_.Read(), Q1D, Q1D, NE);
   auto Y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassAssembleDiagonal2D_element<T_D1D, T_Q1D, T_NBZ>
      (e, b, D, Y, d1d, q1d, nbz);
   });
}

// Shared memory PA Mass Diagonal 3D kernel
MFEM_JIT template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAMassAssembleDiagonal3D(const int NE,
                                         const Array<double> &b,
                                         const Vector &d,
                                         Vector &y,
                                         const int d1d = 0,
                                         const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");

   const auto B = Reshape(b.Read(), Q1D, D1D);
   const auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto Y = Reshape(y.ReadWrite(), D1D, D1D, D1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassAssembleDiagonal3D_element<T_D1D,T_Q1D>
      (e, B, D, Y, d1d, q1d);
   });
}

void PAMassAssembleDiagonal(const int dim, const int D1D,
                            const int Q1D, const int NE,
                            const Array<double> &B,
                            const Vector &D,
                            Vector &Y)
{
   if (dim == 1)
   {
      return PAMassAssembleDiagonal1D(NE,B,D,Y,D1D,Q1D);
   }
   else if (dim == 2)
   {
#ifndef MFEM_USE_JIT
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPAMassAssembleDiagonal2D<2,2,16>(NE,B,D,Y);
         case 0x33: return SmemPAMassAssembleDiagonal2D<3,3,16>(NE,B,D,Y);
         case 0x44: return SmemPAMassAssembleDiagonal2D<4,4,8>(NE,B,D,Y);
         case 0x55: return SmemPAMassAssembleDiagonal2D<5,5,8>(NE,B,D,Y);
         case 0x66: return SmemPAMassAssembleDiagonal2D<6,6,4>(NE,B,D,Y);
         case 0x77: return SmemPAMassAssembleDiagonal2D<7,7,4>(NE,B,D,Y);
         case 0x88: return SmemPAMassAssembleDiagonal2D<8,8,2>(NE,B,D,Y);
         case 0x99: return SmemPAMassAssembleDiagonal2D<9,9,2>(NE,B,D,Y);
         default:   return PAMassAssembleDiagonal2D(NE,B,D,Y,D1D,Q1D);
      }
#else
      const int NBZ = (D1D < 4)  ? 16:
                      (D1D < 6)  ? 8 :
                      (D1D < 8)  ? 4 :
                      (D1D < 10)  ? 2 : 1;
      return SmemPAMassAssembleDiagonal2D(NE,B,D,Y,D1D,Q1D,NBZ);
#endif
   }
   else if (dim == 3)
   {
#ifndef MFEM_USE_JIT
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x23: return SmemPAMassAssembleDiagonal3D<2,3>(NE,B,D,Y);
         case 0x24: return SmemPAMassAssembleDiagonal3D<2,4>(NE,B,D,Y);
         case 0x26: return SmemPAMassAssembleDiagonal3D<2,6>(NE,B,D,Y);
         case 0x34: return SmemPAMassAssembleDiagonal3D<3,4>(NE,B,D,Y);
         case 0x35: return SmemPAMassAssembleDiagonal3D<3,5>(NE,B,D,Y);
         case 0x45: return SmemPAMassAssembleDiagonal3D<4,5>(NE,B,D,Y);
         case 0x48: return SmemPAMassAssembleDiagonal3D<4,8>(NE,B,D,Y);
         case 0x56: return SmemPAMassAssembleDiagonal3D<5,6>(NE,B,D,Y);
         case 0x67: return SmemPAMassAssembleDiagonal3D<6,7>(NE,B,D,Y);
         case 0x78: return SmemPAMassAssembleDiagonal3D<7,8>(NE,B,D,Y);
         case 0x89: return SmemPAMassAssembleDiagonal3D<8,9>(NE,B,D,Y);
         default:   return PAMassAssembleDiagonal3D(NE,B,D,Y,D1D,Q1D);
      }
#else
      return SmemPAMassAssembleDiagonal3D(NE,B,D,Y,D1D,Q1D);
#endif
   }
   MFEM_ABORT("Unknown kernel.");
}

#ifdef MFEM_USE_OCCA
void OccaPAMassApply2D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply2D_cpu;
      if (OccaMassApply2D_cpu.find(id) == OccaMassApply2D_cpu.end())
      {
         const occa::kernel MassApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_CPU", props);
         OccaMassApply2D_cpu.emplace(id, MassApply2D_CPU);
      }
      OccaMassApply2D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply2D_gpu;
      if (OccaMassApply2D_gpu.find(id) == OccaMassApply2D_gpu.end())
      {
         const occa::kernel MassApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply2D_GPU", props);
         OccaMassApply2D_gpu.emplace(id, MassApply2D_GPU);
      }
      OccaMassApply2D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}

void OccaPAMassApply3D(const int D1D,
                       const int Q1D,
                       const int NE,
                       const Array<real_t> &B,
                       const Array<real_t> &Bt,
                       const Vector &D,
                       const Vector &X,
                       Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaMassApply3D_cpu;
      if (OccaMassApply3D_cpu.find(id) == OccaMassApply3D_cpu.end())
      {
         const occa::kernel MassApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_CPU", props);
         OccaMassApply3D_cpu.emplace(id, MassApply3D_CPU);
      }
      OccaMassApply3D_cpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaMassApply3D_gpu;
      if (OccaMassApply3D_gpu.find(id) == OccaMassApply3D_gpu.end())
      {
         const occa::kernel MassApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "MassApply3D_GPU", props);
         OccaMassApply3D_gpu.emplace(id, MassApply3D_GPU);
      }
      OccaMassApply3D_gpu.at(id)(NE, o_B, o_Bt, o_D, o_X, o_Y);
   }
}
#endif // MFEM_USE_OCCA

// PA Mass Apply 1D kernel
static void PAMassApply1D(const int NE,
                          const Array<double> &b_,
                          const Array<double> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(d1d <= MAX_D1D, "");
   MFEM_VERIFY(q1d <= MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply1D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// PA Mass Apply 2D kernel
template<int T_D1D = 0, int T_Q1D = 0>
inline void PAMassApply2D(const int NE,
                          const Array<double> &b_,
                          const Array<double> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply2D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 2D kernel
MFEM_JIT template<int T_D1D = 0, int T_Q1D = 0, int T_NBZ = 0>
static void SmemPAMassApply2D(const int NE,
                              const Array<double> &b_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0,
                              const int nbz = 1)
{
   MFEM_CONTRACT_VAR(nbz);
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int NBZ = T_NBZ ? T_NBZ : 1;
   constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= MD1, "");
   MFEM_VERIFY(Q1D <= MQ1, "");
   const auto b = b_.Read();
   const auto D = d_.Read();
   const auto x = x_.Read();
   auto Y = y_.ReadWrite();
   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApply2D_Element<T_D1D, T_Q1D, T_NBZ, true>
      (e, NE, b, D, x, Y, d1d, q1d, nbz);
   });
}

// PA Mass Apply 3D kernel
template<int T_D1D = 0, int T_Q1D = 0>
static void PAMassApply3D(const int NE,
                          const Array<double> &b_,
                          const Array<double> &bt_,
                          const Vector &d_,
                          const Vector &x_,
                          Vector &y_,
                          const int d1d = 0,
                          const int q1d = 0)
{
   MFEM_VERIFY(T_D1D ? T_D1D : d1d <= MAX_D1D, "");
   MFEM_VERIFY(T_Q1D ? T_Q1D : q1d <= MAX_Q1D, "");

   const auto B = b_.Read();
   const auto Bt = bt_.Read();
   const auto D = d_.Read();
   const auto X = x_.Read();
   auto Y = y_.ReadWrite();

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::PAMassApply3D_Element(e, NE, B, Bt, D, X, Y, d1d, q1d);
   });
}

// Shared memory PA Mass Apply 2D kernel
MFEM_JIT template<int T_D1D = 0, int T_Q1D = 0>
static void SmemPAMassApply3D(const int NE,
                              const Array<double> &b_,
                              const Vector &d_,
                              const Vector &x_,
                              Vector &y_,
                              const int d1d = 0,
                              const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   constexpr int M1Q = T_Q1D ? T_Q1D : MAX_Q1D;
   constexpr int M1D = T_D1D ? T_D1D : MAX_D1D;
   MFEM_VERIFY(D1D <= M1D, "");
   MFEM_VERIFY(Q1D <= M1Q, "");
   auto b = b_.Read();
   auto d = d_.Read();
   auto x = x_.Read();
   auto y = y_.ReadWrite();
   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      internal::SmemPAMassApply3D_Element<T_D1D,T_Q1D>
      (e, NE, b, d, x, y, d1d, q1d);
   });
}


void PAMassApply(const int dim,
                 const int D1D,
                 const int Q1D,
                 const int NE,
                 const Array<double> &B,
                 const Array<double> &Bt,
                 const Vector &D,
                 const Vector &X,
                 Vector &Y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      if (dim == 2)
      {
         return OccaPAMassApply2D(D1D,Q1D,NE,B,Bt,D,X,Y);
      }
      if (dim == 3)
      {
         return OccaPAMassApply3D(D1D,Q1D,NE,B,Bt,D,X,Y);
      }
      MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   const int id = (D1D << 4) | Q1D;

   if (dim == 1)
   {
      return PAMassApply1D(NE,B,Bt,D,X,Y,D1D,Q1D);
   }
   else if (dim == 2)
   {
#ifndef MFEM_USE_JIT
      switch (id)
      {
         case 0x22: return SmemPAMassApply2D<2,2,16>(NE,B,D,X,Y);
         case 0x24: return SmemPAMassApply2D<2,4,16>(NE,B,D,X,Y);
         case 0x33: return SmemPAMassApply2D<3,3,16>(NE,B,D,X,Y);
         case 0x34: return SmemPAMassApply2D<3,4,16>(NE,B,D,X,Y);
         case 0x35: return SmemPAMassApply2D<3,5,16>(NE,B,D,X,Y);
         case 0x36: return SmemPAMassApply2D<3,6,16>(NE,B,D,X,Y);
         case 0x44: return SmemPAMassApply2D<4,4,8>(NE,B,D,X,Y);
         case 0x46: return SmemPAMassApply2D<4,6,8>(NE,B,D,X,Y);
         case 0x48: return SmemPAMassApply2D<4,8,8>(NE,B,D,X,Y);
         case 0x55: return SmemPAMassApply2D<5,5,8>(NE,B,D,X,Y);
         case 0x57: return SmemPAMassApply2D<5,7,8>(NE,B,D,X,Y);
         case 0x58: return SmemPAMassApply2D<5,8,4>(NE,B,D,X,Y);
         case 0x66: return SmemPAMassApply2D<6,6,4>(NE,B,D,X,Y);
         case 0x77: return SmemPAMassApply2D<7,7,4>(NE,B,D,X,Y);
         case 0x88: return SmemPAMassApply2D<8,8,2>(NE,B,D,X,Y);
         case 0x99: return SmemPAMassApply2D<9,9,2>(NE,B,D,X,Y);
         default:   return PAMassApply2D(NE,B,Bt,D,X,Y,D1D,Q1D);
      }
#else
      const int NBZ = (D1D < 4)  ? 16:
                      (D1D < 6)  ? 8 :
                      (D1D < 8)  ? 4 :
                      (D1D < 10)  ? 2 : 1;
      return SmemPAMassApply2D(NE,B,D,X,Y,D1D,Q1D,NBZ);
#endif
   }
   else if (dim == 3)
   {
#ifndef MFEM_USE_JIT
      switch (id)
      {
         case 0x22: return SmemPAMassApply3D<2,2>(NE,B,D,X,Y);
         case 0x23: return SmemPAMassApply3D<2,3>(NE,B,D,X,Y);
         case 0x24: return SmemPAMassApply3D<2,4>(NE,B,D,X,Y);
         case 0x26: return SmemPAMassApply3D<2,6>(NE,B,D,X,Y);
         case 0x34: return SmemPAMassApply3D<3,4>(NE,B,D,X,Y);
         case 0x35: return SmemPAMassApply3D<3,5>(NE,B,D,X,Y);
         case 0x36: return SmemPAMassApply3D<3,6>(NE,B,D,X,Y);
         case 0x37: return SmemPAMassApply3D<3,7>(NE,B,D,X,Y);
         case 0x45: return SmemPAMassApply3D<4,5>(NE,B,D,X,Y);
         case 0x46: return SmemPAMassApply3D<4,6>(NE,B,D,X,Y);
         case 0x48: return SmemPAMassApply3D<4,8>(NE,B,D,X,Y);
         case 0x56: return SmemPAMassApply3D<5,6>(NE,B,D,X,Y);
         case 0x58: return SmemPAMassApply3D<5,8>(NE,B,D,X,Y);
         case 0x67: return SmemPAMassApply3D<6,7>(NE,B,D,X,Y);
         case 0x78: return SmemPAMassApply3D<7,8>(NE,B,D,X,Y);
         case 0x89: return SmemPAMassApply3D<8,9>(NE,B,D,X,Y);
         case 0x9A: return SmemPAMassApply3D<9,10>(NE,B,D,X,Y);
         default:   return PAMassApply3D(NE,B,Bt,D,X,Y,D1D,Q1D);
      }
#else
      return SmemPAMassApply3D(NE,B,D,X,Y,D1D,Q1D);
#endif
   }
   mfem::out << "Unknown kernel 0x" << std::hex << id << std::endl;
   MFEM_ABORT("Unknown kernel.");
}

} // namespace internal

} // namespace mfem
