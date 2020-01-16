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

#include "forall.hpp"
#include "cuda.hpp"
#include "occa.hpp"
#ifdef MFEM_USE_CEED
#include <ceed.h>
#endif

#include <string>
#include <map>

namespace mfem
{

// Place the following variables in the mfem::internal namespace, so that they
// will not be included in the doxygen documentation.
namespace internal
{

#ifdef MFEM_USE_OCCA
// Default occa::device used by MFEM.
occa::device occaDevice;
#endif

#ifdef MFEM_USE_CEED
Ceed ceed = NULL;
#endif

// Backends listed by priority, high to low:
static const Backend::Id backend_list[Backend::NUM_BACKENDS] =
{
   Backend::CEED_CUDA, Backend::OCCA_CUDA, Backend::RAJA_CUDA, Backend::CUDA,
   Backend::HIP,
   Backend::OCCA_OMP, Backend::RAJA_OMP, Backend::OMP,
   Backend::CEED_CPU, Backend::OCCA_CPU, Backend::RAJA_CPU, Backend::CPU
};

// Backend names listed by priority, high to low:
static const char *backend_name[Backend::NUM_BACKENDS] =
{
   "ceed-cuda", "occa-cuda", "raja-cuda", "cuda",
   "hip",
   "occa-omp", "raja-omp", "omp",
   "ceed-cpu", "occa-cpu", "raja-cpu", "cpu"
};

} // namespace mfem::internal


// Initialize the unique global Device variable.
Device Device::device_singleton;


Device::~Device()
{
   if (destroy_mm)
   {
      free(ceed_option);
#ifdef MFEM_USE_CEED
      CeedDestroy(&internal::ceed);
#endif
      mm.Destroy();
   }
}

void Device::Configure(const std::string &device, const int dev)
{
   std::map<std::string, Backend::Id> bmap;
   for (int i = 0; i < Backend::NUM_BACKENDS; i++)
   {
      bmap[internal::backend_name[i]] = internal::backend_list[i];
   }
   std::string::size_type beg = 0, end, option;
   while (1)
   {
      end = device.find(',', beg);
      end = (end != std::string::npos) ? end : device.size();
      const std::string bname = device.substr(beg, end - beg);
      option = bname.find(':');
      if (option==std::string::npos) // No option
      {
         const std::string backend = bname;
         std::map<std::string, Backend::Id>::iterator it = bmap.find(backend);
         MFEM_VERIFY(it != bmap.end(), "invalid backend name: '" << backend << '\'');
         Get().MarkBackend(it->second);
      }
      else
      {
         const std::string backend = bname.substr(0, option);
         const std::string boption = bname.substr(option+1);
         Get().ceed_option = strdup(boption.c_str());
         std::map<std::string, Backend::Id>::iterator it = bmap.find(backend);
         MFEM_VERIFY(it != bmap.end(), "invalid backend name: '" << backend << '\'');
         Get().MarkBackend(it->second);
      }
      if (end == device.size()) { break; }
      beg = end + 1;
   }

   // OCCA_CUDA needs CUDA or RAJA_CUDA:
   if (Allows(Backend::OCCA_CUDA) && !Allows(Backend::RAJA_CUDA))
   {
      Get().MarkBackend(Backend::CUDA);
   }
   if (Allows(Backend::CEED_CUDA))
   {
      Get().MarkBackend(Backend::CUDA);
   }

   // Perform setup.
   Get().Setup(dev);

   // Enable the device
   Enable();

   // Copy all data members from the global 'singleton_device' into '*this'.
   std::memcpy(this, &Get(), sizeof(Device));

   // Only '*this' will call the MemoryManager::Destroy() method.
   destroy_mm = true;
}

void Device::Print(std::ostream &out)
{
   out << "Device configuration: ";
   bool add_comma = false;
   for (int i = 0; i < Backend::NUM_BACKENDS; i++)
   {
      if (backends & internal::backend_list[i])
      {
         if (add_comma) { out << ','; }
         add_comma = true;
         out << internal::backend_name[i];
      }
   }
   out << '\n';
#ifdef MFEM_USE_CEED
   if (Allows(Backend::CEED_MASK))
   {
      const char *ceed_backend;
      CeedGetResource(internal::ceed, &ceed_backend);
      out << "libCEED backend: " << ceed_backend << '\n';
   }
#endif
}

void Device::UpdateMemoryTypeAndClass()
{
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      mem_type = MemoryType::CUDA;
      mem_class = MemoryClass::CUDA;
   }
   else
   {
      mem_type = MemoryType::HOST;
      mem_class = MemoryClass::HOST;
   }
}

void Device::Enable()
{
   if (Get().backends & ~Backend::CPU)
   {
      Get().mode = Device::ACCELERATED;
      Get().UpdateMemoryTypeAndClass();
   }
}

#ifdef MFEM_USE_CUDA
static void DeviceSetup(const int dev, int &ngpu)
{
   ngpu = CuGetDeviceCount();
   MFEM_VERIFY(ngpu > 0, "No CUDA device found!");
   MFEM_GPU_CHECK(cudaSetDevice(dev));
}
#endif

static void CudaDeviceSetup(const int dev, int &ngpu)
{
#ifdef MFEM_USE_CUDA
   DeviceSetup(dev, ngpu);
#endif
}

static void HipDeviceSetup(const int dev, int &ngpu)
{
#ifdef MFEM_USE_HIP
   int deviceId;
   MFEM_GPU_CHECK(hipGetDevice(&deviceId));
   hipDeviceProp_t props;
   MFEM_GPU_CHECK(hipGetDeviceProperties(&props, deviceId));
   MFEM_VERIFY(dev==deviceId,"");
   ngpu = 1;
#endif
}

static void RajaDeviceSetup(const int dev, int &ngpu)
{
#ifdef MFEM_USE_CUDA
   if (ngpu <= 0) { DeviceSetup(dev, ngpu); }
#endif
}

static void OccaDeviceSetup(const int dev)
{
#ifdef MFEM_USE_OCCA
   const int cpu  = Device::Allows(Backend::OCCA_CPU);
   const int omp  = Device::Allows(Backend::OCCA_OMP);
   const int cuda = Device::Allows(Backend::OCCA_CUDA);
   if (cpu + omp + cuda > 1)
   {
      MFEM_ABORT("Only one OCCA backend can be configured at a time!");
   }
   if (cuda)
   {
#if OCCA_CUDA_ENABLED
      std::string mode("mode: 'CUDA', device_id : ");
      internal::occaDevice.setup(mode.append(1,'0'+dev));
#else
      MFEM_ABORT("the OCCA CUDA backend requires OCCA built with CUDA!");
#endif
   }
   else if (omp)
   {
#if OCCA_OPENMP_ENABLED
      internal::occaDevice.setup("mode: 'OpenMP'");
#else
      MFEM_ABORT("the OCCA OpenMP backend requires OCCA built with OpenMP!");
#endif
   }
   else
   {
      internal::occaDevice.setup("mode: 'Serial'");
   }

   std::string mfemDir;
   if (occa::io::exists(MFEM_INSTALL_DIR "/include/mfem/"))
   {
      mfemDir = MFEM_INSTALL_DIR "/include/mfem/";
   }
   else if (occa::io::exists(MFEM_SOURCE_DIR))
   {
      mfemDir = MFEM_SOURCE_DIR;
   }
   else
   {
      MFEM_ABORT("Cannot find OCCA kernels in MFEM_INSTALL_DIR or MFEM_SOURCE_DIR");
   }

   occa::io::addLibraryPath("mfem", mfemDir);
   occa::loadKernels("mfem");
#else
   MFEM_ABORT("the OCCA backends require MFEM built with MFEM_USE_OCCA=YES");
#endif
}

static void CeedDeviceSetup(const char* ceed_spec)
{
#ifdef MFEM_USE_CEED
   CeedInit(ceed_spec, &internal::ceed);
   const char *ceed_backend;
   CeedGetResource(internal::ceed, &ceed_backend);
   if (strcmp(ceed_spec, ceed_backend) && strcmp(ceed_spec, "/cpu/self"))
   {
      mfem::out << std::endl << "WARNING!!!\n"
                "libCEED is not using the requested backend!!!\n"
                "WARNING!!!\n" << std::endl;
   }
#endif
}

void Device::Setup(const int device)
{
   MFEM_VERIFY(ngpu == -1, "the mfem::Device is already configured!");

   ngpu = 0;
   dev = device;
#ifndef MFEM_USE_CUDA
   MFEM_VERIFY(!Allows(Backend::CUDA_MASK),
               "the CUDA backends require MFEM built with MFEM_USE_CUDA=YES");
#endif
#ifndef MFEM_USE_HIP
   MFEM_VERIFY(!Allows(Backend::HIP_MASK),
               "the HIP backends require MFEM built with MFEM_USE_HIP=YES");
#endif
#ifndef MFEM_USE_RAJA
   MFEM_VERIFY(!Allows(Backend::RAJA_MASK),
               "the RAJA backends require MFEM built with MFEM_USE_RAJA=YES");
#endif
#ifndef MFEM_USE_OPENMP
   MFEM_VERIFY(!Allows(Backend::OMP|Backend::RAJA_OMP),
               "the OpenMP and RAJA OpenMP backends require MFEM built with"
               " MFEM_USE_OPENMP=YES");
#endif
#ifndef MFEM_USE_CEED
   MFEM_VERIFY(!Allows(Backend::CEED_MASK),
               "the CEED backends require MFEM built with MFEM_USE_CEED=YES");
#else
   MFEM_VERIFY(!Allows(Backend::CEED_CPU) || !Allows(Backend::CEED_CUDA),
               "Only one CEED backend can be enabled at a time!");
#endif
   if (Allows(Backend::CUDA)) { CudaDeviceSetup(dev, ngpu); }
   if (Allows(Backend::HIP)) { HipDeviceSetup(dev, ngpu); }
   if (Allows(Backend::RAJA_CUDA)) { RajaDeviceSetup(dev, ngpu); }
   // The check for MFEM_USE_OCCA is in the function OccaDeviceSetup().
   if (Allows(Backend::OCCA_MASK)) { OccaDeviceSetup(dev); }
   if (Allows(Backend::CEED_CPU))
   {
      if (!ceed_option)
      {
         CeedDeviceSetup("/cpu/self");
      }
      else
      {
         CeedDeviceSetup(ceed_option);
      }
   }
   if (Allows(Backend::CEED_CUDA))
   {
      if (!ceed_option)
      {
         CeedDeviceSetup("/gpu/cuda/gen");
      }
      else
      {
         CeedDeviceSetup(ceed_option);
      }
   }
}

} // mfem
