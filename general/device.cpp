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

// Backends listed by priority, high to low:
static const Backend::Id backend_list[Backend::NUM_BACKENDS] =
{
   Backend::OCCA_CUDA, Backend::RAJA_CUDA, Backend::CUDA,
   Backend::HIP,
   Backend::OCCA_OMP, Backend::RAJA_OMP, Backend::OMP,
   Backend::OCCA_CPU, Backend::RAJA_CPU, Backend::CPU,
   Backend::DEBUG, Backend::UVM
};

// Backend names listed by priority, high to low:
static const char *backend_name[Backend::NUM_BACKENDS] =
{
   "occa-cuda", "raja-cuda", "cuda", "hip", "occa-omp", "raja-omp", "omp",
   "occa-cpu", "raja-cpu", "cpu", "debug", "uvm"
};

} // namespace mfem::internal


// Initialize the unique global Device variable.
Device Device::device_singleton;


Device::~Device()
{
   if (destroy_mm) { mm.Destroy(); }
   Get().host_mem_type = MemoryType::HOST;
   Get().host_mem_class = MemoryClass::HOST;
   Get().device_mem_type = MemoryType::HOST;
   Get().device_mem_class = MemoryClass::HOST;
}

void Device::Configure(const std::string &device, const int dev)
{
   std::map<std::string, Backend::Id> bmap;
   for (int i = 0; i < Backend::NUM_BACKENDS; i++)
   {
      bmap[internal::backend_name[i]] = internal::backend_list[i];
   }
   std::string::size_type beg = 0, end;
   while (1)
   {
      end = device.find(',', beg);
      end = (end != std::string::npos) ? end : device.size();
      const std::string bname = device.substr(beg, end - beg);
      std::map<std::string, Backend::Id>::iterator it = bmap.find(bname);
      MFEM_VERIFY(it != bmap.end(), "invalid backend name: '" << bname << '\'');
      Get().MarkBackend(it->second);
      if (end == device.size()) { break; }
      beg = end + 1;
   }

   // OCCA_CUDA needs CUDA or RAJA_CUDA:
   if (Allows(Backend::OCCA_CUDA) && !Allows(Backend::RAJA_CUDA))
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
   out << std::endl;
   out << "Memory configuration: "
       << MemoryTypeName[static_cast<int>(host_mem_type)];
   if (Device::Allows(Backend::DEVICE_MASK) ||
       Device::Allows(Backend::DEBUG))
   {
      out << ", " << MemoryTypeName[static_cast<int>(device_mem_type)];
   }
   out << std::endl;
}

void Device::UpdateMemoryTypeAndClass()
{
#ifdef MFEM_USE_UMPIRE
   const bool umpire = true;
#else
   const bool umpire = false;
#endif
   const bool uvm = Device::Allows(Backend::UVM);
   const bool debug = Device::Allows(Backend::DEBUG);

   // If MFEM has been compiled with Umpire support, use it as default
   if (umpire)
   {
      host_mem_type = MemoryType::HOST_UMPIRE;
      host_mem_class = MemoryClass::HOST;
   }

   // In 'debug' mode, switch for the host and device DEBUG memory types
   if (debug)
   {
      host_mem_type = MemoryType::HOST_DEBUG;
      host_mem_class = MemoryClass::DEBUG;
      device_mem_type = MemoryType::DEVICE_DEBUG;
      device_mem_class = MemoryClass::DEBUG;
   }

   // Allow to tune the device memory types, with some restrictions
   if (Device::Allows(Backend::DEVICE_MASK))
   {
      if (!uvm)
      {
         device_mem_type = MemoryType::DEVICE;
         device_mem_class = MemoryClass::DEVICE;
         if (umpire && !debug)
         {
            device_mem_type = MemoryType::DEVICE_UMPIRE;
            device_mem_class = MemoryClass::DEVICE;
         }
      }
      else
      {
         // Use the uvm shortcut if it has been requested
         host_mem_type = MemoryType::HOST_MANAGED;
         host_mem_class = MemoryClass::MANAGED;
         device_mem_type = MemoryType::DEVICE_MANAGED;
         device_mem_class = MemoryClass::MANAGED;
      }
   }

   // Update the memory manager with the new settings
   mm.Configure(host_mem_type, device_mem_type);
}

void Device::Enable()
{
   const bool accelerated =
      Get().backends & ~(Backend::CPU | Backend::DEBUG | Backend::UVM);
   if (accelerated) { Get().mode = Device::ACCELERATED;}
   // Update the host & device memory backends
   Get().UpdateMemoryTypeAndClass();
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
   if (Allows(Backend::CUDA)) { CudaDeviceSetup(dev, ngpu); }
   if (Allows(Backend::HIP)) { HipDeviceSetup(dev, ngpu); }
   if (Allows(Backend::RAJA_CUDA)) { RajaDeviceSetup(dev, ngpu); }
   // The check for MFEM_USE_OCCA is in the function OccaDeviceSetup().
   if (Allows(Backend::OCCA_MASK)) { OccaDeviceSetup(dev); }
}

} // mfem
