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

#ifndef MFEM_DEVICE_HPP
#define MFEM_DEVICE_HPP

#include "globals.hpp"

namespace mfem
{

/// The MFEM Device class that abstracts hardware devices, such as GPUs, and
/// programming models, such as CUDA, OCCA, RAJA and OpenMP.
class Device
{
private:
   enum MODES {HOST, DEVICE};

   MODES mode;
   int dev = 0;
   int ngpu = -1;
   bool cuda = false;
   bool raja = false;
   bool occa = false;
   bool omp = false;
   bool isTracking = true;

   Device(): mode{Device::HOST} {}
   Device(Device const&);
   void operator=(Device const&);
   static Device& Get() { static Device singleton; return singleton; }

   /// Setup switcher based on configuration settings
   void Setup(const int dev = 0);

public:

   /// Configure and enable the device.
   /** The string parameter will enable a backend (cuda, omp, raja, occa) if the
       corresponding substring is present (for now, the order is ignored). The
       dev argument specifies which of the devices (e.g. GPUs) to enable.  */
   static inline void Configure(std::string device, const int dev = 0)
   {
      if (device.find("cuda") != std::string::npos) { Device::UseCuda(); }
      if (device.find("omp")  != std::string::npos) { Device::UseOmp();  }
      if (device.find("raja") != std::string::npos) { Device::UseRaja(); }
      if (device.find("occa") != std::string::npos) { Device::UseOcca(); }
      EnableDevice(dev);
   }

   /// Print the configured device + programming models in order of priority
   static inline void Print(std::ostream &out = mfem::out)
   {
      const bool omp  = Device::UsingOmp();
      const bool cuda = Device::UsingCuda();
      const bool occa = Device::UsingOcca();
      const bool raja = Device::UsingRaja();
      out << "Device configuration: ";
      if (cuda && occa) { out << "OCCA:CUDA\n"; return; }
      if (omp  && occa) { out << "OCCA:OpenMP\n"; return; }
      if (occa)         { out << "OCCA:CPU\n"; return; }
      if (cuda && raja) { out << "RAJA:CUDA\n"; return; }
      if (cuda)         { out << "CUDA\n";  return; }
      if (omp  && raja) { out << "RAJA:OpenMP\n";  return; }
      if (raja)         { out << "RAJA:CPU\n";  return; }
      if (omp)          { out << "OpenMP\n";  return; }
      out << "CPU\n";
   }

   /// Enable the use of the configured device in the code that follows.
   /** In particular, use the device version of the okina kernels encountered,
       with the device versions of the data registered in the memory manager
       (copying host-to-device if necessary). */
   static inline void Enable() { Get().mode = Device::DEVICE; }

   /// Disable the use of the configured device in the code that follows.
   /** In particular, use the host version of the okina kernels encountered,
       with the host versions of the data registered in the memory manager
       (copying device-to-host if necessary). */
   static inline void Disable() { Get().mode = Device::HOST; }

   constexpr static inline bool UsingMM()
   {
#ifdef MFEM_USE_MM
      return true;
#else
      return false;
#endif
   }

   static inline void EnableDevice(const int dev = 0) { Get().Setup(dev); }
   static inline bool DeviceEnabled() { return Get().ngpu > 0; }
   static inline bool DeviceDisabled() { return Get().ngpu == 0; }
   static inline bool DeviceHasBeenEnabled() { return Get().ngpu >= 0; }

   static inline bool UsingDevice() { return DeviceEnabled() && Get().mode == DEVICE; }
   static inline bool UsingHost() { return !UsingDevice(); }

   static inline void DisableTracking() { Get().isTracking = false; };
   static inline void EnableTracking() { Get().isTracking = true; };
   static inline bool IsTracking() { return Get().isTracking; };

   static inline bool UsingCuda() { return Get().cuda; }
   static inline void UseCuda() { Get().cuda = true; }

   static inline bool UsingOmp() { return Get().omp; }
   static inline void UseOmp() { Get().omp = true; }

   static inline bool UsingRaja() { return Get().raja; }
   static inline void UseRaja() { Get().raja = true; }

   static inline bool UsingOcca() { return Get().occa; }
   static inline void UseOcca() { Get().occa = true; }

   static inline bool UsingOkina()
   {
      return UsingCuda() || UsingOmp() || UsingRaja() || UsingOcca();
   }

   ~Device();
};

} // mfem

#endif // MFEM_DEVICE_HPP
