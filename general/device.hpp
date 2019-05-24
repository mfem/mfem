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

/// MFEM backends.
/** Individual backends will generally implement only a subset of the kernels
    implemented by the default CPU backend. The goal of the backends is to
    accelerate data-parallel portions of the code and they can use a device
    memory space (e.g. GPUs) or share the memory space of the host (OpenMP). */
struct Backend
{
   /** @brief In the documentation below, we use square brackets to indicate the
       type of the backend: host or device. */
   enum Id
   {
      /// [host] Default CPU backend: sequential execution on each MPI rank.
      CPU = 1 << 0,
      /// [host] OpenMP backend. Enabled when MFEM_USE_OPENMP = YES.
      OMP = 1 << 1,
      /// [device] CUDA backend. Enabled when MFEM_USE_CUDA = YES.
      CUDA = 1 << 2,
      /** @brief [host] RAJA CPU backend: sequential execution on each MPI rank.
          Enabled when MFEM_USE_RAJA = YES. */
      RAJA_CPU = 1 << 3,
      /** @brief [host] RAJA OpenMP backend. Enabled when MFEM_USE_RAJA = YES
          and MFEM_USE_OPENMP = YES. */
      RAJA_OMP = 1 << 4,
      /** @brief [device] RAJA CUDA backend. Enabled when MFEM_USE_RAJA = YES
          and MFEM_USE_CUDA = YES. */
      RAJA_CUDA = 1 << 5,
      /** @brief [host] OCCA CPU backend: sequential execution on each MPI rank.
          Enabled when MFEM_USE_OCCA = YES. */
      OCCA_CPU = 1 << 6,
      /// [host] OCCA OpenMP backend. Enabled when MFEM_USE_OCCA = YES.
      OCCA_OMP = 1 << 7,
      /** @brief [device] OCCA CUDA backend. Enabled when MFEM_USE_OCCA = YES
          and MFEM_USE_CUDA = YES. */
      OCCA_CUDA = 1 << 8
   };

   /** @brief Additional useful constants. For example, the *_MASK constants can
       be used with Device::Allows(). */
   enum
   {
      /// Number of backends: from (1 << 0) to (1 << (NUM_BACKENDS-1)).
      NUM_BACKENDS = 9,
      /// Biwise-OR of all CUDA backends
      CUDA_MASK = CUDA | RAJA_CUDA | OCCA_CUDA,
      /// Biwise-OR of all RAJA backends
      RAJA_MASK = RAJA_CPU | RAJA_OMP | RAJA_CUDA,
      /// Biwise-OR of all OCCA backends
      OCCA_MASK = OCCA_CPU | OCCA_OMP | OCCA_CUDA,
      /// Biwise-OR of all OpenMP backends
      OMP_MASK = OMP | RAJA_OMP | OCCA_OMP,
      /// Biwise-OR of all device backends
      DEVICE_MASK = CUDA_MASK
   };
};


/** @brief The MFEM Device class abstracts hardware devices, such as GPUs, as
    well as programming models, such as CUDA, OCCA, RAJA and OpenMP. */
/** This class represents a "virtual device" with the following properties:
    - There a single object of this class which is controlled by its static
      methods.
    - Once configured, the object cannot be re-configured during the program
      lifetime.
    - MFEM classes use this object to determine where (host or device) to
      perform an operation and which backend implementation to use.
    - Multiple backends can be configured at the same time; currently, a fixed
      priority order is used to select a specific backend from the list of
      configured backends. See the Backend class and the Configure() method in
      this class for details.
    - The device can be disabled to restrict the backend selection to only the
      default host CPU backend, see the methods Enable() and Disable(). */
class Device
{
private:
   enum MODES {SEQUENTIAL, ACCELERATED};

   MODES mode;
   int dev = 0; ///< Device ID of the configured device.
   int ngpu = -1; ///< Number of detected devices; -1: not initialized.
   unsigned long backends; ///< Bitwise-OR of all configured backends.
   /** Bitwise-OR mask of all allowed backends. All backends are active when the
       Device is enabled. When the Device is disabled, only the host CPU backend
       is allowed. */
   unsigned long allowed_backends;

   Device()
      : mode(Device::SEQUENTIAL),
        backends(Backend::CPU),
        allowed_backends(backends) { }
   Device(Device const&);
   void operator=(Device const&);
   static Device& Get() { static Device singleton; return singleton; }

   /// Setup switcher based on configuration settings
   void Setup(const int dev = 0);

   void MarkBackend(Backend::Id b) { backends |= b; }

public:
   /// Configure the Device backends.
   /** The string parameter @a device must be a comma-separated list of backend
       string names (see below). The @a dev argument specifies the ID of the
       actual devices (e.g. GPU) to use.
       * The available backends are described by the Backend class.
       * The string name of a backend is the lowercase version of the
         Backend::Id enumeration constant with '_' replaced by '-', e.g. the
         string name of 'RAJA_CPU' is 'raja-cpu'.
       * The 'cpu' backend is always enabled with lowest priority.
       * The current backend priority from highest to lowest is: 'occa-cuda',
         'raja-cuda', 'cuda', 'occa-omp', 'raja-omp', 'omp', 'occa-cpu',
         'raja-cpu', 'cpu'.
       * Multiple backends can be configured at the same time.
       * Only one 'occa-*' backend can be configured at a time.
       * The backend 'occa-cuda' enables the 'cuda' backend unless 'raja-cuda'
         is already enabled.
       * After this call, the Device will be disabled. */
   static void Configure(const std::string &device, const int dev = 0);

   /// Print the configuration of the MFEM virtual device object.
   static void Print(std::ostream &out = mfem::out);

   /// Return true if Configure() has been called previously.
   static inline bool IsConfigured() { return Get().ngpu >= 0; }

   /// Return true if an actual device (e.g. GPU) has been configured.
   static inline bool IsAvailable() { return Get().ngpu > 0; }

   /// Enable the use of the configured device in the code that follows.
   /** After this call MFEM classes will use the backend kernels whenever
       possible, transferring data automatically to the device, if necessary.

       If the only configured backend is the default host CPU one, the device
       will remain disabled. */
   static inline void Enable()
   {
      if (Get().backends & ~Backend::CPU)
      {
         Get().mode = Device::ACCELERATED;
         Get().allowed_backends = Get().backends;
      }
   }

   /// Disable the use of the configured device in the code that follows.
   /** After this call MFEM classes will only use default CPU kernels,
       transferring data automatically from the device, if necessary. */
   static inline void Disable()
   {
      Get().mode = Device::SEQUENTIAL;
      Get().allowed_backends = Backend::CPU;
   }

   /// Return true if the Device is enabled.
   static inline bool IsEnabled() { return Get().mode == ACCELERATED; }

   /// The opposite of IsEnabled().
   static inline bool IsDisabled() { return !IsEnabled(); }

   /** @brief Return true if any of the backends in the backend mask, @a b_mask,
       are allowed. The allowed backends are all configured backends minus the
       device backends when the Device is disabled. */
   /** This method can be used with any of the Backend::Id constants, the
       Backend::*_MASK, or combinations of those. */
   static inline bool Allows(unsigned long b_mask)
   { return Get().allowed_backends & b_mask; }
};

} // mfem

#endif // MFEM_DEVICE_HPP
