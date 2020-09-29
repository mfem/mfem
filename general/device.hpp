// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_DEVICE_HPP
#define MFEM_DEVICE_HPP

#include "globals.hpp"
#include "mem_manager.hpp"

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
   enum Id: unsigned long
   {
      /// [host] Default CPU backend: sequential execution on each MPI rank.
      CPU = 1 << 0,
      /// [host] OpenMP backend. Enabled when MFEM_USE_OPENMP = YES.
      OMP = 1 << 1,
      /// [device] CUDA backend. Enabled when MFEM_USE_CUDA = YES.
      CUDA = 1 << 2,
      /// [device] HIP backend. Enabled when MFEM_USE_HIP = YES.
      HIP = 1 << 3,
      /** @brief [host] RAJA CPU backend: sequential execution on each MPI rank.
          Enabled when MFEM_USE_RAJA = YES. */
      RAJA_CPU = 1 << 4,
      /** @brief [host] RAJA OpenMP backend. Enabled when MFEM_USE_RAJA = YES
          and MFEM_USE_OPENMP = YES. */
      RAJA_OMP = 1 << 5,
      /** @brief [device] RAJA CUDA backend. Enabled when MFEM_USE_RAJA = YES
          and MFEM_USE_CUDA = YES. */
      RAJA_CUDA = 1 << 6,
      /** @brief [host] OCCA CPU backend: sequential execution on each MPI rank.
          Enabled when MFEM_USE_OCCA = YES. */
      OCCA_CPU = 1 << 7,
      /// [host] OCCA OpenMP backend. Enabled when MFEM_USE_OCCA = YES.
      OCCA_OMP = 1 << 8,
      /** @brief [device] OCCA CUDA backend. Enabled when MFEM_USE_OCCA = YES
          and MFEM_USE_CUDA = YES. */
      OCCA_CUDA = 1 << 9,
      /** @brief [host] CEED CPU backend. GPU backends can still be used, but
          with expensive memory transfers. Enabled when MFEM_USE_CEED = YES. */
      CEED_CPU  = 1 << 10,
      /** @brief [device] CEED CUDA backend working together with the CUDA
          backend. Enabled when MFEM_USE_CEED = YES and MFEM_USE_CUDA = YES.
          NOTE: The current default libCEED GPU backend is non-deterministic! */
      CEED_CUDA = 1 << 11,
      /** @brief [device] Debug backend: host memory is READ/WRITE protected
          while a device is in use. It allows to test the "device" code-path
          (using separate host/device memory pools and host <-> device
          transfers) without any GPU hardware. As 'DEBUG' is sometimes used
          as a macro, `_DEVICE` has been added to avoid conflicts. */
      DEBUG_DEVICE = 1 << 12
   };

   /** @brief Additional useful constants. For example, the *_MASK constants can
       be used with Device::Allows(). */
   enum
   {
      /// Number of backends: from (1 << 0) to (1 << (NUM_BACKENDS-1)).
      NUM_BACKENDS = 13,

      /// Biwise-OR of all CPU backends
      CPU_MASK = CPU | RAJA_CPU | OCCA_CPU | CEED_CPU,
      /// Biwise-OR of all CUDA backends
      CUDA_MASK = CUDA | RAJA_CUDA | OCCA_CUDA | CEED_CUDA,
      /// Biwise-OR of all HIP backends
      HIP_MASK = HIP,
      /// Biwise-OR of all OpenMP backends
      OMP_MASK = OMP | RAJA_OMP | OCCA_OMP,
      /// Bitwise-OR of all CEED backends
      CEED_MASK = CEED_CPU | CEED_CUDA,
      /// Biwise-OR of all device backends
      DEVICE_MASK = CUDA_MASK | HIP_MASK | DEBUG_DEVICE,

      /// Biwise-OR of all RAJA backends
      RAJA_MASK = RAJA_CPU | RAJA_OMP | RAJA_CUDA,
      /// Biwise-OR of all OCCA backends
      OCCA_MASK = OCCA_CPU | OCCA_OMP | OCCA_CUDA
   };
};


/** @brief The MFEM Device class abstracts hardware devices such as GPUs, as
    well as programming models such as CUDA, OCCA, RAJA and OpenMP. */
/** This class represents a "virtual device" with the following properties:
    - At most one object of this class can be constructed and that object is
      controlled by its static methods.
    - If no Device object is constructed, the static methods will use a default
      global object which is never configured and always uses Backend::CPU.
    - Once configured, the object cannot be re-configured during the program
      lifetime.
    - MFEM classes use this object to determine where (host or device) to
      perform an operation and which backend implementation to use.
    - Multiple backends can be configured at the same time; currently, a fixed
      priority order is used to select a specific backend from the list of
      configured backends. See the Backend class and the Configure() method in
      this class for details. */
class Device
{
private:
   friend class MemoryManager;
   enum MODES {SEQUENTIAL, ACCELERATED};

   static bool device_env, mem_host_env, mem_device_env;
   static Device device_singleton;

   MODES mode;
   int dev = 0; ///< Device ID of the configured device.
   int ngpu = -1; ///< Number of detected devices; -1: not initialized.
   unsigned long backends; ///< Bitwise-OR of all configured backends.
   /// Set to true during configuration, except in 'device_singleton'.
   bool destroy_mm;
   bool mpi_gpu_aware;

   MemoryType host_mem_type;      ///< Current Host MemoryType
   MemoryClass host_mem_class;    ///< Current Host MemoryClass

   MemoryType device_mem_type;    ///< Current Device MemoryType
   MemoryClass device_mem_class;  ///< Current Device MemoryClass

   char *device_option = NULL;
   Device(Device const&);
   void operator=(Device const&);
   static Device& Get() { return device_singleton; }

   /// Setup switcher based on configuration settings
   void Setup(const int dev = 0);

   void MarkBackend(Backend::Id b) { backends |= b; }

   void UpdateMemoryTypeAndClass();

   /// Enable the use of the configured device in the code that follows.
   /** After this call MFEM classes will use the backend kernels whenever
       possible, transferring data automatically to the device, if necessary.

       If the only configured backend is the default host CPU one, the device
       will remain disabled.

       If the device is actually enabled, this method will also update the
       current host/device MemoryType and MemoryClass. */
   static void Enable();

public:
   /** @brief Default constructor. Unless Configure() is called later, the
       default Backend::CPU will be used. */
   /** @note At most one Device object can be constructed during the lifetime of
       a program.
       @note This object should be destroyed after all other MFEM objects that
       use the Device are destroyed. */
   Device();

   /** @brief Construct a Device and configure it based on the @a device string.
       See Configure() for more details. */
   /** @note At most one Device object can be constructed during the lifetime of
       a program.
       @note This object should be destroyed after all other MFEM objects that
       use the Device are destroyed. */
   Device(const std::string &device, const int dev = 0)
      : mode(Device::SEQUENTIAL),
        backends(Backend::CPU),
        destroy_mm(false),
        mpi_gpu_aware(false),
        host_mem_type(MemoryType::HOST),
        host_mem_class(MemoryClass::HOST),
        device_mem_type(MemoryType::HOST),
        device_mem_class(MemoryClass::HOST)
   { Configure(device, dev); }

   /// Destructor.
   ~Device();

   /// Configure the Device backends.
   /** The string parameter @a device must be a comma-separated list of backend
       string names (see below). The @a dev argument specifies the ID of the
       actual devices (e.g. GPU) to use.
       * The available backends are described by the Backend class.
       * The string name of a backend is the lowercase version of the
         Backend::Id enumeration constant with '_' replaced by '-', e.g. the
         string name of 'RAJA_CPU' is 'raja-cpu'. The string name of the debug
         backend (Backend::Id 'DEBUG_DEVICE') is exceptionally set to 'debug'.
       * The 'cpu' backend is always enabled with lowest priority.
       * The current backend priority from highest to lowest is:
         'ceed-cuda', 'occa-cuda', 'raja-cuda', 'cuda', 'hip', 'debug',
         'occa-omp', 'raja-omp', 'omp',
         'ceed-cpu', 'occa-cpu', 'raja-cpu', 'cpu'.
       * Multiple backends can be configured at the same time.
       * Only one 'occa-*' backend can be configured at a time.
       * The backend 'occa-cuda' enables the 'cuda' backend unless 'raja-cuda'
         is already enabled.
       * The backend 'ceed-cpu' delegates to a libCEED CPU backend the setup and
         evaluation of the operator.
       * The backend 'ceed-cuda' delegates to a libCEED CUDA backend the setup
         and evaluation of the operator and enables the 'cuda' backend to avoid
         transfer between host and device.
       * The 'debug' backend should not be combined with other device backends.
   */
   void Configure(const std::string &device, const int dev = 0);

   /// Print the configuration of the MFEM virtual device object.
   void Print(std::ostream &out = mfem::out);

   /// Return true if Configure() has been called previously.
   static inline bool IsConfigured() { return Get().ngpu >= 0; }

   /// Return true if an actual device (e.g. GPU) has been configured.
   static inline bool IsAvailable() { return Get().ngpu > 0; }

   /// Return true if any backend other than Backend::CPU is enabled.
   static inline bool IsEnabled() { return Get().mode == ACCELERATED; }

   /// The opposite of IsEnabled().
   static inline bool IsDisabled() { return !IsEnabled(); }

   /** @brief Return true if any of the backends in the backend mask, @a b_mask,
       are allowed. */
   /** This method can be used with any of the Backend::Id constants, the
       Backend::*_MASK, or combinations of those. */
   static inline bool Allows(unsigned long b_mask)
   { return Get().backends & b_mask; }

   /** @brief Get the current Host MemoryType. This is the MemoryType used by
       most MFEM classes when allocating memory used on the host.
   */
   static inline MemoryType GetHostMemoryType() { return Get().host_mem_type; }

   /** @brief Get the current Host MemoryClass. This is the MemoryClass used
       by most MFEM host Memory objects. */
   static inline MemoryClass GetHostMemoryClass() { return Get().host_mem_class; }

   /** @brief Get the current Device MemoryType. This is the MemoryType used by
       most MFEM classes when allocating memory to be used with device kernels.
   */
   static inline MemoryType GetDeviceMemoryType() { return Get().device_mem_type; }

   /// (DEPRECATED) Equivalent to GetDeviceMemoryType().
   /** @deprecated Use GetDeviceMemoryType() instead. */
   static inline MemoryType GetMemoryType() { return Get().device_mem_type; }

   /** @brief Get the current Device MemoryClass. This is the MemoryClass used
       by most MFEM device kernels to access Memory objects. */
   static inline MemoryClass GetDeviceMemoryClass() { return Get().device_mem_class; }

   /// (DEPRECATED) Equivalent to GetDeviceMemoryClass().
   /** @deprecated Use GetDeviceMemoryClass() instead. */
   static inline MemoryClass GetMemoryClass() { return Get().device_mem_class; }

   static void SetGPUAwareMPI(const bool force = true)
   { Get().mpi_gpu_aware = force; }

   static bool GetGPUAwareMPI() { return Get().mpi_gpu_aware; }
};


// Inline Memory access functions using the mfem::Device DeviceMemoryClass or
// the mfem::Device HostMemoryClass.

/** @brief Return the memory class to be used by the functions Read(), Write(),
    and ReadWrite(), while setting the device use flag in @a mem, if @a on_dev
    is true. */
template <typename T>
MemoryClass GetMemoryClass(const Memory<T> &mem, bool on_dev)
{
   if (!on_dev)
   {
      return Device::GetHostMemoryClass();
   }
   else
   {
      mem.UseDevice(true);
      return Device::GetDeviceMemoryClass();
   }
}

/** @brief Get a pointer for read access to @a mem with the mfem::Device's
    DeviceMemoryClass, if @a on_dev = true, or the mfem::Device's
    HostMemoryClass, otherwise. */
/** Also, if @a on_dev = true, the device flag of @a mem will be set. */
template <typename T>
inline const T *Read(const Memory<T> &mem, int size, bool on_dev = true)
{
   return mem.Read(GetMemoryClass(mem, on_dev), size);
}

/** @brief Shortcut to Read(const Memory<T> &mem, int size, false) */
template <typename T>
inline const T *HostRead(const Memory<T> &mem, int size)
{
   return mfem::Read(mem, size, false);
}

/** @brief Get a pointer for write access to @a mem with the mfem::Device's
    DeviceMemoryClass, if @a on_dev = true, or the mfem::Device's
    HostMemoryClass, otherwise. */
/** Also, if @a on_dev = true, the device flag of @a mem will be set. */
template <typename T>
inline T *Write(Memory<T> &mem, int size, bool on_dev = true)
{
   return mem.Write(GetMemoryClass(mem, on_dev), size);
}

/** @brief Shortcut to Write(const Memory<T> &mem, int size, false) */
template <typename T>
inline T *HostWrite(Memory<T> &mem, int size)
{
   return mfem::Write(mem, size, false);
}

/** @brief Get a pointer for read+write access to @a mem with the mfem::Device's
    DeviceMemoryClass, if @a on_dev = true, or the mfem::Device's
    HostMemoryClass, otherwise. */
/** Also, if @a on_dev = true, the device flag of @a mem will be set. */
template <typename T>
inline T *ReadWrite(Memory<T> &mem, int size, bool on_dev = true)
{
   return mem.ReadWrite(GetMemoryClass(mem, on_dev), size);
}

/** @brief Shortcut to ReadWrite(Memory<T> &mem, int size, false) */
template <typename T>
inline T *HostReadWrite(Memory<T> &mem, int size)
{
   return mfem::ReadWrite(mem, size, false);
}

} // mfem

#endif // MFEM_DEVICE_HPP
