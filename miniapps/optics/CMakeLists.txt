# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if(NOT MFEM_USE_ZLIB)
   message(STATUS "hologram: skipped (zlib required for PNG)")
else()
   set(HOLO_HAVE_CPU_FFT FALSE)
   set(HOLO_HAVE_GPU_FFT FALSE)
   set(HOLO_GPU_FFT_TARGET "")
   find_package(PkgConfig QUIET)
   # One cache entry per precision, including the pkg-config prefix.
   # A shared result keeps the first library across a precision change.
   if(MFEM_USE_SINGLE)
      if(PkgConfig_FOUND)
         pkg_check_modules(HOLO_FFTW3F QUIET fftw3f)
      endif()
      find_path(HOLO_FFTW3F_INCLUDE_DIR fftw3.h
         HINTS ${HOLO_FFTW3F_INCLUDE_DIRS})
      find_library(HOLO_FFTW3F_LIBRARY NAMES fftw3f
         HINTS ${HOLO_FFTW3F_LIBRARY_DIRS})
      set(HOLO_FFTW_INCLUDE_DIR "${HOLO_FFTW3F_INCLUDE_DIR}")
      set(HOLO_FFTW_LIBRARY "${HOLO_FFTW3F_LIBRARY}")
   else()
      if(PkgConfig_FOUND)
         pkg_check_modules(HOLO_FFTW3 QUIET fftw3)
      endif()
      find_path(HOLO_FFTW3_INCLUDE_DIR fftw3.h
         HINTS ${HOLO_FFTW3_INCLUDE_DIRS})
      find_library(HOLO_FFTW3_LIBRARY NAMES fftw3
         HINTS ${HOLO_FFTW3_LIBRARY_DIRS})
      set(HOLO_FFTW_INCLUDE_DIR "${HOLO_FFTW3_INCLUDE_DIR}")
      set(HOLO_FFTW_LIBRARY "${HOLO_FFTW3_LIBRARY}")
   endif()
   if(HOLO_FFTW_INCLUDE_DIR AND HOLO_FFTW_LIBRARY)
      set(HOLO_HAVE_CPU_FFT TRUE)
      message(STATUS "hologram: FFTW found (${HOLO_FFTW_LIBRARY})")
   else()
      message(STATUS "hologram: FFTW not found; -fft on cpu/debug will fail")
   endif()

   if(MFEM_USE_CUDA AND TARGET CUDA::cufft)
      set(HOLO_HAVE_GPU_FFT TRUE)
      set(HOLO_GPU_FFT_TARGET CUDA::cufft)
      message(STATUS "hologram: cuFFT enabled (CUDA::cufft)")
   elseif(MFEM_USE_HIP)
      find_package(hipfft QUIET)
      if(TARGET hip::hipfft)
         set(HOLO_HAVE_GPU_FFT TRUE)
         set(HOLO_GPU_FFT_TARGET hip::hipfft)
         message(STATUS "hologram: hipFFT enabled (hip::hipfft)")
      elseif(TARGET roc::hipfft)
         set(HOLO_HAVE_GPU_FFT TRUE)
         set(HOLO_GPU_FFT_TARGET roc::hipfft)
         message(STATUS "hologram: hipFFT enabled (roc::hipfft)")
      else()
         message(STATUS "hologram: hipFFT not found; -fft on gpu will fail")
      endif()
   elseif(MFEM_USE_CUDA)
      message(STATUS "hologram: CUDA::cufft not found; -fft on gpu will fail")
   endif()

   # Compilers accepting -ffp-contract=off, for C++ and as the nvcc host
   set(HOLO_FP_IDS GNU Clang AppleClang IntelLLVM)
   # -ffp-contract=off for C++, for CUDA with Clang, and through -Xcompiler for
   # nvcc with one of HOLO_FP_IDS as host compiler
   set(HOLO_FP_OPTIONS)
   if(CMAKE_CXX_COMPILER_ID IN_LIST HOLO_FP_IDS)
      list(APPEND HOLO_FP_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
      if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
         list(APPEND HOLO_FP_OPTIONS
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
      endif()
   endif()
   if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
      list(APPEND HOLO_FP_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-ffp-contract=off>)
   endif()

   macro(holo_configure target)
      target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_compile_options(${target} PRIVATE ${HOLO_FP_OPTIONS})
      if(HOLO_HAVE_CPU_FFT)
         target_include_directories(${target} PRIVATE ${HOLO_FFTW_INCLUDE_DIR})
         target_link_libraries(${target} PRIVATE ${HOLO_FFTW_LIBRARY})
         target_compile_definitions(${target} PRIVATE HOLO_USE_CPU_FFT)
      endif()
      if(HOLO_HAVE_GPU_FFT)
         target_link_libraries(${target} PRIVATE ${HOLO_GPU_FFT_TARGET})
         target_compile_definitions(${target} PRIVATE HOLO_USE_GPU_FFT)
      endif()
   endmacro()

   add_mfem_miniapp(hologram
      MAIN hologram.cpp
      EXTRA_HEADERS hologram.hpp sinc.hpp
      LIBRARIES mfem)
   holo_configure(hologram)

   if(MFEM_USE_MPI)
      add_mfem_miniapp(phologram
         MAIN phologram.cpp
         EXTRA_HEADERS hologram.hpp sinc.hpp
         LIBRARIES mfem)
      holo_configure(phologram)
   endif()

   if(MFEM_ENABLE_TESTING)
      # Test running the remaining arguments with -check -o name.png
      # -m name-mask.png, through MPIEXEC with MFEM_MPI_NP ranks when MPI is
      # given
      function(holo_add_test name)
         cmake_parse_arguments(test "MPI" "" "" ${ARGN})
         set(command ${test_UNPARSED_ARGUMENTS} -check
            -o ${CMAKE_CURRENT_BINARY_DIR}/${name}.png
            -m ${CMAKE_CURRENT_BINARY_DIR}/${name}-mask.png)
         if(test_MPI)
            set(command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
               ${MPIEXEC_PREFLAGS} ${command} ${MPIEXEC_POSTFLAGS})
         endif()
         add_test(NAME ${name} COMMAND ${command})
      endfunction()

      set(hologram $<TARGET_FILE:hologram>)
      holo_add_test(hologram_ser ${hologram} -gem -mi 2)
      holo_add_test(hologram_p1_ser ${hologram} -gem -p 1 -mi 2)
      holo_add_test(hologram_debug_ser ${hologram} -gem -d debug -mi 2)
      if(HOLO_HAVE_CPU_FFT)
         holo_add_test(hologram_fft_ser ${hologram} -fft -mi 2)
         holo_add_test(hologram_fft_debug_ser ${hologram} -fft -d debug -mi 2)
      endif()

      if(MFEM_USE_MPI)
         set(phologram $<TARGET_FILE:phologram>)
         set(np ${MFEM_MPI_NP})
         holo_add_test(phologram_np=${np} MPI ${phologram} -gem -mi 2)
         holo_add_test(phologram_p1_np=${np} MPI ${phologram} -gem -p 1 -mi 2)
         holo_add_test(phologram_debug_np=${np} MPI
            ${phologram} -gem -d debug -mi 2)
         if(HOLO_HAVE_CPU_FFT)
            holo_add_test(phologram_fft_np=${np} MPI ${phologram} -fft -mi 2)
         endif()
      endif()

      # CUDA/HIP tests
      set(device)
      if(MFEM_USE_CUDA)
         set(device cuda)
      elseif(MFEM_USE_HIP)
         set(device hip)
      endif()
      if(device)
         holo_add_test(hologram_${device}_ser
            ${hologram} -d ${device} -gem -mi 2)
         if(HOLO_HAVE_GPU_FFT)
            holo_add_test(hologram_fft_${device}_ser
               ${hologram} -d ${device} -fft -mi 2)
         endif()
         if(MFEM_USE_MPI)
            holo_add_test(phologram_${device}_np=${np} MPI
               ${phologram} -d ${device} -gem -mi 2)
            if(HOLO_HAVE_GPU_FFT)
               holo_add_test(phologram_fft_${device}_np=${np} MPI
                  ${phologram} -d ${device} -fft -mi 2)
            endif()
         endif()
      endif()
   endif()
endif()
