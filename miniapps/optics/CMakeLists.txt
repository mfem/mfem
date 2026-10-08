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
   if(PkgConfig_FOUND)
      if(MFEM_USE_SINGLE)
         pkg_check_modules(FFTW3 QUIET fftw3f)
      else()
         pkg_check_modules(FFTW3 QUIET fftw3)
      endif()
   endif()
   if(MFEM_USE_SINGLE)
      set(HOLO_FFTW_LIB_NAME fftw3f)
   else()
      set(HOLO_FFTW_LIB_NAME fftw3)
   endif()
   find_path(HOLO_FFTW_INCLUDE_DIR fftw3.h HINTS ${FFTW3_INCLUDE_DIRS})
   find_library(HOLO_FFTW_LIBRARY NAMES ${HOLO_FFTW_LIB_NAME}
      HINTS ${FFTW3_LIBRARY_DIRS})
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

   macro(holo_configure target)
      target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
      target_compile_options(${target} PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
         $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-ffp-contract=off>
         $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-ffp-contract=off>)
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
      add_test(NAME hologram-logo
         COMMAND sh -c
         "\"$1\" -gem -o \"$2\" -m \"$3\" && test -s \"$2\" && test -s \"$3\""
         _ $<TARGET_FILE:hologram>
         ${CMAKE_CURRENT_BINARY_DIR}/hologram.png
         ${CMAKE_CURRENT_BINARY_DIR}/mask.png)
      add_test(NAME hologram-debug
         COMMAND sh -c
         "\"$1\" -gem -d debug -mi 2 -o \"$2\" -m \"$3\" && test -s \"$2\" && test -s \"$3\""
         _ $<TARGET_FILE:hologram>
         ${CMAKE_CURRENT_BINARY_DIR}/hologram-debug.png
         ${CMAKE_CURRENT_BINARY_DIR}/mask-debug.png)
      # Newton 1/2 losses measured for -gem -mi 2. HOLO_LOSS_PY is not used here.
      set(HOLO_NEWTON_PY_HEAD
         "import re,sys; t=open(sys.argv[1]).read(); m=re.search(r'Newton 1/[0-9]+ \\([0-9]+\\) ([0-9.eE+-]+)', t); v=float(m.group(1)); ref=")
      set(HOLO_NEWTON_PY_TAIL
         "; assert abs(v-ref)<=5e-4*ref, (v,ref)")
      add_test(NAME hologram-gem-p0
         COMMAND sh -c
         "\"$1\" -gem -p 0 -mi 2 -o \"$2\" -m \"$3\" >\"$4\" && test -s \"$2\" && test -s \"$3\" && python3 -c \"${HOLO_NEWTON_PY_HEAD}7.4119133463282916e-02${HOLO_NEWTON_PY_TAIL}\" \"$4\""
         _ $<TARGET_FILE:hologram>
         ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p0.png
         ${CMAKE_CURRENT_BINARY_DIR}/mask-gem-p0.png
         ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p0.log)
      add_test(NAME hologram-gem-p1
         COMMAND sh -c
         "\"$1\" -gem -p 1 -mi 2 -o \"$2\" -m \"$3\" >\"$4\" && test -s \"$2\" && test -s \"$3\" && python3 -c \"${HOLO_NEWTON_PY_HEAD}7.4119133463282874e-02${HOLO_NEWTON_PY_TAIL}\" \"$4\""
         _ $<TARGET_FILE:hologram>
         ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p1.png
         ${CMAKE_CURRENT_BINARY_DIR}/mask-gem-p1.png
         ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p1.log)
      # Loss is printed as "%6.2e" ("Newton 1/2 (5) 7.41e-02")
      set(HOLO_LOSS_PY
         "import re,sys; t=open(sys.argv[1]).read(); m=re.search(r'Newton 1/[0-9]+ \\([0-9]+\\) ([0-9.eE+-]+)', t); v=float(m.group(1)); ref=0.0741191; assert abs(v-ref)<=5e-4*ref, (v,ref)")
      if(HOLO_HAVE_CPU_FFT)
         add_test(NAME hologram-fft
            COMMAND sh -c
            "\"$1\" -fft -mi 2 -o \"$2\" -m \"$3\" >\"$4\" && test -s \"$2\" && test -s \"$3\" && python3 -c \"${HOLO_LOSS_PY}\" \"$4\""
            _ $<TARGET_FILE:hologram>
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft.png
            ${CMAKE_CURRENT_BINARY_DIR}/mask-fft.png
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft.log)
         add_test(NAME hologram-fft-debug
            COMMAND sh -c
            "\"$1\" -d debug -fft -mi 2 -o \"$2\" -m \"$3\" >\"$4\" && test -s \"$2\" && test -s \"$3\""
            _ $<TARGET_FILE:hologram>
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft-debug.png
            ${CMAKE_CURRENT_BINARY_DIR}/mask-fft-debug.png
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft-debug.log)
      endif()
      if(MFEM_USE_MPI)
         add_test(NAME hologram-gem-np${MFEM_MPI_NP}
            COMMAND sh -c
            "\"$1\" \"$2\" \"$3\" \"$4\" -gem -d debug -mi 2 -o \"$5\" -m \"$6\" >\"$7\" && test -s \"$5\" && test -s \"$6\" && python3 -c \"${HOLO_LOSS_PY}\" \"$7\""
            _ ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
            $<TARGET_FILE:phologram>
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/mask-gem-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-mpi.log)
         if(HOLO_HAVE_CPU_FFT)
            add_test(NAME hologram-fft-np${MFEM_MPI_NP}
               COMMAND sh -c
               "\"$1\" \"$2\" \"$3\" \"$4\" -fft -mi 2 -o \"$5\" -m \"$6\" >\"$7\" && test -s \"$5\" && test -s \"$6\" && python3 -c \"${HOLO_LOSS_PY}\" \"$7\""
               _ ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
               $<TARGET_FILE:phologram>
               ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft-mpi.png
               ${CMAKE_CURRENT_BINARY_DIR}/mask-fft-mpi.png
               ${CMAKE_CURRENT_BINARY_DIR}/hologram-fft-mpi.log)
         endif()
         add_test(NAME hologram-gem-p0-np${MFEM_MPI_NP}
            COMMAND sh -c
            "\"$1\" \"$2\" \"$3\" \"$4\" -gem -p 0 -mi 2 -o \"$5\" -m \"$6\" >\"$7\" && test -s \"$5\" && test -s \"$6\" && python3 -c \"${HOLO_NEWTON_PY_HEAD}7.4119133463282624e-02${HOLO_NEWTON_PY_TAIL}\" \"$7\""
            _ ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
            $<TARGET_FILE:phologram>
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p0-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/mask-gem-p0-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p0-mpi.log)
         add_test(NAME hologram-gem-p1-np${MFEM_MPI_NP}
            COMMAND sh -c
            "\"$1\" \"$2\" \"$3\" \"$4\" -gem -p 1 -mi 2 -o \"$5\" -m \"$6\" >\"$7\" && test -s \"$5\" && test -s \"$6\" && python3 -c \"${HOLO_NEWTON_PY_HEAD}7.4119133463282624e-02${HOLO_NEWTON_PY_TAIL}\" \"$7\""
            _ ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
            $<TARGET_FILE:phologram>
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p1-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/mask-gem-p1-mpi.png
            ${CMAKE_CURRENT_BINARY_DIR}/hologram-gem-p1-mpi.log)
      endif()
   endif()
endif()
