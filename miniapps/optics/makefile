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

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/optics/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

ifeq ($(MFEM_USE_ZLIB),YES)
SEQ_MINIAPPS = hologram
PAR_MINIAPPS = phologram
else
SEQ_MINIAPPS =
PAR_MINIAPPS =
endif
SEQ_DEVICE_MINIAPPS = $(SEQ_MINIAPPS)
PAR_DEVICE_MINIAPPS = $(PAR_MINIAPPS)
ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS = $(SEQ_MINIAPPS)
else
   MINIAPPS = $(PAR_MINIAPPS) $(SEQ_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

HOLO_COMMON = $(SRC)hologram.hpp $(SRC)sinc.hpp $(CONFIG_MK)
HOLO_INCLUDES = -I$(if $(SRC),$(SRC),.)
HOLO_FFT_FLAGS =
HOLO_FFT_LIBS =
HOLO_CPU_FFT =
HOLO_GPU_FFT =

ifeq ($(MFEM_USE_SINGLE),YES)
   HOLO_FFTW_PC = fftw3f
else
   HOLO_FFTW_PC = fftw3
endif

ifneq ($(shell pkg-config --exists $(HOLO_FFTW_PC) && echo yes),)
   HOLO_CPU_FFT = YES
   HOLO_FFT_FLAGS += -DHOLO_USE_CPU_FFT $(shell pkg-config --cflags $(HOLO_FFTW_PC))
   HOLO_FFT_LIBS += $(shell pkg-config --libs $(HOLO_FFTW_PC))
endif
ifeq ($(MFEM_USE_CUDA),YES)
   HOLO_GPU_FFT = YES
   HOLO_FFT_FLAGS += -DHOLO_USE_GPU_FFT
   HOLO_FFT_LIBS += -lcufft
else ifeq ($(MFEM_USE_HIP),YES)
   ifneq ($(shell pkg-config --exists hipfft && echo yes),)
      HOLO_GPU_FFT = YES
      HOLO_FFT_FLAGS += -DHOLO_USE_GPU_FFT $(shell pkg-config --cflags hipfft)
      HOLO_FFT_LIBS += $(shell pkg-config --libs hipfft)
   endif
endif

all: $(MINIAPPS)

hologram.o: $(SRC)hologram.cpp $(HOLO_COMMON)
	$(MFEM_CXX) $(MFEM_FLAGS) $(HOLO_INCLUDES) $(HOLO_FFT_FLAGS) \
	   -c $(SRC)hologram.cpp -o $@

phologram.o: $(SRC)phologram.cpp $(HOLO_COMMON)
	$(MFEM_CXX) $(MFEM_FLAGS) $(HOLO_INCLUDES) $(HOLO_FFT_FLAGS) \
	   -c $(SRC)phologram.cpp -o $@

hologram: hologram.o $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ hologram.o $(MFEM_LIBS) \
	   $(HOLO_FFT_LIBS)

phologram: phologram.o $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o $@ phologram.o $(MFEM_LIBS) \
	   $(HOLO_FFT_LIBS)

MFEM_TESTS = MINIAPPS
include $(MFEM_TEST_MK)

# Testing: hologram has no -no-vis option
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
hologram-test-seq: hologram
	@$(call mfem-test,$<,, Optics miniapp gem,-gem -mi 2 -check,SKIP-NO-VIS)
	@$(call mfem-test,$<,, Optics miniapp gem p1,-gem -p 1 -mi 2 -check,\
	   SKIP-NO-VIS)
	@$(call mfem-test,$<,, Optics miniapp gem debug,-gem -d debug -mi 2 -check,\
	   SKIP-NO-VIS)
ifneq ($(HOLO_CPU_FFT),)
	@$(call mfem-test,$<,, Optics miniapp fft,-fft -mi 2 -check,SKIP-NO-VIS)
	@$(call mfem-test,$<,, Optics miniapp fft debug,-fft -d debug -mi 2 -check,\
	   SKIP-NO-VIS)
endif
phologram-test-par: phologram
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp gem,\
	   -gem -mi 2 -check,SKIP-NO-VIS)
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp gem p1,\
	   -gem -p 1 -mi 2 -check,SKIP-NO-VIS)
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp gem debug,\
	   -gem -d debug -mi 2 -check,SKIP-NO-VIS)
ifneq ($(HOLO_CPU_FFT),)
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp fft,\
	   -fft -mi 2 -check,SKIP-NO-VIS)
endif

# Testing: CUDA/HIP runs
hologram-test-seq-%: hologram
	@$(call mfem-test,$<,, Optics miniapp gem $*,-d $* -gem -mi 2 -check,\
	   SKIP-NO-VIS)
ifneq ($(HOLO_GPU_FFT),)
	@$(call mfem-test,$<,, Optics miniapp fft $*,-d $* -fft -mi 2 -check,\
	   SKIP-NO-VIS)
endif
phologram-test-par-%: phologram
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp gem $*,\
	   -d $* -gem -mi 2 -check,SKIP-NO-VIS)
ifneq ($(HOLO_GPU_FFT),)
	@$(call mfem-test,$<, $(RUN_MPI), Parallel optics miniapp fft $*,\
	   -d $* -fft -mi 2 -check,SKIP-NO-VIS)
endif

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_MINIAPPS) $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f hologram.png mask.png hologram-o1.png mask-o1.png \
	   hologram-debug.png mask-debug.png \
	   hologram-fft.png mask-fft.png hologram-fft-debug.png mask-fft-debug.png \
	   hologram-gem-mpi.png mask-gem-mpi.png hologram-fft-mpi.png mask-fft-mpi.png
