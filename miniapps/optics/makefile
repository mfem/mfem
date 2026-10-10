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

ifeq ($(MFEM_USE_SINGLE),YES)
   HOLO_FFTW_PC = fftw3f
else
   HOLO_FFTW_PC = fftw3
endif

ifneq ($(shell pkg-config --exists $(HOLO_FFTW_PC) && echo yes),)
   HOLO_FFT_FLAGS += -DHOLO_USE_CPU_FFT $(shell pkg-config --cflags $(HOLO_FFTW_PC))
   HOLO_FFT_LIBS += $(shell pkg-config --libs $(HOLO_FFTW_PC))
endif
ifeq ($(MFEM_USE_CUDA),YES)
   HOLO_FFT_FLAGS += -DHOLO_USE_GPU_FFT
   HOLO_FFT_LIBS += -lcufft
else ifeq ($(MFEM_USE_HIP),YES)
   ifneq ($(shell pkg-config --exists hipfft && echo yes),)
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

# mfem-test prints the executable, and these cases share one binary.
# $(1) summary name, $(2) MPI launcher, $(3) label, $(4) arguments, $(5) executable.
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
holo-test = \
   printf "   $(3) [$(2) $(1) ... ]: "; \
   $(call $(TIMEFUN),$(TIMECMD),$(2) ./$(5) $(4) > $(1).stderr 2>&1); \
   err="$$3"; \
   if [ "$$3" = 0 ]; then $(PRINT_OK); \
   else if [ "$$3" = 242 ]; then $(PRINT_SKIP); err=0; \
   else $(PRINT_FAILED); cat $(1).stderr; fi; fi; \
   rm -f $(1).stderr; exit $$err
test-print: holo-test=printf "   $(3) [$(2) $(1) ... ]\n"

hologram-test-seq: hologram
	@$(call holo-test,hologram_gem,, Optics miniapp,-gem,hologram)
	@$(call holo-test,hologram_gem_p1,, Optics miniapp,-gem -p 1 \
	   -o hologram-o1.png -m mask-o1.png,hologram)
	@$(call holo-test,hologram_gem_debug,, Optics miniapp,-gem -d debug -mi 2 \
	   -o hologram-debug.png -m mask-debug.png,hologram)
ifneq ($(shell pkg-config --exists $(HOLO_FFTW_PC) && echo yes),)
	@$(call holo-test,hologram_fft,, Optics miniapp,-fft -mi 2 \
	   -o hologram-fft.png -m mask-fft.png,hologram)
	@$(call holo-test,hologram_fft_debug,, Optics miniapp,-d debug -fft -mi 2 \
	   -o hologram-fft-debug.png -m mask-fft-debug.png,hologram)
endif

phologram-test-par: phologram
	@$(call holo-test,phologram_gem_debug, $(RUN_MPI), Optics miniapp,\
	   -gem -d debug -mi 2 -o hologram-gem-mpi.png -m mask-gem-mpi.png,phologram)
ifneq ($(shell pkg-config --exists $(HOLO_FFTW_PC) && echo yes),)
	@$(call holo-test,phologram_fft, $(RUN_MPI), Optics miniapp,-fft -mi 2 \
	   -o hologram-fft-mpi.png -m mask-fft-mpi.png,phologram)
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
