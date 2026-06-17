# =============================================================================
# Makefile  —  mma_mfem MFEM miniapp
#
# This Makefile follows the MFEM miniapp convention:
#   make           — build all test executables and standalone miniapps
#   make tests     — same as above (explicit alias)
#   make check     — build and run the full test suite via run_tests.sh
#   make serial    — run only serial tests (faster, no mpirun required)
#   make clean     — remove build artefacts
#   make distclean — remove the entire build/ directory
#
# Required variable
# -----------------
#   MFEM_BUILD_DIR   Path to the MFEM build or install directory.
#                    The Makefile reads MFEM's own config fragment
#                    (config/config.mk) to inherit all compiler flags,
#                    link flags, and feature macros automatically.
#
#                    Can be set on the command line:
#                      make MFEM_BUILD_DIR=/path/to/mfem/build
#                    or exported in your environment before running make.
#
# Optional variables
# ------------------
#   MFEM_BUILD_DIR   (see above)
#   BUILD_DIR        Where to put objects and executables (default: ./build)
#   CXX              C++ compiler (inherited from MFEM config if not set)
#   CXXFLAGS         Extra compiler flags (appended after MFEM flags)
#   LDFLAGS          Extra linker flags
#   MPIRUN           MPI launcher (default: mpirun)
#   NP               Number of MPI ranks for parallel tests (default: 4)
#
# Example
# -------
#   make MFEM_BUILD_DIR=~/software/mfem -j8
#   make check MFEM_BUILD_DIR=~/software/mfem NP=8
# =============================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Default build directory (mirrors cmake out-of-source convention)
BUILD_DIR ?= build

# MFEM config fragment: provides CXX, MFEM_FLAGS, MFEM_LIBS, etc.
# Try the two standard locations: in-source build and installed prefix.
ifndef MFEM_BUILD_DIR
  $(error MFEM_BUILD_DIR is not set.  Run: make MFEM_BUILD_DIR=/path/to/mfem)
endif

MFEM_CONFIG := $(wildcard \
    $(MFEM_BUILD_DIR)/config/config.mk \
    $(MFEM_BUILD_DIR)/share/mfem/config.mk)

ifeq ($(MFEM_CONFIG),)
  $(error Could not find config/config.mk under $(MFEM_BUILD_DIR). \
          Check that MFEM_BUILD_DIR points to the MFEM build root.)
endif

# Pull in MFEM's own definitions: CXX, MFEM_FLAGS, MFEM_LIBS, MFEM_INC, …
include $(firstword $(MFEM_CONFIG))

AR ?= ar

# ---------------------------------------------------------------------------
# Compiler and flags
# ---------------------------------------------------------------------------

# MFEM_INC / MFEM_LIBS come from config.mk.
# We add the project source directory so #include "MMA_MFEM.hpp" works.
INC    := -I. $(MFEM_INC)
LIBS   := $(MFEM_LIBS) $(LDFLAGS)

# MFEM_FLAGS already contains -std=c++11/14, optimisation level, and any
# feature macros (MFEM_USE_MPI, MFEM_USE_LAPACK, …).
CXXFLAGS_ALL := $(MFEM_FLAGS) $(CXXFLAGS)

# ---------------------------------------------------------------------------
# Sources and targets
# ---------------------------------------------------------------------------

LIB_SRC  := MMA_MFEM.cpp
LIB_OBJ  := $(BUILD_DIR)/MMA_MFEM.o
LIB_A    := $(BUILD_DIR)/libmma_mfem.a

# All test source files (anything matching test_*.cpp)
TEST_SRCS := $(wildcard test_*.cpp)
TEST_BINS := $(patsubst %.cpp,$(BUILD_DIR)/%,$(TEST_SRCS))

# Standalone miniapps in this directory that do not follow the test_*.cpp
# naming convention.
APP_SRCS := ex41p_stokes_adv_diff.cpp
APP_NAMES := $(patsubst %.cpp,%,$(APP_SRCS))
APP_BINS := $(patsubst %.cpp,$(BUILD_DIR)/%,$(APP_SRCS))

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

.PHONY: all tests check serial clean distclean config $(APP_NAMES)

all: $(LIB_A) $(TEST_BINS) $(APP_BINS)

tests: all

# ---------------------------------------------------------------------------
# Build rules
# ---------------------------------------------------------------------------

# Create the build directory on demand
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile the core library object
$(LIB_OBJ): $(LIB_SRC) MMA_MFEM.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_ALL) $(INC) -c $< -o $@

# Archive into a static library so test link lines stay short
$(LIB_A): $(LIB_OBJ)
	$(AR) rcs $@ $^

# Compile and link each test binary
# Pattern rule: build/test_foo  <—  test_foo.cpp + libmma_mfem.a
$(BUILD_DIR)/test_%: test_%.cpp $(LIB_A) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_ALL) $(INC) $< $(LIB_A) $(LIBS) -o $@

# Compile and link standalone miniapps. The convenience target lets users run
# `make ex41p_stokes_adv_diff` while still placing the binary under BUILD_DIR.
$(APP_NAMES): %: $(BUILD_DIR)/%

$(APP_BINS): $(BUILD_DIR)/%: %.cpp $(LIB_A) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_ALL) $(INC) $< $(LIB_A) $(LIBS) -o $@

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

MPIRUN ?= mpirun
NP     ?= 4

# Full suite (build first, then delegate to run_tests.sh)
check: all
	@bash run_tests.sh -b $(BUILD_DIR) -j $(NP)

# Serial-only (no mpirun needed — useful on login nodes)
serial: all
	@bash run_tests.sh -b $(BUILD_DIR) --serial

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.a $(TEST_BINS) $(APP_BINS)

distclean:
	rm -rf $(BUILD_DIR)

# ---------------------------------------------------------------------------
# Convenience: print the resolved configuration
# ---------------------------------------------------------------------------

config:
	@echo "MFEM_BUILD_DIR : $(MFEM_BUILD_DIR)"
	@echo "MFEM_CONFIG    : $(MFEM_CONFIG)"
	@echo "CXX            : $(CXX)"
	@echo "MFEM_FLAGS     : $(MFEM_FLAGS)"
	@echo "MFEM_INC       : $(MFEM_INC)"
	@echo "BUILD_DIR      : $(BUILD_DIR)"
	@echo "Test targets   :"
	@echo "  $(notdir $(TEST_BINS))" | tr ' ' '\n' | sed 's/^/    /'
	@echo "App targets    :"
	@echo "  $(APP_NAMES)" | tr ' ' '\n' | sed 's/^/    /'
