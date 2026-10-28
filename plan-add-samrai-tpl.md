# Plan: Add SAMRAI as a Third-Party Library

## Context

SAMRAI (Structured Adaptive Mesh Refinement Application Infrastructure) needs to be integrated as a third-party library (TPL) into MFEM's CMake build system. Based on user clarifications:
- SAMRAI works in both serial and parallel configurations (no MPI requirement)
- No specific TPL dependencies within MFEM
- No version constraints

This integration will enable MFEM users to leverage SAMRAI's capabilities for structured adaptive mesh refinement applications.

## Implementation Steps

### 1. Add Option to defaults.cmake

- Add `option(MFEM_USE_SAMRAI ...)` near line 65 (in the TPL options section, alphabetically between RAJA and SIDRE)
- Add `SAMRAI_DIR` cache variable near line 258 (in the paths section, alphabetically placed)
- Optionally add `SAMRAI_REQUIRED_PACKAGES` if needed (likely empty based on no dependencies)

### 2. Create FindSAMRAI.cmake Module

Create `/home/vogl2/workspace/mfem-claude/config/cmake/modules/FindSAMRAI.cmake` following the pattern from FindMPFR.cmake and Find_GnuTLS.cmake:
- Define variables: `SAMRAI_FOUND`, `SAMRAI_LIBRARIES`, `SAMRAI_INCLUDE_DIRS`
- Use `mfem_find_package()` utility to search for SAMRAI headers and libraries
- Search for a representative SAMRAI header (e.g., `SAMRAI/SAMRAI_config.h` or similar)
- Search for SAMRAI library (typically `libSAMRAI` or similar)

### 3. Add find_package() Call to CMakeLists.txt

- Add conditional `find_package(SAMRAI REQUIRED)` call around line 460 (after MPFR, before CEED section)
- Follow the pattern: `if (MFEM_USE_SAMRAI) ... endif()`

### 4. Add SAMRAI to MFEM_TPLS List

- Add `SAMRAI` to the `MFEM_TPLS` list at line 631-636
- Place alphabetically (likely between RAJA and SUNDIALS based on existing ordering)

### 5. Add Preprocessor Define

- Add `#cmakedefine MFEM_USE_SAMRAI` to `/home/vogl2/workspace/mfem-claude/config/cmake/config.hpp.in`
- Place alphabetically in the TPL defines section (around line 148 near MPFR)
- Add a descriptive comment like `// Enable SAMRAI support in MFEM.`

## Critical Files

1. `/home/vogl2/workspace/mfem-claude/config/defaults.cmake` - Add option and path variables
2. `/home/vogl2/workspace/mfem-claude/config/cmake/modules/FindSAMRAI.cmake` - New file to create
3. `/home/vogl2/workspace/mfem-claude/CMakeLists.txt` - Add find_package() call and MFEM_TPLS entry
4. `/home/vogl2/workspace/mfem-claude/config/cmake/config.hpp.in` - Add preprocessor define

## Reference Examples

- **Simple TPL without dependencies**: MPFR (lines 457-459 in CMakeLists.txt, FindMPFR.cmake)
- **TPL with mfem_find_package**: Algoim, GSLIB (FindAlgoim.cmake, FindGSLIB.cmake)
- **Option definition pattern**: Lines 28-73 in defaults.cmake
- **Path variable pattern**: Lines 199-259 in defaults.cmake

## Verification Steps

After implementation:
1. Configure MFEM with `-DMFEM_USE_SAMRAI=ON -DSAMRAI_DIR=/path/to/samrai`
2. Verify CMake finds SAMRAI successfully
3. Check that `MFEM_USE_SAMRAI` is defined in generated config/_config.hpp
4. Build MFEM library to ensure linking works
5. Test with a simple program that includes MFEM and checks for SAMRAI availability

## Notes

- The exact SAMRAI header and library names should be determined based on actual SAMRAI installation (may need user input)
- If SAMRAI has an existing CMake config file (SAMRAIConfig.cmake), the Find module could potentially use CONFIG mode
- Since SAMRAI works in both serial/parallel, no MPI guards are needed around the find_package() call
