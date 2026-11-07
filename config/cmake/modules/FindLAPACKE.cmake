find_path(LAPACKE_INCLUDE_DIRS lapacke.h
  HINTS ${LAPACKE_DIR} ENV LAPACKE_DIR PATH_SUFFIXES include)
if (NOT LAPACKE_LIBRARIES)
  find_library(LAPACKE_LIBRARY NAMES lapacke openblas
    HINTS ${LAPACKE_DIR} ENV LAPACKE_DIR PATH_SUFFIXES lib lib64)
  set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
endif()

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
if (LAPACKE_INCLUDE_DIRS AND LAPACKE_LIBRARIES)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${LAPACKE_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  unset(LAPACKE_USABLE CACHE)
  check_cxx_source_compiles("#include <lapacke.h>
    static_assert(sizeof(lapack_int) == sizeof(int), \"LP64 LAPACKE required\");
    int main() {
      auto volatile svd = &LAPACKE_dgesvd;
      auto volatile eigen = &LAPACKE_dggevx;
      return svd && eigen ? 0 : 1;
    }" LAPACKE_USABLE)
  cmake_pop_check_state()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE REQUIRED_VARS
  LAPACKE_INCLUDE_DIRS LAPACKE_LIBRARIES LAPACKE_USABLE)
