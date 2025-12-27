include(MfemCmakeUtilities)

mfem_find_package(CUDSS CUDSS CUDSS_DIR
  "include" cudss.h "lib" cudss
  "Paths to headers required by cuDSS."
  "Libraries required by cuDSS.")
