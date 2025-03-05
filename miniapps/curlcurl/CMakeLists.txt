if (MFEM_USE_MPI)
  list(APPEND CURLCURL_SOURCES
      CurlCurlIntegrator.cpp)
  list(APPEND CURLCURL_HEADERS
      CurlCurlIntegrator.hpp)

  convert_filenames_to_full_paths(CURLCURL_SOURCES)
  convert_filenames_to_full_paths(CURLCURL_HEADERS)

  set(CURLCURL_COMMON_FILES
      EXTRA_SOURCES ${CURLCURL_SOURCES}
      EXTRA_HEADERS ${CURLCURL_HEADERS})

  add_mfem_miniapp(curlcurl
      MAIN curlcurlB.cpp
      ${CURLCURL_COMMON_FILES}
      LIBRARIES mfem)

  add_mfem_miniapp(alfven_system
      MAIN alfven_system.cpp
      ${CURLCURL_COMMON_FILES}
      LIBRARIES mfem)

  add_mfem_miniapp(nl_alfven_system
      MAIN nl_alfven_system.cpp
      ${CURLCURL_COMMON_FILES}
      LIBRARIES mfem)
endif ()
