# Install script for directory: /Users/ben/Documents/SoftwareLibraries/mfem

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/libmfem.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmfem.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmfem.a")
    execute_process(COMMAND "/Library/Developer/CommandLineTools/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmfem.a")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/InstallHeaders/mfem.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/InstallHeaders/mfem-performance.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mfem" TYPE FILE FILES
    "/Users/ben/Documents/SoftwareLibraries/mfem/mfem.hpp"
    "/Users/ben/Documents/SoftwareLibraries/mfem/mfem-performance.hpp"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mfem" TYPE DIRECTORY FILES
    "/Users/ben/Documents/SoftwareLibraries/mfem/general"
    "/Users/ben/Documents/SoftwareLibraries/mfem/linalg"
    "/Users/ben/Documents/SoftwareLibraries/mfem/mesh"
    "/Users/ben/Documents/SoftwareLibraries/mfem/fem"
    FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mfem/config" TYPE FILE RENAME "config.hpp" FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/config/_config.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mfem/config" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/config/tconfig.hpp")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem" TYPE FILE FILES
    "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/CMakeFiles/MFEMConfig.cmake"
    "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/MFEMConfigVersion.cmake"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem/MFEMTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem/MFEMTargets.cmake"
         "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/CMakeFiles/Export/lib/cmake/mfem/MFEMTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem/MFEMTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem/MFEMTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/CMakeFiles/Export/lib/cmake/mfem/MFEMTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/mfem" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/CMakeFiles/Export/lib/cmake/mfem/MFEMTargets-release.cmake")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/share/mfem/test.mk")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/share/mfem" TYPE FILE FILES "/Users/ben/Documents/SoftwareLibraries/mfem/config/test.mk")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Development" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/share/mfem/config.mk")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/share/mfem" TYPE FILE RENAME "config.mk" FILES "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/config/config-install.mk")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/general/cmake_install.cmake")
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/linalg/cmake_install.cmake")
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/mesh/cmake_install.cmake")
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/fem/cmake_install.cmake")
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/config/cmake_install.cmake")
  include("/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/doc/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/ben/Documents/SoftwareLibraries/mfem/bens_build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
