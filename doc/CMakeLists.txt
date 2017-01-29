# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

find_package(Doxygen)

if (DOXYGEN_FOUND)
  set(MFEM_SOURCE_DIR ${PROJECT_SOURCE_DIR})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CodeDocumentation.conf.in
    ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf @ONLY)

  if (UNIX)
    # Only create symlinks if UNIX operating system
    add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen to CodeDocumentation.html"
      VERBATIM)

    add_custom_target(clean-doc
      COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.html
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation
      COMMENT "Removing API documentation"
      VERBATIM)

  else (UNIX)
    add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation.conf
      BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation/html/index.html
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen to CodeDocumentation/html/index.html"
      VERBATIM)

    add_custom_target(clean-doc
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CodeDocumentation
      COMMENT "Removing API documentation"
      VERBATIM)
  endif (UNIX)
endif (DOXYGEN_FOUND)
