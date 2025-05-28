# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# If using `CMake >= 3.26` this functionality of this file can be replaced with `copy_directory_if_different`

# set property that will initiate a configure step if data directory is modified
set_property(DIRECTORY APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR})

set(COPIED_DATA_FILES)
file(GLOB DATA_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(DATA_FILE ${DATA_FILES})
  get_filename_component(DATA_FILENAME ${DATA_FILE} NAME)
  set(COPIED_DATA_FILE ${CMAKE_CURRENT_BINARY_DIR}/${DATA_FILENAME})
  add_custom_command(OUTPUT ${COPIED_DATA_FILE} DEPENDS ${DATA_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${DATA_FILE} ${COPIED_DATA_FILE}
    COMMENT "")
  list(APPEND COPIED_DATA_FILES ${COPIED_DATA_FILE})
endforeach()

add_custom_target(copy_data DEPENDS ${COPIED_DATA_FILES}
  COMMENT "Synching the data directory ...")
