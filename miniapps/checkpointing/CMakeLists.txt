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

foreach(example heterogeneous-state mesh-state forward-euler backward-euler)
  add_mfem_miniapp(checkpoint-${example}
    MAIN ${example}.cpp
    LIBRARIES mfem)
endforeach()

if (MFEM_ENABLE_TESTING)
  foreach(execution direct controller controller-window)
    set(execution_options "")
    set(execution_suffix "")
    if(NOT execution STREQUAL "direct")
      set(execution_options --controller)
      set(execution_suffix -${execution})
      if(execution STREQUAL "controller-window")
        list(APPEND execution_options --window-size 2)
      endif()
    endif()
    add_test(NAME checkpoint-heterogeneous-state${execution_suffix}
      COMMAND checkpoint-heterogeneous-state -n 8 ${execution_options})
    add_test(NAME checkpoint-heterogeneous-state-bounded${execution_suffix}
      COMMAND checkpoint-heterogeneous-state -n 8 -m 9 ${execution_options})

    foreach(example forward-euler backward-euler)
      foreach(storage memory-block file-block memory-snapshots file-snapshots)
        add_test(NAME checkpoint-${example}-${storage}${execution_suffix}
          COMMAND checkpoint-${example} -s 8 -r 2 --storage ${storage}
            ${execution_options})
        if(storage MATCHES "snapshots$")
          add_test(
            NAME checkpoint-${example}-${storage}-bounded${execution_suffix}
            COMMAND checkpoint-${example} -s 8 -r 2 --storage ${storage} -m 9
              ${execution_options})
        endif()
      endforeach()
    endforeach()

    foreach(storage memory-snapshots file-snapshots)
      add_test(NAME checkpoint-mesh-state-${storage}${execution_suffix}
        COMMAND checkpoint-mesh-state -r 4 -c 2 --storage ${storage} -no-pv
          ${execution_options})
      add_test(NAME checkpoint-mesh-state-${storage}-bounded${execution_suffix}
        COMMAND checkpoint-mesh-state -r 4 -c 2 --storage ${storage} -m 2 -no-pv
          ${execution_options})
    endforeach()
  endforeach()
endif()
