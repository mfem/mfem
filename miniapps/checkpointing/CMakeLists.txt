# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

add_mfem_miniapp(checkpoint-heterogeneous-state
  MAIN heterogeneous-state.cpp
  checkpoint_demo.hpp
  LIBRARIES mfem)

add_mfem_miniapp(checkpoint-mesh-state
  MAIN mesh-state.cpp
  checkpoint_demo.hpp
  LIBRARIES mfem)

add_mfem_miniapp(checkpoint-ode-state
  MAIN ode-state.cpp
  LIBRARIES mfem)

add_mfem_miniapp(checkpoint-backward-euler
  MAIN backward-euler.cpp
  checkpoint_demo.hpp
  LIBRARIES mfem)

add_mfem_miniapp(checkpoint-reverse-adjoint
  MAIN reverse-adjoint.cpp
  checkpoint_demo.hpp
  LIBRARIES mfem)

if (MFEM_ENABLE_TESTING)
  add_test(NAME checkpoint-heterogeneous-state
    COMMAND checkpoint-heterogeneous-state -n 8 -c 3 -w 2)
  add_test(NAME checkpoint-mesh-state
    COMMAND checkpoint-mesh-state -r 3 -c 2 -no-pv)
  add_test(NAME checkpoint-ode-state COMMAND checkpoint-ode-state)
  add_test(NAME checkpoint-backward-euler
    COMMAND checkpoint-backward-euler -s 8 -r 3)
  foreach(solver forward-euler backward-euler)
    foreach(schedule store-all revolve wmi)
      add_test(NAME checkpoint-adjoint-${solver}-${schedule}
        COMMAND checkpoint-reverse-adjoint --solver ${solver}
          --schedule ${schedule} --num-steps 12 --checkpoints 3)
    endforeach()
  endforeach()
endif()
