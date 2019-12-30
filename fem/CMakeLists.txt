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

set(SRCS
  bilinearform.cpp
  bilinearform_ext.cpp
  bilininteg.cpp
  bilininteg_diffusion.cpp
  bilininteg_mass.cpp
  coefficient.cpp
  complex_fem.cpp
  datacollection.cpp
  eltrans.cpp
  estimators.cpp
  fe.cpp
  fe_coll.cpp
  fespace.cpp
  geom.cpp
  gridfunc.cpp
  hybridization.cpp
  intrules.cpp
  linearform.cpp
  lininteg.cpp
  nonlinearform.cpp
  nonlininteg.cpp
  staticcond.cpp
  tmop.cpp
  tmop_tools.cpp
  gslib.cpp
  )

set(HDRS
  bilinearform.hpp
  bilinearform_ext.hpp
  bilininteg.hpp
  coefficient.hpp
  complex_fem.hpp
  datacollection.hpp
  eltrans.hpp
  estimators.hpp
  fe.hpp
  fe_coll.hpp
  fem.hpp
  fespace.hpp
  geom.hpp
  gridfunc.hpp
  hybridization.hpp
  intrules.hpp
  linearform.hpp
  lininteg.hpp
  nonlinearform.hpp
  nonlininteg.hpp
  staticcond.hpp
  tbilinearform.hpp
  tbilininteg.hpp
  tcoefficient.hpp
  teltrans.hpp
  tevaluator.hpp
  tfe.hpp
  tfespace.hpp
  tintrules.hpp
  tmop.hpp
  tmop_tools.hpp
  gslib.hpp
  )

if (MFEM_USE_SIDRE)
  list(APPEND SRCS sidredatacollection.cpp)
  list(APPEND HDRS sidredatacollection.hpp)
endif()

if (MFEM_USE_CONDUIT)
  list(APPEND SRCS conduitdatacollection.cpp)
  list(APPEND HDRS conduitdatacollection.hpp)
endif()

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pbilinearform.cpp
    pfespace.cpp
    pgridfunc.cpp
    plinearform.cpp
    pnonlinearform.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # headers added all the time.
  list(APPEND HDRS
    pbilinearform.hpp
    pfespace.hpp
    pgridfunc.hpp
    plinearform.hpp
    pnonlinearform.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
