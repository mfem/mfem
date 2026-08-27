# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.

add_mfem_miniapp(frequency_domain_preconditioners_regression
                 MAIN frequency_domain_preconditioners_regression.cpp
                 EXTRA_SOURCES frequency_domain_preconditioners.cpp
                 EXTRA_HEADERS frequency_domain_preconditioners.hpp
                 LIBRARIES mfem)

if(MFEM_USE_MPI AND MFEM_USE_DOUBLE)
   add_mfem_miniapp(frequency_domain_elasticity_test
                    MAIN frequency_domain_elasticity_test.cpp
                    EXTRA_SOURCES frequency_domain_elasticity.cpp
                    EXTRA_HEADERS frequency_domain_elasticity.hpp
                    LIBRARIES mfem)

   add_mfem_miniapp(frequency_domain_elasticity_solver_regression
                    MAIN frequency_domain_elasticity_solver_regression.cpp
                    EXTRA_SOURCES frequency_domain_elasticity_solver.cpp
                                  frequency_domain_elasticity.cpp
                                  frequency_domain_preconditioners.cpp
                    EXTRA_HEADERS frequency_domain_elasticity_solver.hpp
                                  frequency_domain_elasticity.hpp
                                  frequency_domain_preconditioners.hpp
                    LIBRARIES mfem)

   add_mfem_miniapp(frequency_domain_cantilever
                    MAIN frequency_domain_cantilever.cpp
                    EXTRA_SOURCES frequency_domain_elasticity_solver.cpp
                                  frequency_domain_elasticity.cpp
                                  frequency_domain_preconditioners.cpp
                    EXTRA_HEADERS frequency_domain_elasticity_solver.hpp
                                  frequency_domain_elasticity.hpp
                                  frequency_domain_driver_utils.hpp
                                  frequency_domain_preconditioners.hpp
                    LIBRARIES mfem)

   add_mfem_miniapp(frequency_domain_cantilever_mms_regression
                    MAIN frequency_domain_cantilever_mms_regression.cpp
                    EXTRA_SOURCES frequency_domain_elasticity_solver.cpp
                                  frequency_domain_elasticity.cpp
                                  frequency_domain_preconditioners.cpp
                    EXTRA_HEADERS frequency_domain_elasticity_solver.hpp
                                  frequency_domain_elasticity.hpp
                                  frequency_domain_driver_utils.hpp
                                  frequency_domain_preconditioners.hpp
                    LIBRARIES mfem)
endif()
