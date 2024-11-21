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

set(SRCS
  bilinearform.cpp
  bilinearform_ext.cpp
  bilininteg.cpp
  integ/bilininteg_br2.cpp
  integ/bilininteg_convection_mf.cpp
  integ/bilininteg_convection_pa.cpp
  integ/bilininteg_convection_ea.cpp
  integ/bilininteg_curlcurl_pa.cpp
  integ/bilininteg_dgdiffusion_pa.cpp
  integ/bilininteg_dgtrace_pa.cpp
  integ/bilininteg_dgtrace_ea.cpp
  integ/bilininteg_diffusion_mf.cpp
  integ/bilininteg_diffusion_pa.cpp
  integ/bilininteg_diffusion_ea.cpp
  integ/bilininteg_diffusion_patch.cpp
  integ/bilininteg_divdiv_pa.cpp
  integ/bilininteg_elasticity_ea.cpp
  integ/bilininteg_elasticity_pa.cpp
  integ/bilininteg_gradient_pa.cpp
  integ/bilininteg_interp_pa.cpp
  integ/bilininteg_mass_mf.cpp
  integ/bilininteg_mass_pa.cpp
  integ/bilininteg_mass_ea.cpp
  integ/bilininteg_mixedcurl_pa.cpp
  integ/bilininteg_mixedvecgrad_pa.cpp
  integ/bilininteg_transpose_ea.cpp
  integ/bilininteg_vecdiffusion_mf.cpp
  integ/bilininteg_vecdiffusion_pa.cpp
  integ/bilininteg_vecdiv_pa.cpp
  integ/bilininteg_vecmass_mf.cpp
  integ/bilininteg_vecmass_pa.cpp
  integ/bilininteg_vectorfediv_pa.cpp
  integ/bilininteg_vectorfemass_pa.cpp
  integ/bilininteg_diffusion_kernels.cpp
  integ/bilininteg_elasticity_kernels.cpp
  integ/bilininteg_hcurl_kernels.cpp
  integ/bilininteg_hdiv_kernels.cpp
  integ/bilininteg_hcurlhdiv_kernels.cpp
  integ/bilininteg_mass_kernels.cpp
  integ/lininteg_boundary.cpp
  integ/lininteg_boundary_flux.cpp
  integ/lininteg_domain.cpp
  integ/lininteg_domain_grad.cpp
  integ/lininteg_domain_vectorfe.cpp
  integ/nonlininteg_vecconvection_pa.cpp
  integ/nonlininteg_vecconvection_mf.cpp
  coefficient.cpp
  complex_fem.cpp
  convergence.cpp
  datacollection.cpp
  dgmassinv.cpp
  doftrans.cpp
  eltrans.cpp
  estimators.cpp
  fe.cpp
  fe/face_map_utils.cpp
  fe/fe_base.cpp
  fe/fe_fixed_order.cpp
  fe/fe_h1.cpp
  fe/fe_l2.cpp
  fe/fe_nd.cpp
  fe/fe_nurbs.cpp
  fe/fe_pos.cpp
  fe/fe_rt.cpp
  fe/fe_ser.cpp
  fe_coll.cpp
  fespace.cpp
  geom.cpp
  gridfunc.cpp
  hybridization.cpp
  intrules.cpp
  intrules_cut.cpp
  ceed/interface/basis.cpp
  ceed/interface/restriction.cpp
  ceed/interface/operator.cpp
  ceed/interface/util.cpp
  ceed/integrators/convection/convection.cpp
  ceed/integrators/diffusion/diffusion.cpp
  ceed/integrators/nlconvection/nlconvection.cpp
  ceed/integrators/mass/mass.cpp
  ceed/solvers/algebraic.cpp
  ceed/solvers/full-assembly.cpp
  ceed/solvers/solvers-atpmg.cpp
  kdtree.cpp
  linearform.cpp
  linearform_ext.cpp
  lininteg.cpp
  lor/lor.cpp
  lor/lor_ads.cpp
  lor/lor_ams.cpp
  lor/lor_batched.cpp
  multigrid.cpp
  nonlinearform.cpp
  nonlinearform_ext.cpp
  nonlininteg.cpp
  fespacehierarchy.cpp
  qfunction.cpp
  qinterp/det.cpp
  qinterp/eval_by_nodes.cpp
  qinterp/eval_by_vdim.cpp
  qinterp/grad_by_nodes.cpp
  qinterp/grad_by_vdim.cpp
  qspace.cpp
  quadinterpolator.cpp
  quadinterpolator_face.cpp
  restriction.cpp
  normal_deriv_restriction.cpp
  staticcond.cpp
  tmop.cpp
  tmop/tmop_pa.cpp
  tmop/tmop_pa_da3.cpp
  tmop/tmop_pa_h2d.cpp
  tmop/tmop_pa_h2d_c0.cpp
  tmop/tmop_pa_h2m.cpp
  tmop/tmop_pa_h2m_c0.cpp
  tmop/tmop_pa_h2s.cpp
  tmop/tmop_pa_h2s_c0.cpp
  tmop/tmop_pa_h3d.cpp
  tmop/tmop_pa_h3d_c0.cpp
  tmop/tmop_pa_h3m.cpp
  tmop/tmop_pa_h3m_c0.cpp
  tmop/tmop_pa_h3s.cpp
  tmop/tmop_pa_h3s_c0.cpp
  tmop/tmop_pa_jp2.cpp
  tmop/tmop_pa_jp3.cpp
  tmop/tmop_pa_p2.cpp
  tmop/tmop_pa_p2_c0.cpp
  tmop/tmop_pa_p3.cpp
  tmop/tmop_pa_p3_c0.cpp
  tmop/tmop_pa_tc2.cpp
  tmop/tmop_pa_tc3.cpp
  tmop/tmop_pa_w2.cpp
  tmop/tmop_pa_w2_c0.cpp
  tmop/tmop_pa_w3.cpp
  tmop/tmop_pa_w3_c0.cpp
  tmop_tools.cpp
  tmop_amr.cpp
  gslib.cpp
  transfer.cpp
  hyperbolic.cpp
  )

set(HDRS
  bilinearform.hpp
  bilinearform_ext.hpp
  bilininteg.hpp
  integ/bilininteg_diffusion_kernels.hpp
  integ/bilininteg_elasticity_kernels.hpp
  integ/bilininteg_hcurl_kernels.hpp
  integ/bilininteg_hdiv_kernels.hpp
  integ/bilininteg_hcurlhdiv_kernels.hpp
  integ/bilininteg_mass_kernels.hpp
  coefficient.hpp
  complex_fem.hpp
  convergence.hpp
  datacollection.hpp
  dgmassinv.hpp
  dgmassinv_kernels.hpp
  doftrans.hpp
  eltrans.hpp
  estimators.hpp
  fe.hpp
  fe/face_map_utils.hpp
  fe/fe_base.hpp
  fe/fe_fixed_order.hpp
  fe/fe_h1.hpp
  fe/fe_l2.hpp
  fe/fe_nd.hpp
  fe/fe_nurbs.hpp
  fe/fe_pos.hpp
  fe/fe_rt.hpp
  fe/fe_ser.hpp
  fe_coll.hpp
  fem.hpp
  fespace.hpp
  geom.hpp
  gridfunc.hpp
  hybridization.hpp
  intrules.hpp
  intrules_cut.hpp
  kernel_dispatch.hpp
  kernel_reporter.hpp
  kernels.hpp
  ceed/interface/basis.hpp
  ceed/interface/integrator.hpp
  ceed/interface/interface.hpp
  ceed/interface/operator.hpp
  ceed/interface/restriction.hpp
  ceed/interface/util.hpp
  ceed/integrators/convection/convection.hpp
  ceed/integrators/diffusion/diffusion.hpp
  ceed/integrators/mass/mass.hpp
  ceed/integrators/nlconvection/nlconvection.hpp
  ceed/interface/coefficient.hpp
  ceed/solvers/algebraic.hpp
  ceed/solvers/full-assembly.hpp
  ceed/solvers/solvers-atpmg.hpp
  kdtree.hpp
  linearform.hpp
  linearform_ext.hpp
  lininteg.hpp
  lor/lor.hpp
  lor/lor_ads.hpp
  lor/lor_ams.hpp
  lor/lor_batched.hpp
  lor/lor_h1.hpp
  lor/lor_nd.hpp
  lor/lor_rt.hpp
  lor/lor_util.hpp
  multigrid.hpp
  nonlinearform.hpp
  nonlinearform_ext.hpp
  nonlininteg.hpp
  qfunction.hpp
  qinterp/eval.hpp
  qinterp/grad.hpp
  qspace.hpp
  quadinterpolator.hpp
  quadinterpolator_face.hpp
  restriction.hpp
  normal_deriv_restriction.hpp
  fespacehierarchy.hpp
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
  tmop/tmop_pa.hpp
  tmop_tools.hpp
  tmop_amr.hpp
  gslib.hpp
  transfer.hpp
  hyperbolic.hpp
  )

if (MFEM_USE_SIDRE)
  list(APPEND SRCS sidredatacollection.cpp)
  list(APPEND HDRS sidredatacollection.hpp)
endif()

if (MFEM_USE_CONDUIT)
  list(APPEND SRCS conduitdatacollection.cpp)
  list(APPEND HDRS conduitdatacollection.hpp)
endif()

if (MFEM_USE_ADIOS2)
  list(APPEND SRCS adios2datacollection.cpp)
  list(APPEND HDRS adios2datacollection.hpp)
endif()

if (MFEM_USE_FMS)
  list(APPEND SRCS fmsdatacollection.cpp fmsconvert.cpp)
  list(APPEND HDRS fmsdatacollection.hpp fmsconvert.hpp)
endif()

if (MFEM_USE_MPI)
  list(APPEND SRCS
    pbilinearform.cpp
    pfespace.cpp
    pgridfunc.cpp
    plinearform.cpp
    pnonlinearform.cpp
    prestriction.cpp)
  # If this list (HDRS -> HEADERS) is used for install, we probably want the
  # headers added all the time.
  list(APPEND HDRS
    pbilinearform.hpp
    pfespace.hpp
    pgridfunc.hpp
    plinearform.hpp
    pnonlinearform.hpp
    prestriction.hpp)
endif()

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
