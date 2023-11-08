// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// ------------------------------------------------------------------------------------
// This miniapp presents a space-time adaptive Monodomain solver with local time 
// stepping using a symmetric interior penalty formulation.
//
// We solve the system
//     ∂ₜφ = div(D grad(φ)) + f(φ, s) + I_stim(x,t)  on Ω  × [0,T]
//     ∂ₜs =                  g(φ, s)                on Ω  × [0,T] 
//     0   = D grad(φ)                               on ∂Ω × [0,T]
//
// A detailed description of the discretization is given in
// https://arxiv.org/pdf/2310.07607.pdf .
//
// Example Runs
//   spinner-lts --fibrillation-init -m ./inline-test-quad.mesh -k1 0.1 -k2 0.1 -kappa 8.0 -o 2 -e 1.0 -tf 1000.0
//   spinner-lts --stim-t-max 2.0 --stim-d-max 1.0 --stim-s-max 100.0 -m ./inline-niederer-wedge.mesh -tf 50.0
// ------------------------------------------------------------------------------------

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

template<typename T>
T std_clamp(T v, T low, T up) { // Poort mans std::clamp<T>
    return std::min<T>(up, std::max<T>(low, v));
}

// Turn these off if you want a more accurrate overall timing!
#define MFEM_PERF_FINE_BEGIN(x) //MFEM_PERF_BEGIN(x)
#define MFEM_PERF_FINE_END(x) //MFEM_PERF_END(x)
#define MFEM_PERF_FINE_FUNCTION //MFEM_PERF_FUNCTION
#define MFEM_LOOP_BEGIN(x,y)
#define MFEM_LOOP_ITERATION(x,y)
#define MFEM_LOOP_END(x)

// Print some debug information
#define DEBUG_PRINT(x) //std::cout << x << std::endl;

using namespace mfem;
using namespace std;
#include "cell-solver.hpp"
#include "utils.hpp"
#include "new-kelly-estimator.hpp"

//    LTS Face Action
// This function evaluates the action of a face with linear in time interpolated 
// neighbor values
inline void EvaluateFace(const int ei, const int fi, const int fe1,
                         const int fe2, const int substep,
                         const Array<int> &num_substeps_per_element,
                         const int num_substeps, EvalCache &ec,
                         const std::vector<DenseMatrix> &face_matrices,
                         const FiniteElementSpace *phimfespace,
                         const FiniteElementSpace *sfespace, Mesh *mesh,
                         const Vector &phim_prev, const Vector &phim_predictor) {
  MFEM_PERF_FINE_FUNCTION;

  const int substepgridwidth_1 = num_substeps / num_substeps_per_element[fe1];
  const int substepgridwidth_2 = num_substeps / num_substeps_per_element[fe2];
  const auto substepoffset_1 = std::div(substep, substepgridwidth_1);
  const auto substepoffset_2 = std::div(substep, substepgridwidth_2);

  // Substep both elements at the same time, because the temporal grids align
  // on the current substep. This basically means we do not need to interpolate
  // in time.
  //
  //       tnow      t2
  //        +─────────+
  //        │    |    │
  //    fe1 x---> --->x
  //        │    |    │
  //     fi +─────────+
  //        │         │
  //    fe2 x------->x
  //        │         │
  //        x─────────x
  //
  // where the element to advande "ei" is either fe1 or fe2.
  // In all other cases we have to substep the element "ei" are if either
  //
  //       t1  tnow t2
  //        +─────────+
  //        │    │    │
  // ei=fe1 x-------->x
  //        │    │    │
  //     fi +─────────+
  //        │    │    │
  //    fe2 x---> --->x
  //        │    │    │
  //        x─────────x
  //
  //   or
  //
  //       t1  tnow t2
  //        +─────────+
  //        │    │    │
  //    fe1 x---> --->x
  //        │    │    │
  //     fi +─────────+
  //        │    │    │
  // ei=fe2 x---> --->x
  //        │    │    │
  //        x─────────x
  //
  // where it is a precondition that "ei" is the "smaller-in-time" element.
  if (substepoffset_1.rem == 0 && substepoffset_2.rem == 0) {
    // Get dof indices
    phimfespace->GetElementDofs(fe1, ec.vdofs1);
    phimfespace->GetElementDofs(fe2, ec.vdofs2);
    // Unrolled local action
    const auto &face_mat = face_matrices[fi];
    const auto ndofs1 = ec.facedofblocks[1];
    const auto ndofs2 = ec.facedofblocks[2] - ec.facedofblocks[1];
    if (ei == fe1) {
      for (int i = 0; i < ndofs1; i++) {
        for (int j = 0; j < ndofs1; j++) {
          ec.phimelvalsfacerhs(i) += face_mat(i, j) * phim_prev(ec.vdofs1[j]);
        }
        for (int j = 0; j < ndofs2; j++) {
          ec.phimelvalsfacerhs(i) +=
              face_mat(i, j + ndofs1) * phim_prev(ec.vdofs2[j]);
        }
      }
    } else if (ei == fe2) {
      for (int i = 0; i < ndofs2; i++) {
        for (int j = 0; j < ndofs1; j++) {
          ec.phimelvalsfacerhs(i) +=
              face_mat(i + ndofs1, j) * phim_prev(ec.vdofs1[j]);
        }
        for (int j = 0; j < ndofs2; j++) {
          ec.phimelvalsfacerhs(i) +=
              face_mat(i + ndofs1, j + ndofs1) * phim_prev(ec.vdofs2[j]);
        }
      }
    } else {
      std::cout << "Face-element table corrupted? (Case 1) ei=" << ei
                << " fi=" << fi << std::endl;
      std::exit(-1);
    }
  } else if (ei == fe1) {
    // Get dof indices
    phimfespace->GetElementDofs(fe1, ec.vdofs1);
    phimfespace->GetElementDofs(fe2, ec.vdofs2);

    // Unrolled local action
    const auto &face_mat = face_matrices[fi];
    const auto ndofs1 = ec.facedofblocks[1];
    const auto ndofs2 = ec.facedofblocks[2] - ec.facedofblocks[1];

    const int substep_difference =
        num_substeps_per_element[fe1] / num_substeps_per_element[fe2];
    const auto substep_offset = std::div(substep, substep_difference);

    // Interpolation weights
    const auto ssw2 = (double(substep_offset.rem)) / substep_difference;
    const auto ssw1 = 1.0 - ssw2;

    for (int i = 0; i < ndofs1; i++) {
      for (int j = 0; j < ndofs1; j++) {
        // double interpolation_in_time =
        // ssw1*phim_prev(ec.vdofs1[j])+ssw2*phim_predictor(ec.vdofs1[j]);
        double interpolation_in_time = phim_prev(ec.vdofs1[j]);
        ec.phimelvalsfacerhs(i) += face_mat(i, j) * interpolation_in_time;
      }
      for (int j = 0; j < ndofs2; j++) {
        double interpolation_in_time =
            ssw1 * phim_prev(ec.vdofs2[j]) + ssw2 * phim_predictor(ec.vdofs2[j]);
        ec.phimelvalsfacerhs(i) +=
            face_mat(i, j + ndofs1) * interpolation_in_time;
      }
    }
  } else if (ei == fe2) { // Substep second element only using interpolated
                          // values of the first
    // Get dof indices
    phimfespace->GetElementDofs(fe1, ec.vdofs1);
    phimfespace->GetElementDofs(fe2, ec.vdofs2);

    // Unrolled local action
    const auto &face_mat = face_matrices[fi];
    const auto ndofs1 = ec.facedofblocks[1];
    const auto ndofs2 = ec.facedofblocks[2] - ec.facedofblocks[1];

    const int substep_difference =
        num_substeps_per_element[fe2] / num_substeps_per_element[fe1];
    const auto substep_offset = std::div(substep, substep_difference);

    // Interpolation weights
    const auto ssw2 = (double(substep_offset.rem)) / substep_difference;
    const auto ssw1 = 1.0 - ssw2;

    for (int i = 0; i < ndofs1; i++) {
      for (int j = 0; j < ndofs1; j++) {
        double interpolation_in_time =
            ssw1 * phim_prev(ec.vdofs1[j]) + ssw2 * phim_predictor(ec.vdofs1[j]);
        ec.phimelvalsfacerhs(i) +=
            face_mat(i + ndofs1, j) * interpolation_in_time;
      }
      for (int j = 0; j < ndofs2; j++) {
        // double interpolation_in_time =
        // ssw1*phim_prev(ec.vdofs2[j])+ssw2*phim_predictor(ec.vdofs2[j]);
        double interpolation_in_time = phim_prev(ec.vdofs2[j]);
        ec.phimelvalsfacerhs(i) +=
            face_mat(i + ndofs1, j + ndofs1) * interpolation_in_time;
      }
    }
  } else {
    std::cout << "Face-element table corrupted? (Case 2) ei=" << ei
              << " fi=" << fi << std::endl;
    std::exit(-1);
  }
}

//    LTS Element Action
// Advance the solution with the available predictor
// and existing solution information via explicit Euler step. Also takes care
// of the faces (also the non-conforming case).
void AdvanceElement(const int ei, const int substep,
                    const Array<int> &num_substeps_per_element,
                    const int total_substeps, const GridFunction &phim_prev,
                    GridFunction &phim_next, GridFunction &phim_predictor,
                    const GridFunction &s_prev, GridFunction &s_next,
                    EvalCache &ec, BilinearForm &a, BilinearForm &minv,
                    std::shared_ptr<const AbstractCellSolver> stepper,
                    const IntegrationRule &nodal_ir,
                    const std::vector<DenseMatrix> &face_matrices,
                    const double t, const double Δt_barrier,
                    const Stimulus &stim) {

  auto phimfespace = phim_prev.FESpace();
  auto sfespace = s_prev.FESpace();
  auto mesh = phimfespace->GetMesh();

  // const int substepgridwidth = total_substeps / num_substeps_per_element[ei];
  const double Δt_step = Δt_barrier / num_substeps_per_element[ei];
  const double t_now = t + Δt_barrier * substep / total_substeps;

  DEBUG_PRINT("    Δt_step=" << Δt_step << " t_now=" << t_now
                             << " depth=" << depth);

  MFEM_PERF_FINE_BEGIN("Evaluate Faces");

  MFEM_PERF_FINE_BEGIN("Grab Faces");
  GetFaceIndices(ec.e_faces, ec.f_ori, mesh, ei);
  MFEM_PERF_FINE_END("Grab Faces");

  // TODO replace vdofs with ec.phimdofs
  // MFEM_PERF_BEGIN("Zero out rhs");
  ec.phimelvalsfacerhs = 0.0;
  // MFEM_PERF_END("Zero out rhs");

  MFEM_PERF_FINE_BEGIN("NCList");
  auto &ncfacelist = GetNCFaceList(mesh);
  MFEM_PERF_FINE_END("NCList");

  MFEM_PERF_FINE_BEGIN("Face Loop");
  for (int fi : ec.e_faces) {
    int fe1, fe2;
    mesh->GetFaceElements(fi, &fe1, &fe2);
    DEBUG_PRINT("    Visiting face " << fi << " with elements " << fe1 << " "
                                     << fe2);
    if (fe2 >= 0) { // Not on boundary
      MFEM_PERF_FINE_BEGIN("EvaluateFaceCall");
      EvaluateFace(ei, fi, fe1, fe2, substep, num_substeps_per_element,
                   total_substeps, ec, face_matrices, phimfespace, sfespace, mesh,
                   phim_prev, phim_predictor);
      MFEM_PERF_FINE_END("EvaluateFaceCall");
    } else {
      int Inf1, Inf2, NCFace;
      mesh->GetFaceInfos(fi, &Inf1, &Inf2, &NCFace);
      if (NCFace < 0) {
        // EvaluateBdrFace(ei, fi, ec, face_matrices, phimfespace, phim_prev);
        continue; // Not a master face (i.e. a boundary face)
      }
      // MFEM_PERF_FINE_BEGIN("NCInfo");
      DEBUG_PRINT("      NCFace=" << NCFace);
      auto &masterinfo = ncfacelist.masters[NCFace];
      DEBUG_PRINT("      master index=" << masterinfo.index
                                        << " local=" << int(masterinfo.local));
      DEBUG_PRINT("      slave range=" << masterinfo.slaves_begin << ":"
                                       << masterinfo.slaves_end);
      // MFEM_PERF_FINE_END("NCInfo");
      for (int slave = masterinfo.slaves_begin; slave < masterinfo.slaves_end;
           slave++) {
        auto &slaveinfo = ncfacelist.slaves[slave];
        DEBUG_PRINT("        slave index=" << slaveinfo.index << " local="
                                           << int(slaveinfo.local));
        if (slaveinfo.index < 0) { // Degenerate face-edge constraint
          continue;
        }
        mesh->GetFaceElements(slaveinfo.index, &fe1, &fe2);
        MFEM_PERF_FINE_BEGIN("EvaluateFaceCall2");
        EvaluateFace(ei, slaveinfo.index, fe1, fe2, substep,
                     num_substeps_per_element, total_substeps, ec,
                     face_matrices, phimfespace, sfespace, mesh, phim_prev,
                     phim_predictor);
        MFEM_PERF_FINE_END("EvaluateFaceCall2");
      }
    }
  }
  MFEM_PERF_FINE_END("Face Loop");
  MFEM_PERF_FINE_END("Evaluate Faces");

  phimfespace->GetElementDofs(ei, ec.phimdofs);
  phim_prev.GetSubVector(ec.phimdofs, ec.phimelvals);
  GetElementVDofsLinear(*sfespace, ei, ec.sdofs);
  s_prev.GetSubVector(ec.sdofs, ec.selvals);

  // Buffer local diffusion action of (Ae + Af) phimgf^{t_n}
  // Element diffusion portion
  // MFEM_PERF_FINE_BEGIN("Element Diffusion Action");
  a.ComputeElementMatrix(ei, ec.a_el);
  ec.a_el.Mult(ec.phimelvals, ec.phimelvals_rhs);
  // Face diffusion portion - precomputed in loop above
  ec.phimelvals_rhs += ec.phimelvalsfacerhs;
  // MFEM_PERF_FINE_END("Element Diffusion Action");

  // Apply I_stim
  MFEM_PERF_FINE_BEGIN("Evaluate Stimulus");
  auto geotrafo = mesh->GetElementTransformation(ei);
  if (t_now < stim.t_max) {
    Vector pos(mesh->SpaceDimension());
    for (int node = 0; node < nodal_ir.GetNPoints(); node++) {
      auto ip = nodal_ir.IntPoint(node);
      geotrafo->SetIntPoint(&ip);
      geotrafo->Transform(ip, pos);
      auto detJdV = geotrafo->Weight() * ip.weight;

      const auto dist = pos.Norml2();
      if (dist < stim.dist_max) {
        ec.phimelvals_rhs(node) -=
            stim.stim_max *
            std::max((1.0 - dist / stim.dist_max) * (stim.t_max - t_now), 0.0) *
            detJdV / stim.capacitance; // Test function is 1 at the nodes
      }
    }
  }
  MFEM_PERF_FINE_END("Evaluate Stimulus");

  // Add local reactions
  MFEM_PERF_FINE_BEGIN("Evaluate Reactions");
  for (int node = 0; node < nodal_ir.GetNPoints(); node++) {
    const IntegrationPoint &ip = nodal_ir.IntPoint(node);
    geotrafo->SetIntPoint(&ip);
    auto detJdV = geotrafo->Weight() * ip.weight;
    ec.phimelvals_rhs(node) +=
        stepper->EvalReaction(ec.phimelvals(node), ec.selvals, node, t_now) *
        detJdV / stim.capacitance; // Test function is 1 at the nodes
    // We are done at this point with the data at the quadrature point, so
    // we update them inplace.
    stepper->InternalEulerUpdate(ec.phimelvals(node), ec.selvals, node, t_now,
                                 Δt_step);
  }
  MFEM_PERF_FINE_END("Evaluate Reactions");

  // Adjust local diffusion action with mass matrix inverse
  // MFEM_PERF_FINE_BEGIN("Element Mass Matrix Action");
  minv.ComputeElementMatrix(ei, ec.minv_el);
  ec.minv_el.Mult(ec.phimelvals_rhs, ec.phimelvals_rhs2);
  ec.phimelvals_rhs2 *= Δt_step;
  // MFEM_PERF_FINE_END("Element Mass Matrix Action");

  // Apply corrected local diffusion action
  ec.phimelvals -= ec.phimelvals_rhs2;

  // Store solution
  phim_next.SetSubVector(ec.phimdofs, ec.phimelvals);
  s_next.SetSubVector(ec.sdofs, ec.selvals);
}

// Estimate time step lengths for each element via Gershgorin discs
void ComputeMaxTimeStepLenghtsViaGershgorin(
    mfem::GridFunction &critical_time_step_lengths, mfem::BilinearForm &minv,
    mfem::BilinearForm &a, std::vector<DenseMatrix> &face_matrices,
    std::shared_ptr<const AbstractCellSolver> stepper, EvalCache &ec) {
  DenseMatrix element_jac(ec.facedofblocks[1]);
  DenseMatrix element_jac_massinv(ec.facedofblocks[1]);
  Vector eigenvalues(ec.facedofblocks[1]);
  auto mesh = critical_time_step_lengths.FESpace()->GetMesh();

  MFEM_PERF_FINE_BEGIN("NCList");
  auto &ncfacelist = GetNCFaceList(mesh);
  MFEM_PERF_FINE_END("NCList");

  MFEM_PERF_FINE_BEGIN("Face Loop");
  for (int ei = 0; ei < mesh->GetNE(); ei++) {
    ComputeElementJacobian(element_jac, element_jac_massinv, mesh, ei,
                           ncfacelist, minv, a, face_matrices, stepper, ec);
    // Gershgorin disc theorem
    EigenvaluesGershgorin(element_jac_massinv, eigenvalues);
    critical_time_step_lengths(ei) = 1.0 / eigenvalues.Max();
  }
}

// ----------------------------------- main entry point ---------------------------------------
int main(int argc, char *argv[]) {
#ifdef MFEM_USE_CALIPER
  cali::ConfigManager mgr;
  mgr.start();
#endif

  PrintBanner(std::cout);

  // 2. Parse command-line options.
  const char *mesh_file = "../../data/large-inline-quad.mesh";
  const char *output_name = "RD-LTS-AMR";
  int order = 1;
  double t_final = 20.0;
  double max_elem_error = 0.1;
  double hysteresis = 0.33; // derefinement safety coefficient
  int ref_levels = 0;
  int nc_limit = 1; // maximum level of hanging nodes
  bool visualization = true;
  int which_spatial_estimator = 2;
  double Δt_barrier = 0.1;
  bool use_substepping = false;
  double substep_threshold = 0.1;
  bool use_amr = true;
  // const char* cali_config = "runtime-report";
  double Δtvis = 0.2;

  bool paraview = false;

  // We are using a heuristic to estimate the time step length, hence a safety factor.
  double CFL_factor = 0.9;

  double sigma = -1.0;
  double kappa = (order + 1) * (order + 1);

  bool fib_init = false;

  int derefinement_skips = 1;

  double χ = 1.0;
  double capacitance = 1.0;
  double κ₁ = 0.17 * 0.62 / (0.17 + 0.62);
  double κ₂ = 0.019 * 0.24 / (0.019 + 0.24);
  double κ₃ = 0.019 * 0.24 / (0.019 + 0.24);
  int dtf_type = 0;

  Stimulus stim;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&Δt_barrier, "-dt", "--time-step-length",
                 "Which time step length to use.");
  args.AddOption(&Δtvis, "-dtv", "--vis-time-step-length",
                 "Which time step length should the app visualize.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&sigma, "-sigma", "--sigma", "Sigma pentalty value.");
  args.AddOption(&kappa, "-kappa", "--kappa", "Kappa penalty value.");
  args.AddOption(&max_elem_error, "-e", "--max-err", "Maximum element error");
  args.AddOption(&hysteresis, "-y", "--hysteresis",
                 "Derefinement safety coefficient.");
  args.AddOption(&use_substepping, "", "--substepping", "", "--no-substepping",
                 "Substep based on Ratti-Veroni temporal estimate.");
  args.AddOption(&substep_threshold, "", "--substepping-threshold",
                 "Substep cutoff for Ratti-Veroni temporal estimate.");
  args.AddOption(&ref_levels, "-r", "--ref-levels",
                 "Number of initial uniform refinement levels.");
  args.AddOption(&nc_limit, "-l", "--nc-limit",
                 "Maximum level of hanging nodes.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&which_spatial_estimator, "-est", "--estimator",
                 "Which estimator to use: "
                 "0 = Kelly. Defaults to Kelly.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&paraview, "-pv", "--paraview-datafiles", "-no-pv",
                 "--no-paraview-datafiles",
                 "Save data files for ParaView visualization.");
  args.AddOption(&output_name, "-on", "--output-name",
                 "Folder name of the output.");
  args.AddOption(&use_amr, "-amr", "--adaptive", "-no-amr", "--non-adaptive",
                 "Adapt mesh to solution or not.");

  args.AddOption(&CFL_factor, "", "--cfl-factor",
                 "Safety scaling factor for the CFL condition.");

  args.AddOption(&κ₁, "-k1", "--kappa1", "Fiber conductivity.");
  args.AddOption(&κ₂, "-k2", "--kappa2", "Sheet conductivity.");
  args.AddOption(&κ₃, "-k3", "--kappa3", "Normal conductivity.");
  args.AddOption(&dtf_type, "-dtft", "--diffusion-tensor-field-type",
                 "Which kind of analytical diffusion tensor field to use.");
  args.AddOption(&χ, "-chi", "--chi", "χ");
  args.AddOption(&capacitance, "-cm", "--cm", "capacitance");

  args.AddOption(&stim.t_max, "", "--stim-t-max",
                 "How long to apply the stimulus?");
  args.AddOption(&stim.dist_max, "", "--stim-d-max",
                 "How large is the stimulus?");
  args.AddOption(&stim.stim_max, "", "--stim-s-max",
                 "How strong is the stimulus?");

  args.AddOption(&fib_init, "", "--fibrillation-init", "", "--steady-init",
                 "Fibrillation or steady state?");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);
  stim.capacitance = capacitance; // :)

  // MFEM_PERF_BEGIN("Setup");

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
  //    NURBS meshes are projected to second order meshes.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  const int dim = mesh->Dimension();
  const int sdim = mesh->SpaceDimension();

  // 3. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. By default, or if ref_levels < 0,
  //    we choose it to be the largest number that gives a final mesh with no
  //    more than 50,000 elements.
  for (int l = 0; l < ref_levels; l++) {
    mesh->UniformRefinement();
  }
  if (use_amr) {
    mesh->EnsureNCMesh(true);
  }
  mesh->Finalize(true, true);
  mesh->EnsureNodes();

  // 4. Define a finite element space on the mesh. Here we use discontinuous
  //    finite elements of the specified order >= 0.
  auto stepper = std::make_shared<PathmanathanCordeiroGrayCellSolver>(capacitance);
  L2_FECollection fec(order, dim);
  L2_FECollection fect_est(0, dim);
  FiniteElementSpace phimfespace(mesh, &fec);
  FiniteElementSpace sfespace(mesh, &fec, stepper->InternalDim(),
                              mfem::Ordering::byVDIM);
  FiniteElementSpace t_estfespace(mesh, &fect_est);
  std::vector<FiniteElementSpace *> fespaces;
  fespaces.push_back(&phimfespace);
  fespaces.push_back(&sfespace);
  fespaces.push_back(&t_estfespace);
  FiniteElementSpace fluxfespace(mesh, &fec, sdim);

  cout << "Number of unknowns: " << phimfespace.GetVSize() << "+"
       << sfespace.GetVSize() << endl;

  GridFunction phimgf(&phimfespace); // phim at current time step
  GridFunction sgf(&sfespace); // s at current time step
  GridFunction t_estgf(&t_estfespace); // Temporal error estimate
  t_estgf = 0.0;
  GridFunction s_estgf(&t_estfespace); // Spatial error estimate
  s_estgf = 0.0;
  GridFunction substeps_gf(&t_estfespace); // Helper to visualize the number of taken substeps in paraview
  substeps_gf = -1.0;
  GridFunction critical_time_step_lengths(&t_estfespace); // Helper to see what has been estimated for the local time step length
  critical_time_step_lengths = 0.0;
  stepper->Init(phimgf, sgf); // Default initial condition
  if (fib_init) // Trick taken from "Data-Driven Uncertainty Quantification for
                // Cardiac Electrophysiological Models: Impact of Physiological
                // Variability on Action Potential and Spiral Wave Dynamics"
  {
    FibrillationInit<GridFunction>(phimgf, sgf, stepper->HGateIndex());
    stepper->RescaleVoltage(phimgf);
  }
  // Predictor buffers
  GridFunction phim_predictor(&phimfespace);
  GridFunction s_predictor(&sfespace);

  // Helper to sync grid functions during AMR
  std::vector<GridFunction *> gfs;
  gfs.push_back(&phimgf); 
  gfs.push_back(&sgf); 
  gfs.push_back(&t_estgf);
  gfs.push_back(&s_estgf);
  gfs.push_back(&substeps_gf);
  gfs.push_back(&critical_time_step_lengths);
  gfs.push_back(&phim_predictor);
  gfs.push_back(&s_predictor);

  // Storage for the current time of any element
  vector<double> element_times(mesh->GetNE());
  fill(element_times.begin(), element_times.end(), 0.0);

  // Notify GLVis :)
  char vishost[] = "localhost";
  int visport = 19916;
  socketstream sol_sock(vishost, visport);
  sol_sock.precision(8);
  sol_sock << "solution\n"
           << *mesh << phimgf << std::endl
           << "window_title 'Voltage'" << std::endl
           << "valuerange -90.0 50.0\n"
           << "autoscale off\n"
           << "plot_caption 't= " << 0.0 << "ms'" << std::flush;
  sol_sock << "pause\n";

  // 
  EvalCache ec(phimfespace.GetFE(0)->GetDof(), stepper->InternalDim(),
               mesh->GetElement(0)->GetNFaces());

  // Create form for diffusion
  mfem::DenseMatrix D(sdim);
  D = 0.0;
  D(0, 0) = κ₁ / (χ * capacitance);
  if (sdim > 1)
    D(1, 1) = κ₂ / (χ * capacitance);
  if (sdim > 2)
    D(2, 2) = κ₃ / (χ * capacitance);
  auto Dc = std::make_shared<mfem::MatrixConstantCoefficient>(D);
  auto a = new BilinearForm(&phimfespace);
  auto interior_integ = new DiffusionIntegrator(*Dc);
  a->AddDomainIntegrator(interior_integ);
  auto dginteg = new DGDiffusionIntegrator(*Dc, sigma, kappa);

  mfem::IntegrationRules GLIntRules =
      mfem::IntegrationRules(0, mfem::Quadrature1D::GaussLegendre);
  auto nodal_ir = GLIntRules.Get(mesh->GetElementGeometry(0), 2 * order);
  if (phimfespace.GetFE(0)->GetNodes().GetNPoints() != nodal_ir.GetNPoints()) {
    std::cout << "Quadrature points do not add up!" << std::endl;
    return -1;
  }
  if (mesh->HasGeometry(Geometry::TETRAHEDRON)) {
    std::cout << "Tet not supported." << std::endl;
    return -1;
  }

  auto minv = new BilinearForm(&phimfespace);
  minv->AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));

  std::shared_ptr<ParaViewDataCollection> dc =
      paraview ? std::make_shared<ParaViewDataCollection>(output_name, mesh)
               : nullptr;
  int paraview_cycle = 0;
  if (paraview) {
    dc->SetLevelsOfDetail(order);
    dc->RegisterField("transmembrane potential", &phimgf);
    dc->RegisterField("internal states", &sgf);
    dc->RegisterField("temporal estimator", &t_estgf);
    dc->RegisterField("spatial estimator", &s_estgf);
    dc->RegisterField("log2(num_substeps)", &substeps_gf);
    dc->RegisterField("Δtmax_estimate", &critical_time_step_lengths);
  }

  // Error estimation
  auto spatial_estimator =
      [&](int which_spatial_estimator) -> ErrorEstimator * {
    if (which_spatial_estimator == 0) {
      auto estimator = new CustomKellyErrorEstimator<DGFluxKernel>(
          DGFluxKernel(&fluxfespace), *interior_integ, phimgf, fluxfespace, true);
      return estimator;
    } else if (which_spatial_estimator == 1) {
      auto estimator = new CustomKellyErrorEstimator<DGFluxKernel>(
          DGFluxKernel(&fluxfespace), *interior_integ, phimgf, fluxfespace,
          false);
      return estimator;
    } else if (which_spatial_estimator == 2) {
      return new CustomKellyErrorEstimator<DGWeightedFluxKernel>(
          DGWeightedFluxKernel(&fluxfespace), *interior_integ, phimgf,
          fluxfespace, true);
    } else if (which_spatial_estimator == 3) {
      return new CustomKellyErrorEstimator<DGWeightedFluxKernel>(
          DGWeightedFluxKernel(&fluxfespace), *interior_integ, phimgf,
          fluxfespace, false);
    } else {
      std::cout << "Unknown error estimator. Aborting." << std::endl;
      std::exit(-1);
    }
  }(which_spatial_estimator);

  ThresholdRefiner refiner(*spatial_estimator);
  refiner.SetTotalErrorFraction(0.0);
  refiner.SetLocalErrorGoal(max_elem_error);
  refiner.SetNCLimit(nc_limit);

  ThresholdDerefiner derefiner(*spatial_estimator);
  derefiner.SetThreshold(hysteresis * max_elem_error);
  derefiner.SetNCLimit(nc_limit);

  // Queues
  std::vector<std::pair<std::vector<int>, std::vector<int>>>
      predictor_corrector_queues(1);
  std::vector<int> lowest_level_elements;

  // Reset counter
  { std::ofstream of(std::string(output_name) + "/element-counter.txt"); }

  int tvis_nsteps = std::max(int(ceil(Δtvis / Δt_barrier)), 1);
  int timestep_index = -1;
  for (auto last_timebarrier = 0.0; last_timebarrier < t_final;
       last_timebarrier += Δt_barrier) {
    timestep_index++;

    if (visualization && (timestep_index % tvis_nsteps == 0)) {
      MFEM_PERF_BEGIN("GLVis");
      sol_sock << "solution\n"
               << *mesh << phimgf << std::endl
               << "window_title 'Voltage'" << std::endl
               << "plot_caption 't= " << last_timebarrier << "ms'"
               << std::flush;
      MFEM_PERF_END("GLVis");
    }

    if (paraview && (timestep_index % tvis_nsteps == 0)) {
      if (use_amr) {
        auto local_errors = spatial_estimator->GetLocalErrors();
        for (int i = 0; i < local_errors.Size(); i++)
          s_estgf(i) = local_errors(i);
      }
      MFEM_PERF_BEGIN("Paraview");
      dc->SetCycle(paraview_cycle++);
      dc->SetTime(last_timebarrier);
      dc->Save();
      MFEM_PERF_END("Paraview");
    }

    const double next_timebarrier =
        std::min(last_timebarrier + Δt_barrier, t_final);
    Δt_barrier = next_timebarrier - last_timebarrier;

    cout << "CURRENT CHECKPOINT " << last_timebarrier
         << " ms with Δt_barrier=" << Δt_barrier << endl;

    // AMR
    if (use_amr) {
      MFEM_PERF_BEGIN("AMR");
      // Refine
      MFEM_PERF_BEGIN("Refine");
      refiner.Reset();
      if (refiner.Apply(*mesh)) {
        Update(fespaces, gfs, *a, *minv);
        cout << "  Refinement! Estimated total error: "
             << spatial_estimator->GetTotalError() << endl;
      }
      MFEM_PERF_END("Refine");

      if (timestep_index % derefinement_skips == 0) {
        MFEM_PERF_BEGIN("Coarsen");
        derefiner.Reset();
        if (derefiner.Apply(*mesh)) {
          Update(fespaces, gfs, *a, *minv);
          cout << "  Coarsening! Estimated total error: "
               << spatial_estimator->GetTotalError() << endl;
        }
        MFEM_PERF_END("Coarsen");
      }

      std::cout << "  Number of elements: " << mesh->GetNE() << std::endl;
      cout << "  Number of unknowns: " << phimfespace.GetVSize() << "+"
           << sfespace.GetVSize() << endl;
      MFEM_PERF_END("AMR");
    }

    // Reset temporal error estimator
    t_estgf = 0.0;

    // Store solution of last time step for temporal error estimator
    GridFunction phimprevgf(&phimfespace); // phim at previous time step
    phimprevgf = phimgf;
    GridFunction sprevgf(&sfespace); // s at previous time step
    sprevgf = sgf;

    MFEM_PERF_BEGIN("Assemble Elements");
    MFEM_PERF_BEGIN("Diffusion");
    a->ComputeElementMatrices();
    MFEM_PERF_END("Diffusion");

    MFEM_PERF_BEGIN("Inverse Matrix");
    minv->ComputeElementMatrices();
    MFEM_PERF_END("Inverse Matrix");
    MFEM_PERF_END("Assemble Elements");

    MFEM_PERF_BEGIN("Assemble Faces");
    // Precompute face matrices
    std::vector<DenseMatrix> face_matrices =
        ComputeFaceMatrices(mesh, &phimfespace, dginteg, nullptr);
    MFEM_PERF_END("Assemble Faces");

    MFEM_PERF_BEGIN("CFL Condition");
    if (use_amr || timestep_index == 0) {
        ComputeMaxTimeStepLenghtsViaGershgorin(
            critical_time_step_lengths, *minv, *a, face_matrices, stepper, ec);
    }
    critical_time_step_lengths *= CFL_factor;
    std::cout << "Estimate for smallest stable time step: "
              << critical_time_step_lengths.Min() << std::endl;
    MFEM_PERF_END("CFL Condition");

    // Compute number of substeps
    Array<int> num_substeps_per_element(mesh->GetNE());
    for (int ei = 0; ei < mesh->GetNE(); ei++) {
      if (critical_time_step_lengths(ei) <= 0.0) {
        std::cout << "Element " << ei
                  << " with non-positive CFL condition. Aborting. dt_cfl="
                  << critical_time_step_lengths(ei) << std::endl;
        std::exit(-1);
      }
      const int num_substeps_cfl_exponent =
          int(ceil(log2(Δt_barrier / critical_time_step_lengths(ei))));
      num_substeps_per_element[ei] =
          std::pow(2, std::max(num_substeps_cfl_exponent, 0));
      // constexpr double Δt_cell_transient =
      //     0.01; // TODO determine either from command line or from error
      //           // estimator
      // if (use_substepping && t_estgf(ei) > substep_threshold) {
      //   const auto cell_transient_exponent =
      //       int(ceil(log2(Δt_barrier / Δt_cell_transient)));
      //   if (cell_transient_exponent > num_substeps_cfl_exponent) {
      //     num_substeps_per_element[ei] =
      //         std::pow(2, std::max(cell_transient_exponent, 0));
      //   }
      // }

      if (num_substeps_per_element[ei] == 0) {
        std::cout << "Element " << ei << " with 0 time steps. Aborting."
                  << std::endl;
        std::exit(-1);
      }
    }

    MFEM_PERF_BEGIN("Queue Setup");

    // Resize and clear queues
    const int num_substeps = num_substeps_per_element.Max();
    std::cout << "  smallest time step=" << Δt_barrier / num_substeps
              << std::endl;
    predictor_corrector_queues.resize(num_substeps);
    for (auto &pqpair : predictor_corrector_queues) {
      pqpair.first.clear();
      pqpair.second.clear();
    }

    // Initial queue filling
    lowest_level_elements.clear();
    for (int ei = 0; ei < mesh->GetNE(); ei++) {
      if (num_substeps_per_element[ei] == num_substeps) {
        lowest_level_elements.push_back(ei);
      } else {
        predictor_corrector_queues[0].first.push_back(ei);
      }
    }
    MFEM_PERF_END("Queue Setup");

    // Main loop
    int element_counter = 0;
    MFEM_LOOP_BEGIN(loop_ann, "Main Update Loop");
    std::cout << "  entering substepping..." << std::endl;
    for (int substep = 0; substep < num_substeps; substep++) {
      MFEM_LOOP_ITERATION(loop_ann, substep);

      DEBUG_PRINT("  substep=" << substep);
      // Shoot predictors
      for (auto ei : predictor_corrector_queues[substep].first) {
        DEBUG_PRINT("  Predictor loop " << ei);
        AdvanceElement(ei, substep, num_substeps_per_element, num_substeps,
                       phimgf, phim_predictor, phim_predictor, sgf, s_predictor, ec,
                       *a, *minv, stepper, nodal_ir, face_matrices,
                       last_timebarrier, Δt_barrier, stim);
        element_counter++;
        // Queue up corrector
        const int substepgridwidth =
            num_substeps / num_substeps_per_element[ei];
        // "-1" because we have to update the solution in the substep before the
        // next predictor step
        const int next_correction_substep = substep + substepgridwidth - 1;
        predictor_corrector_queues[next_correction_substep].second.push_back(
            ei);
      }

      // Update elements with largest number of substeps.
      // No predictor step for these necessary for these and we can just update
      // the solution.
      for (auto ei : lowest_level_elements) {
        DEBUG_PRINT("  Fine loop " << ei);
        AdvanceElement(ei, substep, num_substeps_per_element, num_substeps,
                       phimgf, phim_predictor, phim_predictor, sgf, s_predictor, ec,
                       *a, *minv, stepper, nodal_ir, face_matrices,
                       last_timebarrier, Δt_barrier, stim);
        element_counter++;
      }

      // Apply correctors for predicted elements
      for (auto ei : predictor_corrector_queues[substep].second) {
        DEBUG_PRINT("  Apply coarse " << ei);
        phimfespace.GetElementDofs(ei, ec.phimdofs);
        phim_predictor.GetSubVector(ec.phimdofs, ec.phimelvals);
        phimgf.SetSubVector(ec.phimdofs, ec.phimelvals);
        GetElementVDofsLinear(sfespace, ei, ec.sdofs);
        s_predictor.GetSubVector(ec.sdofs, ec.selvals);
        sgf.SetSubVector(ec.sdofs, ec.selvals);
        // Queue up element again
        if (substep != num_substeps - 1) {
          predictor_corrector_queues[substep + 1].first.push_back(ei);
        }
      }

      // Apply correctors for finest elements in time
      for (auto ei : lowest_level_elements) {
        DEBUG_PRINT("  Apply fine " << ei);
        phimfespace.GetElementDofs(ei, ec.phimdofs);
        phim_predictor.GetSubVector(ec.phimdofs, ec.phimelvals);
        phimgf.SetSubVector(ec.phimdofs, ec.phimelvals);
        GetElementVDofsLinear(sfespace, ei, ec.sdofs);
        s_predictor.GetSubVector(ec.sdofs, ec.selvals);
        sgf.SetSubVector(ec.sdofs, ec.selvals);
      }
    }
    MFEM_LOOP_END(loop_ann);
    std::cout << "  done!" << std::endl;

    std::cout << "  Needed " << element_counter
              << " element evaluations for the time step." << std::endl;

    // Modified version of the temporal error estimator in Ratti and Verani
    // (2019) NOTE: The time integral is approximated via midpoint rule.
    //       The space integral is given by the nodal basis.
    // ---------------vvv<NOT REALLY PART OF LTS>vvv-------------------
    // if (use_substepping) {
    //   std::cout << "  temporal error estimation..." << std::endl;
    //   MFEM_PERF_BEGIN("Temporal Estimator");
    //   Vector smidbuf(stepper->InternalDim());
    //   Vector sprevbuf(stepper->InternalDim());
    //   Vector srhsprevbuf(stepper->InternalDim());
    //   Vector srhsmidbuf(stepper->InternalDim());
    //   Vector pos(sdim);
    //   for (int ei = 0; ei < mesh->GetNE(); ei++) {
    //     auto geotrafo = mesh->GetElementTransformation(ei);
    //     // Load solution on element 'ei' for tₙ and tₙ₋₁ (latter indicated by
    //     // suffix 2)
    //     phimfespace.GetElementDofs(ei, ec.phimdofs);
    //     phimgf.GetSubVector(ec.phimdofs, ec.phimelvals);
    //     phimprevgf.GetSubVector(ec.phimdofs, ec.phimelvals2);
    //     GetElementVDofsLinear(sfespace, ei, ec.sdofs);
    //     sgf.GetSubVector(ec.sdofs, ec.selvals);
    //     sprevgf.GetSubVector(ec.sdofs, ec.selvals2);
    //     for (int node = 0; node < nodal_ir.GetNPoints(); node++) {
    //       // Compute integration point weight
    //       auto ip = nodal_ir.IntPoint(node);
    //       geotrafo->SetIntPoint(&ip);
    //       auto detJdV = geotrafo->Weight() * ip.weight;
    //       // Store midpoint for s and previous s
    //       for (int i = 0; i < stepper->InternalDim(); i++) {
    //         sprevbuf(i) = ec.selvals2(stepper->InternalDim() * node + i);
    //         smidbuf(i) = (ec.selvals2(stepper->InternalDim() * node + i) +
    //                       ec.selvals(stepper->InternalDim() * node + i)) /
    //                      2;
    //       }
    //       double V_prev = ec.phimelvals2(node);
    //       double V_mid = (ec.phimelvals(node) + ec.phimelvals2(node)) / 2;
    //       // Previous s rhs
    //       stepper->InternalRHS(srhsprevbuf, V_prev, sprevbuf);
    //       // Midpoint s rhs
    //       stepper->InternalRHS(srhsmidbuf, V_mid, smidbuf);
    //       // Numerical integral evaluation
    //       double node_temporal_estimator = 0.0;
    //       for (int i = 0; i < stepper->InternalDim(); i++) {
    //         double state_val = srhsprevbuf[i] - srhsmidbuf[i];
    //         node_temporal_estimator += state_val * state_val; // Squared...
    //       }
    //       double reaction_val =
    //           stepper->EvalReaction(V_prev, sprevbuf, 0, last_timebarrier) -
    //           stepper->EvalReaction(V_mid, smidbuf, 0,
    //                                 (last_timebarrier + Δt_barrier) / 2);
    //       node_temporal_estimator += reaction_val * reaction_val; // Squared...
    //       t_estgf[ei] += node_temporal_estimator * detJdV;
    //       // = sqrt(node_temporal_estimator)*sqrt(node_temporal_estimator) *  detJdV
    //     }
    //     t_estgf[ei] = sqrt(t_estgf[ei]);
    //   }
    //   MFEM_PERF_END("Temporal Estimator");
    // }
    // ---------------^^^<NOT REALLY PART OF LTS>^^^-------------------

    {
      std::ofstream of(std::string(output_name) + "/element-counter.txt",
                       std::ios_base::app);
      of << last_timebarrier << " " << element_counter << std::endl;
    }

    if (phimgf.CheckFinite() > 0) {
      cout << "Solve failed." << endl;
      std::exit(-1);
    }
  }
  cout << "FINAL " << t_final << " ms" << endl;

  if (paraview) {
    MFEM_PERF_BEGIN("Paraview");
    dc->SetCycle(paraview_cycle++);
    dc->SetTime(t_final);
    dc->Save();
    MFEM_PERF_END("Paraview");
  }

  // 11. Free the used memory.
  delete spatial_estimator;
  delete a;
  delete minv;
  delete dginteg;
  delete mesh;
#ifdef MFEM_USE_CALIPER
  mgr.flush();
#endif
  return 0;
}
