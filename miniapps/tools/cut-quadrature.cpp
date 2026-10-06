// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-3-Clause

// This miniapp demonstrates the complete element-local cut-quadrature flow:
//
//  1. Represent a level set with either a GridFunction or a Coefficient.
//  2. Extract deformation-independent reference-element polynomials.
//  3. Construct scalar and packed host rules with the Algoim backend.
//  4. Retain a reference rule using the complete application-owned cache key.
//  5. Apply current physical metrics while integrating volume and interface
//     measures, including after mesh deformation.
//  6. Invalidate retained data explicitly after changing the level set.
//
// Reference rules contain reference coordinates and weights. They are not
// modified when the mesh moves; physical Jacobians and normals are evaluated
// by the integration layer against the current ElementTransformation.
//
// Sample runs: cut-quadrature -no-vis
//              cut-quadrature -r 1 -o 3 -qo 8 -c 0.3
//              cut-quadrature -vis -p 19916

#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace mfem;

int main(int argc, char *argv[])
{
   const char *mesh_file = "";
   const char *device_config = "cpu";
   int refinement_levels = 0;
   int order = 2;
   int quadrature_order = 6;
   real_t cut_position = 0.45;
   bool visualization = false;
   int visport = 19916;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Planar quadrilateral mesh file; empty uses a two-element "
                  "unit square.");
   args.AddOption(&refinement_levels, "-r", "--refine",
                  "Number of uniform mesh refinements.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element and coefficient interpolation order "
                  "(at least one).");
   args.AddOption(&quadrature_order, "-qo", "--quadrature-order",
                  "Target cut-quadrature order (0 through 19).");
   args.AddOption(&cut_position, "-c", "--cut-position",
                  "Position of the zero level set x = cut-position.");
   args.AddOption(&device_config, "-d", "--device",
                  "MFEM device configuration; Algoim rule construction "
                  "remains on the host.");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable or disable GLVis visualization of the initial "
                  "level set.");
   args.AddOption(&visport, "-p", "--visualization-port", "GLVis server port.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return args.Help() ? 0 : 1;
   }
   args.PrintOptions(std::cout);

   if (refinement_levels < 0 || order < 1 || !std::isfinite(cut_position) ||
       (visualization && (visport < 1 || visport > 65535)))
   {
      std::cerr << "Refinement levels must be nonnegative, order must be positive, "
                << "cut position must be finite, and the visualization port "
                << "must be between 1 and 65535.\n";
      return 1;
   }

#ifndef MFEM_USE_ALGOIM
   std::cout << "This miniapp requires MFEM_USE_ALGOIM=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
#else
   AlgoimCutQuadratureConstructor quadrature_constructor;
   const auto &capabilities = quadrature_constructor.Capabilities();
   if (quadrature_order < capabilities.min_order ||
       quadrature_order > capabilities.max_order)
   {
      std::cerr << "Quadrature order must be between " << capabilities.min_order
                << " and " << capabilities.max_order << ".\n";
      return 1;
   }

   Device device(device_config);
   device.Print();

   // By default, two quadrilateral elements cover [0,1]^2 and x = 0.45 cuts
   // the first element. A mesh file and refinement can change the partition.
   Mesh mesh = mesh_file[0] ? Mesh(mesh_file, 1, 1) :
               Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                                     true, 1.0, 1.0);
   if (mesh.Dimension() != 2 || mesh.SpaceDimension() != 2 || mesh.GetNE() == 0)
   {
      std::cerr << "This miniapp requires a nonempty planar 2D "
                << "quadrilateral mesh.\n";
      return 1;
   }
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      if (mesh.GetElementBaseGeometry(e) != Geometry::SQUARE)
      {
         std::cerr << "This miniapp supports quadrilateral elements only.\n";
         return 1;
      }
   }
   for (int r = 0; r < refinement_levels; r++) { mesh.UniformRefinement(); }

   H1_FECollection collection(order, 2);
   FiniteElementSpace space(&mesh, &collection);
   GridFunction level_set(&space);
   FunctionCoefficient phi([cut_position](const Vector &x)
   { return x(0) - cut_position; });
   level_set.ProjectCoefficient(phi);

   if (visualization)
   {
      socketstream solution("localhost", visport);
      if (solution)
      {
         solution.precision(8);
         solution << "solution\n" << mesh << level_set << std::flush;
      }
      else
      {
         std::cerr << "Unable to connect to GLVis on port " << visport << ".\n";
      }
   }

   // Extractors translate application fields into ElementLevelSet objects.
   // Their revisions are caller-controlled value revisions; they are not
   // inferred from object addresses or GridFunction sequence numbers.
   GridFunctionLevelSetExtractor grid_function_extractor(level_set, 1);

   // A general Coefficient is the alternative source. Unlike the exact
   // GridFunction path above, it is interpolated locally at the requested
   // approximation order selected by --order.
   CoefficientLevelSetExtractor coefficient_extractor(phi, order, 1);

   // A constructor may be shared by concurrent callers, but each thread must
   // own a separate workspace.
   auto workspace = quadrature_constructor.CreateWorkspace();

   // The order is an MFEM target order, not Algoim's native `qo`. Requesting
   // normals stores reference normals needed for physical surface metrics.
   CutQuadratureRequest request;
   request.order = quadrature_order;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;

   // Extract once per element. The batch records the descriptor and status of
   // every extraction so the constructor can validate homogeneity. This miniapp
   // exits if extraction fails for any element.
   const int ne = mesh.GetNE();
   std::vector<ElementLevelSet> local(ne);
   ElementLevelSetBatch batch;
   batch.element_descriptors.SetSize(ne);
   batch.extraction_status.SetSize(ne);
   for (int e = 0; e < ne; e++)
   {
      ElementTransformation &Tr = *mesh.GetElementTransformation(e);
      batch.extraction_status[e] =
         grid_function_extractor.GetElementLevelSet(e, Tr, local[e]);
      if (batch.extraction_status[e] != CutQuadratureStatus::Success)
      {
         std::cerr << "Level-set extraction failed for element " << e << ".\n";
         return 1;
      }
      batch.element_descriptors[e] =
      { local[e].geometry, local[e].basis, local[e].order };
   }

   // This mesh and finite-element space give all successful entries the same
   // geometry, basis, and polynomial order. Coefficients are packed as one
   // element per matrix column.
   batch.descriptor = batch.element_descriptors[0];
   batch.coefficients.SetSize(local[0].coefficients.Size(), ne);
   for (int e = 0; e < ne; e++)
   {
      batch.coefficients.SetCol(e, local[e].coefficients);
   }

   // The return value reports only a whole-call failure. On Success, each
   // entry of packed.status must still be checked before consuming that
   // element's range in the packed points, weights, normals, and offsets.
   BatchedReferenceCutQuadrature packed;
   const CutQuadratureStatus batch_status =
      quadrature_constructor.GenerateReferenceBatch(
         batch, request, packed, *workspace);
   MFEM_VERIFY(batch_status == CutQuadratureStatus::Success,
               "batch construction failed");
   for (int e = 0; e < ne; e++)
   {
      if (packed.status[e] != CutQuadratureStatus::Success)
      {
         std::cerr << "Cut-quadrature construction failed for element "
                   << e << ".\n";
         return 1;
      }
   }

   // A retained scalar result is application-owned. Its complete reuse key is
   // extractor identity, element identity, extractor revision, and exact request
   // equality. The result itself deliberately carries none of this metadata.
   RetainedCutQuadrature retained;
   retained.extractor_id = grid_function_extractor.Id();
   retained.element = 0;
   retained.revision = grid_function_extractor.Revision();
   retained.request = request;
   MFEM_VERIFY(quadrature_constructor.GenerateReference(
                  local[0], request, retained.result, *workspace) ==
               CutQuadratureStatus::Success,
               "scalar construction failed");

   // Sum physical measures over every element using the packed reference rules.
   // Reconstruct each element's rules for the scalar integration helpers. These
   // apply Tr.Weight() for volume and Tr.Weight()*||J^{-T} n_ref|| for interface.
   // The packed data is reused unchanged after deforming the mesh below.
   auto integrate_batch = [&mesh, &packed](real_t &volume, real_t &surface)
   {
      volume = surface = 0.0;
      ConstantCoefficient one(1.0);
      ReferenceCutQuadrature element_rules;
      const int dim = mesh.Dimension();
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         element_rules.status = packed.status[e];
         const int volume_begin = packed.volume.offsets[e];
         const int volume_end = packed.volume.offsets[e + 1];
         element_rules.volume.SetSize(volume_end - volume_begin);
         for (int q = volume_begin; q < volume_end; q++)
         {
            element_rules.volume.IntPoint(q - volume_begin).Set2w(
               packed.volume.points(0, q), packed.volume.points(1, q),
               packed.volume.weights(q));
         }
         element_rules.volume.SetPointIndices();

         const int interface_begin = packed.interface.offsets[e];
         const int interface_end = packed.interface.offsets[e + 1];
         const int nq = interface_end - interface_begin;
         element_rules.interface.rule.SetSize(nq);
         element_rules.interface.reference_normals.SetSize(dim, nq);
         for (int q = interface_begin; q < interface_end; q++)
         {
            const int j = q - interface_begin;
            element_rules.interface.rule.IntPoint(j).Set2w(
               packed.interface.points(0, q), packed.interface.points(1, q),
               packed.interface.weights(q));
            for (int d = 0; d < dim; d++)
            {
               element_rules.interface.reference_normals(d, j) =
                  packed.interface.normals(d, q);
            }
         }
         element_rules.interface.rule.SetPointIndices();

         ElementTransformation &Tr = *mesh.GetElementTransformation(e);
         volume += CutQuadratureIntegrator::IntegrateVolume(
                      one, Tr, element_rules);
         surface += CutQuadratureIntegrator::IntegrateInterface(
                       one, Tr, element_rules);
      }
   };
   real_t volume, surface;
   integrate_batch(volume, surface);

   // This material deformation scales x by 1.2 and y by 0.8. It changes the
   // physical measure through the current Jacobian, but not the reference
   // level-set polynomial or its retained rule, so the reuse key still matches.
   VectorFunctionCoefficient deform(2, [](const Vector &x, Vector &y)
   {
      y.SetSize(2);
      y(0) = 1.2*x(0);
      y(1) = 0.8*x(1);
   });
   mesh.Transform(deform);
   MFEM_VERIFY(retained.IsValid(grid_function_extractor, 0, request),
               "mesh deformation invalidated a reference rule");
   real_t deformed_volume, deformed_surface;
   integrate_batch(deformed_volume, deformed_surface);

   // Field edits require an explicit revision bump. Without IncrementRevision,
   // the application-owned key would still match and silently reuse stale
   // quadrature data.
   level_set += 0.1;
   grid_function_extractor.IncrementRevision();
   MFEM_VERIFY(!retained.IsValid(grid_function_extractor, 0, request),
               "field revision failed to invalidate retained rule");

   // Coefficient extraction evaluates phi through the current (deformed)
   // element transformation and constructs a new local polynomial.
   ElementTransformation &deformed_Tr = *mesh.GetElementTransformation(0);
   ElementLevelSet coefficient_local;
   MFEM_VERIFY(coefficient_extractor.GetElementLevelSet(
                  0, deformed_Tr, coefficient_local) ==
               CutQuadratureStatus::Success,
               "coefficient extraction failed");

   // Device execution is representable in the common API, but the current
   // Algoim backend is host-only and must reject it rather than fall back.
   CutQuadratureRequest device_request = request;
   device_request.execution = CutExecutionMode::Device;
   ReferenceCutQuadrature rejected;
   MFEM_VERIFY(quadrature_constructor.GenerateReference(
                  coefficient_local, device_request, rejected, *workspace) ==
               CutQuadratureStatus::UnsupportedExecutionMode,
               "device construction must be rejected explicitly");

   std::cout << "batch elements: " << packed.status.Size()
             << ", reference points: " << packed.volume.weights.Size()
             << ", total volume: " << volume
             << ", total interface: " << surface
             << ", deformed total volume: " << deformed_volume
             << ", deformed total interface: " << deformed_surface << '\n';
   return 0;
#endif
}
