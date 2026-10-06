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

#include "mfem.hpp"

#include <iostream>
#include <vector>

using namespace mfem;

int main()
{
#ifndef MFEM_USE_ALGOIM
   std::cout << "This miniapp requires MFEM_USE_ALGOIM=YES.\n";
   return MFEM_SKIP_RETURN_VALUE;
#else
   // The two unit-square elements cover [0,1]^2. The zero set x = 0.45 cuts
   // the first element and leaves the second element outside the negative
   // phase selected by the default request below.
   Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::QUADRILATERAL,
                                     true, 1.0, 1.0);
   H1_FECollection collection(2, 2);
   FiniteElementSpace space(&mesh, &collection);
   GridFunction level_set(&space);
   FunctionCoefficient phi([](const Vector &x) { return x(0) - 0.45; });
   level_set.ProjectCoefficient(phi);

   // Extractors translate application fields into ElementLevelSet objects.
   // Their revisions are caller-controlled value revisions; they are not
   // inferred from object addresses or GridFunction sequence numbers.
   GridFunctionLevelSetExtractor grid_function_extractor(level_set, 1);

   // A general Coefficient is the alternative source. Unlike the exact
   // GridFunction path above, it is interpolated locally at the requested
   // approximation order (two here).
   CoefficientLevelSetExtractor coefficient_extractor(phi, 2, 1);

   // A constructor may be shared by concurrent callers, but each thread must
   // own a separate workspace.
   AlgoimCutQuadratureConstructor quadrature_constructor;
   auto workspace = quadrature_constructor.CreateWorkspace();

   // The order is an MFEM target order, not Algoim's native `qo`. Requesting
   // normals stores reference normals needed for physical surface metrics.
   CutQuadratureRequest request;
   request.order = 6;
   request.measures = CutMeasure::Volume | CutMeasure::Interface;
   request.compute_reference_normals = true;

   // Extract once per element. The batch records the descriptor and status of
   // every extraction so the constructor can validate homogeneity and preserve
   // per-element extraction failures.
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
      if (batch.extraction_status[e] == CutQuadratureStatus::Success)
      {
         batch.element_descriptors[e] =
         { local[e].geometry, local[e].basis, local[e].order };
      }
   }

   // This mesh and finite-element space give all successful entries the same
   // geometry, basis, and polynomial order. Coefficients are packed as one
   // element per matrix column; failed-extraction columns would be ignored.
   batch.descriptor = batch.element_descriptors[0];
   batch.coefficients.SetSize(local[0].coefficients.Size(), ne);
   for (int e = 0; e < ne; e++)
   {
      if (batch.extraction_status[e] == CutQuadratureStatus::Success)
      {
         batch.coefficients.SetCol(e, local[e].coefficients);
      }
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

   // Integrating the constant one returns geometric measure. The integrator
   // applies Tr.Weight() exactly once for volume. For the interface it also
   // applies ||J^{-T} n_ref|| and can optionally return physical unit normals.
   ConstantCoefficient one(1.0);
   ElementTransformation &Tr = *mesh.GetElementTransformation(0);
   const real_t volume = CutQuadratureIntegrator::IntegrateVolume(
                            one, Tr, retained.result);
   const real_t surface = CutQuadratureIntegrator::IntegrateInterface(
                             one, Tr, retained.result);

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
   ElementTransformation &deformed_Tr = *mesh.GetElementTransformation(0);
   const real_t deformed_volume = CutQuadratureIntegrator::IntegrateVolume(
                                     one, deformed_Tr, retained.result);

   // Field edits require an explicit revision bump. Without IncrementRevision,
   // the application-owned key would still match and silently reuse stale
   // quadrature data.
   level_set += 0.1;
   grid_function_extractor.IncrementRevision();
   MFEM_VERIFY(!retained.IsValid(grid_function_extractor, 0, request),
               "field revision failed to invalidate retained rule");

   // Coefficient extraction evaluates phi through the current (deformed)
   // element transformation and constructs a new local polynomial.
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
             << ", volume: " << volume
             << ", interface: " << surface
             << ", deformed volume: " << deformed_volume << '\n';
   return 0;
#endif
}
