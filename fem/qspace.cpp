// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "qspace.hpp"
#include "qfunction.hpp"
#include "../general/forall.hpp"

namespace mfem
{

QuadratureSpaceBase::QuadratureSpaceBase(Mesh &mesh_, Geometry::Type geom,
                                         const IntegrationRule &ir)
   : mesh(mesh_), order(ir.GetOrder())
{
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      int_rule[g] = NULL;
   }
   int_rule[geom] = &ir;
}

void QuadratureSpaceBase::ConstructIntRules(int dim)
{
   Array<Geometry::Type> geoms;
   mesh.GetGeometries(dim, geoms);
   for (Geometry::Type geom : geoms)
   {
      int_rule[geom] = &IntRules.Get(geom, order);
   }
}

namespace
{

void ScaleByQuadratureWeights(Vector &weights, const IntegrationRule &ir)
{
   const int N = weights.Size();
   const int n = ir.Size();
   real_t *d_weights = weights.ReadWrite();
   const real_t *d_w = ir.GetWeights().Read();

   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i)
   {
      d_weights[i] *= d_w[i%n];
   });
}

} // anonymous namespace

void QuadratureSpaceBase::ConstructWeights() const
{
   // First get the Jacobian determinants (without the quadrature weight
   // contributions). We also store the pointer to the Vector object, so that
   // we know when the cached weights are invalidated.
   nodes_sequence = mesh.GetNodesSequence();
   weights = GetGeometricFactorWeights();

   // Then scale by the quadrature weights.
   const IntegrationRule &ir = GetIntRule(0);
   ScaleByQuadratureWeights(weights, ir);
}

const Vector &QuadratureSpaceBase::GetWeights() const
{
   if (GetNE() == 0) { return weights; }
   if (weights.Size() == 0 || nodes_sequence != mesh.GetNodesSequence())
   {
      ConstructWeights();
   }
   return weights;
}

real_t QuadratureSpaceBase::Integrate(Coefficient &coeff) const
{
   QuadratureFunction qf(const_cast<QuadratureSpaceBase*>(this));
   coeff.Project(qf);
   return qf.Integrate();
}

void QuadratureSpaceBase::Integrate(VectorCoefficient &coeff,
                                    Vector &integrals) const
{
   const int vdim = coeff.GetVDim();
   QuadratureFunction qf(const_cast<QuadratureSpaceBase*>(this), vdim);
   coeff.Project(qf);
   qf.Integrate(integrals);
}

void QuadratureSpace::ConstructOffsets()
{
   const int num_elem = mesh.GetNE();
   offsets.SetSize(num_elem + 1);

   if (mesh.GetNumGeometries(mesh.Dimension()) == 1)
   {
      Array<Geometry::Type> geoms;
      mesh.GetGeometries(mesh.Dimension(), geoms);
      const int ir_size = int_rule[geoms[0]]->GetNPoints();
      auto d_offsets = offsets.Write();
      mfem::forall(num_elem + 1, [=] MFEM_HOST_DEVICE (int i)
      {
         d_offsets[i] = i*ir_size;
      });
   }
   else
   {
      int offset = 0;
      for (int i = 0; i < num_elem; i++)
      {
         offsets[i] = offset;
         const Geometry::Type geom = mesh.GetElementBaseGeometry(i);
         MFEM_ASSERT(int_rule[geom] != NULL, "Missing integration rule.");
         offset += int_rule[geom]->GetNPoints();
      }
      offsets[num_elem] = offset;
   }
   size = offsets.Last();
}

void QuadratureSpace::Construct()
{
   ConstructIntRules(mesh.Dimension());
   ConstructOffsets();
}

QuadratureSpace::QuadratureSpace(Mesh *mesh_, std::istream &in)
   : QuadratureSpaceBase(*mesh_)
{
   const char *msg = "invalid input stream";
   std::string ident;

   in >> ident; MFEM_VERIFY(ident == "QuadratureSpace", msg);
   in >> ident; MFEM_VERIFY(ident == "Type:", msg);
   in >> ident;
   if (ident == "default_quadrature")
   {
      in >> ident; MFEM_VERIFY(ident == "Order:", msg);
      in >> order;
   }
   else
   {
      MFEM_ABORT("unknown QuadratureSpace type: " << ident);
      return;
   }

   Construct();
}

QuadratureSpace::QuadratureSpace(Mesh &mesh_, const IntegrationRule &ir)
   : QuadratureSpaceBase(mesh_, mesh_.GetElementGeometry(0), ir)
{
   MFEM_VERIFY(mesh.GetNumGeometries(mesh.Dimension()) == 1,
               "Constructor not valid for mixed meshes");
   ConstructOffsets();
}

void QuadratureSpace::Save(std::ostream &os) const
{
   os << "QuadratureSpace\n"
      << "Type: default_quadrature\n"
      << "Order: " << order << '\n';
}

const Vector &QuadratureSpace::GetGeometricFactorWeights() const
{
   auto flags = GeometricFactors::DETERMINANTS;
   // TODO: assumes only one integration rule. This should be fixed once
   // Mesh::GetGeometricFactors acceps a QuadratureSpace instead of
   // IntegrationRule.
   const IntegrationRule &ir = GetIntRule(0);
   auto *geom = mesh.GetGeometricFactors(ir, flags);
   return geom->detJ;
}

FaceQuadratureSpace::FaceQuadratureSpace(Mesh &mesh_, int order_,
                                         FaceType face_type_)
   : QuadratureSpaceBase(mesh_, order_),
     face_type(face_type_),
     num_faces(mesh.GetNFbyType(face_type))
{
   Construct();
}

FaceQuadratureSpace::FaceQuadratureSpace(Mesh &mesh_, const IntegrationRule &ir,
                                         FaceType face_type_)
   : QuadratureSpaceBase(mesh_, mesh_.GetFaceGeometry(0), ir),
     face_type(face_type_),
     num_faces(mesh.GetNFbyType(face_type))
{
   MFEM_VERIFY(mesh.GetNumGeometries(mesh.Dimension() - 1) == 1,
               "Constructor not valid for mixed meshes");
   ConstructOffsets();
}

void FaceQuadratureSpace::ConstructOffsets()
{
   face_indices.SetSize(num_faces);
   offsets.SetSize(num_faces + 1);
   int offset = 0;
   int f_idx = 0;
   for (int i = 0; i < mesh.GetNumFacesWithGhost(); i++)
   {
      const Mesh::FaceInformation face = mesh.GetFaceInformation(i);
      if (face.IsNonconformingCoarse() || !face.IsOfFaceType(face_type))
      {
         continue;
      }
      face_indices[f_idx] = i;
      face_indices_inv[i] = f_idx;
      offsets[f_idx] = offset;
      Geometry::Type geom = mesh.GetFaceGeometry(i);
      MFEM_ASSERT(int_rule[geom] != NULL, "Missing integration rule");
      offset += int_rule[geom]->GetNPoints();

      f_idx++;
   }
   offsets[num_faces] = size = offset;
}

void FaceQuadratureSpace::Construct()
{
   ConstructIntRules(mesh.Dimension() - 1);
   ConstructOffsets();
}

int FaceQuadratureSpace::GetPermutedIndex(int idx, int iq) const
{
   const int f_idx = face_indices[idx];
   if (Geometry::IsTensorProduct(GetGeometry(idx)))
   {
      const int dim = mesh.Dimension();
      const IntegrationRule &ir = GetIntRule(idx);
      const int q1d = (int)floor(pow(ir.GetNPoints(), 1.0/(dim-1)) + 0.5);
      const Mesh::FaceInformation face = mesh.GetFaceInformation(f_idx);
      return ToLexOrdering(dim, face.element[0].local_face_id, q1d, iq);
   }
   else
   {
      return iq;
   }
}

int FaceQuadratureSpace::GetEntityIndex(const ElementTransformation &T) const
{
   auto get_face_index = [this](const int idx)
   {
      const auto it = face_indices_inv.find(idx);
      if (it == face_indices_inv.end()) { return -1; }
      else { return it->second; }
   };

   switch (T.ElementType)
   {
      case ElementTransformation::FACE:
         return get_face_index(T.ElementNo);
      case ElementTransformation::BDR_ELEMENT:
      case ElementTransformation::BDR_FACE:
         return get_face_index(mesh.GetBdrElementFaceIndex(T.ElementNo));
      default:
         MFEM_ABORT("Invalid element type.");
         return -1;
   }
}

void FaceQuadratureSpace::Save(std::ostream &os) const
{
   os << "FaceQuadratureSpace\n"
      << "Type: default_quadrature\n"
      << "Order: " << order << '\n';
}

const Vector &FaceQuadratureSpace::GetGeometricFactorWeights() const
{
   auto flags = FaceGeometricFactors::DETERMINANTS;
   // TODO: assumes only one integration rule. This should be fixed once
   // Mesh::GetFaceGeometricFactors acceps a QuadratureSpace instead of
   // IntegrationRule.
   const IntegrationRule &ir = GetIntRule(0);
   auto *geom = mesh.GetFaceGeometricFactors(ir, flags, face_type);
   return geom->detJ;
}

} // namespace mfem
