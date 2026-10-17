// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "entityset_integral.hpp"
#include "../mesh/mesh.hpp"

#ifdef MFEM_USE_MPI
#include "../mesh/pmesh.hpp"
#endif

namespace mfem
{

namespace
{

void VerifyVectorField(const GridFunction &field, const Mesh &mesh,
                       FiniteElement::MapType map_type,
                       const char *quantity)
{
   const FiniteElementSpace *fes = field.FESpace();
   MFEM_VERIFY(fes != NULL, "The GridFunction has no finite element space.");
   MFEM_VERIFY(field.GetSequence() == fes->GetSequence(),
               "The GridFunction must be updated with its finite element space.");
   MFEM_VERIFY(fes->GetMesh() == &mesh,
               "The grid function and integration domain use different meshes.");
   MFEM_VERIFY(fes->GetTypicalFE()->GetMapType() == map_type,
               quantity << " requires the matching vector finite element space.");
   MFEM_VERIFY(fes->GetVDim() == 1,
               quantity << " requires a vector finite element field with vdim = 1.");
}

real_t GlobalSumImpl(const GridFunction &field, real_t local_value)
{
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh *>
                          (field.FESpace()->GetMesh());
   if (pmesh)
   {
      real_t global_value;
      MPI_Allreduce(&local_value, &global_value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, pmesh->GetComm());
      return global_value;
   }
#endif
   return local_value;
}

int GetEdgeIncidentElement(const Mesh &mesh, int edge)
{
   if (mesh.Dimension() == 1) { return edge; }

   if (mesh.Dimension() == 2)
   {
      int element1, element2;
      mesh.GetFaceElements(edge, &element1, &element2);
      return element1;
   }

   const Table *edge_faces = mesh.GetEdgeFaceTable();
   MFEM_VERIFY(edge_faces != NULL, "Unable to find a face incident to an edge.");
   Array<int> faces;
   edge_faces->GetRow(edge, faces);
   for (const int face : faces)
   {
      int element1, element2;
      mesh.GetFaceElements(face, &element1, &element2);
      if (element1 >= 0 && element1 < mesh.GetNE()) { return element1; }
      if (element2 >= 0 && element2 < mesh.GetNE()) { return element2; }
   }
   MFEM_ABORT("Unable to find a local element incident to an edge.");
   return -1;
}

} // namespace

EntitySetFunctional::EntitySetFunctional(const GridFunction &field,
                                         const EntitySets &sets,
                                         EntitySets::EntityType type,
                                         int set_index)
   : field_(field), sets_(sets), type_(type), set_index_(set_index)
{
   MFEM_VERIFY(set_index >= 0 && sets_.SetExists(type_, set_index),
               "Entity set index out of range.");
   VerifyCompatibleField();
}

void EntitySetFunctional::VerifyCompatibleField() const
{
   const FiniteElementSpace *fes = field_.FESpace();
   MFEM_VERIFY(fes != NULL, "The GridFunction has no finite element space.");
   MFEM_VERIFY(field_.GetSequence() == fes->GetSequence(),
               "The GridFunction must be updated with its finite element space.");
   MFEM_VERIFY(sets_.GetMesh() == fes->GetMesh(),
               "The entity set and GridFunction must use the same mesh.");
}

real_t EntitySetFunctional::GlobalSum(real_t local_value) const
{
#ifdef MFEM_USE_MPI
   const ParMesh *pmesh = dynamic_cast<const ParMesh *>
                          (Field().FESpace()->GetMesh());
   if (pmesh)
   {
      real_t global_value;
      MPI_Allreduce(&local_value, &global_value, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, pmesh->GetComm());
      return global_value;
   }
#endif
   return local_value;
}

L2ElementSetIntegral::L2ElementSetIntegral(const GridFunction &field,
                                           const EntitySets &sets,
                                           int set_index,
                                           int integration_order)
   : EntitySetFunctional(field, sets, EntitySets::ELEMENT, set_index),
     integration_order_(integration_order)
{
   MFEM_VERIFY(integration_order_ >= -1, "Invalid integration order.");

   const FiniteElementSpace *fes = Field().FESpace();
   MFEM_VERIFY(fes->IsDGSpace(),
               "L2ElementSetIntegral requires a discontinuous L2 finite "
               "element space.");
   MFEM_VERIFY(fes->GetVDim() == 1,
               "L2ElementSetIntegral currently requires a scalar field.");
   MFEM_VERIFY(fes->GetTypicalFE()->GetRangeType() == FiniteElement::SCALAR,
               "L2ElementSetIntegral requires scalar finite elements.");
}

L2ElementSetIntegral::L2ElementSetIntegral(const GridFunction &field,
                                           const EntitySets &sets,
                                           const std::string &set_name,
                                           int integration_order)
   : L2ElementSetIntegral(field, sets,
                          sets.GetSetIndex(EntitySets::ELEMENT, set_name),
                          integration_order)
{
}

real_t L2ElementSetIntegral::Eval() const
{
   VerifyCompatibleField();

   const FiniteElementSpace *fes = Field().FESpace();
   real_t local_value = 0.0;

   for (const int element : Entities())
   {
      MFEM_VERIFY(element >= 0 && element < fes->GetNE(),
                  "Entity set contains an invalid element index.");

      const FiniteElement *fe = fes->GetFE(element);
      const int order = (integration_order_ >= 0) ? integration_order_
                        : 2 * fe->GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), order);
      ElementTransformation *trans = fes->GetElementTransformation(element);

      for (int i = 0; i < ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         trans->SetIntPoint(&ip);
         local_value += ip.weight * trans->Weight() * Field().GetValue(element, ip);
      }
   }

   return GlobalSum(local_value);
}

HCurlEdgePathIntegral::HCurlEdgePathIntegral(const GridFunction &field,
                                             const EdgePath &path,
                                             int integration_order)
   : field_(field), path_(path), integration_order_(integration_order)
{
   MFEM_VERIFY(path_.GetMesh() != NULL,
               "The edge path has not been constructed on a mesh.");
   MFEM_VERIFY(integration_order_ >= -1, "Invalid integration order.");
   VerifyCompatibleField();
}

void HCurlEdgePathIntegral::VerifyCompatibleField() const
{
   VerifyVectorField(field_, *path_.GetMesh(), FiniteElement::H_CURL,
                     "HCurlEdgePathIntegral");
}

real_t HCurlEdgePathIntegral::GlobalSum(real_t local_value) const
{
   return GlobalSumImpl(field_, local_value);
}

real_t HCurlEdgePathIntegral::Eval() const
{
   VerifyCompatibleField();
   const FiniteElementSpace *fes = field_.FESpace();
   Mesh &mesh = *fes->GetMesh();
   const Array<int> &edges = path_.GetEdges();
   const Array<int> &orientations = path_.GetEdgeOrientations();
   MFEM_VERIFY(edges.Size() == orientations.Size(),
               "The edge path has inconsistent orientations.");

   const int num_edges = (mesh.Dimension() == 1) ? mesh.GetNE()
                         : mesh.GetNEdges();
   const int space_dim = mesh.SpaceDimension();
   real_t local_value = 0.0;

   for (int i = 0; i < edges.Size(); i++)
   {
      const int edge = edges[i];
      MFEM_VERIFY(edge >= 0 && edge < num_edges,
                  "The edge path contains an invalid edge index.");
      MFEM_VERIFY(orientations[i] == 1 || orientations[i] == -1,
                  "The edge path contains an invalid edge orientation.");

      const int element = GetEdgeIncidentElement(mesh, edge);
      const FiniteElement *fe = fes->GetFE(element);
      const int order = (integration_order_ >= 0) ? integration_order_
                        : 2 * fe->GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, order);
      ElementTransformation *element_trans = fes->GetElementTransformation(element);
      ElementTransformation *edge_trans = (mesh.Dimension() == 1) ?
                                          element_trans :
                                          mesh.GetEdgeTransformation(edge);
      InverseElementTransformation inverse(element_trans);
      Vector point(space_dim), value, tangent(space_dim);

      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &edge_ip = ir.IntPoint(j);
         IntegrationPoint element_ip;
         if (mesh.Dimension() == 1)
         {
            element_ip = edge_ip;
         }
         else
         {
            edge_trans->Transform(edge_ip, point);
            MFEM_VERIFY(inverse.Transform(point, element_ip) ==
                        InverseElementTransformation::Inside,
                        "Unable to map an edge quadrature point to a local element.");
         }

         field_.GetVectorValue(element, element_ip, value);
         MFEM_VERIFY(value.Size() == space_dim,
                     "HCurlEdgePathIntegral requires a physical vector field.");
         edge_trans->SetIntPoint(&edge_ip);
         const DenseMatrix &jacobian = edge_trans->Jacobian();
         for (int d = 0; d < space_dim; d++) { tangent[d] = jacobian(d, 0); }
         local_value += orientations[i] * edge_ip.weight * (value * tangent);
      }
   }

   return GlobalSum(local_value);
}

HDivFaceSetIntegral::HDivFaceSetIntegral(const GridFunction &field,
                                         const OrientedFaceSet &faces,
                                         int integration_order)
   : field_(field), faces_(faces), integration_order_(integration_order)
{
   MFEM_VERIFY(integration_order_ >= -1, "Invalid integration order.");
   VerifyCompatibleField();
}

void HDivFaceSetIntegral::VerifyCompatibleField() const
{
   MFEM_VERIFY(faces_.GetMesh() != NULL,
               "The oriented face set has no mesh.");
   VerifyVectorField(field_, *faces_.GetMesh(), FiniteElement::H_DIV,
                     "HDivFaceSetIntegral");
}

real_t HDivFaceSetIntegral::GlobalSum(real_t local_value) const
{
   return GlobalSumImpl(field_, local_value);
}

real_t HDivFaceSetIntegral::Eval() const
{
   VerifyCompatibleField();
   const FiniteElementSpace *fes = field_.FESpace();
   Mesh &mesh = *fes->GetMesh();
   const int space_dim = mesh.SpaceDimension();
   real_t local_value = 0.0;

   for (int i = 0; i < faces_.Size(); i++)
   {
      const int face = faces_.GetFace(i);
      const int sign = faces_.GetNormalSign(i);
      MFEM_VERIFY(face >= 0 && face < mesh.GetNumFaces(),
                  "The oriented face set contains an invalid face index.");
      MFEM_VERIFY(sign == 1 || sign == -1,
                  "The oriented face set contains an invalid normal sign.");

      FaceElementTransformations *face_trans =
         mesh.GetFaceElementTransformations(face);
      const int element = face_trans->Elem1No;
      const FiniteElement *fe = fes->GetFE(element);
      const int order = (integration_order_ >= 0) ? integration_order_
                        : 2 * fe->GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(mesh.GetFaceGeometry(face), order);
      Vector value, normal(space_dim);

      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &face_ip = ir.IntPoint(j);
         face_trans->SetAllIntPoints(&face_ip);
         ElementTransformation &element_trans =
            face_trans->GetElement1Transformation();
         field_.GetVectorValue(element_trans, element_trans.GetIntPoint(), value);
         MFEM_VERIFY(value.Size() == space_dim,
                     "HDivFaceSetIntegral requires a physical vector field.");
         CalcOrtho(face_trans->Face->Jacobian(), normal);
         local_value += sign * face_ip.weight * (value * normal);
      }
   }

   return GlobalSum(local_value);
}

} // namespace mfem
