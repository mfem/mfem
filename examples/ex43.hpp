//                  MFEM Example 43 - Shared Code

#ifndef MFEM_EX43_HPP
#define MFEM_EX43_HPP

#include "mfem.hpp"

namespace mfem
{

class VertexPatchSmoother : public Solver
{
private:
   Array<Array<int> *> patch_dofs;
   Array<DenseMatrix *> patch_inverses;
   mutable Vector patch_rhs, patch_solution;

public:
   VertexPatchSmoother(const SparseMatrix &op, FiniteElementSpace &fespace)
      : Solver(op.Height())
   {
      Mesh *mesh = fespace.GetMesh();
      Table *vertex_to_element = mesh->GetVertexToElementTable();
      Array<int> marker(fespace.GetVSize());
      marker = -1;
      Array<int> dofs_on_entity, element_edges, edge_orientations;
      Array<int> edge_vertices;
      for (int vertex = 0; vertex < mesh->GetNV(); vertex++)
      {
         Array<int> *dofs = new Array<int>;
         const int *incident_elements = vertex_to_element->GetRow(vertex);
         const int num_incident_elements = vertex_to_element->RowSize(vertex);
         for (int i = 0; i < num_incident_elements; i++)
         {
            const int element = incident_elements[i];
            fespace.GetElementInteriorDofs(element, dofs_on_entity);
            for (int j = 0; j < dofs_on_entity.Size(); j++)
            {
               const int dof = UnsignIndex(dofs_on_entity[j]);
               if (marker[dof] != vertex)
               {
                  marker[dof] = vertex;
                  dofs->Append(dof);
               }
            }
            mesh->GetElementEdges(element, element_edges, edge_orientations);
            for (int j = 0; j < element_edges.Size(); j++)
            {
               const int edge = element_edges[j];
               mesh->GetEdgeVertices(edge, edge_vertices);
               if (edge_vertices.Find(vertex) < 0) { continue; }
               fespace.GetEdgeDofs(edge, dofs_on_entity);
               for (int k = 0; k < dofs_on_entity.Size(); k++)
               {
                  const int dof = UnsignIndex(dofs_on_entity[k]);
                  if (marker[dof] != vertex)
                  {
                     marker[dof] = vertex;
                     dofs->Append(dof);
                  }
               }
            }
         }
         DenseMatrix patch_matrix(dofs->Size());
         op.GetSubMatrix(*dofs, *dofs, patch_matrix);
         DenseMatrix *patch_inverse = new DenseMatrix;
         DenseMatrixInverse inverse(patch_matrix, true);
         inverse.GetInverseMatrix(*patch_inverse);
         patch_dofs.Append(dofs);
         patch_inverses.Append(patch_inverse);
      }
      delete vertex_to_element;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      for (int patch = 0; patch < patch_dofs.Size(); patch++)
      {
         const Array<int> &dofs = *patch_dofs[patch];
         x.GetSubVector(dofs, patch_rhs);
         patch_solution.SetSize(dofs.Size());
         patch_inverses[patch]->Mult(patch_rhs, patch_solution);
         y.AddElementVector(dofs, patch_solution);
      }
      y *= 0.33;
   }

   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void SetOperator(const Operator &) override
   { MFEM_ABORT("VertexPatchSmoother does not support SetOperator"); }
   ~VertexPatchSmoother() override
   {
      for (int patch = 0; patch < patch_dofs.Size(); patch++)
      {
         delete patch_dofs[patch];
         delete patch_inverses[patch];
      }
   }
};

} // namespace mfem

#endif // MFEM_EX43_HPP
