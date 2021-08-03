// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mortarassembler.hpp"
#include "../general/tic_toc.hpp"

#include "cut.hpp"
#include "transferutils.hpp"

#include <cassert>

// Moonolith includes
#include "moonolith_aabb.hpp"
#include "moonolith_serial_hash_grid.hpp"
#include "moonolith_stream_utils.hpp"
#include "par_moonolith_config.hpp"

using namespace mfem::private_;

namespace mfem
{

class MortarAssembler::Impl
{
public:
   std::shared_ptr<FiniteElementSpace> source;
   std::shared_ptr<FiniteElementSpace> destination;
   std::vector<std::shared_ptr<MortarIntegrator>> integrators;
   std::shared_ptr<SparseMatrix> coupling_matrix;
   std::shared_ptr<SparseMatrix> mass_matrix;
};

MortarAssembler::~MortarAssembler() = default;

void MortarAssembler::AddMortarIntegrator(
   const std::shared_ptr<MortarIntegrator> &integrator)
{
   impl_->integrators.push_back(integrator);
}

template <int Dim>
void BuildBoxes(const Mesh &mesh,
                std::vector<::moonolith::AABB<Dim, double>> &element_boxes)
{
#ifndef NDEBUG
   const int dim = mesh.Dimension();
   assert(dim == Dim);
#endif
   element_boxes.resize(mesh.GetNE());

   DenseMatrix pts;
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      mesh.GetPointMatrix(i, pts);
      MinCol(pts, &element_boxes[i].min_[0], false);
      MaxCol(pts, &element_boxes[i].max_[0], false);
   }
}

bool HashGridDetectIntersections(const Mesh &src, const Mesh &dest,
                                 std::vector<moonolith::Integer> &pairs)
{
   const int dim = dest.Dimension();

   switch (dim)
   {
      case 1:
      {
         std::vector<::moonolith::AABB<1, double>> src_boxes, dest_boxes;
         BuildBoxes(src, src_boxes);
         BuildBoxes(dest, dest_boxes);

         ::moonolith::SerialHashGrid<1, double> grid;
         return grid.detect(src_boxes, dest_boxes, pairs);
      }
      case 2:
      {
         std::vector<::moonolith::AABB<2, double>> src_boxes, dest_boxes;
         BuildBoxes(src, src_boxes);
         BuildBoxes(dest, dest_boxes);

         ::moonolith::SerialHashGrid<2, double> grid;
         return grid.detect(src_boxes, dest_boxes, pairs);
      }
      case 3:
      {
         std::vector<::moonolith::AABB<3, double>> src_boxes, dest_boxes;
         BuildBoxes(src, src_boxes);
         BuildBoxes(dest, dest_boxes);

         ::moonolith::SerialHashGrid<3, double> grid;
         return grid.detect(src_boxes, dest_boxes, pairs);
      }
      default:
      {
         assert(false);
         return false;
      }
   }
}

MortarAssembler::MortarAssembler(
   const std::shared_ptr<FiniteElementSpace> &source,
   const std::shared_ptr<FiniteElementSpace> &destination)
   : impl_(new Impl())
{
   impl_->source = source;
   impl_->destination = destination;
}

bool MortarAssembler::Assemble(std::shared_ptr<SparseMatrix> &B)
{
   using namespace std;
   static const bool verbose = false;

   const auto &source_mesh = *impl_->source->GetMesh();
   const auto &destination_mesh = *impl_->destination->GetMesh();

   int dim = source_mesh.Dimension();

   std::vector<::moonolith::Integer> pairs;
   if (!HashGridDetectIntersections(source_mesh, destination_mesh, pairs))
   {
      return false;
   }

   std::shared_ptr<Cut> cut = NewCut(dim);
   if (!cut)
   {
      assert(false && "NOT Supported!");
      return false;
   }

   //////////////////////////////////////////////////
   IntegrationRule source_ir;
   IntegrationRule destination_ir;
   //////////////////////////////////////////////////
   int skip_zeros = 1;
   B = make_shared<SparseMatrix>(impl_->destination->GetNDofs(),
                                 impl_->source->GetNDofs());
   Array<int> source_vdofs, destination_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;
   //////////////////////////////////////////////////
   double local_element_matrices_sum = 0.0;

   long n_intersections = 0;
   long n_candidates = 0;

   bool intersected = false;
   for (auto it = begin(pairs); it != end(pairs); /*inside*/)
   {
      const int source_index = *it++;
      const int destination_index = *it++;

      auto &source_fe = *impl_->source->GetFE(source_index);
      auto &destination_fe = *impl_->destination->GetFE(destination_index);

      ElementTransformation &destination_Trans =
         *impl_->destination->GetElementTransformation(destination_index);
      const int order = source_fe.GetOrder() + destination_fe.GetOrder() +
                        destination_Trans.OrderW();

      // Update the quadrature rule in case it changed the order
      cut->SetIntegrationOrder(order);

      n_candidates++;

      if (cut->BuildQuadrature(*impl_->source, source_index, *impl_->destination,
                               destination_index, source_ir, destination_ir))
      {
         impl_->source->GetElementVDofs(source_index, source_vdofs);
         impl_->destination->GetElementVDofs(destination_index, destination_vdofs);

         ElementTransformation &source_Trans =
            *impl_->source->GetElementTransformation(source_index);

         bool first = true;
         for (auto i_ptr : impl_->integrators)
         {
            if (first)
            {
               i_ptr->AssembleElementMatrix(source_fe, source_ir, source_Trans,
                                            destination_fe, destination_ir,
                                            destination_Trans, cumulative_elemmat);
               first = false;
            }
            else
            {
               i_ptr->AssembleElementMatrix(source_fe, source_ir, source_Trans,
                                            destination_fe, destination_ir,
                                            destination_Trans, elemmat);
               cumulative_elemmat += elemmat;
            }
         }

         local_element_matrices_sum += Sum(cumulative_elemmat);

         B->AddSubMatrix(destination_vdofs, source_vdofs, cumulative_elemmat,
                         skip_zeros);
         intersected = true;
         ++n_intersections;
      }
   }

   if (!intersected)
   {
      return false;
   }

   B->Finalize();

   if (verbose)
   {
      mfem::out << "local_element_matrices_sum: " << local_element_matrices_sum
                << std::endl;
      mfem::out << "B in R^(" << B->Height() << " x " << B->Width() << ")"
                << std::endl;

      mfem::out << "n_intersections: " << n_intersections
                << ", n_candidates: " << n_candidates << '\n';

      cut->Describe();
   }

   return true;
}

bool MortarAssembler::Transfer(GridFunction &src_fun, GridFunction &dest_fun)
{
   return Init() && Apply(src_fun, dest_fun);
}

bool MortarAssembler::Apply(GridFunction &src_fun, GridFunction &dest_fun)
{
   if (!impl_->coupling_matrix)
   {
      mfem::err << "Warning calling apply without calling Init() first!\n";
      if (!Init())
      {
         return false;
      }
   }

   CGSolver Dinv;
   Dinv.SetOperator(*impl_->mass_matrix);
   Dinv.SetRelTol(1e-6);
   Dinv.SetMaxIter(20);

   Vector temp(impl_->coupling_matrix->Height());
   impl_->coupling_matrix->Mult(src_fun, temp);
   Dinv.Mult(temp, dest_fun);
   return true;
}

bool MortarAssembler::Init()
{
   using namespace std;
   static const bool verbose = false;

   StopWatch chrono;

   if (verbose)
   {
      mfem::out << "Assembling coupling operator..." << endl;
   }

   chrono.Start();

   if (!Assemble(impl_->coupling_matrix))
   {
      return false;
   }

   chrono.Stop();
   if (verbose)
   {
      mfem::out << "Done. time: ";
      mfem::out << chrono.RealTime() << " seconds" << endl;
   }

   BilinearForm b_form(impl_->destination.get());

   bool is_vector_fe = false;
   for (auto i_ptr : impl_->integrators)
   {
      if (i_ptr->is_vector_fe())
      {
         is_vector_fe = true;
         break;
      }
   }

   if (is_vector_fe)
   {
      b_form.AddDomainIntegrator(new VectorFEMassIntegrator());
   }
   else
   {
      b_form.AddDomainIntegrator(new MassIntegrator());
   }

   b_form.Assemble();
   b_form.Finalize();

   impl_->mass_matrix = std::shared_ptr<SparseMatrix>(b_form.LoseMat());

   if (verbose)
   {
      Vector brs(impl_->coupling_matrix->Height());
      impl_->coupling_matrix->GetRowSums(brs);

      Vector drs(impl_->mass_matrix->Height());
      impl_->mass_matrix->GetRowSums(drs);

      mfem::out << "sum(B): " << brs.Sum() << std::endl;
      mfem::out << "sum(D): " << drs.Sum() << std::endl;
   }

   return true;
}

} // namespace mfem
