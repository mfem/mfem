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

#include "mortarassembler.hpp"
#include "../../general/tic_toc.hpp"

#include "cut.hpp"
#include "transferutils.hpp"

#include <cassert>

// Moonolith includes
#include "moonolith_aabb.hpp"
#include "moonolith_serial_hash_grid.hpp"
#include "moonolith_stream_utils.hpp"
#include "par_moonolith_config.hpp"

using namespace mfem::internal;

namespace mfem
{

struct MortarAssembler::Impl
{
public:
   std::shared_ptr<FiniteElementSpace> source;
   std::shared_ptr<FiniteElementSpace> destination;
   std::vector<std::shared_ptr<MortarIntegrator>> integrators;
   std::shared_ptr<SparseMatrix> coupling_matrix;
   std::shared_ptr<SparseMatrix> mass_matrix;
   bool verbose{false};
   bool assemble_mass_and_coupling_together{true};
   int max_solver_iterations{400};

   bool is_vector_fe() const
   {
      bool is_vector_fe = false;
      for (auto i_ptr : integrators)
      {
         if (i_ptr->is_vector_fe())
         {
            is_vector_fe = true;
            break;
         }
      }

      return is_vector_fe;
   }

   BilinearFormIntegrator * new_mass_integrator() const
   {
      if (is_vector_fe())
      {
         return new VectorFEMassIntegrator();
      }
      else
      {
         return new MassIntegrator();
      }
   }
};

MortarAssembler::~MortarAssembler() = default;

void MortarAssembler::SetAssembleMassAndCouplingTogether(const bool value)
{
   impl_->assemble_mass_and_coupling_together = value;
}

void MortarAssembler::SetMaxSolverIterations(const int max_solver_iterations)
{
   impl_->max_solver_iterations = max_solver_iterations;
}

void MortarAssembler::AddMortarIntegrator(
   const std::shared_ptr<MortarIntegrator> &integrator)
{
   impl_->integrators.push_back(integrator);
}

void MortarAssembler::SetVerbose(const bool verbose)
{
   impl_->verbose = verbose;
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

int order_multiplier(const Geometry::Type type, const int dim)
{
   return
   (type == Geometry::TRIANGLE || type == Geometry::TETRAHEDRON ||
      type == Geometry::SEGMENT)? 1 : dim;
}

bool MortarAssembler::Assemble(std::shared_ptr<SparseMatrix> &B)
{
   using namespace std;
   const bool verbose = impl_->verbose;

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


   IntegrationRule source_ir;
   IntegrationRule destination_ir;

   int skip_zeros = 1;
   B = make_shared<SparseMatrix>(impl_->destination->GetNDofs(),
                                 impl_->source->GetNDofs());


   std::unique_ptr<BilinearFormIntegrator> mass_integr(impl_->new_mass_integrator());

   if(impl_->assemble_mass_and_coupling_together) {
      impl_->mass_matrix = make_shared<SparseMatrix>(impl_->destination->GetNDofs(), impl_->destination->GetNDofs());
   }



   Array<int> source_vdofs, destination_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;
   double local_element_matrices_sum = 0.0;

   long n_intersections = 0;
   long n_candidates = 0;

   int max_q_order = 0;

   for (auto i_ptr : impl_->integrators)
   {
      max_q_order = std::max(i_ptr->GetQuadratureOrder(), max_q_order);
   }

   bool intersected = false;
   for (auto it = begin(pairs); it != end(pairs); /* inside */)
   {
      const int source_index = *it++;
      const int destination_index = *it++;

      auto &source_fe = *impl_->source->GetFE(source_index);
      auto &destination_fe = *impl_->destination->GetFE(destination_index);

      ElementTransformation &destination_Trans =
         *impl_->destination->GetElementTransformation(destination_index);

      // Quadrature order mangling
      int src_order_mult = order_multiplier(source_fe.GetGeomType(), dim);
      int dest_order_mult = order_multiplier(destination_fe.GetGeomType(), dim);

      const int src_order = src_order_mult * source_fe.GetOrder();
      const int dest_order = dest_order_mult * destination_fe.GetOrder();

      int contraction_order = src_order + dest_order;

     if(impl_->assemble_mass_and_coupling_together) {
        contraction_order = std::max(contraction_order, 2 * dest_order);
     }

     const int order = contraction_order + dest_order_mult * destination_Trans.OrderW() + max_q_order;

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

         if(impl_->assemble_mass_and_coupling_together) {
            mass_integr->SetIntRule(&destination_ir);
            mass_integr->AssembleElementMatrix(destination_fe, destination_Trans, elemmat);
            impl_->mass_matrix->AddSubMatrix(destination_vdofs, destination_vdofs, elemmat, skip_zeros);
         }

         intersected = true;
         ++n_intersections;
      }
   }

   if (!intersected)
   {
      return false;
   }

   B->Finalize();

   if(impl_->assemble_mass_and_coupling_together) {
      impl_->mass_matrix->Finalize();
   }

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

bool MortarAssembler::Transfer(const GridFunction &src_fun,
                               GridFunction &dest_fun)
{
   return Update() && Apply(src_fun, dest_fun);
}

bool MortarAssembler::Apply(const GridFunction &src_fun,
                            GridFunction &dest_fun)
{
   if (!impl_->coupling_matrix)
   {
      if (!Update())
      {
         return false;
      }
   }

   Vector temp(impl_->coupling_matrix->Height());
   impl_->coupling_matrix->Mult(src_fun, temp);

   CGSolver Dinv;
   Dinv.SetMaxIter(impl_->max_solver_iterations);

   if(impl_->verbose) {
      Dinv.SetPrintLevel(3);
   }

   Dinv.SetOperator(*impl_->mass_matrix);
   Dinv.SetRelTol(1e-6);
   Dinv.SetMaxIter(80);
   Dinv.Mult(temp, dest_fun);
   return true;
}

bool MortarAssembler::Update()
{
   using namespace std;
   const bool verbose = impl_->verbose;

   StopWatch chrono;

   if (verbose)
   {
      mfem::out << "\nAssembling coupling operator..." << endl;
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

   if(!impl_->assemble_mass_and_coupling_together) {
      BilinearForm b_form(impl_->destination.get());

      b_form.AddDomainIntegrator(impl_->new_mass_integrator());

      b_form.Assemble();
      b_form.Finalize();

      impl_->mass_matrix = std::shared_ptr<SparseMatrix>(b_form.LoseMat());
   }

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
