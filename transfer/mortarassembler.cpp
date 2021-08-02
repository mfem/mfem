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
   : source_(source), destination_(destination) {}

bool MortarAssembler::Assemble(std::shared_ptr<SparseMatrix> &B)
{
   using namespace std;
   static const bool verbose = false;

   const auto &source_mesh = *source_->GetMesh();
   const auto &destination_mesh = *destination_->GetMesh();

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
   B = make_shared<SparseMatrix>(destination_->GetNDofs(), source_->GetNDofs());
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

      auto &source_fe = *source_->GetFE(source_index);
      auto &destination_fe = *destination_->GetFE(destination_index);

      ElementTransformation &destination_Trans =
         *destination_->GetElementTransformation(destination_index);
      const int order = source_fe.GetOrder() + destination_fe.GetOrder() +
                        destination_Trans.OrderW();

      // Update the quadrature rule in case it changed the order
      cut->SetIntegrationOrder(order);

      n_candidates++;

      if (cut->BuildQuadrature(*source_, source_index, *destination_,
                               destination_index, source_ir, destination_ir))
      {
         source_->GetElementVDofs(source_index, source_vdofs);
         destination_->GetElementVDofs(destination_index, destination_vdofs);

         ElementTransformation &source_Trans =
            *source_->GetElementTransformation(source_index);

         bool first = true;
         for (auto i_ptr : integrators_)
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
   using namespace std;
   static const bool verbose = false;

   StopWatch chrono;

   if (verbose)
   {
      mfem::out << "Assembling coupling operator..." << endl;
   }

   chrono.Start();

   shared_ptr<SparseMatrix> B = nullptr;
   if (!Assemble(B))
   {
      return false;
   }

   chrono.Stop();
   if (verbose)
   {
      mfem::out << "Done. time: ";
      mfem::out << chrono.RealTime() << " seconds" << endl;
   }

   BilinearForm b_form(destination_.get());

   bool is_vector_fe = false;
   for (auto i_ptr : integrators_)
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

   auto &D = b_form.SpMat();

   CGSolver Dinv;
   Dinv.SetOperator(D);
   Dinv.SetRelTol(1e-6);
   Dinv.SetMaxIter(20);

   Vector temp(B->Height());
   B->Mult(src_fun, temp);
   Dinv.Mult(temp, dest_fun);

   if (verbose)
   {
      Vector brs(B->Height());
      B->GetRowSums(brs);

      Vector drs(D.Height());
      D.GetRowSums(drs);

      mfem::out << "sum(B): " << brs.Sum() << std::endl;
      mfem::out << "sum(D): " << drs.Sum() << std::endl;
   }

   return true;
}

} // namespace mfem
