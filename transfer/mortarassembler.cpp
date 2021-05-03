#include "mortarassembler.hpp"
#include "mortarassemble.hpp"
#include "../general/tic_toc.hpp"

#include "transferutils.hpp"

#include <cassert>

// Moonolith includes
#include "par_moonolith_config.hpp"
#include "moonolith_aabb.hpp"
#include "moonolith_stream_utils.hpp"
#include "moonolith_serial_hash_grid.hpp"


namespace mfem
{

template<int Dim>
void BuildBoxes(const Mesh &mesh, std::vector<::moonolith::AABB<Dim, double>> &element_boxes)
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

   switch(dim) {
      case 1:
      {
         std::vector<::moonolith::AABB<1, double>> src_boxes, dest_boxes;
         BuildBoxes(src,  src_boxes);
         BuildBoxes(dest, dest_boxes);

         ::moonolith::SerialHashGrid<1, double> grid;
         return grid.detect(src_boxes, dest_boxes, pairs);
      }
      case 2:
      {
         std::vector<::moonolith::AABB<2, double>> src_boxes, dest_boxes;
         BuildBoxes(src,  src_boxes);
         BuildBoxes(dest, dest_boxes);

         ::moonolith::SerialHashGrid<2, double> grid;
         return grid.detect(src_boxes, dest_boxes, pairs);
      }
      case 3:
      {
         std::vector<::moonolith::AABB<3, double>> src_boxes, dest_boxes;
         BuildBoxes(src,  src_boxes);
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
   const std::shared_ptr<FiniteElementSpace> &master,
   const std::shared_ptr<FiniteElementSpace> &slave)
   : master_(master), slave_(slave)
{ }

bool MortarAssembler::Assemble(std::shared_ptr<SparseMatrix> &B)
{
   using namespace std;
   static const bool verbose = false;

   const auto &master_mesh = *master_->GetMesh();
   const auto &slave_mesh  = *slave_->GetMesh();

   int dim = master_mesh.Dimension();

   std::vector<::moonolith::Integer> pairs;
   if (!HashGridDetectIntersections(master_mesh, slave_mesh, pairs))
   {
      return false;
   }

   DenseMatrix master_pts;
   DenseMatrix slave_pts;
   DenseMatrix intersection2;
   Polyhedron  intersection3;
   Intersector isector;

   //////////////////////////////////////////////////
   int skip_zeros = 1;
   B = make_shared<SparseMatrix>(slave_->GetNDofs(), master_->GetNDofs());
   Array<int> master_vdofs, slave_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;

   IntegrationRule composite_ir;
   IntegrationRule master_ir;
   IntegrationRule slave_ir;
   //////////////////////////////////////////////////

   double total_intersection_volume = 0.0;
   double local_element_matrices_sum = 0.0;

   Array<double> volumes(slave_mesh.GetNE());
   for (int i = 0; i < slave_->GetNE(); ++i)
   {
      if (dim == 2)
      {
         slave_mesh.GetPointMatrix(i, slave_pts);
         volumes[i] = isector.polygon_area_2(slave_pts.Width(), slave_pts.Data());
      }
      else if (dim == 3)
      {
         MakePolyhedron(slave_mesh, i, intersection3);
         volumes[i] = isector.p_mesh_volume_3(intersection3);
      }
      else
      {
         assert(false);
      }
   }

   bool intersected = false;
   for (auto it = begin(pairs); it != end(pairs);    )
   {
      const int master_index = *it++;
      const int slave_index  = *it++;

      auto &master_fe = *master_->GetFE(master_index);
      auto &slave_fe  = *slave_->GetFE(slave_index);

      bool pair_intersected = false;
      if (dim == 2)
      {
         master_mesh.GetPointMatrix(master_index, master_pts);
         slave_mesh.GetPointMatrix(slave_index,     slave_pts);

         if (Intersect2D(master_pts, slave_pts, intersection2))
         {
            total_intersection_volume += fabs(isector.polygon_area_2(intersection2.Width(),
                                                                     intersection2.Data()));
            double weight = volumes[slave_index];

            ElementTransformation &Trans = *slave_->GetElementTransformation(slave_index);
            const int order = master_fe.GetOrder() + slave_fe.GetOrder() + Trans.OrderW();

            MakeCompositeQuadrature2D(intersection2, weight, order, composite_ir);
            pair_intersected = true;
         }

      }
      else if (dim == 3)
      {
         if (Intersect3D(master_mesh, master_index, slave_mesh, slave_index,
                         intersection3))
         {
            total_intersection_volume += isector.p_mesh_volume_3(intersection3);
            double weight = volumes[slave_index];

            ElementTransformation &Trans = *slave_->GetElementTransformation(slave_index);
            const int order = master_fe.GetOrder() + slave_fe.GetOrder() + Trans.OrderW();

            MakeCompositeQuadrature3D(intersection3, weight, order, composite_ir);
            pair_intersected = true;
         }

      }
      else
      {
         assert(false);
         return false;
      }

      if (pair_intersected)
      {
         //make reference quaratures
         ElementTransformation &master_Trans = *master_->GetElementTransformation(
                                                  master_index);
         ElementTransformation &slave_Trans = *slave_->GetElementTransformation(
                                                 slave_index);

         TransformToReference(master_Trans, master_fe.GetGeomType(), composite_ir,
                              master_ir);
         TransformToReference(slave_Trans,  slave_fe.GetGeomType(),  composite_ir,
                              slave_ir);

         master_->GetElementVDofs(master_index, master_vdofs);
         slave_->GetElementVDofs (slave_index,  slave_vdofs);


         bool first = true;
         for (auto i_ptr : integrators_)
         {
            if (first)
            {
               i_ptr->AssembleElementMatrix(master_fe, master_ir, master_Trans, slave_fe,
                                            slave_ir, slave_Trans, cumulative_elemmat);
               first = false;
            }
            else
            {
               i_ptr->AssembleElementMatrix(master_fe, master_ir, master_Trans, slave_fe,
                                            slave_ir, slave_Trans, elemmat);
               cumulative_elemmat += elemmat;
            }
         }

         local_element_matrices_sum += Sum(cumulative_elemmat);


         B->AddSubMatrix(slave_vdofs, master_vdofs, cumulative_elemmat, skip_zeros);
         intersected = true;
      }
   }

   if (!intersected) { return false; }

   B->Finalize();

   if (verbose)
   {
      mfem::out <<  "local_element_matrices_sum: " << local_element_matrices_sum <<
                std::endl;
      mfem::out <<  "intersection volume: " << total_intersection_volume << std::endl;
      mfem::out <<  "B in R^(" << B->Height() <<  " x " << B->Width() << ")" <<
                std::endl;
   }

   return true;

}

bool MortarAssembler::Transfer(GridFunction &src_fun, GridFunction &dest_fun,
                               bool is_vector_fe)
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

   BilinearForm b_form(slave_.get());
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

      mfem::out <<  "sum(B): " << brs.Sum() << std::endl;
      mfem::out <<  "sum(D): " << drs.Sum() << std::endl;
   }

   return true;

}

}
