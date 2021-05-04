#include "mortarassembler.hpp"
#include "../general/tic_toc.hpp"

#include "transferutils.hpp"
#include "cut.hpp"

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

   std::shared_ptr<CutBase> cut;

   if(dim == 2) {
      cut = std::make_shared<Cut2D>();
   } else if(dim == 3) {
      cut = std::make_shared<Cut3D>();
   } else {
      assert(false && "NOT Supported!");
      return false;
   }

   //////////////////////////////////////////////////
   IntegrationRule master_ir;
   IntegrationRule slave_ir;
   //////////////////////////////////////////////////
   int skip_zeros = 1;
   B = make_shared<SparseMatrix>(slave_->GetNDofs(), master_->GetNDofs());
   Array<int> master_vdofs, slave_vdofs;
   DenseMatrix elemmat;
   DenseMatrix cumulative_elemmat;
   //////////////////////////////////////////////////
   double local_element_matrices_sum = 0.0;


   long n_intersections = 0;
   long n_candidates = 0;

   bool intersected = false;
   for (auto it = begin(pairs); it != end(pairs); /*inside*/)
   {
      const int master_index = *it++;
      const int slave_index  = *it++;

      auto &master_fe = *master_->GetFE(master_index);
      auto &slave_fe  = *slave_->GetFE(slave_index);

      ElementTransformation &slave_Trans = *slave_->GetElementTransformation(slave_index);
      const int order = master_fe.GetOrder() + slave_fe.GetOrder() + slave_Trans.OrderW();

      // Update the quadrature rule in case it changed the order
      cut->SetIntegrationOrder(order);

      n_candidates++;

      if (cut->BuildQuadrature(*master_, master_index, *slave_, slave_index,  master_ir, slave_ir))
      {
         master_->GetElementVDofs(master_index, master_vdofs);
         slave_->GetElementVDofs (slave_index,  slave_vdofs);

         ElementTransformation &master_Trans = *master_->GetElementTransformation(
                                                  master_index);

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
         ++n_intersections;
      }
   }

   if (!intersected) { return false; }

   B->Finalize();

   if (verbose)
   {
      mfem::out <<  "local_element_matrices_sum: " << local_element_matrices_sum <<
                std::endl;
      mfem::out <<  "B in R^(" << B->Height() <<  " x " << B->Width() << ")" <<
                std::endl;

      mfem::out << "n_intersections: " << n_intersections << ", n_candidates: " << n_candidates << '\n';

      cut->describe();
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
