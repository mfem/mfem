// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//            --------------------------------------------------
//            Mesh Optimizer Miniapp: Optimize high-order meshes
//            --------------------------------------------------
//
// This miniapp performs mesh optimization using the Target-Matrix Optimization
// Paradigm (TMOP) by P.Knupp et al., and a global variational minimization
// approach. It minimizes the quantity sum_T int_T mu(J(x)), where T are the
// target (ideal) elements, J is the Jacobian of the transformation from the
// target to the physical element, and mu is the mesh quality metric. This
// metric can measure shape, size or alignment of the region around each
// quadrature point. The combination of targets & quality metrics is used to
// optimize the physical node positions, i.e., they must be as close as possible
// to the shape / size / alignment of their targets. This code also demonstrates
// a possible use of nonlinear operators (the class TMOP_QualityMetric, defining
// mu(J), and the class TMOP_Integrator, defining int mu(J)), as well as their
// coupling to Newton methods for solving minimization problems. Note that the
// utilized Newton methods are oriented towards avoiding invalid meshes with
// negative Jacobian determinants. Each Newton step requires the inversion of a
// Jacobian matrix, which is done through an inner linear solver.
//
// Compile with: make mesh-fitting
//
//  Adaptive surface fitting:
//    mesh-fitting -m square01.mesh -o 3 -rs 1 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 5e4 -rtol 1e-5
//    mesh-fitting -m square01-tri.mesh -o 3 -rs 0 -mid 58 -tid 1 -ni 200 -vl 1 -sfc 1e4 -rtol 1e-5
//  Surface fitting with weight adaptation and termination based on fitting error
//    mesh-fitting -m square01.mesh -o 2 -rs 1 -mid 2 -tid 1 -ni 100 -vl 2 -sfc 10 -rtol 1e-20 -st 0 -sfa -sft 1e-5
//
//   Blade shape:
//     mesh-fitting -m blade.mesh -o 4 -mid 2 -tid 1 -ni 30 -ls 3 -art 1 -bnd -qt 1 -qo 8
//
//    New sample runs for p-refinement
//    Take a mesh, add some noise to it, and optimize it
//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -rtol 1e-5 -ji 0.1
//    Randomly p-refine a mesh, add noise to the mesh nodes and optimize it:
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -rtol 1e-5 -ji 0.1 -pref
//    Surface fitting to a circular level-set - no-prefinement right now
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1 -sfa
//    Surface fitting to a circular level-set - with p-refinement
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 1
//    Surface fitting to a circular level-set - with p-refinement by increasing of 2 the element order around the interface
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 2
//    Surface fitting to a circular level-set with p-refinement on a triangular mesh
//     make mesh-fitting -j && ./mesh-fitting -m square01_tri.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1  -pref -sfa -oi 1
//    Surface fitting to a spherical level-set - with p-refinement on a hex mesh
//     make mesh-fitting -j && ./mesh-fitting -m cube.mesh -o 1 -rs 1 -mid 303 -tid 1 -ni 20 -vl 1 -sfc 10 -rtol 1e-5 -ae 1 -sfa -pref -oi 1
//    Surface fitting to a circular level-set - with p-refinement by increasing of 1 the element order around the interface and using a background mesh
//     make mesh-fitting -j && ./mesh-fitting -m square01.mesh -o 1 -rs 1 -mid 2 -tid 1 -ni 50 -vl 1 -sfc 1 -rtol 1e-5 -ae 1 -sfa -pref -oi 1 -sbgmesh

//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -rs 3 -o 1 -oi 1

//    Surface fitting to a squircle level-set - with p-refinement around the interface and using a background mesh
//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -rs 2 -o 1 -oi 1 -sbgmesh -vl 0 -mo 5
//    Surface fitting to a circle level-set 3D function - with p-refinement around the interface and using a background mesh
//    make mesh-fitting -j && ./mesh-fitting -m cube.mesh -rs 2 -o 1 -oi 1 -sbgmesh -vl 0 -mo 3 -mid 303 -preft 1e-8

//    Surface fitting to a squircle level-set - with p-refinement around the interface and order reduction after the fitting step
//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -rs 2 -o 1 -oi 4 -sbgmesh -vl 0 -mo 5

//    squircle 2D:
//    make mesh-fitting -j && ./mesh-fitting -m square01.mesh -rs 2 -o 1 -oi 1 -sbgmesh -vl 0 -mo 5 -preft 1e-11 -lsf 1

//    cercle 3D:
//    make mesh-fitting -j && ./mesh-fitting -m cube.mesh -rs 3 -o 1 -oi 1 -sbgmesh -vl 0 -mo 3 -mid 303 -preft 1e-8
//    squircle 3D:
//    make mesh-fitting -j && ./mesh-fitting -m cube.mesh -rs 2 -o 1 -oi 1 -sbgmesh -vl 0 -mo 3 -mid 303 -preft 1e-11 -lsf 1

//    apollo 2D:
//    make mesh-fitting -j && ./mesh-fitting -m apollo_input_mesh_tri.mesh -rs 1 -o 1 -oi 3 -sbgmesh -vl 2 -mo 4 -mid 2 -no-cus-mat -ni 100 -marking -tid 4 -preft 1e-8 -sft 1e-10 -sfa 2.0 -sfc 1.0 -bgamr 2 -fix-bnd -bgm apollo_amr.mesh -bgls apollo_dist.gf -li 1000 -lsf 2

#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include "mesh-fitting.hpp"

using namespace mfem;
using namespace std;

Array<Mesh *> SetupSurfaceMeshes()
{
   Array<Mesh *> surf_meshes(3);

   // Add a line surface mesh for 2D
   {
      int dim = 1;
      int spaceDim = 2;
      int nvert = 2;
      int nelem = 1;
      surf_meshes[0] = new Mesh(dim, nvert, 1, 0, spaceDim);

      const double c_v[2][2] =
      {
         {0, 1}, {2.0, 3.0}
      };
      const int i_v[1][2] =
      {
         {0, 1}
      };

      for (int j = 0; j < nvert; j++)
      {
         surf_meshes[0]->AddVertex(c_v[j]);
      }
      for (int j = 0; j < nelem; j++)
      {
         int attribute = j + 1;
         surf_meshes[0]->AddSegment(i_v[j], attribute);
      }
      surf_meshes[0]->FinalizeMesh(0, true, false);
   }

   // Add a triangle surface mesh for 3D
   {
      int dim = 2;
      int spaceDim = 3;
      int nvert = 3;
      int nelem = 1;
      surf_meshes[1] = new Mesh(dim, nvert, 1, 0, spaceDim);

      const double c_v[4][3] =
      {
         {0, 1, 2}, {3, 4, 5}, {6, 7, 8}
      };
      const int i_v[1][4] =
      {
         {0, 1, 2}
      };

      for (int j = 0; j < nvert; j++)
      {
         surf_meshes[1]->AddVertex(c_v[j]);
      }
      for (int j = 0; j < nelem; j++)
      {
         int attribute = j + 1;
         surf_meshes[1]->AddTriangle(i_v[j], attribute);
      }
      surf_meshes[1]->FinalizeTriMesh(1, 1, true);
   }
   // Add a quad surface mesh for 3D
   {
      int dim = 2;
      int spaceDim = 3;
      int nvert = 4;
      int nelem = 1;
      surf_meshes[2] = new Mesh(dim, nvert, 1, 0, spaceDim);

      const double c_v[4][3] =
      {
         {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}
      };
      const int i_v[2][4] =
      {
         {0, 1, 2, 3}
      };

      for (int j = 0; j < nvert; j++)
      {
         surf_meshes[2]->AddVertex(c_v[j]);
      }
      for (int j = 0; j < nelem; j++)
      {
         int attribute = j + 1;
         surf_meshes[2]->AddQuad(i_v[j], attribute);
      }
      surf_meshes[2]->FinalizeQuadMesh(1, 1, true);
   }

   // std::cout << surf_meshes[0]->Dimension() << " " <<
   // surf_meshes[0]->SpaceDimension() << " k10dimspacedim\n";
   // surf_meshes[2]->SetCurvature(2);
   // surf_meshes[0]->SetCurvature(2);
   // MFEM_ABORT(" ");

   return surf_meshes;
}

class PRefinementTransfer
{
private:
   FiniteElementSpace *src;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
       The underlying finite elements need to implement the GetTransferMatrix
       methods. */
   PRefinementTransfer(const FiniteElementSpace& src_);

   /// Destructor
   ~PRefinementTransfer();

   /// Update source FiniteElementSpace used to construct the
   /// PRefinementTransfer operator.
   void SetSourceFESpace(const FiniteElementSpace& src_);

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Transfer(GridFunction &targf);
};

PRefinementTransfer::PRefinementTransfer(const FiniteElementSpace &src_)
{
   src = new FiniteElementSpace(src_);
}

PRefinementTransfer::~PRefinementTransfer()
{
   delete src;
}

void PRefinementTransfer::SetSourceFESpace(const FiniteElementSpace &src_)
{
   if (src) { delete src; }
   src = new FiniteElementSpace(src_);
}

void PRefinementTransfer::Transfer(GridFunction &targf)
{
   MFEM_VERIFY(targf.GetSequence() != targf.FESpace()->GetSequence(),
               "GridFunction should not be updated prior to UpdateGF.");
   //MFEM_VERIFY(targf.GetSequence() == src->GetSequence(),
   //            ".");
   Vector srcgf = targf;
   targf.Update();
   PRefinementTransferOperator preft =
      PRefinementTransferOperator(*src, *(targf.FESpace()));
   preft.Mult(srcgf, targf);
}

GridFunction* ProlongToMaxOrder(const GridFunction *x, const int fieldtype)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();
   const int vdim = fespace->GetVDim();

   // find the max order in the space
   int max_order = fespace->GetMaxElementOrder();

   // create a visualization space of max order for all elements
   FiniteElementCollection *fecInt = NULL;
   if (fieldtype == 0)
   {
      fecInt = new H1_FECollection(max_order, mesh->Dimension());
   }
   else if (fieldtype == 1)
   {
      fecInt = new L2_FECollection(max_order, mesh->Dimension());
   }
   FiniteElementSpace *spaceInt = new FiniteElementSpace(mesh, fecInt,
                                                         fespace->GetVDim(),
                                                         fespace->GetOrdering());

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *xInt = new GridFunction(spaceInt);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementVDofs(i, dofs);
      Vector elemvect(0), vectInt(0);
      x->GetSubVector(dofs, elemvect);
      DenseMatrix elemvecMat(elemvect.GetData(), dofs.Size()/vdim, vdim);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *feInt = fecInt->GetFE(geom, max_order);

      feInt->GetTransferMatrix(*fe, T, I);

      spaceInt->GetElementVDofs(i, dofs);
      vectInt.SetSize(dofs.Size());
      DenseMatrix vectIntMat(vectInt.GetData(), dofs.Size()/vdim, vdim);

      //      I.Mult(elemvecMat, vectIntMat);
      Mult(I, elemvecMat, vectIntMat);
      xInt->SetSubVector(dofs, vectInt);
   }

   xInt->MakeOwner(fecInt);
   return xInt;
}

double ComputeIntegrateError(const FiniteElementSpace* fes, GridFunction* lss,
                             const int el)
{
   double error = 0.0;
   const FiniteElement *fe = fes->GetFaceElement(el);  // Face el
   int intorder = 2*fe->GetOrder() + 3;
   const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));

   Vector values ;
   DenseMatrix tr;
   lss->GetFaceValues(el, 0, *ir, values, tr, 1);
   FaceElementTransformations *transf =
      fes->GetMesh()->GetFaceElementTransformations(el, 31);

   for (int i=0; i < ir->GetNPoints();
        i++)    // For each quadrature point of the element
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      transf->SetAllIntPoints(&ip);
      // const IntegrationPoint &e1_ip = transf->GetElement1IntPoint();
      // ElementTransformation *e1_tr = transf->Elem1;
      // ElementTransformation *e2_tr = transf->Elem2;

      //double level_set_value = ls->Eval(*e1_tr, e1_ip);   // With exact ls
      double level_set_value = values(i);   // With GridFunction
      error += ip.weight * transf->Face->Weight() * std::pow(level_set_value, 2.0);
      //        error += ip.weight * transf->Face->Weight() * 1.0; // Should be equal to the lenght of the face
   }
   return error;
}

double ComputeIntegrateErrorBG(const FiniteElementSpace* fes,
                               GridFunction* ls_bg,
                               const int el, GridFunction *lss, FindPointsGSLIB &finder)
{
   double error = 0.0;
   const FiniteElement *fe = fes->GetFaceElement(el);  // Face el
   int intorder = 2*ls_bg->FESpace()->GetMaxElementOrder() + 3;
   const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));

   FaceElementTransformations *transf =
      fes->GetMesh()->GetFaceElementTransformations(el, 31);
   int dim = fes->GetMesh()->Dimension();
   Vector vxyz(
      dim*ir->GetNPoints());  // Coordinates of the quadrature points in the physical space
   Vector interp_values(
      ir->GetNPoints()); // Values of the ls fonction at the quadrature points that will be computed on the bg gridfunction
   // Compute the coords of the quadrature points in the physical space
   for (int i=0; i < ir->GetNPoints();
        i++)    // For each quadrature point of the element
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Vector xyz(dim);
      transf->Transform(ip, xyz);

      vxyz(i*dim) = xyz(0);
      vxyz(i*dim+1) = xyz(1);
      if (dim==3)
      {
         vxyz(i*dim+2) = xyz(2);
      }
   }

   // Compute the interpolated values of the level set grid function on the
   // physical coords of the quadrature points
   double length = 0.0;
   int point_ordering(1);
   finder.Interpolate(vxyz, *ls_bg, interp_values, point_ordering);
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      transf->SetAllIntPoints(&ip);
      double level_set_value = interp_values(i) ;
      error += ip.weight*transf->Face->Weight() * std::pow(level_set_value, 2.0);
      length += ip.weight*transf->Face->Weight();
      //error += ip.weight * transf->Face->Weight() * 1.0; // Should be equal to the lenght of the face
      //std::cout << "Integration point " << vxyz(dim*i) << ", " << vxyz(dim*i+1) << ", level set value " << level_set_value << std::endl;
   }

   //   error *= 1.0/length;

   return error;
}

double TestInterfaceElementOrderReduction(const Mesh *mesh,
                                          int face_el,
                                          int el_order,
                                          Array<Mesh *> surf_el_meshes,
                                          GridFunction* ls_bg,
                                          FindPointsGSLIB &finder,
                                          int etype = 0)
{
   const GridFunction *x = mesh->GetNodes();
   const FiniteElementSpace *fes = x->FESpace();
   int node_ordering = fes->GetOrdering();

   const FiniteElement *fe = fes->GetFaceElement(face_el);
   FaceElementTransformations *transf =
      fes->GetMesh()->GetFaceElementTransformations(face_el, 31);

   int fe_type = fe->GetGeomType();
   // int fe_dim = fe->GetDim();
   int mesh_dim = mesh->Dimension();
   // std::cout << fe_type << " " << fe_dim << " " <<
   // mesh_dim << " k101\n";

   IsoparametricTransformation T;
   DenseMatrix I;
   Geometry::Type geom = fe->GetGeomType();
   T.SetIdentityTransformation(geom);

   FiniteElementCollection *fecInt = new H1_FECollection(el_order,
                                                         mesh->Dimension());
   const auto *feInt = fecInt->GetFE(geom, el_order);
   // Get interpolation matrix from current order to target order
   feInt->GetTransferMatrix(*fe, T, I);

   // Get current coordinates
   DenseMatrix elemvecMat = transf->GetPointMat();
   elemvecMat.Transpose();
   // elemvecMat.Print();

   // Get coordinates for new order
   DenseMatrix IntVals(I.Height(), elemvecMat.Width());
   Mult(I, elemvecMat, IntVals);
   // IntVals.Print();

   // in surf_el_meshes, 0 holds a line mesh, 1 - triangle, and 2 - quad
   int surf_mesh_idx = fe_type - 1;
   Mesh *smesh = surf_el_meshes[surf_mesh_idx];

   smesh->SetCurvature(el_order, false, -1, node_ordering);
   GridFunction *snodes = smesh->GetNodes();
   // FiniteElementSpace *sfespace = snodes->FESpace();
   IsoparametricTransformation *seltrans = new IsoparametricTransformation();
   smesh->GetElementTransformation(0, seltrans);
   DenseMatrix svecMat = seltrans->GetPointMat();
   // svecMat.Print();

   seltrans->GetPointMat().Transpose(IntVals);
   DenseMatrix svecMat2 = seltrans->GetPointMat();
   // svecMat2.Print();

   const int int_order = 2*el_order + 3;
   const IntegrationRule *ir = &(IntRulesLo.Get(geom, int_order));

   // Coordinates of the quadrature points in the physical space
   Vector vxyz(mesh_dim*ir->GetNPoints());
   // Values of the ls fonction at the quadrature points that will be computed on the bg gridfunction
   Vector interp_values(ir->GetNPoints());
   // Compute the coords of the quadrature points in the physical space
   for (int i=0; i < ir->GetNPoints(); i++)
      // For each quadrature point of the element
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Vector xyz(mesh_dim);
      transf->Transform(ip, xyz);

      for (int d = 0; d < mesh_dim; d++)
      {
         vxyz(i*mesh_dim + d) = xyz(d);
      }
   }

   // Compute the interpolated values of the level set grid function on the
   // physical coords of the quadrature points
   double size = 0.0;
   double error = 0.0;
   int point_ordering(1);
   finder.Interpolate(vxyz, *ls_bg, interp_values, point_ordering);
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      seltrans->SetIntPoint(&ip);
      double level_set_value = interp_values(i) ;
      error += ip.weight*seltrans->Weight() * std::pow(level_set_value, 2.0);
      size += ip.weight*seltrans->Weight();
   }

   double orig_size = 0.0;
   if (etype == 1)
   {
      for (int i=0; i<ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         transf->SetAllIntPoints(&ip);
         orig_size += ip.weight*transf->Face->Weight();
      }
      MFEM_VERIFY(orig_size > 0, "Non-positive size of element");
      size -= orig_size;
      size = std::fabs(size);
      size *= 1.0/orig_size;
   }
   double return_val = etype == 0 ? error : size;

   std::cout << error << " "  << size << " k10error-size\n";

   delete seltrans;
   delete fecInt;
   return return_val;
}

double InterfaceElementOrderReduction(const Mesh *mesh,
                                      int face_el,
                                      int el_order,
                                      GridFunction* ls_bg,
                                      FindPointsGSLIB &finder)
{
   const GridFunction *x = mesh->GetNodes();
   const FiniteElementSpace *fes = x->FESpace();

   const FiniteElement *fe = fes->GetFaceElement(face_el);
   FaceElementTransformations *transf =
      fes->GetMesh()->GetFaceElementTransformations(face_el, 31);

   IsoparametricTransformation T;
   DenseMatrix I;
   Geometry::Type geom = fe->GetGeomType();
   T.SetIdentityTransformation(geom);

   FiniteElementCollection *fecInt = new H1_FECollection(el_order,
                                                         mesh->Dimension());
   const auto *feInt = fecInt->GetFE(geom, el_order);
   feInt->GetTransferMatrix(*fe, T, I);

   const int int_order = 2*el_order + 3;
   const IntegrationRule *ir = &(IntRulesLo.Get(geom, int_order));

   int neworder = std::pow(ir->GetNPoints(), 1.0/feInt->GetDim())-1;
   FiniteElementCollection *fecInt2 = new H1_FECollection(neworder,
                                                          mesh->Dimension());
   const auto *feInt2 = fecInt2->GetFE(geom, neworder);
   MFEM_VERIFY(ir->GetNPoints() == feInt2->GetNodes().GetNPoints(),
               "Something is wrong with the FE copying IntRule.");
   DenseMatrix I2;
   T.SetIdentityTransformation(geom);
   feInt2->GetTransferMatrix(*feInt, T, I2);

   DenseMatrix I3(I2.Height(), I.Width());
   Mult(I2, I, I3);

   DenseMatrix elemvecMat = transf->GetPointMat();
   elemvecMat.Transpose();
   //   elemvecMat.Print();

   DenseMatrix IntVals(I3.Height(), elemvecMat.Width());
   Mult(I3, elemvecMat, IntVals);
   //   IntVals.Print();

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(feInt2);
   MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
   const Array<int> &dof_map = tbe->GetDofMap();

   const IntegrationRule &irule = feInt2->GetNodes();

   const int dim = mesh->Dimension();
   Vector vxyz(irule.GetNPoints()*dim);
   for (int i = 0; i < dof_map.Size(); i++)
   {
      for (int d = 0; d < dim; d++)
      {
         vxyz(i*dim+d) = IntVals(dof_map[i], d);
      }
   }

   Vector interp_values(irule.GetNPoints());
   int point_ordering(1);
   finder.Interpolate(vxyz, *ls_bg, interp_values, point_ordering);

   double error = 0.0;
   for (int i=0; i<ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      transf->SetAllIntPoints(&ip);
      double level_set_value = interp_values(i) ;
      error += ip.weight * transf->Face->Weight() * std::pow(level_set_value, 2.0);
   }

   delete fecInt;
   delete fecInt2;

   return error;
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "icf.mesh";
   int mesh_poly_deg     = 1;
   int rs_levels         = 0;
   double jitter         = 0.0;
   int metric_id         = 2;
   int target_id         = 1;
   double surface_fit_const = 0.1;
   int quad_type         = 1;
   int quad_order        = 8;
   int solver_type       = 0;
   int solver_iter       = 200;
   double solver_rtol    = 1e-10;
   int solver_art_type   = 0;
   int lin_solver        = 2;
   int max_lin_iter      = 100;
   bool move_bnd         = true;
   bool visualization    = true;
   int verbosity_level   = 2;
   int adapt_eval        = 1;
   bool exactaction      = false;
   double surface_fit_adapt = 10.0;
   double surface_fit_threshold = 1e-14;
   int mesh_node_ordering = 0;
   bool prefine          = true;
   int pref_order_increase = 1;
   int pref_max_order    = 4;
   int pref_max_iter     = 1;
   double pref_tol       = 1e-13;
   bool surf_bg_mesh     = true;
   bool reduce_order     = true;
   const char *bg_mesh_file = "NULL";
   const char *bg_ls_file = "NULL";
   bool custom_material   = true;
   bool adapt_marking = false;
   int bg_amr_iter = 0;
   int ls_function = 0;

   // 1. Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&mesh_poly_deg, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&jitter, "-ji", "--jitter",
                  "Random perturbation scaling factor.");
   args.AddOption(&metric_id, "-mid", "--metric-id",
                  "Mesh optimization metric. See list in mesh-optimizer.");
   args.AddOption(&target_id, "-tid", "--target-id",
                  "Target (ideal element) type:\n\t"
                  "1: Ideal shape, unit size\n\t"
                  "2: Ideal shape, equal size\n\t"
                  "3: Ideal shape, initial size\n\t"
                  "4: Given full analytic Jacobian (in physical space)\n\t"
                  "5: Ideal shape, given size (in physical space)");
   args.AddOption(&surface_fit_const, "-sfc", "--surface-fit-const",
                  "Surface preservation constant.");
   args.AddOption(&quad_type, "-qt", "--quad-type",
                  "Quadrature rule type:\n\t"
                  "1: Gauss-Lobatto\n\t"
                  "2: Gauss-Legendre\n\t"
                  "3: Closed uniform points");
   args.AddOption(&quad_order, "-qo", "--quad_order",
                  "Order of the quadrature rule.");
   args.AddOption(&solver_type, "-st", "--solver-type",
                  " Type of solver: (default) 0: Newton, 1: LBFGS");
   args.AddOption(&solver_iter, "-ni", "--newton-iters",
                  "Maximum number of Newton iterations.");
   args.AddOption(&solver_rtol, "-rtol", "--newton-rel-tolerance",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&solver_art_type, "-art", "--adaptive-rel-tol",
                  "Type of adaptive relative linear solver tolerance:\n\t"
                  "0: None (default)\n\t"
                  "1: Eisenstat-Walker type 1\n\t"
                  "2: Eisenstat-Walker type 2");
   args.AddOption(&lin_solver, "-ls", "--lin-solver",
                  "Linear solver:\n\t"
                  "0: l1-Jacobi\n\t"
                  "1: CG\n\t"
                  "2: MINRES\n\t"
                  "3: MINRES + Jacobi preconditioner\n\t"
                  "4: MINRES + l1-Jacobi preconditioner");
   args.AddOption(&max_lin_iter, "-li", "--lin-iter",
                  "Maximum number of iterations in the linear solve.");
   args.AddOption(&move_bnd, "-bnd", "--move-boundary", "-fix-bnd",
                  "--fix-boundary",
                  "Enable motion along horizontal and vertical boundaries.");
   args.AddOption(&exactaction, "-ex", "--exact_action",
                  "-no-ex", "--no-exact-action",
                  "Enable exact action of TMOP_Integrator.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&verbosity_level, "-vl", "--verbosity-level",
                  "Set the verbosity level - 0, 1, or 2.");
   args.AddOption(&adapt_eval, "-ae", "--adaptivity-evaluator",
                  "0 - Advection based (DEFAULT), 1 - GSLIB.");
   args.AddOption(&surface_fit_adapt, "-sfa", "--adaptive-surface-fit",
                  "Enable or disable adaptive surface fitting.");
   args.AddOption(&surface_fit_threshold, "-sft", "--surf-fit-threshold",
                  "Set threshold for surface fitting. TMOP solver will"
                  "terminate when max surface fitting error is below this limit");
   args.AddOption(&mesh_node_ordering, "-mno", "--mesh_node_ordering",
                  "Ordering of mesh nodes."
                  "0 (default): byNodes, 1: byVDIM");
   args.AddOption(&prefine, "-pref", "--pref", "-no-pref",
                  "--no-pref",
                  "Randomly p-refine the mesh.");
   args.AddOption(&pref_order_increase, "-oi", "--preforderincrease",
                  "How much polynomial order to increase for p-refinement.");
   args.AddOption(&pref_max_order, "-mo", "--prefmaxorder",
                  "Maximum polynomial order for p-refinement.");
   args.AddOption(&pref_max_iter, "-mi", "--prefmaxiter",
                  "Maximum number of iteration");
   args.AddOption(&pref_tol, "-preft", "--preftol",
                  "Error tolerance on a face.");
   args.AddOption(&surf_bg_mesh, "-sbgmesh", "--surf-bg-mesh",
                  "-no-sbgmesh","--no-surf-bg-mesh",
                  "Use background mesh for surface fitting.");
   args.AddOption(&reduce_order, "-ro", "--reduce-order",
                  "-no-ro","--no-reduce-order",
                  "Reduce the order of elements around the interface.");
   args.AddOption(&bg_mesh_file, "-bgm", "--bgm",
                  "Background Mesh file to use.");
   args.AddOption(&bg_ls_file, "-bgls", "--bgls",
                  "Background level set gridfunction file to use.");
   args.AddOption(&custom_material, "-cus-mat", "--custom-material",
                  "-no-cus-mat", "--no-custom-material",
                  "When true, sets the material based on predetermined logic instead of level-set");
   args.AddOption(&adapt_marking, "-marking", "--adaptive-marking", "-no-amarking",
                  "--no-adaptive-marking",
                  "Enable or disable adaptive marking surface fitting.");
   args.AddOption(&bg_amr_iter, "-bgamr", "--bgamr",
                  "Number of times to AMR refine the background mesh.");
   args.AddOption(&ls_function, "-lsf", "--ls-function",
                  "Choice of level set function.");


   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);


   const char *vis_keys = "Rjaam";
   Array<Mesh *> surf_el_meshes = SetupSurfaceMeshes();

   FunctionCoefficient ls_coeff(circle_level_set);
   if (ls_function==1)
   {
      ls_coeff = FunctionCoefficient(squircle_level_set);
   }
   else if (ls_function==2)
   {
      ls_coeff = FunctionCoefficient(apollo_level_set);
   }
   else if (ls_function==3)
   {
      ls_coeff = FunctionCoefficient(csg_cubecylsph_smooth);
   }

   // 2. Initialize and refine the starting mesh.
   Mesh *mesh = new Mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int dim = mesh->Dimension();
   std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   if (prefine) { mesh->EnsureNCMesh(true); }
   FindPointsGSLIB finder;

   // mesh->SetCurvature(2);
   // surf_el_meshes[0]->SetCurvature(2);
   // MFEM_ABORT("k10testcurvature\n");

   // Setup background mesh for surface fitting: copy then refine the mesh
   Mesh *mesh_surf_fit_bg = NULL;
   if (surf_bg_mesh)
   {
      if (strcmp(bg_mesh_file, "NULL") != 0) //user specified background mesh
      {
         mesh_surf_fit_bg = new Mesh(bg_mesh_file, 1, 1, false);
      }
      else
      {
         mesh_surf_fit_bg = new Mesh(*mesh);
         if (bg_amr_iter == 0)
         {
            for (int ref = 0; ref < 1; ref++) { mesh_surf_fit_bg->UniformRefinement(); } // Refine the mesh in an uniform way x times
         }
      }
      GridFunction *mesh_surf_fit_bg_nodes = mesh_surf_fit_bg->GetNodes();
      if (mesh_surf_fit_bg_nodes == NULL)
      {
         std::cout << "Background mesh does not have nodes. Setting curvature\n";
         mesh_surf_fit_bg->SetCurvature(1, 0, -1, 0);
      }
      else
      {
         const TensorBasisElement *tbe =
            dynamic_cast<const TensorBasisElement *>
            (mesh_surf_fit_bg_nodes->FESpace()->GetFE(0));
         int order = mesh_surf_fit_bg_nodes->FESpace()->GetFE(0)->GetOrder();
         if (tbe == NULL)
         {
            std::cout <<
                      "Background mesh does not have tensor basis nodes. Setting tensor basis\n";
            mesh_surf_fit_bg->SetCurvature(order, 0, -1, 0);
         }
      }
      finder.Setup(*mesh_surf_fit_bg);
   }
   else
   {
      MFEM_ABORT("p-adaptivity is not supported without background mesh");
   }

   // 3. Define a finite element space on the mesh-> Here we use vector finite
   //    elements.
   MFEM_VERIFY(mesh_poly_deg >= 1,"Mesh order should at-least be 1.");
   // Use an H1 space for mesh nodes
   FiniteElementCollection *fec = new H1_FECollection(mesh_poly_deg, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim,
                                                        mesh_node_ordering);

   // use an L2 space for storing the order of elements (piecewise constant).
   L2_FECollection order_coll = L2_FECollection(0, dim);
   FiniteElementSpace order_space = FiniteElementSpace(mesh, &order_coll);
   GridFunction order_gf = GridFunction(&order_space);
   order_gf = mesh_poly_deg*1.0;

   // P-Refine the mesh - randomly
   // We do this here just to make sure that the base mesh-optimization algorithm
   // works for p-refined mesh
   if (prefine && surface_fit_const == 0.0)
   {
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         order_gf(e) = mesh_poly_deg;
         if ((double) rand() / RAND_MAX < 0.5)
         {
            int element_order = fespace->GetElementOrder(e);
            fespace->SetElementOrder(e, element_order + 2);
            order_gf(e) = element_order + 2;
         }
      }
      fespace->Update(false);
   }

   // Curve the mesh based on the (optionally p-refined) finite element space.
   mesh->SetNodalFESpace(fespace);

   // Get the mesh nodes (vertices and other degrees of freedom in the finite
   // element space) as a finite element grid function in fespace. Note that
   // changing x automatically changes the shapes of the mesh elements.
   GridFunction x(fespace);
   mesh->SetNodalGridFunction(&x);

   // Define a gridfunction to save the mesh at maximum order when some of the
   // elements in the mesh are p-refined. We need this for now because some of
   // mfem's output functions do not work for p-refined spaces.
   GridFunction *x_max_order = NULL;
   delete x_max_order;
   x_max_order = ProlongToMaxOrder(&x, 0);
   mesh->SetNodalGridFunction(x_max_order);
   mesh->SetNodalGridFunction(&x);


   // Define a vector representing the minimal local mesh size in the mesh
   // nodes. We index the nodes using the scalar version of the degrees of
   // freedom in fespace. Note: this is partition-dependent.
   //
   // In addition, compute average mesh size and total volume.
   Vector h0(fespace->GetNDofs());
   h0 = infinity();
   double mesh_volume = 0.0;
   Array<int> dofs;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Get the local scalar element degrees of freedom in dofs.
      fespace->GetElementDofs(i, dofs);
      // Adjust the value of h0 in dofs based on the local mesh size.
      const double hi = mesh->GetElementSize(i);
      for (int j = 0; j < dofs.Size(); j++)
      {
         h0(dofs[j]) = min(h0(dofs[j]), hi);
      }
      mesh_volume += mesh->GetElementVolume(i);
   }

   // 8. Add a random perturbation to the nodes in the interior of the domain.
   //    We define a random grid function of fespace and make sure that it is
   //    zero on the boundary and its values are locally of the order of h0.
   //    The latter is based on the DofToVDof() method which maps the scalar to
   //    the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   if (jitter != 0.0)
   {
      rdm.Randomize();
      rdm -= 0.25; // Shift to random values in [-0.5,0.5].
      rdm *= jitter;
      rdm.HostReadWrite();
      // Scale the random values to be of order of the local mesh size.
      for (int i = 0; i < fespace->GetNDofs(); i++)
      {
         for (int d = 0; d < dim; d++)
         {
            rdm(fespace->DofToVDof(i,d)) *= h0(i);
         }
      }
      Array<int> vdofs;
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // Get the vector degrees of freedom in the boundary element.
         fespace->GetBdrElementVDofs(i, vdofs);
         // Set the boundary values to zero.
         for (int j = 0; j < vdofs.Size(); j++) { rdm(vdofs[j]) = 0.0; }
      }
      x -= rdm;
   }

   // For parallel runs, we define the true-vector. This makes sure the data is
   // consistent across processor boundaries.
   x.SetTrueVector();
   x.SetFromTrueVector();

   // 9. Save the starting (prior to the optimization) mesh to a file. This
   //    output can be viewed later using GLVis: "glvis -m perturbed.mesh".
   if (!prefine)
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 10. Store the starting (prior to the optimization) positions.
   GridFunction x0(fespace);
   x0 = x;

   // 11. Form the integrator that uses the chosen metric and target.
   // First pick a metric
   double min_detJ = -0.1;
   TMOP_QualityMetric *metric = NULL;
   switch (metric_id)
   {
      // T-metrics
      case 1: metric = new TMOP_Metric_001; break; //shape-metric
      case 2: metric = new TMOP_Metric_002; break; //shape-metric
      case 58: metric = new TMOP_Metric_058; break; // shape-metric
      case 80: metric = new TMOP_Metric_080(0.5); break; //shape+size
      case 303: metric = new TMOP_Metric_303; break; //shape
      case 328: metric = new TMOP_Metric_328(); break; //shape+size
      default:
         cout << "Unknown metric_id: " << metric_id << endl;
         return 3;
   }

   if (metric_id < 300)
   {
      MFEM_VERIFY(dim == 2, "Incompatible metric for 3D meshes");
   }
   if (metric_id >= 300)
   {
      MFEM_VERIFY(dim == 3, "Incompatible metric for 2D meshes");
   }

   // Next, select a target.
   TargetConstructor::TargetType target_t;
   TargetConstructor *target_c = NULL;
   switch (target_id)
   {
      case 1: target_t = TargetConstructor::IDEAL_SHAPE_UNIT_SIZE; break;
      case 2: target_t = TargetConstructor::IDEAL_SHAPE_EQUAL_SIZE; break;
      case 3: target_t = TargetConstructor::IDEAL_SHAPE_GIVEN_SIZE; break;
      case 4: target_t = TargetConstructor::GIVEN_SHAPE_AND_SIZE; break;
      default: cout << "Unknown target_id: " << target_id << endl; return 3;
   }
   if (target_c == NULL)
   {
      target_c = new TargetConstructor(target_t);
   }
   target_c->SetNodes(x0);

   // Setup the quadrature rules for the TMOP integrator.
   IntegrationRules *irules = NULL;
   switch (quad_type)
   {
      case 1: irules = &IntRulesLo; break;
      case 2: irules = &IntRules; break;
      case 3: irules = &IntRulesCU; break;
      default: cout << "Unknown quad_type: " << quad_type << endl; return 3;
   }
   if (dim == 2)
   {
      cout << "Triangle quadrature points: "
           << irules->Get(Geometry::TRIANGLE, quad_order).GetNPoints()
           << "\nQuadrilateral quadrature points: "
           << irules->Get(Geometry::SQUARE, quad_order).GetNPoints() << endl;
   }
   if (dim == 3)
   {
      cout << "Tetrahedron quadrature points: "
           << irules->Get(Geometry::TETRAHEDRON, quad_order).GetNPoints()
           << "\nHexahedron quadrature points: "
           << irules->Get(Geometry::CUBE, quad_order).GetNPoints()
           << "\nPrism quadrature points: "
           << irules->Get(Geometry::PRISM, quad_order).GetNPoints() << endl;
   }

   // Surface fitting.
   L2_FECollection mat_coll(0, dim);
   H1_FECollection surf_fit_fec(mesh_poly_deg, dim);
   FiniteElementSpace surf_fit_fes(mesh, &surf_fit_fec);
   // Elevate to the same space as mesh for prefinement
   surf_fit_fes.CopySpaceElementOrders(*fespace);
   FiniteElementSpace mat_fes(mesh, &mat_coll);
   GridFunction mat(&mat_fes);
   GridFunction NumFaces(&mat_fes);
   GridFunction surf_fit_mat_gf(&surf_fit_fes);
   GridFunction surf_fit_gf0(&surf_fit_fes);
   Array<bool> surf_fit_marker(surf_fit_gf0.Size());
   ConstantCoefficient surf_fit_coeff(surface_fit_const);
   AdaptivityEvaluator *adapt_surface = NULL;
   AdaptivityEvaluator *adapt_grad_surface = NULL;
   AdaptivityEvaluator *adapt_hess_surface = NULL;

   GridFunction *surf_fit_gf0_max_order = &surf_fit_gf0;
   GridFunction *surf_fit_mat_gf_max_order = &surf_fit_mat_gf;
   GridFunction fitting_error_gf(&mat_fes);

   // Background mesh FECollection, FESpace, and GridFunction
   FiniteElementCollection *surf_fit_bg_fec = NULL;
   FiniteElementSpace *surf_fit_bg_fes = NULL;
   GridFunction *surf_fit_bg_gf0 = NULL;
   FiniteElementSpace *surf_fit_bg_grad_fes = NULL;
   GridFunction *surf_fit_bg_grad = NULL;
   FiniteElementSpace *surf_fit_bg_hess_fes = NULL;
   GridFunction *surf_fit_bg_hess = NULL;

   // If a background mesh is used, we interpolate the Gradient and Hessian
   // from that mesh to the current mesh being optimized.
   FiniteElementSpace *surf_fit_grad_fes = NULL;
   GridFunction *surf_fit_grad = NULL;
   FiniteElementSpace *surf_fit_hess_fes = NULL;
   GridFunction *surf_fit_hess = NULL;

   if (surf_bg_mesh)
   {
      //if the user specified a gridfunction file, use that
      if (strcmp(bg_ls_file, "NULL") != 0) //user specified background mesh
      {
         ifstream bg_ls_stream(bg_ls_file);
         surf_fit_bg_gf0 = new GridFunction(mesh_surf_fit_bg, bg_ls_stream);
         surf_fit_bg_fes = surf_fit_bg_gf0->FESpace();
         surf_fit_bg_fec = const_cast<FiniteElementCollection *>
                           (surf_fit_bg_fes->FEColl());
         *surf_fit_bg_gf0 -= 0.1;
         finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                            x.FESpace()->GetOrdering());
      }
      else
      {
         // Init the FEC, FES and GridFunction of uniform order = 6
         // for the background ls function
         surf_fit_bg_fec = new H1_FECollection(6, dim);
         surf_fit_bg_fes = new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec);
         surf_fit_bg_gf0 = new GridFunction(surf_fit_bg_fes);
         surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);
         if (bg_amr_iter > 0)
         {
            OptimizeMeshWithAMRAroundZeroLevelSet(*mesh_surf_fit_bg,
                                                  ls_coeff,
                                                  bg_amr_iter,
                                                  *surf_fit_bg_gf0);
         }
         finder.Setup(*mesh_surf_fit_bg);
         surf_fit_bg_gf0->ProjectCoefficient(ls_coeff);
      }

      surf_fit_bg_grad_fes =
         new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim);
      surf_fit_bg_grad = new GridFunction(surf_fit_bg_grad_fes);

      surf_fit_bg_hess_fes =
         new FiniteElementSpace(mesh_surf_fit_bg, surf_fit_bg_fec, dim * dim);
      surf_fit_bg_hess = new GridFunction(surf_fit_bg_hess_fes);

      //Setup gradient of the background mesh
      const int size_bg = surf_fit_bg_gf0->Size();
      for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
      {
         GridFunction surf_fit_bg_grad_comp(
            surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
         surf_fit_bg_gf0->GetDerivative(1, d, surf_fit_bg_grad_comp);
      }

      //Setup Hessian on background mesh
      int id = 0;
      for (int d = 0; d < mesh_surf_fit_bg->Dimension(); d++)
      {
         for (int idir = 0; idir < mesh_surf_fit_bg->Dimension(); idir++)
         {
            GridFunction surf_fit_bg_grad_comp(
               surf_fit_bg_fes, surf_fit_bg_grad->GetData() + d * size_bg);
            GridFunction surf_fit_bg_hess_comp(
               surf_fit_bg_fes, surf_fit_bg_hess->GetData()+ id * size_bg);
            surf_fit_bg_grad_comp.GetDerivative(1, idir,
                                                surf_fit_bg_hess_comp);
            id++;
         }
      }
   }

   if (surface_fit_const > 0.0)
   {
      if (surf_bg_mesh && strcmp(bg_mesh_file, "NULL") != 0)
      {
         finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                            x.FESpace()->GetOrdering());
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, surf_fit_gf0,
                                "Interpolated level-set function",
                                1100, 0, 500, 500);
      }
      else
      {
         surf_fit_gf0.ProjectCoefficient(ls_coeff);
      }

      for (int i = 0; i < mesh->GetNE(); i++)
      {
         mat(i) = material_id(i, surf_fit_gf0);
         if (custom_material)
         {
            Vector center(mesh->Dimension());
            mesh->GetElementCenter(i, center);
            mat(i) = 1;
            if (mesh->Dimension() == 2)
            {
               if (center(0) > 0.25 && center(0) < 0.75 && center(1) > 0.25 &&
                   center(1) < 0.75)
               {
                  mat(i) = 0;
               }
            }
            else if (mesh->Dimension() == 3)
            {
               if (center(0) > 0.25 && center(0) < 0.75
                   && center(1) > 0.25 && center(1) < 0.75
                   && center(2) > 0.25 && center(2) < 0.75)
               {
                  mat(i) = 0;
               }
            }
         }
         mesh->SetAttribute(i, static_cast<int>(mat(i) + 1));
      }

      if (prefine)
      {
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
      }

      if (adapt_marking)
      {
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 1);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 0);
         MakeGridFunctionWithNumberOfInterfaceFaces(mesh, mat, NumFaces);
         ModifyAttributeForMarkingDOFS(mesh, mat, 1);
      }
   }

   Array<int> inter_faces;
   Array<int> inter_face_el1, inter_face_el2;
   // Make list of all interface faces
   for (int i=0; i < mesh->GetNumFaces(); i++)
   {
      Array<int> els;
      mesh->GetFaceAdjacentElements(i, els);
      if (els.Size() == 2)
      {
         int mat1 = mat(els[0]);
         int mat2 = mat(els[1]);
         if (mat1 != mat2)
         {
            inter_faces.Append(i);
            inter_face_el1.Append(els[0]);
            inter_face_el2.Append(els[1]);
         }
      }
   }

   if (visualization)
   {
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, order_gf,
                             "Polynomial order before p-refinement",
                             1100, 0, 500, 500);
      mesh->SetNodalGridFunction(&x);
   }

   std::cout <<
             "Max Order || Init order || Pref order increase || Nbr of iterations" <<
             std::endl;
   std::cout << pref_max_order << " || " << mesh_poly_deg << " || " <<
             pref_order_increase << " || " << pref_max_iter << std::endl;
   int iter_pref(0);
   bool faces_to_update(true);
   while (iter_pref<pref_max_iter && faces_to_update)
   {
      std::cout << "REFINEMENT NUMBER: " << iter_pref << std::endl;

      if (iter_pref > 0)
      {
         surface_fit_const = 1e6;
      }

      // Define a TMOPIntegrator based on the metric and target.
      TMOP_Integrator *tmop_integ = new TMOP_Integrator(metric, target_c);
      tmop_integ->SetIntegrationRules(*irules, quad_order);

      if (surface_fit_const > 0.0)
      {
         // Define a function coefficient (based on the analytic description of
         // the level-set)
         if (surf_bg_mesh && strcmp(bg_mesh_file, "NULL") != 0)
         {
            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
         }
         else
         {
            surf_fit_gf0.ProjectCoefficient(ls_coeff);
         }
         //         surf_fit_gf0.ProjectCoefficient(ls_coeff);

         // Now p-refine the elements around the interface
         if (prefine)
         {
            // Define a transfer operator for updating gridfunctions after the mesh
            // has been p-refined`
            PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);
            PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);

            int max_order = fespace->GetMaxElementOrder();
            Array<int> faces_order_increase;

            delete x_max_order;
            x_max_order = ProlongToMaxOrder(&x, 0);
            mesh->SetNodalGridFunction(x_max_order);
            delete surf_fit_gf0_max_order;
            surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
            double min_face_error = std::numeric_limits<double>::max();
            double max_face_error = std::numeric_limits<double>::min();
            for (int i=0; i < inter_faces.Size(); i++)
            {
               int facenum = inter_faces[i];
               double error_bg_face(0);
               if (surf_bg_mesh)
               {
                  error_bg_face = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                          surf_fit_bg_gf0,
                                                          facenum,
                                                          surf_fit_gf0_max_order,
                                                          finder);
               }
               else
               {
                  error_bg_face = ComputeIntegrateError(x_max_order->FESpace(),
                                                        surf_fit_gf0_max_order,
                                                        inter_faces[i]);
               }
               min_face_error = std::min(min_face_error, error_bg_face);
               max_face_error = std::max(max_face_error, error_bg_face);

               if (error_bg_face >= pref_tol)
               {
                  faces_order_increase.Append(facenum);
               }
            }
            std::cout << "Min/Max face error: " << min_face_error << " " <<
                      max_face_error << std::endl;
            mesh->SetNodalGridFunction(&x);
            if (iter_pref>0)
            {
               std::cout << "Number of faces p-refined: " << faces_order_increase.Size() <<
                         std::endl;
               for (int i = 0; i < faces_order_increase.Size(); i++)
               {
                  Array<int> els;
                  mesh->GetFaceAdjacentElements(faces_order_increase[i], els);
                  fespace->SetElementOrder(els[0], max_order + pref_order_increase);
                  fespace->SetElementOrder(els[1], max_order + pref_order_increase);
                  order_gf(els[0]) = max_order + pref_order_increase;
                  order_gf(els[1]) = max_order + pref_order_increase;
               }

               if (faces_order_increase.Size() != 0)
               {
                  // Updates if we increase the order of at least one element
                  fespace->Update(false);
                  surf_fit_fes.CopySpaceElementOrders(*fespace);
                  preft_fespace.Transfer(x);
                  preft_fespace.Transfer(x0);
                  preft_fespace.Transfer(rdm);
                  preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
                  preft_surf_fit_fes.Transfer(surf_fit_gf0);
                  surf_fit_marker.SetSize(surf_fit_gf0.Size());

                  x.SetTrueVector();
                  x.SetFromTrueVector();

                  mesh->SetNodalGridFunction(&x);

                  delete x_max_order;
                  x_max_order = ProlongToMaxOrder(&x, 0);
                  delete surf_fit_gf0_max_order;
                  surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
               }
               else
               {
                  faces_to_update = false;
               }
            }

            /*
            if (visualization)
            {
               x_max_order = ProlongToMaxOrder(&x, 0);
               mesh->SetNodalGridFunction(x_max_order);
               socketstream vis1;
               common::VisualizeField(vis1, "localhost", 19916, order_gf,
                                      "Polynomial order in loop",
                                      00, 600, 300, 300);
               mesh->SetNodalGridFunction(&x);
            }
             */
         }

         if (surf_bg_mesh && strcmp(bg_mesh_file, "NULL") != 0)
         {
            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
         }
         else
         {
            surf_fit_gf0.ProjectCoefficient(ls_coeff);
         }

         if (surf_bg_mesh)
         {
            delete surf_fit_grad_fes;
            delete surf_fit_grad;
            surf_fit_grad_fes =
               new FiniteElementSpace(mesh, &surf_fit_fec, dim);
            surf_fit_grad_fes->CopySpaceElementOrders(*fespace);
            surf_fit_grad = new GridFunction(surf_fit_grad_fes);

            delete surf_fit_hess_fes;
            delete surf_fit_hess;
            surf_fit_hess_fes =
               new FiniteElementSpace(mesh, &surf_fit_fec, dim * dim);
            surf_fit_hess_fes->CopySpaceElementOrders(*fespace);
            surf_fit_hess = new GridFunction(surf_fit_hess_fes);
         }

         for (int j = 0; j < surf_fit_marker.Size(); j++)
         {
            surf_fit_marker[j] = false;
         }
         surf_fit_mat_gf = 0.0;

         Array<int> dof_list;
         Array<int> dofs;
         for (int i = 0; i < inter_faces.Size(); i++)
         {
            int fnum = inter_faces[i];
            if (dim == 2)
            {
               surf_fit_gf0.FESpace()->GetEdgeDofs(fnum, dofs);
            }
            else
            {
               surf_fit_gf0.FESpace()->GetFaceDofs(fnum, dofs);
            }
            dof_list.Append(dofs);
         }

         for (int i = 0; i < dof_list.Size(); i++)
         {
            surf_fit_marker[dof_list[i]] = true;
            surf_fit_mat_gf(dof_list[i]) = 1.0;
         }
         if (iter_pref != 0 && prefine) { delete surf_fit_mat_gf_max_order; }


         if (adapt_eval == 0) { adapt_surface = new AdvectorCG; }
         else if (adapt_eval == 1)
         {
#ifdef MFEM_USE_GSLIB
            adapt_surface = new InterpolatorFP;
            if (surf_bg_mesh)
            {
               adapt_grad_surface = new InterpolatorFP;
               adapt_hess_surface = new InterpolatorFP;
            }
#else
            MFEM_ABORT("MFEM is not built with GSLIB support!");
#endif
         }
         else { MFEM_ABORT("Bad interpolation option."); }
         if (!surf_bg_mesh)
         {
            tmop_integ->EnableSurfaceFitting(surf_fit_gf0, surf_fit_marker,
                                             surf_fit_coeff, *adapt_surface);
         }
         else
         {
            tmop_integ->EnableSurfaceFittingFromSource(
               *surf_fit_bg_gf0, surf_fit_gf0,
               surf_fit_marker, surf_fit_coeff, *adapt_surface,
               *surf_fit_bg_grad, *surf_fit_grad, *adapt_grad_surface,
               *surf_fit_bg_hess, *surf_fit_hess, *adapt_hess_surface);
         }

         if (prefine)
         {
            delete x_max_order;
            x_max_order = ProlongToMaxOrder(&x, 0);
            mesh->SetNodalGridFunction(x_max_order);
            delete surf_fit_gf0_max_order;
            surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
            surf_fit_mat_gf_max_order = ProlongToMaxOrder(&surf_fit_mat_gf, 0);
         }
         if (visualization && iter_pref==0)
         {
            socketstream vis1, vis2;
            common::VisualizeField(vis1, "localhost", 19916, mat,
                                   "Materials before fitting",
                                   600, 680, 300, 300);
            //common::VisualizeField(vis2, "localhost", 19916, *surf_fit_mat_gf_max_order,
            //                       "Dofs to Move",
            //                       900, 680, 300, 300);
         }
         mesh->SetNodalGridFunction(&x);
      }

      if (iter_pref==0)
      {
         if (surf_bg_mesh && strcmp(bg_mesh_file, "NULL") != 0)
         {
            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
         }
         else
         {
            surf_fit_gf0.ProjectCoefficient(ls_coeff);
         }
         //         surf_fit_gf0.ProjectCoefficient(ls_coeff);
         tmop_integ->CopyGridFunction(surf_fit_gf0);
         if (prefine)
         {
            delete surf_fit_gf0_max_order;
            delete x_max_order;
         }
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         x_max_order = ProlongToMaxOrder(&x, 0);
         mesh->SetNodalGridFunction(x_max_order);
         double error_bg_sum = 0.0;
         double min_face_error = std::numeric_limits<double>::max();
         double max_face_error = std::numeric_limits<double>::min();
         fitting_error_gf = 0.0;
         for (int i = 0; i < inter_faces.Size(); i++)
         {
            double error_bg_face(0);
            if (surf_bg_mesh)
            {
               error_bg_face = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                       surf_fit_bg_gf0,
                                                       inter_faces[i],
                                                       surf_fit_gf0_max_order,
                                                       finder);
            }
            else
            {
               error_bg_face = ComputeIntegrateError(x_max_order->FESpace(),
                                                     surf_fit_gf0_max_order,
                                                     inter_faces[i]);
            }

            min_face_error = std::min(min_face_error, error_bg_face);
            max_face_error = std::max(max_face_error, error_bg_face);
            error_bg_sum += error_bg_face;

            fitting_error_gf(inter_face_el1[i]) = error_bg_face;
            fitting_error_gf(inter_face_el2[i]) = error_bg_face;
         }
         if (visualization && iter_pref==0)
         {
            socketstream vis1;
            common::VisualizeField(vis1, "localhost", 19916, fitting_error_gf,
                                   "Fitting Error before any fitting",
                                   900, 680, 300, 300);
         }
         if (visualization && iter_pref==1)
         {
            socketstream vis1;
            common::VisualizeField(vis1, "localhost", 19916, fitting_error_gf,
                                   "Fitting Error after first fitting, before p-ref",
                                   900, 680, 300, 300);
         }
         std::cout << "Integrate fitting error on BG: " << error_bg_sum << " " <<
                   std::endl;
         std::cout << "Min/Max face error: " << min_face_error << " " <<
                   max_face_error << std::endl;
         std::cout << "Max order || Nbr DOFS || Integrate fitting error on BG" <<
                   std::endl;
         std::cout << fespace->GetMaxElementOrder() << " " << fespace->GetNDofs() << " "
                   << error_bg_sum
                   << std::endl;
         mesh->SetNodalGridFunction(&x);
      }

      // 12. Setup the final NonlinearForm (which defines the integral of interest,
      //     its first and second derivatives). Here we can use a combination of
      //     metrics, i.e., optimize the sum of two integrals, where both are
      //     scaled by used-defined space-dependent weights. Note that there are no
      //     command-line options for the weights and the type of the second
      //     metric; one should update those in the code.
      NonlinearForm a(fespace);
      a.AddDomainIntegrator(tmop_integ);

      // Compute the minimum det(J) of the starting mesh.
      min_detJ = infinity();
      const int NE = mesh->GetNE();
      for (int i = 0; i < NE; i++)
      {
         const IntegrationRule &ir =
            irules->Get(fespace->GetFE(i)->GetGeomType(), quad_order);
         ElementTransformation *transf = mesh->GetElementTransformation(i);
         for (int j = 0; j < ir.GetNPoints(); j++)
         {
            transf->SetIntPoint(&ir.IntPoint(j));
            min_detJ = min(min_detJ, transf->Jacobian().Det());
         }
      }
      cout << "Minimum det(J) of the original mesh is " << min_detJ << endl;

      if (min_detJ < 0.0
          && metric_id != 22 && metric_id != 211 && metric_id != 252
          && metric_id != 311 && metric_id != 313 && metric_id != 352)
      {
         MFEM_ABORT("The input mesh is inverted! Try an untangling metric.");
      }
      if (min_detJ < 0.0)
      {
         MFEM_VERIFY(target_t == TargetConstructor::IDEAL_SHAPE_UNIT_SIZE,
                     "Untangling is supported only for ideal targets.");

         const DenseMatrix &Wideal =
            Geometries.GetGeomToPerfGeomJac(fespace->GetFE(0)->GetGeomType());
         min_detJ /= Wideal.Det();

         // Slightly below minJ0 to avoid div by 0.
         min_detJ -= 0.01 * h0.Min();
      }

      // For HR tests, the energy is normalized by the number of elements.
      const double init_energy = a.GetGridFunctionEnergy(x);
      double init_metric_energy = init_energy;
      if (surface_fit_const > 0.0)
      {
         surf_fit_coeff.constant   = 0.0;
         init_metric_energy = a.GetGridFunctionEnergy(x);
         surf_fit_coeff.constant   = surface_fit_const;
      }
      mesh->SetNodalGridFunction(x_max_order);
      // Visualize the starting mesh and metric values.
      // Note that for combinations of metrics, this only shows the first metric.
      if (visualization && iter_pref==0)
      {
         char title[] = "Initial metric values";
         vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 0);
      }
      mesh->SetNodalGridFunction(&x);

      // 13. Fix all boundary nodes, or fix only a given component depending on the
      //     boundary attributes of the given mesh. Attributes 1/2/3 correspond to
      //     fixed x/y/z components of the node. Attribute 4 corresponds to an
      //     entirely fixed node. Other boundary attributes do not affect the node
      //     movement boundary conditions.
      if (move_bnd == false)
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         a.SetEssentialBC(ess_bdr);
      }
      else
      {
         int n = 0;
         for (int i = 0; i < mesh->GetNBE(); i++)
         {
            const int nd = fespace->GetBE(i)->GetDof();
            const int attr = mesh->GetBdrElement(i)->GetAttribute();
            MFEM_VERIFY(!(dim == 2 && attr == 3),
                        "Boundary attribute 3 must be used only for 3D meshes. "
                        "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                        "components, rest for free nodes), or use -fix-bnd.");
            if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
            if (attr == 4) { n += nd * dim; }
         }
         Array<int> ess_vdofs(n);
         n = 0;
         Array<int> vdofs;
         for (int i = 0; i < mesh->GetNBE(); i++)
         {
            const int nd = fespace->GetBE(i)->GetDof();
            const int attr = mesh->GetBdrElement(i)->GetAttribute();
            fespace->GetBdrElementVDofs(i, vdofs);
            if (attr == 1) // Fix x components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[n++] = vdofs[j]; }
            }
            else if (attr == 2) // Fix y components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[n++] = vdofs[j+nd]; }
            }
            else if (attr == 3) // Fix z components.
            {
               for (int j = 0; j < nd; j++)
               { ess_vdofs[n++] = vdofs[j+2*nd]; }
            }
            else if (attr == 4) // Fix all components.
            {
               for (int j = 0; j < vdofs.Size(); j++)
               { ess_vdofs[n++] = vdofs[j]; }
            }
         }
         a.SetEssentialVDofs(ess_vdofs);
      }

      // 14. As we use the Newton method to solve the resulting nonlinear system,
      //     here we setup the linear solver for the system's Jacobian.
      Solver *S = NULL, *S_prec = NULL;
      const double linsol_rtol = 1e-12;
      if (lin_solver == 0)
      {
         S = new DSmoother(1, 1.0, max_lin_iter);
      }
      else if (lin_solver == 1)
      {
         CGSolver *cg = new CGSolver;
         cg->SetMaxIter(max_lin_iter);
         cg->SetRelTol(linsol_rtol);
         cg->SetAbsTol(0.0);
         cg->SetPrintLevel(verbosity_level >= 2 ? 3 : -1);
         S = cg;
      }
      else
      {
         MINRESSolver *minres = new MINRESSolver;
         minres->SetMaxIter(max_lin_iter);
         minres->SetRelTol(linsol_rtol);
         minres->SetAbsTol(0.0);
         if (verbosity_level > 2) { minres->SetPrintLevel(1); }
         minres->SetPrintLevel(verbosity_level == 2 ? 3 : -1);
         if (lin_solver == 3 || lin_solver == 4)
         {
            auto ds = new DSmoother((lin_solver == 3) ? 0 : 1, 1.0, 1);
            ds->SetPositiveDiagonal(true);
            S_prec = ds;
            minres->SetPreconditioner(*S_prec);
         }
         S = minres;
      }

      // Set up an empty right-hand side vector b, which is equivalent to b=0.
      // We use this later when we solve the TMOP problem
      Vector b(0);

      // Perform the nonlinear optimization.
      const IntegrationRule &ir =
         irules->Get(fespace->GetFE(0)->GetGeomType(), quad_order);
      TMOPNewtonSolver solver(ir, solver_type);
      if (surface_fit_const > 0.0 && surface_fit_adapt)
      {
         solver.SetAdaptiveSurfaceFittingScalingFactor(surface_fit_adapt);
         solver.SetAdaptiveSurfaceFittingRelativeChangeThreshold(0.001);
         //         solver.EnableAdaptiveSurfaceFitting();
      }
      if (surface_fit_const > 0.0 && surface_fit_threshold > 0)
      {
         solver.SetTerminationWithMaxSurfaceFittingError(surface_fit_threshold);
      }
      // Provide all integration rules in case of a mixed mesh.
      solver.SetIntegrationRules(*irules, quad_order);
      if (solver_type == 0)
      {
         solver.SetPreconditioner(*S);
      }
      // For untangling, the solver will update the min det(T) values.
      solver.SetMinDetPtr(&min_detJ);
      solver.SetMaxIter(solver_iter);
      solver.SetRelTol(solver_rtol);
      solver.SetAbsTol(0.0);
      if (solver_art_type > 0)
      {
         solver.SetAdaptiveLinRtol(solver_art_type, 0.5, 0.9);
      }
      solver.SetPrintLevel(verbosity_level >= 1 ? 1 : -1);

      solver.SetOperator(a);
      solver.Mult(b, x.GetTrueVector());
      x.SetFromTrueVector();

      delete x_max_order;
      x_max_order = ProlongToMaxOrder(&x, 0);
      mesh->SetNodalGridFunction(x_max_order);

      // 15. Save the optimized mesh to a file. This output can be viewed later
      //     using GLVis: "glvis -m optimized.mesh".
      {
         ofstream mesh_ofs("optimized.mesh");
         mesh_ofs.precision(14);
         mesh->Print(mesh_ofs);
      }
      mesh->SetNodalGridFunction(&x);

      // Report the final energy of the functional.
      const double fin_energy = a.GetGridFunctionEnergy(x);
      double fin_metric_energy = fin_energy;
      if (surface_fit_const > 0.0)
      {
         surf_fit_coeff.constant  = 0.0;
         fin_metric_energy  = a.GetGridFunctionEnergy(x);
         surf_fit_coeff.constant  = surface_fit_const;
      }

      std::cout << std::scientific << std::setprecision(4);
      cout << "Initial strain energy: " << init_energy
           << " = metrics: " << init_metric_energy
           << " + extra terms: " << init_energy - init_metric_energy << endl;
      cout << "  Final strain energy: " << fin_energy
           << " = metrics: " << fin_metric_energy
           << " + extra terms: " << fin_energy - fin_metric_energy << endl;
      cout << "The strain energy decreased by: "
           << (init_energy - fin_energy) * 100.0 / init_energy << " %." << endl;

      mesh->SetNodalGridFunction(x_max_order);
      // Visualize the final mesh and metric values.
      if (visualization && iter_pref==pref_max_iter-1)
      {
         char title[] = "Final metric values";
         vis_tmop_metric_s(mesh_poly_deg, *metric, *target_c, *mesh, title, 500);
      }

      // Visualize fitting surfaces and report fitting errors.
      if (surface_fit_const > 0.0)
      {
         if (visualization && iter_pref==pref_max_iter-1)
         {
            socketstream vis1, vis2;
            common::VisualizeField(vis1, "localhost", 19916, mat, "Materials after fitting",
                                   600, 680, 300, 300);
            //common::VisualizeField(vis2, "localhost", 19916, *surf_fit_mat_gf_max_order,
            //                       "Surface dof",
            //                       900, 680, 300, 300);
         }
         double err_avg, err_max;
         tmop_integ->GetSurfaceFittingErrors(err_avg, err_max);
         std::cout << "Avg fitting error: " << err_avg << std::endl
                   << "Max fitting error: " << err_max << std::endl;

         //         surf_fit_gf0.ProjectCoefficient(ls_coeff);
         if (surf_bg_mesh && strcmp(bg_mesh_file, "NULL") != 0)
         {
            finder.Interpolate(x, *surf_fit_bg_gf0, surf_fit_gf0,
                               x.FESpace()->GetOrdering());
         }
         else
         {
            surf_fit_gf0.ProjectCoefficient(ls_coeff);
         }
         tmop_integ->CopyGridFunction(surf_fit_gf0);
         delete surf_fit_gf0_max_order;
         surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
         double error_bg_sum = 0.0;
         fitting_error_gf = 0.0;
         for (int i=0; i < inter_faces.Size(); i++)
         {
            double error_bg_face(0);
            if (surf_bg_mesh)
            {
               error_bg_face = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                       surf_fit_bg_gf0,
                                                       inter_faces[i],
                                                       surf_fit_gf0_max_order,
                                                       finder);
            }
            else
            {
               error_bg_face = ComputeIntegrateError(x_max_order->FESpace(),
                                                     surf_fit_gf0_max_order,
                                                     inter_faces[i]);
            }
            //std::cout << "Face " << inter_faces[i] << ", " << error_bg_face << std::endl;
            error_bg_sum += error_bg_face;
            fitting_error_gf(inter_face_el1[i]) = error_bg_face;
            fitting_error_gf(inter_face_el2[i]) = error_bg_face;
         }

         if (visualization && iter_pref==0)
         {
            socketstream vis1;
            common::VisualizeField(vis1, "localhost", 19916, fitting_error_gf,
                                   "Fitting Error after fitting",
                                   900, 680, 300, 300);
         }

         std::cout << "Integrate fitting error on BG: " << error_bg_sum << " " <<
                   std::endl;
         std::cout << "Max order || Nbr DOFS || Integrate fitting error on BG" <<
                   std::endl;
         std::cout << fespace->GetMaxElementOrder() << " " << fespace->GetNDofs() << " "
                   << error_bg_sum << std::endl;

         // Reduce the orders of the polynom if needed
         // std::cout << reduce_order << " " << iter_pref << " " << pref_order_increase << " k10info\n";
         if (reduce_order && iter_pref > 0 && pref_order_increase > 1)
         {
            int compt_updates(0);
            for (int i=0; i < inter_faces.Size(); i++)
            {
               int ifnum = inter_faces[i];
               int el1 = inter_face_el1[i];
               int el2 = inter_face_el2[i];

               Array<int> els;
               int el_order = fespace->GetElementOrder(el1);
               double interface_error;
               if (surf_bg_mesh)
               {
                  interface_error = ComputeIntegrateErrorBG(x_max_order->FESpace(),
                                                            surf_fit_bg_gf0,
                                                            inter_faces[i],
                                                            surf_fit_gf0_max_order,
                                                            finder);
               }
               else
               {
                  interface_error = ComputeIntegrateError(x_max_order->FESpace(),
                                                          surf_fit_gf0_max_order,
                                                          inter_faces[i]);
               }
               // std::cout << " Current interface error: " << interface_error << " k101\n";

               // KM TODO: make sure that elements stay valid after coarsening.
               double face_coarsening_error;
               int orig_order = el_order;
               int target_order = el_order;
               bool trycoarsening = true;
               while (el_order > mesh_poly_deg+1 && interface_error < pref_tol &&
                      trycoarsening)
               {
                  if (surf_bg_mesh)
                  {
                     face_coarsening_error = InterfaceElementOrderReduction(mesh,
                                                                            inter_faces[i],
                                                                            el_order-1,
                                                                            surf_fit_bg_gf0,
                                                                            finder);
                     // face_coarsening_error =    TestInterfaceElementOrderReduction(mesh,
                     // inter_faces[i], el_order-1, surf_el_meshes, surf_fit_bg_gf0, finder);
                  }
                  else
                  {
                     face_coarsening_error = ComputeIntegrateError(x_max_order->FESpace(),
                                                                   surf_fit_gf0_max_order,
                                                                   inter_faces[i]);
                  }
                  // std::cout << "Interface coarsening error: " << el_order-1 << " " <<
                  // face_coarsening_error << " k101\n";

                  if (face_coarsening_error < pref_tol)
                  {
                     el_order -= 1;
                     target_order = el_order;
                  }
                  else
                  {
                     trycoarsening = false;
                  }
               }

               if (target_order != orig_order)
               {
                  fespace->SetElementOrder(el1, target_order);
                  fespace->SetElementOrder(el2, target_order);
                  order_gf(el1) = target_order;
                  order_gf(el2) = target_order;
                  compt_updates++;
               }
            }

            // Update the FES and GridFunctions only if some orders have been changed
            if (compt_updates > 0)
            {
               PRefinementTransfer preft_fespace = PRefinementTransfer(*fespace);
               PRefinementTransfer preft_surf_fit_fes = PRefinementTransfer(surf_fit_fes);
               // Updates if we increase the order of at least one element
               fespace->Update(false);
               surf_fit_fes.CopySpaceElementOrders(*fespace);
               preft_fespace.Transfer(x);
               preft_fespace.Transfer(x0);
               preft_fespace.Transfer(rdm);
               preft_surf_fit_fes.Transfer(surf_fit_mat_gf);
               preft_surf_fit_fes.Transfer(surf_fit_gf0);
               surf_fit_marker.SetSize(surf_fit_gf0.Size());

               x.SetTrueVector();
               x.SetFromTrueVector();

               mesh->SetNodalGridFunction(&x);

               delete x_max_order;
               x_max_order = ProlongToMaxOrder(&x, 0);
               delete surf_fit_gf0_max_order;
               surf_fit_gf0_max_order = ProlongToMaxOrder(&surf_fit_gf0, 0);
            }
         }
      }

      // Visualize the mesh displacement.
      /*
      if (visualization)
      {
         osockstream sock(19916, "localhost");
         sock << "solution\n";
         mesh->Print(sock);
         x0 -= x;
         delete x_max_order;
         x_max_order = ProlongToMaxOrder(&x0, 0);
         x_max_order->Save(sock);
         sock.send();
         sock << "window_title 'Displacements'\n"
              << "window_geometry "
              << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
              << "keys jRmclA" << endl;
      }
       */
      mesh->SetNodalGridFunction(&x); // Need this for the loop

      iter_pref++; // Update the loop iterator

      delete S;
      delete S_prec;
   }

   if (visualization)
   {
      mesh->SetNodalGridFunction(x_max_order);
      socketstream vis1, vis2, vis3;
      common::VisualizeField(vis1, "localhost", 19916, order_gf,
                             "Polynomial order after p-refinement",
                             1100, 0, 500, 500);
      mesh->SetNodalGridFunction(&x);
      if (surf_bg_mesh)
      {
         common::VisualizeField(vis2, "localhost", 19916, *surf_fit_bg_grad,
                                "Background Mesh - Level Set Gradrient",
                                0, 680, 300, 300);
         common::VisualizeField(vis3, "localhost", 19916, *surf_fit_bg_gf0,
                                "Background Mesh - Level Set",
                                0, 680, 300, 300);
      }

      // Visualization of the mesh and the orders with Paraview
      {
         ParaViewDataCollection paraview_dc("TEST_CLAIRE", mesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(fespace->GetMaxElementOrder());
         paraview_dc.SetCycle(0);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetTime(0.0); // set the time
         paraview_dc.RegisterField("mesh", x_max_order);
         paraview_dc.RegisterField("order", &order_gf);
         paraview_dc.Save();
      }

      {
         ParaViewDataCollection paraview_dc("TEST_CLAIRE", mesh_surf_fit_bg);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(fespace->GetMaxElementOrder());
         paraview_dc.SetCycle(1);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetTime(0.0); // set the time
         paraview_dc.RegisterField("level set", surf_fit_bg_gf0);
         paraview_dc.Save();
      }
   }

   finder.FreeData();

   delete adapt_surface;
   delete adapt_grad_surface;
   delete adapt_hess_surface;
   delete surf_fit_hess;
   delete surf_fit_hess_fes;
   delete surf_fit_bg_hess;
   delete surf_fit_bg_hess_fes;
   delete surf_fit_grad;
   delete surf_fit_grad_fes;
   delete surf_fit_bg_grad;
   delete surf_fit_bg_grad_fes;
   delete surf_fit_bg_gf0;
   delete surf_fit_bg_fes;
   delete surf_fit_bg_fec;
   delete target_c;
   delete metric;
   delete fespace;
   delete fec;
   delete mesh_surf_fit_bg;
   delete mesh;

   return 0;
}
