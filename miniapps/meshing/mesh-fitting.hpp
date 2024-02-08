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

#include "mesh-optimizer.hpp"
#include "../common/mfem-common.hpp"

using namespace std;
using namespace mfem;
using namespace common;

// Used for exact surface alignment
double circle_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      return r-0.25;
   }
   else
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      const double r = sqrt(xc*xc + yc*yc + zc*zc);
      return r-0.3;
   }
}

double squircle_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      return std::pow(xc, 4.0) + std::pow(yc, 4.0) - std::pow(0.24, 4.0);
   }
   else
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      return std::pow(xc, 4.0) + std::pow(yc, 4.0) +
             std::pow(zc, 4.0) - std::pow(0.24, 4.0);
   }
}

double in_circle(const Vector &x, const Vector &x_center, double radius)
{
   Vector x_current = x;
   x_current -= x_center;
   double dist = x_current.Norml2();
   if (dist < radius)
   {
      return 1.0;
   }
   else if (dist == radius)
   {
      return 0.0;
   }

   return -1.0;
}

double in_trapezium(const Vector &x, double a, double b, double l)
{
   double phi_t = x(1) + (a-b)*x(0)/l - a;
   return (phi_t <= 0.0) ? 1.0 : -1.0;
}

double in_parabola(const Vector &x, double h, double k, double t)
{
   double phi_p1 = (x(0)-h-t/2) - k*x(1)*x(1);
   double phi_p2 = (x(0)-h+t/2) - k*x(1)*x(1);
   return (phi_p1 <= 0.0 && phi_p2 >= 0.0) ? 1.0 : -1.0;
}

double in_rectangle(const Vector &x, double xc, double yc, double w, double h)
{
   double dx = fabs(x(0) - xc);
   double dy = fabs(x(1) - yc);
   return (dx <= w/2 && dy <= h/2) ? 1.0 : -1.0;
}

// Fischer-Tropsch like geometry
double reactor(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   double in_circle1_val = in_circle(x, x_circle1, 0.2);

   double r1 = 0.2;
   double r2 = 1.0;
   double in_trapezium_val = in_trapezium(x, 0.05, 0.1, r2-r1);

   double return_val = max(in_circle1_val, in_trapezium_val);

   double h = 0.4;
   double k = 2;
   double t = 0.15;
   double in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   double in_rectangle_val = in_rectangle(x, 0.99, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   double in_rectangle_val2 = in_rectangle(x, 0.99, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

double in_cube(const Vector &x, double xc, double yc, double zc, double lx,
               double ly, double lz)
{
   double dx = fabs(x(0) - xc);
   double dy = fabs(x(1) - yc);
   double dz = fabs(x(2) - zc);
   return (dx <= lx/2 && dy <= ly/2 && dz <= lz/2) ? 1.0 : -1.0;
}

double in_pipe(const Vector &x, int pipedir, Vector x_pipe_center,
               double radius, double minv, double maxv)
{
   Vector x_pipe_copy = x_pipe_center;
   x_pipe_copy -= x;
   x_pipe_copy(pipedir-1) = 0.0;
   double dist = x_pipe_copy.Norml2();
   double xv = x(pipedir-1);
   if (dist < radius && xv > minv && xv < maxv)
   {
      return 1.0;
   }
   else if (dist == radius || (xv == minv && dist < radius) || (xv == maxv &&
                                                                dist < radius))
   {
      return 0.0;
   }

   return -1.0;
}

double r_intersect(double r1, double r2)
{
   return r1 + r2 - std::pow(r1*r1 + r2*r2, 0.5);
}

double r_union(double r1, double r2)
{
   return r1 + r2 + std::pow(r1*r1 + r2*r2, 0.5);
}

double r_remove(double r1, double r2)
{
   return r_intersect(r1, -r2);
}

double csg_cubecylsph(const Vector &x)
{
   Vector xcc(3);
   xcc = 0.5;
   double cube_x = 0.25*2;
   double cube_y = 0.25*2;
   double cube_z = 0.25*2;

   double in_cube_val = in_cube(x, xcc(0), xcc(1), xcc(2), cube_x, cube_y, cube_z);

   Vector x_circle_c(3);
   x_circle_c = 0.5;

   double sphere_radius = 0.30;
   double in_sphere_val = in_circle(x, x_circle_c, sphere_radius);
   double in_return_val = std::min(in_cube_val, in_sphere_val);

   int pipedir = 1;
   Vector x_pipe_center(3);
   x_pipe_center = 0.5;
   double xmin = 0.5-sphere_radius;
   double xmax = 0.5+sphere_radius;
   double pipe_radius = 0.075;
   double in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);

   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   pipedir = 2;
   in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   pipedir = 3;
   in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   return in_return_val;
}

// Generic class to store pointers to finite element spaces and gridfunctions,
// and update them post h-refinement.
class HRefUpdater
{
protected:
   Array<GridFunction *> gridfuncarr;
   Array<FiniteElementSpace *> fespacearr;

public:
   HRefUpdater() {}

   void AddGridFunctionForUpdate(GridFunction *gf_)
   {
      gridfuncarr.Append(gf_);
   }
   void AddFESpaceForUpdate(FiniteElementSpace *fes_)
   {
      fespacearr.Append(fes_);
   }

   void Update();
};

void HRefUpdater::Update()
{
   // Update FESpace
   for (int i = 0; i < fespacearr.Size(); i++)
   {
      fespacearr[i]->Update();
   }
   // Update GF
   for (int i = 0; i < gridfuncarr.Size(); i++)
   {
      gridfuncarr[i]->Update();
      gridfuncarr[i]->SetTrueVector();
      gridfuncarr[i]->SetFromTrueVector();
   }
}

#ifdef MFEM_USE_MPI
// Take a given list of elements and appends element neighbors to the list.
void ExtendRefinementListToNeighbors(ParMesh &pmesh, Array<int> &intel)
{
   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);
   const int quad_order = 4;

   el_to_refine = 0.0;

   for (int i = 0; i < intel.Size(); i++)
   {
      el_to_refine(intel[i]) = 1.0;
   }

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   el_to_refine.ExchangeFaceNbrData();
   GridFunctionCoefficient field_in_dg(&el_to_refine);
   lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      Array<int> dofs;
      Vector x_vals;
      lhfespace.GetElementDofs(e, dofs);
      const IntegrationRule &ir =
         irRules.Get(pmesh.GetElementGeometry(e), quad_order);
      lhx.GetValues(e, ir, x_vals);
      double max_val = x_vals.Max();
      if (max_val > 0)
      {
         intel.Append(e);
      }
   }

   intel.Sort();
   intel.Unique();
}

// Get material interface elements, face/edge numbers, or DOF indices based on
// user parameter "type".
// type = 0, elements on either side of interface.
// type = 1, face/edge indices
// type = 2, dofs (requires corresponding fespace)
void GetMaterialInterfaceEntities(ParMesh *pmesh, ParGridFunction &mat,
                                  Array<int> &intelf,
                                  int type = 0,
                                  ParFiniteElementSpace *dofspace = NULL)
{
   intelf.SetSize(0);
   mat.ExchangeFaceNbrData();
   const int NElem = pmesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");

   if (type == 2)
   {
      MFEM_VERIFY(dofspace, "dof list requires fepsace");
   }

   ParGridFunction *surf_fit_dofs_gf = NULL;
   if (dofspace)
   {
      surf_fit_dofs_gf = new ParGridFunction(dofspace);
      *surf_fit_dofs_gf = 0.0;
   }

   for (int f = 0; f < pmesh->GetNumFaces(); f++ )
   {
      Array<int> nbrs;
      pmesh->GetFaceAdjacentElements(f,nbrs);
      Vector matvals;
      Array<int> vdofs;
      Vector vec;
      Array<int> els;
      //if there is more than 1 element across the face.
      if (nbrs.Size() > 1)
      {
         matvals.SetSize(nbrs.Size());
         for (int j = 0; j < nbrs.Size(); j++)
         {
            if (nbrs[j] < NElem)
            {
               matvals(j) = mat(nbrs[j]);
               els.Append(nbrs[j]);
            }
            else
            {
               const int Elem2NbrNo = nbrs[j] - NElem;
               mat.ParFESpace()->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs);
               mat.FaceNbrData().GetSubVector(vdofs, vec);
               matvals(j) = vec(0);
            }
         }
         if (matvals(0) != matvals(1))
         {
            if (type == 0) { intelf.Append(els); }
            else if (type == 1) { intelf.Append(f); }
            else if (type == 2)
            {
               Array<int> dofs;
               dofspace->GetFaceDofs(f, dofs);
               Vector vals(dofs.Size());
               vals = 1.0;
               surf_fit_dofs_gf->SetSubVector(dofs, vals);
            }
         }
      }
   }

   if (type == 2)
   {
      // unify across procesor boundaries
      surf_fit_dofs_gf->ExchangeFaceNbrData();
      {
         GroupCommunicator &gcomm = surf_fit_dofs_gf->ParFESpace()->GroupComm();
         Array<double> gf_array(surf_fit_dofs_gf->GetData(), surf_fit_dofs_gf->Size());
         gcomm.Reduce<double>(gf_array, GroupCommunicator::Max);
         gcomm.Bcast(gf_array);
         surf_fit_dofs_gf->ExchangeFaceNbrData();
      }

      for (int i = 0; i < surf_fit_dofs_gf->Size(); i++)
      {
         if ((*surf_fit_dofs_gf)(i) > 0.0)
         {
            intelf.Append(i);
         }
      }
   }
   if (dofspace) { delete surf_fit_dofs_gf; }
}

// Set boundary attributes to 1/2/3 if the boundary element is parallel to x/y/z.
// Otherwise sets the attribute to 4.
void ModifyBoundaryAttributesForNodeMovement(ParMesh *pmesh, ParGridFunction &x)
{
   const int dim = pmesh->Dimension();
   for (int i = 0; i < pmesh->GetNBE(); i++)
   {
      mfem::Array<int> dofs;
      pmesh->GetNodalFESpace()->GetBdrElementDofs(i, dofs);
      mfem::Vector bdr_xy_data;
      mfem::Vector dof_xyz(dim);
      mfem::Vector dof_xyz_compare;
      mfem::Array<int> xyz_check(dim);
      for (int j = 0; j < dofs.Size(); j++)
      {
         for (int d = 0; d < dim; d++)
         {
            dof_xyz(d) = x(pmesh->GetNodalFESpace()->DofToVDof(dofs[j], d));
         }
         if (j == 0)
         {
            dof_xyz_compare = dof_xyz;
            xyz_check = 1;
         }
         else
         {
            for (int d = 0; d < dim; d++)
            {
               if (std::fabs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
               {
                  xyz_check[d] += 1;
               }
            }
         }
      }
      if (dim == 2)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
      else if (dim == 3)
      {
         if (xyz_check[0] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else if (xyz_check[2] == dofs.Size())
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
         else
         {
            pmesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
   }
}

void ModifyAttributeForMarkingDOFS(ParMesh *pmesh, ParGridFunction &mat,
                                   int attr_to_switch)
{
   mat.ExchangeFaceNbrData();
   // Switch attribute if all but 1 of the faces of an element will be marked?
   Array<int> element_attr(pmesh->GetNE());
   element_attr = 0;
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      Array<int> faces, ori;
      if (pmesh->Dimension() == 2)
      {
         pmesh->GetElementEdges(e, faces, ori);
      }
      else
      {
         pmesh->GetElementFaces(e, faces, ori);
      }
      int inf1, inf2;
      int elem1, elem2;
      int diff_attr_count = 0;
      int attr1;
      int attr2;
      attr1 = mat(e);
      bool bdr_element = false;
      element_attr[e] = attr1;
      int target_attr = -1;
      for (int f = 0; f < faces.Size(); f++)
      {
         pmesh->GetFaceElements(faces[f], &elem1, &elem2);
         if (elem2 >= 0)
         {
            attr2 = elem1 == e ? (int)(mat(elem2)) : (int)(mat(elem1));
            if (attr1 != attr2 && attr1 == attr_to_switch)
            {
               diff_attr_count += 1;
               target_attr = attr2;
            }
         }
         else
         {
            pmesh->GetFaceInfos(faces[f], &inf1, &inf2);
            if (inf2 >= 0)
            {
               Vector dof_vals;
               Array<int> dofs;
               mat.GetElementDofValues(pmesh->GetNE() + (-1-elem2), dof_vals);
               attr2 = (int)(dof_vals(0));
               if (attr1 != attr2 && attr1 == attr_to_switch)
               {
                  diff_attr_count += 1;
                  target_attr = attr2;
               }
            }
            else
            {
               bdr_element = true;
            }
         }
      }

      if (diff_attr_count == faces.Size()-1 && !bdr_element)
      {
         element_attr[e] = target_attr;
      }
   }
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      mat(e) = element_attr[e];
      pmesh->SetAttribute(e, element_attr[e]+1);
   }
   mat.ExchangeFaceNbrData();
   pmesh->SetAttributes();
}

void OptimizeMeshWithAMRAroundZeroLevelSet(ParMesh &pmesh,
                                           FunctionCoefficient &ls_coeff,
                                           int amr_iter,
                                           ParGridFunction &distance_s,
                                           const int quad_order = 5,
                                           Array<ParGridFunction *> *pgf_to_update = NULL)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   mfem::L2_FECollection l2fec(0, pmesh.Dimension());
   mfem::ParFiniteElementSpace l2fespace(&pmesh, &l2fec);
   mfem::ParGridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, pmesh.Dimension());
   mfem::ParFiniteElementSpace lhfespace(&pmesh, &lhfec);
   mfem::ParGridFunction lhx(&lhfespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int iter = 0; iter < amr_iter; iter++)
   {
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         DenseMatrix x_grad;
         h1fespace.GetElementDofs(e, dofs);
         const IntegrationRule &ir = irRules.Get(pmesh.GetElementGeometry(e),
                                                 quad_order);
         x.GetValues(e, ir, x_vals);
         double min_val = x_vals.Min();
         double max_val = x_vals.Max();
         // If the zero level set cuts the elements, mark it for refinement
         if (min_val < 0 && max_val >= 0)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < 2; inner_iter++)
      {
         el_to_refine.ExchangeFaceNbrData();
         GridFunctionCoefficient field_in_dg(&el_to_refine);
         lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
         for (int e = 0; e < pmesh.GetNE(); e++)
         {
            Array<int> dofs;
            Vector x_vals;
            lhfespace.GetElementDofs(e, dofs);
            const IntegrationRule &ir =
               irRules.Get(pmesh.GetElementGeometry(e), quad_order);
            lhx.GetValues(e, ir, x_vals);
            double max_val = x_vals.Max();
            if (max_val > 0)
            {
               el_to_refine(e) = 1.0;
            }
         }
      }

      // Make the list of elements to be refined
      Array<int> el_to_refine_list;
      for (int e = 0; e < el_to_refine.Size(); e++)
      {
         if (el_to_refine(e) > 0.0)
         {
            el_to_refine_list.Append(e);
         }
      }

      int loc_count = el_to_refine_list.Size();
      int glob_count = loc_count;
      MPI_Allreduce(&loc_count, &glob_count, 1, MPI_INT, MPI_SUM,
                    pmesh.GetComm());
      MPI_Barrier(pmesh.GetComm());
      if (glob_count > 0)
      {
         pmesh.GeneralRefinement(el_to_refine_list, 1);
      }

      // Update
      h1fespace.Update();
      x.Update();
      x.ProjectCoefficient(ls_coeff);

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.ParFESpace()->Update();
      distance_s.Update();

      if (pgf_to_update != NULL)
      {
         for (int i = 0; i < pgf_to_update->Size(); i++)
         {
            (*pgf_to_update)[i]->ParFESpace()->Update();
            (*pgf_to_update)[i]->Update();
         }
      }
   }
}

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       FunctionCoefficient &ls_coeff,
                                       ParGridFunction &distance_s,
                                       const int nDiffuse = 2,
                                       const int pLapOrder = 5,
                                       const int pLapNewton = 50)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   //Now determine distance
   const double dx = AvgElementSize(pmesh);
   PLapDistanceSolver dist_solver(pLapOrder, pLapNewton);

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver.ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, nDiffuse);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
}

// Set material function based on element attribute or using level-set function.
void SetMaterialGridFunction(ParMesh *pmesh, ParGridFunction &mat,
                             ParGridFunction &surf_fit_gf0,
                             bool material, bool adapt_marking)
{
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      if (material)
      {
         mat(i) = pmesh->GetAttribute(i)-1;
      }
      else
      {
         mat(i) = material_id(i, surf_fit_gf0);
         pmesh->SetAttribute(i, mat(i) + 1);
      }
   }
   mat.ExchangeFaceNbrData();
   pmesh->SetAttributes();

   // Adapt attributes for marking such that if all but 1 face of an element
   // are marked, the element attribute is switched.
   if (adapt_marking)
   {
      ModifyAttributeForMarkingDOFS(pmesh, mat, 0);
      ModifyAttributeForMarkingDOFS(pmesh, mat, 1);
   }
   pmesh->ExchangeFaceNbrData();
}

// Check whether all children across a parent face have the same material.
int CheckMaterialConsistency(ParMesh *pmesh, ParGridFunction &mat)
{
   mat.ExchangeFaceNbrData();
   const int NElem = pmesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   int matcheck = 1;
   int pass = 1;
   for (int f = 0; f < pmesh->GetNumFaces(); f++ )
   {
      Array<int> nbrs;
      pmesh->GetFaceAdjacentElements(f,nbrs);
      Vector matvals;
      Array<int> vdofs;
      Vector vec;
      //if there is more than 1 element across the face.
      matvals.SetSize(nbrs.Size()-1);
      if (nbrs.Size() > 2)
      {
         for (int j = 1; j < nbrs.Size(); j++)
         {
            if (nbrs[j] < NElem)
            {
               matvals(j-1) = mat(nbrs[j]);
            }
            else
            {
               const int Elem2NbrNo = nbrs[j] - NElem;
               mat.ParFESpace()->GetFaceNbrElementVDofs(Elem2NbrNo, vdofs);
               mat.FaceNbrData().GetSubVector(vdofs, vec);
               matvals(j-1) = vec(0);
            }
         }
         double minv = matvals.Min(),
                maxv = matvals.Max();
         matcheck = minv == maxv;
      }
      if (matcheck == 0) { pass = 0; break; }
   }
   int global_pass = pass;
   MPI_Allreduce(&pass, &global_pass, 1, MPI_INT, MPI_MIN,
                 pmesh->GetComm());
   return global_pass;
}

void GetBoundaryElements(ParMesh *pmesh, Array<int> &bel, int attr)
{
   bel.SetSize(0);
   int el, info;
   for (int f = 0; f < pmesh->GetNBE(); f++ )
   {
      pmesh->GetBdrElementAdjacentElement(f, el, info);
      if (pmesh->GetBdrAttribute(f) == attr)
      {
         bel.Append(el);
      }
   }
}
#endif
