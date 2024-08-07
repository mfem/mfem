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
real_t circle_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const real_t xc = x(0) - 0.5, yc = x(1) - 0.5;
      const real_t r = sqrt(xc*xc + yc*yc);
      return r-0.25;
   }
   else
   {
      const real_t xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      const real_t r = sqrt(xc*xc + yc*yc + zc*zc);
      return r-0.3;
   }
}

real_t squircle_level_set(const Vector &x)
{
   const int dim = x.Size();
   real_t pv = 4.0;
   if (dim == 2)
   {
      const real_t xc = x(0) - 0.5, yc = x(1) - 0.5;
      return std::pow(xc, pv) + std::pow(yc, pv) - std::pow(0.24, pv);
   }
   else
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      return std::pow(xc, pv) + std::pow(yc, pv) +
             std::pow(zc, pv) - std::pow(0.24, pv);
   }
}

real_t inclined_line(const Vector &x)
{
   real_t xv = x(0), yv = x(1);
   real_t dy = 0.2*(xv-0.0)/1.0;
   return yv - 0.5 - dy;
}


real_t in_circle(const Vector &x, const Vector &x_center, double radius)
{
   Vector x_current = x;
   x_current -= x_center;
   real_t dist = x_current.Norml2();
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

real_t in_trapezium(const Vector &x, real_t a, real_t b, real_t l)
{
   real_t phi_t = x(1) + (a-b)*x(0)/l - a;
   return (phi_t <= 0.0) ? 1.0 : -1.0;
}

real_t in_parabola(const Vector &x, real_t h, real_t k, real_t t)
{
   real_t phi_p1 = (x(0)-h-t/2) - k*x(1)*x(1);
   real_t phi_p2 = (x(0)-h+t/2) - k*x(1)*x(1);
   return (phi_p1 <= 0.0 && phi_p2 >= 0.0) ? 1.0 : -1.0;
}

real_t in_rectangle(const Vector &x, real_t xc, real_t yc, real_t w, real_t h)
{
   real_t dx = std::abs(x(0) - xc);
   real_t dy = std::abs(x(1) - yc);
   return (dx <= w/2 && dy <= h/2) ? 1.0 : -1.0;
}

// Fischer-Tropsch like geometry
real_t reactor(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   real_t in_circle1_val = in_circle(x, x_circle1, 0.2);

   real_t r1 = 0.2;
   real_t r2 = 1.0;
   real_t in_trapezium_val = in_trapezium(x, 0.05, 0.1, r2-r1);

   real_t return_val = max(in_circle1_val, in_trapezium_val);

   real_t h = 0.4;
   real_t k = 2;
   real_t t = 0.15;
   real_t in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   real_t in_rectangle_val = in_rectangle(x, 0.99, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   real_t in_rectangle_val2 = in_rectangle(x, 0.99, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}



real_t r_intersect(real_t r1, real_t r2)
{
   return r1 + r2 - std::pow(r1*r1 + r2*r2, 0.5);
}

real_t r_union(real_t r1, real_t r2)
{
   return r1 + r2 + std::pow(r1*r1 + r2*r2, 0.5);
}

real_t r_remove(real_t r1, real_t r2)
{
   return r_intersect(r1, -r2);
}

double smooth_circle_ls(const Vector &x, const Vector &x_center, double radius)
{
   Vector x_current = x;
   real_t dist = x_current.DistanceSquaredTo(x_center);
   return dist - radius*radius;
}

real_t smooth_inclined_line(const Vector &x)
{
   real_t xv = x(0), yv = x(1);
   real_t dy = 0.0625*(xv-0.0)/1.0;
   return yv - 0.05 - dy;
}

real_t smooth_line(const Vector &x, real_t a, int coord)
{
   return x(coord) - a;
}

real_t smooth_rectangle(const Vector &x, real_t xc, real_t yc, real_t w, real_t h)
{
   real_t leftline = -smooth_line(x, xc-w/2, 0);
   real_t rightline = smooth_line(x, xc+w/2, 0);
   real_t bottomline = -smooth_line(x, yc-h/2, 1);
   real_t topline = smooth_line(x, yc+h/2, 1);
   real_t xdir = r_union(leftline, rightline);
   // return xdir;
   real_t ydir = r_union(topline, bottomline);
   // return ydir;
   real_t return_val = r_union(xdir, ydir);
   return return_val;
}

real_t smooth_parabola(const Vector &x, real_t h, real_t k, real_t t)
{
   real_t phi_p1 = k*x(1)*x(1) - (x(0)-h-t/2);
   real_t phi_p2 = k*x(1)*x(1) - (x(0)-h+t/2);
   // return phi_p1;
   // return -phi_p2;
   return r_union(-phi_p1, phi_p2);
   // return (phi_p1 <= 0.0 && phi_p2 >= 0.0) ? 1.0 : -1.0;
}

// Fischer-Tropsch like geometry
real_t squarewithcorners(const Vector &x)
{
   real_t in_rect = smooth_rectangle(x, 0.5, 0.5, 0.28, 0.28);
   return in_rect;
}

// Fischer-Tropsch like geometry
real_t reactoranalytic(const Vector &x)
{
   // Circle
   Vector x_circle1(2);
   x_circle1(0) = 0.0;
   x_circle1(1) = 0.0;
   real_t in_circle1_val = smooth_circle_ls(x, x_circle1, 0.3);

   real_t in_inclined_line = smooth_inclined_line(x);
   real_t return_val = r_intersect(in_circle1_val, in_inclined_line);

   real_t in_rect = smooth_rectangle(x, 0.99, 0.0, 0.24, 0.4);
   return_val = r_intersect(return_val, in_rect);

   real_t in_par = smooth_parabola(x, 0.4, 2, 0.3);
   return_val = r_intersect(return_val, in_par);

   real_t in_rect2 = smooth_rectangle(x, 0.99, 0.5, 0.24, 0.4);
   return_val = r_intersect(return_val, in_rect2);

   return return_val;

   real_t r1 = 0.2;
   real_t r2 = 1.0;
   real_t in_trapezium_val = in_trapezium(x, 0.05, 0.1, r2-r1);

   // real_t return_val = max(in_circle1_val, in_trapezium_val);

   real_t h = 0.4;
   real_t k = 2;
   real_t t = 0.15;
   real_t in_parabola_val = in_parabola(x, h, k, t);
   return_val = max(return_val, in_parabola_val);

   real_t in_rectangle_val = in_rectangle(x, 0.99, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   real_t in_rectangle_val2 = in_rectangle(x, 0.99, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

real_t mickymouseanalytic(const Vector &coord)
{
   const int num_circ = 3;
   real_t rad[num_circ] = {0.3, 0.15, 0.2};
   real_t c[num_circ][2] = { {0.6, 0.6}, {0.3, 0.3}, {0.25, 0.75} };

   const real_t xc = coord(0), yc = coord(1);

   // circle 0
   Vector xcir(c[0], 2);
   real_t circ0 = smooth_circle_ls(coord, xcir, rad[0]);
   real_t return_val = circ0;

   for (int i = 1; i < num_circ; i++)
   {
      Vector xcir(c[i], 2);
      real_t circ = smooth_circle_ls(coord, xcir, rad[i]);
      return_val = r_intersect(return_val, circ);
   }

   // remove in center
   {
      circ0 = smooth_circle_ls(coord, xcir, 0.125);
      return_val = r_union(return_val, -circ0);
   }

   real_t in_rect = smooth_rectangle(coord, 0.75, 0.25, 0.1, 0.3);
   return_val = r_intersect(return_val, in_rect);

   real_t in_rect2 = smooth_rectangle(coord, 0.55, 0.175, 0.5, 0.05);
   // return_val = r_intersect(return_val, in_rect2);

   return return_val;

      // rectangle 1
      if (0.7 <= xc && xc <= 0.8 && 0.1 <= yc && yc <= 0.8) { return 1.0; }

      // // rectangle 2
      // if (0.3 <= xc && xc <= 0.8 && 0.15 <= yc && yc <= 0.2) { return 1.0; }
      // return -1.0;
}

real_t in_cube(const Vector &x, real_t xc, real_t yc, real_t zc, real_t lx,
               real_t ly, real_t lz)
{
   real_t dx = std::abs(x(0) - xc);
   real_t dy = std::abs(x(1) - yc);
   real_t dz = std::abs(x(2) - zc);
   return (dx <= lx/2 && dy <= ly/2 && dz <= lz/2) ? 1.0 : -1.0;
}

real_t in_pipe(const Vector &x, int pipedir, Vector x_pipe_center,
               real_t radius, real_t minv, real_t maxv)
{
   Vector x_pipe_copy = x_pipe_center;
   x_pipe_copy -= x;
   x_pipe_copy(pipedir-1) = 0.0;
   real_t dist = x_pipe_copy.Norml2();
   real_t xv = x(pipedir-1);
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


real_t r_circle(const Vector &x, Vector &x_center, real_t radius)
{
   real_t xc = x_center(0);
   real_t yc = x_center(1);
   real_t xv = x(0),
          yv = x(1);
   return std::pow(xc-xv, 2.0) + std::pow(yc-yv, 2.0) - std::pow(radius, 2.0);
}

// cool apollo capsule
real_t apollo_level_set(const Vector &x)
{
   real_t xv = x(0),
          yv = x(1);

   //circle 1
   real_t returnval;
   {
      real_t xcc = 3.4306-0.2311;
      real_t ycc = 0.0;
      real_t rcc = 0.2311;
      Vector xc(2);
      xc(0) = xcc;
      xc(1) = ycc;

      returnval = -r_circle(x, xc, rcc);
   }

   // circle 2
   {
      real_t xcc = 0.5543;
      real_t ycc = 1.7602;
      real_t rcc = 0.1956;
      Vector xc(2);
      xc(0) = xcc;
      xc(1) = ycc;
      real_t circle = -r_circle(x, xc, rcc);

      returnval = r_union(returnval, circle);
   }

   //    // circle 3
   {
      real_t xcc = 0.5543;
      real_t ycc = -1.7602;
      real_t rcc = 0.1956;
      Vector xc(2);
      xc(0) = xcc;
      xc(1) = ycc;
      real_t circle = -r_circle(x, xc, rcc);

      returnval = r_union(returnval, circle);
   }

   // big circle
   {
      real_t xcc = 4.6939;
      real_t ycc = 0.0;
      real_t rcc = 4.6939;
      Vector xc(2);
      xc(0) = xcc;
      xc(1) = ycc;
      real_t circle = -r_circle(x, xc, rcc);

      real_t rem1 = yv-1.8368;
      circle = r_remove(circle, rem1);
      real_t rem1b = -1.8368-yv;
      circle = r_remove(circle, rem1b);
      real_t rem2 = 0.6608-xv;
      circle = r_remove(circle, -rem2);
      returnval = r_union(returnval, circle);
   }

   //line
   {
      real_t line1 =  -yv + 0.1938 + std::tan(M_PI - 33.0*(M_PI/180.0))*(xv-3.3254);
      line1 *= 1;

      real_t rem1 = yv-1.9242;
      line1 = r_remove(line1, rem1);
      real_t rem2 = 0.6608 - xv;
      line1 = r_remove(line1, rem2);
      real_t rem3 = 3.3254 - xv;
      line1 = r_remove(line1, -rem3);
      real_t rem4 = -0.1938 - yv;
      line1 = r_remove(line1, rem4);
      returnval = r_union(returnval, line1);
   }

   //line2
   {
      real_t line1 =  -yv - 0.1938 + std::tan(33.0*(M_PI/180.0))*(xv-3.3254);
      line1 *= -1;

      real_t rem1 = -1.9242-yv;
      line1 = r_remove(line1, rem1);
      real_t rem2 = 0.6608-xv;
      line1 = r_remove(line1, rem2);
      real_t rem3 = xv-3.3254;
      line1 = r_remove(line1, rem3);
      real_t rem4 = yv-0.1938;
      line1 = r_remove(line1, rem4);

      returnval = r_union(returnval, line1);
   }

   return returnval;
}


real_t csg_cubecylsph(const Vector &x)
{
   Vector xcc(3);
   xcc = 0.5;
   real_t cube_x = 0.25*2;
   real_t cube_y = 0.25*2;
   real_t cube_z = 0.25*2;

   real_t in_cube_val = in_cube(x, xcc(0), xcc(1), xcc(2), cube_x, cube_y, cube_z);

   Vector x_circle_c(3);
   x_circle_c = 0.5;

   real_t sphere_radius = 0.30;
   real_t in_sphere_val = in_circle(x, x_circle_c, sphere_radius);
   real_t in_return_val = std::min(in_cube_val, in_sphere_val);

   int pipedir = 1;
   Vector x_pipe_center(3);
   x_pipe_center = 0.5;
   real_t xmin = 0.5-sphere_radius;
   real_t xmax = 0.5+sphere_radius;
   real_t pipe_radius = 0.075;
   real_t in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);

   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   pipedir = 2;
   in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   pipedir = 3;
   in_pipe_x = in_pipe(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   in_return_val = std::min(in_return_val, -1*in_pipe_x);

   return in_return_val;
}

void ModifyAttributeForMarkingDOFS(Mesh *mesh, GridFunction &mat,
                                   int attr_to_switch)
{
   // Switch attribute if all but 1 of the faces of an element will be marked?
   Array<int> element_attr(mesh->GetNE());
   element_attr = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      Array<int> faces, ori;
      if (mesh->Dimension() == 2)
      {
         mesh->GetElementEdges(e, faces, ori);
      }
      else
      {
         mesh->GetElementFaces(e, faces, ori);
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
         mesh->GetFaceElements(faces[f], &elem1, &elem2);
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
            mesh->GetFaceInfos(faces[f], &inf1, &inf2);
            if (inf2 >= 0)
            {
               Vector dof_vals;
               Array<int> dofs;
               mat.GetElementDofValues(mesh->GetNE() + (-1-elem2), dof_vals);
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
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mat(e) = element_attr[e];
      mesh->SetAttribute(e, element_attr[e]+1);
   }
   mesh->SetAttributes();
}

void MakeGridFunctionWithNumberOfInterfaceFaces(
   mfem::Mesh *mesh,
   mfem::GridFunction &mat,
   mfem::GridFunction &NumFaces)
{
   NumFaces = 0.0;

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      mfem::Array<int> faces, ori;
      mfem::Array<int> faces_ele2, ori_ele2;
      mfem::Array<int> faces_ele1, ori_ele1;
      if (mesh->Dimension() == 2)
      {
         mesh->GetElementEdges(e, faces, ori);
      }
      else
      {
         mesh->GetElementFaces(e, faces, ori);
      }
      int inf1, inf2;
      int elem1, elem2;
      int attr1;
      int attr2;
      attr1 = mat(e);

      for (int f = 0; f < faces.Size(); f++)
      {
         mesh->GetFaceElements(faces[f], &elem1, &elem2);

         if (elem2 >= 0)
         {
            attr1 = (elem1 == e) ? static_cast<int>(mat(elem1)) : static_cast<int>(mat(
                                                                                      elem2));
            attr2 = (elem1 == e) ? static_cast<int>(mat(elem2)) : static_cast<int>(mat(
                                                                                      elem1));

            if (attr1 != attr2 )
            {
               NumFaces[e] += 1;
            }
         }
         else
         {
            mesh->GetFaceInfos(faces[f], &inf1, &inf2);
            if (inf2 >= 0)
            {
               mfem::Vector dof_vals;
               mfem::Array<int> dofs;
               mat.GetElementDofValues(mesh->GetNE() + (-1-elem2), dof_vals);

               attr1 = mat(e);
               attr2 = static_cast<int>(dof_vals(0));

               if (attr1 != attr2 )
               {
                  NumFaces[e] += 1;
               }
            }
            else
            {
               // bdr_element = true;
            }
         }
      }
   }

   int counter = 0;
   for (int e = 0; e < mesh->GetNE(); e++)
   {
      counter += (NumFaces(e) > 1);
   }
   std::cout<<"number of element with more than 1 face for fitting: "<<counter<<std::endl;
}

void ModifyBoundaryAttributesForNodeMovement(Mesh *mesh, GridFunction &x)
{
   const int dim = mesh->Dimension();
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      mfem::Array<int> dofs;
      mesh->GetNodalFESpace()->GetBdrElementDofs(i, dofs);
      mfem::Vector bdr_xy_data;
      mfem::Vector dof_xyz(dim);
      mfem::Vector dof_xyz_compare;
      mfem::Array<int> xyz_check(dim);
      for (int j = 0; j < dofs.Size(); j++)
      {
         for (int d = 0; d < dim; d++)
         {
            dof_xyz(d) = x(mesh->GetNodalFESpace()->DofToVDof(dofs[j], d));
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
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
      else if (dim == 3)
      {
         if (xyz_check[0] == dofs.Size())
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 1);
         }
         else if (xyz_check[1] == dofs.Size())
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 2);
         }
         else if (xyz_check[2] == dofs.Size())
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 3);
         }
         else
         {
            mesh->GetNodalFESpace()->GetMesh()->SetBdrAttribute(i, 4);
         }
      }
   }
}

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

   void Update(bool hp = false);
};

void HRefUpdater::Update(bool hp)
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
      auto R = gridfuncarr[i]->FESpace()->GetHpRestrictionMatrix();
      if (hp && gridfuncarr[i]->FESpace()->IsVariableOrder() && R)
      {
         Vector xnew(R->Height());
         R->Mult(*gridfuncarr[i], xnew);
         gridfuncarr[i]->FESpace()->GetProlongationMatrix()->Mult(xnew, *gridfuncarr[i]);
      }
      else
      {
         gridfuncarr[i]->SetTrueVector();
         gridfuncarr[i]->SetFromTrueVector();
      }
   }
}

#ifdef MFEM_USE_MPI
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
               if (std::abs(dof_xyz(d)-dof_xyz_compare(d)) < 1.e-10)
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
                                           Array<ParGridFunction *> *pgf_to_update = NULL,
                                           int ref_neighbors = 2)
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
         real_t min_val = x_vals.Min();
         real_t max_val = x_vals.Max();
         // If the zero level set cuts the elements, mark it for refinement
         if (min_val < 0 && max_val >= 0)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < ref_neighbors; inner_iter++)
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
            real_t max_val = x_vals.Max();
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
                                       const int pLapNewton = 50,
                                       const int solver_type = 0)
{
   mfem::H1_FECollection h1fec(distance_s.ParFESpace()->FEColl()->GetOrder(),
                               pmesh.Dimension());
   mfem::ParFiniteElementSpace h1fespace(&pmesh, &h1fec);
   mfem::ParGridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);
   x.ExchangeFaceNbrData();

   //Now determine distance
   const real_t dx = AvgElementSize(pmesh);
   DistanceSolver *dist_solver = NULL;
   if (solver_type == 0)
   {
      dist_solver = new PLapDistanceSolver(pLapOrder, pLapNewton);
   }
   else
   {
      dist_solver = new NormalizationDistanceSolver();
   }
   dist_solver->print_level.Summary();
   //   PLapDistanceSolver dist_solver(pLapOrder, pLapNewton);
   //   NormalizationDistanceSolver dist_solver;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver->ComputeScalarDistance(ls_filt_coeff, &distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, nDiffuse);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
   delete dist_solver;
}
#endif

void OptimizeMeshWithAMRAroundZeroLevelSet(Mesh &mesh,
                                           FunctionCoefficient &ls_coeff,
                                           int amr_iter,
                                           GridFunction &distance_s,
                                           const int quad_order = 5,
                                           Array<GridFunction *> *gf_to_update = NULL)
{
   mfem::H1_FECollection h1fec(distance_s.FESpace()->FEColl()->GetOrder(),
                               mesh.Dimension());
   mfem::FiniteElementSpace h1fespace(&mesh, &h1fec);
   mfem::GridFunction x(&h1fespace);

   mfem::L2_FECollection l2fec(0, mesh.Dimension());
   mfem::FiniteElementSpace l2fespace(&mesh, &l2fec);
   mfem::GridFunction el_to_refine(&l2fespace);

   mfem::H1_FECollection lhfec(1, mesh.Dimension());
   mfem::FiniteElementSpace lhfespace(&mesh, &lhfec);
   mfem::GridFunction lhx(&lhfespace);

   x.ProjectCoefficient(ls_coeff);

   IntegrationRules irRules = IntegrationRules(0, Quadrature1D::GaussLobatto);
   for (int iter = 0; iter < amr_iter; iter++)
   {
      el_to_refine = 0.0;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         DenseMatrix x_grad;
         h1fespace.GetElementDofs(e, dofs);
         const IntegrationRule &ir = irRules.Get(mesh.GetElementGeometry(e),
                                                 quad_order);
         x.GetValues(e, ir, x_vals);
         double min_val = x_vals.Min();
         double max_val = x_vals.Max();
         // If the zero level set cuts the elements, mark it for refinement
         if (min_val <= 0 && max_val >= 0)
         {
            el_to_refine(e) = 1.0;
         }
      }

      // Refine an element if its neighbor will be refined
      for (int inner_iter = 0; inner_iter < 2; inner_iter++)
      {
         GridFunctionCoefficient field_in_dg(&el_to_refine);
         lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
         for (int e = 0; e < mesh.GetNE(); e++)
         {
            Array<int> dofs;
            Vector x_vals;
            lhfespace.GetElementDofs(e, dofs);
            const IntegrationRule &ir =
               irRules.Get(mesh.GetElementGeometry(e), quad_order);
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
      if (glob_count > 0)
      {
         mesh.GeneralRefinement(el_to_refine_list, 1);
      }

      // Update
      h1fespace.Update();
      x.Update();
      x.ProjectCoefficient(ls_coeff);

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.FESpace()->Update();
      distance_s.Update();

      if (gf_to_update != NULL)
      {
         for (int i = 0; i < gf_to_update->Size(); i++)
         {
            (*gf_to_update)[i]->FESpace()->Update();
            (*gf_to_update)[i]->Update();
         }
      }
   }
}

void ComputeScalarDistanceFromLevelSet(Mesh &mesh,
                                       FunctionCoefficient &ls_coeff,
                                       GridFunction &distance_s,
                                       const int nDiffuse = 2,
                                       const int pLapOrder = 5,
                                       const int pLapNewton = 50,
                                       bool nfilter = true,
                                       int solvertype = 0)
{
   mfem::H1_FECollection h1fec(distance_s.FESpace()->FEColl()->GetOrder(),
                               mesh.Dimension());
   mfem::FiniteElementSpace h1fespace(&mesh, &h1fec);
   mfem::GridFunction x(&h1fespace);

   x.ProjectCoefficient(ls_coeff);

   //Now determine distance
   const double dx = AvgElementSize(mesh);
   DistanceSolver *dist_solver;
   if (solvertype == 0)
   {
      dist_solver = new PLapDistanceSolver(pLapOrder, pLapNewton);
   }
   else
   {
      dist_solver = new NormalizationDistanceSolver();
   }
   //   PLapDistanceSolver dist_solver(pLapOrder, pLapNewton);
   //      NormalizationDistanceSolver dist_solver;

   FiniteElementSpace pfes_s(*distance_s.FESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   GridFunction filt_gf(&pfes_s);
   PDEFilter filter(mesh, nfilter ? 1.0 * dx : 0.0);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver->ComputeScalarDistance(ls_filt_coeff, &distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, nDiffuse);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
}

double pipe_dist(const Vector &x, int pipedir, Vector x_pipe_center,
                 double radius, double minv, double maxv)
{
   Vector x_pipe_copy = x_pipe_center;
   x_pipe_copy -= x;
   x_pipe_copy(pipedir-1) = 0.0;
   double dist = x_pipe_copy.Norml2() - radius*radius;
   return dist;
}

double cube_dist_smooth(const Vector &x, Vector &x_center, Vector &lengths)
{
   double xc = x_center(0);
   double yc = x_center(1);
   double zc = x_center(2);
   double xv = x(0),
          yv = x(1),
          zv = x(2);
   double lx = lengths(0);
   double ly = lengths(1);
   double lz = lengths(2);
   double phi_vertical = std::pow(lx*0.5, 2.0) - std::pow(xv-xc, 2.0);
   double phi_horizontal = std::pow(ly*0.5, 2.0) - std::pow(yv-yc, 2.0);
   double phi_3 = std::pow(lz*0.5, 2.0) - std::pow(zv-zc, 2.0);

   double phi_rectangle = r_intersect(phi_vertical, phi_horizontal);
   return r_intersect(phi_rectangle, phi_3);
}

double csg_cubecylsph_smooth(const Vector &x)
{
   //   double pwrr = 4.0;
   Vector xcc = x;
   xcc = 0.5;
   const int dim = x.Size();
   // const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
   MFEM_VERIFY(dim == 3, "Only 3D supported for this level set");
   Vector x2 = x;
   x2 -= xcc;
   double rsph = 0.375;
   double rcube = 0.3; // 0.325
   double dsph = x2.Norml2() - rsph;
   //   double dcube = std::pow(xc*xc + yc*yc + zc*zc, 0.5) - rcube;
   Vector rcubel(3);
   rcubel = 2*rcube;
   double dcube = cube_dist_smooth(x, xcc, rcubel);
   //    return std::max(dsph, dcube); // return here for sphere + cube
   double dist1 = r_union(dsph, -dcube);
   //   return dist1;

   int pipedir = 1;
   Vector x_pipe_center(3);
   x_pipe_center = 0.5;
   double xmin = 1.0-rsph;
   double xmax = 1.0+rsph;
   double pipe_radius = 0.25;
   double in_pipe_x = pipe_dist(x, pipedir, x_pipe_center, pipe_radius, xmin,
                                xmax);
   //   double dist2 = std::max(dist1, -in_pipe_x);
   double dist2 = r_union(dist1, -in_pipe_x);

   pipedir = 2;
   in_pipe_x = pipe_dist(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   //   double dist3 = std::max(dist2, -in_pipe_x);
   double dist3 = r_union(dist2, -in_pipe_x);

   pipedir = 3;
   in_pipe_x = pipe_dist(x, pipedir, x_pipe_center, pipe_radius, xmin, xmax);
   //   double dist4 = std::max(dist3, -in_pipe_x);
   double dist4 = r_union(dist3, -in_pipe_x);

   return dist4;
}
