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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "mesh-optimizer.hpp"
using namespace std;
using namespace mfem;

// Used for exact surface alignment
double circle_level_set(const Vector &x)
{
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r = sqrt(xc*xc + yc*yc);
      return r-0.3; // circle of radius 0.1
   }
   else
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5, zc = x(2) - 0.5;
      const double r = sqrt(xc*xc + yc*yc + zc*zc);
      return r-0.3;
   }
}

double donut_level_set(const Vector &coord)
{
   MFEM_VERIFY(coord.Size() == 3,"Donut level set for 3D only.");
   // map [0,1] to [-1,1].
   double x = 2*coord(0)-1.0, y = 2*coord(1)-1.0, z = 2*coord(2)-1.0;

   bool doughnut;
   const double R = 0.8, r = 0.15;
   const double t = R - std::sqrt(x*x + y*y);
   doughnut = t*t + z*z - r*r <= 0;

   return (doughnut) ? 1.0 : -1.0;
}

double linear_level_set(const Vector &x)
{
   double y_line = 0.75 - 0.5*x(0);
   return std::pow(y_line-x(1),1.0);
}

double squircle_level_set(const Vector &x)
{
   double power = 4.0;
   const int dim = x.Size();
   if (dim == 2)
   {
      const double xc = x(0) - 0.5, yc = x(1) - 0.5;
      const double r2 = pow(xc, power) + pow(yc, power);
      return r2 - pow(0.1, power);
   }
   else
   {
      MFEM_ABORT("Squircle level set implemented for only 2D right now.");
      return 0.0;
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
   else
   {
      return -1.0;
   }
   return 0.0;
}

double in_trapezium(const Vector &x, double a, double b, double l)
{
   double phi_t = x(1) + (a-b)*x(0)/l - a;
   if (phi_t <= 0.0)
   {
      return 1.0;
   }
   return -1.0;
}

double in_parabola(const Vector &x, double h, double k, double t)
{
   double phi_p1 = (x(0)-h-t/2) - k*x(1)*x(1);
   double phi_p2 = (x(0)-h+t/2) - k*x(1)*x(1);
   if (phi_p1 <= 0.0 && phi_p2 >= 0.0)
   {
      return 1.0;
   }
   return -1.0;
}

double in_rectangle(const Vector &x, double xc, double yc, double w, double h)
{
   double dx = fabs(x(0) - xc);
   double dy = fabs(x(1) - yc);
   if (dx <= w/2 && dy <= h/2)
   {
      return 1.0;
   }
   else
   {
      return -1.0;
   }
}

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

   double in_rectangle_val = in_rectangle(x, 1.0, 0.0, 0.12, 0.35);
   return_val = max(return_val, in_rectangle_val);

   double in_rectangle_val2 = in_rectangle(x, 1.0, 0.5, 0.12, 0.28);
   return_val = max(return_val, in_rectangle_val2);
   return return_val;
}

double in_cube(const Vector &x, double xc, double yc, double zc, double lx, double ly, double lz)
{
   double dx = fabs(x(0) - xc);
   double dy = fabs(x(1) - yc);
   double dz = fabs(x(2) - zc);
   if (dx <= lx/2 && dy <= ly/2 && dz <= lz/2)
   {
      return 1.0;
   }
   else
   {
      return -1.0;
   }
}

double in_pipe(const Vector &x, int pipedir, Vector x_pipe_center, double radius, double minv, double maxv)
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
    else if (dist == radius || (xv == minv && dist < radius) || (xv == maxv && dist < radius))
    {
       return 0.0;
    }
    else
    {
       return -1.0;
    }
    return 0.0;
}

double object_three(const Vector &x)
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

#ifdef MFEM_USE_MPI
Mesh* TrimMesh(Mesh &mesh, FunctionCoefficient &ls_coeff, int order,
               int attr_to_trim)
{
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes_s(&mesh, &fec);
   GridFunction distance_s(&fes_s);
   distance_s.ProjectCoefficient(ls_coeff);
   L2_FECollection mat_coll(0, dim);
   FiniteElementSpace mat_fes(&mesh, &mat_coll);
   GridFunction mat(&mat_fes);

   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.SetAttribute(e, 1);
   }
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mat(e) = material_id(e, distance_s);
      mesh.SetAttribute(e, mat(e) + 1);
   }

   ModifyAttributeForMarkingDOFS(&mesh, mat, 0);
   ModifyAttributeForMarkingDOFS(&mesh, mat, 1);

   mesh.SetAttributes();

   Array<int> attr(1);
   attr[0] = attr_to_trim;
   Array<int> bdr_attr;

   int max_attr     = mesh.attributes.Max();
   int max_bdr_attr = mesh.bdr_attributes.Max();

   if (bdr_attr.Size() == 0)
   {
      bdr_attr.SetSize(attr.Size());
      for (int i=0; i<attr.Size(); i++)
      {
         bdr_attr[i] = max_bdr_attr + attr[i];
      }
   }
   MFEM_VERIFY(attr.Size() == bdr_attr.Size(),
               "Size mismatch in attribute arguments.");

   Array<int> marker(max_attr);
   Array<int> attr_inv(max_attr);
   marker = 0;
   attr_inv = 0;
   for (int i=0; i<attr.Size(); i++)
   {
      marker[attr[i]-1] = 1;
      attr_inv[attr[i]-1] = i;
   }

   // Count the number of elements in the final mesh
   int num_elements = 0;
   for (int e=0; e<mesh.GetNE(); e++)
   {
      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1]) { num_elements++; }
   }

   // Count the number of boundary elements in the final mesh
   int num_bdr_elements = 0;
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 == 0 || a2 == 0)
      {
         if (a1 == 0 && !marker[a2-1]) { num_bdr_elements++; }
         else if (a2 == 0 && !marker[a1-1]) { num_bdr_elements++; }
      }
      else
      {
         if (marker[a1-1] && !marker[a2-1]) { num_bdr_elements++; }
         else if (!marker[a1-1] && marker[a2-1]) { num_bdr_elements++; }
      }
   }

   cout << "Number of Elements:          " << mesh.GetNE() << " -> "
        << num_elements << endl;
   cout << "Number of Boundary Elements: " << mesh.GetNBE() << " -> "
        << num_bdr_elements << endl;

   Mesh *trimmed_mesh = new Mesh(mesh.Dimension(), mesh.GetNV(),
                                 num_elements, num_bdr_elements, mesh.SpaceDimension());
   //   Mesh trimmed_mesh(mesh.Dimension(), mesh.GetNV(),
   //                     num_elements, num_bdr_elements, mesh.SpaceDimension());

   // Copy vertices
   for (int v=0; v<mesh.GetNV(); v++)
   {
      trimmed_mesh->AddVertex(mesh.GetVertex(v));
   }

   // Copy elements
   for (int e=0; e<mesh.GetNE(); e++)
   {
      Element * el = mesh.GetElement(e);
      int elem_attr = el->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nel = mesh.NewElement(el->GetGeometryType());
         nel->SetAttribute(elem_attr);
         nel->SetVertices(el->GetVertices());
         trimmed_mesh->AddElement(nel);
      }
   }

   // Copy selected boundary elements
   for (int be=0; be<mesh.GetNBE(); be++)
   {
      int e, info;
      mesh.GetBdrElementAdjacentElement(be, e, info);

      int elem_attr = mesh.GetElement(e)->GetAttribute();
      if (!marker[elem_attr-1])
      {
         Element * nbel = mesh.GetBdrElement(be)->Duplicate(trimmed_mesh);
         trimmed_mesh->AddBdrElement(nbel);
      }
   }

   // Create new boundary elements
   for (int f=0; f<mesh.GetNumFaces(); f++)
   {
      int e1 = -1, e2 = -1;
      mesh.GetFaceElements(f, &e1, &e2);

      int i1 = -1, i2 = -1;
      mesh.GetFaceInfos(f, &i1, &i2);

      int a1 = 0, a2 = 0;
      if (e1 >= 0) { a1 = mesh.GetElement(e1)->GetAttribute(); }
      if (e2 >= 0) { a2 = mesh.GetElement(e2)->GetAttribute(); }

      if (a1 != 0 && a2 != 0)
      {
         if (marker[a1-1] && !marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(trimmed_mesh);
            //bel->SetAttribute(bdr_attr[attr_inv[a1-1]]);
            bel->SetAttribute(3);
            trimmed_mesh->AddBdrElement(bel);
         }
         else if (!marker[a1-1] && marker[a2-1])
         {
            Element * bel = (mesh.Dimension() == 1) ?
                            (Element*)new Point(&f) :
                            mesh.GetFace(f)->Duplicate(trimmed_mesh);
            //bel->SetAttribute(bdr_attr[attr_inv[a2-1]]);
            bel->SetAttribute(3);
            trimmed_mesh->AddBdrElement(bel);
         }
      }
   }

   trimmed_mesh->FinalizeTopology();
   trimmed_mesh->Finalize();
   trimmed_mesh->RemoveUnusedVertices();

   // Check for curved or discontinuous mesh
   if (mesh.GetNodes())
   {
      // Extract Nodes GridFunction and determine its type
      const GridFunction * Nodes = mesh.GetNodes();
      const FiniteElementSpace * fes = Nodes->FESpace();

      Ordering::Type ordering = fes->GetOrdering();
      int order = fes->FEColl()->GetOrder();
      int sdim = mesh.SpaceDimension();
      bool discont =
         dynamic_cast<const L2_FECollection*>(fes->FEColl()) != NULL;

      // Set curvature of the same type as original mesh
      trimmed_mesh->SetCurvature(order, discont, sdim, ordering);

      const FiniteElementSpace * trimmed_fes = trimmed_mesh->GetNodalFESpace();
      GridFunction * trimmed_nodes = trimmed_mesh->GetNodes();

      Array<int> vdofs;
      Array<int> trimmed_vdofs;
      Vector loc_vec;

      // Copy nodes to trimmed mesh
      int te = 0;
      for (int e = 0; e < mesh.GetNE(); e++)
      {
         Element * el = mesh.GetElement(e);
         int elem_attr = el->GetAttribute();
         if (!marker[elem_attr-1])
         {
            fes->GetElementVDofs(e, vdofs);
            Nodes->GetSubVector(vdofs, loc_vec);

            trimmed_fes->GetElementVDofs(te, trimmed_vdofs);
            trimmed_nodes->SetSubVector(trimmed_vdofs, loc_vec);
            te++;
         }
      }
   }

   return trimmed_mesh;
}

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

void ComputeScalarDistanceFromLevelSet(ParMesh &pmesh,
                                       FunctionCoefficient &ls_coeff,
                                       int amr_iter, ParGridFunction &distance_s)
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

   for (int iter = 0; iter < amr_iter; iter++)
   {
      el_to_refine = 0.0;
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         h1fespace.GetElementDofs(e, dofs);
         x.GetSubVector(dofs, x_vals);
         int refine = 0;
         double min_val = 100;
         double max_val = -100;
         for (int j = 0; j < x_vals.Size(); j++)
         {
            double x_dof_val = x_vals(j);
            min_val = min(x_dof_val, min_val);
            max_val = max(x_dof_val, max_val);
         }
         if (min_val < 0 && max_val > 0)
         {
            refine = 1;
            el_to_refine(e) = 1.0;
         }
      }

      //Refine an element if its neighbor will be refined
      el_to_refine.ExchangeFaceNbrData();
      GridFunctionCoefficient field_in_dg(&el_to_refine);
      lhx.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
      for (int e = 0; e < pmesh.GetNE(); e++)
      {
         Array<int> dofs;
         Vector x_vals;
         lhfespace.GetElementDofs(e, dofs);
         lhx.GetSubVector(dofs, x_vals);
         int refine = 0;
         double max_val = -100;
         for (int j = 0; j < x_vals.Size(); j++)
         {
            double x_dof_val = x_vals(j);
            max_val = max(x_dof_val, max_val);
         }
         if (max_val > 0)
         {
            refine = 1;
            el_to_refine(e) = 1.0;
         }
      }

      //make the list of elements to be refined
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
      h1fespace.Update();
      x.Update();
      x.ProjectCoefficient(ls_coeff);

      l2fespace.Update();
      el_to_refine.Update();

      lhfespace.Update();
      lhx.Update();

      distance_s.ParFESpace()->Update();
      distance_s.Update();
   }

   {
      ostringstream mesh_name;
      mesh_name << "background.mesh";
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.PrintAsOne(mesh_ofs);

      ostringstream gf_name;
      gf_name << "background.gf";
      ofstream gf_ofs(gf_name.str().c_str());
      gf_ofs.precision(8);
      x.SaveAsOne(gf_ofs);
   }

   //Now determine distance
   const double dx = AvgElementSize(pmesh);
   DistanceSolver *dist_solver = NULL;
   int solver_type = 1;
   double t_param = 1.0;

   if (solver_type == 0)
   {
      auto ds = new HeatDistanceSolver(t_param * dx * dx);
      ds->smooth_steps = 0;
      ds->vis_glvis = false;
      dist_solver = ds;
   }
   else if (solver_type == 1)
   {
      const int p = 5;
      const int newton_iter = 50;
      auto ds = new PLapDistanceSolver(p, newton_iter);
      dist_solver = ds;
   }
   else { MFEM_ABORT("Wrong solver option."); }
   dist_solver->print_level = 1;

   ParFiniteElementSpace pfes_s(*distance_s.ParFESpace());

   // Smooth-out Gibbs oscillations from the input level set. The smoothing
   // parameter here is specified to be mesh dependent with length scale dx.
   ParGridFunction filt_gf(&pfes_s);
   PDEFilter filter(pmesh, 1.0 * dx);
   filter.Filter(ls_coeff, filt_gf);
   GridFunctionCoefficient ls_filt_coeff(&filt_gf);

   dist_solver->ComputeScalarDistance(ls_filt_coeff, distance_s);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();

   DiffuseField(distance_s, 5);
   distance_s.SetTrueVector();
   distance_s.SetFromTrueVector();
   for (int i = 0; i < distance_s.Size(); i++)
   {
      distance_s(i) = std::fabs(distance_s(i));
   }
}
#endif
