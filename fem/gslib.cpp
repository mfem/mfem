// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "gslib.hpp"

#ifdef MFEM_USE_GSLIB

// Ignore warnings from the gslib header (GCC version)
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#include "gslib.h"

#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic pop
#endif

namespace mfem
{

FindPointsGSLIB::FindPointsGSLIB()
   : mesh(NULL), meshsplit(NULL), ir_simplex(NULL),
     fdata2D(NULL), fdata3D(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC)
{
   gsl_comm = new comm;
   cr       = new crystal;
#ifdef MFEM_USE_MPI
   int initialized;
   MPI_Initialized(&initialized);
   if (!initialized) { MPI_Init(NULL, NULL); }
   MPI_Comm comm = MPI_COMM_WORLD;;
   comm_init(gsl_comm, comm);
#else
   comm_init(gsl_comm, 0);
#endif
}

FindPointsGSLIB::~FindPointsGSLIB()
{
   delete gsl_comm;
   delete cr;
   delete ir_simplex;
   delete meshsplit;
}

#ifdef MFEM_USE_MPI
FindPointsGSLIB::FindPointsGSLIB(MPI_Comm _comm)
   : mesh(NULL), meshsplit(NULL), ir_simplex(NULL),
     fdata2D(NULL), fdata3D(NULL), cr(NULL), gsl_comm(NULL),
     dim(-1), points_cnt(0), setupflag(false), default_interp_value(0),
     avgtype(AvgType::ARITHMETIC)
{
   gsl_comm = new comm;
   cr      = new crystal;
   comm_init(gsl_comm, _comm);
}
#endif

void FindPointsGSLIB::Setup(Mesh &m, const double bb_t, const double newt_tol,
                            const int npt_max)
{
   MFEM_VERIFY(m.GetNodes() != NULL, "Mesh nodes are required.");
   MFEM_VERIFY(m.GetNumGeometries(m.Dimension()) == 1,
               "Mixed meshes are not currently supported in FindPointsGSLIB.");

   // call FreeData if FindPointsGSLIB::Setup has been called already
   if (setupflag) { FreeData(); }

   crystal_init(cr, gsl_comm);
   mesh = &m;
   dim  = mesh->Dimension();
   const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(0);
   unsigned dof1D = fe->GetOrder() + 1;
   const int gt   = fe->GetGeomType();

   if (gt == Geometry::TRIANGLE || gt == Geometry::TETRAHEDRON ||
       gt == Geometry::PRISM)
   {
      GetSimplexNodalCoordinates();
   }
   else if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
   {
      GetQuadHexNodalCoordinates();
   }
   else
   {
      MFEM_ABORT("Element type not currently supported in FindPointsGSLIB.");
   }

   const int pts_cnt = gsl_mesh.Size()/dim,
             NEtot = pts_cnt/(int)pow(dof1D, dim);

   if (dim == 2)
   {
      unsigned nr[2] = { dof1D, dof1D };
      unsigned mr[2] = { 2*dof1D, 2*dof1D };
      double * const elx[2] = { &gsl_mesh(0), &gsl_mesh(pts_cnt) };
      fdata2D = findpts_setup_2(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                pts_cnt, pts_cnt, npt_max, newt_tol);
   }
   else
   {
      unsigned nr[3] = { dof1D, dof1D, dof1D };
      unsigned mr[3] = { 2*dof1D, 2*dof1D, 2*dof1D };
      double * const elx[3] =
      { &gsl_mesh(0), &gsl_mesh(pts_cnt), &gsl_mesh(2*pts_cnt) };
      fdata3D = findpts_setup_3(gsl_comm, elx, nr, NEtot, mr, bb_t,
                                pts_cnt, pts_cnt, npt_max, newt_tol);
   }
   setupflag = true;
}

void FindPointsGSLIB::FindPoints(const Vector &point_pos)
{
   MFEM_VERIFY(setupflag, "Use FindPointsGSLIB::Setup before finding points.");
   points_cnt = point_pos.Size() / dim;
   gsl_code.SetSize(points_cnt);
   gsl_proc.SetSize(points_cnt);
   gsl_elem.SetSize(points_cnt);
   gsl_ref.SetSize(points_cnt * dim);
   gsl_dist.SetSize(points_cnt);

   if (dim == 2)
   {
      const double *xv_base[2];
      xv_base[0] = point_pos.GetData();
      xv_base[1] = point_pos.GetData() + points_cnt;
      unsigned xv_stride[2];
      xv_stride[0] = sizeof(double);
      xv_stride[1] = sizeof(double);
      findpts_2(gsl_code.GetData(), sizeof(unsigned int),
                gsl_proc.GetData(), sizeof(unsigned int),
                gsl_elem.GetData(), sizeof(unsigned int),
                gsl_ref.GetData(),  sizeof(double) * dim,
                gsl_dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt, fdata2D);
   }
   else
   {
      const double *xv_base[3];
      xv_base[0] = point_pos.GetData();
      xv_base[1] = point_pos.GetData() + points_cnt;
      xv_base[2] = point_pos.GetData() + 2*points_cnt;
      unsigned xv_stride[3];
      xv_stride[0] = sizeof(double);
      xv_stride[1] = sizeof(double);
      xv_stride[2] = sizeof(double);
      findpts_3(gsl_code.GetData(), sizeof(unsigned int),
                gsl_proc.GetData(), sizeof(unsigned int),
                gsl_elem.GetData(), sizeof(unsigned int),
                gsl_ref.GetData(),  sizeof(double) * dim,
                gsl_dist.GetData(), sizeof(double),
                xv_base, xv_stride, points_cnt, fdata3D);
   }

   // Set the element number and reference position to 0 for points not found
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2)
      {
         gsl_elem[i] = 0;
         for (int d = 0; d < dim; d++) { gsl_ref(i*dim + d) = -1.; }
      }
   }

   // Map element number for simplices, and ref_pos from [-1,1] to [0,1] for
   // both simplices and quads.
   MapRefPosAndElemIndices();
}

void FindPointsGSLIB::FindPoints(Mesh &m, const Vector &point_pos,
                                 const double bb_t, const double newt_tol,
                                 const int npt_max)
{
   if (!setupflag || (mesh != &m) )
   {
      Setup(m, bb_t, newt_tol, npt_max);
   }
   FindPoints(point_pos);
}

void FindPointsGSLIB::Interpolate(const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out)
{
   FindPoints(point_pos);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::Interpolate(Mesh &m, const Vector &point_pos,
                                  const GridFunction &field_in, Vector &field_out)
{
   FindPoints(m, point_pos);
   Interpolate(field_in, field_out);
}

void FindPointsGSLIB::FreeData()
{
   if (!setupflag) { return; }
   crystal_free(cr);
   if (dim == 2)
   {
      findpts_free_2(fdata2D);
   }
   else
   {
      findpts_free_3(fdata3D);
   }
   gsl_code.DeleteAll();
   gsl_proc.DeleteAll();
   gsl_elem.DeleteAll();
   gsl_mesh.Destroy();
   gsl_ref.Destroy();
   gsl_dist.Destroy();
   setupflag = false;
}

void FindPointsGSLIB::GetNodeValues(const GridFunction &gf_in,
                                    Vector &node_vals)
{
   MFEM_ASSERT(gf_in.FESpace()->GetVDim() == 1, "Scalar function expected.");

   const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(0);
   const Geometry::Type gt = fe->GetGeomType();
   const int            NE = mesh->GetNE();

   if (gt == Geometry::SQUARE || gt == Geometry::CUBE)
   {
      const GridFunction *nodes     = mesh->GetNodes();
      const FiniteElementSpace *fes = nodes->FESpace();
      const IntegrationRule &ir     = fes->GetFE(0)->GetNodes();
      const int dof_cnt = ir.GetNPoints();

      node_vals.SetSize(NE * dof_cnt);

      const TensorBasisElement *tbe =
         dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
      MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
      const Array<int> &dof_map = tbe->GetDofMap();

      int pt_id = 0;
      Vector vals_el;
      for (int i = 0; i < NE; i++)
      {
         gf_in.GetValues(i, ir, vals_el);
         for (int j = 0; j < dof_cnt; j++)
         {
            node_vals(pt_id++) = vals_el(dof_map[j]);
         }
      }
   }
   else if (gt == Geometry::TRIANGLE || gt == Geometry::TETRAHEDRON ||
            gt == Geometry::PRISM)
   {
      const int dof_cnt = ir_simplex->GetNPoints();
      node_vals.SetSize(NE * dof_cnt);

      int pt_id = 0;
      Vector vals_el;
      for (int j = 0; j < NE; j++)
      {
         gf_in.GetValues(j, *ir_simplex, vals_el);
         for (int i = 0; i < dof_cnt; i++)
         {
            node_vals(pt_id++) = vals_el(i);
         }
      }
   }
   else
   {
      MFEM_ABORT("Element type not currently supported.");
   }
}

void FindPointsGSLIB::GetQuadHexNodalCoordinates()
{
   const GridFunction *nodes     = mesh->GetNodes();
   const FiniteElementSpace *fes = nodes->FESpace();

   const int NE      = mesh->GetNE(),
             dof_cnt = fes->GetFE(0)->GetDof(),
             pts_cnt = NE * dof_cnt;
   gsl_mesh.SetSize(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(fes->GetFE(0));
   MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
   const Array<int> &dof_map = tbe->GetDofMap();

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   int pt_id = 0;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            gsl_mesh(pts_cnt * d + pt_id) = pos(dof_map[j], d);
         }
         pt_id++;
      }
   }
}

void FindPointsGSLIB::GetSimplexNodalCoordinates()
{
   const FiniteElement *fe   = mesh->GetNodalFESpace()->GetFE(0);
   const Geometry::Type gt   = fe->GetGeomType();
   const GridFunction *nodes = mesh->GetNodes();
   const int NE              = mesh->GetNE();
   int NEsplit = 0;

   // Split the reference element into a reference submesh of quads or hexes.
   if (gt == Geometry::TRIANGLE)
   {
      int Nvert = 7;
      NEsplit = 3;
      meshsplit = new Mesh(2, Nvert, NEsplit, 0, 2);

      const double quad_v[7][2] =
      {
         {0, 0}, {0.5, 0}, {1, 0}, {0, 0.5},
         {1./3., 1./3.}, {0.5, 0.5}, {0, 1}
      };
      const int quad_e[3][4] =
      {
         {3, 4, 1, 0}, {4, 5, 2, 1}, {6, 5, 4, 3}
      };

      for (int j = 0; j < Nvert; j++)
      {
         meshsplit->AddVertex(quad_v[j]);
      }
      for (int j = 0; j < NEsplit; j++)
      {
         int attribute = j + 1;
         meshsplit->AddQuad(quad_e[j], attribute);
      }
      meshsplit->FinalizeQuadMesh(1, 1, true);
   }
   else if (gt == Geometry::TETRAHEDRON)
   {
      int Nvert = 15;
      NEsplit = 4;
      meshsplit = new Mesh(3, Nvert, NEsplit, 0, 3);

      const double hex_v[15][3] =
      {
         {0, 0, 0.}, {1, 0., 0.}, {0., 1., 0.}, {0, 0., 1.},
         {0.5, 0., 0.}, {0.5, 0.5, 0.}, {0., 0.5, 0.},
         {0., 0., 0.5}, {0.5, 0., 0.5}, {0., 0.5, 0.5},
         {1./3., 0., 1./3.}, {1./3., 1./3., 1./3.}, {0, 1./3., 1./3.},
         {1./3., 1./3., 0}, {0.25, 0.25, 0.25}
      };
      const int hex_e[4][8] =
      {
         {0, 4, 10, 7, 6, 13, 14, 12},
         {4, 1, 8, 10, 13, 5, 11, 14},
         {13, 5, 11, 14, 6, 2, 9, 12},
         {10, 8, 3, 7, 14, 11, 9, 12}
      };

      for (int j = 0; j < Nvert; j++)
      {
         meshsplit->AddVertex(hex_v[j]);
      }
      for (int j = 0; j < NEsplit; j++)
      {
         int attribute = j + 1;
         meshsplit->AddHex(hex_e[j], attribute);
      }
      meshsplit->FinalizeHexMesh(1, 1, true);
   }
   else if (gt == Geometry::PRISM)
   {
      int Nvert = 14;
      NEsplit = 3;
      meshsplit = new Mesh(3, Nvert, NEsplit, 0, 3);

      const double hex_v[14][3] =
      {
         {0, 0, 0}, {0.5, 0, 0}, {1, 0, 0}, {0, 0.5, 0},
         {1./3., 1./3., 0}, {0.5, 0.5, 0}, {0, 1, 0},
         {0, 0, 1}, {0.5, 0, 1}, {1, 0, 1}, {0, 0.5, 1},
         {1./3., 1./3., 1}, {0.5, 0.5, 1}, {0, 1, 1}
      };
      const int hex_e[3][8] =
      {
         {3, 4, 1, 0, 10, 11, 8, 7},
         {4, 5, 2, 1, 11, 12, 9, 8},
         {6, 5, 4, 3, 13, 12, 11, 10}
      };

      for (int j = 0; j < Nvert; j++)
      {
         meshsplit->AddVertex(hex_v[j]);
      }
      for (int j = 0; j < NEsplit; j++)
      {
         int attribute = j + 1;
         meshsplit->AddHex(hex_e[j], attribute);
      }
      meshsplit->FinalizeHexMesh(1, 1, true);
   }
   else { MFEM_ABORT("Unsupported geometry type."); }

   // Curve the reference submesh.
   H1_FECollection fec(fe->GetOrder(), dim);
   FiniteElementSpace nodal_fes(meshsplit, &fec, dim);
   meshsplit->SetNodalFESpace(&nodal_fes);

   const int dof_cnt = nodal_fes.GetFE(0)->GetDof(),
             pts_cnt = NEsplit * dof_cnt;
   Vector irlist(dim * pts_cnt);

   const TensorBasisElement *tbe =
      dynamic_cast<const TensorBasisElement *>(nodal_fes.GetFE(0));
   MFEM_VERIFY(tbe != NULL, "TensorBasis FiniteElement expected.");
   const Array<int> &dof_map = tbe->GetDofMap();

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   // Create an IntegrationRule on the nodes of the reference submesh.
   ir_simplex = new IntegrationRule(pts_cnt);
   GridFunction *nodesplit = meshsplit->GetNodes();
   int pt_id = 0;
   for (int i = 0; i < NEsplit; i++)
   {
      nodal_fes.GetElementVDofs(i, xdofs);
      nodesplit->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         for (int d = 0; d < dim; d++)
         {
            irlist(pts_cnt * d + pt_id) = pos(dof_map[j], d);
         }
         ir_simplex->IntPoint(pt_id).x = irlist(pt_id);
         ir_simplex->IntPoint(pt_id).y = irlist(pts_cnt + pt_id);
         if (dim == 3)
         {
            ir_simplex->IntPoint(pt_id).z = irlist(2*pts_cnt + pt_id);
         }
         pt_id++;
      }
   }

   // Initialize gsl_mesh with the positions of the split physical elements.
   pt_id = 0;
   Vector locval(dim);
   const int tot_pts_cnt = pts_cnt*NE;
   gsl_mesh.SetSize(tot_pts_cnt*dim);
   for (int j = 0; j < NE; j++)
   {
      for (int i = 0; i < dof_cnt*NEsplit; i++)
      {
         const IntegrationPoint &ip = ir_simplex->IntPoint(i);
         nodes->GetVectorValue(j, ip, locval);
         for (int d = 0; d < dim; d++)
         {
            gsl_mesh(tot_pts_cnt*d + pt_id) = locval(d);
         }
         pt_id++;
      }
   }
}

void FindPointsGSLIB::MapRefPosAndElemIndices()
{
   gsl_mfem_ref = gsl_ref;
   gsl_mfem_elem = gsl_elem;
   const FiniteElement *fe = mesh->GetNodalFESpace()->GetFE(0);
   const Geometry::Type gt = fe->GetGeomType();
   int NEsplit = 0;

   gsl_mfem_ref -= -1.;  // map  [-1, 1] to
   gsl_mfem_ref *= 0.5;  //      [0,  1]
   if (gt == Geometry::SQUARE || gt == Geometry::CUBE) { return; }

   H1_FECollection feclin(1, dim);
   FiniteElementSpace nodal_fes_lin(meshsplit, &feclin, dim);
   GridFunction gf_lin(&nodal_fes_lin);

   if (gt == Geometry::TRIANGLE)
   {
      const double quad_v[7][2] =
      {
         {0, 0}, {0.5, 0}, {1, 0}, {0, 0.5},
         {1./3., 1./3.}, {0.5, 0.5}, {0, 1}
      };
      for (int k = 0; k < dim; k++)
      {
         for (int j = 0; j < gf_lin.Size()/dim; j++)
         {
            gf_lin(j+k*gf_lin.Size()/dim) = quad_v[j][k];
         }
      }
      NEsplit = 3;
   }
   else if (gt == Geometry::TETRAHEDRON)
   {
      const double hex_v[15][3] =
      {
         {0, 0, 0.}, {1, 0., 0.}, {0., 1., 0.}, {0, 0., 1.},
         {0.5, 0., 0.}, {0.5, 0.5, 0.}, {0., 0.5, 0.},
         {0., 0., 0.5}, {0.5, 0., 0.5}, {0., 0.5, 0.5},
         {1./3., 0., 1./3.}, {1./3., 1./3., 1./3.}, {0, 1./3., 1./3.},
         {1./3., 1./3., 0}, {0.25, 0.25, 0.25}
      };
      for (int k = 0; k < dim; k++)
      {
         for (int j = 0; j < gf_lin.Size()/dim; j++)
         {
            gf_lin(j+k*gf_lin.Size()/dim) = hex_v[j][k];
         }
      }
      NEsplit = 4;
   }
   else if (gt == Geometry::PRISM)
   {
      const double hex_v[14][3] =
      {
         {0, 0, 0}, {0.5, 0, 0}, {1, 0, 0}, {0, 0.5, 0},
         {1./3., 1./3., 0}, {0.5, 0.5, 0}, {0, 1, 0},
         {0, 0, 1}, {0.5, 0, 1}, {1, 0, 1}, {0, 0.5, 1},
         {1./3., 1./3., 1}, {0.5, 0.5, 1}, {0, 1, 1}
      };
      for (int k = 0; k < dim; k++)
      {
         for (int j = 0; j < gf_lin.Size()/dim; j++)
         {
            gf_lin(j+k*gf_lin.Size()/dim) = hex_v[j][k];
         }
      }
      NEsplit = 3;
   }
   else
   {
      MFEM_ABORT("Element type not currently supported.");
   }

   // Simplices are split into quads/hexes for GSLIB. For MFEM, we need to find
   // the original element number and map the rst from micro to macro element.
   for (int i = 0; i < points_cnt; i++)
   {
      if (gsl_code[i] == 2) { continue; }
      int local_elem   = gsl_elem[i]%NEsplit;
      gsl_mfem_elem[i] = (gsl_elem[i] - local_elem)/NEsplit; // macro element number

      IntegrationPoint ip;
      Vector mfem_ref(gsl_mfem_ref.GetData()+i*dim, dim);
      ip.Set2(mfem_ref.GetData());
      if (dim == 3) { ip.z = mfem_ref(2); }
      gf_lin.GetVectorValue(local_elem, ip, mfem_ref); // map to rst of macro element
   }
}

void FindPointsGSLIB::Interpolate(const GridFunction &field_in,
                                  Vector &field_out)
{
   const int  gf_order   = field_in.FESpace()->GetFE(0)->GetOrder(),
              mesh_order = mesh->GetNodalFESpace()->GetFE(0)->GetOrder();

   const FiniteElementCollection *fec_in =  field_in.FESpace()->FEColl();
   const H1_FECollection *fec_h1 = dynamic_cast<const H1_FECollection *>(fec_in);
   const L2_FECollection *fec_l2 = dynamic_cast<const L2_FECollection *>(fec_in);

   if (fec_h1 && gf_order == mesh_order &&
       fec_h1->GetBasisType() == BasisType::GaussLobatto)
   {
      InterpolateH1(field_in, field_out);
      return;
   }
   else
   {
      InterpolateGeneral(field_in, field_out);
      if (!fec_l2 || avgtype == AvgType::NONE) { return; }
   }

   // For points on element borders, project the L2 GridFunction to H1 and
   // re-interpolate.
   if (fec_l2)
   {
      Array<int> indl2;
      for (int i = 0; i < points_cnt; i++)
      {
         if (gsl_code[i] == 1) { indl2.Append(i); }
      }
      if (indl2.Size() == 0) { return; } // no points on element borders

      Vector field_out_l2(field_out.Size());
      VectorGridFunctionCoefficient field_in_dg(&field_in);
      int gf_order_h1 = std::max(gf_order, 1); // H1 should be at least order 1
      H1_FECollection fec(gf_order_h1, dim);
      const int ncomp = field_in.FESpace()->GetVDim();
      FiniteElementSpace fes(mesh, &fec, ncomp);
      GridFunction field_in_h1(&fes);

      if (avgtype == AvgType::ARITHMETIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::ARITHMETIC);
      }
      else if (avgtype == AvgType::HARMONIC)
      {
         field_in_h1.ProjectDiscCoefficient(field_in_dg, GridFunction::HARMONIC);
      }
      else
      {
         MFEM_ABORT("Invalid averaging type.");
      }

      if (gf_order_h1 == mesh_order) // basis is GaussLobatto by default
      {
         InterpolateH1(field_in_h1, field_out_l2);
      }
      else
      {
         InterpolateGeneral(field_in_h1, field_out_l2);
      }

      // Copy interpolated values for the points on element border
      for (int j = 0; j < ncomp; j++)
      {
         for (int i = 0; i < indl2.Size(); i++)
         {
            int idx = indl2[i] + j*points_cnt;
            field_out(idx) = field_out_l2(idx);
         }
      }
   }
}

void FindPointsGSLIB::InterpolateH1(const GridFunction &field_in,
                                    Vector &field_out)
{
   FiniteElementSpace ind_fes(mesh, field_in.FESpace()->FEColl());
   GridFunction field_in_scalar(&ind_fes);
   Vector node_vals;

   const int ncomp      = field_in.FESpace()->GetVDim(),
             points_fld = field_in.Size() / ncomp,
             points_cnt = gsl_code.Size();

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   for (int i = 0; i < ncomp; i++)
   {
      const int dataptrin  = i*points_fld,
                dataptrout = i*points_cnt;
      field_in_scalar.NewDataAndSize(field_in.GetData()+dataptrin, points_fld);
      GetNodeValues(field_in_scalar, node_vals);

      if (dim==2)
      {
         findpts_eval_2(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(), fdata2D);
      }
      else
      {
         findpts_eval_3(field_out.GetData()+dataptrout, sizeof(double),
                        gsl_code.GetData(),       sizeof(unsigned int),
                        gsl_proc.GetData(),    sizeof(unsigned int),
                        gsl_elem.GetData(),    sizeof(unsigned int),
                        gsl_ref.GetData(),     sizeof(double) * dim,
                        points_cnt, node_vals.GetData(), fdata3D);
      }
   }
}

void FindPointsGSLIB::InterpolateGeneral(const GridFunction &field_in,
                                         Vector &field_out)
{
   int ncomp   = field_in.VectorDim(),
       nptorig = points_cnt,
       npt     = points_cnt;

   field_out.SetSize(points_cnt*ncomp);
   field_out = default_interp_value;

   if (gsl_comm->np == 1) // serial
   {
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         IntegrationPoint ip;
         ip.Set2(gsl_mfem_ref.GetData()+index*dim);
         if (dim == 3) { ip.z = gsl_mfem_ref(index*dim + 2); }
         Vector localval(ncomp);
         field_in.GetVectorValue(gsl_mfem_elem[index], ip, localval);
         for (int i = 0; i < ncomp; i++)
         {
            field_out(index + i*npt) = localval(i);
         }
      }
   }
   else // parallel
   {
      // Determine number of points to be sent
      int nptsend = 0;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] != 2) { nptsend +=1; }
      }

      // Pack data to send via crystal router
      struct array *outpt = new array;
      struct out_pt { double r[3], ival; uint index, el, proc; };
      struct out_pt *pt;
      array_init(struct out_pt, outpt, nptsend);
      outpt->n=nptsend;
      pt = (struct out_pt *)outpt->ptr;
      for (int index = 0; index < npt; index++)
      {
         if (gsl_code[index] == 2) { continue; }
         for (int d = 0; d < dim; ++d) { pt->r[d]= gsl_mfem_ref(index*dim + d); }
         pt->index = index;
         pt->proc  = gsl_proc[index];
         pt->el    = gsl_mfem_elem[index];
         ++pt;
      }

      // Transfer data to target MPI ranks
      sarray_transfer(struct out_pt, outpt, proc, 1, cr);

      if (ncomp == 1)
      {
         // Interpolate the grid function
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            pt->ival = field_in.GetValue(pt->el, ip, 1);
            ++pt;
         }

         // Transfer data back to source MPI rank
         sarray_transfer(struct out_pt, outpt, proc, 1, cr);
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            field_out(pt->index) = pt->ival;
            ++pt;
         }
         array_free(outpt);
         delete outpt;
      }
      else // ncomp > 1
      {
         // Interpolate data and store in a Vector
         npt = outpt->n;
         pt = (struct out_pt *)outpt->ptr;
         Vector vec_int_vals(npt*ncomp);
         for (int index = 0; index < npt; index++)
         {
            IntegrationPoint ip;
            ip.Set3(&pt->r[0]);
            Vector localval(vec_int_vals.GetData()+index*ncomp, ncomp);
            field_in.GetVectorValue(pt->el, ip, localval);
            ++pt;
         }

         // Save index and proc data in a struct
         struct array *savpt = new array;
         struct sav_pt { uint index, proc; };
         struct sav_pt *spt;
         array_init(struct sav_pt, savpt, npt);
         savpt->n=npt;
         spt = (struct sav_pt *)savpt->ptr;
         pt  = (struct out_pt *)outpt->ptr;
         for (int index = 0; index < npt; index++)
         {
            spt->index = pt->index;
            spt->proc  = pt->proc;
            ++pt; ++spt;
         }

         array_free(outpt);
         delete outpt;

         // Copy data from save struct to send struct and send component wise
         struct array *sendpt = new array;
         struct send_pt { double ival; uint index, proc; };
         struct send_pt *sdpt;
         for (int j = 0; j < ncomp; j++)
         {
            array_init(struct send_pt, sendpt, npt);
            sendpt->n=npt;
            spt  = (struct sav_pt *)savpt->ptr;
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < npt; index++)
            {
               sdpt->index = spt->index;
               sdpt->proc  = spt->proc;
               sdpt->ival  = vec_int_vals(j + index*ncomp);
               ++sdpt; ++spt;
            }

            sarray_transfer(struct send_pt, sendpt, proc, 1, cr);
            sdpt = (struct send_pt *)sendpt->ptr;
            for (int index = 0; index < nptorig; index++)
            {
               int idx = sdpt->index + j*nptorig;
               field_out(idx) = sdpt->ival;
               ++sdpt;
            }
            array_free(sendpt);
         }
         array_free(savpt);
         delete sendpt;
         delete savpt;
      } // ncomp > 1
   } // parallel
}

} // namespace mfem

#endif // MFEM_USE_GSLIB
