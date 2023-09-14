// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace common;

void MakeMaterialConsistentForElementGroups(GridFunction &mat,
                                            int nel_per_group)
{
   FiniteElementSpace *fespace = mat.FESpace();
   Mesh *mesh = fespace->GetMesh();
   int nel = mesh->GetNE();
   int ngroups = nel/nel_per_group;
   MFEM_VERIFY(nel % nel_per_group == 0, "Invalid nel_per_group for custmo mesh.");

   for (int g = 0; g < ngroups; g++)
   {
      double maxv = -1.0;
      for (int i = 0; i < nel_per_group; i++)
      {
         int el_idx = g*nel_per_group + i;
         int el_attr = mesh->GetAttribute(el_idx);
         maxv = std::max(maxv, el_attr*1.0);
      }

      for (int i = 0; i < nel_per_group; i++)
      {
         int el_idx = g*nel_per_group + i;
         mesh->SetAttribute(el_idx, int(maxv));
      }
   }

   for (int e = 0; e < nel; e++)
   {
      mat(e) = mesh->GetAttribute(e)-1.0;
   }
   mesh->SetAttributes();
}

void GetMaterialInterfaceEdgeDofs(Mesh *mesh, GridFunction &mat,
                                  GridFunction &surf_fit_gf0,
                                  Array<Array<int> *> &edge_to_el, //edge to element mapping
                                  Array<int> &inter_face_el_all, //list of all elements with faces on interface
                                  Array<int> &intfdofs, //list of all dofs on interface
                                  Array<int> &intedges, //edge index
                                  Array<int> &intedgeels,  //elements with these edges
                                  Array<Array<int> *> &el_to_el_edge_dep) //dependency
{
   intedges.SetSize(0);
   intedgeels.SetSize(0);
   const int NElem = mesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   int dim = mesh->Dimension();

   GridFunction temp(surf_fit_gf0);
   temp = 0.0;
   for (int ii = 0; ii < intfdofs.Size(); ii++)
   {
      temp(intfdofs[ii]) = 1.0;
   }

   Vector datavec;
   Array<int> dofs;
   //   for (int e = 0; e < mesh->GetNE(); e++)
   //   {
   //       surf_fit_gf0.FESpace()->GetElementDofs(e, dofs);
   //       temp.GetSubVector(dofs, datavec);
   //       if (datavec.Max() == 1.0) { intedgeels.Append(e); }
   //   }

   for (int f = 0; f < mesh->GetNEdges(); f++ )
   {
      surf_fit_gf0.FESpace()->GetEdgeDofs(f, dofs);
      temp.GetSubVector(dofs, datavec);
      if (datavec.Min() == 1.0)
      {
         intedges.Append(f);
      }
   }
   intedges.Sort();
   intedges.Unique();

   // Now we want to identify edges that have elements on the interface
   // that are not marked
   for (int f = 0; f < intedges.Size(); f++)
   {
      int edgnum = intedges[f];
      Array<int> *edge_els = edge_to_el[edgnum];
      Array<int> edge_el_need_to_be_marked;
      Array<int> edge_el_already_marked;
      for (int e = 0; e < edge_els->Size(); e++)
      {
         int elnum = (*edge_els)[e];
         bool el_exists = inter_face_el_all.Find(elnum) != -1;
         if (!el_exists)
         {
            edge_el_need_to_be_marked.Append(elnum);
         }
         else
         {
            edge_el_already_marked.Append(elnum);
         }
      }

      // at this point, every element in need_to_be_marked
      // depends on already_marked.
      for (int e = 0; e < edge_el_need_to_be_marked.Size(); e++)
      {
         int elnum = edge_el_need_to_be_marked[e];
         if (!el_to_el_edge_dep[elnum])
         {
            Array<int> *temp = new Array<int>;
            el_to_el_edge_dep[elnum] = temp;
         }
         el_to_el_edge_dep[elnum]->Append(edge_el_already_marked);
         //           el_to_el_edge_dep[elnum]->Print();
      }
   }
}

void GetMaterialInterfaceFaceDofs(Mesh *mesh, GridFunction &mat,
                                  GridFunction &surf_fit_gf0,
                                  Array<int> &inter_faces,
                                  Array<int> &intdofs)
{
   inter_faces.SetSize(0);
   intdofs.SetSize(0);
   const int NElem = mesh->GetNE();
   MFEM_VERIFY(mat.Size() == NElem, "Material GridFunction should be a piecewise"
               "constant function over the mesh.");
   int dim = mesh->Dimension();
   for (int f = 0; f < mesh->GetNumFaces(); f++ )
   {
      Array<int> els;
      mesh->GetFaceAdjacentElements(f, els);
      if (els.Size() == 2)
      {
         int mat1 = mat(els[0]);
         int mat2 = mat(els[1]);
         Array<int> dofs;
         if (mat1 != mat2)
         {
            inter_faces.Append(f);
            if (dim == 2)
            {
               surf_fit_gf0.FESpace()->GetEdgeDofs(f, dofs);
            }
            else
            {
               surf_fit_gf0.FESpace()->GetFaceDofs(f, dofs);
            }
            intdofs.Append(dofs);
         }
      }
   }
   intdofs.Sort();
   intdofs.Unique();
}


double GetMinDet(Mesh *mesh, FiniteElementSpace *fespace,
                 IntegrationRules *irules, int quad_order)
{
   double tauval = infinity();
   const int NE = mesh->GetNE();
   for (int i = 0; i < NE; i++)
   {
      const IntegrationRule &ir =
         irules->Get(fespace->GetFE(i)->GetGeomType(), quad_order);
      ElementTransformation *transf = mesh->GetElementTransformation(i);

      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }

      const IntegrationRule &ir2 = fespace->GetFE(i)->GetNodes();
      for (int j = 0; j < ir2.GetNPoints(); j++)
      {
         transf->SetIntPoint(&ir2.IntPoint(j));
         tauval = min(tauval, transf->Jacobian().Det());
      }
   }
   return tauval;
}

// Propogates orders from interface elements given by @int_el_list to
// face neighbors. The current orders must be provided by @ordergf, and the
// allowed different either needs to be an integer.
// approach decides whether the difference allowed is maximum or exact.
void PropogateOrdersAcrossInterFace(const GridFunction
                                    &ordergf, //current orders
                                    const Array<int> &int_el_list,
                                    const Array<int> &diff,
                                    const Table &eltoeln,
                                    Array<int> &new_orders,
                                    const int approach = 0)
{
   int nelem = ordergf.FESpace()->GetMesh()->GetNE();
   MFEM_VERIFY(ordergf.Size() == nelem,"Invalid size of current orders");
   new_orders.SetSize(nelem);
   for (int e = 0; e < nelem; e++)
   {
      new_orders[e] = ordergf(e);
   }
   Array<bool> order_propogated(nelem);
   order_propogated = false;

   Array<int> propogate_list(0);
   propogate_list.Append(int_el_list);

   Array<int> propogate_list_new = propogate_list;

   bool done = false;
   int iter = 0;
   int nelupdate = 0;
   while (!done)
   {
      propogate_list = propogate_list_new;
      propogate_list_new.SetSize(0);

      for (int i = 0; i < propogate_list.Size(); i++)
      {
         order_propogated[propogate_list[i]] = true;
      }

      Array<int> propogate_list_current(0);

      for (int i = 0; i < propogate_list.Size(); i++)
      {
         int elem = propogate_list[i];
         Array<int> elem_neighbor_indices;
         eltoeln.GetRow(elem, elem_neighbor_indices);
         int elem_order = new_orders[elem];
         for (int n = 0; n < elem_neighbor_indices.Size(); n++)
         {
            int elem_neighbor_index = elem_neighbor_indices[n];
            if (order_propogated[elem_neighbor_index]) { continue; }
            int current_order = new_orders[elem_neighbor_index];
            int maxdiff = diff.Size() == 0 ? 100 : (n > diff.Size()-1 ? 1 : diff[n]);
            propogate_list_new.Append(elem_neighbor_index);
            // if this element has already been visited in this loop, then
            // keep its current order as we might need it.
            int temp_set_order = propogate_list_current.Find(elem_neighbor_index) != -1 ?
                                 current_order : -1;
            propogate_list_current.Append(elem_neighbor_index);
            // approach = 0; If the interface element is 5, max diff = 2,
            // the neighbor has to be atleast 3 (i.e. 3 || 4 || 5 || 6)
            if (approach == 0)
            {
               std::cout << i << " " << n << " " << maxdiff << " " <<
                         current_order << " " << elem_order-maxdiff <<
                         std::endl;

               if (current_order < elem_order-maxdiff)
               {
                  int set_order = std::max(1, elem_order-maxdiff);
                  if (temp_set_order != -1)
                  {
                     set_order = std::max(set_order, temp_set_order);
                  }
                  new_orders[elem_neighbor_index] = set_order;
                  nelupdate++;
               }
            }
            // approach = 1; If the interface element is 5, max diff = 2,
            // the neighbor has to be exactly 3
            else if (approach == 1)
            {
               if (maxdiff != 100 & current_order != elem_order-maxdiff)
               {
                  int set_order = std::max(1, elem_order-maxdiff);
                  if (temp_set_order != -1)
                  {
                     set_order = std::max(set_order, temp_set_order);
                  }
                  new_orders[elem_neighbor_index] = set_order;
                  nelupdate++;
               }
            }
         }
      }

      if (propogate_list_new.Size() > 0)
      {
         propogate_list_new.Sort();
         propogate_list_new.Unique();
      }
      else
      {
         done = true;
      }
      iter++;
   }
   std::cout << "Number of elements order propogated across face to: " << nelupdate
             <<
             std::endl;
}


// Starts by the max order at interface and propogates orders across its neighbors.
// Then proceeds to the next order in the mesh and propogates across its neighbors.
void PropogateOrdersByMax(const GridFunction &ordergf, //current orders
                          const Array<int> &int_el_list,
                          const Array<int> &diff,
                          const Table &eltoeln,
                          Array<int> &new_orders,
                          const int approach = 0)
{
   int nelem = ordergf.FESpace()->GetMesh()->GetNE();
   MFEM_VERIFY(ordergf.Size() == nelem, "Invalid size of current orders");
   new_orders.SetSize(nelem);
   for (int e = 0; e < nelem; e++)
   {
      new_orders[e] = ordergf(e);
   }
   int cur_max_order = new_orders.Max();
   int n_int_els = int_el_list.Size();

   Array<bool> order_propogated(nelem);
   order_propogated = false;
   Array<int> propogate_list(0);

   for (int e = 0; e < int_el_list.Size(); e++)
   {
      int el = int_el_list[e];
      order_propogated[el] = true; //set true for all interface elements
      if (new_orders[el] == cur_max_order)
      {
         propogate_list.Append(el);
         n_int_els -= 1;
      }
   }

   bool done = false;
   int iter = 0;
   int nelupdate = 0;
   while (!done)
   {
      for (int i = 0; i < propogate_list.Size(); i++)
      {
         order_propogated[propogate_list[i]] = true;
      }

      for (int i = 0; i < propogate_list.Size(); i++)
      {
         int elem = propogate_list[i];
         Array<int> elem_neighbor_indices;
         eltoeln.GetRow(elem, elem_neighbor_indices);
         int elem_order = new_orders[elem];
         for (int n = 0; n < elem_neighbor_indices.Size(); n++)
         {
            int elem_neighbor_index = elem_neighbor_indices[n];
            if (order_propogated[elem_neighbor_index]) { continue; }
            int current_order = new_orders[elem_neighbor_index];
            int maxdiff = diff.Size() == 0 ? 100 : (n > diff.Size()-1 ? 1 : diff[n]);
            // approach = 0; If the interface element is 5, max diff = 2,
            // the neighbor has to be atleast 3 (i.e. 3 || 4 || 5 || 6)
            if (approach == 0)
            {
               if (current_order < elem_order-maxdiff)
               {
                  int set_order = std::max(1, elem_order-maxdiff);
                  new_orders[elem_neighbor_index] = set_order;
                  nelupdate++;
               }
            }
            // approach = 1; If the interface element is 5, max diff = 2,
            // the neighbor has to be exactly 3
            else if (approach == 1)
            {
               if (maxdiff != 100 & current_order != elem_order-maxdiff)
               {
                  int set_order = std::max(1, elem_order-maxdiff);
                  new_orders[elem_neighbor_index] = set_order;
                  nelupdate++;
               }
            }
         }
      }

      propogate_list.SetSize(0);
      if (cur_max_order >= 3)
      {
         cur_max_order -= 1;
         for (int e = 0; e < new_orders.Size(); e++)
         {
            if (new_orders[e] == cur_max_order)
            {
               propogate_list.Append(e);
            }
         }
      }

      if (propogate_list.Size() == 0)
      {
         done = true;
      }
      iter++;
   }
   std::cout << "Number of elements order propogated by max to: " << nelupdate
             <<
             std::endl;
}

void PropogateOrders(const GridFunction &ordergf, //current orders
                     const Array<int> &int_el_list,
                     const Array<int> &diff,
                     const Table &eltoeln,
                     Array<int> &new_orders,
                     const int approach = 0)
{
   PropogateOrdersAcrossInterFace(ordergf, int_el_list, diff, eltoeln, new_orders,
                                  approach);
   GridFunction ordergf2(ordergf);
   for (int e = 0; e < new_orders.Size(); e++)
   {
      ordergf2(e) = new_orders[e];
   }
   PropogateOrdersByMax(ordergf2, int_el_list, diff, eltoeln, new_orders,
                        approach);
}


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

   // For each quadrature point of the element
   for (int i=0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      transf->SetAllIntPoints(&ip);

      double level_set_value = values(i);   // With GridFunction
      error += ip.weight * transf->Face->Weight() * std::pow(level_set_value, 2.0);
   }
   return error;
}

double ComputeIntegrateErrorBG(const FiniteElementSpace* fes,
                               GridFunction* ls_bg,
                               const int el, GridFunction *lss,
                               FindPointsGSLIB &finder,
                               int etype = 0)
{
   double error = 0.0;
   const FiniteElement *fe = fes->GetFaceElement(el);  // Face el
   int intorder = 2*ls_bg->FESpace()->GetMaxElementOrder() + 3;
   const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));

   FaceElementTransformations *transf =
      fes->GetMesh()->GetFaceElementTransformations(el, 31);
   int dim = fes->GetMesh()->Dimension();
   // Coordinates of the quadrature points in the physical space
   Vector vxyz(dim*ir->GetNPoints());
   // Values of the ls fonction at the quadrature points that will be computed on the bg gridfunction
   Vector interp_values(ir->GetNPoints());
   // Compute the coords of the quadrature points in the physical space
   for (int i=0; i < ir->GetNPoints(); i++)
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
   }
   double return_val = error;
   if (etype == 0)
   {
      return_val = error;
   }
   else if (etype == 1)
   {
      return_val = length;
   }
   else if (etype == 2)
   {
      error = error > 0.0 ? std::pow(error, 0.5) : 0.0;
      return_val = error;
   }
   else if (etype == 3)
   {
      error = error > 0.0 ? std::pow(error, 0.5) : 0.0;
      return_val = error/length;
   }
   else
   {
      MFEM_ABORT("Invalid errortype requested.");
   }
   return return_val;
}


void ComputeIntegratedErrorBGonFaces(const FiniteElementSpace* fes,
                                     GridFunction* ls_bg,
                                     Array<int> &inter_faces,
                                     GridFunction *lss,
                                     FindPointsGSLIB &finder,
                                     Vector &error_list,
                                     int etype = 0)
{
   error_list.SetSize(inter_faces.Size());
   for (int f = 0; f < inter_faces.Size(); f++)
   {
      error_list(f) = ComputeIntegrateErrorBG(fes, ls_bg, inter_faces[f],
                                              lss, finder, etype);
   }
}

bool CheckElementValidityAtOrder(Mesh *mesh,
                                 int el,
                                 int el_order)
{
   const GridFunction *x = mesh->GetNodes();
   const FiniteElementSpace *fes = x->FESpace();
   int node_ordering = fes->GetOrdering();
   const int vdim = fes->GetVDim();

   const FiniteElement *fe = fes->GetFE(el);
   ElementTransformation *transf = mesh->GetElementTransformation(el);

   int fe_type = fe->GetGeomType();
   int mesh_dim = mesh->Dimension();

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
   Vector elemvect(0);
   Array<int> dofs;
   fes->GetElementVDofs(el, dofs);
   x->GetSubVector(dofs, elemvect);
   DenseMatrix elemvecMat(elemvect.GetData(), dofs.Size()/vdim, vdim);

   // Get coordinates for new order
   DenseMatrix IntVals(I.Height(), elemvecMat.Width());
   Mult(I, elemvecMat, IntVals);

   IsoparametricTransformation *Tpr = NULL;
   Tpr = new IsoparametricTransformation;
   Tpr->SetFE(feInt);
   Tpr->ElementNo = el;
   Tpr->ElementType = ElementTransformation::ELEMENT;
   Tpr->Attribute = transf->Attribute;
   Tpr->GetPointMat().Transpose(IntVals); // PointMat = PMatI^T

   const int int_order = 2*el_order + 3;
   const IntegrationRule *ir = &(IntRulesLo.Get(geom, int_order));

   double tauval = infinity();
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      Tpr->SetIntPoint(&ip);
      tauval = min(tauval, Tpr->Jacobian().Det());
   }

   delete Tpr;
   delete fecInt;

   return (tauval > 0);
}

// InterfaceElementOrderReduction
// etype = 0 means integrated error using level-set function
// etype = 1 means relative change in length of element.
double InterfaceElementOrderReduction(const Mesh *mesh,
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
   int mesh_dim = mesh->Dimension();

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

   // Get coordinates for new order
   DenseMatrix IntVals(I.Height(), elemvecMat.Width());
   Mult(I, elemvecMat, IntVals);

   // in surf_el_meshes, 0 holds a line mesh, 1 - triangle, and 2 - quad
   int surf_mesh_idx = fe_type - 1;
   Mesh *smesh = surf_el_meshes[surf_mesh_idx];

   smesh->SetCurvature(el_order, false, -1, node_ordering);
   GridFunction *snodes = smesh->GetNodes();
   IsoparametricTransformation *seltrans = new IsoparametricTransformation();
   smesh->GetElementTransformation(0, seltrans);
   seltrans->GetPointMat().Transpose(IntVals);

   const int int_order = 2*el_order + 3;
   const IntegrationRule *ir = &(IntRulesLo.Get(geom, int_order));

   // Coordinates of the quadrature points in the physical space
   Vector vxyz(mesh_dim*ir->GetNPoints());
   // Values of the ls fonction at the quadrature points that will be computed on the bg gridfunction
   Vector interp_values(ir->GetNPoints());
   // Compute the coords of the quadrature points in the physical space
   for (int i=0; i < ir->GetNPoints(); i++)
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

   double return_val = 0.0;
   if (etype == 0)
   {
      return_val = error;
   }
   else if (etype == 1)
   {
      return_val = size;
   }
   else if (etype == 2)
   {
      error = error > 0.0 ? std::pow(error, 0.5) : 0.0;
      return_val = error;
   }
   else if (etype == 3)
   {
      error = error > 0.0 ? std::pow(error, 0.5) : 0.0;
      return_val = error/size;
   }
   else
   {
      MFEM_ABORT("Invalid errortype requested.");
   }


   delete seltrans;
   delete fecInt;
   return return_val;
}

double InterfaceElementOrderReduction2(const Mesh *mesh,
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
