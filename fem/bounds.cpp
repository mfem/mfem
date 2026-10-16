// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of bounds

#include "bounds.hpp"

#include <limits>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace mfem
{

using namespace std;

void PLBound::Setup(const int nb_i, const int ncp_i,
                    const int b_type_i, const int cp_type_i,
                    const real_t tol_i)
{
   MFEM_VERIFY(b_type_i >= 0 && b_type_i <= 2, "Bases not supported. "
               "Please read class description to see supported types.");
   MFEM_VERIFY(cp_type_i == 0 || cp_type_i == 1,
               "Control point type not supported. Please read class "
               "description to see supported types.");
   nb = nb_i;
   ncp = ncp_i;
   b_type = b_type_i;
   cp_type = cp_type_i;
   tol = tol_i;
   lbound.SetSize(nb, ncp);
   ubound.SetSize(nb, ncp);
   nodes.SetSize(nb);
   weights.SetSize(nb);
   control_points.SetSize(ncp);

   auto scalenodes = [](const Vector &in, const real_t a, const real_t b) -> Vector
   {
      Vector outVec(in.Size());
      real_t maxv = in.Max();
      real_t minv = in.Min();
      for (int i = 0; i < in.Size(); i++)
      {
         outVec(i) = a + (b-a)*(in(i)-minv)/(maxv-minv);
      }
      return outVec;
   };
   MFEM_VERIFY(ncp >= 2,"At least 2 control points are required.");

   if (cp_type == 0) // GL + End Point
   {
      control_points(0) = 0.0;
      control_points(ncp-1) = 1.0;
      if (ncp > 2)
      {
         const real_t *x = poly1d.GetPoints(ncp-3, 0);
         MFEM_VERIFY(x, "Error in getting points.");
         for (int i = 0; i < ncp-2; i++)
         {
            control_points(i+1) = x[i];
         }
      }
   }
   else if (cp_type == 1) // Chebyshev
   {
      auto GetChebyshevNodes = [](int n) -> Vector
      {
         Vector cheb(n);
         for (int i = 0; i < n; ++i)
         {
            cheb(i) = -cos(M_PI * (static_cast<real_t>(i) / (n - 1)));
         }
         return cheb;
      };
      control_points = GetChebyshevNodes(ncp);
   }
   else
   {
      MFEM_ABORT("Unsupported interval points. Use [0,1].\n");
   }
   control_points = scalenodes(control_points, 0.0, 1.0); // rescale to [0,1]

   Poly_1D::Basis &basis1d(poly1d.GetBasis(nb-1, b_type));

   // Initialize bounds
   lbound = 0.0;
   ubound = 0.0;

   Vector bmv(nb), bpv(nb), bv(nb); // basis values
   Vector bdmv(nb), bdpv(nb), bdv(nb); // basis derivative values
   Vector vals(3);

   // See Section 3.1.1 of https://arxiv.org/pdf/2501.12349 for explanation of
   // procedure below.
   for (int j = 0; j < ncp; j++)
   {
      real_t x = control_points(j);
      real_t xm = x;
      if (j != 0)
      {
         xm = 0.5*(control_points(j-1)+control_points(j));
      }
      real_t xp = x;
      if (j != ncp-1)
      {
         xp = 0.5*(control_points(j)+control_points(j+1));
      }
      basis1d.Eval(xm, bmv, bdmv);
      basis1d.Eval(xp, bpv, bdpv);
      basis1d.Eval(x, bv);
      real_t dm = x-xm;
      real_t dp = x-xp;
      for (int i = 0; i < nb; i++)
      {
         if (j == 0)
         {
            lbound(i, j) = bv(i);
            ubound(i, j) = bv(i);
         }
         else if (j == ncp-1)
         {
            lbound(i, j) = bv(i);
            ubound(i, j) = bv(i);
         }
         else
         {
            vals(0)  = bv(i);
            vals(1) =  bmv(i) +  dm*bdmv(i);
            vals(2) =  bpv(i) +  dp*bdpv(i);
            lbound(i, j) = vals.Min()-tol; // tolerance for good measure
            ubound(i, j) = vals.Max()+tol; // tolerance for good measure
         }
      }
   }

   IntegrationRule irule(nb);
   if (b_type == 0)
   {
      QuadratureFunctions1D::GaussLegendre(nb, &irule);
      for (int i = 0; i < nb; i++)
      {
         weights(i) = irule.IntPoint(i).weight;
         nodes(i) = irule.IntPoint(i).x;
      }
   }
   else if (b_type == 1)
   {
      QuadratureFunctions1D::GaussLobatto(nb, &irule);
      for (int i = 0; i < nb; i++)
      {
         weights(i) = irule.IntPoint(i).weight;
         nodes(i) = irule.IntPoint(i).x;
      }
   }
   else if (b_type == 2)
   {
      QuadratureFunctions1D::ClosedUniform(nb, &irule);
      for (int i = 0; i < nb; i++)
      {
         weights(i) = irule.IntPoint(i).weight;
         nodes(i) = irule.IntPoint(i).x;
      }
   }

   if (b_type == 2)
   {
      nodes_int.SetSize(nb);
      weights_int.SetSize(nb);
      IntegrationRule irule_int(nb);
      {
         QuadratureFunctions1D::GaussLobatto(nb, &irule_int);
         for (int i = 0; i < nb; i++)
         {
            weights_int(i) = irule_int.IntPoint(i).weight;
            nodes_int(i) = irule_int.IntPoint(i).x;
         }
      }

      SetupBernsteinBasisMat(basisMatNodes, nodes);
      // Setup memory for lu factors
      basisMatLU = basisMatNodes;
      lu_ip.SetSize(nb);
      // Compute lu factors
      LUFactors lu(basisMatLU.GetData(), lu_ip.GetData());
      bool factor = lu.Factor(nb);
      MFEM_VERIFY(factor,"Failure in LU factorization in PLBound.");

      // Setup the Bernstein basis matrix for the GLL integration points. This
      // is used to compute linear fit.
      SetupBernsteinBasisMat(basisMatInt, nodes_int);
   }
   else
   {
      nodes_int.SetDataAndSize(nodes.GetData(), nb);
      weights_int.SetDataAndSize(weights.GetData(), nb);
   }
}

PLBound::PLBound(FiniteElementSpace *fes, int ncp_i, int cp_type_i)
{
   MFEM_VERIFY(!fes->IsVariableOrder(),
               "Variable order meshes not yet supported.");
   const char *name = fes->FEColl()->Name();
   string cname = name;

   cp_type = cp_type_i;
   b_type = BasisType::Invalid;
   nb = fes->GetMaxElementOrder()+1;
   tol = 0.0;

   int minncp = 2;
   if (nb > 12)
   {
      minncp = 2*nb;
   }
   else if (!strncmp(name, "H1_", 3) && strncmp(name, "H1_Trace_", 9))
   {
      // H1 GLL
      b_type = BasisType::GaussLobatto;
      minncp = min_ncp_gll_x[cp_type][nb-2];
   }
   else if (!strncmp(name, "H1Pos_", 6) && strncmp(name, "H1Pos_Trace_", 12))
   {
      // H1 Positive
      b_type = BasisType::Positive;
      minncp = min_ncp_pos_x[cp_type][nb-2];
   }
   else if (!strncmp(name, "L2_", 3) && strncmp(name, "L2_T", 4))
   {
      // L2 Gauss-Legendre
      b_type = BasisType::GaussLegendre;
      minncp = min_ncp_gl_x[cp_type][nb-2];
   }
   else if (!strncmp(name, "L2_T1", 5))
   {
      // L2 GLL
      b_type = BasisType::GaussLobatto;
      minncp = min_ncp_gll_x[cp_type][nb-2];
   }
   else if (!strncmp(name, "L2_T2", 5))
   {
      // L2 Positive
      b_type = BasisType::Positive;
      minncp = min_ncp_pos_x[cp_type][nb-2];
   }
   else
   {
      MFEM_ABORT("Only H1 GLL/Positive & L2 GL/GLL/Positive bases supported.");
   }

   ncp = std::max(minncp, ncp_i);

   Setup(nb, ncp, b_type, cp_type, tol);
}

void PLBound::Get1DBounds(Vector &coeff, Vector &intmin, Vector &intmax) const
{
   real_t x,w;
   intmin.SetSize(ncp);
   intmax.SetSize(ncp);
   intmin = 0.0;
   intmax = 0.0;
   Vector coeffm(nb);
   coeffm = 0.0;

   real_t a0 = 0.0;
   real_t a1 = 0.0;

   Vector nodal_vals, nodal_integ_vals;
   if (b_type == 2) // compute values at equispaced nodes and GLL nodes
   {
      nodal_vals.SetSize(nb);
      nodal_integ_vals.SetSize(nb);
      Vector shape(nb);
      for (int i = 0; i < nb; i++)
      {
         basisMatNodes.GetRow(i, shape);
         nodal_vals(i) = shape*coeff;
         basisMatInt.GetRow(i, shape);
         nodal_integ_vals(i) = shape*coeff;
      }
   }
   else
   {
      nodal_vals.SetDataAndSize(coeff.GetData(), nb);
      nodal_integ_vals.SetDataAndSize(coeff.GetData(), nb);
   }

   // compute L2 projection for linear bases: a0 + a1*x
   if (proj)
   {
      for (int i = 0; i < nb; i++)
      {
         x = 2.0*nodes_int(i)-1;
         w = 2.0*weights_int(i);
         a0 += 0.5*nodal_integ_vals(i)*w;
         a1 += 1.5*nodal_integ_vals(i)*w*x;
      }

      // offset the linear fit from nodal values
      for (int i = 0; i < nb; i++)
      {
         x = 2.0*nodes(i)-1;
         coeffm(i) = nodal_vals(i) - a0 - a1*x;
      }

      // compute coefficients for Bernstein
      if (b_type == 2)
      {
         LUFactors lu(basisMatLU.GetData(), lu_ip.GetData());
         lu.Solve(nb, 1, coeffm.GetData());
      }

      // initialize the bounds to be the linear fit
      for (int j = 0; j < ncp; j++)
      {
         x = 2.0*control_points(j)-1;
         intmin(j) = a0 + a1*x;
         intmax(j) = intmin(j);
      }
   }
   else
   {
      coeffm.SetDataAndSize(coeff.GetData(), nb);
   }

   for (int i = 0; i < nb; i++)
   {
      real_t c = coeffm(i);
      for (int j = 0; j < ncp; j++)
      {
         intmin(j) += min(lbound(i,j)*c, ubound(i,j)*c);
         intmax(j) += max(lbound(i,j)*c, ubound(i,j)*c);
      }
   }
}

void PLBound::Get2DBounds(Vector &coeff, Vector &intmin, Vector &intmax) const
{
   intmin.SetSize(ncp*ncp);
   intmax.SetSize(ncp*ncp);
   intmin = 0.0;
   intmax = 0.0;
   Vector intminT(ncp*nb);
   Vector intmaxT(ncp*nb);
   // Get bounds for each row of the solution
   for (int i = 0; i < nb; i++)
   {
      Vector solcoeff(coeff.GetData()+i*nb, nb);
      Vector intminrow(intminT.GetData()+i*ncp, ncp);
      Vector intmaxrow(intmaxT.GetData()+i*ncp, ncp);
      Get1DBounds(solcoeff, intminrow, intmaxrow);
   }
   Vector intminT2 = intminT;

   // Compute a0 and a1 for each column of nodes
   Vector a0V(ncp), a1V(ncp);
   a0V = 0.0;
   a1V = 0.0;
   real_t x,w,t;
   if (proj)
   {
      if (b_type == 2)
      {
         // Note: DenseMatrix uses column-major ordering so we will need to
         // transpose the matrix.
         DenseMatrix intminTM(intminT.GetData(), ncp, nb),
                     intmaxTM(intmaxT.GetData(), ncp, nb),
                     intmeanTM(ncp, nb);
         DenseMatrix minvalsM(nb, ncp), maxvalsM(nb, ncp), meanintvalsM(nb, ncp);
         MultABt(basisMatNodes, intminTM, minvalsM);
         MultABt(basisMatNodes, intmaxTM, maxvalsM);
         intmeanTM = intminTM;
         intmeanTM += intmaxTM;
         intmeanTM *= 0.5;
         MultABt(basisMatInt, intmeanTM, meanintvalsM);

         // Compute the linear fit along each column and then offset it from
         // the bounds on the coefficient.
         // Note: Since Bernstein bases are positive, we can use the lower
         // bounds to compute the lower bounding polynomial and subtract the
         // linear fit before finding the Bernstein coefficients corresponding
         // to the perturbation. Same for upper bounds. If the bases were not
         // always positive, it is not yet clear if the perturbation
         // coefficients will be this straightforward to compute.
         for (int j = 0; j < ncp; j++) // row of interval points
         {
            for (int i = 0; i < nb; i++)
            {
               x = 2.0*nodes_int(i)-1; // x-coordinate
               w = 2.0*weights_int(i); // weight
               t = meanintvalsM(i,j);
               a0V(j) += 0.5*t*w;
               a1V(j) += 1.5*t*w*x;
            }
            // Offset linear fit
            for (int i = 0; i < nb; i++)
            {
               x = 2.0*nodes(i)-1; // x-coordinate
               minvalsM(i,j) -= a0V(j) + a1V(j)*x;
               maxvalsM(i,j) -= a0V(j) + a1V(j)*x;
            }
            // Compute Bernstein coefficients
            LUFactors lu(basisMatLU.GetData(), lu_ip.GetData());
            lu.Solve(nb, 1, minvalsM.GetColumn(j));
            lu.Solve(nb, 1, maxvalsM.GetColumn(j));
            for (int i = 0; i < nb; i++)
            {
               intminT(i*ncp+j) = minvalsM(i,j);
               intmaxT(i*ncp+j) = maxvalsM(i,j);
            }
         }
      }
      else
      {
         for (int j = 0; j < nb; j++) // row of nodes
         {
            x = 2.0*nodes(j)-1; // x-coordinate
            w = 2.0*weights(j); // weight
            for (int i = 0; i < ncp; i++) // column of interval points
            {
               t = 0.5*(intminT(j*ncp+i)+intmaxT(j*ncp+i));
               a0V(i) += 0.5*t*w;
               a1V(i) += 1.5*t*w*x;
            }
         }
         // offset the linear fit from nodal values
         for (int j = 0; j < nb; j++) // row of nodes
         {
            x = 2.0*nodes(j)-1; // x-coordinate
            for (int i = 0; i < ncp; i++) // column of interval points
            {
               t = a0V(i) + a1V(i)*x;
               intminT(j*ncp+i) -= t;
               intmaxT(j*ncp+i) -= t;
            }
         }
      }

      // Initialize bounds using a0 and a1 values
      for (int j = 0; j < ncp; j++) // row j
      {
         x = 2.0*control_points(j)-1;
         for (int i = 0; i < ncp; i++) // column i
         {
            intmin(j*ncp+i) = a0V(i) + a1V(i)*x;
            intmax(j*ncp+i) = intmin(j*ncp+i);
         }
      }
   }

   // Compute bounds
   int id1 = 0, id2 = 0;
   Vector vals(4);
   for (int j = 0; j < nb; j++)
   {
      for (int i = 0; i < ncp; i++) // ith column
      {
         real_t w0 = intminT(id1++);
         real_t w1 = intmaxT(id2++);
         for (int k = 0; k < ncp; k++) // kth row
         {
            vals(0) = w0*lbound(j,k);
            vals(1) = w0*ubound(j,k);
            vals(2) = w1*lbound(j,k);
            vals(3) = w1*ubound(j,k);
            intmin(k*ncp+i) += vals.Min();
            intmax(k*ncp+i) += vals.Max();
         }
      }
   }
}

void PLBound::Get3DBounds(Vector &coeff, Vector &intmin, Vector &intmax) const
{
   int nb2 = nb*nb,
       ncp2 = ncp*ncp,
       ncp3 = ncp*ncp*ncp;

   intmin.SetSize(ncp3);
   intmax.SetSize(ncp3);
   intmin = 0.0;
   intmax = 0.0;
   Vector intminT(ncp2*nb);
   Vector intmaxT(ncp2*nb);

   // Get bounds for each slice of the solution
   for (int i = 0; i < nb; i++)
   {
      Vector solcoeff(coeff.GetData()+i*nb2, nb2);
      Vector intminrow(intminT.GetData()+i*ncp2, ncp2);
      Vector intmaxrow(intmaxT.GetData()+i*ncp2, ncp2);
      Get2DBounds(solcoeff, intminrow, intmaxrow);
   }
   DenseMatrix intminTM(intminT.GetData(), ncp2, nb),
               intmaxTM(intmaxT.GetData(), ncp2, nb);

   // Compute a0 and a1 for each tower of nodes
   Vector a0V(ncp2), a1V(ncp2);
   a0V = 0.0;
   a1V = 0.0;
   real_t x,w,t;
   if (proj)
   {
      if (b_type == 2) // Bernstein bases
      {
         // Compute the mean coefficients along each tower.
         for (int j = 0; j < ncp2; j++) // slice of interval points
         {
            Vector meanBounds(nb), minBounds(nb), maxBounds(nb);
            intminTM.GetRow(j, minBounds);
            intmaxTM.GetRow(j, maxBounds);
            for (int i = 0; i < nb; i++) // column of nodes
            {
               meanBounds(i) = 0.5*(minBounds(i)+maxBounds(i));
            }
            Vector meanNodalIntVals(nb);
            Vector minNodalVals(nb);
            Vector maxNodalVals(nb);
            Vector row(nb);
            for (int i = 0; i < nb; i++)
            {
               basisMatNodes.GetRow(i, row);
               minNodalVals(i) = row*minBounds;
               maxNodalVals(i) = row*maxBounds;
               basisMatInt.GetRow(i, row);
               meanNodalIntVals(i) = row*meanBounds;
            }
            // linear fit along each tower
            for (int i = 0; i < nb; i++)
            {
               x = 2.0*nodes_int(i)-1; // x-coordinate
               w = 2.0*weights_int(i); // weight
               a0V(j) += 0.5*meanNodalIntVals(i)*w;
               a1V(j) += 1.5*meanNodalIntVals(i)*w*x;
            }
            // offset the linear fit from bounding coefficients
            for (int i = 0; i < nb; i++)
            {
               x = 2.0*nodes(i)-1; // x-coordinate
               minBounds(i) -= a0V(j) + a1V(j)*x;
               maxBounds(i) -= a0V(j) + a1V(j)*x;
            }
            // Compute Bernstein coefficients
            LUFactors lu(basisMatLU.GetData(), lu_ip.GetData());
            lu.Solve(nb, 1, minBounds.GetData());
            lu.Solve(nb, 1, maxBounds.GetData());
            for (int i = 0; i < nb; i++)
            {
               intminT(i*ncp2+j) = minBounds(i);
               intmaxT(i*ncp2+j) = maxBounds(i);
            }
         }
      }
      else
      {
         // nodal bases
         for (int j = 0; j < nb; j++) // tower of nodes
         {
            x = 2.0*nodes(j)-1; // x-coordinate
            w = 2.0*weights(j); // weight
            for (int i = 0; i < ncp2; i++) // slice of interval points
            {
               t = 0.5*(intminT(j*ncp2+i)+intmaxT(j*ncp2+i));
               a0V(i) += 0.5*t*w;
               a1V(i) += 1.5*t*w*x;
            }
         }
         // offset the linear fit from nodal values
         for (int j = 0; j < nb; j++) // row of nodes
         {
            x = 2.0*nodes(j)-1; // x-coordinate
            for (int i = 0; i < ncp2; i++) // column of interval points
            {
               t = a0V(i) + a1V(i)*x;
               intminT(j*ncp2+i) -= t;
               intmaxT(j*ncp2+i) -= t;
            }
         }
      }

      // Initialize bounds using a0 and a1 values
      for (int j = 0; j < ncp; j++) // slice j
      {
         x = 2.0*control_points(j)-1;
         for (int i = 0; i < ncp2; i++) // tower i
         {
            intmin(j*ncp2+i) = a0V(i) + a1V(i)*x;
            intmax(j*ncp2+i) = a0V(i) + a1V(i)*x;
         }
      }
   }

   // Compute bounds
   int id1 = 0, id2 = 0;
   Vector vals(4);
   for (int j = 0; j < nb; j++)
   {
      for (int i = 0; i < ncp2; i++) // ith tower
      {
         real_t w0 = intminT(id1++);
         real_t w1 = intmaxT(id2++);
         for (int k = 0; k < ncp; k++) // kth slice
         {
            vals(0) = w0*lbound(j,k);
            vals(1) = w0*ubound(j,k);
            vals(2) = w1*lbound(j,k);
            vals(3) = w1*ubound(j,k);
            intmin(k*ncp2+i) += vals.Min();
            intmax(k*ncp2+i) += vals.Max();
         }
      }
   }
}

void PLBound::GetNDBounds(int rdim, Vector &coeff,
                          Vector &intmin, Vector &intmax) const
{
   if (rdim == 1)
   {
      Get1DBounds(coeff, intmin, intmax);
   }
   else if (rdim == 2)
   {
      Get2DBounds(coeff, intmin, intmax);
   }
   else if (rdim == 3)
   {
      Get3DBounds(coeff, intmin, intmax);
   }
   else
   {
      MFEM_ABORT("Currently not supported.");
   }
}

void PLBound::SetupBernsteinBasisMat(DenseMatrix &basisMat,
                                     Vector &nodesBern) const
{
   const int nbern = nodesBern.Size();
   L2_SegmentElement el(nbern-1, 2); // we use L2 to leverage lexicographic order
   Array<int> ordering = el.GetLexicographicOrdering();
   basisMat.SetSize(nbern, nbern);
   Vector shape(nbern);
   IntegrationPoint ip;
   for (int i = 0; i < nbern; i++)
   {
      ip.x = nodesBern(i);
      el.CalcShape(ip, shape);
      basisMat.SetRow(i, shape);
   }
}

constexpr int PLBound::min_ncp_gl_x[2][11];
constexpr int PLBound::min_ncp_gll_x[2][11];
constexpr int PLBound::min_ncp_pos_x[2][11];

int PLBound::GetMinimumPointsForGivenBases(int nb_i, int b_type_i,
                                           int cp_type_i) const
{
   MFEM_VERIFY(b_type_i >= 0 && b_type_i <= 2, "Invalid node type. Specify 0 "
               "for GL, 1 for GLL, and 2 for positive " "bases.");
   MFEM_VERIFY(cp_type_i == 0 || cp_type_i == 1, "Invalid control point type. "
               "Specify 0 for GL+end points, 1 for Chebyshev.");
   if (nb_i > 12)
   {
      MFEM_ABORT("GetMinimumPointsForGivenBases can only be used for maximum "
                 "order = 11, i.e. nb=12. 2*nb points should be sufficient to "
                 "bound the bases up to nb = 30.");
   }
   else if (b_type_i == 0)
   {
      return min_ncp_gl_x[cp_type_i][nb_i-2];
   }
   else if (b_type_i == 1)
   {
      return min_ncp_gll_x[cp_type_i][nb_i-2];
   }
   else if (b_type_i == 2)
   {
      return min_ncp_pos_x[cp_type_i][nb_i-2];
   }
   return 0;
}

void PLBound::Print(std::ostream &outp) const
{
   outp << "PLBound nb: " << nb << std::endl;
   outp << "PLBound ncp: " << ncp << std::endl;
   outp << "PLBound b_type: " << b_type << std::endl;
   outp << "PLBound cp_type: " << cp_type << std::endl;
   outp << "Print nodes: " << std::endl;
   nodes.Print(outp);
   outp << "Print weights: " << std::endl;
   weights.Print(outp);
   outp << "Print control_points: " << std::endl;
   control_points.Print(outp);
   outp << "Print lower bounds: " << std::endl;
   lbound.Print(outp);
   outp << "Print upper bounds: " << std::endl;
   ubound.Print(outp);
}

}
