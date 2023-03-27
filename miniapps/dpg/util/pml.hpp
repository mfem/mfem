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

namespace mfem
{

/// Class for setting up a simple Cartesian PML region
class CartesianPML
{
private:
   Mesh *mesh;

   /** 2D array of size (dim,2) representing the length of the PML
       region in each direction */
   Array2D<double> length;

   /// 2D array of size (dim,2) representing the Computational Domain Boundary
   Array2D<double> comp_dom_bdr;

   /// 2D array of size (dim,2) representing the Domain Boundary
   Array2D<double> dom_bdr;

   /** Integer Array identifying elements in the pml
       0: in the pml, 1: not in the pml */
   Array<int> elems;

   /// Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   /** Constructor of the PML region using the mesh @a mesh_ and
       the 2D array of size (dim,2) @a length_ which reprensents the
       length of the PML in each direction. */
   CartesianPML(Mesh *mesh_, const Array2D<double> &length_);

   int dim;
   double omega;
   // Default values for Maxwell
   double epsilon = 1.0;
   double mu = 1.0;
   /// Return Computational Domain Boundary
   const Array2D<double> & GetCompDomainBdr() {return comp_dom_bdr;}

   /// Return Domain Boundary
   const Array2D<double> & GetDomainBdr() {return dom_bdr;}

   /// Return Marker list for elements
   const Array<int> & GetMarkedPMLElements() {return elems;}

   /// Mark element in the PML region
   void SetAttributes(Mesh *mesh_, Array<int> * attrNonPML = nullptr,
                      Array<int> * attrPML = nullptr);

   void SetOmega(double omega_) {omega = omega_;}
   void SetEpsilonAndMu(double epsilon_, double mu_)
   {
      epsilon = epsilon_;
      mu = mu_;
   }
   /// PML complex stretching function
   void StretchFunction(const Vector &x, std::vector<std::complex<double>> &dxs);
};


class PmlCoefficient : public Coefficient
{
private:
   CartesianPML * pml = nullptr;
   double (*Function)(const Vector &, CartesianPML * );
public:
   PmlCoefficient(double (*F)(const Vector &, CartesianPML *), CartesianPML * pml_)
      : pml(pml_), Function(F)
   {}
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      return ((*Function)(transip, pml));
   }
};

class PmlMatrixCoefficient : public MatrixCoefficient
{
private:
   CartesianPML * pml = nullptr;
   void (*Function)(const Vector &, CartesianPML *, DenseMatrix &);
public:
   PmlMatrixCoefficient(int dim, void(*F)(const Vector &, CartesianPML *,
                                          DenseMatrix &),
                        CartesianPML * pml_)
      : MatrixCoefficient(dim), pml(pml_), Function(F)
   {}
   virtual void Eval(DenseMatrix &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(height, width);
      (*Function)(transip, pml, K);
   }
};

/// PML stretching functions: See https://doi.org/10.1006/jcph.1994.1159
// Helmholtz
double detJ_r_function(const Vector & x, CartesianPML * pml);
double detJ_i_function(const Vector & x, CartesianPML * pml);
double abs_detJ_2_function(const Vector & x, CartesianPML * pml);

void Jt_J_detJinv_r_function(const Vector & x, CartesianPML * pml,
                             DenseMatrix & M);
void Jt_J_detJinv_i_function(const Vector & x, CartesianPML * pml,
                             DenseMatrix & M);
void abs_Jt_J_detJinv_2_function(const Vector & x, CartesianPML * pml,
                                 DenseMatrix & M);

// Maxwell
// |J| J^-1 J^-T
void detJ_Jt_J_inv_r_function(const Vector &x, CartesianPML * pml,
                              DenseMatrix &M);
void detJ_Jt_J_inv_i_function(const Vector &x, CartesianPML * pml,
                              DenseMatrix &M);
void abs_detJ_Jt_J_inv_2_function(const Vector &x, CartesianPML * pml,
                                  DenseMatrix &M);

} // namespace mfem
