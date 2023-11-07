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
   Array2D<fptype> length;

   /// 2D array of size (dim,2) representing the Computational Domain Boundary
   Array2D<fptype> comp_dom_bdr;

   /// 2D array of size (dim,2) representing the Domain Boundary
   Array2D<fptype> dom_bdr;

   /** Integer Array identifying elements in the pml
       0: in the pml, 1: not in the pml */
   Array<int> elems;

   /// Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   /** Constructor of the PML region using the mesh @a mesh_ and
       the 2D array of size (dim,2) @a length_ which represents the
       length of the PML in each direction. */
   CartesianPML(Mesh *mesh_, const Array2D<fptype> &length_);

   int dim;
   fptype omega;
   // Default values for Maxwell
   fptype epsilon = 1.0;
   fptype mu = 1.0;
   /// Return Computational Domain Boundary
   const Array2D<fptype> & GetCompDomainBdr() {return comp_dom_bdr;}

   /// Return Domain Boundary
   const Array2D<fptype> & GetDomainBdr() {return dom_bdr;}

   /// Return Marker list for elements
   const Array<int> & GetMarkedPMLElements() {return elems;}

   /// Mark element in the PML region
   void SetAttributes(Mesh *mesh_, Array<int> * attrNonPML = nullptr,
                      Array<int> * attrPML = nullptr);

   void SetOmega(fptype omega_) {omega = omega_;}
   void SetEpsilonAndMu(fptype epsilon_, fptype mu_)
   {
      epsilon = epsilon_;
      mu = mu_;
   }
   /// PML complex stretching function
   void StretchFunction(const Vector &x, std::vector<std::complex<fptype>> &dxs);
};


class PmlCoefficient : public Coefficient
{
private:
   CartesianPML * pml = nullptr;
   fptype (*Function)(const Vector &, CartesianPML * );
public:
   PmlCoefficient(fptype (*F)(const Vector &, CartesianPML *), CartesianPML * pml_)
      : pml(pml_), Function(F)
   {}
   virtual fptype Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      fptype x[3];
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
      fptype x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(height, width);
      (*Function)(transip, pml, K);
   }
};

/// PML stretching functions: See https://doi.org/10.1006/jcph.1994.1159
// Helmholtz
fptype detJ_r_function(const Vector & x, CartesianPML * pml);
fptype detJ_i_function(const Vector & x, CartesianPML * pml);
fptype abs_detJ_2_function(const Vector & x, CartesianPML * pml);

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
