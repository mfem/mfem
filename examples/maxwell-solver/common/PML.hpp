#pragma once
#include "mfem.hpp"
#include "complex_linalg.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


// Class for setting up a simple Cartesian PML region
class CartesianPML
{
private:
   Mesh *mesh;

   // Length of the PML Region in each direction
   Array2D<double> length;

   // Computational Domain Boundary
   Array2D<double> comp_dom_bdr;

   // Domain Boundary
   Array2D<double> dom_bdr;

   // Integer Array identifying elements in the pml
   // 0: in the pml, 1: not in the pml
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   CartesianPML(Mesh *mesh_,Array2D<double> length_);

   int dim;
   double omega;
   // Return Computational Domain Boundary
   Array2D<double> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<double> GetDomainBdr() {return dom_bdr;}

   // Return Marker list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark element in the PML region
   void SetAttributes(Mesh *mesh_);

   void SetOmega(double omega_) {omega = omega_;}

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs, double omega);
};

class ToroidPML
{
private:
   Mesh *mesh;

   Vector zlim, zpml_thickness; // range in axial direction
   Vector rlim, rpml_thickness; // range in radial direction
   Vector alim, apml_thickness; // range in azimuthal direction

   // Integer Array identifying elements in the pml
   // 0: in the pml, 1: not in the pml
   Array<int> elems;

   double GetAngle(const double x, const double y);

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

   bool zstretch = false;
   bool rstretch = false;
   bool astretch = false;

public:
   // Constructor
   ToroidPML(Mesh *mesh_);

   int dim;
   double omega;
   // Return Computational Domain Boundary

   // Return Domain Boundary
   void GetDomainBdrs(Vector & zlim_, Vector & rlim_, Vector & alim_)
   {
      zlim_.SetSize(2); zlim_ = zlim;
      rlim_.SetSize(2); rlim_ = rlim;
      alim_.SetSize(2); alim_ = alim;
   }

   void SetPmlWidth(const Vector & zpml, const Vector & rpml, const Vector & apml)
   {
      MFEM_VERIFY(zpml.Size() == 2 , "Check zpml size");
      MFEM_VERIFY(rpml.Size() == 2 , "Check rpml size");
      MFEM_VERIFY(apml.Size() == 2 , "Check apml size");
      zpml_thickness = zpml;
      rpml_thickness = rpml;
      apml_thickness = apml;
   }

   void SetPmlAxes(const bool zstretch_, 
                   const bool rstretch_, 
                   const bool astretch_ )
   {
      zstretch = zstretch_;
      rstretch = rstretch_;
      astretch = astretch_;
   }

   // // Return Marker list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark element in the PML region
   void SetAttributes(Mesh *mesh_);

   void SetOmega(double omega_) {omega = omega_;}

   // PML complex stretching function
   // void StretchFunction(const Vector &X, vector<complex<double>> &dxs, double omega);
   void StretchFunction(const Vector &X, ComplexDenseMatrix & J, double omega);
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


// This includes scalar coefficients
class PmlMatrixCoefficient : public MatrixCoefficient
{
private:
   CartesianPML * pml = nullptr;
   void (*Function)(const Vector &, CartesianPML * , DenseMatrix &);
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

class PMLMatrixCoefficient : public MatrixCoefficient
{
private:
   ToroidPML * pml = nullptr;
   void (*Function)(const Vector &, ToroidPML * , DenseMatrix &);
public:
   PMLMatrixCoefficient(int dim, void(*F)(const Vector &, ToroidPML *,
                                              DenseMatrix &),
                            ToroidPML * pml_)
      : MatrixCoefficient(dim), pml(pml_), Function(F)
   {}

   using MatrixCoefficient::Eval;

   virtual void Eval(DenseMatrix &M, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      M.SetSize(height,width);
      (*Function)(transip, pml, M);
   }
};

// Helmholtz pml Functions
double pml_detJ_Re(const Vector & x, CartesianPML * pml);
double pml_detJ_Im(const Vector & x, CartesianPML * pml);
void pml_detJ_JT_J_inv_Re(const Vector & x, CartesianPML * pml , DenseMatrix & M);
void pml_detJ_JT_J_inv_Im(const Vector & x, CartesianPML * pml , DenseMatrix & M);

// Maxwell Pml functions
void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_JT_J_inv_abs(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M);
void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_JT_J_inv_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_inv_JT_J_Re(const Vector &x, ToroidPML * pml, DenseMatrix & M);
void detJ_inv_JT_J_Im(const Vector &x, ToroidPML * pml, DenseMatrix & M);