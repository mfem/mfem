// #pragma once
// #include "mfem.hpp"
// #include <fstream>
// #include <iostream>

// using namespace std;
// using namespace mfem;

// // Class for setting up a simple Cartesian PML region
// class CartesianPML
// {
// private:
//    Mesh *mesh;

//    // Length of the PML Region in each direction
//    Array2D<double> length;

//    // Computational Domain Boundary
//    Array2D<double> comp_dom_bdr;

//    // Domain Boundary
//    Array2D<double> dom_bdr;

//    // Integer Array identifying elements in the pml
//    // 0: in the pml, 1: not in the pml
//    Array<int> elems;

//    // Compute Domain and Computational Domain Boundaries
//    void SetBoundaries();

// public:
//    // Constructor
//    CartesianPML(Mesh *mesh_,Array2D<double> length_);

//    int dim;
//    double omega;
//    // Return Computational Domain Boundary
//    Array2D<double> GetCompDomainBdr() {return comp_dom_bdr;}

//    // Return Domain Boundary
//    Array2D<double> GetDomainBdr() {return dom_bdr;}

//    // Return Marker list for elements
//    Array<int> * GetMarkedPMLElements() {return &elems;}

//    // Mark element in the PML region
//    void SetAttributes(Mesh *mesh_);

//    void SetOmega(double omega_) {omega = omega_;}

//    // PML complex stretching function
//    void StretchFunction(const Vector &x, vector<complex<double>> &dxs, double omega);
// };


// class PmlCoefficient : public Coefficient
// {
// private:
//    CartesianPML * pml = nullptr;
//    double (*Function)(const Vector &, CartesianPML * );
// public:
//    PmlCoefficient(double (*F)(const Vector &, CartesianPML *), CartesianPML * pml_)
//       : pml(pml_), Function(F)
//    {}
//    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
//    {
//       double x[3];
//       Vector transip(x, 3);
//       T.Transform(ip, transip);
//       return ((*Function)(transip, pml));
//    }
// };


// // This includes scalar coefficients
// class PmlMatrixCoefficient : public MatrixCoefficient
// {
// private:
//    CartesianPML * pml = nullptr;
//    void (*Function)(const Vector &, CartesianPML * , DenseMatrix &);
// public:
//    PmlMatrixCoefficient(int dim, void(*F)(const Vector &, CartesianPML *,
//                                           DenseMatrix &),
//                         CartesianPML * pml_)
//       : MatrixCoefficient(dim), pml(pml_), Function(F)
//    {}
//    virtual void Eval(DenseMatrix &K, ElementTransformation &T,
//                      const IntegrationPoint &ip)
//    {
//       double x[3];
//       Vector transip(x, 3);
//       T.Transform(ip, transip);
//       K.SetSize(height, width);
//       (*Function)(transip, pml, K);
//    }
// };

// double pml_detJ_Re(const Vector & x, CartesianPML * pml);
// double pml_detJ_Im(const Vector & x, CartesianPML * pml);
// void pml_detJ_JT_J_inv_Re(const Vector & x, CartesianPML * pml , DenseMatrix & M);
// void pml_detJ_JT_J_inv_Im(const Vector & x, CartesianPML * pml , DenseMatrix & M);
