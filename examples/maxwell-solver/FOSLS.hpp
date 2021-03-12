#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
using namespace std;
using namespace mfem;


class ComplexMaxwellFOSLS 
{
private:
   ParFiniteElementSpace * fes = nullptr;
   ParMesh * pmesh = nullptr;
   double omega = 1.0;
   Array<VectorFunctionCoefficient *> loads;
   Array<VectorFunctionCoefficient *> ess_data;
   Array2D<HypreParMatrix *> A;
   BlockVector x,rhs;
   BlockVector X,Rhs;
   Array<int> block_offsets;
   Array<int> block_trueOffsets;

   void FormSystem(bool system = true);
public:
   ComplexMaxwellFOSLS(ParFiniteElementSpace * fes_);
   void SetOmega(double omega_) { omega = omega_; }
   void SetLoadData(Array<VectorFunctionCoefficient *> & loads_);
   void SetEssentialData(Array<VectorFunctionCoefficient *> & ess_data_);
   void GetFOSLSLinearSystem(Array2D<HypreParMatrix *> & A_, 
                             BlockVector & X_,
                             BlockVector & Rhs_);
   void GetFOSLSMatrix(Array2D<HypreParMatrix *> & A_);
};


   // -------------------------------------------------------------------
   // |   |            p             |           u           |   RHS    | 
   // -------------------------------------------------------------------
   // | q | (gradp,gradq) + w^2(p,q) | w(divu,q)-w(u, gradq) |  w(f,q)  |
   // |   |                          |                       |          |
   // | v | w(p,divv) - w(gradp,v)   | (divu,divv) + w^2(u,v)| (f,divv) |

class HelmholtzFOSLS 
{
private:
   Array<ParFiniteElementSpace * > fes;
   bool definite;
   bool complex;
   ParMesh * pmesh = nullptr;
   double omega = 1.0;
   int n; // 
   Array<FunctionCoefficient * > f;
   Array<VectorFunctionCoefficient * > Q;
   Array<FunctionCoefficient * > p_ex_coeff;
   Array<VectorFunctionCoefficient * > u_ex_coeff;
   Array2D<HypreParMatrix *> A;
   BlockVector x,rhs;
   BlockVector X,Rhs;
   Array<int> block_offsets;
   Array<int> block_trueOffsets;

   void FormSystem(bool system = true);
   void Init();
public:
   HelmholtzFOSLS(Array<ParFiniteElementSpace * > & fes_, 
                  bool definite_ = false,
                  bool complex_ = false);
   void SetOmega(double omega_) { omega = omega_; }
   void SetLoadData(Array<FunctionCoefficient * > & f_);
   void SetLoadData(Array<VectorFunctionCoefficient * > & Q_);
   void SetEssentialData(Array<FunctionCoefficient * > & p_ex_coeff_);
   void SetEssentialData(Array<VectorFunctionCoefficient * > & u_ex_coeff_);
   void GetFOSLSLinearSystem(Array2D<HypreParMatrix *> & A_, 
                             BlockVector & X_,
                             BlockVector & Rhs_);
   void GetFOSLSMatrix(Array2D<HypreParMatrix *> & A_);
};