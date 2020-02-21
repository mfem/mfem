#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "additive_schwarz.hpp"
using namespace std;
using namespace mfem;

class ComplexPatchAssembly
{
   FiniteElementSpace *fespace=nullptr;
   SesquilinearForm *bf=nullptr;
public:
   int nrpatch;
   Array<FiniteElementSpace *> patch_fespaces;
   std::vector<Array<int>> patch_dof_map;
   std::vector<Array<int>> complex_patch_dof_map;
   Array<SparseMatrix *> patch_mat;
   Array<KLUSolver * > patch_mat_inv;
   std::vector<Array<int>> ess_tdof_list;

   // constructor
   ComplexPatchAssembly(SesquilinearForm * bf_, Array<int> & ess_tdofs, int part);
   ~ComplexPatchAssembly();
};


class ComplexAddSchwarz : public Solver//
{
private:
   int nrpatch;
   int maxit = 1;
   int part;
   double theta = 0.5;
   ComplexPatchAssembly * p;
   const Operator * A;
public:
   ComplexAddSchwarz(SesquilinearForm * bf_, Array<int> & ess_tdofs, int i = 0);
   void SetNumSmoothSteps(const int iter) { maxit = iter;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ComplexAddSchwarz();
};


