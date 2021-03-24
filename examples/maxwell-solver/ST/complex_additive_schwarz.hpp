#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "additive_schwarz.hpp"
#include "pml.hpp"
using namespace std;
using namespace mfem;

class ComplexPatchAssembly
{
   FiniteElementSpace *fespace=nullptr;
   SesquilinearForm *bf=nullptr;
public:
   int nrpatch, nx, ny, nz;
   Array<FiniteElementSpace *> patch_fespaces;
   Array<FiniteElementSpace *> patch_fespaces_ext;
   Array<Mesh *> patch_meshes_ext;
   std::vector<Array<int>> patch_dof_map;
   std::vector<Array<int>> complex_patch_dof_map;
   std::vector<Array<int>> dof2extdof_map;
   Array<SparseMatrix *> patch_mat;
   Array<SparseMatrix *> patch_mat_ext;
   Array<KLUSolver * > patch_mat_inv_ext;
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
   SesquilinearForm *bf=nullptr;
   int part;
   int type = 0;
   double theta = 0.5;
   ComplexPatchAssembly * p;
   const Operator * A;
   Vector B;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;

   
public:
   ComplexAddSchwarz(SesquilinearForm * bf_, Array<int> & ess_tdofs, int i = 0);
   void SetNumSmoothSteps(const int iter) { maxit = iter;}
   void SetLoadVector(Vector load) { B = load;}
   void SetSmoothType(int itype) { type = itype;}
   void SetDumpingParam(const double dump_param) {theta = dump_param;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ComplexAddSchwarz();
};
