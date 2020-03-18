#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "complex_additive_schwarz.hpp"
using namespace std;
using namespace mfem;

class STPmlPatchAssembly
{
   FiniteElementSpace *fespace=nullptr;
   SesquilinearForm *bf=nullptr;
   double omega = 0.5;
   int nrlayers = 4;
public:
   int nrpatch, nx, ny, nz;
   MeshPartition * p;
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
   std::vector<Array<int>> ess_tdof_list_ext;

   // constructor
   STPmlPatchAssembly(SesquilinearForm * bf_, Array<int> & ess_tdofs, double omega_, int nrlayers_, int part);

   ~STPmlPatchAssembly();
};


class SourceTransferPrecond : public Solver//
{
private:
   int nrpatch;
   int maxit = 1;
   SesquilinearForm *bf=nullptr;
   int type = 0;
   double theta = 0.5;
   double omega = 0.5;
   int nrlayers;
   int part;
   STPmlPatchAssembly * p;
   const Operator * A;
   Vector B;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;

   
public:
   SourceTransferPrecond(SesquilinearForm * bf_, Array<int> & ess_tdofs, double omega_, int nrlayers_, int i = 0);
   void SetNumSmoothSteps(const int iter) { maxit = iter;}
   void SetLoadVector(Vector load) { B = load;}
   void SetSmoothType(int itype) { type = itype;}
   void SetDumpingParam(const double & dump_param) {theta = dump_param;}
   void SetOmega(const double & omega_) {omega = omega_;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~SourceTransferPrecond();
};
