#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ST.hpp"
using namespace std;
using namespace mfem;


class PSTP : public Solver//
{
private:
   int nrpatch;
   int dim;
   SesquilinearForm *bf=nullptr;
   MeshPartition * povlp;
   MeshPartition * pnovlp;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   const Operator * A;
   Vector B;
   DofMap * ovlp_prob = nullptr;
   DofMap * novlp_prob = nullptr;
   Array<SparseMatrix *> PmlMat;
   Array<KLUSolver *> PmlMatInv;
   Array2D<double> Pmllength;
   mutable Array<Vector * > res;

   SparseMatrix * GetPmlSystemMatrix(int ip);
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;
   void PlotLocalSolution(Vector & sol, socketstream & sol_sock, int ip) const;
   void GetCutOffSolution(Vector & sol, int ip, int direction) const;
   void GetCutOffSol(Vector & sol, int ip, int direction) const;

public:
   PSTP(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   void SetLoadVector(Vector load) { B = load;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~PSTP();
};


