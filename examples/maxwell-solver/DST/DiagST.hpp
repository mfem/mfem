#pragma once
#include "Utilities.hpp"
#include "PML.hpp"
using namespace std;
using namespace mfem;

class DiagST : public Solver//
{
private:
   int nrpatch;
   int dim;
   SesquilinearForm *bf=nullptr;
   MeshPartition * povlp=nullptr;
   MeshPartition * pnovlp=nullptr;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   const Operator * A=nullptr;
   Vector B;
   DofMap * ovlp_prob = nullptr;
   DofMap * novlp_prob = nullptr;
   LocalDofMap * lmap=nullptr;
   Array<SparseMatrix *> PmlMat;
   Array<KLUSolver *> PmlMatInv;
   Array2D<double> Pmllength;
   mutable Array<Vector * > res;

   SparseMatrix * GetPmlSystemMatrix(int ip);
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;
   void GetCutOffSolution(Vector & sol, int ip, int direction, bool local=false) const;

public:
   DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   void SetLoadVector(Vector load) { B = load;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~DiagST();
};


