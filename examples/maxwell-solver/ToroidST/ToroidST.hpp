#pragma once

#include "../common/PML.hpp"
#include "DofMaps.hpp"
using namespace std;
using namespace mfem;
class ToroidST : public Solver//
{
private:
   SesquilinearForm *bf=nullptr;
   FiniteElementSpace * fes = nullptr;
   Mesh * mesh = nullptr;
   double omega;
   int nrsubdomains;
   Vector aPmlThickness; 
   double overlap,ovlp; 
   Array<FiniteElementSpace *> fespaces;
   Array<Array<int> *> DofMaps0, DofMaps1, OvlpMaps0, OvlpMaps1;


   Array< SesquilinearForm * > sqf;
   Array< OperatorPtr * > Optr;
   Array<ComplexSparseMatrix *> PmlMat;
   Array<ComplexUMFPackSolver *> PmlMatInv;
   mutable Array<Vector *> f_orig;
   mutable Array<Vector *> forward_transf;
   mutable Array<Vector *> backward_transf;
   void SetupSubdomainProblems();
   void SetMaxwellPmlSystemMatrix(int ip);
   // sweep 1: forward
   // sweep -1: backward
   void SourceTransfer(int ip, const Vector & sol, int sweep) const;
public:
   ToroidST(SesquilinearForm * bf_, const Vector & aPmlThickness_, 
       double omega_, int nrsubdomains_ = 2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ToroidST();
};