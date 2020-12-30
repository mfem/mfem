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

   Array< SesquilinearForm * > sqf;
   Array< OperatorPtr * > Optr;
   Array<ComplexSparseMatrix *> PmlMat;
   Array<ComplexUMFPackSolver *> PmlMatInv;

   void SetupSubdomainProblems();
   void SetMaxwellPmlSystemMatrix(int ip);
   

public:
   ToroidST(SesquilinearForm * bf_, const Vector & aPmlThickness_, 
       double omega_, int nrsubdomains_ = 2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ToroidST();
};