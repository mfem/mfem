#pragma once

#include "mfem.hpp"
#include "blkschwarzp.hpp"
#include "util.hpp"

using namespace mfem;
using namespace std;


namespace Block_AMS
{
   enum BlkSmootherType{HYPRE, SCHWARZ};
}

class Block_AMSSolver : public Solver 
{
public:
   Block_AMSSolver(Array<int> offsets_, std::vector<ParFiniteElementSpace *>fespaces_);
   virtual void SetOperator(const Operator & ) {}
   virtual void SetOperator(Array2D<HypreParMatrix*> Op);
   void SetOperators();
   void SetTheta(const double a);
   void SetSmootherType(Block_AMS::BlkSmootherType type){sType = type;}
   void SetCycleType(const string c_type);
   void SetNumberofCycles(const int k);

   void Mult(const Vector &r, Vector &z) const;
   virtual ~Block_AMSSolver();   
private:
   int nrmeshes = 1;
   std::vector<ParFiniteElementSpace *> fespaces;
   Block_AMS::BlkSmootherType sType=Block_AMS::BlkSmootherType::SCHWARZ; 

   /// The linear system matrix
   Array2D<HypreParMatrix* > A_array;
   Array2D<HypreParMatrix* > Pi;
   HypreParMatrix *Grad, *Pix, *Piy, *Piz;
   HypreParMatrix *l1A00, *l1A11, *Ah;
   Array<int> offsets, offsetsG, offsetsPi;
   BlockOperator *GtAG, *PxtAPx, *PytAPy, *PztAPz;
   BlockOperator  *A, *G, *Px, *Py, *Pz;
   Operator * D;
   HypreBoomerAMG *G00_inv, *Px00_inv, *Py00_inv, *Pz00_inv;
   HypreBoomerAMG *G11_inv, *Px11_inv, *Py11_inv, *Pz11_inv;;
   BlockDiagonalPreconditioner *blkAMG_G, *blkAMG_Px, *blkAMG_Py, *blkAMG_Pz;
   double theta = 1.0;
   string cycle_type = "023414320"; // 0-Smoother, 1-Grad, 2,3,4-Pix,Piy,Piz
   HypreSmoother * Dh;
   int NumberOfCycles=1;
   
   HypreParMatrix *hGtAG, *hPxtAPx, *hPytAPy, *hPztAPz;
   HypreBoomerAMG *AMG_G, *AMG_Px, *AMG_Py, *AMG_Pz;
   Array2D<HypreParMatrix * > hRAP_G;
   Array2D<HypreParMatrix * > hRAP_Px;
   Array2D<HypreParMatrix * > hRAP_Py;
   Array2D<HypreParMatrix * > hRAP_Pz;


   void DiagAddL1norm();
   void Getrowl1norm(HypreParMatrix *A , Vector &l1norm);
   void GetCorrection(BlockOperator* Tr, BlockOperator* op, BlockDiagonalPreconditioner *prec, Vector &r, Vector &z) const;
   void GetCorrection(BlockOperator* Tr, HypreParMatrix* op, HypreBoomerAMG *prec, Vector &r, Vector &z) const;
};

HypreParMatrix* GetDiscreteGradientOp(ParFiniteElementSpace *fespace);
Array2D<HypreParMatrix *> GetNDInterpolationOp(ParFiniteElementSpace *fespace);