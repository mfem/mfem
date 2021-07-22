
#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

IntegrationRule * TensorIntegrationRule(const FiniteElementSpace & fes, int order);
IntegrationRule * TensorIntegrationRule(int dim, int order);

void KronMult(const Vector & x, const Vector & y, Vector & z);
void KronMult(const Vector & x, const Vector & y, const Vector & z, Vector & w);


void AlterLS(DenseMatrix & T, Vector & vecA, Vector & vecB);

void AlterLS(DenseTensor & T, Vector & vecA, Vector & vecB, Vector & vecC);

class ElementTPFunctionCoefficient : public Coefficient//
{
private:
   int dim;
   DenseMatrix A;
   DenseTensor T;
   double coeff_avg;
   Vector VecX;
   Vector VecY;
   Vector VecZ;
   int orient;
   int nint; // total num of integrations points
   int mint=0; // counter for all the integrations points
   int nintx = 0; // counter for the x integrations points
   int ninty = 0; // counter for the y integration points
   int nintz = 0; // counter for the z integration points
   int coord = 0; // (indication flag for x,y or z coordinate)

public:
   ElementTPFunctionCoefficient(FiniteElementSpace &fes, int iel, Coefficient &cf);
   double GetCoeffAvg() {return coeff_avg;}
   void ResetCounters() { mint = nintx = ninty = nintz = 0; }
   void ResetCounter(int c) 
   { 
      switch (c)
      {
         case 0: nintx = 0; break;
         case 1: ninty = 0; break;
         case 2: nintz = 0; break;
         default: mint = 0; break;
      }
   }
   void SetCoord(int coord_) { coord = coord_; }
   void SetOrient(int orient_) { orient = orient_; }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   Vector * GetVecX(){return &VecX;}
   Vector * GetVecY(){return &VecY;}
   Vector * GetVecZ(){return &VecZ;}
   virtual ~ElementTPFunctionCoefficient() { }
};


class TPElementTransformation
{
private:
   int dim;
   FiniteElementSpace * fes = nullptr;
   Array2D<Vector *> TransA1D;
   Array2D<Vector *> TransB1D;
   Array2D<Vector *> TransC1D;
   void Setup2D();
   void Setup3D();
public:
   TPElementTransformation(FiniteElementSpace &fes_);
   Vector * GetTPTransformation(int iel, int coord, int which_coeff)
   {
      switch(which_coeff)
      {
         case 0: return TransA1D[iel][coord]; break;
         case 1: return TransB1D[iel][coord]; break;
         case 2: 
         {
            MFEM_VERIFY(dim == 3, "Wrong coeff for this dimension");
            return TransC1D[iel][coord]; 
            break;
         }
         default: MFEM_ABORT("Wrong coeff selection"); return 0; break; 
      }
   }
   ~TPElementTransformation() { }
};

void GetVertexToEdgeCount(const Mesh * mesh, DenseMatrix & edge_counts);

void GetDiffusionEdgeMatrix(int iedge, FiniteElementSpace * fes, 
                               Vector & Jac1D, Vector & Coeff1D,
                               const IntegrationRule *ir,  
                               DenseMatrix &elmat, int orient);
void GetMassEdgeMatrix(int iedge, FiniteElementSpace * fes, 
                       Vector & Jac1D, Vector & Coeff1D,
                       const IntegrationRule *ir,  
                       DenseMatrix &elmat, int orient);

void TensorProductEssentialDofsMaps(const Array<int> & ess_tdof_list, 
                                    const ParFiniteElementSpace * fes, 
                                    Array<Array<int> *> & tmap, // local edge map
                                    Array<Array<int>* > & non_ess_dofs); // element map