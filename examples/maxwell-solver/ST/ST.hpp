#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "complex_additive_schwarz.hpp"
using namespace std;
using namespace mfem;



// Function coefficient that takes the boundingbox of the mesh as an input
class CutOffFnCoefficient : public Coefficient
{
private:
   double (*Function)(const Vector &, const Vector &, const Vector &, const Array2D<double> &);
   Vector pmin, pmax;
   Array2D<double> h; // specify the with of the cutoff function (h in each direction)
   

public:
   CutOffFnCoefficient(double (*F)(const Vector &, const Vector &, const Vector &, const Array2D<double> &), 
                             const Vector & pmin_, const Vector & pmax_, Array2D<double> & h_)
      : Function(F), pmin(pmin_), pmax(pmax_), h(h_)
   {}
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      return ((*Function)(transip, pmin, pmax, h));
   }
};

double CutOffFncn(const Vector &x, const Vector &pmax, const Vector &pmin, const Array2D<double> & h_);


class DofMap // Constructs dof maps for a given partition
{
   FiniteElementSpace *fespace=nullptr;
   SesquilinearForm * bf=nullptr;
   MeshPartition * partition=nullptr;
public:
   int nrpatch, nx, ny, nz;
   vector<Array<int>> Dof2GlobalDof;
   vector<Array<int>> Dof2PmlDof;
   Array<Mesh *> PmlMeshes;
   Array<FiniteElementSpace *> fespaces;
   Array<FiniteElementSpace *> PmlFespaces;
   // constructor
   // Non PML contructor dof map
   DofMap(SesquilinearForm * bf_, MeshPartition * partition_);
   // PML
   DofMap(SesquilinearForm * bf_ , MeshPartition * partition_, int nrlayers);
   ~DofMap();
};


class STP : public Solver//
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
   const Operator * A=nullptr;
   Vector B;
   DofMap * ovlp_prob = nullptr;
   DofMap * novlp_prob = nullptr;
   Array<SesquilinearForm *> HalfSpaceForms;
   Array<SparseMatrix *> PmlMat;
   Array<SparseMatrix *> HalfSpaceMat;
   Array<KLUSolver *> PmlMatInv;
   Array<KLUSolver *> HalfSpaceMatInv;
   Array2D<double> Pmllength;
   mutable Array<Vector * > res;

   SparseMatrix * GetPmlSystemMatrix(int ip);
   SparseMatrix * GetHalfSpaceSystemMatrix(int ip);
   void SolveHalfSpaceLinearSystem(int ip, Vector & x, Vector & load) const;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;
   void GetCutOffSolution(Vector & sol, int ip, int direction) const;

public:
   STP(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   void SetLoadVector(Vector load) { B = load;}
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~STP();
};


