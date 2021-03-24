#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
using namespace std;
using namespace mfem;



class DST : public Solver//
{
private:
   // Constructor inputs
   SesquilinearForm *bf=nullptr;
   Array2D<double> Pmllength;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   // 
   int nrpatch;
   int dim;
   int nx, ny, nz;
   int ovlpnrlayers;
   MeshPartition * part=nullptr;
   DofMap * dmap = nullptr;
   // Auxiliary global vector for transfers
   std::vector<std::vector<Array<int>>> NovlpElems;
   std::vector<std::vector<Array<int>>> NovlpDofs;
   NeighborDofMaps * NeighborMap = nullptr;
   Array< SesquilinearForm * > sqf;
   Array< OperatorPtr * > Optr;
   Array<ComplexSparseMatrix *> PmlMat;
   Array<ComplexUMFPackSolver *> PmlMatInv;
   Sweep * swp=nullptr;
   mutable Array<Vector *> f_orig;
   mutable Array<Array<Vector * >> f_transf;
   mutable Vector zaux;



   void MarkOverlapElements();
   void MarkOverlapDofs();
   void ComputeOverlapDofMaps();
   void Getijk(int ip, int & i, int & j, int & k ) const;
   int GetPatchId(const Array<int> & ijk) const;
  
   void SetHelmholtzPmlSystemMatrix(int ip);
   void SetMaxwellPmlSystemMatrix(int ip);

   void GetChiRes(Vector & res, int ip, Array2D<int> direct) const;                    
   void GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const;
   void TransferSources(int sweep, int ip, Vector & sol_ext) const;
   int SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
   int GetSweepToTransfer(const int s, Array<int> directions) const;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip,bool localdomain) const;
  
public:
   DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_, int nx_=2, int ny_=2, int nz_=2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~DST();
};


