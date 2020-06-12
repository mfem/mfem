#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
using namespace std;
using namespace mfem;


struct Sweep
{
private:
   int dim;
   std::vector<Array<int>> sweeps;
public:   
   int nsweeps;
   Sweep(int dim_);
   void GetSweep(const int i, Array<int> & sweep)
   {
      MFEM_VERIFY(i<nsweeps, "Sweep number out of bounds");
      sweep.SetSize(dim);
      sweep = sweeps[i];
   }
};



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
   std::vector<Array2D<int>> ovlpelems;
   // Pml local problems
   Array<SparseMatrix *> PmlMat;
   Array<UMFPackSolver *> PmlMatInv;
   Sweep * swp=nullptr;
   mutable Array<Vector *> f_orig;
   mutable Array<Array<Vector * >> f_transf;



   void GetOverlapElements();
   void Getijk(int ip, int & i, int & j, int & k ) const;
   int GetPatchId(const Array<int> & ijk) const;
   SparseMatrix * GetPmlSystemMatrix(int ip);
  
   void GetCutOffSolution(const Vector & sol, Vector & cfsol,
                          int ip, Array2D<int> direct, int nlayers, bool local=false) const;                          
   void GetChiRes(const Vector & res, Vector & cfres,
                  int ip, Array2D<int> direct, int nlayers) const;  
   void GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const;
   void TransferSources(int sweep, int ip, Vector & sol_ext) const;
   void SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
   int GetSweepToTransfer(const int s, Array<int> directions) const;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip,bool localdomain) const;
  
public:
   DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~DST();
};


