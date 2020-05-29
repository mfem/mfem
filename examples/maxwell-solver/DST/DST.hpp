#pragma once
#include "Utilities.hpp"
#include "PML.hpp"
using namespace std;
using namespace mfem;

class DST : public Solver//
{
private:
   int nrpatch;
   int dim;
   SesquilinearForm *bf=nullptr;
   MeshPartition * povlp=nullptr;
   MeshPartition * novlp=nullptr;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   int ovlpnrlayers;
   int nxyz[3];
   const Operator * A=nullptr;
   DofMap * ovlp_prob = nullptr;
   DofMap * nvlp_prob = nullptr;
   Array<SparseMatrix *> PmlMat;
   Array<KLUSolver *> PmlMatInv;
   Array2D<double> Pmllength;
   Array3D<int> subdomains;
   mutable Array<Vector *> f_orig;
   int nsweeps;
   Array2D<int> sweeps;
   mutable Array<Array<Vector * >> f_transf;

   SparseMatrix * GetPmlSystemMatrix(int ip);
  
   void GetCutOffSolution(const Vector & sol, Vector & cfsol,
                          int ip, Array<int> directions, int nlayers, bool local=false) const;
   void GetChiRes(const Vector & res, Vector & cfres,
                  int ip, Array<int> directions, int nlayers) const;  
   
   void TransferSources(int sweep, int ip, Vector & sol_ext) const;
   int GetPatchId(const Array<int> & ijk) const;
   void Getijk(int ip, int & i, int & j, int & k ) const;
   int SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip,bool localdomain) const;
   void PlotMesh(socketstream & mesh_sock, int ip) const;
  
   // void SourceTransfer1(const Vector & Psi0, Array<int> direction, int ip0, Vector & Psi1) const;
   // void SetSubMeshesAttributes();
   // void GetRestrCoeffAttr(const Array<int> & directions, Array<int> & attr) const;
   // double GetSolOvlpNorm(const Vector & sol, const Array<int> & directions, int ip) const;
public:
   DST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~DST();
};


