#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
using namespace std;
using namespace mfem;

class AdditiveST2D : public Solver//
{
private:
   int nrpatch;
   int dim;
   MeshPartition * povlp=nullptr;
   MeshPartition * novlp=nullptr;
   int ovlpnrlayers;
   int nxyz[3];
   const Operator * A=nullptr;
   DofMap * ovlp_prob = nullptr;
   DofMap * nvlp_prob = nullptr;
   Array<SparseMatrix *> PmlMat;
   Array<KLUSolver *> PmlMatInv;
   Array3D<int> subdomains;
   mutable Array<Vector *> f_orig;
   mutable Array<Vector *> usol;
   mutable Array<Array<Vector * >> f_s;
   mutable Array<Array<Vector * >> f_diag;

   SesquilinearForm *bf=nullptr;
   Array2D<double> Pmllength;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;


   SparseMatrix * GetPmlSystemMatrix(int ip);
  
   void GetCutOffSolution(const Vector & sol, Vector & cfsol,
                          int ip, Array<int> directions, int nlayers, bool local=false) const;
   void GetChiRes(const Vector & res, Vector & cfres,
                  int ip, Array<int> directions, int nlayers) const;  
   
   void AdditiveTransferSources(int step, int ip, Vector & sol_ext) const;
   int GetPatchId(const Array<int> & ijk) const;
   void Getijk(int ip, int & i, int & j, int & k ) const;
   int SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
   void PlotSolution(Vector & sol, socketstream & sol_sock, int ip,bool localdomain) const;
   void SaveSolution(Vector & sol, int ip,bool localdomain) const;
   void PlotMesh(socketstream & mesh_sock, int ip) const;
public:
   AdditiveST2D(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_);
   virtual void SetOperator(const Operator &op) {A = &op;}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~AdditiveST2D();
};


