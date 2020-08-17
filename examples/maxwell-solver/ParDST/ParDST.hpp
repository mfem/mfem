#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
#include "DofMapsDST.hpp"
using namespace std;
using namespace mfem;


class ParDST : public Solver//
{
private:
   MPI_Comm comm = MPI_COMM_WORLD;
   int num_procs, myid;
   // Constructor inputs
   int prob_kind;
   ParSesquilinearForm *bf=nullptr;
   ParFiniteElementSpace * pfes = nullptr;
   ParMesh * pmesh = nullptr;
   ParMeshPartition * part = nullptr;
   Array<int> SubdomainRank;
   const FiniteElementCollection * fec = nullptr;
   Array2D<double> Pmllength;
   int dim = 2;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   int ovlpnrlayers;
   int nrsubdomains = 0;
   int nx,ny,nz;
   Array<int> nxyz;

   Sweep * sweeps = nullptr;
   DofMaps * dmaps = nullptr;
   Array< SesquilinearForm * > sqf;
   Array< OperatorPtr * > Optr;
   Array<ComplexSparseMatrix *> PmlMat;
   Array<ComplexUMFPackSolver *> PmlMatInv;
   mutable Array<Vector *> f_orig_re;
   mutable Array<Vector *> f_orig_im;
   mutable Array<Array<Vector * >> f_transf_re;
   mutable Array<Array<Vector * >> f_transf_im;
   mutable Array<Vector * > subdomain_sol;

   void SetupSubdomainProblems();
   std::vector<std::vector<Array<int>>> NovlpElems;
   std::vector<std::vector<Array<int>>> NovlpDofs;
   void MarkSubdomainOverlapDofs();
   void SetHelmholtzPmlSystemMatrix(int ip);   
   void SetMaxwellPmlSystemMatrix(int ip);
   void GetChiRes(Vector & res, int ip, Array2D<int> direct) const;
   void PlotLocal(Vector & sol, socketstream & sol_sock, int ip) const;
   void GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const;

public:
   ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_, int nx_=2, int ny_=2, int nz_=2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ParDST();
};


