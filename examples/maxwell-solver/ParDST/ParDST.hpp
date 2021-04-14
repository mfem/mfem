#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
#include "DofMapsDST.hpp"
using namespace std;
using namespace mfem;


class ParDST : public Solver//
{
public:
   enum BCType
   {
       NEUMANN,
       DIRICHLET
   };
   ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * Q_, int nrlayers_, 
       int nx_=2, int ny_=2, int nz_=2, 
       BCType bc_type_ = BCType::DIRICHLET, Coefficient * LossCoeff_ = nullptr);
   ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, MatrixCoefficient * MQ_, int nrlayers_, int nx_=2, int ny_=2, int nz_=2, 
       BCType bc_type_ = BCType::DIRICHLET, Coefficient * LossCoeff_ = nullptr);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ParDST();
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
   Array<int> RankSubdomains;
   const FiniteElementCollection * fec = nullptr;
   Array2D<double> Pmllength;
   int dim = 2;
   double omega = 0.5;
   Coefficient * Q=nullptr;
   MatrixCoefficient * MQ=nullptr;
   int nrlayers;
   BCType bc_type = BCType::DIRICHLET;
   Coefficient * LossCoeff=nullptr;
   
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
//    Array<ComplexMUMPSSolver *> PmlMatInv;
   mutable Array<Vector *> f_orig;
   mutable Array<Array<Vector * >> f_transf;
   mutable Array<Vector * > subdomain_sol;
   mutable std::vector<std::vector<Vector * >> OvlpSol;
   void SetupSubdomainProblems();
   std::vector<std::vector<Array<int>>> NovlpElems;
   std::vector<std::vector<Array<int>>> NovlpDofs;
   void MarkSubdomainOverlapDofs(const bool comp = false);
   void SetHelmholtzPmlSystemMatrix(int ip);   
   void SetMaxwellPmlSystemMatrix(int ip);
   void GetChiRes(Vector & res, int ip, Array2D<int> direct) const;
   void PlotLocal(Vector & sol, socketstream & sol_sock, int ip) const;
   void PlotGlobal(Vector & sol, socketstream & sol_sock) const;
   double GetSweepNumSteps(const int sweep) const;
   void GetStepSubdomains(const int sweep, const int step, Array2D<int> & subdomains) const;
   void TransferSources(int sweep, const Array<int> & subdomain_ids) const;
   int GetSweepToTransfer(const int s, Array<int> directions) const;
   void CorrectOrientation(int ip, Vector & x) const;
   void Init();

};


