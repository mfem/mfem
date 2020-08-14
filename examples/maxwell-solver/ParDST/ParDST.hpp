#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
#include "DofMapsDST.hpp"
using namespace std;
using namespace mfem;



// class SubdomainToSubdomainMap
// {
// public:
//    // constructor
//    SubdomainToSubdomainMap(ParFiniteElementSpace *fespace_, ParMeshPartition * part_);
//    ~SubdomainToSubdomainMap();

//    void Setup();
//    void TransferToNeighbors(int ip);

// private:
//    void TransferToNeighbor(int ip, int jp);

//    ParFiniteElementSpace *fespace = nullptr;
//    ParMeshPartition * part;
//    MPI_Comm comm = MPI_COMM_WORLD;
//    int num_procs, myid;
//    int mytoffset = 0;
//    int myelemoffset=0;
//    int nrsubdomains=0;
//    vector<int> tdof_offsets;
//    Array<int> subdomain_rank;
// };




// class SubdomainToGlobalMap
// {
// public:
//    // constructor
//    SubdomainToGlobalMap(ParFiniteElementSpace *fespace_, ParMeshPartition * part_);
//    ~SubdomainToGlobalMap();

//    void Setup();
//    void Mult(const std::vector<Vector > & sol, Vector & z);
//    void MultTranspose(const Vector & r, std::vector<Vector> & res);

// private:
//    ParFiniteElementSpace *fespace = nullptr;
//    ParMeshPartition * part;
//    MPI_Comm comm = MPI_COMM_WORLD;
//    int num_procs, myid;
//    int mytoffset = 0;
//    int myelemoffset=0;
//    int nrsubdomains=0;
//    vector<int> tdof_offsets;
//    Array<int> subdomain_rank;

//    Array<int> send_count; 
//    Array<int> send_displ;  
//    Array<int> recv_count;  
//    Array<int> recv_displ;  
//    int sbuff_size = 0;
//    int rbuff_size = 0;

//    // list of all the true dofs in a subdomain
//    vector<Array<int>> SubdomainGlobalTrueDofs; 
//    vector<Array<int>> SubdomainLocalTrueDofs;
//    Array<FiniteElementSpace *> subdomain_fespaces;
//    std::vector<Array<int>> subdomain_dof_map;

//    void ComputeTdofOffsets()
//    {
//       int num_procs;
//       MPI_Comm_size(comm, &num_procs);
//       tdof_offsets.resize(num_procs);
//       mytoffset = fespace->GetMyTDofOffset();
//       MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
//    }
//    int get_rank(int tdof)
//    {
//       int size = tdof_offsets.size();
//       if (size == 1) { return 0; }
//       std::vector<int>::iterator up;
//       up=std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(),tdof); // 
//       return std::distance(tdof_offsets.begin(),up)-1;
//    }
// };


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


