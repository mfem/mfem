#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
using namespace std;
using namespace mfem;

double testcoeff(const Vector & x);
int get_rank(int tdof, std::vector<int> & tdof_offsets);


void ComputeTdofOffsets(const MPI_Comm & comm, const ParFiniteElementSpace * pfes, 
                        std::vector<int> & tdof_offsets);

void GetSubdomainijk(int ip, const Array<int> nxyz, Array<int> & ijk);
void GetDirectionijk(int id, Array<int> & ijk);
int GetSubdomainId(const Array<int> nxyz, Array<int> & ijk);
int GetDirectionId(const Array<int> & ijk);



// class handling two types of dof maps
// 1. Subdomain truedofs ---> Global truedofs
// 2. Subdomain truedofs ---> Neighbor truedofs 
class DofMaps
{
private:
   // The FE space of the problem (H1/Hcurl)
   ParFiniteElementSpace *pfes = nullptr;

   // The given partition of the parmesh
   ParMeshPartition *part = nullptr;
   // partition in x-y-z
   Array<int> nxyz;

   // MPI parameters
   MPI_Comm comm = MPI_COMM_WORLD;
   int num_procs, myid;

   // true dof offset and element offset of the processor
   vector<int> tdof_offsets;
   int mytoffset;
   int myelemoffset;
   
   int dim;
   // Total number of subdomains
   int nrsubdomains;
   
   // Array specifying the subdomain rank
   Array<int> subdomain_rank;

   // Initializing mpi and helper parameters
   void Init();

   // 1. Setting up the subdomains FE spaces
   // 2. Setting up the subdomains-to-subdomains maps
   // 3. Setting up the subdomain-to-global maps
   void Setup();

   // -----------------------------------------------
   //           Subdomain to Subdomain maps
   // -----------------------------------------------
   std::vector<std::vector<Array<int>>> OvlpElems;
   void AddElementToOvlpLists(int l, int iel, 
                            const Array<bool> & neg, 
                            const Array<bool> & pos);
   std::vector<std::vector<Array<int>>> OvlpTDofs;
   std::vector<std::vector<Vector * >> OvlpSol;
   void SubdomainToSubdomainMapsSetup();
   void ComputeOvlpElems();
   void ComputeOvlpTdofs();
   void PrintOvlpTdofs();

   // -----------------------------------------------
   //           Subdomain to Global maps
   // -----------------------------------------------
   std::vector<Array<int>> SubdomainGTrueDofs; // Subdomain Tdofs to Global Tdofs
   std::vector<Array<int>> SubdomainLTrueDofs; // Subdomain Tdofs to Local (on rank) Tdofs

   Array<int> send_count, send_displ;  
   Array<int> recv_count, recv_displ;  
   int sbuff_size = 0;
   int rbuff_size = 0;
   void SubdomainToGlobalMapsSetup();

   // Testing
   void TestSubdomainToGlobalMaps();
   void TestSubdomainToSubdomainMaps();



public:
   // constructor

   // FiniteElementSpaces of the subdomains
   Array<FiniteElementSpace *> fes;

   DofMaps(ParFiniteElementSpace *fespace_, ParMeshPartition * part_);
   ~DofMaps();
   // Transfering from subdomains SubdomainIds to all their neighbors
   void TransferToNeighbors(const Array<int> & SubdomainIds, const Array<Vector *> & x);

   // Prolongation of subdomain solutions to the global solution
   void SubdomainsToGlobal(const Array<Vector*> & x, Vector & y);
   // Restriction of global residual to subdomain residuals
   void GlobalToSubdomains(const Vector & y, Array<Vector*> & x);

};











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

