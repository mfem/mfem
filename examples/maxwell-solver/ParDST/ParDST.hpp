#pragma once
#include "../common/Utilities.hpp"
#include "../common/PML.hpp"
#include "../DST/DST.hpp"
using namespace std;
using namespace mfem;


class ParSubdomainDofInfo
{
public:
   ParFiniteElementSpace *fespace = nullptr;
   ParMeshPartition * part;
   MPI_Comm comm = MPI_COMM_WORLD;
   int mytoffset = 0;
   int nrsubdomains=0;
   
   Array<int> subdomain_rank;
   // list of all the true dofs in a subdomain
   vector<Array<int>> SubdomainGlobalTrueDofs; 
   vector<Array<int>> SubdomainLocalTrueDofs;
   Array<FiniteElementSpace *> subdomain_fespaces;
   // constructor
   ParSubdomainDofInfo(ParFiniteElementSpace *fespace_, ParMeshPartition * part_);
   ~ParSubdomainDofInfo();

private:
   vector<int> tdof_offsets;

   void ComputeTdofOffsets()
   {
      int num_procs;
      MPI_Comm_size(comm, &num_procs);
      tdof_offsets.resize(num_procs);
      mytoffset = fespace->GetMyTDofOffset();
      MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
   }

   int get_rank(int tdof)
   {
      int size = tdof_offsets.size();
      if (size == 1) { return 0; }
      std::vector<int>::iterator up;
      up=std::upper_bound(tdof_offsets.begin(), tdof_offsets.end(),tdof); // 
      return std::distance(tdof_offsets.begin(),up)-1;
   }
};


class ParDST : public Solver//
{
private:
   // Constructor inputs
   ParSesquilinearForm *bf=nullptr;
   ParFiniteElementSpace * pfes = nullptr;
   ParMesh * pmesh = nullptr;
   const FiniteElementCollection * fec = nullptr;
   Array2D<double> Pmllength;
   int dim = 2;
   double omega = 0.5;
   Coefficient * ws;
   int nrlayers;
   int nrsubdomains = 0;
   int nx,ny,nz;

   Sweep * sweeps=nullptr;



   void Getijk(int ip, int & i, int & j, int & k ) const;
   int GetPatchId(const Array<int> & ijk) const;
   void IdentifyCommonDofs(); // TODO (Use global Elem number for matching dofs)
   void TransferToNeighbors(int ip); // TODO
  
public:
   ParDST(ParSesquilinearForm * bf_, Array2D<double> & Pmllength_, 
       double omega_, Coefficient * ws_, int nrlayers_, int nx_=2, int ny_=2, int nz_=2);
   virtual void SetOperator(const Operator &op) {}
   virtual void Mult(const Vector &r, Vector &z) const;
   virtual ~ParDST();
};


