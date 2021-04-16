#pragma once
#include "MeshPartition.hpp"

struct UniqueIndexGen
{
   int counter = 0;
   std::unordered_map<int,int> idx;

   void Set(int i)
   {
      std::unordered_map<int,int>::iterator f = idx.find(i);
      if (f == idx.end())
      {
         idx[i] = counter;
         counter++;
      }
   }

   int Get(int i)
   {
      std::unordered_map<int,int>::iterator f = idx.find(i);
      if (f == idx.end())
      {
         return -1;
      }
      else
      {
         return (*f).second;
      }
   }
   void Reset()
   {
      counter = 0;
      idx.clear();
   }
};


struct Sweep
{
private:
   int dim;
   std::vector<Array<int>> sweeps;
public:   
   int nsweeps;
   Sweep(int dim_);
   ~Sweep();
   void GetSweep(const int i, Array<int> & sweep)
   {
      MFEM_VERIFY(i<nsweeps, "Sweep number out of bounds");
      sweep.SetSize(dim);
      sweep = sweeps[i];
   }
};


// Function coefficient that takes the bounding box of the mesh as an input
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

double CutOffFncn(const Vector &x, const Vector & pmin, 
const Vector & pmax, const Array2D<double> & h_);
double ChiFncn(const Vector &x, const Vector & pmin, 
const Vector & pmax, const Array2D<double> & h_);

class DofMap // Constructs dof maps for a given partition
{
public:
   int nrpatch, nx, ny, nz;
   vector<Array<int>> Dof2GlobalDof;
   vector<Array<int>> Dof2PmlDof;
   Array<Mesh *> PmlMeshes;
   Array<FiniteElementSpace *> fespaces;
   Array<FiniteElementSpace *> PmlFespaces;
   // constructor
   // Non PML constructor dof map
   DofMap(FiniteElementSpace * fes, MeshPartition * partition);
   // PML
   DofMap(FiniteElementSpace * fes , MeshPartition * partition, int nrlayers);
   ~DofMap(){};
};

class LocalDofMap // Constructs dof mapbetween two partitions
{
   const FiniteElementCollection *fec=nullptr;
   MeshPartition * part1=nullptr;
   MeshPartition * part2=nullptr;
public:
   int nrpatch, nx, ny, nz;
   vector<Array<int>> map1;
   vector<Array<int>> map2;
   // constructor
   LocalDofMap(const FiniteElementCollection * fec_, MeshPartition * part1_, 
               MeshPartition * part2_);
   ~LocalDofMap();
};

struct NeighborDofMaps
{
private:
   int dim;
   MeshPartition * part = nullptr;
   FiniteElementSpace * fes = nullptr;
   Mesh * mesh = nullptr;
   std::vector<std::vector<Array<int>>> OvlpElems;
   std::vector<std::vector<Array<int>>> OvlpDofMaps;

   DofMap * dmap = nullptr;
   int nrsubdomains = 0;
   int ovlp_layers = 0;
   Array<int> nxyz;
   void SetElementToOverlap(int ip, int iel, 
                           const Array<bool> & neg, 
                           const Array<bool> & pos);

   void MarkOvlpElements();
   void ComputeNeighborDofMaps();

   void Getijk(int ip, int & i, int & j, int & k) const
   {
      k = ip/(nxyz[0]*nxyz[1]);
      j = (ip-k*nxyz[0]*nxyz[1])/nxyz[0];
      i = (ip-k*nxyz[0]*nxyz[1])%nxyz[0];
   }

   int GetPatchId(const Array<int> & ijk) const
   {
      int d=ijk.Size();
      int z = (d==2)? 0 : ijk[2];
      return part->subdomains(ijk[0],ijk[1],z);
   }
      int GetDirectionId(int i, int j, int k=-1) 
   {
      int n = 3;
      return (k+1)*n*n + (j+1)*n + i+1; 
   }
   void GetDirections(const int id, int & i, int & j, int & k) 
   {
      int n = 3;
      k = id/(n*n) - 1;
      j = (id-(k+1)*n*n)/n - 1;
      i = (id-(k+1)*n*n)%n - 1;
   }

public:   
   NeighborDofMaps(MeshPartition * part_, 
                   FiniteElementSpace * fes_, 
                   DofMap * dmap_,
                   int ovlp_layers_);

   void GetNeighborDofMap(const int ip, const Array<int> & directions,
                          Array<int> & dofmap);
};
