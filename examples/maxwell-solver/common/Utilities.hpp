#pragma once
#include "MeshPartition.hpp"



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
