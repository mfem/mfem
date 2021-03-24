// #pragma once
// #include "MeshPartition2D.hpp"

// struct hash_pair {
//    template <class T1, class T2>
//    size_t operator()(const pair<T1, T2>& p) const{
//       auto hash1 = hash<T1>{}(p.first);
//       auto hash2 = hash<T2>{}(p.second);
//       return hash1 ^ hash2;
//    }
// };

// struct UniqueIndexGenerator
// {
//    int counter = 0;
//    std::unordered_map<pair<int,int>,int, hash_pair> idx;
//    int Get(int i, int j)
//    {
//       pair<int,int> p1(i,j);
//       std::unordered_map<pair<int,int>,int, hash_pair>::iterator f = idx.find(p1);
//       if (f == idx.end())
//       {
//          idx[p1] = counter;
//          return counter++;
//       }
//       else
//       {
//          return (*f).second;
//       }
//    }
//    void Reset()
//    {
//       counter = 0;
//       idx.clear();
//    }
// };



// // Function coefficient that takes the boundingbox of the mesh as an input
// class CutOffFnCoefficient : public Coefficient
// {
// private:
//    double (*Function)(const Vector &, const Vector &, const Vector &, const Array2D<double> &);
//    Vector pmin, pmax;
//    Array2D<double> h; // specify the with of the cutoff function (h in each direction)
   

// public:
//    CutOffFnCoefficient(double (*F)(const Vector &, const Vector &, const Vector &, const Array2D<double> &), 
//                              const Vector & pmin_, const Vector & pmax_, Array2D<double> & h_)
//       : Function(F), pmin(pmin_), pmax(pmax_), h(h_)
//    {}
//    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
//    {
//       double x[3];
//       Vector transip(x, 3);
//       T.Transform(ip, transip);
//       return ((*Function)(transip, pmin, pmax, h));
//    }
// };

// double CutOffFncn(const Vector &x, const Vector & pmin, 
// const Vector & pmax, const Array2D<double> & h_);
// double ChiFncn(const Vector &x, const Vector & pmin, 
// const Vector & pmax, const Array2D<double> & h_);

// class DofMap // Constructs dof maps for a given partition
// {
//    FiniteElementSpace *fespace=nullptr;
//    SesquilinearForm * bf=nullptr;
//    MeshPartition * partition=nullptr;
// public:
//    int nrpatch, nx, ny, nz;
//    vector<Array<int>> Dof2GlobalDof;
//    vector<Array<int>> Dof2PmlDof;
//    Array<Mesh *> PmlMeshes;
//    Array<FiniteElementSpace *> fespaces;
//    Array<FiniteElementSpace *> PmlFespaces;
//    // constructor
//    // Non PML contructor dof map
//    DofMap(SesquilinearForm * bf_, MeshPartition * partition_);
//    // PML
//    DofMap(SesquilinearForm * bf_ , MeshPartition * partition_, int nrlayers);
//    ~DofMap();
// };


// class LocalDofMap // Constructs dof mapbetween two partitions
// {
//    const FiniteElementCollection *fec=nullptr;
//    MeshPartition * part1=nullptr;
//    MeshPartition * part2=nullptr;
// public:
//    int nrpatch, nx, ny, nz;
//    vector<Array<int>> map1;
//    vector<Array<int>> map2;
//    // constructor
//    LocalDofMap(const FiniteElementCollection * fec_, MeshPartition * part1_, 
//                MeshPartition * part2_);
//    ~LocalDofMap();
// };

