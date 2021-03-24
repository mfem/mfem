// // #pragma once
// // #include "Utilities.hpp"
// // #include "PML.hpp"
// // using namespace std;
// // using namespace mfem;

// // class DiagST : public Solver//
// // {
// // private:
// //    int nrpatch;
// //    int dim;
// //    SesquilinearForm *bf=nullptr;
// //    MeshPartition * povlp=nullptr;
// //    double omega = 0.5;
// //    Coefficient * ws;
// //    int nrlayers;
// //    int ovlpnrlayers;
// //    int nxyz[3];
// //    const Operator * A=nullptr;
// //    Vector B;
// //    DofMap * ovlp_prob = nullptr;
// //    Array<SparseMatrix *> PmlMat;
// //    Array<KLUSolver *> PmlMatInv;
// //    Array2D<double> Pmllength;
// //    Array3D<int> subdomains;
// //    mutable Array<Vector *> f_orig;
// //    int ntransf_directions;
// //    int nsweeps;
// //    Array2D<int> sweeps;
// //    Array<int> dirx;
// //    Array<int> diry;
// //    Array<int> dirz;
// //    mutable Array<Array<Vector * >> f_transf;
// //    Array<Array<Vector * >> usol;

// //    SparseMatrix * GetPmlSystemMatrix(int ip);
// //    void PlotSolution(Vector & sol, socketstream & sol_sock, int ip) const;
// //    // void GetCutOffSolution(const Vector & sol, Vector & cfsol,
// //                         //   int ip, Array<int> directions, bool local=false) const;
// //    void GetCutOffSolution(const Vector & sol, Vector & cfsol,
// //                         int ip, Array<int> directions, int ovlpnlayers, bool local=false) const;                       
// //    void GetChiRes(const Vector & res, Vector & cfres,
// //                   int ip, Array<int> directions, int nlayers) const;                       
// //    void TransferSources(int sweep, int ip, Vector & sol_ext) const;
// //    int  GetDirectionId(const Array<int> & ijk) const;
// //    void GetDirectionijk(int id, Array<int> & ijk) const;
// //    void ConstructDirectionsMap();
// //    int GetPatchId(const Array<int> & ijk) const;
// //    void Getijk(int ip, int & i, int & j, int & k ) const;
// //    int SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
// // public:
// //    DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
// //        double omega_, Coefficient * ws_, int nrlayers_);
// //    void SetLoadVector(Vector load) { B = load;}
// //    virtual void SetOperator(const Operator &op) {A = &op;}
// //    virtual void Mult(const Vector &r, Vector &z) const;
// //    virtual ~DiagST();
// // };

// #pragma once
// #include "Utilities.hpp"
// #include "PML.hpp"
// using namespace std;
// using namespace mfem;

// class DiagST : public Solver//
// {
// private:
//    int nrpatch;
//    int dim;
//    SesquilinearForm *bf=nullptr;
//    MeshPartition * povlp=nullptr;
//    MeshPartition * novlp=nullptr;
//    double omega = 0.5;
//    Coefficient * ws;
//    int nrlayers;
//    int ovlpnrlayers;
//    int nxyz[3];
//    const Operator * A=nullptr;
//    Vector B;
//    DofMap * ovlp_prob = nullptr;
//    DofMap * nvlp_prob = nullptr;
//    Array<SparseMatrix *> PmlMat;
//    Array<KLUSolver *> PmlMatInv;
//    Array2D<double> Pmllength;
//    Array3D<int> subdomains;
//    mutable Array<Vector *> f_orig;
//    int ntransf_directions;
//    int nsweeps;
//    Array2D<int> sweeps;
//    Array<int> dirx;
//    Array<int> diry;
//    Array<int> dirz;
//    mutable Array<Array<Vector * >> f_transf;
//    Array<Vector * > usol;

//    SparseMatrix * GetPmlSystemMatrix(int ip);
//    void PlotSolution(Vector & sol, socketstream & sol_sock, int ip, bool localdomain = false, bool pmldomain = false) const;
//    // void GetCutOffSolution(const Vector & sol, Vector & cfsol,
//                         //   int ip, Array<int> directions, bool local=false) const;
//    void GetCutOffSolution(const Vector & sol, Vector & cfsol,
//                         int ip, Array<int> directions, int ovlpnlayers, bool local=false) const;                       
//    void GetChiRes(const Vector & res, Vector & cfres,
//                   int ip, Array<int> directions, int nlayers) const;                       
//    void TransferSources(int sweep, int ip, Vector & sol_ext) const;
//    int  GetDirectionId(const Array<int> & ijk) const;
//    void GetDirectionijk(int id, Array<int> & ijk) const;
//    void ConstructDirectionsMap();
//    int GetPatchId(const Array<int> & ijk) const;
//    void Getijk(int ip, int & i, int & j, int & k ) const;
//    int SourceTransfer(const Vector & Psi0, Array<int> direction, int ip, Vector & Psi1) const;
// public:
//    DiagST(SesquilinearForm * bf_, Array2D<double> & Pmllength_, 
//        double omega_, Coefficient * ws_, int nrlayers_);
//    void SetLoadVector(Vector load) { B = load;}
//    virtual void SetOperator(const Operator &op) {A = &op;}
//    virtual void Mult(const Vector &r, Vector &z) const;
//    virtual ~DiagST();
// };



