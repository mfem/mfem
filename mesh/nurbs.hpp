// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_NURBS
#define MFEM_NURBS

#include "../config/config.hpp"
#include "../general/table.hpp"
#include "../linalg/vector.hpp"
#include "element.hpp"
#include "mesh.hpp"
#ifdef MFEM_USE_MPI
#include "../general/communication.hpp"
#endif
#include <iostream>
#include <set>

namespace mfem
{

class GridFunction;


class KnotVector
{
protected:
   static const int MaxOrder;

   Vector knot;
   int Order, NumOfControlPoints, NumOfElements;

public:
   /// Create KnotVector
   KnotVector() { }
   KnotVector(std::istream &input);
   KnotVector(int Order_, int NCP);
   KnotVector(const KnotVector &kv) { (*this) = kv; }

   KnotVector &operator=(const KnotVector &kv);

   int GetNE()    const { return NumOfElements; }
   int GetNKS()   const { return NumOfControlPoints - Order; }
   int GetNCP()   const { return NumOfControlPoints; }
   int GetOrder() const { return Order; }
   int Size()     const { return knot.Size(); }

   /// Count the number of elements
   void GetElements();

   bool isElement(int i) const { return (knot(Order+i) != knot(Order+i+1)); }

   double getKnotLocation(double xi, int ni) const
   { return (xi*knot(ni+1) + (1. - xi)*knot(ni)); }

   int findKnotSpan(double u) const;

   void CalcShape  (Vector &shape, int i, double xi) const;
   void CalcDShape (Vector &grad,  int i, double xi) const;
   void CalcDnShape(Vector &gradn, int n, int i, double xi) const;
   void CalcD2Shape(Vector &grad2, int i, double xi) const
   { CalcDnShape(grad2, 2, i, xi); }

   /** Gives the locations of the maxima of the knotvector in reference space. The
      function gives the knotspan @a ks, the coordinate in the knotspan @a xi
      and the coordinate of the maximum in parameter space @a u */
   void FindMaxima(Array<int> &ks, Vector &xi, Vector &u);
   /** Global curve interpolation through the points @a x. @a x is an array with the
      length of the spatial dimension containing vectors with spatial coordinates. The
      controlpoints of the interpolated curve are given in @a x in the same form.*/
   void FindInterpolant(Array<Vector*> &x);

   void Difference(const KnotVector &kv, Vector &diff) const;
   void UniformRefinement(Vector &newknots) const;
   /** Return a new KnotVector with elevated degree by repeating the endpoints
       of the knot vector. */
   /// @note The returned object should be deleted by the caller.
   KnotVector *DegreeElevate(int t) const;

   void Flip();

   void Print(std::ostream &out) const;

   /** Prints the non-zero shape functions and their first and second
       derivatives associated with the KnotVector per element. Use GetElements()
       to count the elements before using this function. @a samples is the
       number of samples of the shape functions per element.*/
   void PrintFunctions(std::ostream &out, int samples=11) const;

   /// Destroys KnotVector
   ~KnotVector() { }

   double &operator[](int i) { return knot(i); }
   const double &operator[](int i) const { return knot(i); }
};


class NURBSPatch
{
protected:
   int     ni, nj, nk, Dim;
   double *data; // the layout of data is: (Dim x ni x nj x nk)

   Array<KnotVector *> kv;

   // Special B-NET access functions
   //  - SetLoopDirection(int dir) flattens the multi-dimensional B-NET in the
   //    requested direction. It effectively creates a 1D net.
   //  - The slice(int, int) operator is the access function in that flattened structure.
   //    The first int gives the slice and the second int the element in that slice.
   //  - Both routines are used in 'InsertKnot', 'DegreeElevate' and 'UniformRefinement'.
   //  - In older implementations slice(int int) was implemented as operator()(int, int)
   int nd; // Number of knots in flattened structure
   int ls; // Number of variables per knot in flattened structure
   int sd; // Stride for data access
   int SetLoopDirection(int dir);
   inline       double &slice(int i, int j);
   inline const double &slice(int i, int j) const;

   NURBSPatch(NURBSPatch *parent, int dir, int Order, int NCP);
   void swap(NURBSPatch *np);
   void init(int dim_);

public:
   NURBSPatch(const NURBSPatch &orig);
   NURBSPatch(std::istream &input);
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1, int dim_);
   NURBSPatch(const KnotVector *kv0, const KnotVector *kv1,
              const KnotVector *kv2, int dim_);
   NURBSPatch(Array<const KnotVector *> &kv, int dim_);

   NURBSPatch& operator=(const NURBSPatch&) = delete;

   ~NURBSPatch();

   void Print(std::ostream &out) const;

   void DegreeElevate(int dir, int t);
   void KnotInsert   (int dir, const KnotVector &knot);
   void KnotInsert   (int dir, const Vector     &knot);

   void KnotInsert(Array<Vector *> &knot);
   void KnotInsert(Array<KnotVector *> &knot);

   void DegreeElevate(int t);
   void UniformRefinement();

   // Return the number of components stored in the NURBSPatch
   int GetNC() const { return Dim; }
   int GetNKV() const { return kv.Size(); }
   /// @note The returned object should NOT be deleted by the caller.
   KnotVector *GetKV(int i) { return kv[i]; }

   // Standard B-NET access functions
   inline       double &operator()(int i, int j);
   inline const double &operator()(int i, int j) const;

   inline       double &operator()(int i, int j, int l);
   inline const double &operator()(int i, int j, int l) const;

   inline       double &operator()(int i, int j, int k, int l);
   inline const double &operator()(int i, int j, int k, int l) const;

   static void Get2DRotationMatrix(double angle,
                                   DenseMatrix &T);
   static void Get3DRotationMatrix(double n[], double angle, double r,
                                   DenseMatrix &T);
   void FlipDirection(int dir);
   void SwapDirections(int dir1, int dir2);

   /// Rotate the NURBSPatch.
   /** A rotation of a 2D NURBS-patch requires an angle only. Rotating
       a 3D NURBS-patch requires a normal as well.*/
   void Rotate(double angle, double normal[]= NULL);
   void Rotate2D(double angle);
   void Rotate3D(double normal[], double angle);

   int MakeUniformDegree(int degree = -1);
   /// @note The returned object should be deleted by the caller.
   friend NURBSPatch *Interpolate(NURBSPatch &p1, NURBSPatch &p2);
   /// @note The returned object should be deleted by the caller.
   friend NURBSPatch *Revolve3D(NURBSPatch &patch, double n[], double ang,
                                int times);
};


#ifdef MFEM_USE_MPI
class ParNURBSExtension;
#endif

class NURBSPatchMap;


class NURBSExtension
{
#ifdef MFEM_USE_MPI
   friend class ParNURBSExtension;
#endif
   friend class NURBSPatchMap;

protected:
   int mOrder; // see GetOrder() for description
   Array<int> mOrders;
   int NumOfKnotVectors;
   // global entity counts
   int NumOfVertices, NumOfElements, NumOfBdrElements, NumOfDofs;
   // local entity counts
   int NumOfActiveVertices, NumOfActiveElems, NumOfActiveBdrElems;
   int NumOfActiveDofs;

   Array<int>  activeVert; // activeVert[glob_vert] = loc_vert or -1
   Array<bool> activeElem;
   Array<bool> activeBdrElem;
   Array<int>  activeDof; // activeDof[glob_dof] = loc_dof + 1 or 0

   Mesh *patchTopo;
   int own_topo;
   Array<int> edge_to_knot;
   /** Set of knotvectors containing unique KnotVectors only */
   Array<KnotVector *> knotVectors;
   /** Comprehensive set of knotvectors. This set contains a KnotVector for
       every edge.*/
   Array<KnotVector *> knotVectorsCompr;
   Vector weights;

   // periodic BC info:
   // - dof 2 dof map
   // - master and slave boundary indices
   Array<int> d_to_d;
   Array<int> master;
   Array<int> slave;

   // global offsets, meshOffsets == meshVertexOffsets
   Array<int> v_meshOffsets;
   Array<int> e_meshOffsets;
   Array<int> f_meshOffsets;
   Array<int> p_meshOffsets;

   // global offsets, spaceOffsets == dofOffsets
   Array<int> v_spaceOffsets;
   Array<int> e_spaceOffsets;
   Array<int> f_spaceOffsets;
   Array<int> p_spaceOffsets;

   Table *el_dof, *bel_dof;

   Array<int> el_to_patch;
   Array<int> bel_to_patch;
   Array2D<int> el_to_IJK;  // IJK are "knot-span" indices!
   Array2D<int> bel_to_IJK; // they are NOT element indices!

   std::vector<Array<int>> patch_to_el;
   std::vector<Array<int>> patch_to_bel;

   Array<NURBSPatch *> patches;

   inline int         KnotInd(int edge) const;
   /// @note The returned object should NOT be deleted by the caller.
   inline KnotVector *KnotVec(int edge);
   inline const KnotVector *KnotVec(int edge) const;
   inline const KnotVector *KnotVec(int edge, int oedge, int *okv) const;

   void CheckPatches();
   void CheckBdrPatches();

   /** Checks the direction of the knotvectors in the patch based on
       the patch orientation for patch @a p returns the direction of
       the Knotvectors in @a kvdir.*/
   void CheckKVDirection(int p, Array <int> &kvdir);
   /**  Creates the comprehensive set of KnotVectors. They are the same for 1D. */
   void CreateComprehensiveKV();
   /**  Updates the unique set of KnotVectors */
   void UpdateUniqueKV();

   /** Checks if the comprehensive array of KnotVectors agrees with
       the reduced set of KnotVectors. Returns false if it finds
       a difference. */
   bool ConsistentKVSets();

   void GetPatchKnotVectors   (int p, Array<KnotVector *> &kv);
   void GetBdrPatchKnotVectors(int p, Array<KnotVector *> &kv);

   void SetOrderFromOrders();
   void SetOrdersFromKnotVectors();

   // periodic BC helper functions
   void InitDofMap();
   void ConnectBoundaries();
   void ConnectBoundaries1D(int bnd0, int bnd1);
   void ConnectBoundaries2D(int bnd0, int bnd1);
   void ConnectBoundaries3D(int bnd0, int bnd1);
   int DofMap(int dof) const
   {
      return (d_to_d.Size() > 0 )? d_to_d[dof] : dof;
   };

   // also count the global NumOfVertices and the global NumOfDofs
   void GenerateOffsets();
   // count the global NumOfElements
   void CountElements();
   // count the global NumOfBdrElements
   void CountBdrElements();

   // generate the mesh elements
   void Get1DElementTopo(Array<Element *> &elements) const;
   void Get2DElementTopo(Array<Element *> &elements) const;
   void Get3DElementTopo(Array<Element *> &elements) const;

   // generate the boundary mesh elements
   void Get1DBdrElementTopo(Array<Element *> &boundary) const;
   void Get2DBdrElementTopo(Array<Element *> &boundary) const;
   void Get3DBdrElementTopo(Array<Element *> &boundary) const;

   // FE space generation functions

   // based on activeElem, count NumOfActiveDofs, generate el_dof,
   // el_to_patch, el_to_IJK, activeDof map (global-to-local)
   void GenerateElementDofTable();

   // generate elem_to_global-dof table for the active elements
   // define el_to_patch, el_to_IJK, activeDof (as bool)
   void Generate1DElementDofTable();
   void Generate2DElementDofTable();
   void Generate3DElementDofTable();

   // call after GenerateElementDofTable
   void GenerateBdrElementDofTable();

   // generate the bdr-elem_to_global-dof table for the active bdr. elements
   // define bel_to_patch, bel_to_IJK
   void Generate1DBdrElementDofTable();
   void Generate2DBdrElementDofTable();
   void Generate3DBdrElementDofTable();

   // FE --> Patch translation functions
   void GetPatchNets  (const Vector &Nodes, int vdim);
   void Get1DPatchNets(const Vector &Nodes, int vdim);
   void Get2DPatchNets(const Vector &Nodes, int vdim);
   void Get3DPatchNets(const Vector &Nodes, int vdim);

   // Patch --> FE translation functions
   // Side effects: delete the patches, update the weights from the patches
   void SetSolutionVector  (Vector &Nodes, int vdim);
   void Set1DSolutionVector(Vector &Nodes, int vdim);
   void Set2DSolutionVector(Vector &Nodes, int vdim);
   void Set3DSolutionVector(Vector &Nodes, int vdim);

   // determine activeVert, NumOfActiveVertices from the activeElem array
   void GenerateActiveVertices();

   // determine activeBdrElem, NumOfActiveBdrElems
   void GenerateActiveBdrElems();

   void MergeWeights(Mesh *mesh_array[], int num_pieces);

   void SetPatchToElements();
   void SetPatchToBdrElements();

   // to be used by ParNURBSExtension constructor(s)
   NURBSExtension() { }

public:
   /// Copy constructor: deep copy
   NURBSExtension(const NURBSExtension &orig);
   /// Read-in a NURBSExtension
   NURBSExtension(std::istream &input);
   /** @brief Create a NURBSExtension with elevated order by repeating the
       endpoints of the knot vectors and using uniform weights of 1. */
   /** If a knot vector in @a parent already has order greater than or equal to
       @a newOrder, it will be used unmodified. */
   NURBSExtension(NURBSExtension *parent, int newOrder);
   /** @brief Create a NURBSExtension with elevated knot vector orders (by
       repeating the endpoints of the knot vectors and using uniform weights of
       1) as given by the array @a newOrders. */
   /** If a knot vector in @a parent already has order greater than or equal to
       the corresponding entry in @a newOrder, it will be used unmodified. */
   NURBSExtension(NURBSExtension *parent, const Array<int> &newOrders);
   /// Construct a NURBSExtension by merging a partitioned NURBS mesh
   NURBSExtension(Mesh *mesh_array[], int num_pieces);

   /// Copy assignment not supported
   NURBSExtension& operator=(const NURBSExtension&) = delete;

   // Generate connections between boundaries, such as periodic BCs
   void ConnectBoundaries(Array<int> &master, Array<int> &slave);
   const Array<int> &GetMaster() const { return  master; };
   Array<int> &GetMaster()  { return  master; };
   const Array<int> &GetSlave() const { return  slave; };
   Array<int> &GetSlave()  { return  slave; };
   void MergeGridFunctions(GridFunction *gf_array[], int num_pieces,
                           GridFunction &merged);

   /// Destroy a NURBSExtension
   virtual ~NURBSExtension();

   // Print functions
   void Print(std::ostream &out) const;
   void PrintCharacteristics(std::ostream &out) const;
   void PrintFunctions(const char *filename, int samples=11) const;

   // Meta data functions
   int Dimension() const { return patchTopo->Dimension(); }
   int GetNP()     const { return patchTopo->GetNE(); }
   int GetNBP()    const { return patchTopo->GetNBE(); }

   /// Read-only access to the orders of all knot vectors.
   const Array<int> &GetOrders() const { return mOrders; }
   /** @brief If all orders are identical, return that number. Otherwise, return
       NURBSFECollection::VariableOrder. */
   int GetOrder() const { return mOrder; }

   int GetNKV()  const { return NumOfKnotVectors; }

   int GetGNV()  const { return NumOfVertices; }
   int GetNV()   const { return NumOfActiveVertices; }
   int GetGNE()  const { return NumOfElements; }
   int GetNE()   const { return NumOfActiveElems; }
   int GetGNBE() const { return NumOfBdrElements; }
   int GetNBE()  const { return NumOfActiveBdrElems; }

   int GetNTotalDof() const { return NumOfDofs; }
   int GetNDof()      const { return NumOfActiveDofs; }

   /// Returns knotvectors in each dimension for patch @a p.
   void GetPatchKnotVectors(int p, Array<const KnotVector *> &kv) const;

   void GetBdrPatchKnotVectors(int p, Array<const KnotVector *> &kv) const;

   // Knotvector read-only access function
   const KnotVector *GetKnotVector(int i) const { return knotVectors[i]; }

   // Mesh generation functions
   void GetElementTopo   (Array<Element *> &elements) const;
   void GetBdrElementTopo(Array<Element *> &boundary) const;

   bool HavePatches() const { return (patches.Size() != 0); }

   /// @note The returned object should NOT be deleted by the caller.
   Table *GetElementDofTable() { return el_dof; }
   /// @note The returned object should NOT be deleted by the caller.
   Table *GetBdrElementDofTable() { return bel_dof; }

   void GetVertexLocalToGlobal(Array<int> &lvert_vert);
   void GetElementLocalToGlobal(Array<int> &lelem_elem);

   // Set the attribute for patch @a i, which is set to all elements in the
   // patch.
   void SetPatchAttribute(int i, int attr) { patchTopo->SetAttribute(i, attr); }

   // Get the attribute for patch @a i, which is set to all elements in the
   // patch.
   int GetPatchAttribute(int i) const { return patchTopo->GetAttribute(i); }

   // Set the attribute for patch boundary element @a i, which is set to all
   // boundary elements in the patch.
   void SetPatchBdrAttribute(int i, int attr)
   { patchTopo->SetBdrAttribute(i, attr); }
   // Get the attribute for patch boundary element @a i, which is set to all
   // boundary elements in the patch.
   int GetPatchBdrAttribute(int i) const
   { return patchTopo->GetBdrAttribute(i); }

   // Load functions
   void LoadFE(int i, const FiniteElement *FE) const;
   void LoadBE(int i, const FiniteElement *BE) const;

   const Vector &GetWeights() const { return  weights; }
   Vector       &GetWeights()       { return  weights; }

   // Translation functions: from FE coordinates to IJK patch
   // format and vice versa
   void ConvertToPatches(const Vector &Nodes);
   void SetKnotsFromPatches();
   void SetCoordsFromPatches(Vector &Nodes);

   // Read a GridFunction written patch-by-patch, e.g. with PrintSolution().
   void LoadSolution(std::istream &input, GridFunction &sol) const;
   // Write a GridFunction patch-by-patch.
   void PrintSolution(const GridFunction &sol, std::ostream &out) const;

   // Refinement methods
   // new_degree = max(old_degree, min(old_degree + rel_degree, degree))
   void DegreeElevate(int rel_degree, int degree = 16);
   void UniformRefinement();
   void KnotInsert(Array<KnotVector *> &kv);
   void KnotInsert(Array<Vector *> &kv);

   /// Returns the index of the patch containing element @a elem.
   int GetElementPatch(int elem) const { return el_to_patch[elem]; }

   /** Returns the Cartesian indices (i,j) in 2D or (i,j,k) in 3D of element
       @a elem, in the knot-span tensor product ordering for its patch. */
   void GetElementIJK(int elem, Array<int> & ijk);

   // Returns the degrees of freedom on the patch, in Cartesian order.
   void GetPatchDofs(const int patch, Array<int> &dofs);

   const Array<int>& GetPatchElements(int patch);
   const Array<int>& GetPatchBdrElements(int patch);
};


#ifdef MFEM_USE_MPI
class ParNURBSExtension : public NURBSExtension
{
private:
   int *partitioning;

   Table *GetGlobalElementDofTable();
   Table *Get1DGlobalElementDofTable();
   Table *Get2DGlobalElementDofTable();
   Table *Get3DGlobalElementDofTable();

   void SetActive(const int *partitioning, const Array<bool> &active_bel);
   void BuildGroups(const int *partitioning, const Table &elem_dof);

public:
   GroupTopology gtopo;

   Array<int> ldof_group;

   ParNURBSExtension(const ParNURBSExtension &orig);

   ParNURBSExtension(MPI_Comm comm, NURBSExtension *parent, int *partitioning,
                     const Array<bool> &active_bel);

   // Create a parallel version of 'parent' with partitioning as in
   // 'par_parent'; the 'parent' object is destroyed.
   // The 'parent' can be either a local NURBSExtension or a global one.
   ParNURBSExtension(NURBSExtension *parent,
                     const ParNURBSExtension *par_parent);

   virtual ~ParNURBSExtension() { delete [] partitioning; }
};
#endif


class NURBSPatchMap
{
private:
   const NURBSExtension *Ext;

   int I, J, K, pOffset, opatch;
   Array<int> verts, edges, faces, oedge, oface;

   inline static int F(const int n, const int N)
   { return (n < 0) ? 0 : ((n >= N) ? 2 : 1); }

   inline static int Or1D(const int n, const int N, const int Or)
   { return (Or > 0) ? n : (N - 1 - n); }

   inline static int Or2D(const int n1, const int n2,
                          const int N1, const int N2, const int Or);

   // also set verts, edges, faces, orientations etc
   void GetPatchKnotVectors   (int p, const KnotVector *kv[]);
   void GetBdrPatchKnotVectors(int p, const KnotVector *kv[], int *okv);

public:
   NURBSPatchMap(const NURBSExtension *ext) { Ext = ext; }

   int nx() { return I + 1; }
   int ny() { return J + 1; }
   int nz() { return K + 1; }

   void SetPatchVertexMap(int p, const KnotVector *kv[]);
   void SetPatchDofMap   (int p, const KnotVector *kv[]);

   void SetBdrPatchVertexMap(int p, const KnotVector *kv[], int *okv);
   void SetBdrPatchDofMap   (int p, const KnotVector *kv[], int *okv);

   inline int operator()(const int i) const;
   inline int operator[](const int i) const { return (*this)(i); }

   inline int operator()(const int i, const int j) const;

   inline int operator()(const int i, const int j, const int k) const;
};


// Inline function implementations

inline double &NURBSPatch::slice(int i, int j)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= nd || j < 0 || j > ls)
   {
      mfem_error("NURBSPatch::slice()");
   }
#endif
   return data[j%sd + sd*(i + (j/sd)*nd)];
}

inline const double &NURBSPatch::slice(int i, int j) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= nd || j < 0 || j > ls)
   {
      mfem_error("NURBSPatch::slice()");
   }
#endif
   return data[j%sd + sd*(i + (j/sd)*nd)];
}


inline double &NURBSPatch::operator()(int i, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || nj > 0 || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 1D");
   }
#endif

   return data[i*Dim+l];
}

inline const double &NURBSPatch::operator()(int i, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni ||  nj > 0 || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 1D");
   }
#endif

   return data[i*Dim+l];
}

inline double &NURBSPatch::operator()(int i, int j, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 2D");
   }
#endif

   return data[(i+j*ni)*Dim+l];
}

inline const double &NURBSPatch::operator()(int i, int j, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || nk > 0 ||
       l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 2D");
   }
#endif

   return data[(i+j*ni)*Dim+l];
}

inline double &NURBSPatch::operator()(int i, int j, int k, int l)
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || k < 0 ||
       k >= nk || l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() 3D");
   }
#endif

   return data[(i+(j+k*nj)*ni)*Dim+l];
}

inline const double &NURBSPatch::operator()(int i, int j, int k, int l) const
{
#ifdef MFEM_DEBUG
   if (data == 0 || i < 0 || i >= ni || j < 0 || j >= nj || k < 0 ||
       k >= nk ||  l < 0 || l >= Dim)
   {
      mfem_error("NURBSPatch::operator() const 3D");
   }
#endif

   return data[(i+(j+k*nj)*ni)*Dim+l];
}


inline int NURBSExtension::KnotInd(int edge) const
{
   int kv = edge_to_knot[edge];
   return (kv >= 0) ? kv : (-1-kv);
}

inline KnotVector *NURBSExtension::KnotVec(int edge)
{
   return knotVectors[KnotInd(edge)];
}

inline const KnotVector *NURBSExtension::KnotVec(int edge) const
{
   return knotVectors[KnotInd(edge)];
}

inline const KnotVector *NURBSExtension::KnotVec(int edge, int oedge, int *okv)
const
{
   int kv = edge_to_knot[edge];
   if (kv >= 0)
   {
      *okv = oedge;
      return knotVectors[kv];
   }
   else
   {
      *okv = -oedge;
      return knotVectors[-1-kv];
   }
}


// static method
inline int NURBSPatchMap::Or2D(const int n1, const int n2,
                               const int N1, const int N2, const int Or)
{
   // Needs testing
   switch (Or)
   {
      case 0: return n1 + n2*N1;
      case 1: return n2 + n1*N2;
      case 2: return n2 + (N1 - 1 - n1)*N2;
      case 3: return (N1 - 1 - n1) + n2*N1;
      case 4: return (N1 - 1 - n1) + (N2 - 1 - n2)*N1;
      case 5: return (N2 - 1 - n2) + (N1 - 1 - n1)*N2;
      case 6: return (N2 - 1 - n2) + n1*N2;
      case 7: return n1 + (N2 - 1 - n2)*N1;
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::Or2D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i) const
{
   const int i1 = i - 1;
   switch (F(i1, I))
   {
      case 0: return verts[0];
      case 1: return pOffset + Or1D(i1, I, opatch);
      case 2: return verts[1];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 1D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i, const int j) const
{
   const int i1 = i - 1, j1 = j - 1;
   switch (3*F(j1, J) + F(i1, I))
   {
      case 0: return verts[0];
      case 1: return edges[0] + Or1D(i1, I, oedge[0]);
      case 2: return verts[1];
      case 3: return edges[3] + Or1D(j1, J, -oedge[3]);
      case 4: return pOffset + Or2D(i1, j1, I, J, opatch);
      case 5: return edges[1] + Or1D(j1, J, oedge[1]);
      case 6: return verts[3];
      case 7: return edges[2] + Or1D(i1, I, -oedge[2]);
      case 8: return verts[2];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 2D");
#endif
   return -1;
}

inline int NURBSPatchMap::operator()(const int i, const int j, const int k)
const
{
   // Needs testing
   const int i1 = i - 1, j1 = j - 1, k1 = k - 1;
   switch (3*(3*F(k1, K) + F(j1, J)) + F(i1, I))
   {
      case  0: return verts[0];
      case  1: return edges[0] + Or1D(i1, I, oedge[0]);
      case  2: return verts[1];
      case  3: return edges[3] + Or1D(j1, J, oedge[3]);
      case  4: return faces[0] + Or2D(i1, J - 1 - j1, I, J, oface[0]);
      case  5: return edges[1] + Or1D(j1, J, oedge[1]);
      case  6: return verts[3];
      case  7: return edges[2] + Or1D(i1, I, oedge[2]);
      case  8: return verts[2];
      case  9: return edges[8] + Or1D(k1, K, oedge[8]);
      case 10: return faces[1] + Or2D(i1, k1, I, K, oface[1]);
      case 11: return edges[9] + Or1D(k1, K, oedge[9]);
      case 12: return faces[4] + Or2D(J - 1 - j1, k1, J, K, oface[4]);
      case 13: return pOffset + I*(J*k1 + j1) + i1;
      case 14: return faces[2] + Or2D(j1, k1, J, K, oface[2]);
      case 15: return edges[11] + Or1D(k1, K, oedge[11]);
      case 16: return faces[3] + Or2D(I - 1 - i1, k1, I, K, oface[3]);
      case 17: return edges[10] + Or1D(k1, K, oedge[10]);
      case 18: return verts[4];
      case 19: return edges[4] + Or1D(i1, I, oedge[4]);
      case 20: return verts[5];
      case 21: return edges[7] + Or1D(j1, J, oedge[7]);
      case 22: return faces[5] + Or2D(i1, j1, I, J, oface[5]);
      case 23: return edges[5] + Or1D(j1, J, oedge[5]);
      case 24: return verts[7];
      case 25: return edges[6] + Or1D(i1, I, oedge[6]);
      case 26: return verts[6];
   }
#ifdef MFEM_DEBUG
   mfem_error("NURBSPatchMap::operator() const 3D");
#endif
   return -1;
}

}

#endif
