// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_MESH
#define MFEM_MESH

#include "../config/config.hpp"
#include "../general/stable3d.hpp"
#include "triangle.hpp"
#include "tetrahedron.hpp"
#include "vertex.hpp"
#include "ncmesh.hpp"
#include "../fem/eltrans.hpp"
#include "../fem/coefficient.hpp"
#include "../general/gzstream.hpp"
#include <iostream>
#include <fstream>

namespace mfem
{

// Data type mesh

class KnotVector;
class NURBSExtension;
class FiniteElementSpace;
class GridFunction;
struct Refinement;

#ifdef MFEM_USE_MPI
class ParMesh;
class ParNCMesh;
#endif


class Mesh
{
#ifdef MFEM_USE_MPI
   friend class ParMesh;
   friend class ParNCMesh;
#endif
   friend class NURBSExtension;

protected:
   int Dim;
   int spaceDim;

   int NumOfVertices, NumOfElements, NumOfBdrElements;
   int NumOfEdges, NumOfFaces;
   int NumOfPlanars;

   int BaseGeom, BaseBdrGeom; // element base geometries, -1 if not all the same

   int meshgen; // see MeshGenerator()

   // Counter for Mesh transformations: refinement, derefinement, rebalancing.
   // Used for checking during Update operations on objects depending on the
   // Mesh, such as FiniteElementSpace, GridFunction, etc.
   long sequence;

   Array<Element *> elements;
   // Vertices are only at the corners of elements, where you would expect them
   // in the lowest-order mesh.
   Array<Vertex> vertices;
   Array<Element *> boundary;
   Array<Element *> faces;
   Array<Element *> planars; //only for 4d meshes

   Array<bool> swappedElements; //only for 4d meshes
   Array<bool> swappedFaces; //only for 4d meshes

   struct FaceInfo
   {
      // Inf = 64 * LocalFaceIndex + FaceOrientation
      int Elem1No, Elem2No, Elem1Inf, Elem2Inf;
      int NCFace; /* -1 if this is a regular conforming/boundary face;
                     index into 'nc_faces_info' if >= 0. */
   };
   // NOTE: in NC meshes, master faces have Elem2No == -1. Slave faces on the
   // other hand have Elem2No and Elem2Inf set to the master face's element and
   // its local face number.

   struct NCFaceInfo
   {
      bool Slave; // true if this is a slave face, false if master face
      int MasterFace; // if Slave, this is the index of the master face
      const DenseMatrix* PointMatrix; // if Slave, position within master face
      // (NOTE: PointMatrix points to a matrix owned by NCMesh.)

      NCFaceInfo(bool slave, int master, const DenseMatrix* pm)
         : Slave(slave), MasterFace(master), PointMatrix(pm) {}
   };

   Array<FaceInfo> faces_info;
   Array<NCFaceInfo> nc_faces_info;

   Table *el_to_edge;
   Table *el_to_face;
   Table *el_to_planar;    // for 4D
   Table *el_to_el;
   Array<int> be_to_edge;  // for 2D
   Table *bel_to_edge;     // for 3D
   Table *bel_to_planar;   // for 4D
   Array<int> be_to_face;
   mutable Table *face_edge;
   mutable Table *edge_vertex;

   IsoparametricTransformation Transformation, Transformation2;
   IsoparametricTransformation FaceTransformation, EdgeTransformation;
   FaceElementTransformations FaceElemTr;

   // refinement embeddings for forward compatibility with NCMesh
   CoarseFineTransformations CoarseFineTr;

   // Nodes are only active for higher order meshes, and share locations with
   // the vertices, plus all the higher- order control points within the
   // element and along the edges and on the faces.
   GridFunction *Nodes;
   int own_nodes;

   static const int vtk_quadratic_tet[10];
   static const int vtk_quadratic_hex[27];

#ifdef MFEM_USE_MEMALLOC
   friend class Tetrahedron;
   MemAlloc <Tetrahedron, 1024> TetMemory;
#endif

public:
   typedef Geometry::Constants<Geometry::SEGMENT>     seg_t;
   typedef Geometry::Constants<Geometry::TRIANGLE>    tri_t;
   typedef Geometry::Constants<Geometry::SQUARE>      quad_t;
   typedef Geometry::Constants<Geometry::TETRAHEDRON> tet_t;
   typedef Geometry::Constants<Geometry::CUBE>        hex_t;
   typedef Geometry::Constants<Geometry::PENTATOPE>   pent_t;
   typedef Geometry::Constants<Geometry::TESSERACT>   tess_t;

   enum Operation { NONE, REFINE, DEREFINE, REBALANCE };

   /// A list of all unique element attributes used by the Mesh.
   Array<int> attributes;
   /// A list of all unique boundary attributes used by the Mesh.
   Array<int> bdr_attributes;

   NURBSExtension *NURBSext; ///< Optional NURBS mesh extension.
   NCMesh *ncmesh;           ///< Optional non-conforming mesh extension.

protected:
   Operation last_operation;

   void Init();
   void InitTables();
   void SetEmpty();  // Init all data members with empty values
   void DestroyTables();
   void DeleteTables() { DestroyTables(); InitTables(); }
   void DestroyPointers(); // Delete data specifically allocated by class Mesh.
   void Destroy();         // Delete all owned data.

   Element *ReadElementWithoutAttr(std::istream &);
   static void PrintElementWithoutAttr(const Element *, std::ostream &);

   Element *ReadElement(std::istream &);
   static void PrintElement(const Element *, std::ostream &);

   // Readers for different mesh formats, used in the Load() method.
   // The implementations of these methods are in mesh_readers.cpp.
   void ReadMFEMMesh(std::istream &input, bool mfem_v11, int &curved);
   void ReadLineMesh(std::istream &input);
   void ReadNetgen2DMesh(std::istream &input, int &curved);
   void ReadNetgen3DMesh(std::istream &input);
   void ReadTrueGridMesh(std::istream &input);
   void ReadVTKMesh(std::istream &input, int &curved, int &read_gf);
   void ReadNURBSMesh(std::istream &input, int &curved, int &read_gf);
   void ReadInlineMesh(std::istream &input, int generate_edges = 0);
   void ReadGmshMesh(std::istream &input);
   /* Note NetCDF (optional library) is used for reading cubit files */
#ifdef MFEM_USE_NETCDF
   void ReadCubit(const char *filename, int &curved, int &read_gf);
#endif

   /// Determine the mesh generator bitmask #meshgen, see MeshGenerator().
   void SetMeshGen();

   /// Return the length of the segment from node i to node j.
   double GetLength(int i, int j) const;

   /** Compute the Jacobian of the transformation from the perfect
       reference element at the center of the element. */
   void GetElementJacobian(int i, DenseMatrix &J);

   void MarkForRefinement();
   void MarkTriMeshForRefinement();
   void GetEdgeOrdering(DSTable &v_to_v, Array<int> &order);
   virtual void MarkTetMeshForRefinement(DSTable &v_to_v);

   void PrepareNodeReorder(DSTable **old_v_to_v, Table **old_elem_vert);
   void DoNodeReorder(DSTable *old_v_to_v, Table *old_elem_vert);

   STable3D *GetFacesTable();
   STable3D *GetElementToFaceTable(int ret_ftbl = 0);
   STable4D *GetElementToFaceTable4D(int ret_ftbl = 0);
   STable3D *GetElementToPlanarTable(int ret_ftbl = 0);

   /** Red refinement. Element with index i is refined. The default
       red refinement for now is Uniform. */
   void RedRefinement(int i, const DSTable &v_to_v,
                      int *edge1, int *edge2, int *middle)
   { UniformRefinement(i, v_to_v, edge1, edge2, middle); }

   /** Green refinement. Element with index i is refined. The default
       refinement for now is Bisection. */
   void GreenRefinement(int i, const DSTable &v_to_v,
                        int *edge1, int *edge2, int *middle)
   { Bisection(i, v_to_v, edge1, edge2, middle); }

   /** Bisection. Element with index i is bisected. */
   void Bisection(int i, const DSTable &, int *, int *, int *);

   /** Bisection. Boundary element with index i is bisected. */
   void Bisection(int i, const DSTable &, int *);

   void RedRefinementPentatope(int i, const DSTable & v_to_v, int *middle);
   void RedRefinementBoundaryTet(int i, const DSTable & v_to_v, int *middle);

   /** Uniform Refinement. Element with index i is refined uniformly. */
   void UniformRefinement(int i, const DSTable &, int *, int *, int *);

   /** Averages the vertices with given indexes and saves the result in
       vertices[result]. */
   void AverageVertices (int * indexes, int n, int result);

   void InitRefinementTransforms();
   int FindCoarseElement(int i);

   /// Update the nodes of a curved mesh after refinement
   void UpdateNodes();

   /// Refine quadrilateral mesh.
   virtual void QuadUniformRefinement();

   /// Refine hexahedral mesh.
   virtual void HexUniformRefinement();

   /// Refine NURBS mesh.
   virtual void NURBSUniformRefinement();

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void LocalRefinement(const Array<int> &marked_el, int type = 3);

   /// This function is not public anymore. Use GeneralRefinement instead.
   virtual void NonconformingRefinement(const Array<Refinement> &refinements,
                                        int nc_limit = 0);

   /// NC version of GeneralDerefinement.
   virtual bool NonconformingDerefinement(Array<double> &elem_error,
                                          double threshold, int nc_limit = 0,
                                          int op = 1);

   /// Derefine elements once a list of derefinements is known.
   void DerefineMesh(const Array<int> &derefinements);

   /// Read NURBS patch/macro-element mesh
   void LoadPatchTopo(std::istream &input, Array<int> &edge_to_knot);

   void UpdateNURBS();

   void PrintTopo(std::ostream &out, const Array<int> &e_to_k) const;

   /// Used in GetFaceElementTransformations (...)
   void GetLocalPtToSegTransformation(IsoparametricTransformation &, int);
   void GetLocalSegToTriTransformation (IsoparametricTransformation &loc,
                                        int i);
   void GetLocalSegToQuadTransformation (IsoparametricTransformation &loc,
                                         int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalTriToTetTransformation (IsoparametricTransformation &loc,
                                        int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalQuadToHexTransformation (IsoparametricTransformation &loc,
                                         int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalFaceTransformation(int face_type, int elem_type,
                                   IsoparametricTransformation &Transf,
                                   int inf);
   /** Used in GetFaceElementTransformations to account for the fact that a
       slave face occupies only a portion of its master face. */
   void ApplyLocalSlaveTransformation(IsoparametricTransformation &transf,
                                      const FaceInfo &fi);
   bool IsSlaveFace(const FaceInfo &fi) const;

   /// Returns the orientation of "test" relative to "base"
   static int GetTriOrientation (const int * base, const int * test);
   /// Returns the orientation of "test" relative to "base"
   static int GetQuadOrientation (const int * base, const int * test);

   static int GetTetOrientation (const int * base, const int * test);

   static int GetHexOrientation(const int * base, const int * test);

   static void GetElementArrayEdgeTable(const Array<Element*> &elem_array,
                                        const DSTable &v_to_v,
                                        Table &el_to_edge);

   /** Return vertex to vertex table. The connections stored in the table
       are from smaller to bigger vertex index, i.e. if i<j and (i, j) is
       in the table, then (j, i) is not stored. */
   void GetVertexToVertexTable(DSTable &) const;

   /** Return element to edge table and the indices for the boundary edges.
       The entries in the table are ordered according to the order of the
       nodes in the elements. For example, if T is the element to edge table
       T(i, 0) gives the index of edge in element i that connects vertex 0
       to vertex 1, etc. Returns the number of the edges. */
   int GetElementToEdgeTable(Table &, Array<int> &);

   /// Used in GenerateFaces()
   void AddPointFaceElement(int lf, int gf, int el);

   void AddSegmentFaceElement (int lf, int gf, int el, int v0, int v1);

   void AddTriangleFaceElement (int lf, int gf, int el,
                                int v0, int v1, int v2);

   void AddQuadFaceElement (int lf, int gf, int el,
                            int v0, int v1, int v2, int v3);

   void AddTetrahedralFaceElement(int lf, int gf, int el,
                                  int v0, int v1, int v2, int v3);

   void AddHexahedralFaceElement(int lf, int gf, int el,
                                 int v0, int v1, int v2, int v3,
                                 int v4, int v5, int v6, int v7);


   /** For a serial Mesh, return true if the face is interior. For a parallel
       ParMesh return true if the face is interior or shared. In parallel, this
       method only works if the face neighbor data is exchanged. */
   bool FaceIsTrueInterior(int FaceNo) const
   {
      return FaceIsInterior(FaceNo) || (faces_info[FaceNo].Elem2Inf >= 0);
   }

   //swap first two entries of *a
   inline void Swap(int *a) const;
   // shift cyclically 3 integers left-to-right
   inline static void ShiftL2R(int &, int &, int &);
   // shift cyclically 3 integers so that the smallest is first
   inline static void Rotate3(int &, int &, int &);

   void FreeElement(Element *E);

   void GenerateFaces();
   void GenerateNCFaceInfo();
   void GeneratePlanars();

   /// Begin construction of a mesh
   void InitMesh(int _Dim, int _spaceDim, int NVert, int NElem, int NBdrElem);

   void InitBaseGeom();

   // Used in the methods FinalizeXXXMesh() and FinalizeTopology()
   void FinalizeCheck();

   void Loader(std::istream &input, int generate_edges = 0,
               std::string parse_tag = "");

   // If NURBS mesh, write NURBS format. If NCMesh, write mfem v1.1 format.
   // If section_delimiter is empty, write mfem v1.0 format. Otherwise, write
   // mfem v1.2 format with the given section_delimiter at the end.
   void Printer(std::ostream &out = std::cout,
                std::string section_delimiter = "") const;

   void Make4D(int nx, int ny, int nz, int nt, Element::Type type,
               int generate_edges,
               double sx, double sy, double sz, double st, bool generate_boundary,
               bool which_boundary[8],
               double shX=0.0, double shY=0.0, double shZ=0.0, double shT=0.0);

   /** Creates mesh for the parallelepiped [0,sx]x[0,sy]x[0,sz], divided into
       nx*ny*nz hexahedrals if type=HEXAHEDRON or into 6*nx*ny*nz tetrahedrons
       if type=TETRAHEDRON. If generate_edges = 0 (default) edges are not
       generated, if 1 edges are generated. */
   void Make3D(int nx, int ny, int nz, Element::Type type, int generate_edges,
               double sx, double sy, double sz);

   /** Creates mesh for the rectangle [0,sx]x[0,sy], divided into nx*ny
       quadrilaterals if type = QUADRILATERAL or into 2*nx*ny triangles if
       type = TRIANGLE. If generate_edges = 0 (default) edges are not generated,
       if 1 edges are generated. */
   void Make2D(int nx, int ny, Element::Type type, int generate_edges,
               double sx, double sy);

   /// Creates a 1D mesh for the interval [0,sx] divided into n equal intervals.
   void Make1D(int n, double sx = 1.0);

   /// Initialize vertices/elements/boundary/tables from a nonconforming mesh.
   void InitFromNCMesh(const NCMesh &ncmesh);

   /// Create from a nonconforming mesh.
   Mesh(const NCMesh &ncmesh);

   /// Swaps internal data with another mesh. By default, non-geometry members
   /// like 'ncmesh' and 'NURBSExt' are only swapped when 'non_geometry' is set.
   void Swap(Mesh& other, bool non_geometry = false);

   // used in GetElementData() and GetBdrElementData()
   void GetElementData(const Array<Element*> &elem_array, int geom,
                       Array<int> &elem_vtx, Array<int> &attr) const;

public:

   Mesh() { SetEmpty(); }

   /** Copy constructor. Performs a deep copy of (almost) all data, so that the
       source mesh can be modified (e.g. deleted, refined) without affecting the
       new mesh. If 'copy_nodes' is false, use a shallow (pointer) copy for the
       nodes, if present. */
   explicit Mesh(const Mesh &mesh, bool copy_nodes = true);

   /// Construct a Mesh from the given primary data.
   /** The array @a vertices is used as external data, i.e. the Mesh does not
       copy the data and will not delete the pointer.

       The data from the other arrays is copied into the internal Mesh data
       structures.

       This method calls the method FinalizeTopology(). The method Finalize()
       may be called after this constructor and after optionally setting the
       Mesh nodes. */
   Mesh(double *vertices, int num_vertices,
        int *element_indices, Geometry::Type element_type,
        int *element_attributes, int num_elements,
        int *boundary_indices, Geometry::Type boundary_type,
        int *boundary_attributes, int num_boundary_elements,
        int dimension, int space_dimension= -1);

   /** @anchor mfem_Mesh_init_ctor
       @brief _Init_ constructor: begin the construction of a Mesh object. */
   Mesh(int _Dim, int NVert, int NElem, int NBdrElem = 0, int _spaceDim = -1)
   {
      if (_spaceDim == -1)
      {
         _spaceDim = _Dim;
      }
      InitMesh(_Dim, _spaceDim, NVert, NElem, NBdrElem);
   }

   /** @name Methods for Mesh construction.

       These methods are intended to be used with the @ref mfem_Mesh_init_ctor
       "init constructor". */
   ///@{

   Element *NewElement(int geom);

   void AddVertex(const double *);
   void AddTri(const int *vi, int attr = 1);
   void AddTriangle(const int *vi, int attr = 1);
   void AddQuad(const int *vi, int attr = 1);
   void AddTet(const int *vi, int attr = 1);
   void AddHex(const int *vi, int attr = 1);
   void AddHexAsTets(const int *vi, int attr = 1);
   void AddTes(const int *vi, int attr = 1);
   // 'elem' should be allocated using the NewElement method
   void AddElement(Element *elem)     { elements[NumOfElements++] = elem; }
   void AddBdrElement(Element *elem)  { boundary[NumOfBdrElements++] = elem; }
   void AddBdrSegment(const int *vi, int attr = 1);
   void AddBdrTriangle(const int *vi, int attr = 1);
   void AddBdrQuad(const int *vi, int attr = 1);
   void AddBdrQuadAsTriangles(const int *vi, int attr = 1);
   void AddBdrHex(const int *vi, int attr = 1);

   void GenerateBoundaryElements();
   /// Finalize the construction of a triangular Mesh.
   void FinalizeTriMesh(int generate_edges = 0, int refine = 0,
                        bool fix_orientation = true);
   /// Finalize the construction of a quadrilateral Mesh.
   void FinalizeQuadMesh(int generate_edges = 0, int refine = 0,
                         bool fix_orientation = true);
   /// Finalize the construction of a tetrahedral Mesh.
   void FinalizeTetMesh(int generate_edges = 0, int refine = 0,
                        bool fix_orientation = true);
   /// Finalize the construction of a hexahedral Mesh.
   void FinalizeHexMesh(int generate_edges = 0, int refine = 0,
                        bool fix_orientation = true);
   void FinalizeTesMesh(int generate_edges = 0, int refine = 0,
                        bool fix_orientation = true);

   ///@}

   /** @brief Finalize the construction of the secondary topology (connectivity)
       data of a Mesh. */
   /** This method does not require any actual coordinate data (either vertex
       coordinates for linear meshes or node coordinates for meshes with nodes)
       to be available. However, the data generated by this method is generally
       required by the FiniteElementSpace class.

       After calling this method, setting the Mesh vertices or nodes, it may be
       appropriate to call the method Finalize(). */
   void FinalizeTopology();

   /// Finalize the construction of a general Mesh.
   /** This method will:
       - check and optionally fix the orientation of regular elements
       - check and fix the orientation of boundary elements
       - assume that #vertices are defined, if #Nodes == NULL
       - assume that #Nodes are defined, if #Nodes != NULL.
       @param[in] refine  If true, prepare the Mesh for conforming refinement of
                          triangular or tetrahedral meshes.
       @param[in] fix_orientation
                          If true, fix the orientation of inverted mesh elements
                          by permuting their vertices.

       Before calling this method, call FinalizeTopology() and ensure that the
       Mesh vertices or nodes are set. */
   void Finalize(bool refine = false, bool fix_orientation = false);

   void SetAttributes();

#ifdef MFEM_USE_GECKO
   /** This is our integration with the Gecko library.  This will call the
       Gecko library to find an element ordering that will increase memory
       coherency by putting elements that are in physical proximity closer in
       memory. */
   void GetGeckoElementReordering(Array<int> &ordering);
#endif

   /** Rebuilds the mesh with a different order of elements.  The ordering
       vector maps the old element number to the new element number.  This also
       reorders the vertices and nodes edges and faces along with the elements.  */
   void ReorderElements(const Array<int> &ordering, bool reorder_vertices = true);

   Mesh(int nx, int ny, int nz, int nt, Element::Type type, int generate_edges = 0,
        double sx = 1.0, double sy = 1.0, double sz = 1.0, double st = 1.0)
   {
      bool generate_boundary = true;
      bool which_boundary[8] = {true,true,true,true,true,true,true,true};
      Make4D(nx, ny, nz, nt, type, generate_edges, sx, sy, sz, st, generate_boundary,
             which_boundary);
   }

   /** Creates mesh for the parallelepiped [0,sx]x[0,sy]x[0,sz], divided into
       nx*ny*nz hexahedrals if type=HEXAHEDRON or into 6*nx*ny*nz tetrahedrons
       if type=TETRAHEDRON. If generate_edges = 0 (default) edges are not
       generated, if 1 edges are generated. */
   Mesh(int nx, int ny, int nz, Element::Type type, int generate_edges = 0,
        double sx = 1.0, double sy = 1.0, double sz = 1.0)
   {
      Make3D(nx, ny, nz, type, generate_edges, sx, sy, sz);
   }

   /** Creates mesh for the rectangle [0,sx]x[0,sy], divided into nx*ny
       quadrilaterals if type = QUADRILATERAL or into 2*nx*ny triangles if
       type = TRIANGLE. If generate_edges = 0 (default) edges are not generated,
       if 1 edges are generated. */
   Mesh(int nx, int ny, Element::Type type, int generate_edges = 0,
        double sx = 1.0, double sy = 1.0)
   {
      Make2D(nx, ny, type, generate_edges, sx, sy);
   }

   /** Creates 1D mesh , divided into n equal intervals. */
   explicit Mesh(int n, double sx = 1.0)
   {
      Make1D(n, sx);
   }

   /** Creates mesh by reading a file in MFEM, netgen, or VTK format. If
       generate_edges = 0 (default) edges are not generated, if 1 edges are
       generated. */
   Mesh(const char *filename, int generate_edges = 0, int refine = 1,
        bool fix_orientation = true);

   /** Creates mesh by reading data stream in MFEM, netgen, or VTK format. If
       generate_edges = 0 (default) edges are not generated, if 1 edges are
       generated. */
   Mesh(std::istream &input, int generate_edges = 0, int refine = 1,
        bool fix_orientation = true);

   /// Create a disjoint mesh from the given mesh array
   Mesh(Mesh *mesh_array[], int num_pieces);

   /// Create a uniformly refined (by any factor) version of @a orig_mesh.
   /** @param[in] orig_mesh  The starting coarse mesh.
       @param[in] ref_factor The refinement factor, an integer > 1.
       @param[in] ref_type   Specify the positions of the new vertices. The
                             options are BasisType::ClosedUniform or
                             BasisType::GaussLobatto.

       The refinement data which can be accessed with GetRefinementTransforms()
       is set to reflect the performed refinements.

       @note The constructed Mesh is linear, i.e. it does not have nodes. */
   Mesh(Mesh *orig_mesh, int ref_factor, int ref_type);

   /** This is similar to the mesh constructor with the same arguments, but here
       the current mesh is destroyed and another one created based on the data
       stream again given in MFEM, netgen, or VTK format. If generate_edges = 0
       (default) edges are not generated, if 1 edges are generated. */
   /// \see mfem::igzstream() for on-the-fly decompression of compressed ascii
   /// inputs.
   virtual void Load(std::istream &input, int generate_edges = 0,
                     int refine = 1, bool fix_orientation = true)
   {
      Loader(input, generate_edges);
      Finalize(refine, fix_orientation);
   }

   /// Clear the contents of the Mesh.
   void Clear() { Destroy(); SetEmpty(); }

   /** @brief Get the mesh generator/type.

       @return A bitmask:
       - bit 0 - simplices are present in the mesh (triangles, tets),
       - bit 1 - tensor product elements are present in the mesh (quads, hexes).
   */
   inline int MeshGenerator() { return meshgen; }

   /** @brief Returns number of vertices.  Vertices are only at the corners of
       elements, where you would expect them in the lowest-order mesh. */
   inline int GetNV() const { return NumOfVertices; }

   /// Returns number of elements.
   inline int GetNE() const { return NumOfElements; }

   /// Returns number of boundary elements.
   inline int GetNBE() const { return NumOfBdrElements; }

   /// Return the number of edges.
   inline int GetNEdges() const { return NumOfEdges; }

   /// Return the number of faces in a 3D mesh.
   inline int GetNFaces() const { return NumOfFaces; }

   /// Return the number of planars in a 4D mesh.
   inline int GetNPlanars() const { return NumOfPlanars; }

   /// Return the number of faces (3D), edges (2D) or vertices (1D).
   int GetNumFaces() const;

   /// Utility function: sum integers from all processors (Allreduce).
   virtual long ReduceInt(int value) const { return value; }

   /// Return the total (global) number of elements.
   long GetGlobalNE() const { return ReduceInt(NumOfElements); }

   /// Equals 1 + num_holes - num_loops
   inline int EulerNumber() const
   { return NumOfVertices - NumOfEdges + NumOfFaces - NumOfElements; }
   /// Equals 1 - num_holes
   inline int EulerNumber2D() const
   { return NumOfVertices - NumOfEdges + NumOfElements; }
   inline int EulerNumber4D() const
   { return NumOfVertices - NumOfEdges + NumOfPlanars - NumOfFaces + NumOfElements;}

   int Dimension() const { return Dim; }
   int SpaceDimension() const { return spaceDim; }

   /// @brief Return pointer to vertex i's coordinates.
   /// @warning For high-order meshes (when #Nodes != NULL) vertices may not be
   /// updated and should not be used!
   const double *GetVertex(int i) const { return vertices[i](); }

   /// @brief Return pointer to vertex i's coordinates.
   /// @warning For high-order meshes (when Nodes != NULL) vertices may not
   /// being updated and should not be used!
   double *GetVertex(int i) { return vertices[i](); }

   void GetElementData(int geom, Array<int> &elem_vtx, Array<int> &attr) const
   { GetElementData(elements, geom, elem_vtx, attr); }

   void GetBdrElementData(int geom, Array<int> &bdr_elem_vtx,
                          Array<int> &bdr_attr) const
   { GetElementData(boundary, geom, bdr_elem_vtx, bdr_attr); }

   /** @brief Set the internal Vertex array to point to the given @a vertices
       array without assuming ownership of the pointer. */
   /** If @a zerocopy is `true`, the vertices must be given as an array of 3
       doubles per vertex. If @a zerocopy is `false` then the current Vertex
       data is first copied to the @a vertices array. */
   void ChangeVertexDataOwnership(double *vertices, int len_vertices,
                                  bool zerocopy = false);

   const Element* const *GetElementsArray() const
   { return elements.GetData(); }

   const Element *GetElement(int i) const { return elements[i]; }

   Element *GetElement(int i) { return elements[i]; }

   bool getSwappedElementInfo(int i) const { return swappedElements[i]; }
   bool getSwappedFaceElementInfo(int i) const { return swappedFaces[i]; }

   const Element *GetBdrElement(int i) const { return boundary[i]; }

   Element *GetBdrElement(int i) { return boundary[i]; }

   const Element *GetFace(int i) const { return faces[i]; }

   int GetFaceBaseGeometry(int i) const;

   int GetPlanarBaseGeometry(int i) const;

   int GetBdrPlanarBaseGeometry(int i) const;

   int GetElementBaseGeometry(int i = 0) const
   { return i < GetNE() ? elements[i]->GetGeometryType() : BaseGeom; }

   int GetBdrElementBaseGeometry(int i = 0) const
   { return i < GetNBE() ? boundary[i]->GetGeometryType() : BaseBdrGeom; }

   /// Returns the indices of the dofs of element i.
   void GetElementVertices(int i, Array<int> &dofs) const
   { elements[i]->GetVertices(dofs); }

   /// Returns the indices of the dofs of boundary element i.
   void GetBdrElementVertices(int i, Array<int> &dofs) const
   { boundary[i]->GetVertices(dofs); }

   /// Return the indices and the orientations of all edges of element i.
   void GetElementEdges(int i, Array<int> &edges, Array<int> &cor) const;

   /// Return the indices and the orientations of all edges of bdr element i.
   void GetBdrElementEdges(int i, Array<int> &edges, Array<int> &cor) const;

   /// Return the indices and the orientations of all planars of element i.
   void GetBdrElementPlanars(int i, Array<int> &pls, Array<int> &cor) const;

   /** Return the indices and the orientations of all edges of face i.
       Works for both 2D (face=edge) and 3D faces. */
   void GetFaceEdges(int i, Array<int> &, Array<int> &) const;

   /// Returns the indices of the vertices of face i.
   void GetFaceVertices(int i, Array<int> &vert) const
   {
      if (Dim == 1)
      {
         vert.SetSize(1); vert[0] = i;
      }
      else
      {
         faces[i]->GetVertices(vert);
      }
   }

   /// Returns the indices of the vertices of edge i.
   void GetEdgeVertices(int i, Array<int> &vert) const;

   void GetPlanVertices(int i, Array<int> &vert) const;

   /// Returns the face-to-edge Table (3D)
   Table *GetFaceEdgeTable() const;

   /// Returns the edge-to-vertex Table (3D)
   Table *GetEdgeVertexTable() const;

   /// Return the indices and the orientations of all faces of element i.
   void GetElementFaces(int i, Array<int> &, Array<int> &) const;

   /// Return the index and the orientation of the face of bdr element i. (3D)
   void GetBdrElementFace(int i, int *, int *) const;

   /// Return the indices and the orientations of all planars of element i.
   void GetElementPlanars(int i, Array<int> &pls, Array<int> &cor) const;

   /** Return the vertex index of boundary element i. (1D)
       Return the edge index of boundary element i. (2D)
       Return the face index of boundary element i. (3D) */
   int GetBdrElementEdgeIndex(int i) const;

   /** @brief For the given boundary element, bdr_el, return its adjacent
       element and its info, i.e. 64*local_bdr_index+bdr_orientation. */
   void GetBdrElementAdjacentElement(int bdr_el, int &el, int &info) const;

   /// Returns the type of element i.
   int GetElementType(int i) const;

   /// Returns the type of boundary element i.
   int GetBdrElementType(int i) const;

   /* Return point matrix of element i of dimension Dim X #dofs, where for
      every degree of freedom we give its coordinates in space of dimension
      Dim. */
   void GetPointMatrix(int i, DenseMatrix &pointmat) const;

   /* Return point matrix of boundary element i of dimension Dim X #dofs,
      where for every degree of freedom we give its coordinates in space
      of dimension Dim. */
   void GetBdrPointMatrix(int i, DenseMatrix &pointmat) const;

   static FiniteElement *GetTransformationFEforElementType(int);

   /** Builds the transformation defining the i-th element in the user-defined
       variable. */
   void GetElementTransformation(int i, IsoparametricTransformation *ElTr);

   /// Returns the transformation defining the i-th element
   ElementTransformation *GetElementTransformation(int i);

   /** Return the transformation defining the i-th element assuming
       the position of the vertices/nodes are given by 'nodes'. */
   void GetElementTransformation(int i, const Vector &nodes,
                                 IsoparametricTransformation *ElTr);

   /// Returns the transformation defining the i-th boundary element
   ElementTransformation * GetBdrElementTransformation(int i);
   void GetBdrElementTransformation(int i, IsoparametricTransformation *ElTr);

   /** Returns the transformation defining the given face element.
       The transformation is stored in a user-defined variable. */
   void GetFaceTransformation(int i, IsoparametricTransformation *FTr);

   /// Returns the transformation defining the given face element
   ElementTransformation *GetFaceTransformation(int FaceNo);

   /** Returns the transformation defining the given edge element.
       The transformation is stored in a user-defined variable. */
   void GetEdgeTransformation(int i, IsoparametricTransformation *EdTr);

   /// Returns the transformation defining the given face element
   ElementTransformation *GetEdgeTransformation(int EdgeNo);

   /// Returns (a pointer to a structure containing) the following data:
   ///
   /// 1) Elem1No - the index of the first element that contains this face this
   ///    is the element that has the same outward unit normal vector as the
   ///    face;
   ///
   /// 2) Elem2No - the index of the second element that contains this face this
   ///    element has outward unit normal vector as the face multiplied with -1;
   ///
   /// 3) Elem1, Elem2 - pointers to the ElementTransformation's of the first
   ///    and the second element respectively;
   ///
   /// 4) Face - pointer to the ElementTransformation of the face;
   ///
   /// 5) Loc1, Loc2 - IntegrationPointTransformation's mapping the face
   ///    coordinate system to the element coordinate system (both in their
   ///    reference elements). Used to transform IntegrationPoints from face to
   ///    element. More formally, let:
   ///       TL1, TL2 be the transformations represented by Loc1, Loc2,
   ///       TE1, TE2 - the transformations represented by Elem1, Elem2,
   ///       TF - the transformation represented by Face, then
   ///       TF(x) = TE1(TL1(x)) = TE2(TL2(x)) for all x in the reference face.
   ///
   /// 6) FaceGeom - the base geometry for the face.
   ///
   /// The mask specifies which fields in the structure to return:
   ///    mask & 1 - Elem1, mask & 2 - Elem2
   ///    mask & 4 - Loc1, mask & 8 - Loc2, mask & 16 - Face.
   FaceElementTransformations *GetFaceElementTransformations(int FaceNo,
                                                             int mask = 31);

   FaceElementTransformations *GetInteriorFaceTransformations (int FaceNo)
   {
      if (faces_info[FaceNo].Elem2No < 0) { return NULL; }
      return GetFaceElementTransformations (FaceNo);
   }

   FaceElementTransformations *GetBdrFaceTransformations (int BdrElemNo);

   /// Return true if the given face is interior
   bool FaceIsInterior(int FaceNo) const
   {
      return (faces_info[FaceNo].Elem2No >= 0);
   }
   void GetFaceElements (int Face, int *Elem1, int *Elem2);
   void GetFaceInfos (int Face, int *Inf1, int *Inf2);

   int GetFaceGeometryType(int Face) const;
   int GetFaceElementType(int Face) const;

   /// Check the orientation of the elements
   /** @return The number of elements with wrong orientation. */
   int CheckElementOrientation(bool fix_it = true);
   /// Check the orientation of the boundary elements
   /** @return The number of boundary elements with wrong orientation. */
   int CheckBdrElementOrientation(bool fix_it = true);

   /// Return the attribute of element i.
   int GetAttribute(int i) const { return elements[i]->GetAttribute(); }

   /// Set the attribute of element i.
   void SetAttribute(int i, int attr) { elements[i]->SetAttribute(attr); }

   /// Return the attribute of boundary element i.
   int GetBdrAttribute(int i) const { return boundary[i]->GetAttribute(); }

   const Table &ElementToElementTable();

   const Table &ElementToFaceTable() const;

   const Table &ElementToPlanTable() const;

   const Table &ElementToEdgeTable() const;

   ///  The returned Table must be destroyed by the caller
   Table *GetVertexToElementTable();

   /** Return the "face"-element Table. Here "face" refers to face (3D),
       edge (2D), or vertex (1D).
       The returned Table must be destroyed by the caller. */
   Table *GetFaceToElementTable() const;

   /** This method modifies a tetrahedral mesh so that Nedelec spaces of order
       greater than 1 can be defined on the mesh. Specifically, we
       1) rotate all tets in the mesh so that the vertices {v0, v1, v2, v3}
       satisfy: v0 < v1 < min(v2, v3).
       2) rotate all boundary triangles so that the vertices {v0, v1, v2}
       satisfy: v0 < min(v1, v2).

       @note Refinement does not work after a call to this method! */
   virtual void ReorientTetMesh();

   void ReplaceBoundaryFromFaces();

   int *CartesianPartitioning(int nxyz[]);
   int *GeneratePartitioning(int nparts, int part_method = 1);
   void CheckPartitioning(int *partitioning);

   void CheckDisplacements(const Vector &displacements, double &tmax);

   // Vertices are only at the corners of elements, where you would expect them
   // in the lowest-order mesh.
   void MoveVertices(const Vector &displacements);
   void GetVertices(Vector &vert_coord) const;
   void SetVertices(const Vector &vert_coord);

   // Nodes are only active for higher order meshes, and share locations with
   // the vertices, plus all the higher- order control points within the element
   // and along the edges and on the faces.
   void GetNode(int i, double *coord);
   void SetNode(int i, const double *coord);

   // Node operations for curved mesh.
   // They call the corresponding '...Vertices' method if the
   // mesh is not curved (i.e. Nodes == NULL).
   void MoveNodes(const Vector &displacements);
   void GetNodes(Vector &node_coord) const;
   void SetNodes(const Vector &node_coord);

   /// Return a pointer to the internal node GridFunction (may be NULL).
   GridFunction *GetNodes() { return Nodes; }
   const GridFunction *GetNodes() const { return Nodes; }
   /// Return the mesh nodes ownership flag.
   bool OwnsNodes() const { return own_nodes; }
   /// Set the mesh nodes ownership flag.
   void SetNodesOwner(bool nodes_owner) { own_nodes = nodes_owner; }
   /// Replace the internal node GridFunction with the given GridFunction.
   void NewNodes(GridFunction &nodes, bool make_owner = false);
   /** Swap the internal node GridFunction pointer and ownership flag members
       with the given ones. */
   void SwapNodes(GridFunction *&nodes, int &own_nodes_);

   /// Return the mesh nodes/vertices projected on the given GridFunction.
   void GetNodes(GridFunction &nodes) const;
   /** Replace the internal node GridFunction with a new GridFunction defined
       on the given FiniteElementSpace. The new node coordinates are projected
       (derived) from the current nodes/vertices. */
   void SetNodalFESpace(FiniteElementSpace *nfes);
   /** Replace the internal node GridFunction with the given GridFunction. The
       given GridFunction is updated with node coordinates projected (derived)
       from the current nodes/vertices. */
   void SetNodalGridFunction(GridFunction *nodes, bool make_owner = false);
   /** Return the FiniteElementSpace on which the current mesh nodes are
       defined or NULL if the mesh does not have nodes. */
   const FiniteElementSpace *GetNodalFESpace() const;

   /** Set the curvature of the mesh nodes using the given polynomial degree,
       'order', and optionally: discontinuous or continuous FE space, 'discont',
       new space dimension, 'space_dim' (if != -1), and 'ordering'. */
   void SetCurvature(int order, bool discont = false, int space_dim = -1,
                     int ordering = 1);

   /** Refine all mesh elements. */
   void UniformRefinement();

   /** Refine selected mesh elements. Refinement type can be specified for each
       element. The function can do conforming refinement of triangles and
       tetrahedra and non-conforming refinement (i.e., with hanging-nodes) of
       triangles, quadrilaterals and hexahedrons. If 'nonconforming' = -1,
       suitable refinement method is selected automatically (namely, conforming
       refinement for triangles). Use nonconforming = 0/1 to force the method.
       For nonconforming refinements, nc_limit optionally specifies the maximum
       level of hanging nodes (unlimited by default). */
   void GeneralRefinement(const Array<Refinement> &refinements,
                          int nonconforming = -1, int nc_limit = 0);

   /** Simplified version of GeneralRefinement taking a simple list of elements
       to refine, without refinement types. */
   void GeneralRefinement(const Array<int> &el_to_refine,
                          int nonconforming = -1, int nc_limit = 0);

   /// Refine each element with given probability. Uses GeneralRefinement.
   void RandomRefinement(double prob, bool aniso = false,
                         int nonconforming = -1, int nc_limit = 0);

   /// Refine elements sharing the specified vertex. Uses GeneralRefinement.
   void RefineAtVertex(const Vertex& vert,
                       double eps = 0.0, int nonconforming = -1);

   /** Refine element i if elem_error[i] > threshold, for all i.
       Returns true if at least one element was refined, false otherwise. */
   bool RefineByError(const Array<double> &elem_error, double threshold,
                      int nonconforming = -1, int nc_limit = 0);

   /** Refine element i if elem_error(i) > threshold, for all i.
       Returns true if at least one element was refined, false otherwise. */
   bool RefineByError(const Vector &elem_error, double threshold,
                      int nonconforming = -1, int nc_limit = 0);

   /** Derefine the mesh based on an error measure associated with each
       element. A derefinement is performed if the sum of errors of its fine
       elements is smaller than 'threshold'. If 'nc_limit' > 0, derefinements
       that would increase the maximum level of hanging nodes of the mesh are
       skipped. Returns true if the mesh changed, false otherwise. */
   bool DerefineByError(Array<double> &elem_error, double threshold,
                        int nc_limit = 0, int op = 1);

   /// Same as DerefineByError for an error vector.
   bool DerefineByError(const Vector &elem_error, double threshold,
                        int nc_limit = 0, int op = 1);

   ///@{ @name NURBS mesh refinement methods
   void KnotInsert(Array<KnotVector *> &kv);
   void DegreeElevate(int t);
   ///@}

   /** Make sure that a quad/hex mesh is considered to be non-conforming (i.e.,
       has an associated NCMesh object). Triangles meshes can be both conforming
       (default) or non-conforming. */
   void EnsureNCMesh(bool triangles_nonconforming = false);

   bool Conforming() const { return ncmesh == NULL; }
   bool Nonconforming() const { return ncmesh != NULL; }

   /** Return fine element transformations following a mesh refinement.
       Space uses this to construct a global interpolation matrix. */
   const CoarseFineTransformations &GetRefinementTransforms();

   /// Return type of last modification of the mesh.
   Operation GetLastOperation() const { return last_operation; }

   /** Return update counter. The counter starts at zero and is incremented
       each time refinement, derefinement, or rebalancing method is called.
       It is used for checking proper sequence of Space:: and GridFunction::
       Update() calls. */
   long GetSequence() const { return sequence; }

   /// Print the mesh to the given stream using Netgen/Truegrid format.
   virtual void PrintXG(std::ostream &out = std::cout) const;

   /// Print the mesh to the given stream using the default MFEM mesh format.
   /// \see mfem::ogzstream() for on-the-fly compression of ascii outputs
   virtual void Print(std::ostream &out = std::cout) const { Printer(out); }

   /// Print the mesh in VTK format (linear and quadratic meshes only).
   /// \see mfem::ogzstream() for on-the-fly compression of ascii outputs
   void PrintVTK(std::ostream &out);

   /** Print the mesh in VTK format. The parameter ref > 0 specifies an element
       subdivision number (useful for high order fields and curved meshes).
       If the optional field_data is set, we also add a FIELD section in the
       beginning of the file with additional dataset information. */
   /// \see mfem::ogzstream() for on-the-fly compression of ascii outputs
   void PrintVTK(std::ostream &out, int ref, int field_data=0);

   void GetElementColoring(Array<int> &colors, int el0 = 0);

   /** Prints the mesh with bdr elements given by the boundary of
       the subdomains, so that the boundary of subdomain i has bdr
       attribute i+1. */
   /// \see mfem::ogzstream() for on-the-fly compression of ascii outputs
   void PrintWithPartitioning (int *partitioning,
                               std::ostream &out, int elem_attr = 0) const;

   void PrintElementsWithPartitioning (int *partitioning,
                                       std::ostream &out,
                                       int interior_faces = 0);

   /// Print set of disjoint surfaces:
   /*!
    * If Aface_face(i,j) != 0, print face j as a boundary
    * element with attribute i+1.
    */
   void PrintSurfaces(const Table &Aface_face, std::ostream &out) const;

   void ScaleSubdomains (double sf);
   void ScaleElements (double sf);

   void Transform(void (*f)(const Vector&, Vector&));
   void Transform(VectorCoefficient &deformation);

   /// Remove unused vertices and rebuild mesh connectivity.
   void RemoveUnusedVertices();

   /** Remove boundary elements that lie in the interior of the mesh, i.e. that
       have two adjacent faces in 3D, or edges in 2D. */
   void RemoveInternalBoundaries();

   /** Get the size of the i-th element relative to the perfect
       reference element. */
   double GetElementSize(int i, int type = 0);

   double GetElementSize(int i, const Vector &dir);

   double GetElementVolume(int i);

   /// Returns the minimum and maximum corners of the mesh bounding box. For
   /// high-order meshes, the geometry is refined first "ref" times.
   void GetBoundingBox(Vector &min, Vector &max, int ref = 2);

   void GetCharacteristics(double &h_min, double &h_max,
                           double &kappa_min, double &kappa_max,
                           Vector *Vh = NULL, Vector *Vk = NULL);

   void PrintCharacteristics(Vector *Vh = NULL, Vector *Vk = NULL,
                             std::ostream &out = std::cout);

   virtual void PrintInfo(std::ostream &out = std::cout)
   {
      PrintCharacteristics(NULL, NULL, out);
   }

   void MesquiteSmooth(const int mesquite_option = 0);

   /// Destroys Mesh.
   virtual ~Mesh() { DestroyPointers(); }
};

/** Overload operator<< for std::ostream and Mesh; valid also for the derived
    class ParMesh */
std::ostream &operator<<(std::ostream &out, const Mesh &mesh);


/// Class used to extrude the nodes of a mesh
class NodeExtrudeCoefficient : public VectorCoefficient
{
private:
   int n, layer;
   double p[2], s;
   Vector tip;
public:
   NodeExtrudeCoefficient(const int dim, const int _n, const double _s);
   void SetLayer(const int l) { layer = l; }
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   virtual ~NodeExtrudeCoefficient() { }
};


/// Extrude a 1D mesh
Mesh *Extrude1D(Mesh *mesh, const int ny, const double sy,
                const bool closed = false);


/// Input file stream that remembers the input file name (useful for example
/// when reading NetCDF meshes) and supports optional gzstream decompression.
class named_ifgzstream : public mfem::ifgzstream
{
public:
   const char *filename;
   named_ifgzstream(const char *mesh_name) :
      mfem::ifgzstream(mesh_name), filename(mesh_name) {}
};


// inline functions
inline void Mesh::Swap(int *a) const
{
   int temp = a[0];
   a[0] = a[1];
   a[1] = temp;
}

inline void Mesh::ShiftL2R(int &a, int &b, int &c)
{
   int t = a;
   a = c;  c = b;  b = t;
}

inline void Mesh::Rotate3(int &a, int &b, int &c)
{
   if (a < b)
   {
      if (a > c)
      {
         ShiftL2R(a, b, c);
      }
   }
   else
   {
      if (b < c)
      {
         ShiftL2R(c, b, a);
      }
      else
      {
         ShiftL2R(a, b, c);
      }
   }
}

}

#endif
