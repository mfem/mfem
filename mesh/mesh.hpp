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

#ifndef MFEM_MESH
#define MFEM_MESH

#include "../config/config.hpp"
#include "../general/stable3d.hpp"
#include "../general/globals.hpp"
#include "triangle.hpp"
#include "tetrahedron.hpp"
#include "vertex.hpp"
#include "vtk.hpp"
#include "ncmesh.hpp"
#include "../fem/eltrans.hpp"
#include "../fem/coefficient.hpp"
#include "../general/zstr.hpp"
#ifdef MFEM_USE_ADIOS2
#include "../general/adios2stream.hpp"
#endif
#include <iostream>

namespace mfem
{

// Data type mesh

class GeometricFactors;
class FaceGeometricFactors;
class KnotVector;
class NURBSExtension;
class FiniteElementSpace;
class GridFunction;
struct Refinement;

/** An enum type to specify if interior or boundary faces are desired. */
enum class FaceType : bool {Interior, Boundary};

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
   friend class NCMesh;
   friend class NURBSExtension;

#ifdef MFEM_USE_ADIOS2
   friend class adios2stream;
#endif

protected:
   int Dim;
   int spaceDim;

   int NumOfVertices, NumOfElements, NumOfBdrElements;
   int NumOfEdges, NumOfFaces;
   /** These variables store the number of Interior and Boundary faces. Calling
       fes->GetMesh()->GetNBE() doesn't return the expected value in 3D because
       periodic meshes in 3D have some of their faces marked as boundary for
       visualization purpose in GLVis. */
   mutable int nbInteriorFaces, nbBoundaryFaces;

   int meshgen; // see MeshGenerator()
   int mesh_geoms; // sum of (1 << geom) for all geom of all dimensions

   // Counter for Mesh transformations: refinement, derefinement, rebalancing.
   // Used for checking during Update operations on objects depending on the
   // Mesh, such as FiniteElementSpace, GridFunction, etc.
   long sequence;

   Array<Element *> elements;
   // Vertices are only at the corners of elements, where you would expect them
   // in the lowest-order mesh. In some cases, e.g. in a Mesh that defines the
   // patch topology for a NURBS mesh (see LoadPatchTopo()) the vertices may be
   // empty while NumOfVertices is positive.
   Array<Vertex> vertices;
   Array<Element *> boundary;
   Array<Element *> faces;

   /** @brief This structure stores the low level information necessary to
       interpret the configuration of elements on a specific face. This
       information can be accessed using methods like GetFaceElements(),
       GetFaceInfos(), FaceIsInterior(), etc.

       For accessing higher level deciphered information look at
       Mesh::FaceInformation, and its accessor Mesh::GetFaceInformation().

       Each face contains information on the indices, local reference faces,
       orientations, and potential nonconformity for the two neighboring
       elements on a face.
       Each face can either be an interior, boundary, or shared interior face.
       Each interior face is shared by two elements referred as Elem1 and Elem2.
       For boundary faces only the information on Elem1 is relevant.
       Shared interior faces correspond to faces where Elem1 and Elem2 are
       distributed on different MPI ranks.
       Regarding conformity, three cases are distinguished, conforming faces,
       nonconforming slave faces, and nonconforming master faces. Master and
       slave referring to the coarse and fine elements respectively on a
       nonconforming face.
       Nonconforming slave faces always have the slave element as Elem1 and
       the master element as Elem2. On the other side, nonconforming master
       faces always have the master element as Elem1, and one of the slave
       element as Elem2. Except for ghost nonconforming slave faces, where
       Elem1 is the master side and Elem2 is the slave side.

       The indices of Elem1 and Elem2 can be indirectly extracted from
       FaceInfo::Elem1No and FaceInfo::Elem2No, read the note below for special
       cases on the index of Elem2.

       The local face identifiers are deciphered from FaceInfo::Elem1Inf and
       FaceInfo::Elem2Inf through the formula: LocalFaceIndex = ElemInf/64,
       the semantic of the computed local face identifier can be found in
       fem/geom.cpp. The local face identifier corresponds to an index
       in the Constants<Geometry>::Edges arrays for 2D element geometries, and
       to an index in the Constants<Geometry>::FaceVert arrays for 3D element
       geometries.

       The orientation of each element relative to a face is obtained through
       the formula: Orientation = ElemInf%64, the semantic of the orientation
       can also be found in fem/geom.cpp. The orientation corresponds to
       an index in the Constants<Geometry>::Orient arrays, providing the
       sequence of vertices identifying the orientation of an edge/face. By
       convention the orientation of Elem1 is always set to 0, serving as the
       reference orientation. The orientation of Elem2 relatively to Elem1 is
       therefore determined just by using the orientation of Elem2. An important
       special case is the one of nonconforming faces, the orientation should
       be composed with the PointMatrix, which also contains orientation
       information. A special treatment should be done for 2D, the orientation
       in the PointMatrix is not included, therefore when applying the
       PointMatrix transformation, the PointMatrix should be flipped, except for
       shared nonconforming slave faces where the transformation can be applied
       as is.

       Another special case is the case of shared nonconforming faces. Ghost
       faces use a different design based on so called "ghost" faces.
       Ghost faces, as their name suggest are very well hidden, and they
       usually have a separate interface from "standard" faces.
   */
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
   //
   // A local face is one generated from a local element and has index i in
   // faces_info such that i < GetNumFaces(). Also, Elem1No always refers to the
   // element (slave or master, in the nonconforming case) that generated the
   // face.
   // Classification of a local (non-ghost) face based on its FaceInfo:
   // - Elem2No >= 0 --> local interior face; can be either:
   //    - NCFace == -1 --> conforming face, or
   //    - NCFace >= 0 --> nonconforming slave face; Elem2No is the index of
   //      the master volume element; Elem2Inf%64 is 0, see the note in
   //      Mesh::GenerateNCFaceInfo().
   // - Elem2No < 0 --> local "boundary" face; can be one of:
   //    - NCFace == -1 --> conforming face; can be either:
   //       - Elem2Inf < 0 --> true boundary face (no element on side 2)
   //       - Elem2Inf >= 0 --> shared face where element 2 is a face-neighbor
   //         element with index -1-Elem2No. This state is initialized by
   //         ParMesh::ExchangeFaceNbrData().
   //    - NCFace >= 0 --> nonconforming face; can be one of:
   //       - Elem2Inf < 0 --> master nonconforming face, interior or shared;
   //         In this case, Elem2No is -1; see GenerateNCFaceInfo().
   //       - Elem2Inf >= 0 --> shared slave nonconforming face where element 2
   //         is the master face-neighbor element with index -1-Elem2No; see
   //         ParNCMesh::GetFaceNeighbors().
   //
   // A ghost face is a nonconforming face that is generated by a non-local,
   // i.e. ghost, element. A ghost face has index i in faces_info such that
   // i >= GetNumFaces().
   // Classification of a ghost (non-local) face based on its FaceInfo:
   // - Elem1No == -1 --> master ghost face? These ghost faces also have:
   //   Elem2No == -1, Elem1Inf == Elem2Inf == -1, and NCFace == -1.
   // - Elem1No >= 0 --> slave ghost face; Elem1No is the index of the local
   //   master side element, i.e. side 1 IS NOT the side that generated the
   //   face. Elem2No is < 0 and -1-Elem2No is the index of the ghost
   //   face-neighbor element that generated this slave ghost face. In this
   //   case, Elem2Inf >= 0 and NCFace >= 0.
   // Relevant methods: GenerateFaces(), GenerateNCFaceInfo(),
   //                   ParNCMesh::GetFaceNeighbors(),
   //                   ParMesh::ExchangeFaceNbrData()

   struct NCFaceInfo
   {
      bool Slave; // true if this is a slave face, false if master face
      int MasterFace; // if Slave, this is the index of the master face
      // If not Slave, 'MasterFace' is the local face index of this master face
      // as a face in the unique adjacent element.
      const DenseMatrix* PointMatrix; // if Slave, position within master face
      // (NOTE: PointMatrix points to a matrix owned by NCMesh.)

      NCFaceInfo() = default;

      NCFaceInfo(bool slave, int master, const DenseMatrix* pm)
         : Slave(slave), MasterFace(master), PointMatrix(pm) {}
   };

   Array<FaceInfo> faces_info;
   Array<NCFaceInfo> nc_faces_info;

   Table *el_to_edge;
   Table *el_to_face;
   Table *el_to_el;
   Array<int> be_to_edge;  // for 2D
   Table *bel_to_edge;     // for 3D
   Array<int> be_to_face;

   // Note that the following tables are owned by this class and should not be
   // deleted by the caller. Of these three tables, only face_edge and
   // edge_vertex are returned by access functions.
   mutable Table *face_to_elem;  // Used by FindFaceNeighbors, not returned.
   mutable Table *face_edge;     // Returned by GetFaceEdgeTable().
   mutable Table *edge_vertex;   // Returned by GetEdgeVertexTable().

   IsoparametricTransformation Transformation, Transformation2;
   IsoparametricTransformation BdrTransformation;
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
   static const int vtk_quadratic_pyramid[13];
   static const int vtk_quadratic_wedge[18];
   static const int vtk_quadratic_hex[27];

#ifdef MFEM_USE_MEMALLOC
   friend class Tetrahedron;
   MemAlloc <Tetrahedron, 1024> TetMemory;
#endif

   // used during NC mesh initialization only
   Array<Triple<int, int, int> > tmp_vertex_parents;

public:
   typedef Geometry::Constants<Geometry::SEGMENT>     seg_t;
   typedef Geometry::Constants<Geometry::TRIANGLE>    tri_t;
   typedef Geometry::Constants<Geometry::SQUARE>      quad_t;
   typedef Geometry::Constants<Geometry::TETRAHEDRON> tet_t;
   typedef Geometry::Constants<Geometry::CUBE>        hex_t;
   typedef Geometry::Constants<Geometry::PRISM>       pri_t;
   typedef Geometry::Constants<Geometry::PYRAMID>     pyr_t;

   enum Operation { NONE, REFINE, DEREFINE, REBALANCE };

   /// A list of all unique element attributes used by the Mesh.
   Array<int> attributes;
   /// A list of all unique boundary attributes used by the Mesh.
   Array<int> bdr_attributes;

   NURBSExtension *NURBSext; ///< Optional NURBS mesh extension.
   NCMesh *ncmesh;           ///< Optional nonconforming mesh extension.
   Array<GeometricFactors*> geom_factors; ///< Optional geometric factors.
   Array<FaceGeometricFactors*> face_geom_factors; /**< Optional face geometric
                                                        factors. */

   // Global parameter that can be used to control the removal of unused
   // vertices performed when reading a mesh in MFEM format. The default value
   // (true) is set in mesh_readers.cpp.
   static bool remove_unused_vertices;

protected:
   Operation last_operation;

   void Init();
   void InitTables();
   void SetEmpty();  // Init all data members with empty values
   void DestroyTables();
   void DeleteTables() { DestroyTables(); InitTables(); }
   void DestroyPointers(); // Delete data specifically allocated by class Mesh.
   void Destroy();         // Delete all owned data.
   void ResetLazyData();

   Element *ReadElementWithoutAttr(std::istream &);
   static void PrintElementWithoutAttr(const Element *, std::ostream &);

   Element *ReadElement(std::istream &);
   static void PrintElement(const Element *, std::ostream &);

   // Readers for different mesh formats, used in the Load() method.
   // The implementations of these methods are in mesh_readers.cpp.
   void ReadMFEMMesh(std::istream &input, int version, int &curved);
   void ReadLineMesh(std::istream &input);
   void ReadNetgen2DMesh(std::istream &input, int &curved);
   void ReadNetgen3DMesh(std::istream &input);
   void ReadTrueGridMesh(std::istream &input);
   void CreateVTKMesh(const Vector &points, const Array<int> &cell_data,
                      const Array<int> &cell_offsets,
                      const Array<int> &cell_types,
                      const Array<int> &cell_attributes,
                      int &curved, int &read_gf, bool &finalize_topo);
   void ReadVTKMesh(std::istream &input, int &curved, int &read_gf,
                    bool &finalize_topo);
   void ReadXML_VTKMesh(std::istream &input, int &curved, int &read_gf,
                        bool &finalize_topo, const std::string &xml_prefix="");
   void ReadNURBSMesh(std::istream &input, int &curved, int &read_gf);
   void ReadInlineMesh(std::istream &input, bool generate_edges = false);
   void ReadGmshMesh(std::istream &input, int &curved, int &read_gf);
   /* Note NetCDF (optional library) is used for reading cubit files */
#ifdef MFEM_USE_NETCDF
   void ReadCubit(const char *filename, int &curved, int &read_gf);
#endif

   /// Determine the mesh generator bitmask #meshgen, see MeshGenerator().
   /** Also, initializes #mesh_geoms. */
   void SetMeshGen();

   /// Return the length of the segment from node i to node j.
   double GetLength(int i, int j) const;

   void MarkForRefinement();
   void MarkTriMeshForRefinement();
   void GetEdgeOrdering(DSTable &v_to_v, Array<int> &order);
   virtual void MarkTetMeshForRefinement(DSTable &v_to_v);

   // Methods used to prepare and apply permutation of the mesh nodes assuming
   // that the mesh elements may be rotated (e.g. to mark triangle or tet edges
   // for refinement) between the two calls - PrepareNodeReorder() and
   // DoNodeReorder(). The latter method assumes that the 'faces' have not been
   // updated after the element rotations.
   void PrepareNodeReorder(DSTable **old_v_to_v, Table **old_elem_vert);
   void DoNodeReorder(DSTable *old_v_to_v, Table *old_elem_vert);

   STable3D *GetFacesTable();
   STable3D *GetElementToFaceTable(int ret_ftbl = 0);

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

   /// Bisect a triangle: element with index @a i is bisected.
   void Bisection(int i, const DSTable &, int *, int *, int *);

   /// Bisect a tetrahedron: element with index @a i is bisected.
   void Bisection(int i, HashTable<Hashed2> &);

   /// Bisect a boundary triangle: boundary element with index @a i is bisected.
   void BdrBisection(int i, const HashTable<Hashed2> &);

   /** Uniform Refinement. Element with index i is refined uniformly. */
   void UniformRefinement(int i, const DSTable &, int *, int *, int *);

   /** @brief Averages the vertices with given @a indexes and saves the result
       in #vertices[result]. */
   void AverageVertices(const int *indexes, int n, int result);

   void InitRefinementTransforms();
   int FindCoarseElement(int i);

   /** @brief Update the nodes of a curved mesh after the topological part of a
       Mesh::Operation, such as refinement, has been performed. */
   /** If Nodes GridFunction is defined, i.e. not NULL, this method calls
       NodesUpdated().

       @note Unlike the similarly named public method NodesUpdated() this
       method modifies the mesh nodes (if they exist) and calls NodesUpdated().
   */
   void UpdateNodes();

   /// Helper to set vertex coordinates given a high-order curvature function.
   void SetVerticesFromNodes(const GridFunction *nodes);

   void UniformRefinement2D_base(bool update_nodes = true);

   /// Refine a mixed 2D mesh uniformly.
   virtual void UniformRefinement2D() { UniformRefinement2D_base(); }

   /* If @a f2qf is not NULL, adds all quadrilateral faces to @a f2qf which
      represents a "face-to-quad-face" index map. When all faces are quads, the
      array @a f2qf is kept empty since it is not needed. */
   void UniformRefinement3D_base(Array<int> *f2qf = NULL,
                                 DSTable *v_to_v_p = NULL,
                                 bool update_nodes = true);

   /// Refine a mixed 3D mesh uniformly.
   virtual void UniformRefinement3D() { UniformRefinement3D_base(); }

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
   /// Derefinement helper.
   double AggregateError(const Array<double> &elem_error,
                         const int *fine, int nfine, int op);

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
   void GetLocalTriToWdgTransformation (IsoparametricTransformation &loc,
                                        int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalTriToPyrTransformation (IsoparametricTransformation &loc,
                                        int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalQuadToHexTransformation (IsoparametricTransformation &loc,
                                         int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalQuadToWdgTransformation (IsoparametricTransformation &loc,
                                         int i);
   /// Used in GetFaceElementTransformations (...)
   void GetLocalQuadToPyrTransformation (IsoparametricTransformation &loc,
                                         int i);

   /** Used in GetFaceElementTransformations to account for the fact that a
       slave face occupies only a portion of its master face. */
   void ApplyLocalSlaveTransformation(FaceElementTransformations &FT,
                                      const FaceInfo &fi, bool is_ghost);

   bool IsSlaveFace(const FaceInfo &fi) const;

   /// Returns the orientation of "test" relative to "base"
   static int GetTriOrientation (const int * base, const int * test);
   /// Returns the orientation of "test" relative to "base"
   static int GetQuadOrientation (const int * base, const int * test);
   /// Returns the orientation of "test" relative to "base"
   static int GetTetOrientation (const int * base, const int * test);

   static void GetElementArrayEdgeTable(const Array<Element*> &elem_array,
                                        const DSTable &v_to_v,
                                        Table &el_to_edge);

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
   /** For a serial Mesh, return true if the face is interior. For a parallel
       ParMesh return true if the face is interior or shared. In parallel, this
       method only works if the face neighbor data is exchanged. */
   bool FaceIsTrueInterior(int FaceNo) const
   {
      return FaceIsInterior(FaceNo) || (faces_info[FaceNo].Elem2Inf >= 0);
   }

   void FreeElement(Element *E);

   void GenerateFaces();
   void GenerateNCFaceInfo();

   /// Begin construction of a mesh
   void InitMesh(int Dim_, int spaceDim_, int NVert, int NElem, int NBdrElem);

   // Used in the methods FinalizeXXXMesh() and FinalizeTopology()
   void FinalizeCheck();

   void Loader(std::istream &input, int generate_edges = 0,
               std::string parse_tag = "");

   // If NURBS mesh, write NURBS format. If NCMesh, write mfem v1.1 format.
   // If section_delimiter is empty, write mfem v1.0 format. Otherwise, write
   // mfem v1.2 format with the given section_delimiter at the end.
   void Printer(std::ostream &out = mfem::out,
                std::string section_delimiter = "") const;

   /** Creates mesh for the parallelepiped [0,sx]x[0,sy]x[0,sz], divided into
       nx*ny*nz hexahedra if type=HEXAHEDRON or into 6*nx*ny*nz tetrahedrons if
       type=TETRAHEDRON. The parameter @a sfc_ordering controls how the elements
       (when type=HEXAHEDRON) are ordered: true - use space-filling curve
       ordering, or false - use lexicographic ordering. */
   void Make3D(int nx, int ny, int nz, Element::Type type,
               double sx, double sy, double sz, bool sfc_ordering);

   /** Creates mesh for the rectangle [0,sx]x[0,sy], divided into nx*ny
       quadrilaterals if type = QUADRILATERAL or into 2*nx*ny triangles if
       type = TRIANGLE. If generate_edges = 0 (default) edges are not generated,
       if 1 edges are generated. The parameter @a sfc_ordering controls how the
       elements (when type=QUADRILATERAL) are ordered: true - use space-filling
       curve ordering, or false - use lexicographic ordering. */
   void Make2D(int nx, int ny, Element::Type type, double sx, double sy,
               bool generate_edges, bool sfc_ordering);

   /// Creates a 1D mesh for the interval [0,sx] divided into n equal intervals.
   void Make1D(int n, double sx = 1.0);

   /// Internal function used in Mesh::MakeRefined
   void MakeRefined_(Mesh &orig_mesh, const Array<int> ref_factors,
                     int ref_type);

   /// Initialize vertices/elements/boundary/tables from a nonconforming mesh.
   void InitFromNCMesh(const NCMesh &ncmesh);

   /// Create from a nonconforming mesh.
   explicit Mesh(const NCMesh &ncmesh);

   // used in GetElementData() and GetBdrElementData()
   void GetElementData(const Array<Element*> &elem_array, int geom,
                       Array<int> &elem_vtx, Array<int> &attr) const;

   double GetElementSize(ElementTransformation *T, int type = 0);

   // Internal helper used in MakeSimplicial (and ParMesh::MakeSimplicial).
   void MakeSimplicial_(const Mesh &orig_mesh, int *vglobal);

public:

   Mesh() { SetEmpty(); }

   /** Copy constructor. Performs a deep copy of (almost) all data, so that the
       source mesh can be modified (e.g. deleted, refined) without affecting the
       new mesh. If 'copy_nodes' is false, use a shallow (pointer) copy for the
       nodes, if present. */
   explicit Mesh(const Mesh &mesh, bool copy_nodes = true);

   /// Move constructor, useful for using a Mesh as a function return value.
   Mesh(Mesh &&mesh);

   /// Move assignment operstor.
   Mesh& operator=(Mesh &&mesh);

   /// Explicitly delete the copy assignment operator.
   Mesh& operator=(const Mesh &mesh) = delete;

   /** @name Named mesh constructors.

       Each of these constructors uses the move constructor, and can be used as
       the right-hand side of an assignment when creating new meshes. */
   ///@{

   /** Creates mesh by reading a file in MFEM, Netgen, or VTK format. If
       generate_edges = 0 (default) edges are not generated, if 1 edges are
       generated.

       @note @a filename is not cached by the Mesh object and can be
       safely deleted following this function call.
   */
   static Mesh LoadFromFile(const char *filename,
                            int generate_edges = 0, int refine = 1,
                            bool fix_orientation = true);

   /** Creates 1D mesh , divided into n equal intervals. */
   static Mesh MakeCartesian1D(int n, double sx = 1.0);

   /** Creates mesh for the rectangle [0,sx]x[0,sy], divided into nx*ny
       quadrilaterals if type = QUADRILATERAL or into 2*nx*ny triangles if
       type = TRIANGLE. If generate_edges = 0 (default) edges are not generated,
       if 1 edges are generated. If scf_ordering = true (default), elements are
       ordered along a space-filling curve, instead of row by row. */
   static Mesh MakeCartesian2D(
      int nx, int ny, Element::Type type, bool generate_edges = false,
      double sx = 1.0, double sy = 1.0, bool sfc_ordering = true);

   /** Creates mesh for the parallelepiped [0,sx]x[0,sy]x[0,sz], divided into
       nx*ny*nz hexahedra if type=HEXAHEDRON or into 6*nx*ny*nz tetrahedrons if
       type=TETRAHEDRON. If sfc_ordering = true (default), elements are ordered
       along a space-filling curve, instead of row by row and layer by layer. */
   static Mesh MakeCartesian3D(
      int nx, int ny, int nz, Element::Type type,
      double sx = 1.0, double sy = 1.0, double sz = 1.0,
      bool sfc_ordering = true);

   /// Create a refined (by any factor) version of @a orig_mesh.
   /** @param[in] orig_mesh  The starting coarse mesh.
       @param[in] ref_factor The refinement factor, an integer > 1.
       @param[in] ref_type   Specify the positions of the new vertices. The
                             options are BasisType::ClosedUniform or
                             BasisType::GaussLobatto.

       The refinement data which can be accessed with GetRefinementTransforms()
       is set to reflect the performed refinements.

       @note The constructed Mesh is straight-sided. */
   static Mesh MakeRefined(Mesh &orig_mesh, int ref_factor, int ref_type);

   /// Create a refined mesh, where each element of the original mesh may be
   /// refined by a different factor.
   /** @param[in] orig_mesh   The starting coarse mesh.
       @param[in] ref_factors An array of integers whose size is the number of
                              elements of @a orig_mesh. The @a ith element of
                              @a orig_mesh is refined by refinement factor
                              @a ref_factors[i].
       @param[in] ref_type    Specify the positions of the new vertices. The
                              options are BasisType::ClosedUniform or
                              BasisType::GaussLobatto.

       The refinement data which can be accessed with GetRefinementTransforms()
       is set to reflect the performed refinements.

       @note The constructed Mesh is straight-sided. */
   /// refined @a ref_factors[i] times in each dimension.
   static Mesh MakeRefined(Mesh &orig_mesh, const Array<int> &ref_factors,
                           int ref_type);

   /** Create a mesh by splitting each element of @a orig_mesh into simplices.
       Quadrilaterals are split into two triangles, prisms are split into
       3 tetrahedra, and hexahedra are split into either 5 or 6 tetrahedra
       depending on the configuration.
       @warning The curvature of the original mesh is not carried over to the
       new mesh. Periodic meshes are not supported. */
   static Mesh MakeSimplicial(const Mesh &orig_mesh);

   /// Create a periodic mesh by identifying vertices of @a orig_mesh.
   /** Each vertex @a i will be mapped to vertex @a v2v[i], such that all
       vertices that are coincident under the periodic mapping get mapped to
       the same index. The mapping @a v2v can be generated from translation
       vectors using Mesh::CreatePeriodicVertexMapping.
       @note MFEM requires that each edge of the resulting mesh be uniquely
       identifiable by a pair of distinct vertices. As a consequence, periodic
       boundaries must be connected by at least three edges. */
   static Mesh MakePeriodic(const Mesh &orig_mesh, const std::vector<int> &v2v);

   ///@}

   /// @brief Creates a mapping @a v2v from the vertex indices of the mesh such
   /// that coincident vertices under the given @a translations are identified.
   /** Each Vector in @a translations should be of size @a sdim (the spatial
       dimension of the mesh). Two vertices are considered coincident if the
       translated coordinates of one vertex are within the given tolerance (@a
       tol, relative to the mesh diameter) of the coordinates of the other
       vertex.
       @warning This algorithm does not scale well with the number of boundary
       vertices in the mesh, and may run slowly on very large meshes. */
   std::vector<int> CreatePeriodicVertexMapping(
      const std::vector<Vector> &translations, double tol = 1e-8) const;

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
        int dimension, int space_dimension = -1);

   /** @anchor mfem_Mesh_init_ctor
       @brief _Init_ constructor: begin the construction of a Mesh object. */
   Mesh(int Dim_, int NVert, int NElem, int NBdrElem = 0, int spaceDim_ = -1)
   {
      if (spaceDim_ == -1) { spaceDim_ = Dim_; }
      InitMesh(Dim_, spaceDim_, NVert, NElem, NBdrElem);
   }

   /** @name Methods for Mesh construction.

       These methods are intended to be used with the @ref mfem_Mesh_init_ctor
       "init constructor". */
   ///@{

   /// @note The returned object should be deleted by the caller.
   Element *NewElement(int geom);

   int AddVertex(double x, double y = 0.0, double z = 0.0);
   int AddVertex(const double *coords);
   int AddVertex(const Vector &coords);
   /// Mark vertex @a i as nonconforming, with parent vertices @a p1 and @a p2.
   void AddVertexParents(int i, int p1, int p2);

   int AddSegment(int v1, int v2, int attr = 1);
   int AddSegment(const int *vi, int attr = 1);

   int AddTriangle(int v1, int v2, int v3, int attr = 1);
   int AddTriangle(const int *vi, int attr = 1);
   int AddTri(const int *vi, int attr = 1) { return AddTriangle(vi, attr); }

   int AddQuad(int v1, int v2, int v3, int v4, int attr = 1);
   int AddQuad(const int *vi, int attr = 1);

   int AddTet(int v1, int v2, int v3, int v4, int attr = 1);
   int AddTet(const int *vi, int attr = 1);

   int AddWedge(int v1, int v2, int v3, int v4, int v5, int v6, int attr = 1);
   int AddWedge(const int *vi, int attr = 1);

   int AddPyramid(int v1, int v2, int v3, int v4, int v5, int attr = 1);
   int AddPyramid(const int *vi, int attr = 1);

   int AddHex(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8,
              int attr = 1);
   int AddHex(const int *vi, int attr = 1);
   void AddHexAsTets(const int *vi, int attr = 1);
   void AddHexAsWedges(const int *vi, int attr = 1);
   void AddHexAsPyramids(const int *vi, int attr = 1);

   /// The parameter @a elem should be allocated using the NewElement() method
   /// @note Ownership of @a elem will pass to the Mesh object
   int AddElement(Element *elem);
   /// The parameter @a elem should be allocated using the NewElement() method
   /// @note Ownership of @a elem will pass to the Mesh object
   int AddBdrElement(Element *elem);

   int AddBdrSegment(int v1, int v2, int attr = 1);
   int AddBdrSegment(const int *vi, int attr = 1);

   int AddBdrTriangle(int v1, int v2, int v3, int attr = 1);
   int AddBdrTriangle(const int *vi, int attr = 1);

   int AddBdrQuad(int v1, int v2, int v3, int v4, int attr = 1);
   int AddBdrQuad(const int *vi, int attr = 1);
   void AddBdrQuadAsTriangles(const int *vi, int attr = 1);

   int AddBdrPoint(int v, int attr = 1);

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
   /// Finalize the construction of a wedge Mesh.
   void FinalizeWedgeMesh(int generate_edges = 0, int refine = 0,
                          bool fix_orientation = true);
   /// Finalize the construction of a hexahedral Mesh.
   void FinalizeHexMesh(int generate_edges = 0, int refine = 0,
                        bool fix_orientation = true);
   /// Finalize the construction of any type of Mesh.
   /** This method calls FinalizeTopology() and Finalize(). */
   void FinalizeMesh(int refine = 0, bool fix_orientation = true);

   ///@}

   /** @brief Finalize the construction of the secondary topology (connectivity)
       data of a Mesh. */
   /** This method does not require any actual coordinate data (either vertex
       coordinates for linear meshes or node coordinates for meshes with nodes)
       to be available. However, the data generated by this method is generally
       required by the FiniteElementSpace class.

       After calling this method, setting the Mesh vertices or nodes, it may be
       appropriate to call the method Finalize(). */
   void FinalizeTopology(bool generate_bdr = true);

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
   virtual void Finalize(bool refine = false, bool fix_orientation = false);

   virtual void SetAttributes();

   /** This is our integration with the Gecko library. The method finds an
       element ordering that will increase memory coherency by putting elements
       that are in physical proximity closer in memory. It can also be used to
       obtain a space-filling curve ordering for ParNCMesh partitioning.
       @param[out] ordering Output element ordering.
       @param iterations Total number of V cycles. The ordering may improve with
       more iterations. The best iteration is returned at the end.
       @param window Initial window size. This determines the number of
       permutations tested at each multigrid level and strongly influences the
       quality of the result, but the cost of increasing 'window' is exponential.
       @param period The window size is incremented every 'period' iterations.
       @param seed Seed for initial random ordering (0 = skip random reorder).
       @param verbose Print the progress of the optimization to mfem::out.
       @param time_limit Optional time limit for the optimization, in seconds.
       When reached, ordering from the best iteration so far is returned
       (0 = no limit).
       @return The final edge product cost of the ordering. The function may be
       called in an external loop with different seeds, and the best ordering can
       then be retained. */
   double GetGeckoElementOrdering(Array<int> &ordering,
                                  int iterations = 4, int window = 4,
                                  int period = 2, int seed = 0,
                                  bool verbose = false, double time_limit = 0);

   /** Return an ordering of the elements that approximately follows the Hilbert
       curve. The method performs a spatial (Hilbert) sort on the centers of all
       elements and returns the resulting sequence, which can then be passed to
       ReorderElements. This is a cheap alternative to GetGeckoElementOrdering.*/
   void GetHilbertElementOrdering(Array<int> &ordering);

   /** Rebuilds the mesh with a different order of elements. For each element i,
       the array ordering[i] contains its desired new index. Note that the method
       reorders vertices, edges and faces along with the elements. */
   void ReorderElements(const Array<int> &ordering, bool reorder_vertices = true);

   /// Deprecated: see @a MakeCartesian3D.
   MFEM_DEPRECATED
   Mesh(int nx, int ny, int nz, Element::Type type, bool generate_edges = false,
        double sx = 1.0, double sy = 1.0, double sz = 1.0,
        bool sfc_ordering = true)
   {
      Make3D(nx, ny, nz, type, sx, sy, sz, sfc_ordering);
      Finalize(true); // refine = true
   }

   /// Deprecated: see @a MakeCartesian2D.
   MFEM_DEPRECATED
   Mesh(int nx, int ny, Element::Type type, bool generate_edges = false,
        double sx = 1.0, double sy = 1.0, bool sfc_ordering = true)
   {
      Make2D(nx, ny, type, sx, sy, generate_edges, sfc_ordering);
      Finalize(true); // refine = true
   }

   /// Deprecated: see @a MakeCartesian1D.
   MFEM_DEPRECATED
   explicit Mesh(int n, double sx = 1.0)
   {
      Make1D(n, sx);
      // Finalize(); // reminder: not needed
   }

   /** Creates mesh by reading a file in MFEM, Netgen, or VTK format. If
       generate_edges = 0 (default) edges are not generated, if 1 edges are
       generated. See also @a Mesh::LoadFromFile. */
   explicit Mesh(const char *filename, int generate_edges = 0, int refine = 1,
                 bool fix_orientation = true);

   /** Creates mesh by reading data stream in MFEM, Netgen, or VTK format. If
       generate_edges = 0 (default) edges are not generated, if 1 edges are
       generated. */
   explicit Mesh(std::istream &input, int generate_edges = 0, int refine = 1,
                 bool fix_orientation = true);

   /// Create a disjoint mesh from the given mesh array
   ///
   /// @note Data is copied from the meshes in @a mesh_array.
   Mesh(Mesh *mesh_array[], int num_pieces);

   /// Deprecated: see @a MakeRefined.
   MFEM_DEPRECATED
   Mesh(Mesh *orig_mesh, int ref_factor, int ref_type);

   /** This is similar to the mesh constructor with the same arguments, but here
       the current mesh is destroyed and another one created based on the data
       stream again given in MFEM, Netgen, or VTK format. If generate_edges = 0
       (default) edges are not generated, if 1 edges are generated. */
   /// \see mfem::ifgzstream() for on-the-fly decompression of compressed ascii
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

       The purpose of this is to be able to quickly tell what type of elements
       one has in the mesh. Examination of this bitmask along with knowledge
       of the mesh dimension can be used to identify which element types are
       present.

       @return A bitmask:
       - bit 0 - simplices are present in the mesh (triangles, tets),
       - bit 1 - tensor product elements are present in the mesh (quads, hexes),
       - bit 2 - the mesh has wedge elements.
       - bit 3 - the mesh has pyramid elements.

       In parallel, the result takes into account elements on all processors.
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

   /// Return the number of faces (3D), edges (2D) or vertices (1D).
   int GetNumFaces() const;

   /** @brief Return the number of faces (3D), edges (2D) or vertices (1D)
       including ghost faces. */
   int GetNumFacesWithGhost() const;

   /** @brief Returns the number of faces according to the requested type, does
       not count master nonconforming faces.

       If type==Boundary returns only the number of true boundary faces
       contrary to GetNBE() that returns all "boundary" elements which may
       include actual interior faces.
       Similarly, if type==Interior, only the true interior faces are counted
       excluding all master nonconforming faces. */
   virtual int GetNFbyType(FaceType type) const;

   /// Utility function: sum integers from all processors (Allreduce).
   virtual long long ReduceInt(int value) const { return value; }

   /// Return the total (global) number of elements.
   long long GetGlobalNE() const { return ReduceInt(NumOfElements); }

   /** Compute the Jacobian of the transformation from the perfect
       reference element at the given integration point (defaults to the
       center of the element if no integration point is specified) */
   void GetElementJacobian(int i, DenseMatrix &J,
                           const IntegrationPoint *ip = NULL);

   /** Return vertex to vertex table. The connections stored in the table
    are from smaller to bigger vertex index, i.e. if i<j and (i, j) is
    in the table, then (j, i) is not stored. */
   void GetVertexToVertexTable(DSTable &) const;

   /** @brief Return the mesh geometric factors corresponding to the given
       integration rule.

       The IntegrationRule used with GetGeometricFactors needs to remain valid
       until the internally stored GeometricFactors objects are destroyed (by
       calling Mesh::DeleteGeometricFactors(), Mesh::NodesUpdated(), or the Mesh
       destructor).

       If the device MemoryType parameter @a d_mt is specified, then the
       returned object will use that type unless it was previously allocated
       with a different type.

       The returned pointer points to an internal object that may be invalidated
       by mesh operations such as refinement, vertex/node movement, etc. Since
       not all such modifications can be tracked by the Mesh class (e.g. when
       using the pointer returned by GetNodes() to change the nodes) one needs
       to account for such changes by calling the method NodesUpdated() which,
       in particular, will call DeleteGeometricFactors(). */
   const GeometricFactors* GetGeometricFactors(
      const IntegrationRule& ir,
      const int flags,
      MemoryType d_mt = MemoryType::DEFAULT);

   /** @brief Return the mesh geometric factors for the faces corresponding
       to the given integration rule.

       The IntegrationRule used with GetFaceGeometricFactors needs to remain
       valid until the internally stored FaceGeometricFactors objects are
       destroyed (by either calling Mesh::DeleteGeometricFactors(),
       Mesh::NodesUpdated(), or the Mesh destructor).

       If the device MemoryType parameter @a d_mt is specified, then the
       returned object will use that type unless it was previously allocated
       with a different type.

       The returned pointer points to an internal object that may be invalidated
       by mesh operations such as refinement, vertex/node movement, etc. Since
       not all such modifications can be tracked by the Mesh class (e.g. when
       using the pointer returned by GetNodes() to change the nodes) one needs
       to account for such changes by calling the method NodesUpdated() which,
       in particular, will call DeleteGeometricFactors(). */
   const FaceGeometricFactors* GetFaceGeometricFactors(
      const IntegrationRule& ir,
      const int flags,
      FaceType type,
      MemoryType d_mt = MemoryType::DEFAULT);

   /// Destroy all GeometricFactors stored by the Mesh.
   /** This method can be used to force recomputation of the GeometricFactors,
       for example, after the mesh nodes are modified externally.

       @note In general, the preferred method for resetting the GeometricFactors
       should be to call NodesUpdated(). */
   void DeleteGeometricFactors();

   /** @brief This function should be called after the mesh node coordinates
       have been updated externally, e.g. by modifying the internal nodal
       GridFunction returned by GetNodes(). */
   /** It deletes internal quantities derived from the node coordinates,
       such as the (Face)GeometricFactors.

       @note Unlike the similarly named protected method UpdateNodes() this
       method does not modify the nodes. */
   void NodesUpdated() { DeleteGeometricFactors(); }

   /// Equals 1 + num_holes - num_loops
   inline int EulerNumber() const
   { return NumOfVertices - NumOfEdges + NumOfFaces - NumOfElements; }
   /// Equals 1 - num_holes
   inline int EulerNumber2D() const
   { return NumOfVertices - NumOfEdges + NumOfElements; }

   int Dimension() const { return Dim; }
   int SpaceDimension() const { return spaceDim; }

   /// @brief Return pointer to vertex i's coordinates.
   /// @warning For high-order meshes (when #Nodes != NULL) vertices may not be
   /// updated and should not be used!
   const double *GetVertex(int i) const { return vertices[i](); }

   /// @brief Return pointer to vertex i's coordinates.
   ///
   /// @warning For high-order meshes (when Nodes != NULL) vertices may not
   /// being updated and should not be used!
   ///
   /// @note The pointer returned by this function can be used to
   /// alter vertex locations but the pointer itself should not be
   /// changed by the caller.
   double *GetVertex(int i) { return vertices[i](); }

   void GetElementData(int geom, Array<int> &elem_vtx, Array<int> &attr) const
   { GetElementData(elements, geom, elem_vtx, attr); }

   /// Checks if the mesh has boundary elements
   virtual bool HasBoundaryElements() const { return (NumOfBdrElements > 0); }

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

   /// @brief Return pointer to the i'th element object
   ///
   /// The index @a i should be in the range [0, this->Mesh::GetNE())
   ///
   /// In parallel, @a i is the local element index which is in the
   /// same range mentioned above.
   const Element *GetElement(int i) const { return elements[i]; }

   /// @brief Return pointer to the i'th element object
   ///
   /// @note Provides read/write access to the i'th element object so
   /// that element attributes or connectivity can be adjusted. However,
   /// the Element object itself should not be deleted by the caller.
   Element *GetElement(int i) { return elements[i]; }

   /// @brief Return pointer to the i'th boundary element object
   ///
   /// The index @a i should be in the range [0, this->Mesh::GetNBE())
   ///
   /// In parallel, @a i is the local boundary element index which is
   /// in the same range mentioned above.
   const Element *GetBdrElement(int i) const { return boundary[i]; }

   /// @brief Return pointer to the i'th boundary element object
   ///
   /// @note Provides read/write access to the i'th boundary element object so
   /// that boundary attributes or connectivity can be adjusted. However,
   /// the Element object itself should not be deleted by the caller.
   Element *GetBdrElement(int i) { return boundary[i]; }

   const Element *GetFace(int i) const { return faces[i]; }

   /// Return the Geometry::Type associated with face @a i.
   Geometry::Type GetFaceGeometry(int i) const;

   Geometry::Type GetElementGeometry(int i) const
   {
      return elements[i]->GetGeometryType();
   }

   Geometry::Type GetBdrElementGeometry(int i) const
   {
      return boundary[i]->GetGeometryType();
   }

   /// Deprecated in favor of Mesh::GetFaceGeometry
   MFEM_DEPRECATED Geometry::Type GetFaceBaseGeometry(int i) const
   { return GetFaceGeometry(i); }

   Geometry::Type GetElementBaseGeometry(int i) const
   { return GetElementGeometry(i); }

   Geometry::Type GetBdrElementBaseGeometry(int i) const
   { return GetBdrElementGeometry(i); }

   /** @brief Return true iff the given @a geom is encountered in the mesh.
       Geometries of dimensions lower than Dimension() are counted as well. */
   bool HasGeometry(Geometry::Type geom) const
   { return mesh_geoms & (1 << geom); }

   /** @brief Return the number of geometries of the given dimension present in
       the mesh. */
   /** For a parallel mesh only the local geometries are counted. */
   int GetNumGeometries(int dim) const;

   /// Return all element geometries of the given dimension present in the mesh.
   /** For a parallel mesh only the local geometries are returned.

       The returned geometries are sorted. */
   void GetGeometries(int dim, Array<Geometry::Type> &el_geoms) const;

   /// List of mesh geometries stored as Array<Geometry::Type>.
   class GeometryList : public Array<Geometry::Type>
   {
   protected:
      Geometry::Type geom_buf[Geometry::NumGeom];
   public:
      /// Construct a GeometryList of all element geometries in @a mesh.
      GeometryList(Mesh &mesh)
         : Array<Geometry::Type>(geom_buf, Geometry::NumGeom)
      { mesh.GetGeometries(mesh.Dimension(), *this); }
      /** @brief Construct a GeometryList of all geometries of dimension @a dim
          in @a mesh. */
      GeometryList(Mesh &mesh, int dim)
         : Array<Geometry::Type>(geom_buf, Geometry::NumGeom)
      { mesh.GetGeometries(dim, *this); }
   };

   /// Returns the indices of the vertices of element i.
   void GetElementVertices(int i, Array<int> &v) const
   { elements[i]->GetVertices(v); }

   /// Returns the indices of the vertices of boundary element i.
   void GetBdrElementVertices(int i, Array<int> &v) const
   { boundary[i]->GetVertices(v); }

   /// Return the indices and the orientations of all edges of element i.
   void GetElementEdges(int i, Array<int> &edges, Array<int> &cor) const;

   /// Return the indices and the orientations of all edges of bdr element i.
   void GetBdrElementEdges(int i, Array<int> &edges, Array<int> &cor) const;

   /** Return the indices and the orientations of all edges of face i.
       Works for both 2D (face=edge) and 3D faces. */
   void GetFaceEdges(int i, Array<int> &edges, Array<int> &o) const;

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

   /// Returns the face-to-edge Table (3D)
   ///
   /// @note The returned object should NOT be deleted by the caller.
   Table *GetFaceEdgeTable() const;

   /// Returns the edge-to-vertex Table (3D)
   ///
   /// @note The returned object should NOT be deleted by the caller.
   Table *GetEdgeVertexTable() const;

   /// Return the indices and the orientations of all faces of element i.
   void GetElementFaces(int i, Array<int> &faces, Array<int> &ori) const;

   /** @brief Returns the sorted, unique indices of elements sharing a face with
       element @a elem, including @a elem. */
   Array<int> FindFaceNeighbors(const int elem) const;

   /// Return the index and the orientation of the face of bdr element i. (3D)
   void GetBdrElementFace(int i, int *f, int *o) const;

   /** Return the vertex index of boundary element i. (1D)
       Return the edge index of boundary element i. (2D)
       Return the face index of boundary element i. (3D) */
   int GetBdrElementEdgeIndex(int i) const;

   /** @brief For the given boundary element, bdr_el, return its adjacent
       element and its info, i.e. 64*local_bdr_index+bdr_orientation.

       The returned bdr_orientation is that of the boundary element relative to
       the respective face element.

       @sa GetBdrElementAdjacentElement2() */
   void GetBdrElementAdjacentElement(int bdr_el, int &el, int &info) const;

   /** @brief For the given boundary element, bdr_el, return its adjacent
       element and its info, i.e. 64*local_bdr_index+inverse_bdr_orientation.

       The returned inverse_bdr_orientation is the inverse of the orientation of
       the boundary element relative to the respective face element. In other
       words this is the orientation of the face element relative to the
       boundary element.

       @sa GetBdrElementAdjacentElement() */
   void GetBdrElementAdjacentElement2(int bdr_el, int &el, int &info) const;

   /// Returns the type of element i.
   Element::Type GetElementType(int i) const;

   /// Returns the type of boundary element i.
   Element::Type GetBdrElementType(int i) const;

   /* Return point matrix of element i of dimension Dim X #v, where for every
      vertex we give its coordinates in space of dimension Dim. */
   void GetPointMatrix(int i, DenseMatrix &pointmat) const;

   /* Return point matrix of boundary element i of dimension Dim X #v, where for
      every vertex we give its coordinates in space of dimension Dim. */
   void GetBdrPointMatrix(int i, DenseMatrix &pointmat) const;

   /// @brief Return FiniteElement for reference element of the specified type
   ///
   /// @note The returned object is a pointer to a global object and
   /// should not be deleted by the caller.
   static FiniteElement *GetTransformationFEforElementType(Element::Type);

   /// Builds the transformation defining the i-th element in @a ElTr.
   /// @a ElTr must be allocated in advance and will be owned by the caller.
   void GetElementTransformation(int i, IsoparametricTransformation *ElTr);

   /// Returns a pointer to the transformation defining the i-th element.
   ///
   /// @note The returned object is owned by the class and is shared, i.e.,
   /// calling this function resets pointers obtained from previous calls.
   /// Also, this pointer should NOT be deleted by the caller.
   ElementTransformation *GetElementTransformation(int i);

   /// Builds the transformation defining the i-th element in @a ElTr
   /// assuming position of the vertices/nodes are given by @a nodes.
   /// @a ElTr must be allocated in advance and will be owned by the caller.
   void GetElementTransformation(int i, const Vector &nodes,
                                 IsoparametricTransformation *ElTr);

   /// Returns a pointer to the transformation defining the i-th boundary
   /// element.
   /// @note The returned object is owned by the class and is shared, i.e.,
   /// calling this function resets pointers obtained from previous calls.
   /// Also, the returned object should NOT be deleted by the caller.
   ElementTransformation *GetBdrElementTransformation(int i);

   /// Builds the transformation defining the i-th boundary element in @a ElTr.
   /// @a ElTr must be allocated in advance and will be owned by the caller.
   void GetBdrElementTransformation(int i, IsoparametricTransformation *ElTr);

   /// Builds the transformation defining the i-th face element in @a FTr.
   /// @a FTr must be allocated in advance and will be owned by the caller.
   void GetFaceTransformation(int i, IsoparametricTransformation *FTr);

   /** @brief A helper method that constructs a transformation from the
       reference space of a face to the reference space of an element. */
   /** The local index of the face as a face in the element and its orientation
       are given by the input parameter @a info, as @a info = 64*loc_face_idx +
       loc_face_orientation. */
   void GetLocalFaceTransformation(int face_type, int elem_type,
                                   IsoparametricTransformation &Transf,
                                   int info);

   /// Returns a pointer to the transformation defining the given face element.
   /// @note The returned object is owned by the class and is shared, i.e.,
   /// calling this function resets pointers obtained from previous calls.
   /// Also, the returned object should NOT be deleted by the caller.
   ElementTransformation *GetFaceTransformation(int FaceNo);

   /// Builds the transformation defining the i-th edge element in @a EdTr.
   /// @a EdTr must be allocated in advance and will be owned by the caller.
   void GetEdgeTransformation(int i, IsoparametricTransformation *EdTr);

   /// Returns a pointer to the transformation defining the given edge element.
   /// @note The returned object is owned by the class and is shared, i.e.,
   /// calling this function resets pointers obtained from previous calls.
   /// Also, the returned object should NOT be deleted by the caller.
   ElementTransformation *GetEdgeTransformation(int EdgeNo);

   /// Returns (a pointer to an object containing) the following data:
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
   /// These mask values are defined in the ConfigMasks enum type as part of the
   /// FaceElementTransformations class in fem/eltrans.hpp.
   ///
   /// @note The returned object is owned by the class and is shared, i.e.,
   /// calling this function resets pointers obtained from previous calls.
   /// Also, this pointer should NOT be deleted by the caller.
   virtual FaceElementTransformations *GetFaceElementTransformations(
      int FaceNo,
      int mask = 31);

   /// See GetFaceElementTransformations().
   /// @note The returned object should NOT be deleted by the caller.
   FaceElementTransformations *GetInteriorFaceTransformations (int FaceNo)
   {
      if (faces_info[FaceNo].Elem2No < 0) { return NULL; }
      return GetFaceElementTransformations (FaceNo);
   }

   /// Builds the transformation defining the given boundary face.
   /// @note The returned object should NOT be deleted by the caller.
   FaceElementTransformations *GetBdrFaceTransformations (int BdrElemNo);

   /// Return the local face index for the given boundary face.
   int GetBdrFace(int BdrElemNo) const;

   /// Return true if the given face is interior. @sa FaceIsTrueInterior().
   bool FaceIsInterior(int FaceNo) const
   {
      return (faces_info[FaceNo].Elem2No >= 0);
   }

   /** This enumerated type describes the three main face topologies:
       - Boundary, for faces on the boundary of the computational domain,
       - Conforming, for conforming faces interior to the computational domain,
       - Nonconforming, for nonconforming faces interior to the computational
         domain. */
   enum class FaceTopology { Boundary,
                             Conforming,
                             Nonconforming,
                             NA
                           };

   /** This enumerated type describes the location of the two elements sharing a
       face, Local meaning that the element is local to the MPI rank, FaceNbr
       meaning that the element is distributed on a different MPI rank, this
       typically means that methods with FaceNbr should be used to access the
       relevant information, e.g., ParFiniteElementSpace::GetFaceNbrElementVDofs.
    */
   enum class ElementLocation { Local, FaceNbr, NA };

   /** This enumerated type describes the topological relation of an element to
       a face:
       - Coincident meaning that the element's face is topologically equal to
         the mesh face.
       - Superset meaning that the element's face is topologically coarser than
         the mesh face, i.e., the element's face contains the mesh face.
       - Subset meaning that the element's face is topologically finer than the
         mesh face, i.e., the element's face is contained in the mesh face.
       Superset and Subset are only relevant for nonconforming faces.
       Master nonconforming faces have a conforming element on one side, and a
       fine element on the other side. Slave nonconforming faces have a
       conforming element on one side, and a coarse element on the other side.
    */
   enum class ElementConformity { Coincident, Superset, Subset, NA };

   /** This enumerated type describes the corresponding FaceInfo internal
       representation (encoded cases), c.f. FaceInfo's documentation:
       Classification of a local (non-ghost) face based on its FaceInfo:
         - Elem2No >= 0 --> local interior face; can be either:
            - NCFace == -1 --> LocalConforming,
            - NCFace >= 0 --> LocalSlaveNonconforming,
         - Elem2No < 0 --> local "boundary" face; can be one of:
            - NCFace == -1 --> conforming face; can be either:
               - Elem2Inf < 0 --> Boundary,
               - Elem2Inf >= 0 --> SharedConforming,
            - NCFace >= 0 --> nonconforming face; can be one of:
               - Elem2Inf < 0 --> MasterNonconforming (shared or not shared),
               - Elem2Inf >= 0 --> SharedSlaveNonconforming.
       Classification of a ghost (non-local) face based on its FaceInfo:
         - Elem1No == -1 --> GhostMaster (includes other unused ghost faces),
         - Elem1No >= 0 --> GhostSlave.
    */
   enum class FaceInfoTag { Boundary,
                            LocalConforming,
                            LocalSlaveNonconforming,
                            SharedConforming,
                            SharedSlaveNonconforming,
                            MasterNonconforming,
                            GhostSlave,
                            GhostMaster
                          };

   /** @brief This structure is used as a human readable output format that
       decipheres the information contained in Mesh::FaceInfo when using the
       Mesh::GetFaceInformation() method.

       The element indices in this structure don't need further processing,
       contrary to the ones obtained through Mesh::GetFacesElements and can
       directly be used, e.g., Elem1 and Elem2 indices.
       Likewise the orientations for Elem1 and Elem2 already take into account
       special cases and can be used as is.
   */
   struct FaceInformation
   {
      FaceTopology topology;

      struct
      {
         ElementLocation location;
         ElementConformity conformity;
         int index;
         int local_face_id;
         int orientation;
      } element[2];

      FaceInfoTag tag;
      int ncface;
      const DenseMatrix* point_matrix;

      /** @brief Return true if the face is a local interior face which is NOT
          a master nonconforming face. */
      bool IsLocal() const
      {
         return element[1].location == Mesh::ElementLocation::Local;
      }

      /** @brief Return true if the face is a shared interior face which is NOT
          a master nonconforming face. */
      bool IsShared() const
      {
         return element[1].location == Mesh::ElementLocation::FaceNbr;
      }

      /** @brief return true if the face is an interior face to the computation
          domain, either a local or shared interior face (not a boundary face)
          which is NOT a master nonconforming face.
       */
      bool IsInterior() const
      {
         return topology == FaceTopology::Conforming ||
                topology == FaceTopology::Nonconforming;
      }

      /** @brief Return true if the face is a boundary face. */
      bool IsBoundary() const
      {
         return topology == FaceTopology::Boundary;
      }

      /// @brief Return true if the face is of the same type as @a type.
      bool IsOfFaceType(FaceType type) const
      {
         switch (type)
         {
            case FaceType::Interior:
               return IsInterior();
            case FaceType::Boundary:
               return IsBoundary();
            default:
               return false;
         }
      }

      /// @brief Return true if the face is a conforming face.
      bool IsConforming() const
      {
         return topology == FaceTopology::Conforming;
      }

      /// @brief Return true if the face is a nonconforming fine face.
      bool IsNonconformingFine() const
      {
         return topology == FaceTopology::Nonconforming &&
                (element[0].conformity == ElementConformity::Superset ||
                 element[1].conformity == ElementConformity::Superset);
      }

      /// @brief Return true if the face is a nonconforming coarse face.
      /** Note that ghost nonconforming master faces cannot be clearly
          identified as such with the currently available information, so this
          method will return false for such faces. */
      bool IsNonconformingCoarse() const
      {
         return topology == FaceTopology::Nonconforming &&
                element[1].conformity == ElementConformity::Subset;
      }

      /// @brief cast operator from FaceInformation to FaceInfo.
      operator Mesh::FaceInfo() const;
   };

   /** This method aims to provide face information in a deciphered format, i.e.
       Mesh::FaceInformation, compared to the raw encoded information returned
       by Mesh::GetFaceElements() and Mesh::GetFaceInfos(). */
   FaceInformation GetFaceInformation(int f) const;

   void GetFaceElements (int Face, int *Elem1, int *Elem2) const;
   void GetFaceInfos (int Face, int *Inf1, int *Inf2) const;
   void GetFaceInfos (int Face, int *Inf1, int *Inf2, int *NCFace) const;

   /// Deprecated in favor of Mesh::GetFaceGeometry
   MFEM_DEPRECATED Geometry::Type GetFaceGeometryType(int Face) const
   { return GetFaceGeometry(Face); }

   Element::Type  GetFaceElementType(int Face) const;

   Array<int> GetFaceToBdrElMap() const;

   /// Check (and optionally attempt to fix) the orientation of the elements
   /** @param[in] fix_it  If `true`, attempt to fix the orientations of some
                          elements: triangles, quads, and tets.
       @return The number of elements with wrong orientation.

       @note For meshes with nodes (e.g. high-order or periodic meshes), fixing
       the element orientations may require additional permutation of the nodal
       GridFunction of the mesh which is not performed by this method. Instead,
       the method Finalize() should be used with the parameter
       @a fix_orientation set to `true`.

       @note This method performs a simple check if an element is inverted, e.g.
       for most elements types, it checks if the Jacobian of the mapping from
       the reference element is non-negative at the center of the element. */
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

   /// Set the attribute of boundary element i.
   void SetBdrAttribute(int i, int attr) { boundary[i]->SetAttribute(attr); }

   /// Return the attribute of patch i, for a NURBS mesh.
   int GetPatchAttribute(int i) const;

   /// Set the attribute of patch i, for a NURBS mesh.
   void SetPatchAttribute(int i, int attr);

   /// Return the attribute of patch boundary element i, for a NURBS mesh.
   int GetPatchBdrAttribute(int i) const;

   /// Set the attribute of patch boundary element i, for a NURBS mesh.
   void SetPatchBdrAttribute(int i, int attr);

   const Table &ElementToElementTable();

   const Table &ElementToFaceTable() const;

   const Table &ElementToEdgeTable() const;

   ///  The returned Table should be deleted by the caller
   Table *GetVertexToElementTable();

   /** Return the "face"-element Table. Here "face" refers to face (3D),
       edge (2D), or vertex (1D).
       The returned Table should be deleted by the caller. */
   Table *GetFaceToElementTable() const;

   /** This method modifies a tetrahedral mesh so that Nedelec spaces of order
       greater than 1 can be defined on the mesh. Specifically, we
       1) rotate all tets in the mesh so that the vertices {v0, v1, v2, v3}
       satisfy: v0 < v1 < min(v2, v3).
       2) rotate all boundary triangles so that the vertices {v0, v1, v2}
       satisfy: v0 < min(v1, v2).

       @note Refinement does not work after a call to this method! */
   MFEM_DEPRECATED virtual void ReorientTetMesh();

   /// @note The returned array should be deleted by the caller.
   int *CartesianPartitioning(int nxyz[]);
   /// @note The returned array should be deleted by the caller.
   int *GeneratePartitioning(int nparts, int part_method = 1);
   void CheckPartitioning(int *partitioning_);

   void CheckDisplacements(const Vector &displacements, double &tmax);

   // Vertices are only at the corners of elements, where you would expect them
   // in the lowest-order mesh.
   void MoveVertices(const Vector &displacements);
   void GetVertices(Vector &vert_coord) const;
   void SetVertices(const Vector &vert_coord);

   // Nodes are only active for higher order meshes, and share locations with
   // the vertices, plus all the higher- order control points within the element
   // and along the edges and on the faces.
   void GetNode(int i, double *coord) const;
   void SetNode(int i, const double *coord);

   // Node operations for curved mesh.
   // They call the corresponding '...Vertices' method if the
   // mesh is not curved (i.e. Nodes == NULL).
   void MoveNodes(const Vector &displacements);
   void GetNodes(Vector &node_coord) const;
   /// Updates the vertex/node locations. Invokes NodesUpdated().
   void SetNodes(const Vector &node_coord);

   /// @brief Return a pointer to the internal node GridFunction (may be NULL).
   ///
   /// If the mesh is straight-sided (low-order), it may not have a GridFunction
   /// for the nodes, in which case this function returns NULL. To ensure that
   /// the nodal GridFunction exists, first call EnsureNodes().
   /// @sa SetCurvature().
   ///
   /// @note The returned object should NOT be deleted by the caller.
   GridFunction *GetNodes() { return Nodes; }
   const GridFunction *GetNodes() const { return Nodes; }
   /// Return the mesh nodes ownership flag.
   bool OwnsNodes() const { return own_nodes; }
   /// Set the mesh nodes ownership flag.
   void SetNodesOwner(bool nodes_owner) { own_nodes = nodes_owner; }
   /// Replace the internal node GridFunction with the given GridFunction.
   /** Invokes NodesUpdated(). */
   void NewNodes(GridFunction &nodes, bool make_owner = false);
   /** @brief Swap the internal node GridFunction pointer and ownership flag
       members with the given ones. */
   /** Invokes NodesUpdated(). */
   void SwapNodes(GridFunction *&nodes, int &own_nodes_);

   /// Return the mesh nodes/vertices projected on the given GridFunction.
   void GetNodes(GridFunction &nodes) const;
   /** Replace the internal node GridFunction with a new GridFunction defined
       on the given FiniteElementSpace. The new node coordinates are projected
       (derived) from the current nodes/vertices. */
   virtual void SetNodalFESpace(FiniteElementSpace *nfes);
   /** Replace the internal node GridFunction with the given GridFunction. The
       given GridFunction is updated with node coordinates projected (derived)
       from the current nodes/vertices. */
   void SetNodalGridFunction(GridFunction *nodes, bool make_owner = false);
   /** Return the FiniteElementSpace on which the current mesh nodes are
       defined or NULL if the mesh does not have nodes. */
   const FiniteElementSpace *GetNodalFESpace() const;
   /** @brief Make sure that the mesh has valid nodes, i.e. its geometry is
       described by a vector finite element grid function (even if it is a
       low-order mesh with straight edges).

       @sa GetNodes(). */
   void EnsureNodes();

   /// Set the curvature of the mesh nodes using the given polynomial degree.
   /** Creates a nodal GridFunction if one doesn't already exist.

       @param[in]  order       Polynomial degree of the nodal FE space.
       @param[in]  discont     Whether to use a discontinuous or continuous
                               finite element space (continuous is default).
       @param[in]  space_dim   The space dimension (optional).
       @param[in]  ordering    The Ordering of the finite element space
                               (Ordering::byVDIM is the default). */
   virtual void SetCurvature(int order, bool discont = false, int space_dim = -1,
                             int ordering = 1);

   /// Refine all mesh elements.
   /** @param[in] ref_algo %Refinement algorithm. Currently used only for pure
       tetrahedral meshes. If set to zero (default), a tet mesh will be refined
       using algorithm A, that produces elements with better quality compared to
       algorithm B used when the parameter is non-zero.

       For tetrahedral meshes, after using algorithm A, the mesh cannot be
       refined locally using methods like GeneralRefinement() unless it is
       re-finalized using Finalize() with the parameter @a refine set to true.
       Note that calling Finalize() in this way will generally invalidate any
       FiniteElementSpace%s and GridFunction%s defined on the mesh. */
   void UniformRefinement(int ref_algo = 0);

   /** Refine selected mesh elements. Refinement type can be specified for each
       element. The function can do conforming refinement of triangles and
       tetrahedra and nonconforming refinement (i.e., with hanging-nodes) of
       triangles, quadrilaterals and hexahedra. If 'nonconforming' = -1,
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
   void KnotInsert(Array<Vector *> &kv);
   /* For each knot vector:
         new_degree = max(old_degree, min(old_degree + rel_degree, degree)). */
   void DegreeElevate(int rel_degree, int degree = 16);
   ///@}

   /** Make sure that a quad/hex mesh is considered to be nonconforming (i.e.,
       has an associated NCMesh object). Simplex meshes can be both conforming
       (default) or nonconforming. */
   void EnsureNCMesh(bool simplices_nonconforming = false);

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
   virtual void PrintXG(std::ostream &os = mfem::out) const;

   /// Print the mesh to the given stream using the default MFEM mesh format.
   /// \see mfem::ofgzstream() for on-the-fly compression of ascii outputs
   virtual void Print(std::ostream &os = mfem::out) const { Printer(os); }

   /// Save the mesh to a file using Mesh::Print. The given @a precision will be
   /// used for ASCII output.
   virtual void Save(const char *fname, int precision=16) const;

   /// Print the mesh to the given stream using the adios2 bp format
#ifdef MFEM_USE_ADIOS2
   virtual void Print(adios2stream &os) const;
#endif
   /// Print the mesh in VTK format (linear and quadratic meshes only).
   /// \see mfem::ofgzstream() for on-the-fly compression of ascii outputs
   void PrintVTK(std::ostream &os);
   /** Print the mesh in VTK format. The parameter ref > 0 specifies an element
       subdivision number (useful for high order fields and curved meshes).
       If the optional field_data is set, we also add a FIELD section in the
       beginning of the file with additional dataset information. */
   /// \see mfem::ofgzstream() for on-the-fly compression of ascii outputs
   void PrintVTK(std::ostream &os, int ref, int field_data=0);
   /** Print the mesh in VTU format. The parameter ref > 0 specifies an element
       subdivision number (useful for high order fields and curved meshes).
       If @a bdr_elements is true, then output (only) the boundary elements,
       otherwise output only the non-boundary elements. */
   void PrintVTU(std::ostream &os,
                 int ref=1,
                 VTKFormat format=VTKFormat::ASCII,
                 bool high_order_output=false,
                 int compression_level=0,
                 bool bdr_elements=false);
   /** Print the mesh in VTU format with file name fname. */
   virtual void PrintVTU(std::string fname,
                         VTKFormat format=VTKFormat::ASCII,
                         bool high_order_output=false,
                         int compression_level=0,
                         bool bdr=false);
   /** Print the boundary elements of the mesh in VTU format, and output the
       boundary attributes as a data array (useful for boundary conditions). */
   void PrintBdrVTU(std::string fname,
                    VTKFormat format=VTKFormat::ASCII,
                    bool high_order_output=false,
                    int compression_level=0);

   void GetElementColoring(Array<int> &colors, int el0 = 0);

   /** @brief Prints the mesh with boundary elements given by the boundary of
       the subdomains, so that the boundary of subdomain i has boundary
       attribute i+1. */
   /// \see mfem::ofgzstream() for on-the-fly compression of ascii outputs
   void PrintWithPartitioning (int *partitioning,
                               std::ostream &os, int elem_attr = 0) const;

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

   /** @brief Get the size of the i-th element relative to the perfect
       reference element. */
   double GetElementSize(int i, int type = 0);

   double GetElementSize(int i, const Vector &dir);

   double GetElementVolume(int i);

   void GetElementCenter(int i, Vector &center);

   /// Returns the minimum and maximum corners of the mesh bounding box.
   /** For high-order meshes, the geometry is first refined @a ref times. */
   void GetBoundingBox(Vector &min, Vector &max, int ref = 2);

   void GetCharacteristics(double &h_min, double &h_max,
                           double &kappa_min, double &kappa_max,
                           Vector *Vh = NULL, Vector *Vk = NULL);

   /// Auxiliary method used by PrintCharacteristics().
   /** It is also used in the `mesh-explorer` miniapp. */
   static void PrintElementsByGeometry(int dim,
                                       const Array<int> &num_elems_by_geom,
                                       std::ostream &out);

   /** @brief Compute and print mesh characteristics such as number of vertices,
       number of elements, number of boundary elements, minimal and maximal
       element sizes, minimal and maximal element aspect ratios, etc. */
   /** If @a Vh or @a Vk are not NULL, return the element sizes and aspect
       ratios for all elements in the given Vector%s. */
   void PrintCharacteristics(Vector *Vh = NULL, Vector *Vk = NULL,
                             std::ostream &os = mfem::out);

   /** @brief In serial, this method calls PrintCharacteristics(). In parallel,
       additional information about the parallel decomposition is also printed.
   */
   virtual void PrintInfo(std::ostream &os = mfem::out)
   {
      PrintCharacteristics(NULL, NULL, os);
   }

   /** @brief Find the ids of the elements that contain the given points, and
       their corresponding reference coordinates.

       The DenseMatrix @a point_mat describes the given points - one point for
       each column; it should have SpaceDimension() rows.

       The InverseElementTransformation object, @a inv_trans, is used to attempt
       the element transformation inversion. If NULL pointer is given, the
       method will use a default constructed InverseElementTransformation. Note
       that the algorithms in the base class InverseElementTransformation can be
       completely overwritten by deriving custom classes that override the
       Transform() method.

       If no element is found for the i-th point, elem_ids[i] is set to -1.

       In the ParMesh implementation, the @a point_mat is expected to be the
       same on all ranks. If the i-th point is found by multiple ranks, only one
       of them will mark that point as found, i.e. set its elem_ids[i] to a
       non-negative number; the other ranks will set their elem_ids[i] to -2 to
       indicate that the point was found but assigned to another rank.

       @returns The total number of points that were found.

       @note This method is not 100 percent reliable, i.e. it is not guaranteed
       to find a point, even if it lies inside a mesh element. */
   virtual int FindPoints(DenseMatrix& point_mat, Array<int>& elem_ids,
                          Array<IntegrationPoint>& ips, bool warn = true,
                          InverseElementTransformation *inv_trans = NULL);

   /** @brief Computes geometric parameters associated with a Jacobian matrix
       in 2D/3D. These parameters are
       (1) Area/Volume,
       (2) Aspect-ratio (1 in 2D, and 2 non-dimensional and 2 dimensional
                         parameters in 3D. Dimensional parameters are used
                         for target construction in TMOP),
       (3) skewness (1 in 2D and 3 in 3D), and finally
       (4) orientation (1 in 2D and 3 in 3D).
    */
   void GetGeometricParametersFromJacobian(const DenseMatrix &J,
                                           double &volume,
                                           Vector &aspr,
                                           Vector &skew,
                                           Vector &ori) const;

   /// Swaps internal data with another mesh. By default, non-geometry members
   /// like 'ncmesh' and 'NURBSExt' are only swapped when 'non_geometry' is set.
   void Swap(Mesh& other, bool non_geometry);

   /// Destroys Mesh.
   virtual ~Mesh() { DestroyPointers(); }

#ifdef MFEM_DEBUG
   /// Output an NCMesh-compatible debug dump.
   void DebugDump(std::ostream &out) const;
#endif
};

/** Overload operator<< for std::ostream and Mesh; valid also for the derived
    class ParMesh */
std::ostream &operator<<(std::ostream &out, const Mesh &mesh);


/** @brief Structure for storing mesh geometric factors: coordinates, Jacobians,
    and determinants of the Jacobians. */
/** Typically objects of this type are constructed and owned by objects of class
    Mesh. See Mesh::GetGeometricFactors(). */
class GeometricFactors
{

private:
   void Compute(const GridFunction &nodes,
                MemoryType d_mt = MemoryType::DEFAULT);

public:
   const Mesh *mesh;
   const IntegrationRule *IntRule;
   int computed_factors;

   enum FactorFlags
   {
      COORDINATES  = 1 << 0,
      JACOBIANS    = 1 << 1,
      DETERMINANTS = 1 << 2,
   };

   GeometricFactors(const Mesh *mesh, const IntegrationRule &ir, int flags,
                    MemoryType d_mt = MemoryType::DEFAULT);

   GeometricFactors(const GridFunction &nodes, const IntegrationRule &ir,
                    int flags,
                    MemoryType d_mt = MemoryType::DEFAULT);

   /// Mapped (physical) coordinates of all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x SDIM x NE)
       where
       - NQ = number of quadrature points per element,
       - SDIM = space dimension of the mesh = mesh.SpaceDimension(), and
       - NE = number of elements in the mesh. */
   Vector X;

   /// Jacobians of the element transformations at all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x SDIM x DIM x
       NE) where
       - NQ = number of quadrature points per element,
       - SDIM = space dimension of the mesh = mesh.SpaceDimension(),
       - DIM = dimension of the mesh = mesh.Dimension(), and
       - NE = number of elements in the mesh. */
   Vector J;

   /// Determinants of the Jacobians at all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x NE) where
       - NQ = number of quadrature points per element, and
       - NE = number of elements in the mesh. */
   Vector detJ;
};

/** @brief Structure for storing face geometric factors: coordinates, Jacobians,
    determinants of the Jacobians, and normal vectors. */
/** Typically objects of this type are constructed and owned by objects of class
    Mesh. See Mesh::GetFaceGeometricFactors(). */
class FaceGeometricFactors
{
public:
   const Mesh *mesh;
   const IntegrationRule *IntRule;
   int computed_factors;
   FaceType type;

   enum FactorFlags
   {
      COORDINATES  = 1 << 0,
      JACOBIANS    = 1 << 1,
      DETERMINANTS = 1 << 2,
      NORMALS      = 1 << 3,
   };

   FaceGeometricFactors(const Mesh *mesh, const IntegrationRule &ir, int flags,
                        FaceType type, MemoryType d_mt = MemoryType::DEFAULT);

   /// Mapped (physical) coordinates of all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x SDIM x NF)
       where
       - NQ = number of quadrature points per face,
       - SDIM = space dimension of the mesh = mesh.SpaceDimension(), and
       - NF = number of faces in the mesh. */
   Vector X;

   /// Jacobians of the element transformations at all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x SDIM x DIM x
       NF) where
       - NQ = number of quadrature points per face,
       - SDIM = space dimension of the mesh = mesh.SpaceDimension(),
       - DIM = dimension of the mesh = mesh.Dimension(), and
       - NF = number of faces in the mesh. */
   Vector J;

   /// Determinants of the Jacobians at all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x NF) where
       - NQ = number of quadrature points per face, and
       - NF = number of faces in the mesh. */
   Vector detJ;

   /// Normals at all quadrature points.
   /** This array uses a column-major layout with dimensions (NQ x DIM x NF) where
       - NQ = number of quadrature points per face,
       - SDIM = space dimension of the mesh = mesh.SpaceDimension(), and
       - NF = number of faces in the mesh. */
   Vector normal;
};

/// Class used to extrude the nodes of a mesh
class NodeExtrudeCoefficient : public VectorCoefficient
{
private:
   int n, layer;
   double p[2], s;
   Vector tip;
public:
   NodeExtrudeCoefficient(const int dim, const int n_, const double s_);
   void SetLayer(const int l) { layer = l; }
   using VectorCoefficient::Eval;
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip);
   virtual ~NodeExtrudeCoefficient() { }
};


/// Extrude a 1D mesh
Mesh *Extrude1D(Mesh *mesh, const int ny, const double sy,
                const bool closed = false);

/// Extrude a 2D mesh
Mesh *Extrude2D(Mesh *mesh, const int nz, const double sz);

// shift cyclically 3 integers left-to-right
inline void ShiftRight(int &a, int &b, int &c)
{
   int t = a;
   a = c;  c = b;  b = t;
}

/// @brief Print function for Mesh::FaceInformation.
std::ostream& operator<<(std::ostream& os, const Mesh::FaceInformation& info);

}

#endif
