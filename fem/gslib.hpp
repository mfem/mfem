// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_GSLIB
#define MFEM_GSLIB

#include "../config/config.hpp"
#ifdef MFEM_USE_MPI
#include "pgridfunc.hpp"
#else
#include "gridfunc.hpp"
#endif

#ifdef MFEM_USE_GSLIB

namespace gslib
{
struct comm;
struct crystal;
struct hash_data_3;
struct hash_data_2;
struct gs_data;
}

namespace mfem
{

/** \brief FindPointsGSLIB can robustly evaluate a GridFunction on an arbitrary
 *  collection of points. See Mittal et al., "General Field Evaluation in
 *  High-Order Meshes on GPUs". (2025). Computers & Fluids. for technical
 *  details.
 *
 *  There are three key functions in FindPointsGSLIB:
 *
 *  1. Setup - constructs the internal data structures of gslib. See \ref Setup.
 *
 *  2. FindPoints - for any given arbitrary set of points in physical space,
 *     gslib finds the element number, MPI rank, and the reference space
 *     coordinates inside the element that each point is located in. gslib also
 *     returns a code that indicates whether the point was found inside an
 *     element, on element border, or not found in the domain.
 *     For points returned as found on `element border`, the point is either
 *     on an element edge/face or near the domain boundary, and gslib also
 *     returns a distance to the border. Points near (but outside) the domain
 *     boundary must then be marked as not found using the distance returned
 *     by gslib. See \ref FindPoints.
 *
 *  3. Interpolate - Interpolates any grid function at the points found using 2.
 *     For functions in L2 finite element space, use \ref SetL2AvgType to
 *     specify how to interpolate values at points located at element boundaries
 *     where the function might be multi-valued. See \ref Interpolate.
 *
 *  FindPointsGSLIB also provides interface to use these functions through a
 *  single call.
 *
 *  For custom interpolation (e.g., evaluating strain rate tensor), we provide
 *  functions that use gslib to send element index and corresponding
 *  reference-space coordinates for each point to the mpi rank that the element
 *  is located on. Then, custom interpolation can be defined locally by the user
 *  before sending the values back to mpi ranks where the query originated from.
 *  See \ref DistributePointInfoToOwningMPIRanks and
 *  \ref DistributeInterpolatedValues.
 */
class FindPointsGSLIB
{
public:
   enum AvgType {NONE, ARITHMETIC, HARMONIC}; // Average type for L2 functions

protected:
   Mesh *mesh;
   Array<Mesh *> mesh_split;  // Meshes used to split simplices.
   // IntegrationRules for simplex->Quad/Hex and to project to p_max in-case of
   // p-refinement.
   Array<IntegrationRule *> ir_split;
   Array<FiniteElementSpace *> fes_rst_map; //FESpaces to map Quad/Hex->Simplex
   Array<GridFunction *> gf_rst_map; // GridFunctions to map Quad/Hex->Simplex
   FiniteElementCollection *fec_map_lin;
   void *fdataD;
   struct gslib::crystal *cr;             // gslib's internal data
   struct gslib::comm *gsl_comm;          // gslib's internal data
   int dim, points_cnt;                   // mesh dimension and number of points
   Array<unsigned int> gsl_code, gsl_proc, gsl_elem, gsl_mfem_elem;
   Vector gsl_mesh, gsl_ref, gsl_dist, gsl_mfem_ref;
   Array<unsigned int> recv_proc, recv_index; // data for custom interpolation
   bool setupflag;              // flag to indicate if gslib data has been setup
   double default_interp_value; // used for points that are not found in the mesh
   AvgType avgtype;             // average type used for L2 functions
   Array<int> split_element_map;
   Array<int> split_element_index;
   int        NE_split_total;   // total number of elements after mesh splitting
   int        mesh_points_cnt;  // number of mesh nodes
   // Tolerance to ignore points found beyond the mesh boundary.
   // i.e. if ||x*-x(r)||_2^2 > bdr_tol, we mark point as not found.
   double     bdr_tol;
   // Use CPU functions for Mesh/GridFunction on device for gslib1.0.7
   bool       gpu_to_cpu_fallback = false;

   // Device specific data used for FindPoints
   struct
   {
      bool setup_device = false;
      bool find_device  = false;
      int local_hash_size, dof1d, dof1d_sol, h_o_size, h_nx;
      double newt_tol; // Tolerance specified during setup for Newton solve
      struct gslib::crystal *cr;
      struct gslib::hash_data_3 *hash3;
      struct gslib::hash_data_2 *hash2;
      mutable Vector bb, wtend, gll1d, lagcoeff, gll1d_sol, lagcoeff_sol;
      mutable Array<unsigned int> loc_hash_offset;
      mutable Vector loc_hash_min, loc_hash_fac;
   } DEV;

   /// Use GSLIB for communication and interpolation
   virtual void InterpolateH1(const GridFunction &field_in, Vector &field_out,
                              const int field_out_ordering);
   /// Uses GSLIB Crystal Router for communication followed by MFEM's
   /// interpolation functions
   virtual void InterpolateGeneral(const GridFunction &field_in,
                                   Vector &field_out,
                                   const int field_out_ordering);

   /// Since GSLIB is designed to work with quads/hexes, we split every
   /// triangle/tet/prism/pyramid element into quads/hexes.
   virtual void SetupSplitMeshes();

   /// Setup integration points that will be used to interpolate the nodal
   /// location at points expected by GSLIB.
   virtual void SetupIntegrationRuleForSplitMesh(Mesh *mesh,
                                                 IntegrationRule *irule,
                                                 int order);

   /// Helper function that calls \ref SetupSplitMeshes and
   /// \ref SetupIntegrationRuleForSplitMesh.
   virtual void SetupSplitMeshesAndIntegrationRules(const int order);

   /// Get GridFunction value at the points expected by GSLIB.
   virtual void GetNodalValues(const GridFunction *gf_in, Vector &node_vals) const;

   /// Map {r,s,t} coordinates from [-1,1] to [0,1] for MFEM. For simplices,
   /// find the original element number (that was split into micro quads/hexes)
   /// during the setup phase.
   virtual void MapRefPosAndElemIndices();

   // Device functions
   // FindPoints locally on device for 3D.
   void FindPointsLocal3(const Vector &point_pos, int point_pos_ordering,
                         Array<unsigned int> &gsl_code_dev_l,
                         Array<unsigned int> &gsl_elem_dev_l, Vector &gsl_ref_l,
                         Vector &gsl_dist_l, int npt);

   // FindPoints locally on device for 2D.
   void FindPointsLocal2(const Vector &point_pos, int point_pos_ordering,
                         Array<unsigned int> &gsl_code_dev_l,
                         Array<unsigned int> &gsl_elem_dev_l, Vector &gsl_ref_l,
                         Vector &gsl_dist_l, int npt);

   // Interpolate on device for 3D.
   void InterpolateLocal3(const Vector &field_in,
                          Array<int> &gsl_elem_dev_l,
                          Vector &gsl_ref_l,
                          Vector &field_out,
                          int npt, int ncomp,
                          int nel, int dof1dsol);
   // Interpolate on device for 2D.
   void InterpolateLocal2(const Vector &field_in,
                          Array<int> &gsl_elem_dev_l,
                          Vector &gsl_ref_l,
                          Vector &field_out,
                          int npt, int ncomp,
                          int nel, int dof1dsol);

   // Prepare data for device functions.
   void SetupDevice();

   /** Searches positions given in physical space by @a point_pos.
       These positions can be ordered byNodes: (XXX...,YYY...,ZZZ) or
       byVDim: (XYZ,XYZ,....XYZ) specified by @a point_pos_ordering. */
   void FindPointsOnDevice(const Vector &point_pos,
                           const int point_pos_ordering = Ordering::byNODES);

   /** Interpolation of field values at prescribed reference space positions.
       @param[in] field_in_evec E-vector of grid function to be interpolated.
                                Assumed ordering is NDOFSxVDIMxNEL
       @param[in] nel           Number of elements in the mesh.
       @param[in] ncomp         Number of components in the field.
       @param[in] dof1dsol      Number of degrees of freedom in each reference
                                space direction.
       @param[in] ordering      Ordering of the out field values: byNodes/byVDIM

       @param[out] field_out  Interpolated values. For points that are not found
                              the value is set to #default_interp_value. */
   void InterpolateOnDevice(const Vector &field_in_evec, Vector &field_out,
                            const int nel, const int ncomp,
                            const int dof1dsol, const int ordering);

public:
   FindPointsGSLIB();
   FindPointsGSLIB(Mesh &mesh_in, const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,
                   const int npt_max = 256);

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm comm_);
   FindPointsGSLIB(ParMesh &mesh_in, const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,
                   const int npt_max = 256);
#endif

   virtual ~FindPointsGSLIB();
   FindPointsGSLIB(const FindPointsGSLIB&) = delete;
   FindPointsGSLIB& operator=(const FindPointsGSLIB&) = delete;

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object \p m.
       Note: not tested with periodic (L2).
       Note: the input mesh \p m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      (Optional) Relative size of bounding box around
                            each element.
       @param[in] newt_tol  (Optional) Newton tolerance for the gslib
                            search methods.
       @param[in] npt_max   (Optional) Number of points for simultaneous
                            iteration. This alters performance and
                            memory footprint.*/

   void Setup(Mesh &m, const double bb_t = 0.1, const double newt_tol = 1.0e-12,
              const int npt_max = 256);
   /** Searches positions given in physical space by \p point_pos.
       These positions can be ordered byNodes: (XXX...,YYY...,ZZZ) or
       byVDim: (XYZ,XYZ,....XYZ) specified by \p point_pos_ordering.
       This function populates the following member variables:
       #gsl_code        Return codes for each point: inside element (0),
                        element boundary (1), not found (2).
       #gsl_proc        MPI proc ids where the points were found.
       #gsl_elem        Element ids where the points were found.
                        Defaults to 0 for points that were not found.
       #gsl_mfem_elem   Element ids corresponding to MFEM-mesh where the points
                        were found. #gsl_mfem_elem != #gsl_elem for simplices
                        Defaults to 0 for points that were not found.
       #gsl_ref         Reference coordinates of the found point.
                        Ordered by vdim (XYZ,XYZ,XYZ...). Defaults to -1 for
                        points that were not found. Note: the gslib reference
                        frame is [-1,1].
       #gsl_mfem_ref    Reference coordinates #gsl_ref mapped to [0,1].
                        Defaults to 0 for points that were not found.
       #gsl_dist        Distance between the sought and the found point
                        in physical space. */
   void FindPoints(const Vector &point_pos,
                   const int point_pos_ordering = Ordering::byNODES);
   /// Convenience function when point positions are in a ParticleVector
   void FindPoints(const ParticleVector &point_pos)
   {
      FindPoints(point_pos, point_pos.GetOrdering());
   }
   /// Setup FindPoints and search positions
   void FindPoints(Mesh &m, const Vector &point_pos,
                   const int point_pos_ordering = Ordering::byNODES,
                   const double bb_t = 0.1, const double newt_tol = 1.0e-12,
                   const int npt_max = 256);

   /** Interpolation of field values at prescribed reference space positions.
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              \p field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. For points that are not found
                              the value is set to #default_interp_value. */
   virtual void Interpolate(const GridFunction &field_in, Vector &field_out);
   /// Interpolation of field values, with output ordering specification.
   virtual void Interpolate(const GridFunction &field_in, Vector &field_out,
                            const int field_out_ordering);
   /// Interpolation of field values, with output ordering from ParticleVector
   virtual void Interpolate(const GridFunction &field_in,
                            ParticleVector &field_out)
   {
      Interpolate(field_in, field_out, field_out.GetOrdering());
   }
   /** Search positions and interpolate. The ordering (byNODES or byVDIM) of
       the output values in \p field_out corresponds to the ordering used
       in the input GridFunction \p field_in. */
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out,
                    const int point_pos_ordering = Ordering::byNODES);
   /// Search positions and interpolate with given point and output ordering.
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out, const int point_pos_ordering,
                    const int field_out_ordering);
   /** Setup FindPoints, search positions and interpolate. The ordering (byNODES
       or byVDIM) of the output values in \p field_out corresponds to the
       ordering used in the input GridFunction \p field_in. */
   void Interpolate(Mesh &m, const Vector &point_pos,
                    const GridFunction &field_in, Vector &field_out,
                    const int point_pos_ordering = Ordering::byNODES);

   /// Average type to be used for L2 functions in-case a point is located at
   /// an element boundary where the function might be multi-valued.
   virtual void SetL2AvgType(AvgType avgtype_) { avgtype = avgtype_; }

   /// Set the default interpolation value for points that are not found in the
   /// mesh.
   virtual void SetDefaultInterpolationValue(double interp_value_)
   {
      default_interp_value = interp_value_;
   }

   /// Set the tolerance for detecting points outside the 'curvilinear' boundary
   /// that gslib may return as found on the boundary. Points found on boundary
   /// with distance greater than @ bdr_tol are marked as not found.
   virtual void SetDistanceToleranceForPointsFoundOnBoundary(double bdr_tol_)
   {
      bdr_tol = bdr_tol_;
   }

   /// Enable/Disable use of CPU functions for GPU data if the gslib version
   /// is older.
   virtual void SetGPUtoCPUFallback(bool mode) { gpu_to_cpu_fallback = mode; }

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as it
       calls MPI_Comm_free() for internal gslib communicators. FreeData is
       also called by the class destructor and there are no memory leaks if the
       destructor is called before MPI_Finalize(). If the destructor is called
       after MPI_Finalize(), there will be an error because gslib will try to
       invoke some MPI functions.
   */
   virtual void FreeData();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   virtual const Array<unsigned int> &GetCode() const { return gsl_code; }
   /// Return element number for each point found by FindPoints.
   virtual const Array<unsigned int> &GetElem() const { return gsl_mfem_elem; }
   /// Return MPI rank on which each point was found by FindPoints.
   virtual const Array<unsigned int> &GetProc() const { return gsl_proc; }
   /// Return reference coordinates for each point found by FindPoints.
   virtual const Vector &GetReferencePosition() const { return gsl_mfem_ref;  }
   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   virtual const Vector &GetDist()              const { return gsl_dist; }

   /// Return element number for each point found by FindPoints corresponding to
   /// GSLIB mesh. gsl_mfem_elem != gsl_elem for mesh with simplices.
   virtual const Array<unsigned int> &GetGSLIBElem() const { return gsl_elem; }
   /// Return reference coordinates in [-1,1] (internal range in GSLIB) for each
   /// point found by FindPoints.
   virtual const Vector &GetGSLIBReferencePosition() const { return gsl_ref; }

   /// Get array of indices of not-found points.
   Array<unsigned int> GetPointsNotFoundIndices() const;

   /** @name Methods to support a custom interpolation procedure.
       \brief The physical-space point that the user seeks to interpolate at
       could be located inside an element on another mpi rank.
       To enable a custom interpolation procedure (e.g., strain tensor computation)
       we need a mechanism to first send element indices and reference-space
       coordinates to the mpi-ranks where each point is found. Then the custom
       interpolation can be done locally by the user before sending the
       interpolated values back to the mpi-ranks that the query originated from.
       Example usage looks something like this:

       FindPoints() -> DistributePointInfoToOwningMPIRanks() -> Computation by
       user -> DistributeInterpolatedValues().
   */
   ///@{
   /// Distribute element indices in #gsl_mfem_elem, the reference coordinates
   /// #gsl_mfem_ref, and the code #gsl_code to the corresponding mpi-rank
   /// #gsl_proc for each point. The received information is provided locally
   /// in \p recv_elem, \p recv_ref (ordered by vdim), and \p recv_code.
   /// Note: The user can send empty Array/Vectors to the method as they are
   /// appropriately sized and filled internally.
   virtual void DistributePointInfoToOwningMPIRanks(
      Array<unsigned int> &recv_elem, Vector &recv_ref,
      Array<unsigned int> &recv_code);
   /// Return interpolated values back to the mpi-ranks #recv_proc that had
   /// sent the element indices and corresponding reference-space coordinates.
   /// Specify \p vdim and \p ordering (by nodes or by vdim) based on how the
   /// \p int_vals are structured. The received values are filled in
   /// \p field_out consistent with the original ordering of the points that
   /// were used in \ref FindPoints.
   virtual void DistributeInterpolatedValues(const Vector &int_vals,
                                             const int vdim,
                                             const int ordering,
                                             Vector &field_out) const;
   ///@}

   /// Return the axis-aligned bounding boxes (AABB) computed during \ref Setup.
   /// The size of the returned vector is (nel x nverts x dim), where nel is the
   /// number of elements (after splitting for simplcies), nverts is number of
   /// vertices (4 in 2D, 8 in 3D), and dim is the spatial dimension.
   void GetAxisAlignedBoundingBoxes(Vector &aabb) const;

   /// Return the oriented bounding boxes (OBB) computed during \ref Setup.
   /// Each OBB is represented using the inverse transformation (A^{-1}) and
   /// its center (x_c), such that a point x is inside the OBB if:
   ///                  -1 <= A^{-1}(x-x_c) <= 1.
   /// The inverse transformation is returned in \p obbA, a DenseTensor of
   /// size (dim x dim x nel), and the OBB centers are returned in \p obbC,
   /// a vector of size (nel x dim). The vertices of the OBBs are returned in
   /// \p obbV, a vector of size (nel x nverts x dim) .
   void GetOrientedBoundingBoxes(DenseTensor &obbA, Vector &obbC,
                                 Vector &obbV) const;
};

/** \brief OversetFindPointsGSLIB enables use of findpts for arbitrary number of
    overlapping grids.

    The parameters in this class are the same as FindPointsGSLIB with the
    difference of additional inputs required to account for more than 1 mesh. */
class OversetFindPointsGSLIB : public FindPointsGSLIB
{
protected:
   bool overset;
   unsigned int u_meshid;
   Vector distfint; // Used to store nodal vals of grid func. passed to findpts

public:
   OversetFindPointsGSLIB() : FindPointsGSLIB(),
      overset(true) { }

#ifdef MFEM_USE_MPI
   OversetFindPointsGSLIB(MPI_Comm comm_) : FindPointsGSLIB(comm_),
      overset(true) { }
#endif

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object \p m.
       Note: not tested with periodic meshes (L2).
       Note: the input mesh \p m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] meshid    A unique # for each overlapping mesh. This id is
                            used to make sure that points being searched are not
                            looked for in the mesh that they belong to.
       @param[in] gfmax     (Optional) GridFunction in H1 that is used as a
                            discriminator when one point is located in multiple
                            meshes. The mesh that maximizes gfmax is chosen.
                            For example, using the distance field based on the
                            overlapping boundaries is helpful for convergence
                            during Schwarz iterations.
       @param[in] bb_t      (Optional) Relative size of bounding box around
                            each element.
       @param[in] newt_tol  (Optional) Newton tolerance for the gslib
                            search methods.
       @param[in] npt_max   (Optional) Number of points for simultaneous
                            iteration. This alters performance and
                            memory footprint.*/
   void Setup(Mesh &m, const int meshid, GridFunction *gfmax = NULL,
              const double bb_t = 0.1, const double newt_tol = 1.0e-12,
              const int npt_max = 256);

   /** Searches positions given in physical space by \p point_pos. All output
       Arrays and Vectors are expected to have the correct size.

       @param[in]  point_pos           Positions to be found.
       @param[in]  point_id            Index of the mesh that the point belongs
                                       to (corresponding to \p meshid in Setup).
       @param[in]  point_pos_ordering  Ordering of the points:
                                       byNodes: (XXX...,YYY...,ZZZ) or
                                       byVDim: (XYZ,XYZ,....XYZ) */
   void FindPoints(const Vector &point_pos,
                   const Array<unsigned int> &point_id,
                   const int point_pos_ordering = Ordering::byNODES);

   /** Search positions and interpolate */
   void Interpolate(const Vector &point_pos,
                    const Array<unsigned int> &point_id,
                    const GridFunction &field_in, Vector &field_out,
                    const int point_pos_ordering = Ordering::byNODES);
   using FindPointsGSLIB::Interpolate;
};

/** \brief  Class for gather-scatter (gs) operations on Vectors based on
    corresponding global identifiers.

    This functionality is useful for gs-ops on DOF values across processor
    boundary, where the global identifier would be the corresponding true DOF
    index. Operations currently supported are min, max, sum, and multiplication.
    Note: identifier 0 does not participate in the gather-scatter operation and
    a given identifier can be included multiple times on a given rank.
    For example, consider a vector, v:
    - v = [0.3, 0.4, 0.25, 0.7] on rank1,
    - v = [0.6, 0.1] on rank 2,
    - v = [-0.2, 0.3, 0.7, 0.] on rank 3.

    Consider a corresponding Array<int>, a:
    - a = [1, 2, 3, 1] on rank 1,
    - a = [3, 2] on rank 2,
    - a = [1, 2, 0, 3] on rank 3.

    A gather-scatter "minimum" operation, done as follows:
    GSOPGSLIB gs = GSOPGSLIB(MPI_COMM_WORLD, a);
    gs.GS(v, GSOp::MIN);
    would return into v:
    - v = [-0.2, 0.1, 0., -0.2] on rank 1,
    - v = [0., 0.1] on rank 2,
    - v = [-0.2, 0.1, 0.7, 0.] on rank 3,
    where the values have been compared across all processors based on the
    integer identifier. */
class GSOPGSLIB
{
protected:
   struct gslib::crystal *cr;               // gslib's internal data
   struct gslib::comm *gsl_comm;            // gslib's internal data
   struct gslib::gs_data *gsl_data = NULL;
   int num_ids;

public:
   GSOPGSLIB(Array<long long> &ids);

#ifdef MFEM_USE_MPI
   GSOPGSLIB(MPI_Comm comm_, Array<long long> &ids);
#endif

   virtual ~GSOPGSLIB();

   /// Supported operation types. See class description.
   enum GSOp {ADD, MUL, MIN, MAX};

   /// Update the identifiers used for the gather-scatter operator.
   /// Same \p ids get grouped together and id == 0 does not participate.
   /// See class description.
   void UpdateIdentifiers(const Array<long long> &ids);

   /// Gather-Scatter operation on senddata. Must match length of unique
   /// identifiers used in the constructor. See class description.
   void GS(Vector &senddata, GSOp op);
};

} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_GSLIB
