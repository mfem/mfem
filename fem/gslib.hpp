// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "gridfunc.hpp"
#include <limits>
#include "../general/tic_toc.hpp"

#ifdef MFEM_USE_GSLIB

namespace gslib
{
struct comm;
struct findpts_data_2;
struct crystal;
struct hash_data_3;
struct hash_data_2;
}

namespace mfem
{

/** \brief FindPointsGSLIB can robustly evaluate a GridFunction on an arbitrary
 *  collection of points. There are three key functions in FindPointsGSLIB:
 *
 *  1. Setup - constructs the internal data structures of gslib.
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
 *     by gslib.
 *
 *  3. Interpolate - Interpolates any grid function at the points found using 2.
 *
 *  FindPointsGSLIB provides interface to use these functions individually or
 *  using a single call.
 */
class FindPointsGSLIB
{
public:
   enum AvgType {NONE, ARITHMETIC, HARMONIC}; // Average type for L2 functions
   double setup_split_time = 0.0,
          setup_nodalmapping_time = 0.0,
          setup_findpts_setup_time = 0.0;
   double findpts_findpts_time = 0.0,
          findpts_mapelemrst_time = 0.0,
          findpts_setup_device_arrays_time = 0.0;
   double interpolate_h1_time = 0.0,
          interpolate_general_time = 0.0,
          interpolate_l2_pass2_time = 0.0;
   double min_fpt_kernel_time = 0.0,
          measured_min_fpt_kernel_time = 0.0,
          fpt_kernel_time = 0.0,
          fast_fpt_kernel_time = 0.0;

protected:
   Mesh *mesh;
   Array<Mesh *> mesh_split;  // Meshes used to split simplices.
   // IntegrationRules for simplex->Quad/Hex and to project to highest polynomial
   // order in-case of p-refinement.
   Array<IntegrationRule *> ir_split;
   Array<FiniteElementSpace *>
   fes_rst_map; // FESpaces to map info Quad/Hex->Simplex
   Array<GridFunction *> gf_rst_map; // GridFunctions to map info Quad/Hex->Simplex
   FiniteElementCollection *fec_map_lin;
   //   struct gslib::findpts_data_2 *fdata2D; // gslib's internal data
   //   struct gslib::findpts_data_3 *fdata3D; // gslib's internal data
   void *fdataD;
   struct gslib::crystal *cr;             // gslib's internal data
   struct gslib::comm *gsl_comm;          // gslib's internal data
   int dim, spacedim, points_cnt;
   Array<unsigned int> gsl_code, gsl_proc, gsl_elem, gsl_mfem_elem;
   Array<int> gsl_newton;
   Array<int> gsl_code_dev, gsl_elem_dev, gsl_newton_dev;
   Vector gsl_mesh, gsl_ref, gsl_dist, gsl_mfem_ref;
   bool setupflag;              // flag to indicate whether gslib data has been setup
   double default_interp_value; // used for points that are not found in the mesh
   AvgType avgtype;             // average type used for L2 functions
   Array<int> split_element_map;
   Array<int> split_element_index;
   int        NE_split_total;
   int        mesh_points_cnt;
   // Tolerance to ignore points just outside elements at the boundary.
   double     bdr_tol;
   bool       tensor_product = false;
   int        gpu_code = 0;
   int        newton_iter = 0;

   void * findptsData2;
   void * findptsData3;

   struct
   {
      int local_hash_size;
      int dof1d;
      int dof1dsol;
      double tol;
      int hd_d_size; //local hash data size
      struct gslib::crystal *cr;
      struct gslib::hash_data_3 *hash3;
      struct gslib::hash_data_2 *hash2;
      mutable Vector o_xyz;
      mutable Vector o_c, o_A, o_min, o_max, o_box;
      mutable Vector o_wtend;
      mutable Vector gll1d;
      mutable Vector lagcoeff;
      mutable Vector gll1dsol;
      mutable Vector lagcoeffsol;

      mutable Array<unsigned int> o_code, o_proc, o_el;
      mutable DenseTensor o_r;
      mutable Array<unsigned int> ou_offset;

      mutable Array<int> o_offset;
      mutable int hash_n;
      mutable Vector o_hashMin;
      mutable Vector o_hashMax;
      mutable Vector o_hashFac;
      mutable Vector info;
   } DEV;

   // Stopwatches
   StopWatch setupSW, SW2, SWkernel;

   /// Use GSLIB for communication and interpolation
   virtual void InterpolateH1(const GridFunction &field_in, Vector &field_out);
   /// Uses GSLIB Crystal Router for communication followed by MFEM's
   /// interpolation functions
   virtual void InterpolateGeneral(const GridFunction &field_in,
                                   Vector &field_out);

   /// Since GSLIB is designed to work with quads/hexes, we split every
   /// triangle/tet/prism/pyramid element into quads/hexes.
   virtual void SetupSplitMeshes();
   // Same as above but for surface meshes
   virtual void SetupSplitMeshesSurf();

   /// Setup integration points that will be used to interpolate the nodal
   /// location at points expected by GSLIB.
   virtual void SetupIntegrationRuleForSplitMesh(Mesh *mesh,
                                                 IntegrationRule *irule,
                                                 int order);

   /// Get GridFunction value at the points expected by GSLIB.
   virtual void GetNodalValues(const GridFunction *gf_in, Vector &node_vals);

   virtual void GetNodalValuesSurf(const GridFunction *gf_in, Vector &node_vals);

   /// Map {r,s,t} coordinates from [-1,1] to [0,1] for MFEM. For simplices,
   /// find the original element number (that was split into micro quads/hexes)
   /// during the setup phase.
   virtual void MapRefPosAndElemIndices();
   // Same as above but for surface meshes
   virtual void MapRefPosAndElemIndicesSurf();

   // FindPoints locally on device for 3D.
   void FindPointsLocal3(const Vector &point_pos,
                         int point_pos_ordering,
                         Array<unsigned int> &gsl_code_dev_l,
                         Array<unsigned int> &gsl_elem_dev_l,
                         Vector &gsl_ref_l,
                         Vector &gsl_dist_l,
                         Array<int> &gsl_newton_dev_l,
                         int npt);
   // Faster version of FindPointsLocal3.
   void FindPointsLocal32(Vector &point_pos,
                          int point_pos_ordering,
                          Array<unsigned int> &gsl_code_dev_l,
                          Array<unsigned int> &gsl_elem_dev_l,
                          Vector &gsl_ref_l,
                          Vector &gsl_dist_l,
                          Array<int> &gsl_newton_dev_l,
                          int npt);
   void FindPointsSurfLocal32(const Vector &point_pos,
                              int point_pos_ordering,
                              Array<unsigned int> &gsl_code_dev_l,
                              Array<unsigned int> &gsl_elem_dev_l,
                              Vector &gsl_ref_l,
                              Vector &gsl_dist_l,
                              Array<int> &gsl_newton_dev_l,
                              int npt);
   // FindPoints locally on device for 2D.
   void FindPointsLocal2(const Vector &point_pos,
                         int point_pos_ordering,
                         Array<unsigned int> &gsl_code_dev_l,
                         Array<unsigned int> &gsl_elem_dev_l,
                         Vector &gsl_ref_l,
                         Vector &gsl_dist_l,
                         Array<int> &gsl_newton_dev_l,
                         int npt);

   void FindPointsSurfLocal2(const Vector &point_pos,
                             int point_pos_ordering,
                             Array<unsigned int> &gsl_code_dev_l,
                             Array<unsigned int> &gsl_elem_dev_l,
                             Vector &gsl_ref_l,
                             Vector &gsl_dist_l,
                             Array<int> &gsl_newton_dev_l,
                             int npt);
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
   void InterpolateSurfLocal2( const Vector &field_in,
                               Array<int> &gsl_elem_dev_l,
                               Vector &gsl_ref_l,
                               Vector &field_out,
                               int npt,
                               int ncomp,
                               int nel,
                               int dof1dsol );
   void InterpolateSurfLocal3( const Vector &field_in,
                               Array<int> &gsl_elem_dev_l,
                               Vector &gsl_ref_l,
                               Vector &field_out,
                               int npt,
                               int ncomp,
                               int nel,
                               int dof1dsol );

public:
   FindPointsGSLIB();

#ifdef MFEM_USE_MPI
   FindPointsGSLIB(MPI_Comm comm_);
#endif

   virtual ~FindPointsGSLIB();

   /** Initializes the internal mesh in gslib, by sending the positions of the
       Gauss-Lobatto nodes of the input Mesh object @a m.
       Note: not tested with periodic (L2).
       Note: the input mesh @a m must have Nodes set.

       @param[in] m         Input mesh.
       @param[in] bb_t      (Optional) Relative size of bounding box around
                            each element.
       @param[in] newt_tol  (Optional) Newton tolerance for the gslib
                            search methods.
       @param[in] npt_max   (Optional) Number of points for simultaneous
                            iteration. This alters performance and
                            memory footprint.*/
   void Setup(Mesh &m, const double bb_t = 0.1,
              const double newt_tol = 1.0e-12,
              const int npt_max = 256);

   void SetupSurf(Mesh &m,
                  const double bb_t = 0.1,
                  const double newt_tol = 1.0e-12,
                  const int npt_max = 256);

   /** Searches positions given in physical space by @a point_pos.
       These positions can be ordered byNodes: (XXX...,YYY...,ZZZ) or
       byVDim: (XYZ,XYZ,....XYZ) specified by @a point_pos_ordering.
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
                   int point_pos_ordering = Ordering::byNODES);

   void FindPointsSurf(const Vector &point_pos,
                       int point_pos_ordering = Ordering::byNODES);
   /// Setup FindPoints and search positions
   void FindPoints(Mesh &m, const Vector &point_pos,
                   int point_pos_ordering = Ordering::byNODES,
                   const double bb_t = 0.1,
                   const double newt_tol = 1.0e-12,  const int npt_max = 256);

   /** Interpolation of field values at prescribed reference space positions.
       @param[in] field_in    Function values that will be interpolated on the
                              reference positions. Note: it is assumed that
                              @a field_in is in H1 and in the same space as the
                              mesh that was given to Setup().
       @param[out] field_out  Interpolated values. For points that are not found
                              the value is set to #default_interp_value. */
   virtual void Interpolate(const GridFunction &field_in, Vector &field_out);
   virtual void InterpolateSurf(const GridFunction &field_in, Vector &field_out);
   /** Search positions and interpolate. The ordering (byNODES or byVDIM) of
       the output values in @a field_out corresponds to the ordering used
       in the input GridFunction @a field_in. */
   void Interpolate(const Vector &point_pos, const GridFunction &field_in,
                    Vector &field_out,
                    int point_pos_ordering = Ordering::byNODES);
   /** Setup FindPoints, search positions and interpolate. The ordering (byNODES
       or byVDIM) of the output values in @a field_out corresponds to the
       ordering used in the input GridFunction @a field_in. */
   void Interpolate(Mesh &m, const Vector &point_pos,
                    const GridFunction &field_in, Vector &field_out,
                    int point_pos_ordering = Ordering::byNODES);

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

   /** Cleans up memory allocated internally by gslib.
       Note that in parallel, this must be called before MPI_Finalize(), as it
       calls MPI_Comm_free() for internal gslib communicators. */
   virtual void FreeData();

   /// Return code for each point searched by FindPoints: inside element (0), on
   /// element boundary (1), or not found (2).
   virtual const Array<unsigned int> &GetCode() const { return gsl_code; }
   /// Return element number for each point found by FindPoints.
   virtual const Array<unsigned int> &GetElem() const
   {
      return gsl_mfem_elem.Size() ? gsl_mfem_elem : gsl_elem;
   }
   /// Return MPI rank on which each point was found by FindPoints.
   virtual const Array<unsigned int> &GetProc() const { return gsl_proc; }
   /// Return reference coordinates for each point found by FindPoints.
   virtual const Vector &GetReferencePosition() const { return gsl_mfem_ref;  }
   /// Return distance between the sought and the found point in physical space,
   /// for each point found by FindPoints.
   virtual const Vector &GetDist()              const { return gsl_dist; }

   virtual const Vector &GetInfo()              const { return DEV.info; }

   virtual const Vector &GetGLLMesh()           const { return gsl_mesh; }

   /// Return element number for each point found by FindPoints corresponding to
   /// GSLIB mesh. gsl_mfem_elem != gsl_elem for mesh with simplices.
   virtual const Array<unsigned int> &GetGSLIBElem() const { return gsl_elem; }
   /// Return reference coordinates in [-1,1] (internal range in GSLIB) for each
   /// point found by FindPoints.
   virtual const Vector &GetGSLIBReferencePosition() const { return gsl_ref; }

   virtual void SetupDevice(MemoryType mt); // probably should be internal

   virtual void SurfSetupDevice(MemoryType mt); // probably should be internal

   void FindPointsOnDevice(const Vector &point_pos,
                           int point_pos_ordering = Ordering::byNODES);
   void FindPointsSurfOnDevice(const Vector &point_pos,
                               int point_pos_ordering = Ordering::byNODES);

   void InterpolateOnDevice(const Vector &field_in, Vector &field_out,
                            const int nel, const int ncomp,
                            const int dof1dsol, const int gf_ordering,
                            MemoryType mt);
   void InterpolateSurfOnDevice(const Vector &field_in,
                                Vector &field_out,
                                const int nel,
                                const int ncomp,
                                const int dof1dsol,
                                const int gf_ordering,
                                MemoryType mt);

   // Bounding box mesh.
   // 0 - element-wise axis-aligned bounding box. NE_split_total elem per proc
   // 1 - element-wise oriented     bounding box. NE_split_total elem per proc
   // 2 - proc-wise    axis-aligned bounding box. size depends on setup parameters
   // 3 - global hash mesh.                       depends on setup parameters.
   virtual Mesh* GetBoundingBoxMesh(int type = 0);

   virtual Mesh* GetBoundingBoxMeshSurf(int type = 0);

   virtual Mesh* GetGSLIBMesh();

   void SetGPUCode(int code_) { gpu_code = code_; }

   virtual const Array<int> &GetNewtonIters() const { return gsl_newton; }
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
      Gauss-Lobatto nodes of the input Mesh object @a m.
      Note: not tested with periodic meshes (L2).
      Note: the input mesh @a m must have Nodes set.

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

  /** Searches positions given in physical space by @a point_pos. All output
      Arrays and Vectors are expected to have the correct size.

      @param[in]  point_pos           Positions to be found.
      @param[in]  point_id            Index of the mesh that the point belongs
                                      to (corresponding to @a meshid in Setup).
      @param[in]  point_pos_ordering  Ordering of the points:
                                      byNodes: (XXX...,YYY...,ZZZ) or
                                      byVDim: (XYZ,XYZ,....XYZ) */
  void FindPoints(const Vector &point_pos,
                  Array<unsigned int> &point_id,
                  int point_pos_ordering = Ordering::byNODES);

  /** Search positions and interpolate */
  void Interpolate(const Vector &point_pos, Array<unsigned int> &point_id,
                   const GridFunction &field_in, Vector &field_out,
                   int point_pos_ordering = Ordering::byNODES);
  using FindPointsGSLIB::Interpolate;
};

} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_GSLIB
