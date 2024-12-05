// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_BFIELD_REMHOS
#define MFEM_BFIELD_REMHOS

#include "mfem.hpp"
#include "general/forall.hpp"

#define EMPTY_ZONE_TOL 1e-12

namespace mfem
{

namespace electromagnetics
{

// Class storing information on dofs needed for the low order methods and FCT.
class DofInfo
{
private:
   // 0 is overlap, see ComputeOverlapBounds().
   // 1 is sparcity, see ComputeMatrixSparcityBounds().
   int bounds_type;
   ParMesh *pmesh;
   ParFiniteElementSpace &pfes;

   // The min and max bounds are represented as CG functions of the same order
   // as the solution, thus having 1:1 dof correspondence inside each element.
   H1_FECollection fec_bounds;
   ParFiniteElementSpace pfes_bounds;
   ParGridFunction x_min, x_max;

   // For each DOF on an element boundary, the global index of the DOF on the
   // opposite site is computed and stored in a list. This is needed for lumping
   // the flux contributions, as in the paper. Right now it works on 1D meshes,
   // quad meshes in 2D and 3D meshes of ordered cubes.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs();

   // A list is filled to later access the correct element-global indices given
   // the subcell number and subcell index.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   void FillSubcell2CellDof();

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   // A given DOF gets bounds from the elements it touches (in Gauss-Lobatto
   // sense, i.e., a face dof touches two elements, vertex dofs can touch many).
   void ComputeOverlapBounds(const Vector &el_min, const Vector &el_max,
                             Vector &dof_min, Vector &dof_max,
                             Array<bool> *active_el = NULL);

   // A given DOF gets bounds from its own element and its face-neighbors.
   void ComputeMatrixSparsityBounds(const Vector &el_min, const Vector &el_max,
                                    Vector &dof_min, Vector &dof_max,
                                    Array<bool> *active_el = NULL);

public:
   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element

   DenseMatrix BdrDofs, Sub2Ind;
   DenseTensor NbrDof;

   int numBdrs, numFaceDofs, numSubcells, numDofsSubcell;

   DofInfo(ParFiniteElementSpace &pfes_sltn, int btype = 0);

   // Computes the admissible interval of values for each DG dof from the values
   // of all elements that feature the dof at its physical location.
   void ComputeBounds(const Vector &el_min, const Vector &el_max,
                      Vector &dof_min, Vector &dof_max,
                      Array<bool> *active_el = NULL)
   {
      if (bounds_type == 0)
      {
         ComputeOverlapBounds(el_min, el_max, dof_min, dof_max, active_el);
      }
      else if (bounds_type == 1)
      {
         ComputeMatrixSparsityBounds(el_min, el_max,
                                     dof_min, dof_max, active_el);
      }
      else { MFEM_ABORT("Wrong option for bounds computation."); }
   }

   // Computes the min and max values of u over each element.
   void ComputeElementsMinMax(const Vector &u,
                              Vector &u_min, Vector &u_max,
                              Array<bool> *active_el,
                              Array<bool> *active_dof) const;
};

struct LowOrderMethod
{
   bool subcell_scheme;
   FiniteElementSpace *SubFes0, *SubFes1;
   Array <int> smap;
   SparseMatrix D;
   ParBilinearForm* pk;
   VectorCoefficient* coef;
   VectorCoefficient* subcellCoeff;
   const IntegrationRule* irF;
   BilinearFormIntegrator* VolumeTerms;
};

class SmoothnessIndicator
{
private:
   const int type;
   const double param;
   H1_FECollection fec_sub;
   ParFiniteElementSpace pfes_CG_sub;
   ParFiniteElementSpace &pfes_DG;
   SparseMatrix Mmat, LaplaceOp, *MassMixed;
   BilinearFormIntegrator *MassInt;
   Vector lumpedMH1;
   DenseMatrix ShapeEval;

   void ComputeVariationalMatrix(DofInfo &dof_info);
   void ApproximateLaplacian(const Vector &x, ParGridFunction &y);
   void ComputeFromSparsity(const SparseMatrix &K, const ParGridFunction &x,
                            Vector &x_min, Vector &x_max);

public:
   SmoothnessIndicator(int type_id,
                       ParMesh &subcell_mesh,
                       ParFiniteElementSpace &pfes_DG_,
                       ParGridFunction &u,
                       DofInfo &dof_info);
   ~SmoothnessIndicator();

   void ComputeSmoothnessIndicator(const Vector &u, ParGridFunction &si_vals_u);
   void UpdateBounds(int dof_id, double u_HO,
                     const ParGridFunction &si_vals,
                     double &u_min, double &u_max);

   Vector DG2CG;
};

class Assembly
{
private:
   const int exec_mode;
   const GridFunction &inflow_gf;
   mutable ParGridFunction x_gf;
   BilinearFormIntegrator *VolumeTerms;
   FiniteElementSpace *fes, *SubFes0, *SubFes1;
   Mesh *subcell_mesh;

public:
   Assembly(DofInfo &_dofs, LowOrderMethod &inlom, const GridFunction &inflow,
            ParFiniteElementSpace &pfes, ParMesh *submesh, int mode);

   // Auxiliary member variables that need to be accessed during time-stepping.
   DofInfo &dofs;

   LowOrderMethod &lom;
   // Data structures storing Galerkin contributions. These are updated for
   // remap but remain constant for transport.
   // bdrInt - eq (32).
   // SubcellWeights - above eq (49).
   DenseTensor bdrInt, SubcellWeights;

   void ComputeFluxTerms(const int e_id, const int BdrID,
                         FaceElementTransformations *Trans,
                         LowOrderMethod &lom);

   void ComputeSubcellWeights(const int k, const int m);

   void LinearFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;
   void NonlinFluxLumping(const int k, const int nd,
                          const int BdrID, const Vector &x,
                          Vector &y, const Vector &x_nd,
                          const Vector &alpha) const;

   const FiniteElementSpace * GetFes() {return fes;}

   int GetExecMode() const { return exec_mode;}

   Mesh *GetSubCellMesh() { return subcell_mesh;}
};


// Class for local assembly of M_L M_C^-1 K, where M_L and M_C are the lumped
// and consistent mass matrices and K is the convection matrix. The spaces are
// assumed to be L2 conforming.
class PrecondConvectionIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   PrecondConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

// alpha (q . grad u, v)
class MixedConvectionIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   VectorCoefficient &Q;
   double alpha;

public:
   MixedConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(q) { alpha = a; }
   virtual void AssembleElementMatrix2(const FiniteElement &tr_el,
                                       const FiniteElement &te_el,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


// Low-Order Solver.
class LOSolver
{
protected:
   ParFiniteElementSpace &pfes;
   double dt = -1.0; // usually not known at creation, updated later.

public:
   LOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~LOSolver() { }

   virtual void UpdateTimeStep(double dt_new) { dt = dt_new; }

   virtual void CalcLOSolution(const Vector &u, Vector &du) const = 0;
};

class DiscreteUpwind : public LOSolver
{
protected:
   const SparseMatrix &K;
   mutable SparseMatrix D;
   const Array<int> &K_smap;
   const Vector &M_lumped;
   Assembly &assembly;
   const bool update_D;

   void ComputeDiscreteUpwindMatrix() const;

public:
   DiscreteUpwind(ParFiniteElementSpace &space, const SparseMatrix &adv,
                  const Array<int> &adv_smap, const Vector &Mlump,
                  Assembly &asmbly, bool updateD);

   virtual void CalcLOSolution(const Vector &u, Vector &du) const;
};

// High-Order Solver.
// Conserve mass / provide high-order convergence / may violate the bounds.
class HOSolver
{
protected:
   ParFiniteElementSpace &pfes;

public:
   HOSolver(ParFiniteElementSpace &space) : pfes(space) { }

   virtual ~HOSolver() { }

   virtual void CalcHOSolution(const Vector &u, Vector &du) const = 0;
};

class LocalInverseHOSolver : public HOSolver
{
protected:
   ParBilinearForm &M, &K;

public:
   LocalInverseHOSolver(ParFiniteElementSpace &space,
                        ParBilinearForm &Mbf, ParBilinearForm &Kbf);

   virtual void CalcHOSolution(const Vector &u, Vector &du) const;
};

// Monotone, High-order, Conservative Solver.
class FCTSolver
{
protected:
   ParFiniteElementSpace &pfes;
   SmoothnessIndicator *smth_indicator;
   double dt;
   const bool needs_LO_input_for_products;

   // Computes a compatible slope (piecewise constan = mass_us / mass_u).
   // It could also update s_min and s_max, if required.
   void CalcCompatibleLOProduct(const ParGridFunction &us,
                                const Vector &m, const Vector &d_us_HO,
                                Vector &s_min, Vector &s_max,
                                const Vector &u_new,
                                const Array<bool> &active_el,
                                const Array<bool> &active_dofs,
                                Vector &d_us_LO_new);
   void ScaleProductBounds(const Vector &s_min, const Vector &s_max,
                           const Vector &u_new, const Array<bool> &active_el,
                           const Array<bool> &active_dofs,
                           Vector &us_min, Vector &us_max);

public:
   FCTSolver(ParFiniteElementSpace &space,
             SmoothnessIndicator *si, double dt_, bool needs_LO_prod)
      : pfes(space), smth_indicator(si), dt(dt_),
        needs_LO_input_for_products(needs_LO_prod) { }

   virtual ~FCTSolver() { }

   virtual void UpdateTimeStep(double dt_new) { dt = dt_new; }

   bool NeedsLOProductInput() const { return needs_LO_input_for_products; }

   // Calculate du that satisfies the following:
   // bounds preservation: u_min_i <= u_i + dt du_i <= u_max_i,
   // conservation:        sum m_i (u_i + dt du_ho_i) = sum m_i (u_i + dt du_i).
   // Some methods utilize du_lo as a backup choice, as it satisfies the above.
   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const = 0;

   // Used in the case of product remap.
   // Given the input, calculates d_us, so that:
   // bounds preservation: s_min_i <= (us_i + dt d_us_i) / u_new_i <= s_max_i,
   // conservation: sum m_i (us_i + dt d_us_HO_i) = sum m_i (us_i + dt d_us_i).
   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us)
   {
      MFEM_ABORT("Product remap is not implemented for the chosen solver");
   }
};

class FluxBasedFCT : public FCTSolver
{
protected:
   const SparseMatrix &K, &M;
   const Array<int> &K_smap;

   // Temporary computation objects.
   mutable SparseMatrix flux_ij;
   mutable ParGridFunction gp, gm;

   const int iter_cnt;

   void ComputeFluxMatrix(const ParGridFunction &u, const Vector &du_ho,
                          SparseMatrix &flux_mat) const;
   void AddFluxesAtDofs(const SparseMatrix &flux_mat,
                        Vector &flux_pos, Vector &flux_neg) const;
   void ComputeFluxCoefficients(const Vector &u, const Vector &du_lo,
                                const Vector &m, const Vector &u_min, const Vector &u_max,
                                Vector &coeff_pos, Vector &coeff_neg) const;
   void UpdateSolutionAndFlux(const Vector &du_lo, const Vector &m,
                              ParGridFunction &coeff_pos, ParGridFunction &coeff_neg,
                              SparseMatrix &flux_mat, Vector &du) const;

public:
   FluxBasedFCT(ParFiniteElementSpace &space,
                SmoothnessIndicator *si, double delta_t,
                const SparseMatrix &adv_mat, const Array<int> &adv_smap,
                const SparseMatrix &mass_mat, int fct_iterations = 1)
      : FCTSolver(space, si, delta_t, true),
        K(adv_mat), M(mass_mat), K_smap(adv_smap), flux_ij(adv_mat),
        gp(&pfes), gm(&pfes), iter_cnt(fct_iterations) { }

   virtual void CalcFCTSolution(const ParGridFunction &u, const Vector &m,
                                const Vector &du_ho, const Vector &du_lo,
                                const Vector &u_min, const Vector &u_max,
                                Vector &du) const;

   virtual void CalcFCTProduct(const ParGridFunction &us, const Vector &m,
                               const Vector &d_us_HO, const Vector &d_us_LO,
                               Vector &s_min, Vector &s_max,
                               const Vector &u_new,
                               const Array<bool> &active_el,
                               const Array<bool> &active_dofs, Vector &d_us);
};

int GetLocalFaceDofIndex(int dim, int loc_face_id, int face_orient,
                         int face_dof_id, int face_dof1D_cnt);

void ExtractBdrDofs(int p, Geometry::Type gtype, DenseMatrix &dofs);

void GetMinMax(const ParGridFunction &g, double &min, double &max);

// Utility function to build a map to the offset of the symmetric entry in a
// sparse matrix.
Array<int> SparseMatrix_Build_smap(const SparseMatrix &A);

// Given a matrix K, matrix D (initialized with same sparsity as K) is computed,
// such that (K+D)_ij >= 0 for i != j.
void ComputeDiscreteUpwindingMatrix(const SparseMatrix &K,
                                    Array<int> smap, SparseMatrix& D);

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h,
                    const char *keys = NULL, bool vec = false);

void ComputeBoolIndicators(int NE, const Vector &u,
                           Array<bool> &ind_elem, Array<bool> &ind_dofs);

void ComputeRatio(int NE, const Vector &u_s, const Vector &u,
                  Vector &s, Array<bool> &bool_el, Array<bool> &bool_dof);

void ZeroOutEmptyDofs(const Array<bool> &ind_elem,
                      const Array<bool> &ind_dofs, Vector &u);


} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_REMHOS_TOOLS
