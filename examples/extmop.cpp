//                               ETHOS Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 square-disc.mesh2d
//               ex1 star.mesh2d
//
// Description: This example code performs a simple mesh smoothing based on a
//              topologically defined "mesh Laplacian" matrix.
//
//              The example highlights meshes with curved elements, the
//              assembling of a custom finite element matrix, the use of vector
//              finite element spaces, the definition of different spaces and
//              grid functions on the same mesh, and the setting of values by
//              iterating over the interior and the boundary elements.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>

using namespace mfem;

using namespace std;

// 1. Define the bilinear form corresponding to a mesh Laplacian operator. This
//    will be used to assemble the global mesh Laplacian matrix based on the
//    local matrix provided in the AssembleElementMatrix method. More examples
//    of bilinear integrators can be found in ../fem/bilininteg.hpp.
class VectorMeshLaplacianIntegrator : public BilinearFormIntegrator
{
private:
   int geom, type;
   LinearFECollection lfec;
   IsoparametricTransformation T;
   VectorDiffusionIntegrator vdiff;

public:
   VectorMeshLaplacianIntegrator(int type_) { type = type_; geom = -1; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual ~VectorMeshLaplacianIntegrator() { }
};

// 2. Implement the local stiffness matrix of the mesh Laplacian. This is a
//    block-diagonal matrix with each block having a unit diagonal and constant
//    negative off-diagonal entries, such that the row sums are zero.
void VectorMeshLaplacianIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   if (type == 0)
   {
      int dim = el.GetDim(); // space dimension
      int dof = el.GetDof(); // number of element degrees of freedom

      elmat.SetSize(dim*dof); // block-diagonal element matrix

      for (int d = 0; d < dim; d++)
         for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
               if (k==l)
               {
                  elmat (dof*d+k, dof*d+l) = 1.0;
               }
               else
               {
                  elmat (dof*d+k, dof*d+l) = -1.0/(dof-1);
               }
   }
   else
   {
      if (el.GetGeomType() != geom)
      {
         geom = el.GetGeomType();
         T.SetFE(lfec.FiniteElementForGeometry(geom));
         Geometries.GetPerfPointMat(geom, T.GetPointMat());
      }
      T.Attribute = Trans.Attribute;
      T.ElementNo = Trans.ElementNo;
      vdiff.AssembleElementMatrix(el, T, elmat);
   }
}


class HarmonicModel : public HyperelasticModel
{
public:
   virtual double EvalW(const DenseMatrix &J) const
   {
      return 0.5*(J*J);
   }

   virtual void EvalP(const DenseMatrix &J, DenseMatrix &P) const
   {
      P = J;
   }

   virtual void AssembleH(const DenseMatrix &J, const DenseMatrix &DS,
                          const double weight, DenseMatrix &A) const
   {
      int dof = DS.Height(), dim = DS.Width();

      for (int i = 0; i < dof; i++)
         for (int j = 0; j <= i; j++)
         {
            double a = 0.0;
            for (int d = 0; d < dim; d++)
            {
               a += DS(i,d)*DS(j,d);
            }
            a *= weight;
            for (int d = 0; d < dim; d++)
            {
               A(i+d*dof,j+d*dof) += a;
               if (i != j)
               {
                  A(j+d*dof,i+d*dof) += a;
               }
            }
         }
   }
};


#define BIG_NUMBER 1e+100 // Used when a matrix is outside the metric domain.
#define NBINS 25          // Number of intervals in the metric histogram.
#define GAMMA 0.9         // Used for composite metrics 73, 79, 80.
#define BETA0 0.01        // Used for adaptive pseudo-barrier metrics.
#define TAU0_EPS 0.001    // Used for adaptive shifted-barrier metrics.

class TMOPSmoother
{
private:
   HyperelasticModel *model;
   TargetJacobian *target;
   Mesh *mesh;
   GridFunction *nodes;
   Vector nodes0;

   // Structures used to compute the target matrix W.
   // Some are NULL or empty, depending on W_type.
   // TODO: move the target Jacobian computations in their own class.
   const GridFunction *target_nodes;
   DenseMatrix fixedWinv;
   Vector fixedWinv_mult;
   DenseMatrix Wpos;
   Vector WposV;
   DenseTensor Wdshape;
   Array<int> Wxdofs;

   // dim * zd is total number of dofs in an element.
   // dim * bd is total number of dofs in a boundary element.
   // TODO: assumes all cells are of the same type.
   int dim, NE, zd, bd;
   const int metric_id, max_iter_cnt;

   // Used in adaptive pseudo-barrier metrics.
   double beta;
   // Used in adaptive shifted-barrier metrics.
   double tau0;
   // Used in alignment metrics.
   double alpha_bar;

   // Sample points for metric evaluation in a cell.
   IntegrationRules integ_rules;
   const IntegrationRule *sample_pts;
   int nsp;

   // Temporary position data.
   DenseMatrix pos;
   Vector posV;
   Array<int> xdofs;

   // Jacobian / Ideal / Target matrices at a sample point.
   mutable DenseMatrix A, W, Winv, T, TtT;

   enum {IDEAL, IDEAL_EQ_SIZE, IDEAL_INIT_SIZE,
         IDEAL_CUSTOM_SIZE, TARGET_MESH, ALIGNED} W_type;
   enum {USE_T, USE_A} metric_matrix;
   enum {ALL, ONLY_2D} metric_dim;
   double (TMOPSmoother::*metric_value)() const;
   enum {NONE, HARD, ADAPT_PSEUDO, ADAPT_SHIFT} barrier_type;

   Table node_elem, bnode_nbr1, bnode_nbr2;

   // Checks if a node_id is on the boundary or not.
   // The indices go only over the x components.
   // Also used for fixing positions of dofs.
   Array<bool> is_dof_boundary;

   DenseMatrix shape;
   DenseTensor dshape;

   double NelderMead(int node_id);
   double PatternSearch1D(int node_id, int attr,
                          double pos_left, double pos_right);
   double NodeQuality(int node_id);
   // Print # sample points with detT < 0, detA < 0.
   // Prints min/max/avg/stdev of detT, detA, metric over all sample points.
   // Writes a histogram file showing # sample points per metric interval.
   void ComputeStats();
   double ComputeMinJacobian();
   double MinTangentLength();
   double MaxDistanceMoved();

   void ConstructFixedWinv();
   void ComputeWinv(int z_id, int ip_id, const DenseMatrix &pos);

   // Computes int(c) / volume / NE over the mesh.
   double ComputeAverage(Coefficient &c);

   ///
   /// Metrics.
   /// Notation: tau        = det(T),
   ///           alpha      = det(A),
   ///           phi+(a, b) = 0.5 (a + sqrt(a^2 + b^2)),
   ///           tau0  is an adaptive number, used for ADAPT_SHIFT barriers,
   ///           beta  is an adaptive number, used for ADAPT_PSEUDO barriers,
   ///           gamma is a constant.
   ///
   // |T|^2.
   double metric_1() const;
   // Condition number metric (with hard barrier at tau = 0).
   // 0.5 |T|^2 / tau - 1.
   double metric_2() const;
   // Pseudo-barrier condition number metric.
   // 0.5 |T|^2 / phi+(tau, beta) - 1.
   double metric_5() const;
   // |T - T^-t|^2.
   double metric_7() const;
   // Pseudo-barrier version of metric 7, assumes 2D.
   // (1 + 1 / phi+(tau, beta)^2) |T|^2 - 4 tau / phi+(tau, beta).
   double metric_8() const;
   // 1 / tau^2 |T - tau I|^2.
   double metric_15() const;
   // Shifted-barrier condition number metric.
   // 0.5 (|T|^2 - 2 tau) / (tau - tau0).
   double metric_22() const;
   // G-condition number metric.
   // |T^t T / tau|^2 - 2.
   double metric_50() const;
   // Pseudo-barrier G-condition number metric.
   // |T^t T / phi+(tau, beta)|^2 - 2.
   double metric_51() const;
   // Shifted-barrier G-condition number metric.
   // (|T^t T|^2 - 2 tau^2) / (tau - tau0)^2
   double metric_52() const;
   // |T^t T|^2.
   double metric_54() const;
   // (tau - 1)^2.
   double metric_55() const;
   // 0.5 (tau + 1/tau) - 1.
   double metric_56() const;
   // (1 - gamma)(|T|^2 - 2 tau) + gamma (tau - 1/tau)^2.
   double metric_73() const;
   // 0.5 (tau - 1/tau)^2.
   double metric_77() const;
   // (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1)^2.
   double metric_79() const;
   // (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1/tau)^2.
   double metric_80() const;
   // Shifted-barrier version of metric 7, assumes 2D.
   // (1 + 1 / (tau - tau0)^2) |T|^2 - 4 tau / (tau - tau0).
   double metric_82() const;
   // |A|^2.
   double metric_100() const;
   // |A^-1 - W^-1 |^2.
   double metric_102() const;

public:
   TMOPSmoother(HyperelasticModel *hm, TargetJacobian *tj,
                int metric_id_, int max_iter_cnt_);

   /**
    * @brief Init  Initializes the optimizer's internal structures.
    *
    * @param[in]     mesh        Mesh object used to initialize the metadata.
    * @param[in,out] opt_nodes   The node positions that will be optimized.
    *                            The input values are the starting positions.
    * @param[in]     fixed_dofs  Dofs that will not move during optimization.
    *                            The boundary is always fixed.
    */
   void Init(Mesh &mesh_, GridFunction &opt_nodes,
             const Array<int> *fixed_dofs = NULL);

   /// Used to change W in space and time. Assumes tmesh and the mesh given in
   /// Init() correspond to each other (same #zones, numbering, etc).
   void SetTargetMesh(const Mesh &target_mesh);

   /// Optimizes the mesh given by Init() using the current options.
   void Optimize();
};

TMOPSmoother::TMOPSmoother(HyperelasticModel *hm, TargetJacobian *tj,
                           int metric_id_, int max_iter_cnt_)
   : model(hm), target(tj), mesh(NULL), nodes(NULL), nodes0(),
     target_nodes(NULL), fixedWinv(), fixedWinv_mult(),
     Wpos(), WposV(), Wdshape(), Wxdofs(),
     dim(0), NE(0), zd(0), bd(0),
     metric_id(metric_id_),
     max_iter_cnt(max_iter_cnt_),
     beta(0.0), tau0(0.0), alpha_bar(0.0),
     integ_rules(0, Quadrature1D::GaussLobatto), sample_pts(NULL), nsp(0),
     pos(), posV(), xdofs(),
     A(), W(), Winv(), T(), TtT(),
     W_type(IDEAL),
     metric_matrix(USE_T),
     metric_dim(ALL),
     node_elem(), bnode_nbr1(), bnode_nbr2(),
     is_dof_boundary(),
     shape(), dshape()
{
   switch (metric_id)
   {
   case 1:
      metric_value = &TMOPSmoother::metric_1;
      barrier_type = NONE;
      break;
   case 2:
      metric_value = &TMOPSmoother::metric_2;
      barrier_type = HARD;
      break;
   case 5:
      metric_value = &TMOPSmoother::metric_5;
      barrier_type = ADAPT_PSEUDO;
      break;
   case 7:
      metric_value = &TMOPSmoother::metric_7;
      barrier_type = HARD;
      //W_type = IDEAL_EQ_SIZE;
      W_type = IDEAL_INIT_SIZE;
      break;
   case 8:
      metric_value = &TMOPSmoother::metric_8;
      barrier_type = ADAPT_PSEUDO;
      W_type = IDEAL_INIT_SIZE;
      metric_dim = ONLY_2D;
      break;
   case 15:
      metric_value = &TMOPSmoother::metric_15;
      barrier_type = HARD;
      W_type = ALIGNED;
      metric_dim = ONLY_2D;
      break;
   case 22:
      metric_value = &TMOPSmoother::metric_22;
      barrier_type = ADAPT_SHIFT;
      break;
   case 50:
      metric_value = &TMOPSmoother::metric_50;
      barrier_type = HARD;
      break;
   case 51:
      metric_value = &TMOPSmoother::metric_51;
      barrier_type = ADAPT_PSEUDO;
      break;
   case 52:
      metric_value = &TMOPSmoother::metric_52;
      barrier_type = ADAPT_SHIFT;
      break;
   case 54:
      metric_value = &TMOPSmoother::metric_54;
      barrier_type = NONE;
      break;
   case 55:
      metric_value = &TMOPSmoother::metric_55;
      barrier_type = NONE;
      W_type = IDEAL_EQ_SIZE;
      break;
   case 56:
      metric_value = &TMOPSmoother::metric_56;
      barrier_type = HARD;
      W_type = IDEAL_EQ_SIZE;
      break;
   case 73:
      metric_value = &TMOPSmoother::metric_73;
      barrier_type = HARD;
      W_type = IDEAL_EQ_SIZE;
      //W_type = IDEAL_CUSTOM_SIZE;
      break;
   case 77:
      metric_value = &TMOPSmoother::metric_77;
      barrier_type = HARD;
      W_type = IDEAL_EQ_SIZE;
      break;
   case 79:
      metric_value = &TMOPSmoother::metric_79;
      barrier_type = HARD;
      //W_type = IDEAL_EQ_SIZE;
      W_type = IDEAL_INIT_SIZE;
      //W_type = IDEAL_CUSTOM_SIZE;
      break;
   case 80:
      metric_value = &TMOPSmoother::metric_80;
      barrier_type = HARD;
      W_type = IDEAL_EQ_SIZE;
      break;
   case 82:
      metric_value = &TMOPSmoother::metric_82;
      barrier_type = ADAPT_SHIFT;
      W_type = IDEAL_EQ_SIZE;
      metric_dim = ONLY_2D;
      break;
   case 100:
      metric_value = &TMOPSmoother::metric_100;
      barrier_type = NONE; metric_matrix = USE_A;
      break;
   case 102:
      metric_value = &TMOPSmoother::metric_102;
      barrier_type = HARD; metric_matrix = USE_A;
      W_type = ALIGNED;
      metric_dim = ONLY_2D;
      break;
   default:
      MFEM_ABORT("Metric id " << metric_id << " is not implemented!");
   }

   if (dim == 3 && metric_dim == ONLY_2D)
   {
      MFEM_ABORT("Metric id " << metric_id << " is applicable only in 2D!");
   }
}

void TMOPSmoother::Init(Mesh &mesh_, GridFunction &opt_nodes,
                        const Array<int> *fixed_dofs)
{
   // Initialize and allocate fields.
   mesh = &mesh_;
   nodes = &opt_nodes;
   nodes0 = *nodes;
   dim = mesh->Dimension();
   NE  = mesh->GetNE();
   zd  = nodes->FESpace()->GetFE(0)->GetDof();
   bd  = nodes->FESpace()->GetBE(0)->GetDof();
   // Assumes all cells are of the same type.
   sample_pts = &integ_rules.Get(mesh->GetElementBaseGeometry(0), 8);
   nsp = sample_pts->GetNPoints();
   pos.SetSize(zd, dim); posV.SetDataAndSize(pos.Data(), zd * dim);
   xdofs.SetSize(dim * zd);
   A.SetSize(dim); W.SetSize(dim); Winv.SetSize(dim);
   T.SetSize(dim); TtT.SetSize(dim);

   cout << sample_pts->GetNPoints() << endl;

   // Fill Node-to-Element table, assuming one dof is in at most 20 zones.
   node_elem.SetSize(nodes->FESpace()->GetNDofs(), 20);
   Array<int> dofs(zd);
   for (int i = 0; i < NE; i++)
   {
      nodes->FESpace()->GetElementDofs(i, dofs);
      for (int j = 0; j < zd; j++)
      {
         node_elem.Push(dofs[j], i);
      }
   }
   node_elem.Finalize();

   // Fill (boundary node)-to-(neighbor boundary nodes) table.
   // Note that the row number is dof_id, while inside we put vdof_id.
   MFEM_VERIFY(dim == 2, "Boundary nodes can't move in 3D yet.");
   bnode_nbr1.SetSize(nodes->FESpace()->GetNDofs(), 2);
   bnode_nbr2.SetSize(nodes->FESpace()->GetNDofs(), 2);
   Array<int> vdofs;
   GridFunction &X = *nodes;
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const int attr = mesh->GetBdrElement(i)->GetAttribute();
      int offset;
      if (attr == 1)      { offset = 0; }  // y = 0, we use x components
      else if (attr == 2) { offset = bd; } // x = 0, we use y components.
      else                { continue; }

      nodes->FESpace()->GetBdrElementDofs(i, dofs);
      nodes->FESpace()->GetBdrElementVDofs(i, vdofs);
      for (int j = 0; j < bd; j++)
      {
         const double coord = X(vdofs[offset+j]);
         double minc = -numeric_limits<double>::infinity(),
                maxc =  numeric_limits<double>::infinity();
         int left = -1, right = -1;
         for (int d = 0; d < bd; d++)
         {
            const double d_coord = X(vdofs[offset+d]);
            if (j == d) { continue; }
            if (d_coord < coord && d_coord > minc)
            {
               left = d;
               minc = d_coord;
            }
            if (d_coord > coord && d_coord < maxc)
            {
               right = d;
               maxc  = d_coord;
            }
         }

         if (attr == 1)
         {
            if (left >= 0)  { bnode_nbr1.Push(dofs[j], vdofs[offset+left]); }
            if (right >= 0) { bnode_nbr1.Push(dofs[j], vdofs[offset+right]); }
         }
         else
         {
            if (left >= 0)  { bnode_nbr2.Push(dofs[j], vdofs[offset+left]);  }
            if (right >= 0) { bnode_nbr2.Push(dofs[j], vdofs[offset+right]); }
         }
      }
   }
   bnode_nbr1.Finalize();
   bnode_nbr2.Finalize();

   // Set fixed dofs. The boundary is always fixed.
   is_dof_boundary.SetSize(nodes->FESpace()->GetNDofs());
   is_dof_boundary = false;
   Array<int> bdofs(bd);
   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      const int attr = mesh->GetBdrElement(i)->GetAttribute();
      if (attr == 1 || attr == 2 || attr == 3)
      {
         nodes->FESpace()->GetBdrElementDofs(i, bdofs);
         for (int j = 0; j < bdofs.Size(); j++)
         {
            is_dof_boundary[bdofs[j]] = true;
         }
      }
   }
   if (fixed_dofs)
   {
      for (int j = 0; j < fixed_dofs->Size(); j++)
      {
         is_dof_boundary[(*fixed_dofs)[j]] = true;
      }
   }

   // Shape values and gradients at the sample points.
   // Assumes all cells are of the same type.
   const FiniteElement *fe = nodes->FESpace()->GetFE(0);
   shape.SetSize(fe->GetDof(), nsp);
   dshape.SetSize(fe->GetDof(), dim, nsp);
   Vector sh(fe->GetDof());
   for (int i = 0; i < nsp; i++)
   {
      fe->CalcShape(sample_pts->IntPoint(i), sh);
      for (int j = 0; j < sh.Size(); j++)
      {
         shape(j, i) = sh(j);
      }
      fe->CalcDShape(sample_pts->IntPoint(i), dshape(i));
   }
}

void TMOPSmoother::SetTargetMesh(const Mesh &tmesh)
{
   MFEM_VERIFY(W_type == TARGET_MESH,
               "The metric is not initialized with W_type = TARGET_MESH!");
   MFEM_VERIFY(tmesh.GetNE() == NE,
               "The target mesh has a different number of cells!");
   target_nodes = tmesh.GetNodes();
   int nd = target_nodes->FESpace()->GetFE(0)->GetDof();
   Wpos.SetSize(nd, dim);
   WposV.SetDataAndSize(Wpos.Data(), nd * dim);
   Wdshape.SetSize(nd, dim, nsp);
   const FiniteElement *fe = target_nodes->FESpace()->GetFE(0);
   for (int i = 0; i < nsp; i++)
   {
      fe->CalcDShape(sample_pts->IntPoint(i), Wdshape(i));
   }
   Wxdofs.SetSize(nd * dim);
}

void TMOPSmoother::Optimize()
{
   ConstructFixedWinv();
   MFEM_VERIFY(!(barrier_type == HARD && ComputeMinJacobian() < 0.0),
               "The metric can't be applied to an inverted initial mesh!");

   double tau0_init = max(0.0, ComputeMinJacobian());

   // Initial statistics.
   {
      if (barrier_type == ADAPT_PSEUDO)
      {
         double min_jac = ComputeMinJacobian();
         beta = (min_jac < 0.0) ? BETA0 * (1.0 - min_jac) : 0.0;
      }
      if (barrier_type == ADAPT_SHIFT)
      {
         tau0 = ComputeMinJacobian() - TAU0_EPS;
         if (tau0 > tau0_init) { tau0 = tau0_init; }
      }
      ComputeStats();
   }

   const int s = nodes->FESpace()->GetNDofs();
   const double tol = 1.0e-7;
   for (int i = 0; i < max_iter_cnt; i++)
   {
      if (barrier_type == ADAPT_PSEUDO)
      {
         double min_jac = ComputeMinJacobian();
         if (min_jac < 0.0)
         {
            cout << "pseudo" << endl;
            beta = BETA0 * (1.0 - min_jac);
         }
         else
         {
            cout << "hard" << endl;
            beta = 0.0;
            barrier_type = HARD;
         }
      }

      if (barrier_type == ADAPT_SHIFT)
      {
         tau0 = ComputeMinJacobian() - TAU0_EPS;
         if (tau0 > tau0_init) { tau0 = tau0_init; }
      }

      // Inner iteration - optimize every inner node position.
      double max_move = 0.0;
      for (int j = 0; j < s; j++)
      {
         if (!is_dof_boundary[j])
         {
            max_move = max(max_move, NelderMead(j));
         }
         else
         {
            if (bnode_nbr1.RowSize(j) == 2) // y = 0 boundary.
            {
               Array<int> vdofs(2);
               bnode_nbr1.GetRow(j, vdofs);
               const double c1 = (*nodes)(vdofs[0]), c2 = (*nodes)(vdofs[1]);
               max_move = max(max_move,
                              PatternSearch1D(j, 1, min(c1, c2), max(c1, c2)));
            }
            else if (bnode_nbr2.RowSize(j) == 2) // x = 0 boundary.
            {
               Array<int> vdofs(2);
               bnode_nbr2.GetRow(j, vdofs);
               const double c1 = (*nodes)(vdofs[0]), c2 = (*nodes)(vdofs[1]);
               max_move = max(max_move,
                              PatternSearch1D(j, 2, min(c1, c2), max(c1, c2)));
            }
         }
      }

      if (i % 1 == 0)
      {
         cout << "pass = " << i << " max move = " << max_move << "\n";
      }

      // Stop when all vertices have stopped moving.
      if (max_move < tol)
      {
         break;
      }
   }

   ComputeStats();
   cout << "min tangent length = " << MinTangentLength() << endl;
   cout << "max distance moved = " << MaxDistanceMoved() << endl;
}

inline static void SetPoint(vector<double *> &a, const Vector &b)
{
   for (int d = 0; d < b.Size(); d++)
   {
      *a[d] = b(d);
   }
}

inline static void SetSimplexPoint(vector<double *> &pos,
                                   const DenseMatrix &sim, int pnum)
{
   for (unsigned int d = 0; d < pos.size(); d++)
   {
      *pos[d] = sim(d, pnum);
   }
}

inline static void Inverse2D(const DenseMatrix &M, DenseMatrix &Minv)
{
   double detM = M(0,0) * M(1,1) - M(0,1) * M(1,0);
   Minv(0,0) =   M(1,1) / detM;
   Minv(0,1) = - M(1,0) / detM;
   Minv(1,0) = - M(0,1) / detM;
   Minv(1,1) =   M(0,0) / detM;
}

double TMOPSmoother::NelderMead(int node_id)
{
   // Nelder-Mead parameters.
   const int max_steps = 40;
   const double alp    = 1.0; // Reflection coefficient.
   const double gamma  = 2.0; // Expansion coefficient.
   const double rho    = 0.5; // Contraction coefficient.
   const double sigma  = 0.5; // Reduction coefficient.

   // This determines the size of the initial simplex; it should give a
   // cell-size length scale.
   double size = std::numeric_limits<double>::infinity();
   Array<int> elems(node_elem.RowSize(node_id));
   node_elem.GetRow(node_id, elems);
   int own_nodes = true; // Doesn't matter.
   mesh->SwapNodes(nodes, own_nodes);
   for (int i = 0; i < elems.Size(); i++)
   {
      // This is pow(|det(J)|, 1/dim) at the center.
      size = min(size, mesh->GetElementSize(elems[i], 0));
   }
   mesh->SwapNodes(nodes, own_nodes);

   // Number of points in the simplex;
   const int p_cnt = dim + 1;

   std::vector<double *> node_pos(dim);
   for (int d = 0; d < dim; d++)
   {
      node_pos[d] = &((*nodes)(nodes->FESpace()->DofToVDof(node_id, d)));
   }
   Vector init_pos(dim);
   for (int d = 0; d < dim; d++) { init_pos(d) = *node_pos[d]; }

   // Initial trial locations (simplex with length 1.0 for each edge).
   // Columns are coordinates.
   const double rt3 = sqrt(3), rt6 = sqrt(6);
   DenseMatrix simplex(dim, p_cnt);
   {
      simplex.SetCol(0, init_pos);

      simplex(0, 1) = init_pos(0) - 0.5 * rt3 * size;
      simplex(1, 1) = init_pos(1) + 0.5       * size;
      if (dim == 3) { simplex(2, 1) = init_pos(2); }

      simplex(0, 2) = init_pos(0) - 0.5 * rt3 * size;
      simplex(1, 2) = init_pos(1) - 0.5       * size;
      if (dim == 3) { simplex(2, 2) = init_pos(2); }

      if (dim == 3)
      {
         simplex(0, 1) = init_pos(0) - rt3 / 3.0 * size;
         simplex(1, 1) = init_pos(1);
         simplex(2, 1) = init_pos(2) + rt6 / 3.0 * size;
      }
   }

   // Evaluate at all trial locations.
   Vector nq(p_cnt);
   for (int i = 0; i < dim+1; i++)
   {
      SetSimplexPoint(node_pos, simplex, i);
      nq(i) = NodeQuality(node_id);
   }

   double dist, nqR, nqE, nqC;
   Vector pointM(dim), pointR(dim), pointE(dim), pointC(dim);
   for (int i = 0; true; i++)
   {
      dist = 0.0;
      for (int d = 0; d < dim; d++)
      {
         dist += (simplex(d, 0) - init_pos(d)) * (simplex(d, 0) - init_pos(d));
      }
      dist = sqrt(dist);

      if ((dist != 0.0 && dist < 1.0e-2 * size) || (i > max_steps))
      {
         SetSimplexPoint(node_pos, simplex, 0);
         return dist;
      }

      for (int d = 0; d < dim; d++)
      {
         init_pos(d) = simplex(d, 0);
      }

      // Bubble sort - the best point (which has smallest nq) will be index 0.
      bool swapped = true;
      while(swapped)
      {
         swapped = false;
         for (int i = 1; i < p_cnt; i++)
         {
            if (nq(i-1) > nq(i))
            {
               swapped = true;
               swap(nq(i-1), nq(i));
               for (int d = 0; d < dim; d++)
               {
                  swap(simplex(d, i-1), simplex(d, i));
               }
            }
         }
      }

      // Centroid of all points except the last.
      pointM = 0.0;
      for (int d = 0; d < dim; d++)
      {
         for (int i = 0; i < p_cnt - 1; i++)
         {
            pointM(d) += simplex(d, i);
         }
         pointM(d) /= dim;
      }

      // Reflection of the centroid w.r.t. the last point.
      for (int d = 0; d < dim; d++)
      {
         pointR(d) = pointM(d) + alp * (pointM(d) - simplex(d, dim));
      }
      SetPoint(node_pos, pointR);
      nqR = NodeQuality(node_id);
      // Better than second worst, but not better than best -> replace worst.
      if (nq(0) <= nqR && nqR < nq(p_cnt-2))
      {
         simplex.SetCol(p_cnt-1, pointR); nq(p_cnt-1) = nqR;
         continue;
      }

      // Expansion.
      if (nqR < nq(0))
      {
         for (int d = 0; d < dim; d++)
         {
            pointE(d) = pointM(d) + gamma * (pointM(d) - simplex(d, p_cnt-1));
         }
         SetPoint(node_pos, pointE);
         nqE = NodeQuality(node_id);

         if (nqE < nqR)
         {
            simplex.SetCol(p_cnt-1, pointE); nq(p_cnt-1) = nqE;
            continue;
         }
         else
         {
            simplex.SetCol(p_cnt-1, pointR); nq(p_cnt-1) = nqR;
            continue;
         }
      }

      // Contraction.
      if (nq(p_cnt-2) <= nqR && nqR < nq(p_cnt-1))
      {
         for (int d = 0; d < dim; d++)
         {
            pointC(d) = pointM(d) + rho * (pointR(d) - pointM(d));
         }
         SetPoint(node_pos, pointC);
         nqC = NodeQuality(node_id);

         if (nqC < nqR)
         {
            simplex.SetCol(p_cnt-1, pointC); nq(p_cnt-1) = nqC;
         }
         else
         {
            // Reduction / shrink - replace all but the best.
            for (int i = 1; i < p_cnt; i++)
            {
               for (int d = 0; d < dim; d++)
               {
                  simplex(d, i) = simplex(d, 0) +
                                  sigma * (simplex(d, i) - simplex(d, 0));
               }
               SetSimplexPoint(node_pos, simplex, i);
               nq(i) = NodeQuality(node_id);
            }
         }
         continue;
      }
      if (nqR >= nq(dim))
      {
         for (int d = 0; d < dim; d++)
         {
            pointC(d) = pointM(d) + rho * (simplex(d, p_cnt-1) - pointM(d));
         }
         SetPoint(node_pos, pointC);
         nqC = NodeQuality(node_id);

         if (nqC < nq(p_cnt-1))
         {
            simplex.SetCol(p_cnt-1, pointC); nq(p_cnt-1) = nqC;
         }
         else
         {
            // Reduction / shrink - replace all but the best.
            for (int i = 1; i < p_cnt; i++)
            {
               for (int d = 0; d < dim; d++)
               {
                  simplex(d, i) = simplex(d, 0) +
                                  sigma * (simplex(d, i) - simplex(d, 0));
               }
               SetSimplexPoint(node_pos, simplex, i);
               nq(i) = NodeQuality(node_id);
            }
         }
         continue;
      }
      continue;
   }
}

double TMOPSmoother::PatternSearch1D(int node_id, int attr,
                                     double pos_left, double pos_right)
{
   MFEM_VERIFY(dim == 2, "This function works only with 2D meshes!");

   const int max_iter = 32;
   const double shrink_factor = 0.8;

   const double quality0 = NodeQuality(node_id);
   double quality, min_quality = quality0;

   double *pos;
   if (attr == 1)
   {
      // Change x coordinate.
      pos = &((*nodes)(nodes->FESpace()->DofToVDof(node_id, 0)));
   }
   else
   {
      // Change y coordinate.
      pos = &((*nodes)(nodes->FESpace()->DofToVDof(node_id, 1)));
   }
   const double pos0 = *pos;
   double min_pos = pos0;

   double dleft  = 0.99999 * (pos0 - pos_left);
   double dright = 0.99999 * (pos_right - pos0);

   for (int i = 0; i < max_iter; i++)
   {
      // Check left side.
      *pos    = pos0 - dleft;
      quality = NodeQuality(node_id);
      if (quality < min_quality)
      {
         min_quality = quality;
         min_pos     = *pos;
      }

      // Check right side.
      *pos = pos0 + dright;
      quality = NodeQuality(node_id);
      if (quality < min_quality)
      {
         min_quality = quality;
         min_pos     = *pos;
      }

      // None of the sides was better than the starting point -> shrink.
      if (min_quality >= quality0)
      {
          dleft  *= shrink_factor;
          dright *= shrink_factor;
      }
      else
      {
         MFEM_VERIFY(pos_left < min_pos && min_pos < pos_right,
                     "Boundary self-intersected.");
         *pos = min_pos;
         return fabs(pos0 - min_pos);
      }
   }

   *pos = pos0;
   return 0.0;
}

double TMOPSmoother::NodeQuality(int node_id)
{
   Array<int> elems(node_elem.RowSize(node_id));
   node_elem.GetRow(node_id, elems);

   DenseTensor Ws(dim, dim, nsp);

   double res = 0.0;
   for (int i = 0; i < elems.Size(); i++)
   {
      nodes->FESpace()->GetElementVDofs(elems[i], xdofs);
      nodes->GetSubVector(xdofs, posV);
      if (W_type == TARGET_MESH)
      {
         target_nodes->FESpace()->GetElementVDofs(elems[i], Wxdofs);
         target_nodes->GetSubVector(Wxdofs, WposV);
      }

      if (target)
      {
         target->ComputeElementTargets(i, *(nodes->FESpace()->GetFE(i)),
                                       *sample_pts, Ws);
      }
      for (int j = 0; j < nsp; j++)
      {
         MultAtB(pos, dshape(j), A);
         ComputeWinv(elems[i], j, Wpos);
         Mult(A, Winv, T);

         if (model)
         {
            model->SetTargetJacobian(Ws(j));
            res += sample_pts->IntPoint(j).weight * model->EvalW(T);
         }
         else
         {
            res += sample_pts->IntPoint(j).weight * (this->*metric_value)();
         }
      }
   }
   res /= nsp;
   return res;
}

void TMOPSmoother::ComputeStats()
{
   int T_neg, A_neg;
   double m_min, m_max, m_avg, m_stdev,
          T_min, T_max, T_avg, T_stdev,
          A_min, A_max, A_avg, A_stdev;

   Vector detA(NE * nsp), detT(NE * nsp), mval(NE * nsp);

   T_neg = A_neg = 0;
   m_min = T_min = A_min = numeric_limits<double>::infinity();
   m_max = T_max = A_max = -numeric_limits<double>::infinity();
   m_avg = T_avg = A_avg = 0.0;
   int n_valid_pts = 0;
   for (int i = 0; i < NE; i++)
   {
      nodes->FESpace()->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      if (W_type == TARGET_MESH)
      {
         target_nodes->FESpace()->GetElementVDofs(i, Wxdofs);
         target_nodes->GetSubVector(Wxdofs, WposV);
      }

      for (int j = 0; j < nsp; j++)
      {
         MultAtB(pos, dshape(j), A);
         ComputeWinv(i, j, Wpos);
         Mult(A, Winv, T);

         double m = (this->*metric_value)();
         mval(i * nsp + j) = m;
         if (m != BIG_NUMBER)
         {
            n_valid_pts++;
            m_min = min(m_min, m);
            m_max = max(m_max, m);
            m_avg += m;
         }

         double det = T.Det();
         if (det < 0.0) { T_neg++; }
         T_min =  min(T_min, det);
         T_max =  max(T_max, det);
         T_avg += det;
         detT(i * nsp + j) = det;

         det = A.Det();
         if (det < 0.0) { A_neg++; }
         A_min =  min(A_min, det);
         A_max =  max(A_max, det);
         A_avg += det;
         detA(i * nsp + j) = det;
      }
   }

   int np = NE * nsp;
   m_avg /= n_valid_pts;
   T_avg /= np;
   A_avg /= np;
   m_stdev = T_stdev = A_stdev = 0.0;
   for (int i = 0; i < np; i++)
   {
      if (mval(i) != BIG_NUMBER)
      {
         m_stdev += (mval(i) - m_avg) * (mval(i) - m_avg);
      }
      T_stdev += (detT(i) - T_avg) * (detT(i) - T_avg);
      A_stdev += (detA(i) - A_avg) * (detA(i) - A_avg);
   }
   m_stdev = sqrt(m_stdev / n_valid_pts);
   T_stdev = sqrt(T_stdev / np);
   A_stdev = sqrt(A_stdev / np);

   cout << "mval_min = " << m_min << endl
        << "mval_max = " << m_max << endl << "mval_avg = " << m_avg << endl
        << "mval_stdev = " << m_stdev << endl;
   cout << "detT_neg = " << T_neg << endl << "detT_min = " << T_min << endl
        << "detT_max = " << T_max << endl << "detT_avg = " << T_avg << endl
        << "detT_stdev = " << T_stdev << endl;
   cout << "detA_neg = " << A_neg << endl << "detA_min = " << A_min << endl
        << "detA_max = " << A_max << endl << "detA_avg = " << A_avg << endl
        << "detA_stdev = " << A_stdev << endl;

   // Sort mval and drop the BIG_NUMBERs.
   std::vector<double> std_mval(mval.GetData(), mval.GetData() + n_valid_pts);
   sort(std_mval.begin(), std_mval.end());
   mval.NewDataAndSize(std_mval.data(), n_valid_pts);
   // Write the histogram file.
   ofstream hfile("metric_histogram");
   double bin_width = mval(n_valid_pts - 1) / NBINS;
   double bin_max = bin_width;
   int num_in_bin = 0;
   int i = 0;
   while (i < n_valid_pts)
   {
      if (mval(i) >= bin_max)
      {
         hfile << bin_max - bin_width << " " << num_in_bin << endl
               << bin_max             << " " << num_in_bin << endl;
         num_in_bin = 0;
         bin_max += bin_width;
      }
      else
      {
         num_in_bin++;
         i++;
      }
   }
   hfile << bin_max - bin_width << " " << 0 << endl;

   // Print percentile stats.
   cout << "\tTotal # locations = " << np
        << ", # valid locations = " << n_valid_pts << endl;
   cout << "\t0 locations (i.e., 0.0%) have quality larger than = "
        << mval[n_valid_pts - 1] << endl;
   const double ds[] = {0.995, 0.99, 0.975, 0.95, 0.90, 0.75, 0.50, 0.25};
   for (int i = 0; i < 8; i ++)
   {
      int id = ds[i] * n_valid_pts;
      cout << "\t" << n_valid_pts - id
           << " locations (i.e., " << (1.0-ds[i])*100
           << "%) have quality larger than = " << mval[id] << endl;
   }
}

double TMOPSmoother::ComputeMinJacobian()
{
   double min_det = numeric_limits<double>::infinity();
   for (int i = 0; i < NE; i++)
   {
      nodes->FESpace()->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      if (W_type == TARGET_MESH)
      {
         target_nodes->FESpace()->GetElementVDofs(i, Wxdofs);
         target_nodes->GetSubVector(Wxdofs, WposV);
      }

      for (int j = 0; j < nsp; j++)
      {
         MultAtB(pos, dshape(j), A);
         if (metric_matrix == USE_A)
         {
            min_det = min(min_det, A.Det());
         }
         else
         {
            ComputeWinv(i, j, Wpos);
            Mult(A, Winv, T);

            min_det = min(min_det, T.Det());
         }
      }
   }
   return min_det;
}

double TMOPSmoother::MinTangentLength()
{
   double min_tangent = numeric_limits<double>::infinity();
   for (int i = 0; i < NE; i++)
   {
      nodes->FESpace()->GetElementVDofs(i, xdofs);
      nodes->GetSubVector(xdofs, posV);
      for (int j = 0; j < nsp; j++)
      {
         MultAtB(pos, dshape(j), A);
         // TODO: 3D. This just copies Pat's code.
         double len1 = sqrt(A(0,0)*A(0,0) + A(1,0)*A(1,0));
         double len2 = sqrt(A(0,1)*A(0,1) + A(1,1)*A(1,1));
         double min_length = (len1 > len2) ? len2 : len1;
         min_tangent = min(min_tangent, min_length);
      }
   }
   return min_tangent;
}

double TMOPSmoother::MaxDistanceMoved()
{
   double max_dist = -numeric_limits<double>::infinity();
   int nd = nodes->FESpace()->GetNDofs();
   for (int i = 0; i < nd; i++)
   {
      double x = (*nodes)(nodes->FESpace()->DofToVDof(i, 0));
      double y = (*nodes)(nodes->FESpace()->DofToVDof(i, 1));
      double x0 = nodes0(nodes->FESpace()->DofToVDof(i, 0));
      double y0 = nodes0(nodes->FESpace()->DofToVDof(i, 1));
      max_dist = max(max_dist, sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) ));
   }
   return max_dist;
}

void TMOPSmoother::ConstructFixedWinv()
{
   if (W_type == TARGET_MESH) { return; }

   fixedWinv.SetSize(dim);
   DenseMatrix &W = fixedWinv;

   // Transformation to ideal target.
   double rt3 = sqrt(3.0), rt6 = sqrt(6.0);
   if (mesh->GetElementBaseGeometry(0) == Geometry::SQUARE ||
       mesh->GetElementBaseGeometry(0) == Geometry::CUBE)
   {
      W = 0.0;
      for (int i = 0; i < dim; i++) { W(i, i) = 1.0; }
   }
   else if (mesh->GetElementBaseGeometry(0) == Geometry::TRIANGLE)
   {
      W(0, 0) = 1.0; W(0, 1) = 0.5;
      W(1, 0) = 0.0; W(1, 1) = 0.5*rt3;
   }
   else if (mesh->GetElementBaseGeometry(0) == Geometry::TETRAHEDRON)
   {
      W(0, 0) = 1.0; W(0, 1) = 0.5;     W(0, 2) = 0.5;
      W(1, 0) = 0.0; W(1, 1) = 0.5*rt3; W(1, 2) = 0.5*rt3;
      W(2, 0) = 0.0; W(2, 1) = 0.0;     W(2, 2) = rt6/3.0;
   }

   if (W_type == IDEAL_EQ_SIZE)
   {
      // Sqrt of (average cell area) / (area of ideal element).
      ConstantCoefficient one(1.0);
      W *= sqrt(ComputeAverage(one) / W.Det());
   }
   else if (W_type == IDEAL_INIT_SIZE)
   {
      fixedWinv_mult.SetSize(NE * nsp);

      for (int i = 0; i < NE; i++)
      {
         nodes->FESpace()->GetElementVDofs(i, xdofs);
         nodes->GetSubVector(xdofs, posV);
         for (int j = 0; j < nsp; j++)
         {
            MultAtB(pos, dshape(j), A);
            double alpha = A.Det();
            MFEM_VERIFY(alpha > 0.0, "Initial mesh is inverted!");
            fixedWinv_mult(i*nsp + j) = 1.0 / sqrt(alpha);
         }
      }
      W *= sqrt(1.0 / W.Det());
   }
   else if (W_type == IDEAL_CUSTOM_SIZE)
   {
      fixedWinv_mult.SetSize(NE * nsp);

      class SmallAround00 : public Coefficient
      {
      public:
         double Eval(ElementTransformation &T,
                     const IntegrationPoint &ip)
         {
            Vector c(2);
            T.Transform(ip, c);
            return 0.1 + c(1) * c(1);
         }
      };

      // Define custom function f here.
      //ConstantCoefficient f(2.0);
      SmallAround00 f;
      double f_avg = ComputeAverage(f);
      ConstantCoefficient one(1.0);
      double alpha_avg = ComputeAverage(one);

      W *= sqrt(alpha_avg / f_avg / W.Det());

      for (int i = 0; i < NE; i++)
      {
         ElementTransformation *T =
               nodes->FESpace()->GetElementTransformation(i);
         for (int j = 0; j < nsp; j++)
         {
            fixedWinv_mult(i*nsp + j) =
                  1.0 / sqrt(f.Eval(*T, sample_pts->IntPoint(j)));
         }
      }
   }
   else if (W_type == ALIGNED)
   {
      ConstantCoefficient one(1.0);
      alpha_bar = ComputeAverage(one);
      return;
   }

   DenseMatrixInverse dmi(W);
   dmi.GetInverseMatrix(W);
}

void TMOPSmoother::ComputeWinv(int z_id, int ip_id, const DenseMatrix &pos)
{
   switch (W_type)
   {
      case IDEAL:
      case IDEAL_EQ_SIZE:
         Winv = fixedWinv; return;
      case IDEAL_INIT_SIZE:
      case IDEAL_CUSTOM_SIZE:
         Winv  = fixedWinv;
         Winv *= fixedWinv_mult(z_id * nsp + ip_id); return;
      case TARGET_MESH:
      {
         MultAtB(pos, Wdshape(ip_id), Winv);
         DenseMatrixInverse dmi(Winv);
         dmi.GetInverseMatrix(Winv); return;
      }
      case ALIGNED:
      {
         class VectorField : public VectorCoefficient
         {
         public:
            VectorField(int vdim) : VectorCoefficient(vdim)
            { }
            void Eval(Vector &V, ElementTransformation &T,
                      const IntegrationPoint &ip)
            {
               Vector c(2);
               T.Transform(ip, c);
               double theta = M_PI * c(1) * (1.0 - c(1)) * cos(2 * M_PI * c(0));
               V.SetSize(2);
               V(0) = cos(theta);
               V(1) = sin(theta);
            }
         } vf(dim);

         ElementTransformation *T =
               nodes->FESpace()->GetElementTransformation(z_id);
         Vector uv;

         vf.Eval(uv, *T, sample_pts->IntPoint(ip_id));
         W(0, 0) = uv(0); W(0, 1) = -uv(1);
         W(1, 0) = uv(1); W(1, 1) =  uv(0);
         W *= sqrt (alpha_bar) / uv.Norml2();
         DenseMatrixInverse inv(W);
         inv.GetInverseMatrix(Winv);
         return;
      }
   }
   MFEM_ABORT("Unknown target construction method!");
}

double TMOPSmoother::ComputeAverage(Coefficient &coeff)
{
   LinearForm lf(new FiniteElementSpace(mesh, new L2_FECollection(0, dim)));
   lf.AddDomainIntegrator(new DomainLFIntegrator(coeff, sample_pts));
   lf.Assemble();
   return lf.Sum() / NE;
}

// |T|^2.
double TMOPSmoother::metric_1() const
{
   double fnorm = T.FNorm();
   return fnorm * fnorm;
}

// 0.5 |T|^2 / det(T) - 1.
double TMOPSmoother::metric_2() const
{
   if (T.Det() <= 0.0) { return BIG_NUMBER; }
   double fnorm = T.FNorm();
   return 0.5 * fnorm * fnorm / T.Det() - 1.0;
}

// 0.5 |T|^2 / phi+(det(T), beta) - 1; phi+(a, b) = 0.5 (a + sqrt(a^2 + b^2)).
double TMOPSmoother::metric_5() const
{
   double tau = T.Det();
   // This metric switches to barrier_type = HARD
   // once the mesh becomes non-inverted.
   if (barrier_type == HARD && tau <= 0.0) { return BIG_NUMBER; }
   double phi = 0.5 * (tau + sqrt (tau*tau + beta*beta));
   double fnorm = T.FNorm();
   return 0.5 * fnorm * fnorm / phi - 1.0;
}

// |T - T^-t|^2.
double TMOPSmoother::metric_7() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   DenseMatrix M(dim);
   if (dim == 2)
   {
      M(0,0) = T(0,0) - T(1,1) / tau;
      M(0,1) = T(0,1) + T(1,0) / tau;
      M(1,0) = T(1,0) + T(0,1) / tau;
      M(1,1) = T(1,1) - T(0,0) / tau;
   }
   else
   {
      // TODO - write the entries directly from T.
      DenseMatrixInverse DMI(T);
      DMI.GetInverseMatrix(M);
      M.Transpose();
      M *= -1.0;
      M += const_cast<DenseMatrix &>(T);
   }
   double fnorm = M.FNorm();
   return fnorm * fnorm;
}

// (1 + 1 / phi+(tau, beta)^2) |T|^2 - 4 tau / phi+(tau, beta).
double TMOPSmoother::metric_8() const
{
   double tau = T.Det();
   // This metric switches to barrier_type = HARD
   // once the mesh becomes non-inverted.
   if (barrier_type == HARD && tau <= 0.0) { return BIG_NUMBER; }
   double phi = 0.5 * (tau + sqrt (tau*tau + beta*beta));
   double fnorm = T.FNorm();
   return (1.0 + 1.0 / (phi * phi)) * fnorm * fnorm - 4.0 * tau / phi;
}

// 1 / tau^2 |T - tau I|^2.
double TMOPSmoother::metric_15() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }

   // This metric is 2D only.
   double tau_sq = tau * tau;
   return ( (T(0,0) - tau) * (T(0,0) - tau) + T(0,1) * T(0,1) +
            (T(1,1) - tau) * (T(1,1) - tau) + T(1,0) * T(1,0)   ) / tau_sq;
}

// 0.5 (|T|^2 - 2 det(T)) / (det(T) - tau0).
double TMOPSmoother::metric_22() const
{
   double det = T.Det();
   if (det - tau0 <= 0.0) { return BIG_NUMBER; }
   double fnorm = T.FNorm();
   return 0.5 * (fnorm * fnorm - 2.0 * det) / (det - tau0);
}

// |T^t T / tau|^2 - 2.
double TMOPSmoother::metric_50() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   MultAAt(T, TtT);
   double fnorm = TtT.FNorm();
   return fnorm * fnorm / (tau * tau) - 2.0;
}

// |T^t T / phi+(tau, beta)|^2 - 2.
double TMOPSmoother::metric_51() const
{
   double tau = T.Det();
   // This metric switches to barrier_type = HARD
   // once the mesh becomes non-inverted.
   if (barrier_type == HARD && tau <= 0.0) { return BIG_NUMBER; }
   double phi = 0.5 * (tau + sqrt (tau*tau + beta*beta));
   MultAAt(T, TtT);
   double fnorm = TtT.FNorm();
   return fnorm * fnorm / (phi * phi) - 2.0;
}

// (|T^t T|^2 - 2 tau^2) / (tau - tau0)^2
double TMOPSmoother::metric_52() const
{
   double tau = T.Det();
   if (tau - tau0 <= 0.0) { return BIG_NUMBER; }
   MultAAt(T, TtT);
   double fnorm = TtT.FNorm();
   return (fnorm*fnorm - 2.0*tau*tau) / ( (tau-tau0)*(tau-tau0) );
}

// |T^t T|^2.
double TMOPSmoother::metric_54() const
{
   MultAAt(T, TtT);
   double fnorm = TtT.FNorm();
   return fnorm * fnorm;
}

// (tau - 1)^2.
double TMOPSmoother::metric_55() const
{
   double tau = T.Det();
   return (tau - 1.0) * (tau - 1.0);
}

// 0.5 (tau + 1/tau) - 1.
double TMOPSmoother::metric_56() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   return 0.5 * (tau + 1.0 / tau) - 1.0;
}

// (1 - gamma)(|T|^2 - 2 tau) + gamma (tau - 1/tau)^2.
double TMOPSmoother::metric_73() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   double fnorm = T.FNorm();
   double diff = tau - 1.0 / tau;
   return (1.0 - GAMMA) * (fnorm * fnorm - 2.0 * tau) + GAMMA * diff * diff;
}

// 0.5 (tau - 1/tau)^2.
double TMOPSmoother::metric_77() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   double tmp = (tau - 1.0 / tau);
   return 0.5 * tmp * tmp;
}

// (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1)^2.
double TMOPSmoother::metric_79() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   double fnorm = T.FNorm();
   return (1.0 - GAMMA) * (0.5 * fnorm * fnorm / tau - 1.0) +
          GAMMA * (tau - 1.0) * (tau - 1.0);
}

// (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1/tau)^2.
double TMOPSmoother::metric_80() const
{
   double tau = T.Det();
   if (tau <= 0.0) { return BIG_NUMBER; }
   double fnorm = T.FNorm();
   double diff = tau - 1.0 / tau;
   return (1.0 - GAMMA) * (0.5 * fnorm * fnorm / tau - 1.0) +
          GAMMA * diff * diff;
}

// (1 + 1 / (tau - tau0)^2) |T|^2 - 4 tau / (tau - tau0).
double TMOPSmoother::metric_82() const
{
   double tau = T.Det();
   if (tau - tau0 <= 0.0) { return BIG_NUMBER; }
   double diff = tau - tau0;
   double fnorm = T.FNorm();
   return (1.0 + 1.0 / (diff * diff)) * fnorm * fnorm - 4.0 * tau / diff;
}

// |A|^2.
double TMOPSmoother::metric_100() const
{
   double fnorm = A.FNorm();
   return fnorm * fnorm;
}

// |A^-1 - W^-1 |^2.
double TMOPSmoother::metric_102() const
{
   double alpha = A.Det();
   if (alpha <= 0.0) { return BIG_NUMBER; }

   DenseMatrix M(dim);
   Inverse2D(A, M);
   M -= Winv;
   double fnorm = M.FNorm();
   return fnorm * fnorm;
}


void compute_window(
   const string& window,
   int N,
   Vector& wn)
{
   double a,b,c;

   if ("rectangular" == window)
   {
      a = 1.0;
      b = 0.0;
      c = 0.0;
   }
   else if ("hanning" == window)
   {
      a = 0.5;
      b = 0.5;
      c = 0.0;
   }
   else if ("hamming" == window)
   {
      a = 0.54;
      b = 0.46;
      c = 0.0;
   }
   else if ("blackman" == window)
   {
      a = 0.42;
      b = 0.50;
      c = 0.08;
   }
   else
   {
      printf("window unrecognized: %s\n",window.c_str());
      exit(1);
   }

   for (int i = 0; i <= N; i++)
   {
      double t = (i*M_PI)/(N+1);
      wn[i] = a + b*cos(t) +c*cos(2*t);
   }

}

void compute_chebyshev_coeffs(
   int N,
   Vector& fn,
   double k_pb)
{
   double theta_pb = acos(1.0 -0.5*k_pb);

   // This sigma offset value can (should) be optimized as a function
   // of N, kpb, and the window function.  This is a good value for
   // N=10, kpb = 0.1, and a Hamming window.

   double sigma = 0.482414167;

   fn[0] = (theta_pb +sigma)/M_PI;
   for (int i = 1; i <= N; i++)
   {
      double t = i*(theta_pb+sigma);
      fn[i] = 2.0*sin(t)/(i*M_PI);
   }
}


int main (int argc, char *argv[])
{
   Mesh *mesh;
   char vishost[] = "localhost";
   int  visport   = 19916;
   int  ans;
    vector<double> logvec (10);

   bool dump_iterations = false;

   if (argc == 1)
   {
      cout << "Usage: ex1 <mesh_file>" << endl;
      return 1;
   }

    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    const char *mesh_file = "../data/tipton.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);
    
    mesh = new Mesh(mesh_file, 1, 1);

   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 1000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      cout << "enter ref. levels [" << ref_levels << "] --> " << flush;
      cin >> ref_levels;
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
       logvec[0]=ref_levels;
   }

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements which are tensor products of quadratic finite elements. The
   //    dimensionality of the vector finite element space is specified by the
   //    last parameter of the FiniteElementSpace constructor.
   cout << "Mesh curvature: ";
   if (mesh->GetNodes())
   {
      cout << mesh->GetNodes()->OwnFEC()->Name();
   }
   else
   {
      cout << "(NONE)";
   }
   cout << endl;

   int mesh_poly_deg = 1;
   cout <<
        "Enter polynomial degree of mesh finite element space:\n"
        "0) QuadraticPos (quads only)\n"
        "p) Degree p >= 1\n"
        " --> " << flush;
   cin >> mesh_poly_deg;
   FiniteElementCollection *fec;
   if (mesh_poly_deg <= 0)
   {
      fec = new QuadraticPosFECollection;
      mesh_poly_deg = 2;
   }
   else
   {
      fec = new H1_FECollection(mesh_poly_deg, dim);
   }
    logvec[1]=mesh_poly_deg;
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);

   // 6. Make the mesh curved based on the above finite element space. This
   //    means that we define the mesh elements through a fespace-based
   //    transformation of the reference element.
   mesh->SetNodalFESpace(fespace);

   // 7. Set up the right-hand side vector b. In this case we do not need to use
   //    a LinearForm object because b=0.
   Vector b(fespace->GetVSize());
   b = 0.0;

   // 8. Get the mesh nodes (vertices and other quadratic degrees of freedom in
   //    the finite element space) as a finite element grid function in fespace.
   GridFunction *x;
   x = mesh->GetNodes();

   // 9. Define a vector representing the minimal local mesh size in the mesh
   //    nodes. We index the nodes using the scalar version of the degrees of
   //    freedom in fespace.
   Vector h0(fespace->GetNDofs());
   h0 = numeric_limits<double>::infinity();
   {
      Array<int> dofs;
      // loop over the mesh elements
      for (int i = 0; i < fespace->GetNE(); i++)
      {
         // get the local scalar element degrees of freedom in dofs
         fespace->GetElementDofs(i, dofs);
         // adjust the value of h0 in dofs based on the local mesh size
         for (int j = 0; j < dofs.Size(); j++)
         {
            h0(dofs[j]) = min(h0(dofs[j]), mesh->GetElementSize(i));
         }
      }
   }

   // 10. Add a random perturbation of the nodes in the interior of the domain.
   //     We define a random grid function of fespace and make sure that it is
   //     zero on the boundary and its values are locally of the order of h0.
   //     The latter is based on the DofToVDof() method which maps the scalar to
   //     the vector degrees of freedom in fespace.
   GridFunction rdm(fespace);
   double jitter = 0.; // perturbation scaling factor
   /* k10commenting this for input
    cout << "Enter jitter --> " << flush;
   cin >> jitter;
    */
   rdm.Randomize();
   rdm -= 0.5; // shift to random values in [-0.5,0.5]
   rdm *= jitter;
   {
      // scale the random values to be of order of the local mesh size
      for (int i = 0; i < fespace->GetNDofs(); i++)
         for (int d = 0; d < dim; d++)
         {
            rdm(fespace->DofToVDof(i,d)) *= h0(i);
         }

      Array<int> vdofs;
      // loop over the boundary elements
      for (int i = 0; i < fespace->GetNBE(); i++)
      {
         // get the vector degrees of freedom in the boundary element
         fespace->GetBdrElementVDofs(i, vdofs);
         // set the boundary values to zero
         for (int j = 0; j < vdofs.Size(); j++)
         {
            rdm(vdofs[j]) = 0.0;
         }
      }
   }
   *x -= rdm;

   // 11. Save the perturbed mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m perturbed.mesh".
   {
      ofstream mesh_ofs("perturbed.mesh");
      mesh->Print(mesh_ofs);
   }

   // 12. (Optional) Send the initially perturbed mesh with the vector field
   //     representing the displacements to the original mesh to GLVis.
    /* k10 commenting this for input
     cout << "Visualize the initial random perturbation? [0/1] --> ";
     cin >> ans;
     if (ans)
     {
     osockstream sol_sock(visport, vishost);
     sol_sock << "solution\n";
     mesh->Print(sol_sock);
     rdm.Save(sol_sock);
     sol_sock.send();
     }
     k10*/

   int smoother = 1;
    /*k10
   cout <<
        "Select smoother:\n"
        "1) TMOP\n"
        " --> " << flush;
   cin >> smoother;
     k10*/

   // 14. Simple mesh smoothing can be performed by relaxing the node coordinate
   //     grid function x with the matrix A and right-hand side b. This process
   //     converges to the solution of Ax=b, which we solve below with PCG. Note
   //     that the computed x is the A-harmonic extension of its boundary values
   //     (the coordinates of the boundary vertices). Furthermore, note that
   //     changing x automatically changes the shapes of the elements in the
   //     mesh. The vector field that gives the displacements to the perturbed
   //     positions is saved in the grid function x0.
   GridFunction x0(fespace);
   x0 = *x;

   L2_FECollection mfec(3, mesh->Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace mfes(mesh, &mfec, 1);
   GridFunction metric(&mfes);

   if (smoother == 1)
   {
      int metric_id;
      cout << "Choose optimization metric:\n"
           << "1  : |T|^2 \n"
           << "     shape.\n"
           << "2  : 0.5 |T|^2 / tau  - 1 \n"
           << "     shape, condition number metric.\n"
           << "5  : 0.5 |T|^2 / phi+ - 1 \n"
           << "     shape, pseudo-barrier condition number metric.\n"
           << "7  : |T - T^-t|^2 \n"
           << "     shape+size.\n"
           << "8  : (1 + 1 / phi+^2) |T|^2 - 4 tau / phi+ \n"
           << "     shape+size, pseudo-barrier version of metric 7 in 2D.\n"
           << "15  : 1 / tau^2 |T - tau I|^2 \n"
           << "     shape+size+alignment.\n"
           << "22 : 0.5 (|T|^2 - 2 tau) / (tau - tau0) \n"
           << "     shape, shifted-barrier condition number metric.\n"
           << "50 : |T^t T / tau|^2 - 2 \n"
           << "     shape, G-condition number metric.\n"
           << "51 : |T^t T / phi+ |^2 - 2 \n"
           << "     shape, pseudo-barrier G-condition number metric.\n"
           << "52 : (|T^t T|^2 - 2 tau^2) / (tau - tau0)^2 \n"
           << "     shape, shifted-barrier G-condition number metric.\n"
           << "54 : |T^t T|^2 \n"
           << "     shape.\n"
           << "55 : (tau - 1)^2 \n"
           << "     size.\n"
           << "56 : 0.5 (tau + 1/tau) - 1 \n"
           << "     size.\n"
           << "73 : (1 - gamma)(|T|^2 - 2 tau) + gamma (tau - 1/tau)^2 \n"
           << "     shape+size.\n"
           << "77 : 0.5 (tau - 1/tau)^2 \n"
           << "     size.\n"
           << "79 : (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1)^2 \n"
           << "     shape+size.\n"
           << "80 : (1 - gamma)(|T|^2 / (2 tau) - 1) + gamma (tau - 1/tau)^2 \n"
           << "     shape+size.\n"
           << "82 : (1 + 1 / (tau - tau0)^2) |T|^2 - 4 tau / (tau - tau0) \n"
           << "     shape+size, shifted-barrier version of metric 7 in 2D.\n"
           << "100: |A|^2.\n"
           << "102 :|A^-1 - W^-1 |^2 \n"
           << "     shape+size+alignment.\n"
           << " --> ";
      cin  >> metric_id;
      logvec[2]=metric_id;
       
      TMOPHyperelasticModel007 model;
      TargetJacobian *tj = new TargetJacobian(TargetJacobian::IDEAL_INIT_SIZE);
      tj->SetNodes(*x);
      tj->SetInitialNodes(x0);

      InterpolateHyperElasticModel(model, *tj, *mesh, metric);
      osockstream sol_sock1(visport, vishost);
      sol_sock1 << "solution\n";
      mesh->Print(sol_sock1);
      metric.Save(sol_sock1);
      sol_sock1.send();

      HyperelasticNLFIntegrator *nfi =
            new HyperelasticNLFIntegrator(&model, tj);
      const IntegrationRule *ir =
            &IntRulesLo.Get(fespace->GetFE(0)->GetGeomType(), 8);
      nfi->SetIntegrationRule(*ir);
      NonlinearForm nf(fespace);
      nf.AddDomainIntegrator(nfi);

      cout << "Initial strain energy : " << nf.GetEnergy(*x) << endl;
       logvec[3]=nf.GetEnergy(*x);

       
      //TMOPSmoother tmop(&model, tj, metric_id, 10000);
      TMOPSmoother tmop(NULL, NULL, metric_id, 2000);
      tmop.Init(*mesh, *mesh->GetNodes());
      tmop.Optimize();

      InterpolateHyperElasticModel(model, *tj, *mesh, metric);
      osockstream sol_sock2(visport, vishost);
      sol_sock2 << "solution\n";
      mesh->Print(sol_sock2);
      metric.Save(sol_sock2);
      sol_sock2.send();
      cout << "Final strain energy : " << nf.GetEnergy(*x) << endl;
       logvec[4]=nf.GetEnergy(*x);
   }
   else
   {
      printf("unknown smoothing option, smoother = %d\n",smoother);
      exit(1);
   }

   // Define mesh displacement
   x0 -= *x;
    

   // 15. Save the smoothed mesh to a file. This output can be viewed later
   //     using GLVis: "glvis -m smoothed.mesh".
   {
      ofstream mesh_ofs("smoothed.mesh");
      mesh_ofs.precision(14);
      mesh->Print(mesh_ofs);
   }
    // save subdivided VTK mesh?
    /* k10 commenting this stuff for testing
     if (1)
     {
     cout << "Enter VTK mesh subdivision factor or 0 to skip --> " << flush;
     cin >> ans;
     if (ans > 0)
     {
     ofstream vtk_mesh("smoothed.vtk");
     vtk_mesh.precision(8);
     mesh->PrintVTK(vtk_mesh, ans);
     }
     }

   // 16. (Optional) Send the relaxed mesh with the vector field representing
   //     the displacements to the perturbed mesh by socket to a GLVis server.
   cout << "Visualize the smoothed mesh? [0/1] --> ";
   cin >> ans;
   if (ans)
   {
      osockstream sol_sock(visport, vishost);
      sol_sock << "solution\n";
      mesh->Print(sol_sock);
      x0.Save(sol_sock);
      sol_sock.send();
   }
     */
    
   // 17. Free the used memory.
   delete fespace;
   delete fec;
   delete mesh;
    
    // write log to text file k10 stuff
    cout << "How do you want to write log to a new file:\n"
    "0) New file\n"
    "1) Append\n" << " --> " << flush;
    cin >> ans;
    ofstream outputFile;
    if (ans==0)
    {
        outputFile.open("logtmop.txt");
        outputFile << mesh_file << " ";
    }
    else
    {
        outputFile.open("logtmop.txt",fstream::app);
        outputFile << "\n" << mesh_file << " ";
    }
    
    
    for (int i=0;i<5;i++)
    {
        outputFile << logvec[i] << " ";
    }
    outputFile.close();
    // puase 1 second.. this is because X11 can restart if you push stuff too soon
    usleep(1000000);
    //k10 end

    
}
