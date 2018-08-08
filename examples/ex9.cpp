//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -o 0 -dt 0.01 -tf 4 -s 1 -mt 0
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -o 1 -dt 0.001 -tf 4 -s 1 -mt 0
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -o 1 -dt 0.002 -tf 4 -s 2 -mt 1
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -o 1 -dt 0.005 -tf 4 -s 3 -mt 2 -st 1
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

enum MONOTYPE { NONE, RUSANOV, FLUXSCALING };
enum STENCIL  { FULL, LOCAL, LOCALPLUSDIAG };


class SolutionBounds {

   // set of local dofs which are in stencil of given local dof
   Mesh* mesh;
   FiniteElementSpace* fes;

   STENCIL stencil;

   // metadata for computing local bounds
   
   // Info for all dofs, including ones on face-neighbor cells.
   mutable DenseMatrix DOFs_coord;                   // size #dofs
   
   // Map to compute localized bounds on unstructured grids.
   // For each dof index we have a vector of neighbor dof indices.
   mutable std::map<int, std::vector<int> > map_for_bounds;
   
public:

   Vector x_min;
   Vector x_max;

   SolutionBounds(FiniteElementSpace* _fes, const BilinearForm& K, STENCIL _stencil)
   {
      fes = _fes;
      mesh = fes->GetMesh();
      stencil = _stencil;

      if (stencil > 0) GetBoundsMap(fes, K);
   }

   void Compute(const SparseMatrix &K, const Vector &x)
   {
      x_min.SetSize(x.Size());
      x_max.SetSize(x.Size());
      
      switch (stencil)
      {
         case 0:
            ComputeFromSparsity(K, x);
            break;
         case 1:
         case 2:
            ComputeLocalBounds(x);
            break;
         defualt:
            mfem_error("Unsupported stencil.");
      }
   }

   void ComputeFromSparsity(const SparseMatrix& K, const Vector& x) 
   {
      const int *I = K.GetI(), *J = K.GetJ(), size = K.Size();
      
      for (int i = 0, k = 0; i < size; i++)
      {
         double x_i_min = numeric_limits<double>::infinity();
         double x_i_max = -x_i_min;
         for (int end = I[i+1]; k < end; k++)
         {
            double x_j = x(J[k]);
            
            if (x_j > x_i_max)
               x_i_max = x_j;
            if (x_j < x_i_min)
               x_i_min = x_j;
         }
         x_min(i) = x_i_min;
         x_max(i) = x_i_max;
      }
   }
   
   // Computation of localized bounds.
   void ComputeLocalBounds(const Vector &x)
   {
      const int size = x.Size();
      //const Vector &x_nd = x.FaceNbrData(); // for parallel
      
      for (int i = 0; i < size; i++)
      {
         double x_i_min = +numeric_limits<double>::infinity();
         double x_i_max = -x_i_min;
         for (int j = 0; j < map_for_bounds[i].size(); j++)
         {
            const int dof_id = map_for_bounds[i][j];
            double x_j = x(map_for_bounds[i][j]);
            // const double x_j = (dof_id < size) ? x(map_for_bounds[i][j])
            //                                : x_nd(dof_id - size); // for parallel
            if (x_j > x_i_max) { x_i_max = x_j; }
            if (x_j < x_i_min) { x_i_min = x_j; }
         }
         x_min(i) = x_i_min;
         x_max(i) = x_i_max;
      }
   }

private:
   
   double distance_(const IntegrationPoint &a, const IntegrationPoint &b)
   {
      return sqrt((a.x - b.x) * (a.x - b.x) +
                  (a.y - b.y) * (a.y - b.y) +
                  (a.z - b.z) * (a.z - b.z));
   }
   
   double Distance(const int dof1, const int dof2) const
   {
      const int dim = fes->GetMesh()->Dimension();
      
      if (dim==1)
      {
         return abs(DOFs_coord(0, dof1) - DOFs_coord(0, dof2));
      }
      else if (dim==2)
      {
         const double d1 = DOFs_coord(0, dof1) - DOFs_coord(0, dof2),
         d2 = DOFs_coord(1, dof1) - DOFs_coord(1, dof2);
         return sqrt(d1*d1 + d2*d2);
      }
      else
      {
         const double d1 = DOFs_coord(0, dof1) - DOFs_coord(0, dof2),
         d2 = DOFs_coord(1, dof1) - DOFs_coord(1, dof2),
         d3 = DOFs_coord(2, dof1) - DOFs_coord(2, dof2);
         return sqrt(d1*d1 + d2*d2 + d3*d3);
      }
   }
   
   // Fills DOFs_coord
   void ComputeCoordinates(FiniteElementSpace *fes)
   {
      const int dim = fes->GetMesh()->Dimension();
      const int num_cells = fes->GetNE();
      const int NDOFs     = fes->GetVSize();
      DOFs_coord.SetSize(dim, NDOFs);
      // DOFs_coord.SetSize(dim, NDOFs + fes->num_face_nbr_dofs); // for parallel
      
      Array<int> ldofs;
      DenseMatrix physical_coord;
      
      // Cells for the current process.
      for (int i = 0; i < num_cells; i++)
      {
         const IntegrationRule &ir = fes->GetFE(i)->GetNodes();
         ElementTransformation *el_trans = fes->GetElementTransformation(i);
         
         el_trans->Transform(ir, physical_coord);
         fes->GetElementDofs(i, ldofs);
         
         for (int j = 0; j < ldofs.Size(); j++)
         {
            for (int d = 0; d < dim; d++)
            {
               DOFs_coord(d, ldofs[j]) = physical_coord(d, j);
            }
         }
      }
      
      // Face-neighbor cells.
      /* for parallel
      IsoparametricTransformation el_trans;
      for (int i = 0; i < fes->GetMesh()->face_nbr_elements.Size(); i++)
      {
         const IntegrationRule &ir = fes->GetFaceNbrFE(i)->GetNodes();
         fes->GetMesh()->GetFaceNbrElementTransformation(i, &el_trans);
         
         el_trans.Transform(ir, physical_coord);
         fes->GetFaceNbrElementVDofs(i, ldofs);
         
         for (int j = 0; j < ldofs.Size(); j++)
         {
            ldofs[j] += NDOFs;
            
            for (int d = 0; d < dim; ++d)
            {
               DOFs_coord(d, ldofs[j]) = physical_coord(d, j);
            }
         }
      } */
   }
   
   // Fills map_for_bounds
   void GetBoundsMap(FiniteElementSpace *fes, const BilinearForm &K)
   {
      ComputeCoordinates(fes);
      
      int num_cells = fes->GetMesh()->GetNE();
      int NDOFs     = fes->GetVSize();
      double dist_level, dist = 0;
      const double tol = 1.e-10;
      Array<int> ldofs;
      Array<int> ldofs_external;
      const int *I = K.SpMat().GetI(), *J = K.SpMat().GetJ();
      
      // use the first mesh element as an indicator
      switch (stencil)
      {
         case 1:
            // hk at ref element with some tolerance
            dist_level = 1.0 / fes->GetOrder(0) + tol; 
            break;
         case 2:
            // Include the diagonal neighbors, use the first mesh element as an indicator
            // modified by Hennes, this should be larger than sqrt(3) to support 3D
            dist_level = 1.8 / fes->GetOrder(0) + tol; 
            break;
         default:
            mfem_error("Unsupported stencil.");
      }
      
      // what is the sense of this? I replaced boundsmap with map_for_bounds
      //std::map< int, std::vector<int> > &boundsmap = F.init_state.map_for_bounds;
      
      const FiniteElement *fe_external;
      
      // loop over cells
      for (int k = 0; k < num_cells; k++)
      {
         fes->GetElementDofs(k, ldofs);
         const FiniteElement &fe = *fes->GetFE(k);
         int n_dofs = fe.GetDof();
         const IntegrationRule &ir = fe.GetNodes();
         
// Use for debugging.
#define DOF_ID -1
         
         // Fill map_for_bounds for each dof within the cell.
         for (int i = 0; i < n_dofs; i++)
         {
            //////////////////////
            // ADD INTERNAL DOF //
            //////////////////////
            // For the cell where ith-DOF lives look for DOFs within dist(1).
            // This distance has to be on the reference element
            for (int j = 0; j < n_dofs; j++)
            {
               if (distance_(ir.IntPoint(i), ir.IntPoint(j)) <= dist_level)
               {
                  map_for_bounds[ldofs[i]].push_back(ldofs[j]);
               }
            }
            if (ldofs[i] == DOF_ID)
            {
               for (int j = 0; j < map_for_bounds[DOF_ID].size(); j++)
               {
                  cout << map_for_bounds[DOF_ID][j] << endl;
               }
            }
            
            //////////////////////
            // ADD EXTERNAL DOF //
            //////////////////////
            // There are different sources of external DOF.
            // 1. If one of the already (internal) included DOFs for the
            //    ith position is at a "face" then I have to include all external
            //    DOFs at the face location.
            // 2. If the ith-DOF is at a "face", then I have to include external
            //    DOFs within distance from the i-th location.
            // 3. Periodic BC - points that are at the boundary of the domain or a
            //    DOF next to it have to consider DOFs on the other end of the
            //    domain (NOT IMPLEMENTED YET!!!).
            
            //////////////
            // SOURCE 1 //
            //////////////
            // Loop over the already included internal DOFs (except the ith-DOF).
            // For each, find if its sparsity pattern contains
            // other DOFs with same physical location, and add them to the map.
            /*
             *         vector<int> vector_of_internal_dofs = map_for_bounds[ldofs[i]];
             *         for (int it = 0; it < vector_of_internal_dofs.size(); it++)
             *         {
             *            const int idof = vector_of_internal_dofs[it];
             *            if (idof == ldofs[i]) { continue; }
             * 
             *            // check sparsity pattern
             *            for (int j = I[idof]; j < I[idof + 1]; j++)
             *            {
             *               if (idof != J[j] && Distance(idof, J[j]) <= tol)
             *               {
             *                  boundsmap[ldofs[i]].push_back(J[j]);
         }
         }
         }
         
         if (ldofs[i] == DOF_ID)
         {
         cout << "sdf " << vector_of_internal_dofs.size() << endl;
         for (int j = 0; j < F.init_state.map_for_bounds[DOF_ID].size(); j++)
         {
         cout << boundsmap[DOF_ID][j] << endl;
         }
         }
         */
            
            //////////////
            // SOURCE 2 //
            //////////////
            // Check if the current dof is on a face:
            // Loop over its sparsity pattern and find DOFs at the same location.
            vector<int> DOFs_at_ith_location;
            for (int j = I[ldofs[i]]; j < I[ldofs[i] + 1]; j++)
            {
               dist = Distance(ldofs[i], J[j]);
               if (ldofs[i] == DOF_ID)
               {
                  cout << "checking " << J[j] << " " << dist << endl;
               }
               if (dist <= tol && ldofs[i] != J[j]) // dont include the ith DOF
               {
                  DOFs_at_ith_location.push_back(J[j]);
                  
                  // Now look over the sparcity pattern of J[j] to find more
                  // dofs at the same location
                  // (adds diagonal neighbors, if they are on the same mpi task).
                  const int d = J[j];
                  bool is_new = true;
                  for (int jj = I[d]; jj < I[d+1]; jj++)
                  {
                     if (J[jj] == ldofs[i]) { continue; }
                     for (int dd = 0; dd < DOFs_at_ith_location.size(); dd++)
                     {
                        if (J[jj] == DOFs_at_ith_location[dd])
                        { is_new = false; break; }
                     }
                     if (is_new && Distance(d, J[jj]) < tol)
                     { DOFs_at_ith_location.push_back(J[jj]); }
                  }
               }
            }
            if (ldofs[i] == DOF_ID)
            {
               cout << "same location " << DOFs_at_ith_location.size() << endl;
            }
            // Loop over the dofs at i-th location; for each, loop over DOFs
            // local on its cell to find those within dist(1).
            // Note that distance has to be measured on the reference element.
            for (int it = 0; it < DOFs_at_ith_location.size(); it++)
            {
               int dof = DOFs_at_ith_location[it];
               if (dof < NDOFs)
               {
                  const int cell_id = dof / n_dofs;
                  fes->GetElementDofs(cell_id, ldofs_external);
                  fe_external = fes->GetFE(cell_id);
               }
               /* else
               {
                  const int cell_id = dof / n_dofs - num_cells;
                  fes->GetFaceNbrElementVDofs(cell_id, ldofs_external);
                  fe_external = fes->GetFaceNbrFE(cell_id);
                  
                  for (int j = 0; j < ldofs.Size(); j++)
                  {
                     ldofs_external[j] += NDOFs;
                  }
               }*/ // for parallel
               
               int n_dofs_external = fe_external->GetDof();
               const IntegrationRule &ir_ext = fe_external->GetNodes();
               for (int j = 0; j < n_dofs_external; j++) // here j is local
               {
                  bool is_new = true;
                  for (int dd = 0; dd < map_for_bounds[ldofs[i]].size(); dd++)
                  {
                     if (ldofs_external[j] == map_for_bounds[ldofs[i]][dd])
                     { is_new = false; break; }
                  }
                  
                  int loc_index = dof % n_dofs;
                  if (is_new &&
                     distance_(ir_ext.IntPoint(loc_index),
                               ir_ext.IntPoint(j)) <= dist_level)
                  {
                     map_for_bounds[ldofs[i]].push_back(ldofs_external[j]);
                  }
               }
            }
            if (ldofs[i] == DOF_ID)
            {
               cout << " --- " << endl;
               for (int j = 0; j < map_for_bounds[DOF_ID].size(); j++)
               {
                  cout << map_for_bounds[DOF_ID][j] << endl;
               }
            }
         }
      }
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{   
private:
   FiniteElementSpace* fes;
   SparseMatrix &M, &K;
   const Vector &b, &elDiff, &lumpedM;
   const DenseMatrix &bdrDiff, &dofs;
   DSmoother M_prec;
   CGSolver M_solver;
   MONOTYPE monoType;
   
   SolutionBounds &bnds;

   double dt;

   mutable Vector z;

public:
   FE_Evolution(FiniteElementSpace* fes, SparseMatrix &_M, SparseMatrix &_K, 
                const Vector &_b, const Vector &_elDiff, const DenseMatrix &_bdrDiff, 
                const Vector &_lumpedM, MONOTYPE mono, SolutionBounds& bnds, 
                const DenseMatrix &_dofs);

   virtual void Mult(const Vector &x, Vector &y) const;
   
   virtual void SetDt(double _dt) { dt = _dt; }

   virtual ~FE_Evolution() { }
};


void preprocessLowOrderScheme(FiniteElementSpace* fes, VectorFunctionCoefficient & coef, 
                              MONOTYPE mono, Vector &elDiff, DenseMatrix &bdrDiff, 
                              Vector &lumpedM, DenseMatrix &dofs)
{
   if (mono == NONE)
      return;
   
   Mesh *mesh = fes->GetMesh();
   int i, j, k, p, qOrd, nd, geomType, numBdrs, numDofs,
   dim = mesh->Dimension(), ne = mesh->GetNE();
   double un;
   ElementTransformation *tr;
   Vector shape, vec1, vec2, estim1, estim2, bas, vval(dim), nor(dim);
   DenseMatrix dshape, adjJ;
   Array< int > bdrs, orientation;
   DenseMatrix velEval;
   
   elDiff.SetSize(ne); elDiff = 0.;
   lumpedM.SetSize(ne); lumpedM = 0.;
   adjJ.SetSize(dim,dim);
   
   // fill the dofs array to access the correct dofs for boundaries
   const FiniteElement &dummy = *fes->GetFE(0); // use the first mesh element as an indicator
   geomType = dummy.GetGeomType();
   dummy.ExtractBdrDofs(dofs);
   numBdrs = dofs.Width();
   bdrDiff.SetSize(ne, numBdrs);
   numDofs = dofs.Height();
   bas.SetSize(numDofs);
   
   for (k = 0; k < ne; k++)
   {
      ///////////////////////////
      // Element contributions //
      ///////////////////////////
      const FiniteElement &el = *fes->GetFE(k);
      nd = el.GetDof();
      tr = mesh->GetElementTransformation(k);
      // Assuming order(u)==order(mesh)
      // estim1 can not be integrated exactly due to transforamtion dependent denominator
      // use tr->OrderW() + 2*el.GetOrder() + 2*el.GetOrderJ() instead
      // appropriate qOrd for estim2 is tr->OrderW() + 2*el.GetOrder(), choose max
      qOrd = tr->OrderW() + 2*el.GetOrder() + 2*tr->OrderJ();
      
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), qOrd);
      
      shape.SetSize(nd);
      dshape.SetSize(nd,dim);
      estim1.SetSize(nd);
      estim2.SetSize(nd);
      vec1.SetSize(nd);
      vec2.SetSize(nd);
      estim1 = estim2 = 0.;
      
      coef.Eval(velEval, *tr, *ir);
            
      for (p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         tr->SetIntPoint(&ip);

         el.CalcDShape(ip, dshape);
         CalcAdjugate(tr->Jacobian(), adjJ);
         el.CalcShape(ip, shape);
         
         velEval.GetColumnReference(p, vval);
         adjJ.Mult(vval, vec1);
         dshape.Mult(vec1, vec2);
         for (j = 0; j < nd; j++)
         {
            //divide due to square in L2-norm
            estim1(j) += ip.weight / tr->Weight() * pow(vec2(j), 2.); 
            estim2(j) += ip.weight * tr->Weight() * pow(shape(j), 2.);
         }
         lumpedM(k) += ip.weight * tr->Weight();
      }
      elDiff(k) = std::sqrt(estim1.Max() * estim2.Max());
      lumpedM(k) /= nd;
      
      ////////////////////////////
      // Boundary contributions //
      ////////////////////////////
      if (dim==1)
         mesh->GetElementVertices(k, bdrs);
      else if (dim==2)
         mesh->GetElementEdges(k, bdrs, orientation);
      else if (dim==3)
         mesh->GetElementFaces(k, bdrs, orientation);
      
      for (i = 0; i < numBdrs; i++)
      {
         FaceElementTransformations *Trans = mesh -> GetFaceElementTransformations(bdrs[i]); 
         // qOrd is chosen such that L2-norm of basis is computed accurately.
         // Normal velocity term relies on L^{infty}-norm which is approximated 
         // by its maximum value in the quadrature points of the same rule.
         if (Trans->Elem1No != k)
         {
            if (Trans->Elem2No != k)
               mfem_error("Boundary edge does not belong to this element.");
            else
               qOrd = Trans->Elem2->OrderW() + 2*el.GetOrder();
         }
         else
         {
            qOrd = Trans->Elem1->OrderW() + 2*el.GetOrder();
         }

         const IntegrationRule *irF1 = &IntRules.Get(Trans->FaceGeom, qOrd);
         un = 0.;  bas = 0.;
         
         for (int p = 0; p < irF1->GetNPoints(); p++)
         {
            const IntegrationPoint &ip = irF1->IntPoint(p);
            IntegrationPoint eip1;
            
            Trans->Face->SetIntPoint(&ip);
            
            if (dim == 1)
               nor(0) = 2.*eip1.x - 1.0;
            else
               CalcOrtho(Trans->Face->Jacobian(), nor);
            
            if (Trans->Elem1No != k)
            {
               Trans->Loc2.Transform(ip, eip1);
               el.CalcShape(eip1, shape);
               Trans->Elem2->SetIntPoint(&eip1);
               coef.Eval(vval, *Trans->Elem2, eip1);
               nor *= -1.;
            }
            else
            {
               Trans->Loc1.Transform(ip, eip1);
               el.CalcShape(eip1, shape);
               Trans->Elem1->SetIntPoint(&eip1);
               coef.Eval(vval, *Trans->Elem1, eip1);
            }
            
            nor /= nor.Norml2();

            un = std::max(vval * nor, un);
            for(j = 0; j < numDofs; j++)
               bas(j) += ip.weight * Trans->Face->Weight() * pow(shape(dofs(j,i)), 2.);
         }
         bdrDiff(k,i) = un * bas.Max();
      }
   }
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   MONOTYPE mono = FLUXSCALING;
   STENCIL stencil = LOCAL;
   double t_final = 10.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption((int*)(&mono), "-mt", "--mono",
                  "Type of monotonicity treatment: 0 - no monotonicity treatment,\n\t"
                  "                                1 - matrix-free Rusanov scheme.");
   args.AddOption((int*)(&stencil), "-st", "--stencil",
                  "Type of stencil for high order scheme: 0 - all neighbors,\n\t"
                  "                                       1 - closest neighbors,\n\t"
                  "                                       2 - closest plus diagonal neighbors.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   FiniteElementSpace fes(mesh, &fec);
   
   if (mono != NONE)
   {
      if ((mono != RUSANOV) && (mono != FLUXSCALING))
      {
         cout << "Unsupported option for monotonicity treatment." << endl;
         return 5;
      }
      if (btype != 2)
      {
         cout << "Monotonicity treatment requires use of Bernstein basis." << endl;
         return 5;
      }
      if (order == 0)
         mfem_warning("No need to use monotonicity treatment for polynomial order 0.");
   }

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   //    Also prepare for the use of low and high order schemes.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();
   
   // Precompute data required for low order scheme
   Vector elDiff, lumpedM;
   DenseMatrix bdrDiff, dofs;
   preprocessLowOrderScheme(&fes, velocity, mono, elDiff, bdrDiff, lumpedM, dofs);
   
   // Compute data required to easily find the min-/max-values for the high order scheme
   SolutionBounds bnds(&fes, k, stencil);

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv( &fes, m.SpMat(), k.SpMat(), b, elDiff, bdrDiff, lumpedM,
                     mono, bnds, dofs );
   
   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      // compute solution bounds
      bnds.Compute(k.SpMat(), u);
      adv.SetDt(dt);
      
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << *mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete mesh;
   delete ode_solver;
   delete dc;
   return 0;
   
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace* _fes, SparseMatrix &_M, SparseMatrix &_K, 
                           const Vector &_b, const Vector &_elDiff, const DenseMatrix 
                           &_bdrDiff, const Vector &_lumpedM, MONOTYPE mono, 
                           SolutionBounds& _bnds, const DenseMatrix &_dofs)
   : TimeDependentOperator(_M.Size()), fes(_fes), M(_M), K(_K), b(_b), elDiff(_elDiff), 
   lumpedM(_lumpedM), bdrDiff(_bdrDiff), monoType(mono), z(_M.Size()), bnds(_bnds),
   dofs(_dofs)
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   if (monoType == NONE)
   {
      // No monotonicity treatment, straightforward high-order scheme
      // ydot = M^{-1} (K x + b)
      K.Mult(x, z);
      z += b;
      M_solver.Mult(z, y);
   }
   else if (monoType == RUSANOV)
   {
      // low order matrix-free Rusanov scheme
      Mesh *mesh = fes->GetMesh();
      int i, j, k, nd, numBdrs, dofInd, dim(mesh->Dimension()), numDofs(dofs.Width());
      Array< int > bdrs, orientation;
      double uSum;
      
      // Discretization terms
      K.Mult(x, z);
      z += b;
      
      // Monotonicity terms
      // NOTE: the same polynomial order for each element is assumed for access
      for (k = 0; k < mesh->GetNE(); k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();
         
         if (dim==1)
            mesh->GetElementVertices(k, bdrs);
         else if (dim==2)
            mesh->GetElementEdges(k, bdrs, orientation);
         else if (dim==3)
            mesh->GetElementFaces(k, bdrs, orientation);
         
         numBdrs = bdrs.Size();

         ////////////////////////////
         // Boundary contributions //
         ////////////////////////////
         for (i = 0; i < numBdrs; i++)
         {
            uSum = 0.;
            for (j = 0; j < numDofs; j++)
               uSum += x(k*nd+dofs(i,j));
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+dofs(i,j)) += bdrDiff(k,i)*(uSum - numDofs*x(k*nd+dofs(i,j)));
         }
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         uSum = 0.;
         for (j = 0; j < nd; j++)
            uSum += x(k*nd+j);
         
         for (j = 0; j < nd; j++)
         {
            // element update and inversion of lumped mass matrix
            dofInd = k*nd+j;
            y(dofInd) = ( z(dofInd) + elDiff(k)*(uSum - nd*x(dofInd)) ) / lumpedM(k);
            // z is now the low order discrete time derivative
         }
      }
   }
   else if (monoType == FLUXSCALING)
   {
      // High order reconstruction that yields an updated admissible solution by means of 
      // clipping the solution coefficients within certain bounds and scaling the anti-
      // diffusive fluxes in a way that leads to local conservation of mass.
      Mesh *mesh = fes->GetMesh();
      int i, j, k, nd, numBdrs, dofInd, dim = mesh->Dimension(), numDofs(dofs.Width());
      Array< int > bdrs, orientation;
      double uDof, uSum, sumPos, sumNeg, eps = 1.E-15;
      Vector uClipped, fClipped;
      
      // Discretization terms
      K.Mult(x, z);
      z += b;
      M_solver.Mult(z, y); // y is now the high order discrete time derivative
      
      // Monotonicity terms
      // NOTE: the same polynomial order for each element is assumed for access
      for (k = 0; k < mesh->GetNE(); k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();
         
         if (dim==1)
            mesh->GetElementVertices(k, bdrs);
         else if (dim==2)
            mesh->GetElementEdges(k, bdrs, orientation);
         else if (dim==3)
            mesh->GetElementFaces(k, bdrs, orientation);
         
         numBdrs = bdrs.Size();

         ////////////////////////
         // Face contributions //
         ////////////////////////
         for (i = 0; i < numBdrs; i++)
         {
            uSum = 0.;
            for (j = 0; j < numDofs; j++)
               uSum += x(k*nd+dofs(i,j));
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+dofs(i,j)) += bdrDiff(k,i)*(uSum - numDofs*x(k*nd+dofs(i,j)));
         }
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         uSum = 0.;
         uClipped.SetSize(nd); uClipped = 0.;
         fClipped.SetSize(nd); fClipped = 0.;
         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            uDof = x(dofInd);
            uSum += uDof;
            
            uClipped(j) = std::min(bnds.x_max(dofInd), 
                                   std::max(x(dofInd) + dt * y(dofInd), bnds.x_min(dofInd)));
         }
         
         sumPos = sumNeg = 0.;
         for (j = 0; j < nd; j++)
         {
            // element update and inversion of lumped mass matrix
            dofInd = k*nd+j;
            z(dofInd) = ( z(dofInd) + elDiff(k)*(uSum - nd*x(dofInd)) ) / lumpedM(k);
            // z is now the low order discrete time derivative
            
            // compute coefficients for the high-order corrections
            // NOTE: The multiplication and inversion of the lumped mass matrix is 
            //       avoided here, this is only possible due to its positive diagonal 
            //       entries AND the way this high order scheme works
            fClipped(j) = uClipped(j) - ( x(dofInd) + dt * z(dofInd) );
            
            sumPos += std::max(fClipped(j), 0.);
            sumNeg += std::min(fClipped(j), 0.);
         }
         
         for (j = 0; j < nd; j++)
         {
            if ((sumPos + sumNeg > eps) && (fClipped(j) > eps))
               fClipped(j) *= - sumNeg / sumPos;
            if ((sumPos + sumNeg < -eps) && (fClipped(j) < -eps))
               fClipped(j) *= - sumPos / sumNeg;
            
            dofInd = k*nd+j; 
            // y is high order discrete time derivative
            // z is low order discrete time derivative
            y(dofInd) = z(dofInd) + fClipped(j) / dt;
            // y is now the discrete time derivative featuring the high order anti-diffusive
            // reconstruction that leads to an forward Euler updated admissible solution.
            // The factor dt in the denominator is used for compensation in the ODE solver.
         }
      }
   }
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      case 4:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return .5*(sin(f*X(0))*sin(f*X(1)) + 1.); // modified by Hennes 
      }
      case 4: // new
      {
         double scale = 0.09;
         double G1 = (1./sqrt(scale)) * sqrt(pow(X(0),2.) + pow(X(1)+0.5,2.));
         double G2 = (1./sqrt(scale)) * sqrt(pow(X(0)-0.5,2.) + pow(X(1),2.));

         return ((pow(X(0),2.) + pow(X(1) - 0.5,2.) <= scale) & 
                  (X(0) <= -0.05 | X(0) >= 0.05 | X(1) >= 0.7)) ? 1. : 0. +
                  (1-G1) * (pow(X(0),2.) + pow(X(1) + 0.5,2.) <= scale) + 
                  0.25*(1.+cos(M_PI*G2))*(pow(X(0) - 0.5,2.) + pow(X(1),2.) <= scale);
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: 
      case 4: return 0.0;
   }
   return 0.0;
}
