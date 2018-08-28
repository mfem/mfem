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
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -o 1 -dt 0.0008 -tf 4 -s 3 -mt 2 -st 1
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

enum MONOTYPE { None, AlgUpw, AlgUpw_FS, MaxMax, MaxMax_FS, AlphaBeta, AlphaBeta_FS };
enum STENCIL  { Full, Local, LocalAndDiag };

class SolutionBounds {

   // set of local dofs which are in stencil of given local dof
   Mesh* mesh;
   FiniteElementSpace* fes;

   STENCIL stencil;

   // metadata for computing local bounds
   
   // Info for all dofs, including ones on face-neighbor cells.
   mutable DenseMatrix DOFs_coord;                   // size #dofs
   
public:

   // Map to compute localized bounds on unstructured grids.
   // For each dof index we have a vector of neighbor dof indices.
   mutable std::map<int, std::vector<int> > map_for_bounds;
   mutable std::map<int, std::vector<int> > map_for_SmoothnessIndicator;
   
   Vector x_min;
   Vector x_max;

   SolutionBounds(FiniteElementSpace* _fes, const BilinearForm& K, STENCIL _stencil)
   {
      fes = _fes;
      mesh = fes->GetMesh();
      stencil = _stencil;

      if (stencil > 0) { GetBoundsMap(fes, K); GetAllNeighbors(K.SpMat()); }
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
   
   void GetAllNeighbors(const SparseMatrix& K)
   {
      const int *I = K.GetI(), *J = K.GetJ(), size = K.Size();
      
      for (int i = 0, k = 0; i < size; i++)
      {
         for (int end = I[i+1]; k < end; k++)
         {
            map_for_SmoothnessIndicator[i].push_back(J[k]);
         }
      }
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

class FluxCorrectedTransport
{
private:
   FiniteElementSpace* fes;
   
public:
   // Constructor
   FluxCorrectedTransport(const MONOTYPE _monoType, FiniteElementSpace* _fes, 
                          const SparseMatrix &K, VectorFunctionCoefficient &coef, SolutionBounds &_bnds) : 
                          monoType(_monoType), fes(_fes), KpD(K), bnds(_bnds)
   {
      if (_monoType == None)
         return;
      else if ((_monoType == AlgUpw) || (_monoType == AlgUpw_FS))
      {
         ComputeDiffusionMatrix(K, KpD);
         KpD += K;
         
         // Compute the lumped mass matrix in an algebraic way
         BilinearForm m(fes);
         m.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
         m.Assemble();
         m.Finalize();
         m.SpMat().GetDiag(lumpedM);
      }
      else if ((_monoType == MaxMax) || (_monoType == MaxMax_FS))
      {
         ComputeDiffusionCoefficient(fes, coef, _monoType, elDiff, bdrDiff, lumpedM, dofs);
      }
      else if ((_monoType == AlphaBeta) || (_monoType == AlphaBeta_FS))
      {
         ComputeDiffusionVectors(fes, coef, _monoType, alpha, beta, sortArray, lumpedM, dofs);
      }
   }
   
   // Utility function to build a map to the offset of the symmetric entry in a sparse matrix
   Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
   {
      // assuming that A is finalized
      const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
      Array<int> smap;
      smap.SetSize(I[n]);
      
      for (int row = 0, j = 0; row < n; row++)
      {
         for (int end = I[row+1]; j < end; j++)
         {
            int col = J[j];
            // find the offset, _j, of the (col,row) entry and store it in smap[j]:
            for (int _j = I[col], _end = I[col+1]; true; _j++)
            {
               if (_j == _end)
                  mfem_error("SparseMatrix_Build_smap");
               
               if (J[_j] == row)
               {
                  smap[j] = _j;
                  break;
               }
            }
         }
      }
      return smap;
   }
   
   void ComputeDiffusionMatrix(const SparseMatrix& K, SparseMatrix& D)
   {
      const int s1 = K.Size();
      int* Ip = K.GetI();
      int* Jp = K.GetJ();
      double* Kp = K.GetData();
      Array<int> smap = SparseMatrix_Build_smap(K); // symmetry map
      
      double* Dp = D.GetData();
      
      for (int i = 0, k = 0; i < s1; i++)
      {
         double rowsum = 0.;
         for (int end = Ip[i+1]; k < end; k++)
         {
            int j = Jp[k];
            double kij = Kp[k];
            double kji = Kp[smap[k]];
            double dij = fmax(fmax(0.0,-kij),-kji);
            Dp[k] = dij;
            Dp[smap[k]] = dij;
            if (i != j) rowsum += Dp[k];
         }
         D(i,i) = -rowsum;
      }
   }
   
   void ComputeDiffusionCoefficient(FiniteElementSpace* fes, VectorFunctionCoefficient &coef, MONOTYPE monoType, 
                                    Vector &elDiff, DenseMatrix &bdrDiff, Vector &lumpedM, DenseMatrix &dofs)
   {
      enum ESTIMATE {Schwarz, Hoelder1Inf, Hoelder1Inf_Exact, HoelderInf1, HoelderInf1_Exact};
      ESTIMATE est = Schwarz;
      
      Mesh *mesh = fes->GetMesh();
      int i, j, k, p, qOrdE, qOrdF, nd, numBdrs, numDofs, dim = mesh->Dimension(), ne = mesh->GetNE();
      double vn;
      Array< int > bdrs, orientation;
      
      // use the first mesh element as an indicator for the following bunch
      const FiniteElement &dummy = *fes->GetFE(0);
      nd = dummy.GetDof();
      // fill the dofs array to access the correct dofs for boundaries
      dummy.ExtractBdrDofs(dofs);
      numBdrs = dofs.Width();
      numDofs = dofs.Height();
      
      Vector vval, nor(dim), vec1(dim), vec2(nd), shape(nd), alpha(nd), beta(nd), shapeBdr(numDofs);
      DenseMatrix velEval, adjJ(dim,dim), dshape(nd,dim);

      elDiff.SetSize(ne); elDiff = 0.;
      bdrDiff.SetSize(ne, numBdrs); bdrDiff = 0.;
      lumpedM.SetSize(ne*nd); lumpedM = 0.;
      
      // use the first mesh element as an indicator
      ElementTransformation *tr = mesh->GetElementTransformation(0);
      // Assuming order(u)==order(mesh)
      // beta can not be integrated exactly due to transforamtion dependent denominator
      // use tr->OrderW() + 2*dummy.GetOrder() + 2*dummy.max(tr->OrderGrad(&dummy), 0) instead
      // appropriate qOrdE for alpha is tr->OrderW() + 2*dummy.GetOrder(), choose max
      qOrdE = tr->OrderW() + 2*dummy.GetOrder() + 2*max(tr->OrderGrad(&dummy), 0);
      const IntegrationRule *ir = &IntRules.Get(dummy.GetGeomType(), qOrdE);
      
      // use the first mesh boundary as an indicator 
      FaceElementTransformations *Trans = mesh -> GetFaceElementTransformations(0); 
      // qOrdF is chosen such that L2-norm of basis functions is computed accurately.
      // Normal velocity term relies on L^Inf-norm which is approximated 
      // by its maximum value in the quadrature points of the same rule.
      if (Trans->Elem1No != 0)
      {
         if (Trans->Elem2No != 0)
            mfem_error("Boundary edge does not belong to this element.");
         else
            qOrdF = Trans->Elem2->OrderW() + 2*dummy.GetOrder();
      }
      else
      {
         qOrdF = Trans->Elem1->OrderW() + 2*dummy.GetOrder();
      }
      const IntegrationRule *irF1 = &IntRules.Get(Trans->FaceGeom, qOrdF);
      
      for (k = 0; k < ne; k++)
      {
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         const FiniteElement &el = *fes->GetFE(k);
         tr = mesh->GetElementTransformation(k);
         
         alpha = 0.; beta = 0.;
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
               switch (est)
               {
                  case Schwarz:
                     // divide due to square in L2-norm
                     beta(j) += ip.weight / tr->Weight() * pow(vec2(j), 2.);
                     alpha(j) += ip.weight * tr->Weight() * pow(shape(j), 2.);
                     break;
                  case Hoelder1Inf:
                     // divide because J^-1 = 1 / |J| adj(J) 
                     beta(j) = std::max(beta(j), - vec2(j) / tr->Weight());;
                     alpha(j) += ip.weight * tr->Weight() * shape(j);
                     break;
                  case Hoelder1Inf_Exact:
                     beta(j) = std::max(beta(j), - vec2(j));;
                     alpha(j) += ip.weight * shape(j);
                     break;
                  case HoelderInf1:
                     // divide because J^-1 = 1 / |J| adj(J) 
                     beta(j) += ip.weight * std::max(0., -vec2(j) / tr->Weight());
                     alpha(j) = std::max(alpha(j), tr->Weight() * shape(j));
                     break;
                  case HoelderInf1_Exact:
                     beta(j) += ip.weight * std::max(0., -vec2(j));
                     alpha(j) = std::max(alpha(j), shape(j));
                     break;
                  otherwise:
                     mfem_error("Unsupported estimate option.");
               }
               lumpedM(k*nd+j) += ip.weight * tr->Weight() * shape(j);
            }
         }
         elDiff(k) = std::sqrt(alpha.Max() * beta.Max());
         
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
            Trans = mesh->GetFaceElementTransformations(bdrs[i]); 
            vn = 0.; shapeBdr = 0.;
            
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
               
               vn = std::max(vn, vval * nor);
               for(j = 0; j < numDofs; j++)
                  shapeBdr(j) += ip.weight * Trans->Face->Weight() * pow(shape(dofs(j,i)), 2.);
            }
            bdrDiff(k,i) = vn * shapeBdr.Max();
         }
      }
   }
   
   void ComputeDiffusionVectors(FiniteElementSpace* fes, VectorFunctionCoefficient &coef, MONOTYPE monoType, 
                                Vector &alpha, Vector &beta, Array<int> &sortArray, Vector &LumpedM, 
                                DenseMatrix &dofs)
   {
      Mesh *mesh = fes->GetMesh();
      int i, j, k, p, dofInd, qOrdE, numPtsE, qOrdF, numPtsF, nd, numBdrs, numDofs;
      int dim = mesh->Dimension(), ne = mesh->GetNE();
      double vn, maxDiag;
      Array <int> locDofs, bdrs, orientation;
      
      // use the first mesh element as an indicator for the following bunch
      const FiniteElement &dummy = *fes->GetFE(0);
      nd = dummy.GetDof();
      // fill the dofs array to access the correct dofs for boundaries
      dummy.ExtractBdrDofs(dofs);
      numBdrs = dofs.Width();
      numDofs = dofs.Height();
      
      // use the first mesh element as an indicator
      ElementTransformation *tr = mesh->GetElementTransformation(0);
      // Assuming order(u)==order(mesh)
      // beta can not be integrated exactly due to transforamtion dependent denominator
      // use tr->OrderW() + 2*dummy.GetOrder() + 2*dummy.max(tr->OrderGrad(&dummy), 0) instead
      // appropriate qOrdE for alpha is tr->OrderW() + 2*dummy.GetOrder(), choose max
      qOrdE = tr->OrderW() + 2*dummy.GetOrder() + 2*max(tr->OrderGrad(&dummy), 0);
      const IntegrationRule *ir = &IntRules.Get(dummy.GetGeomType(), qOrdE);
      numPtsE = ir->GetNPoints();
      
      Vector vval, nor(dim), shape(nd), vec1(dim), vec2(nd), D_M(numPtsE), D_M_Inv(numPtsE);
      DenseMatrix velEval, dshape(nd,dim), adjJ(dim,dim), B1(numPtsE,nd), B2(numPtsE*dim,nd), 
                  D_K(dim*numPtsE, numPtsE), B1tDMB1(nd,nd), B2tDK(nd,numPtsE*dim), ret(nd,nd);
      
      D_K = 0.; // I used a DenseMatrix for the blockdiagonal matrix D_K which has diagonal blocks of size dim x 1
      alpha.SetSize(ne*nd); alpha = 0.;
      beta.SetSize(ne*nd); beta = 0.;
      bdrDiff.SetSize(ne, numBdrs); bdrDiff = 0.;
      LumpedM.SetSize(ne*nd); LumpedM = 0.;
      locDofs.SetSize(nd); sortArray.SetSize(ne*nd);
      
      // use the first mesh boundary as an indicator 
      FaceElementTransformations *Trans = mesh -> GetFaceElementTransformations(0); 
      // qOrdF is chosen such that L2-norm of basis functions is computed accurately.
      // Normal velocity term relies on L^Inf-norm which is approximated 
      // by its maximum value in the quadrature points of the same rule.
      if (Trans->Elem1No != 0)
      {
         if (Trans->Elem2No != 0)
            mfem_error("Boundary edge does not belong to this element.");
         else
            qOrdF = Trans->Elem2->OrderW() + 2*dummy.GetOrder();
      }
      else
      {
         qOrdF = Trans->Elem1->OrderW() + 2*dummy.GetOrder();
      }
      const IntegrationRule *irF1 = &IntRules.Get(Trans->FaceGeom, qOrdF);
      numPtsF = irF1->GetNPoints();
      
      
      Vector D_K_bdr(numPtsF), D_M_bdr(numPtsF);
      DenseMatrix B_Int(numPtsF,nd), B_Ext(numPtsF,nd);
      
      // use the first mesh boundary with a neighbor as an indicator
      for (i = 0; i < mesh->GetNumFaces(); i++)
      {
         Trans = mesh->GetFaceElementTransformations(i);
         if (Trans->Elem2No >= 0)
            break;
         //else
            //TODO what if no bdr has a neighbor?
      }
      
      for (p = 0; p < irF1->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = irF1->IntPoint(p);
         IntegrationPoint eip1;
         Trans->Face->SetIntPoint(&ip);
         
         Trans->Loc1.Transform(ip, eip1);
         dummy.CalcShape(eip1, shape);
         B_Int.SetRow(p, shape);
         Trans->Loc2.Transform(ip, eip1);
         dummy.CalcShape(eip1, shape);
         B_Ext.SetRow(p, shape);
      }
      B_Int.Transpose(); // NOTE: B_Int has been transposed.

      for (p = 0; p < numPtsE; p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         dummy.CalcShape(ip, shape);
         dummy.CalcDShape(ip, dshape);
         B1.SetRow(p, shape);
         for (j = 0; j < dim; j++)
         {
            dshape.GetColumn(j, vec2);
            B2.SetRow(p*dim+j, vec2);
         }
      }
      B1.Transpose(); // NOTE: B1 has been transposed.
      
      for (j = 0; j < nd; j++)
         locDofs[j] = j;
      
      for (k = 0; k < ne; k++)
      {
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         const FiniteElement &el = *fes->GetFE(k);
         tr = mesh->GetElementTransformation(k);
         coef.Eval(velEval, *tr, *ir);
         
         for (p = 0; p < numPtsE; p++)
         {
            const IntegrationPoint &ip = ir->IntPoint(p);
            tr->SetIntPoint(&ip);
            
            CalcAdjugate(tr->Jacobian(), adjJ);
            velEval.GetColumnReference(p, vval);
            adjJ.Mult(vval, vec1);
            
            D_M(p) = ip.weight * tr->Weight();
            D_M_Inv(p) = 1. / D_M(p);
            for (j = 0; j < dim; j++)
               D_K(p*dim+j, p) = ip.weight * vec1(j);
         }
         
         B1.Mult(D_M, vec2);
         LumpedM.SetSubVector(locDofs, vec2);
         
         MultADAt(B1, D_M, B1tDMB1);
         
         B1tDMB1.GetDiag(vec2);
         alpha.SetSubVector(locDofs, vec2);
         
         MultAtB(B2, D_K, B2tDK);
         
         // optional sharper bound incorporating the sign: TOO EXPENSIVE for high orders
         /*
         for (int col = 0; col < B2tDK.Width(); col++)
         {
            for (int row = 0; row < B2tDK.Height(); row++)
            {
               B2tDK(row,col) = B2tDK(row,col) < 0. ? 0. : B2tDK(row,col);
            }
         }*/
         
         MultADAt(B2tDK, D_M_Inv, ret);
         ret.GetDiag(vec2);
         beta.SetSubVector(locDofs, vec2);
         
         Array< Pair<double,int> > fractions(nd);
         
         for (j = 0; j < nd; j++) // get sortArray
         {
            dofInd = k*nd+j;
            alpha(dofInd) = sqrt(alpha(dofInd));
            beta(dofInd) = sqrt(beta(dofInd));
            fractions[j].one = beta(dofInd) / alpha(dofInd);
            fractions[j].two = dofInd;
         }
         
         SortPairs<double,int>(fractions, nd);
         for (j = 0; j < nd; j++)
         {
            locDofs[j] += nd;
            sortArray[k*nd+j] = fractions[j].two;
         }
         
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
            Trans = mesh->GetFaceElementTransformations(bdrs[i]); 
            vn = 0.; maxDiag = -numeric_limits<double>::infinity();
            
            for (p = 0; p < irF1->GetNPoints(); p++)
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
                  Trans->Elem2->SetIntPoint(&eip1);
                  coef.Eval(vval, *Trans->Elem2, eip1);
                  nor *= -1.;
               }
               else
               {
                  Trans->Loc1.Transform(ip, eip1);
                  Trans->Elem1->SetIntPoint(&eip1);
                  coef.Eval(vval, *Trans->Elem1, eip1);
               }
               nor /= nor.Norml2();
               
               D_K_bdr(p) = ip.weight * Trans->Face->Weight() * std::max(vn, vval * nor);
               D_M_bdr(p) = ip.weight * Trans->Face->Weight();
               maxDiag = std::max(maxDiag, D_K_bdr(p) / D_M_bdr(p));
            }
            MultADAt(B_Int, D_M_bdr, ret);
            ret.GetDiag(vec2);
            bdrDiff(k,i) = maxDiag * vec2.Max();
         }
      }
   }
   
   // Destructor
   ~FluxCorrectedTransport() { }
   
   // member variables that need to be accessed during time-stepping
   const MONOTYPE monoType;
   
   Vector lumpedM, elDiff, alpha, beta;
   SparseMatrix KpD;
   DenseMatrix bdrDiff, dofs;
   Array<int> sortArray;
   SolutionBounds &bnds;
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
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;
   
   double dt;
   const FluxCorrectedTransport &fct;

public:
   FE_Evolution(FiniteElementSpace* fes, SparseMatrix &_M, SparseMatrix &_K, 
                const Vector &_b, FluxCorrectedTransport &_fct);

   virtual void Mult(const Vector &x, Vector &y) const;
   
   virtual void SetDt(double _dt) { dt = _dt; }
   
   virtual void ComputeHighOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeLowOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeFCTSolution(const Vector &x, const Vector &yH, const Vector &yL, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   MONOTYPE monoType = MaxMax_FS;
   STENCIL stencil = Local;
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
   args.AddOption((int*)(&monoType), "-mt", "--monoType",
                  "Type of monotonicity treatment: 0 - no monotonicity treatment,\n\t"
                  "                                1 - algebraic upwinding - low order,\n\t"
                  "                                2 - algebraic upwinding - FCT,\n\t"
                  "                                3 - MaxMax scheme (matrix-free Rusanov) - low order,\n\t"
                  "                                4 - MaxMax scheme (matrix-free Rusanov) - FCT,\n\t"
                  "                                5 - AlphaBeta scheme (matrix-free) - low order,\n\t"
                  "                                6 - AlphaBeta scheme (matrix-free) - FCT.");
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
   
   if (monoType != None)
   {
      if (((int)monoType != monoType) || (monoType < 0) || (monoType > 6))
      {
         cout << "Unsupported option for monotonicity treatment." << endl;
         delete mesh;
         delete ode_solver;
         return 5;
      }
      if ((btype != 2) && (monoType > 2))
      {
         cout << "Matrix-free monotonicity treatment requires use of Bernstein basis." << endl;
         delete mesh;
         delete ode_solver;
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

   // Compute data required to easily find the min-/max-values for the high order scheme
   SolutionBounds bnds(&fes, k, stencil);
   // Precompute data required for high and low order schemes
   FluxCorrectedTransport fct(monoType, &fes, k.SpMat(), velocity, bnds);
   
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
   FE_Evolution adv( &fes, m.SpMat(), k.SpMat(), b, fct);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      // compute solution bounds
      fct.bnds.Compute(k.SpMat(), u);
      adv.SetDt(dt);
      
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1.e-8*dt);

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

void FE_Evolution::ComputeLowOrderSolution(const Vector &x, Vector &y) const
{
    if ((fct.monoType == AlgUpw) || (fct.monoType == AlgUpw_FS))
   {
      fct.KpD.Mult(x, z);
      z += b;
      for (int k = 0; k < fes->GetNE(); k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         int nd = el.GetDof();
         for (int j = 0; j < nd; j++)
         {
            int dofInd = k*nd+j; 
            y(dofInd) = z(dofInd) / fct.lumpedM(dofInd);
         }
      }
   }
   else if ((fct.monoType == MaxMax) || (fct.monoType == MaxMax_FS)) // low order matrix-free Rusanov scheme
   {
      Mesh *mesh = fes->GetMesh();
      int i, j, k, nd, numBdrs, dofInd, dim(mesh->Dimension()), numDofs(fct.dofs.Width());
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
               uSum += x(k*nd+fct.dofs(i,j));
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+fct.dofs(i,j)) += fct.bdrDiff(k,i)*(uSum - numDofs*x(k*nd+fct.dofs(i,j)));
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
            y(dofInd) = ( z(dofInd) + fct.elDiff(k)*(uSum - nd*x(dofInd)) ) / fct.lumpedM(dofInd);
            // y is now the low order discrete time derivative
         }
      }
   }
   else if ((fct.monoType == AlphaBeta) || (fct.monoType == AlphaBeta_FS))
   {
      Mesh *mesh = fes->GetMesh();
      int i, j, k, dofInd, numBdrs, dim(mesh->Dimension()), numDofs(fct.dofs.Width());
      Array<int> bdrs, orientation;
      double uSum;
      
      bool useSmInd = false; // optional usage of a smoothness indicator to decrease the artificial diffusivity
      // b_ij can use information about the geometry, for structured grids, b_ij = 1 i optimal
      // Ch should be small and dependent on the grid resolution h, which I have not incorporated so far
      double nom, den, q = 1., b_ij = 1., Ch = 1.E-15;
      Vector gamma;
      
      K.Mult(x, z);
      z += b;
      for (k = 0; k < fes->GetNE(); k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         int nd = el.GetDof();
         gamma.SetSize(nd);
         
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
               uSum += x(k*nd+fct.dofs(i,j));
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+fct.dofs(i,j)) += fct.bdrDiff(k,i)*(uSum - numDofs*x(k*nd+fct.dofs(i,j)));
         }
         
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         double sA  = 0., sB = 0., dA = 0., dB = 0.;
         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            sB += fct.beta(dofInd) * x(dofInd);
            dB += fct.beta(dofInd); // possible to optimize this
            
            // compute smoothness indicator
            if (useSmInd)
            {
               nom = 0.; den = 0.;
               for (i = 0; i < fct.bnds.map_for_SmoothnessIndicator[dofInd].size(); i++)
               {
                  double xi = x(fct.bnds.map_for_SmoothnessIndicator[dofInd][i]);
                  nom += b_ij * (x(dofInd) - xi);
                  den += b_ij * std::abs(x(dofInd) - xi);
               }
               gamma(j) = pow((std::abs(nom) + Ch) / (den + Ch), q);
            }
            else
               gamma(j) = 1.;
         }
         
         for (j = 0; j < nd; j++)
         {
            int pj = fct.sortArray[k*nd+j];
            double ga = gamma(k > 0 ? pj % (k*nd) : pj) * fct.alpha(pj);
            
            sA += ga * x(pj);
            sB -= fct.beta(pj) * x(pj);
            z(pj) += fct.beta(pj) * sA + ga * sB;
            
            // diagonal correction
            dA += ga;
            dB -= fct.beta(pj);
            z(pj) -= (fct.beta(pj) * dA + ga * dB) * x(pj);
            
            // invert lumped mass matrix
            y(pj) = z(pj) / fct.lumpedM(pj);
         }
         
         /* // For debugging, testing and analyzing purposes: application of max(a_i b_j, a_j b_i) with full matrix
         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i; double rowsum = 0.;
            for (j = 0; j < nd; j++)
            {
               int dofInd2 = k*nd+j;
               
               double dij = std::max( fct.alpha(dofInd) * fct.beta(dofInd2), fct.alpha(dofInd2) * fct.beta(dofInd) );
               
               z(dofInd) += dij * x(dofInd2);
               rowsum += dij;
            }
            z(dofInd) -= rowsum * x(dofInd);
            y(dofInd) = z(dofInd) / fct.lumpedM(dofInd);
         }*/
      }
   }
}

void FE_Evolution::ComputeHighOrderSolution(const Vector &x, Vector &y) const
{
   // No monotonicity treatment, straightforward high-order scheme
   // ydot = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void FE_Evolution::ComputeFCTSolution(const Vector &x, const Vector &yH, const Vector &yL, Vector &y) const
{
   // High order reconstruction that yields an updated admissible solution by means of 
   // clipping the solution coefficients within certain bounds and scaling the anti-
   // diffusive fluxes in a way that leads to local conservation of mass.
   Mesh *mesh = fes->GetMesh();
   int j, k, nd, dofInd;
   double sumPos, sumNeg, eps = 1.E-16;
   Vector uClipped, fClipped;

   // Monotonicity terms
   // NOTE: the same polynomial order for each element is assumed for access
   for (k = 0; k < mesh->GetNE(); k++)
   {
      const FiniteElement &el = *fes->GetFE(k);
      nd = el.GetDof();

      uClipped.SetSize(nd); uClipped = 0.;
      fClipped.SetSize(nd); fClipped = 0.;
      for (j = 0; j < nd; j++)
      {
         dofInd = k*nd+j;
         uClipped(j) = std::min(fct.bnds.x_max(dofInd), 
                                std::max(x(dofInd) + dt * yH(dofInd), fct.bnds.x_min(dofInd)));
      }
      
      sumPos = sumNeg = 0.;
      for (j = 0; j < nd; j++)
      {
         dofInd = k*nd+j;

         // compute coefficients for the high-order corrections
         // NOTE: The multiplication and inversion of the lumped mass matrix is 
         //       avoided here, this is only possible due to its positive diagonal 
         //       entries AND the way this high order scheme works
         fClipped(j) = uClipped(j) - ( x(dofInd) + dt * yL(dofInd) );
         
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
         // yH is high order discrete time derivative
         // yL is low order discrete time derivative
         y(dofInd) = yL(dofInd) + fClipped(j) / dt;
         // y is now the discrete time derivative featuring the high order anti-diffusive
         // reconstruction that leads to an forward Euler updated admissible solution.
         // The factor dt in the denominator is used for compensation in the ODE solver.
      }
   }
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace* _fes, SparseMatrix &_M, SparseMatrix &_K, 
                           const Vector &_b, FluxCorrectedTransport &_fct)
   : TimeDependentOperator(_M.Size()), fes(_fes), M(_M), K(_K), b(_b), z(_M.Size()), fct(_fct)
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
   if (fct.monoType == None)
   {
      ComputeHighOrderSolution(x, y);
   }
   else if (fct.monoType % 2 == 1)
   {
      ComputeLowOrderSolution(x, y);
   }
   else if (fct.monoType % 2 == 0)
   {
      Vector yH, yL;
      yH.SetSize(x.Size()); yL.SetSize(x.Size());
      
      ComputeHighOrderSolution(x, yH);
      ComputeLowOrderSolution(x, yL);
      ComputeFCTSolution(x, yH, yL, y);
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
      case 4:
      {
         double scale = 0.09;
         double G1 = (1./sqrt(scale)) * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5,2.));
         double G2 = (1./sqrt(scale)) * sqrt(pow(X(0) - 0.5,2.) + pow(X(1), 2.));

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
