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

enum MONOTYPE { None, DiscUpw, DiscUpw_FCT, ResDist, ResDist_FCT };


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
            {
               mfem_error("SparseMatrix_Build_smap");
            }

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

void ComputeDiscreteUpwindingMatrix(const SparseMatrix& K, SparseMatrix& D)
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
         if (i != j) { rowsum += Dp[k]; }
      }
      D(i,i) = -rowsum;
   }
}

Mesh* GetSubcellMesh(Mesh *mesh, int p)
{
   Mesh *ref_mesh;
   if (p==1)
   {
      ref_mesh = mesh;
   }
   else if (mesh->Dimension() > 1)
   {
      int basis_lor = BasisType::ClosedUniform; // to have a uniformly refined mesh
      ref_mesh = new Mesh(mesh, p, basis_lor);
      ref_mesh->SetCurvature(1);
   }
   else
   {
      // TODO generalize to arbitrary 1D segments (different lengths)
      ref_mesh = new Mesh(mesh->GetNE()*p, 1.);
      ref_mesh->SetCurvature(1);
   }
   return ref_mesh;
}
   
const IntegrationRule *GetFaceIntRule(FiniteElementSpace *fes)
{
   int i, qOrdF;
   Mesh* mesh = fes->GetMesh();
   FaceElementTransformations *Trans;
   // use the first mesh boundary with a neighbor as indicator
   for (i = 0; i < mesh->GetNumFaces(); i++)
   {
      Trans = mesh->GetFaceElementTransformations(i);
      qOrdF = Trans->Elem1->OrderW();
      if (Trans->Elem2No >= 0)
      {
         // qOrdF is chosen such that L2-norm of basis functions is computed accurately.
         qOrdF = max(qOrdF, Trans->Elem2->OrderW());
         break;
      }
   }
   // use the first mesh element as indicator
   const FiniteElement &dummy = *fes->GetFE(0);
   qOrdF += 2*dummy.GetOrder();
   
   return &IntRules.Get(Trans->FaceGeom, qOrdF);
}
   
class SolutionBounds
{
   // set of local dofs which are in stencil of given local dof
   Mesh* mesh;
   FiniteElementSpace* fes;

public:

   // For each dof the elements containing that vertex are stored.
   mutable std::map<int, std::vector<int> > map_for_bounds;

   Vector x_min;
   Vector x_max;

   SolutionBounds(FiniteElementSpace* _fes, const BilinearForm& K)
   {
      fes = _fes;
      mesh = fes->GetMesh();

      GetVertexBoundsMap();
   }

   // NOTE: Optimizations possible, combine this with low and high order methods.
   void ComputeVertexBounds(const Vector& x)
   {
      int i, j, k, nd, dofInd, ne = mesh->GetNE();
      Vector xMax, xMin;
      xMax.SetSize(ne); xMin.SetSize(ne);
      
      x_min.SetSize(x.Size());
      x_max.SetSize(x.Size());
      
      for (k = 0; k < ne; k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();
         
         xMin(k) = numeric_limits<double>::infinity();
         xMax(k) = -xMin(k);
         for (j = 0; j < nd; j++)
         {
            xMax(k) = max(xMax(k), x(k*nd+j));
            xMin(k) = min(xMin(k), x(k*nd+j));
         }
      }
      for (k = 0; k < ne; k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();
         
         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            x_min(dofInd) = numeric_limits<double>::infinity();
            x_max(dofInd) = -x_min(dofInd);
            
            for (int i = 0; i < (int)map_for_bounds[dofInd].size(); i++)
            {
               x_max(dofInd) = max(x_max(dofInd), xMax(map_for_bounds[dofInd][i]));
               x_min(dofInd) = min(x_min(dofInd), xMin(map_for_bounds[dofInd][i]));
            }
         }
      }
   }

private:
   
   // Finds a zone id that shares a face with both el1 and el2, but isn't el.
   // NOTE: Here it is assumed that all elements have the same geometry, due to numBdrs.
   int FindCommonAdjacentElement(int el, int el1, int el2, int dim, int numBdrs)
   {
      if (min(el1, el2) < 0) { return -1; }
      
      int i, j, commonNeighbor;
      bool found = false;
      Array<int> bdrs1, bdrs2, orientation, neighborElements1, neighborElements2;
      FaceElementTransformations *Trans;
      
      neighborElements1.SetSize(numBdrs); neighborElements2.SetSize(numBdrs);
      
      // add neighbor elements sharing a vertex/edge/face according to grid dimension
      if (dim==1)
      {
         mesh->GetElementVertices(el1, bdrs1);
         mesh->GetElementVertices(el2, bdrs2);
      }
      else if (dim==2)
      {
         mesh->GetElementEdges(el1, bdrs1, orientation);
         mesh->GetElementEdges(el2, bdrs2, orientation);
      }
      else if (dim==3)
      {
         mesh->GetElementFaces(el1, bdrs1, orientation);
         mesh->GetElementFaces(el2, bdrs2, orientation);
      }

      for (i = 0; i < numBdrs; i++)
      {
         Trans = mesh->GetFaceElementTransformations(bdrs1[i]);
         neighborElements1[i] = Trans->Elem1No != el1 ? Trans->Elem1No : Trans->Elem2No;
         
         Trans = mesh->GetFaceElementTransformations(bdrs2[i]);
         neighborElements2[i] = Trans->Elem1No != el2 ? Trans->Elem1No : Trans->Elem2No;
      }
      
      for (i = 0; i < numBdrs; i++)
      {
         for (j = 0; j < numBdrs; j++)
         {
            if ((neighborElements1[i] == neighborElements2[j]) && (neighborElements1[i] != el))
            {
               if (!found)
               {
                  commonNeighbor = neighborElements1[i];
                  found = true;
               }
               else
               {
                  mfem_error("Found multiple common neighbor elements.");
               }
            }
         }
      }
      if (found)
      {
         return commonNeighbor;
      }
      else { return -1; }
   }
   
   // NOTE: Here it is assumed that the mesh consists of segments, quads or hexes.
   // NOTE: Here it is assumed that all elements have the same geometry, due to numBdrs.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void GetVertexBoundsMap()
   {
      const FiniteElement &dummy = *fes->GetFE(0);
      int i, j, k, dofInd, numBdrs, numDofs, nbr_id, dim = mesh->Dimension(), 
         ne = mesh->GetNE(), nd = dummy.GetDof(), p = dummy.GetOrder();
      Array<int> bdrs, orientation, neighborElements;
      DenseMatrix dofs;
      FaceElementTransformations *Trans;

      dummy.ExtractBdrDofs(dofs);
      numBdrs = dofs.Width();
      numDofs = dofs.Height();

      neighborElements.SetSize(numBdrs);
      
      for (k = 0; k < ne; k++)
      {
         // add the current element for all dofs of the element
         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i;
            map_for_bounds[dofInd].push_back(k);
         }
         
         // add neighbor elements sharing a vertex/edge/face according to grid dimension
         if (dim==1)
            mesh->GetElementVertices(k, bdrs);
         else if (dim==2)
            mesh->GetElementEdges(k, bdrs, orientation);
         else if (dim==3)
            mesh->GetElementFaces(k, bdrs, orientation);
         
         for (i = 0; i < numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);
            
            neighborElements[i] = Trans->Elem1No != k ? Trans->Elem1No : Trans->Elem2No;
            
            for (j = 0; j < numDofs; j++)
            {
               dofInd = k*nd+dofs(j,i);
               map_for_bounds[dofInd].push_back(neighborElements[i]);
            }
         }
         
         // Diagonal neighbors.
         if (dim==2)
         {
            nbr_id = FindCommonAdjacentElement(k, neighborElements[3],
                                                  neighborElements[0], 2, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd].push_back(nbr_id); }

            nbr_id = FindCommonAdjacentElement(k, neighborElements[0],
                                                  neighborElements[1], 2, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+p].push_back(nbr_id); }

            nbr_id = FindCommonAdjacentElement(k, neighborElements[1],
                                                  neighborElements[2], 2, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[(k+1)*nd-1].push_back(nbr_id); }

            nbr_id = FindCommonAdjacentElement(k, neighborElements[2],
                                                  neighborElements[3], 2, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*p*(p+1)].push_back(nbr_id); }
         }
         else if (dim==3)
         {
            Array<int> neighborElem; neighborElem.SetSize(12); // for each edge
            
            neighborElem[0]  = FindCommonAdjacentElement(k, neighborElements[0], neighborElements[1], dim, numBdrs);
            neighborElem[1]  = FindCommonAdjacentElement(k, neighborElements[0], neighborElements[2], dim, numBdrs);
            neighborElem[2]  = FindCommonAdjacentElement(k, neighborElements[0], neighborElements[3], dim, numBdrs);
            neighborElem[3]  = FindCommonAdjacentElement(k, neighborElements[0], neighborElements[4], dim, numBdrs);
            neighborElem[4]  = FindCommonAdjacentElement(k, neighborElements[5], neighborElements[1], dim, numBdrs);
            neighborElem[5]  = FindCommonAdjacentElement(k, neighborElements[5], neighborElements[2], dim, numBdrs);
            neighborElem[6]  = FindCommonAdjacentElement(k, neighborElements[5], neighborElements[3], dim, numBdrs);
            neighborElem[7]  = FindCommonAdjacentElement(k, neighborElements[5], neighborElements[4], dim, numBdrs);
            neighborElem[8]  = FindCommonAdjacentElement(k, neighborElements[4], neighborElements[1], dim, numBdrs);
            neighborElem[9]  = FindCommonAdjacentElement(k, neighborElements[1], neighborElements[2], dim, numBdrs);
            neighborElem[10] = FindCommonAdjacentElement(k, neighborElements[2], neighborElements[3], dim, numBdrs);
            neighborElem[11] = FindCommonAdjacentElement(k, neighborElements[3], neighborElements[4], dim, numBdrs);
            
            // add the neighbors for each edge
            for (j = 0; j <= p; j++)
            {
               if (neighborElem[0] >= 0)
                  map_for_bounds[k*nd+j].push_back(neighborElem[0]);
               if (neighborElem[1] >= 0)
                  map_for_bounds[k*nd+(j+1)*(p+1)-1].push_back(neighborElem[1]);
               if (neighborElem[2] >= 0)
                  map_for_bounds[k*nd+p*(p+1)+j].push_back(neighborElem[2]);
               if (neighborElem[3] >= 0)
                  map_for_bounds[k*nd+j*(p+1)].push_back(neighborElem[3]);
               if (neighborElem[4] >= 0)
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+j].push_back(neighborElem[4]);
               if (neighborElem[5] >= 0)
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+(j+1)*(p+1)-1].push_back(neighborElem[5]);
               if (neighborElem[6] >= 0)
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+p*(p+1)+j].push_back(neighborElem[6]);
               if (neighborElem[7] >= 0)
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+j*(p+1)].push_back(neighborElem[7]);
               if (neighborElem[8] >= 0)
                  map_for_bounds[k*nd+j*(p+1)*(p+1)].push_back(neighborElem[8]);
               if (neighborElem[9] >= 0)
                  map_for_bounds[k*nd+p+j*(p+1)*(p+1)].push_back(neighborElem[9]);
               if (neighborElem[10] >= 0)
                  map_for_bounds[k*nd+(j+1)*(p+1)*(p+1)-1].push_back(neighborElem[10]);
               if (neighborElem[11] >= 0)
                  map_for_bounds[k*nd+p*(p+1)+j*(p+1)*(p+1)].push_back(neighborElem[11]);
            }
            
            // cube vertices
            nbr_id = FindCommonAdjacentElement(neighborElements[0], neighborElem[0], neighborElem[3], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[0], neighborElem[0], neighborElem[1], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+p].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[0], neighborElem[2], neighborElem[3], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+p*(p+1)].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[0], neighborElem[1], neighborElem[2], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+(p+1)*(p+1)-1].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[5], neighborElem[4], neighborElem[7], dim, numBdrs); 
            if (nbr_id >= 0) { map_for_bounds[k*nd+(p+1)*(p+1)*p].push_back(nbr_id);}
            
            nbr_id = FindCommonAdjacentElement(neighborElements[5], neighborElem[4], neighborElem[5], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+(p+1)*(p+1)*p+p].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[5], neighborElem[6], neighborElem[7], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+(p+1)*(p+1)*p+(p+1)*p].push_back(nbr_id); }
            
            nbr_id = FindCommonAdjacentElement(neighborElements[5], neighborElem[5], neighborElem[6], dim, numBdrs);
            if (nbr_id >= 0) { map_for_bounds[k*nd+(p+1)*(p+1)*(p+1)-1].push_back(nbr_id); }
         }
      }
   }
};

class FluxCorrectedTransport
{
public:
   // Constructor builds structures required for low order scheme
   FluxCorrectedTransport(FiniteElementSpace* _fes, const MONOTYPE _monoType, bool &_schemeOpt,
                          const SparseMatrix &K, VectorFunctionCoefficient &coef, SolutionBounds &_bnds) :
      fes(_fes), monoType(_monoType), schemeOpt(_schemeOpt), bnds(_bnds), velocity(coef)
   {
      // NOTE: D is initialized later, due to the need to have identical sparisty with corresponding 
      //       advection operator K or preconditioned volume terms, depending on the scheme.
      
      const FiniteElement &dummy = *fes->GetFE(0);
      int p = dummy.GetOrder();
      
      if (monoType == None) { return; }
      
      else if ((monoType == DiscUpw) || (monoType == DiscUpw_FCT))
      {
         if (schemeOpt)
         {
            mfem_error("PDU not implemented.");
         }
         else
         {
            // NOTE: This is the most basic low order scheme. It works completely
            //       independent of the operator. It is used as comparison, 
            //       because all other schemes are new. This is the only scheme 
            //       NOT using the FluxLumping routines. Monotonicity for flux 
            //       terms is ensured via matrix D. The 1D case is not handled
            //       differnetly from the multi-dimensional one (contrary to PDU)
            // Matrix D is assembled in every RK stage.
         }
      }
      else
      {
         if ((p==1) && schemeOpt)
         {
            mfem_warning("Subcell option does not make sense for order 1. Using cell-based scheme.");
            schemeOpt = false;
         }
         
         Mesh* mesh = fes->GetMesh();
         int k, numDofs, numBdrs, ne = mesh->GetNE(), nd = dummy.GetDof(), dim = mesh->Dimension();
         if (dim==1)
         {
            numSubcells = p;
            numDofsSubcell = 2;
         }
         else if (dim==2)
         {
            numSubcells = p*p;
            numDofsSubcell = 4;
         }
         else if (dim==3)
         {
            numSubcells = p*p*p;
            numDofsSubcell = 8;
         }
          
         // fill the dofs array to access the correct dofs for boundaries later
         dummy.ExtractBdrDofs(dofs);
         numDofs = dofs.Height();
         numBdrs = dofs.Width();

         FillSubcell2CellDof(p, dim);
         
         neighborDof.SetSize(ne*numDofs, numBdrs);
         
         Array <int> bdrs, orientation;
         
         for (k = 0; k < ne; k++)
         {
            if (dim==1) { /* Nothing needs to be done for 1D boundaries */ }
            else if (dim==2)
               mesh->GetElementEdges(k, bdrs, orientation);
            else if (dim==3)
               mesh->GetElementFaces(k, bdrs, orientation);

            FillNeighborDofs(mesh, numDofs, k, nd, p, dim, bdrs);
         }
      }
   }

   // Computes the element-global indices from the indices of the subcell and the indices
   // of dofs on the subcell.
   // NOTE: Here it is assumed that the mesh consists of segments, quads or hexes.
   void FillSubcell2CellDof(int p, int dim)
   {
      subcell2CellDof.SetSize(numSubcells, numDofsSubcell);
      for (int m = 0; m < numSubcells; m++)
      {
         for (int j = 0; j < numDofsSubcell; j++)
         {
            if (dim == 1)
            {
               subcell2CellDof(m,j) = m + j;
            }
            else if (dim == 2)
            {
               switch (j)
               {
                  case 0: subcell2CellDof(m,j) =  m + (m / p); break;
                  case 1: subcell2CellDof(m,j) =  m + (m / p) + 1; break;
                  case 2: subcell2CellDof(m,j) =  m + (m / p) + p + 1; break;
                  case 3: subcell2CellDof(m,j) =  m + (m / p) + p + 2; break;
               }
            }
            else if (dim == 3)
            {
               switch (j)
               {
                  case 0: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)); break;
                  case 1: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + 1; break;
                  case 2: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + p + 1; break;
                  case 3: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + p + 2; break;
                  case 4: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + (p+1)*(p+1); break;
                  case 5: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + (p+1)*(p+1) + 1; break;
                  case 6: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + (p+1)*(p+1) + p + 1; break;
                  case 7: subcell2CellDof(m,j) =  m + (m / p) + (p+1) * (m / (p*p)) + (p+1)*(p+1) + p + 2; break;
               }
            }
         }
      }
   }

   void FillNeighborDofs(Mesh *mesh, int numDofs, int k, int nd, int p, int dim,
                         Array <int> bdrs)
   {
      int i, j, neighborElem, numBdrs = dofs.Width();
      FaceElementTransformations *Trans;
      Array<int> neighborBdrs, orientation;

      if (dim == 1) { return; } // no need to take care of boundary terms
      else if (dim == 2)
      {
         for (j = 0; j < numDofs; j++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[0]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }
            
            mesh->GetElementEdges(neighborElem, neighborBdrs, orientation);
            for (i = 0; i < numBdrs; i++)
               if (neighborBdrs[i] == bdrs[0])
                  break;

            neighborDof(k*numDofs+j, 0) = neighborElem*nd + dofs(numDofs-1-j,i);

            Trans = mesh->GetFaceElementTransformations(bdrs[1]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            mesh->GetElementEdges(neighborElem, neighborBdrs, orientation);
            for (i = 0; i < numBdrs; i++)
               if (neighborBdrs[i] == bdrs[1])
                  break;

            neighborDof(k*numDofs+j, 1) = neighborElem*nd + dofs(numDofs-1-j,i);

            Trans = mesh->GetFaceElementTransformations(bdrs[2]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            mesh->GetElementEdges(neighborElem, neighborBdrs, orientation);
            for (i = 0; i < numBdrs; i++)
               if (neighborBdrs[i] == bdrs[2])
                  break;

            neighborDof(k*numDofs+j, 2) = neighborElem*nd + dofs(numDofs-1-j,i);

            Trans = mesh->GetFaceElementTransformations(bdrs[3]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            mesh->GetElementEdges(neighborElem, neighborBdrs, orientation);
            for (i = 0; i < numBdrs; i++)
               if (neighborBdrs[i] == bdrs[3])
                  break;

            neighborDof(k*numDofs+j, 3) = neighborElem*nd + dofs(numDofs-1-j,i);
         }
      }
      else // dim == 3
      {
         // NOTE: This will only work for meshes of uniformly ordered cube nodes.
         for (j = 0; j < numDofs; j++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[0]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j, 0) = neighborElem*nd + (p+1)*(p+1)*p+j;

            Trans = mesh->GetFaceElementTransformations(bdrs[1]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j,
                        1) = neighborElem*nd + (j/(p+1))*(p+1)*(p+1) + (p+1)*p+(j%(p+1));

            Trans = mesh->GetFaceElementTransformations(bdrs[2]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j, 2) = neighborElem*nd + j*(p+1);

            Trans = mesh->GetFaceElementTransformations(bdrs[3]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j,
                        3) = neighborElem*nd + (j/(p+1))*(p+1)*(p+1) + (j%(p+1));

            Trans = mesh->GetFaceElementTransformations(bdrs[4]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j, 4) = neighborElem*nd + (j+1)*(p+1)-1;

            Trans = mesh->GetFaceElementTransformations(bdrs[5]);
            if (Trans->Elem1No == k)
            {
               neighborElem = Trans->Elem2No;
            }
            else
            {
               neighborElem = Trans->Elem1No;
            }

            neighborDof(k*numDofs+j, 5) = neighborElem*nd + j;
         }
      }
   }

   // Destructor
   ~FluxCorrectedTransport() { }

   // member variables that need to be accessed during time-stepping
   FiniteElementSpace* fes;
   VectorFunctionCoefficient &velocity;
   
   const MONOTYPE monoType;

   bool schemeOpt;
   int numSubcells, numDofsSubcell;
   DenseMatrix dofs, neighborDof, subcell2CellDof;
   SolutionBounds &bnds;
};


void preprocessFluxLumping(const FluxCorrectedTransport &fct, const int k, const IntegrationRule *irF, 
                           DenseMatrix &bdrIntLumped, DenseMatrix &bdrInt)
{
   FiniteElementSpace *fes = fct.fes;
   VectorFunctionCoefficient &coef = fct.velocity;
   DenseMatrix dofs = fct.dofs;
   
   const FiniteElement &dummy = *fes->GetFE(k);
   Mesh *mesh = fes->GetMesh();
   
   int i, j, l, m, idx, numBdrs = dofs.Width(), numDofs = dofs.Height(), 
         nd = dummy.GetDof(), p = dummy.GetOrder(), dim = mesh->Dimension();
   double vn;
   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;
   
   dummy.ExtractBdrDofs(dofs); //TODO rm?
   numBdrs = dofs.Width();
   numDofs = dofs.Height();
   
   Vector vval, nor(dim), shape(nd);
   
   if (dim==1)
      numBdrs = 0; // Nothing needs to be done for 1D boundaries
   else if (dim==2)
      mesh->GetElementEdges(k, bdrs, orientation);
   else if (dim==3)
      mesh->GetElementFaces(k, bdrs, orientation);
   
   for (i = 0; i < numBdrs; i++)
   {
      Trans = mesh->GetFaceElementTransformations(bdrs[i]);
      vn = 0.;
      
      for (l = 0; l < irF->GetNPoints(); l++)
      {
         const IntegrationPoint &ip = irF->IntPoint(l);
         IntegrationPoint eip1;
         Trans->Face->SetIntPoint(&ip);
         
         if (dim == 1)
         {
            nor(0) = 2.*eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Trans->Face->Jacobian(), nor);
         }
         
         if (Trans->Elem1No != k)
         {
            Trans->Loc2.Transform(ip, eip1);
            dummy.CalcShape(eip1, shape);
            Trans->Elem2->SetIntPoint(&eip1);
            coef.Eval(vval, *Trans->Elem2, eip1);
            nor *= -1.;
            Trans->Loc1.Transform(ip, eip1);
         }
         else
         {
            Trans->Loc1.Transform(ip, eip1);
            dummy.CalcShape(eip1, shape);
            Trans->Elem1->SetIntPoint(&eip1);
            coef.Eval(vval, *Trans->Elem1, eip1);
            Trans->Loc2.Transform(ip, eip1);
         }
         
         nor /= nor.Norml2();
         vn = min(0., vval * nor);
         
         for(j = 0; j < numDofs; j++)
         {
            bdrIntLumped(k*nd+dofs(j,i),i) -= ip.weight * 
            Trans->Face->Weight() * shape(dofs(j,i)) * vn;
      
            for (m = 0; m < numDofs; m++)
            {
               bdrInt(k*nd+dofs(j,i),i*nd+dofs(m,i)) -= ip.weight * 
               Trans->Face->Weight() * shape(dofs(j,i)) * shape(dofs(m,i)) * vn;
            }
         }
      }
   }
}

void ComputeResidualWeights(const FluxCorrectedTransport &fct, SparseMatrix &fluctMatrix,
                            DenseMatrix &fluctSub, DenseMatrix &bdrIntLumped, DenseMatrix &bdrInt)
{
   FiniteElementSpace *fes = fct.fes;
   VectorFunctionCoefficient &coef = fct.velocity;
   
   Mesh *mesh = fes->GetMesh();
   int i, j, k, m, p, nd, dofInd, qOrdF, numBdrs = fct.dofs.Width(),
       dim = mesh->Dimension(), ne = mesh->GetNE();
   DenseMatrix elmat;
   ElementTransformation *tr;
   FaceElementTransformations *Trans;
   
   // use the first mesh element as indicator for the following bunch
   const FiniteElement &dummy = *fes->GetFE(0);
   nd = dummy.GetDof();
   p = dummy.GetOrder();
   
   const IntegrationRule *irF = GetFaceIntRule(fes);
   
   BilinearFormIntegrator *fluct;
   fluct = new MixedConvectionIntegrator(coef);
   
   BilinearForm VolumeTerms(fes);
   VolumeTerms.AddDomainIntegrator(new ConvectionIntegrator(coef));
   VolumeTerms.Assemble();
   VolumeTerms.Finalize();
   fluctMatrix = VolumeTerms.SpMat();
   
   Mesh *ref_mesh = GetSubcellMesh(mesh, p);

   const int btype = BasisType::Positive;
   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);
   
   FiniteElementSpace SubFes0(ref_mesh, &fec0);
   FiniteElementSpace SubFes1(ref_mesh, &fec1);

   fluctSub.SetSize(ne*fct.numSubcells, fct.numDofsSubcell);
   bdrIntLumped.SetSize(ne*nd, numBdrs); bdrIntLumped = 0.;
   bdrInt.SetSize(ne*nd, nd*numBdrs); bdrInt = 0.;

   for (k = 0; k < ne; k++)
   {
      ////////////////////////////
      // Boundary contributions //
      ////////////////////////////
      preprocessFluxLumping(fct, k, irF, bdrIntLumped, bdrInt);

      ///////////////////////////
      // Element contributions //
      ///////////////////////////
      for (m = 0; m < fct.numSubcells; m++)
      {
         dofInd = fct.numSubcells*k+m;
         const FiniteElement *el0 = SubFes0.GetFE(dofInd);
         const FiniteElement *el1 = SubFes1.GetFE(dofInd);
         tr = ref_mesh->GetElementTransformation(dofInd);
         fluct->AssembleElementMatrix2(*el1, *el0, *tr, elmat);
         
         for (j = 0; j < fct.numDofsSubcell; j++)
            fluctSub(dofInd, j) = elmat(0,j);
      }
   }
   if (p!=1)
   {
      delete ref_mesh;
   }
   delete fluct;
}


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   FiniteElementSpace* fes;
   BilinearForm &Mbf, &Kbf, &kbdr, &ml;
   SparseMatrix &M, &K, &KBDR;
   const Vector &b;

   Vector start_pos;
   Vector &lumpedM;
   GridFunction &mesh_pos, &vel_pos;

   mutable Vector z;

   double dt, start_t;
   const FluxCorrectedTransport &fct;

public:
   FE_Evolution(FiniteElementSpace* fes,
                BilinearForm &Mbf_, BilinearForm &Kbf_,
                SparseMatrix &_M, SparseMatrix &_K,
                const Vector &_b, FluxCorrectedTransport &_fct,
                GridFunction &mpos, GridFunction &vpos, 
                BilinearForm &_kbdr, SparseMatrix &_KBDR,
                BilinearForm &_ml, Vector &_lumpedM);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetDt(double _dt) { dt = _dt; }
   void SetInitialTimeStepTime(double st) { start_t = st; }
   void SetRemapStartPos(Vector &spos) { start_pos = spos; }
   
   virtual void NeumannSolve(const Vector &b, Vector &x) const;

   virtual void LinearFluxLumping(int k, int nd, const Vector &x, Vector &y,
                                  const DenseMatrix &bdrInt, const DenseMatrix &bdrIntLumped) const;

   virtual void ComputeHighOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeLowOrderSolution(const Vector &x, Vector &y) const;
   virtual void ComputeFCTSolution(const Vector &x, const Vector &yH,
                                   const Vector &yL, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 4;
   const char *mesh_file = "../data/periodic-square.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 3;
   MONOTYPE monoType = ResDist_FCT;
   bool schemeOpt = true;
   double t_final = 4.0;
   double dt = 0.005;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 100;

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
                  "                                1 - discrete upwinding - low order,\n\t"
                  "                                2 - discrete upwinding - FCT,\n\t"
                  "                                3 - residual distribution scheme (matrix-free) - low order,\n\t"
                  "                                4 - residual distribution scheme (matrix-free) - FCT.");
   args.AddOption((int*)(&schemeOpt), "-opt", "--schemeOpt",
                  "Optimized scheme: preconditioned discrete upwinding or subcell version: 0 - basic schemes,\n\t"
                  "                                                                        1 - optimized schemes.");
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

   // Current mesh positions.
   GridFunction *x = mesh->GetNodes();

   // Store initial positions.
   Vector x0(x->Size());
   x0 = *x;

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   FiniteElementSpace fes(mesh, &fec);

   if (monoType != None)
   {
      if (((int)monoType != monoType) || (monoType < 0) || (monoType > 4))
      {
         cout << "Unsupported option for monotonicity treatment." << endl;
         delete mesh;
         delete ode_solver;
         return 5;
      }
      if (btype != 2)
      {
         cout << "Monotonicity treatment requires use of Bernstein basis." << endl;
         delete mesh;
         delete ode_solver;
         return 5;
      }
      if (order == 0)
      {
         delete mesh;
         delete ode_solver;
         cout << "No need to use monotonicity treatment for polynomial order 0." << endl;
         return 5;
      }
   }

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   //    Also prepare for the use of low and high order schemes.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   // Mesh velocity.
   GridFunction v_gf(x->FESpace());
   v_gf.ProjectCoefficient(velocity);
   if (mesh->bdr_attributes.Size() > 0)
   {
      // Zero it out on boundaries (not moving boundaries).
      Array<int> ess_bdr(mesh->bdr_attributes.Max()), ess_vdofs;
      ess_bdr = 1;
      x->FESpace()->GetEssentialVDofs(ess_bdr, ess_vdofs);
      for (int i = 0; i < v_gf.Size(); i++)
      {
         if (ess_vdofs[i] == -1) { v_gf(i) = 0.0; }
      }
   }

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);
   BilinearForm k(&fes);
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity));
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, -1.0, -0.5)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, -1.0, -0.5)));
   
   BilinearForm kbdr(&fes);
   kbdr.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, -1.0, -0.5)));
   kbdr.Assemble(0);
   kbdr.Finalize(0);
   
   // Compute the lumped mass matrix algebraicly
   Vector lumpedM;
   BilinearForm ml(&fes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

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
   SolutionBounds bnds(&fes, k);

   // Precompute data required for high and low order schemes
   FluxCorrectedTransport fct(&fes, monoType, schemeOpt, k.SpMat(), velocity, bnds);

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

   // check for conservation
   Vector tmp(lumpedM); //TODO
   double initialMass = lumpedM * u;

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(&fes, m, k, m.SpMat(), k.SpMat(), b, fct, *x, v_gf, kbdr, kbdr.SpMat(), ml, lumpedM);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      adv.SetDt(dt_real);

      // Move the mesh (and the solution).
      add(x0, dt_real, v_gf, *x);
      adv.SetRemapStartPos(*x);
      adv.SetInitialTimeStepTime(t);

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

   // check for conservation
   double finalMass = tmp * u;
   cout << "Mass loss: " << abs(initialMass - finalMass) << endl;
   
   // Compute errors for problems, where the initial condition is equal to the final solution
   if (problem == 4) // solid body rotation
   {
      cout << "L1-error: " << u.ComputeLpError(1., u0) << ", L-Inf-error: "
           << u.ComputeLpError(numeric_limits<double>::infinity(), u0) << "." << endl;
   }

   // 10. Free the used memory.
   delete mesh;
   delete ode_solver;
   delete dc;

   return 0;
}


void FE_Evolution::NeumannSolve(const Vector &f, Vector &x) const
{   
   int i, iter, n = f.Size(), max_iter = 20;
   Vector y;
   double resid = f.Norml2(), abs_tol = 1.e-4;

   y.SetSize(n);
   x = 0.;

   for (iter = 1; iter <= max_iter; iter++)
   {
      M.Mult(x, y);
      y -= f;
      resid = y.Norml2();
      if (resid <= abs_tol)
      {
         return;
      }
      for (i = 0; i < n; i++)
      {
         x(i) -= y(i) / lumpedM(i);
      }
   }
}

void FE_Evolution::LinearFluxLumping(int k, int nd, const Vector &x, Vector &y, 
                                     const DenseMatrix &bdrInt, const DenseMatrix &bdrIntLumped) const
{
   int i, j, m, idx, dofInd, numBdrs(fct.dofs.Width()), numDofs(fct.dofs.Height());
   double xNeighbor, totalFlux;
   Vector xDiff(numDofs);
   
   for (j = 0; j < numBdrs; j++)
   {
      for (i = 0; i < numDofs; i++)
      {
         dofInd = k*nd+fct.dofs(i,j);
         idx = fct.neighborDof(k*numDofs+i,j);
         xNeighbor = idx < 0 ? 0. : x(idx);
         xDiff(i) = xNeighbor - x(dofInd);
      }
      
      for (i = 0; i < numDofs; i++)
      {
         dofInd = k*nd+fct.dofs(i,j);
         totalFlux = bdrIntLumped(dofInd, j) * xDiff(i);
         y(dofInd) += totalFlux;
      }
   }
}

void FE_Evolution::ComputeLowOrderSolution(const Vector &x, Vector &y) const
{
   if ((fct.monoType == DiscUpw) || (fct.monoType == DiscUpw_FCT))
   {
      int k, j, dofInd, nd;
      SparseMatrix D(K);
      
      ComputeDiscreteUpwindingMatrix(K, D);
      
      // Discretization terms
      K.Mult(x, y);
      y += b;
      
      // Monotonicity terms
      D.AddMult(x, y);
      
      // Division by lumped mass matrix and inclusion of boundary terms in case of PDU
      for (k = 0; k < fes->GetNE(); k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();

         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            y(dofInd) /= lumpedM(dofInd);
         }
      }
   }
   else
   {
      Mesh *mesh = fes->GetMesh();
      int i, j, k, m, nd, dofInd, dofInd2, loc, ne(fes->GetNE()), dim(mesh->Dimension());
      double xMax, xMin, xSum, xNeighbor, sumFluctSubcellP, sumFluctSubcellN,
             sumWeightsP, sumWeightsN, weightP, weightN, rhoP, rhoN, gammaP, gammaN,
             minGammaP, minGammaN, aux, fluct, beta = 10., gamma = 10., eps = 1.E-15;
      Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
             fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN, alpha;

      //
      SparseMatrix fluctMatrix;
      DenseMatrix fluctSub, bdrIntLumped, bdrInt;
      ComputeResidualWeights(fct, fluctMatrix, fluctSub, bdrIntLumped, bdrInt);
         
      // Discretization terms
      y = b;
//       KBDR.AddMult(x, y); //for debug
      fluctMatrix.Mult(x, z);
      if (dim==1)
      {
         K.AddMult(x, y);
         y -= z;
      }
//       z += y;
//       NeumannSolve(z, y); //for debug

      // Monotonicity terms
      for (k = 0; k < ne; k++)
      {
         const FiniteElement &el = *fes->GetFE(k);
         nd = el.GetDof();

         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         xMin = numeric_limits<double>::infinity();
         xMax = -xMin;
         rhoP = rhoN = xSum = 0.;
         alpha.SetSize(nd); alpha = 0.;

         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            xMax = max(xMax, x(dofInd));
            xMin = min(xMin, x(dofInd));
            xSum += x(dofInd);
         }
         
         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            y(dofInd) += alpha(j) * z(dofInd);
            z(dofInd) *= (1. - alpha(j));

            if (fct.schemeOpt)
            {
               rhoP += max(0., z(dofInd));
               rhoN += min(0., z(dofInd));
            }
         }
         
         ////////////////////////////
         // Boundary contributions //
         ////////////////////////////
         if (dim > 1)// Nothing needs to be done for 1D boundaries (due to Bernstein basis)
           LinearFluxLumping(k, nd, x, y, bdrInt, bdrIntLumped);
         
         sumWeightsP = nd*xMax - xSum + eps;
         sumWeightsN = nd*xMin - xSum - eps;
         
         if (fct.schemeOpt)
         {
            fluctSubcellP.SetSize(fct.numSubcells);
            fluctSubcellN.SetSize(fct.numSubcells);
            xMaxSubcell.SetSize(fct.numSubcells);
            xMinSubcell.SetSize(fct.numSubcells);
            nodalWeightsP.SetSize(nd);
            nodalWeightsN.SetSize(nd);
            sumWeightsSubcellP.SetSize(fct.numSubcells);
            sumWeightsSubcellN.SetSize(fct.numSubcells);
            sumFluctSubcellP = sumFluctSubcellN = 0.;
            nodalWeightsP = 0.; nodalWeightsN = 0.;
            
            // compute min-/max-values and the fluctuation for subcells
            for (m = 0; m < fct.numSubcells; m++)
            {
               xMinSubcell(m) = numeric_limits<double>::infinity();
               xMaxSubcell(m) = -xMinSubcell(m);
               fluct = xSum = 0.;
               for (i = 0; i < fct.numDofsSubcell; i++)
               {
                  dofInd = k*nd + fct.subcell2CellDof(m, i);
                  fluct += fluctSub(k*fct.numSubcells+m,i) * x(dofInd);
                  xMaxSubcell(m) = max(xMaxSubcell(m), x(dofInd));
                  xMinSubcell(m) = min(xMinSubcell(m), x(dofInd));
                  xSum += x(dofInd);
               }
               sumWeightsSubcellP(m) = fct.numDofsSubcell * xMaxSubcell(m) - xSum + eps;
               sumWeightsSubcellN(m) = fct.numDofsSubcell * xMinSubcell(m) - xSum - eps;

               fluctSubcellP(m) = max(0., fluct);
               fluctSubcellN(m) = min(0., fluct);
               sumFluctSubcellP += fluctSubcellP(m);
               sumFluctSubcellN += fluctSubcellN(m);
            }
            
            for (m = 0; m < fct.numSubcells; m++)
            {
               for (i = 0; i < fct.numDofsSubcell; i++)
               {
                  loc = fct.subcell2CellDof(m, i);
                  dofInd = k*nd + loc;
                  nodalWeightsP(loc) += fluctSubcellP(m) * ((xMaxSubcell(m) - x(dofInd)) / sumWeightsSubcellP(m)); // eq. (10)
                  nodalWeightsN(loc) += fluctSubcellN(m) * ((xMinSubcell(m) - x(dofInd)) / sumWeightsSubcellN(m)); // eq. (11)
               }
            }
         }

         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i;
            weightP = (xMax - x(dofInd)) / sumWeightsP;
            weightN = (xMin - x(dofInd)) / sumWeightsN;
            
            if (fct.schemeOpt)
            {
               aux = gamma / (rhoP + eps);
               weightP *= 1. - min(aux * sumFluctSubcellP, 1.);
               weightP += min(aux, 1. / (sumFluctSubcellP + eps)) * nodalWeightsP(i);
               
               aux = gamma / (rhoN - eps);
               weightN *= 1. - min(aux * sumFluctSubcellN, 1.);
               weightN += max(aux, 1. / (sumFluctSubcellN - eps)) * nodalWeightsN(i);
            }
            
            for (j = 0; j < nd; j++)
            {
               dofInd2 = k*nd+j;
               if (z(dofInd2) > eps)
               {
                  y(dofInd) += (1. - alpha(j)) * weightP * z(dofInd2);
               }
               else if (z(dofInd2) < -eps)
               {
                  y(dofInd) += (1. - alpha(j)) * weightN * z(dofInd2);
               }
            }
            y(dofInd) = (y(dofInd) + alpha(i) * z(dofInd)) / lumpedM(dofInd);
         }
      }
   }
}

void FE_Evolution::ComputeHighOrderSolution(const Vector &x, Vector &y) const
{
   // No monotonicity treatment, straightforward high-order scheme
   // ydot = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   NeumannSolve(z, y);
}

void FE_Evolution::ComputeFCTSolution(const Vector &x, const Vector &yH,
                                      const Vector &yL, Vector &y) const
{
   // High order reconstruction that yields an updated admissible solution by means of
   // clipping the solution coefficients within certain bounds and scaling the anti-
   // diffusive fluxes in a way that leads to local conservation of mass.
   int j, k, nd, dofInd;
   double sumPos, sumNeg, eps = 1.E-15;
   Vector uClipped, fClipped;
   
   // compute solution bounds
   fct.bnds.ComputeVertexBounds(x);

   // Monotonicity terms
   for (k = 0; k < fes->GetMesh()->GetNE(); k++)
   {
      const FiniteElement &el = *fes->GetFE(k);
      nd = el.GetDof();

      uClipped.SetSize(nd); uClipped = 0.;
      fClipped.SetSize(nd); fClipped = 0.;
      sumPos = sumNeg = 0.;
      
      for (j = 0; j < nd; j++)
      {
         dofInd = k*nd+j;
         uClipped(j) = min(fct.bnds.x_max(dofInd), max(x(dofInd) + dt * yH(dofInd), fct.bnds.x_min(dofInd)));
         // compute coefficients for the high-order corrections
         fClipped(j) = lumpedM(dofInd) * (uClipped(j) - ( x(dofInd) + dt * yL(dofInd) ));

         sumPos += max(fClipped(j), 0.);
         sumNeg += min(fClipped(j), 0.);
      }

      for (j = 0; j < nd; j++)
      {
         if ((sumPos + sumNeg > eps) && (fClipped(j) > eps))
         {
            fClipped(j) *= - sumNeg / sumPos;
         }
         if ((sumPos + sumNeg < -eps) && (fClipped(j) < -eps))
         {
            fClipped(j) *= - sumPos / sumNeg;
         }

         dofInd = k*nd+j;
         // yH is high order discrete time derivative
         // yL is low order discrete time derivative
         y(dofInd) = yL(dofInd) + fClipped(j) / (dt * lumpedM(dofInd));
         // y is now the discrete time derivative featuring the high order anti-diffusive
         // reconstruction that leads to an forward Euler updated admissible solution.
         // The factor dt in the denominator is used for compensation in the ODE solver.
      }
   }
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace* _fes,
                           BilinearForm &Mbf_, BilinearForm &Kbf_,
                           SparseMatrix &_M, SparseMatrix &_K,
                           const Vector &_b, FluxCorrectedTransport &_fct,
                           GridFunction &mpos, GridFunction &vpos,
                           BilinearForm &_kbdr, SparseMatrix &_KBDR,
                           BilinearForm &_ml, Vector &_lumpedM)
   : TimeDependentOperator(_M.Size()), fes(_fes),
     Mbf(Mbf_), Kbf(Kbf_), M(_M), K(_K), b(_b),
     z(_M.Size()), fct(_fct), kbdr(_kbdr), KBDR(_KBDR),
     start_pos(mpos.Size()), mesh_pos(mpos), vel_pos(vpos),
     ml(_ml), lumpedM(_lumpedM) { }

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Move towards x0 with current t.
   const double t = GetTime();
   const double sub_time_step = t - start_t;
   add(start_pos, -sub_time_step, vel_pos, mesh_pos);

   // Reassemble on the new mesh (given by mesh_pos).
   Mbf.BilinearForm::operator=(0.0);
   Mbf.Assemble();
   Kbf.BilinearForm::operator=(0.0);
   Kbf.Assemble(0);
   kbdr.BilinearForm::operator=(0.0);
   kbdr.Assemble(0);
   ml.BilinearForm::operator=(0.0);
   ml.Assemble();
   ml.SpMat().GetDiag(lumpedM);

   if (fct.monoType == 0)
   {
      ComputeHighOrderSolution(x, y);
   }
   else 
   {
      if (fct.monoType % 2 == 1)
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
            //case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 2: v(0) = 0.0; v(1) = 1.0; break;
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
            case 2: v(0) = -w*X(1); v(1) = w*X(0); break;
            case 3: v(0) = -w*X(1); v(1) = w*X(0); v(2) = 0.0; break;
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
      case 5:
      {
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = 1.0; v(1) = 1.0; break;
            case 3: v(0) = 1.0; v(1) = 1.0; v(2) = 1.0; break;
         }
      }
   }
}

double box(std::pair<double,double> p1, std::pair<double,double> p2, double theta, 
      std::pair<double,double> origin, double x, double y)
{
   double xmin=p1.first;
   double xmax=p2.first;
   double ymin=p1.second;
   double ymax=p2.second;
   double ox=origin.first;
   double oy=origin.second;
   
   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);
   
   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;
   
   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax)
      return 1.0;
   else 
      return 0.0;
}

double box3D(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, 
             double theta, double ox, double oy, double x, double y, double z)
{
   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);
   
   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;
   
   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax && z>zmin && z<zmax)
      return 1.0;
   else 
      return 0.0;
}

double get_cross(double rect1, double rect2)
{
   double intersection=rect1*rect2;
   return rect1+rect2-intersection; //union
}

double ring(double rin, double rout, Vector c, Vector y)
{
   double r = 0.;
   int dim = c.Size();
   if (dim != y.Size())
   {
      mfem_error("Origin vector and variable have to be of the same size.");
   }
   for (int i = 0; i < dim; i++)
   {
      r += pow(y(i)-c(i), 2.);
   }
   r = sqrt(r);
   if (r>rin && r<rout)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
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
         double scale = 0.0225;
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = (0.5/sqrt(scale)) * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = (0.5/sqrt(scale)) * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1) - 0.5,2.)) <= 4.*scale)) ? 1. : 0.
                + (1.-cone) * (pow(X(0), 2.) + pow(X(1) + 0.5, 2.) <= 4.*scale)
                + 0.25*(1.+cos(M_PI*hump))*((pow(X(0) + 0.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
      }
      case 5:
      {
         Vector y(dim);
         for (int i = 0; i < dim; i++) { y(i) = 50. * (x(i) + 1.); }
         
         if (dim==1)
         {
            mfem_error("This test is not supported in 1D.");
         }
         else if (dim==2)
         {
            std::pair<double, double> p1;
            std::pair<double, double> p2;
            std::pair<double, double> origin;
            
            // cross
            p1.first=14.; p1.second=3.;
            p2.first=17.; p2.second=26.;
            origin.first = 15.5;
            origin.second = 11.5;
            double rect1=box(p1,p2,-45.,origin,y(0),y(1));
            p1.first=7.; p1.second=10.;
            p2.first=32.; p2.second=13.;
            double rect2=box(p1,p2,-45.,origin,y(0),y(1));
            double cross=get_cross(rect1,rect2);
            // rings
            Vector c(dim);
            c(0) = 40.; c(1) = 40;
            double ring1 = ring(7., 10., c, y);
            c(1) = 20.;
            double ring2 = ring(3., 7., c, y);
            
            return cross + ring1 + ring2;
         }
         else
         {
            // cross
            double rect1 = box3D(7.,32.,10.,13.,10.,13.,-45.,15.5,11.5,y(0),y(1),y(2));
            double rect2 = box3D(14.,17.,3.,26.,10.,13.,-45.,15.5,11.5,y(0),y(1),y(2));
            double rect3 = box3D(14.,17.,10.,13.,3.,26.,-45.,15.5,11.5,y(0),y(1),y(2));
            
            double cross = get_cross(get_cross(rect1, rect2), rect3);
            
            // rings
            Vector c1(dim), c2(dim);
            c1(0) = 40.; c1(1) = 40; c1(2) = 40.;
            c2(0) = 40.; c2(1) = 20; c2(2) = 20.;
            
            double shell1 = ring(7., 10., c1, y);
            double shell2 = ring(3., 7., c2, y);
            
            double dom2 = cross + shell1 + shell2;
            
            // cross
            rect1 = box3D(2.,27.,30.,33.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect2 = box3D(9.,12.,23.,46.,30.,33.,0.,0.,0.,y(0),y(1),y(2));
            rect3 = box3D(9.,12.,30.,33.,23.,46.,0.,0.,0.,y(0),y(1),y(2));
            
            cross = get_cross(get_cross(rect1, rect2), rect3);
            
            double ball1 = ring(0., 7., c1, y);
            double ball2 = ring(0., 3., c2, y);
            double shell3 = ring(7., 10., c2, y);
            
            double dom3 = cross + ball1 + ball2 + shell3;
            
            double dom1 = 1. - get_cross(dom2, dom3);
            
            return dom1 + 2.*dom2 + 3.*dom3;
         }
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
      case 4: 
      case 5: return 0.0;
   }
   return 0.0;
}
