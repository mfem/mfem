//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//
//    Standard transport mode:
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
//    ex9 -m ../data/periodic-square.mesh -p 4 -r 4 -dt 0.001 -o 2 -mt 3
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 2 -dt 0.0025 -o 15 -tf 9 -mt 4
//    ex9 -m ../data/periodic-cube.mesh -p 5 -r 5 -dt 0.0001 -o 1 -tf 0.8 -mt 4

//
//    Standard remap mode:
//    ex9 -m ../data/periodic-square.mesh -p 10 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//    ex9 -m ../data/periodic-square.mesh -p 11 -r 3 -dt 0.005 -tf 0.5 -mt 4 -vs 10
//
//    Lagrangian step followed by mesh return mode:
//    ex9 -m ../data/periodic-square.mesh -p 20 -r 3 -dt 0.005 -tf 4 -mt 4 -vs 10
//    ex9 -m ../data/periodic-square.mesh -p 21 -r 3 -dt 0.005 -tf 4 -mt 4 -vs 10
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

// 0 is standard transport.
// 1 is standard remap (mesh moves, solution is fixed).
// 2 is Lagrangian step followed by mesh return.
int exec_mode;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

enum MONOTYPE { None, DiscUpw, DiscUpw_FCT, ResDist, ResDist_FCT };

struct LowOrderMethod
{
	FiniteElementSpace* fes;
	Array <int> smap;
	SparseMatrix D;
	BilinearForm* pk = NULL;
};

// Utility function to build a map to the offset
// of the symmetric entry in a sparse matrix.
Array<int> SparseMatrix_Build_smap(const SparseMatrix &A)
{
   // Assuming that A is finalized
   const int *I = A.GetI(), *J = A.GetJ(), n = A.Size();
   Array<int> smap;
   smap.SetSize(I[n]);

   for (int row = 0, j = 0; row < n; row++)
   {
      for (int end = I[row+1]; j < end; j++)
      {
         int col = J[j];
         // Find the offset, _j, of the (col,row) entry
         // and store it in smap[j].
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

// Given a matrix K, matrix D (initialized with same sparsity as K) 
// is computed, such that (K+D)_ij >= 0 for i != j.
void ComputeDiscreteUpwindingMatrix(const SparseMatrix& K,
												Array<int> smap, SparseMatrix& D)
{
   const int n = K.Size();
   int* Ip = K.GetI();
   int* Jp = K.GetJ();
   double* Kp = K.GetData();

   double* Dp = D.GetData();

   for (int i = 0, k = 0; i < n; i++)
   {
      double rowsum = 0.;
      for (int end = Ip[i+1]; k < end; k++)
      {
         int j = Jp[k];
         double kij = Kp[k];
         double kji = Kp[smap[k]];
         double dij = fmax(fmax(0.0,-kij),-kji);
         Dp[k] = kij + dij;
         Dp[smap[k]] = kji + dij;
         if (i != j) { rowsum += dij; }
      }
      D(i,i) = K(i,i) -rowsum;
   }
}

// The mesh corresponding to Bezier subcells of order p is constructed.
// NOTE: The mesh is assumed to consist of segments, quads or hexes.
// TODO It seems that for an original mesh with periodic boundaries, this 
// routine produces a mesh that cannot be used for its intended purpose.
Mesh* GetSubcellMesh(Mesh *mesh, int p)
{
   Mesh *ref_mesh;
   if (p==1)
   {
      ref_mesh = mesh;
   }
   else if (mesh->Dimension() > 1)
   {
      int basis_lor = BasisType::ClosedUniform; // Get a uniformly refined mesh.
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

// Appropriate quadrature rule for faces of is obtained.
// TODO check if this gives the desired order. I use the same order
// for all faces. In DGTraceIntegrator it uses the min of OrderW, why?
const IntegrationRule *GetFaceIntRule(FiniteElementSpace *fes)
{
   int i, qOrdF;
   Mesh* mesh = fes->GetMesh();
   FaceElementTransformations *Trans;
   
   // Use the first mesh face with two elements as indicator.
   for (i = 0; i < mesh->GetNumFaces(); i++)
   {
      Trans = mesh->GetFaceElementTransformations(i);
      qOrdF = Trans->Elem1->OrderW();
      if (Trans->Elem2No >= 0)
      {
         // qOrdF is chosen such that L2-norm of basis functions is
         // computed accurately.
         qOrdF = max(qOrdF, Trans->Elem2->OrderW());
         break;
      }
   }
   // Use the first mesh element as indicator.
   const FiniteElement &dummy = *fes->GetFE(0);
   qOrdF += 2*dummy.GetOrder();

   return &IntRules.Get(Trans->FaceGeom, qOrdF);
}

// Class storing information on dofs needed for the low order methods and FCT.
class DofInfo
{
   Mesh* mesh;
   FiniteElementSpace* fes;

public:

   // For each dof the elements containing that vertex are stored.
   mutable std::map<int, std::vector<int> > map_for_bounds;

   Vector xi_min, xi_max; // min/max values for each dof
   Vector xe_min, xe_max; // min/max values for each element
   
   DenseMatrix BdrDofs, NbrDof, Sub2Ind; // TODO should these be Tables?
   int dim, numBdrs, numDofs, numSubcells, numDofsSubcell;

   DofInfo(FiniteElementSpace* _fes)
   {
      fes = _fes;
      mesh = fes->GetMesh();
      dim = mesh->Dimension();
      
		int n = fes->GetVSize();
      int ne = mesh->GetNE();
      
		xi_min.SetSize(n);
		xi_max.SetSize(n);
      xe_min.SetSize(ne);
      xe_max.SetSize(ne);

      // Use the first mesh element as indicator.
      const FiniteElement &dummy = *fes->GetFE(0);
      dummy.ExtractBdrDofs(BdrDofs);
      numDofs = BdrDofs.Height();
      numBdrs = BdrDofs.Width();
      
      GetVertexBoundsMap();  // Fill map_for_bounds.
      FillNeighborDofs();    // Fill NbrDof.
      FillSubcell2CellDof(); // Fill Sub2Ind.
   }

   // Computes the admissible interval of values for one dof from the min and
   // max values of all elements that feature a dof at this physical location.
   // It is assumed that a low order method has computed the min/max values for
   // each element.
   void ComputeVertexBounds(const Vector& x, const int dofInd)
   {
      xi_min(dofInd) = numeric_limits<double>::infinity();
      xi_max(dofInd) = -xi_min(dofInd);

      for (int i = 0; i < (int)map_for_bounds[dofInd].size(); i++)
      {
         xi_max(dofInd) = max(xi_max(dofInd),xe_max(map_for_bounds[dofInd][i]));
         xi_min(dofInd) = min(xi_min(dofInd),xe_min(map_for_bounds[dofInd][i]));
      }
   }
   
   // Destructor
   ~DofInfo() { }

private:

   // Returns element sharing a face with both el1 and el2, but is not el.
   // NOTE: This approach will not work for meshes with hanging nodes.
   // NOTE: The same geometry for all elements is assumed.
   int GetCommonElem(int el, int el1, int el2)
   {
      if (min(el1, el2) < 0) { return -1; }

      int i, j, commonNeighbor;
      bool found = false;
      Array<int> bdrs1, bdrs2, orientation, NbrEl1, NbrEl2;
      FaceElementTransformations *Trans;

      NbrEl1.SetSize(numBdrs); NbrEl2.SetSize(numBdrs);

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

      // get lists of all neighbors of el1 and el2
      for (i = 0; i < numBdrs; i++)
      {
         Trans = mesh->GetFaceElementTransformations(bdrs1[i]);
         NbrEl1[i] = Trans->Elem1No != el1 ? Trans->Elem1No : Trans->Elem2No;

         Trans = mesh->GetFaceElementTransformations(bdrs2[i]);
         NbrEl2[i] = Trans->Elem1No != el2 ? Trans->Elem1No : Trans->Elem2No;
      }

      for (i = 0; i < numBdrs; i++)
      {
         for (j = 0; j < numBdrs; j++)
         {
            // add neighbor elements that share a face
            // with el1 and el2 but are not el
            if ((NbrEl1[i] == NbrEl2[j]) && (NbrEl1[i] != el))
            {
               if (!found)
               {
                  commonNeighbor = NbrEl1[i];
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

   // This fills the map_for_bounds according to our paper.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void GetVertexBoundsMap()
   {
      const FiniteElement &dummy = *fes->GetFE(0);
      int i, j, k, dofInd, nbr;
      int ne = mesh->GetNE(), nd = dummy.GetDof(), p = dummy.GetOrder();
      Array<int> bdrs, orientation, NbrElem;
      FaceElementTransformations *Trans;

      NbrElem.SetSize(numBdrs);

      for (k = 0; k < ne; k++)
      {
         // include the current element for all dofs of the element
         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i;
            map_for_bounds[dofInd].push_back(k);
         }

         if (dim==1)
         {
            mesh->GetElementVertices(k, bdrs);
         }
         else if (dim==2)
         {
            mesh->GetElementEdges(k, bdrs, orientation);
         }
         else if (dim==3)
         {
            mesh->GetElementFaces(k, bdrs, orientation);
         }

         // Include neighbors sharing a face with element k for face dofs.
         for (i = 0; i < numBdrs; i++)
         {
            Trans = mesh->GetFaceElementTransformations(bdrs[i]);

            NbrElem[i] = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
            
            if (NbrElem[i] < 0)
               continue;
            
            for (j = 0; j < numDofs; j++)
            {
               dofInd = k*nd+BdrDofs(j,i);
               map_for_bounds[dofInd].push_back(NbrElem[i]);
            }
         }

         // Include neighbors that have no face in common with element k.
         if (dim==2) // Include neighbor elements for the four vertices.
         {

            nbr = GetCommonElem(k, NbrElem[3], NbrElem[0]);
            if (nbr >= 0) { map_for_bounds[k*nd].push_back(nbr); }

            nbr = GetCommonElem(k, NbrElem[0], NbrElem[1]);
            if (nbr >= 0) { map_for_bounds[k*nd+p].push_back(nbr); }

            nbr = GetCommonElem(k, NbrElem[1], NbrElem[2]);
            if (nbr >= 0) { map_for_bounds[(k+1)*nd-1].push_back(nbr); }

            nbr = GetCommonElem(k, NbrElem[2], NbrElem[3]);
            if (nbr >= 0) { map_for_bounds[k*nd+p*(p+1)].push_back(nbr); }
         }
         else if (dim==3)
         {
            Array<int> EdgeNbrs; EdgeNbrs.SetSize(12);

            EdgeNbrs[0]  = GetCommonElem(k, NbrElem[0], NbrElem[1]);
            EdgeNbrs[1]  = GetCommonElem(k, NbrElem[0], NbrElem[2]);
            EdgeNbrs[2]  = GetCommonElem(k, NbrElem[0], NbrElem[3]);
            EdgeNbrs[3]  = GetCommonElem(k, NbrElem[0], NbrElem[4]);
            EdgeNbrs[4]  = GetCommonElem(k, NbrElem[5], NbrElem[1]);
            EdgeNbrs[5]  = GetCommonElem(k, NbrElem[5], NbrElem[2]);
            EdgeNbrs[6]  = GetCommonElem(k, NbrElem[5], NbrElem[3]);
            EdgeNbrs[7]  = GetCommonElem(k, NbrElem[5], NbrElem[4]);
            EdgeNbrs[8]  = GetCommonElem(k, NbrElem[4], NbrElem[1]);
            EdgeNbrs[9]  = GetCommonElem(k, NbrElem[1], NbrElem[2]);
            EdgeNbrs[10] = GetCommonElem(k, NbrElem[2], NbrElem[3]);
            EdgeNbrs[11] = GetCommonElem(k, NbrElem[3], NbrElem[4]);

            // include neighbor elements for the twelve edges of a square
            for (j = 0; j <= p; j++)
            {
               if (EdgeNbrs[0] >= 0)
               {
                  map_for_bounds[k*nd+j].push_back(EdgeNbrs[0]);
               }
               if (EdgeNbrs[1] >= 0)
               {
                  map_for_bounds[k*nd+(j+1)*(p+1)-1].push_back(EdgeNbrs[1]);
               }
               if (EdgeNbrs[2] >= 0)
               {
                  map_for_bounds[k*nd+p*(p+1)+j].push_back(EdgeNbrs[2]);
               }
               if (EdgeNbrs[3] >= 0)
               {
                  map_for_bounds[k*nd+j*(p+1)].push_back(EdgeNbrs[3]);
               }
               if (EdgeNbrs[4] >= 0)
               {
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+j].push_back(EdgeNbrs[4]);
               }
               if (EdgeNbrs[5] >= 0)
               {
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+(j+1)*(p+1)-1].push_back(EdgeNbrs[5]);
               }
               if (EdgeNbrs[6] >= 0)
               {
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+p*(p+1)+j].push_back(EdgeNbrs[6]);
               }
               if (EdgeNbrs[7] >= 0)
               {
                  map_for_bounds[k*nd+(p+1)*(p+1)*p+j*(p+1)].push_back(EdgeNbrs[7]);
               }
               if (EdgeNbrs[8] >= 0)
               {
                  map_for_bounds[k*nd+j*(p+1)*(p+1)].push_back(EdgeNbrs[8]);
               }
               if (EdgeNbrs[9] >= 0)
               {
                  map_for_bounds[k*nd+p+j*(p+1)*(p+1)].push_back(EdgeNbrs[9]);
               }
               if (EdgeNbrs[10] >= 0)
               {
                  map_for_bounds[k*nd+(j+1)*(p+1)*(p+1)-1].push_back(EdgeNbrs[10]);
               }
               if (EdgeNbrs[11] >= 0)
               {
                  map_for_bounds[k*nd+p*(p+1)+j*(p+1)*(p+1)].push_back(EdgeNbrs[11]);
               }
            }

            // include neighbor elements for the 8 vertices of a square
            nbr = GetCommonElem(NbrElem[0], EdgeNbrs[0], EdgeNbrs[3]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[0], EdgeNbrs[0], EdgeNbrs[1]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+p].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[0], EdgeNbrs[2], EdgeNbrs[3]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+p*(p+1)].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[0], EdgeNbrs[1], EdgeNbrs[2]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+(p+1)*(p+1)-1].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[5], EdgeNbrs[4], EdgeNbrs[7]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+(p+1)*(p+1)*p].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[5], EdgeNbrs[4], EdgeNbrs[5]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+(p+1)*(p+1)*p+p].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[5], EdgeNbrs[6], EdgeNbrs[7]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+(p+1)*(p+1)*p+(p+1)*p].push_back(nbr);
            }

            nbr = GetCommonElem(NbrElem[5], EdgeNbrs[5], EdgeNbrs[6]);
            if (nbr >= 0)
            {
               map_for_bounds[k*nd+(p+1)*(p+1)*(p+1)-1].push_back(nbr);
            }
         }
      }
   }
   
   // For each DOF on an element boundary, the global index of the DOF on the
   // opposite site is computed and stored in a list. This is needed for 
   // lumping the flux contributions as in the paper. Right now it works on 
   // 1D meshes, quad meshes in 2D and 3D meshes of ordered cubes.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs()
   {
      // Use the first mesh element as indicator.
      const FiniteElement &dummy = *fes->GetFE(0);
      int i, j, k, ind, nbr, ne = mesh->GetNE();
      int nd = dummy.GetDof(), p = dummy.GetOrder();
      Array <int> bdrs, NbrBdrs, orientation;
      FaceElementTransformations *Trans;
      
      NbrDof.SetSize(ne*numDofs, numBdrs);
      
      for (k = 0; k < ne; k++)
      {
         if (dim==1)
         {
            mesh->GetElementVertices(k, bdrs);
            
            for (i = 0; i < numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
               NbrDof(k,i) = nbr*nd + BdrDofs(0,(i+1)%2);
            }
         }
         else if (dim==2)
         {
            mesh->GetElementEdges(k, bdrs, orientation);
            
            for (i = 0; i < numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;
               
               for (j = 0; j < numDofs; j++)
               {
                  if (nbr >= 0)
                  {
                     mesh->GetElementEdges(nbr, NbrBdrs, orientation);
                     // Find the local index ind in nbr of the common face.
                     for (ind = 0; ind < numBdrs; ind++)
                     {
                        if (NbrBdrs[ind] == bdrs[i])
                        {
                           break;
                        }
                     }
                     // Here it is utilized that the orientations of the face
                     // for the two elements are opposite of each other.
                     NbrDof(k*numDofs+j,i) = nbr*nd + BdrDofs(numDofs-1-j,ind);
                  }
                  else
                  {
                     NbrDof(k*numDofs+j,i) = -1;
                  }
               }
            }
         }
         else if (dim==3)
         {
            mesh->GetElementFaces(k, bdrs, orientation);
            
            // TODO: This works only for meshes of uniformly ordered cube nodes.
            for (j = 0; j < numDofs; j++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[0]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 0) = nbr*nd + (p+1)*(p+1)*p+j;

               Trans = mesh->GetFaceElementTransformations(bdrs[1]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 1) = nbr*nd + (j/(p+1))*(p+1)*(p+1)
                                        + (p+1)*p+(j%(p+1));

               Trans = mesh->GetFaceElementTransformations(bdrs[2]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 2) = nbr*nd + j*(p+1);

               Trans = mesh->GetFaceElementTransformations(bdrs[3]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 3) = nbr*nd + (j/(p+1))*(p+1)*(p+1)
                                        + (j%(p+1));

               Trans = mesh->GetFaceElementTransformations(bdrs[4]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 4) = nbr*nd + (j+1)*(p+1)-1;

               Trans = mesh->GetFaceElementTransformations(bdrs[5]);
               nbr = Trans->Elem1No == k ? Trans->Elem2No : Trans->Elem1No;

               NbrDof(k*numDofs+j, 5) = nbr*nd + j;
            }
         }
      }
   }
   
   // A list is filled to later access the correct element-global
   // indices given the subcell number and subcell index.
   // NOTE: The mesh is assumed to consist of segments, quads or hexes.
   void FillSubcell2CellDof()
   {
      const FiniteElement &dummy = *fes->GetFE(0);
      int j, m, aux, p = dummy.GetOrder();
      
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
      
      Sub2Ind.SetSize(numSubcells, numDofsSubcell);
      
      for (m = 0; m < numSubcells; m++)
      {
         for (j = 0; j < numDofsSubcell; j++)
         {
            if (dim == 1)
            {
               Sub2Ind(m,j) = m + j;
            }
            else if (dim == 2)
            {
               aux = m + (m/p);
               switch (j)
               {
                  case 0: Sub2Ind(m,j) =  aux; break;
                  case 1: Sub2Ind(m,j) =  aux + 1; break;
                  case 2: Sub2Ind(m,j) =  aux + p+1; break;
                  case 3: Sub2Ind(m,j) =  aux + p+2; break;
               }
            }
            else if (dim == 3)
            {
               aux = m + (m/p)+(p+1)*(m/(p*p));
               switch (j)
               {
                  case 0:
                     Sub2Ind(m,j) = aux; break;
                  case 1:
                     Sub2Ind(m,j) = aux + 1; break;
                  case 2:
                     Sub2Ind(m,j) = aux + p+1; break;
                  case 3:
                     Sub2Ind(m,j) = aux + p+2; break;
                  case 4:
                     Sub2Ind(m,j) = aux + (p+1)*(p+1); break;
                  case 5:
                     Sub2Ind(m,j) = aux + (p+1)*(p+1)+1; break;
                  case 6:
                     Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+1; break;
                  case 7:
                     Sub2Ind(m,j) = aux + (p+1)*(p+1)+p+2; break;
               }
            }
         }
      }
   }
};

class DiscreteUpwinding
{
public:
	FiniteElementSpace* fes;
	Array<int> smap;
	SparseMatrix D;
	
	DiscreteUpwinding(FiniteElementSpace* _fes, const SparseMatrix K) : fes(_fes)
	{
		smap = SparseMatrix_Build_smap(K);
		
		if (exec_mode == 0) // Initialization for transport mode.
      {
			ComputeDiscreteUpwindingMatrix(K, smap, D);
		}
	}
};

class Assembly
{
public:
   Assembly(FiniteElementSpace* _fes, MONOTYPE _monoType, bool _OptScheme,
            VectorFunctionCoefficient &coef, DofInfo &_dofs,
            FiniteElementSpace _SubFes0, FiniteElementSpace _SubFes1,
            Mesh* _ref_mesh, BilinearFormIntegrator* _VolumeTerms,
            const IntegrationRule* _irF) :
            fes(_fes), monoType(_monoType), OptScheme(_OptScheme),
            velocity(coef), dofs(_dofs), SubFes0(_SubFes0),
            SubFes1(_SubFes1), ref_mesh(_ref_mesh), VolumeTerms(_VolumeTerms),
            irF(_irF)
   {
      Mesh *mesh = fes->GetMesh();
      int k, i, m, nd, dim = mesh->Dimension(), ne = fes->GetNE();
      
      // Use the first mesh element as indicator.
      const FiniteElement &dummy = *fes->GetFE(0);
      nd = dummy.GetDof();
      
      Array <int> bdrs, orientation;
      FaceElementTransformations *Trans;

      // TODO There are many zero entries in bdrInt, maybe use better indexing.
      bdrInt.SetSize(ne*nd, nd*dofs.numBdrs); bdrInt = 0.;
      SubcellWeights.SetSize(ne*dofs.numSubcells, dofs.numDofsSubcell);
      
      if (exec_mode == 0) // Initialization for transport mode.
      {
         for (k = 0; k < ne; k++)
         {
            if (dim==1)
            {
               mesh->GetElementVertices(k, bdrs);
            }
            else if (dim==2)
            {
               mesh->GetElementEdges(k, bdrs, orientation);
            }
            else if (dim==3)
            {
               mesh->GetElementFaces(k, bdrs, orientation);
            }
            
            for (i = 0; i < dofs.numBdrs; i++)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               ComputeFluxTerms(k, i, velocity, Trans);
            }
            
            for (m = 0; m < dofs.numSubcells; m++)
            {
               ComputeSubcellWeights(k, m);
            }
         }
      }
   }

   // Destructor
   ~Assembly() { }

   // Auxiliary member variables that need to be accessed during time-stepping.
   FiniteElementSpace* fes;
   MONOTYPE monoType;
   bool OptScheme;
   VectorFunctionCoefficient &velocity;
   DofInfo &dofs;
   FiniteElementSpace SubFes0;
   FiniteElementSpace SubFes1;
   Mesh *ref_mesh;
   BilinearFormIntegrator *VolumeTerms;
   const IntegrationRule* irF;
   
   // Data structures storing Galerkin contributions. These are updated for 
   // remap but remain constant for transport.
   DenseMatrix bdrInt;
   DenseMatrix SubcellWeights;
   
   void ComputeFluxTerms(const int k, const int BdrID, VectorCoefficient &coef,
                         FaceElementTransformations *Trans)
   {
      Mesh *mesh = fes->GetMesh();
      
      int i, j, l, nd, row, dim = mesh->Dimension();
      double aux, vn = 0.;
      
      const FiniteElement &el = *fes->GetFE(k);
      nd = el.GetDof();
      
      Vector vval, nor(dim), shape(nd);
      
      for (l = 0; l < irF->GetNPoints(); l++)
      {
         const IntegrationPoint &ip = irF->IntPoint(l);
         IntegrationPoint eip1;
         Trans->Face->SetIntPoint(&ip);
         
         if (dim == 1)
         {
            Trans->Loc1.Transform(ip, eip1);
            nor(0) = 2.*eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Trans->Face->Jacobian(), nor);
         }
         
         if (Trans->Elem1No != k)
         {
            Trans->Loc2.Transform(ip, eip1);
            el.CalcShape(eip1, shape);
            Trans->Elem2->SetIntPoint(&eip1);
            coef.Eval(vval, *Trans->Elem2, eip1);
            nor *= -1.;
            Trans->Loc1.Transform(ip, eip1);
         }
         else
         {
            Trans->Loc1.Transform(ip, eip1);
            el.CalcShape(eip1, shape);
            Trans->Elem1->SetIntPoint(&eip1);
            coef.Eval(vval, *Trans->Elem1, eip1);
            Trans->Loc2.Transform(ip, eip1);
         }
         
         nor /= nor.Norml2();
         
         if (exec_mode == 0)
         {
            // Transport.
            vn = min(0., vval * nor);
         }
         else
         {
            // Remap.
            vn = max(0., vval * nor);
            vn *= -1.0;
         }
         
         for (i = 0; i < dofs.numDofs; i++)
         {
            row = k*nd+dofs.BdrDofs(i,BdrID);
            aux = ip.weight * Trans->Face->Weight()
                  * shape(dofs.BdrDofs(i,BdrID)) * vn;

            for (j = 0; j < dofs.numDofs; j++)
            {
               bdrInt(row,BdrID*nd+dofs.BdrDofs(j,BdrID)) -= aux
                                    * shape(dofs.BdrDofs(j,BdrID));
            }
         }
      }
   }
   
   void ComputeSubcellWeights(const int k, const int m)
   {
      Vector row;
      DenseMatrix elmat; // These are essentially the same.
      int dofInd = k*dofs.numSubcells+m;
      const FiniteElement *el0 = SubFes0.GetFE(dofInd);
      const FiniteElement *el1 = SubFes1.GetFE(dofInd);
      ElementTransformation *tr = ref_mesh->GetElementTransformation(dofInd);
      VolumeTerms->AssembleElementMatrix2(*el1, *el0, *tr, elmat);
      
      elmat.GetRow(0, row); // Using the fact that elmat has just one row.
      SubcellWeights.SetRow(dofInd, row);
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
   BilinearForm &Mbf, &Kbf, &ml;
   SparseMatrix &M, &K;
   const Vector &b;

   Vector start_pos;
   Vector &lumpedM;
   GridFunction &mesh_pos, &vel_pos;

   VectorGridFunctionCoefficient &coef;

   mutable Vector z;

   double dt, start_t;
   Assembly &asmbl;
	
	LowOrderMethod &lom;

public:
   FE_Evolution(FiniteElementSpace* fes, BilinearForm &Mbf_, BilinearForm &Kbf_,
                SparseMatrix &_M, SparseMatrix &_K, const Vector &_b,
                Assembly &_asmbl, GridFunction &mpos, GridFunction &vpos,
                BilinearForm &_ml, Vector &_lumpedM,
                VectorGridFunctionCoefficient &v_coef, LowOrderMethod &_lom);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void SetDt(double _dt) { dt = _dt; }
   void SetInitialTimeStepTime(double st) { start_t = st; }
   void SetRemapStartPos(const Vector &spos) { start_pos = spos; }
   void GetRemapStartPos(Vector &spos) { spos = start_pos; }

   // Mass matrix solve, addressing the bad Bernstein condition number.
   virtual void NeumannSolve(const Vector &b, Vector &x) const;

   virtual void LinearFluxLumping(const int k, const int nd,
                                  const int BdrID, const Vector &x,
                                  Vector &y, const Vector &alpha) const;

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
   bool OptScheme = true;
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
                  "Monotonicity scheme: 0 - no monotonicity treatment,\n\t"
                  "                     1 - discrete upwinding - LO,\n\t"
                  "                     2 - discrete upwinding - FCT,\n\t"
                  "                     3 - residual distribution - LO,\n\t"
                  "                     4 - residual distribution - FCT.");
   args.AddOption(&OptScheme, "-sc", "--subcell", "-el", "--element-based",
                  "Optimized scheme: PDU or subcell version.");
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

   if (problem < 10)      { exec_mode = 0; }
   else if (problem < 20) { exec_mode = 1; }
   else if (problem < 30) { exec_mode = 2; }
   else { MFEM_ABORT("Unspecified execution mode."); }

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

   // Check for meaningful combinations of parameters.
   bool fail = false;
   if (monoType != None)
   {
      if (((int)monoType != monoType) || (monoType < 0) || (monoType > 4))
      {
         cout << "Unsupported option for monotonicity treatment." << endl;
         fail = true;
      }
      if (btype != 2)
      {
         cout << "Monotonicity treatment requires Bernstein basis." << endl;
         fail = true;
      }
      if (order == 0)
      {
         // Disable monotonicity treatment for piecwise constants.
         mfem_warning("For -o 0, monotonicity treatment is disabled.");
         monoType = None;
      }
   }
   if ((monoType > 2) && (order==1) && OptScheme)
   {
      // Avoid subcell methods for linear elements.
      mfem_warning("For -o 1, subcell scheme is disabled.");
      OptScheme = false;
   }
   if (fail)
   {
      delete mesh;
      delete ode_solver;
      return 5;
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
   VectorGridFunctionCoefficient v_coef(&v_gf);

   BilinearForm m(&fes);
   m.AddDomainIntegrator(new MassIntegrator);

   BilinearForm k(&fes);

   if (exec_mode == 0)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   }
   else if (exec_mode == 1)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(v_coef));
   }
   else if (exec_mode == 2)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(velocity));
   }
   
   // In case of basic discrete upwinding, add boundary terms.
   if (((monoType == DiscUpw) || (monoType == DiscUpw_FCT)) && (!OptScheme))
   {
      if (exec_mode == 0)
      {
         k.AddInteriorFaceIntegrator( new TransposeIntegrator(
            new DGTraceIntegrator(velocity, 1.0, -0.5)) );
         k.AddBdrFaceIntegrator( new TransposeIntegrator(
            new DGTraceIntegrator(velocity, 1.0, -0.5)) );
      }
      else if (exec_mode == 1)
      {
         k.AddInteriorFaceIntegrator(new TransposeIntegrator(
            new DGTraceIntegrator(v_coef, -1.0, -0.5)) );
         k.AddBdrFaceIntegrator( new TransposeIntegrator(
            new DGTraceIntegrator(v_coef, -1.0, -0.5)) );
      }
      else if (exec_mode == 2)
      {
         k.AddInteriorFaceIntegrator( new TransposeIntegrator(
            new DGTraceIntegrator(velocity, -1.0, -0.5)) );
         k.AddBdrFaceIntegrator( new TransposeIntegrator(
            new DGTraceIntegrator(velocity, -1.0, -0.5)) );
      }
   }

   // Compute the lumped mass matrix algebraicly
   Vector lumpedM;
   BilinearForm ml(&fes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, v_coef, -1.0, -0.5));

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);
   b.Assemble();

   // Store topological dof data.
   DofInfo dofs(&fes);

   // Precompute data required for high and low order schemes.
   BilinearFormIntegrator* VolumeTerms;
   if (exec_mode == 0)
   {
      VolumeTerms = new MixedConvectionIntegrator(velocity, -1.0);
   }
   else if (exec_mode == 1)
   {
      VolumeTerms = new MixedConvectionIntegrator(v_coef);
   }
   else if (exec_mode == 2)
   {
      VolumeTerms = new MixedConvectionIntegrator(velocity);
   }
   
   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);
   
   Mesh* ref_mesh = GetSubcellMesh(mesh, order); // TODO if put somewhere else (conditional to OptScheme) remove p==1 in this routine
   FiniteElementSpace SubFes0(ref_mesh, &fec0);
   FiniteElementSpace SubFes1(ref_mesh, &fec1);
   
   const IntegrationRule* irF = GetFaceIntRule(&fes);
   
   Assembly asmbl(&fes, monoType, OptScheme, velocity, dofs,
                  SubFes0, SubFes1, ref_mesh, VolumeTerms, irF); // TODO change type of velocity to allow v_coef

	LowOrderMethod lom;
	lom.fes = &fes;
	
	if ((monoType == DiscUpw) || (DiscUpw_FCT))
	{
		if (!OptScheme)
		{
			lom.smap = SparseMatrix_Build_smap(k.SpMat());
			lom.D = k.SpMat();
			
			if (exec_mode == 0)
			{
				ComputeDiscreteUpwindingMatrix(k.SpMat(), lom.smap, lom.D);
			}
		}
		else
		{
			lom.pk = new BilinearForm(&fes);
			if (exec_mode == 0)
			{
				lom.pk->AddDomainIntegrator(new PrecondConvectionIntegrator(velocity, -1.0));
			}
			else if (exec_mode == 1)
			{
				lom.pk->AddDomainIntegrator(new PrecondConvectionIntegrator(v_coef));
			}
			else if (exec_mode == 2)
			{
				lom.pk->AddDomainIntegrator(new PrecondConvectionIntegrator(velocity));
			}
			lom.pk->Assemble(skip_zeros);
			lom.pk->Finalize(skip_zeros);
			
			lom.smap = SparseMatrix_Build_smap(lom.pk->SpMat());
			lom.D = lom.pk->SpMat();
			
			if (exec_mode == 0)
			{
				ComputeDiscreteUpwindingMatrix(lom.pk->SpMat(), lom.smap, lom.D);
			}
		}
	}
	
// 	Init(monoType, OptScheme, lom);
	
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
   Vector mass(lumpedM);
   double initialMass = lumpedM * u;

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(&fes, m, k, m.SpMat(), k.SpMat(), b, asmbl, *x, v_gf, ml,
                    lumpedM, v_coef, lom);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      adv.SetDt(dt_real);

      if (exec_mode == 1)
      {
         adv.SetRemapStartPos(x0);
      }
      else if (exec_mode == 2)
      {
         // Move the mesh (and the solution) from x0 (one step).
         add(x0, dt_real, v_gf, *x);
         adv.SetRemapStartPos(*x);
      }

      adv.SetInitialTimeStepTime(t);

      ode_solver->Step(u, t, dt_real);
      ti++;

      if (exec_mode == 1)
      {
         add(x0, t, v_gf, *x);
      }
      else if (exec_mode == 2)
      {
         *x = x0;
      }

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
   double finalMass;
   if (exec_mode == 1)
   {
      // TODO figure out why this setup doesn't conserve mass for mode 1.
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);
      finalMass = lumpedM * u;
   }
   else { finalMass = mass * u; }
   cout << "Mass loss: " << abs(initialMass - finalMass) << endl;

   // Compute errors, if the initial condition is equal to the final solution
   if (problem == 4) // solid body rotation
   {
      cout << "L1-error: " << u.ComputeLpError(1., u0) << ", L-Inf-error: "
           << u.ComputeLpError(numeric_limits<double>::infinity(), u0)
           << "." << endl;
   }

   // 10. Free the used memory.
   delete mesh;
   delete ode_solver;
   delete VolumeTerms;
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

void FE_Evolution::LinearFluxLumping(const int k, const int nd,
                                     const int BdrID, const Vector &x,
                                     Vector &y, const Vector &alpha) const
{
   int i, j, idx, dofInd;
   double xNeighbor;
   Vector xDiff(asmbl.dofs.numDofs);

   for (i = 0; i < asmbl.dofs.numDofs; i++)
   {
      dofInd = k*nd+asmbl.dofs.BdrDofs(i,BdrID);
      idx = asmbl.dofs.NbrDof(k*asmbl.dofs.numDofs+i,BdrID);
      // If NbrDof is -1 and bdrInt > 0., this is an inflow boundary. If NbrDof
      // is -1 and bdrInt = 0., this is an outflow, which is handled correctly.
      // TODO use inflow instead of xNeighbor = 0.
      xNeighbor = idx < 0 ? 0. : x(idx);
      xDiff(i) = xNeighbor - x(dofInd);
   }

   for (i = 0; i < asmbl.dofs.numDofs; i++)
   {
      dofInd = k*nd+asmbl.dofs.BdrDofs(i,BdrID);
      for (j = 0; j < asmbl.dofs.numDofs; j++)
      {
         y(dofInd) += asmbl.bdrInt(dofInd, BdrID*nd+asmbl.dofs.BdrDofs(j,BdrID))
          * ( xDiff(i) + (xDiff(j)-xDiff(i))*alpha(asmbl.dofs.BdrDofs(i,BdrID))
                                         * alpha(asmbl.dofs.BdrDofs(j,BdrID)) );
      }
   }
}

void FE_Evolution::ComputeLowOrderSolution(const Vector &x, Vector &y) const
{
   const FiniteElement &dummy = *fes->GetFE(0);
   int i, j, k, dofInd, nd = dummy.GetDof(), ne = fes->GetNE();
   Vector alpha(nd); alpha = 0.;
   
   if ( (asmbl.monoType == DiscUpw) || (asmbl.monoType == DiscUpw_FCT) )
   {
		// Reassemble on the new mesh (given by mesh_pos).
		if (exec_mode > 0)
		{
			if (!asmbl.OptScheme)
			{
				ComputeDiscreteUpwindingMatrix(K, lom.smap, lom.D);
			}
			else
			{
				lom.pk->BilinearForm::operator=(0.0);
				lom.pk->Assemble();
				ComputeDiscreteUpwindingMatrix(lom.pk->SpMat(), lom.smap, lom.D);
			}
		}

      // Discretization and monotonicity terms.
      lom.D.Mult(x, y);
      y += b;

      // Lump fluxes (for PDU), compute min/max, and invert lumped mass matrix.
      for (k = 0; k < ne; k++)
      {
         ////////////////////////////
         // Boundary contributions //
         //////////////////////////// 
         if (asmbl.OptScheme)
         {
            for (i = 0; i < asmbl.dofs.numBdrs; i++)
            {
               LinearFluxLumping(k, nd, i, x, y, alpha);
            }
         }
         
         asmbl.dofs.xe_min(k) = numeric_limits<double>::infinity();
         asmbl.dofs.xe_max(k) = -asmbl.dofs.xe_min(k);
         
         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            asmbl.dofs.xe_max(k) = max(asmbl.dofs.xe_max(k), x(dofInd));
            asmbl.dofs.xe_min(k) = min(asmbl.dofs.xe_min(k), x(dofInd));
            y(dofInd) /= lumpedM(dofInd);
         }
      }
   }
   else // RD(S)
   {
      Mesh *mesh = fes->GetMesh();
      int i, m, dofInd2, loc, ne(fes->GetNE()), dim(mesh->Dimension());
      double xSum, xNeighbor, sumFluctSubcellP, sumFluctSubcellN, sumWeightsP,
             sumWeightsN, weightP, weightN, rhoP, rhoN, gammaP, gammaN,
             minGammaP, minGammaN, aux, fluct, gamma = 10., eps = 1.E-15;
      Vector xMaxSubcell, xMinSubcell, sumWeightsSubcellP, sumWeightsSubcellN,
             fluctSubcellP, fluctSubcellN, nodalWeightsP, nodalWeightsN;
            
      // Discretization terms
      y = b;
      K.Mult(x, z);
		
		if ((exec_mode > 0) && (asmbl.OptScheme))
      {
			asmbl.ref_mesh = GetSubcellMesh(mesh, dummy.GetOrder());
		}

		asmbl.SubcellWeights.Print();
		
      // Monotonicity terms
      for (k = 0; k < ne; k++)
      {
         ////////////////////////////
         // Boundary contributions //
         ////////////////////////////           
         for (i = 0; i < asmbl.dofs.numBdrs; i++)
         {
            LinearFluxLumping(k, nd, i, x, y, alpha);
         }
         
         ///////////////////////////
         // Element contributions //
         ///////////////////////////
         asmbl.dofs.xe_min(k) = numeric_limits<double>::infinity();
         asmbl.dofs.xe_max(k) = -asmbl.dofs.xe_min(k);
         rhoP = rhoN = xSum = 0.;

         for (j = 0; j < nd; j++)
         {
            dofInd = k*nd+j;
            asmbl.dofs.xe_max(k) = max(asmbl.dofs.xe_max(k), x(dofInd));
            asmbl.dofs.xe_min(k) = min(asmbl.dofs.xe_min(k), x(dofInd));
            xSum += x(dofInd);
				if (asmbl.OptScheme)
            {
               rhoP += max(0., z(dofInd));
               rhoN += min(0., z(dofInd));
            }
         }

         sumWeightsP = nd*asmbl.dofs.xe_max(k) - xSum + eps;
         sumWeightsN = nd*asmbl.dofs.xe_min(k) - xSum - eps;

         if (asmbl.OptScheme)
         {
            fluctSubcellP.SetSize(asmbl.dofs.numSubcells);
            fluctSubcellN.SetSize(asmbl.dofs.numSubcells);
            xMaxSubcell.SetSize(asmbl.dofs.numSubcells);
            xMinSubcell.SetSize(asmbl.dofs.numSubcells);
            sumWeightsSubcellP.SetSize(asmbl.dofs.numSubcells);
            sumWeightsSubcellN.SetSize(asmbl.dofs.numSubcells);
            nodalWeightsP.SetSize(nd);
            nodalWeightsN.SetSize(nd);
            sumFluctSubcellP = sumFluctSubcellN = 0.;
            nodalWeightsP = 0.; nodalWeightsN = 0.;

            // compute min-/max-values and the fluctuation for subcells
            for (m = 0; m < asmbl.dofs.numSubcells; m++)
            {
               xMinSubcell(m) = numeric_limits<double>::infinity();
               xMaxSubcell(m) = -xMinSubcell(m);
               fluct = xSum = 0.;
               
               if (exec_mode > 0)
               {
                  asmbl.ComputeSubcellWeights(k, m);
               }
               
               for (i = 0; i < asmbl.dofs.numDofsSubcell; i++)
               {
                  dofInd = k*nd + asmbl.dofs.Sub2Ind(m, i);
                  fluct += asmbl.SubcellWeights(k*asmbl.dofs.numSubcells+m,i)
                           * x(dofInd);
                  xMaxSubcell(m) = max(xMaxSubcell(m), x(dofInd));
                  xMinSubcell(m) = min(xMinSubcell(m), x(dofInd));
                  xSum += x(dofInd);
               }
               sumWeightsSubcellP(m) = asmbl.dofs.numDofsSubcell
                                       * xMaxSubcell(m) - xSum + eps;
               sumWeightsSubcellN(m) = asmbl.dofs.numDofsSubcell
                                       * xMinSubcell(m) - xSum - eps;

               fluctSubcellP(m) = max(0., fluct);
               fluctSubcellN(m) = min(0., fluct);
               sumFluctSubcellP += fluctSubcellP(m);
               sumFluctSubcellN += fluctSubcellN(m);
            }

            for (m = 0; m < asmbl.dofs.numSubcells; m++)
            {
               for (i = 0; i < asmbl.dofs.numDofsSubcell; i++)
               {
                  loc = asmbl.dofs.Sub2Ind(m, i);
                  dofInd = k*nd + loc;
                  nodalWeightsP(loc) += fluctSubcellP(m)
                                        * ((xMaxSubcell(m) - x(dofInd))
                                        / sumWeightsSubcellP(m)); // eq. (10)
                  nodalWeightsN(loc) += fluctSubcellN(m)
                                        * ((xMinSubcell(m) - x(dofInd))
                                        / sumWeightsSubcellN(m)); // eq. (11)
               }
            }
         }

         for (i = 0; i < nd; i++)
         {
            dofInd = k*nd+i;
            weightP = (asmbl.dofs.xe_max(k) - x(dofInd)) / sumWeightsP;
            weightN = (asmbl.dofs.xe_min(k) - x(dofInd)) / sumWeightsN;

            if (asmbl.OptScheme)
            {
               aux = gamma / (rhoP + eps);
               weightP *= 1. - min(aux * sumFluctSubcellP, 1.);
               weightP += min(aux, 1./(sumFluctSubcellP+eps))*nodalWeightsP(i);

               aux = gamma / (rhoN - eps);
               weightN *= 1. - min(aux * sumFluctSubcellN, 1.);
               weightN += max(aux, 1./(sumFluctSubcellN-eps))*nodalWeightsN(i);
            }

            for (j = 0; j < nd; j++)
            {
               dofInd2 = k*nd+j;
               if (z(dofInd2) > eps)
               {
                  y(dofInd) += weightP * z(dofInd2);
               }
               else if (z(dofInd2) < -eps)
               {
                  y(dofInd) += weightN * z(dofInd2);
               }
            }
            y(dofInd) /= lumpedM(dofInd);
         }
      }
   }
}

// No monotonicity treatment, straightforward high-order scheme
// ydot = M^{-1} (K x + b).
void FE_Evolution::ComputeHighOrderSolution(const Vector &x, Vector &y) const
{
   const FiniteElement &dummy = *fes->GetFE(0);
   int i, k, nd = dummy.GetDof(), ne = fes->GetNE();
   Vector alpha(nd); alpha = 1.;
   
   K.Mult(x, z);
   z += b;

   // Incorporate flux terms only if the low order scheme is PDU, RD, or RDS.
   if ((asmbl.monoType != DiscUpw_FCT) || (asmbl.OptScheme))
   {
      // The boundary contributions have been computed in the low order scheme.
      for (k = 0; k < ne; k++)
      {
         for (i = 0; i < asmbl.dofs.numBdrs; i++)
         {
            LinearFluxLumping(k, nd, i, x, z, alpha);
         }
      }
   }
   
   NeumannSolve(z, y);
}

// High order reconstruction that yields an updated admissible solution by means
// of clipping the solution coefficients within certain bounds and scaling the
// antidiffusive fluxes in a way that leads to local conservation of mass. yH,
// yL are the high and low order discrete time derivatives.
void FE_Evolution::ComputeFCTSolution(const Vector &x, const Vector &yH,
                                      const Vector &yL, Vector &y) const
{
	
   int j, k, nd, dofInd;
   double sumPos, sumNeg, eps = 1.E-15;
   Vector uClipped, fClipped;

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
			
			// Compute the bounds for each dof inside the loop.
			asmbl.dofs.ComputeVertexBounds(x, dofInd);
			
         uClipped(j) = min( asmbl.dofs.xi_max(dofInd),
                            max( x(dofInd) + dt * yH(dofInd),
                                 asmbl.dofs.xi_min(dofInd) ) );

         fClipped(j) = lumpedM(dofInd) / dt
								* ( uClipped(j) - (x(dofInd) + dt * yL(dofInd)) );

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

         // Set y to the discrete time derivative featuring the high order anti-
         // diffusive reconstruction that leads to an forward Euler updated 
         // admissible solution. 
         dofInd = k*nd+j;
         y(dofInd) = yL(dofInd) + fClipped(j) / lumpedM(dofInd);
      }
   }
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace* _fes, BilinearForm &Mbf_,
                           BilinearForm &Kbf_, SparseMatrix &_M,
                           SparseMatrix &_K, const Vector &_b, Assembly &_asmbl,
                           GridFunction &mpos, GridFunction &vpos,
                           BilinearForm &_ml, Vector &_lumpedM, 
                           VectorGridFunctionCoefficient &v_coef, LowOrderMethod &_lom) :
   TimeDependentOperator(_M.Size()), fes(_fes),Mbf(Mbf_), Kbf(Kbf_), M(_M),
   K(_K), b(_b), z(_M.Size()), asmbl(_asmbl), start_pos(mpos.Size()),
   mesh_pos(mpos), vel_pos(vpos), ml(_ml), lumpedM(_lumpedM), coef(v_coef), lom(_lom) { }

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   Mesh *mesh = fes->GetMesh();
   int i, k, dim = mesh->Dimension(), ne = fes->GetNE();
   Array <int> bdrs, orientation;
   FaceElementTransformations *Trans;
   
   // Move towards x0 with current t.
   const double t = GetTime();
   const double sub_time_step = t - start_t;

   if (exec_mode == 1)
   {
      add(start_pos, t, vel_pos, mesh_pos);
   }
   else if (exec_mode == 2)
   {
      add(start_pos, -sub_time_step, vel_pos, mesh_pos);
   }

   // Reassemble on the new mesh (given by mesh_pos).
   if (exec_mode > 0)
   {
      ///////////////////////////
      // Element contributions //
      ///////////////////////////
      Mbf.BilinearForm::operator=(0.0);
      Mbf.Assemble();
      Kbf.BilinearForm::operator=(0.0);
      Kbf.Assemble(0);
      ml.BilinearForm::operator=(0.0);
      ml.Assemble();
      ml.SpMat().GetDiag(lumpedM);
      
      ////////////////////////////
      // Boundary contributions //
      ////////////////////////////
      asmbl.bdrInt = 0.;
      for (k = 0; k < ne; k++)
      {
         if (dim==1)
         {
            mesh->GetElementVertices(k, bdrs);
         }
         else if (dim==2)
         {
            mesh->GetElementEdges(k, bdrs, orientation);
         }
         else if (dim==3)
         {
            mesh->GetElementFaces(k, bdrs, orientation);
         }
         
         for (i = 0; i < asmbl.dofs.numBdrs; i++)
         {
            if (exec_mode == 1)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               asmbl.ComputeFluxTerms(k, i, coef, Trans);
            }
            else if (exec_mode == 2)
            {
               Trans = mesh->GetFaceElementTransformations(bdrs[i]);
               asmbl.ComputeFluxTerms(k, i, asmbl.velocity, Trans);
            }
         }
      }
   }

   if (asmbl.monoType == 0)
   {
      ComputeHighOrderSolution(x, y);
   }
   else
   {
      if (asmbl.monoType % 2 == 1)
      {
         ComputeLowOrderSolution(x, y);
      }
      else if (asmbl.monoType % 2 == 0)
      {
         Vector yH, yL;
         yH.SetSize(x.Size()); yL.SetSize(x.Size());

         ComputeLowOrderSolution(x, yL);
         ComputeHighOrderSolution(x, yH);
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
      case 20:
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
      case 21:
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
         break;
      }
      case 10:
      case 11:
      {
         // Taylor-Green velocity, used for mesh motion in remap tests.

         // Map [-1,1] to [0,1].
         for (int d = 0; d < dim; d++) { X(d) = X(d) * 0.5 + 0.5; }

         if (dim == 1) { MFEM_ABORT("Not implemented."); }
         v(0) =  sin(M_PI*X(0)) * cos(M_PI*X(1));
         v(1) = -cos(M_PI*X(0)) * sin(M_PI*X(1));
         if (dim == 3)
         {
            v(0) *= cos(M_PI*X(2));
            v(1) *= cos(M_PI*X(2));
            v(2) = 0.0;
         }
         break;
      }
   }

   if (exec_mode == 2) { v *= -1.0; }
}

double box(std::pair<double,double> p1, std::pair<double,double> p2,
           double theta,
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
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
}

double box3D(double xmin, double xmax, double ymin, double ymax, double zmin,
             double zmax, double theta, double ox, double oy, double x,
             double y, double z)
{
   double pi = M_PI;
   double s=std::sin(theta*pi/180);
   double c=std::cos(theta*pi/180);

   double xn=c*(x-ox)-s*(y-oy)+ox;
   double yn=s*(x-ox)+c*(y-oy)+oy;

   if (xn>xmin && xn<xmax && yn>ymin && yn<ymax && z>zmin && z<zmax)
   {
      return 1.0;
   }
   else
   {
      return 0.0;
   }
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
      case 10:
      case 20:
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
      case 11:
      case 21:
      {
         double scale = 0.0225;
         double coef = (0.5/sqrt(scale));
         double slit = (X(0) <= -0.05) || (X(0) >= 0.05) || (X(1) >= 0.7);
         double cone = coef * sqrt(pow(X(0), 2.) + pow(X(1) + 0.5, 2.));
         double hump = coef * sqrt(pow(X(0) + 0.5, 2.) + pow(X(1), 2.));

         return (slit && ((pow(X(0),2.) + pow(X(1)-.5,2.))<=4.*scale)) ? 1. : 0.
                + (1. - cone) * (pow(X(0), 2.) + pow(X(1)+.5, 2.) <= 4.*scale)
                + .25 * (1. + cos(M_PI*hump))
                       * ((pow(X(0)+.5, 2.) + pow(X(1), 2.)) <= 4.*scale);
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
            double rect1 = box3D(7.,32.,10.,13.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect2 = box3D(14.,17.,3.,26.,10.,13.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));
            double rect3 = box3D(14.,17.,10.,13.,3.,26.,-45.,15.5,11.5,
                                 y(0),y(1),y(2));

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
