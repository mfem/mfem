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



double distance_(const IntegrationPoint &a, const IntegrationPoint &b)
{
   return sqrt((a.x - b.x) * (a.x - b.x) +
               (a.y - b.y) * (a.y - b.y) +
               (a.z - b.z) * (a.z - b.z));
}


class SolutionBounds {

   // set of local dofs which are in stencil of given local dof
   Mesh* mesh;
   FiniteElementSpace* fes;

   // 0 - sparsity pattern of K, 1 - local bounds
   int method;

   // metadata for computing local bounds

   DenseMatrix DOFs_coord;                           // size #dofs
   std::map<int, int> dof2elem;                      // size #dofs
   std::map<int, int> dof2ldof;                      // size #dofs
   std::map<int, std::vector<int> > map_for_bounds;  // size #dofs
   std::map<int, std::map<int, int> > cell_Ni;
   
public:

   Vector x_min;
   Vector x_max;

   SolutionBounds(FiniteElementSpace* fes_,
                  const BilinearForm& K, int method_) {

      fes = fes_;
      mesh = fes->GetMesh();
      method = method_;

      if (method == 1) GetBoundsMap(fes, K);
      
   }


   void Compute(const SparseMatrix &K, const Vector &x) {
            
      x_min.SetSize(x.Size());
      x_max.SetSize(x.Size());
      
      switch (method) {
      case 0:
         ComputeFromSparsity(K,x);
         break;
      case 1:
         ComputeFromLocalDOF(x);
         break;
      }
   }

   void ComputeFromSparsity(const SparseMatrix& K, const Vector& x) {
      
      const int *I = K.GetI(), *J = K.GetJ(), size = K.Size();
      
      for (int i = 0, k = 0; i < size; i++)
      {
         double x_i_min = std::numeric_limits<double>::infinity();
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

   void ComputeFromLocalDOF(const Vector& u) {
      printf("TODO\n");
      exit(1);
   }

private:

   // construct a matrix of coordinates for each DOF & construct maps
   // dof2elem and dof2ldof.
   
   void ComputeCoordinates(FiniteElementSpace* fes) {

      int dim = fes->GetMesh()->Dimension();
      int num_cells = fes->GetNE();
      int NDOFs        = fes->GetVSize();
      int NDOFs_shared = 0; //fes->num_face_nbr_dofs;
      DOFs_coord.SetSize(dim, NDOFs + NDOFs_shared);

      Array<int> ldofs;
      const FiniteElement *fe;
      DenseMatrix physical_coord;

      for (int i = 0; i < num_cells; i++)
      {
         fe = fes->GetFE(i);
         const IntegrationRule &ir = fe->GetNodes();
         ElementTransformation *el_trans = fes->GetElementTransformation(i);

         el_trans->Transform(ir, physical_coord);
         fes->GetElementDofs(i, ldofs);

         for (int j = 0; j < ldofs.Size(); j++)
         {
            for (int d = 0; d < dim; ++d)
            {
               DOFs_coord(d, ldofs[j]) = physical_coord(d, j);
            }
            dof2elem[ldofs[j]] = i;
            dof2ldof[ldofs[j]] = j;
         }
      }

      // Closest neighbors map on a reference cell.
      // dummy reference cell
      fe = fes->GetFE(0);
      int xd = fe->GetDof();
      const IntegrationRule &ir = fe->GetNodes();
      double tol = 1e-10;
      // hk at ref element with some tolerance
      double dist_level = 1.0 / fes->GetOrder(0)+tol;

      for (int i=0; i<xd; i++)
      {
         int count = 0;
         for (int j=0; j<xd; j++)
         {
            if (distance_(ir.IntPoint(i), ir.IntPoint(j)) <= dist_level)
            {
               cell_Ni[i][count++] = j;
            }
         }
      }
      
   }
   
   void GetBoundsMap(FiniteElementSpace *fes,
                     const BilinearForm &K) {

      ComputeCoordinates(fes);

      int dim = mesh->Dimension();
      int num_cells = mesh->GetNE();
      int NDOFs     = fes->GetVSize();
      double dist=0;
      const double tol = 1e-10;
      // hk at ref element with some tolerance
      //double dist_level = 1.0 / fes->GetOrder(0) + tol;
      // Include the diagonal neighbors.
      double dist_level = 1.5 / fes->GetOrder(0) + tol;
      Array<int> ldofs;
      Array<int> ldofs_external;
      const int *I = K.SpMat().GetI(), *J = K.SpMat().GetJ();

      const FiniteElement *fe_external;

      // loop over cells
      for (int k = 0; k < num_cells; k++)
      {
         fes->GetElementDofs(k, ldofs);
         const FiniteElement &fe = *fes->GetFE(k);
         int n_dofs = fe.GetDof();
         const IntegrationRule &ir = fe.GetNodes();

         //loop over DOF within cell
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
            // Loop over the already included internal DOFs (except on ith-DOF).
            // For each I have to find if within the sparsity pattern there is
            // other DOF with same physical location and add them to the map.
            vector<int> vector_of_internal_dofs = map_for_bounds[ldofs[i]];
            for (unsigned int it = 0; it < vector_of_internal_dofs.size(); it++)
            {
               int idof = vector_of_internal_dofs[it];
               if (idof != ldofs[i])
               {
                  // check sparsity pattern
                  for (int j = I[idof]; j < I[idof + 1]; j++)
                  {
                     if (dim==1)
                        dist = abs(DOFs_coord(0, idof) -
                                   DOFs_coord(0, J[j]));
                     else if (dim==2)
                        dist = sqrt(pow(DOFs_coord(0, idof) -
                                        DOFs_coord(0, J[j]), 2) +
                                    pow(DOFs_coord(1, idof) -
                                        DOFs_coord(1, J[j]), 2));
                     else
                        dist = sqrt(pow(DOFs_coord(0, idof) -
                                        DOFs_coord(0, J[j]), 2) +
                                    pow(DOFs_coord(1, idof) -
                                        DOFs_coord(1, J[j]), 2) +
                                    pow(DOFs_coord(2, idof) -
                                        DOFs_coord(2, J[j]), 2));
                     if (dist <= tol && idof != J[j])
                     {
                        map_for_bounds[ldofs[i]].push_back(J[j]);
                     }
                  }
               }
            }
            //////////////
            // SOURCE 2 //
            //////////////
            // Check if the current dof is on a face.
            // For the current ith DOF, loop over DOF in sparsity pattern of K
            // and find DOFs at ith-location.
            vector<int> DOFs_at_ith_location;
            for (int j = I[ldofs[i]]; j < I[ldofs[i] + 1]; j++)
            {
               if (dim==1)
                  dist = abs(DOFs_coord(0, ldofs[i]) -
                             DOFs_coord(0, J[j]));
               else if (dim==2)
                  dist = sqrt(pow(DOFs_coord(0, ldofs[i]) -
                                  DOFs_coord(0, J[j]), 2)   +
                              pow(DOFs_coord(1, ldofs[i]) -
                                  DOFs_coord(1, J[j]), 2));
               else
                  dist = sqrt(pow(DOFs_coord(0, ldofs[i]) -
                                  DOFs_coord(0, J[j]), 2) +
                              pow(DOFs_coord(1, ldofs[i]) -
                                  DOFs_coord(1, J[j]), 2) +
                              pow(DOFs_coord(2, ldofs[i]) -
                                  DOFs_coord(2, J[j]), 2));
               if (dist <= tol && ldofs[i] != J[j]) // dont include the ith DOF
               {
                  DOFs_at_ith_location.push_back(J[j]);   // here j is global
               }
            }
            // Loop over the j-DOFs at i-th location and for each loop over DOFs
            // local on its cell to find those within dist(1).
            // NOTE: this distance has to be on the reference element.
            for (unsigned int it = 0; it < DOFs_at_ith_location.size(); it++)
            {
               int dof = DOFs_at_ith_location[it];
               int cell_id = dof2elem[dof];
               if (dof < NDOFs)
               {
                  fes->GetElementDofs(cell_id, ldofs_external);
                  fe_external = fes->GetFE(cell_id);
               }
               else
               {
                  // not needed in serial?
                  exit(1);
                  
                  // fes->GetFaceNbrElementVDofs(cell_id, ldofs_external);
                  // fe_external = fes->GetFaceNbrFE(cell_id);

                  // for (int j = 0; j < ldofs.Size(); j++)
                  // {
                  //    ldofs_external[j] += NDOFs;
                  // }
               }

               int n_dofs_external = fe_external->GetDof();
               const IntegrationRule &ir_ext = fe_external->GetNodes();
               for (int j = 0; j < n_dofs_external; j++) // here j is local
               {
                  if (distance_(ir_ext.IntPoint(dof2ldof[dof]),
                                ir_ext.IntPoint(j)) <= dist_level)
                  {
                     map_for_bounds[ldofs[i]].push_back(ldofs_external[j]);
                  }
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
   const DenseMatrix &bdrDiff;
   DSmoother M_prec;
   CGSolver M_solver;
   int m_mono_type;
   
   SolutionBounds &bnds;

   double dt;

   mutable Vector z;

public:
   FE_Evolution(FiniteElementSpace* fes, SparseMatrix &_M, SparseMatrix &_K, 
                const Vector &_b, const Vector &_elDiff, const DenseMatrix &_bdrDiff, 
                const Vector &_lumpedM, int mono_type, SolutionBounds& bnds);

   virtual void Mult(const Vector &x, Vector &y) const;
   
   virtual void SetDt(double _dt) { dt = _dt; }

   virtual ~FE_Evolution() { }
};


void extractBdrDofs(int GeomType, int nd, const int bdrID, Array< int > &dofs)
{
   if (nd == 1)
   {
      dofs.SetSize(1);
      dofs[0] = 0;
      return;
   }
   switch (GeomType)
   {
      case Geometry::POINT:
         mfem_error ("No boundaries for geometry POINT.");
      case Geometry::SEGMENT:
      {
         dofs.SetSize(1);
         switch (bdrID)
         {
            case 0:
               dofs[0] = 0; return;
            case 1:
               dofs[0] = nd-1; return;
            default:
               mfem_error ("No more boundaries for geometry SEGMENT.");
         }
      }
      case Geometry::TRIANGLE:
      {
         //TODO
      }
      case Geometry::SQUARE:
      {
         int j = 0, n = sqrt(nd);
         dofs.SetSize(n);
         switch (bdrID)
         {
            case 0:
               for (int i = 0; i < n; i++)
                  dofs[j++] = i;
               return;
            case 1:
               for (int i = n-1; i < n*n; i+=n)
                  dofs[j++] = i;
               return;
            case 2:
               for (int i = n*(n-1); i < n*n; i++)
                  dofs[j++] = i;
               return;
            case 3:
               for (int i = 0; i <= n*(n-1); i+=n)
                  dofs[j++] = i;
               return;
            default:
               mfem_error ("No more boundaries for geometry SQUARE.");
         }
      }
      case Geometry::TETRAHEDRON: 
      {
         //TODO
      }
      case Geometry::CUBE:
      {
         int k = 0, n = cbrt(nd);
         dofs.SetSize(n*n);
         switch (bdrID)
         {
            case 0:
               for (int i = 0; i < n*n; i++)
                  dofs[k++] = i;
               return;
            case 1:
               for (int i = 0; i <= n*n*(n-1); i+=n*n)
                  for (int j = 0; j < n; j++)
                     dofs[k++] = i+j;
               return;
            case 2:
               for (int i = n-1; i < n*n*n; i+=n)
                  dofs[k++] = i;
               return;
            case 3:
               for (int i = 0; i <= n*n*(n-1); i+=n*n)
                  for (int j = n*(n-1); j < n*n; j++)
                     dofs[k++] = i+j;
               return;
            case 4:
               for (int i = 0; i <= n*(n*n-1); i+=n)
                  dofs[k++] = i;
               return;
            case 5:
               for (int i = n*n*(n-1); i < n*n*n; i++)
                  dofs[k++] = i;
               return;
            default:
               mfem_error ("No more boundaries for geometry CUBE.");
         }
      }
      default:
         mfem_error ("extractBdrDofs(...)");
   }
}
/* TODO remove or implement and remove Manuel's code
void getNeighbourDofs(int mono_type, const Vector &x, int dofInd, int nd, 
                      int GeomType, Array< int > &nghbrs)
{
   //nghbrs.SetSize(pow(3, Geometry::Dimension[GeomType])); // works for Q-spaces
   nghbrs.SetSize(nd); // element-based
   int locInd = dofInd % nd;
   for (int i = 0; i < nd; i++)
      nghbrs[i] = dofInd - locInd + i;
   
   switch (GeomType)
   {
      case Geometry::POINT:
         mfem_error ("Nothing to do for geometry POINT.");
      case Geometry::SEGMENT:
      {
         return;//TODO
      }
      case Geometry::TRIANGLE:
      {
         return;//TODO
      }
      case Geometry::SQUARE:
      {
         return;//TODO
      }
      case Geometry::TETRAHEDRON:
      {
         return;//TODO
      }
      case Geometry::CUBE:
      {
         return;//TODO
      }
   }
}

void computeAdmissibleBounds(int mono_type, const Vector &x, int dofInd, int nd,
                             int GeomType, double &xMin, double &xMax)
{
   if (mono_type == 0)
      return;
   else if (mono_type == 1)
   {
      Array< int > nghbrs;
      getNeighbourDofs(mono_type, x, dofInd, nd, GeomType, nghbrs);
      
      xMin = x(dofInd);
      xMax = x(dofInd);
      for (int i = 0; i < nghbrs.Size(); i++)
      {
         if (nghbrs[i] == dofInd)
            continue;
         xMin = std::min(xMin, x(nghbrs[i]));
         xMax = std::min(xMin, x(nghbrs[i]));
      }
   }
}
*/

void preprocessLowOrderScheme(FiniteElementSpace* fes, VectorFunctionCoefficient & coef, 
                              int mono_type, Vector &elDiff, DenseMatrix &bdrDiff, 
                              Vector &lumpedM)
{
   if (mono_type == 0)
      return;
   
   Mesh *mesh = fes->GetMesh();
   int i, j, k, p, qOrd, nd, geomType, numBdrs, numDofs,
   dim = mesh->Dimension(), ne = mesh->GetNE();
   ElementTransformation *tr;
   Vector shape, vec1, vec2, estim1, estim2, vval(dim), nor(dim);
   DenseMatrix dshape, adjJ;
   Array< int > dofs, bdrs, orientation;
   DenseMatrix velEval;
   
   elDiff.SetSize(ne); elDiff = 0.;
   lumpedM.SetSize(ne); lumpedM = 0.;
   adjJ.SetSize(dim,dim);
   
   for (k = 0; k < ne; k++)
   {
      ///////////////////////////
      // Element contributions //
      ///////////////////////////
      const FiniteElement &el = *fes->GetFE(k);
      nd = el.GetDof();
      tr = mesh->GetElementTransformation(k);
      // estim1 can not be integrated exactly due to transforamtion dependent denominator
      // use tr->Order()-1 + 4*el.GetOrder() instead
      // appropriate qOrd for estim2 is tr->Order()-1 + 2*el.GetOrder(), choose max
      qOrd = tr->Order()-1 + 4*el.GetOrder();
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
      
      ////////////////////////
      // Face contributions //
      ////////////////////////
      if (dim==1)
         mesh->GetElementVertices(k, bdrs);
      else if (dim==2)
         mesh->GetElementEdges(k, bdrs, orientation);
      else if (dim==3)
         mesh->GetElementFaces(k, bdrs, orientation);
      
      numBdrs = bdrs.Size();
      geomType = el.GetGeomType();

      if (k==0)
         bdrDiff.SetSize(ne,numBdrs);

      for (i = 0; i < numBdrs; i++)
      {
         extractBdrDofs(geomType, nd, i, dofs);
         //fes->FEColl()->SubDofOrder(geomType, dim-1, 64*i, dofs); TODO
         numDofs = dofs.Size();
         
         FaceElementTransformations *Trans = mesh -> GetFaceElementTransformations(bdrs[i]); 
         // qOrd is chosen such that L2-norm of basis is computed accurately.
         // Normal velocity term relies on L^{infty}-norm which is approximated 
         // by its maximum value in the quadrature points of the same rule.
         if (Trans->Elem1No != k)
         {
            if (Trans->Elem2No != k)
               mfem_error("Boundary edge does not belong to this element.");
            else
               qOrd = Trans->Loc2.Transf.Order() + 2*el.GetOrder();
         }
         else
         {
            qOrd = Trans->Loc1.Transf.Order() + 2*el.GetOrder();
         }

         const IntegrationRule *irF1 = &IntRules.Get(Trans->FaceGeom, qOrd);
         double un = 0.;
         Vector bas(numDofs);
         bas = 0.;
         
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
               bas(j) += ip.weight * Trans->Face->Weight() * pow(shape(dofs[j]), 2.);
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
   int mono_type = 1;
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
   args.AddOption(&mono_type, "-mt", "--mono_type",
                  "Type of monotonicity treatment: 0 - no monotonicity treatment,\n\t"
                  "                                1 - matrix-free Rusanov scheme.");
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
   
   if ((mono_type != 0) && (mono_type != 1) && (mono_type != 2))
   {
      cout << "Unsupported option for monotonicity treatment." << endl;
      return 5;
   }
   if (mono_type > 0)
   {
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
   DenseMatrix bdrDiff;
   preprocessLowOrderScheme(&fes, velocity, mono_type, elDiff, bdrDiff, lumpedM);
   
   int method = 0; // from sparsity
//   int method = 1; // local bounds
   SolutionBounds bnds(&fes, k, method);

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
   FE_Evolution adv(&fes, m.SpMat(), k.SpMat(), b, elDiff, bdrDiff, lumpedM, mono_type, bnds);
   
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
   delete ode_solver;
   delete dc;
   return 0;
   
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace* _fes, SparseMatrix &_M, SparseMatrix &_K, 
                           const Vector &_b, const Vector &_elDiff, const DenseMatrix 
                           &_bdrDiff, const Vector &_lumpedM, int mono_type, 
                           SolutionBounds& _bnds)
   : TimeDependentOperator(_M.Size()), fes(_fes), M(_M), K(_K), b(_b), elDiff(_elDiff), 
   lumpedM(_lumpedM), bdrDiff(_bdrDiff), m_mono_type(mono_type), z(_M.Size()), bnds(_bnds)
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
   if (m_mono_type == 0)
   {
      // No monotonicity treatment, straightforward high-order scheme
      // ydot = M^{-1} (K x + b)
      K.Mult(x, z);
      z += b;
      M_solver.Mult(z, y);
   }
   else if (m_mono_type == 1)
   {
      // low order matrix-free Rusanov scheme
      Mesh *mesh = fes->GetMesh();
      int i, j, k, geomType, nd, numDofs, numBdrs, dofInd, dim = mesh->Dimension();
      Array< int > bdrs, orientation, dofs;
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
         
         geomType = el.GetGeomType();
         numBdrs = bdrs.Size();

         ////////////////////////
         // Face contributions //
         ////////////////////////
         for (i = 0; i < numBdrs; i++)
         {
            extractBdrDofs(geomType, nd, i, dofs);
            //fes->FEColl()->SubDofOrder(geomType, dim-1, 64*i, dofs); TODO
            numDofs = dofs.Size();
            
            uSum = 0.;
            for (j = 0; j < numDofs; j++)
               uSum += x(k*nd+dofs[j]);
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+dofs[j]) += bdrDiff(k,i)*(uSum - numDofs*x(k*nd+dofs[j]));
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
   else if (m_mono_type == 2)
   {
      // High order reconstruction that leads to an updated admissible solution.
      Mesh *mesh = fes->GetMesh();
      int i, j, k, geomType, nd, numDofs, numBdrs, dofInd, dim = mesh->Dimension();
      Array< int > bdrs, orientation, dofs;
      double uDof, uSum, uMin, uMax, sumPos, sumNeg, eps = 1.E-15;
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
         
         geomType = el.GetGeomType();
         numBdrs = bdrs.Size();

         ////////////////////////
         // Face contributions //
         ////////////////////////
         for (i = 0; i < numBdrs; i++)
         {
            extractBdrDofs(geomType, nd, i, dofs);
            //fes->FEColl()->SubDofOrder(geomType, dim-1, 64*i, dofs); TODO
            numDofs = dofs.Size();
            
            uSum = 0.;
            for (j = 0; j < numDofs; j++)
               uSum += x(k*nd+dofs[j]);
            
            // boundary update
            for (j = 0; j < numDofs; j++)
               z(k*nd+dofs[j]) += bdrDiff(k,i)*(uSum - numDofs*x(k*nd+dofs[j]));
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
            // Use this loop also to compute local min-/max-values TODO Kommentar weg
            //computeAdmissibleBounds(m_mono_type, x, dofInd, nd, geomType, uMin, uMax);//mine
            
            uClipped(j) = std::min(bnds.x_max(dofInd), 
                                   std::max(x(dofInd) + dt * y(dofInd), bnds.x_min(dofInd))); //TODO more gneral than Euler
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
            //       entries AND the way this method works
            fClipped(j) = uClipped(j) - ( x(dofInd) + dt * z(dofInd) );
            //TODO more gneral than Euler
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
            // y is now the discrete time derivative featuring the high order anti-
            // diffusive reconstruction that leads to an forward Euler updated admissible // solution. The factor dt in the denominator is used for compensation in the // ODE solver.
            //TODO more gneral than Euler
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
         /*double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);*/
         
         double scale = 0.09;
         double G1 = (1./sqrt(scale)) * sqrt(pow(X(0),2.) + pow(X(1)+0.5,2.));
         double G2 = (1./sqrt(scale)) * sqrt(pow(X(0)-0.5,2.) + pow(X(1),2.));

         return ((pow(X(0),2.) + pow(X(1) - 0.5,2.) <= scale) & 
                  (X(0) <= -0.05 | X(0) >= 0.05 | X(1) >= 0.7)) ? 1. : 0. +
                  (1-G1) * (pow(X(0),2.) + pow(X(1) + 0.5,2.) <= scale) + 
                  0.25*(1.+cos(M_PI*G2))*(pow(X(0) - 0.5,2.) + pow(X(1),2.) <= scale);
      }
      case 3:
      {
         const double f = M_PI;
         return .5*(sin(f*X(0))*sin(f*X(1)) + 1.); //modified by Hennes 
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
      case 3: return 0.0;
   }
   return 0.0;
}
