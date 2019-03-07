// Locate this file in the example folder

// Compile: c++  -g -Wall -I.. ex_kellogg.cpp -o ex_kellogg -L.. -xlmfem

// run ./ex_kellogg -m ../data/square-quad.mesh -o 1
//     ./ex_kellogg -m ../data/square-tri.mesh -o 1

// Description:  This example code demonstrates the use of MFEM to approximate a
//               simple finite element discretization of the Interface Kellogg problem
//               Specifically, we discretize using a conforming FE space of
//               the specified order.
//               The example highlights the use of adaptive mesh refinement (AMR),
//               which handles both regular conforming triangular mesh and non-conforming
//               quadrilateral mesh with arbitrary order of haning nodes.
//               The error estimation algorithm is applicable to any
//               Laplace, Interface and diffusion problems.

#include <fstream>
#include "mfem.hpp"
#include "math.h"

using namespace std;
using namespace mfem;

double kellogg_exsol(Vector &p)
{
   double gamma = 0.1;
   double sigma = -14.9225565104455152;
   double rho = M_PI/4;
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y);
   double theta = atan2(y,x);
   if (theta < 0) { theta += 2*M_PI; }
   double mu = (theta>=0 && theta <M_PI/2 ) *cos((M_PI/2-sigma) * gamma)
               * cos((theta-M_PI/2 + rho) *gamma)
               +(theta >= M_PI/2 && theta < M_PI) * cos(rho * gamma)
               * cos((theta - M_PI + sigma) * gamma)
               +(theta >= M_PI && theta< 1.5 * M_PI) * cos(sigma * gamma)
               * cos((theta - M_PI - rho) * gamma)
               +(theta >= 1.5*M_PI && theta<2*M_PI) * cos((M_PI/2-rho) * gamma)
               * cos((theta - 1.5 * M_PI-sigma) * gamma);
   return pow(r, gamma) * mu;
}

class GradientGridFunctionCoeffiecent : public VectorCoefficient
{
private:
   GridFunction *uh;
   Coefficient *A;

public:
   GradientGridFunctionCoeffiecent(GridFunction &u, Coefficient &a)
      : VectorCoefficient(u.FESpace()->GetMesh()->SpaceDimension()),
        uh(&u), A(&a) { }

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      uh->GetGradient(T, V);//Transformation contains the ip information
      V *= A->Eval(T,
                   ip); //ip does not have to be the point corresponding to one of the DOF of u
   }
};



int RecoverFluxforConformingMesh( BilinearForm &a,
                                  VectorCoefficient &vcoeff,
                                  GridFunction &flux)
{
   FiniteElementSpace *fes = flux.FESpace();
   Array<int> fdofs, edofs;
   Vector fvals, evals;
   Mesh *mesh=fes->GetMesh();

   //Set up interiori dofs. Notice here that the dofs on interiori faces might be set several times
   //however, this will not cause issues since those dofs will be determined later.
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementDofs(i,edofs);
      evals.SetSize(edofs.Size());
      fes->GetFE(i)->Project(vcoeff,*fes->GetElementTransformation(i), evals);
      flux.SetSubVector(edofs,evals);
   }

   //determine dofs on each interiori face
   for (int i = 0; i < fes->GetMesh()->GetNumFaces(); i++)
   {
      int e1, e2;
      mesh->GetFaceElements(i, &e1, &e2);
      //if boundary face, then continue.
      //This algorithm assume the Dirichlet boundary condition problem
      if (e1 < 0 || e2 < 0) { continue; }

      fes->GetFaceDofs(i, fdofs);
      fvals.SetSize(fdofs.Size());

      Array<int> ldofs, rdofs;
      Vector lvals, rvals;
      DenseMatrix lmatrix, rmatrix;

      fes->GetElementDofs(e1, ldofs);
      fes->GetElementDofs(e2, rdofs);

      int ndf = ldofs.Size();
      lmatrix.SetSize(ndf);
      rmatrix.SetSize(ndf);

      a.ComputeElementMatrix(e1,lmatrix);
      a.ComputeElementMatrix(e2,rmatrix);

      lvals.SetSize(ndf);
      rvals.SetSize(ndf);


      fes->GetFE(e1)->Project(vcoeff,*fes->GetElementTransformation(e1), lvals);
      fes->GetFE(e2)->Project(vcoeff,*fes->GetElementTransformation(e2), rvals);

      double weight;
      int lindex, rindex;
      Vector recovered_flux;
      recovered_flux.SetSize(fdofs.Size());

      for (int j = 0; j < fdofs.Size(); j++)
      {
         for (int k = 0; k < ndf; k++)
         {
            if (ldofs[k] == fdofs[j])
            {
               lindex = k;
               break;
            }
         }
         for (int k = 0; k < rdofs.Size(); k++)
         {
            if (rdofs[k] + fdofs[j] == -1)
            {
               rindex = k;
               break;
            }
         }
         weight = lmatrix.Elem(lindex,lindex)/(lmatrix.Elem(lindex,
                                                            lindex) + rmatrix.Elem(rindex,rindex));
         recovered_flux[j] = lvals[lindex] * weight- rvals[rindex] * (1-weight);
      }

      flux.SetSubVector(fdofs,recovered_flux);
   }

   return 1;
}

//Determine dofs for each conforming face on NC (quad) mesh.
int RecoverFluxatConformingFaces( BilinearForm &a,
                                  VectorCoefficient &vcoeff,
                                  GridFunction &flux)
{
   FiniteElementSpace *fes = flux.FESpace();
   Mesh *mesh =fes->GetMesh();
   const mfem::NCMesh::NCList &list=mesh->ncmesh->GetEdgeList();
   int e1, e2;
   Array<int> fdofs;
   Vector fvals;

   for (unsigned i = 0; i< list.conforming.size(); i++)
   {
      const NCMesh::MeshId &cface = list.conforming[i];
      int findex = cface.index;
      fes->GetFaceDofs(findex,fdofs);

      fvals.SetSize(fdofs.Size());
      mesh->GetFaceElements(findex, &e1, &e2);
      //if boundary face, then return.
      if (e1<0 || e2 < 0) { continue;}

      Array<int> ldofs, rdofs;
      Vector lvals, rvals;

      DenseMatrix lmatrix, rmatrix;
      fes->GetElementDofs(e1, ldofs);
      fes->GetElementDofs(e2, rdofs);
      int ndf=ldofs.Size();
      lmatrix.SetSize(ndf);
      rmatrix.SetSize(ndf);

      a.ComputeElementMatrix(e1,lmatrix);
      a.ComputeElementMatrix(e2,rmatrix);

      lvals.SetSize(ndf);
      rvals.SetSize(ndf);

      fes->GetFE(e1)->Project(vcoeff,*fes->GetElementTransformation(e1), lvals);
      fes->GetFE(e2)->Project(vcoeff,*fes->GetElementTransformation(e2), rvals);

      double weight;
      int lindex = -1, rindex = -1;
      Vector recovered_flux;
      recovered_flux.SetSize(fdofs.Size());

      for (int j = 0; j < fdofs.Size(); j++)
      {

         for (int k = 0; k < ldofs.Size(); k++)
         {
            if (ldofs[k] == fdofs[j])
            {
               lindex = k;
               break;
            }
         }
         for (int k = 0; k < rdofs.Size(); k++)
         {

            if (rdofs[k] + fdofs[j] == -1)
            {
               rindex = k;
               break;
            }
         }
         weight = lmatrix.Elem(lindex,lindex)/(lmatrix.Elem(lindex,
                                                            lindex) + rmatrix.Elem(rindex,rindex));
         recovered_flux[j] = lvals[lindex] * weight - rvals[rindex] * (1-weight);
      }
      flux.SetSubVector(fdofs, recovered_flux);
   }
   return 1;
}

//Determine dofs for each master face on NC (quad) mesh.
int RecoverFluxatMasterFaces(BilinearForm &a,
                             VectorCoefficient &vcoeff,
                             GridFunction &flux)
{
   FiniteElementSpace *fes = flux.FESpace();
   Mesh *mesh = fes->GetMesh();
   const SparseMatrix* cP = fes->GetConformingProlongation();
   const SparseMatrix *R = fes->GetConformingRestriction();

   const mfem::NCMesh::NCList &list = mesh->ncmesh->GetEdgeList();
   if (!list.masters.size()) { return 0;}

   Array<int> slave_dofs,slave_edofs, master_dofs, master_edofs,
         master_lindex, slave_lindex;
   DenseMatrix mmaster, mslave,  Ai, dep, S;
   Vector slavevals, slavefvals, mastervals, masterfvals, b;

   Array<int> P(cP->Height());
   P = -1;
   int *Ri;
   Ri = R->GetJ();

   for (int i=0; i< R->Height(); i++)
   {
      //Projrction vector maps the index of global whole DOF to the index of global true DOF.
      //Note that when a DOF is a slave DOF, then its global whole DOF index will be mapped to 0.
      P[Ri[i]] = i;
   }

   for (unsigned mi = 0; mi< list.masters.size(); mi++)
   {
      const NCMesh::Master &master = list.masters[mi];
      fes->GetFaceDofs(master.index, master_dofs);
      if (!master_dofs.Size()) { continue; }

      //get local index of the face dofs
      int mindex = mesh->ncmesh->GetElementFromNCIndex(master.element);
      fes->GetElementDofs(mindex, master_edofs);

      int nfdof = master_dofs.Size();
      int nedof = master_edofs.Size();

      master_lindex.SetSize(nfdof);
      master_lindex = -1;

      for (int i = 0; i< nfdof; i++)
      {
         //there is a optimal way. After finding out the first local index, we can derive the rest, I suppose?
         for ( int j = 0; j< nedof; j++)
         {
            if (master_dofs[i] == master_edofs[j] )
            {
               master_lindex[i] = j;
               break;
            }
         }
      }

      //set up local face stiffness matrix
      a.ComputeElementMatrix(mindex, mmaster);
      S.SetSize(nfdof);

      for (int i=0; i< nfdof; i++)
         for (int j=0; j< nfdof; j++)
         {
            S(i,j) = mmaster.Elem(master_lindex[i], master_lindex[j]);
         }

      masterfvals.SetSize(nfdof);
      mastervals.SetSize(nedof);
      fes->GetFE(mindex)->Project(vcoeff,*fes->GetElementTransformation(mindex),
                                  mastervals);

      for (int i = 0; i < nfdof; i++)
      {
         masterfvals[i] = mastervals[master_lindex[i]];
      }

      Ai.SetSize(nfdof);
      double ele_size = fes->GetMesh()->GetElementSize(mindex);
      ele_size *=
         ele_size; //Here assume that each element is square, the area of the face needs to be used instead.
      S *= 1/ele_size;
      Ai = S;

      b.SetSize(nfdof);
      S.Mult(masterfvals,b);

      int m_sign = 1;
      if (master_dofs[0] < 0)
      {
         m_sign = -1;
      }
      int sign;

      for (int si = master.slaves_begin; si < master.slaves_end; si++)
      {
         const NCMesh::Slave &slave = list.slaves[si];
         fes->GetFaceDofs(slave.index, slave_dofs);

         int sindex, e1;
         mesh->GetFaceElements(slave.index, &sindex, &e1);
         fes->GetElementDofs(sindex, slave_edofs);

         slave_lindex.SetSize(nfdof);
         slave_lindex = -1;

         for (int i = 0; i< nfdof; i++)
         {
            for ( int j = 0; j< nedof; j++)
            {
               if (slave_dofs[i] == slave_edofs[j] )
               {
                  slave_lindex[i] = j;
                  break;
               }
            }
         }

         mslave.SetSize(nedof);
         a.ComputeElementMatrix(sindex, mslave);
         slavevals.SetSize(nedof);
         fes->GetFE(sindex)->Project(vcoeff,*fes->GetElementTransformation(sindex),
                                     slavevals);

         dep.SetSize(nfdof);

         for (int j = 0; j< nfdof; j++)
         {
            int ji = slave_dofs[j]<0 ? -slave_dofs[j]-1:slave_dofs[j];
            for (int k = 0; k< nfdof; k++)
            {
               int ki = master_dofs[k]<0 ? -master_dofs[k]-1: master_dofs[k];
               dep(j,k) = cP->Elem(ji, P[ki]);//positive index to positve index
               S(j,k) = mslave.Elem(slave_lindex[j], slave_lindex[k]);
            }
         }


         if (slave_dofs[0] < 0)
         {
            sign = -m_sign;
         }
         else
         {
            sign = m_sign;
         }

         dep *= sign;//now dep is the local to local dependence matrix

         ele_size = fes->GetMesh()->GetElementSize(sindex);
         ele_size *= ele_size;
         S *= (1/ele_size);

         DenseMatrix deptS;
         deptS.SetSize(nfdof);
         MultAtB(dep, S, deptS);
         DenseMatrix AB;
         AB.SetSize(nfdof);
         Mult(deptS, dep, AB);
         Ai += AB;

         slavefvals.SetSize(nfdof);
         for (int i = 0; i < nfdof; i++)
         {
            slavefvals[i] = slavevals[slave_lindex[i]];
         }

         deptS.AddMult(slavefvals, b);
      }

      //solve the local minimization problems
      DenseMatrixInverse Aiinv(Ai);
      Vector x;
      x.SetSize(b.Size());
      Aiinv.Mult(b,x);
      flux.SetSubVector(master_dofs,x);
   }

   //do the restriction and prolongation. One can also do the restiction locally.
   Vector X;
   X.SetSize(R->Height());
   R->Mult(flux,X);
   cP->Mult(X,flux);
   return 1;
}

//error estimation function based on the recovered flux and numerical solution
double ComputeEnergynormError(BilinearForm &a,
                              VectorCoefficient &flux_coeff,
                              Coefficient &alpha,
                              GridFunction &flux,
                              Vector &eta_Ks,
                              int residual)//residual = 0, only compute flux jump; residual =1, only compute element residual; residual =2, compute both.
{
   FiniteElementSpace *flux_fes = flux.FESpace();

   eta_Ks = 0.;
   double eta = 0;
   double eta_K;
   int ne = flux_fes->GetNE();

   Vector evals;
   Array<int> edofs;
   GridFunction numer_flux(flux_fes);
   numer_flux = 0.;

   for (int i = 0 ; i < ne ; i++)
   {
      eta_K = 0;
      flux_fes->GetElementDofs(i,edofs);
      int ndf = edofs.Size();
      evals.SetSize(ndf);
      flux_fes->GetFE(i)->Project(flux_coeff,*flux_fes->GetElementTransformation(i),
                                  evals);//local projection
      numer_flux.SetSubVector(edofs,evals);

      if (residual == 0 ||residual == 2) //compute flux jump error
      {
         DenseMatrix mmatrix;
         mmatrix.SetSize(ndf);
         a.ComputeElementMatrix(i,mmatrix);

         Vector smooth_evals;
         smooth_evals.SetSize(ndf);
         flux.GetSubVector(edofs,smooth_evals);

         evals -= smooth_evals;
         eta_K += mmatrix.InnerProduct(evals,evals);
      }

      if (residual > 0) //compute element residual
      {
         const FiniteElement* fe = flux_fes->GetFE(i);
         const IntegrationRule *ir;
         int intorder = 2* fe->GetOrder() + 1;
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         int nip = ir->GetNPoints();
         ElementTransformation *T = flux_fes->GetElementTransformation(i);

         double ele_residual = 0;
         double ele_residual_norm=0;
         double alpha_K=1;

         for (int j = 0; j < nip; j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            ele_residual = numer_flux.GetDivergence(*T);
            ele_residual *= ele_residual;
            ele_residual_norm += ip.weight* T->Weight() * ele_residual;
            if (j==0)
            {
               alpha_K = alpha.Eval(*T, ip);
            }
         }

         double ele_size=flux_fes->GetMesh()->GetElementSize(i);
         ele_residual_norm *= (ele_size * ele_size/alpha_K);
         eta_K += ele_residual_norm;
      }

      eta += eta_K;
      eta_Ks[i] = sqrt(eta_K);
   }

   eta = sqrt(eta);
   return eta;
}

double NCImprovedZZErrorEstimation(
   GridFunction &flux,
   VectorCoefficient &flux_coeff,
   Vector &eta_Ks)
{
   FiniteElementSpace *flux_fes = flux.FESpace();
   BilinearForm *a=new BilinearForm(flux_fes);

   Vector inv_attri_list(2);
   inv_attri_list(0)=1;
   inv_attri_list(1)=1/161.4476387975881;
   PWConstCoefficient inv_alpha(inv_attri_list);

   Vector attri_list;
   attri_list.SetSize(2);
   attri_list(0) = 1;
   attri_list(1) = 161.4476387975881;
   PWConstCoefficient alpha(attri_list);

   a->AddDomainIntegrator(new VectorFEMassIntegrator(inv_alpha));
   a->ComputeElementMatrices();

   double eta = 0;
   if (flux_fes->GetMesh()->Conforming())
   {
      RecoverFluxforConformingMesh(*a,flux_coeff,flux);
      eta = ComputeEnergynormError(*a,flux_coeff,alpha, flux, eta_Ks,0);
   }
   else
   {
      Array<int> edofs;
      Vector evals;
      for (int i = 0; i < flux_fes->GetNE(); i++)
      {
         flux_fes->GetElementDofs(i,edofs);
         evals.SetSize(edofs.Size());
         flux_fes->GetFE(i)->Project(flux_coeff,*flux_fes->GetElementTransformation(i),
                                     evals);
         flux.SetSubVector(edofs,evals);
      }

      RecoverFluxatConformingFaces(*a, flux_coeff, flux);
      RecoverFluxatMasterFaces(*a, flux_coeff, flux);
      eta = ComputeEnergynormError(*a,flux_coeff,alpha, flux, eta_Ks,0);
   }

   return eta;
}

double ComputeEnergynorm( GridFunction &flux,
                          VectorCoefficient &flux_coeff)
{
   FiniteElementSpace *fes = flux.FESpace();
   BilinearForm *a=new BilinearForm(fes);

   Vector inv_attri_list(2);
   inv_attri_list(0)=1;
   inv_attri_list(1)=1/161.4476387975881;
   PWConstCoefficient inv_alpha(inv_attri_list);

   a->AddDomainIntegrator(new VectorFEMassIntegrator(inv_alpha));
   a->ComputeElementMatrices();

   int ne = fes->GetNE();
   Vector evals;
   DenseMatrix mmatrix;
   double energy_norm = 0;

   for (int i = 0; i < ne; i++)
   {
      a->ComputeElementMatrix(i, mmatrix);
      evals.SetSize(mmatrix.Width());
      fes->GetFE(i)->Project(flux_coeff,*fes->GetElementTransformation(i),
                             evals);//local projection
      energy_norm += mmatrix.InnerProduct(evals,evals);
   }

   return energy_norm;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   ifstream imesh(mesh_file);
   if (!imesh)
   {
      cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
      return 2;
   }
   Mesh mesh(imesh, 1, 1);
   imesh.close();
   int dim = mesh.Dimension();

   // 3. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }

   // 4. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 5. As in Example 1, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 1. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   Vector attri_list;
   attri_list.SetSize(2);
   attri_list(0) = 1;
   attri_list(1) = 161.4476387975881;
   PWConstCoefficient alpha(attri_list);
   ConstantCoefficient zero(0.0);



   a.AddDomainIntegrator(new DiffusionIntegrator(alpha));
   //b.AddDomainIntegrator(new DomainLFIntegrator(one));

   // 6. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   GridFunction x(&fespace);
   x = 0;

   // 7. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 8. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 9. The main AMR loop. In each iteration we solve the problem on the
   //    current mesh, visualize the solution, estimate the error on all
   //    elements, refine the worst elements and update all objects to work
   //    with the new mesh.

   ofstream myfile;
   if (order==1)
   {
      myfile.open ("kellogg_error_nc_1.txt");
   }
   else if (order==2)
   {
      myfile.open ("kellogg_error_nc_2.txt");
   }
   else if (order==3)
   {
      myfile.open ("kellogg_error_nc_3.txt");
   }
   else if (order==4)
   {
      myfile.open ("kellogg_error_nc_4.txt");
   }

   //const int max_dofs = 8000;//for linear order
   const int max_dofs = 20000;
   for (int it = 0; it <300  ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      cout << "\nAMR iteration " << it << endl;
      cout << "Number of unknowns: " << cdofs << endl;

      // 10. Assemble the stiffness matrix and the right-hand side. Note that
      //     MFEM doesn't care at this point if the mesh is nonconforming (i.e.,
      //     contains hanging nodes). The FE space is considered 'cut' along
      //     hanging edges/faces.
      a.Assemble();
      b.Assemble();

      // 11. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      FunctionCoefficient exsol(kellogg_exsol);
      x.ProjectBdrCoefficient(exsol, ess_bdr);
      Array<int> ess_tdof_list;
      //x.ProjectBdrCoefficient(zero, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 12. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      SparseMatrix A;
      Vector B, X;
      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

#ifndef MFEM_USE_SUITESPARSE
      // 13. Define a simple symmetric Gauss-Seidel preconditioner and use it to
      //     solve the linear system with PCG.
      GSSmoother M(A);
      PCG(A, M, B, X, 3, 200, 1e-12, 0.0);
#else
      // 13. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
      //     the linear system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(A);
      umf_solver.Mult(B, X);
#endif

      // 14. After solving the linear system, reconstruct the solution as a finite
      //     element grid function. Constrained nodes are interpolated from true
      //     DOFs (it may therefore happen that dim(x) >= dim(X)).
      a.RecoverFEMSolution(X, b, x);

      // 15. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
         //sol_sock << "solution\n" << mesh << flush;
      }


      if (cdofs > max_dofs)
      {
         break;
      }

      // 16. Estimate element errors using the local improved Zienkiewicz-Zhu error estimator.
      //     The bilinear form integrator must have the 'ComputeElementFlux'
      //     method defined.
      Vector errors(mesh.GetNE());
      errors =0.;
      double true_error=0;
      {
         int geom_type = mesh.GetElementBaseGeometry(0);
         bool triangles = (geom_type == Geometry::TRIANGLE);
         int RT_order = triangles? order-1: order;
         RT_FECollection smooth_flux_fec(RT_order, dim);
         FiniteElementSpace smooth_flux_fes(&mesh, &smooth_flux_fec);
         GridFunction flux(&smooth_flux_fes);
         GradientGridFunctionCoeffiecent flux_coeff(x, alpha);

         double eta = NCImprovedZZErrorEstimation(flux,flux_coeff, errors);

         cout << it << " "<< cdofs << " " << eta << endl;
         double u_h_norm = ComputeEnergynorm(flux,flux_coeff);
         true_error= abs(0.31923804457854 - u_h_norm);
         true_error = sqrt(true_error);

         cout << it << " "<< cdofs << " " << true_error << endl;
         myfile << it << " "<< cdofs<< " " << true_error << " " << eta << "\n";
      }

      if (true_error < sqrt(0.31923804457854)*0.05)
      {
         break;
      }
      // 17. Make a list of elements whose error is larger than a fraction of
      //     the maximum element error. These elements will be refined.
      Array<Refinement> ref_list;
      const double frac = 0.8;
      double threshold = frac * errors.Max();
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold)
         {
            //ref_list.Append(Refinement(i, aniso_flags[i]));
            ref_list.Append(Refinement(i));
         }
      }

      // 18. Refine the selected elements.
      mesh.GeneralRefinement(ref_list,-1,4);

      // 19. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations since
      //     we'll have a good initial guess of x in the next step. Internally,
      //     FiniteElementSpace::Update() calculates an interpolation matrix
      //     which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 20. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }
   myfile.close();

   return 0;
}
