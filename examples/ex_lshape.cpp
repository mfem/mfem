// Locate this file in the example folder

// Compile with: c++  -g -Wall -I.. ex_lshape.cpp -o ex_lshape -L.. -lmfem
//
// Sample runs:  ./ex_lshape -m ../data/lshape-quad.mesh -o 1
//               ./ex_lshape -m ../data/lshape-tri.mesh -o 1

// Description:  This example code demonstrates the use of MFEM to approximate a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 0 with Dirichlet boundary conditions such that
//                          u(r, θ) = r^2/3 sin(2θ/3).
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


//L-shape analytical solution
double lshape_exsol(Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y);
   double a = atan2(y, x);
   if (a < 0) { a += 2*M_PI; }
   return pow(r, 2.0/3.0) * sin(2.0*a/3.0);
}

// L-shape analytical gradient
void lshape_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double a = atan2(y, x);
   if (a < 0) { a += 2*M_PI; }
   double theta23 = 2.0/3.0*a;
   double r23 = pow(x*x + y*y, 2.0/3.0);
   grad(0) = 2.0/3.0*x*sin(theta23)/(r23) - 2.0/3.0*y*cos(theta23)/(r23);
   grad(1) = 2.0/3.0*y*sin(theta23)/(r23) + 2.0/3.0*x*cos(theta23)/(r23);
}

// Compute the energy norm of the true error in each element and
// returns the global energy norm of the error
double CalculateH10Error(Coefficient &alpha,
                         GridFunction *sol,
                         VectorCoefficient *exgrad,
                         Vector &elemError,
                         int intOrder = 30)
{
   const FiniteElementSpace *fes = sol->FESpace();
   Mesh* mesh = fes->GetMesh();

   Vector e_grad, a_grad, el_dofs;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   const FiniteElement *fe;
   ElementTransformation *transf;

   int dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   Jinv.SetSize(dim);

   double error = 0.0;
   elemError.SetSize(mesh->GetNE());

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int fdof = fe->GetDof();
      transf = mesh->GetElementTransformation(i);
      el_dofs.SetSize(fdof);
      dshape.SetSize(fdof, dim);
      dshapet.SetSize(fdof, dim);

      fes->GetElementVDofs(i, vdofs);
      for (int k = 0; k < fdof; k++)
         if (vdofs[k] >= 0)
         {
            el_dofs(k) =  (*sol)(vdofs[k]);
         }
         else
         {
            el_dofs(k) = -(*sol)(-1-vdofs[k]);
         }

      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intOrder);

      double elErr = 0.0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {//numerical integration
         const IntegrationPoint &ip = ir.IntPoint(j);
         fe->CalcDShape(ip, dshape);
         transf->SetIntPoint(&ip);
         exgrad->Eval(e_grad, *transf, ip);
         CalcInverse(transf->Jacobian(), Jinv);
         Mult(dshape, Jinv, dshapet);
         dshapet.MultTranspose(el_dofs, a_grad);
         e_grad -= a_grad;
         elErr += (ip.weight * transf->Weight() * (e_grad * e_grad));
         elErr *= alpha.Eval(*transf, ip);
      }

      error += elErr;
      elemError[i] = sqrt(fabs(elErr));
   }

   if (error < 0.0) { return -sqrt(-error); }
   return sqrt(error);
}

//vector coefficient class for the numerical flux (A \nabla u_h)
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
      uh->GetGradient(T, V);
      V *= A->Eval(T, ip);
   }
};


//recover flux for conforming triangular mesh
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
         recovered_flux[j] = lvals[lindex] * weight- rvals[rindex] * (1-weight);//solution to 4.3
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
   const mfem::NCMesh::NCList &list=
      mesh->ncmesh->GetEdgeList();
   int e1, e2;
   Array<int> fdofs;
   Vector fvals;

   for ( int i = 0; i< list.conforming.size(); i++)
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
         recovered_flux[j] = lvals[lindex] * weight - rvals[rindex] * (1-weight);//solution to (4.3)
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
      //Projection vector maps the index of global whole DOF to the global index of true DOF.
      //Note that when a DOF is a slave DOF, then its global whold DOF index will be mapped to 0.
      P[Ri[i]] = i;
   }

   for (int mi = 0; mi< list.masters.size(); mi++)
   {
      const NCMesh::Master &master = list.masters[mi];
      fes->GetFaceDofs(master.index, master_dofs);
      if (!master_dofs.Size()) { continue; }

      //get local index of the face dofs
      int mindex = master.element;
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
      a.ComputeElementMatrix(mindex,mmaster);
      S.SetSize(nfdof);

      for (int i=0; i< nfdof; i++)
         for (int j=0; j< nfdof; j++)
         {
            S(i,j) = mmaster.Elem(master_lindex[i], master_lindex[j]); //S = S^- in (4.10), which is the local mass matrix on master face
         }

      masterfvals.SetSize(nfdof);
      mastervals.SetSize(nedof);
      fes->GetFE(mindex)->Project(vcoeff,*fes->GetElementTransformation(mindex),
                                  mastervals);

      for (int i = 0; i < nfdof; i++)//masterfvals = \bsigma_E^- in (4.4)
      {
         masterfvals[i] = mastervals[master_lindex[i]];
      }

      Ai.SetSize(nfdof);//Ai = A in (4.12).
      double ele_size = fes->GetMesh()->GetElementSize(mindex);
      ele_size *=
         ele_size; //Here assume that each element is square, the area of the face needs to be used instead.
      S *= 1/ele_size;
      Ai = S;

      b.SetSize(nfdof);//b will store the right hand side in (4.12)
      S.Mult(masterfvals,b);

      int m_sign = 1;
      if (master_dofs[0] < 0)
      {
         m_sign = -1;
      }
      int sign;//note that the sign of DOF on master and slaves can be arbitrary. However, the global depende matrix only maps
               //the postive DOF to DOF, thus in order to get local depence matrix defined in (4.11), we need to track the sign.

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

         //int ndf = master_dofs.Size();
         dep.SetSize(nfdof);

         for (int j = 0; j< nfdof; j++)
         {
            int ji = slave_dofs[j]<0 ? -slave_dofs[j]-1:slave_dofs[j];
            for (int k = 0; k< nfdof; k++)
            {
               int ki = master_dofs[k]<0 ? -master_dofs[k]-1: master_dofs[k];
               dep(j,k) = cP->Elem(ji, P[ki]);//positive index to positve index
               S(j,k) = mslave.Elem(slave_lindex[j], slave_lindex[k]);//S_i^+ in (4.9)
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

         dep *= sign;//now dep is the local dependence matrix in (4.10)

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
//residual = 0, only compute flux jump; residual =1, only compute element residual;
//residual =2, compute both.
double ComputeEnergynormError(BilinearForm &a,
                              VectorCoefficient &flux_coeff,
                              Coefficient &alpha,
                              GridFunction &flux,
                              Vector &eta_Ks,
                              int residual)
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




double ImprovedZZErrorEstimation(GridFunction &flux,
                                 VectorCoefficient &flux_coeff,
                                 Vector &eta_Ks)
{
   FiniteElementSpace *flux_fes = flux.FESpace();
   BilinearForm *a=new BilinearForm(flux_fes);
   ConstantCoefficient one(1.0);
   a->AddDomainIntegrator(new VectorFEMassIntegrator(one));
   a->ComputeElementMatrices();

   double eta = 0;
   eta_Ks = 0.;

   if (flux_fes->GetMesh()->Conforming())
   {
      RecoverFluxforConformingMesh(*a,flux_coeff,flux);
      eta = ComputeEnergynormError(*a,flux_coeff,one, flux, eta_Ks,0);
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

      RecoverFluxatConformingFaces(*a, flux_coeff,
                                   flux);//if face i is not a boundary face
      RecoverFluxatMasterFaces(*a, flux_coeff,
                               flux);//if face i is not a boundary face
      eta = ComputeEnergynormError(*a,flux_coeff,one, flux, eta_Ks,0);
   }

   return eta;
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

   //   // 3. Since a NURBS mesh can currently only be refined uniformly, we need to
   //   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //   //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   //   if (mesh.NURBSext)
   //   {
   //      for (int i = 0; i < 3; i++)
   //      {
   //         mesh.UniformRefinement();
   //      }
   //      mesh.SetCurvature(2);
   //   }

   // 4. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 5. As in Example 1, we set up bilinear and linear forms corresponding to
   //    the Laplace problem -\Delta u = 0. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

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

   //   ofstream myfile;
   ofstream myfile;
   if (order==1)
   {
      myfile.open ("lshape_error_1.txt");
   }
   else if (order==2)
   {
      myfile.open ("lshape_error_2.txt");
   }
   else if (order==3)
   {
      myfile.open ("lshape_error_3.txt");
   }
   else if (order==4)
   {
      myfile.open ("lshape_error_4.txt");
   }
   else
   {
      myfile.open ("lshape_error_.txt");
   }


   const int max_dofs = 5000;
   for (int it = 0; it < 200 ; it++)
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
      //b.Print();

      // 11. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      FunctionCoefficient exsol(lshape_exsol);
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
         sol_sock << "solution\n" << mesh << x << "pause\n" << flush;
      }

      if (cdofs > max_dofs)
      {
         break;
      }

      // 16. Estimate element errors using the local improved Zienkiewicz-Zhu error estimator.

      Vector errors, true_errors;
      errors.SetSize(mesh.GetNE());
      true_errors.SetSize(mesh.GetNE());

      int geom_type = mesh.GetElementBaseGeometry(0);
      bool triangles = (geom_type == Geometry::TRIANGLE);
      int RT_order = triangles ? order-1 : order;

      RT_FECollection smooth_flux_fec(RT_order, dim);
      FiniteElementSpace smooth_flux_fes(&mesh, &smooth_flux_fec);
      GridFunction flux(&smooth_flux_fes);
      GradientGridFunctionCoeffiecent flux_coeff(x, one);
      double eta = ImprovedZZErrorEstimation(flux,flux_coeff, errors);
      cout << it << " "<< cdofs << " " << eta << endl;

      int intorder = triangles ? 22 : 30;
      VectorFunctionCoefficient exgrad(dim, lshape_exgrad);//2 is the dimension
      double true_error = CalculateH10Error(one, &x, &exgrad, true_errors, intorder);

      cout << it << " "<< cdofs << " " << true_error << endl;
      myfile << it << " "<< cdofs<< " " << true_error << " " << eta << "\n";


      // 17. Make a list of elements whose error is larger than a fraction of
      //     the maximum element error. These elements will be refined.
      Array<Refinement> ref_list;
      const double frac = 0.70;
      double threshold = frac * errors.Max();
      for (int i = 0; i < errors.Size(); i++)
      {
         if (errors[i] >= threshold)
         {
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
