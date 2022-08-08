//                       MFEM Example 0 - Parallel Version
//
// Compile with: make ex0p
//
// Sample runs:  mpirun -np 4 ex0p
//               mpirun -np 4 ex0p -m ../data/fichera.mesh
//               mpirun -np 4 ex0p -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic parallel usage of
//              MFEM to define a simple finite element discretization of the
//              Laplace problem -Delta u = 1 with zero Dirichlet boundary
//              conditions. General 2D/3D serial mesh files and finite element
//              polynomial degrees can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ThermalInterfaceHeatSourceIntegrator : public mfem::LinearFormIntegrator
{
public:
    ThermalInterfaceHeatSourceIntegrator(const ParMesh *pMesh_, mfem::PWConstCoefficient &heatSource, mfem::Array<int> &elem_marker, int oa=2, int ob=2); 
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &T, mfem::Vector &elvect);
    void AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &T, mfem::Vector &elvect);
    void AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
                                    mfem::FaceElementTransformations &T, mfem::Vector &elvect);

protected:
    mfem::PWConstCoefficient Qface_;
    mfem::Array<int> elem_marker_;
    int NEproc_, par_shared_face_count_, oa_, ob_;
};

ThermalInterfaceHeatSourceIntegrator::ThermalInterfaceHeatSourceIntegrator(
   const ParMesh *pMesh_,
   mfem::PWConstCoefficient &heatSource, 
   mfem::Array<int> &elem_marker, int oa, int ob)
    : Qface_(heatSource), elem_marker_(elem_marker), NEproc_(pMesh_->GetNE()), 
      par_shared_face_count_(0), oa_(oa), ob_(ob)
{
}

void ThermalInterfaceHeatSourceIntegrator::AssembleRHSElementVect(
   const mfem::FiniteElement & /*el*/, mfem::ElementTransformation & /*Tr*/, mfem::Vector & /*elvect*/)
{
   mfem_error("ThermalInterfaceHeatSourceIntegrator::AssembleRHSElementVect");
}

void ThermalInterfaceHeatSourceIntegrator::AssembleRHSElementVect(
   const mfem::FiniteElement &el, mfem::FaceElementTransformations &Tr, mfem::Vector &elvect)
{
   AssembleRHSElementVect(el, el, Tr, elvect);
}

void ThermalInterfaceHeatSourceIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
        mfem::FaceElementTransformations &Tr, mfem::Vector &elvect)
{
    int dim, ndof1, ndof2, ndof, ndoftotal;
    double w;
    mfem::Vector temp_elvect1, temp_elvect2;

    // grab sizes
    dim = el1.GetDim();
    ndof1 = el1.GetDof();
    ndof2 = el2.GetDof();
    ndoftotal = ndof1 + ndof2;

    if (Tr.Elem2No >= NEproc_ ||
        Tr.ElementType == ElementTransformation::BDR_FACE)
    {
        ndoftotal = ndof1;
    }

    // output vector
    elvect.SetSize(ndoftotal);
    elvect = 0.0;

    int elem1 = Tr.Elem1No,
        elem2 = Tr.Elem2No,
        marker1 = (elem_marker_)[elem1];

    int marker2;
    if (Tr.Elem2No >= NEproc_)
    {
        marker2 = (elem_marker_)[NEproc_+par_shared_face_count_];
        par_shared_face_count_++;
    }
    else if (Tr.ElementType == mfem::ElementTransformation::BDR_FACE)
    {
        marker2 = marker1;
    }
    else
    {
        marker2 = (elem_marker_)[elem2];
    }

    // Only integrate interfaces
    if ( marker1==marker2 )
    {
      return;
    }

    // set integration rule
    const IntegrationRule *ir = &IntRules.Get(el1.GetGeomType(), oa_*el1.GetOrder()+ob_);

    Vector shape1(ndof1);
    Vector shape2(ndof2);
    Vector wrk1 = shape1;
    Vector wrk2 = shape2;
    temp_elvect1.SetDataAndSize(elvect.GetData(), ndof1);
    temp_elvect2.SetDataAndSize(elvect.GetData()+ndof1, ndof2);

    for (int p = 0; p < ir->GetNPoints(); p++)
    {
       const IntegrationPoint &ip = ir->IntPoint(p);

        // Access the neighboring element's integration point
        const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
        const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

         el1.CalcShape(eip1, shape1);
         w = ip.weight * Qface_.Eval(Tr, ip);
         wrk1.Set(w, shape1);
         temp_elvect1.Add(1., wrk1);

         // if (Tr.ElementType != mfem::ElementTransformation::BDR_FACE)
         if (!(Tr.Elem2No >= NEproc_ ||
            Tr.ElementType == ElementTransformation::BDR_FACE))
         {
            el2.CalcShape(eip2, shape2);
            w = ip.weight * Qface_.Eval(Tr, ip);
            wrk2.Set(w, shape2);
            temp_elvect2.Add(1., wrk2);
         }
    }
}

///---------------------------------------------------

class InterfaceFaceMarker
{
protected:
   ParMesh &pMesh_;                    // Mesh whose elements have to be marked.
   ParFiniteElementSpace *pfes_sltn;  // FESpace associated with the solution.

   // Marking of face dofs by using an averaged continuous GridFunction.
   const bool func_dof_marking = true;

public:
   InterfaceFaceMarker(ParMesh &pm, ParFiniteElementSpace &pfes)
      : pMesh_(pm), pfes_sltn(&pfes)
   { }

   /// Mark elements along the interface on both sides
   /// Assumes the ExchangeFaceNbrData() has been called for pMesh_, marker_gf.
   void MarkElements(const ParGridFunction &marker_gf, Array<int> &elem_marker);

   /// List dofs associated with the interface.
   void InterfaceFaceDofs(const Array<int> &elem_marker,
                            Array<int> &sface_dof_list) const;
};

void InterfaceFaceMarker::MarkElements(const ParGridFunction &marker_gf, 
                                     ::mfem::Array<int> &elem_marker)
{
   elem_marker.SetSize(pMesh_.GetNE() + pMesh_.GetNSharedFaces());

   IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
   
   Vector vals;
   // Check elements on the current MPI rank
   for (int i = 0; i < pMesh_.GetNE(); i++)
   {
      const IntegrationRule &ir = pfes_sltn->GetFE(i)->GetNodes();
      marker_gf.GetValues(i, ir, vals);

      bool NegLSReg(vals.Sum()<0.0 ? true : false);

      if (NegLSReg) 
      {
         elem_marker[i] = 0;
      }
      else
      {
          elem_marker[i] = 1;
      }
   }

   // Check neighbors on the adjacent MPI rank
   for (int i = pMesh_.GetNE(); i < pMesh_.GetNE()+pMesh_.GetNSharedFaces(); i++)
   {
      int shared_fnum = i-pMesh_.GetNE();
      FaceElementTransformations *tr =
         pMesh_.GetSharedFaceTransformations(shared_fnum);
      int Elem2NbrNo = tr->Elem2No - pMesh_.GetNE();

      ElementTransformation *eltr =
         pMesh_.GetFaceNbrElementTransformation(Elem2NbrNo);
      const IntegrationRule &ir =
         IntRulesLo.Get(pMesh_.GetElementBaseGeometry(0), 4*eltr->OrderJ());

      const int nip = ir.GetNPoints();
      vals.SetSize(nip);
      
      int count = 0;
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         vals(j) = marker_gf.GetValue(tr->Elem2No, ip);
      }

      bool NegLSReg(vals.Sum()< 0.0 ? true : false);

      if (NegLSReg)
      {
         elem_marker[i] = 0;
      }
      else
      {
          elem_marker[i] = 1;
      }
   }
}

void InterfaceFaceMarker::InterfaceFaceDofs(const Array<int> &elem_marker,
                                            Array<int> &sface_dof_list) const
{
   sface_dof_list.DeleteAll();

   L2_FECollection mat_coll(0, pMesh_.Dimension());
   ParFiniteElementSpace mat_fes(&pMesh_, &mat_coll);
   ParGridFunction mat(&mat_fes);
   ParGridFunction marker_gf(pfes_sltn);
   for (int i = 0; i < pMesh_.GetNE(); i++)
   {
      // 0 is inside, 1 is outside.
      mat(i) = 0.0;
      if (elem_marker[i] > 0)
      { mat(i) = 1.0; }
   }

   GridFunctionCoefficient coeff_mat(&mat);
   marker_gf.ProjectDiscCoefficient(coeff_mat, GridFunction::ARITHMETIC);

   for (int j = 0; j < marker_gf.Size(); j++)
   {
      if (marker_gf(j) > 0.1 && marker_gf(j) < 0.9)
      {
         sface_dof_list.Append(j);
      }
   }
}

///---------------------------------------------------

int main(int argc, char *argv[])
{
   // Flag for new problem
   bool interfaceSetup(true);

   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command line options.
   const char *mesh_file = "reactorMesh.mesh"; // "../data/star.mesh";
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.ParseCheck();

   // 3. Read the serial mesh from the given mesh file.
   Mesh serial_mesh(mesh_file);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh once in parallel to increase the resolution.
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear(); // the serial mesh is no longer needed
   
   mesh.ExchangeFaceNbrData();
   bool refineMesh(true);
   if (refineMesh)
   {
      mesh.UniformRefinement();
      mesh.UniformRefinement();
   }

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ConstantCoefficient one(1.0);
   ParLinearForm b(&fespace);

   ParGridFunction face_dofs(&fespace);
   face_dofs = 0.0;

   if(!interfaceSetup)
   {
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
   }
   else
   {
      ::mfem::Vector q_Const(mesh.attributes.Max());
      q_Const = 1.0;
      q_Const(1) = 1.0;
      ::mfem::PWConstCoefficient heatSource(q_Const);
      
      mfem::Array<int> elem_marker;
      InterfaceFaceMarker marker(mesh, fespace);

      L2_FECollection mat_coll(0, mesh.Dimension());
      ParFiniteElementSpace mat_fes(&mesh, &mat_coll);
      ParGridFunction lsf(&mat_fes);
      ParGridFunction marker_gf(&mat_fes);

      for (int i = 0; i < mesh.GetNE(); i++)
      {
         int elemAtt = mesh.GetAttribute(i);
         lsf(i) = -1.0;
         if (elemAtt > 1.0)
         { 
            lsf(i) = 1.0;
         }
      }

      GridFunctionCoefficient coeff_lsf(&lsf);
      marker_gf.ProjectDiscCoefficient(coeff_lsf, GridFunction::ARITHMETIC);
      marker_gf.ExchangeFaceNbrData();

      marker.MarkElements(marker_gf, elem_marker);

      // Get a list of dofs associated with shifted boundary (SB) faces.
      mfem::Array<int> sb_dofs; // Array of dofs on SB faces
      marker.InterfaceFaceDofs(elem_marker, sb_dofs); 
      for (int i = 0; i < sb_dofs.Size(); i++)
      {
         face_dofs(sb_dofs[i]) = 1.0;
      }

      b.AddInteriorFaceIntegrator(new ThermalInterfaceHeatSourceIntegrator(&mesh, heatSource, elem_marker));
   }

   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new DiffusionIntegrator);
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // 11. Solve the system using PCG with hypre's BoomerAMG preconditioner.
   HypreBoomerAMG M(A);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   // 12. Recover the solution x as a grid function and save to file. The output
   //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
   a.RecoverFEMSolution(X, b, x);
   // x.Save("sol");
   // mesh.Save("mesh");

   int myid(0);
   MPI_Comm_rank(MPI_COMM_WORLD,&myid);
   L2_FECollection mat_coll(0, mesh.Dimension());
   ParFiniteElementSpace mat_fes(&mesh, &mat_coll);
   ParGridFunction procRank(&mat_fes);
   procRank = myid;

   ParaViewDataCollection vis("InterfaceHeatSourceProb", &mesh);
   vis.RegisterField("temperature", &x);
   vis.RegisterField("interface_dofs", &face_dofs);
   vis.RegisterField("processor_rank", &procRank);
   vis.SetCycle(1);
   vis.SetTime(1);
   vis.Save();

   return 0;
}
