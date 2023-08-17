//                                Contact example
//
// Compile with: make contact
//
// Sample runs:  ./contact -m1 block1.mesh -m2 block2.mesh -at "5 6 7 8"
// Sample runs:  ./contact -m1 block1_d.mesh -m2 block2_d.mesh -at "5 6 7 8"

#include <fstream>
#include <iostream>
#include <array>

#include "mfem.hpp"
#include "problems.hpp"
#include "IPsolver.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
  Mpi::Init(argc, argv);
  Hypre::Init();
  int linSolver = 2;
  int maxIPMiters = 30;
  bool iAmRoot = true;
  int ref_levels = 0;

  OptionsParser args(argc, argv);
  args.AddOption(&linSolver, "-linSolver", "--linearSolver", \
       "IP-Newton linear system solution strategy.");
  args.AddOption(&maxIPMiters, "-IPMiters", "--IPMiters",\
		  "Maximum number of IPM iterations");
  args.AddOption(&ref_levels, "-r", "--mesh_refinement", \
		  "Mesh Refinement");

   
  args.Parse();
  if(!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  else
  {
    if( iAmRoot )
    {
      args.PrintOptions(cout);
    }
  }
	
  // Create an instance of the nlp
  ExContactBlockTL * contact = new ExContactBlockTL(ref_levels);
  int ndofs = contact->GetDimD();
  int nconstraints = contact->GetDimS();
  std::ofstream problemDimStream;
  problemDimStream.open("problemDim.dat", ios::out | ios::trunc);
  problemDimStream << ndofs << endl;
  problemDimStream.close();
  std::ofstream problemDimConstraintsStream;
  problemDimConstraintsStream.open("problemDimConstraints.dat", ios::out | ios::trunc);
  problemDimConstraintsStream << nconstraints << endl;
  problemDimConstraintsStream.close();
  
  // set up a QP-problem
  // E(d) = 1 / 2 d^T K d + f^T d
  // g(d) = J d + g0
  // where K, J, f and g0 are evaluated at d0 (a valid configuration)
  
  // to do: seems more appropriate to evaluate at a valid configuration...
  //        that is one where the Dirichlet conditions hold... need to pull
  //        this data from contactBlockTL...
  Vector d0(ndofs); d0 = 0.0;
  Array<int> DirichletDofs = contact->GetDirichletDofs();
  Array<double> DirichletVals = contact->GetDirichletVals();
  SparseMatrix *K;
  Vector f(ndofs); f = 0.0;
  contact->DdE(d0, f); K = contact->DddE(d0);
  for(int i = 0; i < DirichletDofs.Size(); i++)
  {
    d0(DirichletDofs[i]) = DirichletVals[i];
  }
  SparseMatrix *J;
  Vector g0(nconstraints); g0 = 0.0;
  J = contact->Ddg(d0); contact->g(d0, g0);
  Vector temp(nconstraints);
  J->Mult(d0, temp);
  g0.Add(-1.0, temp);

  // check which rows of the Jacobian are zero!
  Vector ei(nconstraints); ei = 0.0;
  Vector JTei(ndofs); JTei = 0.0;

  double normJTei;

  int reduced_nconstraints = 0; // find actual number of constraints
  
  
  Array<int> nonZeroRows;
  for(int i = 0; i < nconstraints; i++)
  {
    ei(i) = 1.0;
    J->MultTranspose(ei, JTei);
    // nullify contributions from Dirichlet constrined dofs
    for(int j = 0; j < DirichletDofs.Size(); j++)
    {
      JTei(DirichletDofs[j]) = 0.0;
    }
    normJTei = sqrt(InnerProduct(JTei, JTei));
    if (normJTei > 1.e-12)
    {
      reduced_nconstraints += 1;
      nonZeroRows.Append(i);
    }
    ei(i) = 0.0;
  }
  cout << "number of linearized constraints = " << reduced_nconstraints << endl; // 9 constraints 
  
  // remove zero rows of the gap function Jacobian and corresponding gap function entries
  SparseMatrix * Jreduced = new SparseMatrix(reduced_nconstraints, ndofs);
  Vector g0reduced(reduced_nconstraints); g0reduced = 0.0; 
  
  
  for(int i = 0; i < reduced_nconstraints; i++)
  {
    Array<int> col_tmp;
    Vector v_tmp; v_tmp = 0.0;
    J->GetRow(nonZeroRows[i], col_tmp, v_tmp);
    
    /* obtain subset of columns of the given nonZero Jacobian row that are not Dirichlet constrained */
    bool freeDof;
    Array<int> loc_indicies;
    for(int j = 0; j < col_tmp.Size(); j++)
    {
      freeDof = true;
      for(int k = 0; k < DirichletDofs.Size(); k++)
      {
        if(col_tmp[j] == DirichletDofs[k])
        {
          freeDof = false;
        }
      }
      if(freeDof)
      {
        loc_indicies.Append(j);
      }
    }

    Array<int> col_tmp_reduced(loc_indicies.Size());
    Vector v_tmp_reduced(loc_indicies.Size());
    for(int j = 0; j < loc_indicies.Size(); j++)
    {
      col_tmp_reduced[j] = col_tmp[loc_indicies[j]];
      v_tmp_reduced(j)   = v_tmp(loc_indicies[j]);
    }

    Jreduced->SetRow(i, col_tmp_reduced, v_tmp_reduced);
    g0reduced(i) = g0(nonZeroRows[i]);
  }


  QPContactProblem *QPContact = new QPContactProblem(*K, *Jreduced, f, g0reduced);
  
  Mesh * mesh1 = new Mesh("meshes/block1.mesh", 1, 1);
  Mesh * mesh2 = new Mesh("meshes/rotatedblock2.mesh", 1, 1);
  for(int i = 0; i < ref_levels; i++)
  {
     mesh1->UniformRefinement();
     mesh2->UniformRefinement(); 
  }
  
  int numMeshes = 2;
  Mesh *meshArray[numMeshes];
  meshArray[0] = mesh1;
  meshArray[1] = mesh2;
  Mesh mesh(meshArray, numMeshes);
  
  ParMesh pmesh(MPI_COMM_WORLD, mesh);  
  H1_FECollection fec(1, mesh.Dimension());
  ParFiniteElementSpace fespace(&pmesh, &fec, mesh.Dimension(), Ordering::byVDIM);

  InteriorPointSolver * QPContactOptimizer = new InteriorPointSolver(QPContact, &fespace);
  QPContactOptimizer->SetTol(1.e-6);
  QPContactOptimizer->SetLinearSolver(linSolver);
  QPContactOptimizer->SetMaxIter(50);
  Vector x0(ndofs); x0 = 0.0;
  for(int i = 0; i < DirichletDofs.Size(); i++)
  {
    x0(DirichletDofs[i]) = DirichletVals[i];
  }
  Vector xf(ndofs); xf = 0.0;
  QPContactOptimizer->Mult(x0, xf);
  
  double Einitial = QPContact->E(x0);
  double Efinal = QPContact->E(xf);
  cout << "Energy objective at initial point = " << Einitial << endl;
  cout << "Energy objective at QP optimizer = " << Efinal << endl;
  QPContactOptimizer->GetCGIterNumbers().Print(mfem::out, 20);
  MFEM_VERIFY(QPContactOptimizer->GetConverged(), "Interior point solver did not converge.");
  
  
  //Mesh * mesh1 = new Mesh("meshes/block1.mesh", 1, 1);
  //Mesh * mesh2 = new Mesh("meshes/rotatedblock2.mesh", 1, 1);
  //for(int i = 0; i < ref_levels; i++)
  //{
  //   mesh1->UniformRefinement();
  //   mesh2->UniformRefinement(); 
  //}
  //int gdim = mesh1->Dimension();
  //FiniteElementCollection * fec = new H1_FECollection(1, gdim);
  //FiniteElementSpace * fespace1 = new FiniteElementSpace(mesh1, fec, gdim, Ordering::byVDIM);
  //FiniteElementSpace * fespace2 = new FiniteElementSpace(mesh2, fec, gdim, Ordering::byVDIM);
  // 
  //GridFunction x1_gf(fespace1);
  //GridFunction x2_gf(fespace2);

  //int ndof1 = fespace1->GetTrueVSize();
  //int ndof2 = fespace2->GetTrueVSize();
  //int ndof  = ndof1 + ndof2;
  //for(int i = 0; i < ndof1; i++)
  //{
  //  x1_gf(i) = xf(i);
  //}
  //for(int i = ndof1; i < ndof; i++)
  //{
  //  x2_gf(i - ndof1) = xf(i);
  //}

  //mesh1->SetNodalFESpace(fespace1);
  //mesh2->SetNodalFESpace(fespace2);
  //GridFunction *nodes1 = mesh1->GetNodes();
  //GridFunction *nodes2 = mesh2->GetNodes();

  //{
  //  *nodes1 += x1_gf;
  //  *nodes2 += x2_gf;
  //}
  //

  //ParaViewDataCollection paraview_dc1("QPContactBody1", mesh1);
  //paraview_dc1.SetPrefixPath("ParaView");
  //paraview_dc1.SetLevelsOfDetail(1);
  //paraview_dc1.SetDataFormat(VTKFormat::BINARY);
  //paraview_dc1.SetHighOrderOutput(true);
  //paraview_dc1.SetCycle(0);
  //paraview_dc1.SetTime(0.0);
  //paraview_dc1.RegisterField("Body1", &x1_gf);
  //paraview_dc1.Save();
  //
  //ParaViewDataCollection paraview_dc2("QPContactBody2", mesh2);
  //paraview_dc2.SetPrefixPath("ParaView");
  //paraview_dc2.SetLevelsOfDetail(1);
  //paraview_dc2.SetDataFormat(VTKFormat::BINARY);
  //paraview_dc2.SetHighOrderOutput(true);
  //paraview_dc2.SetCycle(0);
  //paraview_dc2.SetTime(0.0);
  //paraview_dc2.RegisterField("Body2", &x2_gf);
  //paraview_dc2.Save();

  //delete fespace1;
  //delete fespace2;
  //delete fec;
  //delete mesh1;
  //delete mesh2;
  
  delete QPContact;
  delete QPContactOptimizer;
  
  delete K;
  delete J;
  delete Jreduced;
  delete contact;
  return 0;
}
