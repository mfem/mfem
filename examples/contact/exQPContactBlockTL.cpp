//                                Quadratic-Programming (QP) Contact example
//
// Compile with: make exQPContactBlockTL
//
// Sample runs:  ./exQPContactBlockTL


#include <fstream>
#include <iostream>
#include <array>

#include "mfem.hpp"
#include "Problems.hpp"
#include "IPsolver.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
  int linSolver = 0;
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
	
  Mesh * mesh1 = new Mesh("block1.mesh", 1, 1);
  Mesh * mesh2 = new Mesh("rotatedblock2.mesh", 1, 1);
  for(int i = 0; i < ref_levels; i++)
  {
     mesh1->UniformRefinement();
     mesh2->UniformRefinement(); 
  }

  // Create an instance of the nlp
  ExContactBlockTL * contact = new ExContactBlockTL(mesh1, mesh2, 1);
  int ndofs = contact->GetDimD();
  int nconstraints = contact->GetDimS();
  
  // set up a QP-problem
  // E(d) = 1 / 2 d^T K d + f^T d
  // g(d) = J d + g0
  // where K, J, f and g0 are evaluated at d0 (a valid configuration)
  
  // to do: seems more appropriate to evaluate at a valid configuration...
  //        that is one where the Dirichlet conditions hold... need to pull
  //        this data from contactBlockTL...
  Vector d0(ndofs); d0 = 0.0;
  Array<int> ess_tdofs1 = contact->GetMesh1DirichletDofs();
  Array<int> ess_tdofs2 = contact->GetMesh2DirichletDofs();
  int sz1 = ess_tdofs1.Size();
  int sz2 = ess_tdofs2.Size();
  Array<int> DirichletDofs(sz1+sz2);
  for (int i = 0; i<sz1; i++)
  {
    DirichletDofs[i] = ess_tdofs1[i];
  }
  for (int i = 0; i<sz2; i++)
  {
    DirichletDofs[i+sz1] = ess_tdofs2[i]+contact->GetVh1().GetTrueVSize();
  }
  GridFunction x1 = contact->GetMesh1GridFunction();
  GridFunction x2 = contact->GetMesh2GridFunction();

  SparseMatrix *K;
  Vector f(ndofs); f = 0.0;
  contact->DdE(d0, f); K = contact->DddE(d0);
  d0.SetVector(x1,0);
  d0.SetVector(x2,x1.Size());
  SparseMatrix *J;
  Vector g0(nconstraints); g0 = 0.0;
  contact->g(d0, g0); J = contact->Ddg(d0);
  Vector temp(nconstraints);
  J->Mult(d0, temp);
  g0.Add(-1.0, temp);

  // check which rows of the Jacobian are zero!
  Vector ei(nconstraints); ei = 0.0;
  Vector JTei(ndofs); JTei = 0.0;

  double normJTei;

  Array<int> nonZeroRows;
  for(int i = 0; i < nconstraints; i++)
  {
    Array<int> col_tmp;
    Vector v_tmp; v_tmp = 0.0;
    J->GetRow(i, col_tmp, v_tmp);
    normJTei = v_tmp.Norml2();
    if (normJTei > 1.e-12)
    {
      nonZeroRows.Append(i);
    }
  }
  mfem::out << J->Height() << " linearized constraints\n";
  mfem::out << nonZeroRows.Size() << " (reduced) linearized constraints\n"; 
  
  // remove zero rows of the gap function Jacobian and corresponding gap function entries
  SparseMatrix * Jreduced = new SparseMatrix(nonZeroRows.Size(), ndofs);
  Vector g0reduced(nonZeroRows.Size()); g0reduced = 0.0; 
  
  
  for(int i = 0; i < nonZeroRows.Size(); i++)
  {
    Array<int> col_tmp;
    Vector v_tmp; v_tmp = 0.0;
    J->GetRow(nonZeroRows[i], col_tmp, v_tmp);
    
    /* obtain subset of columns of the given nonZero Jacobian row that are not Dirichlet constrained */
    bool freeDof;
    Array<int> free_col_indicies;
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
        free_col_indicies.Append(j);
      }
    }

    Array<int> col_tmp_reduced(free_col_indicies.Size());
    Vector v_tmp_reduced(free_col_indicies.Size());
    for(int j = 0; j < free_col_indicies.Size(); j++)
    {
      col_tmp_reduced[j] = col_tmp[free_col_indicies[j]];
      v_tmp_reduced(j)   = v_tmp(free_col_indicies[j]);
    }

    Jreduced->SetRow(i, col_tmp_reduced, v_tmp_reduced);
    g0reduced(i) = g0(nonZeroRows[i]);
  }

  QPOptProblem *QPContact = new QPOptProblem(*K, *Jreduced, f, g0reduced);
  
  InteriorPointSolver * QPContactOptimizer = new InteriorPointSolver(QPContact);
  QPContactOptimizer->SetTol(1.e-6);
  QPContactOptimizer->SetLinearSolver(linSolver);
  Vector x0(ndofs); x0 = 0.0;
  x0.SetVector(x1,0);
  x0.SetVector(x2,x1.Size());

  Vector xf(ndofs); xf = 0.0;
  QPContactOptimizer->Mult(x0, xf);
  
  MFEM_VERIFY(QPContactOptimizer->GetConverged(), "Interior point solver did not converge.");
  double Einitial = QPContact->E(x0);
  double Efinal = QPContact->E(xf);
  cout << "Energy objective at initial point = " << Einitial << endl;
  cout << "Energy objective at QP optimizer = " << Efinal << endl;
 
  
  
  int gdim = mesh1->Dimension();
  FiniteElementCollection * fec = new H1_FECollection(1, gdim);
  FiniteElementSpace * fespace1 = new FiniteElementSpace(mesh1, fec, gdim, Ordering::byVDIM);
  FiniteElementSpace * fespace2 = new FiniteElementSpace(mesh2, fec, gdim, Ordering::byVDIM);
   
  GridFunction x1_gf(fespace1);
  GridFunction x2_gf(fespace2);

  int ndof1 = fespace1->GetTrueVSize();
  int ndof2 = fespace2->GetTrueVSize();
  int ndof  = ndof1 + ndof2;
  for(int i = 0; i < ndof1; i++)
  {
    x1_gf(i) = xf(i);
  }
  for(int i = ndof1; i < ndof; i++)
  {
    x2_gf(i - ndof1) = xf(i);
  }

  mesh1->SetNodalFESpace(fespace1);
  mesh2->SetNodalFESpace(fespace2);
  GridFunction *nodes1 = mesh1->GetNodes();
  GridFunction *nodes2 = mesh2->GetNodes();

  {
    *nodes1 += x1_gf;
    *nodes2 += x2_gf;
  }
  

  ParaViewDataCollection paraview_dc1("QPContactBody1", mesh1);
  paraview_dc1.SetPrefixPath("ParaView");
  paraview_dc1.SetLevelsOfDetail(1);
  paraview_dc1.SetDataFormat(VTKFormat::BINARY);
  paraview_dc1.SetHighOrderOutput(true);
  paraview_dc1.SetCycle(0);
  paraview_dc1.SetTime(0.0);
  paraview_dc1.RegisterField("Body1", &x1_gf);
  paraview_dc1.Save();
  
  ParaViewDataCollection paraview_dc2("QPContactBody2", mesh2);
  paraview_dc2.SetPrefixPath("ParaView");
  paraview_dc2.SetLevelsOfDetail(1);
  paraview_dc2.SetDataFormat(VTKFormat::BINARY);
  paraview_dc2.SetHighOrderOutput(true);
  paraview_dc2.SetCycle(0);
  paraview_dc2.SetTime(0.0);
  paraview_dc2.RegisterField("Body2", &x2_gf);
  paraview_dc2.Save();

  delete fespace1;
  delete fespace2;
  delete fec;
  delete mesh1;
  delete mesh2;
  
  delete QPContact;
  delete QPContactOptimizer;
  
  delete Jreduced;
  delete contact;
  return 0;
}
