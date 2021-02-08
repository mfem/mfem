//                       MFEM Example 28
//
// Compile with: make ex28
//
// Sample runs:  ex28
//               ex28 --visit-datafiles
//               ex28 --order 2
//
// Description:  Demonstrates a sliding boundary condition in an elasticity
//               problem. A trapezoid, roughly as pictured below, is pushed
//               from the right into a rigid notch. Normal displacement is
//               restricted, but tangential movement is allowed, so the
//               trapezoid compresses into the notch.
//
//                                       /-------+
//               normal constrained --->/        | <--- boundary force (2)
//               boundary (4)          /---------+
//                                          ^
//                                          |
//                                normal constrained boundary (1)
//
//               This example demonstrates the use of the ConstrainedSolver
//               framework.
//
//               We recommend viewing Example 2 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** @brief Build a matrix constraining normal components to zero.

    Given a vector space fespace, and the array constrained_att that
    includes the boundary *attributes* that are constrained to have normal
    component zero, this returns a SparseMatrix representing the
    constraints that need to be imposed.

    Each row of the returned matrix corresponds to a node that is
    constrained. The rows are arranged in (contiguous) blocks corresponding
    to the actual constraint; in 3D, a one-row constraint means the node
    is free to move along a plane, a two-row constraint means it is free
    to move along a line (eg the intersection of two normal-constrained
    planes), and a three-row constraint is fully constrained (equivalent
    to MFEM's usual essential boundary conditions).

    The lagrange_rowstarts array is filled in to describe the structure of
    these constraints, so that constraint k is encoded in rows
    lagrange_rowstarts[k] to lagrange_rowstarts[k + 1] - 1, inclusive,
    of the returned matrix.

    When two attributes intersect, this version will combine constraints,
    so in 2D the point at the intersection is fully constrained (ie,
    fixed in both directions). This is the wrong thing to do if the
    two boundaries are (close to) parallel at that point.

    @param[in] fespace              A vector finite element space
    @param[in] constrained_att      Boundary attributes to constrain
    @param[out] lagrange_rowstarts  The rowstarts for separately
                                    eliminated constraints, possible
                                    input to EliminationCGSolver

    @return a constraint matrix
*/
SparseMatrix * BuildNormalConstraints(FiniteElementSpace& fespace,
                                      Array<int>& constrained_att,
                                      Array<int>& lagrange_rowstarts)
{
   int dim = fespace.GetVDim();

   // mapping from dofs (colums of constraint matrix) to constraints
   // (rows of constraint matrix)
   // the indexing is by tdof, but we only bother with one tdof per node
   std::map<int, int> dof_constraint;
   // constraints[j] is a map from attribute to row number
   std::vector<std::map<int, int> > constraints;
   int n_constraints = 0;
   int n_rows = 0;
   for (int att : constrained_att)
   {
      std::set<int> constrained_tdofs;
      for (int i = 0; i < fespace.GetNBE(); ++i)
      {
         if (fespace.GetBdrAttribute(i) == att)
         {
            Array<int> dofs;
            fespace.GetBdrElementDofs(i, dofs);
            for (auto k : dofs)
            {
               int vdof = fespace.DofToVDof(k, 0);
               if (vdof >= 0) { constrained_tdofs.insert(vdof); }
            }
         }
      }
      for (auto k : constrained_tdofs)
      {
         auto it = dof_constraint.find(k);
         if (it == dof_constraint.end())
         {
            dof_constraint[k] = n_constraints++;
            constraints.emplace_back();
            constraints.back()[att] = n_rows++;
         }
         else
         {
            constraints[it->second][att] = n_rows++;
         }
      }
   }

   // reorder so constraints eliminated together are grouped
   // together in rows
   std::map<int, int> reorder_rows;
   int new_row = 0;
   lagrange_rowstarts.DeleteAll();
   lagrange_rowstarts.Append(0);
   for (auto& it : dof_constraint)
   {
      int constraint_index = it.second;
      bool nconstraint = false;
      for (auto& att_it : constraints[constraint_index])
      {
         auto rrit = reorder_rows.find(att_it.second);
         if (rrit == reorder_rows.end())
         {
            nconstraint = true;
            reorder_rows[att_it.second] = new_row++;
         }
      }
      if (nconstraint) { lagrange_rowstarts.Append(new_row); }
   }
   MFEM_VERIFY(new_row == n_rows, "Remapping failed!");
   for (auto& constraint_map : constraints)
   {
      for (auto& it : constraint_map)
      {
         it.second = reorder_rows[it.second];
      }
   }

   SparseMatrix * out = new SparseMatrix(n_rows, fespace.GetTrueVSize());

   // fill in constraint matrix with normal vector information
   Vector nor(dim);
   for (int i = 0; i < fespace.GetNBE(); ++i)
   {
      int att = fespace.GetBdrAttribute(i);
      if (constrained_att.FindSorted(att) != -1)
      {
         ElementTransformation * Tr = fespace.GetBdrElementTransformation(i);
         const FiniteElement * fe = fespace.GetBE(i);
         const IntegrationRule& nodes = fe->GetNodes();

         Array<int> dofs;
         fespace.GetBdrElementDofs(i, dofs);
         MFEM_VERIFY(dofs.Size() == nodes.Size(),
                     "Something wrong in finite element space!");

         for (int j = 0; j < dofs.Size(); ++j)
         {
            Tr->SetIntPoint(&nodes[j]);
            // the normal returned in the next line is scaled by h, which
            // is probably what we want in this application
            CalcOrtho(Tr->Jacobian(), nor);

            int k = dofs[j];
            int vdof = fespace.DofToVDof(k, 0);
            if (vdof >= 0)
            {
               int constraint = dof_constraint[vdof];
               int row = constraints[constraint][att];
               for (int d = 0; d < dim; ++d)
               {
                  int inner_vdof = fespace.DofToVDof(k, d);
                  // an arguably better algorithm does some kind of average
                  // instead of just overwriting when two elements (with
                  // potentially different normals) share a node.
                  out->Set(row, inner_vdof, nor[d]);
               }
            }
         }
      }
   }
   out->Finalize();

   return out;
}


Mesh * build_trapezoid_mesh(double offset)
{
   MFEM_VERIFY(offset < 0.9, "offset is too large!");

   const int dimension = 2;
   const int nvt = 4; // vertices
   const int nbe = 4; // num boundary elements
   Mesh * mesh = new Mesh(dimension, nvt, 1, nbe);

   // vertices
   double vc[dimension];
   vc[0] = 0.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 0.0;
   mesh->AddVertex(vc);
   vc[0] = offset; vc[1] = 1.0;
   mesh->AddVertex(vc);
   vc[0] = 1.0; vc[1] = 1.0;
   mesh->AddVertex(vc);

   // element
   Array<int> vert(4);
   vert[0] = 0; vert[1] = 1; vert[2] = 3; vert[3] = 2;
   mesh->AddQuad(vert, 1);

   // boundary
   Array<int> sv(2);
   sv[0] = 0; sv[1] = 1;
   mesh->AddBdrSegment(sv, 1);
   sv[0] = 1; sv[1] = 3;
   mesh->AddBdrSegment(sv, 2);
   sv[0] = 2; sv[1] = 3;
   mesh->AddBdrSegment(sv, 3);
   sv[0] = 0; sv[1] = 2;
   mesh->AddBdrSegment(sv, 4);

   mesh->FinalizeQuadMesh(1, 0, true);

   return mesh;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int order = 1;
   bool visualization = 1;
   double offset = 0.3;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&offset, "--offset", "--offset",
                  "How much to offset the trapezoid.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh *mesh = build_trapezoid_mesh(offset);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 1,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor.
   FiniteElementCollection *fec = new H1_FECollection(order, dim);
   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec, dim);
   cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
        << endl;

   // 5. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, there are no essential boundary
   //    conditions in the usual sense, but we leave the machinery here for
   //    users to modify if they wish.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system. In this case, b_i equals the boundary integral
   //    of f*phi_i where f represents a "push" force on the right side of the
   //    trapezoid.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector push_force(mesh->bdr_attributes.Max());
      push_force = 0.0;
      push_force(1) = -5.0e-2; // index 1 attribute 2
      f.Set(0, new PWConstCoefficient(push_force));
   }

   // 8. Set up constraint matrix to constrain normal displacement (but
   //    allow tangential displacement) on specified boundaries.
   Array<int> constraint_atts(2);
   constraint_atts[0] = 1;  // attribute 1 bottom
   constraint_atts[1] = 4;  // attribute 4 left side

   LinearForm *b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "r.h.s. ... " << flush;
   b->Assemble();

   // 9. Define the solution vector x as a finite element grid function
   //    corresponding to fespace.
   GridFunction x(fespace);
   x = 0.0;

   // 10. Set up the bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu. We use constant coefficients,
   //     but see ex2 for how to set up piecewise constant coefficients based
   //     on attribute.
   Vector lambda(mesh->attributes.Max());
   lambda = 1.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(mesh->attributes.Max());
   mu = 1.0;
   PWConstCoefficient mu_func(mu);

   BilinearForm *a = new BilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

   // 11. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   cout << "matrix ... " << flush;
   a->Assemble();

   SparseMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   cout << "done." << endl;
   cout << "Size of linear system: " << A.Height() << endl;

   // 12. Define and apply an iterative solver for the constrained system
   //     in saddle-point form with a Gauss-Seidel smoother for the
   //     displacement block.
   Array<int> lagrange_rowstarts;
   SparseMatrix* local_constraints =
      BuildNormalConstraints(*fespace, constraint_atts, lagrange_rowstarts);
   GSSmoother M(A);
   SchurConstrainedSolver * solver =
      new SchurConstrainedSolver(A, *local_constraints, M);
   solver->SetRelTol(1e-6);
   solver->SetMaxIter(1500);
   solver->SetPrintLevel(1);
   solver->PrimalMult(B, X);

   // 13. Recover the solution as a finite element grid function. Move the
   //     mesh to reflect the displacement of the elastic body being
   //     simulated, for purposes of output.
   a->RecoverFEMSolution(X, *b, x);
   mesh->SetNodalFESpace(fespace);
   GridFunction *nodes = mesh->GetNodes();
   *nodes += x;

   // 14. Save the refined mesh and the solution in VisIt format.
   if (visit)
   {
      VisItDataCollection visit_dc("ex28", mesh);
      visit_dc.SetLevelsOfDetail(4);
      visit_dc.RegisterField("displacement", &x);
      visit_dc.Save();
   }

   // 15. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {
      x *= -1; // sign convention for GLVis displacements
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 17. Free the used memory.
   delete local_constraints;
   delete solver;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}
