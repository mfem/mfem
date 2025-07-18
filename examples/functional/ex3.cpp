//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "remap.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command line options.
   // int dim = 2;
   int order = 3;
   int qorder = 4;
   int ref_levels = 0;
   int optType = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&qorder, "-qo", "--quad-order", "Quadrature order");
   // args.AddOption(&dim, "-d", "--dim", "Mesh dimension (2 or 3)");
   args.AddOption(&ref_levels, "-r", "--refine", "Mesh refinement levels");
   args.AddOption(&optType, "-opt", "--opt-type",
                  "Type of remap operator:\n"
                  "\t0: eta (QF)\n"
                  "\t1: eta (QF), rho (QF)\n"
                  "\t2: eta (QF), rho (QF), e (L2 GF)\n"
                  "\t3: eta (QF), rho (QF), e (L2 GF), v (H1 GF)\n");
   args.ParseCheck();

   // 2. Read the mesh from the given mesh file, and refine once uniformly.
   // Mesh mesh = dim == 2
   //             ? Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL)
   //             : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   Mesh mesh("../../data/mobius-strip.mesh", 1, 1);
   const int dim = mesh.Dimension();
   for (int i=0; i<ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   std::vector<std::function<real_t(const Vector &)>> f;
   std::vector<std::function<void(const Vector &, Vector &)>> df;
   Array<int> space_idx(0);
   if (optType >= 0)
   {
      f.push_back(remap::volume_f);
      df.push_back(remap::volume_df);
      space_idx.Append(-1);
   }
   if (optType >= 1)
   {
      f.push_back(remap::mass_f);
      df.push_back(remap::mass_df);
      space_idx.Append(-1);
   }
   if (optType == 2)
   {
      f.push_back(remap::potential_f);
      df.push_back(remap::potential_df);
      space_idx.Append(0);
   }
   if (optType == 3)
   {
      f.push_back(remap::energy_f);
      df.push_back(remap::energy_df);
      space_idx.Append(0);
      for (int i=0; i<dim; i++)
      {
         f.push_back([i](const Vector &u)->real_t { return remap::momentum_f(u, i); });
         df.push_back([i](const Vector &u, Vector &grad_u) { remap::momentum_df(u, grad_u, i); });
         space_idx.Append(1);
      }
   }
   const int numVars = space_idx.Size();

   QuadratureSpace qspace(&mesh, qorder);
   L2_FECollection l2_fec(order, dim, BasisType::Positive);
   H1_FECollection h1_fec(order, dim);
   FiniteElementSpace l2_fespace(&mesh, &l2_fec);
   FiniteElementSpace h1_fespace(&mesh, &h1_fec);
   std::vector<FiniteElementSpace*> fes(0);
   if (optType >= 2) { fes.push_back(&l2_fespace); }
   if (optType == 3) { fes.push_back(&h1_fespace); }

   Array<int> offsets(0);
   offsets.Append(0);
   for (int vid=0; vid<space_idx.Size(); vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0) { offsets.Append(qspace.GetSize()); }
      else { offsets.Append(fes[sid]->GetTrueVSize()); }
   }
   offsets.PartialSum();
   const int numDofs = offsets.Last();

   out << "Number of quadrature points: " << qspace.GetSize() << "\n";
   out << "Number of L2 unknowns: " << l2_fespace.GetTrueVSize() << "\n";
   out << "Number of H1 unknowns: " << h1_fespace.GetTrueVSize() << "\n";
   out << "Space index: "; space_idx.Print(out, space_idx.Size());
   out << "Total number of unknowns: " << numDofs << "\n";
   out << std::endl;
   const int numConst = f.size();

   std::vector<std::unique_ptr<ComposedFunctional>> constraints_vec(numConst);
   StackedFunctional constraints(numDofs);
   for (int i=0; i<numConst; i++)
   {
      constraints_vec[i] = std::make_unique<ComposedFunctional>(
                              f[i], df[i], qspace, fes, space_idx);
      constraints.AddFunctional(*constraints_vec[i]);
   }

   BlockVector x(offsets);
   std::vector<std::unique_ptr<Vector>> disc_vecs(numVars);
   for (int vid=0; vid<numVars; vid++)
   {
      // x.GetBlock(vid) = vid + 1;
      x.GetBlock(vid).Randomize();
      const int sid = space_idx[vid];
      if (sid < 0)
      {
         auto qf = std::make_unique<QuadratureFunction>(&qspace,
                   x.GetBlock(vid).GetData());
         disc_vecs[vid] = std::move(qf);
      }
      else
      {
         auto gf = std::make_unique<GridFunction>(fes[sid]);
         gf->MakeTRef(fes[sid], x.GetBlock(vid).GetData());
         gf->SetFromTrueVector();
         disc_vecs[vid] = std::move(gf);
      }
   }

   Vector constraints_val(space_idx.Size());
   constraints.Mult(x, constraints_val);

   DenseMatrix constraint_grads(constraints.Width(), constraints.Height());
   constraints.GetGradientMatrix(x, constraint_grads);

   /// TEST: Compare the ComposedFunctional with Hard-coded Constraint value and Gradient
   VectorArrayCoefficient qf_gf_cf(numVars);
   for (int vid=0; vid<numVars; vid++)
   {
      const int sid = space_idx[vid];
      if (sid < 0)
      {
         qf_gf_cf.Set(vid, new QuadratureFunctionCoefficient(
                         static_cast<QuadratureFunction&>(*disc_vecs[vid])));
      }
      else
      {
         qf_gf_cf.Set(vid, new GridFunctionCoefficient(static_cast<GridFunction*>
                      (disc_vecs[vid].get())));
      }
   }
   std::vector<std::unique_ptr<QuadratureFunction>> qf_out(numVars);
   for (int vid=0; vid<numVars; vid++) { qf_out[vid] = std::make_unique<QuadratureFunction>(qspace); }

   QuadratureFunction qf_in(qspace, numVars);
   qf_gf_cf.Project(qf_in);

   QuadratureFunction qf_out_comp(qspace);
   QuadratureFunctionCoefficient qf_out_comp_cf(qf_out_comp);
   std::vector<std::unique_ptr<LinearForm>> lfs(fes.size());
   for (int sid=0; sid<fes.size(); sid++)
   {
      lfs[sid] = std::make_unique<LinearForm>(fes[sid]);
      lfs[sid]->AddDomainIntegrator(new QuadratureLFIntegrator(qf_out_comp_cf));
   }


   qf_gf_cf.Project(qf_in);
   for (int fid=0; fid<f.size(); fid++)
   {
      FunctionCoefficient result_cf(f[fid]);
      Vector all_point(numVars);
      Vector grad_point(numVars);
      real_t result = 0.0;
      for (int qid=0; qid<qspace.GetSize(); qid++)
      {
         all_point.MakeRef(qf_in, qid*numVars);
         result += f[fid](all_point)*qspace.GetWeights()[qid];
         df[fid](all_point, grad_point);
         for (int vid=0; vid<numVars; vid++)
         {
            (*qf_out[vid])[qid] = grad_point[vid];
         }
      }
      out << "Difference for " << fid << ": " << std::abs(result -
            constraints_val[fid]) <<
          endl;

      for (int vid=0; vid<numVars; vid++)
      {
         const int sid = space_idx[vid];
         Vector grad_vec;
         constraint_grads.GetColumnReference(fid, grad_vec);
         BlockVector grad_vec_block(grad_vec, offsets);
         if (sid < 0)
         {
            *qf_out[vid] *= qspace.GetWeights();
            out << "Gradient diff [" << fid << "][" << vid << "]: "
                << grad_vec_block.GetBlock(vid).DistanceTo(*qf_out[vid]) << std::endl;
         }
         else
         {
            qf_out_comp.MakeRef(*qf_out[vid], 0);
            lfs[sid]->Assemble();
            out << "Gradient diff [" << fid << "][" << vid << "]: "
                << grad_vec_block.GetBlock(vid).DistanceTo(*lfs[sid]) << std::endl;
         }
      }
      out << std::endl;
   }
   return 0;
}
