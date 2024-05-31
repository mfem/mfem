#include "examples/dfem/dfem_util.hpp"
#include "fem/bilininteg.hpp"
#include "fem/coefficient.hpp"
#include "fem/eltrans.hpp"
#include "fem/fe_coll.hpp"
#include "fem/lininteg.hpp"
#include "fem/pbilinearform.hpp"
#include "fem/pgridfunc.hpp"
#include "fem/plinearform.hpp"
#include "linalg/auxiliary.hpp"
#include "linalg/hypre.hpp"
#include "linalg/solvers.hpp"
#include "mfem.hpp"
#include "mpi.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v)
{
   vector<size_t> idx(v.size());
   iota(idx.begin(), idx.end(), 0);

   sort(idx.begin(), idx.end(),
   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

   return idx;
}

double source_function(const Vector &coords)
{
   const double x = coords(0);
   const double y = coords(1);
   return exp(-1000.0*(x*x + y*y));
}

class Poisson : public Operator
{
public:
   Poisson(ParMesh &mesh, ParFiniteElementSpace &fes,
           Vector *adjoint_load) : Operator()
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;

      this->height = fes.GetTrueVSize();
      this->width = this->height;

      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

      if (!adjoint_load)
      {
         ParLinearForm b(&fes);
         FunctionCoefficient source_coeff(source_function);
         b.AddDomainIntegrator(new DomainLFIntegrator(source_coeff));
         b.Assemble();
         f = b.ParallelAssemble();
      }
      else
      {
         owns_f = false;
         f = adjoint_load;
      }

      a = new ParBilinearForm(&fes);
      a->AddDomainIntegrator(new DiffusionIntegrator);
      a->Assemble();
      a->Finalize();
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      y = 0.0;
      a->TrueAddMult(x, y);
      y.Add(-1.0, *f);
      y.SetSubVector(ess_tdofs, 0.0);
   }

   virtual Operator &GetGradient(const Vector &x) const override
   {
      auto mat = a->ParallelAssemble();
      mat->EliminateBC(ess_tdofs, DiagonalPolicy::DIAG_ONE);
      A = mat;
      return *A;
   }

   ~Poisson()
   {
      delete a;
      delete A;
      if (owns_f)
      {
         delete f;
      }
   }

   Array<int> ess_tdofs;
   ParBilinearForm *a = nullptr;
   mutable Operator *A = nullptr;
   Vector *f = nullptr;
   bool owns_f = true;
};

class DWREstimator
{
   using dual_solve_callback_t = std::function<void(ParGridFunction &, Vector *)>;
   using dqoi_callback_t = std::function<void(ParGridFunction &, Vector &)>;
   using eval_dwr_callback_t =
      std::function<void(ParFiniteElementSpace &fes, ParGridFunction &, ParGridFunction &, ParGridFunction &)>;
public:
   DWREstimator(ParGridFunction &primal_solution,
                ParGridFunction &zh,
                dual_solve_callback_t solve_dual,
                dqoi_callback_t dJ,
                eval_dwr_callback_t compute_dwr,
                ParFiniteElementSpace &marker_fes) :
      solve_dual(solve_dual),
      dJ(dJ),
      compute_dwr(compute_dwr),
      zh(zh),
      primal_solution(primal_solution),
      marker_fes(marker_fes),
      dwr(&marker_fes)
   {}

   ParGridFunction& Estimate()
   {
      Vector jdual;
      dJ(zh, jdual);
      solve_dual(zh, &jdual);

      // out << "jdual :";
      // print_vector(jdual);
      // out << "zh: ";
      // print_vector(zh);

      compute_dwr(marker_fes, primal_solution, zh, dwr);

      return dwr;
   }

   dual_solve_callback_t solve_dual;
   dqoi_callback_t dJ;
   eval_dwr_callback_t compute_dwr;
   ParGridFunction &zh, &primal_solution;
   ParFiniteElementSpace &marker_fes;
   ParGridFunction dwr;
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int num_refinements = 0;
   int order = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&num_refinements, "-r", "--refinements", "");
   args.AddOption(&order, "-o", "--order", "");
   args.ParseCheck();

   Mesh serial_mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   serial_mesh.EnsureNodes();
   serial_mesh.EnsureNCMesh();
   auto nodes = serial_mesh.GetNodes();
   // *nodes -= 0.5;
   for (int i = 0; i < num_refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   H1_FECollection fec(order, mesh.Dimension());
   ParFiniteElementSpace fes(&mesh, &fec);

   H1_FECollection fec_star(order+1, mesh.Dimension());
   ParFiniteElementSpace fes_star(&mesh, &fec_star);

   L2_FECollection l2fec(0, mesh.Dimension());
   ParFiniteElementSpace marker_fes(&mesh, &l2fec);

   ParGridFunction u(&fes), u_star(&fes_star);

   auto solve_poisson = [](ParGridFunction &u, Vector *adjoint_load = nullptr)
   {
      auto fes = u.ParFESpace();
      auto mesh = fes->GetParMesh();

      Poisson poisson(*mesh, *fes, adjoint_load);

      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);

      CGSolver solver(MPI_COMM_WORLD);
      solver.SetPreconditioner(amg);
      solver.SetRelTol(1e-8);
      solver.SetMaxIter(500);

      NewtonSolver newton(MPI_COMM_WORLD);
      newton.SetSolver(solver);
      newton.SetOperator(poisson);
      newton.SetRelTol(1e-6);
      newton.SetPrintLevel(0);

      Vector x(fes->GetTrueVSize());
      u.GetTrueDofs(x);

      Vector zero;
      newton.Mult(zero, x);

      u.SetFromTrueDofs(x);
   };

   auto &solve_dual = solve_poisson;

   auto qoi = [](ParGridFunction &u)
   {
      auto mesh = u.ParFESpace()->GetParMesh();

      double qoi = 0.0;
      for (int e = 0; e < mesh->GetNE(); e++)
      {
         auto fe = u.ParFESpace()->GetFE(e);
         Vector shape(fe->GetDof());
         auto Tr = u.ParFESpace()->GetElementTransformation(e);
         auto ir = &IntRules.Get(fe->GetGeomType(), 4);

         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            Tr->SetIntPoint(&ip);

            const double w = Tr->Weight() * ip.weight;
            const double uq = u.GetValue(*Tr);
            // fe->CalcPhysShape(*Tr, shape)

            // \int u dx
            qoi += uq * w;
         }
      }
      return qoi;
   };

   auto dqoi = [](ParGridFunction &u, Vector &d)
   {
      auto mesh = u.ParFESpace()->GetParMesh();
      ParLinearForm lf(u.ParFESpace());
      ConstantCoefficient one(1.0);
      lf.AddDomainIntegrator(new DomainLFIntegrator(one));
      lf.Assemble();
      d.SetSize(u.ParFESpace()->GetTrueVSize());
      lf.ParallelAssemble(d);
   };

   auto eval_dwr = [](ParFiniteElementSpace &marker_fes, ParGridFunction &uhgf,
                      ParGridFunction &zhgf, ParGridFunction &dwr)
   {
      GridFunctionCoefficient uh(&uhgf);
      GridFunctionCoefficient zh(&zhgf);
      GradientGridFunctionCoefficient grad_uh(&uhgf);
      GradientGridFunctionCoefficient grad_zh(&zhgf);
      InnerProductCoefficient grad_zhTgrad_uh(grad_uh, grad_zh);
      ScalarVectorProductCoefficient zhgrad_uh(zh, grad_uh);
      FunctionCoefficient source_coeff(source_function);

      auto neg = [](double z, double u)
      {
         return -z * u;
      };
      TransformedCoefficient neg_scaled_source_coeff(&zh, &source_coeff, neg);

      ParLinearForm lf(&marker_fes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(grad_zhTgrad_uh));
      lf.AddDomainIntegrator(new DomainLFGradIntegrator(zhgrad_uh));
      lf.AddDomainIntegrator(new DomainLFIntegrator(neg_scaled_source_coeff));
      lf.Assemble();

      ParBilinearForm mass(dwr.ParFESpace());
      mass.AddDomainIntegrator(new MassIntegrator);
      mass.Assemble();
      mass.Finalize();
      auto M = mass.ParallelAssemble();
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetOperator(*M);

      Vector lf_t(dwr.ParFESpace()->GetTrueVSize()),
             dwr_t(dwr.ParFESpace()->GetTrueVSize());
      lf.ParallelAssemble(lf_t);
      cg.Mult(lf_t, dwr_t);
      dwr.SetFromTrueDofs(dwr_t);
   };

   for (int i = 0; i < 6; i++)
   {
      out << "#dofs: " << fes.GetTrueVSize() << "\n";

      u = 0.0;
      u_star = 0.0;

      solve_poisson(u);

      out << "J(u) = " << abs(qoi(u)) << "\n";

      DWREstimator estimator(u, u_star, solve_dual, dqoi, eval_dwr, marker_fes);
      auto dwr = estimator.Estimate();
      print_vector(dwr);

      // Refinement marking
      {
         Array<int> el_to_refine;
         std::vector<double> element_ranking(mesh.GetNE());
         Vector dofs;
         for (int e = 0; e < mesh.GetNE(); e++)
         {
            dwr.GetElementDofValues(e, dofs);
            element_ranking[e] = abs(dofs(0));
         }

         auto v = sort_indexes(element_ranking);
         std::reverse(v.begin(), v.end());

         int element_refinement_limit = std::round(v.size() * 0.2);
         for (auto &i : v)
         {
            if (el_to_refine.Size() <= element_refinement_limit)
            {
               el_to_refine.Append(i);
            }
            else
            {
               break;
            }
         }
         mesh.GeneralRefinement(el_to_refine, 1, 1);
      }

      fes.Update();
      u.Update();
      fes.UpdatesFinished();

      fes_star.Update();
      u_star.Update();
      fes_star.UpdatesFinished();

      marker_fes.Update();
      marker_fes.UpdatesFinished();
      out << "\n";
   }

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs << " " << myid << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << u << "pause\n" << flush;

   return 0;
}
