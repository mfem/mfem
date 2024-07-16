//                       MFEM Example 41 - Parallel Version
//
// Compile with: make ex41p
//
// Sample runs:  mpirun -np 4 ./examples/ex41p -ref 1 -kv 0.001 -o 4
//               mpirun -np 4 ./examples/ex41p -ref 3 -kv 0.001
//
// Description:  This example demonstrates the technique of adaptive mesh
//               refinement using dual-weighted residuals in the context of a
//               steady state Navier-Stokes benchmark problem. The problem setup
//               is the flow around cylinder with a laminar flow field at
//               Reynolds number 20.

#include "mfem.hpp"

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

std::tuple<real_t, real_t> DragLift(
   GridFunction &u_gf,
   GridFunction &p_gf,
   real_t kinematic_viscosity,
   Array<int> &marker,
   IntegrationRule &ir_face)
{
   const auto &u_fes = *u_gf.FESpace();
   const auto &mesh = *u_fes.GetMesh();
   const auto dim = mesh.Dimension();

   Vector draglift(2);
   draglift = 0.0;
   Vector nor(dim);
   DenseMatrix dudx, A(dim, dim);

   for (int be = 0; be < u_fes.GetNBE(); be++)
   {
      const int bdr_el_attr = mesh.GetBdrAttribute(be);
      if (marker[bdr_el_attr-1] == 0)
      {
         continue;
      }

      const FiniteElement &bdr_el = *u_fes.GetBE(be);
      ElementTransformation &Tr = *u_fes.GetBdrElementTransformation(be);

      for (int q = 0; q < ir_face.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir_face.IntPoint(q);
         Tr.SetIntPoint(&ip);
         CalcOrtho(Tr.Jacobian(), nor);

         const real_t scale = nor.Norml2();
         nor /= -scale;

         u_gf.GetVectorGradient(Tr, dudx);
         auto p = p_gf.GetValue(Tr, ip);

         // (-p * I + nu (grad(u)))
         for (int i = 0; i < dim; i++)
         {
            for (int j = 0; j < dim; j++)
            {
               A(i, j) = -p * (i == j) + kinematic_viscosity * dudx(i, j);
            }
         }

         for (int i = 0; i < dim; i++)
         {
            real_t s = 0.0;
            for (int j = 0; j < dim; j++)
            {
               s -= A(i, j) * nor(j);
            }
            draglift(i) += s * Tr.Weight() * ip.weight;
         }
      }
   }

   return {draglift(0), draglift(1)};
}

class DLiftDuIntegrator : public LinearFormIntegrator
{
public:
   DLiftDuIntegrator(real_t nu) : nu(nu) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      MFEM_ABORT("not implemented");
   }

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      Vector nor(spaceDim);

      dpsi.SetSize(dof, spaceDim);

      elvect.SetSize(dof * spaceDim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = 2 * el.GetOrder() + 2;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetAllIntPoints(&ip);

         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         CalcOrtho(Tr.Jacobian(), nor);

         const real_t scale = nor.Norml2();
         nor /= -scale;

         auto TrEl1 = Tr.Elem1;

         el.CalcPhysDShape(*TrEl1, dpsi);

         for (int d = 0; d < dof; d++)
         {
            for (int i = 0; i < spaceDim; i++)
            {
               Vector dpsi_dot_nor(spaceDim);
               dpsi_dot_nor = 0.0;
               for (int j = 0; j < spaceDim; j++)
               {
                  dpsi_dot_nor(i) += dpsi(d, j) * nor(j);
               }
               elvect(dof*i+d) += ip.weight * Tr.Weight() * nu * dpsi_dot_nor(1);
            }
         }
      }
   }

   DenseMatrix dpsi;
   const real_t nu;
};

class DLiftDpIntegrator : public LinearFormIntegrator
{
public:
   DLiftDpIntegrator() {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      MFEM_ABORT("nope");
   }

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      Vector nor(spaceDim);
      identity.SetSize(spaceDim);
      identity = 0.0;

      psi.SetSize(dof);

      elvect.SetSize(dof * spaceDim);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int intorder = 2 * el.GetOrder() + 2;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetAllIntPoints(&ip);

         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         CalcOrtho(Tr.Jacobian(), nor);

         const real_t scale = nor.Norml2();
         nor /= scale;

         el.CalcPhysShape(Tr.GetElement1Transformation(), psi);

         for (int d = 0; d < dof; d++)
         {
            elvect(d) += ip.weight * Tr.Weight() * psi(d) * nor(1);
         }
      }
   }

   DenseMatrix identity;
   Vector psi;
};

class DWRIntegrator : public LinearFormIntegrator
{
public:
   DWRIntegrator(ParGridFunction &uhgf, ParGridFunction &phgf,
                 ParGridFunction &zuhgf, ParGridFunction &zphgf, real_t nu) :
      nu(nu), uhgf(uhgf), phgf(phgf), zuhgf(zuhgf), zphgf(zphgf) {}

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      int dof = el.GetDof();
      int spaceDim = Tr.GetSpaceDim();

      psi.SetSize(dof);
      dzh_duhT.SetSize(spaceDim, spaceDim);
      elvect.SetSize(dof);
      elvect = 0.0;

      Vector wrk0(spaceDim);

      const IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         int intorder = 6;
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, psi);

         uhgf.GetVectorValue(Tr, ip, uh);
         zuhgf.GetVectorValue(Tr, ip, zuh);
         real_t zph = zphgf.GetValue(Tr);
         phgf.GetGradient(Tr, dph);
         uhgf.GetVectorGradient(Tr, duh);
         zuhgf.GetVectorGradient(Tr, dzuh);

         // nu * (grad(uh) : grad(zuh), psi)
         real_t wrk1 = 0.0;
         for (int i = 0; i < duh.NumRows(); i++)
         {
            for (int j = 0; j < duh.NumCols(); j++)
            {
               wrk1 += duh(i, j) * dzuh(i, j);
            }
         }
         add(elvect, ip.weight * Tr.Weight() * nu * wrk1, psi, elvect);

         // (grad(uh), zuh grad(psi)) -> 0

         // (zuh dot(grad(uh), uh), psi)
         wrk1 = 0.0;
         for (int i = 0; i < duh.NumRows(); i++)
         {
            for (int j = 0; j < duh.NumCols(); j++)
            {
               wrk1 += duh(i, j) * uh(j) * zuh(i);
            }
         }
         add(elvect, ip.weight * Tr.Weight() * wrk1, psi, elvect);

         // (zuh grad(ph), psi)
         wrk1 = 0.0;
         for (int i = 0; i < dph.Size(); i++)
         {
            wrk1 += dph(i) * zuh(i);
         }
         add(elvect, ip.weight * Tr.Weight() * wrk1, psi, elvect);

         // (zph * div(uh), psi)
         add(elvect, ip.weight * Tr.Weight() * duh.Trace() * zph, psi, elvect);
      }

      for (int i = 0; i < dof; i ++)
      {
         elvect(i) = abs(elvect(i));
      }
   }

   Vector psi;
   Vector zuh, uh, dph;
   DenseMatrix dzuh, duh, dzh_duhT;

   const real_t nu;
   ParGridFunction uhgf, phgf, zuhgf, zphgf;
};

class NavierStokes : public Operator
{
public:
   NavierStokes(ParFiniteElementSpace &ufes,
                ParFiniteElementSpace &pfes,
                Array<int> &u_ess_tdof,
                Array<int> &p_ess_tdof,
                const real_t kinematic_viscosity) :
      ufes(ufes),
      pfes(pfes),
      u_ess_tdof(u_ess_tdof),
      p_ess_tdof(p_ess_tdof),
      N(&ufes),
      Ne(&ufes),
      g(&pfes, &ufes),
      ge(&pfes, &ufes),
      d(&ufes, &pfes),
      de(&ufes, &pfes),
      offsets({0, ufes.GetTrueVSize(), pfes.GetTrueVSize()}),
           mp(&pfes),
           pq(&pfes)
   {
      offsets.PartialSum();
      pc = new BlockLowerTriangularPreconditioner(offsets);

      this->height = offsets.Last();
      this->width = this->height;

      kinematic_viscosity_coef.constant = kinematic_viscosity;
      inverse_kinematic_viscosity_coef.constant = 1.0 / kinematic_viscosity;

      Array<int> empty;

      N.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      N.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      N.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      N.Setup();

      Ne.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      Ne.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      Ne.SetEssentialTrueDofs(u_ess_tdof);
      Ne.Setup();

      g.AddDomainIntegrator(new GradientIntegrator);
      g.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      g.Assemble();
      g.Finalize();
      g.FormRectangularSystemMatrix(empty, empty, G);

      ge.AddDomainIntegrator(new GradientIntegrator);
      ge.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ge.Assemble();
      ge.Finalize();
      ge.FormRectangularSystemMatrix(p_ess_tdof, u_ess_tdof, Ge);

      d.AddDomainIntegrator(new VectorDivergenceIntegrator);
      d.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      d.Assemble();
      d.Finalize();
      d.FormRectangularSystemMatrix(empty, empty, D);

      de.AddDomainIntegrator(new VectorDivergenceIntegrator);
      de.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      de.Assemble();
      de.Finalize();
      de.FormRectangularSystemMatrix(u_ess_tdof, p_ess_tdof, De);

      zero_coeff.constant = 0.0;
      pq.AddDomainIntegrator(new MassIntegrator(zero_coeff));
      pq.Assemble();
      pq.Finalize();
      pq.FormSystemMatrix(p_ess_tdof, PQ);

      R = new BlockOperator(offsets);
      R->SetBlock(0, 0, &N);
      R->SetBlock(0, 1, G.Ptr());
      R->SetBlock(1, 0, D.Ptr());

      Ae = new BlockOperator(offsets);
      Ae->SetBlock(0, 1, Ge.Ptr());
      Ae->SetBlock(1, 0, De.Ptr());
      Ae->SetBlock(1, 1, PQ.Ptr());

      mp.AddDomainIntegrator(new MassIntegrator(inverse_kinematic_viscosity_coef));
      mp.Assemble();
      mp.Finalize();
      mp.FormSystemMatrix(p_ess_tdof, Mpe);

      mpe_inv = new OperatorJacobiSmoother(mp, p_ess_tdof);
      pc->SetBlock(1, 1, mpe_inv);
      pc->SetBlock(1, 0, De.Ptr());
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      BlockVector yb(y.GetData(), offsets);
      R->Mult(x, y);
      yb.GetBlock(0).SetSubVector(u_ess_tdof, 0.0);
      yb.GetBlock(1).SetSubVector(p_ess_tdof, 0.0);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      BlockVector xb(x.GetData(), offsets);
      Ae->SetBlock(0, 0, &Ne.GetGradient(xb.GetBlock(0)));

      RebuildPC();

      return *Ae;
   }

   void RebuildPC() const
   {
      delete a_inv;

      auto amg = new HypreBoomerAMG;
      HYPRE_BoomerAMGSetSmoothType(*amg, 5);
      amg->SetOperator(*static_cast<HypreParMatrix*>(&Ae->GetBlock(0, 0)));
      amg->SetSystemsOptions(2, true);
      amg->SetPrintLevel(0);
      a_inv = amg;

      pc->SetBlock(0, 0, a_inv);
   }

   ~NavierStokes()
   {
      delete R;
      delete Ae;
      delete mpe_inv;
      delete a_inv;
      delete pc;
   }

   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   Array<int> &u_ess_tdof;
   Array<int> &p_ess_tdof;
   ConstantCoefficient kinematic_viscosity_coef;
   ConstantCoefficient inverse_kinematic_viscosity_coef;
   ParNonlinearForm N, Ne;
   ParBilinearForm pq;
   ParMixedBilinearForm g, ge, d, de;
   Array<int> offsets;
   BlockOperator *R = nullptr;
   mutable BlockOperator *Ae = nullptr;

   OperatorHandle K, Ke;
   OperatorHandle G, Ge;
   OperatorHandle D, De;
   OperatorHandle PQ, Mpe;

   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   ParBilinearForm mp;
   Solver *mpe_inv = nullptr;
   mutable Solver *a_inv = nullptr;
   ConstantCoefficient zero_coeff;
};

class NavierStokesAdjoint : public Operator
{
public:
   NavierStokesAdjoint(ParFiniteElementSpace &ufes,
                       ParFiniteElementSpace &pfes,
                       Array<int> &u_ess_tdof,
                       Array<int> &p_ess_tdof,
                       const real_t kinematic_viscosity) :
      ufes(ufes),
      pfes(pfes),
      u_ess_tdof(u_ess_tdof),
      p_ess_tdof(p_ess_tdof),
      Ne(&ufes),
      ge_tr(&ufes, &pfes),
      de_tr(&pfes, &ufes),
      offsets({0, ufes.GetTrueVSize(), pfes.GetTrueVSize()}),
           mp(&pfes),
           pq(&pfes)
   {
      offsets.PartialSum();
      pc = new BlockLowerTriangularPreconditioner(offsets);

      this->height = offsets.Last();
      this->width = this->height;

      kinematic_viscosity_coef.constant = kinematic_viscosity;
      inverse_kinematic_viscosity_coef.constant = 1.0 / kinematic_viscosity;

      Array<int> empty;
   }

   void Setup(const Vector &x)
   {
      Ne.AddDomainIntegrator(new VectorConvectionNLFIntegrator);
      Ne.AddDomainIntegrator(
         new VectorDiffusionIntegrator(kinematic_viscosity_coef));
      Ne.SetEssentialTrueDofs(u_ess_tdof);
      Ne.Setup();

      ge_tr.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator));
      ge_tr.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      ge_tr.Assemble();
      ge_tr.Finalize();
      ge_tr.FormRectangularSystemMatrix(u_ess_tdof, p_ess_tdof, GeTr);

      de_tr.AddDomainIntegrator(new TransposeIntegrator(new
                                                        VectorDivergenceIntegrator));
      de_tr.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      de_tr.Assemble();
      de_tr.Finalize();
      de_tr.FormRectangularSystemMatrix(p_ess_tdof, u_ess_tdof, DeTr);

      zero_coeff.constant = 0.0;
      pq.AddDomainIntegrator(new MassIntegrator(zero_coeff));
      pq.Assemble();
      pq.Finalize();
      pq.FormSystemMatrix(p_ess_tdof, PQ);

      auto KeTr = static_cast<HypreParMatrix &>(Ne.GetGradient(x)).Transpose();

      Ae = new BlockOperator(offsets);
      Ae->SetBlock(0, 0, KeTr);
      Ae->SetBlock(0, 1, DeTr.Ptr());
      Ae->SetBlock(1, 0, GeTr.Ptr());
      Ae->SetBlock(1, 1, PQ.Ptr());

      mp.AddDomainIntegrator(new MassIntegrator(inverse_kinematic_viscosity_coef));
      mp.Assemble();
      mp.Finalize();
      mp.FormSystemMatrix(p_ess_tdof, Mpe);

      mpe_inv = new OperatorJacobiSmoother(mp, p_ess_tdof);
      pc->SetBlock(1, 1, mpe_inv);
      pc->SetBlock(1, 0, GeTr.Ptr());

      auto amg = new HypreBoomerAMG;
      amg->SetOperator(*static_cast<HypreParMatrix*>(KeTr));
      amg->SetPrintLevel(0);
      a_inv = amg;

      pc->SetBlock(0, 0, a_inv);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      Ae->Mult(x, y);
   }

   ~NavierStokesAdjoint()
   {
      delete Ae;
      delete mpe_inv;
      delete a_inv;
      delete pc;
   }

   ParFiniteElementSpace &ufes;
   ParFiniteElementSpace &pfes;
   Array<int> &u_ess_tdof;
   Array<int> &p_ess_tdof;
   ConstantCoefficient kinematic_viscosity_coef;
   ConstantCoefficient inverse_kinematic_viscosity_coef;
   ParNonlinearForm Ne;
   ParBilinearForm pq;
   ParMixedBilinearForm ge_tr, de_tr;
   Array<int> offsets;
   mutable BlockOperator *Ae = nullptr;

   OperatorHandle Ke;
   OperatorHandle GeTr;
   OperatorHandle DeTr;
   OperatorHandle PQ, Mpe;

   mutable BlockLowerTriangularPreconditioner *pc = nullptr;
   ParBilinearForm mp;
   Solver *mpe_inv = nullptr;
   mutable Solver *a_inv = nullptr;
   ConstantCoefficient zero_coeff;
};

std::tuple<ParGridFunction, ParGridFunction>
SolveForwardProblem(
   ParFiniteElementSpace &h1vfes,
   ParFiniteElementSpace &h1fes,
   real_t kinematic_viscosity)
{
   const int myid = Mpi::WorldRank();
   ParMesh &mesh = *h1fes.GetParMesh();
   const int dim = mesh.Dimension();

   HYPRE_BigInt size = h1fes.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size * (dim + 1) << endl;
   }

   Array<int> vel_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      ess_bdr[1] = 0;
      ess_bdr[4] = 0;
      h1vfes.GetEssentialTrueDofs(ess_bdr, vel_ess_tdof_list);
   }

   Array<int> pres_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[1] = 1;
      h1fes.GetEssentialTrueDofs(ess_bdr, pres_ess_tdof_list);
   }

   NavierStokes ns(h1vfes, h1fes, vel_ess_tdof_list, pres_ess_tdof_list,
                   kinematic_viscosity);

   ParGridFunction u(&h1vfes), p(&h1fes);
   u = 0.0;
   p = 0.0;

   auto inflow = [](const Vector &coords, Vector &u)
   {
      const real_t y = coords(1);
      const real_t U = 0.3;
      u(0) = 4.0 * U * y * (0.41 - y) / pow(0.41, 2.0);
      u(1) = 0.0;
   };

   Array<int> inflow_attr(mesh.bdr_attributes.Max());
   inflow_attr = 0;
   inflow_attr[0] = 1;
   VectorFunctionCoefficient inflow_coef(dim, inflow);
   u.ProjectBdrCoefficient(inflow_coef, inflow_attr);

   BlockVector x(ns.offsets);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));

   GMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-4);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(300);
   krylov.SetMaxIter(1000);
   krylov.SetPreconditioner(*ns.pc);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.SetRelTol(1e-8);
   newton.SetAbsTol(1e-9);
   newton.SetMaxIter(15);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetOperator(ns);
   newton.SetPreconditioner(krylov);

   Vector zero;
   newton.Mult(zero, x);

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   return {u, p};
}

std::tuple<ParGridFunction, ParGridFunction>
SolveDualProblem(
   ParFiniteElementSpace &h1vfes,
   ParFiniteElementSpace &h1fes,
   ParGridFunction &ul,
   ParGridFunction &pl,
   real_t kinematic_viscosity)
{
   const int myid = Mpi::WorldRank();
   ParMesh &mesh = *h1fes.GetParMesh();
   const int dim = mesh.Dimension();

   Array<int> vel_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      ess_bdr[1] = 0;
      ess_bdr[4] = 0;
      h1vfes.GetEssentialTrueDofs(ess_bdr, vel_ess_tdof_list);
   }

   Array<int> pres_ess_tdof_list;
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[1] = 1;
      h1fes.GetEssentialTrueDofs(ess_bdr, pres_ess_tdof_list);
   }

   NavierStokesAdjoint adjoint(h1vfes, h1fes, vel_ess_tdof_list,
                               pres_ess_tdof_list,
                               kinematic_viscosity);

   ParGridFunction u(&h1vfes);
   PRefinementTransferOperator(static_cast<FiniteElementSpace>(*ul.ParFESpace()),
                               static_cast<FiniteElementSpace>(h1vfes)).Mult(ul, u);

   ParGridFunction p(&h1fes);
   PRefinementTransferOperator(static_cast<FiniteElementSpace>(*pl.ParFESpace()),
                               static_cast<FiniteElementSpace>(h1fes)).Mult(pl, p);

   BlockVector x(adjoint.offsets);
   u.GetTrueDofs(x.GetBlock(0));
   p.GetTrueDofs(x.GetBlock(1));

   adjoint.Setup(x.GetBlock(0));

   BlockVector b(adjoint.offsets);

   Vector v(2);
   v = 1.0;
   VectorConstantCoefficient vone(v);
   ConstantCoefficient one(1.0);

   Array<int> qoi_attr(mesh.bdr_attributes.Max());
   qoi_attr = 0;
   qoi_attr[5] = 1;
   qoi_attr[6] = 1;

   ParLinearForm dldu(&h1vfes);
   auto lfi1 = new DLiftDuIntegrator(kinematic_viscosity);
   dldu.AddBdrFaceIntegrator(lfi1, qoi_attr);
   dldu.Assemble();
   dldu.ParallelAssemble(b.GetBlock(0));

   ParLinearForm dldp(&h1fes);
   auto lfi2 = new DLiftDpIntegrator;
   dldp.AddBdrFaceIntegrator(lfi2, qoi_attr);
   dldp.Assemble();
   dldp.ParallelAssemble(b.GetBlock(1));

   GMRESSolver krylov(MPI_COMM_WORLD);
   krylov.SetRelTol(1e-8);
   krylov.SetAbsTol(1e-12);
   krylov.SetKDim(300);
   krylov.SetMaxIter(5000);
   krylov.SetOperator(adjoint);
   krylov.SetPreconditioner(*adjoint.pc);
   krylov.SetPrintLevel(IterativeSolver::PrintLevel().Summary());

   b.GetBlock(0).SetSubVector(vel_ess_tdof_list, 0.0);
   b.GetBlock(1).SetSubVector(pres_ess_tdof_list, 0.0);

   krylov.Mult(b, x);

   u.SetFromTrueDofs(x.GetBlock(0));
   p.SetFromTrueDofs(x.GetBlock(1));

   return {u, p};
}

std::tuple<real_t, real_t>
ComputeQoI(ParGridFunction &u, ParGridFunction &p,
           real_t kinematic_viscosity)
{
   ParFiniteElementSpace &h1vfes = *u.ParFESpace();
   ParMesh &mesh = *h1vfes.GetParMesh();
   const int dim = mesh.Dimension();
   const int polynomial_order = h1vfes.GetOrder(0);

   Array<int> qoi_attr(mesh.bdr_attributes.Max());
   qoi_attr = 0;
   qoi_attr[5] = 1;
   qoi_attr[6] = 1;

   auto ir_face = IntRules.Get(h1vfes.GetMesh()->GetFaceGeometry(0),
                               2 * polynomial_order + 1);
   auto [drag_local, lift_local] = DragLift(u, p, kinematic_viscosity, qoi_attr,
                                            ir_face);

   real_t drag_global = 0.0, lift_global = 0.0;
   MPI_Allreduce(&drag_local, &drag_global, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);
   MPI_Allreduce(&lift_local, &lift_global, 1, MFEM_MPI_REAL_T, MPI_SUM,
                 MPI_COMM_WORLD);

   return {drag_global, lift_global};
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file = "../data/channel-cylinder.msh";
   int polynomial_order = 2;
   const char *device_config = "cpu";
   bool visualization = true;
   real_t kinematic_viscosity = 1.0e-3;
   int uniform_refinements = 3;
   int max_iterations = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&polynomial_order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&uniform_refinements, "-ref", "--refinements",
                  "uniform refinements");
   args.AddOption(&kinematic_viscosity, "-kv", "--kinematic_viscosity",
                  "kinematic viscosity");
   args.AddOption(&max_iterations, "-max_it", "--max_iterations",
                  "maximum refinement iterations");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   Mesh serial_mesh(mesh_file);
   serial_mesh.EnsureNCMesh();
   const int dim = serial_mesh.Dimension();

   for (int i = 0; i < uniform_refinements; i++)
   {
      serial_mesh.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   ParaViewDataCollection paraview_dc("navier_stokes_dwr", &mesh);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);

   for (int iterations = 0; iterations < max_iterations; iterations++)
   {
      H1_FECollection fec(polynomial_order, dim);
      ParFiniteElementSpace h1vfes(&mesh, &fec, dim, Ordering::byNODES);

      H1_FECollection fecl(polynomial_order - 1, dim);
      ParFiniteElementSpace h1fes(&mesh, &fecl);

      ParMixedBilinearForm blf(&h1fes, &h1vfes);
      blf.AddDomainIntegrator(new GradientIntegrator);
      blf.Assemble();
      blf.Finalize();

      auto [u, p] = SolveForwardProblem(h1vfes, h1fes, kinematic_viscosity);

      auto [drag, lift] = ComputeQoI(u, p, kinematic_viscosity);

      const real_t U_mean = 0.2;
      const real_t c0 = 2.0 / (U_mean*U_mean * 0.1);

      if (Mpi::Root())
      {
         printf("CD = %1.8E CL = %1.8E\n", -c0*drag, -c0*lift);
      }

      // Solve the dual equation in an enriched space
      H1_FECollection fecstar(polynomial_order + 1, dim);
      ParFiniteElementSpace h1vfesstar(&mesh, &fecstar, dim, Ordering::byNODES);

      H1_FECollection feclstar(polynomial_order, dim);
      ParFiniteElementSpace h1fesstar(&mesh, &feclstar);

      auto [zu, zp] = SolveDualProblem(h1vfes, h1fes, u, p,
                                       kinematic_viscosity);

      // Compute the DWR
      L2_FECollection l2fec(0, mesh.Dimension());
      ParFiniteElementSpace marker_fes(&mesh, &l2fec);

      ParGridFunction eta(&marker_fes);

      ParLinearForm dwr_lf(&marker_fes);
      dwr_lf.AddDomainIntegrator(new DWRIntegrator(u, p, zu, zp,
                                                   kinematic_viscosity));
      dwr_lf.Assemble();
      eta = dwr_lf;

      paraview_dc.SetCycle(iterations);
      paraview_dc.SetTime(iterations);
      paraview_dc.SetLevelsOfDetail(polynomial_order);
      paraview_dc.RegisterField("velocity", &u);
      paraview_dc.RegisterField("pressure", &p);
      paraview_dc.RegisterField("velocity_dual", &zu);
      paraview_dc.RegisterField("pressure_dual", &zp);
      paraview_dc.RegisterField("error_indicator", &eta);
      paraview_dc.Save();

      // Refinement marking
      {
         const int num_ranks = Mpi::WorldSize();
         const int local_ne = mesh.GetNE();
         const int global_ne = mesh.GetGlobalNE();
         // Limit refinements to 10% of the global number of elements
         int element_refinement_limit = std::round(global_ne * 0.1);

         std::vector<real_t> error_indicator(global_ne);
         std::fill(error_indicator.begin(), error_indicator.end(), 0.0);
         Vector dofs;
         for (int ge = 0; ge < global_ne; ge++)
         {
            const int e = mesh.GetLocalElementNum(ge);
            if (e != -1)
            {
               eta.GetElementDofValues(e, dofs);
               error_indicator[ge] = abs(dofs(0));
            }
         }

         std::vector<real_t> global_error_indicator(global_ne);
         MPI_Allreduce(error_indicator.data(), global_error_indicator.data(),
                       global_ne, MFEM_MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

         auto ranking = sort_indexes(global_error_indicator);
         std::reverse(ranking.begin(), ranking.end());

         std::vector<int> ranking_pruned(element_refinement_limit);
         for (int i = 0; i < element_refinement_limit; i++)
         {
            ranking_pruned[i] = ranking[i];
         }

         Array<int> el_to_refine;
         for (auto &ge : ranking_pruned)
         {
            const int e = mesh.GetLocalElementNum(ge);
            if (e != -1)
            {
               el_to_refine.Append(e);
            }
         }
         mesh.GeneralRefinement(el_to_refine, 1, 1);
      }
   }

   return 0;
}
