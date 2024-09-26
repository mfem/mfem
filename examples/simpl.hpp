#include "mfem.hpp"
#ifndef SIMPL
#define SIMPL

namespace mfem
{

class VectorBdrDirectionalMassIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &k;
   VectorCoefficient &d;
   const int vdim;
   const int oa, ob;
public:
   VectorBdrDirectionalMassIntegrator(
      Coefficient &k, VectorCoefficient &d, const int vdim, const int oa=2,
      const int ob=0):
      BilinearFormIntegrator(NULL), k(k), d(d), vdim(vdim), oa(oa), ob(ob) {}
   VectorBdrDirectionalMassIntegrator(
      Coefficient &k, VectorCoefficient &d, const int vdim,
      const IntegrationRule *ir):
      BilinearFormIntegrator(ir), k(k), d(d), vdim(vdim), oa(0), ob(0) {}

   void AssembleFaceMatrix(const FiniteElement &el,
                           const FiniteElement &dummy,
                           FaceElementTransformations &Tr,
                           DenseMatrix &elmat) override
   {
      int dof = el.GetDof();
      Vector shape(dof);

      elmat.SetSize(dof*vdim);
      elmat = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int intorder = oa * el.GetOrder() + ob;    // <------ user control
         ir = &IntRules.Get(Tr.FaceGeom, intorder); // of integration order
      }

      DenseMatrix elmat_scalar(dof);
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);

         // Set the integration point in the face and the neighboring element
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring element's integration point
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         double val = k.Eval(*Tr.Face, ip)*Tr.Face->Weight() * ip.weight;

         el.CalcShape(eip, shape);
         MultVVt(shape, elmat_scalar);
         elmat_scalar *= val;
         for (int row = 0; row < vdim; row++)
         {
            elmat.AddSubMatrix(dof*row, elmat_scalar);
         }
      }
   }
};

void MarkBoundary(Mesh &mesh, std::function<bool(const Vector &)> marker,
                  int attr)
{
   Array<int> v;
   Vector center(mesh.SpaceDimension());
   Vector coord(mesh.SpaceDimension());
   for (int i=0; i<mesh.GetNBE(); i++)
   {
      center = 0.0;
      mesh.GetBdrElementVertices(i, v);
      for (int j=0; j<v.Size(); j++)
      {
         coord.SetData(mesh.GetVertex(v[j]));
         center.Add(1.0, coord);
      }
      center *= 1.0 / v.Size();
      if (marker(center))
      {
         mesh.SetBdrAttribute(i, attr);
      }
   }
   mesh.SetAttributes();
}

void MarkElement(Mesh &mesh, std::function<bool(const Vector &)> marker,
                 int attr)
{
   Vector center(mesh.SpaceDimension());
   for (int i=0; i<mesh.GetNE(); i++)
   {
      center = 0.0;
      mesh.GetElementCenter(i, center);
      if (marker(center))
      {
         mesh.SetAttribute(i, attr);
      }
   }
   mesh.SetAttributes();
}

void ProjectCoefficient(GridFunction &x, Coefficient &coeff,
                        int attribute)
{
   int i;
   Array<int> vdofs;
   Vector vals;
   FiniteElementSpace* fes = x.FESpace();

   DofTransformation * doftrans = NULL;

   for (i = 0; i < fes->GetNE(); i++)
   {
      if (fes->GetAttribute(i) != attribute)
      {
         continue;
      }

      doftrans = fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
      if (doftrans)
      {
         doftrans->TransformPrimal(vals);
      }
      x.SetSubVector(vdofs, vals);
   }
}

inline void SolveEllipticProblem(BilinearForm &a, LinearForm &b,
                                 GridFunction &x, Array<int> ess_tdof_list, bool use_elasticity=false)
{
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, 1);

   GSSmoother M;
   CGSolver cg;
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.SetPrintLevel(0);
   cg.iterative_mode=true;
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);
}

#ifdef MFEM_USE_MPI
inline void ParSolveEllipticProblem(ParBilinearForm &a, ParLinearForm &b,
                                    ParGridFunction &x, Array<int> ess_tdof_list, bool use_elasticity=false)
{
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, 1);

   HypreBoomerAMG amg(A);
   if (a.FESpace()->GetVDim() > 1)
   {
      amg.SetSystemsOptions(a.FESpace()->GetVDim());
   }
   // if (use_elasticity)
   // {
   //    amg.SetElasticityOptions(a.ParFESpace());
   // }
   // else if (a.FESpace()->GetVDim() > 1)
   // {
   //    amg.SetSystemsOptions(a.FESpace()->GetVDim());
   // }
   amg.SetPrintLevel(0);
   HyprePCG pcg(A);
   pcg.SetTol(1e-8);
   pcg.SetMaxIter(2000);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);
}
#endif

class LinearProblem
{
protected:
   FiniteElementSpace *fes;
   Mesh *mesh;
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm> b;
   std::unique_ptr<LinearForm> adj_b;
   bool isAstationary=false;
   bool isBstationary=false;
   bool isAdjBstationary=false;
   Array<int> ess_tdof_list;

   Array<Coefficient*> ownedCoefficients;
   Array<VectorCoefficient*> ownedVectorCoefficients;

   bool parallel=false;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh=nullptr;
   ParFiniteElementSpace *pfes=nullptr;
   ParBilinearForm *par_a=nullptr;
   ParLinearForm *par_b=nullptr;
   ParLinearForm *par_adj_b=nullptr;
#endif

public:
   LinearProblem(FiniteElementSpace &fes, bool has_dualRHS=false)
      : fes(&fes), mesh(fes.GetMesh()), ess_tdof_list(0),
        ownedCoefficients(0), ownedVectorCoefficients(0)
   {
#ifdef MFEM_USE_MPI
      pmesh = dynamic_cast<ParMesh *>(mesh);
      if (pmesh)
      {
         parallel = true;
         pfes = static_cast<ParFiniteElementSpace *>(&fes);
         par_a = new ParBilinearForm(pfes);
         par_b = new ParLinearForm(pfes);
         a.reset(par_a);
         b.reset(par_b);
         if (has_dualRHS)
         {
            par_adj_b = new ParLinearForm(pfes);
            adj_b.reset(par_adj_b);
         }
      }
      else
      {
         a.reset(new BilinearForm(&fes));
         b.reset(new LinearForm(&fes));
         if (has_dualRHS) {adj_b.reset(new LinearForm(&fes));}
      }
#else
      a.reset(new BilinearForm(&fes));
      b.reset(new LinearForm(&fes));
      if (has_dualRHS) {adj_b.reset(new LinearForm(&fes));}
#endif
   }

   ~LinearProblem() {ownedVectorCoefficients.DeleteAll(); ownedCoefficients.DeleteAll();}

   void SetAstationary(bool isstationary=true) {isAstationary=isstationary;}
   void SetBstationary(bool isstationary=true) {isBstationary=isstationary;}
   void SetAdjBstationary(bool isstationary=true) {isAdjBstationary=isstationary;}
   bool IsAstationary() {return isAstationary;}
   bool IsBstationary() {return isBstationary;}
   bool IsAdjBstationary() {return isAdjBstationary;}
   void AssembleStationaryOperators()
   {
      if (isAstationary) {a->Update(); a->Assemble();}
      if (isBstationary) {b->Assemble();}
      if (isAdjBstationary) {adj_b->Assemble();}
   }

   BilinearForm & GetBilinearForm() {return *a;}
   LinearForm & GetLinearForm() {return *b;}
   LinearForm & GetAdjointLinearForm() {return *adj_b;}
   virtual void Solve(GridFunction &x, bool assembleA, bool assembleB) = 0;
   virtual void SolveDual(GridFunction &x, bool assembleA, bool assembleB) = 0;
   void MakeCoefficientOwner(Coefficient *coeff) {ownedCoefficients.Append(coeff);}
   void MakeVectorCoefficientOwner(VectorCoefficient
                                   *coeff) {ownedVectorCoefficients.Append(coeff);}


   void SetEssentialBoundary(Array<int> ess_bdr)
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         pfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      else
      {
         fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
#else
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
#endif
   }

   void SetEssentialBoundary(Array2D<int> ess_bdr)
   {
      Array<int> ess_bdr_comp;
      Array<int> ess_tdof_list_comp;
      for (int i=-1; i<fes->GetVDim(); i++)
      {
         ess_bdr.GetRow(i+1, ess_bdr_comp);
         ess_tdof_list_comp.SetSize(0);
#ifdef MFEM_USE_MPI
         if (parallel)
         {
            pfes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
         }
         else
         {
            fes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
         }
#else
         fes->GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
#endif
         ess_tdof_list.Append(ess_tdof_list_comp);
      }
   }
};

class LinearEllipticProblem : public LinearProblem
{
protected:
   bool isElasticity=false;
public:
   LinearEllipticProblem(FiniteElementSpace &fes,
                         bool hasDualRHS):LinearProblem(fes, hasDualRHS) {}
   ~LinearEllipticProblem() = default;
   void Solve(GridFunction &x, bool assembleA, bool assembleB) override final
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (assembleA) {par_a->Update(); par_a->Assemble(); }
         if (assembleB) {par_b->Assemble(); }
         ParGridFunction *par_x = static_cast<ParGridFunction*>(&x);
         ParSolveEllipticProblem(*par_a, *par_b, *par_x, ess_tdof_list, isElasticity);
      }
      else
      {
         if (assembleA) {a->Update(); a->Assemble(); }
         if (assembleB) {b->Assemble(); }
         SolveEllipticProblem(*a, *b, x, ess_tdof_list, isElasticity);
      }
#else
      if (assembleA) {a->Update(); a->Assemble(); }
      if (assembleB) {b->Assemble(); }
      SolveEllipticProblem(*a, *b, x, ess_tdof_list, isElasticity);
#endif
   }

   void SolveDual(GridFunction &x, bool assembleA, bool assembleB) override final
   {
      if (!adj_b) {MFEM_ABORT("Adjoint problem undefined");}
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (assembleA) {par_a->Update(); par_a->Assemble(); }
         if (assembleB) {par_adj_b->Assemble(); }
         ParGridFunction *par_x = static_cast<ParGridFunction*>(&x);
         ParSolveEllipticProblem(*par_a, *par_adj_b, *par_x, ess_tdof_list);
      }
      else
      {
         if (assembleA) {a->Update(); a->Assemble(); }
         if (assembleB) {adj_b->Assemble(); }
         SolveEllipticProblem(*a, *adj_b, x, ess_tdof_list);
      }
#else
      if (assembleA) {a->Update(); a->Assemble(); }
      if (assembleB) {adj_b->Assemble(); }
      SolveEllipticProblem(*a, *adj_b, x, ess_tdof_list);
#endif
   }
};

class LinearElasticityProblem final: public LinearEllipticProblem
{
protected:
   Coefficient *lambda;
   Coefficient *mu;

public:
   LinearElasticityProblem(FiniteElementSpace &fes, Coefficient *lambda,
                           Coefficient *mu, bool has_dualRHS=false):LinearEllipticProblem(fes,
                                    has_dualRHS),
      lambda(lambda), mu(mu)
   {
      a->AddDomainIntegrator(new ElasticityIntegrator(*lambda, *mu));
      isElasticity = true;
   }
};

class HelmholtzFilter final: public LinearEllipticProblem
{
protected:
   ConstantCoefficient eps2;
   Coefficient *rho;
   Coefficient *energy;

public:
   HelmholtzFilter(FiniteElementSpace &fes, real_t filter_radius, Coefficient *rho,
                   Coefficient *energy):LinearEllipticProblem(fes, true),
      eps2(std::pow(filter_radius/(2.0*std::sqrt(3)),2))
   {
      a->AddDomainIntegrator(new DiffusionIntegrator(eps2));
      a->AddDomainIntegrator(new MassIntegrator());
      b->AddDomainIntegrator(new DomainLFIntegrator(*rho));
      adj_b->AddDomainIntegrator(new DomainLFIntegrator(*energy));
      isAstationary = true;
   }
};

class L2Projector final : public LinearProblem
{
protected:
   Coefficient *target;

public:
   L2Projector(FiniteElementSpace &fes,
               Coefficient *target):LinearProblem(fes, false),
      target(target)
   {
      a->AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
      b->AddDomainIntegrator(new DomainLFIntegrator(*target));
      isAstationary = true;
   }

   void Solve(GridFunction &x, bool assembleA=false,
              bool assembleB=true) override final
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         if (assembleA) {par_a->Update(); par_a->Assemble(); }
         if (assembleB) {par_b->Assemble(); }
         ParGridFunction *par_x = static_cast<ParGridFunction*>(&x);
         par_a->Mult(*par_b, *par_x);
      }
      else
      {
         if (assembleA) {a->Update(); a->Assemble(); }
         if (assembleB) {b->Assemble(); }
         a->Mult(*b, x);
      }
#else
      if (assembleA) {a->Update(); a->Assemble(); }
      if (assembleB) {b->Assemble(); }
      a->Mult(*b, x);
#endif
   }

   void SolveDual(GridFunction &x, bool assembleA=false,
                  bool assembleB=true) override final
   {
      MFEM_ABORT("Dual problem undefined");
   }
};

enum TopoptProblem
{
   Cantilever2=1,
   Cantilever3=2,
   MBB2=3,
   Torsion3=4,
   Bridge2=5,
   // Below this require adjoint solution of elasticity problem
   // Make sure that they are with negative numbers
   ForceInverter=-2
};

#ifdef MFEM_USE_MPI
ParMesh GetParMeshTopopt(TopoptProblem problem, int ref_serial,
                         int ref_parallel,
                         real_t &filter_radius, real_t &vol_fraction,
                         Array2D<int> &ess_bdr, Array<int> &ess_bdr_filter)
{
   switch (problem)
   {
      case Cantilever2: // Cantilver 2
      {
         if (filter_radius < 0) { filter_radius = 0.05; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }
         Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::Type::QUADRILATERAL, false,
                                           3.0, 1.0);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         ess_bdr.SetSize(3, 4); ess_bdr = 0;
         ess_bdr_filter.SetSize(4); ess_bdr_filter = 0;
         ess_bdr(0, 3) = 1;
         return pmesh;
         break;
      }

      case Cantilever3:
      {
         if (filter_radius < 0) { filter_radius = 0.01; }
         if (vol_fraction < 0) { vol_fraction = 0.12; }
         Mesh mesh = Mesh::MakeCartesian3D(2, 1, 1, Element::Type::HEXAHEDRON, 2.0, 1.0,
                                           1.0);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         ess_bdr.SetSize(4, 6); ess_bdr = 0;
         ess_bdr_filter.SetSize(6); ess_bdr_filter = 0;
         ess_bdr(0, 4) = 1;
         ess_bdr_filter[0] = -1;
         ess_bdr_filter[5] = -1;
         return pmesh;
         break;
      }

      case Torsion3:
      {
         if (filter_radius < 0) { filter_radius = 0.0025; }
         if (vol_fraction < 0) { vol_fraction = 0.01; }
         Mesh mesh = Mesh::MakeCartesian3D(5, 12, 12, Element::Type::HEXAHEDRON, 0.5,
                                           1.2, 1.2);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         ess_bdr.SetSize(4, 6); ess_bdr = 0;
         ess_bdr_filter.SetSize(6); ess_bdr_filter = 0;
         ess_bdr(0, 2) = 1;
         ess_bdr_filter = 1;
         ess_bdr_filter[2] = 0;
         ess_bdr_filter[4] = 0;
         return pmesh;
         break;
      }

      case MBB2:
      {
         if (filter_radius < 0) { filter_radius = 0.05; }
         if (vol_fraction < 0) { vol_fraction = 0.5; }
         Mesh mesh = Mesh::MakeCartesian2D(3, 1, Element::Type::QUADRILATERAL, false,
                                           3.0, 1.0);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         const real_t h = std::pow(2.0, -(ref_serial + ref_parallel));
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            return (x[0] > 3.0 - std::pow(2.0, -5)) && (x[1] < std::pow(h,2.0));
         }, 5);
         ess_bdr.SetSize(3, 5); ess_bdr = 0;
         ess_bdr_filter.SetSize(5); ess_bdr_filter = 0;
         ess_bdr(1, 3) = 1;
         ess_bdr(2, 4) = 1;
         return pmesh;
         break;
      }

      case Bridge2:
      {
         if (filter_radius < 0) { filter_radius = 0.05; }
         if (vol_fraction < 0) { vol_fraction = -0.2; } // negative: lower bound
         Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::Type::QUADRILATERAL, false,
                                           2.0, 1.0);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         const real_t h = std::pow(2.0, -(ref_serial + ref_parallel));
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            // mark top right corner for pin support
            return (x[1] < std::pow(2.0, -5)) && (x[0] > 2.0 - std::pow(h,2.0));
         }, 5);
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            // mark bottom left corner for no material boundary
            return (x[1] < 0.5) && (x[0] < std::pow(h,2.0));
         }, 6);
         MarkElement(pmesh, [](const Vector &x)
         {
            // mark top portion as passive elements
            return x[1]>(1-std::pow(2.0,-5));
         }, 2);
         ess_bdr.SetSize(3, 6); ess_bdr = 0;
         ess_bdr_filter.SetSize(6); ess_bdr_filter = 0;
         ess_bdr(1, 3) = 1;
         ess_bdr(0, 4) = 1;
         // ess_bdr(1, 5) = 1;
         ess_bdr_filter[5] = -1;
         ess_bdr_filter[2] = 1;
         return pmesh;
         break;
      }

      case ForceInverter:
      {
         Mesh mesh = Mesh::MakeCartesian2D(2, 1, Element::Type::QUADRILATERAL, false,
                                           2.0, 1.0);
         for (int i=0; i<ref_serial; i++) {mesh.UniformRefinement(); }
         ParMesh pmesh(MPI_COMM_WORLD, mesh);
         mesh.Clear();
         for (int i=0; i<ref_parallel; i++) {pmesh.UniformRefinement(); }
         //                        X-Roller (3)
         //               ---------------------------------
         //  INPUT (6) -> |                               | <- output (5)
         //               -                               -
         //               |                               |
         //               |                               |
         //               -                               |
         //  FIXED (7)  X |                               |
         //               ---------------------------------
         const real_t h = std::pow(2.0, -(ref_serial + ref_parallel));
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            // output, right top
            return (x[1] > 1.0 - 0.01) && (x[0] > 2.0 - std::pow(h,2.0));
         }, 5);
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            // input, left top
            return (x[1] > 1.0 - 0.01) && (x[0] < std::pow(h,2.0));
         }, 6);
         MarkBoundary(pmesh, [h](const Vector &x)
         {
            // fixed, left bottom
            return (x[1] < 0.01) && (x[0] < std::pow(h,2.0));
         }, 7);
         ess_bdr.SetSize(3, 7); ess_bdr = 0;
         ess_bdr_filter.SetSize(7); ess_bdr_filter = 0;
         ess_bdr(0, 6) = 1;
         ess_bdr(2, 2) = 1;
         return pmesh;
         break;
      }

   }
}
#endif

void SetupTopoptProblem(TopoptProblem problem,
                        LinearElasticityProblem &elasticity, HelmholtzFilter &filter,
                        VectorCoefficient &u_cf, Coefficient &frho_cf)
{
   switch (problem)
   {
      case Cantilever2:
      {
         const Vector center({2.9, 0.5});
         auto *coeff = new VectorFunctionCoefficient(
            2, [center](const Vector &x, Vector &f)
         {
            f = 0.0;
            real_t d = ((x[0] - center[0]) * (x[0] - center[0])
                        + (x[1] - center[1]) * (x[1] - center[1]));
            if (d < 0.0025)
            {
               f[1] = -1.0;
            }
         });
         elasticity.MakeVectorCoefficientOwner(coeff);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator( *coeff));
         break;
      }

      case Cantilever3:
      {
         const Vector center({1.9, 0.0, 0.1});
         auto *coeff = new VectorFunctionCoefficient(
            3, [center](const Vector &x, Vector &f)
         {
            f = 0.0;
            real_t d = ((x[0] - center[0]) * (x[0] - center[0])
                        + (x[2] - center[2]) * (x[2] - center[2]));
            if (d < 0.0025)
            {
               f[2] = -1.0;
            }
         });
         elasticity.MakeVectorCoefficientOwner(coeff);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator( *coeff));
         break;
      }

      case Torsion3:
      {
         const Vector center({0.0, 0.6, 0.6});
         auto *coeff = new VectorFunctionCoefficient(
            3, [center](const Vector &x, Vector &f)
         {
            f = 0.0;
            real_t d = ((x[1] - center[1]) * (x[1] - center[1])
                        + (x[2] - center[2]) * (x[2] - center[2]));
            if (x[0] < 0.01 && d > 0.04 && d < 0.09)
            {
               f[1] = center[2]-x[2];
               f[2] = x[1]-center[1];
            }
         });
         break;
      }

      case MBB2:
      {
         auto *coeff = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (x[0] < 0.05 && x[1] > 0.95)
            {
               f[1] = -1.0;
            }
         });
         elasticity.MakeVectorCoefficientOwner(coeff);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator( *coeff));
         break;
      }

      case Bridge2:
      {
         auto *coeff = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {
            f = 0.0;
            if (x[1] > 1.0 - std::pow(2, -5)) { f[1] = -20.0; }
         });
         Vector g({0.0, -9.8});
         auto *g_cf = new VectorConstantCoefficient(g);
         auto *gfrho_cf = new ScalarVectorProductCoefficient(frho_cf, *g_cf);
         auto *gu_cf = new InnerProductCoefficient(u_cf, *g_cf);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator(*coeff)
         );
         elasticity.MakeVectorCoefficientOwner(coeff);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator(*gfrho_cf)
         );
         elasticity.MakeVectorCoefficientOwner(g_cf);
         elasticity.MakeVectorCoefficientOwner(gfrho_cf);
         elasticity.SetBstationary(false);

         filter.GetAdjointLinearForm().AddDomainIntegrator(
            new DomainLFIntegrator(*gu_cf)
         );
         filter.MakeCoefficientOwner(gu_cf);
         filter.SetAdjBstationary(false);
         break;
      }

      case ForceInverter:
      {
         auto *coeff = new VectorFunctionCoefficient(
            2, [](const Vector &x, Vector &f)
         {

         });
         elasticity.MakeVectorCoefficientOwner(coeff);
         real_t k_in(1.0), k_out(1e-3);
         Vector traction({100.0, 0.0});
         auto *traction_cf = new VectorConstantCoefficient(traction);
         elasticity.MakeVectorCoefficientOwner(traction_cf);
         elasticity.GetLinearForm().AddDomainIntegrator(
            new VectorDomainLFIntegrator(*traction_cf)
         );
         Vector d_in({k_in, 0.0}), d_out({-k_out, 0.0});
         auto *kd_in = new VectorConstantCoefficient(d_in);
         auto *kd_out = new VectorConstantCoefficient(d_out);




         break;
      }
   }
}

} // end of namespace mfem
#endif // end of define SIMPL
