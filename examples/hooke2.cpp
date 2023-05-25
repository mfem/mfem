#include <functional>
#include <iostream>
#include <variant>
#include <vector>

#include "dfem.hpp"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>

using namespace mfem;
using mfem::internal::tensor;
using mfem::internal::dual;
using mfem::internal::make_tensor;

template <int dim>
class AffineSolution
{
public:
   AffineSolution() : A(dim), b(dim)
   {
      // clang-format off
      A(0, 0) = 0.110791568544027; A(0, 1) = 0.230421268325901;
      A(1, 0) = 0.198344644470483; A(1, 1) = 0.060514559793513;
      if constexpr (dim == 3)
      {
         A(0, 2) = 0.15167673653354;
         A(1, 2) = 0.084137393813728;
         A(2, 0) = 0.011544253485023; A(2, 1) = 0.060942846497753;
         A(2, 2) = 0.186383473579596;
      }
      A *= 1e-2;

      b(0) = 0.765645367640828;
      b(1) = 0.992487355850465;
      if constexpr (dim == 3)
      {
         b(2) = 0.162199373722092;
      }
      b *= 1e-2;
      //clang-format on
   };

   /**
    * @brief MFEM-style coefficient function corresponding to this solution
    *
    * @param X Coordinates of point in reference configuration at which solution is sought
    * @param u Exact solution evaluated at \p X
    */
   void operator()(const mfem::Vector& X, mfem::Vector& u) const
   {
      A.Mult(X, u);
      u += b;
   }

   // /**
   //  * @brief Apply forcing that should produce this exact displacement
   //  *
   //  * Given the physics module, apply boundary conditions and a source
   //  * term that are consistent with the exact solution. This is
   //  * independent of the domain. The solution is imposed as an essential
   //  * boundary condition on the parts of the boundary identified by \p
   //  * essential_boundaries. On the complement of
   //  * \p essential_boundaries, the traction corresponding to the exact
   //  * solution is applied.
   //  *
   //  * @tparam p Polynomial degree of the finite element approximation
   //  * @tparam Material Type of the material model used in the problem
   //  *
   //  * @param material Material model used in the problem
   //  * @param sf The SolidMechanics module for the problem
   //  * @param essential_boundaries Boundary attributes on which essential boundary conditions are desired
   //  */
   // template <int p, typename Material>
   // void applyLoads(const Material& material, SolidMechanics<p, dim>& sf,
   //                 std::set<int> essential_boundaries) const
   // {
   //    // essential BCs
   //    auto ebc_func = [*this](const auto& X, auto& u) { this->operator()(X, u); };
   //    sf.setDisplacementBCs(essential_boundaries, ebc_func);

   //    // natural BCs
   //    typename Material::State state;
   //    auto H = make_tensor<dim, dim>([&](int i, int j) { return A(i,j); });
   //    tensor<double, dim, dim> sigma = material(state, H);
   //    auto P = solid_mechanics::CauchyToPiola(sigma, H);
   //    auto traction = [P](auto, auto n0, auto) { return dot(P, n0); };
   //    sf.setPiolaTraction(traction);
   // }

private:
   /// Linear part of solution. Equivalently, the displacement gradient
   mfem::DenseMatrix A;
   /// Constant part of solution. Rigid mody displacement.
   mfem::Vector b;
};


class LambdaOperator : public Operator
{
public:
   LambdaOperator(int size,
                  std::function<void(const Vector&, Vector&)> mult_f) :
      Operator(size),
      mult_f(mult_f)
   {}

   void Mult(const Vector& X, Vector& Y) const
   {
      mult_f(X, Y);
   }

   std::function<void(const Vector&, Vector&)> mult_f;
};

class ADOperator : public Operator
{
public:
   ADOperator(int size = 0) : Operator(size) {}
   virtual void GradientMult(const Vector &dX, Vector &Y) const = 0;
   virtual void AdjointMult(const Vector &L, Vector &Y) const = 0;
};

class ElasticityGradientOperator : public Operator
{
public:
   ElasticityGradientOperator(ADOperator &op) :
      Operator(op.Height()), op(op) {}

   void Mult(const Vector &x, Vector &y) const override
   {
      op.GradientMult(x, y);
   }

   ADOperator &op;
};

void finite_stress_qf(const tensor<double, 2, 2> &dudx, tensor<double, 2, 2> &P)
{
   double lambda, mu;
   {
      lambda = 1.25;
      mu = 1.0;
   }
   static constexpr auto I = mfem::internal::IsotropicIdentity<2>();
   auto F = dudx + I;
   auto E = 0.5 * (transpose(F) * F - I);
   // auto eps = sym(dudx);
   // auto dudx_squared = transpose(dudx) * dudx;
   // auto E = eps + 0.5 * dudx_squared;
   auto S = lambda * tr(E) * I + 2.0 * mu * E;
   P = F * S;
};

// linear elastic
void small_stress_qf(const tensor<double, 2, 2> &dudx,
                     tensor<double, 2, 2> &P)
{
   double lambda, mu;
   {
      lambda = 1.25;
      mu = 1.0;
   }
   static constexpr auto I = mfem::internal::IsotropicIdentity<2>();
   auto eps = sym(dudx);
   auto S = lambda * tr(eps) * I + 2.0 * mu * eps;
   P = S;
};

template <auto quadrature_function>
class ElasticityOperator : public ADOperator
{
public:
   ElasticityOperator(ParMesh &mesh, const int order, bool matfree,
                      bool dump_matrices) :
      ADOperator(),
      mesh(mesh),
      order(order),
      dim(mesh.Dimension()),
      vdim(mesh.Dimension()),
      num_el(mesh.GetNE()),
      h1_fec(order, dim),
      h1_fes(&mesh, &h1_fec, vdim, Ordering::byVDIM),
      matfree(matfree),
      dump_matrices(dump_matrices)
   {
      this->height = h1_fes.GetTrueVSize();
      this->width = this->height;

      ir = const_cast<IntegrationRule *>(
              &IntRules.Get(mesh.GetElementGeometry(0),
                            2 * h1_fes.GetElementOrder(0)));

      ir_face = const_cast<IntegrationRule *>(
                   &IntRules.Get(mesh.GetBdrElementGeometry(0),
                                 2 * h1_fes.GetElementOrder(0)));

      num_qp = ir->GetNPoints();

      int global_tdof_size = h1_fes.GlobalTrueVSize();
      if (Mpi::Root())
      {
         out << "dim = " << mesh.Dimension() << "\n"
             << "vdim = " << h1_fes.GetVDim() << "\n"
             << "#dofs: " << global_tdof_size << "\n"
             << "#qp in IntRule: " << num_qp << std::endl;
      }

      h1_prolongation = h1_fes.GetProlongationMatrix();

      u.SetSpace(&h1_fes);
      current_state.SetSpace(&h1_fes);
      current_iterate.SetSpace(&h1_fes);
      current_iterate_tvec.SetSize(h1_prolongation->Width());

      body_force.SetSpace(&h1_fes);
      body_force = 0.0;

      boundary_load.SetSpace(&h1_fes);
      boundary_load = 0.0;

      // Layout has to be [vdim, dim, num_qp, num_el]
      P_dudx_qp.SetSize(dim * dim * num_qp * num_el);
      out_qp.SetSize(dim * dim * num_qp * num_el);

      element_jacobian_inverse.SetSize(num_qp * dim * dim * num_el);

      matfree_gradient = new ElasticityGradientOperator(*this);

      // Cache inverse Jacobian on each quadrature point
      {
         const GeometricFactors *geom = mesh.GetGeometricFactors(
                                           *ir, GeometricFactors::JACOBIANS);
         auto J = Reshape(geom->J.Read(), num_qp, dim, dim, num_el);
         auto Jinv = Reshape(element_jacobian_inverse.Write(), num_qp, dim, dim, num_el);
         DenseMatrix Jqp(dim, dim), JqpInv(dim, dim);
         for (int e = 0; e < num_el; e++)
         {
            for (int qp = 0; qp < num_qp; qp++)
            {
               for (int i = 0; i < dim; i++)
               {
                  for (int j = 0; j < dim; j++)
                  {
                     Jqp(i, j) = J(qp, i, j, e);
                  }
               }

               CalcInverse(Jqp, JqpInv);

               for (int i = 0; i < dim; i++)
               {
                  for (int j = 0; j < dim; j++)
                  {
                     Jinv(qp, i, j, e) = JqpInv(i, j);
                  }
               }
            }
         }
      }
   }

   void Mult(const Vector &X, Vector &Y) const override
   {
      CALI_MARK_BEGIN("ElasticityOperator::Mult");

      h1_prolongation->Mult(X, u);

      gradient_wrt_x(u, *ir, dudx_qp);

      forall(quadrature_function, num_qp * num_el, dudx_qp, P_dudx_qp);

      integrate_basis_gradient(P_dudx_qp,
                               h1_fes,
                               *ir, P_dudx_qp_dphi,
                               element_jacobian_inverse);

      if (enable_body_force)
      {
         interpolate(body_force, *ir, body_force_qp);
         integrate_basis(body_force_qp, h1_fes, *ir, body_force_phi);
         P_dudx_qp_dphi += body_force_phi;
      }
      if (enable_boundary_load)
      {
         interpolate_boundary(boundary_load, *ir_face, boundary_load_qp);
         integrate_basis_boundary(boundary_load_qp, h1_fes, *ir_face,
                                  boundary_load_phi);
         P_dudx_qp_dphi -= boundary_load_phi;
      }

      h1_prolongation->MultTranspose(P_dudx_qp_dphi, Y);
      Y.SetSubVector(ess_tdof_list, 0.0);

      CALI_MARK_END("ElasticityOperator::Mult");
   }

   Operator &GetGradientNoBC(const Vector &x) const
   {
      assemble_with_bc = false;
      auto& op = GetGradient(x);
      assemble_with_bc = true;
      return op;
   }

   Operator &GetGradient(const Vector &x) const override
   {
      // T -> L
      h1_fes.GetProlongationMatrix()->Mult(x, current_state);

      // Cache dudx
      gradient_wrt_x(current_state, *ir, dudx_qp);

      if (!matfree)
      {
         if (dump_matrices)
         {
            std::ofstream matfree_gradient_out("matfreeA.txt");
            matfree_gradient->PrintMatlab(matfree_gradient_out);
            matfree_gradient_out.close();
         }

         Vector dPddudx_qp(dim * dim * dim * dim * num_qp * num_el);
         CALI_MARK_BEGIN("EnzymeAD Jacobian Assemble");
         forall([&](const tensor<double, 2, 2> &dudx,
                    tensor<double, 2, 2, 2, 2> &dPddudx)
         {
            tensor<double, 2, 2> unused_output{};
            tensor<double, 2, 2> dir{};
            dPddudx = {};
            for (int i = 0; i < dim; i++)
            {
               for (int j = 0; j < dim; j++)
               {
                  dir[i][j] = 1;
                  __enzyme_autodiff<void>(+quadrature_function,
                                          enzyme_dup, &dudx, &(dPddudx[j][i]), // autodiff returns A^t
                                          enzyme_dupnoneed, &unused_output, &dir);
                  dir[i][j] = 0;
               }
            }
         }, num_qp * num_el, dudx_qp, dPddudx_qp);
         CALI_MARK_END("EnzymeAD Jacobian Assemble");

         // Assemble processor local SparseMatrix
         SparseMatrix *mat = new SparseMatrix(h1_fes.GetVSize());
         {
            auto R = h1_fes.GetElementRestriction(ElementDofOrdering::NATIVE);
            const int dim = mesh.Dimension();
            const int num_el = mesh.GetNE();
            const int vdim = h1_fes.GetVDim();
            const int num_qp = ir->GetNPoints();
            const int num_vdofs = R->Height() / num_el;
            const int num_dofs = num_vdofs / vdim;

            const GeometricFactors *geom = mesh.GetGeometricFactors(
                                              *ir, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
            auto detJ = Reshape(geom->detJ.Read(), num_qp, num_el);
            auto invJ = Reshape(element_jacobian_inverse.Read(), num_qp, dim, dim, num_el);

            Vector A_l(num_dofs * dim * num_dofs * dim * num_el);
            A_l = 0.0;

            auto A_e = Reshape(A_l.ReadWrite(), num_dofs, dim, num_dofs, dim, num_el);
            auto D = Reshape(dPddudx_qp.Read(), dim, dim, dim, dim, num_qp, num_el);

            for (int e = 0; e < num_el; e++)
            {
               const DofToQuad &maps = h1_fes.GetFE(e)->GetDofToQuad(*ir, DofToQuad::FULL);
               const auto G = Reshape(maps.G.Read(), num_qp, dim, num_dofs);

               for (int qp = 0; qp < num_qp; qp++)
               {
                  const double JxW = detJ(qp, e) * ir->GetWeights()[qp];

                  // Pullback Gradient into physical space
                  Vector dphidx_data(num_dofs * dim);
                  dphidx_data = 0.0;
                  auto dphidx = Reshape(dphidx_data.ReadWrite(), num_dofs, dim);
                  for (int j = 0; j < num_dofs; j++)
                  {
                     for (int i = 0; i < dim; i++)
                     {
                        double s = 0.0;
                        for (int k = 0; k < dim; k++)
                        {
                           s += G(qp, k, j) * invJ(qp, k, i, e);
                        }
                        dphidx(j, i) += s;
                     }
                  }

                  for (int q = 0; q < dim; q++)
                  {
                     for (int b = 0; b < num_dofs; b++)
                     {
                        for (int i = 0; i < dim; i++)
                        {
                           for (int a = 0; a < num_dofs; a++)
                           {
                              double s = 0.0;
                              for (int l = 0; l < dim; l++)
                              {
                                 for (int k = 0; k < dim; k++)
                                 {
                                    // dN^A/dX_k (D_ijkl dN^C/dX_l)
                                    s += dphidx(a,k) * D(i,k,q,l,qp,e) * dphidx(b,l);
                                 }
                              }
                              A_e(a, i, b, q, e) += s * JxW;
                           }
                        }
                     }
                  }
               }
            }

            for (int e = 0; e < num_el; e++)
            {
               auto tmp = Reshape(A_l.ReadWrite(), num_dofs, dim, num_dofs, dim, num_el);
               DenseMatrix A_e(&tmp(0, 0, 0, 0, e), num_vdofs, num_vdofs);
               Array<int> vdofs;
               h1_fes.GetElementVDofs(e, vdofs);
               mat->AddSubMatrix(vdofs, vdofs, A_e, 1);
            }
            mat->Finalize();

            auto tmp = new HypreParMatrix(h1_fes.GetComm(),
                                          h1_fes.GlobalVSize(),
                                          h1_fes.GetDofOffsets(),
                                          mat);
            delete Amat;
            Amat = RAP(tmp, h1_fes.Dof_TrueDof_Matrix());
            delete tmp;
            delete mat;

            if (assemble_with_bc)
            {
               Amat->EliminateBC(ess_tdof_list, DiagonalPolicy::DIAG_ONE);
            }

            if (dump_matrices)
            {
               std::ofstream assembled_jacobian_out("assembled_jacobian.txt");
               Amat->PrintMatlab(assembled_jacobian_out);
               assembled_jacobian_out.close();
            }

            return *Amat;
         }
      }
      else
      {
         return *matfree_gradient;
      }
   }

   // X: current iterate
   // Y: dR/dU * X
   void GradientMult(const Vector &X, Vector &Y) const override
   {
      CALI_MARK_BEGIN("ElasticityOperator::GradientMult");

      // apply essential bcs
      current_iterate_tvec = X;
      current_iterate_tvec.SetSubVector(ess_tdof_list, 0.0);
      current_iterate.SetFromTrueDofs(current_iterate_tvec);

      gradient_wrt_x(current_iterate, *ir, ddudx_qp);

      CALI_MARK_BEGIN("EnzymeAD MatVec");
      forall([](const tensor<double, 2, 2> &dudx,
                tensor<double, 2, 2> &ddudx,
                tensor<double, 2, 2> &out)
      {
         tensor<double, 2, 2> unused_output{};
         out = {};
         __enzyme_fwddiff<void>(+quadrature_function, &dudx, &ddudx,
                                &unused_output, &out);
      }, num_qp * num_el, dudx_qp, ddudx_qp, out_qp);
      CALI_MARK_END("EnzymeAD MatVec");

      CALI_MARK_BEGIN("integrate_basis_gradient");
      integrate_basis_gradient(out_qp, h1_fes, *ir, y, element_jacobian_inverse);
      CALI_MARK_END("integrate_basis_gradient");

      // L-vector to T-vector
      h1_fes.GetProlongationMatrix()->MultTranspose(y, Y);

      // Re-assign the essential degrees of freedom on the final output vector.
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         Y[ess_tdof_list[i]] = X[ess_tdof_list[i]];
      }

      CALI_MARK_END("ElasticityOperator::GradientMult");
   }

   // dL: current iterate of adjoint state
   // Y: (dF/dU)^t * dL
   void AdjointMult(const Vector &dL, Vector &Y) const override
   {

   }

   void SetEssentialAttributes(const Array<int> attr)
   {
      h1_fes.GetEssentialTrueDofs(attr, ess_tdof_list);
   }

   void SetPrescribedDisplacement(const Array<int> attr)
   {
      h1_fes.GetEssentialTrueDofs(attr, displaced_tdof_list);
   }

   const Array<int> &GetPrescribedDisplacementTDofs()
   {
      return displaced_tdof_list;
   };

   ParGridFunction* GetExternalLoad()
   {
      enable_boundary_load = true;
      return &boundary_load;
   }

   ParGridFunction* GetBodyForce()
   {
      enable_body_force = true;
      return &body_force;
   }

   ParMesh &mesh;
   const int order;
   const int dim;
   const int vdim;
   /// Number of elements in the mesh (rank local)
   int num_el;
   int num_qp = 0;
   /// H1 finite element collection
   H1_FECollection h1_fec;
   /// H1 finite element space
   mutable ParFiniteElementSpace h1_fes;
   // Integration rule
   IntegrationRule *ir = nullptr, *ir_face = nullptr;
   const Operator *h1_element_restriction;
   const Operator *h1_prolongation;

   mutable Vector element_jacobian_inverse;

   Array<int> ess_tdof_list, displaced_tdof_list;

   ParGridFunction body_force, boundary_load;
   mutable ParGridFunction u, current_state, current_iterate;
   mutable Vector current_iterate_tvec, dudx_qp, ddudx_qp,
           P_dudx_qp_dphi, body_force_qp, boundary_load_qp, body_force_phi,
           boundary_load_phi, y,
           P_dudx_qp, out_qp;

   bool enable_boundary_load = false, enable_body_force = false;
   ElasticityGradientOperator *matfree_gradient = nullptr;

   bool matfree;
   bool dump_matrices;
   mutable HypreParMatrix *Amat = nullptr;
   mutable bool assemble_with_bc = true;
};

int main(int argc, char *argv[])
{
   using namespace std;

   Mpi::Init();
   // int num_procs = Mpi::WorldSize();
   // int myid = Mpi::WorldRank();
   Hypre::Init();

   bool visualization = false;
   int polynomial_degree = 1;
   int refinements = 0;
   int problem_type = 0;
   bool matfree = false;
   bool pmg = false;
   bool dump_matrices = false;
   const char *caliper_options = "";
   const char *mesh_file = "../data/beam-quad.mesh";

   std::shared_ptr<VectorFunctionCoefficient> u_ex_coeff;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&polynomial_degree, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&refinements, "-r", "--ref",
                  "");
   args.AddOption(&problem_type, "-p", "--problem",
                  "");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&matfree, "-matfree", "--matfree", "-no-matfree",
                  "--no-matfree",
                  "matrix free");
   args.AddOption(&pmg, "-pmg", "--pmg", "-no-pmg",
                  "--no-pmg",
                  "p-Multigrid");
   args.AddOption(&dump_matrices, "-dump_matrices", "--dump_matrices",
                  "-no-dump_matrices",
                  "--no-dump_matrices",
                  "dump matrices");
   args.AddOption(&caliper_options, "-profile", "--profile", "caliper options");
   args.ParseCheck();

   if (problem_type == 3)
   {
      // MFEM_ASSERT(strcmp("patch2D_quads.mesh", mesh_file) == 0,
      //             "have to use patch2D_quads.mesh");
   }

   cali::ConfigManager caliper_mgr;
   caliper_mgr.add(caliper_options);

   const double length = 1.0;
   const double height = 1.0;
   // const double meas_tol = 1e-8;

   caliper_mgr.start();
   CALI_MARK_FUNCTION_BEGIN;

   Mesh mesh(mesh_file, 1, 1);
   mesh.EnsureNodes();
   int dim = mesh.Dimension();

   for (int i = 0; i < refinements; i++)
   {
      mesh.UniformRefinement();
   }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   int num_el_global = pmesh.GetGlobalNE();
   if (Mpi::Root())
   {
      out << "#elements: " << num_el_global << "\n";
   }

   ElasticityOperator<finite_stress_qf> hooke(pmesh, polynomial_degree, matfree,
                                              dump_matrices);

   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_attr(pmesh.bdr_attributes.Max());
      if (problem_type == 0)
      {
         ess_attr = 0;
      }
      else if (problem_type == 1)
      {
         ess_attr = 1;
      }
      else if (problem_type == 2)
      {
         ess_attr = 0;
         ess_attr[0] = 1;
      }
      else if (problem_type == 3)
      {
         ess_attr = 1;
      }
      hooke.SetEssentialAttributes(ess_attr);
      int fixed_dofs_local = hooke.ess_tdof_list.Size();
      int fixed_dofs_global = 0;
      MPI_Allreduce(&fixed_dofs_local, &fixed_dofs_global, 1, MPI_INT, MPI_SUM,
                    pmesh.GetComm());
      if (Mpi::Root())
      {
         out << "#fixed dofs: " << fixed_dofs_global << "\n";
      }
   }

   if (problem_type == 1)
   {
      // auto body_force = hooke.GetBodyForce();
      // VectorFunctionCoefficient coeff(2, [](const Vector &coords, Vector &u)
      // {
      //    const double x = coords(0);
      //    const double y = coords(1);

      //    const double a = 0.01, b = 0.05;

      //    const double nu = 0.3;
      //    const double E = 1.0;
      //    const double lambda = nu * E / ((1.0 + nu) * (1.0 - 2.0*nu));
      //    const double mu = E / (2.0 * (1.0 + nu));

      //    u(0) = b * (2.0 * lambda + 2.0 * mu);
      //    u(1) = a * (2.0 * lambda + 2.0 * mu);
      // });
      // body_force->ProjectCoefficient(coeff);
   }

   ParGridFunction U_gf(&hooke.h1_fes), Ucmp_gf(&hooke.h1_fes);
   U_gf = 0.0;

   auto boundary_load_ramp = [&](double ramp_scale = 1.0)
   {
      Array<int> boundary_load_attr(pmesh.bdr_attributes.Max());
      boundary_load_attr = 0;
      boundary_load_attr[1] = 1;

      auto boundary_load = hooke.GetExternalLoad();
      VectorFunctionCoefficient boundary_load_coeff(2, [&](const Vector &, Vector &u)
      {
         u(0) = 0.0;
         u(1) = -1.0e-2 * ramp_scale;
      });
      boundary_load->ProjectBdrCoefficient(boundary_load_coeff, boundary_load_attr);
   };

   AffineSolution<2> affine_solution;
   auto patch_test_boundary_load_ramp = [&](ParGridFunction &gf,
                                            double ramp_scale = 1.0)
   {
      u_ex_coeff =
         std::make_shared<VectorFunctionCoefficient>(2,[&](const Vector &coords,
                                                           Vector &u)
      {
         affine_solution(coords, u);
         u *= ramp_scale;
      });

      Array<int> mms_bdr(pmesh.bdr_attributes.Max());
      mms_bdr = 1;
      gf.ProjectBdrCoefficient(*u_ex_coeff, mms_bdr);
   };

   if (problem_type == 1)
   {
      u_ex_coeff =
         std::make_shared<VectorFunctionCoefficient>(2,[](const Vector &coords,
                                                          Vector &u)
      {
         const double x = coords(0);
         const double y = coords(1);
         const double a = 0.01, b = 0.05;

         u(0) = a * (2.0 * x + y);
         u(1) = b * (x + 2.0 * y);
      });

      Array<int> mms_bdr(pmesh.bdr_attributes.Max());
      mms_bdr = 1;
      U_gf.ProjectBdrCoefficient(*u_ex_coeff, mms_bdr);
   }

   Vector U;
   U_gf.GetTrueDofs(U);

   if (problem_type == 0)
   {
      VectorFunctionCoefficient ucoeff(dim, [&](const Vector &c, Vector &u)
      {
         const double x = c(0), y = c(1);
         u(0) = x*x;
         u(1) = y;
         u *= 0.01;
      });
      U_gf.ProjectCoefficient(ucoeff);
      U_gf.GetTrueDofs(U);
      out << "u: ";
      U.Print(out, U.Size());
      Vector R(U.Size());
      out << "r(u): ";
      hooke.Mult(U, R);
      // hooke.Mult(U, R);
      R.Print(out, U.Size());
      hooke.GetGradient(U);
      exit(0);
   }

   HypreBoomerAMG* amg = nullptr;
   Multigrid pmg_solver;

   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetRelTol(1e-14);
   gmres.SetMaxIter(1000);
   gmres.SetPrintLevel(2);
   if (pmg)
   {
   }
   else
   {
      if (!matfree)
      {
         amg = new HypreBoomerAMG;
         amg->SetPrintLevel(0);
         amg->SetElasticityOptions(&hooke.h1_fes);
         gmres.SetPreconditioner(*amg);
      }
   }

   NewtonSolver newton(MPI_COMM_WORLD);
   newton.iterative_mode = true;
   newton.SetSolver(gmres);
   newton.SetOperator(hooke);
   newton.SetRelTol(1e-12);
   newton.SetMaxIter(50);
   newton.SetPrintLevel(1);

   if (problem_type == 2)
   {
      boundary_load_ramp(1.0);

      Vector zero;
      newton.Mult(zero, U);
   }
   else if (problem_type == 3)
   {
      patch_test_boundary_load_ramp(U_gf, 1.0);
      U_gf.GetTrueDofs(U);

      Vector U_tmp(U), f_tmp(U);
      auto J = &hooke.GetGradientNoBC(U_tmp);

      J->Mult(U_tmp, f_tmp);
      f_tmp *= -1.0;

      ParGridFunction f_gf(U_gf);
      f_gf.Distribute(f_tmp);
      patch_test_boundary_load_ramp(f_gf, 1.0);
      f_gf.GetTrueDofs(f_tmp);

      J = &hooke.GetGradient(U_tmp);

      gmres.SetOperator(*J);
      gmres.Mult(f_tmp, U);

      Vector zero;
      newton.Mult(zero, U);
   }
   else
   {
      Vector zero;
      newton.Mult(zero, U);
   }

   U_gf.Distribute(U);

   if (problem_type == 1 || problem_type == 3)
   {
      out << "||u - u_ex||_L2 = " << U_gf.ComputeL2Error(*u_ex_coeff) << "\n";
   }

   CALI_MARK_FUNCTION_END;
   caliper_mgr.flush();

   if (visualization)
   {
      if (Mpi::Root())
      {
         cout << "writing viz files...\n";
      }
      ParaViewDataCollection paraview_dc("hooke", &pmesh);
      paraview_dc.SetPrefixPath("output");
      paraview_dc.SetLevelsOfDetail(polynomial_degree);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("displacement", &U_gf);
      if (problem_type == 1)
      {
         Ucmp_gf.ProjectCoefficient(*u_ex_coeff);
         Ucmp_gf -= U_gf;
         paraview_dc.RegisterField("displacement_cmp", &Ucmp_gf);
      }
      paraview_dc.Save();
   }

   return 0;
}
