#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

string space;

real_t weight(const Vector &x);
real_t rho_exact(const Vector &x);
real_t w_exact(const Vector &x);
real_t compute_mass(GridFunction &gf, Coefficient &coeff, int coeff_order);

int main(int argc, char *argv[])
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 3;
   int lref = order+1;
   int lorder = 0;
   bool use_h1 = false;
   const char *device_config = "cpu";
   bool use_ea = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&use_h1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                  "--no-ea-version", "Use element assembly version.");
   args.ParseCheck();

   // Configure device
   Device device(device_config);

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor = Mesh::MakeRefined(mesh, lref, basis_lor);

   // Create spaces
   unique_ptr<FiniteElementCollection> fec, fec_lor;
   if (use_h1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         cerr << "Switching the H1 LOR space order from 0 to 1\n";
      }
      fec = make_unique<H1_FECollection>(order, dim);
      fec_lor = make_unique<H1_FECollection>(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = make_unique<L2_FECollection>(order, dim);
      fec_lor = make_unique<L2_FECollection>(lorder, dim);
   }
   FiniteElementSpace fespace(&mesh, fec.get());
   FiniteElementSpace fespace_lor(&mesh_lor, fec_lor.get());

   FunctionCoefficient axi_coeff(weight);

   GridFunction rho(&fespace);
   FunctionCoefficient rho_coeff(rho_exact);
   rho.ProjectCoefficient(rho_coeff);
   const real_t rho_ho_mass = compute_mass(rho, axi_coeff, 4);

   cout << "\n       LOR L2 Projection - Axisymmetric Weighting\n";
   cout << left << '\n';
   cout << setw(20) << " "
        << setw(20) << "Mass"
        << setw(20) << "Difference" << '\n'
        << "============================================================\n";

   cout << setw(20) << "rho HO"
        << setw(20) << scientific << setprecision(4) << rho_ho_mass
        << setw(20) << "---" << '\n';

   L2ProjectionGridTransfer gt(
      fespace, fespace_lor, {axi_coeff, 4}, {axi_coeff, 4});
   gt.UseEA(use_ea);

   GridFunction rho_lor(&fespace_lor);
   gt.ForwardOperator().Mult(rho, rho_lor);
   const real_t rho_lor_mass = compute_mass(rho_lor, axi_coeff, 4);
   const real_t rho_err = std::abs(rho_lor_mass - rho_ho_mass);

   cout << setw(20) << "rho LOR"
        << setw(20) << scientific << setprecision(4) << rho_lor_mass
        << setprecision(2)
        << std::abs(rho_err) << " (" << rho_err/abs(rho_ho_mass) << "%)\n";

   GridFunction P_rho_lor(&fespace);
   gt.BackwardOperator().Mult(rho_lor, P_rho_lor);

   const real_t P_rho_lor_mass = compute_mass(P_rho_lor, axi_coeff, 4);
   const real_t P_rho_lor_err = std::abs(P_rho_lor_mass - rho_ho_mass);

   cout << setw(20) << "P(rho LOR)"
        << setw(20) << scientific << setprecision(4) << P_rho_lor_mass
        << setprecision(2)
        << std::abs(P_rho_lor_err)
        << " (" << P_rho_lor_err/abs(rho_ho_mass) << "%)\n";

   P_rho_lor -= rho;
   cout << "\n|HO - P(R(HO))|_∞ = " << P_rho_lor.Normlinf() << "\n\n";

   // When projecting 'w' ("velocity"), we want to conserve 'rho w'
   // ("momentum"). In other words, we should have that
   //
   //    mass(rho w) = mass(rho_L w_L)
   //
   // where the mass may also be weighted with a coefficient such as the
   // axisymmetric weight.
   GridFunctionCoefficient rho_gf_coeff(&rho);
   GridFunctionCoefficient rho_lor_gf_coeff(&rho_lor);

   ProductCoefficient prod_coeff(axi_coeff, rho_gf_coeff);
   ProductCoefficient prod_coeff_lor(axi_coeff, rho_lor_gf_coeff);

   GridFunction w(&fespace);
   FunctionCoefficient w_coeff(w_exact);
   w.ProjectCoefficient(w_coeff);

   const real_t rho_w_ho_mass = compute_mass(w, prod_coeff, order + 4);

   cout << "\n       LOR L2 Projection - Axisymmetric Density Weighting\n";
   cout << left << '\n';
   cout << setw(20) << " "
        << setw(20) << "Mass"
        << setw(20) << "Difference" << '\n'
        << "============================================================\n";

   cout << setw(20) << "rho w HO"
        << setw(20) << scientific << setprecision(4) << rho_w_ho_mass
        << setw(20) << "---" << '\n';

   L2ProjectionGridTransfer gt2(fespace, fespace_lor, {prod_coeff, order + 4},
   {prod_coeff_lor, lorder + 4});
   gt2.UseEA(use_ea);

   GridFunction w_lor(&fespace_lor);
   gt2.ForwardOperator().Mult(w, w_lor);
   const real_t rho_w_lor_mass = compute_mass(w_lor, prod_coeff_lor, lorder + 4);
   const real_t rho_w_err = std::abs(rho_w_lor_mass - rho_w_ho_mass);

   cout << setw(20) << "rho w LOR"
        << setw(20) << scientific << setprecision(4) << rho_w_lor_mass
        << setprecision(2)
        << std::abs(rho_w_err) << " (" << rho_w_err/abs(rho_w_ho_mass) << "%)\n";

   GridFunction P_w_lor(&fespace);
   gt2.BackwardOperator().Mult(w_lor, P_w_lor);

   const real_t rho_P_w_lor_mass = compute_mass(P_w_lor, prod_coeff, order + 4);
   const real_t rho_P_w_lor_err = std::abs(rho_P_w_lor_mass - rho_w_ho_mass);

   cout << setw(20) << "P(rho LOR)"
        << setw(20) << scientific << setprecision(4) << rho_P_w_lor_mass
        << setprecision(2)
        << std::abs(rho_P_w_lor_err)
        << " (" << rho_P_w_lor_err/abs(rho_w_ho_mass) << "%)\n";

   P_w_lor -= w;
   cout << "\n|HO - P(R(HO))|_∞ = " << P_w_lor.Normlinf() << "\n\n";

   return 0;
}

real_t weight(const Vector &x)
{
   return x(0)*x(0) + x(1)*x(1) + 1.0;
}

real_t rho_exact(const Vector &x)
{
   return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0) + 20.0;
}

real_t w_exact(const Vector &x)
{
   return x(1)+0.25*cos(2*M_PI*x.Norml2());
}

real_t compute_mass(GridFunction &gf, Coefficient &coeff, int coeff_order)
{
   FiniteElementSpace &fes = *gf.FESpace();
   Mesh &mesh = *fes.GetMesh();

   const int order = 2*fes.GetMaxElementOrder()
                     + mesh.GetTypicalElementTransformation()->OrderW()
                     + coeff_order;

   DomainLFIntegrator *integ  = new DomainLFIntegrator(coeff);
   integ->SetIntegrationRule(
      IntRules.Get(mesh.GetTypicalElementGeometry(), order));

   LinearForm lf(&fes);
   lf.AddDomainIntegrator(integ);
   lf.Assemble();
   return lf(gf);
}
