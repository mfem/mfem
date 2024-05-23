////////////////////////////////////////////////////////////////////////////////
// TEST ENVIRONMENT
// OS: Ubuntu 22.04
// mfem: HEAD detached at v4.6
//       build config: MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES MFEM_DEBUG=NO
// hypre: HEAD detached at v2.24.0
//        build config: --with-openmp
// metis: 4.0.3 (symlinked as metis-4.0)
//        build options: defaults
// cmake version 3.29.0
// g++ (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
////////////////////////////////////////////////////////////////////////////////
// STEPS TO REPRODUCE
// 1. cd examples
// 2. make repro
// 3. ./repro
////////////////////////////////////////////////////////////////////////////////

#include <mfem.hpp>

auto compute_lambda(double E, double nu) -> double {
    // E: Young's modulus, nu: Poisson's ratio
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
}

auto compute_mu(double E, double nu) -> double {
    // E: Young's modulus, nu: Poisson's ratio
    return E / (2.0 * (1.0 + nu));
}

auto compute_displacement() -> void {
    // mesh has 1 physical volume: label 1, and 6 physical surfaces, labels 1..6
    auto mesh = mfem::Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::TETRAHEDRON, 10.0, 10.0, 10.0);
    int const bdr_attr_count = 6;

    auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

    auto serial_nodal_vector_fec = mfem::H1_FECollection(1, 3);
    auto serial_nodal_vector_space = mfem::FiniteElementSpace(&mesh, &serial_nodal_vector_fec, 3);
    auto nodal_vector_space = mfem::ParFiniteElementSpace(serial_nodal_vector_space, pmesh);

    auto displacement_gf = mfem::ParGridFunction(&nodal_vector_space);
    displacement_gf = 0.0;

    auto dirichlet_bdr_marker = mfem::Array<int>(bdr_attr_count);
    dirichlet_bdr_marker = 0;
    // attribute 1 is the fixed boundary
    dirichlet_bdr_marker[0] = 1;

    auto zero_displacement = mfem::Vector(3);
    zero_displacement = 0.0;
    auto dirichlet_coeff = mfem::VectorConstantCoefficient(zero_displacement);
    displacement_gf.ProjectBdrCoefficient(dirichlet_coeff, dirichlet_bdr_marker);

    auto neumann_bdr_marker = mfem::Array<int>(bdr_attr_count);
    neumann_bdr_marker = 0;
    // attribute 6 is the pull down boundary
    neumann_bdr_marker[5] = 1;

    auto pull_down = mfem::Vector(3);
    pull_down = 0.0;
    pull_down(1) = -1000.0;
    auto neumann_coeff = mfem::VectorConstantCoefficient(pull_down);

    // Material properties
    double youngs = 1.0e7;
    double poissons = 0.3;

    // Set up forms
    mfem::ParBilinearForm a_form(displacement_gf.ParFESpace());
    auto lambda = compute_lambda(youngs, poissons);
    std::cout << "lambda: " << lambda << std::endl;
    auto lambda_coeff = mfem::ConstantCoefficient(lambda);
    auto mu = compute_mu(youngs, poissons);
    std::cout << "mu: " << mu << std::endl;
    auto mu_coeff = mfem::ConstantCoefficient(mu);
    a_form.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));
    a_form.Assemble();

    mfem::ParLinearForm b_form(displacement_gf.ParFESpace());
    b_form.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(neumann_coeff), neumann_bdr_marker);
    b_form.Assemble();

    mfem::Array<int> ess_tdof_list;
    displacement_gf.ParFESpace()->GetEssentialTrueDofs(dirichlet_bdr_marker, ess_tdof_list);

    mfem::HypreParMatrix a_matrix;
    mfem::Vector b_vector;
    mfem::Vector x_vector;
    a_form.FormLinearSystem(dirichlet_bdr_marker, displacement_gf, b_form, a_matrix, x_vector, b_vector);

    // Define and apply a parallel PCG solver for A X = B with the
    // BoomerAMG preconditioner from hypre.
    mfem::HypreBoomerAMG amg(a_matrix);
    amg.SetPrintLevel(0);

    mfem::HyprePCG pcg(a_matrix);
    pcg.SetTol(1e-09);
    pcg.SetMaxIter(5000);
    pcg.SetPrintLevel(0);
    pcg.SetPreconditioner(amg);
    pcg.Mult(b_vector, x_vector);

    // Recover the solution as a finite element grid function.
    a_form.RecoverFEMSolution(x_vector, b_form, displacement_gf);
}

auto compute_eigenmodes() -> void {
    // mesh has 1 physical volume: label 1, and 6 physical surfaces, labels 1..6
    auto mesh = mfem::Mesh::MakeCartesian3D(50, 10, 10, mfem::Element::TETRAHEDRON, 50.0, 10.0, 10.0);
    int const bdr_attr_count = 6;

    auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

    auto serial_nodal_vector_fec = mfem::H1_FECollection(1, 3);
    auto serial_nodal_vector_space = mfem::FiniteElementSpace(&mesh, &serial_nodal_vector_fec, 3);
    auto nodal_vector_space = mfem::ParFiniteElementSpace(serial_nodal_vector_space, pmesh);

    auto dirichlet_bdr_marker = mfem::Array<int>(bdr_attr_count);
    dirichlet_bdr_marker = 0;
    // attribute 1 is the fixed boundary
    dirichlet_bdr_marker[0] = 1;

    // Material properties
    double youngs = 1.0e7;
    double poissons = 0.3;
    auto mass_density = 2710e-9;

    // Set up forms
    mfem::ParBilinearForm a_form(&nodal_vector_space);
    auto lambda = compute_lambda(youngs, poissons);
    std::cout << "lambda: " << lambda << std::endl;
    auto lambda_coeff = mfem::ConstantCoefficient(lambda);
    auto mu = compute_mu(youngs, poissons);
    std::cout << "mu: " << mu << std::endl;
    auto mu_coeff = mfem::ConstantCoefficient(mu);
    a_form.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));
    a_form.Assemble();
    a_form.EliminateEssentialBCDiag(dirichlet_bdr_marker, 1.0);
    a_form.Finalize();

    mfem::ParBilinearForm m_form(&nodal_vector_space);
    auto mass_density_coeff = mfem::ConstantCoefficient(mass_density);
    m_form.AddDomainIntegrator(new mfem::VectorMassIntegrator(mass_density_coeff));
    m_form.Assemble();
    m_form.EliminateEssentialBCDiag(dirichlet_bdr_marker, std::numeric_limits<double>::min());
    m_form.Finalize();

    mfem::HypreParMatrix *a_matrix = a_form.ParallelAssemble();
    mfem::HypreParMatrix *m_matrix = m_form.ParallelAssemble();

    auto amg = mfem::HypreBoomerAMG(*a_matrix);
    amg.SetPrintLevel(0);
    int const vector_dimension = 3;
    amg.SetSystemsOptions(vector_dimension);

    auto lobpcg = mfem::HypreLOBPCG(MPI_COMM_WORLD);
    lobpcg.SetNumModes(4);
    lobpcg.SetPreconditioner(amg);
    lobpcg.SetRandomSeed(0x10ADBEEF);
    lobpcg.SetMaxIter(5000);
    lobpcg.SetTol(1e-09);
    lobpcg.SetPrecondUsageMode(1);
    lobpcg.SetPrintLevel(2);
    lobpcg.SetMassMatrix(*m_matrix);
    lobpcg.SetOperator(*a_matrix);

    lobpcg.Solve();

    auto eigenvalues = mfem::Array<double>();
    lobpcg.GetEigenvalues(eigenvalues);

    delete a_matrix;
    delete m_matrix;
}

int main(int argc, char* argv[]) {
    mfem::Mpi::Init();
    mfem::Hypre::Init();

    // Any other combination of these functions works except for this particular order;
    // both succeed when called alone, as well as together but compute_eigenmodes() first.
    compute_displacement();
    compute_eigenmodes();
}
