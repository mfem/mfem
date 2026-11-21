#include "mfem.hpp"
#include "vector_boundary_AMP_LFIntegrator.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace mfem;
using namespace std;

real_t TOL = 1e-10;

// Example tangential vector field for 2D: v = (-y, x)
class TangentialField2D : public VectorCoefficient {
private:
    int mode; // 1: Constant, 2: Rotation
    real_t expected_value; 
public:
    TangentialField2D(int mode_) : VectorCoefficient(2), mode(mode_)
    {
        if (mode == 1) {
            expected_value = 2.0; // For constant field on unit square
        } else if (mode == 2) {
            expected_value = 0.0; // For rotation field on unit square
        } else {
            MFEM_ABORT("Invalid mode for TangentialField2D. Choose 1 or 2.");
        }
    }

    virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override {
        Vector transip(2);
        T.Transform(ip, transip);
        real_t x = transip(0);
        real_t y = transip(1);
        V.SetSize(2);

        if (mode == 1) {
            // BENCHMARK 1: Constant Field V = [1, 0]
            // Expected Result on Unit Square: 2.0 (sum of load vector components)
            V(0) = 1.0;
            V(1) = 0.0;
        }
            else if (mode == 2) {
                // BENCHMARK 2: Rotation Field V = [-y, x]
            // Expected Result on Unit Square: 0.0 (sum of load vector components)
            V(0) = -y;
            V(1) = x;
        }
        else {
            MFEM_ABORT("Invalid mode for TangentialField2D. Choose 1 or 2.");
        }
    }
    
    real_t GetExpectedValue() const { return expected_value; }
};

class TangentialField3D : public VectorCoefficient {
private:
    int mode; // 1: Constant, 2: Rotation, 3: Full Antisymmetric
    real_t expected_value; // Expected integral value for unit cube
public:
    TangentialField3D(int mode_) : VectorCoefficient(3), mode(mode_)
    {
        if (mode == 1) {
            expected_value = 4.0; // For constant field on unit cube
        } else if (mode == 2) {
            expected_value = 0.0; // For axisymmetric rotation on unit cube
        } else if (mode == 3) {
            expected_value = 0.0; // For full antisymmetric field on unit cube
        } else {
            MFEM_ABORT("Invalid mode for TangentialField3D. Choose 1, 2, or 3.");
        }
    }

    virtual void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override {
        Vector transip(3);
        T.Transform(ip, transip);
        real_t x = transip(0);
        real_t y = transip(1);
        real_t z = transip(2);
        V.SetSize(3);

        if (mode == 1) {
            // BENCHMARK 1: Constant Field
            // Expected Result on Unit Cube: 4.0
            V(0) = 1.0;
            V(1) = 0.0;
            V(2) = 0.0;
        }
        else if (mode == 2) {
            // BENCHMARK 2: Axisymmetric Rotation [-y, x, 0]
            // Expected Result on Unit Cube: 0.0
            V(0) = -y;
            V(1) = x;
            V(2) = 0.0;
        }
        else if (mode == 3) {
            // BENCHMARK 3: Full Antisymmetric (a=1, b=0.5, c=0.25)
            // Expected Result on Unit Cube: 0.0
            // Logic: Tests all components interacting across all 6 faces.
            real_t a = 1.0, b = 0.5, c = 0.25;
            V(0) = a*y - b*z;
            V(1) = b*x - c*z;
            V(2) = c*y - a*x;
        }
        else {
            MFEM_ABORT("Invalid mode for TangentialField3D. Choose 1, 2, or 3.");
        }
    }

    real_t GetExpectedValue() const { return expected_value; }
};

class TangentCoefficient : public VectorCoefficient
{
    mutable Vector tangent; ///< Tangent vector
public:
    TangentCoefficient(int dim) : VectorCoefficient(dim) {}

    /// Evaluate the vector coefficient at @a ip.
    void Eval(Vector &V, ElementTransformation &T,
              const IntegrationPoint &ip) override
    {
        int sdim = T.GetSpaceDim();
        T.SetIntPoint(&ip);
        tangent.SetSize(sdim);

        const DenseMatrix &Jac = T.Jacobian();

        V.SetSize(sdim);

        if (sdim == 2) {
            // 2D: tangent is the first column of the Jacobian
            V(0) = Jac(0, 0);
            V(1) = Jac(1, 0);
        } else if (sdim == 3) {
            // 3D: tangent is the first column of the Jacobian (along xi)
            V(0) = Jac(0, 0);
            V(1) = Jac(1, 0);
            V(2) = Jac(2, 0);
        } else {
            MFEM_ABORT("TangentCoefficient only implemented for 2D or 3D.");
        }

        // Normalize the tangent vector
        V /= V.Norml2();
    }
};

void test_tangential_integrator_2D(int mode, int n_elem, int order) {
    Mesh mesh = Mesh::MakeCartesian2D(n_elem, n_elem, Element::QUADRILATERAL, true, 1.0, 1.0);
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    H1_FECollection fec(order, 2);
    ParFiniteElementSpace fes(&pmesh, &fec, 2); // vector-valued space

    TangentialField2D tangential_field(mode);

    // Set up the boundary linear form
    ParLinearForm blf(&fes);
    // The integrator expects three coefficients: VG, VU, Q
    // The integrator assembles the tangential part of Q * VU (the VG term is
    // dropped, see the note in the integrator), so use VU = tangential_field,
    // Q = 1 and VG = 0.
    Vector zero_vec(2); zero_vec = 0.0;
    ConstantCoefficient one_coeff(1.0);
    VectorConstantCoefficient zero_vec_coeff(zero_vec);

    blf.AddBoundaryIntegrator(
        new VectorBoundaryTangentialAMPLFIntegrator(
            &zero_vec_coeff,   // VG
            &tangential_field, // VU
            &one_coeff         // Q
        )
    );
    blf.Assemble();

    Vector B(fes.GetTrueVSize());
    blf.ParallelAssemble(B);

    real_t local_sum = B.Sum();
    real_t global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

    // Compute expected value analytically for the unit square boundary and compare
    real_t expected = tangential_field.GetExpectedValue();

    MPI_Barrier(pmesh.GetComm()); // Ensure all ranks have printed before final message
    if (Mpi::Root())
        out << "2D Tangential Integrator: computed = " << global_sum << ", expected = " << expected << endl;
    MFEM_VERIFY(fabs(global_sum - expected) < TOL, "Tangential integrator test failed!");
}

void test_tangential_integrator_3D(int mode, int n_elem, int order) {
    Mesh mesh = Mesh::MakeCartesian3D(n_elem, n_elem, n_elem, Element::HEXAHEDRON, 1.0, 1.0, 1.0);
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();
    H1_FECollection fec(order, 3);
    ParFiniteElementSpace fes(&pmesh, &fec, 3); // vector-valued space

    TangentialField3D tangential_field(mode);

    // Set up the boundary linear form
    ParLinearForm blf(&fes);
    Vector zero_vec(3); zero_vec = 0.0;
    VectorConstantCoefficient zero_vec_coeff(zero_vec);
    ConstantCoefficient one_coeff(1.0);

    blf.AddBoundaryIntegrator(
        new VectorBoundaryTangentialAMPLFIntegrator(
            &zero_vec_coeff,   // VG
            &tangential_field, // VU
            &one_coeff         // Q
        )
    );
    blf.Assemble();

    Vector B(fes.GetTrueVSize());
    blf.ParallelAssemble(B);

    real_t local_sum = B.Sum();
    real_t global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh.GetComm());

    // Compute expected value analytically for the unit square boundary and compare
    real_t expected = tangential_field.GetExpectedValue();
    if (Mpi::Root())
        out << "3D Tangential Integrator: computed = " << global_sum << ", expected = " << expected << endl;
    MFEM_VERIFY(fabs(global_sum - expected) < TOL, "Tangential integrator test failed!");
}

void export_field(int dim) {
    Mesh *mesh = nullptr;
    ParMesh *pmesh = nullptr;
    H1_FECollection *fec = nullptr;
    ParFiniteElementSpace *fes = nullptr;
    TangentCoefficient *tangential_field = nullptr;

    if (dim == 2) {
        mesh = new Mesh(Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL, true, 1.0, 1.0));
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        mesh->Clear();
        fec = new H1_FECollection(1, 2);
        fes = new ParFiniteElementSpace(pmesh, fec, 2); // vector-valued space
        tangential_field = new TangentCoefficient(2);
    } else if (dim == 3) {
        mesh = new Mesh(Mesh::MakeCartesian3D(15, 15, 15, Element::HEXAHEDRON, 1.0, 1.0, 1.0));
        pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
        mesh->Clear();
        fec = new H1_FECollection(1, 3);
        fes = new ParFiniteElementSpace(pmesh, fec, 3); // vector-valued space
        tangential_field = new TangentCoefficient(3);
    } else {
        mfem_error("Unsupported dimension for export_field. Choose 2 or 3.");
        return;
    }

    ParGridFunction gf(fes);
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1; // Mark all boundary attributes as essential
    gf.ProjectBdrCoefficient(*tangential_field, ess_bdr);

    ParaViewDataCollection paraview_dc("tangential_field", mesh);
    paraview_dc.RegisterField("tangential_field", &gf);
    paraview_dc.Save();

    delete tangential_field;
    delete fes;
    delete fec;
    delete mesh;
    delete pmesh;
}

int main(int argc, char *argv[]) {

    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();

    int n_elem = 5;
    int order = 1;
    OptionsParser args(argc, argv);
    args.AddOption(&n_elem, "-n", "--num-elements",
                   "Number of elements in each direction for the test meshes.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) for the test meshes.");
    args.Parse();
    
    //int dim = 3; 
    //export_field(dim);

    // 2D Tests
    test_tangential_integrator_2D(1, n_elem, order);
    test_tangential_integrator_2D(2, n_elem, order);
    if (Mpi::Root())
    {        
        out << "Tangential integrator tests 2D passed!" << endl;
        out << endl;
    }
    
    // 3D Tests
    test_tangential_integrator_3D(1, n_elem, order);
    test_tangential_integrator_3D(2, n_elem, order);
    test_tangential_integrator_3D(3, n_elem, order);
    if (Mpi::Root())
        out << "Tangential integrator tests 3D passed!" << endl;
    return 0;
}