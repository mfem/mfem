#ifndef MFEM_ELASTICITY_UTILS_HPP
#define MFEM_ELASTICITY_UTILS_HPP

#include "mfem.hpp"

namespace mfem
{
    namespace elasticity
    {

        using VecFuncT = void(const Vector &x, real_t t, Vector &u);
        using ScalarFuncT = real_t(const Vector &x, real_t t);

        class FixedConstraint
        {
        public:
            FixedConstraint(const Array<int> &attr, int component_ = -1) : attr(attr), component(component_) {}

            FixedConstraint(const FixedConstraint &other)
                : attr(other.attr), component(other.component) {}

            FixedConstraint(FixedConstraint &&other) noexcept
                : attr(std::move(other.attr)), component(other.component) {}

            Array<int> attr; ///< Array marking the boundary attributes for fixed constraints.
            int component;   ///< Component of the field to which the fixed boundary condition applies (-1 for all components).
        };


        /**
         * @brief DerivedAccelerationCoefficient computes the acceleration consistent with a prescribed velocity from discrete data.
         * This is used to apply a Dirichlet boundary condition on the velocity in an implicit time integration scheme.
         * The derived acceleration is computed based on the time integration method and the provided prescribed velocity, previous velocity and previous acceleration.
         */
        class DerivedAccelerationCoefficient : public VectorCoefficient
        {
        public:
            DerivedAccelerationCoefficient(GridFunction *prescribed_velocity_gf, GridFunction *previous_velocity_gf, GridFunction *previous_acceleration_gf, SecondOrderODESolver *time_integrator_)
                : VectorCoefficient(prescribed_velocity_gf->FESpace()->GetVDim()),
                  time_integrator(time_integrator_)
            {
                // Create VectorGridFunctionCoefficients for the prescribed velocity, previous velocity and previous acceleration
                prescribed_velocity_coeff = new VectorGridFunctionCoefficient(prescribed_velocity_gf);
                previous_velocity_coeff = new VectorGridFunctionCoefficient(previous_velocity_gf);
                previous_acceleration_coeff = new VectorGridFunctionCoefficient(previous_acceleration_gf);

                // Get the time factors from the time integrator
                time_integrator->GetFactors(time_factors);

                // Initialize the temporary vectors used in Eval
                u_bc.SetSize(vdim);
                un.SetSize(vdim);
                an.SetSize(vdim);
            }

            ~DerivedAccelerationCoefficient()
            {
                delete prescribed_velocity_coeff;
                delete previous_velocity_coeff;
                delete previous_acceleration_coeff;
            }

            VectorCoefficient *GetPrescribedVelocityCoefficient() const { return prescribed_velocity_coeff; }
            VectorCoefficient *GetPreviousVelocityCoefficient() const { return previous_velocity_coeff; }
            VectorCoefficient *GetPreviousAccelerationCoefficient() const { return previous_acceleration_coeff; }

            void SetTimeStep(real_t dt)
            {
                cached_dt = dt;
            }

            // Given a prescribed discrete velocity, the derived acceleration depends on the time integration method.
            void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip) override
            {
                if (prescribed_velocity_coeff == nullptr || previous_velocity_coeff == nullptr || previous_acceleration_coeff == nullptr || time_integrator == nullptr)
                {
                    MFEM_ABORT("DerivedAccelerationCoefficient requires prescribed velocity, previous velocity, previous acceleration and time integrator to be set.");
                }

                V.SetSize(vdim);

                // Evaluate the grid functions at the current integration point
                prescribed_velocity_coeff->Eval(u_bc, T, ip);
                previous_velocity_coeff->Eval(un, T, ip);
                previous_acceleration_coeff->Eval(an, T, ip);

                if (dynamic_cast<NewmarkSolver *>(time_integrator) != nullptr)
                {
                    // Extract the time factors from the time integrator
                    real_t alpha_2 = time_factors[1];
                    real_t alpha_4 = time_factors[3];

                    for (int i = 0; i < GetVDim(); i++)
                    {
                        V(i) = 1.0 / alpha_4 * (u_bc(i) - un(i)) / cached_dt - alpha_2 / alpha_4 * an(i);
                    }
                }
                else if (dynamic_cast<GeneralizedAlpha2Solver *>(time_integrator) != nullptr)
                {
                    // Extract the time factors from the time integrator
                    real_t alpha_1 = time_factors[1];
                    real_t alpha_2 = time_factors[2];
                    real_t alpha_4 = time_factors[4];

                    for (int i = 0; i < GetVDim(); i++)
                    {
                        V(i) = alpha_1 / alpha_4 * (u_bc(i) - un(i)) / cached_dt - alpha_2 / alpha_4 * an(i);
                    }
                }
                else
                {
                    MFEM_ABORT("Unsupported time integrator for DerivedAccelerationCoefficient.");
                }
            }

        private:
            VectorGridFunctionCoefficient *prescribed_velocity_coeff;
            VectorGridFunctionCoefficient *previous_velocity_coeff;
            VectorGridFunctionCoefficient *previous_acceleration_coeff;

            SecondOrderODESolver *time_integrator;
            std::vector<real_t> time_factors;

            mutable Vector u_bc, un, an;
            real_t cached_dt;
        };

        /**
         * @brief Container for a Robin boundary condition of the form:
         *   alpha * du/dt + sigma*n = g
         *
         * Holds alpha (as a ConstantCoefficient for bilinear form assembly) and
         * the forcing vector g. Used by ElasticityOperator to build M_gamma and
         * the Robin forcing contribution to the RHS.
         */
        class RobinBC_T
        {
        public:
            RobinBC_T(const Array<int> &attr_, real_t alpha_, VectorCoefficient *g_, bool own_ = true)
                : attr(attr_), g(g_), own(own_)
            {
                alpha_coef = new ConstantCoefficient(alpha_);
            }

            RobinBC_T(RobinBC_T &&obj) noexcept
                : attr(std::move(obj.attr)), g(obj.g), alpha_coef(obj.alpha_coef), own(obj.own)
            {
                obj.g         = nullptr;
                obj.alpha_coef = nullptr;
            }

            ~RobinBC_T()
            {
                delete alpha_coef;
                alpha_coef = nullptr;
                if (own) { delete g; g = nullptr; }
            }

            void SetAlpha(real_t alpha_) { alpha_coef->constant = alpha_; }
            real_t GetAlpha() const      { return alpha_coef->constant; }
            void SetTime(real_t t)       { if (g) { g->SetTime(t); } }

            Array<int>             attr;                    ///< Boundary attributes.
            VectorCoefficient     *g          = nullptr;    ///< Robin forcing (e.g. alpha*u_f + sigma_f*n). May be OWNED.
            ConstantCoefficient   *alpha_coef = nullptr;    ///< Scalar Robin penalty coefficient alpha. OWNED.
            bool                   own;
        };

    } // namespace elasticity
} // namespace mfem

#endif // MFEM_ELASTICITY_UTILS_HPP