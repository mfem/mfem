#ifndef MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP
#define MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP

#include <memory>
#include <vector>

#include "../fem/fem.hpp"
#include "mortarintegrator.hpp"


namespace mfem {

	class ParMortarAssembler {
	public:

		ParMortarAssembler(
			const MPI_Comm comm,
			const std::shared_ptr<ParFiniteElementSpace> &source,
			const std::shared_ptr<ParFiniteElementSpace> &destination);

		/*!
		 * @brief assembles the coupling matrix B. B : source -> destination If u is a coefficient
		 * associated with source and v with destination
		 * Then v = M^(-1) * B * u; where M is the mass matrix in destination.
		 * Works with L2_FECollection, H1_FECollection and DG_FECollection (experimental with RT_FECollection and ND_FECollection).
		 * @param B the assembled coupling operator. B can be passed uninitialized.
		 * @return true if there was an intersection and the operator has been assembled. False otherwise.
		 */
		 bool Assemble(std::shared_ptr<HypreParMatrix> &B);

		///@brief if the transfer is to be performed multiple times use Assemble instead
		 bool Transfer(ParGridFunction &src_fun, ParGridFunction &dest_fun, bool is_vector_fe = false);

		 inline void AddMortarIntegrator(const std::shared_ptr<MortarIntegrator> &integrator)
		 {
		 	integrators_.push_back(integrator);
		 }

	private:
		MPI_Comm comm_;
		std::shared_ptr<ParFiniteElementSpace> source_;
		std::shared_ptr<ParFiniteElementSpace> destination_;
		std::vector< std::shared_ptr<MortarIntegrator> > integrators_;
	};

}

#endif //MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP
