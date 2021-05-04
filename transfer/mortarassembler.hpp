#ifndef MFEML2P_MORTAR_ASSEMBLER_HPP
#define MFEML2P_MORTAR_ASSEMBLER_HPP

#include <memory>
#include "../fem/fem.hpp"
#include "mortarintegrator.hpp"


namespace mfem {

	class MortarAssembler {
	public:

		MortarAssembler(
			const std::shared_ptr<FiniteElementSpace> &source,
			const std::shared_ptr<FiniteElementSpace> &destination);

		/*!
		 * @brief assembles the coupling matrix B. B : source -> destination If u is a coefficient
		 * associated with source and v with destination
		 * Then v = M^(-1) * B * u; where M is the mass matrix in destination.
		 * Works with L2_FECollection, H1_FECollection and DG_FECollection (experimental with RT_FECollection and ND_FECollection).
		 * @param B the assembled coupling operator. B can be passed uninitialized.
		 * @return true if there was an intersection and the operator has been assembled. False otherwise.
		 */
		 bool Assemble(std::shared_ptr<SparseMatrix> &B);

		///@brief if the transfer is to be performed multiple times use Assemble instead
		 bool Transfer(GridFunction &src_fun, GridFunction &dest_fun, bool is_vector_fe = false);

		 inline void AddMortarIntegrator(const std::shared_ptr<MortarIntegrator> &integrator)
		 {
		 	integrators_.push_back(integrator);
		 }

		private:
			MPI_Comm comm_;
			std::shared_ptr<FiniteElementSpace> source_;
			std::shared_ptr<FiniteElementSpace> destination_;
			std::vector< std::shared_ptr<MortarIntegrator> > integrators_;
	};

}


#endif //MFEML2P_MORTAR_ASSEMBLER_HPP
