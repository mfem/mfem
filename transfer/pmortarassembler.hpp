#ifndef MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP
#define MFEM_L2P_PAR_MORTAR_ASSEMBLER_HPP

#include <memory>
#include <vector>

#include "../fem/fem.hpp"
#include "mortarintegrator.hpp"


namespace mfem
{

/*!
 * @brief This class implements the parallel variational transfer between finite element spaces.
 * Variational transfer has been shown to have better approximation properties than standard interpolation.
 * This facilities can be used for supporting applications wich require the handling of non matching meshes.
 * For instance: General multi-physics problems, fluid structure interaction, or even visulization of average
 * quanties within subvolumes.
 * This particular code is also used with LLNL for large scale multilevel Monte Carlo simulations.
 */
class ParMortarAssembler
{
public:

   /*!
    * @brief constructs the object with source and destination spaces
    * @param source the source space from where we want to transfer the discrete field
    * @param destination the source space to where we want to transfer the discrete field
    */
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

   /*!
    * @brief transfer a function from source to destination. if the transfer is to be performed multiple times use Assemble instead
    * @param src_fun the function associated with the source finite element space
    * @param[out] dest_fun the function associated with the destination finite element space
    * @param is_vector_fe set to true if vector FEM are used
    * @return true if there was an intersection and the output can be used.
    */
   bool Transfer(ParGridFunction &src_fun, ParGridFunction &dest_fun,
                 bool is_vector_fe = false);


   /*!
    * @brief This method must be called before Assemble or Transfer.
    * It will assemble the operator in all intersections found.
    * @param integrator the integrator object
    */
   inline void AddMortarIntegrator(const std::shared_ptr<MortarIntegrator>
                                   &integrator)
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
