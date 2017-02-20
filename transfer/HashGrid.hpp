#ifndef MFEM_L2P_HASH_GRID_HPP
#define MFEM_L2P_HASH_GRID_HPP 

#include <vector>
#include "../fem/fem.hpp"
#include "Box.hpp"



namespace mfem {

	class HashGrid {
	public:

		long Hash(const Vector &point) const;
		long Hash(const std::vector<long> &coord) const;
		void HashRange(const Vector &min, const Vector &max, std::vector<long> &hashes);
		HashGrid(const Box &box, const std::vector<int> &dims);

		inline long NCells() const
		{
			return n_cells_;
		}

	private:
		Box box_;
		Vector range_;
		std::vector<int> dims_;
		long n_cells_;
	};

	void BuildBoxes(const Mesh &mesh, std::vector<Box> &element_boxes, Box &mesh_box);
	bool HashGridDetectIntersections(const Mesh &src, const Mesh &dest, std::vector<int> &pairs);
	
	/// @brief Inefficient n^2 algorithm. Do not use unless it is for test purposes
	bool DetectIntersections(const Mesh &src, const Mesh &dest, std::vector<int> &pairs);
}

#endif //MFEM_L2P_HASH_GRID_HPP
