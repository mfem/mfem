
#include <assert.h>
#include "HashGrid.hpp"


namespace mfem {
	long HashGrid::Hash(const Vector &point) const
	{
		double x = (point(0) - box_.GetMin(0))/range_(0);
		long result = floor(x * dims_[0]);

		long totalDim = dims_[0];

		for(int i = 1; i < range_.Size(); ++i) {
			result *= dims_[i];

			x = (point(i) - box_.GetMin(i))/range_(i);
			result += floor(x * dims_[i]);
			totalDim *= dims_[i];
		}

		if(result >= totalDim || result < 0) {
			printf("error -> %d\n", result);
		}

		return result;
	}

	long HashGrid::Hash(const std::vector<long> &coord) const
	{
		long result   = coord[0];
		long totalDim = dims_[0];

		for(int i = 1; i < range_.Size(); ++i) {
			result *= dims_[i];
			result += coord[i];
			totalDim *= dims_[i];
		}

		return result;
	}

	void HashGrid::HashRange(const Vector &min, const Vector &max, std::vector<long> &hashes)
	{
		const int dim = min.Size();
		std::vector<long> imin(dim), imax(dim);

			//generate tensor indices
		for(int i = 0; i < dim; ++i) {
			double x = (min(i) - box_.GetMin(i))/range_(i);
			imin[i] = floor(x * dims_[i]);
		}

		for(int i = 0; i < dim; ++i) {
			double x = (max(i) - box_.GetMin(i))/range_(i);
			imax[i] = floor(x * dims_[i]);
		}

		std::vector<long> offsets(dim);
		for(int i = 0; i < dim; ++i) {
			offsets[i] = imax[i] - imin[i];
		}

			//FIXME make more general for greater dim
		if(dim == 1) {
			std::vector<long> coord(1);
			for(int i = imin[0]; i <= imax[0]; ++i) {
				coord[0] = i;
				hashes.push_back(Hash(coord)); 
			}

		} else if(dim == 2) {
			std::vector<long> coord(2);
			for(int i = imin[0]; i <= imax[0]; ++i) {
				for(int j = imin[1]; j <= imax[1]; ++j) {
					coord[0] = i;
					coord[1] = j;
					hashes.push_back(Hash(coord)); 
				}
			}
		} else if(dim == 3) {
			std::vector<long> coord(3);
			for(int i = imin[0]; i <= imax[0]; ++i) {
				for(int j = imin[1]; j <= imax[1]; ++j) {
					for(int k = imin[2]; k <= imax[2]; ++k) {
						coord[0] = i;
						coord[1] = j;
						coord[2] = k;
						hashes.push_back(Hash(coord)); 
					}
				}
			}
		} else {
			assert(false && "dim > 3 not supported yet!");
		}
	}

	HashGrid::HashGrid(const Box &box, const std::vector<int> &dims)
	: box_(box), dims_(dims), n_cells_(1)
	{
		box_.Enlarge(1e-8);

		range_  = box_.GetMax();
		range_ -= box_.GetMin();

		for(int i = 0; i < dims_.size(); ++i) {
			n_cells_ *= dims_[i];
		}
	}

	void BuildBoxes(const Mesh &mesh, std::vector<Box> &element_boxes, Box &mesh_box)
	{
		const int dim = mesh.Dimension();
		element_boxes.resize(mesh.GetNE());
		mesh_box.Reset(dim);

		DenseMatrix pts;
		for(int i = 0; i < mesh.GetNE(); ++i) {
			mesh.GetPointMatrix(i, pts);
			element_boxes[i].Reset(dim);
			element_boxes[i] += pts;
			mesh_box += element_boxes[i];
		}
	}

	bool HashGridDetectIntersections(const Mesh &src, const Mesh &dest, std::vector<int> &pairs)
	{
		const int dim = dest.Dimension();
		Box src_box(dim);
		Box dest_box(dim);
		
		std::vector<Box> src_boxes;
		std::vector<Box> dest_boxes;

		BuildBoxes(src,  src_boxes,  src_box);
		BuildBoxes(dest, dest_boxes, dest_box);

		const int n_x_dim = std::max(1, int(pow(src.GetNE(), 1./dim)));	
		std::vector<int> dims(dim);
		for(int i = 0; i < dim; ++i) {
			dims[i] = n_x_dim;
		}

		HashGrid hgrid(src_box, dims);
		std::vector< std::vector<int> > src_table(hgrid.NCells());

		std::vector<long> hashes;
		for(int i = 0; i < src.GetNE(); ++i) {
			const Box &b = src_boxes[i];
			hashes.clear();
			hgrid.HashRange(b.GetMin(), b.GetMax(), hashes);

			for(auto h : hashes) {
				src_table[h].push_back(i);
			}
		}

		std::vector<int> candidates;
		for(int i = 0; i < dest.GetNE(); ++i) {
			const Box &b = dest_boxes[i];

			hashes.clear();
			hgrid.HashRange(b.GetMin(), b.GetMax(), hashes);

			candidates.clear();

			for(auto h : hashes) {
				if(h < 0) continue;
				
				for(int j : src_table[h]) {
					const Box &bj = src_boxes[j];

					if(b.Intersects(bj)) {
						candidates.push_back(j);
					}
				}
			}

			std::sort(candidates.begin(), candidates.end());
			auto last = std::unique(candidates.begin(), candidates.end());
			candidates.erase(last, candidates.end());

			for(auto c : candidates) {
				pairs.push_back(c);
				pairs.push_back(i);
				
			}
		}

		return !pairs.empty();
	}

	bool DetectIntersections(const Mesh &src, const Mesh &dest, std::vector<int> &pairs)
	{
		const int src_dim  = src.Dimension();
		const int dest_dim = dest.Dimension();

		assert(src_dim == dest_dim && "must have same dimensions");

		DenseMatrix src_pts;
		DenseMatrix dest_pts;

		Box src_box (src_dim);
		Box dest_box(dest_dim);

		for(int i = 0; i < dest.GetNE(); ++i) {
			dest.GetPointMatrix(i, dest_pts);
			dest_box.Reset();
			dest_box += dest_pts;

			for(int j = 0; j < src.GetNE(); ++j) {
				src.GetPointMatrix(j, src_pts);
				src_box.Reset();
				src_box += src_pts;


				if(dest_box.Intersects(src_box)) {
					pairs.push_back(j);
					//destination is contiguous
					pairs.push_back(i);
				}
			}
		}

		return !pairs.empty();
	}
}