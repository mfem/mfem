

#include "Refiners.hpp"

#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iterator>


namespace mfem {

	MaximumMarkingRefiner::MaximumMarkingRefiner(ErrorEstimator &est) :
		estimator( est )
	{
		gamma = .5;
		max_elements = std::numeric_limits<long>::max();

		threshold = 0.0;
		num_marked_elements = 0L;
		current_sequence = -1;

		non_conforming = -1;
		nc_limit = 0;
	};

	int MaximumMarkingRefiner::ApplyImpl(Mesh &mesh)
	{
		num_marked_elements = 0;
		marked_elements.SetSize(0);
		current_sequence = mesh.GetSequence();

		const long num_elements = mesh.GetGlobalNE();
		if (num_elements >= max_elements) { return STOP; }

		const int NE = mesh.GetNE();
		const Vector &local_err = estimator.GetLocalErrors();
		MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

		threshold = gamma * local_err.Max();	

		for (int el = 0; el < NE; el++)
		{
			if (local_err(el) > threshold)
			{
				marked_elements.Append(Refinement(el));
			}
		}

		num_marked_elements = mesh.ReduceInt(marked_elements.Size());
		if (num_marked_elements == 0) { return STOP; }

		mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
		return CONTINUE + REFINED;
	}

	/* Implementation from Pfeiler and Praetorius ( arXiv:1907.13078 ) 
	 * with minor tweak to work with mfem::Vector as the original data structure.
	 */

	const double DoerflerMarkingRefiner::xStarKernel( Iterator_t subX_begin, Iterator_t subX_end, double goal )
	{
		// QuickMark , step ( i )-( ii ): partition by median element
		auto length { std::distance( subX_begin , subX_end )};
		auto subX_middle { subX_begin + length /2 };
		std::nth_element( subX_begin, subX_middle, subX_end, std::greater<double>());
		auto pivot_val {* subX_middle };
		// QuickMark , step ( iii )
		auto sigma_g = std::accumulate( subX_begin, subX_middle, ( double )0.0);
		// QuickMark , step ( iv ), ( v ) and ( vi )
		if ( sigma_g >= goal )
			return xStarKernel( subX_begin, subX_middle, goal );
		if ( sigma_g + pivot_val >= goal )
			return pivot_val ;
		return xStarKernel(++subX_middle, subX_end, goal - sigma_g - pivot_val );
	}

	const double DoerflerMarkingRefiner::compute_threshold ( const mfem::Vector & eta , double theta )
	{
		std::vector<double> x { eta.GetData(), eta.GetData() + eta.Size() };
		double goal = theta * std ::accumulate( x.cbegin(), x.cend(), ( double )0.0);
		return xStarKernel( x.begin(), x.end(), goal);
	} 

	int DoerflerMarkingRefiner::ApplyImpl(Mesh &mesh)
	{
		threshold = 0.0;
		num_marked_elements = 0;
		marked_elements.SetSize(0);
		current_sequence = mesh.GetSequence();

		const long num_elements = mesh.GetGlobalNE();
		if (num_elements >= max_elements) { return STOP; }

		const int NE = mesh.GetNE();
		const Vector &local_err = estimator.GetLocalErrors();
		MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

		double threshold = compute_threshold( local_err, gamma );

		for (int el = 0; el < NE; el++)
		{
			if (local_err(el) >= threshold)
			{
				marked_elements.Append(Refinement(el));
			}
		}

		num_marked_elements = mesh.ReduceInt(marked_elements.Size());
		MFEM_ASSERT( num_marked_elements == n, "Marking algorithm n is not the same as number of actually marked elements" );
		if (num_marked_elements == 0) { return STOP; }

		mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
		return CONTINUE + REFINED;
	}
}
