#include "mfem.hpp"
#include "amr.hpp"

using namespace mfem;
using namespace std;


RegionalThresholdRefiner::RegionalThresholdRefiner(ErrorEstimator &est)
  : estimator(est)
{
  aniso_estimator = dynamic_cast<AnisotropicErrorEstimator*>(&estimator);
  total_norm_p = infinity();
  total_err_goal = 0.0;
  total_fraction = 0.5;
  local_err_goal = 0.0;
  max_elements = std::numeric_limits<long long>::max();
  amr_levels=max_elements;
  xRange_levels=max_elements; //if xRange_levels is trun on, it will ignore xRang if levels<xRange_levels
  xRange=false;
  yRange_levels=max_elements;
  yRange=false;
  xmax=std::numeric_limits<double>::max();
  ymax=xmax;
  xmin=std::numeric_limits<double>::lowest();
  ymin=xmin;

  threshold = 0.0;
  num_marked_elements = 0LL;
  current_sequence = -1;

  non_conforming = -1;
  nc_limit = 0;
}

int RegionalThresholdRefiner::ApplyRef(Mesh &mesh, int attrib_select, double total_fraction_in, double total_fraction_out)
{
  threshold = 0.0;
  double threshold_in = 0.0;
  double threshold_out = 0.0;
  num_marked_elements = 0LL;
  marked_elements.SetSize(0);
  current_sequence = mesh.GetSequence();

  const long long num_elements = mesh.GetGlobalNE();
  if (num_elements >= max_elements) { return STOP; }

  const int NE = mesh.GetNE();
  const Vector &local_err = estimator.GetLocalErrors();
  MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

  double vert[3];
  double yMean, xMean;
  long elementLevel;

  const double total_err = GetNorm(local_err, mesh);
  if (total_err <= total_err_goal) { return STOP; }

  if (total_norm_p < infinity())
    {
      threshold_in = std::max(total_err * total_fraction_in *
                           std::pow(num_elements, -1.0/total_norm_p),
                           local_err_goal);
      threshold_out = std::max(total_err * total_fraction_out *
                           std::pow(num_elements, -1.0/total_norm_p),
                           local_err_goal);
    }
  else
    {
      threshold_in = std::max(total_err * total_fraction_in, local_err_goal);
      threshold_out = std::max(total_err * total_fraction_out, local_err_goal);
    }


  for (int el = 0; el < NE; el++)
    { 
      const int attrib = mesh.GetElement(el)->GetAttribute();

      if (attrib == attrib_select) {
        // amr_levels = levels_inside;
        threshold = threshold_in;
      } else {
        // amr_levels = levels_outside;
        threshold = threshold_out;
      }
      // if (attrib != attrib_select) {
      //   continue;
      // }
      
      if ((yRange || xRange) && mesh.Nonconforming())
        {
          FiniteElementSpace * fes = mesh.GetNodes()->FESpace();
          Array<int> dofs;
          fes->GetElementDofs(el, dofs);
          int ndof=dofs.Size();
          yMean=0.0;
          xMean=0.0;
          for (int j = 0; j < ndof; j++)
            {
              mesh.GetNode(dofs[j], vert);
              yMean+=vert[1];
              xMean+=vert[0];
            }
          yMean=yMean/ndof;
          xMean=xMean/ndof;

          elementLevel=mesh.ncmesh->GetElementDepth(el);
        
          if (local_err(el) > threshold && elementLevel < amr_levels &&
              mesh.ncmesh->GetElementDepth(el) < amr_levels && 
              ((yMean>ymin && yMean<ymax) || elementLevel<yRange_levels) &&
              ((xMean>xmin && xMean<xmax) || elementLevel<xRange_levels)
              )
            {
              marked_elements.Append(Refinement(el));
            }

        }
      else if (mesh.Nonconforming())
        {
          //std::cout <<"el="<<el<<" level="<<mesh.ncmesh->GetElementDepth(el)<< '\n';
          if (local_err(el) > threshold && 
              mesh.ncmesh->GetElementDepth(el) < amr_levels)
            {
              marked_elements.Append(Refinement(el));
            }
        }
      else
        {
          if (local_err(el) > threshold )
            {
              marked_elements.Append(Refinement(el));
            }
        }
    }

  if (aniso_estimator)
    {
      const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
      if (aniso_flags.Size() > 0)
        {
          for (int i = 0; i < marked_elements.Size(); i++)
            {
              Refinement &ref = marked_elements[i];
              ref.ref_type = aniso_flags[ref.index];
            }
        }
    }

  num_marked_elements = mesh.ReduceInt(marked_elements.Size());
  if (num_marked_elements == 0LL) { return STOP; }

  mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
  return CONTINUE + REFINED;
}


double RegionalThresholdRefiner::GetNorm(const Vector &local_err, Mesh &mesh) const
{
#ifdef MFEM_USE_MPI
  ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
  if (pmesh)
    {
      return ParNormlp(local_err, total_norm_p, pmesh->GetComm());
    }
#endif
  return local_err.Normlp(total_norm_p);
}

void RegionalThresholdRefiner::Reset()
{
  estimator.Reset();
  current_sequence = -1;
  num_marked_elements = 0LL;
}
