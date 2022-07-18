#include "material_metrics.hpp"

namespace mfem {

double ParticleTopology::ComputeMetric(const Vector &x){
   std::vector<double> dist_vector;
   dist_vector.resize(particle_positions_.size());
   // 1. Compute the distance to each particle.
   for (int i = 0; i < particle_positions_.size(); i++)
   {
      Vector y (3);
      particle_orientations_[i].Mult(x, y);
      dist_vector[i] = particle_positions_[i].DistanceTo(y);
   }
   // 2. Choose smallest number in the vector dist_vector.
   double min_dist = *std::min_element(dist_vector.begin(), dist_vector.end());
   return min_dist;
}

void ParticleTopology::Initialize(std::vector<double> &random_positions, 
                                        std::vector<double> &random_rotations) {
   // 1. Initialize the particle positions.
   particle_positions_.resize(number_of_particles_);
   particle_orientations_.resize(number_of_particles_);
   for (int i = 0; i < number_of_particles_; i++)
   {
      // 2.1 Read the positions.
      int idx_pos = i * 3;
      Vector particle_position({random_positions[idx_pos], 
                                random_positions[idx_pos + 1], 
                                random_positions[idx_pos + 2]});

      // 2.2 Read the random rotations.
      int idx_rot = i * 9;
      DenseMatrix R(3,3);
      R(0,0) = random_rotations[idx_rot + 0];
      R(0,1) = random_rotations[idx_rot + 1];
      R(0,2) = random_rotations[idx_rot + 2];
      R(1,0) = random_rotations[idx_rot + 3];
      R(1,1) = random_rotations[idx_rot + 4];
      R(1,2) = random_rotations[idx_rot + 5];
      R(2,0) = random_rotations[idx_rot + 6];
      R(2,1) = random_rotations[idx_rot + 7];
      R(2,2) = random_rotations[idx_rot + 8];

      // 2.3 Fill the orientation vector.
      DenseMatrix res(3,3);
      MultADBt(R, particle_shape_, R, res);
      particle_orientations_[i] = res;

      // 2.4 Scale position for distance metric
      Vector scaled_position(3);
      res.Mult(particle_position, scaled_position);
      particle_positions_[i] = scaled_position;
   }
}

double Edge::GetDistanceTo(const Vector& x) const {
   // Implements formula used in [2, Example 5].
   const double a = start_.DistanceTo(x);
   const double b = end_.DistanceTo(x);
   const double c = start_.DistanceTo(end_);
   const double s1 = (pow(a,2) + pow(b,2)) / 2;
   const double s2 = pow(c,2) / 4;
   const double s3 = pow((pow(a,2) - pow(b,2))/(2*c),2);
   return sqrt(abs(s1 - s2 - s3));
}

double OctetTrussTopology::ComputeMetric(const Vector &x) {
      // 1. Fill a vector with x and it's ghost points mimicking the periodicity
      //    of the topology.
      std::vector<Vector> periodic_points;
      CreatePeriodicPoints(x, periodic_points);
      std::vector<double> dist_vector;

      // 2. Compute the distance to each periodic points to all edges.
      for (auto point : periodic_points) {
         for (auto edge : edges_){
            dist_vector.push_back(edge.GetDistanceTo(point));
         }
      }
      // 3. Choose the smallest number in the vector dist_vector.
      double min_dist = *std::min_element(dist_vector.begin(), 
                                          dist_vector.end());
      return min_dist;
   }

void OctetTrussTopology::Initialize() {
   // 1. Create the points defining the topology (begin and end points of the
   //    edges).
   Vector p1 ({0,0,0});
   Vector p2 ({0,1,1});
   Vector p3 ({1,0,1});
   Vector p4 ({1,1,0});
   points_.push_back(p1);
   points_.push_back(p2);
   points_.push_back(p3);
   points_.push_back(p4);

   // 2. Create the edges.
   for (size_t i = 0; i < points_.size(); i++){
      for (size_t j = i+1; j < points_.size(); j++){
         Edge edge(points_[i],points_[j]);
         edges_.push_back(edge);
      }
   }
}

void OctetTrussTopology::CreatePeriodicPoints(const Vector&x, 
                           std::vector<Vector>& periodic_points) {
   Vector xx (x);
   // Compute the diplaced ghost points. Computation assumes domain [0,1]^3.
   Vector dispcement_x ({1,0,0});
   Vector dispcement_y ({0,1,0});
   Vector dispcement_z ({0,0,1});
   Vector x_shifted_x_pos = x;
   x_shifted_x_pos += dispcement_x;
   Vector x_shifted_x_neg = x;
   x_shifted_x_neg -= dispcement_x;
   Vector x_shifted_y_pos = x;
   x_shifted_y_pos += dispcement_y;
   Vector x_shifted_y_neg = x;
   x_shifted_y_neg -= dispcement_y;
   Vector x_shifted_z_pos = x;
   x_shifted_z_pos += dispcement_z;
   Vector x_shifted_z_neg = x;
   x_shifted_z_neg -= dispcement_z;
   // Fill the vector with all relevant points
   periodic_points.push_back(xx);
   periodic_points.push_back(x_shifted_x_pos);
   periodic_points.push_back(x_shifted_x_neg);
   periodic_points.push_back(x_shifted_y_pos);
   periodic_points.push_back(x_shifted_y_neg);
   periodic_points.push_back(x_shifted_z_pos);
   periodic_points.push_back(x_shifted_z_neg);
}

} // namespace mfem
