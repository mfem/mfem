// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

#include "material_metrics.hpp"

namespace mfem
{

real_t ParticleTopology::ComputeMetric(const Vector &x)
{
   std::vector<real_t> dist_vector;
   dist_vector.resize(particle_positions_.size());
   // 1. Compute the distance to each particle.
   for (size_t i = 0; i < particle_positions_.size(); i++)
   {
      Vector y(3);
      particle_orientations_[i].Mult(x, y);
      dist_vector[i] = particle_positions_[i].DistanceTo(y);
   }
   // 2. Choose smallest number in the vector dist_vector.
   real_t min_dist = *std::min_element(dist_vector.begin(), dist_vector.end());
   return min_dist;
}

void ParticleTopology::Initialize(std::vector<real_t> &random_positions,
                                  std::vector<real_t> &random_rotations)
{
   // 1. Initialize the particle positions.
   particle_positions_.resize(number_of_particles_);
   particle_orientations_.resize(number_of_particles_);
   for (size_t i = 0; i < number_of_particles_; i++)
   {
      // 2.1 Read the positions.
      size_t idx_pos = i * 3;
      Vector particle_position({random_positions[idx_pos],
                                random_positions[idx_pos + 1],
                                random_positions[idx_pos + 2]});

      // 2.2 Read the random rotations.
      size_t idx_rot = i * 9;
      DenseMatrix R(3, 3);
      R(0, 0) = random_rotations[idx_rot + 0];
      R(0, 1) = random_rotations[idx_rot + 1];
      R(0, 2) = random_rotations[idx_rot + 2];
      R(1, 0) = random_rotations[idx_rot + 3];
      R(1, 1) = random_rotations[idx_rot + 4];
      R(1, 2) = random_rotations[idx_rot + 5];
      R(2, 0) = random_rotations[idx_rot + 6];
      R(2, 1) = random_rotations[idx_rot + 7];
      R(2, 2) = random_rotations[idx_rot + 8];

      // 2.3 Fill the orientation vector.
      DenseMatrix res(3, 3);
      MultADBt(R, particle_shape_, R, res);
      particle_orientations_[i] = res;

      // 2.4 Scale position for distance metric
      Vector scaled_position(3);
      res.Mult(particle_position, scaled_position);
      particle_positions_[i] = scaled_position;
   }
}

real_t Edge::GetDistanceTo(const Vector &x) const
{
   // Implements formula used in [2, Example 5].
   const real_t a = start_.DistanceTo(x);
   const real_t b = end_.DistanceTo(x);
   const real_t c = start_.DistanceTo(end_);
   const real_t s1 = (pow(a, 2) + pow(b, 2)) / 2;
   const real_t s2 = pow(c, 2) / 4;
   const real_t s3 = pow((pow(a, 2) - pow(b, 2)) / (2 * c), 2);
   return sqrt(abs(s1 - s2 - s3));
}

real_t OctetTrussTopology::ComputeMetric(const Vector &x)
{
   // Define the point in the vector which differentiates between the periodic
   // points and the inner points.
   constexpr size_t periodic_edges = 6;

   // 1. Fill a vector with x and it's ghost points mimicking the periodicity
   //    of the topology.
   std::vector<Vector> periodic_points;
   CreatePeriodicPoints(x, periodic_points);
   std::vector<real_t> dist_vector;

   // 2. Compute the distance to each periodic points to the outer edges.
   for (const auto &point : periodic_points)
   {
      for (size_t i = 0; i < periodic_edges; i++)
      {
         dist_vector.push_back(edges_[i].GetDistanceTo(point));
      }
   }

   // 3. Add distance between x and the remaining inner edges
   for (size_t i = periodic_edges; i < edges_.size(); i++)
   {
      dist_vector.push_back(edges_[i].GetDistanceTo(x));
   }

   // 3. Choose the smallest number in the vector dist_vector.
   real_t min_dist = *std::min_element(dist_vector.begin(), dist_vector.end());
   return min_dist;
}

void OctetTrussTopology::Initialize()
{
   // 1. Create the points defining the topology (begin and end points of the
   //    edges). Outer structure
   real_t p1_data[3] = {0, 0, 0};
   real_t p2_data[3] = {0, 1, 1};
   real_t p3_data[3] = {1, 0, 1};
   real_t p4_data[3] = {1, 1, 0};

   Vector p1(p1_data, 3);
   Vector p2(p2_data, 3);
   Vector p3(p3_data, 3);
   Vector p4(p4_data, 3);

   points_.push_back(p1);
   points_.push_back(p2);
   points_.push_back(p3);
   points_.push_back(p4);

   // 2. Create the inner structure
   real_t p5_data[3] = {0, 0.5, 0.5};   // left
   real_t p6_data[3] = {1, 0.5, 0.5};   // right
   real_t p7_data[3] = {0.5, 0, 0.5};   // bottom
   real_t p8_data[3] = {0.5, 1, 0.5};   // top
   real_t p9_data[3] = {0.5, 0.5, 0};   // front
   real_t p10_data[3] = {0.5, 0.5, 1};  // back

   Vector p5(p5_data, 3);
   Vector p6(p6_data, 3);
   Vector p7(p7_data, 3);
   Vector p8(p8_data, 3);
   Vector p9(p9_data, 3);
   Vector p10(p10_data, 3);

   points_.push_back(p5);
   points_.push_back(p6);
   points_.push_back(p7);
   points_.push_back(p8);
   points_.push_back(p9);
   points_.push_back(p10);

   // 3. Create the outer edges.
   for (size_t i = 0; i < 4; i++)
   {
      for (size_t j = i + 1; j < 4; j++)
      {
         Edge edge(points_[i], points_[j]);
         edges_.push_back(edge);
      }
   }

   // 4. Create the inner edges from p5 and p6 to p7, p8, p9, p10; plus the
   //    latter four connected in a cycle.
   Edge edge5(points_[4], points_[6]);
   Edge edge6(points_[4], points_[7]);
   Edge edge7(points_[4], points_[8]);
   Edge edge8(points_[4], points_[9]);
   Edge edge9(points_[5], points_[6]);
   Edge edge10(points_[5], points_[7]);
   Edge edge11(points_[5], points_[8]);
   Edge edge12(points_[5], points_[9]);
   Edge edge13(points_[6], points_[8]);
   Edge edge14(points_[6], points_[9]);
   Edge edge15(points_[7], points_[8]);
   Edge edge16(points_[7], points_[9]);

   // Push into edges vector
   edges_.push_back(edge5);
   edges_.push_back(edge6);
   edges_.push_back(edge7);
   edges_.push_back(edge8);
   edges_.push_back(edge9);
   edges_.push_back(edge10);
   edges_.push_back(edge11);
   edges_.push_back(edge12);
   edges_.push_back(edge13);
   edges_.push_back(edge14);
   edges_.push_back(edge15);
   edges_.push_back(edge16);
}

void OctetTrussTopology::CreatePeriodicPoints(
   const Vector &x, std::vector<Vector> &periodic_points) const
{
   Vector xx(x);
   // Compute the displaced ghost points. Computation assumes domain [0,1]^3.
   real_t d_x[3] = {1, 0, 0};
   real_t d_y[3] = {0, 1, 0};
   real_t d_z[3] = {0, 0, 1};

   Vector dispcement_x(d_x, 3);
   Vector dispcement_y(d_y, 3);
   Vector dispcement_z(d_z, 3);

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

}  // namespace mfem
