/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Event-origin sampler for Origin Ensemble reconstruction.
*/

#ifndef STIR_OE_ORIGINENSEMBLE_H
#define STIR_OE_ORIGINENSEMBLE_H

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <random>
#include <vector>

namespace stir
{

/*!
  \brief Sample event origins using the N+1/M Origin Ensemble transition.

  Each event references a continuous origin proposal, for example LORProposal.
  Attenuation enters the supplied voxel sensitivity. For distinct old and proposed voxels,
  the acceptance probability is
  \f$\min(1, s_{old}(N_{new}+1)/(s_{new}N_{old}))\f$,
  where counts include the event at its old origin before the move.

  One sweep visits every event once. Call accumulate_sample() only after any
  desired burn-in, then get_mean_image() to retrieve mean occupancies divided
  by sensitivity. No pseudocount is added to the returned image.

  A proposal outside the image or at zero sensitivity is a self-loop. These
  proposals are not redrawn during a sweep. Initialization makes at most 100000
  attempts per event to find an observable origin, then throws std::runtime_error.
  Invalid inputs throw std::invalid_argument. The sampler is serial and owns
  its random-number generator; equal inputs and seeds give equal sequences
  within the same standard-library implementation.
*/
class OriginEnsemble
{
public:
  using Proposal = std::function<std::size_t(std::mt19937&)>;
  static const std::size_t invalid_voxel;

  OriginEnsemble(const std::vector<double>& sensitivity,
                 const std::vector<Proposal>& proposals,
                 const std::vector<std::size_t>& event_proposals,
                 unsigned int seed);

  //! Attempt one origin update for every event.
  void sweep();

  //! Include the current ensemble in the image average.
  void accumulate_sample();

  //! Return the sample-mean occupancy divided by sensitivity.
  /*! Throws std::logic_error if no sample has been accumulated. */
  std::vector<double> get_mean_image() const;

  const std::vector<std::uint64_t>& get_counts() const { return counts; }
  std::uint64_t get_num_samples() const { return num_samples; }

  //! Compute the acceptance probability for a move between distinct voxels.
  /*! Counts are taken before removing the event from its old voxel. */
  static double
  acceptance_probability(std::uint64_t old_count, std::uint64_t new_count, double old_sensitivity, double new_sensitivity);

private:
  std::vector<double> sensitivity;
  std::vector<Proposal> proposals;
  std::vector<std::size_t> event_proposals;
  std::vector<std::size_t> origins;
  std::vector<std::uint64_t> counts;
  std::vector<double> mean_counts;
  std::uint64_t num_samples;
  std::mt19937 generator;
  std::uniform_real_distribution<double> uniform;
};

/*!
  \brief Direct continuous proposal on a finite detector-to-detector LOR.

  Grid coordinates and endpoints are ordered z,y,x, in the same physical units.
  Grid origin is the coordinate of voxel index zero; minimum is the first image
  index. Voxel faces occur half a spacing from their centers. The chord is clipped
  analytically to these faces, without constructing a voxel intersection list.

  With tof_sigma == 0, points are uniform on the clipped chord. With positive
  sigma, one timing displacement is sampled as Uniform(tof_low,tof_high) plus
  Normal(0,tof_sigma), then mapped by point = midpoint - unit(point2-point1)*t.
  Timing limits and sigma are distances along the LOR, not times. A draw outside
  the finite chord yields OriginEnsemble::invalid_voxel; it is never redrawn here.
*/
class LORProposal
{
public:
  struct Grid
  {
    std::array<double, 3> origin;
    std::array<double, 3> spacing;
    std::array<int, 3> minimum;
    std::array<std::size_t, 3> size;
  };

  LORProposal(const Grid& grid,
              const std::array<double, 3>& point1,
              const std::array<double, 3>& point2,
              double tof_low = 0.,
              double tof_high = 0.,
              double tof_sigma = 0.);

  std::size_t operator()(std::mt19937& generator);
  bool has_intersection() const { return intersects; }

private:
  Grid grid;
  std::array<double, 3> center;
  std::array<double, 3> minus_unit;
  double chord_low;
  double chord_high;
  bool intersects;
  bool use_tof;
  std::uniform_real_distribution<double> position_distribution;
  std::normal_distribution<double> noise_distribution;
};

} // namespace stir

#endif
