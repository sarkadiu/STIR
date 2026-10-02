/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Implementation of the Origin Ensemble event-origin sampler.
*/

#include "stir/OE/OriginEnsemble.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace stir
{

const std::size_t OriginEnsemble::invalid_voxel = std::numeric_limits<std::size_t>::max();

OriginEnsemble::OriginEnsemble(const std::vector<double>& sensitivity_v,
                               const std::vector<Proposal>& proposals_v,
                               const std::vector<std::size_t>& event_proposals_v,
                               const unsigned int seed)
    : sensitivity(sensitivity_v),
      proposals(proposals_v),
      event_proposals(event_proposals_v),
      counts(sensitivity.size(), 0),
      mean_counts(sensitivity.size(), 0.),
      num_samples(0),
      generator(seed),
      uniform(0., 1.)
{
  if (sensitivity.empty() || proposals.empty() || event_proposals.empty())
    throw std::invalid_argument("Origin Ensemble requires voxels, origin proposals, and events");

  for (const auto value : sensitivity)
    if (!std::isfinite(value) || value < 0.)
      throw std::invalid_argument("Origin Ensemble sensitivity must be finite and nonnegative");

  for (const auto& proposal : proposals)
    if (!proposal)
      throw std::invalid_argument("Origin Ensemble origin proposals cannot be empty functions");

  // Validate every event before drawing any initial origin.
  for (const auto proposal : event_proposals)
    if (proposal >= proposals.size())
      throw std::invalid_argument("Origin Ensemble event references a nonexistent origin proposal");

  origins.reserve(event_proposals.size());
  for (const auto proposal : event_proposals)
    {
      std::size_t voxel = invalid_voxel;
      for (unsigned int attempt = 0; attempt < 100000; ++attempt)
        {
          const auto candidate = proposals[proposal](generator);
          if (candidate < sensitivity.size() && sensitivity[candidate] > 0.)
            {
              voxel = candidate;
              break;
            }
        }
      if (voxel == invalid_voxel)
        throw std::runtime_error("Origin Ensemble could not initialize event " + std::to_string(origins.size())
                                 + " after 100000 proposals; its LOR/TOF support may miss the observable image");
      if (counts[voxel] == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Origin Ensemble voxel count overflow");
      origins.push_back(voxel);
      ++counts[voxel];
    }
}

double
OriginEnsemble::acceptance_probability(const std::uint64_t old_count,
                                       const std::uint64_t new_count,
                                       const double old_sensitivity,
                                       const double new_sensitivity)
{
  if (old_count == 0)
    throw std::invalid_argument("Origin Ensemble old voxel must contain the moving event");
  if (!std::isfinite(old_sensitivity) || !std::isfinite(new_sensitivity) || old_sensitivity <= 0. || new_sensitivity <= 0.)
    throw std::invalid_argument("Origin Ensemble move sensitivities must be positive and finite");

  // Ordinary moves use only division and multiplication. Convert before adding
  // one to avoid wrapping N_new at the uint64 limit.
  const double count_ratio = (static_cast<double>(new_count) + 1.) / static_cast<double>(old_count);
  const double ratio = old_sensitivity / new_sensitivity * count_ratio;
  if (std::isfinite(ratio) && ratio > 0.)
    return std::min(1., ratio);

  // A logarithmic fallback handles extreme sensitivity ratios without losing
  // a small representable acceptance through an intermediate underflow.
  const long double log_ratio
      = std::log(static_cast<long double>(old_sensitivity)) - std::log(static_cast<long double>(new_sensitivity))
        + std::log(static_cast<long double>(new_count) + 1.L) - std::log(static_cast<long double>(old_count));
  return log_ratio >= 0.L ? 1. : static_cast<double>(std::exp(log_ratio));
}

void
OriginEnsemble::sweep()
{
  for (std::size_t event = 0; event < origins.size(); ++event)
    {
      const auto new_voxel = proposals[event_proposals[event]](generator);
      const auto old_voxel = origins[event];
      if (new_voxel >= sensitivity.size() || sensitivity[new_voxel] == 0. || new_voxel == old_voxel)
        continue;

      const double acceptance
          = acceptance_probability(counts[old_voxel], counts[new_voxel], sensitivity[old_voxel], sensitivity[new_voxel]);
      if (uniform(generator) < acceptance)
        {
          if (counts[new_voxel] == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Origin Ensemble voxel count overflow");
          --counts[old_voxel];
          ++counts[new_voxel];
          origins[event] = new_voxel;
        }
    }
}

void
OriginEnsemble::accumulate_sample()
{
  if (num_samples == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("Origin Ensemble sample count overflow");
  ++num_samples;
  // An online mean avoids overflowing a sum over a long sampling run.
  const double samples = static_cast<double>(num_samples);
  for (std::size_t voxel = 0; voxel < counts.size(); ++voxel)
    mean_counts[voxel] += (static_cast<double>(counts[voxel]) - mean_counts[voxel]) / samples;
}

std::vector<double>
OriginEnsemble::get_mean_image() const
{
  if (num_samples == 0)
    throw std::logic_error("Origin Ensemble image requires at least one accumulated sample");

  std::vector<double> image(mean_counts.size(), 0.);
  for (std::size_t voxel = 0; voxel < image.size(); ++voxel)
    if (sensitivity[voxel] > 0.)
      {
        image[voxel] = mean_counts[voxel] / sensitivity[voxel];
        if (!std::isfinite(image[voxel]))
          throw std::overflow_error("Origin Ensemble mean image exceeds the finite output range");
      }
  return image;
}

LORProposal::LORProposal(const Grid& grid_v,
                         const std::array<double, 3>& point1,
                         const std::array<double, 3>& point2,
                         const double tof_low,
                         const double tof_high,
                         const double tof_sigma)
    : grid(grid_v),
      chord_low(0.),
      chord_high(0.),
      intersects(false),
      use_tof(tof_sigma > 0.),
      position_distribution(0., 1.),
      noise_distribution(0., 1.)
{
  if (!std::isfinite(tof_low) || !std::isfinite(tof_high) || !std::isfinite(tof_sigma) || tof_high < tof_low || tof_sigma < 0.
      || !std::isfinite(tof_high - tof_low))
    throw std::invalid_argument("LOR proposal timing limits and sigma must be finite, ordered, and sigma nonnegative");

  std::array<double, 3> box_low, box_high, delta;
  std::size_t voxel_count = 1;
  for (std::size_t axis = 0; axis < 3; ++axis)
    {
      if (!std::isfinite(grid.origin[axis]) || !std::isfinite(grid.spacing[axis]) || grid.spacing[axis] <= 0.
          || grid.size[axis] == 0)
        throw std::invalid_argument("LOR proposal requires finite grid origin, positive spacing, and nonempty dimensions");
      if (grid.size[axis] > std::numeric_limits<std::size_t>::max() / voxel_count)
        throw std::invalid_argument("LOR proposal grid size exceeds the flattened index range");
      voxel_count *= grid.size[axis];
      if (!std::isfinite(point1[axis]) || !std::isfinite(point2[axis]))
        throw std::invalid_argument("LOR proposal detector endpoints must be finite");
      center[axis] = point1[axis] / 2. + point2[axis] / 2.;
      delta[axis] = point2[axis] - point1[axis];
      box_low[axis] = grid.origin[axis] + grid.spacing[axis] * (static_cast<double>(grid.minimum[axis]) - .5);
      box_high[axis]
          = grid.origin[axis]
            + grid.spacing[axis] * (static_cast<double>(grid.minimum[axis]) + static_cast<double>(grid.size[axis]) - .5);
      if (!std::isfinite(box_low[axis]) || !std::isfinite(box_high[axis]) || box_high[axis] <= box_low[axis])
        throw std::invalid_argument("LOR proposal grid voxel boundaries exceed the finite coordinate range");
    }
  const double length = std::hypot(std::hypot(delta[0], delta[1]), delta[2]);
  if (!std::isfinite(length) || length <= 0.)
    throw std::invalid_argument("LOR proposal endpoints must define a nonzero finite segment");
  chord_low = -length / 2.;
  chord_high = length / 2.;
  for (std::size_t axis = 0; axis < 3; ++axis)
    {
      minus_unit[axis] = -delta[axis] / length;
      if (minus_unit[axis] == 0.)
        {
          if (center[axis] < box_low[axis] || center[axis] >= box_high[axis])
            return;
          continue;
        }
      const double first = (box_low[axis] - center[axis]) / minus_unit[axis];
      const double second = (box_high[axis] - center[axis]) / minus_unit[axis];
      chord_low = std::max(chord_low, std::min(first, second));
      chord_high = std::min(chord_high, std::max(first, second));
      if (chord_high <= chord_low)
        return;
    }
  intersects = true;
  if (use_tof)
    {
      position_distribution = std::uniform_real_distribution<double>(tof_low, tof_high);
      noise_distribution = std::normal_distribution<double>(0., tof_sigma);
    }
  else
    position_distribution = std::uniform_real_distribution<double>(chord_low, chord_high);
}

std::size_t
LORProposal::operator()(std::mt19937& generator)
{
  if (!intersects)
    return OriginEnsemble::invalid_voxel;
  double displacement = position_distribution(generator);
  if (use_tof)
    displacement += noise_distribution(generator);
  if (!std::isfinite(displacement) || displacement < chord_low || displacement > chord_high)
    return OriginEnsemble::invalid_voxel;

  std::array<std::size_t, 3> index;
  for (std::size_t axis = 0; axis < 3; ++axis)
    {
      const double point = center[axis] + minus_unit[axis] * displacement;
      const double voxel
          = std::floor((point - grid.origin[axis]) / grid.spacing[axis] + .5) - static_cast<double>(grid.minimum[axis]);
      if (!std::isfinite(voxel) || voxel < 0. || voxel >= static_cast<double>(grid.size[axis]))
        return OriginEnsemble::invalid_voxel;
      index[axis] = static_cast<std::size_t>(voxel);
    }
  return (index[0] * grid.size[1] + index[1]) * grid.size[2] + index[2];
}

} // namespace stir
