/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Tests for the Origin Ensemble sampler, including exact detailed balance.
*/

#include "stir/OE/OriginEnsemble.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace
{

using stir::OriginEnsemble;
using stir::LORProposal;
using Row = std::vector<std::pair<std::size_t, double>>;

// Discrete factories are used only for the sampler's mathematical tests.
std::vector<OriginEnsemble::Proposal>
make_proposals(const std::vector<Row>& rows)
{
  std::vector<OriginEnsemble::Proposal> result;
  for (const auto& row : rows)
    {
      std::vector<std::size_t> voxels;
      std::vector<double> weights;
      for (const auto& candidate : row)
        {
          voxels.push_back(candidate.first);
          weights.push_back(candidate.second);
        }
      std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
      result.push_back([voxels, distribution](std::mt19937& generator) mutable { return voxels[distribution(generator)]; });
    }
  return result;
}

LORProposal::Grid
simple_grid()
{
  return { { { 0., 0., 0. } }, { { 1., 1., 1. } }, { { 0, 0, -2 } }, { { 1, 1, 5 } } };
}

// These tests deliberately use only the standard library, so the core sampler
// can also be checked without a scanner, data files, or STIR registries.
class OriginEnsembleTests
{
public:
  int run()
  {
    test_validation();
    test_acceptance();
    test_output();
    test_conservation_and_seed();
    test_weighted_proposal();
    test_exact_ensemble();
    test_uniform_lor();
    test_tof_lor();
    test_oblique_grid();
    test_lor_validation();
    test_invalid_self_loops();
    if (failures == 0)
      std::cout << "All Origin Ensemble sampler tests passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

private:
  unsigned int failures = 0;

  void check(const bool condition, const std::string& description)
  {
    if (!condition)
      {
        ++failures;
        std::cerr << "FAILED: " << description << '\n';
      }
  }

  void check_close(const double actual, const double expected, const double tolerance, const std::string& description)
  {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, description);
  }

  template <class Exception, class Function>
  void expect_exception(Function function, const std::string& description)
  {
    try
      {
        function();
        check(false, description + " (no exception)");
      }
    catch (const Exception&)
      {}
    catch (...)
      {
        check(false, description + " (wrong exception)");
      }
  }

  void test_validation()
  {
    const std::vector<double> sensitivity{ 1., 2. };
    const auto proposals = make_proposals({ { { 0, 1. }, { 1, 1. } } });
    const std::vector<std::size_t> events{ 0 };
    expect_exception<std::invalid_argument>([&]() { OriginEnsemble sampler({}, proposals, events, 1); }, "empty image rejected");
    expect_exception<std::invalid_argument>([&]() { OriginEnsemble sampler(sensitivity, {}, events, 1); },
                                            "empty proposal list rejected");
    expect_exception<std::invalid_argument>([&]() { OriginEnsemble sampler(sensitivity, proposals, {}, 1); },
                                            "empty event list rejected");
    expect_exception<std::invalid_argument>(
        [&]() { OriginEnsemble sampler(sensitivity, { OriginEnsemble::Proposal() }, events, 1); },
        "empty origin callback rejected");
    expect_exception<std::invalid_argument>([&]() { OriginEnsemble sampler(sensitivity, proposals, { 1 }, 1); },
                                            "event proposal outside range rejected");

    const double invalid_values[] = { -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() };
    for (const auto value : invalid_values)
      {
        expect_exception<std::invalid_argument>([&]() { OriginEnsemble sampler({ value, 2. }, proposals, events, 1); },
                                                "negative or nonfinite sensitivity rejected");
      }
  }

  void test_acceptance()
  {
    check_close(OriginEnsemble::acceptance_probability(4, 0, 2., 1.), 0.5, 1.e-14, "N+1/M with sensitivity ratio");
    check_close(OriginEnsemble::acceptance_probability(2, 0, 1., 1.), 0.5, 1.e-14, "old count includes moving event");
    check_close(OriginEnsemble::acceptance_probability(1, 0, 1., 1.), 1., 0., "one event moving to empty voxel accepted");
    check_close(OriginEnsemble::acceptance_probability(1, 4, 1., 1.), 1., 0., "acceptance capped at one");
    check_close(OriginEnsemble::acceptance_probability(1, 0, 1.e300, 1.e-300), 1., 0., "large sensitivity ratio stays finite");
    check_close(OriginEnsemble::acceptance_probability(1, 0, 1.e-300, 1.e300), 0., 0., "small sensitivity ratio stays finite");
    const double recovered = OriginEnsemble::acceptance_probability(1, UINT64_C(10000000000000000000), 1.e-200, 1.e130);
    check(recovered > 0. && std::abs(recovered / 1.e-311 - 1.) < 1.e-10,
          "log fallback recovers representable acceptance after intermediate underflow");
    check_close(OriginEnsemble::acceptance_probability(
                    std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max(), 1., 1.),
                1.,
                0.,
                "count increment does not wrap at uint64 limit");
    expect_exception<std::invalid_argument>([]() { OriginEnsemble::acceptance_probability(0, 1, 1., 1.); },
                                            "empty old voxel rejected");
    expect_exception<std::invalid_argument>([]() { OriginEnsemble::acceptance_probability(1, 0, 1., 0.); },
                                            "zero move sensitivity rejected");
  }

  void test_output()
  {
    OriginEnsemble sampler({ 1., 2., 0. }, make_proposals({ { { 1, 1. } } }), { 0, 0, 0, 0 }, 19);
    expect_exception<std::logic_error>([&]() { sampler.get_mean_image(); }, "image before sampling rejected");
    sampler.sweep();
    check(sampler.get_num_samples() == 0, "sweep does not accumulate a burn-in sample");
    sampler.accumulate_sample();
    sampler.sweep();
    sampler.accumulate_sample();
    const auto image = sampler.get_mean_image();
    check(sampler.get_num_samples() == 2, "sample counter tracks only explicit accumulation");
    check_close(image[0], 0., 0., "empty observable voxel has no output pseudocount");
    check_close(image[1], 2., 0., "mean image divides detected counts by sensitivity");
    check_close(image[2], 0., 0., "unobservable voxel remains zero");
  }

  void test_conservation_and_seed()
  {
    const std::vector<double> sensitivity{ 1., 2., 3., 0. };
    const std::vector<Row> rows{ { { 0, 1. }, { 1, 2. } }, { { 2, 1. } }, { { 0, 3. }, { 1, 1. }, { 2, 2. } } };
    std::vector<std::size_t> events;
    for (std::size_t event = 0; event < 96; ++event)
      events.push_back(event % rows.size());
    OriginEnsemble first(sensitivity, make_proposals(rows), events, 31415);
    OriginEnsemble second(sensitivity, make_proposals(rows), events, 31415);
    check(first.get_counts() == second.get_counts(), "seed repeats initial assignments");
    for (int sweep = 0; sweep < 500; ++sweep)
      {
        first.sweep();
        second.sweep();
        const auto& counts = first.get_counts();
        check(std::accumulate(counts.begin(), counts.end(), std::uint64_t(0)) == events.size(), "sweep conserves event count");
        check(counts[2] >= 32 && counts[3] == 0, "origins respect sparse candidate support");
        check(counts == second.get_counts(), "equal seed repeats every sweep");
        if (sweep >= 50)
          {
            first.accumulate_sample();
            second.accumulate_sample();
          }
      }
    const auto image = first.get_mean_image();
    check(image == second.get_mean_image(), "equal seed repeats ensemble mean");
    double total = 0.;
    for (std::size_t voxel = 0; voxel < image.size(); ++voxel)
      total += image[voxel] * sensitivity[voxel];
    check_close(total, 96., 1.e-11, "sampled image preserves sensitivity-weighted count total");
  }

  void test_weighted_proposal()
  {
    // For one event and equal sensitivities every proposal is accepted. The
    // sampled origin distribution must therefore equal the row weights.
    OriginEnsemble sampler({ 1., 1. }, make_proposals({ { { 0, 1. }, { 1, 9. } } }), { 0 }, 12345);
    for (int sample = 0; sample < 80000; ++sample)
      {
        sampler.sweep();
        sampler.accumulate_sample();
      }
    const auto image = sampler.get_mean_image();
    check_close(image[0], 0.1, 0.007, "proposal uses geometric weights for first voxel");
    check_close(image[1], 0.9, 0.007, "proposal uses geometric weights for second voxel");

    unsigned int second_voxel = 0;
    for (unsigned int seed = 0; seed < 10000; ++seed)
      {
        OriginEnsemble initial({ 1., 1. }, make_proposals({ { { 0, 1. }, { 1, 9. } } }), { 0 }, seed);
        second_voxel += static_cast<unsigned int>(initial.get_counts()[1]);
      }
    check_close(second_voxel / 10000., 0.9, 0.02, "initialization also uses weighted proposal");
  }

  void test_exact_ensemble()
  {
    const std::vector<double> sensitivity{ 2., 3. };
    const std::vector<Row> rows{ { { 0, 1. }, { 1, 2. } }, { { 0, 3. }, { 1, 1. } } };
    const std::vector<std::size_t> events{ 0, 1, 0 };
    const double weights[2][2] = { { 1., 2. }, { 3., 1. } };
    const double factorials[] = { 1., 1., 2., 6. };
    std::vector<double> probabilities(8, 0.);
    double total = 0.;

    // Enumerate every labeled-event assignment. The transition targets
    // pi(x) proportional to product_e P[e,x_e] product_v N_v! / s_v^N_v.
    for (unsigned int state = 0; state < 8; ++state)
      {
        unsigned int counts[2] = { 0, 0 };
        double weight = 1.;
        for (unsigned int event = 0; event < 3; ++event)
          {
            const auto voxel = (state >> event) & 1U;
            ++counts[voxel];
            weight *= weights[events[event]][voxel];
          }
        for (unsigned int voxel = 0; voxel < 2; ++voxel)
          weight *= factorials[counts[voxel]] / std::pow(sensitivity[voxel], counts[voxel]);
        probabilities[state] = weight;
        total += weight;
      }
    for (auto& probability : probabilities)
      probability /= total;

    std::vector<double> exact_image(2, 0.);
    for (unsigned int state = 0; state < 8; ++state)
      {
        std::uint64_t counts[2] = { 0, 0 };
        for (unsigned int event = 0; event < 3; ++event)
          ++counts[(state >> event) & 1U];
        for (unsigned int voxel = 0; voxel < 2; ++voxel)
          exact_image[voxel] += probabilities[state] * counts[voxel] / sensitivity[voxel];

        for (unsigned int event = 0; event < 3; ++event)
          {
            const unsigned int old_voxel = (state >> event) & 1U;
            const unsigned int new_voxel = 1U - old_voxel;
            const unsigned int next_state = state ^ (1U << event);
            const double* row_weights = weights[events[event]];
            const double row_sum = row_weights[0] + row_weights[1];
            const double forward = probabilities[state] * row_weights[new_voxel] / row_sum
                                   * OriginEnsemble::acceptance_probability(
                                       counts[old_voxel], counts[new_voxel], sensitivity[old_voxel], sensitivity[new_voxel]);
            const double reverse
                = probabilities[next_state] * row_weights[old_voxel] / row_sum
                  * OriginEnsemble::acceptance_probability(
                      counts[new_voxel] + 1, counts[old_voxel] - 1, sensitivity[new_voxel], sensitivity[old_voxel]);
            check_close(forward, reverse, 1.e-13, "exact labeled-event detailed balance");
          }
      }

    OriginEnsemble sampler(sensitivity, make_proposals(rows), events, 67890);
    for (int burn_in = 0; burn_in < 1000; ++burn_in)
      sampler.sweep();
    for (int sample = 0; sample < 100000; ++sample)
      {
        sampler.sweep();
        sampler.accumulate_sample();
      }
    const auto image = sampler.get_mean_image();
    check_close(image[0], exact_image[0], 0.015, "sampled first voxel matches exactly enumerated ensemble");
    check_close(image[1], exact_image[1], 0.015, "sampled second voxel matches exactly enumerated ensemble");
  }

  std::vector<double> sample_lor(LORProposal proposal,
                                 const std::size_t num_voxels,
                                 const unsigned int samples,
                                 const unsigned int seed,
                                 double& invalid_fraction)
  {
    std::mt19937 generator(seed);
    std::vector<double> frequencies(num_voxels, 0.);
    unsigned int invalid = 0;
    bool within_range = true;
    for (unsigned int draw = 0; draw < samples; ++draw)
      {
        const auto voxel = proposal(generator);
        if (voxel == OriginEnsemble::invalid_voxel)
          ++invalid;
        else if (voxel < num_voxels)
          ++frequencies[voxel];
        else
          within_range = false;
      }
    check(within_range, "LOR proposal returns only flattened image indices or the invalid sentinel");
    for (auto& frequency : frequencies)
      frequency /= samples;
    invalid_fraction = static_cast<double>(invalid) / samples;
    return frequencies;
  }

  static double normal_cdf(const double value) { return .5 * std::erfc(-value / std::sqrt(2.)); }

  static double normal_integral(const double value)
  {
    return value * normal_cdf(value) + std::exp(-value * value / 2.) / std::sqrt(2. * std::acos(-1.));
  }

  static double timing_cdf(const double value, const double low, const double high, const double sigma)
  {
    if (high == low)
      return normal_cdf((value - low) / sigma);
    return sigma / (high - low) * (normal_integral((value - low) / sigma) - normal_integral((value - high) / sigma));
  }

  void test_uniform_lor()
  {
    const auto grid = simple_grid();
    LORProposal proposal(grid, { { 0., 0., -10. } }, { { 0., 0., 10. } });
    check(proposal.has_intersection(), "finite axial LOR intersects grid voxel faces");
    double invalid = 0.;
    const auto frequencies = sample_lor(proposal, 5, 80000, 151, invalid);
    check_close(invalid, 0., 0., "uniform proposal samples only the analytically clipped chord");
    for (const auto mass : frequencies)
      check_close(mass, .2, .007, "uniform chord maps to analytic equal voxel intersection masses");

    LORProposal finite_segment(grid, { { 0., 0., -1. } }, { { 0., 0., 1. } });
    const auto shorter = sample_lor(finite_segment, 5, 80000, 152, invalid);
    check_close(shorter[0] + shorter[4], 0., 0., "clipping retains finite detector segment limits");
    check_close(shorter[1], .25, .007, "finite segment's first half voxel has correct uniform mass");
    check_close(shorter[2], .5, .007, "finite segment's center voxel has correct uniform mass");
    check_close(shorter[3], .25, .007, "finite segment's last half voxel has correct uniform mass");
  }

  void test_tof_lor()
  {
    const auto grid = simple_grid();
    const double cases[2][3] = { { .75, .75, .7 }, { -.7, .9, .8 } };
    for (unsigned int test = 0; test < 2; ++test)
      {
        const double low = cases[test][0], high = cases[test][1], sigma = cases[test][2];
        LORProposal proposal(grid, { { 0., 0., -10. } }, { { 0., 0., 10. } }, low, high, sigma);
        double invalid = 0.;
        const auto frequencies = sample_lor(proposal, 5, 100000, 160 + test, invalid);
        double valid_probability = 0.;
        for (int voxel = -2; voxel <= 2; ++voxel)
          {
            // x = -t for endpoints ordered in increasing x: integrate timing
            // density across the reversed voxel-face interval.
            const double expected = timing_cdf(-voxel + .5, low, high, sigma) - timing_cdf(-voxel - .5, low, high, sigma);
            valid_probability += expected;
            check_close(frequencies[voxel + 2], expected, .007, "continuous TOF mass matches analytic bin/Gaussian convolution");
          }
        check_close(invalid, 1. - valid_probability, .007, "out-of-chord TOF draws retain their original self-loop mass");
        check(invalid > 0., "TOF proposals outside image are not rejection-resampled");
      }
    LORProposal sign(grid, { { 0., 0., -10. } }, { { 0., 0., 10. } }, 1., 1., .1);
    double invalid = 0.;
    const auto frequencies = sample_lor(sign, 5, 20000, 162, invalid);
    check(frequencies[1] > .99, "positive timing displacement moves toward point1 using the minus-unit convention");
    LORProposal reversed(grid, { { 0., 0., 10. } }, { { 0., 0., -10. } }, 1., 1., .1);
    const auto reverse_frequencies = sample_lor(reversed, 5, 20000, 162, invalid);
    check(reverse_frequencies[3] > .99, "reversing detector endpoints reverses TOF localization");

    LORProposal finite(grid, { { 0., 0., -1. } }, { { 0., 0., 1. } }, 0., 0., 2.);
    const auto finite_frequencies = sample_lor(finite, 5, 80000, 163, invalid);
    check_close(finite_frequencies[0] + finite_frequencies[4], 0., 0., "TOF cannot propose beyond the finite detector segment");
    check_close(invalid,
                1. - (normal_cdf(.5) - normal_cdf(-.5)),
                .007,
                "finite detector segment excludes the correct Gaussian tail probability");
  }

  void test_oblique_grid()
  {
    const LORProposal::Grid grid{ { { 10., -5., 2. } }, { { 2., 3., 4. } }, { { -1, 2, -2 } }, { { 3, 4, 5 } } };
    LORProposal proposal(grid, { { 6., -2., -10. } }, { { 14., 13., 14. } });
    double invalid = 0.;
    const auto frequencies = sample_lor(proposal, 60, 100000, 171, invalid);
    check_close(invalid, 0., 0., "oblique anisotropic chord maps inside the grid");

    // Independently split the line at voxel faces in the test only, to obtain
    // exact continuous masses. Production LORProposal performs no traversal.
    const double first_index[3] = { -2., 1., -3. };
    const double delta_index[3] = { 4., 5., 6. };
    std::vector<double> cuts{ .125, .875 };
    for (unsigned int axis = 0; axis < 3; ++axis)
      for (std::size_t face = 0; face <= grid.size[axis]; ++face)
        {
          const double cut = (grid.minimum[axis] + static_cast<double>(face) - .5 - first_index[axis]) / delta_index[axis];
          if (cut > .125 && cut < .875)
            cuts.push_back(cut);
        }
    std::sort(cuts.begin(), cuts.end());
    std::vector<double> expected(60, 0.);
    for (std::size_t interval = 1; interval < cuts.size(); ++interval)
      {
        const double middle = (cuts[interval - 1] + cuts[interval]) / 2.;
        std::size_t index[3];
        for (unsigned int axis = 0; axis < 3; ++axis)
          index[axis]
              = static_cast<std::size_t>(std::floor(first_index[axis] + delta_index[axis] * middle + .5) - grid.minimum[axis]);
        const std::size_t flat = (index[0] * grid.size[1] + index[1]) * grid.size[2] + index[2];
        expected[flat] += (cuts[interval] - cuts[interval - 1]) / .75;
      }
    for (std::size_t voxel = 0; voxel < expected.size(); ++voxel)
      check_close(frequencies[voxel],
                  expected[voxel],
                  .007,
                  "oblique mapped mass matches origin/spacing/index bounds and z,y,x flattening");
  }

  void test_lor_validation()
  {
    const auto grid = simple_grid();
    const std::array<double, 3> first{ { 0., 0., -10. } }, second{ { 0., 0., 10. } };
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(grid, first, first); }, "zero-length LOR rejected");
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(grid, first, second, 1., -1., .5); },
                                            "reversed timing interval rejected");
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(grid, first, second, 0., 0., -1.); },
                                            "negative timing sigma rejected");
    expect_exception<std::invalid_argument>(
        [&]() {
          LORProposal proposal(grid, first, second, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 1.);
        },
        "timing-bin width overflow rejected");
    for (const double value : { 0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN() })
      {
        auto invalid_grid = grid;
        invalid_grid.spacing[1] = value;
        expect_exception<std::invalid_argument>([&]() { LORProposal proposal(invalid_grid, first, second); },
                                                "invalid anisotropic spacing rejected");
      }
    auto empty = grid;
    empty.size[2] = 0;
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(empty, first, second); }, "empty grid rejected");
    auto huge = grid;
    huge.size = { { std::numeric_limits<std::size_t>::max(), 2, 1 } };
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(huge, first, second); },
                                            "flattened grid overflow rejected");
    auto bad_origin = grid;
    bad_origin.origin[0] = std::numeric_limits<double>::infinity();
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(bad_origin, first, second); },
                                            "nonfinite grid origin rejected");
    auto bad_endpoint = first;
    bad_endpoint[0] = std::numeric_limits<double>::quiet_NaN();
    expect_exception<std::invalid_argument>([&]() { LORProposal proposal(grid, bad_endpoint, second); },
                                            "nonfinite detector endpoint rejected");
    expect_exception<std::invalid_argument>(
        [&]() { LORProposal proposal(grid, first, second, 0., 0., std::numeric_limits<double>::infinity()); },
        "nonfinite timing sigma rejected");
    std::mt19937 generator(1);
    LORProposal outside(grid, { { 1., 0., -10. } }, { { 1., 0., 10. } });
    check(!outside.has_intersection() && outside(generator) == OriginEnsemble::invalid_voxel,
          "LOR missing the voxel box has no valid proposal");
    LORProposal upper_face(grid, { { .5, 0., -10. } }, { { .5, 0., 10. } });
    check(!upper_face.has_intersection(), "line on excluded upper voxel face has no image support");
    LORProposal lower_face(grid, { { -.5, 0., -10. } }, { { -.5, 0., 10. } });
    check(lower_face.has_intersection() && lower_face(generator) < 5, "included lower voxel face maps to the first image plane");
  }

  void test_invalid_self_loops()
  {
    for (const std::size_t invalid : { OriginEnsemble::invalid_voxel, std::size_t(999), std::size_t(1) })
      {
        unsigned int calls = 0;
        OriginEnsemble::Proposal proposal = [&calls, invalid](std::mt19937&) { return ++calls == 1 ? std::size_t(0) : invalid; };
        OriginEnsemble sampler({ 1., 0. }, { proposal }, { 0 }, 181);
        sampler.sweep();
        check(calls == 2, "invalid/outside/zero-sensitivity proposal is drawn only once per sweep");
        check(sampler.get_counts()[0] == 1 && sampler.get_counts()[1] == 0, "invalid proposal preserves event origin and count");
      }
    unsigned int calls = 0;
    OriginEnsemble::Proposal unreachable = [&calls](std::mt19937&) {
      ++calls;
      return OriginEnsemble::invalid_voxel;
    };
    expect_exception<std::runtime_error>([&]() { OriginEnsemble sampler({ 1. }, { unreachable }, { 0 }, 182); },
                                         "unreachable initialization gives a bounded clear failure");
    check(calls == 100000, "initialization retry limit is exactly 100000 proposals");
    LORProposal outside(simple_grid(), { { 1., 0., -10. } }, { { 1., 0., 10. } });
    expect_exception<std::runtime_error>([&]() { OriginEnsemble sampler(std::vector<double>(5, 1.), { outside }, { 0 }, 183); },
                                         "direct LOR without image intersection cannot initialize an origin");
  }
};

} // namespace

int
main()
{
  OriginEnsembleTests tests;
  return tests.run();
}
