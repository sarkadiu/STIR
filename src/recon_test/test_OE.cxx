/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.
*/
/*! \file
    \ingroup recon_test
    \brief Self-contained list-mode integration tests for attenuation-corrected OE.
*/

#include "stir/OE/OEReconstruction.h"
#include "stir/RunTests.h"
#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfo.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/listmode/ListModeData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/Succeeded.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

START_NAMESPACE_STIR

namespace
{

// Explicit bin assignments exercise the reconstruction's list-mode adapter,
// including TOF and time records, without needing a scanner-specific data file.
class TestEvent : public ListEvent
{
public:
  TestEvent()
      : prompt(true)
  {}
  bool is_prompt() const override { return prompt; }
  bool is_valid_template(const ProjDataInfo&) const override { return true; }
  void get_bin(Bin& output, const ProjDataInfo&) const override { output = bin; }
  LORAs2Points<float> get_LOR() const override
  {
    return LORAs2Points<float>(CartesianCoordinate3D<float>(0, -100, 0), CartesianCoordinate3D<float>(0, 100, 0));
  }
  Bin bin;
  bool prompt;
};

class TestTime : public ListTime
{
public:
  TestTime()
      : milliseconds(0)
  {}
  unsigned long get_time_in_millisecs() const override { return milliseconds; }
  Succeeded set_time_in_millisecs(unsigned long value) override
  {
    milliseconds = value;
    return Succeeded::yes;
  }

private:
  unsigned long milliseconds;
};

class TestRecord : public ListRecord
{
public:
  TestRecord()
      : time_record(false)
  {}
  bool is_time() const override { return time_record; }
  bool is_event() const override { return !time_record; }
  TestEvent& event() override { return event_data; }
  const TestEvent& event() const override { return event_data; }
  TestTime& time() override { return time_data; }
  const TestTime& time() const override { return time_data; }

  static TestRecord make_event(const Bin& bin, bool prompt = true)
  {
    TestRecord result;
    result.event_data.bin = bin;
    result.event_data.prompt = prompt;
    return result;
  }
  static TestRecord make_time(unsigned long milliseconds)
  {
    TestRecord result;
    result.time_record = true;
    result.time_data.set_time_in_millisecs(milliseconds);
    return result;
  }

private:
  bool time_record;
  TestEvent event_data;
  TestTime time_data;
};

class TestListModeData : public ListModeData
{
public:
  TestListModeData(const shared_ptr<const ExamInfo>& exam,
                   const shared_ptr<const ProjDataInfo>& geometry,
                   const std::vector<TestRecord>& records)
      : ListModeData(exam, geometry),
        records(records),
        position(0)
  {}
  std::string get_name() const override { return "OE in-memory test events"; }
  bool has_delayeds() const override { return true; }
  Succeeded reset() override
  {
    position = 0;
    return Succeeded::yes;
  }
  SavedPosition save_get_position() override { return static_cast<SavedPosition>(position); }
  Succeeded set_get_position(const SavedPosition& value) override
  {
    if (value > records.size())
      return Succeeded::no;
    position = value;
    return Succeeded::yes;
  }

protected:
  shared_ptr<ListRecord> get_empty_record_helper_sptr() const override { return shared_ptr<ListRecord>(new TestRecord); }
  Succeeded get_next(ListRecord& record) const override
  {
    if (position == records.size())
      return Succeeded::no;
    dynamic_cast<TestRecord&>(record) = records[position++];
    return Succeeded::yes;
  }

private:
  std::vector<TestRecord> records;
  mutable std::size_t position;
};

typedef DiscretisedDensity<3, float> Image;

shared_ptr<ProjMatrixByBinUsingRayTracing>
make_matrix()
{
  shared_ptr<ProjMatrixByBinUsingRayTracing> matrix(new ProjMatrixByBinUsingRayTracing);
  matrix->set_num_tangential_LORs(1);
  return matrix;
}

// The public matrix accessor is not virtual, so disable both caches and count
// its virtual row calculations. Any matrix request made while sampling then
// changes this count. The delegate supplies ordinary ray-tracing rows.
class CountingRayTracingMatrix : public ProjMatrixByBinUsingRayTracing
{
public:
  CountingRayTracingMatrix()
      : delegate(make_matrix()),
        setup_count(0),
        row_count(0),
        only_non_tof(true)
  {
    enable_cache(false);
    delegate->enable_cache(false);
  }
  void set_up(const shared_ptr<const ProjDataInfo>& geometry, const shared_ptr<const Image>& image) override
  {
    ++setup_count;
    only_non_tof = only_non_tof && geometry->get_num_tof_poss() == 1;
    ProjMatrixByBinUsingRayTracing::set_up(geometry, image);
    delegate->set_up(geometry, image);
  }
  std::size_t get_setup_count() const { return setup_count; }
  std::size_t get_row_count() const { return row_count; }
  bool has_only_non_tof_requests() const { return only_non_tof; }

private:
  void calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin& row) const override
  {
    ++row_count;
    only_non_tof = only_non_tof && row.get_bin().timing_pos_num() == 0;
    const Bin bin(row.get_bin());
    delegate->get_proj_matrix_elems_for_one_bin(row, bin);
  }
  shared_ptr<ProjMatrixByBinUsingRayTracing> delegate;
  std::size_t setup_count;
  mutable std::size_t row_count;
  mutable bool only_non_tof;
};

std::vector<float>
image_values(const Image& image)
{
  return std::vector<float>(image.begin_all(), image.end_all());
}

// Independently compute the mu-map line integral by summing sparse non-TOF
// matrix elements over its Cartesian grid. No bin normalisation or forward
// projector wrapper is used, so this checks transmission direction, units,
// and complete mu-map support in the OE integration.
std::vector<double>
reference_sensitivity(const shared_ptr<ProjDataInfo>& geometry, const shared_ptr<Image>& image, const shared_ptr<const Image>& mu)
{
  const auto non_tof = geometry->create_non_tof_clone();
  shared_ptr<Image> scaled_mu(mu->clone());
#ifdef NEWSCALE
  *scaled_mu *= 0.1F;
#else
  const auto& cartesian_mu = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*mu);
  *scaled_mu *= cartesian_mu.get_voxel_size().x() / 10.F;
#endif
  const auto attenuation_matrix = make_matrix();
  attenuation_matrix->set_restrict_to_cylindrical_FOV(false);
  attenuation_matrix->enable_cache(false);
  attenuation_matrix->set_up(non_tof, scaled_mu);
  ProjMatrixElemsForOneBin attenuation_row;
  BasicCoordinate<3, int> mu_minimum, mu_maximum;
  if (!scaled_mu->get_regular_range(mu_minimum, mu_maximum))
    throw std::runtime_error("OE test requires a regular attenuation image");

  const auto matrix = make_matrix();
  matrix->set_up(non_tof, image);
  BasicCoordinate<3, int> minimum, maximum;
  if (!image->get_regular_range(minimum, maximum))
    throw std::runtime_error("OE test requires a regular image");
  const std::size_t nx = maximum[3] - minimum[3] + 1;
  const std::size_t ny = maximum[2] - minimum[2] + 1;
  std::vector<double> result(image->size_all(), 0.);
  ProjMatrixElemsForOneBin row;
  for (int segment = geometry->get_min_segment_num(); segment <= geometry->get_max_segment_num(); ++segment)
    for (int view = geometry->get_min_view_num(); view <= geometry->get_max_view_num(); ++view)
      for (int axial = geometry->get_min_axial_pos_num(segment); axial <= geometry->get_max_axial_pos_num(segment); ++axial)
        for (int tangential = geometry->get_min_tangential_pos_num(); tangential <= geometry->get_max_tangential_pos_num();
             ++tangential)
          {
            attenuation_matrix->get_proj_matrix_elems_for_one_bin(attenuation_row, Bin(segment, view, axial, tangential, 0, 1.F));
            double line_integral = 0;
            for (const auto& element : attenuation_row)
              {
                const auto coords = element.get_coords();
                if (coords[1] < mu_minimum[1] || coords[1] > mu_maximum[1] || coords[2] < mu_minimum[2]
                    || coords[2] > mu_maximum[2] || coords[3] < mu_minimum[3] || coords[3] > mu_maximum[3])
                  continue;
                line_integral += static_cast<double>(element.get_value()) * (*scaled_mu)[coords[1]][coords[2]][coords[3]];
              }
            const double transmission = std::exp(-line_integral);
            matrix->get_proj_matrix_elems_for_one_bin(row, Bin(segment, view, axial, tangential, 0, 1.F));
            for (const auto& element : row)
              {
                const auto coords = element.get_coords();
                if (coords[1] < minimum[1] || coords[1] > maximum[1] || coords[2] < minimum[2] || coords[2] > maximum[2]
                    || coords[3] < minimum[3] || coords[3] > maximum[3])
                  continue;
                const std::size_t voxel
                    = (static_cast<std::size_t>(coords[1] - minimum[1]) * ny + static_cast<std::size_t>(coords[2] - minimum[2]))
                          * nx
                      + static_cast<std::size_t>(coords[3] - minimum[3]);
                result[voxel] += transmission * element.get_value();
              }
          }
  return result;
}

void
configure(OEReconstruction& reconstruction, const shared_ptr<TestListModeData>& events, const shared_ptr<const Image>& mu)
{
  reconstruction.set_input_data(events);
  reconstruction.set_attenuation_image_sptr(mu);
  reconstruction.set_projection_matrix_sptr(make_matrix());
  reconstruction.set_disable_output(true);
  reconstruction.set_random_seed(37);
  reconstruction.set_number_of_burn_in_sweeps(50);
  reconstruction.set_number_of_samples(200);
  reconstruction.set_sweeps_between_samples(2);
}

} // namespace

class OETests : public RunTests
{
public:
  void run_tests() override;

private:
  void check_sensitivity(const Image& actual, const std::vector<double>& expected, const std::string& description);
  void check_conserved_events(const Image& image, const Image& sensitivity, std::size_t event_count);
};

void
OETests::check_sensitivity(const Image& actual, const std::vector<double>& expected, const std::string& description)
{
  if (!check(actual.size_all() == expected.size(), description + ": image size"))
    return;
  auto value = actual.begin_all();
  double maximum_error = 0;
  for (const double reference : expected)
    maximum_error = std::max(maximum_error, std::fabs(*value++ - reference) / std::max(1., std::fabs(reference)));
  check(maximum_error < 2.E-5, description);
}

void
OETests::check_conserved_events(const Image& image, const Image& sensitivity, std::size_t event_count)
{
  double sum = 0;
  auto efficiency = sensitivity.begin_all();
  bool finite_nonnegative = true;
  for (auto value = image.begin_all(); value != image.end_all(); ++value, ++efficiency)
    {
      finite_nonnegative = finite_nonnegative && std::isfinite(*value) && *value >= 0;
      sum += static_cast<double>(*value) * *efficiency;
    }
  check(finite_nonnegative, "MMSE image is finite and nonnegative");
  check(std::fabs(sum - event_count) < 2.E-5 * std::max<std::size_t>(1, event_count),
        "sensitivity-weighted MMSE image conserves all prompt events");
}

void
OETests::run_tests()
{
  shared_ptr<ExamInfo> exam(new ExamInfo(ImagingModality::PT));
  shared_ptr<Scanner> scanner(new Scanner(Scanner::E953));
  scanner->set_num_rings(1);
  scanner->set_intrinsic_azimuthal_tilt(0.F);
  scanner->set_max_num_timing_poss(3);
  scanner->set_size_of_timing_poss(100.F);
  scanner->set_timing_resolution(200.F);
  scanner->set_up();
  shared_ptr<ProjDataInfo> geometry(ProjDataInfo::construct_proj_data_info(scanner, 1, 0, 4, 3, false, 1));
  check(geometry->get_num_tof_poss() == 3, "test geometry has three TOF bins");
  shared_ptr<Image> image(new VoxelsOnCartesianGrid<float>(
      exam, *geometry, 1.F, CartesianCoordinate3D<float>(0, 0, 0), CartesianCoordinate3D<int>(1, 5, 5)));
  shared_ptr<Image> mu(image->get_empty_copy());
  mu->fill(.096F);
  // Nonuniform attenuation prevents a single global transmission factor from
  // satisfying the sensitivity reference comparison.
  (*mu)[0][0][0] = .15F;
  const auto original_mu = image_values(*mu);
  const Bin central(0, 0, 0, 0, 0, 1.F);
  shared_ptr<TestListModeData> events(new TestListModeData(
      exam, geometry, { TestRecord::make_event(central), TestRecord::make_event(central), TestRecord::make_event(central) }));
  OEReconstruction reconstruction;
  configure(reconstruction, events, mu);
  shared_ptr<CountingRayTracingMatrix> counted_matrix(new CountingRayTracingMatrix);
  reconstruction.set_projection_matrix_sptr(counted_matrix);
  if (!check(reconstruction.set_up(image) == Succeeded::yes, "attenuated TOF OE set_up"))
    return;
  check(reconstruction.get_num_events() == 3, "all three prompt events are loaded");
  check(counted_matrix->get_setup_count() > 0 && counted_matrix->get_row_count() > 0,
        "the supplied projection matrix is used to compute sensitivity");
  check(!counted_matrix->is_cache_enabled(), "matrix instrumentation keeps caching disabled after setup");
  check(counted_matrix->has_only_non_tof_requests(), "sensitivity matrix setup and row requests use non-TOF geometry only");
  const auto full_reference = reference_sensitivity(geometry, image, mu);
  check_sensitivity(reconstruction.get_sensitivity(),
                    full_reference,
                    "attenuated sensitivity equals transmission times the complete non-TOF acquisition matrix");
  check(image_values(*mu) == original_mu, "OE set_up preserves the caller's attenuation image");
  const auto rows_before_sampling = counted_matrix->get_row_count();
  if (!check(reconstruction.reconstruct(image) == Succeeded::yes, "attenuated TOF OE reconstruction"))
    return;
  check(counted_matrix->get_row_count() == rows_before_sampling,
        "direct continuous-LOR burn-in and sample sweeps make no projection-matrix requests");
  check_conserved_events(*image, reconstruction.get_sensitivity(), 3);
  const auto first_image = image_values(*image);
  if (check(reconstruction.set_up(image) == Succeeded::yes, "OE can be set up again after EOF"))
    {
      check(reconstruction.reconstruct(image) == Succeeded::yes, "reconstruction after repeated set_up");
      check(image_values(*image) == first_image, "repeated set_up resets input position and reproduces the seeded ensemble");
    }

  // Sensitivity integrates the geometric acquisition, rather than a finite
  // timing histogram. Neither TOF resolution, timing-window coverage, nor
  // mashing should change it when the spatial acquisition remains identical.
  for (int variant = 0; variant < 3; ++variant)
    {
      shared_ptr<Scanner> timing_scanner(new Scanner(*scanner));
      timing_scanner->set_timing_resolution(variant == 0 ? 500.F : 200.F);
      timing_scanner->set_max_num_timing_poss(variant == 0 ? 3 : 9);
      timing_scanner->set_up();
      const int mash = variant == 2 ? 3 : 1;
      shared_ptr<ProjDataInfo> timing_geometry(ProjDataInfo::construct_proj_data_info(timing_scanner, 1, 0, 4, 3, false, mash));
      check(timing_geometry->get_num_tof_poss() == (variant == 1 ? 9 : 3), "TOF-invariance fixture timing-bin count");
      shared_ptr<TestListModeData> timing_events(new TestListModeData(
          exam,
          timing_geometry,
          { TestRecord::make_event(central), TestRecord::make_event(central), TestRecord::make_event(central) }));
      shared_ptr<Image> timing_image(image->get_empty_copy());
      OEReconstruction timing_reconstruction;
      configure(timing_reconstruction, timing_events, mu);
      if (check(timing_reconstruction.set_up(timing_image) == Succeeded::yes, "alternative TOF metadata OE set_up"))
        check_sensitivity(timing_reconstruction.get_sensitivity(),
                          full_reference,
                          "non-TOF sensitivity is invariant to timing resolution, timing-window coverage, and TOF mashing");
    }

  shared_ptr<Image> zero_mu(mu->get_empty_copy());
  zero_mu->fill(0.F);
  OEReconstruction zero_attenuation;
  configure(zero_attenuation, events, zero_mu);
  if (check(zero_attenuation.set_up(image) == Succeeded::yes, "zero mu-map OE set_up"))
    {
      const auto unattenuated_reference = reference_sensitivity(geometry, image, zero_mu);
      check_sensitivity(
          zero_attenuation.get_sensitivity(), unattenuated_reference, "zero mu-map gives the unattenuated matrix sensitivity");
      bool attenuation_reduces_sensitivity = false;
      for (std::size_t voxel = 0; voxel < full_reference.size(); ++voxel)
        {
          check(full_reference[voxel] <= unattenuated_reference[voxel] + 1.E-5,
                "attenuation never increases detection sensitivity");
          attenuation_reduces_sensitivity
              = attenuation_reduces_sensitivity || full_reference[voxel] < unattenuated_reference[voxel] - 1.E-4;
        }
      check(attenuation_reduces_sensitivity, "nonzero mu-map reduces sensitivity for observable voxels");
    }

  // Both maps have the same larger grid and identical material within the
  // emission image. Only the second map has an extra absorbing shell outside
  // that image, so its lower sensitivity tests that the entire LOR is traced.
  shared_ptr<Image> padded_mu(new VoxelsOnCartesianGrid<float>(
      exam, *geometry, 1.F, CartesianCoordinate3D<float>(0, 0, 0), CartesianCoordinate3D<int>(1, 9, 9)));
  padded_mu->fill(0.F);
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x)
      (*padded_mu)[0][y][x] = (*mu)[0][y][x];
  shared_ptr<Image> shell_mu(padded_mu->clone());
  for (int y = -4; y <= 4; ++y)
    for (int x = -4; x <= 4; ++x)
      if (std::abs(y) > 2 || std::abs(x) > 2)
        (*shell_mu)[0][y][x] = .096F;
  OEReconstruction no_shell;
  OEReconstruction with_shell;
  configure(no_shell, events, padded_mu);
  configure(with_shell, events, shell_mu);
  if (check(no_shell.set_up(image) == Succeeded::yes, "padded mu-map OE set_up")
      && check(with_shell.set_up(image) == Succeeded::yes, "outside-emission attenuation shell OE set_up"))
    {
      check_sensitivity(with_shell.get_sensitivity(),
                        reference_sensitivity(geometry, image, shell_mu),
                        "attenuation shell sensitivity matches the full mu-grid matrix reference");
      const auto without_extra_material = image_values(no_shell.get_sensitivity());
      const auto with_extra_material = image_values(with_shell.get_sensitivity());
      bool shell_reduces_sensitivity = false;
      for (std::size_t voxel = 0; voxel < without_extra_material.size(); ++voxel)
        {
          check(with_extra_material[voxel] <= without_extra_material[voxel] + 1.E-5,
                "material outside the emission image never increases sensitivity");
          shell_reduces_sensitivity
              = shell_reduces_sensitivity || with_extra_material[voxel] < without_extra_material[voxel] - 1.E-4;
        }
      check(shell_reduces_sensitivity, "material outside the emission image contributes to attenuation correction");
    }

  // Multiple rings exercise axial indexing and the two directions of oblique
  // LORs, rather than only the direct one-plane geometry above.
  shared_ptr<Scanner> axial_scanner(new Scanner(*scanner));
  axial_scanner->set_num_rings(3);
  axial_scanner->set_up();
  shared_ptr<ProjDataInfo> axial_geometry(ProjDataInfo::construct_proj_data_info(axial_scanner, 1, 2, 4, 3, false, 1));
  shared_ptr<Image> axial_image(new VoxelsOnCartesianGrid<float>(
      exam, *axial_geometry, 1.F, CartesianCoordinate3D<float>(0, 0, 0), CartesianCoordinate3D<int>(-1, 5, 5)));
  shared_ptr<Image> axial_mu(axial_image->get_empty_copy());
  axial_mu->fill(.096F);
  (*axial_mu)[2][0][0] = .15F;
  const std::vector<TestRecord> axial_records{
    TestRecord::make_event(Bin(0, 0, axial_geometry->get_min_axial_pos_num(0), 0, 0, 1.F)),
    TestRecord::make_event(Bin(0, 2, axial_geometry->get_max_axial_pos_num(0), 0, 1, 1.F)),
    TestRecord::make_event(Bin(1, 0, axial_geometry->get_min_axial_pos_num(1), 0, -1, 1.F)),
    TestRecord::make_event(Bin(1, 1, axial_geometry->get_max_axial_pos_num(1), 0, 0, 1.F)),
    TestRecord::make_event(Bin(-1, 1, axial_geometry->get_min_axial_pos_num(-1), 0, 1, 1.F)),
    TestRecord::make_event(Bin(2, 2, axial_geometry->get_min_axial_pos_num(2), 0, 0, 1.F)),
    TestRecord::make_event(Bin(-2, 3, axial_geometry->get_min_axial_pos_num(-2), 0, -1, 1.F))
  };
  shared_ptr<TestListModeData> axial_events(new TestListModeData(exam, axial_geometry, axial_records));
  OEReconstruction axial_reconstruction;
  configure(axial_reconstruction, axial_events, axial_mu);
  if (check(axial_reconstruction.set_up(axial_image) == Succeeded::yes, "multi-ring oblique TOF OE set_up"))
    {
      check(axial_reconstruction.get_num_events() == axial_records.size(), "multi-ring OE retains all direct and oblique events");
      check_sensitivity(axial_reconstruction.get_sensitivity(),
                        reference_sensitivity(axial_geometry, axial_image, axial_mu),
                        "multi-ring sensitivity matches the complete axial and oblique non-TOF matrix reference");
      if (check(axial_reconstruction.reconstruct(axial_image) == Succeeded::yes, "multi-ring oblique TOF reconstruction"))
        check_conserved_events(*axial_image, axial_reconstruction.get_sensitivity(), axial_records.size());
    }

  // A custom axial size and finer spacing move the image index midpoint away
  // from the default scanner-ring offset. Direct LORs have fixed axial
  // coordinates, so each event must remain in its corresponding image plane.
  shared_ptr<Image> custom_axial_image(new VoxelsOnCartesianGrid<float>(exam,
                                                                        *axial_geometry,
                                                                        CartesianCoordinate3D<float>(2.F, 1.F, 1.F),
                                                                        CartesianCoordinate3D<float>(0, 0, 0),
                                                                        CartesianCoordinate3D<int>(11, 5, 5)));
  const std::vector<TestRecord> custom_axial_records{ TestRecord::make_event(Bin(0, 0, 0, 0, -1, 1.F)),
                                                      TestRecord::make_event(Bin(0, 1, 1, 0, 0, 1.F)),
                                                      TestRecord::make_event(Bin(0, 2, 2, 0, 1, 1.F)) };
  shared_ptr<TestListModeData> custom_axial_events(new TestListModeData(exam, axial_geometry, custom_axial_records));
  OEReconstruction custom_axial_reconstruction;
  configure(custom_axial_reconstruction, custom_axial_events, axial_mu);
  if (check(custom_axial_reconstruction.set_up(custom_axial_image) == Succeeded::yes, "custom axial grid TOF OE set_up"))
    {
      check_sensitivity(custom_axial_reconstruction.get_sensitivity(),
                        reference_sensitivity(axial_geometry, custom_axial_image, axial_mu),
                        "custom axial size and spacing sensitivity matches the non-TOF reference");
      if (check(custom_axial_reconstruction.reconstruct(custom_axial_image) == Succeeded::yes,
                "custom axial grid TOF reconstruction"))
        {
          check_conserved_events(*custom_axial_image, custom_axial_reconstruction.get_sensitivity(), custom_axial_records.size());
          // Ring spacing is four custom voxels. The image midpoint is plane 5;
          // the first, middle, and last direct ring LORs occupy planes 1,5,9.
          for (int z = 0; z < 11; ++z)
            {
              double plane_count = 0;
              for (int y = -2; y <= 2; ++y)
                for (int x = -2; x <= 2; ++x)
                  plane_count += static_cast<double>((*custom_axial_image)[z][y][x])
                                 * custom_axial_reconstruction.get_sensitivity()[z][y][x];
              const double expected = z == 1 || z == 5 || z == 9 ? 1. : 0.;
              check(std::fabs(plane_count - expected) < 2.E-5,
                    "custom axial proposals place each direct-ring event in the sensitivity tracer's selected plane");
            }
        }
    }

  const std::vector<TestRecord> timed_records{ TestRecord::make_time(0),
                                               TestRecord::make_event(central),
                                               TestRecord::make_time(1000),
                                               TestRecord::make_event(central),
                                               TestRecord::make_event(central),
                                               TestRecord::make_event(central, false),
                                               TestRecord::make_event(Bin(0, 0, 0, 0, 1, 1.F)),
                                               TestRecord::make_event(Bin(0, 1, 0, 0, -1, 1.F)),
                                               TestRecord::make_event(Bin(0, 0, 0, 0, 2, 1.F)), // outside TOF geometry
                                               TestRecord::make_time(2000),
                                               TestRecord::make_event(central),
                                               TestRecord::make_time(3000),
                                               TestRecord::make_event(central) };
  shared_ptr<TestListModeData> timed_events(new TestListModeData(exam, geometry, timed_records));
  OEReconstruction selected;
  configure(selected, timed_events, mu);
  selected.set_time_interval(1., 2.);
  if (check(selected.set_up(image) == Succeeded::yes, "OE time-selection set_up"))
    {
      check(selected.get_num_events() == 4, "[start,end) excludes delayed events, outside bins, and endpoint events");
      check_sensitivity(selected.get_sensitivity(), full_reference, "time selection leaves full acquisition sensitivity intact");
      if (check(selected.reconstruct(image) == Succeeded::yes, "OE reconstructs events in distinct TOF bins"))
        check_conserved_events(*image, selected.get_sensitivity(), 4);
      const auto& frames = image->get_exam_info().get_time_frame_definitions();
      if (check(frames.get_num_frames() == 1, "selected output has one time frame"))
        {
          check_if_equal(frames.get_start_time(1), 1., "selected output frame starts at the requested time");
          check_if_equal(frames.get_end_time(1), 2., "selected output frame ends at the requested time");
        }
      check(events->get_exam_info().get_time_frame_definitions().get_num_frames() == 0,
            "output frame metadata preserves the input exam information");
    }

  OEReconstruction invalid;
  invalid.set_input_data(events);
  invalid.set_disable_output(true);
  check(invalid.set_up(image) == Succeeded::no, "OE rejects a missing attenuation image");
  shared_ptr<Image> empty_mu(new VoxelsOnCartesianGrid<float>);
  invalid.set_attenuation_image_sptr(empty_mu);
  check(invalid.set_up(image) == Succeeded::no, "OE rejects an empty attenuation image before projecting it");
  shared_ptr<Image> rectangular_mu(new VoxelsOnCartesianGrid<float>(
      exam, *geometry, 1.F, CartesianCoordinate3D<float>(0, 0, 0), CartesianCoordinate3D<int>(1, 5, 9)));
  rectangular_mu->fill(.096F);
  invalid.set_attenuation_image_sptr(rectangular_mu);
  check(invalid.set_up(image) == Succeeded::no, "OE rejects an unsupported rectangular attenuation grid");
  shared_ptr<Image> nonnegative_indices_mu(
      new VoxelsOnCartesianGrid<float>(exam,
                                       IndexRange<3>(CartesianCoordinate3D<int>(0, 0, 0), CartesianCoordinate3D<int>(0, 4, 4)),
                                       CartesianCoordinate3D<float>(0, 0, 0),
                                       dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*mu).get_voxel_size()));
  nonnegative_indices_mu->fill(.096F);
  invalid.set_attenuation_image_sptr(nonnegative_indices_mu);
  check(invalid.set_up(image) == Succeeded::no,
        "OE rejects zero-origin attenuation indices that do not span the transaxial center");
  const float invalid_coefficients[]{ -.01F, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity() };
  for (const float coefficient : invalid_coefficients)
    {
      shared_ptr<Image> invalid_mu(mu->clone());
      *invalid_mu->begin_all() = coefficient;
      invalid.set_attenuation_image_sptr(invalid_mu);
      check(invalid.set_up(image) == Succeeded::no, "OE rejects negative or nonfinite attenuation coefficients");
      check(invalid.get_num_events() == 0, "failed OE set_up leaves no active ensemble");
    }
  invalid.set_attenuation_image_sptr(mu);
  for (const int samples : { 0, -1 })
    {
      invalid.set_number_of_samples(samples);
      check(invalid.set_up(image) == Succeeded::no, "OE rejects nonpositive sample counts");
    }
  invalid.set_number_of_samples(1);
  invalid.set_number_of_burn_in_sweeps(-1);
  check(invalid.set_up(image) == Succeeded::no, "OE rejects negative burn-in sweeps");
  invalid.set_number_of_burn_in_sweeps(0);
  invalid.set_sweeps_between_samples(0);
  check(invalid.set_up(image) == Succeeded::no, "OE rejects nonpositive sweeps between samples");
  invalid.set_sweeps_between_samples(1);
  const std::vector<std::pair<double, double>> invalid_intervals{ { -1., 2. },
                                                                  { 1., 1. },
                                                                  { 2., 1. },
                                                                  { 0., -2. },
                                                                  { std::numeric_limits<double>::quiet_NaN(), 2. },
                                                                  { 0., std::numeric_limits<double>::infinity() } };
  for (const auto& interval : invalid_intervals)
    {
      invalid.set_time_interval(interval.first, interval.second);
      check(invalid.set_up(image) == Succeeded::no, "OE rejects invalid time intervals");
      check(invalid.get_num_events() == 0, "invalid parameters leave no active ensemble");
    }
}

END_NAMESPACE_STIR

int
main()
{
  stir::OETests tests;
  try
    {
      tests.run_tests();
    }
  catch (const std::exception& exception)
    {
      tests.check(false, std::string("unexpected exception in OE integration tests: ") + exception.what());
    }
  return tests.main_return_value();
}
