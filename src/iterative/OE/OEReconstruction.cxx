/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.
*/
/*! \file
    \ingroup recon_buildblock
    \brief Implementation of attenuation-corrected origin ensembles.
*/

#include "stir/OE/OEReconstruction.h"
#include "stir/listmode/ListModeData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/listmode/ListEvent.h"
#include "stir/listmode/ListTime.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "stir/TOF_conversions.h"
#include "stir/LORCoordinates.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/format.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

START_NAMESPACE_STIR

namespace
{
typedef std::tuple<int, int, int, int, int> BinKey;

BinKey
bin_key(const Bin& bin)
{
  return std::make_tuple(bin.segment_num(), bin.view_num(), bin.axial_pos_num(), bin.tangential_pos_num(), bin.timing_pos_num());
}

// Sensitivity visits the full acquisition matrix. Do not retain it in the
// projector's unbounded cache: event proposals do not use matrix rows.
class ScopedDisableMatrixCache
{
public:
  explicit ScopedDisableMatrixCache(ProjMatrixByBin& matrix)
      : matrix(matrix),
        was_enabled(matrix.is_cache_enabled())
  {
    matrix.enable_cache(false);
  }
  ~ScopedDisableMatrixCache() { matrix.enable_cache(was_enabled); }

private:
  ProjMatrixByBin& matrix;
  bool was_enabled;
};
} // namespace

const char* const OEReconstruction::registered_name = "OE";

OEReconstruction::OEReconstruction()
{
  set_defaults();
}

OEReconstruction::OEReconstruction(const std::string& parameter_filename)
{
  set_defaults();
  initialise(parameter_filename);
}

OEReconstruction::~OEReconstruction() = default;

std::string
OEReconstruction::method_info() const
{
  return "Origin ensemble PET (MMSE, direct TOF LOR sampling, attenuation correction)";
}

void
OEReconstruction::invalidate()
{
  this->_already_set_up = false;
  ensemble.reset();
  sensitivity_sptr.reset();
  num_events = 0;
}

void
OEReconstruction::set_defaults()
{
  base_type::set_defaults();
  input_filename.clear();
  attenuation_image_filename.clear();
  input_data_sptr.reset();
  attenuation_image_sptr.reset();
  projection_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing);
  proj_data_info_sptr.reset();
  target_parameters.set_defaults();
  number_of_burn_in_sweeps = 100;
  number_of_samples = 100;
  sweeps_between_samples = 1;
  random_seed = 1;
  maximum_segment_number = -1;
  start_time = 0;
  end_time = -1;
  invalidate();
}

void
OEReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("OE Parameters");
  this->parser.add_key("input filename", &input_filename);
  this->parser.add_key("attenuation image filename", &attenuation_image_filename);
  this->parser.add_parsing_key("projection matrix type", &projection_matrix_sptr);
  this->parser.add_key("number of burn-in sweeps", &number_of_burn_in_sweeps);
  this->parser.add_key("number of samples", &number_of_samples);
  this->parser.add_key("sweeps between samples", &sweeps_between_samples);
  this->parser.add_key("random seed", &random_seed);
  this->parser.add_key("maximum segment number", &maximum_segment_number);
  this->parser.add_key("start time", &start_time);
  this->parser.add_key("end time", &end_time);
  target_parameters.add_to_keymap(this->parser);
  this->parser.add_stop_key("END OE Parameters");
}

bool
OEReconstruction::parameters_are_valid() const
{
  if (number_of_burn_in_sweeps < 0 || number_of_samples <= 0 || sweeps_between_samples <= 0 || random_seed < 0
      || maximum_segment_number < -1 || !std::isfinite(start_time) || start_time < 0 || !std::isfinite(end_time)
      || (end_time != -1 && end_time <= start_time))
    {
      warning("OE: invalid sampling, segment-range, seed, or time-interval parameters");
      return false;
    }
  return true;
}

bool
OEReconstruction::post_processing()
{
  if (base_type::post_processing() || !parameters_are_valid())
    return true;
  target_parameters.check_values();
  if (input_filename.empty() || attenuation_image_filename.empty())
    {
      warning("OE requires an input list-mode filename and an attenuation image filename");
      return true;
    }
  input_data_sptr = read_from_file<ListModeData>(input_filename);
  attenuation_image_sptr = read_from_file<target_type>(attenuation_image_filename);
  return !input_data_sptr || !attenuation_image_sptr;
}

void
OEReconstruction::set_input_data(const shared_ptr<ExamData>& data)
{
  const auto listmode = std::dynamic_pointer_cast<ListModeData>(data);
  if (!listmode)
    throw std::invalid_argument("OE requires ListModeData input");
  invalidate();
  input_data_sptr = listmode;
}

const ExamData&
OEReconstruction::get_input_data() const
{
  if (!input_data_sptr)
    throw std::logic_error("OE input data have not been set");
  return *input_data_sptr;
}

void
OEReconstruction::set_attenuation_image_sptr(const shared_ptr<const target_type>& image)
{
  invalidate();
  attenuation_image_sptr = image;
}

void
OEReconstruction::set_projection_matrix_sptr(const shared_ptr<ProjMatrixByBin>& matrix)
{
  invalidate();
  projection_matrix_sptr = matrix;
}

void
OEReconstruction::set_number_of_burn_in_sweeps(int value)
{
  invalidate();
  number_of_burn_in_sweeps = value;
}

void
OEReconstruction::set_number_of_samples(int value)
{
  invalidate();
  number_of_samples = value;
}

void
OEReconstruction::set_sweeps_between_samples(int value)
{
  invalidate();
  sweeps_between_samples = value;
}

void
OEReconstruction::set_random_seed(int value)
{
  invalidate();
  random_seed = value;
}

void
OEReconstruction::set_maximum_segment_number(int value)
{
  invalidate();
  maximum_segment_number = value;
}

void
OEReconstruction::set_time_interval(double start, double end)
{
  invalidate();
  start_time = start;
  end_time = end;
}

const OEReconstruction::target_type&
OEReconstruction::get_sensitivity() const
{
  if (!this->_already_set_up || !sensitivity_sptr)
    throw std::logic_error("OE sensitivity requested before successful set_up");
  return *sensitivity_sptr;
}

std::size_t
OEReconstruction::get_num_events() const
{
  return num_events;
}

void
OEReconstruction::read_events(std::vector<Bin>& bins, std::vector<std::size_t>& event_proposals)
{
  if (input_data_sptr->reset() != Succeeded::yes)
    throw std::runtime_error("OE could not rewind the list-mode input");
  const auto record = input_data_sptr->get_empty_record_sptr();
  if (!record || !record->event().is_valid_template(*proj_data_info_sptr))
    throw std::runtime_error("OE list-mode reader cannot use this projection geometry");
  std::map<BinKey, std::size_t> proposal_indices;
  double current_time = 0;
  std::size_t ignored = 0;
  while (input_data_sptr->get_next_record(*record) == Succeeded::yes)
    {
      if (record->is_time())
        {
          current_time = record->time().get_time_in_secs();
          if (end_time >= 0 && current_time >= end_time)
            break;
        }
      if (current_time < start_time || !record->is_event() || !record->event().is_prompt())
        continue;
      Bin bin;
      bin.set_bin_value(1.F);
      record->event().get_bin(bin, *proj_data_info_sptr);
      if (bin.get_bin_value() <= 0)
        {
          ++ignored;
          continue;
        }
      if (bin.get_bin_value() != 1.F)
        throw std::runtime_error("OE requires unit-weight prompt events");
      if (bin.segment_num() < proj_data_info_sptr->get_min_segment_num()
          || bin.segment_num() > proj_data_info_sptr->get_max_segment_num()
          || bin.view_num() < proj_data_info_sptr->get_min_view_num() || bin.view_num() > proj_data_info_sptr->get_max_view_num()
          || bin.axial_pos_num() < proj_data_info_sptr->get_min_axial_pos_num(bin.segment_num())
          || bin.axial_pos_num() > proj_data_info_sptr->get_max_axial_pos_num(bin.segment_num())
          || bin.tangential_pos_num() < proj_data_info_sptr->get_min_tangential_pos_num()
          || bin.tangential_pos_num() > proj_data_info_sptr->get_max_tangential_pos_num()
          || bin.timing_pos_num() < proj_data_info_sptr->get_min_tof_pos_num()
          || bin.timing_pos_num() > proj_data_info_sptr->get_max_tof_pos_num())
        {
          ++ignored;
          continue;
        }
      const auto inserted = proposal_indices.emplace(bin_key(bin), bins.size());
      if (inserted.second)
        bins.push_back(bin);
      event_proposals.push_back(inserted.first->second);
    }
  if (event_proposals.empty())
    throw std::runtime_error("OE found no usable prompt events in the selected interval and geometry");
  info(format("OE: read {} prompt events in {} distinct bins; ignored {} events outside the selected geometry",
              event_proposals.size(),
              bins.size(),
              ignored));
}

Succeeded
OEReconstruction::set_up(const shared_ptr<target_type>& target_image_sptr)
{
  invalidate();
  if (!parameters_are_valid() || !target_image_sptr || !input_data_sptr || !attenuation_image_sptr || !projection_matrix_sptr)
    {
      warning("OE set_up requires an image, list-mode data, a mu-map, and a projection matrix");
      return Succeeded::no;
    }
  const auto modality = input_data_sptr->get_exam_info_sptr()->imaging_modality.get_modality();
  if (modality != ImagingModality::PT && modality != ImagingModality::Unknown)
    {
      warning("OE currently supports PET only");
      return Succeeded::no;
    }
  const auto* mu_grid = dynamic_cast<const VoxelsOnCartesianGrid<float>*>(attenuation_image_sptr.get());
  BasicCoordinate<3, int> mu_minimum, mu_maximum;
  if (!mu_grid || mu_grid->size_all() == 0 || !mu_grid->get_regular_range(mu_minimum, mu_maximum))
    {
      warning("OE requires a nonempty regular Cartesian attenuation image");
      return Succeeded::no;
    }
  for (int axis = 1; axis <= 3; ++axis)
    if (!std::isfinite(mu_grid->get_voxel_size()[axis]) || mu_grid->get_voxel_size()[axis] <= 0
        || !std::isfinite(mu_grid->get_origin()[axis]))
      {
        warning("OE attenuation image requires finite origins and positive finite voxel sizes");
        return Succeeded::no;
      }
  // STIR's matrix tracer uses a centered square transaxial projection region,
  // even with cylindrical restriction disabled. Reject incompatible maps
  // instead of silently omitting attenuation in a longer image dimension.
  if (std::abs(mu_grid->get_origin().x()) > .05F || std::abs(mu_grid->get_origin().y()) > .05F || mu_minimum[2] != mu_minimum[3]
      || mu_maximum[2] != mu_maximum[3] || mu_minimum[2] >= 0 || mu_maximum[2] <= 0
      || std::abs(mu_grid->get_voxel_size().x() - mu_grid->get_voxel_size().y()) > 1.E-4F)
    {
      warning("OE attenuation requires a centered square transaxial grid with equal x/y voxel sizes; "
              "pad or resample the registered mu-map to a compatible grid");
      return Succeeded::no;
    }
  for (auto iter = attenuation_image_sptr->begin_all(); iter != attenuation_image_sptr->end_all(); ++iter)
    if (!std::isfinite(*iter) || *iter < 0)
      {
        warning("OE attenuation coefficients must be finite and nonnegative (in cm^-1)");
        return Succeeded::no;
      }

  BasicCoordinate<3, int> minimum, maximum;
  const auto* target_grid = dynamic_cast<const VoxelsOnCartesianGrid<float>*>(target_image_sptr.get());
  if (!target_grid || !target_image_sptr->get_regular_range(minimum, maximum) || target_image_sptr->size_all() == 0)
    {
      warning("OE requires a nonempty regular Cartesian image grid");
      return Succeeded::no;
    }
  LORProposal::Grid proposal_grid;
  for (int axis = 1; axis <= 3; ++axis)
    {
      proposal_grid.origin[axis - 1] = target_grid->get_origin()[axis];
      proposal_grid.spacing[axis - 1] = target_grid->get_voxel_size()[axis];
      proposal_grid.minimum[axis - 1] = minimum[axis];
      proposal_grid.size[axis - 1] = static_cast<std::size_t>(maximum[axis] - minimum[axis] + 1);
      if (!std::isfinite(proposal_grid.origin[axis - 1]) || !std::isfinite(proposal_grid.spacing[axis - 1])
          || proposal_grid.spacing[axis - 1] <= 0)
        {
          warning("OE emission image requires finite origins and positive finite voxel sizes");
          return Succeeded::no;
        }
    }
  proj_data_info_sptr = input_data_sptr->get_proj_data_info_sptr()->create_shared_clone();
  if (maximum_segment_number >= 0)
    proj_data_info_sptr->reduce_segment_range(std::max(proj_data_info_sptr->get_min_segment_num(), -maximum_segment_number),
                                              std::min(proj_data_info_sptr->get_max_segment_num(), maximum_segment_number));
  const auto& scanner = *proj_data_info_sptr->get_scanner_ptr();
  if (proj_data_info_sptr->get_num_tof_poss() > 1
      && (!scanner.is_tof_ready() || !std::isfinite(scanner.get_timing_resolution()) || scanner.get_timing_resolution() <= 0
          || !std::isfinite(scanner.get_size_of_timing_pos()) || scanner.get_size_of_timing_pos() <= 0))
    {
      warning("OE TOF reconstruction requires valid scanner timing metadata");
      return Succeeded::no;
    }
  ScopedDisableMatrixCache cache_guard(*projection_matrix_sptr);
  // A normalized TOF kernel integrates to one across the acquisition timing
  // range. As in STIR's list-mode objective, sensitivity uses non-TOF geometry.
  const auto non_tof_info = proj_data_info_sptr->create_non_tof_clone();
  projection_matrix_sptr->set_up(non_tof_info, target_image_sptr);
  std::vector<Bin> occupied_bins;
  std::vector<std::size_t> event_proposals;
  read_events(occupied_bins, event_proposals);
  std::map<BinKey, std::size_t> occupied_spatial_bins;
  for (std::size_t i = 0; i < occupied_bins.size(); ++i)
    {
      Bin spatial_bin(occupied_bins[i]);
      spatial_bin.timing_pos_num() = 0;
      occupied_spatial_bins.emplace(bin_key(spatial_bin), i);
    }
  std::vector<double> sensitivity(target_image_sptr->size_all(), 0.);
  const std::size_t size_x = maximum[3] - minimum[3] + 1;
  const std::size_t size_y = maximum[2] - minimum[2] + 1;

  // PET transmission is independent of timing and the origin along the LOR.
  // BinNormalisationFromAttenuationImage only accepts non-TOF geometry and
  // provides transmission through undo(ones); get_bin_efficiency is unimplemented.
  // Trace through the supported mu-map region, including material outside the emission
  // image FOV. A separate matrix supports the mu-map's own voxel spacing and
  // origin, and avoids the legacy attenuation projector's geometry limits.
  shared_ptr<ProjMatrixByBinUsingRayTracing> attenuation_matrix(new ProjMatrixByBinUsingRayTracing);
  attenuation_matrix->set_restrict_to_cylindrical_FOV(false);
  // The sensitivity loop requests one viewgram at a time. Its attenuation
  // forward projector must use the same trivial view/segment grouping.
  attenuation_matrix->set_do_symmetry_90degrees_min_phi(false);
  attenuation_matrix->set_do_symmetry_180degrees_min_phi(false);
  attenuation_matrix->set_do_symmetry_swap_segment(false);
  attenuation_matrix->set_do_symmetry_swap_s(false);
  attenuation_matrix->set_do_symmetry_shift_z(false);
  attenuation_matrix->enable_cache(false);
  shared_ptr<ForwardProjectorByBin> attenuation_projector(new ForwardProjectorByBinUsingProjMatrixByBin(attenuation_matrix));
  BinNormalisationFromAttenuationImage attenuation(attenuation_image_sptr, attenuation_projector);
  if (attenuation.set_up(input_data_sptr->get_exam_info_sptr(), non_tof_info) != Succeeded::yes)
    return Succeeded::no;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> trivial(new TrivialDataSymmetriesForBins(non_tof_info));
  ProjMatrixElemsForOneBin matrix_row;
  info("OE: computing attenuation-weighted non-TOF sensitivity over all spatial acquisition bins");
  for (int segment = proj_data_info_sptr->get_min_segment_num(); segment <= proj_data_info_sptr->get_max_segment_num(); ++segment)
    for (int view = proj_data_info_sptr->get_min_view_num(); view <= proj_data_info_sptr->get_max_view_num(); ++view)
      {
        auto transmission = non_tof_info->get_empty_related_viewgrams(ViewgramIndices(view, segment), trivial);
        transmission.fill(1.F);
        attenuation.undo(transmission);
        const auto& viewgram = *transmission.begin();
        for (int axial = proj_data_info_sptr->get_min_axial_pos_num(segment);
             axial <= proj_data_info_sptr->get_max_axial_pos_num(segment);
             ++axial)
          for (int tangential = proj_data_info_sptr->get_min_tangential_pos_num();
               tangential <= proj_data_info_sptr->get_max_tangential_pos_num();
               ++tangential)
            {
              const double efficiency = viewgram[axial][tangential];
              if (!std::isfinite(efficiency) || efficiency < 0 || efficiency > 1.00001)
                throw std::runtime_error("OE obtained an invalid attenuation transmission factor");
              const Bin bin(segment, view, axial, tangential, 0, 1.F);
              if (efficiency == 0)
                {
                  if (occupied_spatial_bins.count(bin_key(bin)))
                    throw std::runtime_error("OE observed an event in a bin with zero attenuation transmission");
                  continue;
                }
              projection_matrix_sptr->get_proj_matrix_elems_for_one_bin(matrix_row, bin);
              for (const auto& element : matrix_row)
                {
                  const double weight = element.get_value();
                  if (!std::isfinite(weight) || weight < 0)
                    throw std::runtime_error("OE requires finite nonnegative projection matrix weights");
                  const auto coords = element.get_coords();
                  if (weight == 0 || coords[1] < minimum[1] || coords[1] > maximum[1] || coords[2] < minimum[2]
                      || coords[2] > maximum[2] || coords[3] < minimum[3] || coords[3] > maximum[3])
                    continue;
                  const std::size_t voxel = (static_cast<std::size_t>(coords[1] - minimum[1]) * size_y
                                             + static_cast<std::size_t>(coords[2] - minimum[2]))
                                                * size_x
                                            + static_cast<std::size_t>(coords[3] - minimum[3]);
                  sensitivity[voxel] += efficiency * weight;
                }
            }
        info(format("OE sensitivity: segment {}, view {}", segment, view), 3);
      }
  sensitivity_sptr.reset(target_image_sptr->get_empty_copy());
  auto output = sensitivity_sptr->begin_all();
  for (const double value : sensitivity)
    {
      if (!std::isfinite(value) || value > std::numeric_limits<float>::max())
        throw std::runtime_error("OE sensitivity overflow");
      *output++ = static_cast<float>(value);
    }
  std::vector<OriginEnsemble::Proposal> proposals;
  proposals.reserve(occupied_bins.size());
  // Match the sensitivity tracer's scanner-centred to image-centred axial
  // convention, including custom Z zoom and output size. The image origin
  // is subtracted when mapping the physical position to a voxel.
  const double axial_center_offset = (static_cast<double>(minimum[1]) + maximum[1]) * target_grid->get_voxel_size().z() / 2.;
  const bool is_tof = proj_data_info_sptr->get_num_tof_poss() > 1;
  const double tof_sigma = is_tof ? tof_delta_time_to_mm(scanner.get_timing_resolution()) / 2.355 : 0.;
  for (const auto& bin : occupied_bins)
    {
      // The canonical bin LOR keeps its direction consistent with STIR's TOF
      // bin sign. Its scanner-centred z is shifted into the image convention.
      LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
      proj_data_info_sptr->get_LOR(lor, bin);
      const LORAs2Points<float> points(lor);
      std::array<double, 3> point1, point2;
      for (int axis = 1; axis <= 3; ++axis)
        {
          point1[axis - 1] = points.p1()[axis] + (axis == 1 ? axial_center_offset : 0.);
          point2[axis - 1] = points.p2()[axis] + (axis == 1 ? axial_center_offset : 0.);
        }
      const double low = is_tof ? proj_data_info_sptr->tof_bin_boundaries_mm[bin.timing_pos_num()].low_lim : 0.;
      const double high = is_tof ? proj_data_info_sptr->tof_bin_boundaries_mm[bin.timing_pos_num()].high_lim : 0.;
      LORProposal proposal(proposal_grid, point1, point2, low, high, tof_sigma);
      if (!proposal.has_intersection())
        throw std::runtime_error(format("OE event LOR (segment {}, view {}, axial {}, tangential {}) misses the image",
                                        bin.segment_num(),
                                        bin.view_num(),
                                        bin.axial_pos_num(),
                                        bin.tangential_pos_num()));
      proposals.emplace_back(std::move(proposal));
    }
  ensemble.reset(new OriginEnsemble(sensitivity, proposals, event_proposals, static_cast<unsigned int>(random_seed)));
  num_events = event_proposals.size();
  if (end_time >= 0)
    {
      ExamInfo exam_info(*input_data_sptr->get_exam_info_sptr());
      exam_info.time_frame_definitions = TimeFrameDefinitions(std::vector<std::pair<double, double>>{ { start_time, end_time } });
      target_image_sptr->set_exam_info(exam_info);
    }
  if (base_type::set_up(target_image_sptr) != Succeeded::yes)
    {
      invalidate();
      return Succeeded::no;
    }
  return Succeeded::yes;
}

Succeeded
OEReconstruction::reconstruct()
{
  if (!input_data_sptr)
    {
      warning("OE input data have not been set");
      return Succeeded::no;
    }
  shared_ptr<target_type> image(target_parameters.create(*input_data_sptr));
  if (set_up(image) != Succeeded::yes)
    return Succeeded::no;
  return reconstruct(image);
}

Succeeded
OEReconstruction::reconstruct(const shared_ptr<target_type>& target_image_sptr)
{
  if (!target_image_sptr)
    return Succeeded::no;
  this->check(*target_image_sptr);
  if (!ensemble || !parameters_are_valid())
    return Succeeded::no;
  for (int sweep = 0; sweep < number_of_burn_in_sweeps; ++sweep)
    ensemble->sweep();
  for (int sample = 0; sample < number_of_samples; ++sample)
    {
      for (int sweep = 0; sweep < sweeps_between_samples; ++sweep)
        ensemble->sweep();
      ensemble->accumulate_sample();
      info(format("OE: accumulated sample {} of {}", sample + 1, number_of_samples), 2);
    }
  const auto result = ensemble->get_mean_image();
  auto output = target_image_sptr->begin_all();
  for (const double value : result)
    {
      if (!std::isfinite(value) || value > std::numeric_limits<float>::max())
        throw std::runtime_error("OE reconstructed image overflow");
      *output++ = static_cast<float>(value);
    }
  this->target_data_sptr = target_image_sptr;
  if (this->post_filter_sptr && this->post_filter_sptr->apply(*target_image_sptr) != Succeeded::yes)
    return Succeeded::no;
  if (!this->_disable_output)
    {
      if (this->output_filename_prefix.empty() || !this->output_file_format_ptr)
        {
          warning("OE output requires a filename prefix and output file format");
          return Succeeded::no;
        }
      return this->output_file_format_ptr->write_to_file(this->output_filename_prefix, *target_image_sptr);
    }
  return Succeeded::yes;
}

END_NAMESPACE_STIR
