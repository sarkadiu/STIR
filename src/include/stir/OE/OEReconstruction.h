/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.
*/
/*! \file
    \ingroup recon_buildblock
    \brief Attenuation-corrected list-mode origin-ensemble PET reconstruction.
*/

#ifndef __stir_OE_OEReconstruction_H__
#define __stir_OE_OEReconstruction_H__

#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ParseAndCreateFrom.h"
#include "stir/OE/OriginEnsemble.h"
#include "stir/Bin.h"
#include <memory>
#include <vector>

START_NAMESPACE_STIR

class ListModeData;
class ProjDataInfo;
class ProjMatrixByBin;

/*! \brief MMSE origin ensembles with direct TOF-weighted LOR position proposals.
    \ingroup recon_buildblock

    For an event move i -> j, the probability of acceptance is
    min(1, (c_j + 1) / c_i * sensitivity_i / sensitivity_j).
    Sensitivity sums the attenuated non-TOF matrix over all spatial acquisition
    bins, including bins without observed events. This assumes complete timing
    coverage of the normalized TOF kernel. The result is the mean voxel
    occupancy divided by sensitivity. See documentation/OE.md.
*/
class OEReconstruction : public RegisteredParsingObject<OEReconstruction, Reconstruction<DiscretisedDensity<3, float>>>
{
public:
  typedef DiscretisedDensity<3, float> target_type;
  static const char* const registered_name;

  OEReconstruction();
  explicit OEReconstruction(const std::string& parameter_filename);
  ~OEReconstruction() override;

  std::string method_info() const override;
  Succeeded reconstruct() override;
  Succeeded reconstruct(const shared_ptr<target_type>& target_image_sptr) override;
  Succeeded set_up(const shared_ptr<target_type>& target_image_sptr) override;

  void set_input_data(const shared_ptr<ExamData>& data) override;
  const ExamData& get_input_data() const override;
  void set_attenuation_image_sptr(const shared_ptr<const target_type>& image);
  //! Matrix used only for non-TOF sensitivity, never for event proposals.
  void set_projection_matrix_sptr(const shared_ptr<ProjMatrixByBin>& matrix);
  void set_number_of_burn_in_sweeps(int value);
  void set_number_of_samples(int value);
  void set_sweeps_between_samples(int value);
  void set_random_seed(int value);
  void set_maximum_segment_number(int value);
  //! Time interval [start, end) in seconds from scan start; end == -1 means EOF.
  void set_time_interval(double start, double end);

  const target_type& get_sensitivity() const;
  std::size_t get_num_events() const;

protected:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

private:
  typedef Reconstruction<target_type> base_type;
  bool parameters_are_valid() const;
  void invalidate();
  void read_events(std::vector<Bin>& bins, std::vector<std::size_t>& event_proposals);

  std::string input_filename;
  std::string attenuation_image_filename;
  shared_ptr<ListModeData> input_data_sptr;
  shared_ptr<const target_type> attenuation_image_sptr;
  shared_ptr<ProjMatrixByBin> projection_matrix_sptr;
  shared_ptr<ProjDataInfo> proj_data_info_sptr;
  shared_ptr<target_type> sensitivity_sptr;
  ParseAndCreateFrom<target_type, ListModeData> target_parameters;
  std::unique_ptr<OriginEnsemble> ensemble;
  int number_of_burn_in_sweeps;
  int number_of_samples;
  int sweeps_between_samples;
  int random_seed;
  int maximum_segment_number;
  double start_time;
  double end_time;
  std::size_t num_events;
};

END_NAMESPACE_STIR
#endif
