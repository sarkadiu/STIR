# Copyright (C) 2026 STIR contributors
# SPDX-License-Identifier: Apache-2.0
# See STIR/LICENSE.txt for details.

set(dir iterative_OE)
set(dir_LIB_SOURCES ${dir}_LIB_SOURCES)
set(${dir_LIB_SOURCES} OEReconstruction.cxx OriginEnsemble.cxx)
include(stir_lib_target)
target_link_libraries(iterative_OE PUBLIC ${STIR_RECON_BUILDBLOCK_LIB})
# Preserve finite-value validation and small TOF probabilities when the STIR
# GCC Release defaults enable fast-math. The link option also prevents FTZ/DAZ.
target_compile_options(iterative_OE PUBLIC $<$<CXX_COMPILER_ID:GNU>:-fno-fast-math>)
target_link_options(iterative_OE INTERFACE $<$<CXX_COMPILER_ID:GNU>:-fno-fast-math>)

set(dir_EXE_SOURCES ${dir}_EXE_SOURCES)
set(${dir_EXE_SOURCES} OE.cxx)
include(stir_exe_targets)
