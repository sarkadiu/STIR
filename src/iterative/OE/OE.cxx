/*
    Copyright (C) 2026 STIR contributors
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details.
*/
/*! \file \brief Command-line list-mode origin-ensemble PET reconstruction. */
#include "stir/OE/OEReconstruction.h"
#include "stir/Succeeded.h"
#include <cstdlib>
#include <exception>
#include <iostream>

int
main(int argc, char** argv)
{
  if (argc != 2)
    {
      std::cerr << "Usage: OE parameter_file\n";
      return EXIT_FAILURE;
    }
  try
    {
      stir::OEReconstruction reconstruction(argv[1]);
      return reconstruction.reconstruct() == stir::Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
    }
  catch (const std::exception& exception)
    {
      std::cerr << "OE: " << exception.what() << '\n';
      return EXIT_FAILURE;
    }
}
