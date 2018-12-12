#ifndef GRAVITY_FFTGRAVITY_HPP_
#define GRAVITY_FFTGRAVITY_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fftgravity.hpp
//  \brief defines FFTGravity class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/athena_fft.hpp"

class MeshBlock;
class ParameterInput;
class Coordinates;
class FFTBlock;
class FFTDriver;
class GravitySolverTaskList;

//! \class FFTGravity
//  \brief FFT gravity solver for each block

class FFTGravity : public FFTBlock {
public:
  FFTGravity(FFTDriver *pfd, LogicalLocation iloc, int igid,
           RegionSize msize, RegionSize bsize)
  : FFTBlock(pfd, iloc, igid, msize, bsize) {}
  ~FFTGravity() {}
  void ApplyKernel(int mode) final;
};


//! \class FFTDriver
//  \brief FFT gravity solver

class FFTGravityDriver : public FFTDriver{
public:
  FFTGravityDriver(Mesh *pm, ParameterInput *pin);
  ~FFTGravityDriver();
  void Solve(int stage, int mode); // unlike MultigridDriver::Solve(), not inherited

private:
  Real four_pi_G_;
  GravitySolverTaskList *gtlist_;
};

#endif // GRAVITY_FFTGRAVITY_HPP_
