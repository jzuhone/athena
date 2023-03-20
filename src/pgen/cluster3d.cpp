/*============================================================================*/
/*! \file cluster3d.cpp
 *  \brief Problem generator for galaxy cluster merger. */
/*============================================================================*/

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../globals.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../eos/eos.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh_refinement.hpp"

#include "hdf5.h"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#define ONE_THIRD (1./3.)
#define ONE_SIXTH (1./6.)
#define RAD_CONV 3.2407792899999994e-22
#define DENS_CONV 1.4775575897980712e+31
#define PRES_CONV 1.54543684e+15
#define GPOT_CONV 1.04594017e-16
#define GRAV_CONV 322743.41425179
#define VPOT_CONV 1.2740166e-14

// Refinement condition methods
int RefinementCondition(MeshBlock *pmb);
Real ComputeCurvature(MeshBlock *pmb, int ivar);

// BCs of grid in each dimension
void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// Gravitational acceleration function
void ClusterAccel(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
                  const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
                  AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar);

namespace {
  // Radial profiles of the cluster(s)
  static Real *r1, *r2, *dens1, *dens2, *pres1, *pres2;
  static Real *grav1, *grav2, *gpot1, *gpot2;
  static Real rmax1, rmax2, mass1, mass2;
  static int num_points1, num_points2;
  // Variables for the cluster trajectories
  static Real xctr1, xctr2, xctr3, xsub1, xsub2, xsub3, vsub1, vsub2, vsub3;
  static Real x1min, x1max, x2min, x2max, x3min, x3max;
  static Real xmain1, xmain2, xmain3, vmain1, vmain2, vmain3, dt_old;
  static Real oamain1, oamain2, oamain3, amain1, amain2, amain3;
  static Real oasub1, oasub2, oasub3, asub1, asub2, asub3;
  // Variables and arrays for the magnetic field input
  static int NAx, NAy, NAz;
  static Real *Axcoords, *Aycoords, *Azcoords, Adx, Ady, Adz;  
  static Real Axmin, Axmax, Aymin, Aymax, Azmin, Azmax;
  // Runtime parameters
  std::string mag_file;
  static int num_halo, main_cluster_fixed, subhalo_gas, sphere_reflevel;
  static Real r_scale, r_cut, min_refine_density, ref_radius1_sq, ref_radius2_sq;
  // Other parameters
  static int res_flag = 0, a_in_cgs = 0;
  // Methods
  Real interpolate(Real a[], int n, Real x[], Real xx);
  void ReadNumPoints(std::string filename, int *num_points);
  void ReadProfiles(int which, std::string filename, Real r[],
                    Real dens[], Real pres[], Real gpot[], Real grav[]);
  Real VecPot(Real *field, Real xx, Real yy, Real zz);
  void ReadFieldPoints(std::string filename, int *nbx, int *nby, int *nbz);
  void ReadFieldCoords(std::string filename, Real xcoords[], Real ycoords[], Real zcoords[]);
  void ReadField(std::string filename, const int ibegin, const int jbegin,
		 const int kbegin, const int iend, const int jend,
		 const int kend, Real Axf[], Real Ayf[], Real Azf[]);  
  Real TSCWeight(Real x);
  Real interp_grav_pot(const Real x1, const Real x2, const Real x3);
  Real noninertial_accel(int axis, const Real x1, const Real x2, const Real x3);
  void update_accel();
} // namespace

/*----------------------------------------------------------------------------
 * problem: 3d galaxy cluster
 */

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{

  Real x_init1, y_init1, x_init2, y_init2;
  Real vx_init1, vy_init1, vx_init2, vy_init2;

  num_halo = pin->GetOrAddInteger("problem", "num_halo", 1);
  main_cluster_fixed = pin->GetOrAddInteger("problem", "main_cluster_fixed",1);
  r_scale = pin->GetOrAddReal("problem", "r_scale", 300.);
  r_cut = pin->GetOrAddReal("problem", "r_cut", 800.);
  subhalo_gas = pin->GetOrAddInteger("problem", "subhalo_gas", 0);
  min_refine_density = pin->GetOrAddReal("problem", "min_refine_density", 0.0);
  ref_radius1_sq = pin->GetOrAddReal("problem", "ref_radius1", 0.0);
  ref_radius2_sq = pin->GetOrAddReal("problem", "ref_radius2", 0.0);
  ref_radius1_sq *= ref_radius1_sq;
  ref_radius2_sq *= ref_radius2_sq;
  sphere_reflevel = pin->GetOrAddReal("problem", "sphere_reflevel", 0) + root_level;
  x1min = mesh_size.x1min;
  x2min = mesh_size.x2min;
  x3min = mesh_size.x3min;
  x1max = mesh_size.x1max;
  x2max = mesh_size.x2max;
  x3max = mesh_size.x3max;

  xctr1 = 0.5*(mesh_size.x1max + mesh_size.x1min);
  xctr2 = 0.5*(mesh_size.x2max + mesh_size.x2min);
  xctr3 = 0.5*(mesh_size.x3max + mesh_size.x3min);

  AllocateRealUserMeshDataField(7);
  ruser_mesh_data[0].NewAthenaArray(3); // xmain*
  ruser_mesh_data[1].NewAthenaArray(3); // vmain*
  ruser_mesh_data[2].NewAthenaArray(3); // oamain*
  ruser_mesh_data[3].NewAthenaArray(3); // xsub*
  ruser_mesh_data[4].NewAthenaArray(3); // vsub*
  ruser_mesh_data[5].NewAthenaArray(3); // oasub*
  ruser_mesh_data[6].NewAthenaArray(1); // dt_old

  if (num_halo == 1) {
    
    xmain1 = xctr1;
    xmain2 = xctr2;
    xmain3 = xctr3;
    vmain1 = 0.0;
    vmain2 = 0.0;
    vmain3 = 0.0;

  } else {
    
    if (ncycle == 0) {
      if (main_cluster_fixed) {
        x_init1 = xctr1;
        y_init1 = xctr2;
        vx_init1 = 0.0;
        vy_init1 = 0.0;
      } else {
        x_init1 = pin->GetOrAddReal("problem", "x_init1", xctr1);
        y_init1 = pin->GetOrAddReal("problem", "y_init1", xctr2);
        vx_init1 = pin->GetOrAddReal("problem", "vx_init1", 0.0);
        vy_init1 = pin->GetOrAddReal("problem", "vy_init1", 0.0);
      }
      x_init2 = pin->GetReal("problem", "x_init2");
      y_init2 = pin->GetReal("problem", "y_init2");
      vx_init2 = pin->GetReal("problem", "vx_init2");
      vy_init2 = pin->GetReal("problem", "vy_init2");
      xmain1 = x_init1;
      xmain2 = y_init1;
      xmain3 = xctr3;
      xsub1 = x_init2;
      xsub2 = y_init2;
      xsub3 = xctr3;
      vmain1 = vx_init1;
      vmain2 = vy_init1;
      vmain3 = 0.0;
      vsub1 = vx_init2;
      vsub2 = vy_init2;
      vsub3 = 0.0;
      oamain1 = 0.0;
      oamain2 = 0.0;
      oamain3 = 0.0;
      oasub1 = 0.0;
      oasub2 = 0.0;
      oasub3 = 0.0;
      dt_old = -1.0;
      update_accel();
    } else {
      res_flag = 1;
    }
 
  }
    
  std::string filename1 = pin->GetString("problem","profile1");
    
  if (Globals::my_rank == 0) {
    ReadNumPoints(filename1, &num_points1);
    std::cout << "num_points1 = " << num_points1 << std::endl;
  }

#ifdef MPI_PARALLEL
  MPI_Bcast(&num_points1, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  r1 = new Real [num_points1];
  dens1 = new Real [num_points1];
  pres1 = new Real [num_points1];
  gpot1 = new Real [num_points1];
  grav1 = new Real [num_points1];

  if (Globals::my_rank == 0) {
    ReadProfiles(1, filename1, r1, dens1, pres1, gpot1, grav1);
    for (int i = 0; i < num_points1; i++) {
      gpot1[i] *= -1.; 
      grav1[i] *= -1.;
    }
  }

  if (r1[num_points1-1] > 1.0e10) {
    // profiles are in CGS units
    for (int i = 0; i < num_points1; i++) {
      r1[i] *= RAD_CONV; 
      dens1[i] *= DENS_CONV;
      pres1[i] *= PRES_CONV;
      gpot1[i] *= GPOT_CONV;
      grav1[i] *= GRAV_CONV;
    }
  }

#ifdef MPI_PARALLEL
  MPI_Bcast(r1, num_points1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(dens1, num_points1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(pres1, num_points1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(gpot1, num_points1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(grav1, num_points1, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
  rmax1 = r1[num_points1-1];
  mass1 = grav1[num_points1-1]*rmax1*rmax1;
  
  if (num_halo == 2) {

    std::string filename2 = pin->GetString("problem","profile2");
    
    if (Globals::my_rank == 0) {
      ReadNumPoints(filename1, &num_points2);
      std::cout << "num_points2 = " << num_points2 << std::endl;
    }

#ifdef MPI_PARALLEL
    MPI_Bcast(&num_points2, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    r2 = new Real [num_points2];
    dens2 = new Real [num_points2];
    pres2 = new Real [num_points2];
    gpot2 = new Real [num_points2];
    grav2 = new Real [num_points2];

    if (Globals::my_rank == 0) {
      ReadProfiles(2, filename2, r2, dens2, pres2, gpot2, grav2);
      for (int i = 0; i < num_points2; i++) {
        gpot2[i] *= -1.;
        grav2[i] *= -1.;
      }
    }

    if (r2[num_points2-1] > 1.0e10) {
      // profiles are in CGS units
      for (int i = 0; i < num_points2; i++) {
        r2[i] *= RAD_CONV; 
        dens2[i] *= DENS_CONV;
        pres2[i] *= PRES_CONV;
        gpot2[i] *= GPOT_CONV;
        grav2[i] *= GRAV_CONV;
      }
    }
  
#ifdef MPI_PARALLEL
    MPI_Bcast(r2, num_points2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(dens2, num_points2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(pres2, num_points2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(gpot2, num_points2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(grav2, num_points2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
    rmax2 = r2[num_points2-1];
    mass2 = grav2[num_points2-1]*rmax2*rmax2;

  }

  if (MAGNETIC_FIELDS_ENABLED && ncycle == 0) {
      
    if (Globals::my_rank == 0)
       std::cout << "Reading magnetic field." << std::endl;

    mag_file = pin->GetString("problem","mag_file");
    
    if (Globals::my_rank == 0) 
      ReadFieldPoints(mag_file, &NAx, &NAy, &NAz);

#ifdef MPI_PARALLEL
    MPI_Bcast(&NAx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NAy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NAz, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if (Globals::my_rank == 0)
      std::cout << "NAx = " << NAx << " NAy = " << NAy << " NAz = "
                << NAz << std::endl;

    Axcoords = new Real [NAx];
    Aycoords = new Real [NAy];
    Azcoords = new Real [NAz];

    if (Globals::my_rank == 0) {
      ReadFieldCoords(mag_file, Axcoords, Aycoords, Azcoords);
      std::cout << "Finished reading potential coordinates." << std::endl;
    }

    if (Axcoords[NAx-1]-Axcoords[0] > 1.0e10) {
      // A-field is in cgs
      a_in_cgs = 1;
      for (int i=0; i < NAx; i++)
        Axcoords[i] *= RAD_CONV;
      for (int j=0; j < NAy; j++)
        Aycoords[j] *= RAD_CONV;
      for (int k=0; k < NAz; k++)
        Azcoords[k] *= RAD_CONV;
    }

#ifdef MPI_PARALLEL
    MPI_Bcast(Axcoords, NAx, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(Aycoords, NAy, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
    MPI_Bcast(Azcoords, NAz, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

    Adx = Axcoords[1] - Axcoords[0];
    Ady = Aycoords[1] - Aycoords[0];
    Adz = Azcoords[1] - Azcoords[0];
    Axmin = Axcoords[0]-0.5*Adx;
    Aymin = Aycoords[0]-0.5*Ady;
    Azmin = Azcoords[0]-0.5*Adz;
    Axmax = Axcoords[NAx-1]+0.5*Adx;
    Aymax = Aycoords[NAy-1]+0.5*Ady;
    Azmax = Azcoords[NAz-1]+0.5*Adz;

    std::stringstream msg;

    if ( mesh_size.x1min < Axmin+2*Adx || mesh_size.x1max >= Axmax-2*Adx ||
         mesh_size.x2min < Aymin+2*Ady || mesh_size.x2max >= Aymax-2*Ady ||
         mesh_size.x3min < Azmin+2*Adz || mesh_size.x3max >= Azmax-2*Adz ) {
      msg << "Input grid is smaller than the simulation domain !!" << std::endl;
      ATHENA_ERROR(msg);
    }

  }

#ifdef MPI_PARALLEL
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (Globals::my_rank == 0)
    std::cout << "Finished with initialization." << std::endl;

  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiodeInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiodeInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiodeOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiodeInnerX3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiodeOuterX3);
  EnrollUserExplicitSourceFunction(ClusterAccel);
  if (adaptive == true)
    EnrollUserRefinementCondition(RefinementCondition);
    
  return;

}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  std::stringstream msg;
  
  Real gm1 = peos->GetGamma() - 1.0;
  int nsubzones = pin->GetOrAddInteger("problem","nsubzones", 3);
  Real nsubzninv = 1.0/nsubzones;
  Real nsubvolinv = nsubzninv*nsubzninv*nsubzninv;

  if (block_size.nx2 == 1 || block_size.nx3 == 1) {
    msg << "### FATAL ERROR in function [MeshBlock::ProblemGenerator]"
        << std::endl << "[cluster3d]: This problem can only be run in 3D!";
    throw std::runtime_error(msg.str().c_str());
  }

  for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
  for (int i=is; i<=ie; i++) {

    Real sum_pres = 0.0;
    Real sum_dens1 = 0.0;
    Real sum_dens2 = 0.0;
  
    for (int kk = 0; kk < nsubzones; kk++) {
      Real xx3 = pcoord->x3f(k) + (kk+0.5)*pcoord->dx3v(k)*nsubzninv;
      for (int jj = 0; jj < nsubzones; jj++) {
	      Real xx2 = pcoord->x2f(j) + (jj+0.5)*pcoord->dx2v(j)*nsubzninv;
	      for (int ii = 0; ii < nsubzones; ii++) {
          Real xx1 = pcoord->x1f(i) + (ii+0.5)*pcoord->dx1v(i)*nsubzninv;
          
          Real rr1 = sqrt(SQR(xx1-xmain1)+SQR(xx2-xmain2)+SQR(xx3-xmain3));
          sum_dens1 += interpolate(dens1, num_points1, r1, rr1);
          sum_pres += interpolate(pres1, num_points1, r1, rr1);

          if (num_halo == 2 && subhalo_gas == 1) {
            Real rr2 = sqrt(SQR(xx1-xsub1)+SQR(xx2-xsub2)+SQR(xx3-xsub3));
            sum_dens2 += interpolate(dens2, num_points2, r2, rr2);
            sum_pres += interpolate(pres2, num_points2, r2, rr2);
          }

	      }
      }
    }

    sum_dens1 *= nsubvolinv;
    sum_dens2 *= nsubvolinv;
    sum_pres *= nsubvolinv;

    phydro->u(IDN,k,j,i) = sum_dens1+sum_dens2;
    phydro->u(IEN,k,j,i) = sum_pres/gm1;
    phydro->u(IM1,k,j,i) = sum_dens1*vmain1+sum_dens2*vsub1;
    phydro->u(IM2,k,j,i) = sum_dens1*vmain2+sum_dens2*vsub2;
    phydro->u(IM3,k,j,i) = 0.0;

  }}}

  if (MAGNETIC_FIELDS_ENABLED) {
	  
    AthenaArray<Real> Ax, Ay, Az;
    double *Axf, *Ayf, *Azf;

    // Compute the beginning and ending indices on the vector potential grid
    // +/- 1 are necessary because we will be computing derivatives

    int ibegin = (int)((block_size.x1min-Axmin)/Adx)-2;
    int jbegin = (int)((block_size.x2min-Aymin)/Ady)-2;
    int kbegin = (int)((block_size.x3min-Azmin)/Adz)-2;

    int iend   = (int)((block_size.x1max-Axmin)/Adx)+2;
    int jend   = (int)((block_size.x2max-Aymin)/Ady)+2;
    int kend   = (int)((block_size.x3max-Azmin)/Adz)+2;

    int nlocx  = iend-ibegin+1;
    int nlocy  = jend-jbegin+1;
    int nlocz  = kend-kbegin+1;

    int fdims[3] = { nlocx, nlocy, nlocz };
    int fbegin[3] = { ibegin, jbegin, kbegin };
    int nloc = nlocx*nlocy*nlocz;

    Ax.NewAthenaArray(block_size.nx1,block_size.nx2,block_size.nx3);
    Ay.NewAthenaArray(block_size.nx1,block_size.nx2,block_size.nx3);
    Az.NewAthenaArray(block_size.nx1,block_size.nx2,block_size.nx3);
    Ax.ZeroClear();
    Ay.ZeroClear();
    Az.ZeroClear();

    Axf = new double [nloc];
    Ayf = new double [nloc];
    Azf = new double [nloc];

    ReadField( mag_file, ibegin, jbegin, kbegin,
               iend, jend, kend, Axf, Ayf, Azf );

    if (a_in_cgs) {
      for (int i=0; i < nloc; i++) {
        Axf[i] *= VPOT_CONV;
        Ayf[i] *= VPOT_CONV;
        Azf[i] *= VPOT_CONV;
      }
    }
   
    int sample_res  = (int)std::pow(2.0, pmy_mesh->max_level - loc.level);
    Real sample_fact = 1.0/sample_res;
    Real xl1, xl2, xl3;
    
    Real dx1 = pcoord->dx1v(0);
    Real dx2 = pcoord->dx2v(0);
    Real dx3 = pcoord->dx3v(0);

    for (int k=ks; k<=ke+1; k++) {
      xl3 = pcoord->x3f(k);
      for (int j=js; j<=je+1; j++) {
        xl2 = pcoord->x2f(j);
	      for (int i=is; i<=ie+1; i++) {
          xl1 = pcoord->x1f(i);
  
          for (int ii = 0; ii < sample_res; ii++) {
            Real dxx1 = (ii+0.5)*dx1*sample_fact;
            Ax(k,j,i) += VecPot(Axf,xl1+dxx1,xl2,xl3);
          }
          
          for (int jj = 0; jj < sample_res; jj++) {
            Real dxx2 = (jj+0.5)*dx2*sample_fact;
            Ay(k,j,i) += VecPot(Ayf,xl1,xl2+dxx2,xl3);
          }
          
          for (int kk = 0; kk < sample_res; kk++) {
            Real dxx3 = (kk+0.5)*dx3*sample_fact;
            Az(k,j,i) += VecPot(Azf,xl1,xl2,xl3+dxx3);
          }
          
          Ax(k,j,i) *= sample_fact;
          Ay(k,j,i) *= sample_fact;
          Az(k,j,i) *= sample_fact;
          
        }
      }
    }

    for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie+1; i++) {

      pfield->b.x1f(k,j,i) = (Az(k,j+1,i)-Az(k,j,i))/dx2 -
	      (Ay(k+1,j,i)-Ay(k,j,i))/dx3;

      }
    }
    } 

    for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je+1; j++) {
      for (int i=is; i<=ie; i++) {

      pfield->b.x2f(k,j,i) = (Ax(k+1,j,i)-Ax(k,j,i))/dx3 -
        (Az(k,j,i+1)-Az(k,j,i))/dx1;

      }
    }
    }

    for (int k=ks; k<=ke+1; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

      pfield->b.x3f(k,j,i) = (Ay(k,j,i+1)-Ay(k,j,i))/dx1 -
        (Ax(k,j+1,i)-Ax(k,j,i))/dx2;

      }
    }
    }
    
    Ax.DeleteAthenaArray();
    Ay.DeleteAthenaArray();
    Az.DeleteAthenaArray();

    delete [] Axf;
    delete [] Ayf;
    delete [] Azf;

  }

  // update total energy
  
  for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
  for (int i=is; i<=ie; i++) {
	  
    phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i)) + 
                                 SQR(phydro->u(IM2,k,j,i)) +
				 SQR(phydro->u(IM1,k,j,i)))/phydro->u(IDN,k,j,i);
                                 
    if (MAGNETIC_FIELDS_ENABLED) {
      phydro->u(IEN,k,j,i) += 0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                                   SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                                   SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));
    }
    
  }}}

  return;
}

void MeshBlock::UserWorkInLoop(void)
{

  Real dt = pmy_mesh->dt;
  Real time = pmy_mesh->time;
  int ncycle = pmy_mesh->ncycle;
  Real wterm, woldterm;
    
  if (num_halo == 1 || lid > 0)
    return;

  if (Globals::my_rank == 0) {

    if (main_cluster_fixed == 0) {
      std::ofstream fp_main;
      fp_main.open("main_trajectory.dat", std::ios::app);
      fp_main << time << "\t" << xmain1 << "\t" << xmain2 << "\t"
              << xmain3 << "\t" << vmain1 << "\t" << vmain2 << "\t"
              << vmain3 << "\t" << amain1 << "\t" << amain2 << "\t"
              << amain3 << "\t" << oamain1 << "\t" << oamain2 << "\t"
              << oamain3 << std::endl;
      fp_main.close();
    }
    
    std::ofstream fp_sub;
    fp_sub.open("sub_trajectory.dat", std::ios::app);
    fp_sub << time << "\t" << "\t" << xsub1 << "\t" << xsub2 << "\t"
           << xsub3 << "\t" << vsub1 << "\t" << vsub2 << "\t"
           << vsub3 << "\t" << asub1 << "\t" << asub2 << "\t"
           << asub3 << "\t" << oasub1 << "\t" << oasub2 << "\t"
           << oasub3 << std::endl;
    fp_sub.close();

  }

  if (ncycle == 0) {
    wterm = 0.5*dt;
    woldterm = 0.0;
  } else {
    wterm = 0.5*dt + ONE_THIRD*dt_old + ONE_SIXTH*dt*dt/dt_old;
    woldterm = ONE_SIXTH*(dt_old*dt_old-dt*dt)/dt_old;
  }

  if (main_cluster_fixed == 0) {
    vmain1 += wterm*amain1 + woldterm*oamain1;
    vmain2 += wterm*amain2 + woldterm*oamain2;
    vmain3 += wterm*amain3 + woldterm*oamain3;
    
    xmain1 += dt*vmain1;
    xmain2 += dt*vmain2;
    xmain3 += dt*vmain3;
    
    oamain1 = amain1;
    oamain2 = amain2;
    oamain3 = amain3;
  }

  vsub1 += wterm*asub1 + woldterm*oasub1;
  vsub2 += wterm*asub2 + woldterm*oasub2;
  vsub3 += wterm*asub3 + woldterm*oasub3;

  xsub1 += dt*vsub1;
  xsub2 += dt*vsub2;
  xsub3 += dt*vsub3;

  oasub1 = asub1;
  oasub2 = asub2;
  oasub3 = asub3;

  update_accel();
 
  dt_old = dt;

  pmy_mesh->ruser_mesh_data[0](0) = xmain1;
  pmy_mesh->ruser_mesh_data[0](1) = xmain2;
  pmy_mesh->ruser_mesh_data[0](2) = xmain3;
  pmy_mesh->ruser_mesh_data[1](0) = vmain1;
  pmy_mesh->ruser_mesh_data[1](1) = vmain2;
  pmy_mesh->ruser_mesh_data[1](2) = vmain3;
  pmy_mesh->ruser_mesh_data[2](0) = oamain1;
  pmy_mesh->ruser_mesh_data[2](1) = oamain2;
  pmy_mesh->ruser_mesh_data[2](2) = oamain3;
  pmy_mesh->ruser_mesh_data[3](0) = xsub1;
  pmy_mesh->ruser_mesh_data[3](1) = xsub2;
  pmy_mesh->ruser_mesh_data[3](2) = xsub3;
  pmy_mesh->ruser_mesh_data[4](0) = vsub1;
  pmy_mesh->ruser_mesh_data[4](1) = vsub2;
  pmy_mesh->ruser_mesh_data[4](2) = vsub3;
  pmy_mesh->ruser_mesh_data[5](0) = oasub1;
  pmy_mesh->ruser_mesh_data[5](1) = oasub2;
  pmy_mesh->ruser_mesh_data[5](2) = oasub3;
  pmy_mesh->ruser_mesh_data[6](0) = dt_old;
 
}

void ClusterAccel(
    MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar)
{

  AthenaArray<Real> &x1flux = pmb->phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux = pmb->phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux = pmb->phydro->flux[X3DIR];

  int il = pmb->is-(NGHOST-1), iu = pmb->ie+(NGHOST-1);
  int jl = pmb->js-(NGHOST-1), ju = pmb->je+(NGHOST-1);
  int kl = pmb->ks-(NGHOST-1), ku = pmb->ke+(NGHOST-1);

  if (res_flag == 1) { 
    xmain1 = pmb->pmy_mesh->ruser_mesh_data[0](0);
    xmain2 = pmb->pmy_mesh->ruser_mesh_data[0](1);
    xmain3 = pmb->pmy_mesh->ruser_mesh_data[0](2);
    vmain1 = pmb->pmy_mesh->ruser_mesh_data[1](0);
    vmain2 = pmb->pmy_mesh->ruser_mesh_data[1](1);
    vmain3 = pmb->pmy_mesh->ruser_mesh_data[1](2);
    oamain1 = pmb->pmy_mesh->ruser_mesh_data[2](0);
    oamain2 = pmb->pmy_mesh->ruser_mesh_data[2](1);
    oamain3 = pmb->pmy_mesh->ruser_mesh_data[2](2);
    xsub1 = pmb->pmy_mesh->ruser_mesh_data[3](0);
    xsub2 = pmb->pmy_mesh->ruser_mesh_data[3](1);
    xsub3 = pmb->pmy_mesh->ruser_mesh_data[3](2);
    vsub1 = pmb->pmy_mesh->ruser_mesh_data[4](0);
    vsub2 = pmb->pmy_mesh->ruser_mesh_data[4](1);
    vsub3 = pmb->pmy_mesh->ruser_mesh_data[4](2);
    oasub1 = pmb->pmy_mesh->ruser_mesh_data[5](0);
    oasub2 = pmb->pmy_mesh->ruser_mesh_data[5](1);
    oasub3 = pmb->pmy_mesh->ruser_mesh_data[5](2);
    dt_old = pmb->pmy_mesh->ruser_mesh_data[6](0);
    update_accel();
    res_flag = 0;
  }

  for (int k = kl; k <= ku; ++k){
  for (int j = jl; j <= ju; ++j){
  for (int i = il; i <= iu; ++i){

    Real phic = interp_grav_pot(pmb->pcoord->x1v(i),
                                pmb->pcoord->x2v(j),
                                pmb->pcoord->x3v(k));

    Real phil = interp_grav_pot(pmb->pcoord->x1f(i),
                                pmb->pcoord->x2v(j),
                                pmb->pcoord->x3v(k));
    Real phir = interp_grav_pot(pmb->pcoord->x1f(i+1),
                                pmb->pcoord->x2v(j),
                                pmb->pcoord->x3v(k));
        
    Real src = -(phir-phil)/pmb->pcoord->dx1v(i);
    if (main_cluster_fixed == 1 && num_halo == 2) {
      src -= noninertial_accel(1, pmb->pcoord->x1v(i),
			       pmb->pcoord->x2v(j),
			       pmb->pcoord->x3v(k));
    }
    cons(IM1,k,j,i) += src*prim(IDN,k,j,i)*dt;
    if (NON_BAROTROPIC_EOS) {
      src = -(x1flux(IDN,k,j,i)*(phic - phil) +
              x1flux(IDN,k,j,i+1)*(phir - phic))/pmb->pcoord->dx1v(i);
      if (main_cluster_fixed == 1 && num_halo == 2) {
        Real gl = -noninertial_accel(1, pmb->pcoord->x1f(i),
                  pmb->pcoord->x2v(j),
                  pmb->pcoord->x3v(k));
        Real gr = -noninertial_accel(1, pmb->pcoord->x1f(i+1),
                  pmb->pcoord->x2v(j),
                  pmb->pcoord->x3v(k));
        src += (x1flux(IDN,k,j,i)*gl +
          x1flux(IDN,k,j,i+1)*gr);
      }
      cons(IEN,k,j,i) += src*dt;
    }
    
    phil = interp_grav_pot(pmb->pcoord->x1v(i),
                           pmb->pcoord->x2f(j),
                           pmb->pcoord->x3v(k));
    phir = interp_grav_pot(pmb->pcoord->x1v(i),
                           pmb->pcoord->x2f(j+1),
                           pmb->pcoord->x3v(k));

    src = -(phir-phil)/pmb->pcoord->dx2v(j);
    if (main_cluster_fixed == 1 && num_halo == 2) {
      src -= noninertial_accel(2, pmb->pcoord->x1v(i),
			       pmb->pcoord->x2v(j),
			       pmb->pcoord->x3v(k));
    }
    cons(IM2,k,j,i) += src*prim(IDN,k,j,i)*dt;
    if (NON_BAROTROPIC_EOS) {
      src = -(x2flux(IDN,k,j,i)*(phic - phil) +
              x2flux(IDN,k,j+1,i)*(phir - phic))/pmb->pcoord->dx2v(j);
      if (main_cluster_fixed == 1 && num_halo == 2) {
        Real gl = -noninertial_accel(2, pmb->pcoord->x1v(i),
                  pmb->pcoord->x2f(j),
                  pmb->pcoord->x3v(k));
        Real gr = -noninertial_accel(2, pmb->pcoord->x1v(i),
                  pmb->pcoord->x2f(j+1),
                  pmb->pcoord->x3v(k));
        src += (x2flux(IDN,k,j,i)*gl +
                x2flux(IDN,k,j+1,i)*gr);
      }
      cons(IEN,k,j,i) += src*dt;
    }

    phil = interp_grav_pot(pmb->pcoord->x1v(i),
                           pmb->pcoord->x2v(j),
                           pmb->pcoord->x3f(k));
    phir = interp_grav_pot(pmb->pcoord->x1v(i),
                           pmb->pcoord->x2v(j),
                           pmb->pcoord->x3f(k+1));

    src = -(phir-phil)/pmb->pcoord->dx3v(k);
    if (main_cluster_fixed == 1 && num_halo == 2) {
      src -= noninertial_accel(3, pmb->pcoord->x1v(i),
			       pmb->pcoord->x2v(j),
			       pmb->pcoord->x3v(k));
    }
    cons(IM3,k,j,i) += src*prim(IDN,k,j,i)*dt;
    if (NON_BAROTROPIC_EOS) {
      src = -(x3flux(IDN,k,j,i)*(phic - phil) +
              x3flux(IDN,k+1,j,i)*(phir - phic))/pmb->pcoord->dx3v(k);
      if (main_cluster_fixed == 1 && num_halo == 2) {
        Real gl = -noninertial_accel(3, pmb->pcoord->x1v(i),
                  pmb->pcoord->x2v(j),
                  pmb->pcoord->x3f(k));
        Real gr = -noninertial_accel(3, pmb->pcoord->x1v(i),
                  pmb->pcoord->x2v(j),
                  pmb->pcoord->x3f(k+1));
        src += (x3flux(IDN,k,j,i)*gl +
		            x3flux(IDN,k+1,j,i)*gr);
      }
      cons(IEN,k,j,i) += src*dt;
    }

  }}}
 
}

namespace {

  void update_accel()
  {

    Real gmain, gsub;

    Real xc = xsub1-xmain1;
    Real yc = xsub2-xmain2;
    Real zc = xsub3-xmain3;
    Real rc = sqrt(SQR(xc)+SQR(yc)+SQR(zc));

    if (rc < rmax1) {
      gmain = -interpolate(grav1, num_points1, r1, rc);
    } else {
      gmain = -mass1/(rc*rc);
    }
    asub1 = gmain*xc/rc;
    asub2 = gmain*yc/rc;
    asub3 = gmain*zc/rc;

    if (rc < rmax2) {
      gsub = -interpolate(grav2, num_points2, r2, rc);
    } else {
      gsub = -mass2/(rc*rc);
    }
    amain1 = -gsub*xc/rc;
    amain2 = -gsub*yc/rc;
    amain3 = -gsub*zc/rc;

  }

  Real interp_grav_pot(const Real x1, const Real x2, const Real x3)
  {

    Real local_gpot;
    
    Real rr_main = sqrt(SQR(x1-xmain1)+SQR(x2-xmain2)+SQR(x3-xmain3));
    if (rr_main < rmax1) {
      local_gpot = -interpolate(gpot1, num_points1, r1, rr_main);
    } else {
      local_gpot = -mass1/rr_main;
    }
    
    if (num_halo == 2) {
      Real rr_sub = sqrt(SQR(x1-xsub1)+SQR(x2-xsub2)+SQR(x3-xsub3));
      if (rr_sub < rmax2) {
        local_gpot -= interpolate(gpot2, num_points2, r2, rr_sub);
      } else {
        local_gpot -= mass2/rr_sub;
      }
    }

    return local_gpot;

  }

  Real noninertial_accel(int axis, const Real x1, const Real x2, const Real x3)
  {

    Real accel, rr_main;

    if (axis == 1) {
      accel = amain1;
    } else if (axis == 2) {
      accel = amain2;
    } else if (axis == 3) {
      accel = amain3;
    }

    rr_main = sqrt(SQR(x1-xmain1)+SQR(x2-xmain2)+SQR(x3-xmain3));
    if (rr_main > r_cut) {
      accel *= std::exp(-(rr_main-r_cut)/r_scale);
    }

    return accel;

  }

  Real interpolate(Real a[], int n, Real x[], Real xx)
  {

    Real ret;

    Real r = (log10(xx)-log10(x[0]))*n/(log10(x[n-1])-log10(x[0]));
    int i = (int)r;
    if (a[i] > 0.0) {
      ret = a[i]*pow(a[i+1]/a[i], r-(Real)i);
    } else {
      ret = 0.0;
    }

    return ret;

  }

  void ReadFieldPoints(std::string filename, int *nbx, int *nby, int *nbz)
  {

    hid_t   file_id, dataset, dataspace;
    herr_t  status;
    hsize_t dims[3], maxdims[3];

    int rank;

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataset   = H5Dopen(file_id, "/vector_potential_x", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    rank      = H5Sget_simple_extent_dims(dataspace, dims, maxdims);

    *nbx = (int)dims[0];
    *nby = (int)dims[1];
    *nbz = (int)dims[2];

    H5Fclose(file_id);

    return;

  }

  void ReadNumPoints(std::string filename, int *num_points)
  {

    hid_t   file_id, dataset, dataspace;
    herr_t  status;
    hsize_t dims[1], maxdims[1];

    int rank;

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataset   = H5Dopen(file_id, "/fields/radius", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    rank      = H5Sget_simple_extent_dims(dataspace, dims, maxdims);

    *num_points = (int)dims[0];

    H5Fclose(file_id);

    return;

  }

  void ReadFieldCoords(std::string filename, Real xcoords[], Real ycoords[], 
                       Real zcoords[])
  {

    hid_t   file_id, dataset, dataspace, memspace;
    herr_t  status;

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::cout << "Read xcoords." << std::endl;

    dataset = H5Dopen(file_id, "/xcoord", H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                    H5S_ALL, H5P_DEFAULT, xcoords);
    H5Dclose(dataset);

    std::cout << "Read ycoords." << std::endl;

    dataset = H5Dopen(file_id, "/ycoord", H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                    H5S_ALL, H5P_DEFAULT, ycoords);
    H5Dclose(dataset);

    std::cout << "Read zcoords." << std::endl;

    dataset   = H5Dopen(file_id, "/zcoord", H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                    H5S_ALL, H5P_DEFAULT, zcoords);
    H5Dclose(dataset);

    H5Fclose(file_id);

    return;

  }

  void ReadField(std::string filename, const int ibegin, const int jbegin,
                       const int kbegin, const int iend, const int jend,
                       const int kend, Real Axf[], Real Ayf[], Real Azf[])
  {

    hid_t   file_id, dataset, dataspace, memspace, dxfer_template;
    herr_t  status;

    hsize_t start[3], stride[3], count[3], dims[3];

    int rank, ierr;

    rank = 3;

    start[0] = ibegin;
    start[1] = jbegin;
    start[2] = kbegin;

    stride[0] = 1;
    stride[1] = 1;
    stride[2] = 1;

    count[0] = iend-ibegin+1;
    count[1] = jend-jbegin+1;
    count[2] = kend-kbegin+1;

    dims[0] = count[0];
    dims[1] = count[1];
    dims[2] = count[2];

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    std::cout << "Read vector field." << std::endl;

    dataset = H5Dopen(file_id, "magnetic_vector_potential_x", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
    memspace = H5Screate_simple(rank, dims, NULL);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, Axf);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    dataset = H5Dopen(file_id, "magnetic_vector_potential_y", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
    memspace = H5Screate_simple(rank, dims, NULL);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, Ayf);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    dataset = H5Dopen(file_id, "magnetic_vector_potential_z", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    status = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start,
                                 stride, count, NULL);
    memspace = H5Screate_simple(rank, dims, NULL);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, Azf);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);

    H5Fclose(file_id);

    return;

  }

  void ReadProfiles(int which, std::string filename, Real r[],
        Real dens[], Real pres[], Real gpot[],
        Real grav[])
  {

    hid_t   file_id, dataset;
    herr_t  status;

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dataset   = H5Dopen(file_id, "/fields/radius", H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
        H5S_ALL, H5P_DEFAULT, r);
    H5Dclose(dataset);

    if (which == 1 || subhalo_gas == 1) {
      dataset   = H5Dopen(file_id, "/fields/density", H5P_DEFAULT);
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                      H5S_ALL, H5P_DEFAULT, dens);
      H5Dclose(dataset);

      dataset   = H5Dopen(file_id, "/fields/pressure", H5P_DEFAULT);
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                       H5S_ALL, H5P_DEFAULT, pres);
      H5Dclose(dataset);
    }

    dataset   = H5Dopen(file_id, "/fields/gravitational_potential", 
                        H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
        H5S_ALL, H5P_DEFAULT, gpot);
    H5Dclose(dataset);

    dataset   = H5Dopen(file_id, "/fields/gravitational_field",
                        H5P_DEFAULT);
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL,
                    H5S_ALL, H5P_DEFAULT, grav);
    H5Dclose(dataset);

    H5Fclose(file_id);

    return;

  }

  Real VecPot(Real *field, Real xx, Real yy, Real zz, 
              const int fdims[], const int fbegin[] )
  {

    // Indices into the coordinate vectors
    const int ii = (int)((xx-Axmin)/Adx);
    const int jj = (int)((yy-Aymin)/Ady);
    const int kk = (int)((zz-Azmin)/Adz);

    // Indices into the local vector potential patch
    const int ib = ii - fbegin[0];
    const int jb = jj - fbegin[1];
    const int kb = kk - fbegin[2];

    Real pot = 0.0;
  
    std::stringstream msg;

    if ( ib == 0 || ib == fdims[0]-1 ||
         jb == 0 || jb == fdims[1]-1 ||
         kb == 0 || kb == fdims[2]-1 ) {
      msg << "Input grid is smaller than the simulation domain !!" << std::endl;
      ATHENA_ERROR(msg);
    }

    for (int i = -1; i <= 1; i++) {
      Real dx = (xx-Axcoords[ii+i])/Adx;
      for (int j = -1; j <= 1; j++) {
        Real dy = (yy-Aycoords[jj+j])/Ady;
        for (int k = -1; k <= 1; k++) {
          Real dz = (zz-Azcoords[kk+k])/Adz;
          int idx = (ib+i)*fdims[2]*fdims[1] + (jb+j)*fdims[2] + (kb+k);
          pot += field[idx]*TSCWeight(dx)*TSCWeight(dy)*TSCWeight(dz);

        }
      }
    }

    return pot;

  }

  Real TSCWeight(Real x)
  {

    Real weight;

    Real xx = fabs(x);

    if (xx <= 0.5) {
      weight = 0.75 - xx*xx;
    } else if (xx >= 0.5 && xx <= 1.5) {
      weight = 0.5*SQR(1.5-xx);
    } else {
      weight = 0.0;
    }

    return weight;

  }

} // namespace

void DiodeInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,il-i) = prim(n,k,j,il);
      }
    }}
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma simd
    for (int i=1; i<=ngh; ++i) {
      prim(IVX,k,j,il-i) = std::min(prim(IVX,k,j,il-i), 0.0);
    }
  }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(il-i)) = b.x1f(k,j,il);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(il-i)) = b.x2f(k,j,il);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(il-i)) = b.x3f(k,j,il);
      }
    }}
  }

  return;
}

void DiodeOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,iu+i) = prim(n,k,j,iu);
      }
    }}
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma simd
    for (int i=1; i<=ngh; ++i) {
      prim(IVX,k,j,iu+i) = std::max(prim(IVX,k,j,iu+i), 0.0);
    }
  }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(iu+i+1)) = b.x1f(k,j,(iu+1));
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(iu+i)) = b.x2f(k,j,iu);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(iu+i)) = b.x3f(k,j,iu);
      }
    }}
  }

  return;
}

void DiodeInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        prim(n,k,jl-j,i) = prim(n,k,jl,i);
      }
    }}
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=1; j<=ngh; ++j) {
#pragma simd
    for (int i=il; i<=iu; ++i) {
      prim(IVY,k,jl-j,i) = std::min(prim(IVY,k,jl-j,i), 0.0);
    }
  }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(jl-j),i) = b.x1f(k,jl,i);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(jl-j),i) = b.x2f(k,jl,i);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(jl-j),i) = b.x3f(k,jl,i);
      }
    }}
  }

  return;
}

void DiodeOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        prim(n,k,ju+j,i) = prim(n,k,ju,i);
      }
    }}
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=1; j<=ngh; ++j) {
#pragma simd
    for (int i=il; i<=iu; ++i) {
      prim(IVY,k,ju+j,i) = std::max(prim(IVY,k,ju+j,i), 0.0);
    }
  }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(ju+j),i) = b.x1f(k,(ju  ),i);
      }
    }}

    for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(ju+j+1),i) = b.x2f(k,(ju+1),i);
      }
    }}

    for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(ju+j  ),i) = b.x3f(k,(ju  ),i);
      }
    }}
  }

  return;
}

void DiodeInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        prim(n,kl-k,j,i) = prim(n,kl,j,i);
      }
    }}
  }
  
  for (int k=1; k<=ngh; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma simd
    for (int i=il; i<=iu; ++i) {
      prim(IVZ,kl-k,j,i) = std::min(prim(IVZ,kl-k,j,i), 0.0);
    }
  }}
 
  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((kl-k),j,i) = b.x1f(kl,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x2f((kl-k),j,i) = b.x2f(kl,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x3f((kl-k),j,i) = b.x3f(kl,j,i);
      }
    }}
  }

  return;
}

void DiodeOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        prim(n,ku+k,j,i) = prim(n,ku,j,i);
      }
    }}
  }

  for (int k=1; k<=ngh; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma simd
    for (int i=il; i<=iu; ++i) {
      prim(IVZ,ku+k,j,i) = std::max(prim(IVZ,ku+k,j,i), 0.0);
    }
  }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((ku+k  ),j,i) = b.x1f((ku  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x2f((ku+k  ),j,i) = b.x2f((ku  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma simd
      for (int i=il; i<=iu; ++i) {
        b.x3f((ku+k+1),j,i) = b.x3f((ku+1),j,i);
      }
    }}
  }

  return;
}

int RefinementCondition(MeshBlock *pmb)
{
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxcurv = 0.0;
  Real max_dens = 0.0;
  int level = pmb->loc.level;
  int refine = 0;
  Real xdist1_sq, ydist1_sq, zdist1_sq;
  Real xdist2_sq, ydist2_sq, zdist2_sq;

  // First, try second-derivative refinement on                                                                      
  // pressure and density. Only do it if the                                                                        
  // density is high enough.                                                                                         

  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        max_dens = std::max(w(IDN,k,j,i), max_dens);
      }
    }
  }
  Real dcurv = ComputeCurvature(pmb, IDN);
  Real pcurv = ComputeCurvature(pmb, IEN);
  maxcurv = std::max(dcurv, pcurv);

  if (max_dens > min_refine_density) {
    if (maxcurv > 0.6) {
      // refine : curvature > 0.6                                                                                  
      refine = 1;
    } else if (maxcurv < 0.3) {
      // derefinement: curvature < 0.3                                                                              
      refine = -1;
    }
  } else {
    refine = -1;
  }

  // Next, check to see whether we are within either of the refining radii
  // of the cluster(s)
  
  Real bxl1 = pmb->block_size.x1min-xmain1;
  Real bxl2 = pmb->block_size.x1min-xsub1;
  Real bxr1 = pmb->block_size.x1max-xmain1;
  Real bxr2 = pmb->block_size.x1max-xsub1;
  Real byl1 = pmb->block_size.x2min-xmain2;
  Real byl2 = pmb->block_size.x2min-xsub2;
  Real byr1 = pmb->block_size.x2max-xmain2;
  Real byr2 = pmb->block_size.x2max-xsub2;
  Real bzl1 = pmb->block_size.x3min-xmain3;
  Real bzl2 = pmb->block_size.x3min-xsub3;
  Real bzr1 = pmb->block_size.x3max-xmain3;
  Real bzr2 = pmb->block_size.x3max-xsub3;
             
  if (bxl1*bxr1 > 0.) {
    xdist1_sq = std::min(SQR(bxl1), SQR(bxr1));
  } else {
    xdist1_sq = 0.;
  }
  if (byl1*byr1 > 0.) {
    ydist1_sq = std::min(SQR(byl1), SQR(byr1));
  } else {
    ydist1_sq = 0.;
  }
  if (bzl1*bzr1 > 0.) {
    zdist1_sq = std::min(SQR(bzl1), SQR(bzr1));
  } else {
    zdist1_sq = 0.;
  }

  if (bxl2*bxr2 > 0.) {
    xdist2_sq = std::min(SQR(bxl2), SQR(bxr2));
  } else {
    xdist2_sq = 0.;
  }
  if (byl2*byr2 > 0.) {
    ydist2_sq = std::min(SQR(byl2), SQR(byr2));
  } else {
    ydist2_sq = 0.;
  }
  if (bzl2*bzr2 > 0.) {
    zdist2_sq = std::min(SQR(bzl2), SQR(bzr2));
  } else {
    zdist2_sq = 0.;
  }

  Real dist1_sq = xdist1_sq+ydist1_sq+zdist1_sq;
  Real dist2_sq = xdist2_sq+ydist2_sq+zdist2_sq;

  if (dist1_sq < ref_radius1_sq || dist2_sq < ref_radius2_sq) {     
    // If we're inside a sphere and need to refine, do it.   
    refine = 1;
  } else if (refine < 1) {
    // Mark for de-refine, but only if we don't want second-derivative refinement
    refine = -1;
  }

  return refine;

}

Real ComputeCurvature(MeshBlock *pmb, int ivar)
{
  Real eps = 1.0e-2;
  Real curv = 0.0;
  
  AthenaArray<Real> du, au, du2, du3, du4;

  int nx_tot = (pmb->ie-pmb->is)+1 + 2*(NGHOST);
  int ny_tot = (pmb->je-pmb->js)+1 + 2*(NGHOST);
  int nz_tot = (pmb->ke-pmb->ks)+1 + 2*(NGHOST);

  Real delx1 = 0.5/pmb->pcoord->dx1v(0);
  Real delx2 = 0.5/pmb->pcoord->dx2v(0);
  Real delx3 = 0.5/pmb->pcoord->dx3v(0);

  du.NewAthenaArray(3,nz_tot,ny_tot,nx_tot);
  au.NewAthenaArray(3,nz_tot,ny_tot,nx_tot);
  du2.NewAthenaArray(9);
  du3.NewAthenaArray(9);
  du4.NewAthenaArray(9);

  AthenaArray<Real> &w = pmb->phydro->w;

  for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
    for (int j=pmb->js-1; j<=pmb->je+1; j++) {
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        du(0,k,j,i) = (w(ivar,k,j,i+1) - w(ivar,k,j,i-1))*delx1;
	      au(0,k,j,i) = (std::abs(w(ivar,k,j,i+1)) + std::abs(w(ivar,k,j,i-1)))*delx1;
        du(1,k,j,i) = (w(ivar,k,j+1,i) - w(ivar,k,j-1,i))*delx2;
        au(1,k,j,i) = (std::abs(w(ivar,k,j+1,i)) + std::abs(w(ivar,k,j-1,i)))*delx2;
        du(2,k,j,i) = (w(ivar,k+1,j,i) - w(ivar,k-1,j,i))*delx3;
        au(2,k,j,i) = (std::abs(w(ivar,k+1,j,i)) + std::abs(w(ivar,k-1,j,i)))*delx3;
      }
    }
  }

  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {

        du2(0) = (du(0,k,j,i+1) - du(0,k,j,i-1))*delx1;
        du3(0) = (std::abs(du(0,k,j,i+1)) + std::abs(du(0,k,j,i-1)))*delx1;
	      du4(0) = (au(0,k,j,i+1) + au(0,k,j,i-1))*delx1;

        du2(1) = (du(0,k,j+1,i) - du(0,k,j-1,i))*delx2;
        du3(1) = (std::abs(du(0,k,j+1,i)) + std::abs(du(0,k,j-1,i)))*delx2;
	      du4(1) = (au(0,k,j+1,i) + au(0,k,j-1,i))*delx2;

        du2(2) = (du(1,k,j,i+1) - du(1,k,j,i-1))*delx1;
        du3(2) = (std::abs(du(1,k,j,i+1)) + std::abs(du(1,k,j,i-1)))*delx1;
	      du4(2) = (au(1,k,j,i+1) + au(1,k,j,i-1))*delx1;

        du2(3) = (du(1,k,j+1,i) - du(1,k,j-1,i))*delx2;
        du3(3) = (std::abs(du(1,k,j+1,i)) + std::abs(du(1,k,j-1,i)))*delx2;
	      du4(3) = (au(1,k,j+1,i) + au(1,k,j-1,i))*delx2;

        du2(4) = (du(0,k+1,j,i) - du(0,k-1,j,i))*delx3;
        du3(4) = (std::abs(du(0,k+1,j,i)) + std::abs(du(0,k-1,j,i)))*delx3;
	      du4(4) = (au(0,k+1,j,i) + au(0,k-1,j,i))*delx3;

        du2(5) = (du(1,k+1,j,i) - du(1,k-1,j,i))*delx3;
        du3(5) = (std::abs(du(1,k+1,j,i)) + std::abs(du(1,k-1,j,i)))*delx3;
	      du4(5) = (au(1,k+1,j,i) + au(1,k-1,j,i))*delx3;

        du2(6) = (du(2,k,j,i+1) - du(2,k,j,i-1))*delx1;
        du3(6) = (std::abs(du(2,k,j,i+1)) + std::abs(du(2,k,j,i-1)))*delx1;
	      du4(6) = (au(2,k,j,i+1) + au(2,k,j,i-1))*delx1;

        du2(7) = (du(2,k,j+1,i) - du(2,k,j-1,i))*delx2;
        du3(7) = (std::abs(du(2,k,j+1,i)) + std::abs(du(2,k,j-1,i)))*delx2;
        du4(7) = (au(2,k,j+1,i) + au(2,k,j-1,i))*delx2;

        du2(8) = (du(2,k+1,j,i) - du(2,k-1,j,i))*delx3;
        du3(8) = (std::abs(du(2,k+1,j,i)) + std::abs(du(2,k-1,j,i)))*delx3;
	      du4(8) = (au(2,k+1,j,i) + au(2,k-1,j,i))*delx3;

        Real num = 0.0;
        Real denom = 0.0;
        
        for (int kk = 0; kk < 9; kk++) {
          num += SQR(du2(kk));
          denom += SQR(du3(kk)+eps*du4(kk));
        }
        
        if (denom == 0.0 && num != 0.0) {
          curv = 1.0e99;
        } else if (denom != 0.0) {
          curv = std::max(curv, num/denom);
        }

      }
    }
  }
  
  du.DeleteAthenaArray();
  au.DeleteAthenaArray();
  du2.DeleteAthenaArray();
  du3.DeleteAthenaArray();
  du4.DeleteAthenaArray();

  return std::sqrt(curv);
}
