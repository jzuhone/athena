# Regression test based on Newtonian MHD linear wave convergence problem
#
# Runs a linear wave convergence test in 3D including SMR and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# linearwave_errors.dat)

# Modules
import scripts.utils.athena as athena
import sys
import os
from shutil import move
sys.path.insert(0, '../../vis/python')
import athena_read                             # noqa
athena_read.check_nan_flag = True
_fluxes = ['hlld', 'roe']
_exec = os.path.join('bin', 'athena')


# Prepare Athena++
def prepare(**kwargs):
    for flux in _fluxes:
        athena.configure('b',
                         prob='linear_wave',
                         coord='cartesian',
                         flux=flux, **kwargs)
        athena.make()
        move(_exec, _exec + '_' + flux)
        os.system('mv obj obj_' + flux)


# Run Athena++
def run(**kwargs):
    for flux in _fluxes:
        move(_exec + '_' + flux, _exec)
        os.system('mv obj_' + flux + ' obj')
        # L-going fast/Alfven/slow waves
        for w in (0, 1, 2):
            tlim = max(0.5, w)
            for i in (32, 64):
                arguments = ['time/ncycle_out=100',
                             'problem/wave_flag=' + repr(w),
                             'problem/vflow=0.0',
                             'mesh/refinement=static',
                             'mesh/nx1=' + repr(i),
                             'mesh/nx2=' + repr(i/2),
                             'mesh/nx3=' + repr(i/2),
                             'meshblock/nx1=' + repr(i/4),
                             'meshblock/nx2=' + repr(i/4),
                             'meshblock/nx3=' + repr(i/4),
                             'output2/dt=-1',
                             'time/tlim=' + repr(tlim),
                             'problem/compute_error=true']
                athena.run('mhd/athinput.linear_wave3d', arguments)
        # entropy wave
        for i in (32, 64):
            arguments = ['time/ncycle_out=100',
                         'problem/wave_flag=3',
                         'problem/vflow=1.0',
                         'mesh/refinement=static',
                         'mesh/nx1=' + repr(i),
                         'mesh/nx2=' + repr(i/2),
                         'mesh/nx3=' + repr(i/2),
                         'meshblock/nx1=' + repr(i/4),
                         'meshblock/nx2=' + repr(i/4),
                         'meshblock/nx3=' + repr(i/4),
                         'output2/dt=-1',
                         'time/tlim=1.0',
                         'problem/compute_error=true']
            athena.run('mhd/athinput.linear_wave3d', arguments)
        # L/R-going fast wave
        w = 0
        arguments = ['time/ncycle_out=100',
                     'problem/wave_flag=' + repr(w),
                     'output2/dt=-1', 'time/tlim=0.5', 'problem/compute_error=true']
        athena.run('mhd/athinput.linear_wave3d', arguments)
        w = 6
        arguments = ['time/ncycle_out=100',
                     'problem/wave_flag=' + repr(w),
                     'output2/dt=-1', 'time/tlim=0.5', 'problem/compute_error=true']
        athena.run('mhd/athinput.linear_wave3d', arguments, lcov_test_suffix=flux)
        # clear object directory
        os.system('rm -rf obj')
    return 'skip_lcov'


# Analyze outputs
def analyze():
    # read data from error file
    filename = 'bin/linearwave-errors.dat'
    all_data = athena_read.error_dat(filename)
    all_data = all_data.reshape(len(_fluxes), -1, all_data.shape[-1])
    analyze_status = True

    for i, flux in enumerate(_fluxes):
        data = all_data[i]
        flux_str = 'With flux "{0:}": '.format(flux)
        # check largest maximum error scaled to RMS is within bounds at each highest res
        if data[1][13] > 8.0:
            print(flux_str + "maximum relative error in L-going fast wave too large",
                  data[1][13])
            analyze_status = False
        # check error in M1 for Alfven wave since density constant
        if data[3][15]/data[3][6] > 8.0:
            print(flux_str + "maximum relative error in L-going Alfven wave too large",
                  data[3][15]/data[3][6])
        if data[5][13] > 8.0:
            print(flux_str + "maximum relative error in L-going slow wave too large",
                  data[5][13])
            analyze_status = False

        # check RMS error and convergence of all four waves
        if data[1][4] > 4.5e-8:
            print(flux_str + "RMS error in L-going fast wave too large", data[1][4])
            analyze_status = False
        if data[1][4]/data[0][4] > 0.4:
            print(flux_str + "not converging for L-going fast wave",
                  data[0][4], data[1][4])
            analyze_status = False

        if data[3][4] > 4.0e-8:
            print(flux_str + "RMS error in L-going Alfven wave too large",
                  data[3][4])
            analyze_status = False
        if data[3][4]/data[2][4] > 0.4:
            print(flux_str + "not converging for L-going Alfven wave",
                  data[2][4], data[3][4])
            analyze_status = False

        if data[5][4] > 5.0e-8:
            print(flux_str + "RMS error in L-going slow wave too large", data[5][4])
            analyze_status = False
        if data[5][4]/data[4][4] > 0.4:
            print(flux_str + "not converging for L-going slow wave",
                  data[4][4], data[5][4])
            analyze_status = False

        if data[7][4] > 2.75e-8:
            print(flux_str + "RMS error in entropy wave too large", data[7][4])
            analyze_status = False
        if data[7][4]/data[6][4] > 0.4:
            print(flux_str + "not converging for entropy wave", data[6][4], data[7][4])
            analyze_status = False

        # check error identical for waves in each direction
        if data[8][4] != data[9][4]:
            print(flux_str + "error in L/R-going fast waves not equal",
                  data[8][4], data[9][4])
            analyze_status = False

    return analyze_status
