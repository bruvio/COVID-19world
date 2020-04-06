#!/Users/bruvio/python37/bin python
# -*- coding: utf-8 -*-

import stat
import os

from subprocess import call

cwd = os.getcwd()
run_scrap_data = cwd+'/'+"Scrap_data.py"
run_build_table = cwd+'/'+"build_country_table.py"
run_world_fits = cwd+'/'+"world_data.py"
run_doubling_time = cwd+'/'+"country-situation-report.py"
run_plots = cwd+'/'+"plots.py"

st1 = os.stat(run_scrap_data)
st2 = os.stat(run_build_table)
st3 = os.stat(run_world_fits)
st4 = os.stat(run_doubling_time)
st5 = os.stat(run_plots)
os.chmod(run_scrap_data, st1.st_mode | stat.S_IEXEC)
os.chmod(run_build_table, st2.st_mode | stat.S_IEXEC)
os.chmod(run_world_fits, st3.st_mode | stat.S_IEXEC)
os.chmod(run_doubling_time, st4.st_mode | stat.S_IEXEC)
os.chmod(run_plots, st5.st_mode | stat.S_IEXEC)

ret = call('python {}'.format(run_scrap_data).split())
if ret == 0:
    call('python {}'.format(run_build_table).split())
    call('python {}'.format(run_world_fits).split())
    call('python {}'.format(run_doubling_time).split())
    call('python {}'.format(run_plots).split())
else:
         print ("Command failed with return code ", ret)

# os.system(cwd+'/'+run_scrap_data)
# os.system(cwd+'/'+run_build_table)
# os.system(cwd+'/'+run_world_fits)
# os.system(cwd+'/'+run_doubling_time)