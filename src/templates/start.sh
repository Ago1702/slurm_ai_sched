#!/bin/bash
export CLUS_DIR=$(pwd)
export MACHINE_NAME="slurmsimcont"
export RUN_NAME="test1"
export dtstart={{dtstart}}
export replica=1

slurmsim -v run_sim  -d \
            -e ${CLUS_DIR}/etc \
            -a ${CLUS_DIR}/etc/sacctmgr.script \
            -w ${CLUS_DIR}/workload/first_job.events \
            -r ${CLUS_DIR}/results \
#            -rt 300 \
            -dtstart $dtstart