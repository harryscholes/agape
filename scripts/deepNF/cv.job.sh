#!/bin/bash -l
#
# Submission script for deepNF cross-validation jobs.
#
# This script sets up the environment and executes deepNF cross-validation SGE
# jobs on the CS cluster.
#
# Usage:
#   ./cv.job.sh <architecture> <ontology> <level>


#########
# Setup #
#########

date=$(date +%Y-%m-%d)
current_time='date +%H:%M:%S'

if [ $SGE_TASK_ID = 'undefined' ]; then
    sge_task_label=''
else
    sge_task_label=_$SGE_TASK_ID
fi

# Do work in scratch
output_dir=/scratch0/$USER/${JOB_NAME}_${JOB_ID}${sge_task_label}
mkdir -p $output_dir
cd $output_dir


###############
# Job options #
###############

code=$HOME/git/agape/scripts/deepNF/cv.py
conda_env=
results_path=$HOME/results


###############
# SGE options #
###############

# Set executing shell
#$ -S /bin/bash

# Job name
# #$ -N deepNF_cv_n_jobs4
# Working directory
#$ -wd /home/hscholes/results
# Notifications
#$ -M ucbthsc@ucl.ac.uk
#$ -m baes

# Wall time h:m:s
#$ -l h_rt=5:00:0
# Resource reservation
#$ -R y

# OpenMP threads
#$ -pe smp 4
# Memory
#$ -l h_vmem=7.7G,tmem=7.7G
# Scratch space
#$ -l tscratch=10G


export OMP_NUM_THREADS=4

echo
echo ---------------
echo - JOB_DETAILS -
echo ---------------
echo DATE $date
echo USER $USER
echo HOSTNAME $HOSTNAME
echo JOB_NAME $JOB_NAME
echo JOB_ID $JOB_ID
echo SGE_TASK_ID $SGE_TASK_ID
echo PWD $PWD
echo


###############
# Job details #
###############

echo STARTED $current_time

# Setup environment
if [[ $conda_env != '' ]]; then
    echo Setting conda env: $conda_env
    source activate $conda_env
else
    echo Using default conda env
fi

python=$(which python)
echo Python interpreter: $python

export JOBLIB_START_METHOD='forkserver'
echo JOBLIB_START_METHOD $JOBLIB_START_METHOD

# Run code
echo CODE_STARTED $(eval $current_time)

$python -u $code \
    -a $1 \
    -g $2 \
    -l $3 \
    -m $AGAPEDATA/deepNF/models/180510_fda4344 \
    -r results/180510_fda4344 \
    -n 5 \
    -j 4 \
    -s 1

echo CODE_DONE $current_time

# Save output
tar zcvf $results_path/${date}_${JOB_NAME}_${JOB_ID}${sge_task_label}.tar.gz $output_dir

echo DONE $(eval $current_time)
