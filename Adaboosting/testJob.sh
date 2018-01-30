#! /bin/bash
# exec 1>PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS Options ========
#PBS -N "AmazonReview_Adaboosting_Job"
#PBS -q mamba
#PBS -l walltime=7:50:00
#PBS -l nodes=3:ppn=5
#PBS -V
# ==== Main ======
cd $PBS_O_WORKDIR

mkdir log

{
 module load python/3.5.1

 python3 /users/clolla/machine_learning/Algorithms/SL_Adaboosting/AmazonReviewAnalysis/SL_Adaboosting_AmazonReview.py
} > log/out_adaboosting3_"$PBS_JOBNAME"_$PBS_JOBID 2>log/err_adaboosting3_"$PBS_JOBNAME"_$PBS_JOBID

