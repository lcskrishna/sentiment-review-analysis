#! /bin/bash
# exec 1>PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS Options ========
#PBS -N "AmazonReview_KNN_Job"
#PBS -q mamba
#PBS -l walltime=4:00:00
#PBS -l nodes=2:ppn=2
#PBS -V
# ==== Main ======
cd $PBS_O_WORKDIR

mkdir log

{
 module load python/3.5.1

 python3 /users/clolla/machine_learning/Algorithms/SL_KNN/AmazonReviewAnalysis/SL_KNN_AmazonReview.py
} > log/out_knn_amazon2_"$PBS_JOBNAME"_$PBS_JOBID 2>log/err_knn_amazon2_"$PBS_JOBNAME"_$PBS_JOBID

