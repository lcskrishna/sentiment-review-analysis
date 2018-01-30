#! /bin/bash
# exec 1>PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS Options ========
#PBS -N "AmazonReview_NN_Job"
#PBS -q mamba
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -V
# ==== Main ======
cd $PBS_O_WORKDIR

mkdir log

{
 module load python/3.5.1

 python3 /users/clolla/machine_learning/Algorithms/SL_NeuralNetwork/AmazonReviewAnalysis/SL_NeuralNetwork_AmazonReview.py
} > log/out_neuralnetwork_"$PBS_JOBNAME"_$PBS_JOBID 2>log/err_neuralnetwork_"$PBS_JOBNAME"_$PBS_JOBID

