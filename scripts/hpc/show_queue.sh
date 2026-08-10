#!/bin/bash
# Watch this project's jobs in the SLURM queue.
#
# Modelled on the AIP reference repo's show_AIP_squeue.sh, which greps for its
# own "AIP_" job-name prefix. Our sbatch sets --job-name=hybrid-eval-train, so
# this filters on "hybrid-eval" instead.
#
#   bash scripts/hpc/show_queue.sh
#
# Columns: jobid, partition, job name, user, state, elapsed, nodes, reason/node.
# State codes worth knowing: PD = pending (see the last column for why),
# R = running, CG = completing.
watch "squeue -o '%.18i %.9P %.20j %.12u %.2t %.10M %.6D %R' | grep -E 'hybrid-eval|JOBID'"
