#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=24:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/vsch/continual/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --qos unkillable

#> first experiment to be able to scale minimal losses for other schedules:
#> --task_schedule=parallel

# 1. Create your environement locally
module load python/3.7.4
module load httpproxy

export continual_dataset="h2z_d"


cd /home/vsch/continual-translation
# zip -r $SCRATCH/ct-env.zip ct-env > /dev/null #! uncomment to load new packages
source /home/vsch/continual-translation/ctenv/bin/activate

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /scratch/vsch/continual/$continual_dataset.zip $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/$continual_dataset.zip -d $SLURM_TMPDIR > /dev/null

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python train.py \
    --sbatch_file=$0 \
    --git_hash="899e741f7e1f40fec4d66f7c0ef3c96419a18ab8" \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --name "parallel_continual_0" \
    --model continual \
    --checkpoints_dir "/scratch/vsch/continual/checkpoints" \
    --display_freq 5000 \
    --batch_size 5 \
    --netG "continual" \
    --task_schedule "parallel" \
    --message "same as parallel_h2z_0, smaller learning rate for rotation" \
    --lr_rot 0.0001 \
    --lambda_A 10.0 \
    --lambda_B 10.0 \
    --lambda_I 0.5 \
    --lambda_R 10.0 \
    --lambda_D 10.0


# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/checkpoints $SCRATCH
