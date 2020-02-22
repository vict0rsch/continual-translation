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
#> lambda_A|B to 5

# 1. Create your environement locally
module load python/3.7.4
module load httpproxy

cd /home/vsch/continual-translation
# zip -r $SCRATCH/ct-env.zip ct-env > /dev/null #! uncomment to load new packages
cp /$SCRATCH/ct-env.zip $SLURM_TMPDIR
unzip $SLURM_TMPDIR/ct-env.zip -d $SLURM_TMPDIR > /dev/null
source $SLURM_TMPDIR/ct-env/bin/activate

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /scratch/vsch/continual/s2w_d.zip $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/s2w_d.zip -d $SLURM_TMPDIR > /dev/null

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python train.py \
    --git_hash="409661853ff978bf2f02e534ca0fb86a5f8c2a33" \
    --dataroot $SLURM_TMPDIR/s2w_d \
    --name "parallel_continual_1" \
    --model continual \
    --checkpoints_dir "/scratch/vsch/continual/checkpoints" \
    --display_freq 1000 \
    --batch_size 5 \
    --netG "continual" \
    --task_schedule "parallel" \
    --lambda_A 5.0 \
    --lambda_B 5.0 \
    --lambda_I 0.5 \
    --lambda_R 1.0 \
    --lambda_D 1.0

#> lambda_I_A = lambda_A * lambda_I


# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/checkpoints $SCRATCH
