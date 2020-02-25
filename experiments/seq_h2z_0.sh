#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=24:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/vsch/continual/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --qos high

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
    --git_hash="27f5972157a1f69d9feade9c8d3505b2930e758d" \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --name "seq_continual_1" \
    --model continual \
    --checkpoints_dir "/scratch/vsch/continual/checkpoints" \
    --display_freq 5000 \
    --batch_size 5 \
    --netG "continual" \
    --task_schedule "sequential" \
    --message "seq h2z exp with lambda_R and _D are set to 10" \
    --lambda_A 10 \
    --lambda_B 10 \
    --lambda_I 0.5 \
    --lambda_R 10 \
    --lambda_D 10 \
    --d_loss_threshold 0.15 \
    --r_acc_threshold 0.6 \
    --lr_rot 0.001


# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/checkpoints $SCRATCH
