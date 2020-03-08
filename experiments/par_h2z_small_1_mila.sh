#!/bin/bash
#SBATCH --cpus-per-task=6             # Ask for 6 CPUs
#SBATCH --gres=gpu:v100:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH -o /network/tmp1/schmidtv/continual/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --partition unkillable

#> first experiment to be able to scale minimal losses for other schedules:
#> --task_schedule=parallel

# 1. Create your environement locally
module load anaconda/3

source $CONDA_ACTIVATE

conda activate base
conda deactivate
conda activate clouds

export continual_dataset="h2z_d"


cd /network/home/schmidtv/continual-translation
# zip -r $SCRATCH/ct-env.zip ct-env > /dev/null #! uncomment to load new packages

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /network/tmp1/schmidtv/continual/$continual_dataset.zip $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/$continual_dataset.zip -d $SLURM_TMPDIR > /dev/null

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python train.py \
    --num_threads 6 \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --model continual \
    --checkpoints_dir "/network/tmp1/schmidtv/continual/checkpoints" \
    --display_freq 5000 \
    --batch_size 3 \
    --netG "continual" \
    --git_hash="200517bdb5775dd616c7a6aa674395caf7eac6ac" \
    --name "par_continual_small_0" \
    --task_schedule "parallel" \
    --message "par h2z exp with gray & SMALL | par_continual_small_0.sh" \
    --lambda_DA 2 \
    --lambda_DB 2 \
    --lambda_CA 10 \
    --lambda_CB 10 \
    --lambda_I 0.5 \
    --lambda_R 1 \
    --lambda_D 1 \
    --lambda_G 1 \
    --lambda_J 1 \
    --depth_loss_threshold 0.1 \
    --gray_loss_threshold 0.3 \
    --rotation_acc_threshold 0.9 \
    --jigsaw_acc_threshold 0.7 \
    --lr_rotation 0.001 \
    --lr_depth 0.001 \
    --lr_gray 0.0005 \
    --n_epochs_decay 100 \
    --n_epochs 200 \
    --small_data 250


# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/checkpoints $SCRATCH
