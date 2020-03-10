#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=8                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=24:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/vsch/continual/slurm-%j.out  # Write the log in $SCRATCH

#> first experiment to be able to scale minimal losses for other schedules:
#> --task_schedule=parallel

# 1. Create your environement locally
module load python/3.7.4
module load httpproxy

export continual_dataset="h2z_d"


cd $HOME/continual-translation
# zip -r $SCRATCH/ct-env.zip ct-env > /dev/null #! uncomment to load new packages
source $HOME/continual-translation/ctenv/bin/activate

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp /scratch/vsch/continual/$continual_dataset.zip $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/$continual_dataset.zip -d $SLURM_TMPDIR > /dev/null

python train.py \
    --num_threads 8 \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --model continual \
    --checkpoints_dir "/scratch/vsch/continual/checkpoints" \
    --display_freq 2000 \
    --batch_size 3 \
    --netG "continual" \
    --git_hash="e1f12153226be09dbe10ad613cd28f673f789c66" \
    --name "rept_h2z_2" \
    --task_schedule "reprensentational-traduction" \
    --message "rept h2z rot D | rept_h2z_2.sh" \
    --lambda_CA 10 \
    --lambda_DA 1 \
    --lambda_CB 10 \
    --lambda_DB 1 \
    --lambda_I 0.5 \
    --lambda_R 1 \
    --lambda_D 1 \
    --lambda_G 1 \
    --lambda_J 1 \
    --depth_loss_threshold 0.15 \
    --gray_loss_threshold 0.3 \
    --rotation_acc_threshold 0.85 \
    --jigsaw_acc_threshold 0.85 \
    --lr_rotation 0.001 \
    --lr_depth 0.001 \
    --lr_gray 0.0005 \
    --lr_jigsaw 0.001 \
    --n_epochs_decay 100 \
    --n_epochs 200 \
    --D_rotation
