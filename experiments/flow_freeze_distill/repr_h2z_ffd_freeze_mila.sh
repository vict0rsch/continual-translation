#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH -o /network/tmp1/schmidtv/continual/slurm-%j.out  # Write the log in $SCRATCH
#SBATCH --partition unkillable

export continual_dataset="h2z_d"

if [[ $HOME == *"schmidtv"* ]];
then
    echo "Welcome to Mila"
    module load anaconda/3
    source $CONDA_ACTIVATE
    conda activate base
    conda deactivate
    conda activate clouds
    cp /network/tmp1/schmidtv/continual/$continual_dataset.zip $SLURM_TMPDIR
else
    echo "Welcome to Beluga"
    module load python/3.7.4
    module load httpproxy
    source $HOME/continual-translation/ctenv/bin/activate
    cp /scratch/vsch/continual/$continual_dataset.zip $SLURM_TMPDIR
fi

cd $HOME/continual-translation

unzip $SLURM_TMPDIR/$continual_dataset.zip -d $SLURM_TMPDIR > /dev/null


python train.py \
    --num_threads 8 \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --model continual \
    --checkpoints_dir "/network/tmp1/schmidtv/continual/checkpoints" \
    --display_freq 2000 \
    --batch_size 3 \
    --netG "continual" \
    --git_hash="9bda085070c0a16f3191117bcc4a0ffbb1dd67ce" \
    --name "repr_h2z_ffd_freeze_mila" \
    --task_schedule "representational" \
    --message "repr_h2z_ffd_freeze_mila.sh" \
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
    --D_rotation \
    --repr_mode freeze
