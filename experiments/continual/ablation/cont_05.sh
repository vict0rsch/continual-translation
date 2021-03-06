#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /scratch/vsch/continual/slurm-%j.out


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
    cp $SCRATCH/continual/$continual_dataset.zip $SLURM_TMPDIR
fi

cd $HOME/continual-translation
unzip $SLURM_TMPDIR/$continual_dataset.zip -d $SLURM_TMPDIR > /dev/null


python train.py \
    --num_threads 10 \
    --dataroot $SLURM_TMPDIR/$continual_dataset \
    --model continual \
    --checkpoints_dir $SCRATCH/continual/checkpoints \
    --display_freq 2000 \
    --batch_size 5 \
    --netG "continual" \
    --git_hash="b14b2b12eb6d0f60b5bb32fdf479a5dac88d8709" \
    --name "cont_05" \
    --task_schedule "continual" \
    --message "better lr-annealing smaller lr no-rot_D + radam + cont_05.sh" \
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
    --lr 0.0005 \
    --n_epochs_decay 100 \
    --n_epochs 200 \
    --encoder_merge_ratio 0.5
