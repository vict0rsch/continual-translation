#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o ~/scratch/continual/slurm-%j.out


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
    --model cycle_gan \
    --checkpoints_dir $SCRATCH/continual/checkpoints \
    --display_freq 2000 \
    --batch_size 5 \
    --netG "resnet_9blocks"
    --init_type "kaiming" \
    --name "radam_cyclegan_h2z" \
    --message "radam_cyclegan_h2z.sh" \
    --lr 0.0005 \
    --n_epochs_decay 100 \
    --n_epochs 200
