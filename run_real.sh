source activate clone

for i in c17 c22 c26 c29 c30 c39
do
    python train_real.py --algo leiden --real_ds $i --rl_epoch 0
done