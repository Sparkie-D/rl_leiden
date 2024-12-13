source activate clone

for i in 17 22 26 29 30 39
do
    python eval_lineage.py --dataset_id ${i}
done