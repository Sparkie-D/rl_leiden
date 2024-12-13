source activate clone

task=SingleSample_CNV
method=leiden

datadir=../data/${task}

type=CNV
                                                                                            # task
# for i in c17 c22 c26 c29 c30 c39                                                            # lineage_trace_data
# for i in data1 data2 data3 data5 data6 data7                                              # CNV_multiSample_martix
for i in GSM5276940 # GSM5276943 SOL003 SOL006 SOL008 SOL012 SOL016 SOL1303 SOL1306 SOL1307 # SingleSample_CNV
do
    # for col in celltype Stage sample anatomical_location # CNV_multiSample_martix
    for col in celltype # SingleSample_CNV
    do
        python eval_sample.py \
            --task_type ${task} \
            --data_name ${i} \
            --data_type ${type} \
            --method ${method} \
            --meta_column ${col}
    done
done