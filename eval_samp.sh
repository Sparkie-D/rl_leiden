source activate clone


for method in leiden
do
    for data_type in CNV expr
    do
        python eval_sample.py   --task_type CNV_multiSample_martix \
                                --method ${method} \
                                --data_type ${data_type} \
                                --data_names data1,data2,data5,data6,data7 \
                                --meta_columns celltype,Stage,sample,anatomical_location
        # python eval_sample.py   --task_type SingleSample_CNV \
        #                         --method ${method} \
        #                         --data_type ${data_type} \
        #                         --data_names GSM5276940,GSM5276943,SOL003,SOL006,SOL008,SOL012,SOL016,SOL1303,SOL1306,SOL1307 \
        #                         --meta_columns celltype
    done
done