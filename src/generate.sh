profile=consep
export H_PROFILE=hv_${profile}

python create_config.py \
    --profile $H_PROFILE \
    --id 1.0 \
    --input_prefix /data/input/ \
    --output_prefix /data/output/ \
    --data_dir data_hv_${profile}/ \
    --train_dir train/Annotations \
    --valid_dir test/Annotations \
    --preproc \
    --norm_brightness \
    --mode train \
    --image train_1 \
    --extract \
    --export \
    --extract_type mirror \
    --step_size 80 \
    --win_size 540 \
    --input_norm \
    --input_augs p_linear \
    --data_modes train,test \
    --inf_auto_find_chkpt \
    --inf_auto_metric 'valid_dice_Inflammatory' \
    --inf_data_list data_hv_${profile}/test/Images/ \
    --remap_labels \
    --outline '' \
    --skip_types 'Misc,Spindle'

# --inf_auto_find_chkpt \
# --inf_model dm_hv_class_${profile}.npz \