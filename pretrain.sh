gpu=0
dis_type='DisSegGraph'
loss_mode='gan'
seg_len=90
loss_type=2
if [ $loss_type == 1 ]; then
    loss_arr=(1.0 0.1 0.0)
elif [ $loss_type == 2 ]; then
    loss_arr=(1.0 0.1 0.1)
else
    loss_arr=(1.0 0.0 0.0)
fi
mus_ebd_dim=72
dis_name='time_cond_cnn'
kernel_size=(1 3)
stride=(1 2)
cond_axis=1
CUDA_VISIBLE_DEVICES=$gpu \
    python3 train_gan.py --learning_rate 1e-4 \
            --dis_learning_rate 2e-5 \
            --mse_rate 1 \
            --dis_rate 0.01 \
            --loss_mode $loss_mode \
            --is_load_model False \
            --is_reg False \
            --reg_scale 5e-5 \
            --rnn_keep_list 1.0 1.0 1.0\
            --dis_type $dis_type \
            --dis_name $dis_name \
            --loss_rate_list ${loss_arr[0]} ${loss_arr[1]} ${loss_arr[2]}\
            --kernel_size ${kernel_size[0]} ${kernel_size[1]} \
            --stride ${stride[0]} ${stride[1]}\
            --act_type lrelu \
            --optimizer Adam \
            --cond_axis $cond_axis \
            --seg_list $seg_len \
            --seq_shift 1 \
            --gen_hop $seg_len \
            --fold_list 0 \
            --type_list all-f4 \
            --model_path '' \
            --max_max_epoch 20 \
            --save_data_epoch 5 \
            --save_model_epoch 5 \
            --is_save_train False \
            --mot_scale 100. \
            --norm_way zscore \
            --teacher_forcing_ratio 0. \
            --tf_decay 1. \
            --batch_size 128 \
            --mus_ebd_dim $mus_ebd_dim \
            --has_random_seed False \
            --is_all_norm False \
            --add_info ./output/pretrain