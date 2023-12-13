declare TRIAL_SEED_VALUE=2
declare NUM_OF_EXEMPLAR=0
declare NUM_OF_STEPS=5001
declare BATCH_SIZE=32
declare RESNET_34=true
declare RESNET_18=false
declare BACKBONE_name="ResNet34"

declare hparam_LOSS_lambda_c=1
declare hparam_LOSS_lambda_d=30
declare hparam_LOSS_lambda_t=1

declare hparam_PROTO_index=5  # sigma=0.5, mean_MovingAvg_eta=0.1, mean_Balance_beta=1.0
declare hparam_PROTO_sigma=(0.3 0.3 0.3 0.5 0.5 0.5 0.8 0.8 0.8 1.0 1.0 1.0 1.5 1.5 1.5)
declare PROTO_mean_MovingAvg_eta=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1)
declare hparam_PROTO_mean_Balance_beta=(0.1 0.5 1.0 0.1 0.5 1.0 0.1 0.5 1.0 0.1 0.5 1.0 0.1 0.5 1.0) 
declare hparam_PROTO_name=("030101" "030501" "031001" "050101" "050501" "051001" "080101" "080501" "081001" "100101" "100501" "101001" "150101" "150501" "151001")

declare hparam_PROTO_cov_index=0  # cov_Shrinkage_alpha=0.05, cov_MovingAvg_eta=0.1, cov_Balance_beta=1.0
declare PROTO_cov_Shrinkage_alpha=(0.05 0.05 0.05 0.05 0.10 0.10 0.10 0.10 0.20 0.20 0.20 0.20 0.40 0.40 0.40 0.40 0.60 0.60 0.60 0.60)
declare PROTO_cov_MovingAvg_eta=(0.10 0.20 0.30 0.50 0.10 0.20 0.30 0.50 0.10 0.20 0.30 0.50 0.10 0.20 0.30 0.50 0.10 0.20 0.30 0.50)
declare PROTO_cov_Balance_beta=1.0
declare hparam_PROTO_cov_name=("005010" "005020" "005030" "005050" "010010" "010020" "010030" "010050" "020010" "020020" "020030" "020050" "040010" "040020" "040030" "040050" "060010" "060020" "060030" "060050")

declare EXP_NAME='TRIPLET_DIST_W_PROTO_w_'$NUM_OF_EXEMPLAR'_exemplar_'$BACKBONE_name'_'$NUM_OF_STEPS'_steps_'$BATCH_SIZE'_batch_size_lr_5e-5_optimizer_adam_iid_'$hparam_LOSS_lambda_c'_'$hparam_LOSS_lambda_d'_'$hparam_LOSS_lambda_t'_loss_Proto_shifting_COV_AUG_'${hparam_PROTO_cov_name[$hparam_PROTO_cov_index]}'10'
declare CODE_PTH='/DGCIL_TRIPS'
declare DATA_PTH='/DATA'
cd $CODE_PTH

echo "Start Job Running."

###############################
#
# Use both calculated covariance & stored covariance, no data augmentation (our FINAL method)
#
################################

# session 0 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_0 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 0 \
      --model_type iid \
      --num_old_cls 0 \
      --num_new_cls 15 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 1 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_1 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 1 \
      --model_type iid \
      --load_old_info ${EXP_NAME}_session_0 \
      --num_old_cls 15 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 2 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_2 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 2 \
      --model_type iid \
      --load_old_info ${EXP_NAME}_session_1 \
      --num_old_cls 25 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 3 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_3 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 3 \
      --model_type iid \
      --load_old_info ${EXP_NAME}_session_2 \
      --num_old_cls 35 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 4 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_4 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 4 \
      --model_type iid \
      --load_old_info ${EXP_NAME}_session_3 \
      --num_old_cls 45 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 5 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_5 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --batch_size $BATCH_SIZE \
      --model_save 50 \
      --current_session 5 \
      --model_type iid \
      --load_old_info ${EXP_NAME}_session_4 \
      --num_old_cls 55 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --TRIPLET_w_cross_entropy \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --margin 0 \
      --triplet_dist_type euclidean_dist \
      --TRIPLET_feature_offset 1 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --hparam_LOSS_lambda_t $hparam_LOSS_lambda_t \
      --PROTO_semantic_shifting \
      --hparam_PROTO_sigma ${hparam_PROTO_sigma[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_Balance_beta ${hparam_PROTO_mean_Balance_beta[$hparam_PROTO_index]} \
      --hparam_PROTO_mean_MovingAvg_eta ${PROTO_mean_MovingAvg_eta[$hparam_PROTO_index]} \
      --PROTO_using_delta \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta



