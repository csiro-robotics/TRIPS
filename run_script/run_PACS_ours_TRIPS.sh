declare TRIAL_SEED_VALUE=2
declare NUM_OF_EXEMPLAR=0
declare NUM_OF_STEPS=5001
declare BATCH_SIZE=24
declare RESNET_34=true
declare RESNET_18=false
declare BACKBONE_name="ResNet34"

declare hparam_LOSS_lambda_c=1
declare hparam_LOSS_lambda_d=30
declare hparam_LOSS_lambda_t=1

declare hparam_PROTO_index=0  # sigma=0.5, mean_MovingAvg_eta=0.1, mean_Balance_beta=1.0
declare hparam_PROTO_sigma=(0.5)
declare PROTO_mean_MovingAvg_eta=(0.1)
declare hparam_PROTO_mean_Balance_beta=(1.0) 
declare hparam_PROTO_mean_name=("050110")

declare hparam_PROTO_cov_index=0  # cov_Shrinkage_alpha=0.05, cov_MovingAvg_eta=0.1, cov_Balance_beta=1.0
declare PROTO_cov_Shrinkage_alpha=(0.05)
declare PROTO_cov_MovingAvg_eta=(0.10)
declare PROTO_cov_Balance_beta=1.0
declare hparam_PROTO_cov_name=("005010")

declare EXP_NAME='TRIPLET_DIST_W_PROTO_w_'$NUM_OF_EXEMPLAR'_exemplar_'$BACKBONE_name'_'$NUM_OF_STEPS'_steps_'$BATCH_SIZE'_batch_size_lr_5e-5_optimizer_adam_iid_'$hparam_LOSS_lambda_c'_'$hparam_LOSS_lambda_d'_'$hparam_LOSS_lambda_t'_loss_DATA_AUG_Proto_shifting_COV_AUG_'${hparam_PROTO_cov_name[$hparam_PROTO_cov_index]}'10'

declare CODE_PTH='/DGCIL_TRIPS'
declare DATA_PTH='/DATA'
cd $CODE_PTH

echo "Start Job Running."

###############################
#
# our TRIPS method
#
################################

# session 0 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_0 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 0 \
      --model_type iid \
      --num_old_cls 0 \
      --num_new_cls 3 \
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
      --Data_Augmentation \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 1 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_1 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 1 \
      --load_old_info ${EXP_NAME}_session_0 \
      --model_type iid \
      --num_old_cls 3 \
      --num_new_cls 2 \
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
      --Data_Augmentation \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta

# session 2 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_2 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm TRIPLET_DIST_W_PROTO \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 2 \
      --load_old_info ${EXP_NAME}_session_1 \
      --model_type iid \
      --num_old_cls 5 \
      --num_new_cls 2 \
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
      --Data_Augmentation \
      --PROTO_augmentation \
      --PROTO_augmentation_w_COV \
      --hparam_PROTO_cov_Shrinkage_alpha ${PROTO_cov_Shrinkage_alpha[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_MovingAvg_eta ${PROTO_cov_MovingAvg_eta[$hparam_PROTO_cov_index]} \
      --hparam_PROTO_cov_Balance_beta $PROTO_cov_Balance_beta



