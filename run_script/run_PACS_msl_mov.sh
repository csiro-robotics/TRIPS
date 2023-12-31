declare TRIAL_SEED_VALUE=2
declare NUM_OF_EXEMPLAR=0
declare NUM_OF_STEPS=5001
declare BATCH_SIZE=24
declare hparam_LOSS_lambda_c=1
declare hparam_LOSS_lambda_d=30
declare RESNET_34=true
declare RESNET_18=false
declare BACKBONE_name="ResNet34"

declare EXP_NAME='MSL_MOV_w_'$NUM_OF_EXEMPLAR'_exemplar_ResNet34_'$NUM_OF_STEPS'_steps_'$BATCH_SIZE'_batch_size_default_lr_optimizer_iid_'$hparam_LOSS_lambda_c'_'$hparam_LOSS_lambda_d'_loss_DATA_AUG'

declare CODE_PTH='/DGCIL_TRIPS'
declare DATA_PTH='/DATA'
cd $CODE_PTH

# session 0 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_0 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MSL_MOV_DIST \
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
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --off_swad \
      --Data_Augmentation

# session 1 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_1 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MSL_MOV_DIST \
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
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --off_swad \
      --Data_Augmentation

# session 2 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_2 \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MSL_MOV_DIST \
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
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --hparam_LOSS_mannual_setting \
      --hparam_LOSS_lambda_c $hparam_LOSS_lambda_c \
      --hparam_LOSS_lambda_d $hparam_LOSS_lambda_d \
      --off_swad \
      --Data_Augmentation