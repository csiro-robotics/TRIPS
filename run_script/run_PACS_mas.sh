declare TRIAL_SEED_VALUE=2
declare NUM_OF_EXEMPLAR=0
declare NUM_OF_STEPS=5001
declare RESNET_34=true
declare RESNET_18=false
declare BACKBONE_name="ResNet34"

declare EXP_NAME='MAS_w_'$NUM_OF_EXEMPLAR'_exemplar_ResNet34_'$NUM_OF_STEPS'_steps_default_lr_optimizer_iid'

declare CODE_PTH='/DGCIL_TRIPS'
declare DATA_PTH='/DATA'
cd $CODE_PTH

# session 0 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_0_offswad \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --current_session 0 \
      --num_old_cls 0 \
      --num_new_cls 3 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000

# session 1 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_1_offswad \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --current_session 1 \
      --load_old_info ${EXP_NAME}_session_0_offswad \
      --model_type iid \
      --num_old_cls 3 \
      --num_new_cls 2 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000
      
# session 2 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_2_offswad \
      --data_dir $DATA_PTH \
      --dataset PACS_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --current_session 2 \
      --load_old_info ${EXP_NAME}_session_1_offswad \
      --model_type iid \
      --num_old_cls 5 \
      --num_new_cls 2 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000



