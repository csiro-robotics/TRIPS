declare TRIAL_SEED_VALUE=2
declare NUM_OF_EXEMPLAR=0
declare NUM_OF_STEPS=5001
declare BATCH_SIZE=32
declare RESNET_34=true
declare RESNET_18=false
declare BACKBONE_name="ResNet34"

declare EXP_NAME='MAS_w_'$NUM_OF_EXEMPLAR'_exemplar_ResNet34_default_lr_optimizer_0_iid'

declare CODE_PTH='/DGCIL_TRIPS'
declare DATA_PTH='/DATA'
cd $CODE_PTH

# session 0 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_0 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 0 \
      --num_old_cls 0 \
      --num_new_cls 15 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000

# session 1 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_1 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 1 \
      --load_old_info ${EXP_NAME}_session_0 \
      --model_type iid \
      --num_old_cls 15 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000
      
# session 2 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_2 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 2 \
      --load_old_info ${EXP_NAME}_session_1 \
      --model_type iid \
      --num_old_cls 25 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000

# session 3 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_3 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 3 \
      --load_old_info ${EXP_NAME}_session_2 \
      --model_type iid \
      --num_old_cls 35 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000

# session 4 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_4 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 4 \
      --load_old_info ${EXP_NAME}_session_3 \
      --model_type iid \
      --num_old_cls 45 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000

# session 5 --------------------------------------------------------------

python train_all.py ${EXP_NAME}_session_5 \
      --data_dir $DATA_PTH \
      --dataset OfficeHome_INC \
      --algorithm MAS \
      --deterministic \
      --trial_seed $TRIAL_SEED_VALUE \
      --checkpoint_freq 100 \
      --steps $NUM_OF_STEPS \
      --model_save 50 \
      --batch_size $BATCH_SIZE \
      --current_session 5 \
      --load_old_info ${EXP_NAME}_session_4 \
      --model_type iid \
      --num_old_cls 55 \
      --num_new_cls 10 \
      --num_of_exemplar $NUM_OF_EXEMPLAR \
      --store_mas_importance \
      --resnet18 $RESNET_18 \
      --resnet34 $RESNET_34 \
      --off_swad \
      --hparam_mas_scale 1000





