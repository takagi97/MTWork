#! /usr/bin/bash
set -e

device=0,1,2,3,4,5,6,7
data=wiki

if [ $data == "wiki" ]; then
        arch=transformer_lm_wiki103
        lr=0.0001
        warmup=16000
        max_tokens=2048
        tokens_per_sample=${max_tokens}
        update_freq=4
        dropout=0.1
        weight_decay=0.01
        keep_last_epochs=2
        criterion=adaptive_loss
        max_epoch=
        max_update=286000
        data_dir=wikitext-103
        fp16=1
elif [ $data == "ptb" ]; then
        arch=transformer_lm_small
        lr=0.0007
        warmup=2000
        max_tokens=4096
        tokens_per_sample=${max_tokens}
        update_freq=1
        criterion=label_smoothed_cross_entropy
        dropout=0.1
        weight_decay=0.01
        keep_last_epochs=2
        max_epoch=20
        max_update=
        data_dir=penn
        adam_betas="'(0.9, 0.997)'"
        fp16=0
else
        echo "unknown task=$data"
        exit
fi
#tag=$arch"_"${max_tokens}"_"${dropout}"_"${lr}
save_dir="output/lm/${data}"

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train_lm.sh
gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

if [ $data == "wiki" ]; then
    cmd="python3 -u train.py ../data-bin/${data_dir}
      --task language_modeling
      --max-lr 1.0
      --t-mult 2
      --lr-period-updates 270000
      --lr-scheduler cosine
      --lr-shrink 0.75
      --warmup-init-lr 1e-07
      --min-lr 1e-09
      --optimizer nag
      --clip-norm 0.1
      --seed 1
      --sample-break-mode none
      --skip-invalid-size-inputs-valid-test
      --ddp-backend=no_c10d
      --criterion ${criterion}
      --weight-decay $weight_decay
      --distributed-world-size $gpu_num
      --keep-last-epochs $keep_last_epochs
      --tensorboard-logdir $save_dir
      --save-dir ${save_dir}
      --arch ${arch}
      --warmup-updates ${warmup}
      --max-tokens ${max_tokens}
      --update-freq ${update_freq}
      --tokens-per-sample ${tokens_per_sample}
      --lr ${lr}"
elif [ $data == "ptb" ]; then
     cmd="python3 -u train.py ../data-bin/$data_dir
      --optimizer adam
      --clip-norm 0.0
      --lr-scheduler inverse_sqrt
      --warmup-init-lr 1e-07
      --min-lr 1e-09
      --label-smoothing 0.1
      --no-progress-bar
      --log-interval 100
      --ddp-backend no_c10d
      --task language_modeling
      --distributed-world-size $gpu_num
      --criterion ${criterion}
      --arch $arch
      --weight-decay $weight_decay
      --warmup-updates $warmup
      --lr $lr
      --max-tokens $max_tokens
      --dropout $dropout
      --update-freq $update_freq
      --save-dir $save_dir
      --keep-last-epochs $keep_last_epochs
      --tensorboard-logdir $save_dir"
      cmd=${cmd}" --adam-betas "${adam_betas}
else
        echo "unknown task=$data"
        exit
fi

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
