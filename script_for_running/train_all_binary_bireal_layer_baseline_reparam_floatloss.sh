#!/bin/bash -e
# debug pkill python
#data=PACS
#data_path=/media/disk3
#root_path=/media/disk2
root_path=/shuju
data_path=/shuju
#CUDA_VISIBLE_DEVICES=1 python train.py --dataset=${data} --data_path=${root_path}/yjm/data/PACS \
#      --net resnet18_binary --target=photo  --n_classes=7 \
#      --epochs=100  --batch_size=64 --suffix=debug --cal_var --var_loss_weights=0.01 --var_type=last
# all
#root_path=/userhome/
dataset=( office office_home VLCS PACS DomainNet)
domains_all=('amazon dslr webcam' \
'Art Clipart Product Real_World' \
'CALTECH LABELME PASCAL SUN' \
'art_painting cartoon photo sketch' \
'clipart infograph painting real quickdraw sketch')
n_classes=( 31 65 5 7 345 )
opt_types=( Adam )
varw=( 0 )
lr_list=( 0.001 )
var_type=( '-' )
weight_decay=( 1e-6 )
fpws=( 0.1 0.01 0.001 0.0001 )
resls=( 0.1 )
#
(
for ((s=1234;s<1237;s++)) # 3
do
for ((vtype=0;vtype<1;vtype++))
do
for ((j=3;j<4;j++))#${#dataset[@]}
do
IFS=' ' read -r -a domains <<< "${domains_all[j]}"
for ((lr_id=0;lr_id<1;lr_id++)) #2
do
for ((opt_idx=0;opt_idx<1;opt_idx++)) # 3
do
for ((resl_idx=0;resl_idx<1;resl_idx++))
do
for ((wd_id=0;wd_id<1;wd_id++)) #2
do
for ((k=0;k<3;k++))#varw  ${#varw[*]}
do
for ((fpw_idx=0;fpw_idx<1;fpw_idx++)) #2
do
for ((i=0;i<${#domains[*]};i++))#${#domains[*]}
do
    lr=${lr_list[lr_id]}
    varw_rate=${varw[k]}
    opt_type=${opt_types[opt_idx]}
    target=${domains[i]}
    data=${dataset[j]}
    echo "run on $target"
    nettype=${var_type[vtype]}

    OUT=''
    for I in ${domains_all[j]}
    do
        if [ "$I" = "$target" ]
        then
        continue
        fi
        if [ "$OUT" = '' ]
        then
        OUT=$I
        else
        OUT=${OUT:+$OUT}-$I
        fi
    done

    wd=${weight_decay[wd_id]}
    a=$(($((2*${fpw_idx}))+resl_idx))
    gid=${i}
    resl=${resls[$resl_idx]}
    fpw=${fpws[$fpw_idx]}
    CUDA_VISIBLE_DEVICES=${gid} python train_mixup_reparam.py --dataset=${data} --data_path=${data_path}/yjm/data/${data} \
    --net resnet18_binary_reparam --target=$target  --n_classes=${n_classes[j]} --log_path=${root_path}/yjm/DG \
    --epochs=100  --batch_size=64 --suffix=Vcvpr_pretrained_reparam_init${s} --seed=$s \
    --pretrained='./models/model_best_resnet_oldv.pth.tar' \
    --var_type=${nettype} \
    --loss_res_weight=${resl} --loss_fp_weight=${fpw} --two_step \
    --cal_var_loss --var_loss_weights=${varw_rate} --optimizer_type=${opt_type} --weight_decay=${wd} \
    --learning_rate=${lr} | tee -a ./Vcvpr_abalation_onf_resnet18_binary_reparam_init_fpw${fpw}_resl${resl}_${opt_type}_w_${wd}_seed${s}_${data}_target_${target}_varw_rate${varw[k]}_lr${lr}_type${nettype}.txt&&
    echo "done $target" &
done
wait
done
done
done
done
done
done
done
done
done
)