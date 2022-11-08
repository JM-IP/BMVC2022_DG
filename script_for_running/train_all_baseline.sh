#!/bin/bash -e
#data_path=/media/disk3
#root_path=/media/disk2
root_path=/shuju
data_path=/shuju

dataset=( office office_home VLCS PACS DomainNet)
domains_all=('amazon dslr webcam' \
'Art Clipart Product Real_World' \
'CALTECH LABELME PASCAL SUN' \
'art_painting cartoon photo sketch' \
'clipart infograph painting real quickdraw sketch')
n_classes=( 31 65 5 7 345 )

# baseline on birealnet
varw=( 0 )
lr_list=( 0.001 )
var_type=( '-' )
weight_decay=( 1e-6 )
(

for ((s=1234;s<1237;s++)) # 3
do
for ((k=0;k<1;k++))#varw  ${#varw[*]}
do
for ((j=0;j<4;j++))#${#dataset[@]}
do
IFS=' ' read -r -a domains <<< "${domains_all[j]}"
for ((type=0;type<1;type++))
do
for ((lr_id=0;lr_id<1;lr_id++)) #2
do
for ((wd_id=0;wd_id<1;wd_id++)) #2
do
for ((i=0;i<${#domains[*]};i++))#${#domains[*]}
do
    lr=${lr_list[lr_id]}
    varw_rate=${varw[k]}
    wd=${weight_decay[wd_id]}
    target=${domains[i]}
    data=${dataset[j]}
    echo "run on $target"
    gid=${i}
    nettype=${var_type[type]}
    (CUDA_VISIBLE_DEVICES=${gid} python train.py --dataset=${data} --data_path=${data_path}/yjm/data/${data} \
    --net resnet18_binary --target=$target  --n_classes=${n_classes[j]} --log_path=${root_path}/yjm/DG \
    --epochs=100  --batch_size=64 --suffix=Vcvpr_pretrained${s} --seed=$s --pretrained \
    --var_type=${nettype} \
    --cal_var_loss --var_loss_weights=${varw_rate} --optimizer_type=Adam --weight_decay=${wd} \
    --learning_rate=${lr} | tee -a ./Vcvpr_baseline_w_${wd}_seed${s}_${data}_target_${target}_varw_rate${varw[k]}_lr${lr}_type${nettype}.txt&&
    echo "done $target")&
done
wait
done
done
done
done
done
done
)

