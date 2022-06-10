config_file=configs/Object365-Detection/faster_rcnn_R_50_FPN_1x.yaml
config_file=configs/Object365-Detection/surv30_faster_rcnn_R50_FPN_1x.yml
save='surv30-exp1-r50-fpn'
config_file=configs/Object365-Detection/surv30_uni_faster_rcnn_R50_FPN_1x.yml
config_file=configs/Object365-Detection/surv30_uni_faster_rcnn_R50_FPN_short.yml
save='surv30-uni-r50-fpn'
save='surv30-uni-r50-fpn-with-test'

python -u  ./tools/train_net.py \
    --config-file $config_file \
    --num-gpus 8 \
    OUTPUT_DIR output/$save/ \
    2>&1 | tee -a logs/$save.log