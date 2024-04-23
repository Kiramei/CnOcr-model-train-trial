cnocr train -m densenet_lite_136-gru --index-dir index --train-config-fp train.json

cnocr evaluate --model-name last.ckpt  -i index/dev.tsv \ --image-folder data --batch-size 128 