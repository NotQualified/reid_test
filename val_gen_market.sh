CUDA_VISIBLE_DEVICES=1,2,3 python only_evaluate.py -d market_gen -a resnet50 -b 256 -j 0 --epochs 60 --logs-dir logs/market1501_gen  --data-dir /home/hawkeyenew1/hht/data --evaluate
