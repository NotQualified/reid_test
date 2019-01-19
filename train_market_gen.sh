CUDA_VISIBLE_DEVICES=2,3 python new_soft.py -d market_gen -a resnet50 -b 256 -j 0 --epochs 120 --logs-dir logs/market1501_gen --features 256 --data-dir /home/hawkeyenew1/hht/data
