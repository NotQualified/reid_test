CUDA_VISIBLE_DEVICES=0,1 python new_soft.py -d market_gen -a resnet50 -b 256 -j 0 --epochs 120 --logs-dir logs/market1501 --features 512 --data-dir /home/hawkeyenew1/hht/data
