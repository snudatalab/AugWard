cd src

python run.py --exp augplain --data PTC_MR --aug NodeDrop --aug-size 0.1 --batch-size 32 --dropout 0 --gpu 0 --lr 0.01 --workers 1 --trials 5 --epochs 100 --verbose 10
python run.py --exp augward --data PTC_MR --aug NodeDrop --aug-size 0.1 --batch-size 32 --dropout 0 --gpu 0  --lr 0.01 --workers 1 --trials 5 --epochs 100 --verbose 10 --alpha 0.5 --weighted 50 --weighted-cr 10