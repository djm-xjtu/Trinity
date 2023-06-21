rm model.h5
rm model.json
/usr/local/bin/python3.8 generate.py --width 128 --height 64 --length 6 --symbols symbols.txt --count 128000 --output-dir train
/usr/local/bin/python3.8 generate.py --width 128 --height 64 --length 6 --symbols symbols.txt --count 25600 --output-dir test
/usr/local/bin/python3.8 train.py --width 128 --height 64 --length 6 --symbols symbols.txt --batch-size 32 --epochs 10 --output-model deng --train-dataset train --validate test
/usr/local/bin/python3.8 classify.py  --model-name model --captcha-dir ../img --output stuff.csv --symbols symbols.txt
