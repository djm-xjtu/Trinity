rm model.h5
rm model.json
/usr/local/bin/python3.8 train.py --width 128 --height 64 --length 4 --symbols symbols.txt --batch-size 4 --epochs 2 --output-model model --train-dataset train --validate test
/usr/local/bin/python3.8 classify.py  --model-name model --captcha-dir ../img --output stuff.txt --symbols symbols.txt