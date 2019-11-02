CUDA_VISIBLE_DEVICES=0 python main.py --status train \
		--train data/ctb6.0/origin/train.ctb60.char.bmes \
		--dev data/ctb6.0/origin/dev.ctb60.char.bmes \
		--test data/ctb6.0/origin/test.ctb60.char.bmes \
		--savemodel ./model \