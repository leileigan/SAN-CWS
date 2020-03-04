CUDA_VISIBLE_DEVICES=1 python main.py \
                       --char_embedding /mnt/data/ganleilei/NeuralSegmentation/gigaword_chn.all.a2b.uni.ite50.vec \
                       --bichar_embedding /mnt/data/ganleilei/NeuralSegmentation/gigaword_chn.all.a2b.bi.ite50.vec \
                       --status train \
                       --train ../SubWordCWS/data/ctb6.0/origin/train.ctb60.char.bmes \
                       --dev ../SubWordCWS/data/ctb6.0/origin/dev.ctb60.char.bmes \
                       --test ../SubWordCWS/data/ctb6.0/origin/test.ctb60.char.bmes \
                       --use_san false \
                       --use_attention true \
                       --savemodel /mnt/data/ganleilei/models/cws_models/save.model \
                       --use_bert true \
                       --cross_domain true \
                       --pos_to_idx ../SubWordCWS/data/pos_count \
                       --token_replace_prob ../SubWordCWS/data/pd.prob \
                       --use_tencent_dic true \
                       --dropout 0.4
