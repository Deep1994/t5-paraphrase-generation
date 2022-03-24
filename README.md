# t5-paraphrase-generation
​T5 Model for generating paraphrases of english sentences. Trained on the Quora Paraphrase dataset.

## How to use
First, you should install [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) via:
```
pip install simpletransformers
```

Then, you can fine-tune the T5 model via:
```
CUDA_VISIBLE_DEVICES=0 python train.py --train
```

After training finished, you can evaluate the trained model via:
```
CUDA_VISIBLE_DEVICES=0 python train.py --eval
```
Note that, if you don't pass the decoding argument, the model default use top k/ top p sampling, for beam search decoding, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --eval --beam_search
```

All the arguments about train and eval are in 'train.py'.


For BLEU calculation, use:
```
python bleu.py
```

## Result
Model: t5-small
beam sarch: BLEU = 29.34 63.8/38.7/26.1/18.1
top k/ top p sampling: BLEU = 21.48 56.6/29.8/18.0/11.4

Although top k/ top p sampling has lower bleu, the diversity of generated paraphrase is higher. On the contrary, beam search has higher bleu but lower diversity.
