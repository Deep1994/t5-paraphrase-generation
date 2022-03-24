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

For BLEU calculation, use:
```
python bleu.py
```
