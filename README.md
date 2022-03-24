# t5-paraphrase-generation
​T5 Model for generating paraphrases of english sentences. Trained on the Quora Paraphrase dataset. If you want to use our pretrained model, please see this [instruction](https://huggingface.co/Deep1994/t5-paraphrase-quora) for detail.

## Online demo website
Click [t5-paraphrase](https://huggingface.co/spaces/Deep1994/t5-paraphrase) to have a try online.

<img width="1245" alt="image" src="https://user-images.githubusercontent.com/24366782/159983658-f213f1df-08f8-4378-bf5f-482966f99625.png">

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
python bleu.py --predictions_file your_predictions_file.txt
```

## Result
```
Model: t5-small
beam sarch: BLEU = 29.34 63.8/38.7/26.1/18.1
top k/ top p sampling: BLEU = 21.48 56.6/29.8/18.0/11.4
```

Although top k/ top p sampling has lower bleu, the diversity of generated paraphrase is higher. On the contrary, beam search has higher bleu but lower diversity.

```
Original Question ::
What is the best comedy TV serial/series?

Beam search: 
0: What is the best comedy TV series?
1: What are some of the best comedy TV series?
2: Which is the best comedy TV series?
3: What are the best comedy TV series?
4: What are some of the best comedy TV shows?

Top k/ Top p sampling:
0: What are some of the best comedy TV dramas?
1: What are the best comedy TV series or series?
2: What are the best comedy television serials?
3: What is the best comedy series?
4: Which are some best comedy TV series series?
```
