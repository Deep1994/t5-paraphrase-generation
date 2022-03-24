"""
calculate the BLEU on the predictions
beam sarch: BLEU = 29.34 63.8/38.7/26.1/18.1
top k/ top p sampling: BLEU = 21.48 56.6/29.8/18.0/11.4
"""
from sacrebleu.metrics import BLEU
import argparse, os

def main(args):
    srcs = []
    refs = []
    preds = []
    with open(os.path.join('predictions', args.predictions_file), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            srcs.append(line[0])
            refs.append(line[1])
            # pick the first one as the final output.
            preds.append(line[2])
    print(len(srcs), len(refs), len(preds))

    bleu = BLEU()
    bleu_score = bleu.corpus_score(preds, [refs])
    print(bleu_score)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', type=str, default='preds_for_bleu.txt')
    args = parser.parse_args()
    
    main(args)