"""
calculate the BLEU on the predictions
"""
from sacrebleu.metrics import BLEU
import argparse

def main(args):
    srcs = []
    refs = []
    preds = []
    with open(args.predictions_dir, 'r') as f:
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
    parser.add_argument('--predictions_dir', type=str, default='predictions/preds_for_bleu.txt')
    args = parser.parse_args()
    
    main(args)