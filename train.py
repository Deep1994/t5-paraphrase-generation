"""
References:
    1. https://towardsdatascience.com/training-t5-for-paraphrase-generation-ab3b5be151a2
    2. https://simpletransformers.ai/docs/installation/
    3. https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/config/model_args.py
    4. https://simpletransformers.ai/docs/usage/
    5. https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c
"""

import argparse
from ast import arg
from sys import prefix
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
import os
from datetime import datetime
from simpletransformers.t5 import T5Model, T5Args
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # Set the seed
    set_seed(args.seed)

    # Read data
    dataset_df = pd.read_csv(args.data_dir)
    dataset_df = dataset_df[dataset_df.columns[3:]][dataset_df['is_duplicate']==1]
    dataset_df = dataset_df[dataset_df.columns[:-1]]

    # Renaming the columns
    dataset_df.columns = ["input_text", "target_text"]
    # print(dataset_df)

    # Adding a prefix. Here we shall keep "paraphrase" as a prefix.
    dataset_df["prefix"] = "paraphrase"

    # Data split
    train_df, eval_df = train_test_split(dataset_df, test_size=0.1, random_state=args.seed)

    # Save the train/val data
    train_df.to_csv(os.path.join('data', 'qqp_train.csv'))
    eval_df.to_csv(os.path.join('data', 'qqp_val.csv'))

    # Configure the model
    model_args = T5Args()
    model_args.num_train_epochs = args.num_train_epochs
    model_args.evaluate_generated_text = False
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_verbose = False
    model_args.overwrite_output_dir = True
    model_args.max_length = args.max_length
    model_args.max_seq_length = args.max_length
    model_args.num_beams = None
    model_args.do_sample = True
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.save_steps = -1
    model_args.fp16 = False
    model_args.learning_rate = args.learning_rate
    model_args.manual_seed = args.seed
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    # model_args.use_multiprocessed_decoding = False
    model_args.num_return_sequences = args.num_return_sequences
    model_args.output_dir = args.output_dir

    # Train the model
    if args.train:
        print("*"*10 + "Load model" + "*"*10)
        model = T5Model("t5", "t5-small", args=model_args)
        print("*"*10 + "Done!" + "*"*10)
        print("*"*10 + "Training start!" + "*"*10)
        model.train_model(train_df, eval_data=eval_df)
        print("*"*10 + "Done!" + "*"*10)

    # Eval
    if args.eval:
        root_dir = os.getcwd()
        trained_model_path = os.path.join(root_dir, args.output_dir)
        print("*"*10 + "Load trained model from: " + "*"*10)
        print(trained_model_path)

        ## beam search decoding
        if args.beam_search:
            model_args.do_sample = False
            model_args.num_beams = args.num_beams
            model_args.no_repeat_ngram_size = args.no_repeat_ngram_size
            trained_model = T5Model("t5", trained_model_path, args=model_args)
            
        ## top k/ top p sampling
        else:
            trained_model = T5Model("t5", trained_model_path, args=model_args)

        ## test a sentence
        # prefix = "paraphrase"
        # s = trained_model.predict([f"{prefix}: How can I be a good PHD?"])
        # for p in s[0]:
        #     print(p)

        to_predict = [
        prefix + ": " + str(input_text)
        for prefix, input_text in zip(eval_df["prefix"].tolist(), eval_df["input_text"].tolist())
        ]
        truth = eval_df["target_text"].tolist()
        print("*"*10 + "Evaluation start!" + "*"*10)
    
        preds = trained_model.predict(to_predict)
        
        # Saving the predictions if needed
        print("*"*10 + "Svae the predictions" + "*"*10)
        os.makedirs("predictions", exist_ok=True)
        with open(os.path.join('predictions', args.pred_file), "w") as f:
            for i, text in enumerate(eval_df["input_text"].tolist()):
                f.write(str(text) + "\t" + truth[i] + "\t")
                for pred in preds[i]:
                    f.write(str(pred) + "\t")
                f.write("\n")
        print("*"*10 + "Done!" + "*"*10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/train.csv')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--pred_file', type=str, default='predictions.txt')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2)

    args = parser.parse_args()

    main(args)
