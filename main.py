from __future__ import absolute_import, division, print_function

import argparse
import codecs
import logging
import os
import random
import sys

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler

from core.model.xlmr_for_token_classification import XLMRForTokenClassification
from core.utils.train_utils import add_xlmr_args, evaluate_model, predict_model
from core.utils.data_utils import NerProcessor, create_dataset, convert_examples_to_features


def get_labels(sentences):
    label_set = set([])
    for t in sentences:
        label_set.update(t.label)
    return sorted(list(label_set))


def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory (%s) already exists and is not empty." % args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(args.output_dir, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
                         % args.gradient_accumulation_steps)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval and not args.prediction:
        raise ValueError("At least one of `do_train`, `do_eval` or prediction must be present.")

    processor = NerProcessor()
    label_list = processor.get_labels(args.data_dir)
    print(*label_list, sep="\n")

    num_labels = len(label_list) + 1  # add one for IGNORE label

    train_examples = None
    num_train_optimization_steps = 0

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    
    # preparing model configs
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__

    #device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    device = "cuda:0"
    logger.info(device)

    # creating model
    model = XLMRForTokenClassification(pretrained_path=args.pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size,
                                       dropout_p=args.dropout, device=device)

    model.to(device)
    no_decay = ['bias', 'final_layer_norm.weight']

    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # freeze model if necessary
    if args.freeze_model:
        logger.info("Freezing XLM-R model...")
        for n, p in model.named_parameters():
            if 'xlmr' in n and p.requires_grad:
                p.requires_grad = False

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    #global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, model.encode_word)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_data = create_dataset(train_features)

        train_sampler = RandomSampler(train_data)
    
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # getting validation samples
        val_examples = processor.get_dev_examples(args.data_dir)
        val_features = convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, model.encode_word)

        val_data = create_dataset(val_features)
        
        best_val_f1 = 0.0

        for epoch_no in range(1, args.num_train_epochs+1):
            logger.info("Epoch %d" % epoch_no)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            model.train()
            steps = len(train_dataloader)
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids, l_mask, valid_ids, = batch
                loss = model(input_ids, label_ids, l_mask, valid_ids)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if step % 100 == 99:
                    logger.info('Step = %d/%d; Loss = %.4f' % (step+1, steps, tr_loss / (step+1)))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()

            logger.info("\nTesting on validation set...")
            f1, report = evaluate_model(model, val_data, label_list, args.eval_batch_size, device)
            if f1 > best_val_f1:
                best_val_f1 = f1
                logger.info("\nFound better f1=%.4f on validation set. Saving model\n" % f1)
                logger.info("%s\n" % report)
                
                torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
            
            else:
                logger.info("\nNo better F1 score: {}\n".format(f1))
    else:# load a saved model
        state_dict = torch.load(open(os.path.join(args.output_dir, 'model.pt'), 'rb'))
        model.load_state_dict(state_dict)
        logger.info("Loaded saved model")

    model.to(device)

    if args.do_eval:
        if args.eval_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
        elif args.eval_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_word)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_data = create_dataset(eval_features)
        f1_score, report = evaluate_model(model, eval_data, label_list, args.eval_batch_size, device)

        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            writer.write(report)
            logger.info("Done.")

    if args.prediction and args.eval_on:
        eval_examples = processor.get_test_examples(args.data_dir)

        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_word)

        eval_data = create_dataset(eval_features)
        predictions = predict_model(model, eval_data, label_list, args.eval_batch_size, device)

        with codecs.open(args.prediction, "w", "utf8") as fout:
            for tokens, predicted_labels in zip(eval_examples, predictions):
                # for token_word, token_label, predicted_label in zip(tokens.text_a.split(" "), token.label, predicted_labels):
                for token, predicted_label in zip(tokens.text_a.split(" "), predicted_labels):
                    # fout.write("%s\t%s\t%s\n" % (token_word, token_label,predicted_label))
                    fout.write("%s\t%s\n" % (token, predicted_label))
                fout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except ValueError as er:
        print("[ERROR] %s" % er)
