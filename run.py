from typing import List, Optional
import datasets
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser, \
    TrainerCallback
from transformers.trainer_utils import PredictionOutput
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import pandas as pd
import numpy as np
import torch

NUM_PREPROCESSING_WORKERS = 2

#  customize the way the loss is computed based on the logits
#  https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss
class CustomTrainer(Trainer):
    def __init__(self, model, args, data_collator, train_dataset=None, eval_dataset=None, tokenizer=None, compute_metrics=None, callbacks=None, optimizers=None, **kwargs):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, compute_metrics, callbacks, optimizers, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def sample_dataset(dataset, sample_ids):
    df = dataset.to_pandas()
    df.set_index('id', inplace=True)
    df = df.loc[sample_ids]
    return datasets.Dataset.from_pandas(df)

def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int]):
    """
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """
    td_df = pd.DataFrame({"guid": train_ids,
                        f"logits_epoch_{epoch}": train_logits,
                        "gold": train_golds})

    logging_dir = os.path.join(output_dir, f"training_dynamics")
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    
class SaveEvalMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # evaluate on the validation dataset
        train_predictions = self.trainer.predict(test_dataset=self.trainer.train_dataset, metric_key_prefix="train")
        log_training_dynamics(self.trainer.args.output_dir, int(state.epoch)-1, self.trainer.train_dataset['id'] , list(train_predictions.predictions) , train_predictions.label_ids)
        # return the original control object
        return control
    
def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa', 'hqa', 'mnli'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--save_training_dynamics', action='store_true', default=False,
                      help='Save training dynamics (logits) for each epoch')
    argp.add_argument('--data_sampling_path', default=None, type=str
                      , help='Use data sampling for training')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    elif args.dataset.endswith('.tsv'):
        dataset_id = None
        folder = "/".join(args.dataset.split('/')[:-1])
        eval_split = 'validation_matched' if args.task == 'mnli' else 'validation'
        # Load from local tsv file
        dataset_files = {'train': args.dataset, eval_split: folder + f'/{eval_split}.tsv', 'test': folder + '/test.tsv'}
        dataset = datasets.load_dataset('csv', data_files=dataset_files, delimiter='\t')
        dataset = dataset.filter(lambda ex: isinstance(ex['hypothesis'], (str,list)))
        print("loaded custom dataset with schema:", dataset)
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',), 'hqa': ('hotpot_qa',), 'mnli': ('glue','mnli')}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if args.task == 'mnli' else 'validation'
        # Load the raw data
        if dataset_id == ('hotpot_qa',):
            dataset = datasets.load_dataset(*dataset_id, 'distractor')
        else:
            dataset = datasets.load_dataset(*dataset_id)

    # If you want to use data sampling for training, you should define your own "prepare_train_dataset" function.
    if args.data_sampling_path is not None:
        #read data sampling file
        sample_ids = np.load(args.data_sampling_path)
        # if len(sample_ids) == len(set(sample_ids)):
        #     dataset['train'] = dataset['train'].filter(lambda ex: ex['id'] in sample_ids)
        # else:
        #     dataset['train'] = sample_dataset(dataset['train'], sample_ids)
        dataset['train'] = sample_dataset(dataset['train'], sample_ids) # this approach faster

    # breakpoint()
    # exit()

    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' or args.task == 'mnli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification,
                     'hqa': AutoModelForQuestionAnswering,
                     'mnli': AutoModelForSequenceClassification}
    
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa' or args.task == 'hqa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli' or args.task == 'mnli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    if args.save_training_dynamics:
        for dataset_group in dataset.keys():
            if "id" not in dataset[dataset_group].column_names:
                dataset[dataset_group] = dataset[dataset_group].add_column(name="id", column=[i for i in range(len(dataset[dataset_group]))])
    
    # for split, dataset in dataset.items():
    #     dataset.to_csv(f"velurib-snli/{split}.tsv", index=None, sep="\t")

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if "id" in train_dataset.column_names:
            train_remove_columns = train_dataset.column_names.remove("id")
        else:
            train_remove_columns = train_dataset.column_names
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_remove_columns
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if "id" in eval_dataset.column_names:
            eval_remove_columns = eval_dataset.column_names.remove("id")    
        else:
            eval_remove_columns = eval_dataset.column_names    
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_remove_columns
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli' or args.task == 'mnli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    
    if args.save_training_dynamics:
        saveEvalMetricscallback = SaveEvalMetricsCallback(trainer=trainer)
        trainer.add_callback(saveEvalMetricscallback)
    
    # Train and/or evaluate
    if training_args.do_train:
        print("Training...")
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
