import os
import torch
import argparse
import optuna
import wandb
import torch.distributed as dist
from functools import partial
from utils.general_utils import (
    create_directory_if_not_exists,
    print_study_statistics,
    get_scheduler,
    save_config,
    update_metric
)
from utils.logging import initialize_logging
from utils.config import (
    initialize_model_config,
    load_config,
    load_model,
    set_command_line_args,
    setup_tokenizer,
    set_device_settings,
    set_seed,
    set_cuda_settings,
    configure_optuna_trial,
    create_and_run_study,
)
from utils.dataset import create_dataloaders, load_datasets
from utils.model import save_model, train, evaluate
from distributed_training.ddp_setup import (
    setup,
    cleanup,
    run_distributed,
    run_optuna_distributed,
)
from omegaconf import OmegaConf
from torch.optim import AdamW
from dataset.collate import CustomCollateFn

from transformers.utils import logging

logging.set_verbosity(
    40
)  # To suppress warnings from long OSM data sequences in the tokenizer

PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
WANDB_ID = wandb.util.generate_id()

os.environ["WANDB_PROJECT"] = "OpenStreetSat"

parser = argparse.ArgumentParser(description="OpenStreetSatellite Project")

parser.add_argument(
    "--config_path", type=str, default="config.yaml", help="Path to Config"
)
parser.add_argument("--log", action="store_true", help="Log to wandb")
parser.add_argument(
    "--distributedgpu", action="store_true", help="Use distributed GPU if available"
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed for reproducibility of results"
)
parser.add_argument(
    "--llm", type=str, default="phi", help="Language model to use (phi or gpt)"
)
parser.add_argument(
    "--optuna", action="store_true", help="Use optuna for hyperparameter tuning"
)
parser.add_argument(
    "--number_trials", type=int, default=10, help="Number of trial for optuna"
)

parser.add_argument(
    "--save", type=str, default="best_model", help="Name of the model to save"
)
parser.add_argument(
    "--similarity_model", type=str, default= "best_model", help="Name of the model to load on the similarity model"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=5.0,
    help="Threshold for pruning trials with optuna",
)
parser.add_argument(
    "--dataset", type=str, default="osm", help="Dataset to use (osm or ucm)"
)
parser.add_argument(
    "--scheduler",
    type=int,
    default=0,
    help="Scheduler to use (0: Constant, 1: OneCycle, 2: Linear with warmup)",
)
parser.add_argument(
    "--vision",
    type=str,
    default="clip",
    help="Vision model to use (clip or remote or vit)",
)

args = parser.parse_args()


def initialize_env(args, master_process=False, trial=None):
    config = load_config(PATH_PROJECT, args.config_path)
    set_device_settings(config)

    if args.optuna:
        trial = configure_optuna_trial(trial, config, args.distributedgpu)

    if args.log and master_process:
        print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    if master_process:
        if args.seed is not None:
            set_seed(args.seed)
            print(f"Seed set to: {args.seed}")
        else:
            print("No seed set.")

    return config, trial


def get_complete_path(directory):
    output_path = os.path.join(
        PATH_PROJECT,
        directory,
    )
    create_directory_if_not_exists(output_path)

    return output_path


def main(rank, trial=None):

    # - Flag used mainly for distributedgpu to know if it is the master process (rank 0) running
    # - Set the device that is using the rank 0 (main device)
    # if we are not using distributedgpu, its a single gpu, so we are the master process (default to True)
    master_process = rank == 0 if args.distributedgpu else True

    # - Flag to allow saving the config only once
    allow_config_save = True

    config, trial = initialize_env(args, master_process, trial)
    
    evaluate_bleu = config.trainer.evaluate_bleu
    
    if config.trainer.metric_to_monitor=="bleu4":
        assert evaluate_bleu is True, "You must calculate the bleu if you want to monitor it!"

    output_model_path = get_complete_path(os.path.join(config.output_dir, args.save))

    if args.log and master_process:
        group_by = f"hyperparameter_tuning_{WANDB_ID}" if args.optuna else "single_run"

        initialize_logging(config, "OpenStreetSat", group_by=group_by)

    model_device = rank if args.distributedgpu else config.device

    llm = config.llm_models[args.llm]
    if master_process:
        print(f"Using {args.llm} model: {llm.path}")

    vision_model = config.vision_model[args.vision]
    if master_process:
        print(f"Using {args.vision} model: {vision_model.path}")

    # Load the tokenizer
    tokenizer = setup_tokenizer(llm.path, args.llm)

    osmcap_config = initialize_model_config(
        config,
        args.llm,
        args.vision,
        args.distributedgpu,
        tokenizer,
        model_device,
        llm,
        vision_model,
        rank,
    ) 

    image_osm_config = None
    if osmcap_config.trainer.image_osm_similarity.enabled:
        # TODO: Refactor this to be more dynamic, right now is a quick solution to get the path
        osmcap_config.img_osm_path = os.path.join(PATH_PROJECT, config.similarity_output_dir, args.similarity_model)
        similarity_config = load_config(
            PATH_PROJECT,
            os.path.join(config.similarity_output_dir, args.similarity_model, "config.yaml"),
        )

        llm = config.llm_models["gpt"]
        vision_model = config.vision_model["clip"]
        image_osm_config = initialize_model_config(
            similarity_config,
            "gpt",
            "clip",
            args.distributedgpu,
            tokenizer,
            model_device,
            llm,
            vision_model,
            rank,
        )

    dataset_config = config.dataset[args.dataset.lower()]

    osm_extra_params = {
        **config.trainer,
        **config.transformations,
    }
    
    datasets = load_datasets(
        dataset_name=args.dataset.lower(),
        dataset_config=dataset_config,
        inference=False,
        tokenizer=tokenizer,
        splits=("train", "val", "test"),
        **osm_extra_params,
    )

    train_dataset, val_dataset, test_dataset = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )

    training_datasets = {
        "train": (train_dataset, config.trainer.batch_size), 
        "val": (val_dataset, 32),
        "test": (test_dataset, 32)                            
    }

    custom_collate_fn = partial(CustomCollateFn, sentence_embedding=config.trainer.sentence_embedding)

    dataloaders = create_dataloaders(
        training_datasets,
        args.distributedgpu,
        seed=args.seed,
        num_workers=0,
        collate_fn=custom_collate_fn(inference=False),
    )

    trainloader, valloader, testloader = dataloaders["train"], dataloaders["val"], dataloaders["test"]
    
    if evaluate_bleu:
        dataset_inference = load_datasets(
            dataset_name=args.dataset.lower(),
            dataset_config=dataset_config,
            inference=True,
            tokenizer=tokenizer,
            splits=("train", "val", "test"),
            **osm_extra_params,
        )

        train_dataset_inference, val_dataset_inference, test_dataset_inference = (
            dataset_inference["train"],
            dataset_inference["val"],
            dataset_inference["test"],
        )
        
        inference_datasets = {
            "train": (train_dataset_inference, config.trainer.batch_size), 
            "val": (val_dataset_inference, 32),
            "test": (test_dataset_inference, 32)                            
        }

        inference_dataloaders = create_dataloaders(
            inference_datasets,
            args.distributedgpu,
            seed=args.seed,
            num_workers=0,
            collate_fn=custom_collate_fn(inference=True),
        )

        trainloader_inference, valloader_inference, testloader_inference = (
            inference_dataloaders["train"],
            inference_dataloaders["val"],
            inference_dataloaders["test"],
        )

    else:
        trainloader_inference = None
        valloader_inference = None
        testloader_inference = None
    
    if master_process:
        print(f"Using dataset: {args.dataset}")

    model = load_model(
        osmcap_config,
        config,
        master_process,
        args.distributedgpu,
        rank,
        image_osm_config,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
    )

    scheduler = get_scheduler(optimizer, config, len(train_dataset), args.scheduler)

    if master_process:
        print(f"Using type {args.scheduler} scheduler")
    
    best_metric = 1000 if config.trainer.metric_to_monitor=="perplexity" else 0
    
    optuna_early_stopping_counter = 0
    for epoch in range(config.trainer.epochs):
        avg_train_perplexity, metrics_train = train(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader_train=trainloader,
            dataloader_inference=trainloader_inference,
            distributedgpu=args.distributedgpu,
            master_process=master_process,
            use_mixed_precision=config.trainer.use_mixed_precision,
            start_word=config.trainer.start_word,
        )

        if master_process:
            print(f"Avg. Training Perplexity {avg_train_perplexity}")
            if evaluate_bleu:
                print(f"Bleu score on Train {metrics_train[0]}")
                print(f"Compatibility metric on Train {metrics_train[1]}")
        
        if args.log and master_process:
            wandb.log({"train_perplexity": avg_train_perplexity}, commit=False)
            if evaluate_bleu:
                wandb.log({"train_bleu_4": metrics_train[0][-1], "train_comp_score": metrics_train[1]}, commit=False)

        avg_val_perplexity, metrics_val = evaluate(

            epoch=epoch,
            dataloader_eval=valloader,
            dataloader_eval_inference=valloader_inference,
            model=model,
            distributedgpu=args.distributedgpu,
            master_process=master_process,
            start_word=config.trainer.start_word,
            split="val"
        )

        if master_process:
            print(f"Avg. Validation Perplexity {avg_val_perplexity}")
            if evaluate_bleu:
                print(f"Bleu score on Val {metrics_val[0]}")
                print(f"Compatibility metric on Val {metrics_val[1]}")
                
        if args.log and master_process:
            wandb.log({"eval_perplexity": avg_val_perplexity}, commit=False)
            if evaluate_bleu:
                wandb.log({"eval_bleu_4": metrics_val[0][-1], "val_comp_score": metrics_val[1]})
        
        metric = avg_val_perplexity if config.trainer.metric_to_monitor == "perplexity" else metrics_val[0][-1]
        # - If using optuna we don't save the model, because we are not sure if it is the best one
        # we are only interested in hyperparameter tuning
        # - Only save in master process
        if master_process and not args.optuna:
            if update_metric(metric=metric, best_metric=best_metric, minimize=config.trainer.metric_to_monitor == "perplexity"):
                print(
                    f"Saving best model, metric went from {best_metric} to {metric}"
                )
                
                save_path = os.path.join(output_model_path, "best_model.pth")
                save_model(
                    model, save_path, master_process, args.distributedgpu
                )

                if allow_config_save:
                    allow_config_save = False
                    output_config_path = os.path.join(output_model_path, "config.yaml")

                    # Save the configs set in the command line for quicker use
                    set_command_line_args(config, args)
                    save_config(config, output_config_path)

                    if master_process:
                        print(f"Saved config!")
                
                # Save also the epoch checkpoint so that is not getting overwritten 
                save_path = os.path.join(output_model_path, "epoch_"+str(epoch)+".pth")
                save_model(
                    model, save_path, master_process, args.distributedgpu
                )
                # Update metric
                best_metric = metric
            
            else:
                if config.trainer.save_every_epoch:
                    save_path = os.path.join(output_model_path, "epoch_"+str(epoch)+".pth")
                    save_model(
                        model, save_path, master_process, args.distributedgpu
                    )

        if args.optuna:
            trial.report(metric, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early stopping for optuna, this is for the overfitting and quicken up the trials
            if update_metric(metric=metric, best_metric=best_metric, minimize=config.trainer.metric_to_monitor == "val_loss"):
                best_metric = metric
                optuna_early_stopping_counter = 0
            else:
                optuna_early_stopping_counter += 1

            if optuna_early_stopping_counter >= config.optuna_early_stopping:
                print(f"Early stopping trial")
                raise optuna.exceptions.TrialPruned()

    # No need to evaluate the test set if we are using optuna because theres no saved model

    
    if not args.optuna:
        # Evaluate the best model
        model.load_state_dict(torch.load(os.path.join(output_model_path, "best_model.pth")))

        avg_test_perplexity, metrics_test = evaluate(
            epoch=epoch,
            dataloader_eval=testloader,
            dataloader_eval_inference=testloader_inference,
            model=model,
            distributedgpu=args.distributedgpu,
            master_process=master_process,
            start_word=config.trainer.start_word,
            split="test",
        )

        if master_process:
            print(f"Avg. Testing Perplexity {avg_test_perplexity}")
            if evaluate_bleu:
                print(f"Bleu score on Test {metrics_test[0]}")
                print(f"Compatibility metric on Test {metrics_test[1]}")
        
        if args.log and master_process:
            wandb.log({"test_perplexity": avg_test_perplexity})
            if evaluate_bleu:
                wandb.log({"test_bleu_4": metrics_test[0][-1], "test_comp_score": metrics_test[1]})

    if args.log and master_process:
        wandb.finish()

    if args.optuna:
        # optuna requires a return value (a metric to minimize such as acc, loss, etc) to report the results
        return metric


def run_optimize(rank, world_size, return_dict):

    master_process = rank == 0

    setup(rank, world_size)

    if master_process:
        study = create_and_run_study(
            main, rank=rank, number_trials=args.number_trials, threshold=args.threshold
        )
        return_dict["study"] = study
    else:
        for _ in range(args.number_trials):
            try:
                main(rank=rank)
            except optuna.TrialPruned:
                pass

    dist.barrier()
    cleanup()


def run_distributed_setup(rank, world_size):
    setup(rank, world_size)
    main(rank)
    dist.barrier()
    cleanup()


if __name__ == "__main__":

    # Enable backends for TF32 magic
    set_cuda_settings()

    if args.distributedgpu:
        world_size = torch.cuda.device_count()
        if args.optuna:
            study = run_optuna_distributed(run_optimize, world_size)

            print_study_statistics(study, f"study_statistics_{WANDB_ID}.txt")
        else:
            run_distributed(run_distributed_setup, world_size)

    else:
        if args.optuna:
            study = create_and_run_study(
                main, rank=0, number_trials=args.number_trials, threshold=args.threshold
            )

            print_study_statistics(study, f"study_statistics_{WANDB_ID}.txt")
        else:
            main(rank=0)
