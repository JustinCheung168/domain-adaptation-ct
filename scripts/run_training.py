#!/usr/bin/env python3

from transformers import Trainer, TrainingArguments, set_seed
import datetime
from src.learn.dann import ResNetForMultiLabel

def run_dann_training_experiment(lambda_scheduler, num_epochs: int, lr: float, optim: str, weight_decay: float, seed: int, output_dir: str, from_checkpoint=None):
    """
    Top-level function for training a model.

    Args:
    - num_epochs: Total number of epochs to train for.
    """
    
    # Set seed
    torch.manual_seed(seed)
    set_seed(seed)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Initialize model
    config = br.ResNetConfig()
    model = br.ResNetForMultiLabel(config=config, num_d1_classes=11, num_d2_classes=2, lamb = 0)

    scheduler_name = lambda_scheduler.__name__

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{scheduler_name}_results_{date_str}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        learning_rate=lr,
        weight_decay=weight_decay,
        seed=seed,
        optim=optim,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset= train_data,
        eval_dataset= val_data,
        compute_metrics=br.make_metrics_fn(model),
        callbacks=[br.LambdaUpdateCallback(model, lambda_scheduler, num_epochs)]
    )
    if from_checkpoint != None:
        trainer.train(resume_from_checkpoint=from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(f"{output_dir}/{scheduler_name}_final_model_{date_str}")

    metrics = trainer.evaluate()

    return metrics

if __name__ == "__main__":
    run_experiment(
        br.lambda_scheduler,
        num_epochs = 50,
        lr = 0.1,
        optim = 'sgd',
        weight_decay = 1e-4,
        seed = 42,
        output_dir = f'./data/D21_results/',
    )
    
