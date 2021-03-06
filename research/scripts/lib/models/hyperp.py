import shutil
import tempfile
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import torchvision.transforms as transforms
from .ss import SiameseNetwork
from data.data import SiameseNetworkDataset
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
import mlflow




def train_ss(config, train_dl, data_dir=None, num_epochs=10, num_gpus=0):
    mlflow.pytorch.autolog()
    net = SiameseNetwork(config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=".", name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])


    with mlflow.start_run(tags={"model":"facenet","method":"validation"}) as run:
        trainer.fit(net,train_dl)
        mlflow.log_params(config)



def tune_ss(folder_dataset,num_samples=10, num_epochs=10, gpus_per_trial=0,train_batch_size=20):
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
    train_dl = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size= train_batch_size)


    config = {
        "l1": tune.choice([64, 128]),
        "l2": tune.choice([ 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "threshold":0.7
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["l1", "l2", "lr"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_ss,
            train_dl=train_dl,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_ss")

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis
