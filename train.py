import argparse
import torch
import csv
import os

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM
from model.efficient_net import EfficientNet

def save_batch_metrics(epoch, batch_idx, loss, correct, filename="batch_metrics.csv"):
    """Save batch-level loss and accuracy to a CSV file."""
    accuracy = correct.float().mean().item()
    loss_value = loss.mean().item()
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "batch_idx", "loss", "accuracy"])
        writer.writerow([epoch, batch_idx, loss_value, accuracy])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers (for WideResNet).")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate (for WideResNet).")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet (for WideResNet).")
    parser.add_argument("--csv_file_name", default='default.csv', type=str, help="The name of the loss-accuracy curve data file")
    parser.add_argument("--use_sam", default=False, type=bool, help="True if you want to use SAM optimizer.")
    parser.add_argument("--optimizer", default="SGD", type=str, choices=["SGD", "Adam"], help="Choose optimizer: SGD or Adam.")
    parser.add_argument("--model", default="WideResNet", type=str, choices=["WideResNet", "EfficientNet"], help="Choose model: WideResNet or EfficientNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)

    # Model selection
    if args.model == "WideResNet":
        model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    elif args.model == "EfficientNet":
        model = EfficientNet(in_channels=3, labels=10).to(device)
    else:
        raise ValueError("Invalid model choice.")

    # Optimizer selection
    if args.optimizer == "SGD":
        base_optimizer = torch.optim.SGD
        optimizer_params = {'lr': args.learning_rate, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    elif args.optimizer == "Adam":
        base_optimizer = torch.optim.Adam
        optimizer_params = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    else:
        raise ValueError("Invalid optimizer choice.")

    if args.use_sam:
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, **optimizer_params)
    else:
        optimizer = base_optimizer(model.parameters(), **optimizer_params)

    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        batch_idx = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            if args.use_sam:
                # First forward-backward step with SAM
                enable_running_stats(model)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # Second forward-backward step with SAM
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard forward-backward step without SAM
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                save_batch_metrics(epoch, batch_idx, loss.cpu(), correct.cpu(), filename=args.csv_file_name)
                scheduler(epoch)
                batch_idx += 1

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()