import argparse

from torch import nn
from torch import optim

import checkpoint
import loader
import nnetwork
import trainer


def main():
    ap = argparse.ArgumentParser(description='This script allow to train a pre-trained model')

    ap.add_argument(
        'data_dir',
        default='./flowers'
    )
    ap.add_argument(
        '--save_dir',
        dest='save_dir',
        default='./checkpoint.pth'
    )
    ap.add_argument(
        '--learning_rate',
        dest='learning_rate',
        type=float,
        default=0.001
    )
    ap.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=1
    )
    ap.add_argument(
        '--arch',
        dest='arch',
        type=str,
        default='vgg16'
    )
    ap.add_argument(
        '--hidden_units',
        dest='hidden_units',
        type=int,
        default=120
    )
    ap.add_argument(
        '--gpu',
        dest='gpu',
        action='store_true'
    )

    args = ap.parse_args()

    if args.data_dir:
        # Load data
        data = loader.load(args.data_dir)

        # Setup Network
        model = nnetwork.setup(
            arch=args.arch,
            hidden_units=args.hidden_units,
            gpu=args.gpu
        )

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)

        # Train Model
        trainer.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=data['train_loader'],
            validation_loader=data['validation_loader'],
            epochs=args.epochs,
            gpu=args.gpu
        )

        # Save Checkpoint
        checkpoint.save(
            save_dir=args.save_dir,
            model=model,
            arch=args.arch,
            hidden_units=args.hidden_units,
            train_data=data['train_data']
        )


if __name__ == "__main__":
    main()