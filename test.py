import argparse
import json

import checkpoint
import predictor


def main():
    ap = argparse.ArgumentParser(description='This script allow to predict using a pre-trained model')

    ap.add_argument(
        'image_path',
        default='/home/workspace/paind-project/flowers/test/1/image_06752.jpg'
    )
    ap.add_argument(
        'checkpoint',
        default='/home/workspace/paind-project/checkpoint.pth'
    )
    ap.add_argument(
        '--top_k',
        dest='top_k',
        type=int,
        default=5
    )
    ap.add_argument(
        '--category_names',
        dest='category_names',
        default='cat_to_name.json'
    )
    ap.add_argument(
        '--gpu',
        dest='gpu',
        action='store_true'
    )

    args = ap.parse_args()

    if args.image_path and args.checkpoint:
        model = checkpoint.load(
            checkpoint_path=args.checkpoint,
            gpu=args.gpu
        )

        probs, classes = predictor.predict(
            image_path=args.image_path,
            model=model,
            top_k=args.top_k,
            gpu=args.gpu
        )

        with open(args.category_names, 'r') as json_file:
            cat_to_name = json.load(json_file)

        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]

        for c, p in zip(classes, probs):
            print(c, p)


if __name__ == "__main__":
    main()
