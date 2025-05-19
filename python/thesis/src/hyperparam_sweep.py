import argparse
import wandb
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='config/sweep.yaml'
    )
    parser.add_argument(
        '--project',
        required=True
    )
    parser.add_argument(
        '--entity',
        required=True
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(
        cfg,
        project=args.project,
        entity=args.entity
    )

    print(f'Created sweep {args.entity}/{args.project}/{sweep_id}')


if __name__ == '__main__':
    main()
