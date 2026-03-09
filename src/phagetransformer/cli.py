#!/usr/bin/env python3
"""PhageTransformer CLI — unified entry point for all commands.

Usage:
    phagetransformer init                                      # download model
    phagetransformer predict   --input phages.fasta --model_dir ./models
    phagetransformer train     --dataset_dir ./data --host_genome_dir ./genomes
"""

import sys


def main():
    commands = {
        'init':     ('phagetransformer.init_model', 'Download pre-trained model weights'),
        'predict':  ('phagetransformer.predict',    'Predict phage hosts'),
        'train':    ('phagetransformer.train',      'Train a new model'),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("PhageTransformer — hierarchical DNA classifier for phage host prediction\n")
        print("Usage: phagetransformer <command> [options]\n")
        print("Commands:")
        for cmd, (_, desc) in commands.items():
            print(f"  {cmd:<12s} {desc}")
        print(f"\nRun 'phagetransformer <command> --help' for command-specific options.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: '{cmd}'")
        print(f"Available commands: {', '.join(commands)}")
        sys.exit(1)

    # Remove the command name so each module's argparse sees the right args
    sys.argv = [f"phagetransformer {cmd}"] + sys.argv[2:]

    # Import and run the command's main()
    module_path = commands[cmd][0]
    import importlib
    module = importlib.import_module(module_path)
    module.main()


if __name__ == '__main__':
    main()
