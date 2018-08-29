"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment

def parse():
    parser = argparse.ArgumentParser(description="expectation learning")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_exp', action='store_true', help='train or test')
    parser.add_argument('--test_exp', action='store_true', help='train or test')
    parser.add_argument('--keep', action='store_true', help='whether use the trained model')
    parser.add_argument('--batch_size', default=1, help='batch_size')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_exp:
        env_name = args.env_name or 'Pong-v0'
        env = Environment(env_name, args)
        from agent_dir.Model import model
        agent = model(env, args)
        agent.train()
    if args.test_exp:
        env = Environment('Pong-v0', args, test=True)
        from agent_dir.Model import model
        agent = model(env, args)
        test(agent, env)

if __name__ == '__main__':
    args = parse()
    run(args)
