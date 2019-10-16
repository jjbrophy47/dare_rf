"""
Simulation to see how often the color changes of the majority group when m
balls from two groups of red (1) and blue (0) balls are deleted at random.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def majority_changes(n_balls=10000, n_remove=1000, majority='red', n_change=0):
    red_balls = int(n_balls / 2)
    balls_to_remove = np.random.randint(2, size=n_remove)

    for ball_to_remove in balls_to_remove:
        red_balls -= ball_to_remove
        n_balls -= 1

        if majority == 'red' and red_balls < n_balls - red_balls:
            majority = 'blue'
            n_change += 1

        elif majority == 'blue' and red_balls > n_balls - red_balls:
            majority = 'red'
            n_change += 1

    return n_change

    print('changes: {}'.format(n_change))


def main(min_balls=100, max_balls=1000000, step_multiplier=10, remove_frac=0.1, n_runs=100,
         out_dir='output/majority'):
    n_balls = min_balls

    ball_sizes = []
    mean_changes = []
    std_changes = []
    change_to_removal_ratios = []

    while n_balls <= max_balls:
        n_remove = int(n_balls * remove_frac)

        n_change_list = []
        for i in range(n_runs):
            n_change_list.append(majority_changes(n_balls=n_balls, n_remove=n_remove))
        n_change_list = np.array(n_change_list)
        n_change_mean, n_change_std = np.mean(n_change_list), np.std(n_change_list)
        print('n_balls: {}, n_remove: {}, n_change: {:.3f} +/- {:.3f}'.format(n_balls, n_remove,
              n_change_mean, n_change_std))

        ball_sizes.append(n_balls)
        mean_changes.append(n_change_mean)
        std_changes.append(n_change_std)
        change_to_removal_ratios.append(n_change_mean / n_remove)
        n_balls *= step_multiplier

    remove_pct = '{}'.format(int(remove_frac * 100))

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(ball_sizes, mean_changes, marker='.')
    axs[0].set_xlabel('# balls')
    axs[0].set_ylabel('mean # changes')
    axs[0].set_title('# runs: {}, removal: {}%'.format(n_runs, remove_pct))
    axs[0].set_xscale('log')

    axs[1].plot(ball_sizes, change_to_removal_ratios, marker='.')
    axs[1].set_xlabel('# balls')
    axs[1].set_ylabel('mean # changes / # removed')
    axs[1].set_title('# runs: {}, removal: {}%'.format(n_runs, remove_pct))
    axs[1].set_xscale('log')

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'removal_{}.pdf'.format(remove_pct)), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_frac', type=float, default=0.1, help='percentage of balls to remove.')
    args = parser.parse_args()
    print(args)
    main(remove_frac=args.remove_frac)
