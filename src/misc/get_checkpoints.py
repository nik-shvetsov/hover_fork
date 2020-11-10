import json
import os
import sys

def show_best_checkpoint(path, compare_value='epoch_num', comparator='max'):
    with open(os.path.join(path, "stats.json"), "r") as read_file:
        data = json.load(read_file)
        checkpoints = [epoch_stat[compare_value] for epoch_stat in data]
        if comparator is 'max':
            chckp = max((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        elif comparator is 'min':
            chckp = min((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        return (chckp, data)

if __name__ == '__main__':
    if sys.argv[1]:
        chkp, data = show_best_checkpoint(sys.argv[1])
    elif sys.argv[2]:
        chkp, data = show_best_checkpoint(sys.argv[1], sys.argv[2])
    elif sys.argv[3]:
        chkp, data = show_best_checkpoint(sys.argv[1], sys.argv[2], sys.argv[3])
    print(data[chkp[1]])