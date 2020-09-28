import json
import os

def show_best_checkpoint(path, compare_value, comparator):
    with open(os.path.join(path, "stats.json"), "r") as read_file:
        data = json.load(read_file) 
        checkpoints = [epoch_stat[compare_value] for epoch_stat in data]
        if comparator is 'max':
            chckp = max((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        elif comparator is 'min':
            chckp = min((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        return (chckp, data)

if __name__ == '__main__':
    chkp, data = show_best_checkpoint('./', 'epoch_num', 'max')
    print(data[chkp[1]])