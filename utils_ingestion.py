from sklearn.model_selection import train_test_split
import random
import copy

import pdb


def strat_splits(n_classes, n_draws, pct_splits, n_insts_2_classes, class_2_insts, insts):
    """
    Args:
    - n_classes (n_genera)
    - n_draws (n_sids_ood_eval)
    - pct_splits (pct_ood_eval) -- percentage for val/test, evenly distributed between both e.g. 10% = 5% val, 5% test
    - n_insts_2_classes
    - class_2_insts (genus_2_sids) -- dictionary mapping classes [str] to lists of instances [List(str)]
    - insts -- set of instances e.g. set of species ids (whole dataset)

    Returns
    - [(set(insts), set(insts), set(insts)] --- Train, Val, Test
    """

    def compute_class_hits(n_draws, n_classes):
        """
        num_draws, num_classes --> list(class_hits) e.g. [1,1,1,0,0,0,0,0], [1,1,1,1,0,0], [2,1,1,1,1,1], etc.
        """

        class_hits = [n_draws // n_classes] * n_classes

        plus_ones = n_draws % n_classes
        for i in range(plus_ones):
            class_hits[i] += 1

        return class_hits

    insts_rem = copy.deepcopy(insts)
    insts_eval = []

    n_classes_rem = n_classes
    n_draws_rem = n_draws

    count_min_strat2 = 1 / pct_splits
    i = 0
    while True:
        i += 1
        classes_i = n_insts_2_classes[i]
        if not classes_i:
            # n_insts_2_classes[i] is empty i.e. no classes at count i
            continue

        n_classes_i = len(classes_i)
        n_instances_i = n_classes_i * i
        n_draws_i = round(n_instances_i * pct_splits)

        random.shuffle(classes_i)
        class_hits = compute_class_hits(n_draws_i, n_classes_i)

        for idx, k in enumerate(class_hits):
            c = classes_i[idx]
            inst_hits = random.sample(class_2_insts[c], k)
            insts_eval += inst_hits

        n_classes_rem -= n_classes_i
        n_draws_rem -= n_draws_i

        if i >= count_min_strat2 - 1 and (n_draws_rem >= n_classes_rem or n_classes_rem == 0):
            break

    if n_draws_rem > 0 and n_classes_rem > 0:
        # construct classes_rem & instances_rem (instances_rem structured as tuples with inst_count for sorting, sorting is important for the zipper delegation between val/test)
        classes_rem = []
        insts_counts_rem = []  # List((instance, count))  ~ `count` is the number of instances in the corresponding class

        for count in sorted(list(n_insts_2_classes.keys())):
            if count <= i:
                continue
            else:
                classes = n_insts_2_classes[count]
                for c in classes:
                    classes_rem += [c] * count

                    insts_c = class_2_insts[c]  # list of instances
                    for inst in insts_c:
                        insts_counts_rem += [(inst, count)]

        _, insts_counts_strat2 = train_test_split(
            insts_counts_rem,
            stratify=classes_rem,
            test_size=n_draws_rem,
            shuffle=True,
            random_state=None,
        )

        insts_counts_strat2.sort(key=lambda x: (x[1], x[0]))
        insts_strat2, _ = zip(*insts_counts_strat2)
        insts_eval += insts_strat2

    insts_val = set()
    insts_test = set()

    # "zipper delegation"
    for idx, sid in enumerate(insts_eval):
        if idx % 2 == 0:
            insts_val.add(sid)
        else:
            insts_test.add(sid)

    insts_rem -= insts_val
    insts_rem -= insts_test

    # insts_val and insts_test appear to be getting sorted in alphabetical order, but not insts_rem -- why??? (upon viewing vars in jupyter notebook)
    return insts_rem, insts_val, insts_test
