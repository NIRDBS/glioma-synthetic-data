"""
The shuffled-bag-label control:
permute which label goes with which bag, holding every instance's features,
bag membership, and the CV split completely fixed, then refit and re-run
Stage 5's spatial validation. If lag>=3 coherence survives a permutation
that has severed the label-bag link entirely, the original result cannot be
attributed to the true label -- it must come from something else (real
spatial autocorrelation in the raw features, an acquisition artifact, etc).
This is the only available way to validate the spatial claim now that no
tumor-size ground truth exists to check a variogram range against
(see [[project_no_tumor_ground_truth]]).

The permutation is EXACT -- a random reassignment of the true label
multiset across bags, not independent resampling -- so class balance stays
identical to the real run. This isolates exactly the causal question above
rather than also changing what a spatially-uninformative baseline would
look like.
"""
import numpy as np


def permuted_bag_label_map(bag_ids, y_bag_per_instance, seed):
    """
    {bag_id: permuted_label} -- computed ONCE, globally, not per fold, so a
    bag that spans two CV folds (the 4 duplicated
    slides) gets the SAME permuted label in both -- required for
    aggregate_bag_scores_across_folds's own "a bag's label must not change
    across folds" assertion to keep holding under the permutation too.
    """
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance)
    unique_bags = np.array(sorted(set(bag_ids.tolist())))
    true_labels = np.array([y_bag_per_instance[bag_ids == b][0] for b in unique_bags])

    rng = np.random.RandomState(seed)
    permuted_labels = rng.permutation(true_labels)
    return dict(zip(unique_bags.tolist(), permuted_labels.tolist()))


def apply_permuted_labels(bag_ids, label_map):
    """Broadcast label_map back onto every instance, in bag_ids' own row
    order -- the array that gets substituted for the loader's y_class
    in-place, before any fold splitting happens."""
    bag_ids = np.asarray(bag_ids)
    return np.array([label_map[b] for b in bag_ids])
