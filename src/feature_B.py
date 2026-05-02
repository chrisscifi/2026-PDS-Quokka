import skimage as ski
import numpy as np


def border_calc(mask):
    FEATURE_NAMES = ['compactness', 'solidity', 'border_smoothness']

    area = mask.sum()
    if area == 0:
        return {k: np.nan for k in FEATURE_NAMES}

    features = {}

    perimeter = area - ski.morphology.erosion(mask, ski.morphology.disk(3)).sum()

    # compactness
    features['compactness'] = perimeter ** 2 / (area * 12)

    # convex share of area
    convex_hull = ski.morphology.convex_hull_image(mask)
    convex_area = convex_hull.sum()
    features['solidity'] = area / convex_area

    # smoothness of the border
    convex_perimeter = convex_area - ski.morphology.erosion(convex_hull, ski.morphology.disk(3)).sum()
    features['border_smoothness'] = convex_perimeter / perimeter

    return features
