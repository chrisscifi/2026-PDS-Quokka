import numpy as np
import skimage as ski
from scipy.ndimage import label
import os


#1. Load image
def load_mask(path):
    img = ski.io.imread(path, as_gray=True)
    mask = (img > 0).astype(np.uint8)
    return mask


#2. Keep largest component 
def largest_component(mask):
    labeled, num = label(mask)

    if num == 0:
        return mask

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1

    return (labeled == largest).astype(np.uint8)


#3. Crop to object
def crop_to_object(mask):
    ys, xs = np.where(mask == 1)

    if len(ys) == 0:
        return mask

    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


#4. Split by centroid
def split_mask(mask):
    ys, xs = np.where(mask == 1)
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    left  = mask[:, :cx]
    right = mask[:, cx:]

    top    = mask[:cy, :]
    bottom = mask[cy:, :]

    return left, right, top, bottom


#5. Align shapes (simple safe version)
def align(A, B):
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    return A[:h, :w], B[:h, :w]


#6. Asymmetry score
def asymmetry(A, B):
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)

    diff = np.abs(A.astype(int) - B.astype(int))

    return diff.sum() / (A.sum() + B.sum() + 1e-8)


#7. Main function
def compute_asymmetry(mask):
    FEATURE_NAMES = ['horizontal_asymmetry', 'vertical_asymmetry', 'combined_asymmetry']

    mask = (np.asarray(mask) > 0).astype(np.uint8)
    mask = largest_component(mask)
    mask = crop_to_object(mask)

    if mask.sum() == 0:
        return {k: np.nan for k in FEATURE_NAMES}

    left, right, top, bottom = split_mask(mask)

    # align
    left, right = align(left, right)
    top, bottom = align(top, bottom)

    # flip
    right = np.fliplr(right)
    bottom = np.flipud(bottom)

    # compute
    h = asymmetry(left, right)
    v = asymmetry(top, bottom)

    features = {}
    features['horizontal_asymmetry'] = float(h)
    features['vertical_asymmetry'] = float(v)
    features['combined_asymmetry'] = float((h + v) / 2)

    return features

if __name__ == "__main__":
    folder = "data/masks"

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)

            try:
                res = compute_asymmetry(load_mask(path))

                print(f"{filename} -> H: {res['horizontal_asymmetry']:.3f}, "
                      f"V: {res['vertical_asymmetry']:.3f}, "
                      f"C: {res['combined_asymmetry']:.3f}")

            except Exception as e:
                print(f"Nope {filename}: {e}")
