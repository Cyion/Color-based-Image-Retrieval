# Imports
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from PIL import Image
from skimage.color import rgb2hsv

# Read Images
rgb_images = [np.array(Image.open(filename)) for filename in glob.glob(os.path.join('images', '*.png'))]

# Convert rgb-Images to hsv-Images
hsv_images = [rgb2hsv(img) for img in rgb_images]

# Quantization of hsv-Images (18 h-values, 3 s-values and 3 v-values, the 4 grey-values will be omitted)
h_bins = np.linspace(0, 1, 19)[1:18]
s_bins = np.linspace(0, 1, 4)[1:3]
v_bins = np.linspace(0, 1, 4)[1:3]

h_images = [img[:, :, 0] for img in hsv_images]
s_images = [img[:, :, 1] for img in hsv_images]
v_images = [img[:, :, 2] for img in hsv_images]

h_images_q = [np.digitize(img, h_bins) for img in h_images]
s_images_q = [np.digitize(img, s_bins) for img in s_images]
v_images_q = [np.digitize(img, v_bins) for img in v_images]

# Generate histograms
hist_bins_images = []
for i in range(len(h_images_q)):
    hist_bins_images.append(9 * h_images_q[i] + 3 * s_images_q[i] + v_images_q[i])

img_hists = []
for i in range(len(hist_bins_images)):
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(rgb_images[i])
    axes[0].axis(False)
    hist, bins, patches = axes[1].hist(hist_bins_images[i].ravel(), bins=np.arange(0, 162, 1), density=True)
    plt.suptitle(f'Image {i + 1}')
    fig.savefig(f'results/hist{i + 1}.png')
    fig.tight_layout()
    plt.show()
    img_hists.append(hist)


# Compare histograms
# Minkowski distance between two histograms (r=1: Manhattan distance, r=2: Euclidean distance)
def distance(hist1, hist2, r):
    abs_diff = abs(hist1 - hist2)
    return sum(abs_diff ** r)


# Compare all histograms against each other with r = 1, 2
for i in range(len(img_hists)):
    for j in range(len(img_hists)):
        manhattan = distance(img_hists[i], img_hists[j], 1)
        euclidean = distance(img_hists[i], img_hists[j], 2)
        total = manhattan + euclidean
        print(f'image {i + 1}, image {j + 1} (r = 1): {manhattan}, (r = 2): {euclidean}, (total): {total}')
