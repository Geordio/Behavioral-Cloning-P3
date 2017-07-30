
import numpy as np
import matplotlib.pyplot as plt
import random

angle_offset = 0.25
measurements = []
centre_col = 0
left_col = 1
right_col = 2
angle_col = 3
angle_offset = 0.25
straight_thres = 0.1
drop_prob = 0.2

# plot a histogram of the dataset
# in addition, print some useful stats
def plot_angles(angles):

    max_angle = max(angles)
    min_angle = min(angles)
    mean_angle = np.mean(angles)
    total_samples = len(angles)
    print('max_angle: {}' .format(max_angle))
    print('min_angle: {}' .format(min_angle))
    print('mean_angle: {}' .format(mean_angle))
    print('total_samples: {}' .format(total_samples))
# width can go from -1 to + 1, but theres not really anything worth plotting above  +/- 0.5
    hist, bins = np.histogram(angles, bins=np.arange(-0.675,0.675,0.05))
    centres = (bins[:-1] + bins[1:]) / 2
    width_of_bar = (bins[1] - bins[0]) * 0.9
    plt.bar(centres, hist, align='center', width=width_of_bar)
    plt.title('Dataset Angle Distribution')
    plt.show()


# helper method to plot images for visualisation
def plot_figure(array_to_plot, labels_array, n_rows, n_columns, figuresize):
    fig, axes = plt.subplots( n_rows, n_columns,figsize=figuresize)
    axes = axes.ravel()

    for i in range(len(array_to_plot)):
        print('i {}'.format(i))
        print('title {}'.format(labels_array[i]))
        axes[i].imshow(array_to_plot[i])
        axes[i].set_title(labels_array[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def get_angles_only(dataset):
    #   i = 0
    centre_angles = []
    centre_paths = []
    left_angles = []
    left_paths = []
    right_angles = []
    right_paths = []
    for sample in dataset:
        angle, centre, left, right = extract_data(sample)

        centre_angles.append(angle)
    return centre_angles

# method to extract the image paths and angles
# returns angle, path_centre, path_left, path_right
def extract_data(line):
    path_centre = line[centre_col]
    path_left = line[left_col]
    path_right = line[right_col]
    angle = float(line[angle_col])
    return angle, path_centre, path_left, path_right

# process a batch but not the images, only the angles.
# used to help visualise
def process_batch_no_images(batch_samples):
    images = []
    angles = []
    for batch_sample in batch_samples:
        angle, path_centre, path_left, path_right = extract_data(batch_sample)

        # add the side images
        left_image_angle = angle + angle_offset
        right_image_angle = angle - angle_offset

        angles.append(angle)
        angles.append(left_image_angle)
        angles.append(right_image_angle)
        # flip images

        angles.append(-angle)
        flipNotFlip = random.randint(0, 1)
        if flipNotFlip > 0:
            angles.append(-left_image_angle)
        else:
            angles.append(-right_image_angle)
    return angles
