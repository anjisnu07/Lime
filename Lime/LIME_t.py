import numpy as np
import copy
import cv2
from skimage.segmentation import quickshift, mark_boundaries
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import json
from tqdm.auto import tqdm

# Load model and class names
model = InceptionV3(weights='imagenet', include_top=True)
with open("imagenet_class_index.json") as f:
    class_names = json.load(f)

# Step 1: Generate Superpixel Areas
def generate_superpixels(image):
    resized_image = cv2.resize(image, (299, 299))
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    superpixels = quickshift(resized_image_rgb, kernel_size=4, max_dist=200, ratio=0.2)
    return superpixels

# Step 2: Create Perturbed Images and Data Labels
def data_labels(image, superpixels, num_samples, distance_metric='cosine', kernel_width=None):
    resized_image = cv2.resize(image, (299, 299))
    n_features = np.unique(superpixels).shape[0]
    data = np.random.randint(0, 2, num_samples * n_features).reshape((num_samples, n_features))
    data[0, :] = 1  

    perturbed_images = []
    fudged_image = resized_image.copy()
    for x in np.unique(superpixels):
        fudged_image[superpixels == x] = np.mean(resized_image[superpixels == x], axis=0)

    for row in data:
        temp = copy.deepcopy(resized_image)
        zeros = np.where(row == 0)[0]
        for z in zeros:
            temp[superpixels == z] = fudged_image[superpixels == z]
        perturbed_images.append(temp)

    perturbed_images = np.array(perturbed_images)
    images_preprocessed = preprocess_input(perturbed_images)
    predictions = model.predict(images_preprocessed)

    distances = pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()
    if kernel_width is None:
        kernel_width = np.sqrt(n_features) * 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    return data, predictions, weights

# Step 3: Fit a Simple Interpretable Model
def fit_interpretable_model(data, predictions, weights):
    model = Ridge(alpha=0, fit_intercept=True)
    model.fit(data, predictions, sample_weight=weights)
    return model

# Step 4: Identify Important Superpixels
def plot_important_segments(image, superpixels, model, num_segments_to_plot=5):
    coefs = model.coef_[0]
    top_features = np.argsort(np.abs(coefs))[::-1][:num_segments_to_plot]

    mask = np.zeros_like(superpixels)
    for feature in top_features:
        mask[superpixels == feature] = 1

    resized_mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(image, resized_mask))
    plt.axis('off')
    plt.show()

    return resized_mask


original_image = cv2.imread('Image/tiger-n02129604/n02129604_1055.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
superpixels = generate_superpixels(original_image)

# Plotting superpixels
def plot_superpixel_image(image, superpixels):
    superpixels_resized = cv2.resize(superpixels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(image, superpixels_resized))
    plt.axis('off')
    plt.show()

plot_superpixel_image(original_image, superpixels)

# perturbed images and labels
num_samples = 1000
distance_metric = 'cosine' 
kernel_width = None  

data, predictions, weights = data_labels(original_image, superpixels, num_samples, distance_metric)


interpretable_model = fit_interpretable_model(data, predictions, weights)


mask = plot_important_segments(original_image, superpixels, interpretable_model)


print(f"Original image shape: {original_image.shape}")
print(f"Mask shape: {mask.shape}")
combined_segment = mask[:, :, np.newaxis] * original_image
print(f"Combined segment shape before resize: {combined_segment.shape}")

combined_segment_resized = cv2.resize(combined_segment, (299, 299))
combined_segment_preprocessed = preprocess_input(combined_segment_resized)
combined_segment_predictions = model.predict(np.expand_dims(combined_segment_preprocessed, axis=0))

top_predicted_classes = np.argsort(-combined_segment_predictions[0])[:5]
print("Top 5 Predicted Classes:")
for idx in top_predicted_classes:
    class_name = class_names[str(idx)][1]
    probability = combined_segment_predictions[0][idx]
    print(f"{class_name}: {probability:.2f}")
