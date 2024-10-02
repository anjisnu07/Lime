# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:28:27 2023

@author: SUKUMAR
"""

import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings
import math

print('Notebook running: keras ', keras.__version__)
np.random.seed(222)

#%% Load pretrained model of InceptionNet

from keras.applications import inception_v3
inceptionV3_model = inception_v3.InceptionV3()

#%% Load image and pre-process for InceptionNet

Xi = skimage.io.imread("Image\cock.jpeg", plugin='matplotlib')
Xi = skimage.transform.resize(Xi, (299,299)) 
Xi = (Xi - 0.5)*2 #Inception pre-processing
skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing

#%% Do the initial prediction of the image

preds = inceptionV3_model.predict(Xi[np.newaxis,:,:,:])
#print(decode_predictions(preds)[0]) #Top 5 classes
print(decode_predictions(preds, top=5)[0]) #Top 5 classes

top_pred_classes = preds[0].argsort()[-5:][::-1]
print(top_pred_classes) #Index of top 5 classes

#%% Create the superpixels

superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]
print(num_superpixels)

skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))

#%%FINDING OUT THE NO. OF PIXELS IN EACH OF THE SUPERPIXELS
sp_vol = np.empty(num_superpixels)
for i in range(num_superpixels) :
    sp_vol[i] = 0
print(superpixels.shape)
total = 0
for i in range(299):
    for j in range(299):
        sp = superpixels[i][j]
        sp_vol[sp] = sp_vol[sp] + 1
print(sp_vol)

for i in range(num_superpixels) :
    total += sp_vol[i]
print(total)

#%% define the perturbation functions

def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

def new_perturb_image(img,perturbation,segments,color_code):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  for i in range(299):
        for j in range(299):
            if(perturbation[segments[i][j]] == 0):
                for k in range(3):
                    perturbed_image[i][j][k] = color_code
  return perturbed_image

# =============================================================================
# #%% Create a perturbation with all ones and check the classification of the perturbed image
# #   The classification should be exactly same with the original image classification
# 
# all_one_perturbation = np.ones(num_superpixels)[np.newaxis,:]
# pertubed_original_image = perturb_image(Xi,all_one_perturbation[0],superpixels)
# original_image_pred = inceptionV3_model.predict(pertubed_original_image[np.newaxis,:,:,:])
# print(decode_predictions(original_image_pred, top=20)[0]) 
# =============================================================================

#%% Create the perturbed images and show a sample perturbed image

num_perturb = 500
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
print(perturbations[250]) #Show example of perturbation
skimage.io.imshow(perturb_image(Xi/2+0.5,perturbations[250],superpixels))

#%% Create the prediction list for all these perturbed images, and decode a sample prediction

predictions = []
count = 0
for pert in perturbations:
    count = count + 1
    perturbed_img = perturb_image(Xi,pert,superpixels)
    #perturbed_img = new_perturb_image(Xi,pert,superpixels,1.0)
    pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:], verbose=0)
    predictions.append(pred)
    print(str(count) + '. ', end="")

print("")
predictions = np.array(predictions)
print(predictions.shape)

print(decode_predictions(predictions[250]))

#%% Create a list of distances of all these perturbed images from the unperturbed image

all_ones_pert = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
print(all_ones_pert[0])
all_ones_img = perturb_image(Xi,all_ones_pert[0],superpixels)
all_ones_pred = inceptionV3_model.predict(all_ones_img[np.newaxis,:,:,:])
skimage.io.imshow(perturb_image(Xi/2+0.5,all_ones_pert[0],superpixels))
print(decode_predictions(all_ones_pred, top=10)[0])
distances = sklearn.metrics.pairwise_distances(perturbations,all_ones_pert, metric='cosine').ravel()
print(distances.shape)

#%% Compute distances between the images, instead of the perturbations

all_ones_pert = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled
new_distances_1 = []
new_distances_2 = []
complete_image_norm = np.linalg.norm(perturb_image(Xi,all_ones_pert,superpixels))
print("computing", end="")
for pert in perturbations :
    print(".", end="")
    perturbed_img = perturb_image(Xi,pert,superpixels)
    original_img = perturb_image(Xi,all_ones_pert,superpixels)
    dist = np.sqrt(np.sum(np.square(perturbed_img - original_img)))
    new_distances_1.append(dist)
    
    perturbed_image_norm = np.linalg.norm(perturb_image(Xi,pert,superpixels))
    cosine = abs(complete_image_norm/perturbed_image_norm)
    new_distances_2.append(cosine)
    
    #print(str(dist) + " , " + str(cosine))
print()

#%%
max_distance = np.max(new_distances_1)
min_distance = np.min(new_distances_1)
print(max_distance)
print(min_distance)

new_distances_1 = (max_distance - new_distances_1) / (max_distance - min_distance)
#print(new_distances_1)

#%% Create the weight matrix. Naturally, weight will be inversely proportional to the distance

kernel_width = 0.25
#weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
weights = np.sqrt(np.exp(-(new_distances_1**2)/(2*kernel_width**2))) #Kernel function
#weights = np.exp(-(new_distances_1**2)/(2*kernel_width**2)) #Kernel function

#%% Now fit a liner model to approximate these weighted predictions with the perturbed images as input

class_to_explain = top_pred_classes[0]
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]
#print(coeff)

#%% Based on these coefficients, select the top 5 features with highest coefficients

num_top_features = 5
top_features = np.argsort(abs(coeff))[-num_top_features:] 
top_features = np.flip(top_features)
print("TOP FEATURES")
print(top_features)
print("TOP COEFFICIENTS")
print(coeff[top_features])
print("PIXEL COUNTS")
print(sp_vol[top_features])
#print(coeff)

mask = np.zeros(num_superpixels) 
mask[top_features]= True #Activate top superpixels

#%% Display the selected superpixels with black perturbation, and feed the image with only the selected superpixels enabled, back to InceptionNet and see the classification

skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels) )
black_perturbed_img = perturb_image(Xi,mask,superpixels)
black_pred = inceptionV3_model.predict(black_perturbed_img[np.newaxis,:,:,:])
print(decode_predictions(black_pred)[0])

#%% Display the selected superpixels with white perturbation, and feed the image with only the selected superpixels enabled, back to InceptionNet and see the classification

skimage.io.imshow(new_perturb_image(Xi/2+0.5,mask,superpixels,1.0) )
white_perturbed_img = new_perturb_image(Xi,mask,superpixels,0.8)
white_pred = inceptionV3_model.predict(white_perturbed_img[np.newaxis,:,:,:])
print(decode_predictions(white_pred)[0])

#%% Calculate the effectiveness of the explanation

def compute_explanation_effectiveness(initial_pred, final_pred):
  score = 0
  for i in range(initial_pred.size):
      x = initial_pred[i]
      y = final_pred[i]
      score += x * (1 - abs(x - y))

  return score

print(compute_explanation_effectiveness(preds[0], black_pred[0]))
print(compute_explanation_effectiveness(preds[0], white_pred[0]))

#%% Calculate the Euclidean distance of the explanation from the original explanation

def compute_explanation_L2Norm(initial_pred, final_pred):
  score = 0
  for i in range(initial_pred.size):
      x = initial_pred[i]
      y = final_pred[i]
      score += (x - y)*(x - y)
  score = 1 - math.sqrt(score)

  return score

print(compute_explanation_L2Norm(preds[0], black_pred[0]))

#%% Calculate the Cosine similarity of the explanation with the original explanation

def compute_explanation_CosineSimilarity(initial_pred, final_pred):
  score = 0
  x_square = 0
  y_square = 0
  for i in range(initial_pred.size):
      x = initial_pred[i]
      y = final_pred[i]
      score += x * y
      x_square += x * x
      y_square += y * y
  score = score / (math.sqrt(x_square * y_square))

  return score

print(compute_explanation_CosineSimilarity(preds[0], black_pred[0]))

#%% Calculate the KL Divergence of the explanation

def compute_explanation_KLdivergence(initial_pred, final_pred):
  score = 0
  for i in range(initial_pred.size):
      x = initial_pred[i]
      y = final_pred[i]
      score += x * math.log((x/y), 2)

  return score

print(compute_explanation_KLdivergence(preds[0], black_pred[0]))
print(compute_explanation_KLdivergence(preds[0], white_pred[0]))

#%% Calculate the Bhattacharyya Distance of the explanation

def compute_bhattacharyya_distance(initial_pred, final_pred):
  score = 0
  for i in range(initial_pred.size):
      x = initial_pred[i]
      y = final_pred[i]
      score += math.sqrt(x * y)
  bh_dist = -math.log(score)

  return bh_dist

#%% Calculate the rank correlation between two prediction arrays

def get_rank_array(input_array) :
    #vec1 = np.array([1,2,7,4,9,3,5,8,6])
    vec1 = input_array
    vec2 = np.flip(np.sort(vec1))
    #print(vec1)
    #print(vec2)
    vec3 = np.zeros(np.size(vec1))
    for i in range(np.size(vec1)) :
        vec3[i] = np.where(vec2 == vec1[i])[0][0]
    #print(vec3)
    return vec3

def get_rank_correlation(arr1, arr2) :
    rho = np.corrcoef(get_rank_array(arr1), get_rank_array(arr2))
    return rho[0][1]

#%%Now we will mask the top superpixels and feed the image to inceptionNet to see the results

masked_image = np.ones(num_superpixels)
masked_image[top_features] = 0
skimage.io.imshow(perturb_image(Xi/2+0.5,masked_image,superpixels) )
masked_preds = inceptionV3_model.predict(perturb_image(Xi,masked_image,superpixels)[np.newaxis,:,:,:])
print(decode_predictions(masked_preds)[0]) #Top 5 classes

print(compute_explanation_effectiveness(preds[0], masked_preds[0]))
print(compute_explanation_KLdivergence(preds[0], masked_preds[0]))

#%% display the top superpixels passed as an array

def display_superpixel_array(splist) :
    masked_original = np.zeros(num_superpixels)
    masked_original[splist] = 1
    
    newXi = perturb_image(Xi/2+0.5,masked_original,superpixels)
    skimage.io.imshow(newXi)

#%% Display the top superpixels only
#splist = [27]
splist = np.flip(np.argsort(abs(coeff))[-19:])
print(splist)
display_superpixel_array(splist)

#%% What this function does is in a for loop, it gradually activates the top features, one-by-one, on a full black/white image,
# and record the scores for both the cases. This helps us in understanding how the classifier
# is recognizing / classifying the image.

def continuous_evaluation(splist) :
    
    pixel_counts = []
    black_score_trend = []
    black_l2norm_trend = []
    black_cosine_similarity_trend = []
    black_divergence_trend = []
    black_rc_trend = []
    black_bh_dist_trend = []
    white_score_trend = []
    white_divergence_trend = []
    white_rc_trend = []
    white_bh_dist_trend = []
    
    masked_original = np.zeros(num_superpixels)
    for i in range (num_superpixels) :
        print(i)
        masked_original[splist[i]] = 1
        
        if(i == 0):
            pixel_counts.append(sp_vol[splist[i]])
        else :
            cum_count = pixel_counts[i-1] + sp_vol[splist[i]]
            pixel_counts.append(cum_count)
        
        black_perturbed_img = perturb_image(Xi,masked_original,superpixels)
        black_pred = inceptionV3_model.predict(black_perturbed_img[np.newaxis,:,:,:], verbose=0)
        white_perturbed_img = new_perturb_image(Xi,masked_original,superpixels,1.0)
        white_pred = inceptionV3_model.predict(white_perturbed_img[np.newaxis,:,:,:], verbose=0)
        
        black_score = compute_explanation_effectiveness(preds[0], black_pred[0])
        black_l2norm = compute_explanation_L2Norm(preds[0], black_pred[0])
        black_cosine_similarity = compute_explanation_CosineSimilarity(preds[0], black_pred[0])
        black_divergence = compute_explanation_KLdivergence(preds[0], black_pred[0])
        black_rc = get_rank_correlation(preds[0], white_pred[0])
        black_bh_dist = compute_bhattacharyya_distance(preds[0], white_pred[0])
        black_score_trend.append(black_score)
        black_l2norm_trend.append(black_l2norm)
        black_cosine_similarity_trend.append(black_cosine_similarity)
        black_divergence_trend.append(black_divergence)
        black_rc_trend.append(black_rc)
        black_bh_dist_trend.append(black_bh_dist)
        
        white_score = compute_explanation_effectiveness(preds[0], white_pred[0])
        white_divergence = compute_explanation_KLdivergence(preds[0], white_pred[0])
        white_rc = get_rank_correlation(preds[0], white_pred[0])
        white_bh_dist = compute_bhattacharyya_distance(preds[0], white_pred[0])
        white_score_trend.append(white_score)
        white_divergence_trend.append(white_divergence)
        white_rc_trend.append(white_rc)
        white_bh_dist_trend.append(white_bh_dist)
        
        #i = i + 1
        #if(i == num_superpixels) :
        #    test_file = open ("D:/Sukumar/Research_Work/Task-3/test_file.txt", 'a')
        #    test_file.write(str(decode_predictions(black_pred, top=10)))
        #    test_file.write(str(decode_predictions(white_pred, top=10)))
        #    test_file.write(str(preds[0]))
        #    test_file.close()
    print("BLACK SCORE" + '\n' + "*****************************")
    print(black_score_trend)
    print("L2_NORM" + '\n' + "*****************************")
    print(black_l2norm_trend)
    print("COSINE SIMILARITY" + '\n' + "*****************************")
    print(black_cosine_similarity_trend)
   # print("WHITE SCORE" + '\n' + "*****************************")
   # print(white_score_trend)
   # print("BLACK DIVERGENCE" + '\n' + "*****************************")
   # print(black_divergence_trend)
   # print("WHITE DIVERGENCE" + '\n' + "*****************************")
   # print(white_divergence_trend)
   # print("BLACK RANK CORRELATION" + '\n' + "*****************************")
   # print(black_rc_trend)
   # print("WHITE RANK CORRELATION" + '\n' + "*****************************")
   # print(white_rc_trend)
   # print("BLACK BHATTACHARYYA DISTANCE" + '\n' + "*****************************")
   # print(black_bh_dist_trend)
   # print("WHITE BHATTACHARYYA DISTANCE" + '\n' + "*****************************")
   # print(white_bh_dist_trend)
    for i in range (num_superpixels) :
       print(str(i) + " , " + str(pixel_counts[i]) + " , " + str(black_score_trend[i]) + " , " + str(black_l2norm_trend[i]) + " , " + str(black_cosine_similarity_trend[i]))

#%% Evaluate the score with 1 features, then 2 features, and so on...

sorted_features = np.argsort(abs(coeff))[-num_superpixels:] 
sorted_features = np.flip(sorted_features)
continuous_evaluation(sorted_features)

#%% Analyze the top-n features

num_top_features = 12
top_features = np.argsort(abs(coeff))[-num_top_features:] 
top_features = np.flip(top_features)
print(top_features)
print(coeff[top_features])
#print(coeff)

mask = np.zeros(num_superpixels) 
mask[top_features]= True #Activate top superpixels

# Display the selected superpixels with black perturbation, and feed the image with only the selected superpixels enabled, back to InceptionNet and see the classification
skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels) )
black_perturbed_img = perturb_image(Xi,mask,superpixels)
black_pred = inceptionV3_model.predict(black_perturbed_img[np.newaxis,:,:,:])
print(decode_predictions(black_pred)[0])


# Calculate the effectiveness of the explanation
print(compute_explanation_effectiveness(preds[0], black_pred[0]))
