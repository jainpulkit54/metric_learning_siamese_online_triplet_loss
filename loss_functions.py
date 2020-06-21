import torch
import torch.nn as nn
import numpy as np

def triplet_loss(emb_anchor, emb_positive, emb_negative, margin = 1):
	
	max_fn = nn.ReLU()
	positive_pair_loss = torch.sum((emb_anchor - emb_positive)**2, dim = 1)
	negative_pair_loss = torch.sum((emb_anchor - emb_negative)**2, dim = 1)
	loss = max_fn(margin + positive_pair_loss - negative_pair_loss)
	loss = torch.mean(loss)
	return loss

def pdist(embeddings, squared = True):

	# The shape of embeddings will be [batch_size, 2]
	max_fn = nn.ReLU()
	embeddings_transpose = torch.transpose(embeddings, 0, 1)
	dot_product = torch.mm(embeddings, embeddings_transpose)
	square_norm = torch.diagonal(dot_product, 0)
	distances = square_norm.view(1,-1) - 2.0 * dot_product + square_norm.view(-1,1)
	# Because of computation errors, some distances might be negative, so we force all of them to be >= 0
	distances = max_fn(distances)

	if not squared:
		distances = distances + 1e-16 # Adding small epsilon for numerical stability as gradient of square root function if value is 0 will be infinite
		distances = torch.sqrt(distances)

	return distances

def get_anchor_positive_triplet_mask(labels):
	# Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
	
	# Inputs
	# labels: Shape is [batch_size]

	# Outputs
	# ap_mask: Shape is [batch_size, batch_size]
	
	labels = labels.numpy()
	ap_eye = np.eye(labels.shape[0], dtype = np.bool)
	ap_eye = np.logical_not(ap_eye)
	ap_mask = np.equal(labels.reshape(-1,1), labels.reshape(1,-1))
	ap_mask = np.logical_and(ap_mask, ap_eye)
	ap_mask = torch.from_numpy(ap_mask)

	return ap_mask

def get_anchor_negative_triplet_mask(labels):
	# Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

	# Inputs
	# labels: Shape is [batch_size]

	# Outputs
	# an_mask: Shape is [batch_size, batch_size]
	
	labels = labels.numpy()
	an_mask = np.logical_not(np.equal(labels.reshape(-1,1), labels.reshape(1,-1)))
	an_mask = torch.from_numpy(an_mask)
	return an_mask

def batch_hard_online_triplet_loss(labels, embeddings, margin = 1, squared = False):

	# For each of the anchors, we get the hardest positive and the hardest negative
	# These hardest positives will actually be the moderate positives since we are processing in batches
	# These hardest negatives will actually be the moderate negatives since we are processing in batches

	# Inputs
	# labels: Shape is [batch_size]
	# embeddings: Shape is [batch_size, emb_dim]
	# margin: Margin used for calculating the triplet loss
	# squared: If True, uses the Squared Euclidean distance else actual Euclidean distance

	# Outputs
	# triplet_loss: A scalar value obtained using PK terms where:
	# P --> Number of classes
	# K --> Number of Samples in each class

	max_fn = nn.ReLU()
	distance_matrix = pdist(embeddings, squared) # Returns a matrix of size [batch_size, batch_size]
	# For each anchor, get the hardest positive
	# For this, we would first require to calculate the mask of the anchor-positive pairs
	ap_mask = get_anchor_positive_triplet_mask(labels)
	ap_distances = distance_matrix * ap_mask.cuda().float()
	hardest_positives, _ = torch.max(ap_distances, dim = 1, keepdims = True) # Shape is [batch_size, 1]
	# For each anchor, get the hardest negative
	# For this, we would first require to calculate the mask of the anchor-negative pairs
	an_mask = get_anchor_negative_triplet_mask(labels)
	an_mask = an_mask.cuda().float()
	max_dist_in_each_row, _ = torch.max(distance_matrix, dim = 1, keepdims = True) # Shape is [batch_size, 1]
	an_distances = distance_matrix + max_dist_in_each_row * (1 - an_mask)
	hardest_negatives, _ = torch.min(an_distances, dim = 1, keepdims = True) # Shape is [batch_size, 1]

	triplet_loss = max_fn(margin + hardest_positives - hardest_negatives)
	triplet_loss = torch.mean(triplet_loss)

	return triplet_loss

def get_valid_triplet_mask(labels):
	
	# Returns a 3D mask where mask[a, p, n] is True iff the triplet (a,p,n) is valid
	# A triplet (i,j,k) is valid if:
	# -- i,j,k are distinct
	# -- labels[i] == labels[j] and labels[i] != labels[k]

	# Checking if (i,j,k) are distinct
	labels = labels.numpy()
	batch_size = labels.shape[0]
	indices_equal = np.eye(batch_size, dtype = np.bool)
	indices_not_equal = np.logical_not(indices_equal)

	i_not_equal_j = indices_not_equal.reshape(batch_size, batch_size, 1)
	i_not_equal_k = indices_not_equal.reshape(batch_size, 1, batch_size)
	j_not_equal_k = indices_not_equal.reshape(1, batch_size, batch_size)

	distinct_indices = np.logical_and(np.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
	
	labels_equal = np.equal(labels.reshape(-1,1), labels.reshape(1,-1))
	i_equal_j = labels_equal.reshape(batch_size, batch_size, 1)
	i_equal_k = labels_equal.reshape(batch_size, 1, batch_size)

	valid_labels = np.logical_and(i_equal_j, np.logical_not(i_equal_k))
	
	# Now combining the two masks
	mask = np.logical_and(distinct_indices, valid_labels)
	mask = torch.from_numpy(mask)

	return mask

def batch_all_online_triplet_loss(labels, embeddings, margin = 1, squared = False):

	# For each of the valid triplets, average the loss only on the postive ones
	
	# Inputs
	# labels: Shape is [batch_size]
	# embeddings: Shape is [batch_size, emb_dim]
	# margin: Margin used for calculating the triplet loss
	# squared: If True, uses the Squared Euclidean distance else actual Euclidean distance

	# Outputs
	# triplet_loss: A scalar value obtained using PK(PK-K)(K-1) terms where:
	# P --> Number of classes
	# K --> Number of Samples in each class

	max_fn = nn.ReLU()
	distance_matrix = pdist(embeddings, squared) # Returns a matrix of size [batch_size, batch_size]
	anchor_positive_distance = torch.unsqueeze(distance_matrix, dim = 2)
	anchor_negative_distance = torch.unsqueeze(distance_matrix, dim = 1)
	triplet_loss = margin + anchor_positive_distance - anchor_negative_distance
	mask = get_valid_triplet_mask(labels)
	mask = mask.cuda().float()
	triplet_loss = torch.mul(triplet_loss, mask)
	
	# Removing the negative losses (i.e., the easy triplets)
	triplet_loss = max_fn(triplet_loss)

	# Counting the number of positive triplets (i.e., places where loss > 0)
	positive_loss_triplets = torch.ge(triplet_loss, 1e-16).float() # i.e., 0 will be 1e-16 or a very small value
	num_positive_loss_triplets = torch.sum(positive_loss_triplets)
	num_valid_triplets = torch.sum(mask)
	fraction_positive_loss_triplets = num_positive_loss_triplets/(num_valid_triplets + 1e-16)
	triplet_loss = torch.sum(triplet_loss)
	triplet_loss = triplet_loss/num_positive_loss_triplets
	return triplet_loss, fraction_positive_loss_triplets