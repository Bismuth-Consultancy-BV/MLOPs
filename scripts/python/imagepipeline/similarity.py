from transformers import AutoProcessor, CLIPModel
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as G
from torchmetrics import StructuralSimilarityIndexMeasure
import mlops_image_utils as imutils

# calculates similarity of cip embeddings of two PIL images
def clip_embedding_similarity(image1, image2, model_path):

	model = CLIPModel.from_pretrained(model_path)
	processor = AutoProcessor.from_pretrained(model_path)

	image1, image2 = imutils.ensure_same_pil_image_dimensions(image1, image2)

	inputs1 = processor(images=image1, return_tensors="pt")
	image_features1 = model.get_image_features(**inputs1)

	inputs2 = processor(images=image2, return_tensors="pt")
	image_features2 = model.get_image_features(**inputs2)

	# Reshape the tensors to be 2-D
	tensor1_2d = image_features1.view(2, -1)
	tensor2_2d = image_features2.view(2, -1)

	# Normalize the tensors along the last dimension
	normalized_tensor1 = F.normalize(image_features1)
	normalized_tensor2 = F.normalize(image_features2)

	# Calculate the cosine similarity between the normalized tensors
	similarity = torch.matmul(normalized_tensor1, normalized_tensor2.transpose(0, 1))
	sim = similarity.item()
	#sim = (sim +  1.0) * .5
	sim = max(sim, 0.0)

	return sim

def ssim(pil_image1, pil_image2, data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03):

    pil_image1, pil_image2 = imutils.ensure_same_pil_image_dimensions(pil_image1, pil_image2)

    # Create an instance of the SSIM metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range, kernel_size, sigma, K=(k1, k2))

    # Compute SSIM
    ssim_value_tensor = ssim_metric(G.to_tensor(pil_image1).unsqueeze(0), G.to_tensor(pil_image2).unsqueeze(0))
    ssim_value = ssim_value_tensor.item()

    # remap cosine similarity (-1. to +1.) into 0 to 1 range
    #ssim_value = (ssim_value + 1.) * .5
    ssim_value = max(ssim_value, 0.0)

    # Print the similarity measure
    return ssim_value
