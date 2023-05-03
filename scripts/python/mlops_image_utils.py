
def colors_numpy_array_to_pil(input_colors):
	from PIL import Image
	# Transpose into (width, height, channels)
	input_colors = input_colors.transpose(1, 2, 0)
	# Correct Orientation
	input_colors = input_colors[:, ::-1, :]
	# Gamma Correct
	input_colors = pow(input_colors, 1.0/2.2)
	# Convert to RGB space
	input_colors = (input_colors * 255).round().astype("uint8")	
	return Image.fromarray(input_colors)