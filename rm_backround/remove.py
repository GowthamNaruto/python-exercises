from rembg import remove
from PIL import Image

# Specify the path to import and export
input_path = "./images/kr.jpg"
output_path = "./images/kr.png"

# Read the given file
inp = Image.open(input_path)

# Remove the background by running the remove function
output = remove(inp)

# Save the output file
output.save(output_path)
