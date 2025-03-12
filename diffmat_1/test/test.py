from pathlib import Path

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.core.io import read_image

# Enable on-screen logging
config_logger(level='default')

# Input and output file paths
sbs_file_path = Path(r"C:\Users\lvxiaoyu\Desktop\Substance_graph.sbs")
img_path = Path(r"C:\Users\lvxiaoyu\Desktop\test.png")
result_dir = Path(r"./testtest")

# Specify a location for storing pre-cached texture images from SAT
external_input_dir = result_dir / 'external_input'

# Translate the source material graph (using 512x512 resolution)
translator = MGT(sbs_file_path, res=9, external_noise=False)
graph = translator.translate(external_input_folder=external_input_dir, device='cuda')

# Compile the graph to generate a differentiable program
graph.compile()

# Read the target image (convert into a BxCxHxW tensor) and run gradient-based optimization for 1k iterations
target_img = read_image(img_path, device='cuda')[:3].unsqueeze(0)
optimizer = Optimizer(graph, lr=5e-4,metric="combine")
optimizer.optimize(target_img, num_iters=1000, result_dir=result_dir)