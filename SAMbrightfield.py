import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

path = "z:\\Users\\Benedetta\\3. IF imaging and analysis\\Testing\\"

#Open a tif file
im = Image.open(path+"AED_40x_005.tif")

# Convert to numpy array with explicit dtype and ensure it's in the native byte order
im_array_raw = np.asarray(im, dtype=np.float32).copy()  # Create a copy with native byte order

# Print the min and max values to understand the data range
print(f"Original image range: {im_array_raw.min()} to {im_array_raw.max()}")

# Normalize for display (0-1 range for floats)
im_array = (im_array_raw - im_array_raw.min()) / (im_array_raw.max() - im_array_raw.min())

# Get the shape of the array
print(f"Image shape: {im_array_raw.shape}")

# Check if grayscale (1 channel) and convert to RGB (3 channels) for the model
if len(im_array_raw.shape) == 2:
    print("Converting grayscale image to RGB...")
    # For display purposes, normalize to 0-255 range and convert to uint8
    im_array_uint8 = (im_array * 255).astype(np.uint8)
    # Convert grayscale to RGB by repeating the channel 3 times
    im_array_rgb = np.stack([im_array_uint8, im_array_uint8, im_array_uint8], axis=2)
    print(f"RGB image shape: {im_array_rgb.shape}")
elif len(im_array_raw.shape) == 3 and im_array_raw.shape[2] == 1:
    print("Converting single-channel image to RGB...")
    im_array_uint8 = (im_array * 255).astype(np.uint8)
    im_array_rgb = np.concatenate([im_array_uint8, im_array_uint8, im_array_uint8], axis=2)
    print(f"RGB image shape: {im_array_rgb.shape}")
else:
    # Already RGB or multi-channel
    im_array_uint8 = (im_array * 255).astype(np.uint8)
    im_array_rgb = im_array_uint8
    print(f"Using existing multi-channel image: {im_array_rgb.shape}")

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(im_array_rgb)

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from IPython.display import display, clear_output

class PointSelector:
    def __init__(self, image):
        self.image = image
        self.points = []
        self.labels = []  # Default label is 1
        self.fig_created = False
        self.setup_figure()
        
    def setup_figure(self):
        # Create figure and display image
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.imshow(self.image)
        
        # Add instruction text
        self.fig.text(0.5, 0.01, 'Click to add points. Press Enter when done.', 
                     ha='center', va='bottom', fontsize=12)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Add reset button
        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        self.ax.set_title(f'Selected Points: {len(self.points)}')
        plt.axis('on')
        self.fig_created = True
        
    def on_click(self, event):
        # Ignore clicks outside the image
        if event.inaxes != self.ax:
            return
            
        # Add the point
        x, y = int(event.xdata), int(event.ydata)
        self.points.append([x, y])
        self.labels.append(1)  # Default label is 1
        
        # Update visualization
        self.show_points()
        
    def on_key(self, event):
        if event.key == 'enter':
            # In Jupyter, we don't want to close the figure
            # Just signal that we're done
            clear_output(wait=True)
            print(f"Selection complete! {len(self.points)} points selected.")
            self.done = True
            
    def reset(self, event):
        self.points = []
        self.labels = []
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(f'Selected Points: {len(self.points)}')
        plt.draw()
            
    def show_points(self):
        # Clear and redraw image
        self.ax.clear()
        self.ax.imshow(self.image)
        
        # Plot points
        points_array = np.array(self.points)
        if len(points_array) > 0:
            self.ax.scatter(
                points_array[:, 0],
                points_array[:, 1],
                color='red',
                marker='*',
                s=200,
                edgecolor='white',
                linewidth=1.25
            )
            
        self.ax.set_title(f'Selected Points: {len(self.points)}')
        plt.draw()
        
    def get_points(self):
        return np.array(self.points), np.array(self.labels)

def show_points(coords, labels, ax):
    """Show points on the image."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color='green',
        marker='*',
        s=200,
        edgecolor='white',
        linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color='red',
        marker='*',
        s=200,
        edgecolor='white',
        linewidth=1.25
    )

# Example
# Create the point selector
selector = PointSelector(im_array_rgb)

