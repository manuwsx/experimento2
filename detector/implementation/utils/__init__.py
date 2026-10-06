from .load import load_mesh
from .sampling import sampling_fun
from .views import get_min_camera_distance, sample_standard_viewpoints, sample_fibonacci_viewpoints
from .model_wrappers import DINOWrapper

models = {
    "dino": DINOWrapper,
}