

CUDA_VISIBLE_DEVICES=0 python none.py
CUDA_VISIBLE_DEVICES=0 python joint.py
CUDA_VISIBLE_DEVICES=0 python main.py
CUDA_VISIBLE_DEVICES=0 python none_KAN.py
CUDA_VISIBLE_DEVICES=0 python main_KAN.py

For main.py and main_KAN.py we have two parameters alpha and interpolation_method.
We have 3 values for interpolation_method including linear, spline and polynomial.
 For alpha, I have tried 3 values 0.3, 0.5 and 0.7.
Other values can be tried.

