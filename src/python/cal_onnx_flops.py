import onnxruntime as ort
import numpy as np
import onnx_tool

onnx_model = "checkpoints/modelx4.onnx"

onnx_tool.model_profile(onnx_model)
# Load your ONNX model
# session = ort.InferenceSession(onnx_model)

# # Run profiling
# options = ort.SessionOptions()
# options.enable_profiling = True

# # Assuming you have an input sample
# input_data = np.random.normal(0, 1, (1, 544, 960, 4)).astype(np.float32) # Prepare your input sample based on the model's input shape

# # Run the model to generate the profile
# output = session.run(None, {"input": input_data})

# # Retrieve and read the profile
# profile_file = session.end_profiling()

# with open(profile_file, "r") as f:
#     profile_data = f.read()

# # Analyze profile data for FLOPs
# print(profile_data)

