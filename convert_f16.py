import onnx
from onnxconverter_common import float16
import argparse
import onnxruntime
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument("--infer_size", type=int, default=800, help="inference size")
    args = parser.parse_args()
    return args


def benchmark(model_path, infer_size):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, infer_size, infer_size), np.float32)
    # if 'f16' in model_path:
        # input_data = input_data.astype(np.float16)
        
    # Warming up
    _ = session.run([], {input_name: input_data})
    # Benchmarking
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")



def main():
    args = get_args()

    input_model_path = args.input_model
    output_model_path = args.output_model
    infer_size = args.infer_size
    
    model = onnx.load(input_model_path)
    model_fp16 = float16.convert_float_to_float16(model, min_positive_val=1e-10, max_finite_val=1e5, keep_io_types=True)
    onnx.save(model_fp16, output_model_path)

    # benchmark
    print("benchmarking fp32 model...")
    benchmark(input_model_path, infer_size)

    print("benchmarking fp16 model...")
    benchmark(output_model_path, infer_size)


if __name__ == "__main__":
    main()