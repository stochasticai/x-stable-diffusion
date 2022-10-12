import tensorrt as trt
import os, sys, argparse 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
from time import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_unet_path", default='./models/unet/unet.onnx', type=str, help="Onnx unet model path")
    parser.add_argument("--save_path", default='unet.engine', type=str, help="TensorRT saved path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")

    return parser.parse_args()

def convert(args):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file(args.onnx_unet_path)
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit('ONNX model parsing failed')

    latents_shape = (args.batch_size*2, 4, args.img_size[0] // 8, args.img_size[1] // 8)
    min_latents_shape = (1, 4, args.img_size[0] // 8, args.img_size[1] // 8)
    embed_shape = (args.batch_size*2, args.max_seq_length, 768)
    min_embed_shape = (1, args.max_seq_length, 768)
    timestep_shape = (args.batch_size*2,)
    min_timestep_shape = (1,)

    config = TRT_BUILDER.create_builder_config()
    if args.batch_size != 1:
        config.max_workspace_size = 2 << 40
        TRT_BUILDER.max_batch_size = args.batch_size*2
    
    profile = TRT_BUILDER.create_optimization_profile() 
    profile.set_shape("sample", min_latents_shape, latents_shape, latents_shape) 
    profile.set_shape("encoder_hidden_states", min_embed_shape, embed_shape, embed_shape) 
    profile.set_shape("timestep", min_timestep_shape, timestep_shape, timestep_shape) 
    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = TRT_BUILDER.build_serialized_network(network, config)
            
    ## save TRT engine
    with open(args.save_path, 'wb') as f:
        f.write(serialized_engine)
    print(f'Engine is saved to {args.save_path}')

if __name__ == "__main__":
    args = get_args()
    assert args.batch_size <=2, "Can not fit higher batch size than 2 when converting the model"
    assert args.max_seq_length <= 77, "Maximum sequence length is 64"

    convert(args)