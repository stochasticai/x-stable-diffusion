# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np
import torch

from diffusers.models.attention import AttentionBlock, SpatialTransformer
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Downsample2D, Upsample2D
from diffusers.testing_utils import torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class EmbeddingsTests(unittest.TestCase):
    def test_timestep_embeddings(self):
        embedding_dim = 256
        timesteps = torch.arange(16)

        t1 = get_timestep_embedding(timesteps, embedding_dim)

        # first vector should always be composed only of 0's and 1's
        assert (t1[0, : embedding_dim // 2] - 0).abs().sum() < 1e-5
        assert (t1[0, embedding_dim // 2 :] - 1).abs().sum() < 1e-5

        # last element of each vector should be one
        assert (t1[:, -1] - 1).abs().sum() < 1e-5

        # For large embeddings (e.g. 128) the frequency of every vector is higher
        # than the previous one which means that the gradients of later vectors are
        # ALWAYS higher than the previous ones
        grad_mean = np.abs(np.gradient(t1, axis=-1)).mean(axis=1)

        prev_grad = 0.0
        for grad in grad_mean:
            assert grad > prev_grad
            prev_grad = grad

    def test_timestep_defaults(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim)
        t2 = get_timestep_embedding(
            timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, max_period=10_000
        )

        assert torch.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_flip_sin_cos(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        t1 = torch.cat([t1[:, embedding_dim // 2 :], t1[:, : embedding_dim // 2]], dim=-1)

        t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)

        assert torch.allclose(t1.cpu(), t2.cpu(), 1e-3)

    def test_timestep_downscale_freq_shift(self):
        embedding_dim = 16
        timesteps = torch.arange(10)

        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0)
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1)

        # get cosine half (vectors that are wrapped into cosine)
        cosine_half = (t1 - t2)[:, embedding_dim // 2 :]

        # cosine needs to be negative
        assert (np.abs((cosine_half <= 0).numpy()) - 1).sum() < 1e-5

    def test_sinoid_embeddings_hardcoded(self):
        embedding_dim = 64
        timesteps = torch.arange(128)

        # standard unet, score_vde
        t1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1, flip_sin_to_cos=False)
        # glide, ldm
        t2 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0, flip_sin_to_cos=True)
        # grad-tts
        t3 = get_timestep_embedding(timesteps, embedding_dim, scale=1000)

        assert torch.allclose(
            t1[23:26, 47:50].flatten().cpu(),
            torch.tensor([0.9646, 0.9804, 0.9892, 0.9615, 0.9787, 0.9882, 0.9582, 0.9769, 0.9872]),
            1e-3,
        )
        assert torch.allclose(
            t2[23:26, 47:50].flatten().cpu(),
            torch.tensor([0.3019, 0.2280, 0.1716, 0.3146, 0.2377, 0.1790, 0.3272, 0.2474, 0.1864]),
            1e-3,
        )
        assert torch.allclose(
            t3[23:26, 47:50].flatten().cpu(),
            torch.tensor([-0.9801, -0.9464, -0.9349, -0.3952, 0.8887, -0.9709, 0.5299, -0.2853, -0.9927]),
            1e-3,
        )


class Upsample2DBlockTests(unittest.TestCase):
    def test_upsample_default(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=False)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.2173, -1.2079, -1.2079, 0.2952, 1.1254, 1.1254, 0.2952, 1.1254, 1.1254])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=True)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.7145, 1.3773, 0.3492, 0.8448, 1.0839, -0.3341, 0.5956, 0.1250, -0.4841])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_conv_out_dim(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 64, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.2703, 0.1656, -0.2538, -0.0553, -0.2984, 0.1044, 0.1155, 0.2579, 0.7755])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_upsample_with_transpose(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 32, 32)
        upsample = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        with torch.no_grad():
            upsampled = upsample(sample)

        assert upsampled.shape == (1, 32, 64, 64)
        output_slice = upsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.3028, -0.1582, 0.0071, 0.0350, -0.4799, -0.1139, 0.1056, -0.1153, -0.1046])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class Downsample2DBlockTests(unittest.TestCase):
    def test_downsample_default(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=False)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.0513, -0.3889, 0.0640, 0.0836, -0.5460, -0.0341, -0.0169, -0.6967, 0.1179])
        max_diff = (output_slice.flatten() - expected_slice).abs().sum().item()
        assert max_diff <= 1e-3
        # assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-1)

    def test_downsample_with_conv(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913],
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_pad1(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True, padding=1)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 32, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([0.9267, 0.5878, 0.3337, 1.2321, -0.1191, -0.3984, -0.7532, -0.0715, -0.3913])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_downsample_with_conv_out_dim(self):
        torch.manual_seed(0)
        sample = torch.randn(1, 32, 64, 64)
        downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
        with torch.no_grad():
            downsampled = downsample(sample)

        assert downsampled.shape == (1, 16, 32, 32)
        output_slice = downsampled[0, -1, -3:, -3:]
        expected_slice = torch.tensor([-0.6586, 0.5985, 0.0721, 0.1256, -0.1492, 0.4436, -0.2544, 0.5021, 1.1522])
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class AttentionBlockTests(unittest.TestCase):
    def test_attention_block_default(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        attentionBlock = AttentionBlock(
            channels=32,
            num_head_channels=1,
            rescale_output_factor=1.0,
            eps=1e-6,
            num_groups=32,
        ).to(torch_device)
        with torch.no_grad():
            attention_scores = attentionBlock(sample)

        assert attention_scores.shape == (1, 32, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-1.4975, -0.0038, -0.7847, -1.4567, 1.1220, -0.8962, -1.7394, 1.1319, -0.5427], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)


class SpatialTransformerTests(unittest.TestCase):
    def test_spatial_transformer_default(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        spatial_transformer_block = SpatialTransformer(
            in_channels=32,
            n_heads=1,
            d_head=32,
            dropout=0.0,
            context_dim=None,
        ).to(torch_device)
        with torch.no_grad():
            attention_scores = spatial_transformer_block(sample)

        assert attention_scores.shape == (1, 32, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-1.2447, -0.0137, -0.9559, -1.5223, 0.6991, -1.0126, -2.0974, 0.8921, -1.0201], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_context_dim(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 64, 64, 64).to(torch_device)
        spatial_transformer_block = SpatialTransformer(
            in_channels=64,
            n_heads=2,
            d_head=32,
            dropout=0.0,
            context_dim=64,
        ).to(torch_device)
        with torch.no_grad():
            context = torch.randn(1, 4, 64).to(torch_device)
            attention_scores = spatial_transformer_block(sample, context)

        assert attention_scores.shape == (1, 64, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-0.2555, -0.8877, -2.4739, -2.2251, 1.2714, 0.0807, -0.4161, -1.6408, -0.0471], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)

    def test_spatial_transformer_dropout(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sample = torch.randn(1, 32, 64, 64).to(torch_device)
        spatial_transformer_block = (
            SpatialTransformer(
                in_channels=32,
                n_heads=2,
                d_head=16,
                dropout=0.3,
                context_dim=None,
            )
            .to(torch_device)
            .eval()
        )
        with torch.no_grad():
            attention_scores = spatial_transformer_block(sample)

        assert attention_scores.shape == (1, 32, 64, 64)
        output_slice = attention_scores[0, -1, -3:, -3:]

        expected_slice = torch.tensor(
            [-1.2448, -0.0190, -0.9471, -1.5140, 0.7069, -1.0144, -2.1077, 0.9099, -1.0091], device=torch_device
        )
        assert torch.allclose(output_slice.flatten(), expected_slice, atol=1e-3)
