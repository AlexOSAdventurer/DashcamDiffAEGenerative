import torch
import torch.nn.functional as F
from torch import nn
import math

# We assume channels can be cleanly divided by 16 if greater than 16
def UNetLayerNormalization(channels, block_list_base_channels):
    return torch.nn.GroupNorm(block_list_base_channels, channels)

def Upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

def Downsample(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)

def dropout():
    return torch.nn.Dropout(p=0.1)

def skip_connection(input_channels, output_channels):
    return torch.nn.Identity() if (input_channels == output_channels) else conv_fc(input_channels, output_channels)

def conv_nd(input_channels, output_channels, stride=1, padding=1):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=padding, stride=stride)

def conv_fc(input_channels, output_channels):
    return torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

def activation():
    return torch.nn.SiLU()

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class UNetBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, block_list_base_channels, latent_space):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_list_base_channels = block_list_base_channels
        self.latent_space = latent_space
        self.block_list = torch.nn.Sequential(
            UNetLayerNormalization(input_channels, block_list_base_channels),
            activation(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels, block_list_base_channels),
            activation(),
            dropout(),
            conv_nd(output_channels, output_channels)
        )
        self.skip_connection = skip_connection(input_channels, output_channels)

    def forward(self, x):
        return self.block_list(x) + self.skip_connection(x)

class UNetBlockConditional(torch.nn.Module):
    def __init__(self, input_channels, output_channels, block_list_base_channels, latent_space):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_list_base_channels = block_list_base_channels
        self.semantic_latent_channels = latent_space
        self.timestep_dims = latent_space
        self.semantic_affine = torch.nn.Sequential(
            activation(),
            torch.nn.Linear(self.semantic_latent_channels, output_channels)
        )
        self.timestep_mlp = torch.nn.Sequential(
            activation(),
            torch.nn.Linear(self.timestep_dims, 2 * output_channels)
        )
        self.block_list_1 = torch.nn.Sequential(
            UNetLayerNormalization(input_channels, block_list_base_channels),
            activation(),
            conv_nd(input_channels, output_channels),
            UNetLayerNormalization(output_channels, block_list_base_channels))
        self.block_list_2 = torch.nn.Sequential(
            activation(),
            dropout(),
            conv_nd(output_channels, output_channels)
        )
        self.skip_connection = skip_connection(input_channels, output_channels)

    def forward(self, x, t, z_sem):
        mid_point = self.block_list_1(x)
        t_emb = self.timestep_mlp(timestep_embedding(t, self.timestep_dims))
        t_s, t_b = torch.chunk(torch.unsqueeze(torch.unsqueeze(t_emb, dim=-1), dim=-1), 2, dim=1)
        z_sem_scaling = torch.unsqueeze(torch.unsqueeze(self.semantic_affine(z_sem), dim=-1), dim=-1)
        conditioned = t_s * mid_point
        conditioned = conditioned + t_b
        conditioned = z_sem_scaling * conditioned
        final_point = self.block_list_2(conditioned)
        skipped = self.skip_connection(x)
        return final_point + skipped

class DownsamplerBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, factor=2):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.factor = factor
        self.conv = conv_nd(input_channels, output_channels, stride=factor)

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(torch.nn.Module):
    def __init__(self, channels, block_list_base_channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.block_list_base_channels = block_list_base_channels
        self.num_heads = num_heads

        self.norm = UNetLayerNormalization(channels, block_list_base_channels)
        self.qkv = conv_fc(channels, channels * 3)
        self.attention = QKVAttentionDiffAE(self.num_heads)

        self.proj_out = zero_module(conv_fc(channels, channels))

    def forward(self, x):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, c * 3, -1)
        h = self.attention(qkv).reshape(b, c, *spatial)
        h = self.proj_out(h)
        return x + h

class QKVAttentionDiffAE(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        batch_size, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(batch_size * self.num_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch)) # Scale is ch^0.25, not ch^0.5. Not sure why, probably has to do with some literature I am not familiar with
        # My understanding is that this stuff is verbatim from Attention is all you need
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(batch_size, -1, length)

class UNetBlockGroup(torch.nn.Module):
    def __init__(self, input_channels, output_channels, block_list_base_channels, latent_space, num_res_blocks, upsample=False, upsample_target_channels = None, downsample=False, conditional=False, num_heads=None, conc=False, conc_channels=0):
        super().__init__()
        block_type = UNetBlockConditional if conditional else UNetBlock
        self.block_list = torch.nn.ModuleList([])
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.block_list_base_channels = block_list_base_channels
        self.latent_space = latent_space
        self.num_res_blocks = num_res_blocks
        self.upsample = upsample
        self.upsample_target_channels = upsample_target_channels
        self.downsample = downsample
        self.conditional = conditional
        self.conc = conc
        self.conc_channels = conc_channels
        self.block_list.append(block_type(input_channels + conc_channels, output_channels, block_list_base_channels, latent_space))
        for i in range(self.num_res_blocks - 1):
            self.block_list.append(block_type(output_channels + conc_channels, output_channels, block_list_base_channels, latent_space))
            if (num_heads is not None):
                self.block_list.append(AttentionBlock(output_channels, block_list_base_channels, num_heads))
        assert (not (self.upsample and self.downsample)), "You can't be both upsampling and downsampling!"
        if upsample:
            self.upsample_conv = conv_nd(self.output_channels, self.upsample_target_channels)
        if downsample:
            self.downsampler = DownsamplerBlock(self.output_channels, self.output_channels)

    def forward(self, x, t = None, z_sem = None, conc_x = None, return_unscaled_output=False):
        if self.conditional:
            for module in self.block_list:
                if isinstance(module, AttentionBlock):
                    x = module(x)
                else:
                    if self.conc:
                        x = torch.cat([x,conc_x], dim=1)
                    x = module(x, t, z_sem)
        else:
            for module in self.block_list:
                if isinstance(module, AttentionBlock):
                    x = module(x)
                else:
                    if self.conc:
                        x = torch.cat([x,conc_x], dim=1)
                    x = module(x)

        if self.upsample:
            res = self.upsample_conv(Upsample(x))
            if return_unscaled_output:
                x = (res, x)
            else:
                x = res
        if self.downsample:
            #res = Downsample(x)
            res = self.downsampler(x)
            if return_unscaled_output:
                x = (res, x)
            else:
                x = res
        if (return_unscaled_output and not (self.upsample or self.downsample)):
            return (x,x)
        return x

class UNetEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_list = torch.nn.ModuleList([])
        self.input_size = config["input_size"]
        self.input_channels = config["input_channels"]
        self.block_list_base_channels = config["encoder_model"]["block_list_base_channels"]
        self.block_list_channels_mult = config["encoder_model"]["block_list_channels_mult"]
        self.latent_space = config["encoder_model"]["latent_space"]
        self.num_res_blocks = config["encoder_model"]["num_res_blocks"]
        self.attention_heads = config["encoder_model"]["attention_heads"]
        self.attention_resolutions = config["encoder_model"]["attention_resolutions"]

        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )

        current_resolution = self.input_size
        previous_channels = self.block_list_base_channels
        for entry in self.block_list_channels_mult:
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            self.block_list.append(UNetBlockGroup(previous_channels, current_channels, self.block_list_base_channels, None, self.num_res_blocks, upsample=False, downsample=True, conditional=False, num_heads=current_heads))
            previous_channels = current_channels
            current_resolution = current_resolution // 2

        self.final_output_module = torch.nn.Sequential(
            UNetLayerNormalization(previous_channels, self.block_list_base_channels),
            activation(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            conv_fc(previous_channels, self.latent_space),
            torch.nn.Flatten()
        )

    def forward(self, x):
        x = self.firstSide(x)
        for module in self.block_list:
            x = module(x)
        x = self.final_output_module(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.input_size = 64
        #self.input_channels = 3
        #self.block_list_base_channels = 32
        #self.block_list_channels_mult = [1, 2, 2, 4]
        #self.num_res_blocks = 2
        #self.attention_heads = [4]
        #self.attention_resolutions = [8]
        self.config = config
        self.input_size = config["input_size"]
        self.input_channels = config["input_channels"]
        self.block_list_base_channels = config["noise_model"]["block_list_base_channels"]
        self.block_list_channels_mult = config["noise_model"]["block_list_channels_mult"]
        self.latent_space = config["noise_model"]["latent_space"]
        self.num_res_blocks = config["noise_model"]["num_res_blocks"]
        self.attention_heads = config["noise_model"]["attention_heads"]
        self.attention_resolutions = config["noise_model"]["attention_resolutions"]
        self.stable_diffusion_preload = ("stable_diffusion_preload_model_source" in config["noise_model"])
        if (self.stable_diffusion_preload):
            self.stable_diffusion_source = config["noise_model"]["stable_diffusion_preload_model_source"]

        self.firstSide = torch.nn.Sequential(
            conv_nd(self.input_channels, self.block_list_base_channels)
        )
        self.lastSide = torch.nn.Sequential(
            conv_nd(self.block_list_base_channels, self.input_channels)
        )
        self.downSide = torch.nn.ModuleList([])
        self.upSide = torch.nn.ModuleList([])

        current_resolution = self.input_size
        previous_channels = self.block_list_base_channels
        for i, entry in enumerate(self.block_list_channels_mult):
            last_block = (i == (len(self.block_list_channels_mult) - 1))
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            self.downSide.append(UNetBlockGroup(previous_channels, current_channels, self.block_list_base_channels, self.latent_space, self.num_res_blocks, upsample=False, downsample=(not last_block), conditional=True, num_heads=current_heads))
            previous_channels = current_channels
            if not last_block:
                current_resolution = current_resolution // 2
        current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
        self.middleModule = UNetBlockGroup(previous_channels, previous_channels, self.block_list_base_channels, self.latent_space, self.num_res_blocks, upsample=False, upsample_target_channels = previous_channels, downsample=False, conditional=True, num_heads=current_heads)
        current_resolution = current_resolution * 2
        block_list_channels_mult_reversed = self.block_list_channels_mult[::-1]
        for i, entry in enumerate(block_list_channels_mult_reversed):
            print(i, entry)
            last_block = (i == (len(block_list_channels_mult_reversed) - 1))
            current_channels = self.block_list_base_channels * entry
            current_heads = self.attention_heads[self.attention_resolutions.index(current_resolution)] if current_resolution in self.attention_resolutions else None
            #next_channels = self.block_list_base_channels * next_entry
            #self.upSide.append(UNetBlockGroup(previous_channels, current_channels, self.num_res_blocks, upsample=True, upsample_target_channels = next_channels, downsample=False, conditional=True, num_heads=current_heads, conc=True, conc_channels=))
            self.upSide.append(UNetBlockGroup(previous_channels, current_channels, self.block_list_base_channels, self.latent_space, self.num_res_blocks, upsample=(not last_block), upsample_target_channels = current_channels, downsample=False, conditional=True, num_heads=current_heads, conc=True, conc_channels=current_channels))
            if (not last_block):
                current_resolution = current_resolution * 2
            previous_channels = current_channels
            #self.upSide.append(UNetBlockGroup(previous_channels, self.block_list_base_channels, self.num_res_blocks, upsample=False, downsample=False, conditional=True, num_heads=current_heads, conc=True, conc_channels=))

        if (self.stable_diffusion_preload):
            print("Loading stable diffusion data...")
            diffusion_source = torch.load(self.stable_diffusion_source)
            new_state_dict = self.state_dict()
            total_sourced_parameters = 0
            # Down Blocks
            for i in range(len(self.block_list_channels_mult)):
                source_keys = [k for k in diffusion_source.keys() if (f"down_blocks.{i}" in k)]
                skip_connection = False
                for k in source_keys:
                    skip_connection = skip_connection or ("conv_shortcut" in k)

                for j in range(self.num_res_blocks):
                    assert (f"downSide.{i}.block_list.{j}.block_list_1.0.weight" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_1.0.bias" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_1.2.weight" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_1.2.bias" in new_state_dict)

                    assert (f"downSide.{i}.block_list.{j}.block_list_1.3.weight" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_1.3.bias" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_2.2.weight" in new_state_dict)
                    assert (f"downSide.{i}.block_list.{j}.block_list_2.2.bias" in new_state_dict)

                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.0.weight"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.norm1.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.norm1.weight"].flatten())
                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.0.bias"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.norm1.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.norm1.bias"].flatten())

                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.2.weight"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.conv1.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.conv1.weight"].flatten())
                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.2.bias"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.conv1.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.conv1.bias"].flatten())

                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.3.weight"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.norm2.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.norm2.weight"].flatten())
                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_1.3.bias"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.norm2.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.norm2.bias"].flatten())

                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_2.2.weight"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.conv2.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.conv2.weight"].flatten())
                    new_state_dict[f"downSide.{i}.block_list.{j}.block_list_2.2.bias"] = diffusion_source[f"down_blocks.{i}.resnets.{j}.conv2.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.{j}.conv2.bias"].flatten())

                if skip_connection:
                    assert(f"downSide.{i}.block_list.0.skip_connection.weight" in new_state_dict)
                    assert(f"downSide.{i}.block_list.0.skip_connection.bias" in new_state_dict)
                    new_state_dict[f"downSide.{i}.block_list.0.skip_connection.weight"] = diffusion_source[f"down_blocks.{i}.resnets.0.conv_shortcut.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.0.conv_shortcut.weight"].flatten())
                    new_state_dict[f"downSide.{i}.block_list.0.skip_connection.bias"] = diffusion_source[f"down_blocks.{i}.resnets.0.conv_shortcut.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.resnets.0.conv_shortcut.bias"].flatten())


                # Downsample
                if (f"down_blocks.{i}.downsamplers.0.conv.weight" in diffusion_source):
                    assert(f"downSide.{i}.downsampler.conv.weight" in new_state_dict)
                    assert(f"downSide.{i}.downsampler.conv.bias" in new_state_dict)
                    new_state_dict[f"downSide.{i}.downsampler.conv.weight"] = diffusion_source[f"down_blocks.{i}.downsamplers.0.conv.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.downsamplers.0.conv.weight"].flatten())
                    new_state_dict[f"downSide.{i}.downsampler.conv.bias"] = diffusion_source[f"down_blocks.{i}.downsamplers.0.conv.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"down_blocks.{i}.downsamplers.0.conv.bias"].flatten())

            # Middle Block
            for j in range(self.num_res_blocks):
                assert (f"middleModule.block_list.{j}.block_list_1.0.weight" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_1.0.bias" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_1.2.weight" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_1.2.bias" in new_state_dict)

                assert (f"middleModule.block_list.{j}.block_list_1.3.weight" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_1.3.bias" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_2.2.weight" in new_state_dict)
                assert (f"middleModule.block_list.{j}.block_list_2.2.bias" in new_state_dict)

                new_state_dict[f"middleModule.block_list.{j}.block_list_1.0.weight"] = diffusion_source[f"mid_block.resnets.{j}.norm1.weight"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.norm1.weight"].flatten())
                new_state_dict[f"middleModule.block_list.{j}.block_list_1.0.bias"] = diffusion_source[f"mid_block.resnets.{j}.norm1.bias"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.norm1.bias"].flatten())

                new_state_dict[f"middleModule.block_list.{j}.block_list_1.2.weight"] = diffusion_source[f"mid_block.resnets.{j}.conv1.weight"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.conv1.weight"].flatten())
                new_state_dict[f"middleModule.block_list.{j}.block_list_1.2.bias"] = diffusion_source[f"mid_block.resnets.{j}.conv1.bias"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.conv1.bias"].flatten())

                new_state_dict[f"middleModule.block_list.{j}.block_list_1.3.weight"] = diffusion_source[f"mid_block.resnets.{j}.norm2.weight"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.norm2.weight"].flatten())
                new_state_dict[f"middleModule.block_list.{j}.block_list_1.3.bias"] = diffusion_source[f"mid_block.resnets.{j}.norm2.bias"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.norm2.bias"].flatten())

                new_state_dict[f"middleModule.block_list.{j}.block_list_2.2.weight"] = diffusion_source[f"mid_block.resnets.{j}.conv2.weight"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.conv2.weight"].flatten())
                new_state_dict[f"middleModule.block_list.{j}.block_list_2.2.bias"] = diffusion_source[f"mid_block.resnets.{j}.conv2.bias"]
                total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"mid_block.resnets.{j}.conv2.bias"].flatten())

            # Up Blocks
            for i in range(len(self.block_list_channels_mult)):
                print(i)
                source_keys = [k for k in diffusion_source.keys() if (f"up_blocks.{i}" in k)]
                skip_connection = False
                for k in source_keys:
                    skip_connection = skip_connection or ("conv_shortcut" in k)

                for j in range(self.num_res_blocks):
                    assert (f"upSide.{i}.block_list.{j}.block_list_1.0.weight" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_1.0.bias" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_1.2.weight" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_1.2.bias" in new_state_dict)

                    assert (f"upSide.{i}.block_list.{j}.block_list_1.3.weight" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_1.3.bias" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_2.2.weight" in new_state_dict)
                    assert (f"upSide.{i}.block_list.{j}.block_list_2.2.bias" in new_state_dict)

                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.0.weight"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.norm1.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.norm1.weight"].flatten())
                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.0.bias"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.norm1.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.norm1.bias"].flatten())

                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.2.weight"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.conv1.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.conv1.weight"].flatten())
                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.2.bias"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.conv1.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.conv1.bias"].flatten())

                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.3.weight"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.norm2.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.norm2.weight"].flatten())
                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_1.3.bias"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.norm2.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.norm2.bias"].flatten())

                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_2.2.weight"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.conv2.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.conv2.weight"].flatten())
                    new_state_dict[f"upSide.{i}.block_list.{j}.block_list_2.2.bias"] = diffusion_source[f"up_blocks.{i}.resnets.{j}.conv2.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.{j}.conv2.bias"].flatten())

                if skip_connection:
                    assert(f"upSide.{i}.block_list.0.skip_connection.weight" in new_state_dict)
                    assert(f"upSide.{i}.block_list.0.skip_connection.bias" in new_state_dict)
                    new_state_dict[f"upSide.{i}.block_list.0.skip_connection.weight"] = diffusion_source[f"up_blocks.{i}.resnets.0.conv_shortcut.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.0.conv_shortcut.weight"].flatten())
                    new_state_dict[f"upSide.{i}.block_list.0.skip_connection.bias"] = diffusion_source[f"up_blocks.{i}.resnets.0.conv_shortcut.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.resnets.0.conv_shortcut.bias"].flatten())
                # Upsample
                if (f"up_blocks.{i}.upsamplers.0.conv.weight" in diffusion_source):
                    assert(f"upSide.{i}.upsample_conv.weight" in new_state_dict)
                    assert(f"upSide.{i}.upsample_conv.bias" in new_state_dict)
                    new_state_dict[f"upSide.{i}.upsample_conv.weight"] = diffusion_source[f"up_blocks.{i}.upsamplers.0.conv.weight"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.upsamplers.0.conv.weight"].flatten())
                    new_state_dict[f"upSide.{i}.upsample_conv.bias"] = diffusion_source[f"up_blocks.{i}.upsamplers.0.conv.bias"]
                    total_sourced_parameters = total_sourced_parameters + len(diffusion_source[f"up_blocks.{i}.upsamplers.0.conv.bias"].flatten())

            print(f"Total sourced parameters: {total_sourced_parameters}")
            self.load_state_dict(new_state_dict)



    def forward(self, x, t=None, cond=None):
        x = self.firstSide(x)
        intermediate_outputs = []
        for module in self.downSide:
            x = module(x, t, cond, return_unscaled_output=True)
            intermediate_outputs.append(x[1])
            x = x[0]
        x = self.middleModule(x, t, cond)
        for module in self.upSide:
            conc_x = intermediate_outputs.pop()
            #x = torch.cat([x, intermediate_outputs.pop()], dim=1)
            x = module(x, t, cond, conc_x = conc_x)
        x = self.lastSide(x)
        return x

class Autoencoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = UNet(config)
        self.encoder = UNetEncoder(config)

    def forward(self, x, t, cond):
        return self.unet(x, t, cond)

class DDIM(torch.nn.Module):
    def __init__(self, latent_space):
        super().__init__()
        self.mlp_skip_net = MLPSkipNet(latent_space)

    def forward(self, x, t):
        return self.mlp_skip_net(x, t)

class MLPSkipNet(torch.nn.Module):
    def __init__(self, latent_space):
        super().__init__()
        self.num_channels = latent_space
        self.num_hidden_channels = 2048
        self.num_time_layers = 2
        self.num_time_emb_channels = 64
        self.num_condition_bias = 1
        self.num_regular_layers = 10

        layers = []
        in_channels = self.num_time_emb_channels
        out_channels = self.num_channels
        for i in range(self.num_time_layers):
            layers.append(nn.Linear(in_channels, out_channels))
            if (i != (self.num_time_layers - 1)):
                layers.append(activation())
            in_channels = out_channels
        self.time_embed = torch.nn.Sequential(*layers)

        self.regular_layers = torch.nn.ModuleList([])
        in_channels = self.num_channels
        out_channels = self.num_hidden_channels
        for i in range(self.num_regular_layers):
            if (i == (self.num_regular_layers - 1)):
                self.regular_layers.append(MLPBlock(in_channels, self.num_channels, norm=False, cond=False, act=False))
            else:
                self.regular_layers.append(MLPBlock(in_channels, out_channels, norm=True, cond=True, act=True, cond_channels=self.num_channels, cond_bias=self.num_condition_bias))
            in_channels = out_channels + self.num_channels

    def forward(self, x, t):
        t = timestep_embedding(t, self.num_time_emb_channels)
        t_cond = self.time_embed(t)
        h = x
        for i in range(self.num_regular_layers):
            if (i != 0):
                h = torch.cat([h, x], dim=1)
            h = self.regular_layers[i].forward(h, cond=t_cond)
        return h

class MLPBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm, cond, act, cond_channels=None, cond_bias=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = torch.nn.LayerNorm(out_channels) if norm else torch.nn.Identity()
        self.act = activation() if act else torch.nn.Identity()
        self.use_cond = cond
        self.cond_channels = cond_channels
        self.cond_bias = cond_bias

        self.linear = torch.nn.Linear(self.in_channels, self.out_channels)
        if self.use_cond:
            self.linear_emb = torch.nn.Linear(self.cond_channels, self.out_channels)
            self.cond_layers = torch.nn.Sequential(self.act, self.linear_emb)

    def forward(self, x, cond=None):
        x = self.linear(x)
        if (self.use_cond):
            cond = self.cond_layers(cond)
            x = x * (self.cond_bias + cond)
        x = self.norm(x)
        x = self.act(x)
        return x
