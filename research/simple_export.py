
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch, os, glob, json, math, scipy, fire
from torch import nn
import numpy as np
import onnxruntime as ort
import inspect
from models import *
from exporters import *

def test_onnx(
        folder: str, 
        inputs: dict,
        max_len: int, 
        cfg: int, 
        temperature: float,
        top_k: int, 
        top_p: float, 
        sampling_rate: int
    ):
    ort_session = ort.InferenceSession(f"{folder}/text_encoder.onnx")
    ort_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    input_ids_np = inputs['input_ids'].detach().numpy()
    attention_mask_np = inputs['attention_mask'].detach().numpy()

    # Run the model
    ort_inputs = {
        'input_ids': input_ids_np,
        'attention_mask': attention_mask_np,
        'cfg': np.array([cfg], dtype=np.int64)
    }
    encoded, attention_mask = ort_session.run(None, ort_inputs)

    ort_session = ort.InferenceSession(f"{folder}/pre_loop.onnx")
    ort_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    batch_size = torch.tensor(1, dtype=torch.int64).detach().numpy()
    max_length = torch.tensor(max_len, dtype=torch.int64).detach().numpy()

    # Run the model
    ort_inputs = {
        'batch_size': batch_size,
        'max_length': max_length
    }
    decoder_input_ids, decoder_delay_pattern_mask = ort_session.run(None, ort_inputs)

    ort_session = ort.InferenceSession(f"{folder}/sampler.onnx")
    ort_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for i in range(max_len-1):
        print(i, end='\r')
        # Run the model
        ort_inputs = {
            'decoder_input_ids.1': decoder_input_ids, 
            'attention_mask': attention_mask.astype(np.int64), 
            'encoder_hidden_states.1': encoded, 
            'delay_pattern_mask': decoder_delay_pattern_mask, 
            'cfg': np.array([cfg], dtype=np.int64), 
            'temperature': np.array([temperature], dtype=np.float32), 
            'topk': np.array([top_k], dtype=np.int64), 
            'topp': np.array([top_p], dtype=np.float32)
        }

        decoder_input_ids, _ = ort_session.run(None, ort_inputs)

    ort_session = ort.InferenceSession(f"{folder}/audio_token_decoder.onnx")
    ort_session.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Run the model
    ort_inputs = {
        'output_ids': decoder_input_ids, # We either need to remove the first tokens or add tokens to the decoder delay_pattern mask, check og for inspo
        'decoder_delay_pattern_mask': decoder_delay_pattern_mask,
        'pad_token_id': np.array([2048], dtype=np.int64)
    }

    output_values = ort_session.run(None, ort_inputs)[0]

    scipy.io.wavfile.write("onnx_res.wav", rate=sampling_rate, data=output_values[0].detach().numpy().T)

def export_onnx(
        folder,
        model: MusicgenForConditionalGeneration,
        processor: AutoProcessor,
        text_encoder_wrapper: TextEncoderWrapper, 
        pre_loop: PreLoop, 
        sample: Sample, 
        audio_decoder_wrapper: DecodeAudioWrapper
    ):
    export_configs(processor, model, folder)
    export_text_encoder(text_encoder_wrapper, folder)
    export_preloop(pre_loop, folder)
    export_sampler(sample, folder)
    export_audio_decoder(audio_decoder_wrapper, folder)

def test_torch(
        inputs: dict, 
        text_encoder_wrapper: TextEncoderWrapper, 
        pre_loop: PreLoop, 
        sample: Sample, 
        audio_decoder_wrapper: DecodeAudioWrapper, 
        max_len: int, 
        cfg: int, 
        temperature: float,
        top_k: int, 
        top_p: float, 
        sampling_rate: int
    ):
    with torch.no_grad():
        encoded, attention_mask = text_encoder_wrapper(inputs['input_ids'], inputs['attention_mask'], torch.tensor([cfg], dtype=torch.int64))

        decoder_input_ids, decoder_delay_pattern_mask = pre_loop(torch.tensor(1, dtype=torch.int64), None, None, torch.tensor(max_len, dtype=torch.int64))

        for i in range(max_len-1):
            print(i, end='\r')
        # for i in range(3):
            decoder_input_ids, _ = sample(
                decoder_input_ids, 
                attention_mask, 
                encoded, 
                decoder_delay_pattern_mask, 
                torch.tensor([cfg], dtype=torch.int64), 
                torch.tensor([temperature], dtype=torch.float32), 
                torch.tensor([top_k], dtype=torch.int64), 
                torch.tensor([top_p], dtype=torch.float32)
            )

        output_values = audio_decoder_wrapper(decoder_input_ids, decoder_delay_pattern_mask, torch.tensor([2048], dtype=torch.int64))

    scipy.io.wavfile.write("torch_output.wav", rate=sampling_rate, data=output_values[0].detach().numpy().T)

def main(
        export_onnx_ = False,
        test_onnx_ = False,
        test_torch_script = False,
        do_model_sample = False,
        sampling_rate = 32000,
        cfg = 5,
        temperature = 0.7,
        top_k = 500,
        top_p = 0.0,
        max_len = 256,
        folder = './musicgen-stereo-small'
    ):
    os.makedirs(folder, exist_ok=True)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-small")

    if do_model_sample or export_onnx or test_torch_script:
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small")
        model.eval()

    inputs = processor(
        text=["80s pop track with bassy drums and synth asd asd asd"],
        padding=True,
        return_tensors="pt",
    )
    if do_model_sample:
        res = model.generate(**inputs, do_sample=True, guidance_scale=cfg, max_new_tokens=max_len, temperature=temperature, top_k=top_k, top_p=top_p)

        scipy.io.wavfile.write("org_output.wav", rate=sampling_rate, data=res[0].detach().numpy().T)

    # Init the classes
    text_encoder_wrapper = TextEncoderWrapper(model.text_encoder)
    pre_loop = PreLoop(model.config.decoder.num_codebooks, model.config.decoder.audio_channels)
    sample = Sample(model.decoder, model.enc_to_dec_proj)
    audio_decoder_wrapper = DecodeAudioWrapper(model.audio_encoder)

    if test_torch_script:
        test_torch(
            inputs,
            text_encoder_wrapper,
            pre_loop,
            sample,
            audio_decoder_wrapper,
            max_len,
            cfg,
            temperature,
            top_k,
            top_p,
            sampling_rate
        )

    if export_onnx_:
        export_onnx(
            folder,
            model,
            processor,
            text_encoder_wrapper,
            pre_loop,
            sample,
            audio_decoder_wrapper
        )

    if test_onnx_:
        test_onnx(
            folder,
            inputs,
            max_len,
            cfg,
            temperature,
            top_k,
            top_p,
            sampling_rate
        )

if __name__ == "__main__":
    fire.Fire()