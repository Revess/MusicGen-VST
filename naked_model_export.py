from transformers import AutoProcessor, MusicgenForConditionalGeneration, PreTrainedModel, MusicgenForCausalLM
from torch import nn
import torch, os, glob, json, math
import onnxruntime as ort
import numpy as np

class NakedMusicGen(torch.nn.Module):
    # TODO: Find a way to also take care of melodic input encoding.

    def __init__(self, 
                text_encoder: PreTrainedModel, 
                enc_to_dec_proj: torch.nn.Linear, 
                decoder: MusicgenForCausalLM
            ):
        super().__init__()
        self.text_encoder = text_encoder
        self.enc_to_dec_proj = enc_to_dec_proj
        self.decoder = decoder
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                max_length: int,
                num_codebooks: int,
                decoder_start_token_id: int
            ):
        # Note to self, all inputs must be used
        # TODO: add generation params over here somewhere aswell like CFG topk and temperature.

        ## Input the data to the text encoder
        encoded = self.text_encoder(input_ids)

        # Apply the guidance scale or something
        # When the guidance scale is > 1 then we need to apply zeros to the mask and the last hidden state
        encoded.last_hidden_state = torch.concatenate([encoded.last_hidden_state, torch.zeros_like(encoded.last_hidden_state)], dim=0)
        attention_mask = torch.concatenate([attention_mask, torch.zeros_like(attention_mask)], dim=0)

        # Prepare for decoder inputs
        decoder_input_ids = torch.ones((input_ids.size(0) * num_codebooks, 1), dtype=torch.long) * decoder_start_token_id

        # Just remember this as when the user wants a really small sample 
        # Really small is max_len < (2 * num_codebooks - 1)
        # decoder_input_ids.reshape(bsz * num_codebooks, -1), decoder_ids_shifted.reshape(bsz * num_codebooks, -1)

        # Build delay pattern
        decoder_input_ids = decoder_input_ids.reshape(-1, num_codebooks, decoder_input_ids.shape[-1])
        bsz, num_codebooks, seq_len = decoder_input_ids.shape
        channel_codebooks = num_codebooks // 2 # We know it is 2 because we just care about the stereo version. TODO: impl a version for mono etc.
        decoder_ids_shifted = torch.ones((bsz, num_codebooks, max_length), dtype=torch.long) * -1

        # Now fill the shifted ids with the prompt
        for codebook in range(channel_codebooks):
            decoder_ids_shifted[:, 2 * codebook, codebook : seq_len + codebook] = decoder_input_ids[:, 2 * codebook]
            decoder_ids_shifted[:, 2 * codebook + 1, codebook : seq_len + codebook] = decoder_input_ids[:, 2 * codebook + 1]

        delay_pattern = torch.triu(
            torch.ones((channel_codebooks, max_length), dtype=torch.bool), diagonal = max_length - channel_codebooks + 1
        )
        delay_pattern = delay_pattern + torch.tril(torch.ones((channel_codebooks, max_length), dtype=torch.bool))
        delay_pattern = delay_pattern.repeat_interleave(2, dim=0)

        mask = ~delay_pattern.to(input_ids.device)
        decoder_input_ids = mask * decoder_ids_shifted + ~mask * decoder_start_token_id
        first_codebook_ids = decoder_input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = seq_len
        pattern_mask = decoder_input_ids.reshape(bsz * num_codebooks, -1)
        decoder_input_ids = decoder_input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)

        # Prepare inputs for generation fn
        # Apply mask
        decoder_input_ids = torch.where(pattern_mask[..., :decoder_input_ids.shape[-1]] == -1, decoder_input_ids, pattern_mask[..., :decoder_input_ids.shape[-1]])

        # Prep for the CGF
        decoder_input_ids = decoder_input_ids.repeat((2,1))

        # The translation layer
        encoder_hidden_states = encoded[0]
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]

        # The sampling
        decoder_input_ids = decoder_input_ids
        decoder_attention_mask = None
        encoder_hidden_states = encoder_hidden_states
        attention_mask = attention_mask
        decoder_inputs_embeds = None
        output_attentions = False
        output_hidden_states = False
        use_cache = True
        past_key_values = None
        return_dict = True
        labels = None
        head_mask = None

        # TODO: use the sampling loop similar to the code of Transformers7

        output = model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            head_mask=head_mask
        )

        return output

if __name__ == "__main__":
    folder = 'musicgen-stereo-small'
    os.makedirs(folder, exist_ok=True)
    # Undress the selected model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small")

    naked_model = NakedMusicGen(
        model.text_encoder, 
        model.enc_to_dec_proj, 
        model.decoder
    )

    naked_model.eval()

    with open(f'{folder}/config.json', 'r') as file_:
        model_config = json.load(file_)

    with open(f'{folder}/generation_config.json', 'r') as file_:
        generation_config = json.load(file_)

    max_length = 256 #This will be variable later
    outputs = processor.tokenizer(["80s pop track with bassy drums and synth"])
    input_ids, attention_mask = torch.tensor(outputs['input_ids']), torch.tensor(outputs['attention_mask'])



    # Start the export
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'}
    }

    dummy_input_ids = torch.randint(0, 100, ( 2, 12), dtype=torch.int64)
    dummy_attention_mask = torch.randint(0, 100, ( 2, 12), dtype=torch.int64)
    dummy_max_length = torch.tensor(256, dtype=torch.int64)
    dummy_num_codebooks = torch.tensor(8, dtype=torch.int64)
    dummy_decoder_start_token_id = torch.tensor(2048, dtype=torch.int64)

    torch.onnx.export(
        naked_model,
        (dummy_input_ids, dummy_attention_mask, dummy_max_length, dummy_num_codebooks, dummy_decoder_start_token_id),
        f"{folder}/naked_model.onnx",
        input_names=[
            'input_ids', 
            'attention_mask', 
            'max_length',
            'num_codebooks',
            'decoder_start_token_id'
        ],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17
    )

    # Test the exported model
    ort_session = ort.InferenceSession(f"{folder}/naked_model.onnx")

    # Prepare input data (assuming you already have input_ids and attention_mask as PyTorch tensors)
    input_ids_np = input_ids.detach().numpy()  # Convert to NumPy arrays if they're in PyTorch tensors
    attention_mask_np = attention_mask.detach().numpy()  # Convert to NumPy arrays if they're in PyTorch tensors

    # Run the model
    ort_inputs = {
        # 'input_ids': np.expand_dims(np.concatenate((input_ids_np, attention_mask_np), axis=0), 0),
        'input_ids': input_ids_np,
        'attention_mask': attention_mask_np,
        'max_length': np.array(256),
        'num_codebooks': np.array(model_config['decoder']['num_codebooks']),
        'decoder_start_token_id': np.array(generation_config['decoder_start_token_id'])
    }
    encoded = ort_session.run(None, ort_inputs)[0]
    print(encoded)


    exit()