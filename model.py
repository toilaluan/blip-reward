import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Optional, Tuple, Union
import torch
from peft import LoraConfig, TaskType, get_peft_model

class QEncoder(Blip2ForConditionalGeneration):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        n_query_tokens = query_tokens.shape[1]
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)], dim=1
        )
        outputs = self.language_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
        ).last_hidden_state
        outputs = outputs[:, :n_query_tokens, :]
        outputs = outputs.mean(dim=1)
        return outputs


class QformerAesthetic(nn.Module):
    def __init__(self, pretrained_qformer=True, config=None):
        super(QformerAesthetic, self).__init__()
        if pretrained_qformer:
            self.qencoder = QEncoder.from_pretrained("ethzanalytics/blip2-flan-t5-xl-sharded", torch_dtype=torch.bfloat16)
        else:
            self.qencoder = QEncoder(config)
        self.qencoder.language_model.decoder = None
        for param in self.qencoder.parameters():
            param.requires_grad = False
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.qencoder.language_model = get_peft_model(self.qencoder.language_model, peft_config)
        self.qencoder.language_model.print_trainable_parameters()
        self.out_dim = self.qencoder.config.text_config.d_model
        self.head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim // 4),
            nn.Dropout(0.1),
            nn.Linear(self.out_dim // 4, 1)
        )
        # initial MLP param
        for name, param in self.head.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.out_dim+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
    
    def forward(self, pixel_values, input_ids, attention_mask=None, **kwargs):
        hidden_embed = self.qencoder(pixel_values, input_ids, attention_mask)
        out = self.head(hidden_embed)
        return out

if __name__ == "__main__":
    
    from data.pair_dataset import PairDataset
    from torch.utils.data import DataLoader
    dataset = PairDataset("/workspace/t5-reward/dataset/validation")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    inputs = next(iter(dataloader))
    print(inputs.keys())
    print(inputs['img_better']['input_ids'].shape)
    print(inputs['img_better']['pixel_values'].shape)
    model = QformerAesthetic()
    out = model(inputs['img_better']['pixel_values'], inputs['img_better']['input_ids'])
    print(out.shape)

    
    