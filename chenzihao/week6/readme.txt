------------------------------------------------------------------------------------

BERT

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

model config:

- vocab_size: 21128
- hidden_size: 768 
- intermediate_size: 3072
- max_position_embeddings: 512
- num_attention_heads: 12 (does not change parameters)
- num_hidden_layers: 12
- type_vocab_size: 2


- pooler shit**

we denote L as max_seq_len, V as vocan size, D as embedding dimension, N as number of layers

class LayerNorm(nn.Module):
    '''
    feature : [dim] 就是 embedding dim
    x : [B, seq_len, dim]
    y : [B, seq_len, dim]
    '''
    def __init__(self, feature, eps = 1e-6) -> None:
        # 这里其实feature没啥用，只是为了norm统一性
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature)) # scale factor, not uniform
        self.b_2 = nn.Parameter(torch.zeros(feature)) # bias
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

Total parameters: 102290312

Embedding :
Name: bert.embeddings.word_embeddings.weight, Shape: torch.Size([21128, 768])
Name: bert.embeddings.position_embeddings.weight, Shape: torch.Size([512, 768])
Name: bert.embeddings.token_type_embeddings.weight, Shape: torch.Size([2, 768])
Name: bert.embeddings.LayerNorm.weight, Shape: torch.Size([768])
Name: bert.embeddings.LayerNorm.bias, Shape: torch.Size([768])

V*D + 2*D + L*D + 2*D
= 16622592

Transformer Blocks:
N * (MHSA + LayerNorm + FFN + LayerNorm)
= N * ( 4 * (D*D+D) + 2*D + D * 4D + 4D + 4D*D + D + 2*D )
= 12 * (12*D*D + 13D)
= 85054464

Name: bert.encoder.layer.0.attention.self.query.weight, Shape: torch.Size([768, 768])
Name: bert.encoder.layer.0.attention.self.query.bias, Shape: torch.Size([768])
Name: bert.encoder.layer.0.attention.self.key.weight, Shape: torch.Size([768, 768])
Name: bert.encoder.layer.0.attention.self.key.bias, Shape: torch.Size([768])
Name: bert.encoder.layer.0.attention.self.value.weight, Shape: torch.Size([768, 768])
Name: bert.encoder.layer.0.attention.self.value.bias, Shape: torch.Size([768])

Name: bert.encoder.layer.0.attention.output.dense.weight, Shape: torch.Size([768, 768])
Name: bert.encoder.layer.0.attention.output.dense.bias, Shape: torch.Size([768])

Name: bert.encoder.layer.0.attention.output.LayerNorm.weight, Shape: torch.Size([768])
Name: bert.encoder.layer.0.attention.output.LayerNorm.bias, Shape: torch.Size([768])

Name: bert.encoder.layer.0.intermediate.dense.weight, Shape: torch.Size([3072, 768])
Name: bert.encoder.layer.0.intermediate.dense.bias, Shape: torch.Size([3072])
Name: bert.encoder.layer.0.output.dense.weight, Shape: torch.Size([768, 3072])
Name: bert.encoder.layer.0.output.dense.bias, Shape: torch.Size([768])

Name: bert.encoder.layer.0.output.LayerNorm.weight, Shape: torch.Size([768])
Name: bert.encoder.layer.0.output.LayerNorm.bias, Shape: torch.Size([768])

Final:
Name: cls.predictions.bias, Shape: torch.Size([21128])
Name: cls.predictions.transform.dense.weight, Shape: torch.Size([768, 768])
Name: cls.predictions.transform.dense.bias, Shape: torch.Size([768])
Name: cls.predictions.transform.LayerNorm.weight, Shape: torch.Size([768])
Name: cls.predictions.transform.LayerNorm.bias, Shape: torch.Size([768])
Name: cls.predictions.decoder.weight, Shape: torch.Size([21128, 768])
Name: cls.predictions.decoder.bias, Shape: torch.Size([21128])

// parameters
torch.Size([21128]) // vocab bias
torch.Size([768, 768]) // Pool_fc
torch.Size([768])
torch.Size([768]) // LayerNorm
torch.Size([768])

V + D*D + 3*D 

++++++++++++++++++s

Total:

V*D + L*D + 4 *D + 12 * (12*D*D + 13*D)  + V + D*D + 3*D

-----------------------------------------------------------------------------------------

Transformer :

check later

-----------------------------------------------------------------------------------------

GPT : 

check later
