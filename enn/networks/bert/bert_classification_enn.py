import torch
import jax
import jax.numpy as jnp
import haiku as hk
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, BertModel
import enn
from enn.networks.bert.base import BertInput
from enn.networks.bert.bert import make_bert_enn  # Your Haiku-based ENN model
import cls_heads

# Convert PyTorch tensor to JAX-compatible format
def torch_to_jax(torch_tensor):
    return jnp.array(torch_tensor.detach().cpu().numpy())

class BertConfigCustom:
    def __init__(self, transformers_config):
        self.vocab_size = transformers_config.vocab_size
        self.hidden_size = transformers_config.hidden_size
        self.num_hidden_layers = transformers_config.num_hidden_layers
        self.num_attention_heads = transformers_config.num_attention_heads
        self.intermediate_size = transformers_config.intermediate_size
        self.max_position_embeddings = transformers_config.max_position_embeddings
        self.type_vocab_size = transformers_config.type_vocab_size
        self.initializer_range = transformers_config.initializer_range
        self.hidden_dropout_prob = transformers_config.hidden_dropout_prob
        self.attention_probs_dropout_prob = (
            transformers_config.attention_probs_dropout_prob
        )

def tokenize_input(
    text: str, tokenizer: BertTokenizer, max_length: int = 512
) -> BertInput:
    """Tokenizes input text and converts it into BertInput format."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",  # Convert to NumPy arrays
    )

    return BertInput(
        token_ids=encoding["input_ids"],
        segment_ids=encoding["token_type_ids"],
        input_mask=encoding["attention_mask"],
    )

def embedding_params_fix(haiku_params, jax_bert_params):
    """Manually convert embeddings parameters from jax_bert_params to Haiku format."""

    haiku_params["BERT/word_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.word_embeddings.weight"
    ]
    haiku_params["BERT/token_type_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.token_type_embeddings.weight"
    ]
    haiku_params["BERT/position_embeddings"]["embeddings"] = jax_bert_params[
        "embeddings.position_embeddings.weight"
    ]
    haiku_params["BERT/embeddings_ln"]["scale"] = jax_bert_params[
        "embeddings.LayerNorm.weight"
    ]
    haiku_params["BERT/embeddings_ln"]["offset"] = jax_bert_params[
        "embeddings.LayerNorm.bias"
    ]

    return haiku_params

def layer_params_fix(haiku_params, jax_bert_params, layer_id):
    conversion_dict = {
        "query_": "attention.self.query",
        "keys_": "attention.self.key",
        "values_": "attention.self.value",
        "attention_output_dense_": "attention.output.dense",
        "attention_output_ln_": "attention.output.LayerNorm",
        "intermediate_output_": "intermediate.dense",
        "layer_output_": "output.dense",
        "layer_output_ln_": "output.LayerNorm",
    }

    for name, hf_name in conversion_dict.items():
        haiku_name = f"BERT/~_bert_layer/{name}{layer_id}"
        jax_name = f"encoder.layer.{layer_id}.{hf_name}"

        # Check if weight needs transposition
        if (
            "dense" in hf_name
            or "query" in hf_name
            or "key" in hf_name
            or "value" in hf_name
        ):
            haiku_params[haiku_name]["w"] = jax_bert_params[
                jax_name + ".weight"
            ].T  # Transpose
        else:
            haiku_params[haiku_name]["w"] = jax_bert_params[jax_name + ".weight"]

        haiku_params[haiku_name]["b"] = jax_bert_params[jax_name + ".bias"]

    return haiku_params


# def layer_params_fix(haiku_params, jax_bert_params, layer_id):

#     conversion_dict = {
#         "query_": "attention.self.query",
#         "keys_": "attention.self.key",
#         "values_": "attention.self.value",
#         "attention_output_dense_": "attention.output.dense",
#         "attention_output_ln_": "attention.output.LayerNorm",
#         "intermediate_output_": "intermediate.dense",
#         "layer_output_": "output.dense",
#         "layer_output_ln_": "output.LayerNorm",
#     }

#     for name in conversion_dict.keys():
#         haiku_name = f"BERT/~_bert_layer/" + name + str(layer_id)
#         jax_name = f"encoder.layer.{layer_id}.{conversion_dict[name]}"

#         haiku_params[haiku_name]["w"] = jax_bert_params[jax_name + ".weight"]
#         haiku_params[haiku_name]["b"] = jax_bert_params[jax_name + ".bias"]

#     return haiku_params

def mlm_params_fix(haiku_params, jax_bert_params):
    haiku_params["BERT/mlm_dense"]['w'] = jax_bert_params["cls.predictions.transform.dense.weight"]
    haiku_params["BERT/mlm_dense"]['b'] = jax_bert_params["cls.predictions.transform.dense.bias"]
    haiku_params["BERT/mlm_ln"]["scale"] = jax_bert_params["cls.predictions.transform.LayerNorm.weight"]
    haiku_params["BERT/mlm_ln"]["offset"] = jax_bert_params["cls.predictions.transform.LayerNorm.bias"]
    haiku_params["BERT/mlm_bias"]["b"] = jax_bert_params["cls.predictions.bias"]

    return haiku_params

# # ========== STEP 5: Match and Load Pretrained Weights ==========
# def load_pretrained_weights(haiku_params, jax_bert_params):
#     """Map Hugging Face BERT weights to Haiku model structure."""
#     new_params = {}

#     for name, value in haiku_params.items():
#         torch_name = name.replace('BERT', '')

#         if "embeddings" in torch_name:
#             torch_name = 'embeddings.' + torch_name
#         if 'embeddings_ln' in torch_name:
#             torch_name = 'embeddings.LayerNorm.'


#         torch_name = name.replace("/", ".")  # Adjust naming convention if needed
#         if torch_name in jax_bert_params:
#             new_params[name] = jax_bert_params[torch_name]  # Assign pretrained weights
#         else:
#             print(f"⚠️ Warning: No match found for {name}")
#             new_params[name] = value  # Keep original Haiku param

#     return new_params


# # Load pretrained weights into Haiku model
# haiku_params = load_pretrained_weights(haiku_params, jax_bert_params)

def create_enn_bert_for_classification_ft() -> tuple:
    '''Create ENN BERT for classification fine tuning, loading
    weights from Huggingface bert-base-uncased. Classification head
    weights are only initialized. '''

    # ========== STEP 1: Load Pretrained Hugging Face BERT ==========
    pretrained_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    torch_bert = BertModel.from_pretrained(pretrained_model_name)
    bert_config = BertConfig.from_pretrained(pretrained_model_name)

    # Extract BERT weights from Hugging Face and convert them
    # Note: this excludes the pooling layer that bert-base-uncased provides
    jax_bert_params = {
        k: torch_to_jax(v) for k, v in torch_bert.state_dict().items() if "pooler" not in k
    }

    # print(f"Huggingface JAX BERT Parameters:")
    # for param in jax_bert_params.keys():
    #     print(param)

    # ========== STEP 2: Define Custom BERT Configuration for Haiku ==========
    bert_config_custom = BertConfigCustom(bert_config)

    # ========== STEP 3: Tokenize input ==========

    # Example text input
    text = "Hello, this is an example sentence for tokenization."
    bert_input = tokenize_input(text, tokenizer)

    # ========== STEP 4: Initialize Haiku-based ENN BERT Model ==========
    enn_model = make_bert_enn(bert_config_custom, is_training=True)

    # Generate JAX PRNG key
    rng_key = jax.random.PRNGKey(0)

    # Initialize Haiku model parameters and state
    haiku_params, haiku_state = enn_model.init(rng_key, bert_input, index=rng_key)

    # print(f"\nHaiku ENN Model Parameters:")
    # for param in haiku_params.keys():
    #     print(param)

    # ========= STEP 5: Match loaded parameters to Haiku model ==========
    for layer in range(bert_config_custom.num_hidden_layers):
        haiku_params = layer_params_fix(haiku_params, jax_bert_params, layer)

    haiku_params = embedding_params_fix(haiku_params, jax_bert_params)

    # ========== STEP 6: Apply Model to Tokenized Input ==========
    output, new_state = enn_model.apply(
        haiku_params, haiku_state, bert_input, index=rng_key
    )
    # Print output shape
    # print(f"✅ Output.train.shape: {output.train.shape}")
    # print(f"type(output) = {type(output)}")
    # print(f"Output: \n{output}")

    return (enn_model, haiku_params, haiku_state)

if __name__ == '__main__':
    new_head = cls_heads.make_head_enn(
        agent='epinet',
        num_classes=2,
    )

    print(f"DEBUG: type(new_head) = {type(new_head)}")
    print(f"DEBUG: new_head.__dir__() = \n{new_head.__dir__()}")
    print(f"DEBUG: new_head: \n{new_head}")
