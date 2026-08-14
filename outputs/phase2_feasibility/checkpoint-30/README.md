---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1022
- loss:Distillation
pipeline_tag: sentence-similarity
library_name: PyLate
---

# PyLate

This is a [PyLate](https://github.com/lightonai/pylate) model trained on the embeddings-fine-tuning-filtered-it dataset. It maps sentences & paragraphs to sequences of 128-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Document Length:** 512 tokens
- **Query Length:** 32 tokens
- **Output Dimensionality:** 128 tokens
- **Similarity Function:** MaxSim
- **Training Dataset:**
    - embeddings-fine-tuning-filtered-it
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [PyLate Documentation](https://lightonai.github.io/pylate/)
- **Repository:** [PyLate on GitHub](https://github.com/lightonai/pylate)
- **Hugging Face:** [PyLate models on Hugging Face](https://huggingface.co/models?library=PyLate)

### Full Model Architecture

```
ColBERT(
  (0): Transformer({'max_seq_length': 511, 'do_lower_case': False, 'architecture': 'ModernBertModel'})
  (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'use_residual': False})
)
```

## Usage
First install the PyLate library:

```bash
pip install -U pylate
```

### Retrieval

Use this model with PyLate to index and retrieve documents. The index uses [FastPLAID](https://github.com/lightonai/fast-plaid) for efficient similarity search.

#### Indexing documents

Load the ColBERT model and initialize the PLAID index, then encode and index your documents:

```python
from pylate import indexes, models, retrieve

# Step 1: Load the ColBERT model
model = models.ColBERT(
    model_name_or_path="pylate_model_id",
)

# Step 2: Initialize the PLAID index
index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
    override=True,  # This overwrites the existing index if any
)

# Step 3: Encode the documents
documents_ids = ["1", "2", "3"]
documents = ["document 1 text", "document 2 text", "document 3 text"]

documents_embeddings = model.encode(
    documents,
    batch_size=32,
    is_query=False,  # Ensure that it is set to False to indicate that these are documents, not queries
    show_progress_bar=True,
)

# Step 4: Add document embeddings to the index by providing embeddings and corresponding ids
index.add_documents(
    documents_ids=documents_ids,
    documents_embeddings=documents_embeddings,
)
```

Note that you do not have to recreate the index and encode the documents every time. Once you have created an index and added the documents, you can re-use the index later by loading it:

```python
# To load an index, simply instantiate it with the correct folder/name and without overriding it
index = indexes.PLAID(
    index_folder="pylate-index",
    index_name="index",
)
```

#### Retrieving top-k documents for queries

Once the documents are indexed, you can retrieve the top-k most relevant documents for a given set of queries.
To do so, initialize the ColBERT retriever with the index you want to search in, encode the queries and then retrieve the top-k documents to get the top matches ids and relevance scores:

```python
# Step 1: Initialize the ColBERT retriever
retriever = retrieve.ColBERT(index=index)

# Step 2: Encode the queries
queries_embeddings = model.encode(
    ["query for document 3", "query for document 1"],
    batch_size=32,
    is_query=True,  #  # Ensure that it is set to False to indicate that these are queries
    show_progress_bar=True,
)

# Step 3: Retrieve top-k documents
scores = retriever.retrieve(
    queries_embeddings=queries_embeddings,
    k=10,  # Retrieve the top 10 matches for each query
)
```

### Reranking
If you only want to use the ColBERT model to perform reranking on top of your first-stage retrieval pipeline without building an index, you can simply use rank function and pass the queries and documents to rerank:

```python
from pylate import rank, models

queries = [
    "query A",
    "query B",
]

documents = [
    ["document A", "document B"],
    ["document 1", "document C", "document B"],
]

documents_ids = [
    [1, 2],
    [1, 3, 2],
]

model = models.ColBERT(
    model_name_or_path="pylate_model_id",
)

queries_embeddings = model.encode(
    queries,
    is_query=True,
)

documents_embeddings = model.encode(
    documents,
    is_query=False,
)

reranked_documents = rank.rerank(
    documents_ids=documents_ids,
    queries_embeddings=queries_embeddings,
    documents_embeddings=documents_embeddings,
)
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### embeddings-fine-tuning-filtered-it

* Dataset: embeddings-fine-tuning-filtered-it
* Size: 1,022 training samples
* Columns: <code>query</code>, <code>documents</code>, and <code>scores</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | documents                           | scores                              |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------|:------------------------------------|
  | type    | string                                                                            | list                                | list                                |
  | details | <ul><li>min: 7 tokens</li><li>mean: 18.25 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>size: 11 elements</li></ul> | <ul><li>size: 11 elements</li></ul> |
* Samples:
  | query                                                                                                                                                                    | documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | scores                                                   |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------|
  | <code>Come si chiama il canale del parto nella donna?</code>                                                                                                             | <code>["La vagina svolge tre funzioni principali: 1. È il punto in cui viene inserito il pene durante l'atto sessuale. 2. È il percorso che il bambino percorre per uscire dal corpo della donna durante il parto, noto come canale del parto. 3. È il passaggio attraverso cui il sangue mestruale (la regola) esce dal corpo dall'utero.", 'Menzionato in? 1 canale accessorio. 2 canale alimentare. 3 canale anale. 4 occhio artificiale. 5 canale atrioventricolare. 6 setto atrioventricolare. 7 canale del parto. 8 setto bulbare. 9 canale. 10 canale di Corti. 11 canale carotideo. 12 canale cervicale. 13 canale condiloide. 14 diplopia. 15 occhio secco. 16 osso etmoide. 17 occhio. 18 orbita oculare. 19 lavaggio oculare.', "La vagina è l'apertura che si trova tra l'uretra e l'ano. Normalmente ha una profondità di circa 10-15 centimetri. All'estremità posteriore della vagina si trova il collo dell'utero, o cervice. La cervice è spessa circa 2,5 centimetri e non è altro che un'apertura muscolare che conduce all...</code> | <code>[10.625, 6.875, 5.75, 6.0625, 2.0625, ...]</code>  |
  | <code>Su quale fiume sorge Bristol (UK)?</code>                                                                                                                          | <code>['“Netham a Bristol, dove le imbarcazioni provenienti dal fiume Avon hanno accesso al Porto Galleggiante di Bristol. La costruzione iniziò nel 1804 per realizzare la New Cut, un canale marea dove confluisce il fiume Malago, e per deviare il corso dell’Avon lungo il Feeder Canal verso il porto; un sistema progettato e costruito da William Jessop e successivamente migliorato da Isambard Kingdom Brunel. Una diga convoglia il fiume nella New Cut e le imbarcazioni utilizzano il paraggio adiacente. L’accesso al porto è possibile solo di giorno, quando il custode del paraggio aprirà le sponde, a meno che il livello dell’acqua nel fiume non lo permetta.”', 'Includevano frumento, lana, tessuti, cemento, mattoni e tegole. A differenza di Bristol, Bridgwater non fu mai coinvolto nel commercio degli schiavi e, nel 1797, fu la prima città del Regno Unito a chiedere al governo di abolirlo. La nave di Bridgwater "Emanuel" fu una delle tre che parteciparono alla spedizione del 1577 organizzata da Mart...</code> | <code>[6.5625, -1.4375, 6.625, 5.5, 6.5, ...]</code>     |
  | <code>Chi è stato nominato presidente della South Carolina Educational Television Commission dall'attuale ambasciatore degli Stati Uniti presso le Nazioni Unite?</code> | <code>['Nimrata "Nikki" Haley (nata Randhawa; nata il 20 gennaio 1972) è la 29ª e attuale ambasciatrice degli Stati Uniti alle Nazioni Unite. Ha ricoperto la carica di 116ª governatrice della Carolina del Sud dal gennaio 2011 al gennaio 2017. Prima del suo mandato da governatrice, Haley è stata membro della Camera dei rappresentanti della Carolina del Sud.', 'Scuola Governativa del South Carolina per le Arti e le Scienze Umanistiche', 'Inviato Speciale degli Stati Uniti per le questioni femminili globali', 'Ambasciatore Speciale per il Monitoraggio e la Lotta contro la Tratta delle Persone', 'Ambasciatore degli Stati Uniti in Sri Lanka e nelle Maldive', ...]</code>                                                                                                                                                                                                                                                                                                                                                        | <code>[8.5, -3.3125, -1.0, -1.5625, -1.4375, ...]</code> |
* Loss: <code>pylate.losses.distillation.Distillation</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1.0
- `max_steps`: 30
- `learning_rate`: 1e-05
- `warmup_steps`: 0.05
- `bf16`: True
- `disable_tqdm`: True
- `per_device_eval_batch_size`: 4

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1.0
- `max_steps`: 30
- `learning_rate`: 1e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.05
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: True
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 4
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0039 | 1    | 1.0577        |
| 0.0078 | 2    | 0.8940        |
| 0.0117 | 3    | 1.0470        |
| 0.0156 | 4    | 1.0108        |
| 0.0195 | 5    | 1.3500        |
| 0.0234 | 6    | 1.0891        |
| 0.0273 | 7    | 1.2901        |
| 0.0312 | 8    | 0.8924        |
| 0.0352 | 9    | 1.1397        |
| 0.0391 | 10   | 1.7140        |
| 0.0430 | 11   | 1.4728        |
| 0.0469 | 12   | 1.1991        |
| 0.0508 | 13   | 1.6058        |
| 0.0547 | 14   | 1.1141        |
| 0.0586 | 15   | 0.9642        |
| 0.0625 | 16   | 0.5263        |
| 0.0664 | 17   | 0.6198        |
| 0.0703 | 18   | 0.5091        |
| 0.0742 | 19   | 0.9336        |
| 0.0781 | 20   | 1.1491        |
| 0.0820 | 21   | 1.1060        |
| 0.0859 | 22   | 0.8594        |
| 0.0898 | 23   | 1.4065        |
| 0.0938 | 24   | 1.5763        |
| 0.0977 | 25   | 0.8889        |
| 0.1016 | 26   | 1.5383        |
| 0.1055 | 27   | 0.7008        |
| 0.1094 | 28   | 1.3316        |
| 0.1133 | 29   | 1.2452        |
| 0.1172 | 30   | 2.0546        |


### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.3.0
- PyLate: 1.5.0
- Transformers: 5.3.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2


## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084"
}
```

#### PyLate
```bibtex
@inproceedings{DBLP:conf/cikm/ChaffinS25,
  author       = {Antoine Chaffin and
                  Rapha{"{e}}l Sourty},
  editor       = {Meeyoung Cha and
                  Chanyoung Park and
                  Noseong Park and
                  Carl Yang and
                  Senjuti Basu Roy and
                  Jessie Li and
                  Jaap Kamps and
                  Kijung Shin and
                  Bryan Hooi and
                  Lifang He},
  title        = {PyLate: Flexible Training and Retrieval for Late Interaction Models},
  booktitle    = {Proceedings of the 34th {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2025, Seoul, Republic of Korea, November
                  10-14, 2025},
  pages        = {6334--6339},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://github.com/lightonai/pylate},
  doi          = {10.1145/3746252.3761608},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->