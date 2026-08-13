---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2487135
- loss:CachedContrastive
pipeline_tag: sentence-similarity
library_name: PyLate
metrics:
- accuracy
model-index:
- name: PyLate
  results:
  - task:
      type: col-berttriplet
      name: Col BERTTriplet
    dataset:
      name: mmarco it wiki eval
      type: mmarco-it-wiki-eval
    metrics:
    - type: accuracy
      value: 0.9725000262260437
      name: Accuracy
---

# PyLate

This is a [PyLate](https://github.com/lightonai/pylate) model trained. It maps sentences & paragraphs to sequences of 128-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Document Length:** 512 tokens
- **Query Length:** 32 tokens
- **Output Dimensionality:** 128 tokens
- **Similarity Function:** MaxSim
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [PyLate Documentation](https://lightonai.github.io/pylate/)
- **Repository:** [PyLate on GitHub](https://github.com/lightonai/pylate)
- **Hugging Face:** [PyLate models on Hugging Face](https://huggingface.co/models?library=PyLate)

### Full Model Architecture

```
ColBERT(
  (0): Transformer({'max_seq_length': 31, 'do_lower_case': False, 'architecture': 'ModernBertModel'})
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

## Evaluation

### Metrics

#### Col BERTTriplet
* Dataset: `mmarco-it-wiki-eval`
* Evaluated with <code>pylate.evaluation.colbert_triplet.ColBERTTripletEvaluator</code>

| Metric       | Value      |
|:-------------|:-----------|
| **accuracy** | **0.9725** |

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

#### Unnamed Dataset


* Size: 2,487,135 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 6 tokens</li><li>mean: 12.59 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 31.97 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                                              | positive                                                                                                                                                                                                                                                                                                                                                                                                                          | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
  |:-------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>rasoi da donna bic</code>                                    | <code>Cerca prodotto risultato. 1 prodotto - Rasoio usa e getta a 2 lame da donna BIC Silky Touch, confezione da 10 best seller. 2 prodotti - Rasoio monouso a 3 lame BIC Soleil Twilight da donna, confezione da 4 Immagine del prodotto. 3 prodotti - BIC Soleil Colour Collection Rasoio da donna usa e getta a 3 lame, confezione da 8 Immagine del prodotto.</code>                                                          | <code>I numeri bancari come Bank Identifier Code (BIC), Bank Identification Number (BIN) e Routing transit number (RTN) classificano una banca per lo smistamento automatico degli assegni e così via.</code>                                                                                                                                                                                                                                                        |
  | <code>quali sono gli ingredienti per gli involtini di pizza</code> | <code>Nascondi immagini. 1 1. Preriscaldare il forno a 425Ãâ€šÃ‚Â°F. Stendete l'impasto della pizza in un rettangolo grande. 2 2. Cospargere l'impasto con sale all'aglio, basilico, formaggi e peperoni. 3 3. Disporre i rotoli su teglie leggermente unte.</code>                                                                                                                                                               | <code>I nostri corsi di pizzaiolo sono riconosciuti dall'Associazione Italiana Pizzaioli. I nostri istruttori possono viaggiare da te e dalla tua attività di pizzaiolo, ovunque tu sia nel mondo. Per maggiori dettagli fare clic sul collegamento alla nostra pagina Web di seguito: ÃÂ¢Ã‚â‚¬Ã‚Å“Consulenza aziendale-Corso di pizza direttamente presso la vostra sede.</code>                                                                                    |
  | <code>dov'è la nazione di Fisher River Creek?</code>               | <code>Nazione di Fisher River Cree. Fisher River (Ochekwi-Sipi) è una riserva Cree First Nations situata a circa 193 km a nord della capitale del Manitoba, Winnipeg. La Fisher River Cree Nation è composta da due riserve; Fiume Fisher 44 e fiume Fisher 44A. La popolazione di riserva è 1709, la popolazione fuori riserva è 1389 per un totale di 3098 membri della band. Fisher River è 15.614 acri (6.319 ettari).</code> | <code>Bacino del fiume Snake superiore: Snake River sotto il lago Jackson: 1.1: Bacino del fiume Snake superiore: Fiume Gros Ventre a Zenith, WY: 1.2: Bacino del fiume Snake superiore: Snake River sotto Flat Creek vicino a Jackson, WY: 1.3: Bacino del fiume Snake superiore: Snake River sopra il bacino idrico vicino a Alpine, WY: 1.4: Upper Snake River Basin: Grays River sopra il bacino idrico vicino a Alpine, WY: 1.5: Upper Snake River Basin</code> |
* Loss: <code>pylate.losses.cached_contrastive.CachedContrastive</code>

### Evaluation Dataset

#### Unnamed Dataset


* Size: 2,000 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.55 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 27 tokens</li><li>mean: 31.99 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 31.97 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                                              | positive                                                                                                                                                                                                                                                                                                                      | negative                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:-------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>che cos'è la consulenza?</code>                              | <code>Definizione di Counseling. Secondo l'American Counseling Association, la consulenza è definita come una relazione professionale che consente a diversi individui, famiglie e gruppi di raggiungere obiettivi di salute mentale, benessere, istruzione e carriera.</code>                                                | <code>ÃƒÂ¢Ã‚â‚¬Ã‚Â¢ Classificato in Salute \| Differenza tra consulenza e terapia. Counseling vs Terapia. La vita non è perfetta come ci aspettiamo che sia. La vita, come si dice, può essere bella; può essere brutto. Una verità è che viverla da soli è una sfida. O la vita ti controllerà o tu controllerai la tua vita.</code>                                                                                  |
  | <code>cos'è un .rtf rispetto a un .docx?</code>                    | <code>Sia il formato RTF che il formato DOC sono sviluppati da Microsoft per Word. RTF è un formato più vecchio di DOC. Al giorno d'oggi, il formato DOC è più popolare del formato RTF. Tuttavia, RTF è ancora utilizzato da alcune persone per i suoi vantaggi. Le differenze tra i due formati sono le seguenti:</code>    | <code>Per prima cosa, sono più compressi dei file .doc. Un documento Word che utilizza il formato .docx potrebbe essere metà o tre quarti delle dimensioni di un file .doc, afferma. Ciò consente di risparmiare spazio sul disco rigido e larghezza di banda. Inoltre, se hai provato ad aprire un file danneggiato in un formato .doc, Word semplicemente non poteva aprirlo.</code>                                 |
  | <code>con quali altri sistemi funziona il sistema linfatico</code> | <code>Le strutture primarie del sistema linfatico e immunitario degli arti inferiori sono i vasi linfatici. Anche le grandi ossa della gamba sono importanti, poiché contengono midollo osseo che produce un gran numero di linfociti.\\n\\nIl sistema linfatico lavora a stretto contatto con il sistema immunitario.</code> | <code>Lo studio del flusso sanguigno è chiamato emodinamica. Lo studio delle proprietà del flusso sanguigno è chiamato emoreologia. Il sistema circolatorio è spesso visto come composto da due sistemi separati: il sistema cardiovascolare, che distribuisce il sangue, e il sistema linfatico, che fa circolare la linfa. Il passaggio della linfa ad esempio richiede molto più tempo di quello del sangue.</code> |
* Loss: <code>pylate.losses.cached_contrastive.CachedContrastive</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 512
- `num_train_epochs`: 1.0
- `learning_rate`: 1e-05
- `warmup_steps`: 0.05
- `bf16`: True
- `disable_tqdm`: True
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 512
- `num_train_epochs`: 1.0
- `max_steps`: -1
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
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
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
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss | Validation Loss | accuracy |
|:------:|:----:|:-------------:|:---------------:|:--------:|
| 0.5167 | 2510 | 0.1512        | -               | -        |
| 0.5187 | 2520 | 0.1725        | -               | -        |
| 0.5208 | 2530 | 0.1871        | -               | -        |
| 0.5228 | 2540 | 0.1671        | -               | -        |
| 0.5249 | 2550 | 0.1642        | -               | -        |
| 0.5270 | 2560 | 0.1627        | -               | -        |
| 0.5290 | 2570 | 0.1774        | -               | -        |
| 0.5311 | 2580 | 0.1714        | -               | -        |
| 0.5331 | 2590 | 0.1798        | -               | -        |
| 0.5352 | 2600 | 0.1753        | -               | -        |
| 0.5373 | 2610 | 0.1599        | -               | -        |
| 0.5393 | 2620 | 0.1953        | -               | -        |
| 0.5414 | 2630 | 0.1751        | -               | -        |
| 0.5434 | 2640 | 0.1816        | -               | -        |
| 0.5455 | 2650 | 0.1654        | -               | -        |
| 0.5476 | 2660 | 0.1880        | -               | -        |
| 0.5496 | 2670 | 0.1831        | -               | -        |
| 0.5517 | 2680 | 0.1933        | -               | -        |
| 0.5537 | 2690 | 0.1653        | -               | -        |
| 0.5558 | 2700 | 0.1575        | -               | -        |
| 0.5578 | 2710 | 0.1717        | -               | -        |
| 0.5599 | 2720 | 0.1644        | -               | -        |
| 0.5620 | 2730 | 0.1949        | -               | -        |
| 0.5640 | 2740 | 0.1702        | -               | -        |
| 0.5661 | 2750 | 0.1766        | -               | -        |
| 0      | 0    | -             | -               | 0.9665   |
| 0.5661 | 2750 | -             | 0.0947          | -        |
| 0.5681 | 2760 | 0.1756        | -               | -        |
| 0.5702 | 2770 | 0.1724        | -               | -        |
| 0.5723 | 2780 | 0.1593        | -               | -        |
| 0.5743 | 2790 | 0.1828        | -               | -        |
| 0.5764 | 2800 | 0.1646        | -               | -        |
| 0.5784 | 2810 | 0.1686        | -               | -        |
| 0.5805 | 2820 | 0.1793        | -               | -        |
| 0.5825 | 2830 | 0.1776        | -               | -        |
| 0.5846 | 2840 | 0.1708        | -               | -        |
| 0.5867 | 2850 | 0.1688        | -               | -        |
| 0.5887 | 2860 | 0.1794        | -               | -        |
| 0.5908 | 2870 | 0.1696        | -               | -        |
| 0.5928 | 2880 | 0.1677        | -               | -        |
| 0.5949 | 2890 | 0.1628        | -               | -        |
| 0.5970 | 2900 | 0.1574        | -               | -        |
| 0.5990 | 2910 | 0.1760        | -               | -        |
| 0.6011 | 2920 | 0.1536        | -               | -        |
| 0.6031 | 2930 | 0.1560        | -               | -        |
| 0.6052 | 2940 | 0.1663        | -               | -        |
| 0.6072 | 2950 | 0.1465        | -               | -        |
| 0.6093 | 2960 | 0.1694        | -               | -        |
| 0.6114 | 2970 | 0.1469        | -               | -        |
| 0.6134 | 2980 | 0.1525        | -               | -        |
| 0.6155 | 2990 | 0.1708        | -               | -        |
| 0.6175 | 3000 | 0.1644        | -               | -        |
| 0      | 0    | -             | -               | 0.9700   |
| 0.6175 | 3000 | -             | 0.0887          | -        |
| 0.6196 | 3010 | 0.1603        | -               | -        |
| 0.6217 | 3020 | 0.1839        | -               | -        |
| 0.6237 | 3030 | 0.1613        | -               | -        |
| 0.6258 | 3040 | 0.1790        | -               | -        |
| 0.6278 | 3050 | 0.1699        | -               | -        |
| 0.6299 | 3060 | 0.1640        | -               | -        |
| 0.6319 | 3070 | 0.1582        | -               | -        |
| 0.6340 | 3080 | 0.1707        | -               | -        |
| 0.6361 | 3090 | 0.1651        | -               | -        |
| 0.6381 | 3100 | 0.1517        | -               | -        |
| 0.6402 | 3110 | 0.1729        | -               | -        |
| 0.6422 | 3120 | 0.1845        | -               | -        |
| 0.6443 | 3130 | 0.1470        | -               | -        |
| 0.6464 | 3140 | 0.1592        | -               | -        |
| 0.6484 | 3150 | 0.1629        | -               | -        |
| 0.6505 | 3160 | 0.1597        | -               | -        |
| 0.6525 | 3170 | 0.1510        | -               | -        |
| 0.6546 | 3180 | 0.1718        | -               | -        |
| 0.6566 | 3190 | 0.1562        | -               | -        |
| 0.6587 | 3200 | 0.1670        | -               | -        |
| 0.6608 | 3210 | 0.1598        | -               | -        |
| 0.6628 | 3220 | 0.1723        | -               | -        |
| 0.6649 | 3230 | 0.1338        | -               | -        |
| 0.6669 | 3240 | 0.1635        | -               | -        |
| 0.6690 | 3250 | 0.1642        | -               | -        |
| 0      | 0    | -             | -               | 0.9705   |
| 0.6690 | 3250 | -             | 0.0872          | -        |
| 0.6711 | 3260 | 0.1570        | -               | -        |
| 0.6731 | 3270 | 0.1722        | -               | -        |
| 0.6752 | 3280 | 0.1653        | -               | -        |
| 0.6772 | 3290 | 0.1617        | -               | -        |
| 0.6793 | 3300 | 0.1671        | -               | -        |
| 0.6814 | 3310 | 0.1505        | -               | -        |
| 0.6834 | 3320 | 0.1668        | -               | -        |
| 0.6855 | 3330 | 0.1401        | -               | -        |
| 0.6875 | 3340 | 0.1580        | -               | -        |
| 0.6896 | 3350 | 0.1561        | -               | -        |
| 0.6916 | 3360 | 0.1639        | -               | -        |
| 0.6937 | 3370 | 0.1452        | -               | -        |
| 0.6958 | 3380 | 0.1511        | -               | -        |
| 0.6978 | 3390 | 0.1616        | -               | -        |
| 0.6999 | 3400 | 0.1481        | -               | -        |
| 0.7019 | 3410 | 0.1590        | -               | -        |
| 0.7040 | 3420 | 0.1482        | -               | -        |
| 0.7061 | 3430 | 0.1449        | -               | -        |
| 0.7081 | 3440 | 0.1621        | -               | -        |
| 0.7102 | 3450 | 0.1479        | -               | -        |
| 0.7122 | 3460 | 0.1344        | -               | -        |
| 0.7143 | 3470 | 0.1578        | -               | -        |
| 0.7163 | 3480 | 0.1626        | -               | -        |
| 0.7184 | 3490 | 0.1611        | -               | -        |
| 0.7205 | 3500 | 0.1556        | -               | -        |
| 0      | 0    | -             | -               | 0.9695   |
| 0.7205 | 3500 | -             | 0.0872          | -        |
| 0.7225 | 3510 | 0.1579        | -               | -        |
| 0.7246 | 3520 | 0.1613        | -               | -        |
| 0.7266 | 3530 | 0.1556        | -               | -        |
| 0.7287 | 3540 | 0.1626        | -               | -        |
| 0.7308 | 3550 | 0.1368        | -               | -        |
| 0.7328 | 3560 | 0.1461        | -               | -        |
| 0.7349 | 3570 | 0.1423        | -               | -        |
| 0.7369 | 3580 | 0.1514        | -               | -        |
| 0.7390 | 3590 | 0.1618        | -               | -        |
| 0.7410 | 3600 | 0.1506        | -               | -        |
| 0.7431 | 3610 | 0.1495        | -               | -        |
| 0.7452 | 3620 | 0.1628        | -               | -        |
| 0.7472 | 3630 | 0.1474        | -               | -        |
| 0.7493 | 3640 | 0.1468        | -               | -        |
| 0.7513 | 3650 | 0.1539        | -               | -        |
| 0.7534 | 3660 | 0.1476        | -               | -        |
| 0.7555 | 3670 | 0.1530        | -               | -        |
| 0.7575 | 3680 | 0.1734        | -               | -        |
| 0.7596 | 3690 | 0.1502        | -               | -        |
| 0.7616 | 3700 | 0.1422        | -               | -        |
| 0.7637 | 3710 | 0.1579        | -               | -        |
| 0.7657 | 3720 | 0.1507        | -               | -        |
| 0.7678 | 3730 | 0.1628        | -               | -        |
| 0.7699 | 3740 | 0.1455        | -               | -        |
| 0.7719 | 3750 | 0.1628        | -               | -        |
| 0      | 0    | -             | -               | 0.9700   |
| 0.7719 | 3750 | -             | 0.0833          | -        |
| 0.7740 | 3760 | 0.1594        | -               | -        |
| 0.7760 | 3770 | 0.1413        | -               | -        |
| 0.7781 | 3780 | 0.1386        | -               | -        |
| 0.7802 | 3790 | 0.1569        | -               | -        |
| 0.7822 | 3800 | 0.1508        | -               | -        |
| 0.7843 | 3810 | 0.1598        | -               | -        |
| 0.7863 | 3820 | 0.1681        | -               | -        |
| 0.7884 | 3830 | 0.1434        | -               | -        |
| 0.7904 | 3840 | 0.1426        | -               | -        |
| 0.7925 | 3850 | 0.1482        | -               | -        |
| 0.7946 | 3860 | 0.1683        | -               | -        |
| 0.7966 | 3870 | 0.1389        | -               | -        |
| 0.7987 | 3880 | 0.1600        | -               | -        |
| 0.8007 | 3890 | 0.1422        | -               | -        |
| 0.8028 | 3900 | 0.1576        | -               | -        |
| 0.8049 | 3910 | 0.1325        | -               | -        |
| 0.8069 | 3920 | 0.1271        | -               | -        |
| 0.8090 | 3930 | 0.1396        | -               | -        |
| 0.8110 | 3940 | 0.1480        | -               | -        |
| 0.8131 | 3950 | 0.1333        | -               | -        |
| 0.8152 | 3960 | 0.1511        | -               | -        |
| 0.8172 | 3970 | 0.1539        | -               | -        |
| 0.8193 | 3980 | 0.1499        | -               | -        |
| 0.8213 | 3990 | 0.1410        | -               | -        |
| 0.8234 | 4000 | 0.1672        | -               | -        |
| 0      | 0    | -             | -               | 0.9700   |
| 0.8234 | 4000 | -             | 0.0838          | -        |
| 0.8254 | 4010 | 0.1285        | -               | -        |
| 0.8275 | 4020 | 0.1593        | -               | -        |
| 0.8296 | 4030 | 0.1629        | -               | -        |
| 0.8316 | 4040 | 0.1506        | -               | -        |
| 0.8337 | 4050 | 0.1335        | -               | -        |
| 0.8357 | 4060 | 0.1535        | -               | -        |
| 0.8378 | 4070 | 0.1509        | -               | -        |
| 0.8399 | 4080 | 0.1451        | -               | -        |
| 0.8419 | 4090 | 0.1403        | -               | -        |
| 0.8440 | 4100 | 0.1481        | -               | -        |
| 0.8460 | 4110 | 0.1388        | -               | -        |
| 0.8481 | 4120 | 0.1348        | -               | -        |
| 0.8501 | 4130 | 0.1368        | -               | -        |
| 0.8522 | 4140 | 0.1500        | -               | -        |
| 0.8543 | 4150 | 0.1375        | -               | -        |
| 0.8563 | 4160 | 0.1548        | -               | -        |
| 0.8584 | 4170 | 0.1522        | -               | -        |
| 0.8604 | 4180 | 0.1576        | -               | -        |
| 0.8625 | 4190 | 0.1344        | -               | -        |
| 0.8646 | 4200 | 0.1426        | -               | -        |
| 0.8666 | 4210 | 0.1388        | -               | -        |
| 0.8687 | 4220 | 0.1609        | -               | -        |
| 0.8707 | 4230 | 0.1455        | -               | -        |
| 0.8728 | 4240 | 0.1591        | -               | -        |
| 0.8748 | 4250 | 0.1628        | -               | -        |
| 0      | 0    | -             | -               | 0.9705   |
| 0.8748 | 4250 | -             | 0.0835          | -        |
| 0.8769 | 4260 | 0.1568        | -               | -        |
| 0.8790 | 4270 | 0.1566        | -               | -        |
| 0.8810 | 4280 | 0.1489        | -               | -        |
| 0.8831 | 4290 | 0.1372        | -               | -        |
| 0.8851 | 4300 | 0.1476        | -               | -        |
| 0.8872 | 4310 | 0.1396        | -               | -        |
| 0.8893 | 4320 | 0.1474        | -               | -        |
| 0.8913 | 4330 | 0.1577        | -               | -        |
| 0.8934 | 4340 | 0.1595        | -               | -        |
| 0.8954 | 4350 | 0.1376        | -               | -        |
| 0.8975 | 4360 | 0.1542        | -               | -        |
| 0.8995 | 4370 | 0.1478        | -               | -        |
| 0.9016 | 4380 | 0.1546        | -               | -        |
| 0.9037 | 4390 | 0.1611        | -               | -        |
| 0.9057 | 4400 | 0.1608        | -               | -        |
| 0.9078 | 4410 | 0.1532        | -               | -        |
| 0.9098 | 4420 | 0.1326        | -               | -        |
| 0.9119 | 4430 | 0.1564        | -               | -        |
| 0.9140 | 4440 | 0.1454        | -               | -        |
| 0.9160 | 4450 | 0.1502        | -               | -        |
| 0.9181 | 4460 | 0.1469        | -               | -        |
| 0.9201 | 4470 | 0.1476        | -               | -        |
| 0.9222 | 4480 | 0.1369        | -               | -        |
| 0.9242 | 4490 | 0.1461        | -               | -        |
| 0.9263 | 4500 | 0.1499        | -               | -        |
| 0      | 0    | -             | -               | 0.9715   |
| 0.9263 | 4500 | -             | 0.0830          | -        |
| 0.9284 | 4510 | 0.1292        | -               | -        |
| 0.9304 | 4520 | 0.1325        | -               | -        |
| 0.9325 | 4530 | 0.1356        | -               | -        |
| 0.9345 | 4540 | 0.1478        | -               | -        |
| 0.9366 | 4550 | 0.1457        | -               | -        |
| 0.9387 | 4560 | 0.1478        | -               | -        |
| 0.9407 | 4570 | 0.1293        | -               | -        |
| 0.9428 | 4580 | 0.1466        | -               | -        |
| 0.9448 | 4590 | 0.1380        | -               | -        |
| 0.9469 | 4600 | 0.1378        | -               | -        |
| 0.9490 | 4610 | 0.1399        | -               | -        |
| 0.9510 | 4620 | 0.1435        | -               | -        |
| 0.9531 | 4630 | 0.1543        | -               | -        |
| 0.9551 | 4640 | 0.1366        | -               | -        |
| 0.9572 | 4650 | 0.1547        | -               | -        |
| 0.9592 | 4660 | 0.1314        | -               | -        |
| 0.9613 | 4670 | 0.1320        | -               | -        |
| 0.9634 | 4680 | 0.1509        | -               | -        |
| 0.9654 | 4690 | 0.1280        | -               | -        |
| 0.9675 | 4700 | 0.1487        | -               | -        |
| 0.9695 | 4710 | 0.1537        | -               | -        |
| 0.9716 | 4720 | 0.1513        | -               | -        |
| 0.9737 | 4730 | 0.1528        | -               | -        |
| 0.9757 | 4740 | 0.1562        | -               | -        |
| 0.9778 | 4750 | 0.1574        | -               | -        |
| 0      | 0    | -             | -               | 0.9725   |
| 0.9778 | 4750 | -             | 0.0827          | -        |

</details>

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

#### CachedContrastive
```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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