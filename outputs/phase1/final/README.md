---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:2673467
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
      value: 0.9755000472068787
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

## Evaluation

### Metrics

#### Col BERTTriplet
* Dataset: `mmarco-it-wiki-eval`
* Evaluated with <code>pylate.evaluation.colbert_triplet.ColBERTTripletEvaluator</code>

| Metric       | Value      |
|:-------------|:-----------|
| **accuracy** | **0.9755** |

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


* Size: 2,673,467 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.46 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 31.97 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | negative                                                                                                                                                                                                                                                                                                                                                 |
  |:------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>cosa significa affermazione?</code> | <code>Un'asserzione è un atto linguistico in cui si afferma che qualcosa vale, per esempio che ci sono infiniti numeri primi, o, rispetto a un certo tempo t, che c'è una congestione del traffico sul ponte di Brooklyn a t, o, di qualche persona x rispetto ad un certo tempo t, che x ha mal di denti in t.</code>                                                                                                                                                                                                                                                                                            | <code>Frasi di esempio e utilizzo di esempio. 1 Sovrintendente Alton L. Frailey: Tuttavia, l'affermazione secondo cui l'insegnante ha cercato di costringerla a negarle Dio o di minacciarla Dio, non è stata confermata. 2 Trey Gowdy: Non sapeva se le fonti fossero legittime, non sapeva se le informazioni fossero state corroborate o meno.</code> |
  | <code>cosa significa idiota?</code>       | <code>Un idiota, stupido, ottuso o (arcaicamente) mome è una persona intellettualmente disabile, o qualcuno che agisce in modo autodistruttivo o significativamente controproducente. Si dice che un idiota sia idiota e soffra di idiozia. Un somaro è un idiota che è specificamente incapace di apprendere. Un idiota differisce da uno sciocco (che è poco saggio) e da un ignorante (che è ignorante/ignorante), nessuno dei quali si riferisce a qualcuno con scarsa intelligenza.</code>                                                                                                                   | <code>Idiota! Sei un idiota!. Questo è ciò che significa. anata: tu (in modo educato) wa: significa che la parola prima è il soggetto della frase (quindi anata/tu è il soggetto della frase) baka: idiota desu: sono . 18 persone lo hanno trovato utile.</code>                                                                                        |
  | <code>la distonia è ereditaria?</code>    | <code>1 La distonia può essere un sintomo di altre malattie, alcune delle quali possono essere ereditarie. 2 La distonia acquisita spesso si stabilizza e non si diffonde ad altre parti del corpo. 3 La distonia che si verifica a seguito di farmaci spesso cessa se i farmaci vengono interrotti rapidamente. La distonia DYT1 è una rara forma di distonia generalizzata ereditaria dominante che può essere causata da una mutazione nel gene DYT1. 2 Questa forma di distonia inizia tipicamente nell'infanzia, colpisce prima gli arti e progredisce, spesso causando una disabilità significativa.</code> | <code>Alcune persone con distonia sono preoccupate per il livello di dolore che provano. Il dolore è spesso associato alla distonia del collo e alla distonia generalizzata e può colpire anche persone con altri tipi di distonia. È un'esperienza molto individuale e varia molto da persona a persona.</code>                                         |
* Loss: <code>pylate.losses.cached_contrastive.CachedContrastive</code>

### Evaluation Dataset

#### Unnamed Dataset


* Size: 2,000 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.65 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 22 tokens</li><li>mean: 31.97 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                      | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | negative                                                                                                                                                                                                                                                                                                                                  |
  |:-------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>può un aneurisma sintomi?</code>     | <code>Sintomi di aneurisma cerebrale rotto. Quando si rompe un aneurisma, chiamato emorragia subaracnoidea, le persone spesso si lamentano del peggior mal di testa della loro vita. Altri sintomi di aneurisma cerebrale rotto includono: nausea e vomito. Torcicollo o dolore al collo. Visione offuscata o visione doppia. Dolore sopra e dietro l'occhio.</code>                                                                                                                               | <code>Aneurisma Mal di testa. Un aneurisma è la formazione di una piccola sacca di sangue all'interno delle pareti di un'arteria. Uno dei luoghi più comuni in cui è più probabile che si sviluppi è nelle arterie del cervello. Questa condizione può essere considerata pericolosa per la vita, soprattutto se perde o si rompe.</code> |
  | <code>chi possiede google?</code>          | <code>La A alla Z di Alphabet, l'azienda che ora possiede Google. Quindi ecco il succo: Google ha scorporato Google, ribattezzandosi Alphabet, che ora possiede Google. I fondatori di Google Larry Page e Sergey Brin guideranno Alphabet rispettivamente come CEO e presidente, mentre Sundar Pichai guiderà Google come nuovo CEO.</code>                                                                                                                                                       | <code>WhatsApp Messenger è un'app di messaggistica mobile multipiattaforma che ti consente di scambiare messaggi senza dover pagare per gli SMS. WhatsApp Messenger è disponibile per iPhone, BlackBerry, Windows Phone, Android e Nokia.</code>                                                                                          |
  | <code>cosa sono i draghi cromatici?</code> | <code>I draghi cromatici sono veri draghi, di solito di allineamento malvagio, al contrario dei draghi metallici di allineamento buono. La maggior parte dei draghi cromatici cerca semplicemente di placare la loro infinita brama di tesori, cibo e spargimento di sangue. Le scaglie di questi draghi corrispondono tutte al colore indicato nel nome. I draghi cromatici sono veri draghi, di solito di allineamento malvagio, al contrario dei draghi metallici di allineamento buono.</code> | <code>I draghi barbuti adulti sono piuttosto vivaci e apprezzano un grande vivaio. Un vivaio da 3 piedi o un serbatoio da 40 galloni dovrebbe essere adeguato per un drago barbuto adulto, ma più spazio puoi dare loro, meglio è, specialmente quando si ospitano più draghi barbuti insieme.</code>                                     |
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
| 0.7679 | 4010 | 0.1259        | -               | -        |
| 0.7698 | 4020 | 0.1488        | -               | -        |
| 0.7717 | 4030 | 0.1386        | -               | -        |
| 0.7736 | 4040 | 0.1430        | -               | -        |
| 0.7756 | 4050 | 0.1283        | -               | -        |
| 0.7775 | 4060 | 0.1467        | -               | -        |
| 0.7794 | 4070 | 0.1410        | -               | -        |
| 0.7813 | 4080 | 0.1388        | -               | -        |
| 0.7832 | 4090 | 0.1240        | -               | -        |
| 0.7851 | 4100 | 0.1351        | -               | -        |
| 0.7871 | 4110 | 0.1309        | -               | -        |
| 0.7890 | 4120 | 0.1245        | -               | -        |
| 0.7909 | 4130 | 0.1463        | -               | -        |
| 0.7928 | 4140 | 0.1369        | -               | -        |
| 0.7947 | 4150 | 0.1406        | -               | -        |
| 0.7966 | 4160 | 0.1482        | -               | -        |
| 0.7985 | 4170 | 0.1405        | -               | -        |
| 0.8005 | 4180 | 0.1292        | -               | -        |
| 0.8024 | 4190 | 0.1395        | -               | -        |
| 0.8043 | 4200 | 0.1384        | -               | -        |
| 0.8062 | 4210 | 0.1302        | -               | -        |
| 0.8081 | 4220 | 0.1403        | -               | -        |
| 0.8100 | 4230 | 0.1329        | -               | -        |
| 0.8119 | 4240 | 0.1366        | -               | -        |
| 0.8139 | 4250 | 0.1388        | -               | -        |
| 0      | 0    | -             | -               | 0.9765   |
| 0.8139 | 4250 | -             | 0.0623          | -        |
| 0.8158 | 4260 | 0.1419        | -               | -        |
| 0.8177 | 4270 | 0.1255        | -               | -        |
| 0.8196 | 4280 | 0.1296        | -               | -        |
| 0.8215 | 4290 | 0.1293        | -               | -        |
| 0.8234 | 4300 | 0.1279        | -               | -        |
| 0.8254 | 4310 | 0.1377        | -               | -        |
| 0.8273 | 4320 | 0.1192        | -               | -        |
| 0.8292 | 4330 | 0.1338        | -               | -        |
| 0.8311 | 4340 | 0.1365        | -               | -        |
| 0.8330 | 4350 | 0.1391        | -               | -        |
| 0.8349 | 4360 | 0.1221        | -               | -        |
| 0.8368 | 4370 | 0.1459        | -               | -        |
| 0.8388 | 4380 | 0.1272        | -               | -        |
| 0.8407 | 4390 | 0.1240        | -               | -        |
| 0.8426 | 4400 | 0.1373        | -               | -        |
| 0.8445 | 4410 | 0.1257        | -               | -        |
| 0.8464 | 4420 | 0.1360        | -               | -        |
| 0.8483 | 4430 | 0.1402        | -               | -        |
| 0.8502 | 4440 | 0.1181        | -               | -        |
| 0.8522 | 4450 | 0.1318        | -               | -        |
| 0.8541 | 4460 | 0.1301        | -               | -        |
| 0.8560 | 4470 | 0.1236        | -               | -        |
| 0.8579 | 4480 | 0.1437        | -               | -        |
| 0.8598 | 4490 | 0.1400        | -               | -        |
| 0.8617 | 4500 | 0.1332        | -               | -        |
| 0      | 0    | -             | -               | 0.9760   |
| 0.8617 | 4500 | -             | 0.0616          | -        |
| 0.8637 | 4510 | 0.1492        | -               | -        |
| 0.8656 | 4520 | 0.1295        | -               | -        |
| 0.8675 | 4530 | 0.1370        | -               | -        |
| 0.8694 | 4540 | 0.1415        | -               | -        |
| 0.8713 | 4550 | 0.1326        | -               | -        |
| 0.8732 | 4560 | 0.1352        | -               | -        |
| 0.8751 | 4570 | 0.1273        | -               | -        |
| 0.8771 | 4580 | 0.1252        | -               | -        |
| 0.8790 | 4590 | 0.1371        | -               | -        |
| 0.8809 | 4600 | 0.1388        | -               | -        |
| 0.8828 | 4610 | 0.1366        | -               | -        |
| 0.8847 | 4620 | 0.1256        | -               | -        |
| 0.8866 | 4630 | 0.1370        | -               | -        |
| 0.8885 | 4640 | 0.1293        | -               | -        |
| 0.8905 | 4650 | 0.1239        | -               | -        |
| 0.8924 | 4660 | 0.1377        | -               | -        |
| 0.8943 | 4670 | 0.1281        | -               | -        |
| 0.8962 | 4680 | 0.1400        | -               | -        |
| 0.8981 | 4690 | 0.1288        | -               | -        |
| 0.9000 | 4700 | 0.1282        | -               | -        |
| 0.9020 | 4710 | 0.1453        | -               | -        |
| 0.9039 | 4720 | 0.1407        | -               | -        |
| 0.9058 | 4730 | 0.1311        | -               | -        |
| 0.9077 | 4740 | 0.1339        | -               | -        |
| 0.9096 | 4750 | 0.1345        | -               | -        |
| 0      | 0    | -             | -               | 0.9755   |
| 0.9096 | 4750 | -             | 0.0605          | -        |
| 0.9115 | 4760 | 0.1387        | -               | -        |
| 0.9134 | 4770 | 0.1340        | -               | -        |
| 0.9154 | 4780 | 0.1350        | -               | -        |
| 0.9173 | 4790 | 0.1218        | -               | -        |
| 0.9192 | 4800 | 0.1289        | -               | -        |
| 0.9211 | 4810 | 0.1285        | -               | -        |
| 0.9230 | 4820 | 0.1368        | -               | -        |
| 0.9249 | 4830 | 0.1336        | -               | -        |
| 0.9268 | 4840 | 0.1464        | -               | -        |
| 0.9288 | 4850 | 0.1408        | -               | -        |
| 0.9307 | 4860 | 0.1279        | -               | -        |
| 0.9326 | 4870 | 0.1290        | -               | -        |
| 0.9345 | 4880 | 0.1342        | -               | -        |
| 0.9364 | 4890 | 0.1351        | -               | -        |
| 0.9383 | 4900 | 0.1474        | -               | -        |
| 0.9403 | 4910 | 0.1284        | -               | -        |
| 0.9422 | 4920 | 0.1296        | -               | -        |
| 0.9441 | 4930 | 0.1199        | -               | -        |
| 0.9460 | 4940 | 0.1520        | -               | -        |
| 0.9479 | 4950 | 0.1391        | -               | -        |
| 0.9498 | 4960 | 0.1368        | -               | -        |
| 0.9517 | 4970 | 0.1243        | -               | -        |
| 0.9537 | 4980 | 0.1276        | -               | -        |
| 0.9556 | 4990 | 0.1274        | -               | -        |
| 0.9575 | 5000 | 0.1375        | -               | -        |
| 0      | 0    | -             | -               | 0.9755   |
| 0.9575 | 5000 | -             | 0.0610          | -        |
| 0.9594 | 5010 | 0.1357        | -               | -        |
| 0.9613 | 5020 | 0.1182        | -               | -        |
| 0.9632 | 5030 | 0.1178        | -               | -        |
| 0.9651 | 5040 | 0.1272        | -               | -        |
| 0.9671 | 5050 | 0.1574        | -               | -        |
| 0.9690 | 5060 | 0.1291        | -               | -        |
| 0.9709 | 5070 | 0.1333        | -               | -        |
| 0.9728 | 5080 | 0.1223        | -               | -        |
| 0.9747 | 5090 | 0.1303        | -               | -        |
| 0.9766 | 5100 | 0.1316        | -               | -        |
| 0.9786 | 5110 | 0.1215        | -               | -        |
| 0.9805 | 5120 | 0.1376        | -               | -        |
| 0.9824 | 5130 | 0.1281        | -               | -        |
| 0.9843 | 5140 | 0.1524        | -               | -        |
| 0.9862 | 5150 | 0.1240        | -               | -        |
| 0.9881 | 5160 | 0.1378        | -               | -        |
| 0.9900 | 5170 | 0.1184        | -               | -        |
| 0.9920 | 5180 | 0.1310        | -               | -        |
| 0.9939 | 5190 | 0.1147        | -               | -        |
| 0.9958 | 5200 | 0.1190        | -               | -        |
| 0.9977 | 5210 | 0.1416        | -               | -        |
| 0.9996 | 5220 | 0.1243        | -               | -        |

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