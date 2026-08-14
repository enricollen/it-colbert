---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:899998
- loss:Distillation
pipeline_tag: sentence-similarity
library_name: PyLate
metrics:
- kl_divergence
model-index:
- name: PyLate
  results:
  - task:
      type: col-bertdistillation
      name: Col BERTDistillation
    dataset:
      name: kd holdout
      type: kd-holdout
    metrics:
    - type: kl_divergence
      value: 1.0753214359283447
      name: Kl Divergence
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

#### Col BERTDistillation
* Dataset: `kd-holdout`
* Evaluated with <code>pylate.evaluation.colbert_distillation.ColBERTDistillationEvaluator</code>

| Metric            | Value      |
|:------------------|:-----------|
| **kl_divergence** | **1.0753** |

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
* Size: 899,998 training samples
* Columns: <code>query</code>, <code>documents</code>, and <code>scores</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | documents                           | scores                              |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------|:------------------------------------|
  | type    | string                                                                            | list                                | list                                |
  | details | <ul><li>min: 8 tokens</li><li>mean: 18.03 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>size: 11 elements</li></ul> | <ul><li>size: 11 elements</li></ul> |
* Samples:
  | query                                            | documents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | scores                                                    |
  |:-------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
  | <code>Chi ha fondato l'EWTN?</code>              | <code>['Mother Angelica, suora cattolica e fondatrice della rete televisiva Eternal Word Television Network (EWTN). L’EWTN è una delle organizzazioni conservatrici affiliate alla corrente legata al Concilio Vaticano II e rappresenta una rete televisiva cattolica globale.', 'TheUrbanDaily 25 maggio 2011. Nel agosto 1979, con il sogno di offrire contenuti mirati alla comunità afroamericana, un prestito di 15.000 dollari e un investimento di sei cifre da parte della TCI, Robert L. Johnson fondò Black Entertainment Television.', 'Enemy Territory Fortress (modifica per il gioco) ETF: Impianto di trattamento degli effluenti: ETF: Eclipse Trust Framework: ETF: Elektrotehnicki Fakultet u Beogradu: ETF: Forza Europea di Missione: ETF: Scheda elettronica bersaglio (US Navy CVN-68) ETF: Strumentazione elettronica per test', 'Nel 1994, con l’aiuto di Kevin Wendle, co-fondatore della Fox Network, e di Dan Baker, ex collaboratore creativo di Disney, CNET realizzò quattro programmi televisivi pilota dedica...</code> | <code>[10.5625, 1.0625, -5.125, 1.625, -2.0, ...]</code>  |
  | <code>L'hockey è popolare in Bielorussia?</code> | <code>['L\'hockey su ghiaccio è uno sport di squadra di contatto che si pratica sul ghiaccio, solitamente in una pista, in cui due squadre di pattinatori usano i loro bastoni per tirare un disco di gomma vulcanizzata nella porta avversaria per segnare punti. Questo sport è noto per essere veloce e fisico, con squadre che di solito sono composte da sei giocatori: un portiere e cinque giocatori che pattinano su e giù per il ghiaccio cercando di prendere il disco e segnare una rete contro la squadra avversaria. L\'hockey su ghiaccio è molto popolare in Canada, in Europa centrale e orientale, in Scandinavia e nelle regioni settentrionali degli Stati Uniti. È lo sport invernale nazionale ufficiale del Canada, dove è stata creata la versione moderna dello sport, e gode di grande popolarità; oltre al Canada, l\'hockey è lo sport invernale più popolare in Finlandia, Lettonia, Repubblica Ceca, Svezia, Slovacchia, Bielorussia e Svizzera. In Nord America, la National Hockey League (NHL) è il livello pi...</code> | <code>[9.1875, 1.1875, 6.0625, 2.5, 2.9375, ...]</code>   |
  | <code>Con chi ha lavorato A.J. Cook?</code>      | <code>['Andrea Joy "A. J." Cook (nata il 22 luglio 1978) è un\'attrice canadese nota per il suo ruolo di Supervisory Special Agent Jennifer "JJ" Jareau nella serie televisiva di genere crime della CBS "Criminal Minds". Ha anche recitato in film come "Le vergini suicide" (1999), "Out Cold" (2001) e "Final Destination 2" (2003).', 'James Clement Cook (nato il 13 marzo 1959 a Huntington, New York) è un musicista, scrittore e produttore cinematografico e televisivo americano. Dal 1996 al 1999, ha progettato e costruito strumenti musicali sperimentali per il Blue Man Group. È anche conosciuto come Jimmy Cook. Dopo lo tsunami asiatico del 2004, ha viaggiato in Sri Lanka (ex Ceylon) per partecipare alla distribuzione di cibo con Food For Life Global, la più grande organizzazione di soccorso alimentare vegetariana del mondo. Fu lì che incontrò Nandarani Devi, una devota Hare Krishna che gestiva un orfanotrofio per bambini provenienti da entrambe le parti del conflitto etnico tra le comunità singales...</code> | <code>[10.1875, 3.4375, 1.9375, 3.5625, 8.25, ...]</code> |
* Loss: <code>pylate.losses.distillation.Distillation</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1.0
- `learning_rate`: 1e-05
- `warmup_steps`: 0.05
- `bf16`: True
- `disable_tqdm`: True
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 32
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
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
- `load_best_model_at_end`: True
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
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | kd-holdout_kl_divergence |
|:------:|:-----:|:-------------:|:------------------------:|
| 0.0002 | 50    | 1.1157        | -                        |
| 0.0004 | 100   | 1.1386        | -                        |
| 0.0007 | 150   | 1.1591        | -                        |
| 0.0009 | 200   | 1.1744        | -                        |
| 0.0011 | 250   | 1.1342        | -                        |
| 0.0013 | 300   | 1.1091        | -                        |
| 0.0016 | 350   | 1.1879        | -                        |
| 0.0018 | 400   | 1.1480        | -                        |
| 0.002  | 450   | 1.1997        | -                        |
| 0.0022 | 500   | 1.1016        | -                        |
| 0.0024 | 550   | 1.1933        | -                        |
| 0.0027 | 600   | 1.1274        | -                        |
| 0.0029 | 650   | 1.1686        | -                        |
| 0.0031 | 700   | 1.1612        | -                        |
| 0.0033 | 750   | 1.0999        | -                        |
| 0.0036 | 800   | 1.0781        | -                        |
| 0.0038 | 850   | 1.1520        | -                        |
| 0.004  | 900   | 1.0917        | -                        |
| 0.0042 | 950   | 1.1713        | -                        |
| 0.0044 | 1000  | 1.0783        | -                        |
| 0.0047 | 1050  | 1.0969        | -                        |
| 0.0049 | 1100  | 1.0605        | -                        |
| 0.0051 | 1150  | 1.1488        | -                        |
| 0.0053 | 1200  | 1.1371        | -                        |
| 0.0056 | 1250  | 1.1642        | -                        |
| 0.0058 | 1300  | 1.1258        | -                        |
| 0.006  | 1350  | 1.1587        | -                        |
| 0.0062 | 1400  | 1.1274        | -                        |
| 0.0064 | 1450  | 1.1561        | -                        |
| 0.0067 | 1500  | 1.1141        | -                        |
| 0.0069 | 1550  | 1.1182        | -                        |
| 0.0071 | 1600  | 1.1189        | -                        |
| 0.0073 | 1650  | 1.0893        | -                        |
| 0.0076 | 1700  | 1.0842        | -                        |
| 0.0078 | 1750  | 1.0958        | -                        |
| 0.008  | 1800  | 1.0724        | -                        |
| 0.0082 | 1850  | 1.1396        | -                        |
| 0.0084 | 1900  | 1.1186        | -                        |
| 0.0087 | 1950  | 1.0889        | -                        |
| 0.0089 | 2000  | 1.1226        | -                        |
| 0      | 0     | -             | 1.1185                   |
| 0.0089 | 2000  | -             | -                        |
| 0.0091 | 2050  | 1.0828        | -                        |
| 0.0093 | 2100  | 1.0901        | -                        |
| 0.0096 | 2150  | 1.0925        | -                        |
| 0.0098 | 2200  | 1.1032        | -                        |
| 0.01   | 2250  | 1.1098        | -                        |
| 0.0102 | 2300  | 1.1128        | -                        |
| 0.0104 | 2350  | 1.0785        | -                        |
| 0.0107 | 2400  | 1.1687        | -                        |
| 0.0109 | 2450  | 1.1072        | -                        |
| 0.0111 | 2500  | 1.0975        | -                        |
| 0.0113 | 2550  | 1.0572        | -                        |
| 0.0116 | 2600  | 1.1076        | -                        |
| 0.0118 | 2650  | 1.0960        | -                        |
| 0.012  | 2700  | 1.1168        | -                        |
| 0.0122 | 2750  | 1.1010        | -                        |
| 0.0124 | 2800  | 1.0929        | -                        |
| 0.0127 | 2850  | 1.1059        | -                        |
| 0.0129 | 2900  | 1.1364        | -                        |
| 0.0131 | 2950  | 1.1287        | -                        |
| 0.0133 | 3000  | 1.1352        | -                        |
| 0.0136 | 3050  | 1.0568        | -                        |
| 0.0138 | 3100  | 1.1223        | -                        |
| 0.014  | 3150  | 1.1049        | -                        |
| 0.0142 | 3200  | 1.0955        | -                        |
| 0.0144 | 3250  | 1.1096        | -                        |
| 0.0147 | 3300  | 1.1258        | -                        |
| 0.0149 | 3350  | 1.0997        | -                        |
| 0.0151 | 3400  | 1.1119        | -                        |
| 0.0153 | 3450  | 1.0794        | -                        |
| 0.0156 | 3500  | 1.0604        | -                        |
| 0.0158 | 3550  | 1.0950        | -                        |
| 0.016  | 3600  | 1.0603        | -                        |
| 0.0162 | 3650  | 1.0765        | -                        |
| 0.0164 | 3700  | 1.1336        | -                        |
| 0.0167 | 3750  | 1.0727        | -                        |
| 0.0169 | 3800  | 1.0712        | -                        |
| 0.0171 | 3850  | 1.0986        | -                        |
| 0.0173 | 3900  | 1.1542        | -                        |
| 0.0176 | 3950  | 1.1110        | -                        |
| 0.0178 | 4000  | 1.1629        | -                        |
| 0      | 0     | -             | 1.1058                   |
| 0.0178 | 4000  | -             | -                        |
| 0.018  | 4050  | 1.1361        | -                        |
| 0.0182 | 4100  | 1.0627        | -                        |
| 0.0184 | 4150  | 1.0706        | -                        |
| 0.0187 | 4200  | 1.1362        | -                        |
| 0.0189 | 4250  | 1.0360        | -                        |
| 0.0191 | 4300  | 1.1017        | -                        |
| 0.0193 | 4350  | 1.0430        | -                        |
| 0.0196 | 4400  | 1.1209        | -                        |
| 0.0198 | 4450  | 1.0369        | -                        |
| 0.02   | 4500  | 1.1443        | -                        |
| 0.0202 | 4550  | 1.1286        | -                        |
| 0.0204 | 4600  | 1.0812        | -                        |
| 0.0207 | 4650  | 1.1069        | -                        |
| 0.0209 | 4700  | 1.1525        | -                        |
| 0.0211 | 4750  | 1.1155        | -                        |
| 0.0213 | 4800  | 1.1127        | -                        |
| 0.0216 | 4850  | 1.1191        | -                        |
| 0.0218 | 4900  | 1.0550        | -                        |
| 0.022  | 4950  | 1.1142        | -                        |
| 0.0222 | 5000  | 1.0981        | -                        |
| 0.0224 | 5050  | 1.0687        | -                        |
| 0.0227 | 5100  | 1.1681        | -                        |
| 0.0229 | 5150  | 1.0386        | -                        |
| 0.0231 | 5200  | 1.0544        | -                        |
| 0.0233 | 5250  | 1.0768        | -                        |
| 0.0236 | 5300  | 1.0534        | -                        |
| 0.0238 | 5350  | 1.0582        | -                        |
| 0.024  | 5400  | 1.1362        | -                        |
| 0.0242 | 5450  | 1.0498        | -                        |
| 0.0244 | 5500  | 1.0680        | -                        |
| 0.0247 | 5550  | 1.0720        | -                        |
| 0.0249 | 5600  | 1.0220        | -                        |
| 0.0251 | 5650  | 1.0709        | -                        |
| 0.0253 | 5700  | 1.1202        | -                        |
| 0.0256 | 5750  | 1.0859        | -                        |
| 0.0258 | 5800  | 1.0963        | -                        |
| 0.026  | 5850  | 1.1289        | -                        |
| 0.0262 | 5900  | 1.0678        | -                        |
| 0.0264 | 5950  | 1.1045        | -                        |
| 0.0267 | 6000  | 1.0949        | -                        |
| 0      | 0     | -             | 1.0948                   |
| 0.0267 | 6000  | -             | -                        |
| 0.0269 | 6050  | 1.1390        | -                        |
| 0.0271 | 6100  | 1.0970        | -                        |
| 0.0273 | 6150  | 1.1053        | -                        |
| 0.0276 | 6200  | 1.0884        | -                        |
| 0.0278 | 6250  | 1.0904        | -                        |
| 0.028  | 6300  | 1.0698        | -                        |
| 0.0282 | 6350  | 1.0825        | -                        |
| 0.0284 | 6400  | 1.0965        | -                        |
| 0.0287 | 6450  | 1.1389        | -                        |
| 0.0289 | 6500  | 1.0904        | -                        |
| 0.0291 | 6550  | 1.0232        | -                        |
| 0.0293 | 6600  | 1.0627        | -                        |
| 0.0296 | 6650  | 1.0573        | -                        |
| 0.0298 | 6700  | 1.1116        | -                        |
| 0.03   | 6750  | 1.1553        | -                        |
| 0.0302 | 6800  | 1.0242        | -                        |
| 0.0304 | 6850  | 1.0745        | -                        |
| 0.0307 | 6900  | 1.1004        | -                        |
| 0.0309 | 6950  | 1.0596        | -                        |
| 0.0311 | 7000  | 1.1150        | -                        |
| 0.0313 | 7050  | 1.0637        | -                        |
| 0.0316 | 7100  | 1.0212        | -                        |
| 0.0318 | 7150  | 1.0607        | -                        |
| 0.032  | 7200  | 1.0730        | -                        |
| 0.0322 | 7250  | 1.1427        | -                        |
| 0.0324 | 7300  | 1.0869        | -                        |
| 0.0327 | 7350  | 1.0820        | -                        |
| 0.0329 | 7400  | 1.1184        | -                        |
| 0.0331 | 7450  | 1.1179        | -                        |
| 0.0333 | 7500  | 1.1163        | -                        |
| 0.0336 | 7550  | 1.0951        | -                        |
| 0.0338 | 7600  | 1.0966        | -                        |
| 0.034  | 7650  | 1.1285        | -                        |
| 0.0342 | 7700  | 1.0277        | -                        |
| 0.0344 | 7750  | 1.0824        | -                        |
| 0.0347 | 7800  | 1.0509        | -                        |
| 0.0349 | 7850  | 1.0634        | -                        |
| 0.0351 | 7900  | 1.0859        | -                        |
| 0.0353 | 7950  | 1.1146        | -                        |
| 0.0356 | 8000  | 1.1417        | -                        |
| 0      | 0     | -             | 1.0841                   |
| 0.0356 | 8000  | -             | -                        |
| 0.0358 | 8050  | 1.0434        | -                        |
| 0.036  | 8100  | 1.0862        | -                        |
| 0.0362 | 8150  | 1.1517        | -                        |
| 0.0364 | 8200  | 1.1028        | -                        |
| 0.0367 | 8250  | 1.0706        | -                        |
| 0.0369 | 8300  | 1.0364        | -                        |
| 0.0371 | 8350  | 1.0862        | -                        |
| 0.0373 | 8400  | 1.1320        | -                        |
| 0.0376 | 8450  | 1.0878        | -                        |
| 0.0378 | 8500  | 1.0762        | -                        |
| 0.038  | 8550  | 1.0595        | -                        |
| 0.0382 | 8600  | 1.1006        | -                        |
| 0.0384 | 8650  | 1.1037        | -                        |
| 0.0387 | 8700  | 1.0901        | -                        |
| 0.0389 | 8750  | 1.0382        | -                        |
| 0.0391 | 8800  | 1.1115        | -                        |
| 0.0393 | 8850  | 1.0625        | -                        |
| 0.0396 | 8900  | 1.0674        | -                        |
| 0.0398 | 8950  | 1.0583        | -                        |
| 0.04   | 9000  | 1.1536        | -                        |
| 0.0402 | 9050  | 1.0629        | -                        |
| 0.0404 | 9100  | 1.0546        | -                        |
| 0.0407 | 9150  | 1.0728        | -                        |
| 0.0409 | 9200  | 1.1199        | -                        |
| 0.0411 | 9250  | 1.0383        | -                        |
| 0.0413 | 9300  | 1.0809        | -                        |
| 0.0416 | 9350  | 1.0430        | -                        |
| 0.0418 | 9400  | 1.1009        | -                        |
| 0.042  | 9450  | 1.0861        | -                        |
| 0.0422 | 9500  | 1.1199        | -                        |
| 0.0424 | 9550  | 1.1128        | -                        |
| 0.0427 | 9600  | 1.0533        | -                        |
| 0.0429 | 9650  | 1.0692        | -                        |
| 0.0431 | 9700  | 1.0444        | -                        |
| 0.0433 | 9750  | 1.0575        | -                        |
| 0.0436 | 9800  | 1.0387        | -                        |
| 0.0438 | 9850  | 1.0116        | -                        |
| 0.044  | 9900  | 1.0893        | -                        |
| 0.0442 | 9950  | 1.0828        | -                        |
| 0.0444 | 10000 | 1.0395        | -                        |
| 0      | 0     | -             | 1.0753                   |
| 0.0444 | 10000 | -             | -                        |

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