---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:999998
- loss:Distillation
- loss:CachedContrastive
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
      value: 1.109534740447998
      name: Kl Divergence
---

# PyLate

This is a [PyLate](https://github.com/lightonai/pylate) model trained on the kd and contrastive datasets. It maps sentences & paragraphs to sequences of 128-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Document Length:** 512 tokens
- **Query Length:** 32 tokens
- **Output Dimensionality:** 128 tokens
- **Similarity Function:** MaxSim
- **Training Datasets:**
    - kd
    - contrastive
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
| **kl_divergence** | **1.1095** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets

#### kd

* Dataset: kd
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

#### contrastive

* Dataset: contrastive
* Size: 100,000 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 12.07 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                                    | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | negative                                                                                                                                                                                                                                                                                                                                                             |
  |:---------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>dov'è l'hotel con vista?</code>                    | <code>La trama è semplice: Jack Torrance (Jack Nicholson) diventa il custode dell'Overlook Hotel nelle montagne isolate del Colorado. Jack, essendo un padre di famiglia, porta sua moglie (Shelley Duvall) e suo figlio (Danny Lloyd) in albergo per tenergli compagnia durante le lunghe notti isolate. Durante il loro soggiorno, accadono cose strane quando il figlio di Jack, Danny, vede immagini raccapriccianti alimentate da una forza chiamata "lo splendore" e Jack ne è pesantemente colpito. Insieme al blocco dello scrittore e ai demoni dell'hotel che lo perseguitano, Jack ha un completo crollo mentale e la situazione prende una sinistra svolta per il peggio.</code>                                                                                                                                   | <code>Scopri cosa rende ufficiale un certificato di nascita. Quando ti prepari per le tante pietre miliari della vita che richiedono una prova di identificazione personale, non farlo. trascurare la necessità di presentare una copia ufficiale del certificato di nascita.</code>                                                                                 |
  | <code>cosa significa tsh nell'analisi del sangue?</code> | <code>Ormone stimolante la tiroide (TSH). Guida. Un esame del sangue dell'ormone stimolante la tiroide (TSH) viene utilizzato per verificare la presenza di problemi alla ghiandola tiroidea. Il TSH viene prodotto quando l'ipotalamo rilascia una sostanza chiamata ormone di rilascio della tireotropina (TRH). Il TRH innesca quindi la ghiandola pituitaria per rilasciare TSH. Il TSH fa sì che la ghiandola tiroide produca due ormoni: triiodotironina (T3) e tiroxina (T4). T3 e T4 aiutano a controllare il metabolismo del corpo. Trova la causa di una ghiandola tiroidea ipoattiva (ipotiroidismo). 2 I livelli di TSH possono aiutare a determinare se l'ipotiroidismo è dovuto a una ghiandola tiroidea danneggiata oa qualche altra causa (come un problema con la ghiandola pituitaria o l'ipotalamo).</code> | <code>valori normali di tsh l'intervallo normale del rapporto del test di tsh può variare tra 0 4 4 0 milli unità internazionali per litro il valore di tsh dipende da vari fattori come i rapporti di laboratorio relativi ai sintomi e al trattamento per la condizione</code>                                                                                     |
  | <code>quali anellidi vivono nell'oceano</code>           | <code>Gli anellidi si trovano in tutto il mondo in tutti i tipi di habitat, in particolare nelle acque oceaniche, nelle acque dolci e nei terreni umidi. La maggior parte dei policheti vive nell'oceano, dove galleggia, scava, vaga sul fondo o vive in tubi che costruiscono; i loro colori vanno dal brillante all'opaco e alcune specie possono produrre luce. Il piumino (Manayunkia speciosa) abita i Grandi Laghi e alcuni fiumi degli Stati Uniti.</code>                                                                                                                                                                                                                                                                                                                                                             | <code>Temperatura dell'acqua della costa dell'Oceano Atlantico mensile: Temperatura dell'acqua dell'Oceano Atlantico a gennaio. Temperatura dell'acqua dell'Oceano Atlantico a febbraio. Temperatura dell'acqua dell'Oceano Atlantico a marzo. Temperatura dell'acqua dell'Oceano Atlantico ad aprile. Temperatura dell'acqua dell'Oceano Atlantico a maggio.</code> |
* Loss: <code>pylate.losses.cached_contrastive.CachedContrastive</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 1.0
- `learning_rate`: 2e-06
- `warmup_steps`: 0.01
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
- `learning_rate`: 2e-06
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.01
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
| Epoch     | Step     | Training Loss | kd-holdout_kl_divergence |
|:---------:|:--------:|:-------------:|:------------------------:|
| 0.0002    | 50       | 1.0382        | -                        |
| 0.0004    | 100      | 0.9728        | -                        |
| 0.0006    | 150      | 0.9731        | -                        |
| 0.0008    | 200      | 1.0121        | -                        |
| 0.001     | 250      | 1.0293        | -                        |
| 0.0012    | 300      | 1.0618        | -                        |
| 0.0014    | 350      | 1.0125        | -                        |
| 0.0016    | 400      | 0.9260        | -                        |
| 0.0018    | 450      | 0.9249        | -                        |
| 0.002     | 500      | 1.1016        | -                        |
| 0         | 0        | -             | 1.1427                   |
| 0.002     | 500      | -             | -                        |
| 0.0022    | 550      | 1.0417        | -                        |
| 0.0024    | 600      | 1.0145        | -                        |
| 0.0026    | 650      | 1.0236        | -                        |
| 0.0028    | 700      | 0.9882        | -                        |
| 0.003     | 750      | 1.0197        | -                        |
| 0.0032    | 800      | 0.9853        | -                        |
| 0.0034    | 850      | 0.9507        | -                        |
| 0.0036    | 900      | 0.9434        | -                        |
| 0.0038    | 950      | 1.1349        | -                        |
| 0.004     | 1000     | 0.9695        | -                        |
| 0         | 0        | -             | 1.1327                   |
| 0.004     | 1000     | -             | -                        |
| 0.0042    | 1050     | 1.0855        | -                        |
| 0.0044    | 1100     | 1.0314        | -                        |
| 0.0046    | 1150     | 1.0612        | -                        |
| 0.0048    | 1200     | 1.1407        | -                        |
| 0.005     | 1250     | 1.0300        | -                        |
| 0.0052    | 1300     | 1.0708        | -                        |
| 0.0054    | 1350     | 1.0284        | -                        |
| 0.0056    | 1400     | 1.0221        | -                        |
| 0.0058    | 1450     | 0.9330        | -                        |
| 0.006     | 1500     | 1.0126        | -                        |
| 0         | 0        | -             | 1.1263                   |
| 0.006     | 1500     | -             | -                        |
| 0.0062    | 1550     | 1.0826        | -                        |
| 0.0064    | 1600     | 1.0069        | -                        |
| 0.0066    | 1650     | 1.0490        | -                        |
| 0.0068    | 1700     | 0.9712        | -                        |
| 0.007     | 1750     | 0.9995        | -                        |
| 0.0072    | 1800     | 1.1085        | -                        |
| 0.0074    | 1850     | 1.0571        | -                        |
| 0.0076    | 1900     | 0.8824        | -                        |
| 0.0078    | 1950     | 0.9436        | -                        |
| **0.008** | **2000** | **1.042**     | **-**                    |
| 0         | 0        | -             | 1.1215                   |
| **0.008** | **2000** | **-**         | **-**                    |
| 0.0082    | 2050     | 0.8972        | -                        |
| 0.0084    | 2100     | 1.0481        | -                        |
| 0.0086    | 2150     | 0.9877        | -                        |
| 0.0088    | 2200     | 1.0232        | -                        |
| 0.009     | 2250     | 1.0150        | -                        |
| 0.0092    | 2300     | 1.0111        | -                        |
| 0.0094    | 2350     | 1.0297        | -                        |
| 0.0096    | 2400     | 1.0106        | -                        |
| 0.0098    | 2450     | 1.0746        | -                        |
| 0.01      | 2500     | 0.9936        | -                        |
| 0         | 0        | -             | 1.1175                   |
| 0.01      | 2500     | -             | -                        |
| 0.0102    | 2550     | 0.9698        | -                        |
| 0.0104    | 2600     | 1.1069        | -                        |
| 0.0106    | 2650     | 0.9949        | -                        |
| 0.0108    | 2700     | 1.0606        | -                        |
| 0.011     | 2750     | 1.0282        | -                        |
| 0.0112    | 2800     | 0.9940        | -                        |
| 0.0114    | 2850     | 1.0088        | -                        |
| 0.0116    | 2900     | 1.0379        | -                        |
| 0.0118    | 2950     | 0.9058        | -                        |
| 0.012     | 3000     | 1.0706        | -                        |
| 0         | 0        | -             | 1.1149                   |
| 0.012     | 3000     | -             | -                        |
| 0.0122    | 3050     | 0.9957        | -                        |
| 0.0124    | 3100     | 0.9399        | -                        |
| 0.0126    | 3150     | 0.9483        | -                        |
| 0.0128    | 3200     | 0.9923        | -                        |
| 0.013     | 3250     | 0.9051        | -                        |
| 0.0132    | 3300     | 0.9962        | -                        |
| 0.0134    | 3350     | 0.9659        | -                        |
| 0.0136    | 3400     | 0.8749        | -                        |
| 0.0138    | 3450     | 0.9912        | -                        |
| 0.014     | 3500     | 0.9669        | -                        |
| 0         | 0        | -             | 1.1120                   |
| 0.014     | 3500     | -             | -                        |
| 0.0142    | 3550     | 0.9475        | -                        |
| 0.0144    | 3600     | 1.0022        | -                        |
| 0.0146    | 3650     | 1.0534        | -                        |
| 0.0148    | 3700     | 0.9702        | -                        |
| 0.015     | 3750     | 1.0257        | -                        |
| 0.0152    | 3800     | 0.9371        | -                        |
| 0.0154    | 3850     | 1.0842        | -                        |
| 0.0156    | 3900     | 1.0428        | -                        |
| 0.0158    | 3950     | 1.0495        | -                        |
| 0.016     | 4000     | 0.9542        | -                        |
| 0         | 0        | -             | 1.1095                   |
| 0.016     | 4000     | -             | -                        |

* The bold row denotes the saved checkpoint.

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