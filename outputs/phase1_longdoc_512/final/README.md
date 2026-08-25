---
tags:
- ColBERT
- PyLate
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:498000
- loss:CachedContrastive
base_model: nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl
pipeline_tag: sentence-similarity
library_name: PyLate
metrics:
- accuracy
model-index:
- name: PyLate model based on nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl
  results:
  - task:
      type: col-berttriplet
      name: Col BERTTriplet
    dataset:
      name: mmarco it wiki eval
      type: mmarco-it-wiki-eval
    metrics:
    - type: accuracy
      value: 0.9620000720024109
      name: Accuracy
---

# PyLate model based on nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl

This is a [PyLate](https://github.com/lightonai/pylate) model finetuned from [nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl](https://huggingface.co/nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl). It maps sentences & paragraphs to sequences of 128-dimensional dense vectors and can be used for semantic textual similarity using the MaxSim operator.

## Model Details

### Model Description
- **Model Type:** PyLate model
- **Base model:** [nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl](https://huggingface.co/nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl) <!-- at revision d241b1266c2fda05252f9ceff1dab1b959fb1e0d -->
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

| Metric       | Value     |
|:-------------|:----------|
| **accuracy** | **0.962** |

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


* Size: 498,000 training samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 11.39 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 31.95 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 31.96 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  |:----------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Cos'è l'autonomia chilometrica?</code>              | <code># Autonomia chilometrica<br><br>L'autonomia chilometrica indica lo spazio che un aeromobile può percorrere senza subire rifornimento. L’obiettivo è quello di ottenere la massima autonomia chilometrica (MAK) pertanto l’aeromobile deve viaggiare a una certa velocità e ad una certa quota in relazione alla tipologia del velivolo. Strettamente legata all'autonomia chilometrica, è presente l'autonomia oraria. La scelta di determinare l'una o l'altra dipende dalla missione che l'aeromobile deve compiere. Si fanno differenziazioni per l'autonomia tra velivoli a elica e velivoli con propulsione a getto ma in ogni caso sono presenti parametri caratteristici quali consumo specifico, orario e chilometrico.<br><br>## L'autonomia nei velivoli a elica<br><br>Nel caso di velivolo a elica il consumo specifico varia tra  e nel caso di elicotteri dipende dalla potenza che il motore trasmette all’albero. Il consumo orario dipende dal quantitativo di carburante consumato in un certo periodo di tempo secondo la relazione:<br><br>e...</code>             | <code># Collegiata di San Pietro (Massa)<br><br>La chiesa collegiata di San Pietro a Massa era un edificio religioso, demolito nel 1807 per decreto di Felice ed Elisa Bonaparte Baciocchi, che si trovava in piazza Aranci. Ad essa era legato l'oratorio di San Sebastiano, distrutto dai bombardamenti alleati del febbraio 1945. Dalla documentazione d'archivio è noto che la chiesa, già esistente in età medievale come pieve, subì importanti interventi nel corso del '500, crollò nel 1671 e venne ricostruita nello stesso luogo tra il 1697 e il 1701. I lavori di riqualificazione di piazza degli Aranci, tra il 2011 e il 2012, hanno permesso di riportarne alla luce i resti (coperti nuovamente al termine dei lavori).<br><br>## Storia<br><br>La Insigne Collegiata di San Pietro demolita nel 1807 era una grande chiesa barocca che stava per diventare cattedrale. Essa sorgeva sul luogo della più antica pieve di San Pietro crollata nel 1671 (o 1672 secondo altri documenti). La prima testimonianza che c'informa della sua esisten...</code> |
  | <code>che foto c'è sulla banconota da due dollari?</code> | <code>Sul retro di una banconota da due dollari americani c'è una foto della firma della Dichiarazione di Indipendenza. Se non sbaglio, il resto delle banconote da un dollaro ha tutte immagini di edifici. Perché la banconota da due dollari era così diversa? Domanda #31581.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>La banconota da venti dollari degli Stati Uniti ($ 20) è una denominazione della valuta statunitense. Il settimo presidente degli Stati Uniti (1829ÃÂ¢Ã‚â‚¬ââ€37), Andrew Jackson è apparso sul lato anteriore della banconota dal 1928, motivo per cui il biglietto da venti dollari è spesso chiamato Jackson, mentre il La Casa Bianca è rappresentata sul retro. L'obiettivo era quello di avere una donna sul disegno di legge da $ 20 entro il 2020, il centenario del 19° emendamento che dava alle donne il diritto di voto. A partire dall'8 aprile 2015, i quattro candidati principali erano Eleanor Roosevelt, Rosa Parks, Harriet Tubman e Wilma Mankiller. Il 12 maggio 2015, Tubman è stato annunciato come il candidato vincitore.</code>                                                                                                                                                                                                                                                                                                            |
  | <code>Quali antologie ha pubblicato?</code>               | <code># Carlos Pintado<br><br>Carlos Pintado (L'Avana, 1974) è uno scrittore cubano.<br><br>## Biografia<br><br>Laureato in Lingua e Letteratura inglese nel 1996 presso l'Istituto pedagogico di Pinar del Río, si è trasferito negli Stati Uniti nel 1997.<br><br>Nel 2006 ha vinto il premio internazionale di poesia Sant Jordi di Barcellona con il suo libro Autorretrato en azul.<br><br>Le sue poesie, racconti e articoli sono stati tradotti in inglese, tedesco, turco, italiano e francese, e sono apparsi in diverse antologie, tra cui: Ante el espejo (poesia latinoamericana, Fundación Inquietud Europea, Madrid, 2008), Adiós (Madrid, 2006), Aldabonazo en Trocadero 162 (Ed. Aduana Vieja, Madrid, 2008), Una voz en el abismo (Perù, 2007), Antología de la poesia cubana del exilio (Aduana Vieja, Madrid, 2011).<br><br>Le sue opere sono state pubblicate in diverse nazioni (Spagna, Cuba, Turchia, Messico, Germania, Perù, Argentina e Stati Uniti) su riviste in lingua spagnola, tra cui Blancomóvil,  Enfocarte,  13trenes, Decir del Agua, La Haban...</code> | <code># Traversodon stahleckeri<br><br>Traversodon stahleckeri è un terapside estinto, appartenente ai cinodonti. Visse nel Triassico medio (circa 242 - 235 milini di anni fa) e i suoi resti fossili sono stati ritrovati in Sudamerica.<br><br>## Descrizione<br><br>Questo animale doveva essere piuttosto grande e dalla struttura abbastanza robusta; il cranio era dotato di grandi finestre temporali e di una cresta sagittale scanalata. Il muso era piuttosto corto e stretto, dotato di piccoli denti incisiviformi e di due grandi canini superiori allungati. I denti postcanini erano molariformi; quelli dell'osso mascellare erano più larghi che lunghi e dotati di un solco trasversale (tranne il primo). Il solco trasversale dei postcanini era situato sulla parte posteriore dei denti superiori e sulla parte anteriore dei denti inferiori. I postcanini della mandibola erano più lunghi che larghi. Il processo cornoide della mandibola era elevato ma abbastanza sottile. Lo scheletro postcranico era caratterizzato da un omero ...</code> |
* Loss: <code>pylate.losses.cached_contrastive.CachedContrastive</code>

### Evaluation Dataset

#### Unnamed Dataset


* Size: 2,000 evaluation samples
* Columns: <code>query</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | query                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 11.41 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 31.98 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 31.98 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | query                                                                | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | negative                                                                                                                                                                                                                                                                                                                                                                                        |
  |:---------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>cosa rappresenta l'anno del segno della capra</code>           | <code>La capra arriva ottava nello zodiaco cinese. Gli anni della Capra includono: 1931, 1943, 1955, 1967, 1979, 1991, 2003, 2015, 2027... Secondo l'astrologia cinese, ogni anno è associato a un segno animale, che si verifica in un ciclo di 12 anni. Ad esempio, il 2015 è stato l'anno della Capra. Il ciclo è sempre l'anno del cavallo, l'anno della capra, poi l'anno della scimmia. Se sei nato in un anno di capra, si dice che questi siano fortunati per te: Colori fortunati: verde, rosso, viola Numeri fortunati: 2, 7</code> | <code>2017 Mercurio retrogrado. Durante l'anno 2017, Mercurio diventa retrogrado per quattro volte come di seguito: il primo è un riporto dell'anno precedente. Dal 19 dicembre 2016 all'8 gennaio 2017 - Dal segno di terra Capricorno, al segno di fuoco Sagittario. 9 aprile - 3 maggio 2017- Dal segno di terra Toro, al segno di fuoco Ariete.</code>                                      |
  | <code>posso ottenere un atto rimosso dal rapporto di credito?</code> | <code>Gli account possono essere rimossi dal tuo rapporto di credito per diversi motivi. Come per qualsiasi modifica al rapporto di credito, è meglio comprendere le cause di base e le spiegazioni per la rimozione dell'account in modo da essere preparati per eventuali sorprese future.</code>                                                                                                                                                                                                                                           | <code>1 I ritardi di pagamento scompariranno dal tuo rapporto di credito dopo sette anni. Non sarai in grado di ottenere un punteggio perfetto, o anche uno buono, se hai eventi negativi come un fallimento o un conto di riscossione sul tuo rapporto di credito. Questi difetti possono rimanere sul tuo rapporto di credito per dieci anni e possono rimanere su di esso per sempre.</code> |
  | <code>sono le isole cayman noi territorio</code>                     | <code>Le Isole Cayman (territorio britannico d'oltremare) si trovano nel Mar dei Caraibi appena a sud di Cuba occidentale e a nord-ovest della Giamaica.</code>                                                                                                                                                                                                                                                                                                                                                                               | <code>Temperatura media nelle Isole Cayman in ottobre: ​​25-31Ãâ€šÃ‚Â°C, 77-88Ãƒâ€šÃ‚Â°F. Temperatura del mare nelle Isole Cayman in ottobre: ​​29Ãƒâ€šÃ‚Â°C, 84Ãƒâ€šÃ‚Â°F. Precipitazioni medie in ottobre: ​​234 mm e 9,2 pollici. Nota: ottobre vede la fine della stagione degli uragani e segna l'inizio dell'alta stagione delle vacanze nelle Isole Cayman.</code>                       |
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
| 0.0103 | 10   | 103.5755      | -               | -        |
| 0.0206 | 20   | 71.8932       | -               | -        |
| 0.0308 | 30   | 48.3615       | -               | -        |
| 0.0411 | 40   | 28.6564       | -               | -        |
| 0.0514 | 50   | 13.1034       | -               | -        |
| 0.0617 | 60   | 6.7400        | -               | -        |
| 0.0719 | 70   | 4.6197        | -               | -        |
| 0.0822 | 80   | 3.0824        | -               | -        |
| 0.0925 | 90   | 2.1174        | -               | -        |
| 0.1028 | 100  | 1.8567        | -               | -        |
| 0      | 0    | -             | -               | 0.9205   |
| 0.1028 | 100  | -             | 0.7784          | -        |
| 0.1131 | 110  | 1.4278        | -               | -        |
| 0.1233 | 120  | 1.3173        | -               | -        |
| 0.1336 | 130  | 1.2483        | -               | -        |
| 0.1439 | 140  | 1.1166        | -               | -        |
| 0.1542 | 150  | 1.1559        | -               | -        |
| 0.1644 | 160  | 1.0608        | -               | -        |
| 0.1747 | 170  | 1.0338        | -               | -        |
| 0.1850 | 180  | 1.0253        | -               | -        |
| 0.1953 | 190  | 0.9815        | -               | -        |
| 0.2055 | 200  | 0.9472        | -               | -        |
| 0      | 0    | -             | -               | 0.9365   |
| 0.2055 | 200  | -             | 0.4218          | -        |
| 0.2158 | 210  | 0.9414        | -               | -        |
| 0.2261 | 220  | 0.9140        | -               | -        |
| 0.2364 | 230  | 0.9895        | -               | -        |
| 0.2467 | 240  | 0.9259        | -               | -        |
| 0.2569 | 250  | 0.8899        | -               | -        |
| 0.2672 | 260  | 0.8227        | -               | -        |
| 0.2775 | 270  | 0.9031        | -               | -        |
| 0.2878 | 280  | 0.8600        | -               | -        |
| 0.2980 | 290  | 0.8759        | -               | -        |
| 0.3083 | 300  | 0.8077        | -               | -        |
| 0      | 0    | -             | -               | 0.9460   |
| 0.3083 | 300  | -             | 0.3521          | -        |
| 0.3186 | 310  | 0.8757        | -               | -        |
| 0.3289 | 320  | 0.8645        | -               | -        |
| 0.3392 | 330  | 0.8149        | -               | -        |
| 0.3494 | 340  | 0.7985        | -               | -        |
| 0.3597 | 350  | 0.8450        | -               | -        |
| 0.3700 | 360  | 0.7748        | -               | -        |
| 0.3803 | 370  | 0.7877        | -               | -        |
| 0.3905 | 380  | 0.7548        | -               | -        |
| 0.4008 | 390  | 0.7688        | -               | -        |
| 0.4111 | 400  | 0.7749        | -               | -        |
| 0      | 0    | -             | -               | 0.9515   |
| 0.4111 | 400  | -             | 0.3065          | -        |
| 0.4214 | 410  | 0.7407        | -               | -        |
| 0.4317 | 420  | 0.7332        | -               | -        |
| 0.4419 | 430  | 0.7533        | -               | -        |
| 0.4522 | 440  | 0.7334        | -               | -        |
| 0.4625 | 450  | 0.7270        | -               | -        |
| 0.4728 | 460  | 0.7287        | -               | -        |
| 0.4830 | 470  | 0.7287        | -               | -        |
| 0.4933 | 480  | 0.7180        | -               | -        |
| 0.5036 | 490  | 0.7235        | -               | -        |
| 0.5139 | 500  | 0.7084        | -               | -        |
| 0      | 0    | -             | -               | 0.9535   |
| 0.5139 | 500  | -             | 0.2868          | -        |
| 0.5242 | 510  | 0.7205        | -               | -        |
| 0.5344 | 520  | 0.7186        | -               | -        |
| 0.5447 | 530  | 0.6932        | -               | -        |
| 0.5550 | 540  | 0.7070        | -               | -        |
| 0.5653 | 550  | 0.6947        | -               | -        |
| 0.5755 | 560  | 0.7217        | -               | -        |
| 0.5858 | 570  | 0.7007        | -               | -        |
| 0.5961 | 580  | 0.6736        | -               | -        |
| 0.6064 | 590  | 0.6726        | -               | -        |
| 0.6166 | 600  | 0.7156        | -               | -        |
| 0      | 0    | -             | -               | 0.9565   |
| 0.6166 | 600  | -             | 0.2729          | -        |
| 0.6269 | 610  | 0.6830        | -               | -        |
| 0.6372 | 620  | 0.6926        | -               | -        |
| 0.6475 | 630  | 0.6624        | -               | -        |
| 0.6578 | 640  | 0.7127        | -               | -        |
| 0.6680 | 650  | 0.6901        | -               | -        |
| 0.6783 | 660  | 0.7144        | -               | -        |
| 0.6886 | 670  | 0.6990        | -               | -        |
| 0.6989 | 680  | 0.6673        | -               | -        |
| 0.7091 | 690  | 0.6436        | -               | -        |
| 0.7194 | 700  | 0.6977        | -               | -        |
| 0      | 0    | -             | -               | 0.9585   |
| 0.7194 | 700  | -             | 0.2636          | -        |
| 0.7297 | 710  | 0.7335        | -               | -        |
| 0.7400 | 720  | 0.6650        | -               | -        |
| 0.7503 | 730  | 0.6798        | -               | -        |
| 0.7605 | 740  | 0.6631        | -               | -        |
| 0.7708 | 750  | 0.6515        | -               | -        |
| 0.7811 | 760  | 0.6595        | -               | -        |
| 0.7914 | 770  | 0.6870        | -               | -        |
| 0.8016 | 780  | 0.6610        | -               | -        |
| 0.8119 | 790  | 0.6586        | -               | -        |
| 0.8222 | 800  | 0.6902        | -               | -        |
| 0      | 0    | -             | -               | 0.9620   |
| 0.8222 | 800  | -             | 0.2586          | -        |
| 0.8325 | 810  | 0.7027        | -               | -        |
| 0.8428 | 820  | 0.6664        | -               | -        |
| 0.8530 | 830  | 0.6642        | -               | -        |
| 0.8633 | 840  | 0.6534        | -               | -        |
| 0.8736 | 850  | 0.6646        | -               | -        |
| 0.8839 | 860  | 0.6640        | -               | -        |
| 0.8941 | 870  | 0.7163        | -               | -        |
| 0.9044 | 880  | 0.6605        | -               | -        |
| 0.9147 | 890  | 0.6169        | -               | -        |
| 0.9250 | 900  | 0.6632        | -               | -        |
| 0      | 0    | -             | -               | 0.9620   |
| 0.9250 | 900  | -             | 0.2531          | -        |
| 0.9353 | 910  | 0.6719        | -               | -        |
| 0.9455 | 920  | 0.6481        | -               | -        |
| 0.9558 | 930  | 0.6556        | -               | -        |
| 0.9661 | 940  | 0.6515        | -               | -        |
| 0.9764 | 950  | 0.6751        | -               | -        |
| 0.9866 | 960  | 0.6851        | -               | -        |
| 0.9969 | 970  | 0.6934        | -               | -        |

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