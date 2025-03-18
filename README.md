# Rag

## Dependencies
`Python 3.12` was used.

To create a virtual environment and install dependencies from `requirements.txt` run:
```
source ./setup.sh
```

Activate virtual environment:
```
source env/bin/activate
```
Tools used:
- Scrapy
- Selenium
- scrapy-selenium
- PyMUPDF
- spaCy
- OCRmyPDF, tesseract

```
yay -Sy ocrmypdf tesseract
sudo pacman -S tesseract-data-ron
```

## Crawler
```
scrapy crawl upb
```

## Docker setup
Install `docker` and `docker-compose`
 ```
 sudo pacman -Sy docker docker-compose
 sudo systemctl enable docker.service
 sudo systemctl start docker.service  
 sudo usermod -aG docker $USER
 ```
To [set up a custom admin password](https://opensearch.org/docs/latest/security/configuration/demo-configuration/#setting-up-a-custom-admin-password), do one of the following:

- run: `export OPENSEARCH_INITIAL_ADMIN_PASSWORD=<passwd>`
- create an `.env` file with: `OPENSEARCH_INITIAL_ADMIN_PASSWORD=<passwd>`

Create and start the cluster as a background process
```
docker compose up -d
```

## OpenSearch Dashboard
Open: `http://localhost:5601/` \
Default username: `admin` \
Password: `$OPENSEARCH_INITIAL_ADMIN_PASSWORD`

## OpenSearch setup
#### [Neural search tutorial](https://opensearch.org/docs/latest/search-plugins/neural-search-tutorial/)

#### Update ML-related cluster settings
```
PUT _cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99",
    "plugins.ml_commons.allow_registering_model_via_url": "true"
  }
}
```

### Registering a model

#### Register a model group
```
POST /_plugins/_ml/model_groups/_register
{
  "name": "NLP_model_group",
  "description": "A model group for NLP models",
  "access_mode": "public"
}
```

You will need the `model_group_id`
```
{
  "model_group_id": "ckKnd5UB_e6dONcE9pdb",
  "status": "CREATED"
}
```

To get all model groups:
```
POST /_plugins/_ml/model_groups/_search
{
  "query": {
    "match_all": {}
  }
}
```


#### Register the model to the model group
I used the [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) sentence-transformer model from Hugging Face, trained on data for 50+ languages, including Romanian.

```
POST /_plugins/_ml/models/_register
{
  "name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "version": "1.0.1",
  "model_group_id": "RWo2V5UB7VJulTW8i0FV",
  "model_format": "TORCH_SCRIPT"
}
```
Response:
```
{
  "task_id": "f0Krd5UB_e6dONcEBpec",
  "status": "CREATED"
}
```
Use the `task_id`:
```
GET /_plugins/_ml/tasks/f0Krd5UB_e6dONcEBpec
```

```
{
  "model_id": "g0Krd5UB_e6dONcEC5dk",
  "task_type": "REGISTER_MODEL",
  "function_name": "TEXT_EMBEDDING",
  "state": "COMPLETED",
  "worker_node": [
    "CpF0gs_vT1uHwx_LxmrMVg"
  ],
  "create_time": 1741469451791,
  "last_update_time": 1741469598173,
  "is_async": true
}
```

#### Deploy the model
```
POST /_plugins/_ml/models/g0Krd5UB_e6dONcEC5dk/_deploy
```

### Create an ingest pipeline

Create an ingest pipeline to transform the text into embeddings using the registered embedding model before storing it.

You will need the `model_id` of the model you registered.

```
PUT /_ingest/pipeline/nlp-ingest-pipeline
{
  "description": "An NLP ingest pipeline",
  "processors": [
    {
      "text_embedding": {
        "model_id": "g0Krd5UB_e6dONcEC5dk",
        "field_map": {
          "text": "embedding"
        }
      }
    }
  ]
}
```

The pipeline automatically generates an embedding for `"text"` and stores it in the `"embedding"` field.


### Create a KNN index
```
PUT /rag-knn-index
{
  "settings": {
    "index.knn": true,
    "default_pipeline": "nlp-ingest-pipeline"
  },
  "mappings": {
    "properties": {
      "id": {
        "type": "text"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "engine": "lucene",
          "space_type": "l2",
          "name": "hnsw",
          "parameters": {}
        }
      },
      "text": {
        "type": "text"
      },
      "url": {
        "type": "keyword"
      },
      "type": {
        "type": "keyword"
      },
      "filename": {
        "type": "keyword"
      },
      "page_number": {
        "type": "integer"
      }
    }
  }
}
```

## Search data

Get all elements
```
GET /rag-knn-index/_search
{
  "query": {
    "match_all": {}
  }
}
```

Get number of elements
```
GET /rag-knn-index/_count
{
  "query": {
    "match_all": {}
  }
}
```

Semantic search
```
GET /rag-knn-index/_search
{
  "_source": {
    "excludes": [
      "embedding"
    ]
  },
  "query": {
    "neural": {
      "embedding": {
        "query_text": "control financiar preventiv",
        "model_id": "g0Krd5UB_e6dONcEC5dk",
        "k": 5
      }
    }
  }
}
```

## Ingest data

