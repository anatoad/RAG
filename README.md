# Rag

## Description


## Crawler
```
scrapy crawl upb
```

## Dependencies
Replicate the environment:
```
pip install -r requirements.txt
```

Tools used:
- Scrapy
- Selenium
- scrapy-selenium
- PyMUPDF
- spaCy
- OCRmyPDF

```
yay -Sy ocrmypdf tesseract
sudo pacman -S tesseract-data-ron
```

## Docker

 ```
 sudo pacman -Sy docker docker-compose
 sudo systemctl enable docker.service
 sudo systemctl start docker.service  
 sudo usermod -aG docker $USER
 docker compose up -d
 ```
[Setting up a custom admin password](https://opensearch.org/docs/latest/security/configuration/demo-configuration/#setting-up-a-custom-admin-password)
```
export OPENSEARCH_INITIAL_ADMIN_PASSWORD="<passwd>"
```

## OpenSearch setup
#### Update ML-related cluster settings:
```
PUT _cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99"
  }
}
```

### [Neural search tutorial](https://opensearch.org/docs/latest/search-plugins/neural-search-tutorial/)

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

```
POST /_plugins/_ml/model_groups/_search
{
  "query": {
    "match_all": {}
  }
}
```

```
{
  "model_group_id": "RWo2V5UB7VJulTW8i0FV",
  "status": "CREATED"
}
```

#### Register the model to the model group:
```
POST /_plugins/_ml/models/_register
{
  "name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "version": "1.0.1",
  "model_group_id": "RWo2V5UB7VJulTW8i0FV",
  "model_format": "TORCH_SCRIPT"
}
```

```
{
  "task_id": "i31YV5UBxZa1z7yEnrPw",
  "status": "CREATED"
}
```

```
GET /_plugins/_ml/tasks/i31YV5UBxZa1z7yEnrPw
```

```
{
  "model_id": "j31YV5UBxZa1z7yEpLM-",
  "task_type": "REGISTER_MODEL",
  "function_name": "TEXT_EMBEDDING",
  "state": "COMPLETED",
  "worker_node": [
    "fUYVAX5xSGelP5vt1gVnwg"
  ],
  "create_time": 1740927180392,
  "last_update_time": 1740927493212,
  "is_async": true
}
```

#### Deploy the model:
```
POST /_plugins/_ml/models/j31YV5UBxZa1z7yEpLM-/_deploy
```

### Create an ingest pipeline
```
PUT /_ingest/pipeline/nlp-ingest-pipeline
{
  "description": "An NLP ingest pipeline",
  "processors": [
    {
      "text_embedding": {
        "model_id": "j31YV5UBxZa1z7yEpLM-",
        "field_map": {
          "text": "embedding"
        }
      }
    }
  ]
}
```

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
        "type": "keyword",
      },
      "page_number": {
        "type": "integer"
      },
    }
  }
}
```

### Search data
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
        "model_id": "j31YV5UBxZa1z7yEpLM-",
        "k": 5
      }
    }
  }
}
```
