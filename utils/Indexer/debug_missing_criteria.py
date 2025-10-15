#!/usr/bin/env python3
"""
Debug missing criteria in Elasticsearch
Usage: python debug_missing_criteria.py --config path/to/config.json
"""
import argparse
import json
from pathlib import Path
from elasticsearch import Elasticsearch


def load_config(path: str) -> dict:
    """Load config and normalize format"""
    cfg = json.loads(Path(path).read_text())
    
    # Check if it's the indexer config format (has elasticsearch.hosts as list)
    if "elasticsearch" in cfg and isinstance(cfg["elasticsearch"].get("hosts"), list):
        # Indexer format - already good
        return cfg
    
    # Otherwise it's matcher config format - needs conversion
    if "elasticsearch" in cfg:
        es_conf = cfg["elasticsearch"]
        # Convert matcher format to indexer format
        normalized = {
            "elasticsearch": {
                "hosts": [es_conf.get("host", "https://localhost:9200")],
                "username": es_conf.get("username"),
                "password": es_conf.get("password"),
                "request_timeout": es_conf.get("request_timeout", 300),
                "retry_on_timeout": es_conf.get("retry_on_timeout", True),
                "max_retries": 3
            }
        }
        
        # Add ca_certs if docker_certs is in paths
        if "paths" in cfg and "docker_certs" in cfg["paths"]:
            normalized["elasticsearch"]["ca_certs"] = cfg["paths"]["docker_certs"]
        elif "ca_certs" in es_conf:
            normalized["elasticsearch"]["ca_certs"] = es_conf["ca_certs"]
        
        # Keep other useful fields
        normalized["paths"] = cfg.get("paths", {})
        normalized["elasticsearch_indices"] = {
            "index_trials": es_conf.get("index_trials", "clinical_trials"),
            "index_trials_eligibility": es_conf.get("index_trials_eligibility", "trials_eligibility")
        }
        
        return normalized
    
    return cfg


def make_es_client(cfg: dict) -> Elasticsearch:
    """Create ES client from normalized config"""
    es_conf = cfg["elasticsearch"]
    
    es_kwargs = {
        "hosts": es_conf["hosts"],
        "basic_auth": (es_conf["username"], es_conf["password"]),
        "request_timeout": es_conf.get("request_timeout", 300),
        "retry_on_timeout": es_conf.get("retry_on_timeout", True),
        "max_retries": es_conf.get("max_retries", 3),
        "verify_certs": True,
    }
    
    if "ca_certs" in es_conf:
        es_kwargs["ca_certs"] = es_conf["ca_certs"]
    
    return Elasticsearch(**es_kwargs)


def check_trial_in_index(es: Elasticsearch, index_name: str, nct_id: str):
    """Check if a specific NCT ID exists in the index and how"""
    
    print(f"\n{'='*70}")
    print(f"CHECKING: {nct_id} in {index_name}")
    print(f"{'='*70}")
    
    # Try multiple query strategies
    queries = [
        ("Exact term match", {"term": {"nct_id": nct_id}}),
        ("Keyword match", {"term": {"nct_id.keyword": nct_id}}),
        ("Text match", {"match": {"nct_id": nct_id}}),
        ("Wildcard ID", {"wildcard": {"_id": f"*{nct_id}*"}}),
        ("Match all + filter", {
            "bool": {
                "filter": {"term": {"nct_id": nct_id}}
            }
        }),
    ]
    
    found = False
    for query_name, query_body in queries:
        try:
            response = es.search(
                index=index_name,
                body={
                    "size": 5,
                    "query": query_body,
                    "_source": True
                }
            )
            
            hits = response['hits']['total']['value']
            print(f"\n{query_name}: {hits} hits")
            
            if hits > 0:
                found = True
                print(f"  ✓ FOUND!")
                
                for i, hit in enumerate(response['hits']['hits']):
                    doc = hit['_source']
                    print(f"\n  Document {i+1}:")
                    print(f"    _id: {hit['_id']}")
                    print(f"    _score: {hit.get('_score', 'N/A')}")
                    print(f"    Fields: {list(doc.keys())}")
                    
                    # Check NCT ID field
                    if 'nct_id' in doc:
                        print(f"    nct_id value: '{doc['nct_id']}'")
                    elif 'NCTId' in doc:
                        print(f"    NCTId value: '{doc['NCTId']}'")
                    else:
                        print(f"    ⚠️ No nct_id field found!")
                    
                    # Check criteria content
                    criteria_fields = [k for k in doc.keys() if 'criteria' in k.lower() or 'chunk' in k.lower()]
                    print(f"    Criteria/chunk fields: {criteria_fields}")
                    
                    for field in criteria_fields[:5]:  # Show first 5
                        content = doc[field]
                        if isinstance(content, list):
                            print(f"      {field}: list with {len(content)} items")
                            if content:
                                print(f"        First item: {str(content[0])[:100]}...")
                        else:
                            content_str = str(content)
                            print(f"      {field}: {len(content_str)} chars")
                            if len(content_str) < 100:
                                print(f"        Content: {content_str}")
                            else:
                                print(f"        Preview: {content_str[:200]}...")
                
                break  # Found it, no need to continue
                
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
    
    if not found:
        print(f"\n  ❌ NOT FOUND in index using any query method")
    
    return found


def check_file_vs_index(es: Elasticsearch, index_name: str, nct_id: str, trials_folder: str):
    """Compare file content with what's in the index"""
    
    print(f"\n{'='*70}")
    print(f"FILE vs INDEX COMPARISON: {nct_id}")
    print(f"{'='*70}")
    
    # Check criteria folder
    criteria_folder = Path(trials_folder) / "processed_criteria" / "processed_criteria" / nct_id
    
    if not criteria_folder.exists():
        print(f"\n❌ Criteria folder NOT found: {criteria_folder}")
        return
    
    print(f"\n✓ Criteria folder exists: {criteria_folder}")
    
    # Get all JSON files in the folder
    criteria_files = list(criteria_folder.glob("*.json"))
    print(f"  Found {len(criteria_files)} criteria files")
    
    if not criteria_files:
        print("  ❌ No criteria files found!")
        return
        
    # Load and show content from first few files
    print("\n📄 Sample Criteria Files:")
    for file_path in criteria_files[:3]:  # Show first 3 files
        try:
            with open(file_path, 'r') as f:
                criteria_data = json.load(f)
            
            print(f"\n  File: {file_path.name}")
            print(f"  Size: {file_path.stat().st_size:,} bytes")
            
            if isinstance(criteria_data, dict):
                print(f"  Keys: {list(criteria_data.keys())}")
                
                # Check NCT ID in file
                nct_in_file = criteria_data.get('nct_id')
                if nct_in_file and nct_in_file != nct_id:
                    print(f"  ⚠️ NCT ID MISMATCH: file has '{nct_in_file}' but expected '{nct_id}'")
                
                # Show criterion text
                if 'criterion' in criteria_data:
                    print(f"  Criterion: {criteria_data['criterion'][:100]}...")
                
                # Check eligibility type
                if 'eligibility_type' in criteria_data:
                    print(f"  Type: {criteria_data['eligibility_type']}")
                
            else:
                print(f"  ⚠️ Unexpected format: {type(criteria_data)}")
                
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error in {file_path.name}: {e}")
            continue
        except Exception as e:
            print(f"  ❌ Error reading {file_path.name}: {e}")
            continue
    
    # Check remaining files count
    if len(criteria_files) > 3:
        print(f"\n  ... and {len(criteria_files)-3} more criteria files")
    
    # Now check index
    print(f"\n🔍 Index Status:")
    in_index = check_trial_in_index(es, index_name, nct_id)
    
    if not in_index:
        print(f"\n⚠️ CONCLUSION: Files exist but NOT indexed!")
        print(f"   → This trial needs to be re-indexed")
        print(f"   → Run: python reindex_single_trial.py --nct-id {nct_id}")
    else:
        print(f"\n✓ CONCLUSION: Trial is properly indexed")
    
    # Now check index
    print(f"\n🔍 Index Status:")
    in_index = check_trial_in_index(es, index_name, nct_id)
    
    if not in_index:
        print(f"\n⚠️ CONCLUSION: File exists but NOT indexed!")
        print(f"   → This trial needs to be re-indexed")
        print(f"   → Run: python reindex_single_trial.py --nct-id {nct_id}")
    else:
        print(f"\n✓ CONCLUSION: Trial is properly indexed")


def get_sample_documents(es: Elasticsearch, index_name: str, n=3):
    """Get sample documents to understand structure"""
    
    print(f"\n{'='*70}")
    print(f"SAMPLE DOCUMENTS from {index_name}")
    print(f"{'='*70}")
    
    try:
        response = es.search(
            index=index_name,
            body={
                "size": n,
                "query": {"match_all": {}}
            }
        )
        
        total = response['hits']['total']['value']
        print(f"\nTotal documents in index: {total:,}")
        
        for i, hit in enumerate(response['hits']['hits'], 1):
            doc = hit['_source']
            print(f"\nSample {i}:")
            print(f"  _id: {hit['_id']}")
            print(f"  Fields: {list(doc.keys())}")
            
            # Show NCT ID
            nct_id = doc.get('nct_id') or doc.get('NCTId') or doc.get('nctId') or 'N/A'
            print(f"  NCT ID: {nct_id}")
            
            # Show size of criteria fields
            for key in doc.keys():
                if 'criteria' in key.lower() or 'chunk' in key.lower():
                    content = doc[key]
                    if isinstance(content, list):
                        print(f"  {key}: list with {len(content)} items")
                    else:
                        print(f"  {key}: {len(str(content))} chars")
    
    except Exception as e:
        print(f"Error getting samples: {e}")


def find_all_nct_ids_in_index(es: Elasticsearch, index_name: str):
    """Get all NCT IDs actually in the index"""
    
    print(f"\n{'='*70}")
    print(f"ALL NCT IDs in {index_name}")
    print(f"{'='*70}")
    
    try:
        # Use scroll API for large result sets
        response = es.search(
            index=index_name,
            body={
                "size": 1000,
                "_source": ["nct_id", "NCTId", "nctId"],
                "query": {"match_all": {}}
            },
            scroll='2m'
        )
        
        scroll_id = response['_scroll_id']
        nct_ids = set()
        
        # Process first batch
        for hit in response['hits']['hits']:
            doc = hit['_source']
            nct_id = doc.get('nct_id') or doc.get('NCTId') or doc.get('nctId')
            if nct_id:
                nct_ids.add(nct_id)
        
        # Process remaining batches
        while len(response['hits']['hits']) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll='2m')
            for hit in response['hits']['hits']:
                doc = hit['_source']
                nct_id = doc.get('nct_id') or doc.get('NCTId') or doc.get('nctId')
                if nct_id:
                    nct_ids.add(nct_id)
        
        # Clear scroll
        es.clear_scroll(scroll_id=scroll_id)
        
        print(f"\nTotal unique NCT IDs found: {len(nct_ids)}")
        print(f"\nFirst 20 NCT IDs:")
        for nct_id in sorted(list(nct_ids))[:20]:
            print(f"  - {nct_id}")
        
        return nct_ids
        
    except Exception as e:
        print(f"Error finding NCT IDs: {e}")
        return set()


def check_missing_trials(es: Elasticsearch, index_name: str, expected_nct_ids: list, trials_folder: str):
    """Check which trials from expected list are missing from index"""
    
    print(f"\n{'='*70}")
    print(f"CHECKING FOR MISSING TRIALS")
    print(f"{'='*70}")
    
    # Get all NCT IDs in index
    indexed_ids = find_all_nct_ids_in_index(es, index_name)
    
    expected_set = set(expected_nct_ids)
    missing_ids = expected_set - indexed_ids
    
    print(f"\n📊 SUMMARY:")
    print(f"  Expected trials: {len(expected_set)}")
    print(f"  Indexed trials: {len(indexed_ids)}")
    print(f"  Missing trials: {len(missing_ids)}")
    print(f"  Coverage: {len(indexed_ids)/len(expected_set)*100:.1f}%")
    
    if missing_ids:
        print(f"\n❌ MISSING TRIALS ({len(missing_ids)}):")
        missing_with_files = []
        missing_no_files = []
        
        for nct_id in sorted(list(missing_ids)):
            file_path = Path(trials_folder) / f"{nct_id}_criteria.json"
            if not file_path.exists():
                file_path = Path(trials_folder) / f"{nct_id}.json"
            
            if file_path.exists():
                missing_with_files.append(nct_id)
            else:
                missing_no_files.append(nct_id)
        
        if missing_with_files:
            print(f"\n  Missing but FILE EXISTS ({len(missing_with_files)}) - NEED REINDEXING:")
            for nct_id in missing_with_files[:20]:
                print(f"    - {nct_id}")
            if len(missing_with_files) > 20:
                print(f"    ... and {len(missing_with_files) - 20} more")
        
        if missing_no_files:
            print(f"\n  Missing and NO FILE ({len(missing_no_files)}) - NEED PROCESSING:")
            for nct_id in missing_no_files[:20]:
                print(f"    - {nct_id}")
            if len(missing_no_files) > 20:
                print(f"    ... and {len(missing_no_files) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Debug missing criteria in Elasticsearch")
    parser.add_argument("--config", required=True, help="Path to config JSON (either format)")
    parser.add_argument("--index", help="Index name to check (defaults from config)")
    parser.add_argument("--check-nct", help="Check specific NCT ID")
    parser.add_argument("--samples", action="store_true", help="Show sample documents")
    parser.add_argument("--check-missing", help="Path to file with expected NCT IDs (one per line)")
    parser.add_argument("--trials-folder", help="Path to folder with criteria JSON files")
    
    args = parser.parse_args()
    
    # Load and normalize config
    cfg = load_config(args.config)
    
    # Get trials folder from config if not provided
    if not args.trials_folder:
        args.trials_folder = cfg.get("paths", {}).get("trials_json_folder", "../data/lung_processed_trials")
    
    # Create ES client
    try:
        es = make_es_client(cfg)
    except Exception as e:
        print(f"❌ Failed to create ES client: {e}")
        print(f"\nConfig: {json.dumps(cfg.get('elasticsearch', {}), indent=2)}")
        return
    
    # Check connection
    if not es.ping():
        print("❌ Cannot connect to Elasticsearch")
        print(f"\nHosts: {cfg['elasticsearch']['hosts']}")
        return
    
    print("✅ Connected to Elasticsearch")
    
    # Get index name
    if not args.index:
        if "elasticsearch_indices" in cfg:
            args.index = cfg["elasticsearch_indices"].get("index_trials_eligibility", "trials_eligibility")
        else:
            args.index = "trials_eligibility"
        print(f"Using index from config: {args.index}")
    
    # Check if index exists
    if not es.indices.exists(index=args.index):
        print(f"❌ Index '{args.index}' does not exist!")
        print("\nAvailable indices:")
        indices = es.cat.indices(format="json")
        for idx in sorted(indices, key=lambda x: x['index']):
            if not idx['index'].startswith('.'):
                print(f"  - {idx['index']} ({idx['docs.count']} docs)")
        return
    
    print(f"✓ Index '{args.index}' exists\n")
    
    # Execute requested action
    if args.check_nct:
        check_file_vs_index(es, args.index, args.check_nct, args.trials_folder)
    
    elif args.samples:
        get_sample_documents(es, args.index)
    
    elif args.check_missing:
        # Load expected NCT IDs
        with open(args.check_missing, 'r') as f:
            expected_ids = [line.strip() for line in f if line.strip().startswith('NCT')]
        
        check_missing_trials(es, args.index, expected_ids, args.trials_folder)
    
    else:
        # Default: check the 3 ground truth trials
        ground_truth_trials = ["NCT03948763", "NCT06345729", "NCT07174908"]
        
        print(f"Checking {len(ground_truth_trials)} ground truth trials...")
        
        for nct_id in ground_truth_trials:
            check_file_vs_index(es, args.index, nct_id, args.trials_folder)
        
        # Also show samples
        get_sample_documents(es, args.index, n=2)


if __name__ == "__main__":
    main()