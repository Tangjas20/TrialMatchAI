#!/usr/bin/env python3
"""
List all Elasticsearch indexes with details
Usage: python list_indexes.py --config config.json
"""
import argparse
import json
from pathlib import Path
from elasticsearch import Elasticsearch


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def make_es_client(cfg: dict) -> Elasticsearch:
    es_conf = cfg["elasticsearch"]
    es_kwargs = {
        "hosts": es_conf["hosts"],
        "basic_auth": (es_conf["username"], es_conf["password"]),
        "verify_certs": True,
    }
    if "ca_certs" in es_conf:
        es_kwargs["ca_certs"] = es_conf["ca_certs"]
    return Elasticsearch(**es_kwargs)


def list_indexes(es: Elasticsearch, pattern: str = "*"):
    """List all indexes matching pattern"""
    try:
        # Get detailed index information
        indices = es.cat.indices(index=pattern, format="json", h="index,docs.count,store.size,status")
        
        if not indices:
            print("No indexes found.")
            return
        
        # Sort by index name
        indices = sorted(indices, key=lambda x: x['index'])
        
        print(f"\n{'Index Name':<40} {'Docs':<15} {'Size':<15} {'Status':<10}")
        print("=" * 80)
        
        for idx in indices:
            index_name = idx.get('index', 'N/A')
            doc_count = idx.get('docs.count', '0')
            size = idx.get('store.size', 'N/A')
            status = idx.get('status', 'N/A')
            
            print(f"{index_name:<40} {doc_count:<15} {size:<15} {status:<10}")
        
        print(f"\nTotal indexes: {len(indices)}")
        
    except Exception as e:
        print(f"Error listing indexes: {e}")


def get_index_details(es: Elasticsearch, index_name: str):
    """Get detailed information about a specific index"""
    try:
        if not es.indices.exists(index=index_name):
            print(f"Index '{index_name}' does not exist.")
            return
        
        # Get index stats
        stats = es.indices.stats(index=index_name)
        total = stats['_all']['total']
        
        # Get index mapping
        mapping = es.indices.get_mapping(index=index_name)
        
        print(f"\n{'='*60}")
        print(f"Details for index: {index_name}")
        print(f"{'='*60}")
        print(f"\nDocument Count: {total['docs']['count']:,}")
        print(f"Total Size: {total['store']['size_in_bytes'] / (1024**2):.2f} MB")
        print(f"Deleted Docs: {total['docs']['deleted']:,}")
        
        # Show field mappings
        properties = mapping[index_name]['mappings'].get('properties', {})
        print(f"\nFields ({len(properties)}):")
        for field_name, field_info in sorted(properties.items())[:20]:  # Show first 20
            field_type = field_info.get('type', 'object')
            print(f"  - {field_name}: {field_type}")
        
        if len(properties) > 20:
            print(f"  ... and {len(properties) - 20} more fields")
        
    except Exception as e:
        print(f"Error getting index details: {e}")


def delete_index(es: Elasticsearch, index_name: str, confirm: bool = False):
    """Delete an index (with confirmation)"""
    try:
        if not es.indices.exists(index=index_name):
            print(f"Index '{index_name}' does not exist.")
            return
        
        if not confirm:
            response = input(f"Are you sure you want to delete '{index_name}'? (yes/no): ")
            if response.lower() != 'yes':
                print("Deletion cancelled.")
                return
        
        es.indices.delete(index=index_name)
        print(f"✅ Index '{index_name}' deleted successfully.")
        
    except Exception as e:
        print(f"Error deleting index: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage Elasticsearch indexes")
    parser.add_argument("--config", required=True, help="Path to config JSON with ES credentials")
    parser.add_argument("--pattern", default="*", help="Index pattern to list (default: all)")
    parser.add_argument("--details", help="Get detailed info for specific index")
    parser.add_argument("--delete", help="Delete specific index (requires confirmation)")
    parser.add_argument("--force", action="store_true", help="Skip confirmation for delete")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    es = make_es_client(cfg)
    
    # Check connection
    if not es.ping():
        print("❌ Cannot connect to Elasticsearch")
        print(cfg)
        return
    
    print("✅ Connected to Elasticsearch")
    
    if args.delete:
        delete_index(es, args.delete, confirm=args.force)
    elif args.details:
        get_index_details(es, args.details)
    else:
        list_indexes(es, args.pattern)


if __name__ == "__main__":
    main()