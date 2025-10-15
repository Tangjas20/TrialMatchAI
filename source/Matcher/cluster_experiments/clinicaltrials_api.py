# Matcher/utils/clinicaltrials_api.py
"""
Fetch trial metadata from ClinicalTrials.gov API v2.
"""

import requests
import time
import json
from typing import Dict, List, Optional
from Matcher.utils.logging_config import setup_logging
import argparse
import os

logger = setup_logging()


class ClinicalTrialsAPI:
    """
    Interface to ClinicalTrials.gov API v2.
    
    Docs: https://clinicaltrials.gov/data-api/api
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Args:
            rate_limit_delay: Seconds to wait between requests (be polite to API)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def fetch_trial_locations(self, nct_id: str) -> Optional[Dict]:
        """
        Fetch location data for a single trial.
        
        Returns:
            {
                'nct_id': 'NCT12345678',
                'countries': ['United States', 'China'],
                'locations': [
                    {'facility': 'MD Anderson', 'city': 'Houston', 'state': 'Texas', 'country': 'United States'},
                    {'facility': 'Peking University', 'city': 'Beijing', 'country': 'China'}
                ]
            }
        """
        try:
            url = f"{self.BASE_URL}/{nct_id}"
            
            # Request specific fields to reduce payload
            params = {
                'fields': 'NCTId,LocationCity,LocationState,LocationCountry,LocationFacility'
            }
            
            logger.debug(f"Fetching {nct_id}...")
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 404:
                logger.warning(f"{nct_id}: Not found in ClinicalTrials.gov")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Parse locations
            result = self._parse_locations(nct_id, data)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API error for {nct_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {nct_id}: {e}")
            return None
    
    def _parse_locations(self, nct_id: str, api_data: Dict) -> Dict:
        """Parse location data from API response."""
        
        result = {
            'nct_id': nct_id,
            'countries': [],
            'locations': []
        }
        
        try:
            # API v2 structure
            protocol_section = api_data.get('protocolSection', {})
            contacts_locations = protocol_section.get('contactsLocationsModule', {})
            locations = contacts_locations.get('locations', [])
            
            countries_set = set()
            
            for loc in locations:
                facility = loc.get('facility', '')
                city = loc.get('city', '')
                state = loc.get('state', '')
                country = loc.get('country', '')
                
                if country:
                    countries_set.add(country)
                    
                    result['locations'].append({
                        'facility': facility,
                        'city': city,
                        'state': state,
                        'country': country
                    })
            
            result['countries'] = sorted(list(countries_set))
            
            logger.debug(f"{nct_id}: Found {len(result['locations'])} locations in {len(result['countries'])} countries")
            
        except Exception as e:
            logger.warning(f"Failed to parse locations for {nct_id}: {e}")
        
        return result
    
    def fetch_batch_locations(self, nct_ids: List[str], cache_file: str = None) -> Dict[str, Dict]:
        """
        Fetch locations for multiple trials with caching.
        
        Args:
            nct_ids: List of NCT IDs
            cache_file: Optional JSON file to cache results
        
        Returns:
            {
                'NCT12345678': {'nct_id': ..., 'countries': [...], 'locations': [...]},
                'NCT87654321': {...}
            }
        """
        results = {}
        
        # Load cache if exists
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading cached locations from {cache_file}")
            with open(cache_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} cached trials")
        
        # Fetch missing trials
        missing_ids = [nct_id for nct_id in nct_ids if nct_id not in results]
        
        if missing_ids:
            logger.info(f"Fetching {len(missing_ids)} trials from ClinicalTrials.gov...")
            
            for i, nct_id in enumerate(missing_ids, 1):
                logger.info(f"  [{i}/{len(missing_ids)}] {nct_id}")
                
                location_data = self.fetch_trial_locations(nct_id)
                
                if location_data:
                    results[nct_id] = location_data
                
                # Save cache periodically
                if cache_file and i % 10 == 0:
                    with open(cache_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.debug(f"  Cached {len(results)} trials")
        
        # Final cache save
        if cache_file:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Cached {len(results)} trials to {cache_file}")
        
        return results


# Standalone script to fetch locations
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fetch trial locations from ClinicalTrials.gov")
    parser.add_argument(
        '--nct-ids-file',
        required=True,
        help='File with NCT IDs, one per line'
    )
    parser.add_argument(
        '--output-file',
        default='trial_locations_cache.json',
        help='Output JSON file for cached locations'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Seconds between API requests (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Read NCT IDs
    with open(args.nct_ids_file, 'r') as f:
        nct_ids = [line.strip() for line in f if line.strip().startswith('NCT')]
    
    logger.info(f"Found {len(nct_ids)} NCT IDs")
    
    # Fetch locations
    api = ClinicalTrialsAPI(rate_limit_delay=args.rate_limit)
    locations = api.fetch_batch_locations(nct_ids, cache_file=args.output_file)
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("LOCATION SUMMARY")
    logger.info(f"{'='*70}")
    
    from collections import Counter
    
    all_countries = []
    for data in locations.values():
        all_countries.extend(data.get('countries', []))
    
    country_counts = Counter(all_countries)
    
    logger.info(f"\nTotal trials with location data: {len(locations)}/{len(nct_ids)}")
    logger.info(f"\nTop 10 countries:")
    for country, count in country_counts.most_common(10):
        logger.info(f"  {country}: {count} trials")
    
    # Geographic distribution
    us_only = sum(1 for d in locations.values() 
                  if d.get('countries') == ['United States'])
    china_only = sum(1 for d in locations.values() 
                     if d.get('countries') == ['China'])
    both = sum(1 for d in locations.values() 
               if 'United States' in d.get('countries', []) 
               and 'China' in d.get('countries', []))
    
    logger.info(f"\nGeographic distribution:")
    logger.info(f"  US only: {us_only}")
    logger.info(f"  China only: {china_only}")
    logger.info(f"  Both US & China: {both}")
    logger.info(f"  Other: {len(locations) - us_only - china_only - both}")