"""
API Health Check Module
Validates connectivity to all 19 API providers
"""

import os
import requests
from typing import Dict, List
import config


def check_api_health() -> Dict[str, Dict]:
    """
    Check health status of all 19 API providers
    
    Returns:
        dict: Status for each provider
            {
                'provider_name': {
                    'status': 'available' | 'error' | 'needs_key',
                    'message': str,
                    'configured': bool
                }
            }
    """
    results = {}
    configured = config.is_configured()
    
    # LLM Providers
    llm_providers = {
        'groq': {
            'name': 'Groq LLM',
            'key': config.GROQ_API_KEY,
            'test_url': 'https://api.groq.com/openai/v1/models',
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'openrouter': {
            'name': 'OpenRouter',
            'key': config.OPENROUTER_API_KEY,
            'test_url': 'https://openrouter.ai/api/v1/models',
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'huggingface': {
            'name': 'HuggingFace',
            'key': config.HUGGING_FACE_TOKEN,
            'test_url': 'https://huggingface.co/api/whoami',
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'together': {
            'name': 'Together AI',
            'key': config.TOGETHER_API_KEY,
            'test_url': 'https://api.together.xyz/v1/models',
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        }
    }
    
    # Music Generation Providers
    music_providers = {
        'replicate': {
            'name': 'Replicate',
            'key': config.REPLICATE_API_TOKEN,
            'test_url': 'https://api.replicate.com/v1/models',
            'headers': lambda k: {'Authorization': f'Token {k}'}
        },
        'beatoven': {
            'name': 'Beatoven.ai',
            'key': config.BEATOVEN_API_KEY,
            'test_url': None,  # No public health endpoint
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'loudly': {
            'name': 'Loudly Music',
            'key': config.LOUDLY_API_KEY,
            'test_url': None,
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'musicapi': {
            'name': 'MusicAPI.ai',
            'key': config.MUSICAPI_KEY,
            'test_url': None,
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        },
        'udio': {
            'name': 'Udio',
            'key': config.UDIO_API_KEY,
            'test_url': None,
            'headers': lambda k: {'Authorization': f'Bearer {k}'}
        }
    }
    
    # Audio Analysis Providers
    analysis_providers = {
        'hume': {
            'name': 'Hume AI',
            'key': config.HUME_API_KEY,
            'test_url': None,
            'headers': lambda k: {'X-Hume-Api-Key': k}
        },
        'eden': {
            'name': 'Eden AI',
            'key': config.EDEN_API_KEY,
            'test_url': 'https://api.edenai.run/v2/info/providers',
            'headers': lambda k: {'Authorization': f'Bearer {k}' if k.startswith('Bearer ') else f'Bearer {k}'}
        },
        'audd': {
            'name': 'Audd.io',
            'key': config.AUDD_API_KEY,
            'test_url': None,
            'headers': lambda k: {}
        }
    }
    
    all_providers = {**llm_providers, **music_providers, **analysis_providers}
    
    # Check each provider
    for provider_id, provider_info in all_providers.items():
        api_key = provider_info['key']
        is_configured = configured.get(provider_id, False)
        
        if not api_key or api_key == "":
            results[provider_id] = {
                'name': provider_info['name'],
                'status': 'needs_key',
                'message': 'API key not configured',
                'configured': False
            }
            continue
        
        # If no test URL, just check if key exists
        if provider_info['test_url'] is None:
            results[provider_id] = {
                'name': provider_info['name'],
                'status': 'available',
                'message': 'API key configured (no health check endpoint)',
                'configured': True
            }
            continue
        
        # Try to connect to API
        try:
            headers = provider_info['headers'](api_key)
            response = requests.get(
                provider_info['test_url'],
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                results[provider_id] = {
                    'name': provider_info['name'],
                    'status': 'available',
                    'message': 'API connection successful',
                    'configured': True
                }
            elif response.status_code == 401:
                results[provider_id] = {
                    'name': provider_info['name'],
                    'status': 'error',
                    'message': 'Invalid API key',
                    'configured': False
                }
            else:
                results[provider_id] = {
                    'name': provider_info['name'],
                    'status': 'error',
                    'message': f'HTTP {response.status_code}',
                    'configured': True
                }
        except requests.exceptions.Timeout:
            results[provider_id] = {
                'name': provider_info['name'],
                'status': 'error',
                'message': 'Connection timeout',
                'configured': True
            }
        except requests.exceptions.RequestException as e:
            results[provider_id] = {
                'name': provider_info['name'],
                'status': 'error',
                'message': f'Connection error: {str(e)[:50]}',
                'configured': True
            }
    
    # Always available providers
    results['ollama'] = {
        'name': 'Ollama (Local)',
        'status': 'available',
        'message': 'Local installation (if available)',
        'configured': True
    }
    
    results['free'] = {
        'name': 'Free Generator',
        'status': 'available',
        'message': 'Always available (built-in)',
        'configured': True
    }
    
    results['suno'] = {
        'name': 'Suno AI',
        'status': 'needs_key',
        'message': 'API not yet public (coming soon)',
        'configured': False
    }
    
    return results


def get_health_summary(results: Dict) -> Dict:
    """
    Get summary statistics from health check results
    
    Args:
        results: Results from check_api_health()
        
    Returns:
        dict: Summary statistics
    """
    total = len(results)
    available = sum(1 for r in results.values() if r['status'] == 'available')
    needs_key = sum(1 for r in results.values() if r['status'] == 'needs_key')
    error = sum(1 for r in results.values() if r['status'] == 'error')
    
    return {
        'total': total,
        'available': available,
        'needs_key': needs_key,
        'error': error,
        'percentage': round((available / total) * 100, 1) if total > 0 else 0
    }


def get_providers_by_category(results: Dict) -> Dict[str, List[str]]:
    """
    Group providers by category
    
    Args:
        results: Results from check_api_health()
        
    Returns:
        dict: Providers grouped by category
    """
    llm = ['groq', 'openrouter', 'huggingface', 'together', 'ollama']
    music = ['suno', 'replicate', 'beatoven', 'loudly', 'musicapi', 'udio', 'free']
    analysis = ['hume', 'eden', 'audd']
    
    return {
        'llm': {k: results[k] for k in llm if k in results},
        'music': {k: results[k] for k in music if k in results},
        'analysis': {k: results[k] for k in analysis if k in results}
    }
