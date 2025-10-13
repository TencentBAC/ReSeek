#!/usr/bin/env python3
import requests
import json
import argparse
from pprint import pprint

def test_retrieve(queries, topk=5):
    """测试检索 API"""
    url = "http://127.0.0.1:8100/retrieve"
    
    payload = {
        "queries": queries if isinstance(queries, list) else [queries],
        "topk": topk,
        "return_scores": True
    }
    
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Response:")
            pprint(result)
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection Error: Cannot connect to server")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test retrieve API')
    parser.add_argument('query', nargs='+', help='Query string(s)')
    parser.add_argument('-k', '--topk', type=int, default=5, help='Top K results')
    
    args = parser.parse_args()
    
    # 如果只有一个查询，作为字符串；多个则作为列表
    queries = args.query[0] if len(args.query) == 1 else args.query
    
    test_retrieve(queries, args.topk)