#!/usr/bin/env python

"""
Script to run the PII Detection API server.
"""

from pii_protection.api import start_api
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("Starting PII Detection API server...")
    print("Documentation available at: http://localhost:8000/docs")
    print("Health check at: http://localhost:8000/health")
    start_api()

if __name__ == "__main__":
    main() 