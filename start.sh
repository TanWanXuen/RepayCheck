#!/bin/bash
set -e

echo "Waiting for DB to be ready..."
until mysqladmin ping -h db -u fyp -pfyp --silent; do
    sleep 2
done

echo "DB is up. Starting app..."
#python -u app.py
gunicorn app:app -b 0.0.0.0:5000 --timeout 120 --workers 2 --threads 2
