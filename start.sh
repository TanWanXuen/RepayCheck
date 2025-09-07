#!/bin/bash
set -e

echo "Waiting for DB to be ready..."
until mysql -h db -u fyp -pfyp --skip-ssl -e "SELECT 1;" >/dev/null 2>&1; do
  echo "DB is unavailable - sleeping"
  sleep 5
done

echo "DB is up. Starting app..."
gunicorn app:app -b 0.0.0.0:5000 --timeout 120 --workers 2 --threads 2
