#!/bin/bash

# Lancer le serveur Uvicorn en arrière-plan
uvicorn clipai_runpod_engine.handler:app --host 0.0.0.0 --port 8000 &

# Lancer le worker en utilisant 'exec' pour qu'il devienne le processus principal
exec python3 -m clipai_runpod_engine.engine.worker