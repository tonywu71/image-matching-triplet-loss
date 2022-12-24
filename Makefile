# --- CLEAR ---
clear-saved_models:
	rm -rv saved_models/*

clear-logs:
	rm -rv logs/*

clear-embeddings:
	rm -rv embeddings/*

clear-all:
	make clear-saved_models && make clear-logs && make clear-embeddings


# --- BACKUP ---
zip-models:
	zip -r saved_models.zip saved_models/

zip-logs:
	zip -r logs.zip logs/
