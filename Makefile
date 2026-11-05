SRC ?= data/videos/mixkit-61.mp4
.PHONY: demo demo-pose test
demo:
	python demo.py --source "$(SRC)"
demo-pose:
	python demo.py --source "$(SRC)" --pose
test:
	pytest -q

.PHONY: site serve app
site:
	python scripts/build_site.py
serve: site
	@echo "open http://localhost:8000"
	cd site && python -m http.server 8000
app:
	python app.py
