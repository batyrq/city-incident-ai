SRC ?= data/videos/mixkit-61.mp4
.PHONY: demo demo-pose test
demo:
	python demo.py --source "$(SRC)"
demo-pose:
	python demo.py --source "$(SRC)" --pose
test:
	pytest -q

.PHONY: site run
site:
	python scripts/build_site.py
run: site
	python server.py
