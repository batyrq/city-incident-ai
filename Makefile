SRC ?= data/videos/mixkit-61.mp4
.PHONY: demo demo-pose test
demo:
	python demo.py --source "$(SRC)"
demo-pose:
	python demo.py --source "$(SRC)" --pose
test:
	pytest -q
