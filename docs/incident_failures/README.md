# Incident detector failures (test split)

Image-level mistakes of the **default approach per class** (`configs/incident_detectors.json`), at the threshold chosen on the validation split.
Picked by `python -m eval.incidents_failures`: the most confident false positives and the lowest-scored false negatives. Score = the detector's score (0-1), not a calibrated probability.
The reason on each line is my reading of the image, not something the model told me; "looks like" marks a guess.
Images from the two datasets that state no licence (zinnia82 flood, hiennguyen9874 crash negatives) are **not committed**; their lines are kept so the analysis is complete.

| Image | Class | Score | Why it happened |
|---|---|---|---|
| `fire_FP1_s1.00.jpg` | fire, false positive | 1.00 | Night city under a dark, orange-lit smoke or cloud layer (D-Fire "no label"); to CLIP this looks like "fire". |
| `fire_FP2_s1.00.jpg` | fire, false positive | 1.00 | Red-lit cloud at dusk, the sunset look I wanted as a hard negative; the D-Fire label says no fire. |
| `fire_FN1_s0.00.jpg` | fire, false negative | 0.00 | Almost black night frame; the fire is a thin bright streak at the right edge, far too small for a whole-image score. |
| `fire_FN2_s0.01.jpg` | fire, false negative | 0.01 | Low sun over a car park; if the label is right the fire is tiny or hidden in the glare, nothing fire-like is visible to me. |
| `fire_FN3_s0.01.jpg` | fire, false negative | 0.01 | Dark frame with one small bright spot at the top left; again a tiny object in a whole-image classifier. |
| `smoke_FP1_s0.93.jpg` | smoke, false positive | 0.93 | Burnt, grey landscape under a hazy sky (D-Fire "no label"); the aftermath of a fire looks like smoke to CLIP. |
| `smoke_FP2_s0.88.jpg` | smoke, false positive | 0.88 | Dark Mixkit neon-hotel frame with a hazy glow; the same clip causes the one smoke false alarm in the video test. |
| `smoke_FP3_s0.86.jpg` | smoke, false positive | 0.86 | Next frame of the same clip, same cause. |
| `smoke_FN1_s0.01.jpg` | smoke, false negative | 0.01 | Hillside town; the smoke is a faint haze at the right edge. |
| `smoke_FN2_s0.02.jpg` | smoke, false negative | 0.02 | Road with low fog and a truck; labelled smoke (looks like exhaust or fog), too faint and ambiguous to separate from mist. |
| `smoke_FN3_s0.02.jpg` | smoke, false negative | 0.02 | A small white plume far away over a field; tiny in the frame. |
| `flood_FP1_s1.00.jpg` | flood, false positive | 1.00 | A road under brown water with a boat, labelled "river"; it is plainly a flooded road, so this is a label problem (not committed: dataset licence not stated). |
| `flood_FP2_s1.00.jpg` | flood, false positive | 1.00 | Overflowing river inside a park, also labelled "river"; same label ambiguity (not committed). |
| `flood_FP3_s0.95.jpg` | flood, false positive | 0.95 | Close-up of a cat labelled "animal"; I do not know why it scores 0.95 (not committed). |
| `flood_FN1_s0.39.jpg` | flood, false negative | 0.39 | Large puddle in a road with snow patches; shallow water, and it is a judgement call whether this is a flood (not committed). |
| `flood_FN2_s0.45.jpg` | flood, false negative | 0.45 | Wet road at dusk with standing water and red reflections; the water is shallow and merges with the wet asphalt (not committed). |
| `flood_FN3_s0.50.jpg` | flood, false negative | 0.50 | Dark flooded cave with a swimmer; not a street scene and very dark (not committed). |
| `crash_FP1_s0.96.jpg` | crash, false positive | 0.96 | An overturned car, labelled "non-accident": this looks like a label error in the source data (not committed: licence not stated). |
| `crash_FP2_s0.91.jpg` | crash, false positive | 0.91 | Ordinary city street seen through a dashcam with a data overlay. No crash. This is the camera-style shortcut: the probe learned "dashcam or CCTV frame", see the video audit (not committed). |
| `crash_FN1_s0.45.jpg` | crash, false negative | 0.45 | Overturned mail truck photographed from the roadside; a clean press-style photo, unlike the surveillance frames the positives are made of. |
| `crash_FN2_s0.45.jpg` | crash, false negative | 0.45 | Close-up of two damaged cars after a collision, again a normal photograph and not a surveillance frame. |
