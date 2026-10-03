# vball: External Data, Models and Tools Reference

Suggested repo location: `docs/references/external_data.md`

This is a **reference database**, not a task list. Use it to look up sources, label schemas, formats, licences and fetch recipes while working on vball (rally cutting, then action tagging, then player stats).

Compiled 2026-10-03 from live sources. Anything marked ⚠ was not confirmed against the primary source and must be checked before relying on it.

---

## Conventions for Claude Code

* Fetched data goes in `$VBALL_DATA/external/<source_id>/` (default `data/external/`). Roboflow sets go in `data/external/roboflow/<source_id>/`.
* Never commit raw external data or weights to git.
* Every fetched folder gets a `_source.txt` (URL, fetch date, licence). The script in §9 does this automatically.
* Auto-fetchable sources: run the script in §9. Manual sources: follow the **Fetch** line in each entry.
* Before training on a source, check its **Licence** and **View** rows. Camera viewpoint (broadcast side view vs. tripod behind the end line) causes more domain gap than league level does.
* Target domain for vball: **indoor** university match footage from a fixed camera.
* vball native formats:
  * Ball track: TrackNet CSV `Frame,Visibility,X,Y`.
  * Rally ground truth: CSV from the web app ("Download GT CSV", keys S/E/U), consumed by `vball eval`.

---

## 0. Quick index

| id | type | labels | view | size | licence | fetch |
|---|---|---|---|---|---|---|
| `deadfast-tracknet` | weights | ball (TrackNetV3 fine-tune) | beach | 1 ckpt | MIT | auto (HF) |
| `ddecks-beach-volley-vision` | code + tools | ball, rally, serve, players | beach (YouTube) | n/a | MIT | auto (git) |
| `vballnet-dataset` | dataset | ball xy per frame | indoor amateur | ⚠ unknown | ⚠ unknown | auto (tgz) |
| `asigatchov-fast-volleyball-tracking-inference` | ONNX weights + code | ball | indoor amateur | several ONNX | MIT | auto (git) |
| `nttcom-wasb-sbdt` | code + data | ball xy, 5 sports incl. volleyball | ⚠ check | ⚠ | ⚠ see repo | git + repo docs |
| `vnl-stes` | dataset + code | 6 events, frame + (x,y) | broadcast (VNL) | 1,028 rallies, 6,137 events | ⚠ sourced with Volleyball World permission | manual |
| `multisports` | dataset | 12 volleyball classes, person tubes | broadcast | 3,200 videos all sports | gated | HF gated |
| `ibrahim-volleyball` | dataset | 9 individual + 8 group actions, boxes | broadcast | 4,830 clips, 55 videos | ⚠ | manual |
| `tugraz-volleyball-2014` | dataset | 7 classes, boxes | ⚠ league HD | 36,178 ann, 18,960 frames | cite papers | manual / Roboflow mirror |
| `volleyvision-*` | Roboflow sets | ball / actions / players / court | mostly non side view | 19k / 14k / ... | CC BY-NC-ND (README) | Roboflow key |
| `vren` | dataset | rally notation (actions + locations) | broadcast (pro + NCAA D-I) | ⚠ | ⚠ | auto (git) |
| `sportsmot` | dataset | player MOT boxes | broadcast | 28,831 volleyball imgs | ⚠ | CodaLab sign-up |
| `openvolley-pydatavolley` | library | parses DataVolley `.dvw` | n/a | n/a | ⚠ | `uv add` |
| `ovscout2` | app | creates `.dvw` from your video | fixed camera | n/a | MIT | release zip |

---

## 1. Ball tracking: data, weights, code

### 1.1 `deadfast-tracknet`: deadfast/beach-volley-vision-models (currently used by vball)

* **URL:** https://huggingface.co/deadfast/beach-volley-vision-models
* **What:** TrackNetV3 checkpoint fine-tuned for beach volleyball. Base model: `qaz812345/TrackNetV3`. Consumed upstream by `ddecks/beach-volley-vision`.
* **Training labels:** hand-annotated beach rally clips in TrackNet CSV format (`Frame,Visibility,X,Y`). The ddecks README says about 18,000 visible ball frames across 12 clips.
* **Inpainting:** if the pipeline uses inpainting, `InpaintNet_best.pt` comes from upstream TrackNetV3.
* **View:** beach. Expect a domain gap on indoor gym footage.
* **Licence:** MIT.
* **Fetch:** direct link https://huggingface.co/deadfast/beach-volley-vision-models/resolve/main/tracknet_best.pt (handled automatically by §9).

### 1.2 `ddecks-beach-volley-vision`

* **URL:** https://github.com/ddecks/beach-volley-vision (MIT)
* **Pipeline:**
  * TrackNet ball detection feeds a rally detection state machine.
  * Players: RF-DETR (Apache 2.0) detection plus ByteTrack tracking.
  * Court calibration from manual corner annotation to a homography.
  * Net line registration.
  * Serve detector fusing a depth-on-side spatial signal with ball trajectory.
  * DAG orchestrator with cached stages.
* **Rally detector details** (ball-visibility heuristic):
  * sustained-flight gate (filters brief tosses);
  * max-velocity gate (rejects impossible ball jumps, which prevents rallies merging);
  * frame-geometry-relative motion normalisation;
  * two modes, condense vs. split.
* **Post-processing:** trajectory outlier suppressor using a Kalman filter with gravity.
* **Fast inference:** non-overlapping stride 8 with batch 16, about 8–30× faster than a sliding window.
* **Scripts worth reading:**
  * `process_match.py`: YouTube → Balltime → hard examples → annotation.
  * `extract_hard_examples.py`
  * `preannotate_tracknet.py`, `annotate_ball.py`
  * `convert_balltime.py`: Balltime JSON → frame-level events.
  * `compare_balltime.py`: own rallies vs. Balltime ground truth.
  * `finetune_tracknet.py`, `calibrate_court.py`, `auto_tune.py`
* **Docs:**
  * `docs/annotation-guide.md`
  * `docs/annotation-pipeline.md` (Label Studio legacy format)
  * `docs/gpu-training.md`
  * `TechnicalDirection6-15.md`
  * `RESOURCES.md` (⚠ not read; may list more sources)
* **Data:** `data/labeling_clips/` holds clips plus annotation CSVs. ⚠ Check what is actually committed; models are gitignored.
* **Fetch:** auto (git).

### 1.3 asigatchov VballNet family (indoor, amateur gym footage)

* **Author:** Alexander (`asigatchov`). Demo: https://demo.vb-ai.ru
* **`vballnet-dataset`:** https://volleyball-orel.ru/static/system/docs/data_20250711_2330.tgz
  * Amateur volleyball videos with ball positions hand-annotated frame by frame.
  * ⚠ Size and exact label format not verified; probably TrackNet-style CSV. Inspect after download.
* **Labelling tool:** https://github.com/asigatchov/vball-net-pytorch/blob/main/src/utils/imgLabel.py (Python + OpenCV).
* **Write-up:** https://medium.com/@asigatchov/building-vball-net-a-lightweight-volleyball-ball-tracker-200-fps-on-cpu-20f5724c0c18 (200+ FPS on CPU with OpenVINO).
* **Repos:**
  * `fast-volleyball-tracking-inference` (MIT): https://github.com/asigatchov/fast-volleyball-tracking-inference
    * ONNX inference at about 100 FPS on an i5-10400F CPU.
    * Outputs `ball.csv`. `track_calculator.py` turns that into `track_*.json`; `make_reels.py` cuts reels; there is also an OpenVINO runtime.
    * ONNX models:
      * `VballNetFastV1_155_h288_w512.onnx`
      * `VballNetFastV1_seq9_grayscale_233_h288_w512.onnx`
      * `VballNetV1_150_h288_w512.onnx`
      * `VballNetV1_seq9_grayscale_330_h288_w512.onnx`
      * OpenVINO: `ov/VballNetV2_seq9_grayscale_ov.xml`
    * Input: 9 stacked grayscale frames at 288×512.
  * `vball-net`: https://github.com/asigatchov/vball-net
    * Built on TrackNetV4 (TensorFlow).
    * VballNetV1 adds a MotionPromptLayer that builds attention maps from consecutive frames.
    * VballNetFastV1 uses depthwise separable convolutions.
    * Output: heatmaps of shape `(B, 3, H, W)`.
  * `vball-net-pytorch`: https://github.com/asigatchov/vball-net-pytorch (PyTorch rewrite with training code).
  * `Court-Keypoint-Detection`: see §3.3.
  * `VAA`: annotation tool, see §5.
* **Independent validation:** `williamyen042/passform` reports that on practice footage VballNet went from 15% of frames at noise confidence to 50–80% of frames at 0.65. A 50 KB checkpoint beat 136 MB of YOLOv8x, because the task is driven by motion.
* **Inference recipe** (from the README):

```bash
uv run src/inference_onnx_seq_gray_v2.py --video_path "$VIDEO" \
  --model_path models/VballNetV1_seq9_grayscale_330_h288_w512.onnx \
  --output_dir output --only_csv
uv run src/track_calculator.py --csv_path output/<video_stem>/ball.csv --output_dir output
```

### 1.4 `nttcom-wasb-sbdt`: WASB (Widely Applicable Strong Baseline), BMVC 2023

* **URL:** https://github.com/nttcom/WASB-SBDT
* **Method:**
  * high-resolution feature extraction;
  * position-aware training;
  * inference with temporal consistency (online tracking with a local motion model).
* **Evaluation:** benchmarked against 6 SOTA ball trackers on 5 sports. The volleyball dataset was **newly introduced** by the authors. Distance threshold τ = 4 px.
* **Use:** a second source of volleyball ball labels, plus clean reimplementations of TrackNetV2, DeepBall and others for A/B tests.
* **Fetch:** git clone (auto), then follow the repo's dataset instructions.
* ⚠ Dataset view and licence not checked.

### 1.5 TrackNet upstream

* `qaz812345/TrackNetV3` (MIT), already vendored at `third_party/TrackNetV3`.
* Original TrackNet paper: arXiv 1907.03698.

### 1.6 openvolley `ovml` / `ovmlpy` (R)

* **URLs:** https://github.com/openvolley/ovml and https://github.com/openvolley/ovmlpy (MIT)
* **Models:** YOLO v3, v4 and v7, plus an **experimental volleyball-specific** ball network.
* `ovmlpy` runs the Python implementation and is faster. `ovml` also has an ONNX-backed Ultralytics detector option.
* ⚠ Weights download via `ovml_yolo(weights_file = "auto")`. Check whether the volleyball net weights are reachable outside R.

### 1.7 Roboflow single-frame ball sets (no temporal context; use for hard negatives or a fallback detector)

| id | workspace/project | version | notes |
|---|---|---|---|
| `volleyvision-ball` | `shukur-sabzaliev1/volleyball_v2` | ⚠ 2 (from model badge) | 19k images, duplicates removed. Earlier 25k version: `volleyvision/volleyball-tracking` v9 |
| `daniel-ball` | `daniel-eqnnu/volleyball_ball_object_detection_dataset-3jez2` | ⚠ 1 | CC BY 4.0, RGB + grayscale versions. Companion sets exist for players/referee and court keypoints |
| `myarmy-ball` | `myarmy/volleyball-2vfip` | 2 | 2,615 images, CC BY 4.0 |
| `aivolleyballref-ball` | `aivolleyballref/volleyball_detection` | 2 | 1,091 images |
| `aimbotcod-ball` | `aimbotcod/volley-ball-recognition` | 5 | 1,280 images |
| (low) | `incodeia/voleball-ball-detection-fludy`, `test-rnrvz/volleyball-ball-tracker`, `neurone/ball-detection-1bjgl` (CC BY 4.0) | ⚠ | unverified sizes |
| (beach) | `anaya-brown/beach_volley_bb` | 1 | 102 images, CC BY 4.0 |
| (skip) | `omar-7mdnt/indoorvolleyball` | 1 | listed 0 images; classes Ball, Block, Dig, Serve, Set |

### 1.8 Other ball work

* `tprlab/vball` (https://github.com/tprlab/vball): old approach (highest blobs plus a ball / not-ball classifier). Low priority.
* **Volleyball-1,2** (from a sports survey, arXiv 2206.01038): two sequences of 10,000 and 19,500 frames with manually annotated ball boxes. No public link located.

---

## 2. Event and action datasets

### 2.1 `vnl-stes`: VNL-STES (CVPR Workshops 2025)

* **Links:**
  * Project: https://hoangqnguyen.github.io/stes/
  * Code: https://github.com/hoangqnguyen/spot
  * Data: https://bit.ly/vnlvolley1
  * Paper: https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf
* **Content:** 8 full matches from VNL 2022 and 2023, cut into 1,028 rally videos (251,110 frames). Full HD MP4 at 25 FPS. Varied jerseys, courts and audiences.
* **Labels:** 6,137 events, each with an exact frame and an (x, y) position.

  | class | events | % |
  |---|---|---|
  | Serve | 1,071 | 17.45 |
  | Receive | 1,558 | 25.39 |
  | Set | 1,393 | 22.70 |
  | Spike | 1,321 | 21.53 |
  | Block | 550 | 8.96 |
  | Score | 244 | 3.97 |

* **Splits** (each split uses different matches):

  | split | rallies | events | frames |
  |---|---|---|---|
  | train | 811 | 4,901 | 202,909 |
  | val | 102 | 600 | 21,846 |
  | test | 115 | 636 | 26,355 |

* **Annotation pipeline:** download → rally extraction via **court keypoint detection** → pseudolabels from a pretrained model → manual correction of frame and (x, y).
* **Metrics:** mTAP at 0–4 frame tolerance; mSAP at 2–6 px.
* **STES model:**
  * Backbone: RegNet-Y with a Gate Shift Module.
  * Temporal: 3-layer bidirectional GRU, per-frame classification.
  * Spatial: MLP head regressing (x, y).
  * Loss: weighted cross-entropy plus L1 on event frames only.
* **Results** (mTAP@0–4F): E2E-Spot 55.32, T-DEED 58.58, E2E Spatial 64.81, **STES 68.44**. Spatial mSAP@2–6P: STES 80.21.
* **Known confusions:** spike vs. block, and score vs. receive (ball occluded at ground contact).
* **View:** broadcast. Video was sourced with permission from Volleyball World. ⚠ The paper suggests the full-resolution version is not distributed because of size; check what the bit.ly link actually serves.
* **Fetch:** manual. Resolve the bit.ly link in a browser. If it points to Google Drive, use `uvx gdown --folder <url> -O data/external/vnl-stes`.

### 2.2 `multisports`: MultiSports (ICCV 2021), volleyball subset

* **Links:**
  * Data (gated): https://huggingface.co/datasets/MCG-NJU/MultiSports
  * Site: https://deeperaction.github.io/datasets/multisports.html
  * Eval code: https://github.com/MCG-NJU/MultiSports
  * Paper: arXiv 2105.07404
* **Content:** 66 classes across basketball, volleyball, football and aerobic gymnastics. Frame-wise person tubes at 25 FPS with precise temporal boundaries defined in a handbook. Whole dataset: 3,200 videos and 37,701 action instances.
* **Volleyball class ids:**

  | id | class | id | class |
  |---|---|---|---|
  | 21 | serve | 27 | adjust |
  | 22 | block | 28 | save |
  | 23 | first pass | 29 | second attack |
  | 24 | defend | 30 | spike |
  | 25 | protect | 31 | dink |
  | 26 | second pass | 32 | no offensive attack |

* **Annotation pickle keys:** `nframes`, `resolution` `(h, w)`, `gttubes` (⚠ plus label and split lists; check the repo).
* **View:** broadcast (Olympics and World Cup footage from YouTube).
* **Fetch:** run `hf auth login`, accept the terms on the HF page, then use §9 with `--gated`.

### 2.3 `ibrahim-volleyball`: Volleyball Dataset (Ibrahim et al., CVPR 2016)

* **Link:** https://github.com/mostafa-saad/deep-activity-rec. The README has the download links, with a note about quota issues on mirrors.
* **Content:** 55 YouTube matches (London 2012 Olympics and others), 4,830 annotated clips.
* **Structure:**
  * Each video is a folder `0..54`.
  * Each annotated frame is a folder of 41 images: 20 before, the target, 20 after. The original authors used 5 before and 4 after.
* **Labels:** on the middle frame, player boxes plus one individual action, and one group activity.
  * Individual action counts: standing 38,696, moving 5,121, waiting 3,601, blocking 2,458, digging 2,333, setting 1,332, falling 1,241, spiking 1,216, jumping 341. **There is no serve class.**
  * Group activities: left/right × set, spike, pass, winpoint.
* **Annotation file:** `annotations.txt` per video. Line format is `{FrameID} {GroupActivity} {PlayerAnnotation}...` (⚠ per-player field order, believed to be `X Y W H Action`; verify in README).
* **Splits by video id:**
  * Train: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
  * Val: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
  * Test: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47
* **Extensions:**
  * Bagautdinov et al.: boxes for every frame of each clip.
  * DECOMPL (arXiv 2303.06439): 497 relabels, about 10% of clips, leaving 4,821 clips.
  * "Skeleton-based relational reasoning for group activity analysis": hand-annotated **ball positions** on these frames.
* **Predecessor:** HierVolleyball (1,525 frames from 15 videos).
* **View:** broadcast.

### 2.4 `tugraz-volleyball-2014`: Volleyball Activity Dataset 2014 (TU Graz ICG)

* **Link:** https://tugraz.at/index.php?id=17751
* **Content:** 6 videos from the Austrian Volley League 2011/12. 36,178 annotations in 18,960 frames. HD 1920×1080 at 25 FPS, DX50 codec.
* **Classes:** Serve, Reception, Setting, Attack, Block, Stand, Defense/Move.
* **Cite:** Waltner, Mauthner, Bischof, 2014 (DVS/GSSS and AAPR/OAGM papers; BibTeX on the page).
* **Roboflow mirror:** `shukur-sabzaliev-zc3en/volleyball-activity-dataset` (v3 original, v4 resized to 640). **Warning:** the mirror's train/val/test split is random over frames. Recombine everything and re-split sequentially by video.
* **Fetch:** manual from the TU Graz page, or the Roboflow mirror via §9.

### 2.5 `volleyvision-*`: VolleyVision (shukkkur)

* **Repo:** https://github.com/shukkkur/VolleyVision. The README says CC BY-NC-ND 4.0; the repo licence file says AGPL-3.0. Treat it as non-commercial.

| set | Roboflow workspace/project | versions | classes / size |
|---|---|---|---|
| ball | `shukur-sabzaliev1/volleyball_v2` | see §1.7 | 1 class, 19k images |
| actions | `shukur-sabzaliev-42xvj/volleyball-actions` | v5 original, v3 640², v4 1024² | block, defense, serve, set, spike; 14k images |
| players | `shukur-sabzaliev-42xvj/players-dataset` | v1 original, v4 resized, v2 augmented | 1 class |
| court seg | `shukur-sabzaliev-bh7pq/court-segmented` | ⚠ 1 | semantic segmentation |
| TU Graz mirror | `shukur-sabzaliev-zc3en/volleyball-activity-dataset` | v3 original, v4 640 | 7 classes |

* **Reported metrics:**
  * Ball: YOLOv7-tiny mAP50 74.1; Roboflow model 92.3.
  * Actions: YOLOv8m mAP50 92.31.
  * Players: YOLOv8m mAP50 97.2.
  * Court: mIoU 97.2.
* **Known gaps** (from the README):
  * The defense class is undertrained.
  * Side-view (official broadcast) data is underrepresented.
  * The player detector sometimes fires on coaches, referees and spectators.
* **Weights in repo:** `Stage I - Volleyball/yV7-tiny/weights`, `Stage II - Players & Actions/actions/yV8_medium`.

### 2.6 `vren`: VREN (Volleyball Rally Dataset with Expression Notation Language)

* **Links:** https://github.com/haotianxia/VREN, paper arXiv 2209.13846.
* **Content:** pro and NCAA D-I indoor rallies, manually labelled in a volleyball notation language that covers player actions and locations through to the rally outcome.
* **Tasks defined:** rally outcome prediction, set type and hit type prediction, tactics and attack-zone statistics.
* **Use:** tactical and statistics layers rather than vision training.

### 2.7 Paper only, or nearly empty

* **PathFinder / PathFinderPlus** (arXiv 2309.14753): setting-tactic classification from ball trajectories. Uses 537 clips from 720p men's national team matches. ⚠ Data availability unknown.
* **digdeep** (https://huggingface.co/datasets/ryan1288/digdeep_data, AGPL-3.0; code https://github.com/ryan1288/digdeep/):
  * Aimed at removing downtime between plays (rally cutting).
  * Snippet naming scheme: `YouTube_Indoor_20241012_Wide_001.mp4`, with metadata for type, date and camera type.
  * As of 2026-10-03 the repo holds only about 4 KB. Watch it, but nothing to fetch yet.
* **SVW:** 4,100 videos, 44 categories, includes volleyball. Low priority.

---

## 3. Players, tracking, court

### 3.1 `sportsmot`: SportsMOT (ICCV 2023)

* **Links:** https://deeperaction.github.io/datasets/sportsmot.html and https://github.com/MCG-NJU/SportsMOT. Mirror and stats: https://datasetninja.com/sports-mot
* **Content:** 240 sequences, 150k+ frames, 1.6M boxes across basketball, volleyball and football. Volleyball is 28,831 images.
* **Labels:** only on-court players are tracked; referees, coaches and spectators are excluded.
* **Baseline:** MixSort tracker.
* **Fetch:** manual. Sign up on CodaLab, then use Participate → Get Data.

### 3.2 Player detection sources

* VolleyVision players set (§2.5).
* Ibrahim boxes (§2.3).
* Roboflow `daniel-eqnnu` companion players/referee set (§1.7, ⚠ exact project id not captured).
* ddecks uses RF-DETR (Apache 2.0) with ByteTrack.

### 3.3 Court geometry

* `asigatchov/Court-Keypoint-Detection` (https://github.com/asigatchov/Court-Keypoint-Detection): YOLOv11 keypoints trained on **back-view cameras in indoor gyms**. Best viewpoint match for tripod footage.
* Roboflow `protom/volleyball-court-detection-tpvsi`: keypoint detection, CC BY 4.0, ⚠ version 1.
* VolleyVision court segmentation (§2.5): mask → `cv2.findContours` → `cv2.approxPolyDP`.
* openvolley `ovideo::ov_transform_points`: image coordinates to court coordinates (R).

---

## 4. Scouting data as weak labels

### 4.1 `openvolley-pydatavolley`

* **Install:** `uv add openvolley-pydatavolley` (GitHub: https://github.com/openvolley/py-datavolley).
* **Reads:** DataVolley `.dvw` and VolleyStation `.vsm` files.
* **API:**
  * `import datavolley as dv`
  * `dv.read_dv(path)`
  * `dv.example_file()`
  * `dv.read_dv(path, validation_mode=dv.ValidationMode.LENIENT, normalize_types=True, return_issues=True)` returns `(plays, issues)`
* **Older package:** `pydatavolley` (https://github.com/openvolley/pydatavolley), used as `read_dv.DataVolley(path).get_plays()` and returning a DataFrame.
* **Columns** (from the older package):
  * Identity and timing: `match_id`, `video_file_number`, `video_time`, `code`, `team`
  * Player: `player_number`, `player_name`, `player_id`
  * Touch: `skill`, `evaluation_code`, `attack_code`, `set_code`, `set_type`, `start_zone`, `end_zone`, `end_subzone`, `num_players_numeric`
  * Score and rotation: `setter_position`, `home_team_score`, `visiting_team_score`, `home_setter_position`, `visiting_setter_position`, `home_p1..`
  * Rally: `custom_code`, `point_won_by`, `serving_team`, `receiving_team`, `rally_number`, `possesion_number`
  * ⚠ `video_time` looks like seconds from video start (example values 494–495).
* **Use:** a `.dvw` file plus its match video gives you timestamped touches (skill, player number, zones) for free. Rally boundaries come from `rally_number` and the serve rows.
* **Precision:** timestamps are coarse (⚠ about 1 s), so use them as weak labels or event windows and refine with an event spotter.

### 4.2 `ovscout2` (MIT)

* **Links:** https://github.com/openvolley/ovscout2. Manual: https://openvolley.r-universe.dev/ovscout2/doc/ovscout2-user-manual.html
* **What:** a free R/Shiny scouting app. It needs video from a **fixed viewpoint** and saves an industry-standard `.dvw` after each rally.
* **Workflow:** the scout clicks action locations on the video, so labels include court positions.
* **Windows standalone:** https://github.com/openvolley/ovscout2/releases/download/v0.1.0/ovscout2-win-x64.zip
* **R install:** `install.packages("ovscout2", repos = c("https://openvolley.r-universe.dev", "https://cloud.r-project.org"))` (⚠ pattern taken from sibling packages). May also need `ovscout2::ov_install_lighttpd()`.
* **Related openvolley packages (MIT):**
  * `ovscout`: video sync and scout editing (`ov_shiny_video_sync`).
  * `ovva`: video analysis and playlists from `.dvw`.
  * `ovideo`: frames and court transforms.

### 4.3 Balltime exports

* ddecks `convert_balltime.py` / `compare_balltime.py` turn Balltime JSON into frame-level events and rally ground truth.
* ⚠ Check Balltime's terms of service before reusing its output.

### 4.4 DVMate (commercial, iPad)

* Imports `.dvw` from DataVolley or Click and Scout and syncs it to video.
* Also syncs VERT jump data.
* Reference only.

---

## 5. Annotation tools

| tool | link | good for |
|---|---|---|
| vball web app | built in | rally S/E ground truth (`N/P/S/E/U`), feeds `vball eval` |
| asigatchov VAA | https://github.com/asigatchov/VAA | rally timelines, nested action labels, YOLO boxes, superframe preview, frame-accurate navigation |
| asigatchov imgLabel.py | see §1.3 | fast ball xy clicking |
| ddecks annotate scripts | see §1.2 | model-assisted ball correction and hard-example mining |
| CVAT + SAM 2 tracker | https://docs.cvat.ai/docs/enterprise/segment-anything-2-tracker | propagate player masks/boxes across frames |
| ovscout2 | see §4.2 | full action scouting producing `.dvw` |

**CVAT notes:**
* The SAM 2 tracker runs as a user-side AI agent for CVAT Online, or server-side in Enterprise.
* Tasks must be created from a video or from same-size image sequences.
* Skeletons cannot be tracked.

---

## 6. Model codebases

* **T-DEED** (https://github.com/arturxe2/T-DEED):
  * Precise event spotting.
  * RegNetY backbone with Gate-Shift-Fuse modules, then an SGP-layer encoder-decoder.
  * Classification head plus temporal displacement head.
  * Won the SoccerNet Ball Action Spotting 2024 challenge.
* **STES** (https://github.com/hoangqnguyen/spot): spatiotemporal event spotting on VNL (§2.1).
* **E2E-Spot** (Hong et al.): end-to-end precise spotting baseline. ⚠ Repo URL not captured.
* **VideoMAE** (https://github.com/MCG-NJU/VideoMAE) and **VideoMAE V2** (https://github.com/OpenGVLab/VideoMAEv2): work well on small datasets of about 3–4k videos. Usable as a clip classifier for rally / no-rally or action windows.
* **masouduut94/volleyball_analytics** (https://github.com/masouduut94/volleyball_analytics, GPL-2.0): YOLOv8 + VideoMAE with scene classification, action detection and a FastAPI service. ⚠ README not read.
* **williamyen042/passform** (https://github.com/williamyen042/passform): passing analysis.
  * Pipeline: ball track → trajectory turn → contact frame → player touching the ball = passer → MediaPipe pose at that frame → joint angles.
* **TrackNetV3 / VballNet / WASB:** see §1.

---

## 7. Method notes from the literature

**Rally segmentation signals**
* Ball-visibility state machine with flight and velocity gates (ddecks).
* Whistle detection to cut matches into rallies (volleyball spike-detection and block-classification paper).
* Court transition caused by camera operation (Itazuri et al., CVPR-W 2017). Broadcast only.
* Court keypoint detection (VNL rally extraction).
* Audio impact detection clustered into rally intervals (TennisExpert, arXiv 2603.13397, tennis).

**End-line camera pipeline** (IEEJ 2025 reception evaluation)
* Detect court and net, then track the ball through the rally.
* Derive toss and spike events from the ball trajectory.

**Contact frames**
* A turn in the ball trajectory marks a touch (passform). Pairs naturally with player boxes for touch attribution.

**Event spotting tolerances**
* Volleyball events are evaluated within 0–4 frames at 25 FPS (VNL).
* Rare classes (block, score) are weakest.

**Tracking vs. pixels**
* On SoccerNet-GAR (soccer group activity), a tracking-only model beat a fine-tuned VideoMAEv2 video baseline by 9.1 points with 438× fewer parameters.
* Relevant when classifying actions from ball and player tracks.

**Domain gap**
* Pretrained TrackNet models are tuned on professional broadcast tennis and badminton. The asigatchov write-up calls the gap to amateur volleyball footage severe.

---

## 8. Label crosswalk

Suggested canonical vocabulary: `serve, receive, set, attack, block, dig, freeball, point, other`.

| source | granularity | serve | receive | set | attack | block | dig | other / notes |
|---|---|---|---|---|---|---|---|---|
| VNL-STES | frame + (x,y) | Serve | Receive | Set | Spike | Block | n/a ⚠ | Score → point |
| MultiSports | person tube + temporal extent | serve | first pass ⚠ | second pass | spike, second attack, dink | block | defend ⚠ | protect, adjust, save, no offensive attack: check handbook |
| Ibrahim | person box, middle frame | none | none | setting | spiking | blocking | digging | standing, moving, waiting, jumping, falling |
| TU Graz 2014 | person box per frame | Serve | Reception | Setting | Attack | Block | Defense/Move ⚠ | Stand |
| VolleyVision actions | single-image box | serve | none | set | spike | block | defense ⚠ | |
| DataVolley `.dvw` | timestamp (~1 s ⚠) + player number + zones | S | R | E | A | B | D | F freeball (⚠ verify skill codes via `skill` column) |

---

## 9. Fetch script

The script lives at `scripts/fetch_external.py`.

**Prerequisites**
* `git` must be on PATH.
* Roboflow sets need `ROBOFLOW_API_KEY`.
* Gated Hugging Face sets need `hf auth login` and the terms accepted on the dataset page.
* Downloads were not test-run at compile time. Inspect each folder after fetching.

**Run**

```bash
uv run --with huggingface_hub --with roboflow python scripts/fetch_external.py               # git + archives + HF
uv run --with huggingface_hub --with roboflow python scripts/fetch_external.py --roboflow    # + Roboflow sets
uv run --with huggingface_hub --with roboflow python scripts/fetch_external.py --gated       # + gated HF sets
uv run --with huggingface_hub --with roboflow python scripts/fetch_external.py --only vballnet-dataset deadfast-tracknet
uv run --with huggingface_hub --with roboflow python scripts/fetch_external.py --list
```

---

## 10. Licence summary

| source | licence / terms |
|---|---|
| deadfast weights, ddecks repo, TrackNetV3, fast-volleyball-tracking-inference | MIT |
| openvolley packages (ovml, ovmlpy, ovscout, ovscout2, ovva) | MIT |
| RF-DETR | Apache 2.0 |
| VolleyVision data | CC BY-NC-ND 4.0 (README); repo licence file AGPL-3.0 |
| Roboflow sets marked CC BY 4.0 (daniel, myarmy, protom, neurone, beach_volley_bb) | CC BY 4.0 |
| masouduut94/volleyball_analytics | GPL-2.0 |
| digdeep data | AGPL-3.0 |
| VNL-STES | video sourced with permission from Volleyball World; ⚠ check data terms |
| MultiSports | gated; accept terms on Hugging Face |
| Ibrahim Volleyball, TU Graz 2014, SportsMOT, VREN, WASB, vballnet-dataset | ⚠ unknown; check before any redistribution |

vball is a personal project, so non-commercial terms are fine for now. Recheck this table if the repo or trained weights are ever published.
