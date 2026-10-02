# Volleyball AI clone: research notes (2026-10-02)

## Target: what Balltime "Volleyball AI" does
- Removes dead time (~65% of footage) and splits video into rallies
- Detects actions: serve, pass, set, attack, block, dig
- Assigns players to teams, identifies individual players (filter by athlete)
- Ball tracking: trajectories, serve speed, attack height, jump height
- Outputs: per-player/rotation/team stats, highlight reels, tendency/heat maps, rotation charts
- Works on indoor, beach, grass; phone footage; results within ~1 hour
- Sources: https://academy.balltime.com/getting-started/what-is-volleyball-ai , https://balltime.com/faq

## Open-source repos
| Repo | What it gives | Licence |
|---|---|---|
| github.com/ddecks/beach-volley-vision | Closest end-to-end pipeline: TrackNetV3 ball (fine-tuned, ~18k frames), RF-DETR + ByteTrack players, manual-corner homography, rally state machine, serve detection, score attribution, labelling UI with model-assisted ball labels | MIT |
| huggingface.co/deadfast/beach-volley-vision-models | Fine-tuned TrackNetV3 weights for beach volleyball | MIT |
| github.com/masouduut94/volleyball_analytics | VideoMAE game-state classifier (serve / play / no-play), YOLOv8 6-class (ball, serve, reception, set, block, spike), court segmentation, FastAPI backend; datasets on Google Drive | GPL-2.0 |
| github.com/shukkkur/VolleyVision | YOLOv7-tiny / YOLOv8 ball detection, 5-class action detection, player detection, court segmentation | CC BY-NC-ND (personal use OK, no derivatives redistribution) |
| github.com/qaz812345/TrackNetV3 | Upstream heatmap ball tracker + InpaintNet trajectory repair | MIT |
| openvolley (R: datavolley, ovideo, ovscout2, ovml; Python: ovmlpy) | DataVolley .dvw parsing, video sync, scouting app, YOLO-based player/ball detection | various OSS |
| github.com/PKU-ICST-MIPL/FineSports_CVPR2024 | Fine-grained multi-person sports actions (method reference) | check |

## Labelled datasets
| Dataset | Labels | Size | Notes |
|---|---|---|---|
| VNL-STES (CVPRW 2025) https://hoangqnguyen.github.io/stes/ | Touch events with time + (x,y) location | 8 VNL games, 1,028 rallies, 6,137 events | Best match for touch-level event spotting; broadcast view |
| Volleyball dataset (Ibrahim et al., CVPR 2016) github.com/mostafa-saad/deep-activity-rec | Player boxes, 9 individual actions, 8 group activities | 55 videos, 4,830 labelled clips | Broadcast; old but widely used; skeleton versions exist (PoseC3D) |
| RIT-18 github.com/junwenchen/RIT-18 | 18 compositional rally activities | 51 games, 1,530 clips | YouTube |
| Volleyball Activity Dataset 2014 (TU Graz) | Serve, reception, setting, attack, block, stand, defense | 6 videos, 36k annotations | Austrian league, HD |
| VREN arxiv.org/abs/2209.13846 | Rally-level symbolic notation (actions, positions, ball path) | NCAA / pro | Tactics / prediction, not pixels |
| SportsMOT (ICCV 2023) | Player tracking boxes incl. volleyball | 240 seqs total | Player tracker training/eval |
| SportsJumpMotion github.com/yinmayoo185/SportsJumpMotion | Jump events + jersey-based IDs (basketball, volleyball) | | Player ID / jump spotting |
| Roboflow Universe (many) | Ball boxes (25k imgs), actions (14k imgs), players, court seg, court keypoints | varies | Mostly CC BY 4.0; quality varies |
| VolleyVision / volleyball_analytics Drive datasets | Ball, actions, game-state clips | | See repos |
| DataVolley .dvw scout files + video | Every touch: time, player #, skill, quality, zones | Unlimited if you have them | Weak labels via ovscout sync; example match: GKS Katowice v MKS Bedzin (PlusLiga 2018/19) in openvolley |

## Gaps
- No public dataset of amateur/club single-phone footage with full touch-level labels; Balltime's advantage is a large proprietary set.
- Jersey number OCR for volleyball is thin; generic sports jersey datasets + fine-tuning needed.
- Serve speed / jump height need court homography + camera calibration; no labelled ground truth publicly available.
