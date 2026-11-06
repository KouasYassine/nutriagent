From 035d945d8d922f850a6edea8322e2a5dadc78daf Mon Sep 17 00:00:00 2001
From: Yassine Kouas <yassine.kouas@enpc.fr>
Date: Fri, 6 Nov 2026 20:39:35 +0100
Subject: [PATCH] README: final validation-calibrated uncertainty results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E123t5saottygiEnp298CX
---
 README.md | 26 ++++++++++++++------------
 1 file changed, 14 insertions(+), 12 deletions(-)

diff --git a/README.md b/README.md
index a2868e4..b1454a0 100644
--- a/README.md
+++ b/README.md
@@ -27,21 +27,23 @@ Test set: 349 dishes never used for training or model selection. The table shows
 
 ### Uncertainty: are the 90% ranges honest?
 
-Three ways to build a 90% range were compared, all calibrated on the validation split and checked on the test split.
+Three ways to build a 90% range were compared, all calibrated on the validation split (349 dishes) and checked on the test split (349 dishes).
 
-| Calories, 90% target | Overall | Small plates | Medium | Large |
-|---|---|---|---|---|
-| Fixed width (absolute score) | 89.1% | 98.3% | 89.7% | **79.3%** |
-| Width scaled with portion (relative score) | 86.5% | **77.8%** | 90.5% | 91.4% |
-| **Calibrated per portion size (Mondrian)** | _see `results/uncertainty/coverage.md`_ | | | |
+| Method | Calories: overall | Small plates | Medium | Large | Mean width | **Worst group, all 5 nutrients** |
+|---|---|---|---|---|---|---|
+| Fixed width (absolute score) | 89.1% | 98.3% | 89.7% | 79.3% | 243 kcal | **71.6%** (carbs, large plates) |
+| Width scaled with portion (relative score) | 86.5% | 77.8% | 90.5% | 91.4% | 291 kcal | **77.8%** (calories, small plates) |
+| **Calibrated per portion size (Mondrian)** | **88.0%** | **86.3%** | **88.8%** | **88.8%** | **272 kcal** | **86.3%** (calories, small plates) |
 
-Plate size groups are terciles of true calories.
+Plate size groups are terciles of true calories, about 116 test dishes each.
 
-- **Overall coverage** is close to 90% for every method, as conformal prediction guarantees.
-- **Per plate size, the first two methods miss in opposite directions:**
-  - fixed-width ranges are too narrow for large plates (79% coverage);
-  - portion-scaled ranges are too narrow for near-empty plates (78% coverage).
-- **Mondrian conformal prediction** calibrates small, medium and large *predicted* portions separately. This guarantees 90% coverage within each group, not only on average, so it is the method the app uses.
+- **The first two methods each fail on one plate size:**
+  - fixed-width ranges are badly overconfident on large plates;
+  - portion-scaled ranges are overconfident on near-empty plates.
+- **Mondrian conformal** calibrates small, medium and large *predicted* portions separately. It raises the worst-case coverage from 72% to 86% across all nutrients, and gives small plates ranges about half as wide as the fixed-width method (104 vs 193 kcal). The app uses this method.
+- **The remaining gap to 90%** is within sampling noise for about 116 dishes per group (standard error ≈ 2.8 points).
+
+Full table for all nutrients: [`results/uncertainty/coverage.md`](results/uncertainty/coverage.md).
 
 ## How it works
 
-- 
2.43.0