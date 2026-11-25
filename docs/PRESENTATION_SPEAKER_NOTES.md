# Speaker Notes: Predicting Serverless Cold Starts

Presentation to accompany the project. Each section is one slide: what to say, then likely follow-up questions with answers.


## Slide 1: Predicting Serverless Cold Starts

**SAY (about 1 minute)**  
Good morning, sir. Our project is about cold starts in serverless computing. In one line: we predict, five minutes ahead, which serverless functions will have a cold start, and then we test what that prediction is worth for pre-warming containers.  
Roadmap: the problem, the data, the features and models, the prediction results, the pre-warming simulation, the robustness checks, and finally limitations and conclusions.

**IF ASKED: what is serverless?**  
A function is a small program that runs when triggered (an HTTP request, a timer, a file upload). The platform runs it inside a container. Examples: AWS Lambda, Azure Functions, Google Cloud Functions, Huawei Cloud FunctionGraph. Pizza-oven analogy: at 3 a.m. the oven is switched off to save electricity, so the first customer waits while it heats up. That wait is a cold start; a ready container gives a warm start.

**IF ASKED: is this a replication of the reference paper?**  
Reference: N. Saravana Kumar and S. Selvakumara Samy, Cold Start Prediction and Provisioning Optimization in Serverless Computing Using Deep Learning, Concurrency and Computation: Practice and Experience, 2025. We adapt its ABiGRU model to the Huawei trace. It is an adaptation, not an exact reproduction. We do not try to beat the paper’s 90.36 percent accuracy, because its data (10,000 AWS Lambda rows, split not described) and definitions are different, so the numbers are not comparable.

**IF ASKED: what is the contribution?**  
A leakage-audited benchmark on a public production trace; strong simple rule baselines; multi-seed results with significance tests; an honest negative result for brand-new functions; and a pre-warming simulator compared against keep-alive and the Azure hybrid-histogram policy. We do not claim a new state-of-the-art architecture.


## Slide 2: What is a cold start?

**SAY (about 1.5 minutes)**  
A cold start is the extra delay when a function is called and no ready container exists. The platform must create the container, load the function code and its dependencies, initialize the runtime, and only then run the handler. Huawei’s documentation splits the wait into four parts: pod allocation, code deployment, dependency loading, and scheduling and network overhead.  
Why it matters: it adds latency, from milliseconds to seconds, to user-facing requests. Providers cannot just keep everything warm, because idle containers cost memory and CPU. So it is a latency versus cost trade-off. If we could predict cold starts, we could start containers shortly before they are needed (pre-warming) and keep idle capacity small.

**IF ASKED: why is this hard?**  
1. Irregular demand: some functions run on timers, some on user traffic, some rarely.  
2. No future information: the prediction may use only the past.  
3. Class imbalance: cold starts are a minority event, about 13 percent of function-time-slots in our data.  
4. New functions have no history to forecast from.  
5. Costs differ: one cold start may cost 0.05 s, another 20 s, so counting events is not the same as saving time.  
6. Public traces are incomplete; ours contains cold starts only.

**IF ASKED: what do platforms do today?**  
Mostly keep-alive: keep a container warm for some minutes after use. We compare against 15 and 60 minute keep-alive. The Azure paper (Shahrad et al., USENIX ATC 2020) uses a per-function histogram of idle times; we implement an adapted version as a baseline (slide 15).

**IF ASKED: what other work exists?**  
The reference paper reviews traffic prediction, reinforcement learning for proactive initialization, container caching, SARIMA autoscaling, function fusion, FaaSLight code loading, and LSTM plus RL warm-up. Its own limits include dependence on workload patterns and difficulty in setting warm-up parameters.


## Slide 3: Predict cold starts one bucket ahead

**SAY (about 1.5 minutes)**  
Formal statement: for a function f and a 5-minute bucket t, estimate the probability that f has at least one recorded cold start in bucket t, using only information from buckets before t. Then turn the probability into a yes or no with a threshold chosen on validation data. So it is binary classification, one step ahead.  
The label y(f,t) is 1 if the function has at least one cold-start row in that bucket, else 0.

**IF ASKED: why 5 minutes?**  
Long enough to contain events for most active functions, short enough to act on for pre-warming. 12 buckets are one hour, 288 buckets are one day, 31 days give 8,928 buckets. It matches the original project setting and was not tuned. Other bucket sizes were not tested.

**IF ASKED: why F1 and not accuracy?**  
Only about 13 percent of function-buckets are positive (57,739 of 443,520 test rows). Always predicting no gives 0.8698 accuracy, so accuracy is misleading. We report precision (low means wasted pre-warming), recall (low means missed cold starts), F1 (balance), MCC (fair under imbalance), PR-AUC (ranking quality for rare positives) and ROC-AUC.

**IF ASKED: how is the threshold chosen?**  
For each model, the threshold that maximizes F1 on the validation days, scanning 400 quantile thresholds between the 50th and 99.95th percentile of validation scores. Rules output 0 or 1, so they use 0.5. The test days are never used for tuning.

**IF ASKED: what do you NOT claim?**  
We cannot tell whether an individual request was cold or warm. The data only contains cold-start events, so no event means no recorded cold start, not that the function ran warm.


## Slide 4: Huawei Cloud public trace, Region 5

**SAY (about 1.5 minutes)**  
The data is the public Huawei Cloud 2025 release (GitHub sir-lab/data-release), from the paper Serverless Cold Starts and Where to Find Them, EuroSys 2025. We use Region 5: 31 daily CSV files, 805,745 rows, where every row is one cold-start event. A function is identified by function name plus user ID plus pool name, which gives 527 functions.  
We turn the CSVs into two matrices of 527 functions by 8,928 buckets: the count of cold starts, and the summed cold-start cost (seconds). Each row has the time, the cold-start cost, the pod and cluster, a dependency flag, and the code, scheduling and allocation cost parts.

**IF ASKED: the key limitation of the data?**  
Every row is a cold start. There are no warm invocations, no request counts and no CPU or memory utilization. So an empty bucket does not prove the function was idle. The target is will a cold-start row appear, not is this request cold or warm. A consequence: in the simulator, avoided cold starts are conservative, because we cannot model a warm container serving later requests.

**IF ASKED: data cleaning?**  
No missing values in the CSVs (verified). Two function identities have missing pool information because their pod name does not match the usual pattern; XGBoost handles missing values natively and the neural models set them to 0 after standardization. Empty buckets are zeros. No rows are removed.

**IF ASKED: other regions?**  
The loader supports other regions (CS_REGION and region_prep.py, which reads the zip day by day without extracting). But regions 1 to 4 have not been modeled, so every result here is Region 5 only.

**IF ASKED: class balance?**  
About 13 percent of function-buckets are positive in the test frame.


## Slide 5: Strict time-based splits

**SAY (about 2 minutes)**  
Splits are always by time, never random. In real use a model predicts the future from the past; a random split would let it see the day after a bucket and give optimistic scores.  
Days 0 to 6 are history only: they feed the 7-day-ago features and the static per-function statistics, so they are not training targets. Train is days 7 to 23 (2,110,176 rows from 431 functions). Validation is days 24 to 26 (372,384 rows), used for early stopping and for choosing thresholds. Test is days 27 to 30 (443,520 rows for the main slice), used only for final scores.

Function roles (decided from days 0 to 23 counts, and never used as a feature):  
Normal, 385 functions: at least 10 cold starts in days 0 to 23. Used for training and as the main test slice.  
Holdout, 96 functions: a random 20 percent of the eligible functions (seed 42), excluded from training. Used for the unseen-function test (110,592 test rows, 14,085 positives).  
Low-history, 46 functions: fewer than 10 cold starts. Included in training, reported as a separate slice.

**IF ASKED: how do you know there is no leakage?**  
Every feature for bucket t reads only buckets before t. We tested this by zeroing a function’s series from t onward and recomputing: the maximum feature difference was 0.0. Static function statistics use days 0 to 6 only (a first version used days 0 to 23; it was found and fixed before any reported result). Pool statistics come from non-holdout functions only. Thresholds are tuned on validation, and the neural scaler uses training rows only.

**IF ASKED: why not cross-validation?**  
Time series: rows of one function are strongly correlated and order matters. Instead we added a rolling-origin check with three expanding windows (slide 16).


## Slide 6: 57 leakage-safe features

**SAY (about 2 minutes)**  
All 57 features are identical for every model, so comparisons are fair, and each uses only buckets before the one being predicted. The groups: clock (hour sine and cosine, slot: 3); hardware (pool CPU, pool memory, cluster: 3); pool profile (1); pool live load of other functions in the same pool (3); recent history lag1 to lag12 (12); seasonal history, the event at the same time 1, 2, 3 and 7 days earlier at offsets of minus 1, 0 and plus 1 bucket (12); seasonal votes slot_hits_7d and its plus-minus 1 version (2); rolling counts over 15 minutes, 1 hour, 6 hours, 1 day and 7 days (5); time since last event (1); intensity, log count and log cost for the last bucket and for the same slot yesterday (4); function profile from days 0 to 6 (9); function hourly rate (1); function age (1). Total 57.

Most important: slot_hits_7d, how many of the last 7 days had a cold start in the same slot, has importance 0.495 in XGBoost, and d1_+0, was there one at this time yesterday, has 0.433. Together about 93 percent. The next, lag6, is only 0.012. So the model is essentially a learned, smoother version of the seasonal rule.

**IF ASKED: do static and pool features help?**  
No. Removing the 11 function-profile features (9 early statistics, the function's hourly rate and its age; 46 left) gives F1 0.9427 versus 0.9426 for all 57 (3 seeds each). Removing every pool feature, including pool CPU and memory (51 left) gives 0.9423 versus 0.9420, one seed. Neither changes the result measurably.

**IF ASKED: what is the cold set?**  
The 10 features of clock, hardware, pool profile and pool live load, with no own history. It is used only for the new-function experiment (slide 11).

**IF ASKED: normalization and feature selection?**  
Trees need no scaling. Neural models standardize the 57 features with training mean and standard deviation, clipped to plus or minus 6, with missing values set to 0. No automated feature selection was done; ablations remove groups instead.


## Slide 7: Rules, gradient boosting and neural networks

**SAY (about 3 minutes)**  
Rules (no training): last bucket; same slot yesterday; same slot 7 days ago; yesterday OR 7 days ago; any event in the last hour; and 7-day majority, which predicts a cold start if at least 4 of the last 7 days had one in this slot. The rules set the minimum standard and showed that most of the task is periodicity.

XGBoost: many small decision trees built one after another, each correcting the previous ones. Four variants: full (57 features), no_static (46), no_pool (51), cold_newfunc (10). Settings: up to 1,500 trees, early stopping after 40 trees without improvement on validation PR-AUC, depth 8, learning rate 0.05, subsample 0.8, column subsample 0.8, min child weight 5, L2 1.0, no class weights. Trees kept: about 356 for full. Training takes 8 to 29 seconds.

Neural models, all one class with two branches. Static branch: the 57 standardized features through 128 then 64 units. Sequence branch: the last 24 buckets (2 hours), each with 7 channels (event, log cost, time sine and cosine, event one day earlier, event seven days earlier, pool activity). Head: dropout then a 64-unit layer to one logit with sigmoid. MLP: static branch only, no sequence (19,905 parameters), an ablation. GRU: 2-layer, hidden 64 (62,977). BiGRU: reads the window in both directions, only inside the past window so there is no leakage (130,625). ABiGRU: BiGRU plus additive attention over the 24 steps plus max-pooling (142,977).  
Training: binary cross-entropy, RMSprop learning rate 0.001 (the paper’s optimizer), gradient clipping 1.0, batch 2,048, 700,000 random rows per epoch, at most 14 epochs, early stopping with patience 3 on validation PR-AUC, best weights restored. Time: MLP about 17 s, GRU 392 s, BiGRU 868 s, ABiGRU about 908 s on an Apple GPU.

**IF ASKED: how does ABiGRU differ from the paper?**  
The paper sums forward and backward outputs and uses a different attention form with a sliding max-pool window. We concatenate directions, use standard additive attention (dimension 32) and global max-pool. Many details are unspecified in the paper, so exact reproduction cannot be verified.

**IF ASKED: seeds?**  
XGBoost full and no_static: seeds 42, 1, 2. All neural models: seeds 0, 1, 2. No ensemble. The models are independent; none uses another’s output.


## Slide 8: Learned models beat the best rule

**SAY (about 3 minutes)**  
These are 3-seed means on the held-out test days 27 to 30. All six learned models land between 0.9409 and 0.9427 F1. The best rule, the 7-day majority, gets 0.9194. So the learned models are about 0.022 F1 better, and this holds in every seed. Paired function-level bootstrap (1,000 resamples): XGBoost minus rule +0.0226, 95 percent interval [+0.0121, +0.0351], p below 0.001; ABiGRU minus rule +0.0236, p below 0.001.  
XGBoost and ABiGRU are tied: ABiGRU minus XGBoost is +0.0010, interval [minus 0.0002, plus 0.0024], p = 0.13.

**IF ASKED: is 0.022 F1 a big improvement?**  
It is small in absolute terms but consistent and significant. The main message is that a strong simple rule already gets 0.919, so gains of any model must be read against that baseline.

**IF ASKED: confidence intervals?**  
Cluster bootstrap over whole functions, because rows of one function are correlated. Seed-0 intervals: ABiGRU [0.926, 0.960], XGBoost full [0.924, 0.958], 7-day rule [0.891, 0.944]. They overlap because function-level variation is large; the paired test is the sharper comparison.

**IF ASKED: confusion matrix for ABiGRU (seed 0, test)?**  
TP 55,263, FP 4,204, FN 2,476, TN 381,577; precision 0.929, recall 0.957.

**IF ASKED: where do learned models NOT win?**  
Reactivation buckets: buckets with no event in the previous hour (306,100 rows, only 1,285 positives, 0.4 percent). There the 7-day rule has the highest F1 (0.870 versus 0.85 for XGBoost and 0.83 to 0.84 for neural), although the learned models rank events much better (PR-AUC 0.91 to 0.93 versus 0.76); the intervals overlap, so it is not conclusive. Low-history functions: only 91 positives, best F1 0.343, very wide intervals.

**IF ASKED: overfitting?**  
XGBoost full: train F1 0.9656, validation 0.9399, test 0.9420. The rule, which learns nothing, shows a similar gap of 0.019, so most of the gap is later days being slightly harder, not overfitting. Validation and test F1 agree.

**IF ASKED: calibration?**  
Expected calibration error 0.0018 to 0.0031; probabilities can be used directly.


## Slide 9: F1 by model across three seeds

**SAY (about 1.5 minutes)**  
This is figure 1. It shows F1 of each learned model over the three seeds: the mean, each individual seed, and the min to max range, against the 7-day rule, for all test buckets and for reactivation buckets. The spread between seeds (standard deviation 0.0002 to 0.0008) is far smaller than the gap to the rule (about 0.022), so the gap is real. The ordering inside the learned models is not meaningful.

Seed details: XGBoost seeds 42, 1, 2 (full: 0.9420, 0.9431, 0.9427; no_static: 0.9430, 0.9424, 0.9427). Neural seeds 0, 1, 2. Each seed’s threshold is tuned on its own validation predictions.

**IF ASKED: is ABiGRU better than the other neural models?**  
ABiGRU is ahead of BiGRU, GRU and MLP by 0.0009 to 0.0014 F1 on average, consistent across seeds, so the attention adds a very small gain. ABiGRU versus MLP is +0.0021 (p below 0.001), but that is about a tenth of an F1 point and far smaller than the bootstrap interval of about plus or minus 0.017. GRU and BiGRU versus MLP show no meaningful difference. With 10 comparisons the Bonferroni level is 0.005; only learned model versus rule and ABiGRU versus MLP pass it.

**IF ASKED: reactivation panel of the figure?**  
XGBoost is about 0.013 to 0.023 F1 ahead of the neural models there, and the 7-day rule still has the highest F1 (0.870).

**IF ASKED: reproducibility?**  
XGBoost re-runs and retrained neural seed-0 runs on the same machine gave identical predictions (difference 0.0). Different hardware (CPU or NVIDIA GPU) may give slightly different neural results.


## Slide 10: Mostly daily and weekly repetition

**SAY (about 2 minutes)**  
This is the key insight. The plot shows the top features of XGBoost by gain share. slot_hits_7d (0.495) and d1_+0 (0.433) together carry about 93 percent; every other feature is at most 0.012 (lag6) and most are 0.003 or less. In plain words: if a function had a cold start at this time of day on most recent days, or yesterday at this time, it will probably have one now. The workload is dominated by daily and weekly repetition, so most of the predictive signal is in two simple features.  
Consequence: extra model capacity buys almost nothing. MLP (no sequence) 0.9409 versus GRU 0.9410, BiGRU 0.9414, ABiGRU 0.9423. The ablations agree: dropping function-profile features or pool features changes nothing.

**IF ASKED: so why use machine learning at all?**  
It still beats the best rule by about 0.022 F1 in every seed, and in every rolling-origin window. It also ranks events much better (PR-AUC 0.987 versus 0.856 for the rule), which matters for budgeted pre-warming. But we are explicit that the gain is modest and a tuned tree model is enough.

**IF ASKED: does this contradict the reference paper’s claim that attention helps?**  
Not a contradiction, but on this dataset attention adds only a very small, consistent gain (about 0.001 F1), not a large one. The paper uses different data and its comparison is against other publications’ numbers, not rerun models.

**IF ASKED: why does XGBoost give no_static a slightly higher F1?**  
0.9427 versus 0.9426: within noise (paired p about 0.05, difference +0.0010).


## Slide 11: Brand-new functions are not predictable

**SAY (about 1.5 minutes)**  
This is our honest negative result. We trained a model with only the 10 cold features (clock, hardware, pool profile, pool live load) and no own history, and evaluated it on the 96 holdout functions on the test days. F1 is 0.239 (95 percent interval 0.175 to 0.300), MCC 0.068, PR-AUC 0.149, ROC-AUC 0.567 where 0.5 is chance. A simple pool-hour-rate score is even weaker: ROC-AUC 0.504. So hardware and pool context do not tell us when a brand-new function will cold start.

**IF ASKED: but holdout functions scored 0.97 earlier?**  
Yes, and the difference is the point. Holdout functions were never seen in training, but when their own recent history is available at inference (the normal case after a few days of operation) all models reach F1 about 0.969 (XGBoost no-pool 0.970) versus 0.948 for the 7-day rule. So the models generalize to new functions as soon as a few days of history exist. What is unsolved is a function with no history at all.

**IF ASKED: what should a platform do for new functions?**  
Our results only show that hardware and pool context is not enough. A practical option, which we did not test, is to fall back to keep-alive until a few days of history accumulate.

**IF ASKED: what about functions with little history?**  
The low-history slice (46 functions, 91 positives in test): best F1 0.343 with XGBoost no-static, best rule 0.243, intervals very wide because of the few positives.


## Slide 12: From prediction to saved cold starts

**SAY (about 2 minutes)**  
F1 does not tell us what the prediction is worth, so we built a simulator (v2/prewarm_sim.py). It replays test days 27 to 30 for the 385 eligible functions, 1,152 five-minute buckets. Before each bucket a policy decides, for each function, whether to keep one container ready. If the function then has a cold start in that bucket, the pre-warm avoids its first cold start and saves that bucket’s average cold-start time. Further cold starts in the same bucket are not avoided, because one container serves one cold start. If no cold start happens, the pre-warm is wasted and one container idles for 5 minutes.  
Metrics: cold starts avoided; wasted pre-warms (share of pre-warms never used); idle container-hours (cost proxy); latency saved is also in the full table.  
Totals in the test window: 82,031 cold starts and 81,939 seconds of cold-start latency. Ceiling for one container per function-bucket: 70.4 percent of cold starts avoided (60.2 percent of latency), because extra cold starts in the same bucket cannot be avoided.

Policies compared: no pre-warming; keep-alive 15 and 60 minutes (warm if a cold start happened in the last 3 or 12 buckets); the hybrid histogram from Azure; rules (same slot yesterday, 7-day majority); models at their validation-tuned thresholds; and fixed budgets of the top N functions per bucket.

**IF ASKED: what are the assumptions?**  
A container started at the end of bucket t minus 1 is ready in bucket t (mean cold start is around a second). Because the trace has only cold starts, the simulator cannot model a pre-warmed container serving later requests or requests routed to another cluster, so the avoided counts are conservative. The platform’s own keep-alive is not changed. This is an offline estimate, not a live measurement.

**IF ASKED: why one container per function?**  
That is the policy unit: warm or not warm for a function in a bucket. Multi-container scaling is not modeled.


## Slide 13: Same benefit, far less idle capacity

**SAY (about 3 minutes)**  
Read the table by trade-off. Keep-alive 60 minutes avoids 68.8 percent of cold starts but warms about 119 containers per bucket, wastes 58.9 percent and idles 6,747 container-hours. The hybrid histogram avoids 70.1 percent with about 126 containers, 60.2 percent waste and 7,263 idle hours. XGBoost avoids 66.4 percent with about 50 containers, 5.9 percent waste and 283 idle hours. ABiGRU avoids 67.4 percent with 51.6 containers, 7.1 percent waste and 350 hours. The 7-day rule avoids 64.4 percent with 7.7 percent waste and 366 hours.  
So the models get within 1.4 to 3.7 points of keep-alive or the hybrid histogram with about 20 to 25 times fewer idle container-hours (6,747 over 283 is about 24; 7,263 over 283 is about 26; 6,747 over 350 is about 19). Against the best rule, the models add 2 to 3 points of cold starts avoided. The ceiling is 70.4 percent.

Other rows of the full table: no pre-warming 0; keep-alive 15 minutes 58.6 percent avoided, 51.5 percent waste, 4,263 idle hours; same slot yesterday 63.8 percent, 9.4 percent, 454 hours; MLP at its threshold 66.7 percent, 6.6 percent, 320 hours. Latency saved: keep-alive 60 58.7 percent, hybrid histogram 59.8, rule 54.3, XGBoost 55.9, ABiGRU 56.9 (ceiling 60.2).

**IF ASKED: why is XGBoost less wasteful than ABiGRU?**  
Different thresholds: XGBoost at 0.538, ABiGRU at 0.437, each tuned for F1 on validation. ABiGRU warms slightly more functions and avoids one more point, at a bit more waste.

**IF ASKED: is the comparison fair?**  
Keep-alive and histogram are not trained, they only see cold-start timestamps. As classifiers on the main slice, the 60-minute keep-alive and the hybrid histogram have recall 0.98 to 0.996 but precision only about 0.40 to 0.41 (F1 0.57 to 0.58) versus about 0.94 for the models. On reactivation buckets the hybrid histogram's precision is 0.023, and keep-alive never fires there because it needs an event in the last hour, so they cannot target rare events.

**CAVEAT to state: the simulator uses one seed per model (XGBoost seed 42, neural seed 0), not the 3-seed means, and it is an offline estimate.**


## Slide 14: The trade-off between benefit and waste

**SAY (about 1.5 minutes)**  
Figure 7 plots cold starts avoided against containers per bucket (and against the share of wasted pre-warms). Fixed-budget curves rank functions by score and warm the top N in every bucket; threshold policies and keep-alive appear as markers. Keep-alive and the hybrid histogram sit far to the right: they buy their last few points with 119 to 126 containers per bucket. The learned policies reach 66 to 67 percent at about 50.  
Threshold policies, which warm a different number of functions in each bucket, sit above every fixed-N budget curve: a fixed N wastes containers in quiet buckets and runs short in busy ones.

Numbers if asked: at N = 40 containers per bucket, the models avoid 50.8 percent versus 49.8 percent for the 7-day vote score, with 9.5 versus 11.3 percent waste. The fixed-budget table goes up to N = 125, where the models reach 70.3 to 70.4 percent, the ceiling.

**IF ASKED: what is the practical reading?**  
A platform can pick an operating point: more containers for more coverage, up to the ceiling. Because the models are well calibrated (ECE 0.002 to 0.003), their probabilities can be used directly for budgeted pre-warming.

**IF ASKED: how large is the advantage of models over the best rule?**  
Two to three percentage points of cold starts avoided at similar container counts, and slightly lower waste for XGBoost (5.9 versus 7.7 percent).


## Slide 15: Hybrid histogram policy from Azure

**SAY (about 2 minutes)**  
We added this baseline so that we compare against a published, production-inspired policy, not only plain keep-alive. Shahrad et al., Serverless in the Wild, USENIX ATC 2020, keep a histogram of idle times per application (1-minute bins, 4-hour range). If the histogram is representative (coefficient of variation of bin counts above 2), the container is unloaded after use, pre-warmed at the 5th percentile of idle times minus 10 percent, and kept until the 99th percentile plus 10 percent. If many idle times fall outside the range, an ARIMA forecast sets the window (85 and 115 percent of the prediction). Otherwise a fixed keep-alive of the full range is used.  
Our four adaptations: (1) the trace holds only cold starts, so idle time is the gap between consecutive buckets that contain a cold start, not between invocations; (2) bins are our 5-minute buckets, range 48 buckets (4 hours); (3) ARIMA is replaced by the median of the last 5 out-of-range gaps, because statsmodels is not in the pipeline; (4) strictly causal: parameters used after event E[i] come only from gaps that ended at or before E[i].  
Two variants were run: fallback 4 hours (as in the paper) and fallback 60 minutes. Results: 70.1 percent avoided in both; waste 60.2 and 60.1 percent; idle hours 7,263 and 7,220.

Branch usage over the test period: fixed keep-alive 4 functions, histogram 124, out-of-range forecaster 260. Because the window it opens is long, it behaves like a keep-alive: recall 0.996, precision 0.398, F1 0.568.

**IF ASKED: is this the real Azure policy?**  
No, it is an adaptation. The real ARIMA branch was not run, and the idle times are cold-start gaps, not invocation gaps. This is listed in our not-calculated items. It is still a fair, strictly causal baseline for this event-only trace.

**IF ASKED: why does it beat keep-alive 60 slightly?**  
Its windows are a little longer (126 versus 119 containers per bucket): 70.1 versus 68.8 percent avoided, with more idle hours (7,263 versus 6,747).


## Slide 16: Robustness of the findings

**SAY (about 2.5 minutes)**  
Each check answers a different doubt.  
Calibration: are the probabilities trustworthy? On test, all learned models have expected calibration error 0.0018 to 0.0031, Brier score 0.0107 to 0.0113, and mean predicted probability 0.130 against an observed rate of 0.130. Isotonic recalibration fitted on validation changes ECE by at most 0.0013, so the raw probabilities can be used directly, for example for budgeted pre-warming.  
Significance: is the gap real? Function-level paired bootstrap with 1,000 resamples. Learned models beat the 7-day rule by +0.0215 to +0.0236 F1, p below 0.001. ABiGRU versus XGBoost: +0.0010, p = 0.13 (tied). ABiGRU versus MLP: +0.0021, p below 0.001. With 10 comparisons the Bonferroni level is 0.005, which only the model-versus-rule and ABiGRU-versus-MLP comparisons pass. Row-level McNemar tests are in extra_eval.md but are optimistic because rows of one function are correlated.  
Rolling origin: does it depend on the test window? XGBoost full versus the 7-day rule on three 4-day test windows with expanding training. Fold 1 (train 7 to 15, test 19 to 22): 0.9550 versus 0.9383, gap +0.0166. Fold 2 (train 7 to 19, test 23 to 26): 0.9399 versus 0.9127, gap +0.0271. Fold 3 (the original split): 0.9420 versus 0.9194, gap +0.0226. Mean 0.9456 versus 0.9235. The advantage holds in every window. Neural models were not rolled (about 15 minutes per model per fold).  
Tuning: is XGBoost under-tuned? Grid of 16 configurations of depth, learning rate and min child weight, selected on validation PR-AUC, which only ranged 0.9862 to 0.9866. The selected setting (depth 6, learning rate 0.1, min child weight 5) scored test F1 0.9429 versus 0.9420 for the default, a difference like seed noise.  
Also: overfitting check, train F1 0.9656 versus test 0.9420 for XGBoost, and the rule shows a similar 0.019 gap.

**IF ASKED: what was not done?**  
Rolling origin and tuning of neural models, and the paper’s baselines rerun on our data.


## Slide 17: 12 neural runs, 144 epochs, all documented

**SAY (about 1.5 minutes)**  
This is figure 8: training loss and validation PR-AUC per epoch for all 12 neural runs (4 models times 3 seeds), with the kept epoch marked. In total 144 epochs were run. Early stopping used validation PR-AUC with patience 3, and the best epoch’s weights were restored.  
Epochs run (seeds 0 / 1 / 2): MLP 11 / 8 / 14; GRU 13 / 14 / 8; BiGRU 14 / 9 / 13; ABiGRU 13 / 14 / 13. Kept epoch: MLP 8 / 5 / 12; GRU 10 / 13 / 5; BiGRU 13 / 6 / 10; ABiGRU 10 / 13 / 10. Seed-0 BiGRU reached the 14-epoch cap, so it was not stopped early. Validation PR-AUC starts around 0.978 after the first epoch and reaches 0.9846 to 0.9856 at best.  
Evidence kept in the repository: console logs (v2/results/logs), per-epoch records (info JSON files), the training report with SHA-256 fingerprints of every saved model, and the histories stored inside each checkpoint. A script cross-checks logs, records and checkpoint histories, and all 12 runs agree epoch by epoch.

**IF ASKED: how long does training take?**  
Seed 0 on an Apple Silicon laptop (MPS): MLP about 17 s, GRU about 392 s, BiGRU about 868 s, ABiGRU about 908 s. XGBoost takes 8 to 29 s. CUDA path is implemented but untested (no NVIDIA GPU was available).

**IF ASKED: are the saved models the ones behind the numbers?**  
Yes. Five runs finished before saving was added and were retrained with the same seed; every retrained run reproduced its original predictions exactly (difference 0.0). Each checkpoint stores weights, the scaler mean and standard deviation, the 57 feature names, window length, seed, history and torch version.

**IF ASKED: parameter counts?**  
MLP 19,905; GRU 62,977; BiGRU 130,625; ABiGRU 142,977.


## Slide 18: What is not done yet

**SAY (about 2 minutes)**  
We state the limits openly.  
Limitations:  
1. One region, one trace. All results are Region 5. Code for other regions exists (CS_REGION and region_prep.py, which reads zips day by day), but regions 1 to 4 have not been modeled, so we make no claims about them. Function counts, roles and thresholds would differ per region.  
2. Brand-new functions with no history are unsolved (F1 0.24, ROC-AUC 0.567).  
3. The trace is event-only: no warm invocations, no request counts, no CPU or memory. So we predict whether a cold start occurs, not load, and the simulator’s avoided counts are conservative.  
4. The pre-warming results are an offline simulation, not a live test, and use one seed per model (XGBoost 42, neural seed 0).  
5. The 5-minute bucket size was not tuned or compared with other sizes.  
6. The hybrid histogram is an adaptation without a real ARIMA branch.  
7. Neural models were not rolled over windows or hyperparameter-tuned; only XGBoost and the rules were.  
8. Not measured: inference latency and memory; the paper’s baselines rerun on our data.  
9. Minor disclosed inconsistency: pool_yday includes the function’s own event of the same slot yesterday (a legitimate past value) while pool_last1 and pool_last12 exclude the function itself.

Future work (proposals, not done):  
Evaluate Regions 1 to 4 with the existing region-aware pipeline; better handling of new functions, for example a fallback policy until history exists; cost-aware decisions that weight by cold-start latency; multi-seed pre-warming simulation; a real ARIMA branch for the histogram baseline; a live pre-warming test.

**IF ASKED: would the result hold on other datasets?**  
We do not know. The strength of seasonal rules suggests that other workloads with strong daily periodicity would behave similarly, but we have no evidence for Azure or other Huawei regions yet.


## Slide 19: Three takeaways

**SAY (about 2 minutes)**  
Three takeaways.  
1. Cold starts are predictable from history. F1 about 0.94 (XGBoost 0.9426 to 0.9427, ABiGRU 0.9423) against 0.9194 for the best rule; the gap is significant and holds in every seed and every rolling window.  
2. Simple is enough. XGBoost ties the neural sequence models; sequence encoders add almost nothing (ABiGRU +0.001 over XGBoost, not significant, p = 0.13) because the signal is daily and weekly repetition (slot_hits_7d and d1_+0 carry about 93 percent of importance).  
3. Prediction-driven pre-warming is far cheaper than keep-alive. 66 to 67 percent of cold starts avoided (ceiling 70.4) with 283 to 350 idle container-hours versus 6,747 for keep-alive 60 and 7,263 for the hybrid histogram, about 20 to 25 times fewer.

Answers to our research questions: RQ1 learned models beat strong seasonal rules, yes by about 2.4 F1 points; RQ2 sequence encoders add practically nothing; RQ3 on reactivation events they rank better but the 7-day rule has the higher F1, inconclusive; RQ4 new functions from hardware and pool context alone, no; RQ5 predictions can drive cost-aware pre-warming, shown by the simulator.

**IF ASKED: so is deep learning useless here?**  
On this data it is not needed. XGBoost trains in about 30 seconds; ABiGRU takes about 15 minutes for the same accuracy. We still report all of them because comparing them was an objective.

**IF ASKED: what would you do next?**  
Regions 1 to 4, a fallback for new functions, latency-weighted decisions, and a live pre-warming experiment.

**IF ASKED: where is everything?**  
README and docs/PROJECT_DOCUMENTATION.md; results in v2/results; figures in v2/figures; trained models in v2/models; an IEEE-style paper draft in report_overleaf. Trained models can be used without retraining via python3 v2/predict.py.

Finish: Thank you. We are happy to take questions.
