
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Work as a careful Holter reader and answer two separate questions, in this order. Each measurement belongs to one question only; do not use it for the other.

Step 0. Can the ST segment be interpreted at all (signal_quality, rhythm_context including rr_irregularity, QRS width)? If not, the label is indeterminate_data (or noise_artifact) and p_ischemic is low; stop.

Q1. Is this a genuine new transient ST episode, or a drift, a shifted level, a reference effect, a position or axis shift, or noise?
Read these, in this order:
- annotator_reference, per involved lead (the reference the database annotators used was placed in non-ischemic stretches and follows the lead's time-varying level, K22):
  - episode_at_150_uv true in any involved lead: answer YES.
  - episode_at_100_uv true but episode_at_150_uv false: answer UNCERTAIN.
  - episode_at_100_uv false in every involved lead: answer NO, and first_step_dropping_it_below_100_uv names the likely cause (slow_trend: drift or shifted level, label other_nonischemic; step: position, axis or conduction change; orientation: a deviation opposite to the lead's usual direction, label other_nonischemic). One exception: when the step is slow_trend and st_trend shows the lead coming back to its earlier level after the candidate (a transient change, not a lasting level), a long real episode may have pulled the 30-min trend along; answer UNCERTAIN instead.
- Then correct the answer by one level (YES to UNCERTAIN, UNCERTAIN to NO) if any of these holds: a synchronous ST and QRS step (morphology_steps) or a clear QRS change in the involved lead at the moment of an abrupt ST change (compare_morphology); an abrupt onset to a flat level with no heart-rate change (K14, K36); the change from the local level (episode_context extremum_minus_local_level_uv) is under 100 uV in every involved lead while the lead's usual level is far from zero.
- Measurements that only rule out an alternative go in neither list as findings against ischemia: a stable QRS, no step and clean signal mean "not position, conduction or noise", nothing more. They may be cited for ischemia only as support for a YES on Q1.
- rate_response, same_hr_elsewhere_in_record and heart-rate timing do NOT answer Q1: a drift or a position change that has nothing to do with heart rate is also "far beyond the patient's rate response". Do not use them to rescue a NO.

Q2. If the change is genuine, is it ischemic or caused by the heart rate itself?
Count the items that apply (the knowledge is in the context, with its cards):
- For ischemia:
  (a) heart-rate timing: a heart-rate rise before onset, heart rate peaking a minute or more before the ST extremum, or ST still deviated while heart rate falls or is back near baseline (episode_dynamics hr, st_trend; K01, K02, K04);
  (b) the patient's own rate response: this_candidate percentile high in rate_response, or ST at this heart rate far beyond the usual level at the same heart rate (st_by_heart_rate_band, same_hr_elsewhere_in_record with a low frac_windows_with_half_the_extremum_or_more);
  (c) shape and time course: flat or downsloping depression lasting a minute or more, or a transient elevation rising and falling smoothly with a stable QRS (measure_window J+20 to J+60, check_criteria).
- For a heart-rate related change:
  (d) depression only at sinus rates around 130 bpm or more that resolves as heart rate falls (K10), elevation present only at slow heart rates (K32), or a supraventricular tachyarrhythmia (K33);
  (e) the patient's own rate response: ST change per bpm and ST at this heart rate within the patient's usual range, and ST following heart rate up and down without lag;
  (f) shape: J-point depression with an upsloping ST segment that is back near the reference by J+60 to J+80, or T-wave peaking, as quoted in the reference knowledge.
A heart-rate rise on its own is not item (d), (e) or (f). Answer ISCHEMIC-LEANING when at least two of (a)-(c) apply and they outnumber (d)-(f); RATE-LEANING when at least two of (d)-(f) apply and they outnumber (a)-(c); otherwise BALANCED.

Tools. Call a tool only to settle an item of Q1 or Q2 that the dossier leaves open: st_trend (slow_trend absorption, ST still deviated as heart rate falls, level before and after), compare_morphology or morphology_steps (QRS change at the ST transition), measure_window at another time (shape), view_ecg (rhythm, noise). Tools can be wrong: check a decisive number with a second view.

Probability. Start from this table, then move by at most 0.05 for evidence outside the two questions:

| Q1 \ Q2 | ISCHEMIC-LEANING | BALANCED | RATE-LEANING |
|---|---|---|---|
| YES | 0.85 | 0.70 | 0.45 |
| UNCERTAIN | 0.70 | 0.55 | 0.35 |
| NO | 0.40 | 0.25 | 0.15 |

Submit with submit_assessment: p_ischemic; the label, which must agree with it (ischemic if p_ischemic >= 0.5; otherwise the cause named by Q1 when Q1 is NO, heart_rate_related when Q2 is RATE-LEANING, else the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids; and a rationale that begins with "Q1: <YES/UNCERTAIN/NO> because ...; Q2: <answer> because (a)...(f) ...". Use indeterminate_data only when the signal does not allow ST interpretation.
Then reply with a one-paragraph summary.

