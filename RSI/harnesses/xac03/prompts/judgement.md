
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Turn it into a probability with the five steps below, in this order, and write the result of each step in your rationale ("Step 2: anchor 0.50 (V5 -170 uV)", "Step 3: +0.10 a, -0.05 g", ...).

Step 1 - Can the ST segment be read? From signal_quality, rhythm_context (including rr_irregularity) and the images: if the involved leads are unusable (noise, saturation, flat line), the rhythm is irregular although beats are labelled normal, or the QRS is wide throughout, the label is indeterminate_data or noise_artifact with a low p_ischemic. Otherwise go on.

Step 2 - Anchor on the size of the change. Take the involved lead with the largest |peak_uv| in the candidate line. Check it against episode_context: if |extremum_minus_local_level_uv| of that lead is less than half of |peak_uv|, the lead was already off the record reference before the change [K22, K23]; then use the local change as the size. Same bands for depression and elevation:
- below 100 uV: start at 0.25 (below the ACC/AHA 0.1 mV criterion);
- 100-149 uV: start at 0.35 (lowering the criterion from 0.1 to 0.05 mV raised positional changes and cut reader agreement [K17]: the smaller the change, the weaker the evidence);
- 150-199 uV: start at 0.50 (specificity 0.95 for flat or downsloping depression of 0.15 mV or more [K26]);
- 200 uV or more: start at 0.65 (specificity 1.0 for flat or downsloping depression over 0.20 mV or elevation of 0.15 mV or more [K26]); add up to 0.10 more as the change grows well beyond that.

Step 3 - Adjust for what this candidate shows. Each item present moves p by about 0.05 to 0.15 (use the larger step when the finding is clear in the numbers and the image); keep the total adjustment within +-0.30, except the decisive items marked (D).
For ischemia (+):
- a. Timing: heart rate peaks before the ST extremum (episode_dynamics hr_max_minus_st_peak_s NEGATIVE means heart rate peaked first; positive means after) and ST is still deviated after heart rate has fallen back, or ST stays level until a heart rate and then falls steeply [K01, K02].
- b. The ST change develops at a steady, normal heart rate, with little or no heart-rate rise before onset [K29, K08].
- c. Flat or downsloping depression from J+20 to J+80 [K26], or depression still deep at J+80 despite an upslope [K24].
- d. Smooth, gradual onset and offset over minutes [K16]; or elevation in one lead with opposite-sign deviation in another [K28].
- e. A QRS change that grows and recedes together with the ST change in the most deviated lead and is gone once ST has recovered fits ischemia itself [K18, K19]: count it +0.05 at most, never against.
Against ischemia (-):
- f. (D) Position or artifact step: ST starts and stops abruptly (within a window or two) at the same moment as a QRS step, with little heart-rate change (morphology_steps synchronous_st_and_qrs_steps, or the raw strip), or P and R waves grow together, or PQ shifts with ST [K14, K27]; or an abrupt shift that then stays level for hours without heart-rate change [K36]. QRS steps that are near but not synchronous with the ST change, or a QRS change that develops gradually, are not item f. If f is clearly present, p may go down to 0.10-0.25 whatever the size.
- g. Upsloping depression that is small at J+80 [K25].
- h. Elevation present only at slow heart rates that disappears whenever heart rate rises [K32].
- i. ST follows heart rate closely up and down after a small, short heart-rate rise; a small short rise makes demand ischemia less likely than a large long one [K04].
- j. (D) Bizarre, wider QRS complexes during the change (conduction change, Long-Term ST Database classes): p may go down to 0.10-0.25 whatever the size.
- k. episode_context shows the candidate is a fragment of a longer change, the return between two episodes, or a reference effect rather than a new change.
(D) on the ischemic side: when the size is 150 uV or more and a and c (or a and d) are both clearly present, p may go up to 0.90.

Step 4 - Not evidence in either direction. Do not count these as for or against; mention them at most once as a limitation:
- rate_response showing the candidate's ST change per bpm, or its ST level in a heart-rate band, far beyond the patient's other surges: those surges are taken outside ST episodes, so nearly every candidate lies far beyond them, ischemic or not. Only the opposite finding counts: a response inside the usual range (between the outside-episode p25 and p75) is item i-like evidence, -0.05.
- A heart-rate rise before onset by itself: most ischemic episodes in daily life follow one [K03]; only its timing and size count (items a, b, i).
- A QRS amplitude or correlation change by itself, without its timing [K14, K18, K19].
- Record-wide properties, the same for every candidate of this patient: lead set or non-standard lead names, eligibility flags (low R wave, baseline ST off zero), and the number or look of the patient's other candidates [K43].
- Not meeting the ACC/AHA duration rule: it is a criterion for a positive finding, not proof of a non-ischemic change.

Step 5 - Tools, then submit. Call a tool only when an item of Step 3 is unclear and could move p by 0.10 or more: measure_window or compare_morphology before, at and after the change (does a QRS change recover with ST? flat or upsloping?), view_ecg at onset (abrupt step, artifact, P waves), st_trend around the candidate (gradual or step, fragment of a longer change). Tools can be wrong: check a decisive number with a second view. Then submit with submit_assessment: p_ischemic = anchor + adjustments, kept within 0.05-0.95; the label must agree (ischemic if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids (Step 3 items only); and a short rationale with the anchor and each adjustment. Use indeterminate_data only when the signal does not allow ST interpretation.
Then reply with a one-paragraph summary.

