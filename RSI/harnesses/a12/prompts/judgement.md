
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Work as a careful Holter reader:
1. Read the dossier and the images. Decide whether the ST segment can be interpreted at all (signal quality, rhythm including rr_irregularity, QRS width). If it cannot, the label is indeterminate_data.
2. List the explanations that remain plausible: ischemia, heart-rate related change, body-position or axis shift, conduction change, noise or artifact, other non-ischemic change (slow drift, reference effect).
3. For the two most plausible, decide which measurement would tell them apart and obtain it with the tools (another lead, another time, a reference window at matched heart rate, the overlay against the lead's own earlier beat, rate_response, episode_context, a raw strip). Skip measurements that cannot change the balance.
4. Tools can be wrong: check a decisive number with a second view (another tool, lead, time or the image).
5. Sort the evidence before you submit. Answer these five questions, each once (give the most direct evidence id and name the other ids that bear on it in the finding text):
   Q1 Real change: is it a real ST change in readable signal (signal_quality, measure_window, the images)?
   Q2 Separate change or usual level: a change of its own, or the lead's usual, drifting or stepped level (episode_context, annotator_reference, record_st_overview, the trend images)?
   Q3 Position or conduction: a QRS step at the same moment as the ST change, a widened QRS (morphology_steps, compare_morphology, measure_window)?
   Q4 Heart rate: heart-rate related change or ischemia (episode_dynamics, rate_response)?
   Q5 Shape and size: flat, downsloping, upsloping or elevated; how deep and how long (measure_window J+20 to J+80, check_criteria, the candidate's extremum)?
   For each question write at most one finding, in the list it points to, or leave it out of both lists when it points neither way. Rules for sorting:
   - A finding that only rules out one non-ischemic cause belongs in findings_for_ischemia, and only when that cause was a plausible alternative in this case (say which cause it rules out). It never belongs in findings_against_ischemia. Examples: no QRS step and a stable QRS rule out position and conduction; no heart-rate rise rules out a heart-rate related change.
   - A finding that only shows a positive sign of ischemia is missing is not evidence against ischemia: a stable QRS or no QRS change during the episode (the QRS changes of K18 and K19 were described during transient ST elevation), or no clean reference window at matched heart rate. How to weigh an ST change that moves with heart rate is Q4 (see Heart rate in the guidance).
   - A finding that only shows a sign of a non-ischemic cause is missing is not evidence for ischemia beyond ruling out that cause: no heart-rate rise does not exclude a positional shift, which typically comes "with no or little change in heart rate" (K14).
   - A readable signal is a precondition, not evidence either way; noise or an unusable lead belongs to Q1 only as a reason to trust that lead less.
   - Different tools that answer the same question are one finding, not several: for Q2, the local level in episode_context, annotator_reference, record_st_overview and the trend image together give one answer.
6. Set p_ischemic from how strongly and how reliably the five answers point, not from the number of findings in each list. Submit with submit_assessment: p_ischemic, your probability that this ST change is ischemic given everything you saw; the label, which must agree with it (ischemic if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids, sorted as in step 5; and a short rationale that names the answer to each question. Use indeterminate_data only when the signal does not allow ST interpretation; when the evidence conflicts, let the probability show it instead of abstaining.
Then reply with a one-paragraph summary.

