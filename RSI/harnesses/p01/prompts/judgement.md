
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Work as a careful Holter reader:
1. Read the dossier and the images. Decide whether the ST segment can be interpreted at all (signal quality, rhythm including rr_irregularity, QRS width). If it cannot, the label is indeterminate_data.
2. List the explanations that remain plausible: ischemia, heart-rate related change, body-position or axis shift, conduction change, noise or artifact, other non-ischemic change (slow drift, reference effect).
3. For the two most plausible, decide which measurement would tell them apart and obtain it with the tools (another lead, another time, a reference window at matched heart rate, the overlay against the lead's own earlier beat, rate_response, episode_context, a raw strip). Skip measurements that cannot change the balance.
4. Tools can be wrong: check a decisive number with a second view (another tool, lead, time or the image).
5. Submit a final assessment with submit_assessment: p_ischemic, your probability that this ST change is ischemic given everything you saw; the label, which must agree with it (ischemic if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids; and a short rationale. Use indeterminate_data only when the signal does not allow ST interpretation; when the evidence conflicts, let the probability show it instead of abstaining.
Then reply with a one-paragraph summary.

