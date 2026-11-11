
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Work as a careful Holter reader:
1. Read the dossier and the images. Decide whether the ST segment can be interpreted at all (signal quality, rhythm including rr_irregularity, QRS width). If it cannot, the label is indeterminate_data.
2. List the explanations that remain plausible: ischemia, heart-rate related change, body-position or axis shift, conduction change, noise or artifact, other non-ischemic change (slow drift, reference effect).
3. For the two most plausible, decide which measurement would tell them apart and obtain it with the tools (another lead, another time, a reference window at matched heart rate, the overlay against the lead's own earlier beat, rate_response, episode_context, a raw strip). Skip measurements that cannot change the balance.
4. Tools can be wrong: check a decisive number with a second view (another tool, lead, time or the image).
5. Weigh the findings for and against ischemia and decide where the balance lies: clearly for ischemia, leaning for, mixed, leaning against, or clearly for a named non-ischemic cause.
6. Set p_ischemic inside the range that annotator_reference (in the dossier) gives. The database annotators measured ST from local references placed in non-ischaemic stretches (K22); annotator_reference rebuilds such a reference. Read episode_at_100_uv and episode_at_150_uv in every involved lead and take the row of the lead that stands out most:
   - episode_at_150_uv true in at least one involved lead (the change still stands out as an episode): p_ischemic from 0.45 to 0.95.
   - episode_at_100_uv true in at least one involved lead, episode_at_150_uv false in all (uncertain zone): p_ischemic from 0.25 to 0.80.
   - episode_at_100_uv false in every involved lead (the reference absorbs the change): p_ischemic from 0.05 to 0.60.
   Inside the range, place p_ischemic by the balance from step 5: near the top when it is clearly for ischemia, in the middle when it is mixed, near the bottom when it clearly favours a named non-ischemic cause; use the whole range, in steps of 0.01. Do not leave the range; the range is not a label, and the label still follows from p_ischemic. Name the row in the rationale (for example "annotator_reference row 150: 0.45-0.95"). If annotator_reference returned an error or no involved lead, use 0.05 to 0.95.
7. Submit a final assessment with submit_assessment: p_ischemic from step 6; the label, which must agree with it (ischemic if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids; and a short rationale. Use indeterminate_data only when the signal does not allow ST interpretation; when the evidence conflicts, let the probability show it instead of abstaining.
Then reply with a one-paragraph summary.

