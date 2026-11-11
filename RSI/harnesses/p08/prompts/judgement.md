
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. Work as a careful Holter reader:
1. Read the dossier and the images. Decide whether the ST segment can be interpreted at all (signal quality, rhythm including rr_irregularity, QRS width). If it cannot, the label is indeterminate_data.
2. Read the annotator_reference row now and write it down; it sets the range of p_ischemic in step 7. The database annotators measured ST from local references placed in non-ischaemic stretches (K22); annotator_reference rebuilds such a reference. Look at every involved lead:
   - Row 150: episode_at_150_uv true in at least one involved lead (the change still stands out as an episode): range 0.45 to 0.95.
   - Row 100: episode_at_100_uv true in at least one involved lead and episode_at_150_uv false in all (uncertain zone): range 0.25 to 0.80.
   - Row absorbed: episode_at_100_uv false in every involved lead (the reference absorbs the change): range 0.05 to 0.60.
   - If annotator_reference returned an error or no involved lead: range 0.05 to 0.95.
   Also note R, the largest peak_deviation_uv.annotator_like_reference among the involved leads (how far the change stands out from that reference).
3. List the explanations that remain plausible: ischemia, heart-rate related change, body-position or axis shift, conduction change, noise or artifact, other non-ischemic change (slow drift, reference effect).
4. For the two most plausible, decide which measurement would tell them apart and obtain it with the tools (another lead, another time, a reference window at matched heart rate, the overlay against the lead's own earlier beat, rate_response, episode_context, a raw strip). Skip measurements that cannot change the balance.
5. Tools can be wrong: check a decisive number with a second view (another tool, lead, time or the image).
6. Weigh the findings for and against ischemia and decide where the balance lies: clearly for ischemia, leaning for, mixed, leaning against, or clearly for a named non-ischemic cause.
7. Place p_ischemic inside the range of the row from step 2, using the whole range in steps of 0.01:
   - Start from the balance of step 6: near the top when it is clearly for ischemia, above the middle when leaning for, in the middle when mixed, below the middle when leaning against, near the bottom when it clearly favours a named non-ischemic cause.
   - Then adjust by R, in both directions, by a few hundredths. Larger R moves p up inside the range: in healthy men, 24-hour Holter specificity was 1.0 when depression had to exceed 0.20 mV (or elevation reach 0.15 mV), and 0.95 at 0.15 mV depression (K26). Smaller R moves p down inside the range (in the absorbed row, R far below 100 uV more so than R close to it): with a smaller deviation criterion (0.05 mV depression instead of 0.1 mV), reader agreement fell (kappa 0.72 to 0.32) and positional ST changes were found in 42% instead of 16% of patients (K17).
   - Do not leave the range; the range is not a label, and the label still follows from p_ischemic. Name the row, R and the balance in the rationale (for example "row 150, R 230 uV, leaning for: 0.78").
8. Submit a final assessment with submit_assessment: p_ischemic from step 7; the label, which must agree with it (ischemic if p_ischemic >= 0.5, otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids; and a short rationale. Use indeterminate_data only when the signal does not allow ST interpretation; when the evidence conflicts, let the probability show it instead of abstaining.
Then reply with a one-paragraph summary.

