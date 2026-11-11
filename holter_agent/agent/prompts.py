"""Prompt text for the agent sessions. Kept stable across calls so the prefix can be cached."""
from __future__ import annotations

import json

from .state import LABELS

_LABEL_TEXT = """\
Labels:
- ischemic: ST change best explained by myocardial ischemia.
- heart_rate_related: ST change that follows heart-rate changes.
- axis_or_position_shift: ST shift caused by a change of electrical axis or body position.
- conduction_change: ST shift caused by an intermittent conduction change.
- noise_artifact: the change is produced by noise, saturation or electrode problems.
- other_nonischemic: a real ST change with a non-ischemic cause not listed above (e.g. slow drift).
- indeterminate_data: the evidence needed to decide is missing or unusable (signal quality, leads).
- indeterminate_ambiguous: the evidence is there but the ST change cannot be interpreted \
(e.g. paced or bundle-branch-block beats) or the findings conflict.
"""

# Labels for graded output (harness 0.8): no "ambiguous" abstention, and heart_rate_related defined by its cause
# rather than by tracking heart rate, which demand ischaemia does as well.
_LABEL_TEXT_V08 = """\
Labels (the label must agree with p_ischemic):
- ischemic: ST change best explained by myocardial ischemia.
- heart_rate_related: ST change caused by the heart-rate change itself, without ischemia.
- axis_or_position_shift: ST shift caused by a change of electrical axis or body position.
- conduction_change: ST shift caused by an intermittent conduction change.
- noise_artifact: the change is produced by noise, saturation or electrode problems.
- other_nonischemic: a real ST change with a non-ischemic cause not listed above (e.g. slow drift, a change \
relative to a reference level that is not a new episode).
- indeterminate_data: the signal does not allow ST interpretation (quality, leads, rhythm, wide QRS).
"""

_CONTEXT = """\
You review candidate ST-segment episodes in 24-hour 12-lead Holter recordings and decide whether \
each reflects myocardial ischemia or a non-ischemic cause. You cannot see the ECG; you work only \
through tools that return measurements. The ST reference level and the episode rule are fixed for \
this run and cannot be changed. Times are seconds from the start of the record; ST values are in \
microvolts.

Reference knowledge (quoted from the sources):
- Long-Term ST Database annotation classes (Jager et al., Med Biol Eng Comput 2003): axis shifts from \
body-position changes are "characterised by a change in the Q-, R- or S-wave amplitude"; intermittent \
bundle-branch block or other conduction changes "by bizarre and wider QRS complexes"; slow drifts come \
from heart-rate related diurnal changes or medication. Heart-rate related ST changes typically show "J \
point depression with positive slope; moving of T-wave into ST segment; T-wave peaking; or parallel \
shift of ST segment"; ischemic changes "horizontal flattening; down sloping; scooping; or elevation". \
The annotators separated heart-rate related from ischemic changes partly with the subject's clinical \
information, which you do not have; when the ECG evidence cannot separate them, say so.
- ACC/AHA 1999 ambulatory ECG guideline: ischemia is "flat or downsloping ST-segment depression >=0.1 \
mV, with a gradual onset and offset that lasts for a minimum period of 1 minute", episodes separated \
by at least 1 minute (5 minutes recommended). Left ventricular hypertrophy, pre-excitation, left \
bundle-branch block or intraventricular conduction delay >=0.10 s make ST unsuitable for ischemia \
detection; digoxin and some antidepressants distort the ST segment.
- Panza et al. (J Am Coll Cardiol 1992): 89% of ischemic episodes during daily life were preceded by a \
heart-rate rise of >=10 bpm, largest in the 5 minutes before onset. A heart-rate rise therefore does \
not by itself make an episode non-ischemic.
"""

# Guidance 0.7a: how to use the knowledge above. Written after reading the agent's failures on LTSTDB
# development records (outputs/failure_analysis): criteria for a positive finding were used as
# requirements, typical features of a class as exclusion rules, and rules for 12-lead ECGs on 2-3 lead
# records. Each rule restates what the quoted sources say or follows from them; none is fitted to the
# LTSTDB labels.
_READING_RULES_A = """
How to use this knowledge:
- A criterion for a positive finding is not a requirement. Not meeting the ACC/AHA criterion (too short, \
too small, upsloping, or an ST elevation, which it does not address) means the episode is not a \
guideline-positive ST depression; it does not show that the change is non-ischemic.
- ST elevation can be ischemic: "elevation" is one of the ischemic changes of the Long-Term ST Database \
classes above, and the international criteria for vasospastic angina (COVADIS; Beltrame et al., Circ J \
2016) count transient "ST segment elevation >=0.1 mV", "ST segment depression >=0.1 mV" or "new \
negative U waves" in at least 2 contiguous leads during a spontaneous episode.
- Heart rate: a heart-rate rise before or during an episode does not make it heart-rate related, and the \
absence of a rise does not argue against ischemia (in Panza et al., 11% of ischemic episodes were not \
preceded by a rise of >=10 bpm). Test heart-rate dependence directly: compare the ST deviation in the \
episode with the ST deviation at the same heart rate elsewhere in the record. If ST at the same heart rate \
is near the reference elsewhere, heart rate alone does not explain the change. An ST change that deepens \
while heart rate is stable or falling argues against a heart-rate related change, not for it.
- Leads: a record may have only 2 or 3 leads, often with non-standard names. Rules that need two \
contiguous leads or a lead territory assume a standard 12-lead ECG; where they cannot be applied, a \
change seen in only one monitored lead is not evidence against ischemia.
- QRS changes: an axis or body-position shift changes the Q-, R- or S-wave amplitude at the same time as \
the ST shift. A QRS change flagged on a small QRS complex, or a QRS change that is small next to the ST \
change, does not by itself explain the ST change. A large ST shift can move the automatically detected \
J point, which inflates the QRS duration and the J-to-ST slope; tools warn when this may have happened.
- Typical features are not exclusion rules: "J point depression with positive slope" is typical of \
heart-rate related change, but an upsloping ST does not exclude ischemia, and the measured slope depends \
on the automatic J point (compare the J+40 and J+80 levels and the image).
- Recurrence of similar episodes in a record is not by itself evidence for or against ischemia.
- Use only the criteria quoted here and in tool outputs; do not invent thresholds or rules. Choose the \
label that explains most of the findings, not the one suggested by the first criterion that is not met.
"""

# Guidance 0.7b. Version a only removed reasons to rule ischemia out; on 70 development candidates the
# blind first impression became "ischemic" for 57 (v06: 13) and specificity fell from 0.98 to 0.32. Version b
# states each rule in both directions, asks for positive evidence of ischemia in the ST change itself,
# and treats the same-heart-rate comparison as supporting evidence only.
_READING_RULES_B = """
How to use this knowledge:
- Criteria and typical features are evidence, not verdicts, in both directions. Meeting the ACC/AHA \
criterion does not by itself make a change ischemic, and not meeting it (too short, too small, upsloping, \
or an ST elevation, which it does not address) does not make it non-ischemic.
- Ischemia needs positive evidence of its own, not only the absence of other causes. Ruling out noise, \
axis or position shift and conduction change still leaves heart-rate related change, slow drift and \
ischemia. Decide between them on the ST change itself: its shape (horizontal or downsloping depression, \
scooping or elevation, versus J-point depression with positive slope, T-wave peaking or a parallel \
shift), its time course (a distinct episode with gradual onset and offset, versus a slow drift or a \
deviation already present before the candidate), and its relation to heart rate.
- ST elevation can be ischemic: "elevation" is one of the ischemic changes of the Long-Term ST Database \
classes above, and the international criteria for vasospastic angina (COVADIS; Beltrame et al., Circ J \
2016) count transient "ST segment elevation >=0.1 mV", "ST segment depression >=0.1 mV" or "new \
negative U waves" in at least 2 contiguous leads during a spontaneous episode.
- Heart rate: a heart-rate rise before or during an episode does not by itself make it heart-rate \
related, and the absence of a rise does not by itself argue against ischemia (in Panza et al., 11% of \
ischemic episodes were not preceded by a rise of >=10 bpm). A heart-rate related change follows heart \
rate within the episode: it grows as heart rate rises and recedes as it falls. An ST change that deepens \
while heart rate is stable or falling does not follow heart rate. A smaller ST deviation at the same \
heart rate elsewhere in the record shows that the heart-rate level alone does not account for the \
change; it does not show what does, so it supports but does not establish ischemia.
- Leads: a record may have only 2 or 3 leads, often with non-standard names. Rules that need two \
contiguous leads or a lead territory assume a standard 12-lead ECG; where they cannot be applied, a \
change seen in only one monitored lead is not evidence against ischemia.
- QRS changes: an axis or body-position shift changes the Q-, R- or S-wave amplitude at the same time as \
the ST shift. A QRS change that coincides with the ST change and is comparable in size supports that \
cause; a QRS change flagged on a small QRS complex, or one that is small next to the ST change, does not \
by itself explain the ST change. A large ST shift can move the automatically detected J point, which \
inflates the QRS duration and the J-to-ST slope; tools warn when this may have happened.
- Slope: "J point depression with positive slope" is typical of heart-rate related change, but typical \
features are not exclusion rules, and the measured slope depends on the automatic J point (compare the \
J+40 and J+80 levels and the image).
- Recurrence of similar episodes in a record is not by itself evidence for or against ischemia.
- Use only the criteria quoted here and in tool outputs; do not invent thresholds or rules. Weigh the \
evidence for each candidate label; if the ECG evidence fits ischemia and a non-ischemic cause equally \
well, say so with indeterminate_ambiguous rather than defaulting to either.
"""

# Guidance 0.8 (harness 0.8). Written after reading 70 GPT-6 Astra cases under v07b
# (outputs/runs/devsub_gpt-6-astra_high_both_base_v07b/review/): a strong model applied the heuristic
# sentences of v07b mechanically (heart-rate tracking, "deepens while heart rate is stable or falling",
# "comparable in size", "deviation already present before the candidate"), abstained in 38% of cases, and
# judged absolute contours and deviations from a record-wide reference rather than the change itself. These
# rules say what the ECG can and cannot separate, make the remaining checks operational, and ask for a
# probability instead of abstention. Sources: the quotes above; ST/HR hysteresis (ST at the same heart rate
# differs between rising and falling heart rate in exercise testing); Smrdel & Jager 2002 (axis shifts
# detected as steps).
_READING_RULES_C = """
How to use this knowledge:
- Weigh the evidence; do not apply rules mechanically. Criteria and the typical features quoted above are \
evidence in either direction, never verdicts. When the evidence is mixed, express it as a probability.
- Heart rate. ST changes that grow and recede with heart rate occur in heart-rate related change and in \
demand ischemia alike (most ischemic episodes in daily life follow a heart-rate rise). ST at the same heart \
rate can differ between rising and falling heart rate (ST/heart-rate hysteresis), so a short dissociation \
around a heart-rate step is not evidence either way. What the ECG can add: an ST change per bpm far beyond \
this patient's response to other heart-rate surges outside ST episodes, or ST at a given heart rate far \
beyond the patient's usual level at that heart rate (rate_response), argues for ischemia; a response within \
the patient's usual range fits a heart-rate related change but does not exclude demand ischemia. If that is \
all the evidence there is, give a probability near the middle rather than deciding by rule.
- Position and conduction. A body-position or axis shift changes the Q-, R- or S-wave amplitude or shape in \
the same lead and at the same time as the ST change, which is then often abrupt (step-like). Judge a QRS \
change relative to that lead's QRS size and check its timing; a gradual ST change with a stable QRS argues \
against position. A conduction change widens the QRS.
- Shape. Judge the shape of the change, not the absolute contour: compare the averaged beat during the \
episode with the same lead's beat before it. Above about 100 bpm the fixed J+80 point can lie on the T wave; \
use J+20 to J+60 and the image.
- Context. Candidate boundaries come from a threshold detector on deviations from a record-wide reference. \
Check (episode_context) whether the candidate is a fragment of a longer change, the return between two \
episodes, or a change of a lead whose usual level is far from zero, and judge the change itself (extremum \
minus the local level before it, and the absolute level).
- ST elevation can be ischemic (Long-Term ST Database classes; COVADIS criteria for vasospastic angina, \
Beltrame et al., Circ J 2016: transient "ST segment elevation >=0.1 mV"). A transient elevation that rises \
and falls smoothly with a stable QRS is positive evidence; a step-like shift at a heart-rate or posture \
change is not.
- Leads: a record may have only 2 or 3 leads with non-standard names; rules for contiguous 12-lead groups do \
not apply there.
- Use only the knowledge quoted here and the tool outputs; do not invent thresholds.
"""

GUIDANCE = ("v06", "v07a", "v07b", "v08")

_ROLES_V08 = {
    "judgement": """\
The case comes with a dossier: standard checks already run for this candidate, with evidence ids and images. \
Work as a careful Holter reader:
1. Read the dossier and the images. Decide whether the ST segment can be interpreted at all (signal quality, \
rhythm including rr_irregularity, QRS width). If it cannot, the label is indeterminate_data.
2. List the explanations that remain plausible: ischemia, heart-rate related change, body-position or axis \
shift, conduction change, noise or artifact, other non-ischemic change (slow drift, reference effect).
3. For the two most plausible, decide which measurement would tell them apart and obtain it with the tools \
(another lead, another time, a reference window at matched heart rate, the overlay against the lead's own \
earlier beat, rate_response, episode_context, a raw strip). Skip measurements that cannot change the balance.
4. Tools can be wrong: check a decisive number with a second view (another tool, lead, time or the image).
5. Submit a final assessment with submit_assessment: p_ischemic, your probability that this ST change is \
ischemic given everything you saw; the label, which must agree with it (ischemic if p_ischemic >= 0.5, \
otherwise the most likely non-ischemic cause); the findings for and against ischemia with their evidence ids; \
and a short rationale. Use indeterminate_data only when the signal does not allow ST interpretation; when the \
evidence conflicts, let the probability show it instead of abstaining.
Then reply with a one-paragraph summary.""",
    "verification": """\
A first reader assessed this candidate. Its assessment, with its probability and its findings for and \
against ischemia, is shown below as a ledger record, followed by the evidence it cites; it may be wrong. You \
are a second reader. Do not repeat its measurements; test its weakest point:
1. Name the finding the assessment rests on most, and the most plausible explanation it did not favour.
2. Obtain the measurement that best tells these two apart and that the first reader did not make (for \
example a heart-rate matched comparison outside ST episodes, rate_response, episode_context, the overlay \
against the same lead's beat before the change, the QRS of the same lead at the same time, a raw strip for \
rhythm or noise).
3. Submit a final assessment with your own p_ischemic, label and findings. You may keep or change the \
probability and the label; a changed label must cite evidence you obtained in this session. Say what your \
new evidence showed; neither doubt nor agreement is evidence by itself.
Then reply with a one-paragraph summary.""",
}

# How the model sees the recording, per modality (the second sentence of the shared context).
_SEE_NUMBERS = "You cannot see the ECG; you work only through tools that return measurements."
_SEE_IMAGES = ("You see the ECG only through tools, which return measurements and images drawn from the same "
               "data.")

_IMAGE_GUIDE = {
    "intro": """
Images. Some results come with an image drawn from the same signal and computation as the numbers; \
the numbers and the image are two views of one observation:""",
    "strip": """
- view_ecg: raw ECG on standard paper, 25 mm/s and 10 mm/mV (small square 40 ms and 0.1 mV, large square \
200 ms and 0.5 mV), one row per lead, beat labels of the recording system above the strip.""",
    "beats": """
- measure_window: the averaged beat of each lead at 50 mm/s and 20 mm/mV (small square 20 ms and 50 uV), \
with the baseline as a green dashed line; vertical lines mark the baseline point (green), the J point \
(orange) and the ST measurement point (red); the blue dotted line is the lead's usual ST level.""",
    "overlay": """
- compare_morphology: the two averaged beats of each lead overlaid (grey: reference time, black: the \
other time), same scale as measure_window.""",
    "trend": """
- st_trend, and the first message of a case: ST deviation of each lead over time, with lines at 0, \
+-50 and +-100 uV, and heart rate below.""",
    "outro": """
Use each view to check the other: for example, whether the J point and baseline point sit where they \
should, and whether a deviation is a real ST change or a baseline shift, noise or a QRS change. If they \
disagree, say so, cite both, and settle it with another tool, lead or time before relying on either.
""",
}
_IMAGE_ONLY_NOTE = """\
In this run the measured values are withheld: tool results give only times and leads, and you must read \
amplitudes, intervals and shapes from the images. Only the tools that return images, and \
find_reference_window, are available; checks below that name other tools have to be done on the images.
"""

ROLES = {
    "judgement": """\
Work through the candidate the way a careful Holter reader does:
1. Check whether the record allows ST interpretation at all (eligibility_check) and the rhythm around \
the candidate (rhythm_context).
2. Rule out non-ischemic explanations, each with its own evidence:
   - signal problems: noise, saturation, flat lines, too few clean beats (signal_quality, measure_window);
   - QRS morphology or axis change at the same time as the ST shift, pointing to a body-position or \
axis shift or a conduction change (morphology_steps; compare_morphology against a clean window from \
find_reference_window);
   - heart-rate dependence (episode_dynamics), keeping in mind that ischemia is also often preceded by a \
heart-rate rise.
3. Characterise the ST change itself: onset and offset (gradual or step-like), lead territory and \
reciprocal leads, ST slope, and the ACC/AHA criterion (check_criteria).
4. Tools can be wrong. Cross-check any finding that decides the label with a second tool, another \
lead or another time before relying on it.
5. Submit a final assessment with submit_assessment, citing the evidence ids (E1, E2, ...) that \
support it. If later evidence contradicts it, submit a new final assessment that cites that new \
evidence; a change of label without new evidence is rejected. If the evidence is missing or \
conflicting, use one of the indeterminate labels rather than guessing.

Keep rationales short: which findings from which evidence ids support the label. When the final \
assessment is recorded and nothing important is left unchecked, stop calling tools and reply with \
a one-paragraph summary.""",
    "first_impression": """\
This step records a first impression before any investigation. Using only the overview and the \
candidate summary you are given, call submit_assessment once with stage "initial". Cite the \
evidence ids of the overview and the candidate. Do not reply with anything else.""",
    "verification": """\
An earlier review assigned the label below to a candidate. Your job is to test that label \
independently, as a second reader would, using the tools. The assessment and the evidence it cites \
are shown as ledger records; they may be wrong.
1. Re-check the finding the label rests on with a different measurement: another tool, another lead, \
another time or another reference window. Repeating an identical call returns the earlier evidence id \
and adds nothing.
2. Test the strongest alternative label with its own evidence:
   - if the label is ischemic: signal problems, QRS or axis change, heart-rate dependence, rhythm that \
makes ST uninterpretable, or an ST change that fails the criteria. Ischemic episodes are rare in \
Holter recordings, so a positive label needs particular scrutiny;
   - if the label is non-ischemic: check that the claimed cause is actually present and coincides in \
time with the ST change, and that the ST change is not better explained by ischemia;
   - if the label is indeterminate: check whether the missing or conflicting evidence can be resolved.
3. Submit a final assessment. If your checks support the label, submit the same label again. Change \
the label only if evidence you obtained in this session supports a different one, and cite those \
evidence ids; doubt alone is not evidence. Then reply with a one-paragraph summary.""",
    "oneshot": """\
You are given the outputs of a fixed set of checks for one candidate. Decide the label from these \
outputs alone (no further tools are available) and call submit_assessment once with stage \
"final", citing the evidence ids you relied on.""",
    "recheck": """\
Review the assessment below once more. No further tools are available. Call submit_assessment once \
with stage "final": keep the label or change it, citing evidence ids that are already in the \
ledger.""",
}


_V07A_EDITS = {
    "judgement": (
        ("   - heart-rate dependence (episode_dynamics), keeping in mind that ischemia is also often preceded by a \
heart-rate rise.",
         "   - heart-rate dependence (episode_dynamics: ST at the same heart rate elsewhere in the record, ST vs \
heart rate), keeping in mind that ischemia is also often preceded by a heart-rate rise."),
        ("reciprocal leads, ST slope, and the ACC/AHA criterion (check_criteria).",
         "reciprocal leads, ST slope, and the ACC/AHA criterion (check_criteria), which is a criterion for a \
positive finding, not a requirement."),
    ),
    "verification": (
        ("makes ST uninterpretable, or an ST change that fails the criteria. Ischemic episodes are rare in \
Holter recordings, so a positive label needs particular scrutiny;",
         "makes ST uninterpretable;"),
        ("Change \
the label only if evidence you obtained in this session supports a different one, and cite those \
evidence ids; doubt alone is not evidence.",
         "Change \
the label only if evidence you obtained in this session shows the alternative directly (for example a QRS \
change of comparable size at the same time, ST at the same heart rate elsewhere as large as in the \
episode, noise, or a rhythm that makes ST uninterpretable), and cite those evidence ids. An unmet \
criterion for a positive finding, a feature typical of another class, or doubt is not such evidence; \
check your reason against the rules on how to use the knowledge before changing the label."),
    ),
}


_V07B_EDITS = {
    "judgement": _V07A_EDITS["judgement"] + (
        ("If the evidence is missing or \
conflicting, use one of the indeterminate labels rather than guessing.",
         "Label a change ischemic only \
if the ST change itself shows evidence of ischemia, not only because other causes were not found. If the \
evidence is missing or conflicting, use one of the indeterminate labels rather than guessing."),
    ),
    "verification": (
        ("makes ST uninterpretable, or an ST change that fails the criteria. Ischemic episodes are rare in \
Holter recordings, so a positive label needs particular scrutiny;",
         "makes ST uninterpretable, a slow drift; and check that the ST change itself shows evidence of \
ischemia (shape, time course, relation to heart rate), not only that other causes were not found;"),
        _V07A_EDITS["verification"][1],
    ),
}


def _edited_roles(edits: dict) -> dict[str, str]:
    out = dict(ROLES)
    for role, pairs in edits.items():
        for old, new in pairs:
            assert old in out[role], (role, old[:60])
            out[role] = out[role].replace(old, new)
    return out


ROLES_BY_GUIDANCE = {"v06": ROLES, "v07a": _edited_roles(_V07A_EDITS), "v07b": _edited_roles(_V07B_EDITS),
                     "v08": {**ROLES, **_ROLES_V08}}
ROLES_V07 = ROLES_BY_GUIDANCE["v07b"]  # the default


def context(modality: str = "numbers", guidance: str = "v07b") -> str:
    """The shared context; its second sentence says how the model sees the recording. Guidance v07 adds
    the rules on how to use the reference knowledge."""
    assert _SEE_NUMBERS in _CONTEXT and guidance in GUIDANCE
    text = _CONTEXT
    if guidance != "v06":  # the records are not all 12-lead (LTSTDB: 2-3 leads)
        text = text.replace("24-hour 12-lead Holter recordings", "24-hour Holter recordings (2 to 12 leads)")
        text += {"v07a": _READING_RULES_A, "v07b": _READING_RULES_B, "v08": _READING_RULES_C}[guidance]
    return text if modality == "numbers" else text.replace(_SEE_NUMBERS, _SEE_IMAGES)


# How to read the toolkit 0.6 additions (sessions with analysis tools; tool sets "ecg" and "full").
TOOL_GUIDE = {
    "ecg": ("\nFurther checks available in this run:\n"
            "- shape_changes compares the QRS shape and the ST-T shape before onset and at the extremum. A QRS "
            "change that is large relative to the ST-T change points to an axis or body-position shift; ischaemic "
            "changes show mainly in the ST-T shape. It also gives the ST-segment level, rise and curvature per lead.\n"
            "- episode_dynamics also gives the ST/HR slope from onset to the extremum and back. In the Long-Term "
            "ST Database, ischaemic episodes had steeper ST/HR slopes than heart-rate related ones (Faganeli "
            "Pucer 2012); a heart-rate rise alone does not make an episode heart-rate related."),
    "est": ("\n- ischemia_estimate gives a statistical estimate of the probability that the candidate is "
            "ischemic, from a model fitted on other records of the Long-Term ST Database. It summarises how "
            "candidates with similar tool outputs were labelled there. Weigh it together with the ECG findings; "
            "if they disagree, say which you follow and why."),
    "full": ("\n- patient_context gives the patient's clinical information when the record has it (diagnoses, "
             "history, previous tests, medication). It changes how likely ischaemia is for this patient; it does "
             "not show whether this particular episode is ischaemic, which the ECG evidence must decide."),
}


def system(role: str, modality: str = "numbers", image_kinds: tuple[str, ...] = ("strip", "beats", "overlay", "trend"),
           toolset: str = "base", guidance: str = "v07b") -> str:
    """System prompt of a session: shared context, image guide (if images are shown), role, tool guide
    (tool sets beyond "base", sessions with tools), labels."""
    roles = ROLES_BY_GUIDANCE[guidance]
    guide = ""
    if modality != "numbers":
        guide = (_IMAGE_GUIDE["intro"] + "".join(_IMAGE_GUIDE[k] for k in ("strip", "beats", "overlay", "trend")
                                                  if k in image_kinds) + _IMAGE_GUIDE["outro"])
        if modality == "images":
            guide += _IMAGE_ONLY_NOTE
    tools = ""
    base = toolset.split("+")[0]
    if role in ("judgement", "verification") and base != "base":
        tools = TOOL_GUIDE["ecg"] + (TOOL_GUIDE["full"] if base == "full" else "") + "\n"
    if role in ("judgement", "verification") and toolset.endswith("+est"):
        tools += TOOL_GUIDE["est"] + "\n"
    labels = _LABEL_TEXT_V08 if guidance == "v08" else _LABEL_TEXT
    return context(modality, guidance) + guide + "\n" + roles[role] + "\n" + tools + "\n" + labels


def role_of(system_prompt: str) -> str | None:
    """Which session a system prompt belongs to."""
    return next((r for roles in ROLES_BY_GUIDANCE.values() for r, text in roles.items() if text in system_prompt),
                None)


# The prompts of a numbers-only run.
INVESTIGATOR_SYSTEM = system("judgement")
INITIAL_SYSTEM = system("first_impression")
VERIFIER_SYSTEM = system("verification")
ONESHOT_SYSTEM = system("oneshot")
RECHECK_SYSTEM = system("recheck")


SUBMIT_TOOL = {
    "name": "submit_assessment",
    "description": "Record an initial or final assessment for the candidate under review. A later final "
                   "assessment for the same candidate is stored as a revision next to the earlier one.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "stage": {"type": "string", "enum": ["initial", "final"]},
            "label": {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "number", "description": "0 to 1"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["candidate_id", "stage", "label", "confidence", "evidence_ids", "rationale"],
        "additionalProperties": False,
    },
}


_FINDINGS = {"type": "array", "items": {"type": "object", "properties": {
    "evidence_id": {"type": "string"}, "finding": {"type": "string"}},
    "required": ["evidence_id", "finding"], "additionalProperties": False}}
GRADED_LABELS = tuple(lab for lab in LABELS if lab != "indeterminate_ambiguous")
SUBMIT_TOOL_GRADED = {
    "name": "submit_assessment",
    "description": "Record an assessment of the candidate under review. p_ischemic is your probability that the ST "
                   "change is ischemic. label is the most likely explanation and must agree with p_ischemic: "
                   "'ischemic' when p_ischemic >= 0.5, otherwise the most likely non-ischemic cause; "
                   "'indeterminate_data' only when the signal does not allow ST interpretation. Cite the evidence ids "
                   "of each finding. A later final assessment is stored as a revision next to the earlier one.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "stage": {"type": "string", "enum": ["initial", "final"]},
            "p_ischemic": {"type": "number", "description": "0 to 1"},
            "label": {"type": "string", "enum": list(GRADED_LABELS)},
            "findings_for_ischemia": _FINDINGS,
            "findings_against_ischemia": _FINDINGS,
            "rationale": {"type": "string"},
        },
        "required": ["candidate_id", "stage", "p_ischemic", "label", "findings_for_ischemia",
                     "findings_against_ischemia", "rationale"],
        "additionalProperties": False,
    },
}


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def case_header(record_id: str, overview: dict, overview_eid: str, candidate: dict, candidate_eid: str) -> str:
    return (f"Record {record_id}. Overview (evidence {overview_eid}):\n{_j(overview)}\n\n"
            f"Candidate {candidate['id']} (evidence {candidate_eid}):\n{_j(candidate)}")


def ledger_entry(assessment, source: str) -> str:
    """A prior assessment shown as a structured ledger record, not as the model's own words.

    Errors presented in an external role are corrected far more often than the same
    content inside the model's own reasoning (Chen et al. 2026, arXiv:2606.05976), and
    structured tool-like records cause less deference than authoritative prose
    (Roy et al. 2026, arXiv:2608.29800).
    """
    rec = {"source": source, "candidate_id": assessment.candidate_id, "label": assessment.label,
           "confidence": assessment.confidence, "evidence_ids": assessment.evidence_ids,
           "rationale": assessment.rationale}
    if getattr(assessment, "p_ischemic", None) is not None:  # graded output (harness 0.8)
        rec = {k: v for k, v in rec.items() if k != "confidence"}
        rec["p_ischemic"] = assessment.p_ischemic
        rec["findings"] = getattr(assessment, "evidence_table", None)
    return _j({"ledger_record": rec})


def investigate_task(header: str, candidate_id: str, prior: str | None = None) -> str:
    text = header
    if prior:
        text += (f"\n\nA label for this candidate from another system is recorded in the ledger below. It may "
                 f"be wrong; check it against the evidence.\n{prior}")
    return text + f"\n\nInvestigate {candidate_id} and submit a final assessment."


def verify_task(header: str, candidate_id: str, ledger: str, cited: list[tuple[str, str, dict, dict]]) -> str:
    lines = [f"{eid} {tool} {_j(args)}: {_j(result)}" for eid, tool, args, result in cited]
    text = f"{header}\n\nAssessment under review:\n{ledger}"
    if lines:
        text += "\n\nEvidence it cites:\n" + "\n".join(lines)
    return text + f"\n\nTest this label for {candidate_id}, then submit a final assessment."


def dossier_block(evidence: list[tuple[str, str, dict, dict]]) -> str:
    """Standard checks run before the judgement session (harness 0.8), as ledger lines."""
    lines = [f"{eid} {tool} {_j(args)}: {_j(result)}" for eid, tool, args, result in evidence]
    return ("\n\nDossier: standard checks already run for this candidate (images of measure_window and "
            "compare_morphology are attached after the trend image, in this order):\n" + "\n".join(lines))


def oneshot_task(header: str, candidate_id: str, evidence: list[tuple[str, str, dict]]) -> str:
    lines = [f"{eid} {tool}: {_j(result)}" for eid, tool, result in evidence]
    return f"{header}\n\nCheck outputs:\n" + "\n".join(lines) + f"\n\nSubmit a final assessment for {candidate_id}."


def recheck_task(header: str, candidate_id: str, ledger: str) -> str:
    return f"{header}\n\nAssessment to review:\n{ledger}\n\nSubmit a final assessment for {candidate_id}."


def initial_task(header: str, candidate_id: str) -> str:
    return f"{header}\n\nRecord your initial impression of {candidate_id}."


NUDGE_FINAL = ("No final assessment has been recorded for this candidate in this session. "
               "Call submit_assessment with stage 'final', citing evidence ids.")
LAST_TURN_NOTE = ("This is your last turn in this session. Submit your final assessment now with "
                  "submit_assessment, citing the evidence ids you have.")
STALL_NOTE = ("The last calls produced no new evidence (repeated calls return the earlier evidence id). "
              "Use a different check or submit your final assessment.")
