
Images. Some results come with an image drawn from the same signal and computation as the numbers; the numbers and the image are two views of one observation:
- view_ecg: raw ECG on standard paper, 25 mm/s and 10 mm/mV (small square 40 ms and 0.1 mV, large square 200 ms and 0.5 mV), one row per lead, beat labels of the recording system above the strip.
- measure_window: the averaged beat of each lead at 50 mm/s and 20 mm/mV (small square 20 ms and 50 uV), with the baseline as a green dashed line; vertical lines mark the baseline point (green), the J point (orange) and the ST measurement point (red); the blue dotted line is the lead's usual ST level.
- compare_morphology: the two averaged beats of each lead overlaid (grey: reference time, black: the other time), same scale as measure_window.
- st_trend, and the first message of a case: ST deviation of each lead over time, with lines at 0, +-50 and +-100 uV, and heart rate below.
