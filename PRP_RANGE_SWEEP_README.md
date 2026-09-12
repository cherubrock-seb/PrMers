# PRP range sweep

Purpose: map AEVUM PRP plan throughput from 100M to 220M on the real GPUs.

Critical methodology change versus Pass 3: the reference is the **true AEVUM AUTO plan on the same source**. Pass 3's 100M plan-search case used a forced Type4 reference (`4:512:8:512:202`), so its large ratios must not be interpreted as gains over the current production AUTO policy until this sweep confirms them.

Default prime exponents:
100000007, 120000007, 130000001, 140000041, 150000001, 160000003, 180000017, 197000003, 200000033, 210000017, 220000013.

For every exponent:
- run true AUTO;
- word-exact single-step preflight for each explicit plan;
- alternating AUTO/candidate A/B;
- warm-up pair discarded;
- median of >=3 timed pairs;
- confirm the best positive candidate with >=5 timed pairs;
- mark >=1.05x confirmed as a production candidate;
- collect a short kernel-event profile of AUTO and the confirmed best plan.

No full PRP/Gerbicz test is run in this sweep. That test is intentionally excluded because the existing p=786433 Gerbicz validation path is known to restart independently of these plan candidates. This sweep is for exact engine/plan selection only.

Override exponent list with `AEVUM_SWEEP_EXPONENTS=...`.
