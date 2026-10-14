# Paper implementation card: <paper title>

- **Paper / OpenReview:** <URL>
- **Primary module:** <module>
- **Secondary modules:** <none or modules>
- **Owner / reviewer:** <GitHub handles>
- **Status:** proposed | reading | baseline-ready | implementing | tested | reproduced | partial-reproduction | blocked
- **Upstream code / licence:** <URL, licence, or not found>
- **Backbone IDs:** <registry IDs>
- **Dataset IDs and split:** <registry IDs>
- **Hardware assumptions:** <device/ops/memory>

## Research summary

### Problem
<What resource/quality bottleneck is being addressed?>

### Method
<Explain the algorithm without copying large portions of the paper.>

### Expected behaviour
<Expected quality/runtime/memory trade-off from the paper. Distinguish reported from locally measured values.>

## Implementation

- **Input contract:** <shapes/dtypes/state>
- **Output contract:** <shapes/dtypes/metadata>
- **Configuration:** <config path and important flags>
- **Reproduction command:** <one exact command>
- **Tests:** <test command>

## Results

| Source | Quality | Latency | Throughput | Peak memory | Energy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Paper-reported | | | | | | Mark units and exact setup |
| Local baseline | | | | | | |
| Local method | | | | | | |

## Differences and limitations

- <missing assets, approximations, unsupported devices/shapes, or divergence from the paper>

## Reproduction checklist

- [ ] Exact paper link and version verified
- [ ] Base model and dataset revisions pinned
- [ ] Baseline runs on the same setup
- [ ] Tests and seed handling added
- [ ] Resource and quality metrics reported
- [ ] Licence/data terms checked
