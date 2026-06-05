# Notebook Inspection Report

## Task Prompt

Inspect all notebooks in the experiment directory with `jq`, list all architectures and training history, then suggest new names for all notebooks for easy lookup.

Also captured during the task:
- Baseline CNN architecture:
  - Conv2D 32 3x3 ReLU
  - MaxPool 2x2
  - Conv2D 64 3x3 ReLU
  - MaxPool 2x2
  - Flatten
  - Dense 64
  - Dropout 0.5
  - Output Softmax
- Wolfmonheim architecture:
  - BatchNorm
  - Conv2D 64 3x3 ReLU `same` padding
  - MaxPool 2x2
  - Conv2D 128 3x3 ReLU `same` padding
  - MaxPool 2x2
  - Conv2D 256 3x3 ReLU `same` padding
  - MaxPool 2x2
  - Conv2D 256 3x3 ReLU `same` padding
  - MaxPool 2x2
  - Flatten
  - Dense 256 ReLU
  - Dropout 0.5
  - Dense Softmax
- Requested notebook cell lookups:
  - architecture definition: `jq '.cells[6].source'`
  - architecture summary: `jq '.cells[13].outputs[0].text'`
  - training epochs: `jq '.cells[15].outputs[0].text'`
  - noisy dataset evaluation accuracy: `jq '.cells[-1].outputs[2].text[0]'`

## Commands Used

```bash
for f in ./*.ipynb; do
  printf '=== %s ===\n' "$(basename "$f")"
  jq -r '
    "cells=" + ((.cells|length)|tostring),
    "arch=" + (((.cells[6].source // []) | join(""))),
    "summary=" + (((.cells[13].outputs[0].text // []) | join(""))),
    "epochs=" + (((.cells[15].outputs[0].text // []) | join(""))),
    "eval=" + (((.cells[-1].outputs[2].text[0] // "")))
  ' "$f"
  printf '\n'
done
```

```bash
jq -r '
  "cells=" + ((.cells|length)|tostring),
  "arch=" + (((.cells[6].source // []) | join(""))),
  "summary=" + (((.cells[13].outputs[0].text // []) | join(""))),
  "epochs=" + (((.cells[15].outputs[0].text // []) | join(""))),
  "eval=" + (((.cells[-1].outputs[2].text[0] // "")))
' ./opus_baseline.ipynb
```

```bash
for f in ./opus_baseline-128dense.ipynb ./opus_baseline-2wolfconv.ipynb ./opus_baseline-batchnorm.ipynb; do
  echo "=== $(basename \"$f\") ==="
  jq -r '
    "cells=" + ((.cells|length)|tostring),
    "arch=" + (((.cells[6].source // []) | join(""))),
    "summary=" + (((.cells[13].outputs[0].text // []) | join(""))),
    "epochs=" + (((.cells[15].outputs[0].text // []) | join(""))),
    "eval=" + (((.cells[-1].outputs[2].text[0] // "")))
  ' "$f"
  echo
done
```

```bash
jq -r '(.cells[15].outputs[0].text | length), (.cells[15].outputs[0].text[0]), (.cells[15].outputs[0].text[-1])' ./opus_baseline-batchnorm.ipynb
```

## Notebook Inventory

| Current notebook | Actual architecture | Training history | Noisy test accuracy | Suggested rename |
|---|---|---|---:|---|
| `opus_baseline.ipynb` | C32-MP-C64-MP-Flat-D64-Do0.5-SM | 25 epochs, best epoch 19, early stop | 0.9497 | `opus_baseline-c32c64-d64-noisy950.ipynb` |
| `opus_baseline-128dense.ipynb` | C32-MP-C64-MP-Flat-D128-Do0.5-SM | 22 epochs, best epoch 16, early stop | 0.9861 | `opus_baseline-c32c64-d128-noisy986.ipynb` |
| `opus_baseline-2wolfconv.ipynb` | C64-MP-C128-MP-Flat-D128-Do0.5-SM | 24 epochs, best epoch 18, early stop | 0.9606 | `opus_baseline-c64c128-d128-noisy961.ipynb` |
| `opus_baseline-batchnorm.ipynb` | Same as baseline-128dense; BN is commented out | Training output captured only the start of epoch 1 | 0.1744 | `opus_baseline-c32c64-d128-noisy174.ipynb` |
| `opus_wolfmonheim.ipynb` | BN-C64-same-MP-C128-same-MP-C256-same-MP-C256-same-MP-Flat-D256-Do0.5-SM | 13 epochs, best epoch 7, early stop | 0.2319 | `opus_wolf-bn-c64c128c256c256-d256-same-noisy232.ipynb` |
| `opus_wolfmonheim-halfconv.ipynb` | BN-C32-MP-C64-MP-C128-MP-C128-MP-Flat-D256-Do0.5-SM | 8 epochs, best epoch 2, early stop | 0.5717 | `opus_wolf-bn-c32c64c128c128-d256-noisy572.ipynb` |
| `opus_wolfmonheim-halfconvno4.ipynb` | BN-C32-MP-C64-MP-C128-MP-Flat-D256-Do0.5-SM | 16 epochs, best epoch 10, early stop | 0.1642 | `opus_wolf-bn-c32c64c128-d256-noisy164.ipynb` |
| `opus_wolfmonheim-no3convbatchnorm.ipynb` | C64-MP-C128-MP-Flat-D256-Do0.5-SM; BN commented out | 24 epochs, best epoch 18, early stop | 0.9650 | `opus_wolf-c64c128-d256-noisy965.ipynb` |
| `opus_wolfmonheim-no4conv.ipynb` | BN-C64-same-MP-C128-same-MP-C256-same-MP-Flat-D256-Do0.5-SM | 8 epochs, best epoch 2, early stop | 0.6789 | `opus_wolf-bn-c64c128c256-d256-same-noisy679.ipynb` |
| `opus_wolfmonheim-no4conv128dense.ipynb` | BN-C64-MP-C128-MP-C256-MP-Flat-D128-Do0.5-SM | 8 epochs, best epoch 2, early stop | 0.6822 | `opus_wolf-bn-c64c128c256-d128-noisy682.ipynb` |
| `opus_wolfmonheim-nobatchnorm.ipynb` | C64-same-MP-C128-same-MP-C256-same-MP-C256-same-MP-Flat-D256-Do0.5-SM | 8 epochs, best epoch 2, early stop | 0.3092 | `opus_wolf-c64c128c256c256-d256-same-noisy309.ipynb` |

## Notes On The Findings

- The baseline family benefits most from the denser classifier head. The 128-unit dense layer consistently outperformed the 64-unit baseline on noisy data.
- The larger Wolfmonheim stacks have much larger flattened representations and very large dense layers, which likely increases overfitting even when validation accuracy looks near-perfect.
- The best noisy-set results came from the smaller baseline-style models, especially `opus_baseline-128dense.ipynb` and `opus_baseline-2wolfconv.ipynb`.
- The notebook named `opus_baseline-batchnorm.ipynb` is misleading because the BatchNorm line is commented out.
- The notebook named `opus_wolfmonheim-no3convbatchnorm.ipynb` is also misleading for the same reason: BatchNorm is commented out and the model only uses two conv blocks.

## Summary

The likely cause of the accuracy improvement is a better balance between feature-extractor depth and classifier capacity, not simply more convolutional layers. In this set of experiments, the smaller baseline models with a stronger dense head generalized better to the noisy dataset than the very large Wolfmonheim variants. The biggest risk in the Wolfmonheim models is the large flattened tensor feeding a large dense layer, which creates a much larger parameter count and makes overfitting more likely.

If the goal is easy lookup, the most useful filename scheme is:
- family prefix: `opus_baseline` or `opus_wolf`
- conv stack shorthand: `c32c64`, `c64c128c256`, and so on
- dense size: `d64`, `d128`, `d256`
- padding flag where relevant: `same`
- noisy accuracy rounded to three digits: `noisy950`, `noisy682`, and so on
