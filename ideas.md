# Remarks to improve the models:

## Give more power to the rarest classes

* giving different weights to the classes in the criterion.
* weights inversly proportional to the frequency

example:
```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# y_train: numpy array of integer class labels (0..C-1), shape (N,)
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
# weights is length C, e.g. [0.2, 5.0, 3.2]
print("class weights:", weights)
```

```python
import torch
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
# during training: loss = criterion(logits, targets)
```

## Limit gradient explosion

using clipping between backward and step optimizer in function train_one_epoch()

```python
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

## Using label smoothing

give 0.9 for a correct class and 0.05 for each incorrect class while training the model.
Then the model explore more.

```python
criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
```

## Looking at the autocorrelation plot to find the correct window size.