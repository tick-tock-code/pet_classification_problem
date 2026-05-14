# Model Improvement Notes

Current status: the custom PyTorch CNN is learning, and the 30-epoch run should finish before changing settings. Use `best_model.pt` as the meaningful checkpoint because it tracks the lowest validation loss.

## Recommended Priority

1. **Finish the current run**
   - Do not change architecture, optimizer, or augmentation mid-run.
   - Record the best validation loss and validation accuracy from `best_model.pt`.

2. **Improve the validation split**
   - Add fixed random seeds for reproducible comparisons.
   - Prefer a stratified train/validation split so all 37 classes are represented consistently.
   - This improves confidence in whether a change genuinely helped.

3. **Try transfer learning**
   - Use a pretrained model such as `torchvision.models.efficientnet_b0`, `resnet18`, or `mobilenet_v3_small`.
   - Replace the classifier head with a 37-class output layer.
   - This is likely the largest performance jump because the pretrained model already knows general image features.

4. **Tune augmentation**
   - Keep augmentation realistic for pet photos.
   - Good defaults: horizontal flip, mild rotation, mild `ColorJitter`, and possibly mild random resized crop.
   - Avoid vertical flips and large rotations unless the real test images may contain those cases.
   - Consider MixUp or CutMix later, but they require label/training-loop changes.

5. **Tune hyperparameters**
   - Keep the search small and controlled.
   - Try learning rates: `0.001`, `0.0003`, `0.0001`.
   - Try weight decay: `0.001`, `0.0005`, `0.0001`.
   - Try batch size `64` only if GPU memory allows.
   - Compare runs using the same split and same epoch budget.

6. **Add training controls**
   - Add early stopping after roughly 7-10 epochs without validation-loss improvement.
   - Log the current learning rate each epoch to see when `ReduceLROnPlateau` activates.
   - Keep saving both `autosave_model.pt` and `best_model.pt`.

## Architecture Direction

The current custom CNN is useful for learning and is now a reasonable baseline:

```text
3 -> 32 -> 32 -> pool
32 -> 64 -> 64 -> pool
64 -> 128 -> 128 -> pool
128 -> 256 -> 256 -> pool
AdaptiveAvgPool2d
Linear(256, 37)
```

Further custom architecture work is lower priority than transfer learning. If the goal is best accuracy, compare the current CNN against a pretrained EfficientNet or MobileNet before designing a more advanced CNN from scratch.

## Practical Next Experiment

After the current run finishes:

1. Save the final training log.
2. Record the best validation metrics.
3. Implement a transfer-learning version using `efficientnet_b0`.
4. Keep the same data split, augmentation, metrics, checkpointing, and epoch budget.
5. Compare the custom CNN and transfer-learning model on validation loss and accuracy.
