# Restored test outputs — pending final training

`evaluate.py` has been verified end-to-end on the real 400-image
`Test_NoisyLR` set (all 400 processed, correct 256x256 [0,1] outputs,
timing reported — see README). The actual restored outputs are not
included here because they require `models/final_model.pth`, which does
not yet exist (no real training has occurred — see
`models/PLACEHOLDER.md`). Generate them with:

    python evaluate.py --input_dir /path/to/Test_NoisyLR/NoisyLR --output_dir outputs/restored_test
