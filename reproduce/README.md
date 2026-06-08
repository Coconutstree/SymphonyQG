# Reproduction
Here, we provide the code for reproducing main experimental results reported in the paper. Please make sure datasets have been downloaded and preprocessed.

This branch uses a global fixed center `c_0` for **4-bit ExRaBitQ** quantization. Existing `.index`
files built by the original 1-bit RaBitQ or per-node-center layout are incompatible and must be
rebuilt before running query benchmarks.

1. Install the Python binding from the repository root:

```bash
cd python
pip install -r requirements.txt
sh build.sh
cd ..
```

2. Download and preprocess datasets according to `./data/README.md`.

3. Edit `./reproduce/settings.py` to choose datasets, degrees, construction
   `EF`, and iteration counts.

4. Remove or overwrite old indices from the previous RaBitQ or per-node-center layout, 
   then rebuild with **4-bit ExRaBitQ** quantization:

```bash
python ./reproduce/indexing.py
```

This saves fixed-`c_0` indices with 4-bit ExRaBitQ codes as:

```text
./data/<DATASET>/symphonyqg_<DEGREE>.index
```

Note: The new indices use 4-bit quantization per dimension, providing better accuracy than
the original 1-bit RaBitQ quantization.

5. Run the QPS-recall benchmark:

```bash
python ./reproduce/run.py
```

Results are saved as CSV files under `./results/<DATASET>/symphonyqg/`.

6. Optional: compute average distance ratio using the generated search results:

```bash
python ./reproduce/ratio.py
```

## 4-bit ExRaBitQ Performance Characteristics

The 4-bit ExRaBitQ quantization provides the following improvements over the original 1-bit RaBitQ:

- **Better Recall**: 4-bit quantization captures more information about vector direction,
  resulting in improved recall at the same query latency
- **Higher Precision**: More precise distance approximation with multi-bit codes
- **Memory Trade-off**: Uses 2x more memory for quantization codes (4 bits/dim vs 1 bit/dim),
  but still maintains efficient global-center layout

Expected benchmark results show that 4-bit ExRaBitQ achieves:
- ~95-99% recall@10 with moderate query latency
- Comparable or better QPS-recall curves compared to the original RaBitQ
- Significant memory savings vs. full-precision storage
