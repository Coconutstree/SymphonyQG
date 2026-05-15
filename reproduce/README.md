# Reproduction
Here, we provide the code for reproducing main experimental results reported in the paper. Please make sure datasets have been downloaded and preprocessed.

This branch uses a global fixed center `c_0` for RaBitQ. Existing `.index`
files built by the original per-node-center layout are incompatible and must be
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

4. Remove or overwrite old indices from the previous layout, then rebuild:

```bash
python ./reproduce/indexing.py
```

This saves fixed-`c_0` indices as:

```text
./data/<DATASET>/symphonyqg_<DEGREE>.index
```

5. Run the QPS-recall benchmark:

```bash
python ./reproduce/run.py
```

Results are saved as CSV files under `./results/<DATASET>/symphonyqg/`.

6. Optional: compute average distance ratio using the generated search results:

```bash
python ./reproduce/ratio.py
```
