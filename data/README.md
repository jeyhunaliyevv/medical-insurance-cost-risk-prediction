# Dataset

`insurance.csv` is the public *Medical Cost Personal Dataset* (1,338 rows: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).

- Upstream: <https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv>
- Also available on Kaggle as "Medical Cost Personal Datasets".

**The file is not stored in this repository.** `src/data.py` (and the notebook) download it on first use and verify it against a pinned SHA-256 checksum. If the download fails, save the file manually as `data/insurance.csv`.

**Why it is not redistributed:** a clear redistribution license was not verified for this project, so only the download code is published here.

**Nature of the data:** it is a teaching dataset; one R package's documentation describes it as simulated from US Census demographics. Results should therefore not be read as statements about real insurance portfolios. No employer, client or private data is used anywhere in this project.

**Deduplication:** The original CSV has 1,338 rows. The pipeline validates it and removes one exact duplicate in memory **before** creating the 1,337-row modelling dataset. The original downloaded CSV is kept unchanged.

