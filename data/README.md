# Data (not committed)

The dataset is too large for GitHub and is **not** included in this repo.

## Download

**UCI ML Repository - Online Retail II** (official source, ~44 MB):

```bash
curl -L -o online_retail_ii.zip \
  "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip online_retail_ii.zip          # produces "online_retail_II.xlsx"
mv "online_retail_II.xlsx" data/    # place it here
rm online_retail_ii.zip
```

Expected path: `data/online_retail_II.xlsx`

Alternative: Kaggle - search `online retail ii` (same UCI data re-uploaded).

## License / citation

Dataset from the UCI Machine Learning Repository. Please cite the source when
redistributing analyses:

> Daqing Li, Sai Raj Kairam, and Dr. Sergey Butakov. (2020). Online Retail II
> [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW2X

Then run `./run_all.sh` to regenerate every table and figure from scratch.
