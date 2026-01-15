# RML2016.10a Dataset Processing Script

This project contains a script to process the RML2016.10a dataset according to the method used in the `amc.py` file's `process_data()` function and convert it to parquet format.

## Files Created

- `process_rml_to_parquet_fixed.py`: Main processing script that:
  - Loads the RML2016.10a_dict.pkl file
  - Splits the data using the same method as in amc.py (90% train, 10% test)
  - Applies the same transformations (iq2ap and normalize)
  - Saves the data in parquet format with the required columns

- `RML2016_10a_train.parquet`: Training dataset with 360 samples
- `RML2016_10a_test.parquet`: Test dataset with 40 samples

## Parquet File Structure

Both parquet files contain the following columns:

- `dataset_name`: String indicating the dataset ("RML2016.10a_train" or "RML2016.10a_test")
- `infer_class`: Integer representing the modulation class label (0-3 for the 4 modulation types)
- `snr`: Signal-to-noise ratio (integer for test set, 0 for training set as placeholder)
- `iq`: List of IQ data pairs in the form [['i','q'],...] representing amplitude-phase transformed values

## Usage

To process your own RML2016.10a_dict.pkl file:

```bash
python process_rml_to_parquet_fixed.py
```

Note: Make sure the RML2016.10a_dict.pkl file is located in the `Data/` directory.

## Requirements

- pandas
- pyarrow
- numpy
- scikit-learn

## Data Transformation

The script applies the same preprocessing as in amc.py:
1. Splits each modulation-SNR combination using `train_test_split` with test_size=0.1
2. Converts IQ data to Amplitude-Phase representation using `iq2ap`
3. Normalizes the data using min-max normalization
4. Formats the data for parquet export with the specified column structure