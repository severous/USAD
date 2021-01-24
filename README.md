# USAD
Non-offical implementation
### dataset
- SWat dataset

Train: [SWaT_Dataset_Normal_v1.csv](https://drive.google.com/open?id=1rVJ5ry5GG-ZZi5yI4x9lICB8VhErXwCw)

Test:  [SWaT_Dataset_Attack_v0.csv](https://drive.google.com/open?id=1iDYc0OEmidN712fquOBRFjln90SbpaE7)
- MSI dataset

The MSI dataset is in data folder
### preprocessing
`python data_preprocess.py <dataset>`
where <dataset> is one of MSL or SWaT.
### Train and Test
run 'train_usad_swat.py'
### Notice
This implementation do not use the down-sampling rate mentioned in the article USAD
