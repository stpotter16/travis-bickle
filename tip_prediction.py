import contextlib

import coiled
import dask.dataframe as dd
from dask.distributed import Client
from xgboost.dask import DaskDMatrix, predict, train

ENVIRONMENT_NAME = 'travis-bickle-env'
REQUIREMENTS = ["dask[complete]", "xgboost", "s3fs"]


@contextlib.contextmanager
def client_context():
    try:
        cluster = coiled.Cluster(
            name='travis-bickle',
            configuration='sam-potter/travis-bickle-config'
        )
        client = Client(cluster)
        yield client
    finally:
        cluster_name = list(coiled.list_clusters().keys())[-1]
        coiled.delete_cluster(name=cluster_name)
        client.close()


def compute_rmse(prediction, test_labels):
    rmse = ((prediction - test_labels) ** 2).mean() ** 0.5
    return rmse.compute()


def featurize_data(df):
    df['tip_frac'] = df['tip_amount'] / df['total_amount']
    df['pickup_hour'] = dd.to_datetime(df['tpep_pickup_datetime']).dt.hour


def filter_data(df):
    use_cols = [
        'passenger_count',
        'trip_distance',
        'fare_amount',
        'tip_amount',
        'total_amount',
        'tip_frac',
        'pickup_hour'
    ]
    filtered_df = df[use_cols]
    mask = (df['tip_amount'] > 0) & (df['total_amount'] > 0) & (df['total_amount'] < 10000)
    filtered_df = filtered_df[mask]
    return filtered_df


def load_raw_data():
    df = dd.read_csv(
        "s3://nyc-tlc/trip data/yellow_tripdata_2019-*.csv",
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        dtype={
            "payment_type": "UInt8",
            "VendorID": "UInt8",
            "passenger_count": "UInt8",
            "RatecodeID": "UInt8",
            "store_and_fwd_flag": "category",
            "PULocationID": "UInt16",
            "DOLocationID": "UInt16",
        },
        storage_options={"anon": True},
        blocksize="16 MiB",
    ).persist()
    return df


def make_dmatrix(client, X, y):
    return DaskDMatrix(client, X, y)


def make_software_env():
    coiled.create_software_environment(
        name=ENVIRONMENT_NAME,
        pip=REQUIREMENTS,
    )


def split_data(df):
    train, test = df.random_split([0.8, 0.2])
    train_labels = train.tip_frac
    test_labels = test.tip_frac
    
    del train['tip_frac']
    del test['tip_frac']

    return train, train_labels, test, test_labels


def train_tree(client, dmatrix):
    params = {
        'verbosity': 2,
        'tree_method': 'hist',
        'objective': 'reg:squarederror'
    }
    return train(
        client,
        params,
        dmatrix,
        evals=[(dmatrix, 'train')],
    )


def main():
    make_software_env()
    with client_context() as client:
        print(client.dashboard_link)
        print('Loading raw data')
        df = load_raw_data()
        print('Featurizing data')
        featurize_data(df)
        print('Filtering data')
        filtered_df = filter_data(df)
        print('Splitting into train/test sets')
        train, train_labels, test, test_labels = split_data(filtered_df)
        print(train.dtypes)
        print('Forming dmatrix')
        d_train = make_dmatrix(client, train, train_labels)
        print('Training classifier')
        training_output = train_tree(client, d_train)
        print('Predicting on holdout data')
        prediction = predict(client, training_output, test)
        print('Computing RMSE')
        rmse = compute_rmse(prediction, test_labels)
        print(f'RMSE of model against hold out data: {rmse}')


if __name__ == '__main__':
    main()
