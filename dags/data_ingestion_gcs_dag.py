import os
import logging
from functools import reduce
from datetime import datetime
from airflow.utils.dates import days_ago
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateExternalTableOperator, BigQueryDeleteTableOperator
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your DAG parameters
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "energy-consumption-419814")
BUCKET_NAME = Variable.get("GCS_BUCKET_NAME", default_var="dam_resorvoir_data_lake")

def download_files(local_data_dir):
    """Download CSV files from GitHub to the specified local directory.

    Args:
        local_data_dir (str): The local directory to download the files to.
    """
    import requests
    try:
        github_repo_url = "https://raw.githubusercontent.com/jadeadams517/California_Reservoir_Prediction/main/data/"
        urls = {
            "folsom_evaporation.csv": f"{github_repo_url}folsomlake/folsom_evaporation.csv",
            "folsom_storage.csv": f"{github_repo_url}folsomlake/folsom_storage.csv",
            "5SI.csv": f"{github_repo_url}5SI.csv",
            "8SI.csv": f"{github_repo_url}8SI.csv"
        }

        os.makedirs(local_data_dir, exist_ok=True)

        for file_name, url in urls.items():
            file_path = os.path.join(local_data_dir, file_name)
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
        logger.info("Files downloaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading files: {e}")

def process_data(local_data_dir):
    """Process the downloaded CSV files and save the result as a Parquet file.

    Args:
        local_data_dir (str): The local directory where the downloaded files are stored.

    Returns:
        str: The path to the processed Parquet file.
    """
    try:
        storage_csv_file = os.path.join(local_data_dir, 'folsom_storage.csv')
        evaporation_csv_file = os.path.join(local_data_dir, 'folsom_evaporation.csv')
        north_sierra_csv_file = os.path.join(local_data_dir, '8SI.csv')
        south_sierra_csv_file = os.path.join(local_data_dir, '5SI.csv')

        # Load CSV files into Pandas DataFrames
        storage = pd.read_csv(storage_csv_file, header=7)
        evaporation = pd.read_csv(evaporation_csv_file, header=7)
        northsierra = pd.read_csv(north_sierra_csv_file)
        southsierra = pd.read_csv(south_sierra_csv_file)

        drop_list = ['Location', 'Parameter', 'Timestep', 'Aggregation', 'Units']
        storage.drop(drop_list, axis=1, inplace=True)
        evaporation.drop(drop_list, axis=1, inplace=True)

        storage.rename(columns={'Result': 'storage'}, inplace=True)
        evaporation.rename(columns={'Result': 'evaporation'}, inplace=True)

        start_date = "1989-10-01"
        end_date = "2021-10-01"

        storage = storage[(storage['Datetime (UTC)'] >= start_date) & (storage['Datetime (UTC)'] <= end_date)]
        evaporation = evaporation[(evaporation['Datetime (UTC)'] >= start_date) & (evaporation['Datetime (UTC)'] <= end_date)]

        storage['date'] = pd.to_datetime(storage['Datetime (UTC)']).dt.date
        evaporation['date'] = pd.to_datetime(evaporation['Datetime (UTC)']).dt.date
        storage.drop('Datetime (UTC)', axis=1, inplace=True)
        evaporation.drop('Datetime (UTC)', axis=1, inplace=True)

        storage.set_index('date', inplace=True)
        evaporation.set_index('date', inplace=True)

        northsierra.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        northsierra['date'] = pd.to_datetime(northsierra['date'])
        northsierra.set_index('date', inplace=True)

        drop_list = ['STATION_ID', 'DURATION', 'SENSOR_NUMBER', 'SENS_TYPE', 'DATE TIME', 'DATA_FLAG', 'UNITS']
        southsierra.drop(drop_list, axis=1, inplace=True)
        southsierra['OBS DATE'] = southsierra['OBS DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        southsierra = southsierra[(southsierra['OBS DATE'] >= start_date) & (southsierra['OBS DATE'] <= end_date)]
        southsierra.rename(columns={'OBS DATE': 'date'}, inplace=True)
        southsierra.set_index('date', inplace=True)

        storage = storage.resample('1M').mean().round(2)
        evaporation = evaporation.resample('1M').mean().round(2)
        southsierra = southsierra.resample('1M').sum().round(2)
        northsierra = northsierra.resample('1M').sum().round(2)

        northsierra.drop([pd.to_datetime('2021-10-31'), pd.to_datetime('2021-11-30')], inplace=True)
        southsierra.drop(pd.to_datetime('2021-10-31'), inplace=True)

        df_list = [northsierra, evaporation, storage]
        merged_data = reduce(lambda left, right: pd.merge(left, right, on=['date'], how='outer'), df_list)

        parquet_file = os.path.join(local_data_dir, "dam_reservoir_storage.parquet")
        table = pa.Table.from_pandas(merged_data)
        pq.write_table(table, parquet_file)

        logger.info("Data processed and saved as Parquet file successfully.")
        return parquet_file

    except Exception as e:
        logger.error(f"Error processing data: {e}")

def upload_to_gcs(bucket_name, object_name, local_file):
    """Upload the specified local file to Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        object_name (str): The name of the object to be created in GCS.
        local_file (str): The path to the local file to be uploaded.
    """
    try:
        storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB
        storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 5 MB

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_file)

        logger.info("File uploaded to GCS successfully.")
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="data_ingestion_gcs_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=['dtc-de'],
) as dag:

    local_data_dir = "/opt/airflow/data"
    BUCKET = "your-gcs-bucket-name"
    path_to_local_home = "/opt/airflow/data"
    parquet_file = "dam_reservoir_storage.parquet"

    download_dataset_task = PythonOperator(
        task_id="download_dataset_task",
        python_callable=download_files,
        op_kwargs={"local_data_dir": local_data_dir},
    )

    format_to_parquet_task = PythonOperator(
        task_id="format_to_parquet_task",
        python_callable=process_data,
        op_kwargs={"local_data_dir": local_data_dir},
    )

    clean_up_files = BashOperator(
        task_id="clean_up_files",
        bash_command="cd /opt/airflow/data && rm -f *.csv",
    )

    local_to_gcs_task = PythonOperator(
        task_id="local_to_gcs_task",
        python_callable=upload_to_gcs,
        op_kwargs={
            "bucket_name": BUCKET,
            "object_name": f"raw/{parquet_file}",
            "local_file": f"{path_to_local_home}/{parquet_file}",
        },
    )

    delete_downloaded_files = BashOperator(
        task_id="delete_downloaded_files",
        bash_command="cd /opt/airflow/data && rm -f *.parquet",
    )

    download_dataset_task >> format_to_parquet_task >> clean_up_files >> local_to_gcs_task >> delete_downloaded_files
