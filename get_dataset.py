import os
import wget

daily_raw_url = "http://yanran.li/files/ijcnlp_dailydialog.zip"
data_path = './data/'


def download_dailydialog(daily_raw_fname: str, data_path: str):
    """Download the raw DailyDialog dataset
    Args:
        daily_raw_fname (str): Raw DailyDialog dataset URL
        data_path (str): Path to save
    """
    wget.download(daily_raw_fname, data_path)
    # Manually unzip the train/dev/test files

if __name__ == '__main__':
    os.makedirs(data_path, exist_ok=True)
    download_dailydialog(daily_raw_url, data_path)