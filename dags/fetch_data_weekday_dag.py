from datetime import timedelta
from pathlib import Path
import sys

import pendulum
import yaml
from airflow.decorators import dag, task


@dag(
    dag_id="fetch_forex_data_weekdays",
    description="Fetch fresh EUR/USD data from OANDA on weekdays (Mon-Fri)",
    schedule="0 0 * * 1-5",  # daily, Monday-Friday
    start_date=pendulum.datetime(2026, 4, 29, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mohit",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["oanda", "forex", "daily", "weekday"],
)
def fetch_data_weekday_dag():
    @task(task_id="fetch_data")
    def run_fetch_data() -> str:
        repo_root = Path(__file__).resolve().parents[1]

        # Ensure project modules are importable in Airflow worker context.
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from src.fetch_data import fetch_data

        params = yaml.safe_load((repo_root / "params.yaml").read_text(encoding="utf-8"))["fetch"]
        output_path = params["output_path"]

        fetch_data(output_path)
        return f"Data fetch completed. Output: {output_path}"

    run_fetch_data()


dag = fetch_data_weekday_dag()
