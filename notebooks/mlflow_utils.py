"""MLflow: tracking URI из Docker и хелперы для ноутбука."""
from __future__ import annotations

import os

# В Docker-образе Jupyter нет git — MLflow пытается записать commit SHA и шумит в лог.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from typing import Any

import mlflow
import numpy as np
from mlflow.models import infer_signature

__version__ = "7"  # bump при изменении API (для importlib.reload в ноутбуке)

SPELL_FEATURE_NAMES = ("mana_cost", "damage")
MLFLOW_ARTIFACT_ROOT = "mlflow-artifacts:/"


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def mlflow_ui_url() -> str:
    """URL для браузера с хоста (Jupyter в Docker → localhost)."""
    return get_tracking_uri().replace("http://mlflow:5000", "http://localhost:5000")


def get_serve_uri() -> str:
    return os.environ.get("MLFLOW_SERVE_URI", "http://mlflow-serve:5001")


def get_alpaca_serve_uri() -> str:
    """URI по умолчанию (Docker DNS). Для запросов используйте resolve_alpaca_serve_uri()."""
    return os.environ.get("MLFLOW_ALPACA_SERVE_URI", "http://mlflow-serve-alpaca:5002")


def alpaca_serve_ui_url() -> str:
    """URL с хоста Windows (порт проброшен в compose)."""
    return "http://localhost:5002"


def wait_for_alpaca_serve(*, max_wait_sec: int = 600, poll_sec: int = 10) -> str:
    """Ждёт /health Alpaca serve (5002) и возвращает URI."""
    return _wait_for_lm_serve(
        resolve_alpaca_serve_uri,
        max_wait_sec=max_wait_sec,
        poll_sec=poll_sec,
        log_container="mlflow-serve-alpaca",
    )


def resolve_alpaca_serve_uri(*, timeout: float = 5.0) -> str:
    """Первый доступный endpoint serve Alpaca (порт 5002)."""
    return _resolve_lm_serve_uri(
        env_var="MLFLOW_ALPACA_SERVE_URI",
        default_uri="http://mlflow-serve-alpaca:5002",
        docker_host="mlflow-serve-alpaca",
        host_port=5002,
        label="alpaca",
        timeout=timeout,
    )


def _resolve_lm_serve_uri(
    *,
    env_var: str,
    default_uri: str,
    docker_host: str,
    host_port: int,
    label: str,
    timeout: float = 5.0,
) -> str:
    from urllib.parse import urlparse

    candidates: list[str] = []
    env_uri = os.environ.get(env_var)
    if env_uri:
        candidates.append(env_uri)
    candidates.extend(
        [
            default_uri,
            f"http://host.docker.internal:{host_port}",
            f"http://localhost:{host_port}",
            f"http://127.0.0.1:{host_port}",
        ]
    )
    seen: set[str] = set()
    for raw in candidates:
        uri = raw.rstrip("/")
        if uri in seen or not urlparse(uri).hostname:
            continue
        seen.add(uri)
        try:
            import urllib.request

            with urllib.request.urlopen(f"{uri}/health", timeout=timeout) as resp:
                if resp.status == 200:
                    return uri
        except (OSError, Exception):
            continue
    raise ConnectionError(
        f"MLflow serve ({label}) недоступен на порту {host_port}. "
        f"docker compose up -d {docker_host}\n"
        f"Пробовали: {', '.join(seen)}"
    )


def _wait_for_lm_serve(
    resolve_fn,
    *,
    max_wait_sec: int = 600,
    poll_sec: int = 10,
    log_container: str,
) -> str:
    import time

    deadline = time.time() + max_wait_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            return resolve_fn()
        except ConnectionError as exc:
            last_err = exc
            time.sleep(poll_sec)
    raise ConnectionError(
        f"Serve не поднялся за {max_wait_sec}s. docker compose logs {log_container}"
    ) from last_err


def _predict_lm_via_serve(
    prompt: str,
    *,
    resolve_fn,
    serve_uri: str | None,
    max_new_tokens: int,
    timeout: float,
    log_container: str,
) -> str:
    import time

    import requests

    uri = (serve_uri or resolve_fn()).rstrip("/")
    payloads = [
        {"inputs": prompt},
        {"inputs": [prompt]},
        {
            "inputs": prompt,
            "params": {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
            },
        },
    ]
    last_err: Exception | None = None
    for attempt in range(5):
        for payload in payloads:
            try:
                resp = requests.post(
                    f"{uri}/invocations", json=payload, timeout=timeout
                )
                resp.raise_for_status()
                return _parse_serve_response(resp.json())
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
        time.sleep(min(15, 3 * (attempt + 1)))
    raise ConnectionError(
        f"MLflow serve не ответил на {uri}/invocations. "
        f"docker compose logs {log_container}"
    ) from last_err


def get_deepseek_serve_uri() -> str:
    return os.environ.get(
        "MLFLOW_DEEPSEEK_SERVE_URI", "http://mlflow-serve-deepseek:5003"
    )


def deepseek_serve_ui_url() -> str:
    return "http://localhost:5003"


def resolve_deepseek_serve_uri(*, timeout: float = 5.0) -> str:
    return _resolve_lm_serve_uri(
        env_var="MLFLOW_DEEPSEEK_SERVE_URI",
        default_uri="http://mlflow-serve-deepseek:5003",
        docker_host="mlflow-serve-deepseek",
        host_port=5003,
        label="deepseek",
        timeout=timeout,
    )


def wait_for_deepseek_serve(*, max_wait_sec: int = 600, poll_sec: int = 10) -> str:
    return _wait_for_lm_serve(
        resolve_deepseek_serve_uri,
        max_wait_sec=max_wait_sec,
        poll_sec=poll_sec,
        log_container="mlflow-serve-deepseek",
    )


def predict_deepseek_via_serve(
    prompt: str,
    *,
    serve_uri: str | None = None,
    max_new_tokens: int = 80,
    timeout: float = 300.0,
) -> str:
    """Генерация через serve DeepSeek (порт 5003), без загрузки модели в ноутбук."""
    return _predict_lm_via_serve(
        prompt,
        resolve_fn=resolve_deepseek_serve_uri,
        serve_uri=serve_uri,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        log_container="mlflow-serve-deepseek",
    )


def predict_alpaca_via_serve(
    prompt: str,
    *,
    serve_uri: str | None = None,
    max_new_tokens: int = 80,
    timeout: float = 300.0,
) -> str:
    """Генерация через serve Alpaca/distilgpt2 (порт 5002)."""
    return _predict_lm_via_serve(
        prompt,
        resolve_fn=resolve_alpaca_serve_uri,
        serve_uri=serve_uri,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        log_container="mlflow-serve-alpaca",
    )


def _parse_serve_response(data: Any) -> str:
    if isinstance(data, list) and data:
        return str(data[0])
    if isinstance(data, dict):
        if "predictions" in data:
            return _parse_serve_response(data["predictions"])
        return str(
            data.get("generated_text")
            or data.get("output")
            or data.get("text")
            or data
        )
    return str(data)


def predict_spells_via_serve(
    rows: list[list[float]],
    *,
    columns: tuple[str, ...] = SPELL_FEATURE_NAMES,
    serve_uri: str | None = None,
    timeout: float = 30.0,
) -> Any:
    """Предсказание через MLflow Model Server (POST /invocations), без загрузки модели в память."""
    import requests

    uri = (serve_uri or get_serve_uri()).rstrip("/")
    payload = {"dataframe_split": {"columns": list(columns), "data": rows}}
    resp = requests.post(f"{uri}/invocations", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _experiment_needs_recreate(artifact_location: str | None) -> bool:
    """Пути вида /mlflow/artifacts/N — только на сервере, Jupyter пишет локально и падает."""
    if not artifact_location:
        return False
    loc = artifact_location.strip()
    if loc.startswith("mlflow-artifacts:"):
        return False
    if loc.startswith("file:") or "/mlflow/artifacts" in loc:
        return True
    return loc.startswith("/mlflow")


def _ensure_experiment(client: Any, experiment_name: str) -> str:
    from mlflow.exceptions import RestException

    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None:
        stage = getattr(exp, "lifecycle_stage", "active")
        if stage == "deleted":
            raise RuntimeError(
                f"Experiment '{experiment_name}' в корзине MLflow (имя занято).\n"
                "Выполните: docker compose restart mlflow\n"
                "или: docker compose exec mlflow sqlite3 /mlflow/mlflow.db "
                f"\"DELETE FROM experiments WHERE name='{experiment_name}';\""
            )
        if _experiment_needs_recreate(exp.artifact_location):
            print(
                f"Удаляем experiment '{experiment_name}' "
                f"(artifact_location={exp.artifact_location!r}) — runs будут удалены."
            )
            client.delete_experiment(exp.experiment_id)
            exp = None

    if exp is None:
        try:
            return client.create_experiment(
                experiment_name,
                artifact_location=MLFLOW_ARTIFACT_ROOT,
            )
        except RestException as exc:
            if "UNIQUE" in str(exc):
                raise RuntimeError(
                    f"Имя '{experiment_name}' занято в БД MLflow.\n"
                    "Выполните: docker compose restart mlflow"
                ) from exc
            raise

    return exp.experiment_id


def setup_mlflow(experiment_name: str = "spells_classifiers") -> str:
    """Подключение к MLflow Server и выбор experiment."""
    from mlflow.tracking import MlflowClient

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    exp_id = _ensure_experiment(client, experiment_name)
    mlflow.set_experiment(experiment_id=exp_id)

    exp = client.get_experiment(exp_id)
    print(f"artifact_location: {exp.artifact_location}")
    print(f"MLflow tracking: {uri}")
    print(f"UI в браузере: {mlflow_ui_url()}")
    print(f"Experiment: {experiment_name}")
    return uri


def log_sklearn_spell_model(
    sk_model: Any,
    *,
    classifier_name: str,
    X_sample: np.ndarray,
    metrics: dict[str, float] | None = None,
    artifact_path: str = "model",
) -> None:
    """Сохраняет sklearn-модель в текущий MLflow run (артефакт + signature)."""
    X_ex = np.asarray(X_sample[:5])
    signature = infer_signature(X_ex, sk_model.predict(X_ex))

    for key, value in (metrics or {}).items():
        mlflow.log_metric(key, float(value))

    mlflow.set_tag("classifier", classifier_name)
    mlflow.set_tag("task", "spell_fireball_vs_lightning")
    mlflow.log_param("features", ",".join(SPELL_FEATURE_NAMES))

    mlflow.sklearn.log_model(
        sk_model,
        artifact_path=artifact_path,
        signature=signature,
        input_example=X_ex[:1],
    )
    print(f"  → модель сохранена в MLflow: artifact '{artifact_path}'")


REGISTERED_MODEL_NAME = "spells-classifier"
ALPACA_REGISTERED_MODEL_NAME = "alpaca-causal-lm"
DEEPSEEK_REGISTERED_MODEL_NAME = "deepseek-causal-lm"
COMPARE_METRICS = ("test_f1", "test_roc_auc", "accuracy_real", "holdout_f1")


def list_classifier_runs(
    experiment_name: str = "spells_classifiers",
    *,
    metric_names: tuple[str, ...] = COMPARE_METRICS,
) -> "pd.DataFrame":
    """Таблица runs experiment с метриками (для сравнения моделей в UI/ноутбуке)."""
    import pandas as pd
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' не найден. Сначала обучите модели.")

    rows: list[dict[str, Any]] = []
    for run in client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    ):
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "classifier": run.data.params.get("classifier") or run.info.run_name,
        }
        for key in metric_names:
            row[key] = run.data.metrics.get(key)
        rows.append(row)

    return pd.DataFrame(rows)


def select_best_run(
    runs_df: "pd.DataFrame",
    *,
    primary_metric: str = "test_f1",
    fallback_metrics: tuple[str, ...] = ("test_roc_auc", "accuracy_real"),
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Выбирает лучший run по primary_metric, затем по fallback."""
    if runs_df.empty:
        raise RuntimeError("Нет завершённых runs в experiment.")

    metrics_to_try = (primary_metric,) + fallback_metrics
    last_error: str | None = None

    for metric in metrics_to_try:
        if metric not in runs_df.columns:
            continue
        subset = runs_df.dropna(subset=[metric])
        if subset.empty:
            last_error = f"ни у одного run нет метрики '{metric}'"
            continue
        best_idx = (
            subset[metric].idxmax() if higher_is_better else subset[metric].idxmin()
        )
        row = subset.loc[best_idx]
        return {
            "run_id": row["run_id"],
            "classifier": row["classifier"],
            "metric": metric,
            "value": float(row[metric]),
        }

    raise RuntimeError(
        f"Не удалось выбрать лучший run: {last_error or 'нет метрик'}. "
        f"Выполните ячейку с test_f1 / accuracy_real."
    )


def log_alpaca_causal_lm(
    model: Any | None = None,
    tokenizer: Any | None = None,
    *,
    model_dir: str | None = None,
    artifact_path: str = "model",
) -> None:
    """Логирует HuggingFace causal LM в текущий MLflow run (flavor transformers)."""
    # protobuf 5.x + upb ломает mlflow.transformers (FieldDescriptor.label)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import mlflow

    if model_dir:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if model is None or tokenizer is None:
        raise ValueError("Нужны model+tokenizer или model_dir")

    mlflow.set_tag("task", "alpaca_instruction_lm")
    if model_dir:
        mlflow.log_param("local_model_dir", model_dir)

    import mlflow.transformers

    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path=artifact_path,
        task="text-generation",
    )
    print(f"  → causal LM в MLflow (flavor transformers): '{artifact_path}'")


def register_alpaca_from_local(
    model_dir: str,
    *,
    experiment_name: str = "alpaca_llm_finetune",
    target_stage: str = "Production",
    model_name: str = ALPACA_REGISTERED_MODEL_NAME,
) -> str:
    """Перелогировать локальную папку в MLflow и зарегистрировать (если serve падал без MLmodel)."""
    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run(run_name="alpaca-causal-lm-reregister") as run:
        log_alpaca_causal_lm(model_dir=model_dir)
        run_id = run.info.run_id
    register_and_promote_run(
        run_id, model_name=model_name, target_stage=target_stage
    )
    print(
        "Перезапустите serve: docker compose up -d mlflow-serve-alpaca --force-recreate"
    )
    return run_id


def register_deepseek_from_local(
    model_dir: str,
    *,
    experiment_name: str = "alpaca_llm_finetune",
    target_stage: str = "Production",
    model_name: str = DEEPSEEK_REGISTERED_MODEL_NAME,
) -> str:
    """Перелогировать DeepSeek в Registry и поднять serve на :5003."""
    setup_mlflow(experiment_name=experiment_name)
    with mlflow.start_run(run_name="deepseek-causal-lm-reregister") as run:
        log_alpaca_causal_lm(model_dir=model_dir)
        run_id = run.info.run_id
    register_and_promote_run(
        run_id, model_name=model_name, target_stage=target_stage
    )
    print(
        "Перезапустите serve: docker compose up -d mlflow-serve-deepseek --force-recreate"
    )
    return run_id


def register_and_promote_run(
    run_id: str,
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    target_stage: str = "Staging",
    artifact_path: str = "model",
) -> Any:
    """Регистрирует модель из run и переводит версию на следующий этап (Staging/Production)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    version_info = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=version_info.version,
        stage=target_stage,
        archive_existing_versions=(target_stage == "Production"),
    )
    print(
        f"Model Registry: {model_name} v{version_info.version} → stage '{target_stage}'"
    )
    return version_info


def load_registered_spell_model(
    model_name: str = REGISTERED_MODEL_NAME,
    *,
    stage: str = "Staging",
):
    """Загружает зарегистрированную модель по stage (None → Staging → Production)."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.sklearn.load_model(model_uri)
