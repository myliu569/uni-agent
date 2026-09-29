from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import Mock

import pytest

from examples.agent_aware_router import run_infer
from examples.agent_aware_router.run_infer import init_config as init_router_config
from examples.inference.parallel_infer_verl import init_config

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def router_args():
    return Namespace(
        temperature=0.4,
        top_p=0.85,
        top_k=20,
        allowed_request_sampling_param_keys=["temperature", "top_p", "top_k"],
        n=2,
        nnodes=1,
        n_gpus_per_node=1,
        model_path="/tmp/test-model",
        engine="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_rollout_routing_replay=False,
        tool_parser="qwen3_coder",
        gateway_count=1,
        concurrency=4,
        task_config="/not-loaded-until-task-preparation.yaml",
        log_dir="/tmp/test-inference",
        num_workers=1,
        max_model_len=32768,
        max_num_seqs=16,
        enable_mooncake=False,
        mooncake_config_path="/not-read-when-disabled.json",
        mooncake_save_decode_cache=False,
        device="gpu",
        kv_events=False,
        router_config_path="uni_agent/agent_aware_router/configs/agent_aware_router.yaml",
        simulated_runner_fqn=None,
        load_threshold=0.5,
        prompt_length=2048,
        response_length=8192,
    )


@pytest.mark.parametrize(
    "entrypoint,simulated_runner",
    [(init_config, None), (init_router_config, None), (init_router_config, "example.simulated_runner")],
)
def test_inference_sampling_uses_run_options_and_preserves_length_configuration(
    entrypoint, simulated_runner, router_args
):
    args = router_args
    args.simulated_runner_fqn = simulated_runner
    config = entrypoint(
        args,
        served_model_name="policy",
    )

    rollout = config.actor_rollout_ref.rollout
    for sampling in (rollout, rollout.val_kwargs):
        assert sampling.temperature == 0.4
        assert sampling.top_p == 0.85
        assert sampling.top_k == 20
    assert rollout.val_kwargs.do_sample is True
    expected_prompt_length = args.prompt_length if entrypoint is init_router_config else 4096
    expected_response_length = args.response_length
    assert rollout.prompt_length == config.data.max_prompt_length == expected_prompt_length
    assert rollout.response_length == config.data.max_response_length == expected_response_length
    if entrypoint is init_config:
        assert rollout.max_model_len == expected_prompt_length + expected_response_length
    assert rollout.custom.agent_framework.allowed_request_sampling_param_keys == ["temperature", "top_p", "top_k"]
    task_runner = rollout.custom.agent_framework.agent_runners.task
    if simulated_runner:
        assert task_runner.runner_fqn == simulated_runner
        assert not task_runner.runner_kwargs
    else:
        assert task_runner.runner_kwargs.task_config_path == args.task_config


@pytest.mark.parametrize("enabled,save_decode", [(False, False), (True, False), (True, True)])
def test_mooncake_engine_config(router_args, enabled, save_decode):
    router_args.enable_mooncake = enabled
    router_args.mooncake_save_decode_cache = save_decode
    rollout = init_router_config(router_args, served_model_name="policy").actor_rollout_ref.rollout
    assert rollout.enable_prefix_caching
    if not enabled:
        assert "kv_transfer_config" not in rollout.engine_kwargs.vllm
    else:
        assert rollout.engine_kwargs.vllm.kv_transfer_config == {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"save_decode_cache": save_decode},
        }


@pytest.mark.parametrize("address", [None, "auto"])
@pytest.mark.parametrize("enabled", [False, True])
def test_mooncake_ray_job_env(router_args, tmp_path, monkeypatch, address, enabled):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "store.json").write_text("{}")
    router_args.enable_mooncake = enabled
    router_args.mooncake_config_path = "store.json" if enabled else "missing.json"
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "stale-driver-path")
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "127.0.0.1:50053")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    if address:
        monkeypatch.setenv("RAY_ADDRESS", address)
    else:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: False)
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    run_infer._init_ray(router_args)
    kwargs = init.call_args.kwargs
    if enabled:
        env = kwargs["runtime_env"]["env_vars"]
        assert env["MOONCAKE_CONFIG_PATH"] == str(tmp_path / "store.json")
        assert router_args.mooncake_config_path == env["MOONCAKE_CONFIG_PATH"]
        assert env["MOONCAKE_PREFERRED_SEGMENT"] == "127.0.0.1:50053"
        assert env["PYTHONHASHSEED"] == "42"
    else:
        assert "runtime_env" not in kwargs
    if address:
        assert kwargs["address"] == "auto"
        assert "_system_config" not in kwargs
    else:
        assert "idle_worker_killing_time_threshold_ms" in kwargs["_system_config"]


@pytest.mark.parametrize("contents", [None, "not json", "[]", "null"])
def test_invalid_mooncake_config_fails_before_ray(router_args, tmp_path, monkeypatch, contents):
    router_args.enable_mooncake = True
    path = tmp_path / "store.json"
    router_args.mooncake_config_path = str(path)
    if contents is not None:
        path.write_text(contents)
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    with pytest.raises(ValueError, match="Mooncake"):
        run_infer._init_ray(router_args)
    init.assert_not_called()


def test_mooncake_expands_home(router_args, tmp_path, monkeypatch):
    # Patch expanduser rather than changing the process HOME.
    path = tmp_path / "store.json"
    path.write_text("{}")
    router_args.enable_mooncake = True
    router_args.mooncake_config_path = "~/store.json"
    monkeypatch.setattr(run_infer.Path, "expanduser", lambda self: path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "old")
    assert run_infer._prepare_mooncake_env(router_args)["MOONCAKE_CONFIG_PATH"] == str(path)


@pytest.mark.parametrize("matches", [False, True])
def test_already_initialized_ray_requires_matching_env(router_args, tmp_path, monkeypatch, matches):
    path = tmp_path / "store.json"
    path.write_text("{}")
    router_args.enable_mooncake = True
    router_args.mooncake_config_path = str(path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", "old")
    env = run_infer._prepare_mooncake_env(router_args)
    monkeypatch.setattr(run_infer.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        run_infer.ray,
        "get_runtime_context",
        lambda: Namespace(runtime_env={"env_vars": env if matches else {}}),
    )
    init = Mock()
    monkeypatch.setattr(run_infer.ray, "init", init)
    if matches:
        run_infer._init_ray(router_args)
    else:
        with pytest.raises(ValueError, match="already initialized"):
            run_infer._init_ray(router_args)
    init.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_report_records_submitted_mooncake_config(router_args, tmp_path, enabled):
    router_args.enable_mooncake = enabled
    router_args.mooncake_save_decode_cache = True
    router_args.result_path = str(tmp_path / "result.json")
    router_args.data_path = "/tmp/data.parquet"
    run_infer._report(
        {"scores": [1.0], "per_uid": {"task": [1.0]}, "uid_status": {"task": "finished"}},
        wall=1.0,
        num_prompts=1,
        n=1,
        args=router_args,
        served_model_name="policy",
    )
    metadata = json.loads((tmp_path / "result.json").read_text())["mooncake"]
    assert metadata["enabled"] == enabled
    if enabled:
        rollout = init_router_config(router_args, served_model_name="policy").actor_rollout_ref.rollout
        assert metadata["kv_transfer_config"] == rollout.engine_kwargs.vllm.kv_transfer_config
        assert metadata["config_path"] == router_args.mooncake_config_path
    else:
        assert metadata["config_path"] is None and metadata["kv_transfer_config"] is None
