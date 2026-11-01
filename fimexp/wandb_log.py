"""Private W&B logging: configs and scalar metrics only.

Disabled: code saving, git/diff capture, console capture, system metadata (host, program, args, git),
system/GPU stats, requirements upload, model/checkpoint upload. Credentials come from the local
`wandb login` (or WANDB_API_KEY in the user's own shell) and are never logged.
Online runs refuse to start unless the target project already exists and its access is in `allowed_access`.
"""

import os

PRIVATE_SETTINGS = dict(
    save_code=False, disable_code=True, disable_git=True, disable_job_creation=True, console="off",
    x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
)
SAFE_CONFIG_KEYS = ("model", "base_precision", "format", "split", "data", "lora", "training", "evaluation", "run")


ACCESS_QUERY = "query P($entity: String!, $name: String!) { project(name: $name, entityName: $entity) { id access } }"


def _gql(query: str, variables: dict) -> dict:
    import wandb
    r = wandb.Api()._service_api.execute_graphql(query, variables)
    return r.get("data", r)


def project_access(entity: str, project: str) -> str | None:
    """Access level of an existing project via the W&B GraphQL API, or None if it does not exist."""
    return (_gql(ACCESS_QUERY, {"entity": entity, "name": project}).get("project") or {}).get("access")


def create_private_project(entity: str, project: str) -> str | None:
    """Create the project with access PRIVATE (only if it does not exist yet), then read the access back."""
    from wandb.apis._generated import CREATE_PROJECT_GQL, UpsertModelInput
    if project_access(entity, project) is not None:
        raise RuntimeError(f"{entity}/{project} already exists; not modifying it")
    _gql(CREATE_PROJECT_GQL, {"input": UpsertModelInput(name=project, entity_name=entity, access="PRIVATE").model_dump()})
    return project_access(entity, project)


def safe_config(effective: dict) -> dict:
    """Experiment settings only: no paths to local files' contents, no dataset text, no code."""
    return {k: effective[k] for k in SAFE_CONFIG_KEYS if k in effective}


def init(wcfg: dict, run_name: str, effective: dict, out_dir):
    """Start a W&B run (or return None if disabled). `out_dir` holds wandb's local files."""
    if not wcfg or not wcfg.get("enabled"):
        return None
    import wandb
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    mode = wcfg.get("mode", "online")
    if mode == "online":
        access = project_access(wcfg["entity"], wcfg["project"])
        if access is None:
            raise RuntimeError(f"W&B project {wcfg['entity']}/{wcfg['project']} does not exist; create it as private first")
        if access not in wcfg["allowed_access"]:
            raise RuntimeError(f"W&B project access is {access!r}, not in {wcfg['allowed_access']}; refusing to upload")
    return wandb.init(entity=wcfg.get("entity"), project=wcfg["project"], name=run_name, dir=str(out_dir), mode=mode,
                      config=safe_config(effective), settings=wandb.Settings(**PRIVATE_SETTINGS))
