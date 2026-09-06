# LLM Clients - Consistency Improvements

## Open

_None._

## Fixed

1. ~~`validate_model()` is never called~~ — every client's `get_llm()` calls `warn_if_unknown_model()`, which warns (not errors) on unknown models. Covered by `tests/test_model_validation.py`.
2. ~~Inconsistent parameter handling~~ — GoogleClient accepts unified `api_key` and maps it to `google_api_key`.
3. ~~`base_url` accepted but ignored~~ — all clients pass `base_url` to their LLM constructors.
4. ~~Update validators.py with models from CLI~~ — synced in v0.2.2.

Provider routing in `factory.py` is covered offline by `tests/test_llm_client_factory.py`.
