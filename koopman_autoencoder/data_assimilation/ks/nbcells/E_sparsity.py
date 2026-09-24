# ruff: noqa: F821
# mypy: disable-error-code="name-defined"
_ = sweep_plot(
    "E_sparsity",
    "fraction of grid observed",
    "E. Recovery vs spatial sparsity",
    logx=True,
)
