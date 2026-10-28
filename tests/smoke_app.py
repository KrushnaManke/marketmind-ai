"""Smoke-test every dashboard page WITHOUT streamlit/plotly installed.

Streamlit and Plotly are replaced by strict stubs that
  * run each page top to bottom (catches NameError, bad pandas code, missing files),
  * check that every column name passed to px.* / st.dataframe(column_config=...) really exists,
  * exercise the 'score a customer' form and the SQL box (buttons return True).
It cannot check real Streamlit/Plotly API behaviour or visual layout: run `streamlit run app/Home.py` for that.

    python tests/smoke_app.py
"""
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


class StopScript(Exception):
    pass


class Ctx:
    """Context manager that also accepts any st.* call (so `with col: col.metric(...)` works)."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(ST, name)


class Sidebar(Ctx):
    pass


def _cols(spec):
    n = spec if isinstance(spec, int) else len(spec)
    return [Ctx() for _ in range(n)]


class FakeST:
    def __init__(self):
        self.calls, self.sidebar = [], Sidebar()
        self.column_config = types.SimpleNamespace(**{k: (lambda *a, **kw: {"kind": k, **kw}) for k in
                                                      ("ProgressColumn", "NumberColumn", "CheckboxColumn", "TextColumn")})

    # decorators
    def cache_data(self, f=None, **kw): return f if f else (lambda g: g)
    cache_resource = cache_data

    # layout
    def columns(self, spec, **kw): return _cols(spec)
    def tabs(self, labels): return [Ctx() for _ in labels]
    def expander(self, *a, **kw): return Ctx()
    def form(self, *a, **kw): return Ctx()
    def stop(self): raise StopScript()

    # inputs (return defaults)
    def selectbox(self, label, options, index=0, **kw):
        options = list(options); assert options, f"selectbox '{label}' has no options"; return options[index]
    def multiselect(self, label, options, default=None, **kw): return list(default) if default is not None else []
    def slider(self, label, a=0, b=1, value=None, **kw): return value if value is not None else a
    def checkbox(self, label, value=False, **kw): return value
    def text_area(self, label, value="", **kw): return value
    def button(self, *a, **kw): return True
    def form_submit_button(self, *a, **kw): return True
    def download_button(self, *a, **kw): return None

    # outputs
    def metric(self, label, value, delta=None, **kw):
        assert isinstance(value, (str, int, float)), f"metric '{label}' got {type(value)}"
    def dataframe(self, df, **kw):
        assert isinstance(df, pd.DataFrame), "st.dataframe expects a DataFrame"
        for k in (kw.get("column_config") or {}):
            assert k in df.columns, f"column_config key '{k}' not in dataframe columns {list(df.columns)}"
    def plotly_chart(self, fig, **kw): pass
    def set_page_config(self, **kw): pass
    def __getattr__(self, name):           # title, caption, markdown, write, subheader, info, warning, error, success, divider, code ...
        return lambda *a, **kw: None


ST = FakeST()


def fake_px_factory(name):
    def fn(data_frame=None, *a, **kw):
        if name != "imshow":
            assert isinstance(data_frame, pd.DataFrame), f"px.{name}: first arg must be a DataFrame"
            for arg in ("x", "y", "color", "facet_col", "text"):
                v = kw.get(arg)
                if isinstance(v, str):
                    assert v in data_frame.columns, f"px.{name}({arg}='{v}') not in columns {list(data_frame.columns)}"
                if arg == "text" and isinstance(v, pd.Series):
                    assert len(v) == len(data_frame), "text series length mismatch"
        return MagicMock(name=f"fig_{name}")
    return fn


def install_stubs():
    st = types.ModuleType("streamlit")
    st.__dict__.update({k: getattr(ST, k) for k in dir(ST) if not k.startswith("__") and k != "calls"})
    st.__getattr__ = lambda name: getattr(ST, name)
    sys.modules["streamlit"] = st
    px = types.ModuleType("plotly.express")
    for n in ("bar", "line", "scatter", "histogram", "imshow", "box", "area"):
        setattr(px, n, fake_px_factory(n))
    go = MagicMock(name="plotly.graph_objects")
    pl = types.ModuleType("plotly"); pl.express, pl.graph_objects = px, go
    sys.modules.update({"plotly": pl, "plotly.express": px, "plotly.graph_objects": go})


def main():
    install_stubs()
    sys.path.insert(0, str(APP))
    pages = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]
    failed = 0
    for p in pages:
        try:
            runpy.run_path(str(p), run_name="__main__")
            print(f"  OK   {p.relative_to(ROOT)}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {p.relative_to(ROOT)}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=4)
    print(f"\n{len(pages) - failed}/{len(pages)} pages ran")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
