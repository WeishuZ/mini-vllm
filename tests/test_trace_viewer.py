from mini_vllm.trace_viewer import (
    build_demo_engine,
    build_trace,
    render_html,
    write_trace_html,
)


def test_trace_captures_prefix_hits_and_preemptions():
    trace = build_trace(build_demo_engine(n_requests=24), max_steps=60)

    assert trace["frames"]
    assert all(f["narration"] for f in trace["frames"])
    assert any(f["work"]["prefix_hits"] for f in trace["frames"])
    assert any(f["work"]["preempted"] for f in trace["frames"])
    assert trace["metadata"]["prefix_caching"] is True


def test_trace_html_is_standalone(tmp_path):
    trace = build_trace(build_demo_engine(n_requests=12), max_steps=20)
    html = render_html(trace)

    assert "__TRACE_JSON__" not in html
    assert "mini-vLLM scheduler trace" in html
    assert "const TRACE =" in html

    path = write_trace_html(trace, tmp_path / "trace.html")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == html
