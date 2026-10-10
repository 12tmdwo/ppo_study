"""plot_compare.py 검사.

실행: pytest ppo/test_plot_compare.py
"""
from plot_compare import find_runs, passed


def test_find_runs_picks_latest_per_seed(tmp_path):
    for name in ("2026-10-10_09-00-00_seed1", "2026-10-10_10-00-00_seed1", "2026-10-10_09-30-00_seed2",
                 "2026-10-09_19-45-58", "notes_seedling"):
        (tmp_path / name).mkdir()
    (tmp_path / "2026-10-10_11-00-00_seed3").write_text("폴더가 아닌 파일")
    runs = find_runs(tmp_path)
    assert set(runs) == {1, 2}, "이름이 '_seed<번호>'로 끝나는 폴더만 골라야 합니다."
    assert runs[1].name == "2026-10-10_10-00-00_seed1", "같은 씨앗값이 여러 개면 가장 최근 것을 골라야 합니다."


def test_find_runs_missing_root(tmp_path):
    assert find_runs(tmp_path / "없는_폴더") == {}


def test_passed_thresholds():
    assert passed(4.5, 290.0)
    assert passed(4.92, 300.0)
    assert not passed(4.49, 300.0)
    assert not passed(4.92, 289.9)
